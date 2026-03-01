//! Dispatch Latency Benchmark — measures kernel launch overhead.
//! Run: cargo run --release --bin bench_dispatch --features cuda

use nabla::prelude::*;
use nabla_bench::{rand_tensor, gpu_sync};
use std::time::Instant;

const N: usize = 4096;
const WARMUP: usize = 50;

fn main() {
    let a = rand_tensor(N, N);
    let b = rand_tensor(N, N);

    // --- Test 1: Burst dispatch ---
    for &(op, count) in &[("exp", 1000usize), ("add", 1000)] {
        for _ in 0..WARMUP { let _ = a.exp(); }
        gpu_sync();
        let mut ring = vec![a.clone(); 4];
        let start = Instant::now();
        for i in 0..count {
            ring[i % 4] = match op { "exp" => a.exp(), _ => &a + &b };
        }
        gpu_sync();
        let ns_per_op = start.elapsed().as_nanos() as f64 / count as f64;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("{{\"test\":\"burst\",\"op\":\"{op}\",\"count\":{count},\"ns_per_op\":{ns_per_op:.0},\"total_ms\":{total_ms:.3}}}");
        drop(ring);
    }

    // --- Test 2: Fused vs unfused (4 ops) ---
    let iters = 1000;
    for _ in 0..WARMUP { let _ = a.exp().sin().cos().tanh(); }
    gpu_sync();

    let mut ring = vec![a.clone(); 4];
    let start = Instant::now();
    for i in 0..iters { ring[i % 4] = a.exp().sin().cos().tanh(); }
    gpu_sync();
    let unfused_ms = start.elapsed().as_secs_f64() * 1000.0;

    for _ in 0..WARMUP { let _: Tensor<f32> = fuse!(a.exp().sin().cos().tanh(); a); }
    gpu_sync();
    let start = Instant::now();
    for i in 0..iters { ring[i % 4] = fuse!(a.exp().sin().cos().tanh(); a); }
    gpu_sync();
    let fused_ms = start.elapsed().as_secs_f64() * 1000.0;
    drop(ring);

    let speedup = unfused_ms / fused_ms;
    println!("{{\"test\":\"fuse_vs_unfuse\",\"unfused_ms\":{unfused_ms:.3},\"fused_ms\":{fused_ms:.3},\"speedup\":{speedup:.2},\"iters\":{iters}}}");

    // --- Test 3: Simulated training step (12 layers x 3 ops = 36 kernels) ---
    let dim = 512;
    let weights: Vec<Tensor<f32>> = (0..12).map(|_| rand_tensor(dim, dim)).collect();
    let x = rand_tensor(dim, dim);
    let steps = 500;

    for _ in 0..WARMUP {
        let mut h = x.clone();
        for w in &weights { h = h.emul(w).exp(); h = &h + &x; }
        gpu_sync();
    }

    let start = Instant::now();
    for _ in 0..steps {
        let mut h = x.clone();
        for w in &weights { h = h.emul(w).exp(); h = &h + &x; }
    }
    gpu_sync();
    let eager_ms = start.elapsed().as_secs_f64() * 1000.0;
    let eager_us_per_step = eager_ms / steps as f64 * 1000.0;
    println!("{{\"test\":\"training_step\",\"mode\":\"eager\",\"steps\":{steps},\"total_ms\":{eager_ms:.3},\"us_per_step\":{eager_us_per_step:.1}}}");

    #[cfg(feature = "cuda")]
    {
        use nabla::{TrainingGraph, cuda_synchronize};
        let mut tg = TrainingGraph::with_warmup(3);
        let capture_ok = {
            let mut step_fn = || {
                let mut h = x.clone();
                for w in &weights { h = h.emul(w).exp(); h = &h + &x; }
            };
            if tg.step(&mut step_fn).is_err() { false }
            else {
                let mut ok = true;
                for _ in 1..5 {
                    let mut sf = || {
                        let mut h = x.clone();
                        for w in &weights { h = h.emul(w).exp(); h = &h + &x; }
                    };
                    if tg.step(&mut sf).is_err() { ok = false; break; }
                }
                ok
            }
        };
        if capture_ok {
            cuda_synchronize();
            let start = Instant::now();
            let mut noop = || {};
            let mut replay_ok = true;
            for _ in 0..steps {
                if tg.step(&mut noop).is_err() { replay_ok = false; break; }
            }
            cuda_synchronize();
            let graph_ms = start.elapsed().as_secs_f64() * 1000.0;
            let graph_us_per_step = graph_ms / steps as f64 * 1000.0;
            if replay_ok {
                let graph_speedup = eager_ms / graph_ms;
                println!("{{\"test\":\"training_step\",\"mode\":\"cuda_graph\",\"steps\":{steps},\"total_ms\":{graph_ms:.3},\"us_per_step\":{graph_us_per_step:.1},\"speedup\":{graph_speedup:.2}}}");
            } else {
                println!("{{\"test\":\"training_step\",\"mode\":\"cuda_graph\",\"status\":\"replay_failed\"}}");
            }
        } else {
            println!("{{\"test\":\"training_step\",\"mode\":\"cuda_graph\",\"status\":\"capture_failed\"}}");
        }
    }

    eprintln!();
    eprintln!("=== Dispatch Benchmark Summary ===");
    eprintln!("Burst dispatch: exp x1000, add x1000 (see JSON above)");
    eprintln!("Fuse 4-op speedup: {speedup:.2}x ({unfused_ms:.3} ms -> {fused_ms:.3} ms over {iters} iters)");
    eprintln!("Training step (36 kernels): {eager_us_per_step:.1} us/step eager");
}
