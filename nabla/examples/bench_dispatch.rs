//! Dispatch Overhead Benchmark — measures CPU-side kernel dispatch latency
//!
//! Measures:
//! 1. Individual op dispatch overhead (many small ops, sync per batch)
//! 2. Fused chain dispatch
//! 3. TrainingGraph capture/replay vs eager
//!
//! Run: cargo run --release --example bench_dispatch --features cuda

use nabla::prelude::*;
use std::time::Instant;

#[cfg(feature = "cuda")]
#[inline]
fn gpu_sync() { nabla::cuda_synchronize(); }
#[cfg(not(feature = "cuda"))]
#[inline]
fn gpu_sync() {}

const N: usize = 4096;
const WARMUP: usize = 50;

#[inline]
fn per_iter_ms(start: Instant, iters: usize) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn rand_tensor(rows: usize, cols: usize) -> Tensor<f32> {
    use std::cell::Cell;
    thread_local! { static SEED: Cell<u64> = const { Cell::new(42) }; }
    Tensor::from_fn(rows, cols, |_, _| {
        SEED.with(|s| {
            let x = s.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s.set(x);
            (x >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        })
    })
}

fn main() {
    println!("=== Dispatch Overhead Benchmark ===");
    println!("Matrix: {N}x{N} f32");
    println!();

    let a = rand_tensor(N, N);
    let b = rand_tensor(N, N);

    // ── Test 1: Single op dispatch overhead ──────────────────────────
    // Launch many ops WITHOUT sync between them to measure pure CPU dispatch cost
    println!("--- Test 1: Burst dispatch (no inter-op sync) ---");
    println!("  Measures CPU-side dispatch latency per kernel launch");
    println!();

    for &(name, burst_count) in &[
        ("exp x100",    100usize),
        ("exp x1000",   1000),
        ("add x100",    100),
        ("add x1000",   1000),
    ] {
        // Warmup
        for _ in 0..WARMUP {
            let _ = a.exp();
        }
        gpu_sync();

        let start = Instant::now();
        let mut ring = vec![a.clone(); 4];
        if name.starts_with("exp") {
            for i in 0..burst_count {
                ring[i % 4] = a.exp();
            }
        } else {
            for i in 0..burst_count {
                ring[i % 4] = &a + &b;
            }
        }
        gpu_sync();
        let elapsed = start.elapsed();
        let per_op_ns = elapsed.as_nanos() as f64 / burst_count as f64;
        println!("  {:<20} {:>8.0} ns/op  ({:.3} ms total)", name, per_op_ns, elapsed.as_secs_f64() * 1000.0);
        drop(ring);
    }

    // ── Test 2: Fused vs unfused dispatch ────────────────────────────
    println!();
    println!("--- Test 2: Fused vs unfused (4 ops) ---");
    println!("  Unfused = 4 separate kernel launches");
    println!("  Fused   = 1 kernel launch via fuse!");
    println!();

    let iters = 1000;

    // Warmup
    for _ in 0..WARMUP {
        let _ = a.exp().sin().cos().tanh();
    }
    gpu_sync();

    // Unfused: 4 separate launches
    let start = Instant::now();
    let mut ring = vec![a.clone(); 4];
    for i in 0..iters {
        ring[i % 4] = a.exp().sin().cos().tanh();
    }
    gpu_sync();
    let unfused_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Warmup fused
    for _ in 0..WARMUP {
        let _: Tensor<f32> = fuse!(a.exp().sin().cos().tanh(); a);
    }
    gpu_sync();

    // Fused: 1 launch
    let start = Instant::now();
    for i in 0..iters {
        ring[i % 4] = fuse!(a.exp().sin().cos().tanh(); a);
    }
    gpu_sync();
    let fused_ms = start.elapsed().as_secs_f64() * 1000.0;
    drop(ring);

    println!("  Unfused (4 launches)  {:>8.3} ms / {iters} iters  ({:.1} ns/iter)", unfused_ms, unfused_ms / iters as f64 * 1e6);
    println!("  Fused   (1 launch)    {:>8.3} ms / {iters} iters  ({:.1} ns/iter)", fused_ms, fused_ms / iters as f64 * 1e6);
    println!("  Speedup: {:.2}x", unfused_ms / fused_ms);

    // ── Test 3: Simulated training step dispatch ─────────────────────
    println!();
    println!("--- Test 3: Simulated training step ---");
    println!("  12 layers x (emul + exp + add) = 36 kernel launches per step");
    println!();

    let dim = 512;
    let weights: Vec<Tensor<f32>> = (0..12).map(|_| rand_tensor(dim, dim)).collect();
    let x = rand_tensor(dim, dim);

    // Warmup
    for _ in 0..WARMUP {
        let mut h = x.clone();
        for w in &weights {
            h = h.emul(w).exp();
            h = &h + &x;
        }
        gpu_sync();
    }

    let steps = 500;

    // Eager dispatch
    let start = Instant::now();
    for _ in 0..steps {
        let mut h = x.clone();
        for w in &weights {
            h = h.emul(w).exp();
            h = &h + &x;
        }
    }
    gpu_sync();
    let eager_ms = start.elapsed().as_secs_f64() * 1000.0;

    // CUDA Graph
    #[cfg(feature = "cuda")]
    {
        use nabla::{TrainingGraph, cuda_synchronize};

        let mut tg = TrainingGraph::with_warmup(3);
        // Warmup + capture: try the first step to detect capture errors
        let mut step_fn = || {
            let mut h = x.clone();
            for w in &weights {
                h = h.emul(w).exp();
                h = &h + &x;
            }
        };
        match tg.step(&mut step_fn) {
            Err(_) => {
                println!("  Eager dispatch:    {:>8.3} ms / {steps} steps  ({:.1} us/step)", eager_ms, eager_ms / steps as f64 * 1000.0);
                println!("  CUDA Graph: skipped (stream capture unsupported — cuMemAlloc during capture)");
            }
            Ok(_) => {
                // Continue remaining warmup iterations
                for _ in 1..5 {
                    let mut step_fn2 = || {
                        let mut h = x.clone();
                        for w in &weights {
                            h = h.emul(w).exp();
                            h = &h + &x;
                        }
                    };
                    if tg.step(&mut step_fn2).is_err() {
                        break;
                    }
                }
                cuda_synchronize();

                // Measure replay
                let start = Instant::now();
                let mut noop = || {};
                let mut replay_ok = true;
                for _ in 0..steps {
                    if tg.step(&mut noop).is_err() {
                        replay_ok = false;
                        break;
                    }
                }
                cuda_synchronize();
                let graph_ms = start.elapsed().as_secs_f64() * 1000.0;

                println!("  Eager dispatch:    {:>8.3} ms / {steps} steps  ({:.1} us/step)", eager_ms, eager_ms / steps as f64 * 1000.0);
                if replay_ok {
                    println!("  CUDA Graph replay: {:>8.3} ms / {steps} steps  ({:.1} us/step)", graph_ms, graph_ms / steps as f64 * 1000.0);
                    println!("  Speedup: {:.2}x", eager_ms / graph_ms);
                } else {
                    println!("  CUDA Graph: skipped (stream capture unsupported — cuMemAlloc during capture)");
                }
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("  Eager dispatch:    {:>8.3} ms / {steps} steps  ({:.1} us/step)", eager_ms, eager_ms / steps as f64 * 1000.0);
        println!("  (CUDA Graph not available on CPU backend)");
    }

    // ── Test 4: Full bench_gpu comparison (existing benchmarks) ──────
    println!();
    println!("--- Test 4: Core ops (4096x4096 f32) ---");
    println!("  Compare with spec.md W20 baseline numbers");
    println!();

    let iters = 100;

    for &(name, is_unary) in &[
        ("exp", true),
        ("sin", true),
        ("tanh", true),
        ("add", false),
        ("emul", false),
    ] {
        // Warmup
        let mut ring = if is_unary {
            (0..20).map(|_| a.exp()).collect::<Vec<_>>()
        } else {
            (0..20).map(|_| &a + &b).collect::<Vec<_>>()
        };
        gpu_sync();

        let start = Instant::now();
        for i in 0..iters {
            ring[i % 20] = if is_unary {
                match name {
                    "exp" => a.exp(),
                    "sin" => a.sin(),
                    "tanh" => a.tanh(),
                    _ => a.exp(),
                }
            } else {
                match name {
                    "add" => &a + &b,
                    "emul" => a.emul(&b),
                    _ => &a + &b,
                }
            };
        }
        gpu_sync();
        let ms = per_iter_ms(start, iters);
        println!("  {:<12} {:.3} ms/iter", name, ms);
        drop(ring);
    }

    // Matmul
    let m1 = rand_tensor(N, N);
    let m2 = rand_tensor(N, N);
    let mut ring = (0..20).map(|_| &m1 * &m2).collect::<Vec<_>>();
    gpu_sync();
    let start = Instant::now();
    for i in 0..iters {
        ring[i % 20] = &m1 * &m2;
    }
    gpu_sync();
    let ms = per_iter_ms(start, iters);
    println!("  {:<12} {:.3} ms/iter", "matmul 4096", ms);
    drop(ring);

    // Fused
    let mut ring: Vec<Tensor<f32>> = (0..20).map(|_| fuse!(a.exp().sin(); a)).collect();

    gpu_sync();
    let start = Instant::now();
    for i in 0..iters {
        ring[i % 20] = fuse!(a.exp().sin(); a);
    }
    gpu_sync();
    let ms = per_iter_ms(start, iters);
    println!("  {:<12} {:.3} ms/iter", "fuse exp+sin", ms);
    drop(ring);

    // Reductions
    for _ in 0..20 { let _ = a.sum_all(); }
    gpu_sync();
    let start = Instant::now();
    for _ in 0..iters { let _ = a.sum_all(); }
    gpu_sync();
    let ms = per_iter_ms(start, iters);
    println!("  {:<12} {:.3} ms/iter", "sum_all", ms);

    for _ in 0..20 { let _ = a.max_all(); }
    gpu_sync();
    let start = Instant::now();
    for _ in 0..iters { let _ = a.max_all(); }
    gpu_sync();
    let ms = per_iter_ms(start, iters);
    println!("  {:<12} {:.3} ms/iter", "max_all", ms);

    println!();
    println!("=== Benchmark complete ===");
}
