//! Dispatch Scaling Benchmark — dispatch overhead vs tensor size & op count.
//! Run: cargo run --release --bin bench_dispatch_scaling --features cuda

use nabla::prelude::*;
use nabla_bench::{rand_tensor, gpu_sync, kaiming, bench_ms};
use nabla_train::prelude::{Optimizer, Sgd};
use std::time::Instant;

const WARMUP: usize = 20;
const SIZES: [usize; 8] = [32, 64, 128, 256, 512, 1024, 2048, 4096];
const CHAIN_OPS: usize = 1000;

type B = DefaultBackend;

fn exp1_op_chain() {
    eprintln!("\n=== Exp1: Small tensor x {CHAIN_OPS} chained exp ops ===");
    eprintln!("{:<8} {:>12} {:>14} {:>10}", "size", "total_ms", "ops/sec", "ns/op");
    eprintln!("{}", "-".repeat(48));

    for &sz in &SIZES {
        let a = rand_tensor(sz, sz);
        for _ in 0..WARMUP { let _ = a.exp(); }
        gpu_sync();
        let mut ring = vec![a.clone(); 4];
        let start = Instant::now();
        for i in 0..CHAIN_OPS { ring[i % 4] = ring[(i + 3) % 4].exp(); }
        gpu_sync();
        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let ops_per_sec = CHAIN_OPS as f64 / elapsed.as_secs_f64();
        let ns_per_op = elapsed.as_nanos() as f64 / CHAIN_OPS as f64;
        println!("{{\"test\":\"exp1_chain\",\"size\":{sz},\"ops\":{CHAIN_OPS},\"total_ms\":{total_ms:.3},\"ops_per_sec\":{ops_per_sec:.0},\"ns_per_op\":{ns_per_op:.0}}}");
        eprintln!("{sz:<8} {total_ms:>12.3} {ops_per_sec:>14.0} {ns_per_op:>10.0}");
        drop(ring);
    }
}

fn exp2_mlp_forward() {
    let w1 = kaiming(784, 256);
    let w2 = kaiming(256, 128);
    let w3 = kaiming(128, 10);

    eprintln!("\n=== Exp2: MLP forward (784→256→128→10) ===");
    eprintln!("{:<12} {:>10} {:>12}", "batch", "ms/fwd", "fwd/sec");
    eprintln!("{}", "-".repeat(36));

    for &(batch, iters) in &[(1usize, 500usize), (32, 100), (128, 50), (1024, 20)] {
        let x = rand_tensor(batch, 784);
        let fwd = |x: &Tensor<f32>| {
            let h1 = (x * &w1).leaky_relu(0.01);
            let h2 = (&h1 * &w2).leaky_relu(0.01);
            &h2 * &w3
        };
        for _ in 0..10 { let _ = fwd(&x); }
        gpu_sync();
        let mut ring = vec![fwd(&x); 4];
        let start = Instant::now();
        for i in 0..iters { ring[i % 4] = fwd(&x); }
        gpu_sync();
        let elapsed = start.elapsed();
        drop(ring);
        let ms_per_fwd = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let fwd_per_sec = iters as f64 / elapsed.as_secs_f64();
        println!("{{\"test\":\"exp2_forward\",\"batch\":{batch},\"ms_per_fwd\":{ms_per_fwd:.4},\"fwd_per_sec\":{fwd_per_sec:.0},\"iters\":{iters}}}");
        eprintln!("{batch:<12} {ms_per_fwd:>10.4} {fwd_per_sec:>12.0}");
    }
}

fn exp3_train_step() {
    eprintln!("\n=== Exp3: Training step (fwd+bwd+sgd) ===");
    eprintln!("{:<12} {:>12} {:>12}", "batch", "ms/step", "steps/sec");
    eprintln!("{}", "-".repeat(38));

    for &(batch, steps) in &[(1usize, 100usize), (32, 30), (128, 10)] {
        let mut w1 = kaiming(784, 256);
        let mut w2 = kaiming(256, 128);
        let mut w3 = kaiming(128, 10);
        let params = vec![&w1, &w2, &w3];
        let mut sgd: Sgd<f32, B> = Sgd::from_params(0.01, &params);
        drop(params);
        let x = rand_tensor(batch, 784);
        let target = rand_tensor(batch, 10);

        let mut do_step = |w1: &mut Tensor<f32>, w2: &mut Tensor<f32>, w3: &mut Tensor<f32>| {
            let tape = Tape::new();
            if let (Ok(x_var), Ok(t_var)) = (tape.variable(x.clone()), tape.variable(target.clone())) {
                let w1v = tape.variable(w1.clone()).ok();
                let w2v = tape.variable(w2.clone()).ok();
                let w3v = tape.variable(w3.clone()).ok();
                if let (Some(w1v), Some(w2v), Some(w3v)) = (w1v, w2v, w3v) {
                    let h1 = x_var.matmul(&w1v).leaky_relu(0.01);
                    let h2 = h1.matmul(&w2v).leaky_relu(0.01);
                    let out = h2.matmul(&w3v);
                    let diff = out.sub_var(&t_var);
                    let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
                    if loss.backward().is_ok() {
                        let g0 = w1v.grad_ref().ok();
                        let g1 = w2v.grad_ref().ok();
                        let g2 = w3v.grad_ref().ok();
                        if let (Some(r0), Some(r1), Some(r2)) = (&g0, &g1, &g2) {
                            sgd.step(&mut vec![w1, w2, w3], &[&**r0, &**r1, &**r2]);
                        }
                    }
                }
            }
        };

        for _ in 0..3 { do_step(&mut w1, &mut w2, &mut w3); }
        gpu_sync();

        let ms = bench_ms(0, steps, || do_step(&mut w1, &mut w2, &mut w3));
        let steps_per_sec = 1000.0 / ms;
        println!("{{\"test\":\"exp3_train\",\"batch\":{batch},\"ms_per_step\":{ms:.4},\"steps_per_sec\":{steps_per_sec:.0},\"steps\":{steps}}}");
        eprintln!("{batch:<12} {ms:>12.4} {steps_per_sec:>12.0}");
    }
}

fn main() {
    exp1_op_chain();
    exp2_mlp_forward();
    exp3_train_step();
}
