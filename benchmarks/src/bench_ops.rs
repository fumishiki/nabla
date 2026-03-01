//! GPU Operations Benchmark — nabla side of nabla vs PyTorch comparison.
//! Run: cargo run --release --bin bench_ops --features cuda

use nabla::prelude::*;
use nabla_bench::{rand_tensor, gpu_sync};
use std::time::Instant;

const N: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 100;

struct BenchResult { op: &'static str, ms: f64, gbps: f64 }

fn bench<F: FnMut() -> Tensor<f32>>(op: &'static str, read_bytes: usize, write_bytes: usize, mut f: F) -> BenchResult {
    let mut ring: Vec<Tensor<f32>> = (0..WARMUP).map(|_| f()).collect();
    gpu_sync();
    let start = Instant::now();
    for i in 0..ITERS { ring[i % WARMUP] = f(); }
    gpu_sync();
    let elapsed = start.elapsed();
    drop(ring);
    let per_iter_us = elapsed.as_micros() as f64 / ITERS as f64;
    let ms = per_iter_us / 1000.0;
    let gbps = (read_bytes + write_bytes) as f64 / (per_iter_us * 1e-6) / 1e9;
    println!("{{\"op\":\"{op}\",\"ms\":{ms:.4},\"gbps\":{gbps:.1}}}");
    BenchResult { op, ms, gbps }
}

fn bench_scalar<F: FnMut() -> f32>(op: &'static str, read_bytes: usize, mut f: F) -> BenchResult {
    for _ in 0..WARMUP { let _ = f(); }
    gpu_sync();
    let start = Instant::now();
    for _ in 0..ITERS { let _ = f(); }
    gpu_sync();
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / ITERS as f64;
    let ms = per_iter_us / 1000.0;
    let gbps = read_bytes as f64 / (per_iter_us * 1e-6) / 1e9;
    println!("{{\"op\":\"{op}\",\"ms\":{ms:.4},\"gbps\":{gbps:.1}}}");
    BenchResult { op, ms, gbps }
}

fn main() {
    let n_bytes = N * N * 4;
    let n1k_bytes = 1024 * 1024 * 4;
    let a = rand_tensor(N, N);
    let b = rand_tensor(N, N);
    let m1 = rand_tensor(1024, 1024);
    let m2 = rand_tensor(1024, 1024);
    let mut results: Vec<BenchResult> = Vec::with_capacity(16);

    results.push(bench("exp", n_bytes, n_bytes, || a.exp()));
    results.push(bench("sin", n_bytes, n_bytes, || a.sin()));
    results.push(bench("cos", n_bytes, n_bytes, || a.cos()));
    results.push(bench("tanh", n_bytes, n_bytes, || a.tanh()));
    results.push(bench("add", 2 * n_bytes, n_bytes, || &a + &b));
    results.push(bench("sub", 2 * n_bytes, n_bytes, || &a - &b));
    results.push(bench("emul", 2 * n_bytes, n_bytes, || a.emul(&b)));
    results.push(bench("fuse_exp_sin", n_bytes, n_bytes, || fuse!(a.exp().sin(); a)));
    results.push(bench_scalar("sum_all", n_bytes, || a.sum_all()));
    results.push(bench_scalar("max_all", n_bytes, || a.max_all()));
    results.push(bench("matmul_1024", 2 * n1k_bytes, n1k_bytes, || &m1 * &m2));
    results.push(bench("matmul_4096", 2 * n_bytes, n_bytes, || &a * &b));

    eprintln!();
    eprintln!("{:<20} {:>10} {:>10}", "Operation", "ms", "GB/s");
    eprintln!("{}", "-".repeat(42));
    for r in &results { eprintln!("{:<20} {:>10.4} {:>10.1}", r.op, r.ms, r.gbps); }
}
