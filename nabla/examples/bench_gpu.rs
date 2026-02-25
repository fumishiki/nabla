//! GPU Benchmark: nabla CUDA vs PyTorch comparison
//!
//! Run: cargo run --release --example bench_gpu --features cuda
//! Compare with: python3 scripts/bench_pytorch.py

use nabla::prelude::*;
use std::time::Instant;

const N: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 100;

fn bench<F: FnMut()>(name: &str, mut f: F) {
    for _ in 0..WARMUP { f(); }
    let start = Instant::now();
    for _ in 0..ITERS { f(); }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / ITERS as f64;
    let per_iter_ms = per_iter_us / 1000.0;
    let bytes = N * N * 4; // f32 = 4 bytes
    let gbps = (2.0 * bytes as f64) / (per_iter_us * 1e-6) / 1e9;
    println!("{:<25} {:>8.3} ms  {:>8.1} GB/s", name, per_iter_ms, gbps);
}

fn rand_tensor(rows: usize, cols: usize) -> Tensor<f32> {
    use std::cell::Cell;
    // Simple LCG for reproducible pseudo-random data
    thread_local! { static SEED: Cell<u64> = Cell::new(42); }
    Tensor::from_fn(rows, cols, |_, _| {
        SEED.with(|s| {
            let x = s.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s.set(x);
            (x >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        })
    })
}

fn main() {
    println!("nabla GPU Benchmark — {}×{} f32", N, N);
    println!("Warmup: {WARMUP}, Iterations: {ITERS}");
    println!("{}", "=".repeat(50));

    let a = rand_tensor(N, N);
    let b = rand_tensor(N, N);

    // --- Element-wise unary ops ---
    println!("\n--- Element-wise Unary ---");
    bench("exp", || { let _ = a.exp(); });
    bench("sin", || { let _ = a.sin(); });
    bench("cos", || { let _ = a.cos(); });
    bench("tanh", || { let _ = a.tanh(); });
    bench("sqrt(abs)", || { let _ = a.abs().sqrt(); });
    bench("ln(abs+1)", || { let _ = a.abs().log1p(); });
    bench("neg", || { let _ = -&a; });
    bench("abs", || { let _ = a.abs(); });

    // --- Element-wise binary ops ---
    println!("\n--- Element-wise Binary ---");
    bench("add", || { let _ = &a + &b; });
    bench("sub", || { let _ = &a - &b; });
    bench("emul", || { let _ = a.emul(&b); });

    // --- Fused ops ---
    println!("\n--- Fused (single kernel) ---");
    bench("fuse exp+sin", || {
        let _: Tensor<f32> = fuse!(a.exp().sin(); a);
    });
    bench("fuse 4-op", || {
        let _: Tensor<f32> = fuse!(a.exp().sin().cos().tanh(); a);
    });

    // --- Reductions ---
    println!("\n--- Reductions ---");
    bench("sum_all", || { let _ = a.sum_all(); });
    bench("max_all", || { let _ = a.max_all(); });

    // --- MatMul ---
    println!("\n--- MatMul ---");
    let m1 = rand_tensor(1024, 1024);
    let m2 = rand_tensor(1024, 1024);
    bench("matmul 1024", || { let _ = &m1 * &m2; });

    let m3 = rand_tensor(N, N);
    let m4 = rand_tensor(N, N);
    bench("matmul 4096", || { let _ = &m3 * &m4; });

    println!("\n{}", "=".repeat(50));
    println!("Done.");
}
