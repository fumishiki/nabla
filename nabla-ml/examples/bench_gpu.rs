//! GPU Benchmark
//! Run: cargo run --release --example bench_gpu --features cuda

use nabla::prelude::*;
use std::time::Instant;

#[cfg(any(feature = "cpu", feature = "cuda"))]
use half::f16;

#[cfg(feature = "cuda")]
#[inline]
fn gpu_sync() {
    nabla::cuda_synchronize();
}
#[cfg(not(feature = "cuda"))]
#[inline]
fn gpu_sync() {}

const N: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 100;

fn bench<F: FnMut() -> Tensor<f32>>(name: &str, mut f: F) {
    let mut ring: Vec<Tensor<f32>> = (0..WARMUP).map(|_| f()).collect();
    gpu_sync();
    let start = Instant::now();
    for i in 0..ITERS {
        ring[i % WARMUP] = f();
    }
    gpu_sync();
    let elapsed = start.elapsed();
    drop(ring);
    let per_iter_us = elapsed.as_micros() as f64 / ITERS as f64;
    let per_iter_ms = per_iter_us / 1000.0;
    let bytes = N * N * 4;
    let gbps = (2.0 * bytes as f64) / (per_iter_us * 1e-6) / 1e9;
    println!("{:<25} {:>8.3} ms  {:>8.1} GB/s", name, per_iter_ms, gbps);
}

fn bench_scalar<F: FnMut() -> f32>(name: &str, mut f: F) {
    for _ in 0..WARMUP {
        let _ = f();
    }
    gpu_sync();
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = f();
    }
    gpu_sync();
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / ITERS as f64;
    let per_iter_ms = per_iter_us / 1000.0;
    let bytes = N * N * 4;
    let gbps = (bytes as f64) / (per_iter_us * 1e-6) / 1e9;
    println!("{:<25} {:>8.3} ms  {:>8.1} GB/s", name, per_iter_ms, gbps);
}

fn rand_tensor(rows: usize, cols: usize) -> Tensor<f32> {
    use std::cell::Cell;
    thread_local! { static SEED: Cell<u64> = const { Cell::new(42) }; }
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        let v = SEED.with(|s| {
            let x = s
                .get()
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(x);
            (x >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        });
        data.push(v);
    }
    Tensor::from_vec(data, rows, cols)
}

fn main() {
    println!("nabla GPU Benchmark — {}×{} f32", N, N);
    println!("Warmup: {WARMUP}, Iterations: {ITERS}");
    println!("{}", "=".repeat(50));

    let a = rand_tensor(N, N);
    let b = rand_tensor(N, N);

    println!("\n--- Element-wise Unary ---");
    bench("exp", || a.exp());
    bench("sin", || a.sin());
    bench("cos", || a.cos());
    bench("tanh", || a.tanh());
    bench("sqrt(abs)", || a.abs().sqrt());
    bench("ln(abs+1)", || a.abs().log1p());
    bench("neg", || -&a);
    bench("abs", || a.abs());

    println!("\n--- Element-wise Binary ---");
    bench("add", || &a + &b);
    bench("sub", || &a - &b);
    bench("emul", || a.emul(&b));

    println!("\n--- Fused (single kernel) ---");
    bench("fuse exp+sin", || fuse!(a.exp().sin(); a));
    bench("fuse 4-op", || fuse!(a.exp().sin().cos().tanh(); a));

    println!("\n--- Mega-fused (multi-output, 1 kernel) ---");
    bench("2x fuse! (baseline)", || {
        let _ = fuse!(a.exp().sin(); a) as Tensor<f32>;
        fuse!(b.tanh(); b)
    });
    bench("mega_fuse 2-out", || {
        let _r = mega_fuse!(
            a.exp().sin();
            b.tanh();
            inputs: a, b
        );
        match _r.into_iter().last() {
            Some(last) => last,
            None => panic!("mega_fuse returned no outputs"),
        }
    });
    bench("4x fuse! (baseline)", || {
        let _ = fuse!(a.exp(); a) as Tensor<f32>;
        let _ = fuse!(b.sin(); b) as Tensor<f32>;
        let _ = fuse!(a.cos(); a) as Tensor<f32>;
        fuse!(b.tanh(); b)
    });
    bench("mega_fuse 4-out", || {
        let _r = mega_fuse!(
            a.exp();
            b.sin();
            a.cos();
            b.tanh();
            inputs: a, b
        );
        match _r.into_iter().last() {
            Some(last) => last,
            None => panic!("mega_fuse returned no outputs"),
        }
    });

    println!("\n--- Reductions ---");
    bench_scalar("sum_all", || a.sum_all());
    bench_scalar("max_all", || a.max_all());

    println!("\n--- MatMul ---");
    let m1 = rand_tensor(1024, 1024);
    let m2 = rand_tensor(1024, 1024);
    bench("matmul f32 1024", || &m1 * &m2);

    let m3 = rand_tensor(2048, 2048);
    let m4 = rand_tensor(2048, 2048);
    bench("matmul f32 2048", || &m3 * &m4);

    let m5 = rand_tensor(4096, 4096);
    let m6 = rand_tensor(4096, 4096);
    bench("matmul f32 4096", || &m5 * &m6);

    // FP16 matmul via Tensor Cores (CUBLAS_COMPUTE_16F)
    #[cfg(any(feature = "cpu", feature = "cuda"))]
    {
        let h1: Tensor<f16> = Tensor::fill(4096, 4096, f16::from_f32(0.01));
        let h2: Tensor<f16> = Tensor::fill(4096, 4096, f16::from_f32(0.01));
        let mut ring_f16: Vec<Tensor<f16>> = (0..WARMUP).map(|_| &h1 * &h2).collect();
        gpu_sync();
        let start_f16 = Instant::now();
        for i in 0..ITERS {
            ring_f16[i % WARMUP] = &h1 * &h2;
        }
        gpu_sync();
        let ms_f16 = start_f16.elapsed().as_micros() as f64 / ITERS as f64 / 1000.0;
        println!("{:<25} {:>8.3} ms", "matmul f16 4096", ms_f16);
    }

    println!("\n{}", "=".repeat(50));
    println!("Done.");
}
