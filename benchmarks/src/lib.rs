//! Shared utilities for nabla benchmarks.

use nabla::prelude::*;
use std::cell::Cell;

/// Deterministic pseudo-random tensor (LCG seeded at 42).
pub fn rand_tensor(rows: usize, cols: usize) -> Tensor<f32> {
    thread_local! { static SEED: Cell<u64> = const { Cell::new(42) }; }
    let n = rows * cols;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        SEED.with(|s| {
            let x = s
                .get()
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s.set(x);
            data.push((x >> 33) as f32 / (1u64 << 31) as f32 - 1.0);
        });
    }
    Tensor::from_vec(data, rows, cols)
}

/// Kaiming-initialized random tensor: rand * sqrt(2/fan_in).
pub fn kaiming(rows: usize, cols: usize) -> Tensor<f32> {
    rand_tensor(rows, cols).map(|x| x * (2.0 / rows as f32).sqrt())
}

/// GPU stream synchronization barrier (no-op without cuda feature).
#[cfg(feature = "cuda")]
#[inline]
pub fn gpu_sync() {
    nabla::cuda_synchronize();
}

#[cfg(not(feature = "cuda"))]
#[inline]
pub fn gpu_sync() {}

/// Unwrap Result, panicking with the error message on failure.
#[macro_export]
macro_rules! must {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        }
    };
}

/// Time `iters` iterations of `f` (after discarding `warmup` rounds).
/// Returns milliseconds per iteration.
pub fn bench_ms(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    gpu_sync();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    gpu_sync();
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}
