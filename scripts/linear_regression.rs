#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "../nabla", features = ["cpu"] }
//! ```

use nabla::prelude::*;

fn main() -> nabla::error::Result<()> {
    // y = W*x + b, W_true=[1.5, -2.0], b_true=0.5
    let n = 50;
    let w_true: Tensor<f64> = mat![[1.5_f64, -2.0]];
    let b_true = 0.5_f64;

    // Deterministic pseudo-data via simple LCG
    let mut seed: u64 = 42;
    let mut rng = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    };

    let x: Tensor<f64> = Tensor::from_fn(n, 2, |_, _| rng());
    let y_true: Tensor<f64> = Tensor::from_fn(n, 1, |i, _| {
        w_true.get(0, 0) * x.get(i, 0) + w_true.get(0, 1) * x.get(i, 1) + b_true
    });

    // GD via gradient_prep
    let mut w: Tensor<f64> = Tensor::zeros(1, 2);
    let mut b_val = 0.0_f64;
    let lr = 0.01;

    for step in 0..100 {
        // Forward: pred = X @ W^T + b
        let pred: Tensor<f64> = Tensor::from_fn(n, 1, |i, _| {
            w.get(0, 0) * x.get(i, 0) + w.get(0, 1) * x.get(i, 1) + b_val
        });
        let residual = &pred - &y_true;
        let mse = residual.emul(&residual).sum_all() / (n as f64);

        if step % 20 == 0 {
            println!("step {step:3}: MSE = {mse:.6}");
        }

        // Manual gradient: dL/dW = 2/n * residual^T @ X, dL/db = 2/n * sum(residual)
        let scale = 2.0 / (n as f64);
        let dw0 = scale * (0..n).map(|i| residual.get(i, 0) * x.get(i, 0)).sum::<f64>();
        let dw1 = scale * (0..n).map(|i| residual.get(i, 0) * x.get(i, 1)).sum::<f64>();
        let db = scale * (0..n).map(|i| residual.get(i, 0)).sum::<f64>();

        w.set(0, 0, w.get(0, 0) - lr * dw0);
        w.set(0, 1, w.get(0, 1) - lr * dw1);
        b_val -= lr * db;
    }

    println!("\nLearned W = [{:.4}, {:.4}]", w.get(0, 0), w.get(0, 1));
    println!("Learned b = {b_val:.4}");
    println!("True    W = [1.5000, -2.0000], b = 0.5000");

    Ok(())
}
