//! # Least Squares — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~12 | Julia LOC: ~10
//! Julia: `A \ b` | nabla: `a.solve_lstsq(&b)?` — QR-based, numerically stable
//!
//! Run: cargo run --example 02_least_squares --features cpu

use nabla::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Overdetermined system: 4 data points, fit y = a + b*x
    // Design matrix [1, x_i]
    let a = mat![[1.0_f64, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let b = mat![[1.1_f64], [2.0], [2.9], [4.1]]; // noisy y = 1 + x

    // QR-based least squares (equivalent to Julia's A \ b)
    let x = a.solve_lstsq(&b)?;
    println!("Intercept: {:.4}, Slope: {:.4}", x.get(0, 0), x.get(1, 0));

    // Residual: r = b - A*x
    let residual = &b - &(&a * &x);
    let norm_sq = residual.emul(&residual).sum_all();
    println!("Residual norm^2: {:.6}", norm_sq);

    Ok(())
}
