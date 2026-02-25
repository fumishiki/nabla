#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "..", features = ["cpu"] }
//! ```

use nabla::prelude::*;

fn main() -> nabla::Result<()> {
    // 4x4 matrix with rank-2 structure + noise
    let a = mat![
        [1.0_f64, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [3.0, 6.0, 9.1, 12.0],
        [4.0, 8.0, 12.0, 16.1]
    ];

    let svd = a.factorize_svd()?;
    let s = svd.s();
    println!("Singular values: {s:.4?}");

    // Rank-2 approximation: U[:,:2] * diag(s[:2]) * Vt[:2,:]
    let u = svd.u();
    let vt = svd.vt();
    let (m, n) = a.shape();
    let rank = 2;
    let approx = Tensor::from_fn(m, n, |i, j| {
        (0..rank).map(|k| u.get(i, k) * s[k] * vt.get(k, j)).sum::<f64>()
    });

    // Frobenius error
    let diff = &a - &approx;
    let frob = diff.emul(&diff).sum_all().sqrt();
    println!("Rank-{rank} approximation error (Frobenius): {frob:.6}");

    Ok(())
}
