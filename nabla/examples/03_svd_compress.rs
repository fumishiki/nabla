//! # SVD Compression — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~12 | Julia LOC: ~10
//! Julia advantage: `svd(A)` is one line
//! nabla advantage: explicit factorization types, type-safe access
//!
//! Run: cargo run --example 03_svd_compress --features cpu

use nabla::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 4x3 matrix — SVD decomposition
    let a = mat![
        [1.0_f64, 2.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ];

    let svd = a.svd()?;
    let s = svd.s();
    println!("Singular values: {:.4}, {:.4}, {:.4}", s[0], s[1], s[2]);

    // Full reconstruction
    let recon = svd.reconstruct_rank(s.len());
    let err = (&a - &recon).norm();
    println!("Reconstruction error: {err:.2e}");

    // Rank-2 approximation (drop smallest singular value)
    let approx = svd.reconstruct_rank(2);
    let approx_err = (&a - &approx).norm();
    println!("Rank-2 approx error: {approx_err:.4} (dropped singular value: {:.4})", s[2]);

    Ok(())
}
