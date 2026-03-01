//! # SVD Compression — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~12 | Julia LOC: ~10
//! Julia advantage: `svd(A)` is one line
//! nabla advantage: explicit factorization types, type-safe access
//!
//! Run: cargo run --example 03_svd_compress --features cpu

#[cfg(feature = "cpu")]
use nabla::prelude::*;

#[cfg(feature = "cpu")]
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
    let diff_norm = |m: &Tensor<f64>| (&a - m).norm();

    // Full reconstruction
    let recon = svd.reconstruct_rank(s.len());
    let err = diff_norm(&recon);
    println!("Reconstruction error: {err:.2e}");

    // Rank-2 approximation (drop smallest singular value)
    let approx = svd.reconstruct_rank(2);
    let approx_err = diff_norm(&approx);
    println!(
        "Rank-2 approx error: {approx_err:.4} (dropped singular value: {:.4})",
        s[2]
    );

    Ok(())
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("example 03_svd_compress requires --features cpu");
}
