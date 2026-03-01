//! # SVD Compression -- one factorization, many uses
//! Run: cargo run --example 03_svd_compress --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let a = mat![f64: 1.0, 2.0, 0.0; 0.0, 1.0, 1.0; 1.0, 0.0, 1.0; 0.0, 1.0, 0.0];

    let svd = a.svd()?;
    let s = svd.s();
    println!("Singular values: {:.4}, {:.4}, {:.4}", s[0], s[1], s[2]);
    let diff_norm = |m: &Tensor<f64, Cpu>| (&a - m).norm();

    let recon = svd.reconstruct_rank(s.len());
    let err = diff_norm(&recon);
    println!("Reconstruction error: {err:.2e}");

    let approx = svd.reconstruct_rank(2);
    let approx_err = diff_norm(&approx);
    println!(
        "Rank-2 approx error: {approx_err:.4} (dropped singular value: {:.4})",
        s[2]
    );
}
