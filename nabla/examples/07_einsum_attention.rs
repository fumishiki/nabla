//! # Einsum Attention — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~25 | Julia LOC: ~25 (Tullio.jl or manual)
//! Julia advantage: broadcast `.` syntax for softmax
//! nabla advantage: `einsum!` compiles to optimal matmul, zero overhead
//!
//! Run: cargo run --example 07_einsum_attention --features cpu

use nabla::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Scaled dot-product attention: softmax(Q*K^T / sqrt(d)) * V
    let d = 4; // head dimension
    let seq = 3; // sequence length

    // Q, K, V matrices (seq x d)
    let q = Tensor::<f64>::from_fn(seq, d, |i, j| ((i * d + j) as f64) * 0.1);
    let k = Tensor::<f64>::from_fn(seq, d, |i, j| ((i * d + j + 1) as f64) * 0.1);
    let v = Tensor::<f64>::from_fn(seq, d, |i, j| ((i * d + j) as f64) * 0.2);

    // scores = Q * K^T / sqrt(d)
    let scale = 1.0 / (d as f64).sqrt();
    let scores: Tensor<f64> = einsum!(s[i,j] = q[i,k] * k[j,k]); // Q * K^T
    let scores = &scores * scale;

    // Row-wise softmax
    let exp_scores = scores.exp();
    let row_sums: Vec<f64> = (0..seq)
        .map(|i| (0..seq).map(|c| exp_scores.get(i, c)).sum())
        .collect();
    let softmax = Tensor::from_fn(seq, seq, |i, j| exp_scores.get(i, j) / row_sums[i]);

    // Attention output = softmax * V
    let out: Tensor<f64> = einsum!(o[i,j] = softmax[i,k] * v[k,j]);

    println!("Attention weights (softmax):");
    for i in 0..seq {
        let row: Vec<String> = (0..seq)
            .map(|j| format!("{:.4}", softmax.get(i, j)))
            .collect();
        println!("  [{}]", row.join(", "));
    }
    println!("Output shape: {:?}", out.shape());

    Ok(())
}
