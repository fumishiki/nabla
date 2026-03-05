//! # Einsum Attention -- DSL for attention
//! Run: cargo run --example 07_einsum_attention --features cpu

use nabla::prelude::*;

#[cfg(feature = "cpu")]
#[nabla::main(cpu)]
fn main() {
    let d = 4;
    let seq = 3;

    let q = Tensor::<f64>::from_fn(seq, d, |i, j| ((i * d + j) as f64) * 0.1);
    let k = Tensor::<f64>::from_fn(seq, d, |i, j| ((i * d + j + 1) as f64) * 0.1);
    let v = Tensor::<f64>::from_fn(seq, d, |i, j| ((i * d + j) as f64) * 0.2);

    let scale = 1.0 / (d as f64).sqrt();
    let scores: Tensor<f64> = einsum!(s[i,j] = q[i,k] * k[j,k]);
    let scores = &scores * scale;

    let softmax = scores.softmax(1);

    let out: Tensor<f64> = einsum!(o[i,j] = softmax[i,k] * v[k,j]);

    println!("Attention weights (softmax):");
    for i in 0..seq {
        let row: Vec<String> = (0..seq)
            .map(|j| format!("{:.4}", softmax.get(i, j)))
            .collect();
        println!("  [{}]", row.join(", "));
    }
    println!("Output shape: {:?}", out.shape());
}

#[cfg(not(feature = "cpu"))]
fn main() {
    println!("07_einsum_attention requires the cpu feature.");
}
