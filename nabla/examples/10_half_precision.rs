//! # Half Precision — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~20 | Julia LOC: ~20 (Float16 is built-in)
//! Julia advantage: `Float16` is a first-class numeric type
//! nabla advantage: `half` crate interop, same Tensor API for all precisions
//!
//! Run: cargo run --example 10_half_precision --features cpu

use nabla::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // f16 tensor creation
    let a = Tensor::<f16>::from_fn(3, 3, |i, j| f16::from_f32((i * 3 + j + 1) as f32));
    let b = Tensor::<f16>::from_fn(3, 3, |i, j| f16::from_f32(if i == j { 1.0 } else { 0.0 }));

    // Element-wise ops work on f16
    let c = &a * &b; // matmul
    println!("A * I (f16):");
    for i in 0..3 {
        let row: Vec<String> = (0..3).map(|j| format!("{:.1}", c.get(i, j))).collect();
        println!("  [{}]", row.join(", "));
    }

    // Element-wise math
    let exp_a = a.exp();
    let ln_a = a.ln();
    println!("\nexp(A[0,0]) = {:.4} (f16)", exp_a.get(0, 0));
    println!("ln(A[0,0])  = {:.4} (f16)", ln_a.get(0, 0));

    // Compare with f32 for precision
    let a32 = Tensor::<f32>::from_fn(3, 3, |i, j| (i * 3 + j + 1) as f32);
    let exp32 = a32.exp();
    let diff = (exp_a.get(0, 0).to_f32() - exp32.get(0, 0)).abs();
    println!("\nPrecision diff |exp_f16 - exp_f32| at [0,0]: {diff:.6}");
    println!("f16 range: [{}, {}]", f16::MIN, f16::MAX);

    Ok(())
}
