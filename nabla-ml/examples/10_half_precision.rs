//! # Half Precision -- same API, different dtype
//! Run: cargo run --example 10_half_precision --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let a = Tensor::<f16>::from_fn(3, 3, |i, j| f16::from_f32((i * 3 + j + 1) as f32));
    let b = Tensor::<f16>::identity(3);

    let c = math!(a * b);
    println!("A * I (f16):");
    for i in 0..3 {
        let row: Vec<String> = (0..3).map(|j| format!("{:.1}", c.get(i, j))).collect();
        println!("  [{}]", row.join(", "));
    }

    let exp_a = a.exp();
    let ln_a = a.ln();
    println!("\nexp(A[0,0]) = {:.4} (f16)", exp_a.get(0, 0));
    println!("ln(A[0,0])  = {:.4} (f16)", ln_a.get(0, 0));

    let a32 = Tensor::<f32>::from_fn(3, 3, |i, j| (i * 3 + j + 1) as f32);
    let exp32 = a32.exp();
    let diff = (exp_a.get(0, 0).to_f32() - exp32.get(0, 0)).abs();
    println!("\nPrecision diff |exp_f16 - exp_f32| at [0,0]: {diff:.6}");
    println!("f16 range: [{}, {}]", f16::MIN, f16::MAX);
}
