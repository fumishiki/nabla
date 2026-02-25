//! # Autograd MLP — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~30 | Julia LOC: ~30 (Zygote.jl)
//! Julia advantage: implicit differentiation via `gradient()`
//! nabla advantage: explicit tape gives control over what is tracked
//!
//! Run: cargo run --example 04_autograd_mlp --features cpu

#[cfg(feature = "cpu")]
use nabla::prelude::*;

#[cfg(feature = "cpu")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 1-hidden-layer MLP: input(2) -> hidden(3) -> output(1)
    let tape = Tape::<f64, Cpu>::new();

    // Weights (small fixed values for reproducibility)
    let w1 = tape.variable(mat![[0.1_f64, 0.2], [0.3, 0.4], [0.5, 0.6]]); // 3x2
    let w2 = tape.variable(mat![[0.1_f64, 0.2, 0.3]]); // 1x3
    let x = tape.variable(mat![[1.0_f64], [2.0]]); // 2x1

    // Forward: h = tanh(W1 * x), y = W2 * h
    let h = (&w1 * &x).tanh();
    let y = &w2 * &h;

    // MSE loss against target = 1.0
    let target = tape.variable(Tensor::fill(1, 1, 1.0_f64));
    let diff = &y - &target;
    let loss = diff.emul(&diff).sum_all_var();

    println!("Forward pass — loss: {:.6}", loss.data().get(0, 0));

    // Backward
    loss.backward()?;

    let g1 = w1.grad().expect("grad for W1");
    let g2 = w2.grad().expect("grad for W2");
    println!("dL/dW1 shape: {:?}", g1.shape());
    println!("dL/dW2 shape: {:?}", g2.shape());
    println!(
        "dL/dW2 = [{:.6}, {:.6}, {:.6}]",
        g2.get(0, 0),
        g2.get(0, 1),
        g2.get(0, 2)
    );

    Ok(())
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("example 04_autograd_mlp requires --features cpu");
}
