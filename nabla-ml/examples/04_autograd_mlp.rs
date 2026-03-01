//! # Autograd MLP -- DSL + ad! macro
//! Run: cargo run --example 04_autograd_mlp --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let (loss, g1, g2, _gx) = ad!(Cpu;
        w1 = mat![f64: [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        w2 = mat![f64: [0.1, 0.2, 0.3]],
        x  = mat![f64: [1.0], [2.0]]
        => |tape| {
            let h = math!((w1 * x).tanh());
            let y = math!(w2 * h);
            let target = tape.variable(Tensor::fill(1, 1, 1.0_f64))?;
            let diff = math!(y - target);
            math!(diff.emul(diff).sum_all_var())
        }
    );

    println!("Forward pass -- loss: {:.6}", loss.data().get(0, 0));
    println!("dL/dW1 shape: {:?}", g1.shape());
    println!("dL/dW2 shape: {:?}", g2.shape());
    println!(
        "dL/dW2 = [{:.6}, {:.6}, {:.6}]",
        g2.get(0, 0),
        g2.get(0, 1),
        g2.get(0, 2)
    );
}
