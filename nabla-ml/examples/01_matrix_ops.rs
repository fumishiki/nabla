//! # Matrix Ops -- DSL wins on brevity
//! Run: cargo run --example 01_matrix_ops --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let tol = 1e-10;
    let a = mat![f64: 2.0, 1.0; 5.0, 3.0];
    let b = mat![f64: 1.0, 0.0; 0.0, 1.0];
    let eye = Tensor::<f64>::identity(2);

    let c = math!(a * b);
    println!(
        "A * B = [{}, {}; {}, {}]",
        c.get(0, 0), c.get(0, 1), c.get(1, 0), c.get(1, 1)
    );

    let rhs = mat![f64: 3.0; 8.0];
    let x = a.solve(&rhs)?;
    println!("A \\ b = [{}, {}]", x.get(0, 0), x.get(1, 0));

    let lu = a.partial_piv_lu()?;
    let recon = lu.reconstruct();
    println!("LU reconstruct matches: {}", math!((recon - a).abs().sum_all()) < tol);

    println!("eye = I: {}", math!((eye - b).abs().sum_all()) < tol);
}
