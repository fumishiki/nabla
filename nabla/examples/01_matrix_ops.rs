//! # Matrix Operations — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~18 | Julia LOC: ~12
//! Julia advantage: implicit typing, no `?` or `&`
//! nabla advantage: compile-time shape checks, explicit error handling
//!
//! Run: cargo run --example 01_matrix_ops --features cpu

#[cfg(feature = "cpu")]
use nabla::prelude::*;

#[cfg(feature = "cpu")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create matrices
    let a = mat![[2.0_f64, 1.0], [5.0, 3.0]];
    let b = mat![[1.0_f64, 0.0], [0.0, 1.0]];
    let eye = Tensor::<f64>::identity(2);

    // Matmul: C = A * B
    let c = &a * &b;
    println!(
        "A * B = [{}, {}; {}, {}]",
        c.get(0, 0),
        c.get(0, 1),
        c.get(1, 0),
        c.get(1, 1)
    );

    // Solve: A * x = b  (column vector)
    let rhs = mat![[3.0_f64], [8.0]];
    let x = a.solve(&rhs)?;
    println!("A \\ b = [{}, {}]", x.get(0, 0), x.get(1, 0));

    // LU factorization + reconstruct
    let lu = a.partial_piv_lu()?;
    let recon = lu.reconstruct();
    println!(
        "LU reconstruct matches: {}",
        (&recon - &a).abs().sum_all() < 1e-10
    );

    // Identity check
    println!("eye = I: {}", (&eye - &b).abs().sum_all() < 1e-10);

    Ok(())
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("example 01_matrix_ops requires --features cpu");
}
