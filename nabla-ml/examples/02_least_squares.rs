//! # Least Squares -- one call DSL
//! Run: cargo run --example 02_least_squares --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let a = mat![f64: 1.0, 0.0; 1.0, 1.0; 1.0, 2.0; 1.0, 3.0];
    let b = mat![f64: 1.1; 2.0; 2.9; 4.1];

    let x = a.solve_lstsq(&b)?;
    println!("Intercept: {:.4}, Slope: {:.4}", x.get(0, 0), x.get(1, 0));

    let residual = math!(b - a * x);
    let norm_sq = residual.emul(&residual).sum_all();
    println!("Residual norm^2: {:.6}", norm_sq);
}
