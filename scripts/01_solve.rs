#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "..", features = ["cpu"] }
//! ```

use nabla::prelude::*;

fn main() -> nabla::Result<()> {
    let a = mat![[2.0_f64, 1.0], [5.0, 7.0]];
    let b = mat![[11.0_f64], [13.0]];
    let x = a.solve(&b)?;
    println!("A = [[2,1],[5,7]], b = [[11],[13]]");
    println!("x = [{:.4}, {:.4}]", x.get(0, 0), x.get(1, 0));

    // Verify: A*x ≈ b
    let residual = &b - &(&a * &x);
    let err = residual.emul(&residual).sum_all().sqrt();
    println!("||Ax - b|| = {err:.2e}");

    Ok(())
}
