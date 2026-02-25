#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "..", features = ["cpu"] }
//! ```

use nabla::prelude::*;

fn main() -> nabla::Result<()> {
    // Solve dy/dt = -2y, y(0) = 1  on [0, 2]
    // Exact solution: y(t) = exp(-2t)
    let y0 = mat![[1.0_f64]];

    let sol = nabla::ode::rk4(
        |_t, y| Ok(&y * -2.0_f64),
        &y0,
        (0.0, 2.0),
        0.01,
    )?;

    let y_final = sol.final_state().expect("empty solution");
    let exact = (-4.0_f64).exp();
    let err = (y_final.get(0, 0) - exact).abs();

    println!("y(2) = {:.8}  (RK4)", y_final.get(0, 0));
    println!("exact = {exact:.8}");
    println!("|error| = {err:.2e}");

    Ok(())
}
