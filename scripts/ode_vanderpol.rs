#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "../nabla", features = ["cpu"] }
//! ```

use nabla::prelude::*;

fn main() -> nabla::error::Result<()> {
    // Van der Pol oscillator: dy1/dt = y2, dy2/dt = mu*(1-y1^2)*y2 - y1
    let mu = 1000.0_f64;

    let y0: Tensor<f64> = mat![[2.0_f64], [0.0]];
    let config = IfEulerScalarConfig { dt: 0.01, stiffness: mu };

    let sol = if_euler_scalar(
        |_t, y| {
            let y1 = y.get(0, 0);
            let y2 = y.get(1, 0);
            let mut ny = Tensor::<f64>::zeros(2, 1);
            ny.set(0, 0, y2);
            ny.set(1, 0, mu * (1.0 - y1 * y1) * y2 - y1);
            Ok(ny)
        },
        &y0,
        (0.0, 10.0),
        &config,
    )?;

    let final_state = sol.final_state().expect("no states");
    println!("Van der Pol (mu={mu}), t=0..10, dt=0.01");
    println!("y1 = {:.6}", final_state.get(0, 0));
    println!("y2 = {:.6}", final_state.get(1, 0));
    println!("steps = {}", sol.len());

    Ok(())
}
