//! # Lorenz Attractor -- ODE in a few lines
//! Run: cargo run --example 05_ode_lorenz --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    let y0: Tensor<f64> = mat![f64: 1.0; 1.0; 1.0];

    let sol = dormand_prince(
        |_t, y| {
            vec_unpack!(y, x, y_val, z);
            mat![
                [sigma * (y_val - x)],
                [x * (rho - z) - y_val],
                [x * y_val - beta * z]
            ]
        },
        &y0,
        (0.0, 2.0),
        &AdaptiveConfig::default().with_dt(0.01),
    )?;

    println!("Steps: {}", sol.len());
    let final_state = sol.final_state().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "non-empty solution expected")
    })?;
    vec_unpack!(final_state, fx, fy, fz);
    println!("Final (x, y, z) = ({fx:.4}, {fy:.4}, {fz:.4})");

    for i in 0..5.min(sol.len()) {
        let s = &sol.states[i];
        vec_unpack!(s, sx, sy, sz);
        println!("t={:.4}: ({sx:.4}, {sy:.4}, {sz:.4})", sol.times[i]);
    }
}
