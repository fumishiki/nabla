//! # Lorenz Attractor — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~25 | Julia LOC: ~20 (DifferentialEquations.jl)
//! Julia advantage: `ODEProblem` + `solve()` one-liner
//! nabla advantage: zero dependency, adaptive step built-in
//!
//! Run: cargo run --example 05_ode_lorenz --features cpu

use nabla::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    // y = [x, y, z] as 3x1 column vector
    let y0: Tensor<f64> = mat![[1.0_f64], [1.0], [1.0]];

    let sol = nabla::ode::dormand_prince(
        |_t, y| {
            let x = y.get(0, 0);
            let y_val = y.get(1, 0);
            let z = y.get(2, 0);
            Ok(mat![
                [sigma * (y_val - x)],
                [x * (rho - z) - y_val],
                [x * y_val - beta * z]
            ])
        },
        &y0,
        (0.0, 2.0),
        &AdaptiveConfig {
            dt_init: 0.01,
            ..AdaptiveConfig::default()
        },
    )?;

    println!("Steps: {}", sol.len());
    let final_state = sol.final_state().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "non-empty solution expected")
    })?;
    println!(
        "Final (x, y, z) = ({:.4}, {:.4}, {:.4})",
        final_state.get(0, 0),
        final_state.get(1, 0),
        final_state.get(2, 0)
    );

    // Print first 5 time points
    for i in 0..5.min(sol.len()) {
        let s = &sol.states[i];
        println!(
            "t={:.4}: ({:.4}, {:.4}, {:.4})",
            sol.times[i],
            s.get(0, 0),
            s.get(1, 0),
            s.get(2, 0)
        );
    }

    Ok(())
}
