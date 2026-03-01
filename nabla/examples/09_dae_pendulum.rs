//! # DAE Circuit — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~30 | Julia LOC: ~30 (DifferentialEquations.jl + mass matrix)
//! Julia advantage: `DAEProblem` with mass matrix is more general
//! nabla advantage: direct semi-explicit formulation, Newton corrector built-in
//!
//! Run: cargo run --example 09_dae_pendulum --features cpu

#[cfg(feature = "cpu")]
use nabla::prelude::*;

#[cfg(feature = "cpu")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Simple RC circuit as index-1 DAE:
    //   Differential: C * dV/dt = I             (capacitor)
    //   Algebraic:    V + R * I = V_source(t)   (KVL constraint)
    //
    // x = [V] (voltage across capacitor), z = [I] (current)
    // V_source(t) = 1.0 (step input)

    let r = 1.0; // resistance
    let c = 0.1; // capacitance

    let x0 = mat![[0.0_f64]]; // capacitor starts discharged
    let z0 = mat![[1.0_f64]]; // initial current = V_source / R

    let sol = dae_solve(
        // f(x, z, t) -> dx/dt
        |_x, z, _t| {
            let i = z.get(0, 0);
            mat![[i / c]] // dV/dt = I/C
        },
        // g(x, z, t) -> algebraic constraint residual
        |x, z, _t| {
            let v = x.get(0, 0);
            let i = z.get(0, 0);
            let v_source = 1.0; // step input
            mat![[v + r * i - v_source]] // KVL: V + R*I = V_source
        },
        x0,
        z0,
        (0.0, 0.5),
        &DaeConfig {
            dt: 0.001,
            tol: 1e-10,
            max_iter: 100,
            saveat: None,
        },
    )?;

    println!("Steps: {}", sol.len());

    // Print a few time points
    let n = sol.len();
    for &idx in &[0, n / 4, n / 2, 3 * n / 4, n - 1] {
        let v = sol.states[idx].get(0, 0);
        println!(
            "t={:.3}: V={:.6} (exact: {:.6})",
            sol.times[idx],
            v,
            1.0 - (-sol.times[idx] / (r * c)).exp()
        );
    }

    // Final voltage should approach V_source = 1.0
    let final_v = sol.final_state().expect("non-empty").get(0, 0);
    println!("Final V = {final_v:.6} (target: 1.0)");

    Ok(())
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("example 09_dae_pendulum requires --features cpu");
}
