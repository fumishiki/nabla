//! # DAE Circuit -- semi-explicit in a few lines
//! Run: cargo run --example 09_dae_pendulum --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let r = 1.0;
    let c = 0.1;

    let x0 = mat![f64: 0.0];
    let z0 = mat![f64: 1.0];

    let sol = dae_solve(
        |_x, z, _t| {
            let i = z.get(0, 0);
            mat![[i / c]]
        },
        |x, z, _t| {
            let v = x.get(0, 0);
            let i = z.get(0, 0);
            let v_source = 1.0;
            mat![[v + r * i - v_source]]
        },
        x0,
        z0,
        (0.0, 0.5),
        &DaeConfig::default().with_dt(0.001).with_tol(1e-10).with_max_iter(100),
    )?;

    println!("Steps: {}", sol.len());

    let n = sol.len();
    for &idx in &[0, n / 4, n / 2, 3 * n / 4, n - 1] {
        let v = sol.states[idx].get(0, 0);
        println!(
            "t={:.3}: V={:.6} (exact: {:.6})",
            sol.times[idx], v, 1.0 - (-sol.times[idx] / (r * c)).exp()
        );
    }

    let final_v = sol
        .final_state()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "non-empty"))?
        .get(0, 0);
    println!("Final V = {final_v:.6} (target: 1.0)");
}
