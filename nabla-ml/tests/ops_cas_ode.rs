#![cfg(feature = "cpu")]
#![allow(unused_imports)]

use nabla::cas::{Expr, diff, eval, eval_tensor, simplify};
use nabla::ode::{AdaptiveConfig, dormand_prince, rk4};
use nabla::prelude::*;
use nabla::{between, frange};
use std::collections::HashMap;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

fn linear_f64(rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::from_fn(rows, cols, |i, j| (i * cols + j + 1) as f64)
}

fn assert_approx_grid(got: &Tensor<f64>, expected: &Tensor<f64>, tol: f64) {
    assert_eq!(got.shape(), expected.shape(), "shape mismatch");
    let (r, c) = got.shape();
    for i in 0..r {
        for j in 0..c {
            assert!(
                (got.get(i, j) - expected.get(i, j)).abs() < tol,
                "mismatch at ({i},{j}): got {}, expected {}",
                got.get(i, j),
                expected.get(i, j)
            );
        }
    }
}

#[test]
fn cas_diff_simplify_eval_roundtrip() {
    let x = Expr::var("x");
    let expr = Expr::sin(&Expr::pow(&x, &Expr::lit(2.0)));
    let d = simplify(&diff(&expr, "x"));
    let mut vars = HashMap::new();
    vars.insert("x", 1.0_f64);
    let result = eval(&d, &vars).expect("eval failed");
    assert!((result - 2.0 * 1.0_f64.cos()).abs() < 1e-10);
}


#[test]
fn cas_eval_unbound_error() {
    assert!(eval(&Expr::var("x"), &HashMap::new()).is_err());
}


#[test]
fn cas_eval_tensor_roundtrip() {
    let x = Expr::var("x");
    let s = simplify(&(&x * &Expr::lit(2.0)));
    let t: Tensor<f64> = linear_f64(2, 2);
    let mut vars: HashMap<&str, &Tensor<f64>> = HashMap::new();
    vars.insert("x", &t);
    let result = eval_tensor(&s, &vars).expect("eval_tensor failed");
    assert!((result.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((result.get(1, 1) - 8.0).abs() < 1e-10);
}


#[test]
fn cas_eval_tensor_generic_type() {
    let x = Expr::var("x");
    let expr = Expr::exp(&x);
    let t: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j + 1) as f64);
    let mut vars: HashMap<&str, &Tensor<f64>> = HashMap::new();
    vars.insert("x", &t);
    let result = eval_tensor(&expr, &vars).expect("eval_tensor failed");
    assert!((result.get(0, 0) - 1.0_f64.exp()).abs() < 1e-10);
    assert!((result.get(1, 1) - 3.0_f64.exp()).abs() < 1e-10);
}


#[test]
fn cas_simplify_constant_folding() {
    use nabla::cas::{Expr, simplify};
    assert_eq!(
        format!("{}", simplify(&(&Expr::lit(2.0) + &Expr::lit(3.0)))),
        "5"
    );
    assert_eq!(
        format!("{}", simplify(&(&Expr::lit(2.0) * &Expr::lit(3.0)))),
        "6"
    );
}

#[test]
fn cas_simplify_eqsat_exp_ln() {
    use nabla::cas::{Expr, simplify};
    let x = Expr::var("x");
    // exp(ln(x)) = x
    assert_eq!(format!("{}", simplify(&Expr::exp(&Expr::ln(&x)))), "x");
    // ln(exp(x)) = x
    assert_eq!(format!("{}", simplify(&Expr::ln(&Expr::exp(&x)))), "x");
}


#[test]
fn cas_diff_simplify_matches_diff_simplify() {
    use nabla::cas::{Expr, diff, diff_simplify, eval, simplify};
    // Verify diff_simplify matches diff().simplify() numerically
    let x = Expr::var("x");
    let expr = &x * &Expr::sin(&x);
    let a = diff_simplify(&expr, "x");
    let b = simplify(&diff(&expr, "x"));
    for v in [0.1, 0.5, 1.0, 2.0] {
        let mut vars = HashMap::new();
        vars.insert("x", v);
        let va = eval(&a, &vars).expect("eval a");
        let vb = eval(&b, &vars).expect("eval b");
        assert!((va - vb).abs() < 1e-9, "mismatch at x={v}: {va} vs {vb}");
    }
}


#[test]
fn cas_diff_multivar_product_rule() {
    use nabla::cas::{Expr, diff_simplify, eval};
    // d(x*y)/dx = y
    let expr = &Expr::var("x") * &Expr::var("y");
    let d = diff_simplify(&expr, "x");
    let mut vars = HashMap::new();
    vars.insert("x", 2.0);
    vars.insert("y", 3.0);
    let val = eval(&d, &vars).expect("eval");
    assert!((val - 3.0).abs() < 1e-10);
}


#[test]
fn cas_diff_multivar_other_var() {
    use nabla::cas::{Expr, diff_simplify, eval};
    // d(x^2)/dy = 0
    let expr = Expr::pow(&Expr::var("x"), &Expr::lit(2.0));
    let d = diff_simplify(&expr, "y");
    let mut vars = HashMap::new();
    vars.insert("x", 5.0);
    vars.insert("y", 1.0);
    let val = eval(&d, &vars).expect("eval");
    assert!(val.abs() < 1e-10);
}


#[test]
fn cas_diff_chain_rule_exp() {
    use nabla::cas::{Expr, diff_simplify, eval};
    // d(exp(x^2))/dx = 2*x*exp(x^2)
    let x = Expr::var("x");
    let x2 = Expr::pow(&x, &Expr::lit(2.0));
    let expr = Expr::exp(&x2);
    let d = diff_simplify(&expr, "x");
    let xv = 1.5_f64;
    let expected = 2.0 * xv * (xv * xv).exp();
    let mut vars = HashMap::new();
    vars.insert("x", xv);
    let val = eval(&d, &vars).expect("eval");
    assert!(
        (val - expected).abs() < 1e-6,
        "got {val}, expected {expected}"
    );
}

#[test]
fn ode_rk4_accuracy() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let sol = rk4(|_t, y| Ok(-y), &y0, (0.0, 1.0), 0.01).expect("rk4 failed");
    assert!(
        (sol.final_state().expect("final_state failed").get(0, 0) - (-1.0_f64).exp()).abs() < 1e-6
    );
}


#[test]
fn ode_dormand_prince_adaptive() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let config = AdaptiveConfig {
        dt_init: 0.1,
        ..Default::default()
    };
    let sol =
        dormand_prince(|_t, y| Ok(-y), &y0, (0.0, 1.0), &config).expect("dormand_prince failed");
    assert!(
        (sol.final_state().expect("final_state failed").get(0, 0) - (-1.0_f64).exp()).abs() < 1e-4
    );
    assert!(sol.len() < 100);
}


#[test]
fn ode_error_invalid_inputs() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    assert!(rk4(|_t, y| Ok(-y), &y0, (0.0, 1.0), -0.1).is_err());
}


#[test]
fn ode_rk4_generic_type() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let sol = rk4(|_t, y| Ok(-y), &y0, (0.0, 0.5), 0.01).expect("rk4 failed");
    assert!(
        (sol.final_state().expect("final_state failed").get(0, 0) - (-0.5_f64).exp()).abs() < 1e-5
    );
}


#[test]
fn bdf1_linear_decay() {
    // dy/dt = -100y, y(0) = 1 => y(t) = exp(-100t)
    use nabla::ode::{Bdf1Config, bdf1};
    let y0 = Tensor::<f64>::fill(1, 1, 1.0);
    let sol = bdf1(
        |_t, y| Ok(y * (-100.0_f64)),
        &y0,
        (0.0, 0.1),
        &Bdf1Config {
            dt: 0.001,
            tol: 1e-10,
            max_iter: 100,
            saveat: None,
        },
    )
    .expect("bdf1 should converge");
    let y_final = sol.final_state().expect("final_state").get(0, 0);
    let expected = (-100.0_f64 * 0.1).exp();
    assert!(
        (y_final - expected).abs() < 0.1,
        "bdf1: got {y_final}, expected ~{expected}"
    );
}


#[test]
fn if_euler_scalar_stiff_stable() {
    // lambda=100, dt=0.1 — forward Euler diverges but IF Euler stays stable
    let y0 = mat![[1.0_f64]];
    let config = IfEulerScalarConfig {
        dt: 0.1,
        stiffness: 100.0,
        saveat: None,
    };
    let sol = if_euler_scalar(
        |_t, _y| Ok(Tensor::<f64>::zeros(1, 1)),
        &y0,
        (0.0, 0.5),
        &config,
    )
    .expect("if_euler_scalar failed");
    let y_final = sol.final_state().expect("no final state");
    assert!(y_final[(0, 0)] >= 0.0, "IF Euler should remain stable");
    assert!(
        y_final[(0, 0)] < 0.01,
        "should have decayed significantly, got {}",
        y_final[(0, 0)]
    );
}


#[test]
fn dae_simple_constraint() {
    // x' = 1, 0 = z - x  =>  x(t) = t, z(t) = t
    // x0 = 0, z0 = 0, t ∈ [0, 1]
    use nabla::ode::{DaeConfig, dae_solve};
    let x0 = Tensor::<f64>::fill(1, 1, 0.0);
    let z0 = Tensor::<f64>::fill(1, 1, 0.0);
    let sol = dae_solve(
        |_x, _z, _t| Tensor::<f64>::fill(1, 1, 1.0), // f: x' = 1
        |x, z, _t| {
            // g: 0 = z - x
            let zv = z.get(0, 0);
            let xv = x.get(0, 0);
            Tensor::<f64>::fill(1, 1, zv - xv)
        },
        x0,
        z0,
        (0.0, 1.0),
        &DaeConfig {
            dt: 0.01,
            tol: 1e-10,
            max_iter: 50,
            saveat: None,
        },
    )
    .expect("dae_solve failed");
    let x_final = sol.final_state().expect("no final state").get(0, 0);
    assert!(
        (x_final - 1.0).abs() < 0.05,
        "dae x(1) should be ~1.0, got {x_final}"
    );
}


#[test]
fn dae_quadratic_constraint() {
    // x' = z, 0 = z - 2*t  =>  z(t) = 2t, x(t) = t^2
    // x0 = 0, z0 = 0
    use nabla::ode::{DaeConfig, dae_solve};
    let x0 = Tensor::<f64>::fill(1, 1, 0.0);
    let z0 = Tensor::<f64>::fill(1, 1, 0.0);
    let sol = dae_solve(
        |_x, z, _t| z.clone(), // f: x' = z
        |_x, z, t| {
            // g: 0 = z - 2t
            let zv = z.get(0, 0);
            Tensor::<f64>::fill(1, 1, zv - 2.0 * t)
        },
        x0,
        z0,
        (0.0, 1.0),
        &DaeConfig {
            dt: 0.01,
            tol: 1e-10,
            max_iter: 50,
            saveat: None,
        },
    )
    .expect("dae_solve failed");
    let x_final = sol.final_state().expect("no final state").get(0, 0);
    // x(1) = 1^2 = 1.0
    assert!(
        (x_final - 1.0).abs() < 0.05,
        "dae x(1) should be ~1.0, got {x_final}"
    );
}


#[test]
fn metd_linear_decay() {
    // dy/dt = -y  →  L = [-1] (1x1 matrix), N = 0
    // y(0) = 1, exact: y(t) = exp(-t), so y(1) ≈ 0.3679
    use nabla::ode::{MetdConfig, metd_solve};
    let l: Tensor<f64> = mat![[-1.0_f64]];
    let y0: Tensor<f64> = mat![[1.0_f64]];
    let cfg = MetdConfig { dt: 0.01, order: 8, saveat: None };
    let sol = metd_solve(
        &l,
        |_t, _y| Tensor::<f64>::zeros(1, 1), // N(t,y) = 0
        y0,
        (0.0, 1.0),
        &cfg,
    )
    .expect("metd_solve failed");
    let y_final = sol.final_state().expect("empty solution");
    let val = y_final.get(0, 0);
    let expected = (-1.0_f64).exp(); // e^{-1} ≈ 0.3679
    assert!(
        (val - expected).abs() < 1e-6,
        "metd_linear_decay: got {val}, expected {expected}"
    );
}


#[test]
fn stormer_verlet_harmonic() {
    // Simple harmonic oscillator: V(q) = q^2/2, grad_V(q) = q
    // H = p^2/2 + q^2/2 conserved
    // q(0)=1, p(0)=0, exact: q(t)=cos(t), p(t)=-sin(t), H=0.5
    use nabla::ode::{StormerVerletConfig, stormer_verlet};
    let cfg = StormerVerletConfig {
        dt: 0.01,
        mass: 1.0,
        saveat: None,
    };
    let sol = stormer_verlet(
        |q| q.clone(), // grad_V(q) = q
        mat![[1.0_f64]],
        mat![[0.0_f64]],
        (0.0, 2.0 * std::f64::consts::PI),
        &cfg,
    )
    .expect("stormer_verlet failed");
    // Check energy conservation: H = (q^2 + p^2)/2 ≈ 0.5 at every step
    for (q, p) in sol.q_states.iter().zip(sol.p_states.iter()) {
        let qv = q.get(0, 0);
        let pv = p.get(0, 0);
        let h = (qv * qv + pv * pv) * 0.5;
        assert!(
            (h - 0.5).abs() < 0.01,
            "stormer_verlet_harmonic: energy {h} deviates from 0.5"
        );
    }
}


#[test]
fn parareal_van_der_pol() {
    use nabla::ode::{PararealConfig, parareal_solve};

    // Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0
    // Rewrite as system: x1' = x2, x2' = mu*(1-x1^2)*x2 - x1
    // For scalar parareal we solve just x1' = x2 with x2 coupled analytically via Euler.
    // Simpler: use a scalar test ODE y' = -y (exponential decay) and verify convergence.
    let t0 = 0.0;
    let t1 = 2.0;
    let y0 = 1.0;

    // Coarse propagator: single Euler step per interval
    let coarse = |ta: f64, tb: f64, ya: f64| -> f64 {
        let h = tb - ta;
        ya + h * (-ya) // y' = -y, forward Euler
    };

    // Fine propagator: 100 Euler sub-steps per interval (accurate)
    let fine = |ta: f64, tb: f64, ya: f64| -> f64 {
        let n_sub = 100;
        let h = (tb - ta) / n_sub as f64;
        let mut y = ya;
        for _ in 0..n_sub {
            y += h * (-y);
        }
        y
    };

    let config = PararealConfig {
        n_intervals: 8,
        max_iter: 5,
        tol: 1e-8,
        saveat: None,
    };
    let result = parareal_solve(t0, t1, y0, &config, coarse, fine);
    assert!(result.is_ok());
    let vals = result.expect("parareal should converge");

    // Exact solution: y(t) = exp(-t)
    let exact_final = (-t1).exp();
    let computed_final = vals[vals.len() - 1];

    // Fine propagator with 100 sub-steps has ~1e-4 error per interval,
    // parareal corrects to fine-level accuracy
    assert!(
        (computed_final - exact_final).abs() < 1e-3,
        "parareal final value {computed_final} vs exact {exact_final}"
    );

    // Check that all intermediate values are reasonable
    let dt = (t1 - t0) / config.n_intervals as f64;
    for (i, &v) in vals.iter().enumerate() {
        let t = t0 + i as f64 * dt;
        let exact = (-t).exp();
        assert!(
            (v - exact).abs() < 1e-2,
            "parareal checkpoint {i}: computed={v}, exact={exact}"
        );
    }
}

// #[nabla_grad] proc macro — source-transform forward AD

#[nabla_grad]
#[allow(dead_code)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
