// Stiff ODE solvers: if_euler, bdf1, if_euler_scalar, metd_solve.
// All require feature = "cpu" (gated at the mod-level import in mod.rs).

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    alloc_trajectory, apply_saveat, diff_inf_norm, phi1, sc, validate, Bdf1Config, Bdf2Config,
    IfEulerScalarConfig, MetdConfig, OdeSolution,
};

#[inline]
fn is_full_step(h: f64, dt: f64) -> bool {
    (h - dt).abs() < 1e-14
}

/// Run fixed-point iteration until convergence.
/// Returns the converged value, or an error if `max_iter` exhausted.
#[allow(clippy::too_many_arguments)]
fn fixed_point_converge<T: Scalar, F>(
    f_eval: &F,
    t_next: f64,
    base: &Tensor<T, Cpu>,
    coeff: T,
    y_init: Tensor<T, Cpu>,
    max_iter: usize,
    tol: f64,
    err_msg: &str,
) -> Result<Tensor<T, Cpu>>
where
    F: Fn(f64, &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>>,
{
    let mut y_new = y_init;
    for _ in 0..max_iter {
        let fy = f_eval(t_next, &y_new)?;
        let y_next = base + &(&fy * coeff);
        let norm = diff_inf_norm(&y_next, &y_new);
        y_new = y_next;
        if norm < tol {
            return Ok(y_new);
        }
    }
    Err(Error::invalid(err_msg))
}

fn sym_from_diag(v: &Tensor<f64, Cpu>, diag: &[f64]) -> Tensor<f64, Cpu> {
    let n = diag.len();
    Tensor::<f64, Cpu>::from_fn(n, n, |i, j| {
        (0..n).map(|k| v.get(i, k) * diag[k] * v.get(j, k)).sum()
    })
}

fn exp_phi1_scalars<T: Scalar>(stiffness: f64, step: f64) -> (T, T) {
    let z = -stiffness * step;
    let exp_z_t: T = sc(z.exp());
    let h_phi1_t: T = sc(step * phi1(z));
    (exp_z_t, h_phi1_t)
}

#[inline]
fn require_square(name: &str, a: &Tensor<f64, Cpu>) -> Result<usize> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(Error::invalid(format!("{name}: matrix must be square")));
    }
    Ok(n)
}

#[inline]
fn require_col_vec<T: Scalar>(name: &str, y: &Tensor<T, Cpu>, n: usize) -> Result<()> {
    if y.shape() != (n, 1) {
        return Err(Error::invalid(format!(
            "{name}: expected column vector shape ({n}, 1), got {:?}",
            y.shape()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// IF Euler — exponential integrator for stiff ODEs
// ---------------------------------------------------------------------------

/// Integrating Factor Euler for stiff ODEs: `y' = A*y + g(t, y)`.
///
/// Uses the phi_1-function integrator:
///   `y_{n+1} = exp(h*A) * y_n + h * phi_1(h*A) * g(t_n, y_n)`
///
/// Unconditionally stable for any step size when A has non-positive real eigenvalues.
///
/// **Limitations**: A must be symmetric (self-adjoint eigendecomposition).
/// Only `f64` / `Cpu` backend.
///
/// # Arguments
/// - `a`: symmetric stiffness matrix A (n x n)
/// - `g`: non-stiff remainder g(t, y), where y is an (n x 1) column vector
/// - `y0`: initial state (n x 1 column vector)
/// - `t_span`: (`t_start`, `t_end`)
/// - `dt`: step size
///
/// # Errors
///
/// Returns an error if `a` is not square, `y0` shape is wrong, eigendecomposition
/// fails, or `g` returns an error.
pub fn if_euler<F>(
    a: &Tensor<f64, Cpu>,
    g: F,
    y0: &Tensor<f64, Cpu>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<OdeSolution<f64, Cpu>>
where
    F: Fn(f64, &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>,
{
    validate(t_span, dt)?;

    let n = require_square("if_euler", a)?;
    require_col_vec("if_euler", y0, n)?;

    // Eigendecompose symmetric A = V * diag(lambda) * V^T
    let sym = crate::linalg::Symmetric::new(a.clone(), crate::linalg::Side::Lower)?;
    let eig = sym.eigen()?;
    let lambdas = eig.eigenvalues();
    let v = eig.eigenvectors();

    // Precompute exp(h*A) = V * diag(exp(h*lambda_i)) * V^T
    // Precompute phi1(h*A) = V * diag(phi1(h*lambda_i)) * V^T
    let exp_diag: Vec<f64> = lambdas.iter().map(|&l| (dt * l).exp()).collect();
    let phi1_diag: Vec<f64> = lambdas.iter().map(|&l| phi1(dt * l)).collect();

    // exp_ha = V * diag(exp_diag) * V^T
    let exp_ha = sym_from_diag(v, &exp_diag);
    // phi1_ha = V * diag(phi1_diag) * V^T
    let phi1_ha = sym_from_diag(v, &phi1_diag);

    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let h = dt.min(t_span.1 - t);
        let gv = g(t, &y)?;

        if is_full_step(h, dt) {
            // Standard step: use precomputed matrices
            y = &(&exp_ha * &y) + &(&(&phi1_ha * &gv) * h);
        } else {
            // Final fractional step: recompute for this h
            let exp_d: Vec<f64> = lambdas.iter().map(|&l| (h * l).exp()).collect();
            let phi_d: Vec<f64> = lambdas.iter().map(|&l| phi1(h * l)).collect();
            let exp_h = sym_from_diag(v, &exp_d);
            let phi_h = sym_from_diag(v, &phi_d);
            y = &(&exp_h * &y) + &(&(&phi_h * &gv) * h);
        }

        t += h;
        times.push(t);
        states.push(y.clone());
    }

    Ok(OdeSolution { times, states })
}

// ---------------------------------------------------------------------------
// BDF-1 — Backward Euler implicit solver
// ---------------------------------------------------------------------------

/// BDF-1 (Backward Euler) solver for stiff ODEs: `dy/dt = f(t, y)`.
///
/// Solves the implicit equation `y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})`
/// via fixed-point (Picard) iteration at each step.
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, if `f` returns an error,
/// or if the fixed-point iteration fails to converge within `max_iter`.
pub fn bdf1<T: Scalar, F>(
    f: F,
    y0: &Tensor<T, Cpu>,
    t_span: (f64, f64),
    config: &Bdf1Config,
) -> Result<OdeSolution<T, Cpu>>
where
    F: Fn(f64, &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>>,
{
    validate(t_span, config.dt)?;
    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let h = config.dt.min(t_span.1 - t);
        let h_t: T = sc(h);
        let t_next = t + h;

        // Fixed-point: y^(k+1) = y_n + h * f(t_{n+1}, y^(k))
        let y_new = fixed_point_converge(
            &f,
            t_next,
            &y,
            h_t,
            y.clone(),
            config.max_iter,
            config.tol,
            "bdf1: fixed-point iteration did not converge — reduce dt or increase max_iter",
        )?;

        t = t_next;
        y = y_new;
        times.push(t);
        states.push(y.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}

// ---------------------------------------------------------------------------
// METD — Matrix Exponential Time Differencing (1st order)
// ---------------------------------------------------------------------------

/// phi_1(Z) via truncated Taylor series: `sum_{k=0}^{K} Z^k / (k+1)!`
///
/// Numerically stable for any Z (avoids computing inv(Z)).
fn phi1_matrix(z: &Tensor<f64, Cpu>, order: usize) -> Tensor<f64, Cpu> {
    let n = z.nrows();
    let mut result = Tensor::<f64, Cpu>::identity(n); // k=0 term: I / 1!
    let mut zk = Tensor::<f64, Cpu>::identity(n); // Z^0

    let mut factorial = 1.0_f64; // (k+1)! accumulator
    for k in 1..=order {
        zk = &zk * z; // Z^k
        factorial *= (k + 1) as f64;
        result = &result + &(&zk * (1.0 / factorial));
    }
    result
}

/// Matrix Exponential Time Differencing solver for stiff matrix-valued ODEs.
///
/// Solves `dy/dt = L*y + N(t, y)` where L is a constant n x n matrix (linear stiff part)
/// and N(t, y) is the nonlinear remainder.
///
/// Algorithm (1st-order METD):
///   `y_{n+1} = exp(h*L) * y_n + h * phi_1(h*L) * N(t_n, y_n)`
///
/// where `phi_1(Z) = sum_{k=0}^{K} Z^k / (k+1)!`.
///
/// # Errors
///
/// Returns an error if L is not square, y0 shape is wrong, expm fails, or
/// the time span/step size are invalid.
pub fn metd_solve<F>(
    l: &Tensor<f64, Cpu>,
    nonlinear: F,
    y0: Tensor<f64, Cpu>,
    t_span: (f64, f64),
    cfg: &MetdConfig,
) -> Result<OdeSolution<f64, Cpu>>
where
    F: Fn(f64, &Tensor<f64, Cpu>) -> Tensor<f64, Cpu>,
{
    validate(t_span, cfg.dt)?;

    let n = require_square("metd_solve", l)?;
    require_col_vec("metd_solve", &y0, n)?;

    // Precompute exp(h*L) and h * phi_1(h*L) for the standard step size
    let hl = l * cfg.dt;
    let exp_hl = crate::linalg::expm(&hl)?;
    let phi1_hl = phi1_matrix(&hl, cfg.order);

    let (mut times, mut states) = alloc_trajectory(t_span, cfg.dt, &y0);
    let mut t = t_span.0;
    let mut y = y0;

    while t < t_span.1 {
        let h = cfg.dt.min(t_span.1 - t);
        let nv = nonlinear(t, &y);

        if is_full_step(h, cfg.dt) {
            // Standard step: use precomputed matrices
            y = &(&exp_hl * &y) + &(&(&phi1_hl * &nv) * h);
        } else {
            // Final fractional step: recompute for this h
            let hl_frac = l * h;
            let exp_frac = crate::linalg::expm(&hl_frac)?;
            let phi1_frac = phi1_matrix(&hl_frac, cfg.order);
            y = &(&exp_frac * &y) + &(&(&phi1_frac * &nv) * h);
        }

        t += h;
        times.push(t);
        states.push(y.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &cfg.saveat))
}

// ---------------------------------------------------------------------------
// IF Euler Scalar — exponential integrator for scalar-stiff ODEs
// ---------------------------------------------------------------------------

/// Integrating-factor Euler for scalar-stiff ODEs: `dy/dt = -lambda*y + N(t, y)`.
///
/// Uses the exact integrating factor for the linear part:
///   `y_{n+1} = exp(-lambda*h) * y_n + h * phi_1(-lambda*h) * N(t_n, y_n)`
///
/// where `phi_1(z) = (exp(z) - 1) / z` (limit 1 when z -> 0).
///
/// Unconditionally stable for any step size when lambda > 0.
///
/// # Errors
///
/// Returns an error if the time span or step size are invalid, or if `nonlinear`
/// returns an error.
pub fn if_euler_scalar<T, F>(
    nonlinear: F,
    y0: &Tensor<T, Cpu>,
    t_span: (f64, f64),
    config: &IfEulerScalarConfig,
) -> Result<OdeSolution<T, Cpu>>
where
    T: Scalar,
    F: Fn(f64, &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>>,
{
    validate(t_span, config.dt)?;

    let h = config.dt;
    let (exp_z_t, h_phi1_t) = exp_phi1_scalars::<T>(config.stiffness, h);

    let (mut times, mut states) = alloc_trajectory(t_span, h, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let step = h.min(t_span.1 - t);
        let ny = nonlinear(t, &y)?;

        if is_full_step(step, h) {
            // Standard step: use precomputed scalars
            y = &(&y * exp_z_t) + &(&ny * h_phi1_t);
        } else {
            // Final fractional step: recompute for this step size
            let (exp_zf_t, h_phi1f_t) = exp_phi1_scalars::<T>(config.stiffness, step);
            y = &(&y * exp_zf_t) + &(&ny * h_phi1f_t);
        }

        t += step;
        times.push(t);
        states.push(y.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}

// ---------------------------------------------------------------------------
// BDF-2 — second-order Backward Differentiation Formula
// ---------------------------------------------------------------------------

/// BDF-2 solver for stiff ODEs: `dy/dt = f(t, y)`.
///
/// Second-order implicit multistep method:
///   `y_{n+1} = (4/3) y_n - (1/3) y_{n-1} + (2/3) h f(t_{n+1}, y_{n+1})`
///
/// The first step is bootstrapped with one BDF-1 (Backward Euler) step to
/// obtain `y_1`, after which BDF-2 is used for all subsequent steps.
///
/// The implicit equation at each step is solved via fixed-point (Picard)
/// iteration.
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, if `f` returns an error,
/// or if the fixed-point iteration fails to converge within `max_iter`.
pub fn bdf2<T: Scalar, F>(
    f: F,
    y0: &Tensor<T, Cpu>,
    t_span: (f64, f64),
    config: &Bdf2Config,
) -> Result<OdeSolution<T, Cpu>>
where
    F: Fn(f64, &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>>,
{
    validate(t_span, config.dt)?;
    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, y0);
    let mut t = t_span.0;
    let mut y_prev = y0.clone(); // y_{n-1}

    // -----------------------------------------------------------------------
    // Step 1: BDF-1 bootstrap to get y_1
    // -----------------------------------------------------------------------
    let h0 = config.dt.min(t_span.1 - t);
    let h0_t: T = sc(h0);
    let t1 = t + h0;

    let mut y_cur = fixed_point_converge(
        &f,
        t1,
        &y_prev,
        h0_t,
        y_prev.clone(),
        config.max_iter,
        config.tol,
        "bdf2: BDF-1 bootstrap step did not converge — reduce dt or increase max_iter",
    )?;

    t = t1;
    times.push(t);
    states.push(y_cur.clone());

    // -----------------------------------------------------------------------
    // Steps 2..N: BDF-2
    //   y_{n+1} = (4/3) y_n - (1/3) y_{n-1} + (2/3) h f(t_{n+1}, y_{n+1})
    // -----------------------------------------------------------------------
    let four_thirds: T = sc(4.0 / 3.0);
    let one_third: T = sc(1.0 / 3.0);
    let two_thirds_factor = 2.0 / 3.0;

    while t < t_span.1 {
        let h = config.dt.min(t_span.1 - t);
        let coeff: T = sc(two_thirds_factor * h);
        let t_next = t + h;

        // Predictor: explicit BDF-2 base = (4/3) y_n - (1/3) y_{n-1}
        let base = &(&y_cur * four_thirds) - &(&y_prev * one_third);

        // Fixed-point iteration for the implicit part
        let y_new = fixed_point_converge(
            &f,
            t_next,
            &base,
            coeff,
            y_cur.clone(),
            config.max_iter,
            config.tol,
            "bdf2: fixed-point iteration did not converge — reduce dt or increase max_iter",
        )?;

        y_prev = y_cur;
        y_cur = y_new;
        t = t_next;
        times.push(t);
        states.push(y_cur.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}
