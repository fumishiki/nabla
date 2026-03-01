
use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    alloc_trajectory, apply_saveat, diff_inf_norm, phi1, sc, validate, wrap_rhs, Bdf1Config,
    Bdf2Config, IfEulerScalarConfig, IntoOdeRhs, MetdConfig, OdeSolution,
};

#[inline]
fn is_full_step(h: f64, dt: f64) -> bool {
    (h - dt).abs() < 1e-14
}

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


/// Matrix integrating-factor Euler method for stiff linear systems.
pub fn if_euler<R: IntoOdeRhs<f64, Cpu>, F>(
    a: &Tensor<f64, Cpu>,
    g: F,
    y0: &Tensor<f64, Cpu>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<OdeSolution<f64, Cpu>>
where
    F: Fn(f64, &Tensor<f64, Cpu>) -> R,
{
    let g = wrap_rhs(g);
    validate(t_span, dt)?;

    let n = require_square("if_euler", a)?;
    require_col_vec("if_euler", y0, n)?;

    // Eigendecompose symmetric A = V * diag(lambda) * V^T
    let sym = crate::linalg::Symmetric::new(a.clone(), crate::linalg::Side::Lower)?;
    let eig = sym.eigen()?;
    let lambdas = eig.eigenvalues();
    let v = eig.eigenvectors();

    let dir = super::time_direction(t_span);
    // Precompute exp(h*A) = V * diag(exp(h*lambda_i)) * V^T (with direction)
    let signed_dt = dir * dt;
    let exp_diag: Vec<f64> = lambdas.iter().map(|&l| (signed_dt * l).exp()).collect();
    let phi1_diag: Vec<f64> = lambdas.iter().map(|&l| phi1(signed_dt * l)).collect();

    let exp_ha = sym_from_diag(v, &exp_diag);
    let phi1_ha = sym_from_diag(v, &phi1_diag);
    let remaining = |t: f64| (t_span.1 - t) * dir;
    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while remaining(t) > 1e-14 {
        let h = dir * dt.min(remaining(t).abs());
        let gv = g(t, &y)?;

        if is_full_step(h.abs(), dt) {
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


/// BDF-1 (Backward Euler) implicit method with fixed-point iteration.
pub fn bdf1<T: Scalar, R: IntoOdeRhs<T, Cpu>, F>(
    f: F,
    y0: &Tensor<T, Cpu>,
    t_span: (f64, f64),
    config: &Bdf1Config,
) -> Result<OdeSolution<T, Cpu>>
where
    F: Fn(f64, &Tensor<T, Cpu>) -> R,
{
    let f = wrap_rhs(f);
    validate(t_span, config.dt)?;
    let dir = super::time_direction(t_span);
    let remaining = |t: f64| (t_span.1 - t) * dir;
    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while remaining(t) > 1e-14 {
        let h = dir * config.dt.min(remaining(t).abs());
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

/// Matrix exponential time-differencing solver for stiff semi-linear systems.
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

    let dir = super::time_direction(t_span);
    // Precompute exp(h*L) and h * phi_1(h*L) for the standard step size (with direction)
    let hl = l * (dir * cfg.dt);
    let exp_hl = crate::linalg::expm(&hl)?;
    let phi1_hl = phi1_matrix(&hl, cfg.order);
    let remaining = |t: f64| (t_span.1 - t) * dir;
    let (mut times, mut states) = alloc_trajectory(t_span, cfg.dt, &y0);
    let mut t = t_span.0;
    let mut y = y0;

    while remaining(t) > 1e-14 {
        let h = dir * cfg.dt.min(remaining(t).abs());
        let nv = nonlinear(t, &y);

        if is_full_step(h.abs(), cfg.dt) {
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


/// Scalar integrating-factor Euler method for stiff ODEs with scalar stiffness.
pub fn if_euler_scalar<T, R: IntoOdeRhs<T, Cpu>, F>(
    nonlinear: F,
    y0: &Tensor<T, Cpu>,
    t_span: (f64, f64),
    config: &IfEulerScalarConfig,
) -> Result<OdeSolution<T, Cpu>>
where
    T: Scalar,
    F: Fn(f64, &Tensor<T, Cpu>) -> R,
{
    let nonlinear = wrap_rhs(nonlinear);
    validate(t_span, config.dt)?;

    let dir = super::time_direction(t_span);
    let remaining = |t: f64| (t_span.1 - t) * dir;
    let h = config.dt;
    let (exp_z_t, h_phi1_t) = exp_phi1_scalars::<T>(config.stiffness, dir * h);

    let (mut times, mut states) = alloc_trajectory(t_span, h, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while remaining(t) > 1e-14 {
        let step = dir * h.min(remaining(t).abs());
        let ny = nonlinear(t, &y)?;

        if is_full_step(step.abs(), h) {
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


/// BDF-2 (second-order BDF) implicit method with BDF-1 bootstrap.
pub fn bdf2<T: Scalar, R: IntoOdeRhs<T, Cpu>, F>(
    f: F,
    y0: &Tensor<T, Cpu>,
    t_span: (f64, f64),
    config: &Bdf2Config,
) -> Result<OdeSolution<T, Cpu>>
where
    F: Fn(f64, &Tensor<T, Cpu>) -> R,
{
    let f = wrap_rhs(f);
    validate(t_span, config.dt)?;
    let dir = super::time_direction(t_span);
    let remaining = |t: f64| (t_span.1 - t) * dir;
    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, y0);
    let mut t = t_span.0;
    let mut y_prev = y0.clone(); // y_{n-1}

    // -----------------------------------------------------------------------
    // Step 1: BDF-1 bootstrap to get y_1
    // -----------------------------------------------------------------------
    let h0 = dir * config.dt.min(remaining(t).abs());
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

    while remaining(t) > 1e-14 {
        let h = dir * config.dt.min(remaining(t).abs());
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
