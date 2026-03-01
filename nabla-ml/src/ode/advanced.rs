use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    DaeConfig, IntoOdeRhs, OdeSolution, PararealConfig, StormerVerletConfig, SymplecticSolution,
    alloc_trajectory, apply_saveat, diff_inf_norm, inf_norm_vec, validate,
};

fn wrap_rhs_parareal<T, R, F>(f: F) -> impl Fn(f64, f64, &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>>
where
    T: Scalar,
    R: IntoOdeRhs<T, Cpu>,
    F: Fn(f64, f64, &Tensor<T, Cpu>) -> R,
{
    move |t0, t1, y| f(t0, t1, y).into_rhs()
}

/// Semi-explicit DAE solver using implicit Euler with Newton iteration.
pub fn dae_solve<F, G>(
    f: F,
    g: G,
    x0: Tensor<f64, Cpu>,
    z0: Tensor<f64, Cpu>,
    t_span: (f64, f64),
    cfg: &DaeConfig,
) -> Result<OdeSolution<f64, Cpu>>
where
    F: Fn(&Tensor<f64, Cpu>, &Tensor<f64, Cpu>, f64) -> Tensor<f64, Cpu>,
    G: Fn(&Tensor<f64, Cpu>, &Tensor<f64, Cpu>, f64) -> Tensor<f64, Cpu>,
{
    use crate::linalg::LinalgExt;

    validate(t_span, cfg.dt)?;
    let (mut times, mut states) = alloc_trajectory(t_span, cfg.dt, &x0);
    let n_z = z0.nrows();
    let eps = 1e-7;

    let mut t = t_span.0;
    let mut x = x0;
    let mut z = z0;

    while t < t_span.1 {
        let h = cfg.dt.min(t_span.1 - t);
        let t_next = t + h;

        // Predictor: x_pred = x_n + h * f(x_n, z_n, t_n)
        let fx = f(&x, &z, t);
        let x_pred = &x + &(&fx * h);

        // Newton iteration for z at t_{n+1}
        let mut z_k = z.clone();
        let mut converged = false;
        for _ in 0..cfg.max_iter {
            let gv = g(&x_pred, &z_k, t_next);

            // Check convergence: ||g|| < tol
            let norm = inf_norm_vec(&gv);
            if norm < cfg.tol {
                converged = true;
                break;
            }

            // Finite-difference Jacobian J_z in R^{n_z x n_z}
            let jac = Tensor::<f64, Cpu>::from_fn(n_z, n_z, |i, j| {
                let mut z_plus = z_k.clone();
                let orig = z_plus.get(j, 0);
                z_plus.set(j, 0, orig + eps);
                let g_plus = g(&x_pred, &z_plus, t_next);
                (g_plus.get(i, 0) - gv.get(i, 0)) / eps
            });

            // Solve J_z * dz = g for dz, then z_{k+1} = z_k - dz
            let dz = jac.solve(&gv)?;
            z_k = &z_k - &dz;
        }

        if !converged {
            return Err(Error::invalid(
                "dae_solve: Newton iteration did not converge — reduce dt or increase max_iter",
            ));
        }

        // Corrector: x_{n+1} = x_n + h * f(x_pred, z_{n+1}, t_{n+1})
        let fx_corr = f(&x_pred, &z_k, t_next);
        x = &x + &(&fx_corr * h);
        z = z_k;
        t = t_next;

        times.push(t);
        states.push(x.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &cfg.saveat))
}

/// Stormer-Verlet symplectic integrator for Hamiltonian systems.
pub fn stormer_verlet<V>(
    grad_v: V,
    q0: Tensor<f64, Cpu>,
    p0: Tensor<f64, Cpu>,
    t_span: (f64, f64),
    cfg: &StormerVerletConfig,
) -> Result<SymplecticSolution<f64, Cpu>>
where
    V: Fn(&Tensor<f64, Cpu>) -> Tensor<f64, Cpu>,
{
    validate(t_span, cfg.dt)?;

    let inv_mass = 1.0 / cfg.mass;

    let capacity = ((t_span.1 - t_span.0) / cfg.dt).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut qs = Vec::with_capacity(capacity);
    let mut ps = Vec::with_capacity(capacity);

    let mut t = t_span.0;
    let mut q = q0;
    let mut p = p0;

    times.push(t);
    qs.push(q.clone());
    ps.push(p.clone());

    while t < t_span.1 {
        let h = cfg.dt.min(t_span.1 - t);
        let half_h = h * 0.5;

        // p_{n+1/2} = p_n - (h/2) * grad_V(q_n)
        let gv = grad_v(&q);
        let p_half = &p - &(&gv * half_h);

        // q_{n+1} = q_n + h * p_{n+1/2} / mass
        q = &q + &(&p_half * (h * inv_mass));

        // p_{n+1} = p_{n+1/2} - (h/2) * grad_V(q_{n+1})
        let gv_new = grad_v(&q);
        p = &p_half - &(&gv_new * half_h);

        t += h;
        times.push(t);
        qs.push(q.clone());
        ps.push(p.clone());
    }

    Ok(SymplecticSolution {
        times,
        q_states: qs,
        p_states: ps,
    })
}

/// Parareal parallel-in-time solver for scalar ODEs.
pub fn parareal_solve<G, F>(
    t0: f64,
    t1: f64,
    y0: f64,
    config: &PararealConfig,
    coarse: G,
    fine: F,
) -> Result<Vec<f64>>
where
    G: Fn(f64, f64, f64) -> f64 + Sync,
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    use rayon::prelude::*;

    validate_parareal(t0, t1, config)?;

    let n = config.n_intervals;
    let dt = (t1 - t0) / n as f64;

    // Time grid: t[i] = t0 + i * dt
    let t_grid: Vec<f64> = (0..=n).map(|i| t0 + i as f64 * dt).collect();

    // Step 1: Sequential coarse sweep -> Y^0
    let mut y = vec![0.0; n + 1];
    y[0] = y0;
    for i in 0..n {
        y[i + 1] = coarse(t_grid[i], t_grid[i + 1], y[i]);
    }

    // Step 2: Parareal iterations
    for _ in 0..config.max_iter {
        // (a) Fine propagator in parallel: F(t_i, t_{i+1}, Y^k[i]) for each interval
        let fine_vals: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| fine(t_grid[i], t_grid[i + 1], y[i]))
            .collect();

        // (b) Coarse propagator (sequential, using current Y^k): G(t_i, t_{i+1}, Y^k[i])
        let coarse_vals: Vec<f64> = (0..n)
            .map(|i| coarse(t_grid[i], t_grid[i + 1], y[i]))
            .collect();

        // (c) Correction: Y^{k+1}[i+1] = F_i + G(t_i, t_{i+1}, Y^{k+1}[i]) - G_i
        let mut y_new = vec![0.0; n + 1];
        y_new[0] = y0;
        let mut max_diff = 0.0_f64;

        for i in 0..n {
            let g_new = coarse(t_grid[i], t_grid[i + 1], y_new[i]);
            y_new[i + 1] = fine_vals[i] + g_new - coarse_vals[i];
            let diff = (y_new[i + 1] - y[i + 1]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        y = y_new;

        // (d) Convergence check
        if max_diff < config.tol {
            break;
        }
    }

    Ok(y)
}

fn validate_parareal(t0: f64, t1: f64, config: &PararealConfig) -> Result<()> {
    if t0 >= t1 {
        return Err(Error::invalid("parareal_solve: t0 must be less than t1"));
    }
    let ensure_positive = |name: &'static str, value: usize| -> Result<()> {
        if value == 0 {
            Err(Error::invalid(format!(
                "parareal_solve: {name} must be > 0"
            )))
        } else {
            Ok(())
        }
    };
    ensure_positive("n_intervals", config.n_intervals)?;
    ensure_positive("max_iter", config.max_iter)?;
    Ok(())
}

/// Parareal parallel-in-time solver for tensor-valued ODEs.
pub fn parareal_solve_tensor<T, RG, RF, G, F>(
    t0: f64,
    t1: f64,
    y0: &Tensor<T, Cpu>,
    config: &PararealConfig,
    coarse: G,
    fine: F,
) -> Result<Vec<Tensor<T, Cpu>>>
where
    T: Scalar,
    RG: IntoOdeRhs<T, Cpu>,
    RF: IntoOdeRhs<T, Cpu>,
    G: Fn(f64, f64, &Tensor<T, Cpu>) -> RG + Sync,
    F: Fn(f64, f64, &Tensor<T, Cpu>) -> RF + Send + Sync,
{
    use rayon::prelude::*;

    let coarse = wrap_rhs_parareal(coarse);
    let fine = wrap_rhs_parareal(fine);
    validate_parareal(t0, t1, config)?;

    let n = config.n_intervals;
    let dt = (t1 - t0) / n as f64;

    // Time grid: t[i] = t0 + i * dt
    let t_grid: Vec<f64> = (0..=n).map(|i| t0 + i as f64 * dt).collect();

    // Step 1: Sequential coarse sweep -> Y^0
    let mut y: Vec<Tensor<T, Cpu>> = Vec::with_capacity(n + 1);
    y.push(y0.clone());
    for i in 0..n {
        let yi = coarse(t_grid[i], t_grid[i + 1], &y[i])?;
        y.push(yi);
    }

    // Step 2: Parareal iterations
    for _ in 0..config.max_iter {
        // (a) Fine propagator in parallel
        let fine_vals: Vec<Result<Tensor<T, Cpu>>> = (0..n)
            .into_par_iter()
            .map(|i| fine(t_grid[i], t_grid[i + 1], &y[i]))
            .collect();

        // Unwrap results.
        let fine_vals: Vec<Tensor<T, Cpu>> = fine_vals.into_iter().collect::<Result<Vec<_>>>()?;

        // (b) Coarse propagator (sequential, using current Y^k)
        let coarse_vals: Vec<Tensor<T, Cpu>> = (0..n)
            .map(|i| coarse(t_grid[i], t_grid[i + 1], &y[i]))
            .collect::<Result<Vec<_>>>()?;

        // (c) Correction: Y^{k+1}[i+1] = F_i + G(t_i, t_{i+1}, Y^{k+1}[i]) - G_i
        let mut y_new: Vec<Tensor<T, Cpu>> = Vec::with_capacity(n + 1);
        y_new.push(y0.clone());
        let mut max_diff = 0.0_f64;

        for i in 0..n {
            let g_new = coarse(t_grid[i], t_grid[i + 1], &y_new[i])?;
            let corrected = &(&fine_vals[i] + &g_new) - &coarse_vals[i];
            let diff = diff_inf_norm(&corrected, &y[i + 1]);
            if diff > max_diff {
                max_diff = diff;
            }
            y_new.push(corrected);
        }

        y = y_new;

        // (d) Convergence check
        if max_diff < config.tol {
            break;
        }
    }

    Ok(y)
}
