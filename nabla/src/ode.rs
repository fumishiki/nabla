// ode.rs — Ordinary Differential Equation solvers: Euler, RK4, Dormand-Prince RK45.
//
// All solvers accept a generic right-hand-side function
// `F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>` over any Scalar type T and Backend B.
//
// Step-size control (dt, t_span, error thresholds) is always f64.
// Tensor arithmetic uses the generic Backend interface.
//
// The module is `#[cfg(feature = "cpu")]`-gated by the lib.rs declaration.

#![allow(clippy::many_single_char_names)] // ODE variables (t, y, h, k, f) are standard notation.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // capacity hints
#![allow(clippy::missing_errors_doc)]

use nabla_core::backend::Backend;
#[cfg(feature = "cpu")]
use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;
#[cfg(feature = "cpu")]
use crate::linalg::LinalgExt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Solution trajectory returned by every ODE solver.
///
/// `times[i]` and `states[i]` correspond to each accepted step, starting from
/// the initial condition at `t_span.0`.
pub struct OdeSolution<T: Scalar, B: Backend> {
    /// Accepted time points (monotonically increasing).
    pub times: Vec<f64>,
    /// State vector at each accepted time point.
    pub states: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> OdeSolution<T, B> {
    /// Number of accepted steps stored in the solution.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` when no steps have been stored.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Borrow the state at the final accepted time point, or `None` if empty.
    #[must_use]
    #[inline]
    pub fn final_state(&self) -> Option<&Tensor<T, B>> {
        self.states.last()
    }
}

/// Configuration for the adaptive Dormand-Prince RK45 solver.
pub struct AdaptiveConfig {
    /// Absolute error tolerance.
    pub atol: f64,
    /// Relative error tolerance.
    pub rtol: f64,
    /// Minimum allowable step size.
    pub dt_min: f64,
    /// Maximum allowable step size.
    pub dt_max: f64,
    /// Initial trial step size.
    pub dt_init: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            atol: 1e-6,
            rtol: 1e-3,
            dt_min: 1e-12,
            dt_max: 1.0,
            dt_init: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate a fixed-step span/dt combination.
fn validate(t_span: (f64, f64), dt: f64) -> Result<()> {
    if t_span.0 >= t_span.1 {
        return Err(Error::invalid("t_span.0 must be less than t_span.1"));
    }
    if dt <= 0.0 {
        return Err(Error::invalid("dt must be positive"));
    }
    if dt > t_span.1 - t_span.0 {
        return Err(Error::invalid("dt exceeds t_span range"));
    }
    Ok(())
}

/// Validate an adaptive-step configuration.
fn validate_adaptive(t_span: (f64, f64), config: &AdaptiveConfig) -> Result<()> {
    if t_span.0 >= t_span.1 {
        return Err(Error::invalid("t_span.0 must be less than t_span.1"));
    }
    if config.dt_init <= 0.0 {
        return Err(Error::invalid("dt_init must be positive"));
    }
    if config.dt_min <= 0.0 {
        return Err(Error::invalid("dt_min must be positive"));
    }
    if config.dt_max <= 0.0 {
        return Err(Error::invalid("dt_max must be positive"));
    }
    if config.dt_min > config.dt_max {
        return Err(Error::invalid("dt_min must be <= dt_max"));
    }
    Ok(())
}

/// Estimate step count and return pre-allocated (times, states) vectors.
#[inline]
fn alloc_trajectory<T: Scalar, B: Backend>(
    t_span: (f64, f64),
    dt: f64,
    y0: &Tensor<T, B>,
) -> (Vec<f64>, Vec<Tensor<T, B>>) {
    let capacity = ((t_span.1 - t_span.0) / dt).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity);
    times.push(t_span.0);
    states.push(y0.clone());
    (times, states)
}

/// Convert an f64 Butcher-tableau / step-size constant to scalar type T.
#[inline]
fn sc<T: Scalar>(v: f64) -> T {
    T::from_f64(v)
}

/// `lincomb!(base; c1, k1; c2, k2; ...)` = `base + c1*k1 + c2*k2 + ...`
macro_rules! lincomb {
    ($base:expr; $($c:expr, $k:expr);+ $(;)?) => {{
        let mut _acc = $base.clone();
        $( _acc = &_acc + &(&$k * sc::<T>($c)); )+
        _acc
    }};
}

/// Mixed absolute/relative error norm over all elements.
///
/// `max_i |err_i| / (atol + rtol * max(|y_i|, |y_new_i|))`
///
/// Tensor elements are read back via `get` (GPU backends do a readback here, which is
/// acceptable because this function is used only for adaptive step-size control).
fn error_norm<T: Scalar, B: Backend>(
    err: &Tensor<T, B>,
    y: &Tensor<T, B>,
    y_new: &Tensor<T, B>,
    atol: f64,
    rtol: f64,
) -> f64
where
    T::Real: Into<f64>,
{
    let (rows, cols) = err.shape();
    let mut max_val = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let abs_err: f64 = nabla_core::scalar::math_utils::abs::<T>(&err.get(i, j)).into();
            let abs_y: f64 = nabla_core::scalar::math_utils::abs::<T>(&y.get(i, j)).into();
            let abs_yn: f64 = nabla_core::scalar::math_utils::abs::<T>(&y_new.get(i, j)).into();
            let scale = atol + rtol * abs_y.max(abs_yn);
            let e = abs_err / scale;
            if e > max_val {
                max_val = e;
            }
        }
    }
    max_val
}

// ---------------------------------------------------------------------------
// Euler — first-order fixed-step integrator
// ---------------------------------------------------------------------------

/// Forward Euler method: first-order, fixed step size `dt`.
///
/// Stores the state after every accepted step, beginning with `(t_span.0, y0)`.
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, or if `f` returns an error.
pub fn euler<T: Scalar, B: Backend, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
{
    validate(t_span, dt)?;
    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let h = dt.min(t_span.1 - t);
        let k = f(t, &y)?;
        y = &y + &(&k * sc::<T>(h));
        t += h;
        times.push(t);
        states.push(y.clone());
    }

    Ok(OdeSolution { times, states })
}

// ---------------------------------------------------------------------------
// RK4 — classic 4th-order fixed-step integrator
// ---------------------------------------------------------------------------

/// Classic 4-stage Runge-Kutta method: 4th-order, fixed step size `dt`.
///
/// Stores the state after every accepted step, beginning with `(t_span.0, y0)`.
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, or if `f` returns an error.
pub fn rk4<T: Scalar, B: Backend, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
{
    validate(t_span, dt)?;
    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let h = dt.min(t_span.1 - t);
        let h2 = h / 2.0;

        let k1 = f(t, &y)?;
        let k2 = f(t + h2, &(&y + &(&k1 * sc::<T>(h2))))?;
        let k3 = f(t + h2, &(&y + &(&k2 * sc::<T>(h2))))?;
        let k4 = f(t + h, &(&y + &(&k3 * sc::<T>(h))))?;

        // dy = (k1 + 2*k2 + 2*k3 + k4) * h/6
        let dy = &(&(&k1 + &(&k2 * sc::<T>(2.0))) + &(&k3 * sc::<T>(2.0))) + &k4;
        y = &y + &(&dy * sc::<T>(h / 6.0));
        t += h;

        times.push(t);
        states.push(y.clone());
    }

    Ok(OdeSolution { times, states })
}

// ---------------------------------------------------------------------------
// Dormand-Prince — adaptive RK45 with FSAL
// ---------------------------------------------------------------------------
//
// Butcher tableau (Dormand & Prince, 1980):
//
//   c2=1/5   | a21=1/5
//   c3=3/10  | a31=3/40      a32=9/40
//   c4=4/5   | a41=44/45     a42=−56/15    a43=32/9
//   c5=8/9   | a51=19372/6561 a52=−25360/2187 a53=64448/6561 a54=−212/729
//   c6=1     | a61=9017/3168  a62=−355/33   a63=46732/5247  a64=49/176  a65=−5103/18656
//   c7=1     | a71=35/384     a73=500/1113  a74=125/192     a75=−2187/6784 a76=11/84
//
// 5th-order output weights b (= a7.*):  same as row 7 above.
// 4th-order embedded weights b*:
//   bs1=5179/57600  bs3=7571/16695  bs4=393/640
//   bs5=−92097/339200  bs6=187/2100  bs7=1/40
//
// Error = h * Σ_i (b_i − b*_i) k_i  (FSAL: k1_next = k7_current)

// Dormand-Prince Butcher tableau constants.
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19_372.0 / 6_561.0;
const A52: f64 = -25_360.0 / 2_187.0;
const A53: f64 = 64_448.0 / 6_561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9_017.0 / 3_168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46_732.0 / 5_247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5_103.0 / 18_656.0;

// 5th-order weights (b = a7.*).
const B1: f64 = 35.0 / 384.0;
// B2 = 0
const B3: f64 = 500.0 / 1_113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2_187.0 / 6_784.0;
const B6: f64 = 11.0 / 84.0;
// B7 = 0 (unused in 5th-order output)

// 4th-order embedded weights (b*).
const BS1: f64 = 5_179.0 / 57_600.0;
// BS2 = 0
const BS3: f64 = 7_571.0 / 16_695.0;
const BS4: f64 = 393.0 / 640.0;
const BS5: f64 = -92_097.0 / 339_200.0;
const BS6: f64 = 187.0 / 2_100.0;
const BS7: f64 = 1.0 / 40.0;

// Error coefficient: e_i = b_i - b*_i.
const E1: f64 = B1 - BS1;
const E3: f64 = B3 - BS3;
const E4: f64 = B4 - BS4;
const E5: f64 = B5 - BS5;
const E6: f64 = B6 - BS6;
const E7: f64 = -BS7; // B7 = 0

/// Dormand-Prince adaptive RK45 solver with FSAL (First Same As Last).
///
/// Step size is controlled by a mixed absolute/relative error norm.  When a
/// step is accepted, `k1` for the next step reuses the last stage `k7` of the
/// current step (FSAL optimisation: 6 function evaluations per accepted step
/// instead of 7).
///
/// Stores the initial state plus every accepted state.
///
/// # Errors
///
/// Returns an error if the configuration is invalid or if `f` returns an error.
pub fn dormand_prince<T: Scalar, B: Backend, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &AdaptiveConfig,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    T::Real: Into<f64>,
{
    validate_adaptive(t_span, config)?;

    let mut times = vec![t_span.0];
    let mut states = vec![y0.clone()];

    let mut t = t_span.0;
    let mut y = y0.clone();
    let mut dt = config
        .dt_init
        .min(config.dt_max)
        .max(config.dt_min)
        .min(t_span.1 - t_span.0);

    // FSAL: compute k1 once, then reuse k7 of accepted step as k1 of next step.
    let mut k1 = f(t, &y)?;

    while t < t_span.1 {
        // Clamp step to not overshoot end of interval.
        let h = dt.min(t_span.1 - t);

        // Stage 2: t + h/5
        let k2 = f(t + h / 5.0, &lincomb!(y; h * A21, k1))?;

        // Stage 3: t + 3h/10
        let k3 = f(t + 3.0 * h / 10.0, &lincomb!(y; h * A31, k1; h * A32, k2))?;

        // Stage 4: t + 4h/5
        let k4 = f(t + 4.0 * h / 5.0, &lincomb!(y; h * A41, k1; h * A42, k2; h * A43, k3))?;

        // Stage 5: t + 8h/9
        let k5 = f(t + 8.0 * h / 9.0, &lincomb!(y; h * A51, k1; h * A52, k2; h * A53, k3; h * A54, k4))?;

        // Stage 6: t + h
        let k6 = f(t + h, &lincomb!(y; h * A61, k1; h * A62, k2; h * A63, k3; h * A64, k4; h * A65, k5))?;

        // 5th-order solution.
        let y_new = lincomb!(y; h * B1, k1; h * B3, k3; h * B4, k4; h * B5, k5; h * B6, k6);

        // Stage 7 (FSAL): evaluate at t + h with the 5th-order solution.
        let k7 = f(t + h, &y_new)?;

        // Error estimate: h * Σ_i e_i * k_i  (e_i = b_i - b*_i).
        let err_base = &k1 * sc::<T>(h * E1);
        let err = lincomb!(err_base; h * E3, k3; h * E4, k4; h * E5, k5; h * E6, k6; h * E7, k7);

        let err_norm = error_norm(&err, &y, &y_new, config.atol, config.rtol);

        // Accept or reject the step.
        if err_norm <= 1.0 {
            // Advance state.
            t += h;
            y = y_new;
            times.push(t);
            states.push(y.clone());

            // FSAL: k1 for next step = k7 of this step.
            k1 = k7;
        }

        // Adjust step size: PI controller with safety factor 0.9.
        // factor = 0.9 * err_norm^(-1/5), clamped to [0.2, 5.0].
        let factor = if err_norm == 0.0 {
            5.0_f64
        } else {
            (0.9 * err_norm.powf(-0.2)).clamp(0.2, 5.0)
        };

        dt = (h * factor).clamp(config.dt_min, config.dt_max);

        // Safety: avoid infinite loop if step cannot advance past dt_min.
        if h <= config.dt_min && err_norm > 1.0 {
            return Err(Error::invalid(
                "step size reached dt_min with error above tolerance; system may be stiff",
            ));
        }
    }

    Ok(OdeSolution { times, states })
}

// ---------------------------------------------------------------------------
// IF Euler — exponential integrator for stiff ODEs
// ---------------------------------------------------------------------------

/// `phi_1(z) = (e^z - 1) / z`, with `phi_1(0) = 1`.
#[cfg(feature = "cpu")]
#[inline]
fn phi1(z: f64) -> f64 {
    if z.abs() < 1e-8 { 1.0 } else { z.exp_m1() / z }
}

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
#[cfg(feature = "cpu")]
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

    let n = a.nrows();
    if a.ncols() != n {
        return Err(Error::invalid("if_euler: A must be square"));
    }
    if y0.shape() != (n, 1) {
        return Err(Error::mismatch((n, 1), y0.shape()));
    }

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
    let exp_ha = Tensor::<f64, Cpu>::from_fn(n, n, |i, j| {
        (0..n).map(|k| v.get(i, k) * exp_diag[k] * v.get(j, k)).sum()
    });
    // phi1_ha = V * diag(phi1_diag) * V^T
    let phi1_ha = Tensor::<f64, Cpu>::from_fn(n, n, |i, j| {
        (0..n).map(|k| v.get(i, k) * phi1_diag[k] * v.get(j, k)).sum()
    });

    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let h = dt.min(t_span.1 - t);
        let gv = g(t, &y)?;

        if (h - dt).abs() < 1e-14 {
            // Standard step: use precomputed matrices
            y = &(&exp_ha * &y) + &(&(&phi1_ha * &gv) * h);
        } else {
            // Final fractional step: recompute for this h
            let exp_d: Vec<f64> = lambdas.iter().map(|&l| (h * l).exp()).collect();
            let phi_d: Vec<f64> = lambdas.iter().map(|&l| phi1(h * l)).collect();
            let exp_h = Tensor::<f64, Cpu>::from_fn(n, n, |i, j| {
                (0..n).map(|k| v.get(i, k) * exp_d[k] * v.get(j, k)).sum()
            });
            let phi_h = Tensor::<f64, Cpu>::from_fn(n, n, |i, j| {
                (0..n).map(|k| v.get(i, k) * phi_d[k] * v.get(j, k)).sum()
            });
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

/// Configuration for the BDF-1 (Backward Euler) implicit solver.
pub struct Bdf1Config {
    /// Fixed step size.
    pub dt: f64,
    /// Fixed-point iteration convergence tolerance.
    pub tol: f64,
    /// Maximum fixed-point iterations per step.
    pub max_iter: usize,
}

impl Default for Bdf1Config {
    fn default() -> Self {
        Self { dt: 1e-3, tol: 1e-8, max_iter: 50 }
    }
}

/// Infinity norm of the difference between two tensors: `max |a_ij - b_ij|`.
#[cfg(feature = "cpu")]
fn diff_inf_norm<T: Scalar>(a: &Tensor<T, Cpu>, b: &Tensor<T, Cpu>) -> f64 {
    let (rows, cols) = a.shape();
    let mut mx = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let d: f64 = nabla_core::scalar::math_utils::abs(
                &(a.get(i, j) - b.get(i, j)),
            )
            .into();
            if d > mx {
                mx = d;
            }
        }
    }
    mx
}

/// BDF-1 (Backward Euler) solver for stiff ODEs: `dy/dt = f(t, y)`.
///
/// Solves the implicit equation `y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})`
/// via fixed-point (Picard) iteration at each step.
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, if `f` returns an error,
/// or if the fixed-point iteration fails to converge within `max_iter`.
#[cfg(feature = "cpu")]
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
        let mut y_new = y.clone();
        let mut converged = false;
        for _ in 0..config.max_iter {
            let fy = f(t_next, &y_new)?;
            let y_next = &y + &(&fy * h_t);
            let norm = diff_inf_norm(&y_next, &y_new);
            y_new = y_next;
            if norm < config.tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(Error::invalid(
                "bdf1: fixed-point iteration did not converge — reduce dt or increase max_iter",
            ));
        }

        t = t_next;
        y = y_new;
        times.push(t);
        states.push(y.clone());
    }

    Ok(OdeSolution { times, states })
}

// ---------------------------------------------------------------------------
// DAE — semi-explicit index-1 differential-algebraic equation solver
// ---------------------------------------------------------------------------

/// Configuration for the semi-explicit index-1 DAE solver.
#[derive(Clone, Copy)]
pub struct DaeConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Newton iteration convergence tolerance.
    pub tol: f64,
    /// Maximum Newton iterations per step.
    pub max_iter: usize,
}

impl Default for DaeConfig {
    fn default() -> Self {
        Self { dt: 0.01, tol: 1e-8, max_iter: 50 }
    }
}

/// Semi-explicit index-1 DAE solver using BDF-1 predictor + Newton corrector.
///
/// Solves the system:
///   `x'(t) = f(x, z, t)`  — differential variables (n_x-dim column vector)
///   `0     = g(x, z, t)`  — algebraic constraint  (n_z-dim column vector)
///
/// Algorithm per step:
///   1. Predictor: `x_pred = x_n + dt * f(x_n, z_n, t_n)`
///   2. Newton for z: solve `g(x_pred, z, t_{n+1}) = 0` via finite-difference Jacobian
///   3. Corrector: `x_{n+1} = x_n + dt * f(x_pred, z_{n+1}, t_{n+1})`
///
/// Returns the trajectory of x (differential variables) in `OdeSolution`.
///
/// # Errors
///
/// Returns an error if `t_span`/`dt` are invalid, if `f`/`g` return errors,
/// if the Jacobian solve fails, or if Newton iteration does not converge.
#[cfg(feature = "cpu")]
pub fn dae_solve<F, G>(
    f: F,
    g: G,
    x0: Tensor<f64, Cpu>,
    z0: Tensor<f64, Cpu>,
    t_span: (f64, f64),
    cfg: DaeConfig,
) -> Result<OdeSolution<f64, Cpu>>
where
    F: Fn(&Tensor<f64, Cpu>, &Tensor<f64, Cpu>, f64) -> Tensor<f64, Cpu>,
    G: Fn(&Tensor<f64, Cpu>, &Tensor<f64, Cpu>, f64) -> Tensor<f64, Cpu>,
{
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

            // Finite-difference Jacobian J_z ∈ ℝ^{n_z × n_z}
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

    Ok(OdeSolution { times, states })
}

/// Infinity norm of a column vector: `max |v_i|`.
#[cfg(feature = "cpu")]
fn inf_norm_vec(v: &Tensor<f64, Cpu>) -> f64 {
    let n = v.nrows();
    let mut mx = 0.0_f64;
    for i in 0..n {
        let a = v.get(i, 0).abs();
        if a > mx {
            mx = a;
        }
    }
    mx
}

// ---------------------------------------------------------------------------
// IF Euler Scalar — exponential integrator for scalar-stiff ODEs
// ---------------------------------------------------------------------------

/// Configuration for the scalar integrating-factor Euler solver.
pub struct IfEulerScalarConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Stiffness coefficient lambda (the linear coefficient in `y' = -lambda*y + N(t,y)`).
    pub stiffness: f64,
}

impl Default for IfEulerScalarConfig {
    fn default() -> Self {
        Self { dt: 1e-3, stiffness: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// METD — Matrix Exponential Time Differencing (1st order)
// ---------------------------------------------------------------------------

/// Configuration for the METD (Matrix Exponential Time Differencing) solver.
#[derive(Clone, Copy)]
pub struct MetdConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Number of terms in the phi_1 series expansion (default 8).
    pub order: usize,
}

impl Default for MetdConfig {
    fn default() -> Self {
        Self { dt: 0.01, order: 8 }
    }
}

/// phi_1(Z) via truncated Taylor series: `sum_{k=0}^{K} Z^k / (k+1)!`
///
/// Numerically stable for any Z (avoids computing inv(Z)).
#[cfg(feature = "cpu")]
fn phi1_matrix(z: &Tensor<f64, Cpu>, order: usize) -> Tensor<f64, Cpu> {
    let n = z.nrows();
    let mut result = Tensor::<f64, Cpu>::identity(n); // k=0 term: I / 1!
    let mut zk = Tensor::<f64, Cpu>::identity(n);     // Z^0

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
#[cfg(feature = "cpu")]
pub fn metd_solve<F>(
    l: &Tensor<f64, Cpu>,
    nonlinear: F,
    y0: Tensor<f64, Cpu>,
    t_span: (f64, f64),
    cfg: MetdConfig,
) -> Result<OdeSolution<f64, Cpu>>
where
    F: Fn(f64, &Tensor<f64, Cpu>) -> Tensor<f64, Cpu>,
{
    validate(t_span, cfg.dt)?;

    let n = l.nrows();
    if l.ncols() != n {
        return Err(Error::invalid("metd_solve: L must be square"));
    }
    if y0.nrows() != n || y0.ncols() != 1 {
        return Err(Error::mismatch((n, 1), y0.shape()));
    }

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

        if (h - cfg.dt).abs() < 1e-14 {
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

    Ok(OdeSolution { times, states })
}

// ---------------------------------------------------------------------------
// Störmer-Verlet — symplectic integrator for Hamiltonian systems
// ---------------------------------------------------------------------------

/// Configuration for the Störmer-Verlet (leapfrog) symplectic integrator.
#[derive(Clone, Copy)]
pub struct StormerVerletConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Particle mass (default 1.0). Used as grad_T(p) = p / mass.
    pub mass: f64,
}

impl Default for StormerVerletConfig {
    fn default() -> Self {
        Self { dt: 0.01, mass: 1.0 }
    }
}

/// Störmer-Verlet (leapfrog) symplectic integrator for Hamiltonian systems.
///
/// Solves `H(q, p) = T(p) + V(q)` where `T(p) = p^2 / (2m)`.
///
/// Algorithm:
///   `p_{n+1/2} = p_n - (h/2) * grad_V(q_n)`
///   `q_{n+1}   = q_n + h * p_{n+1/2} / mass`
///   `p_{n+1}   = p_{n+1/2} - (h/2) * grad_V(q_{n+1})`
///
/// Returns `(times, q_trajectory, p_trajectory)`.
///
/// # Errors
///
/// Returns an error if the time span or step size are invalid.
#[cfg(feature = "cpu")]
pub fn stormer_verlet<V>(
    grad_v: V,
    q0: Tensor<f64, Cpu>,
    p0: Tensor<f64, Cpu>,
    t_span: (f64, f64),
    cfg: StormerVerletConfig,
) -> Result<(Vec<f64>, Vec<Tensor<f64, Cpu>>, Vec<Tensor<f64, Cpu>>)>
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

    Ok((times, qs, ps))
}

// ---------------------------------------------------------------------------
// Parareal — parallel-in-time ODE solver
// ---------------------------------------------------------------------------

/// Configuration for the Parareal parallel-in-time ODE solver.
pub struct PararealConfig {
    /// Number of time sub-intervals (= rayon parallelism degree).
    pub n_intervals: usize,
    /// Maximum Parareal correction iterations (3-10 typical).
    pub max_iter: usize,
    /// Convergence tolerance: `max |Y^{k+1}[i] - Y^k[i]| < tol`.
    pub tol: f64,
}

impl Default for PararealConfig {
    fn default() -> Self {
        Self { n_intervals: 8, max_iter: 10, tol: 1e-6 }
    }
}

/// Parareal parallel-in-time ODE solver for scalar ODEs.
///
/// Decomposes `[t0, t1]` into `n_intervals` sub-intervals, using a cheap
/// *coarse* propagator `G` for sequential sweeps and an accurate *fine*
/// propagator `F` evaluated in parallel via rayon.
///
/// Both propagators have signature `fn(t_start, t_end, y_start) -> y_end`.
///
/// Returns a `Vec<f64>` of `n_intervals + 1` checkpoint values at the
/// sub-interval boundaries (including the initial condition at `t0`).
///
/// # Errors
///
/// Returns an error if `t0 >= t1`, `n_intervals == 0`, or `max_iter == 0`.
#[cfg(feature = "cpu")]
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

    if t0 >= t1 {
        return Err(Error::invalid("parareal_solve: t0 must be less than t1"));
    }
    if config.n_intervals == 0 {
        return Err(Error::invalid("parareal_solve: n_intervals must be > 0"));
    }
    if config.max_iter == 0 {
        return Err(Error::invalid("parareal_solve: max_iter must be > 0"));
    }

    let n = config.n_intervals;
    let dt = (t1 - t0) / n as f64;

    // Time grid: t[i] = t0 + i * dt
    let t_grid: Vec<f64> = (0..=n).map(|i| t0 + i as f64 * dt).collect();

    // Step 1: Sequential coarse sweep → Y^0
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
#[cfg(feature = "cpu")]
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
    let z = -config.stiffness * h;
    let exp_z = z.exp();
    let phi1_val = phi1(z);

    let exp_z_t: T = sc(exp_z);
    let h_phi1_t: T = sc(h * phi1_val);

    let (mut times, mut states) = alloc_trajectory(t_span, h, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();

    while t < t_span.1 {
        let step = h.min(t_span.1 - t);
        let ny = nonlinear(t, &y)?;

        if (step - h).abs() < 1e-14 {
            // Standard step: use precomputed scalars
            y = &(&y * exp_z_t) + &(&ny * h_phi1_t);
        } else {
            // Final fractional step: recompute for this step size
            let zf = -config.stiffness * step;
            let exp_zf_t: T = sc(zf.exp());
            let h_phi1f_t: T = sc(step * phi1(zf));
            y = &(&y * exp_zf_t) + &(&ny * h_phi1f_t);
        }

        t += step;
        times.push(t);
        states.push(y.clone());
    }

    Ok(OdeSolution { times, states })
}
