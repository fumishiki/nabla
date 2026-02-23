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

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::scalar::Scalar;
use crate::tensor::Tensor;

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

/// Convert an f64 Butcher-tableau / step-size constant to scalar type T.
#[inline]
fn sc<T: Scalar>(v: f64) -> T {
    T::from_f64(v)
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
            let abs_err: f64 = crate::scalar::math_utils::abs::<T>(&err.get(i, j)).into();
            let abs_y: f64 = crate::scalar::math_utils::abs::<T>(&y.get(i, j)).into();
            let abs_yn: f64 = crate::scalar::math_utils::abs::<T>(&y_new.get(i, j)).into();
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

    let capacity = ((t_span.1 - t_span.0) / dt).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity);

    let mut t = t_span.0;
    let mut y = y0.clone();

    times.push(t);
    states.push(y.clone());

    while t < t_span.1 {
        // Clamp step so we do not overshoot t_span.1.
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

    let capacity = ((t_span.1 - t_span.0) / dt).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity);

    let mut t = t_span.0;
    let mut y = y0.clone();

    times.push(t);
    states.push(y.clone());

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

    let mut times = Vec::new();
    let mut states = Vec::new();

    let mut t = t_span.0;
    let mut y = y0.clone();
    let mut dt = config
        .dt_init
        .min(config.dt_max)
        .max(config.dt_min)
        .min(t_span.1 - t_span.0);

    times.push(t);
    states.push(y.clone());

    // FSAL: compute k1 once, then reuse k7 of accepted step as k1 of next step.
    let mut k1 = f(t, &y)?;

    while t < t_span.1 {
        // Clamp step to not overshoot end of interval.
        let h = dt.min(t_span.1 - t);

        // Stage 2: t + h/5
        let y2 = &y + &(&k1 * sc::<T>(h * A21));
        let k2 = f(t + h / 5.0, &y2)?;

        // Stage 3: t + 3h/10
        let y3 = &(&y + &(&k1 * sc::<T>(h * A31))) + &(&k2 * sc::<T>(h * A32));
        let k3 = f(t + 3.0 * h / 10.0, &y3)?;

        // Stage 4: t + 4h/5
        let y4 = &(&(&y + &(&k1 * sc::<T>(h * A41))) + &(&k2 * sc::<T>(h * A42)))
            + &(&k3 * sc::<T>(h * A43));
        let k4 = f(t + 4.0 * h / 5.0, &y4)?;

        // Stage 5: t + 8h/9
        let y5 = &(&(&(&y + &(&k1 * sc::<T>(h * A51))) + &(&k2 * sc::<T>(h * A52)))
            + &(&k3 * sc::<T>(h * A53)))
            + &(&k4 * sc::<T>(h * A54));
        let k5 = f(t + 8.0 * h / 9.0, &y5)?;

        // Stage 6: t + h
        let y6 = &(&(&(&(&y + &(&k1 * sc::<T>(h * A61))) + &(&k2 * sc::<T>(h * A62)))
            + &(&k3 * sc::<T>(h * A63)))
            + &(&k4 * sc::<T>(h * A64)))
            + &(&k5 * sc::<T>(h * A65));
        let k6 = f(t + h, &y6)?;

        // 5th-order solution.
        let y_new = &(&(&(&(&y + &(&k1 * sc::<T>(h * B1))) + &(&k3 * sc::<T>(h * B3)))
            + &(&k4 * sc::<T>(h * B4)))
            + &(&k5 * sc::<T>(h * B5)))
            + &(&k6 * sc::<T>(h * B6));

        // Stage 7 (FSAL): evaluate at t + h with the 5th-order solution.
        let k7 = f(t + h, &y_new)?;

        // Error estimate: h * Σ_i e_i * k_i  (e_i = b_i - b*_i).
        let err = &(&(&(&(&(&k1 * sc::<T>(h * E1)) + &(&k3 * sc::<T>(h * E3)))
            + &(&k4 * sc::<T>(h * E4)))
            + &(&k5 * sc::<T>(h * E5)))
            + &(&k6 * sc::<T>(h * E6)))
            + &(&k7 * sc::<T>(h * E7));

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
