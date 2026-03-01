#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]

#[cfg(feature = "cpu")]
mod advanced;
#[cfg(feature = "cpu")]
mod sde;
#[cfg(feature = "cpu")]
mod stiff;

#[cfg(feature = "cpu")]
pub use advanced::*;
#[cfg(feature = "cpu")]
pub use sde::*;
#[cfg(feature = "cpu")]
pub use stiff::*;

use nabla_core::backend::Backend;
#[cfg(feature = "cpu")]
use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

/// Allows ODE right-hand-side closures to return `Tensor<T,B>` or `Result<Tensor<T,B>>`.
pub trait IntoOdeRhs<T: Scalar, B: Backend> {
    /// Convert into a `Result<Tensor>` for uniform RHS handling.
    fn into_rhs(self) -> Result<Tensor<T, B>>;
}

impl<T: Scalar, B: Backend> IntoOdeRhs<T, B> for Tensor<T, B> {
    #[inline]
    fn into_rhs(self) -> Result<Tensor<T, B>> {
        Ok(self)
    }
}

impl<T: Scalar, B: Backend> IntoOdeRhs<T, B> for Result<Tensor<T, B>> {
    #[inline]
    fn into_rhs(self) -> Result<Tensor<T, B>> {
        self
    }
}

pub(crate) fn wrap_rhs<T, B, R, F>(f: F) -> impl Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>
where
    T: Scalar,
    B: Backend,
    R: IntoOdeRhs<T, B>,
    F: Fn(f64, &Tensor<T, B>) -> R,
{
    move |t, y| f(t, y).into_rhs()
}

/// Container for ODE solver output (times and states).
pub struct OdeSolution<T: Scalar, B: Backend> {
    /// Time points at which the solution was recorded.
    pub times: Vec<f64>,
    /// State tensors corresponding to each time point.
    pub states: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> OdeSolution<T, B> {
    /// Number of time points in the solution.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the solution contains no time points.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Reference to the last computed state, if any.
    #[must_use]
    #[inline]
    pub fn final_state(&self) -> Option<&Tensor<T, B>> {
        self.states.last()
    }

    /// Evaluate the solution at time `t` via linear interpolation.
    #[must_use]
    pub fn eval(&self, t: f64) -> Tensor<T, B> {
        let n = self.times.len();
        assert!(n > 0, "OdeSolution::eval called on empty solution");

        if n == 1 || t <= self.times[0] {
            return self.states[0].clone();
        }
        if t >= self.times[n - 1] {
            return self.states[n - 1].clone();
        }

        let idx = match self
            .times
            .binary_search_by(|probe| probe.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => return self.states[i].clone(),
            Err(i) => i - 1,
        };

        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let dt_seg = t1 - t0;

        if dt_seg.abs() < 1e-15 {
            return self.states[idx].clone();
        }

        let alpha = (t - t0) / dt_seg;
        let one_minus: T = sc(1.0 - alpha);
        let alpha_t: T = sc(alpha);
        &(&self.states[idx] * one_minus) + &(&self.states[idx + 1] * alpha_t)
    }
}

/// Container for symplectic integrator output (positions and momenta).
pub struct SymplecticSolution<T: Scalar, B: Backend> {
    /// Time points at which the solution was recorded.
    pub times: Vec<f64>,
    /// Generalized position states at each time point.
    pub q_states: Vec<Tensor<T, B>>,
    /// Generalized momentum states at each time point.
    pub p_states: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> SymplecticSolution<T, B> {
    /// Number of time points in the solution.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the solution contains no time points.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Reference to the last (q, p) pair, if any.
    #[must_use]
    #[inline]
    pub fn final_state(&self) -> Option<(&Tensor<T, B>, &Tensor<T, B>)> {
        self.q_states
            .last()
            .and_then(|q| self.p_states.last().map(|p| (q, p)))
    }
}

impl<T: Scalar, B: Backend> std::ops::Index<usize> for OdeSolution<T, B> {
    type Output = Tensor<T, B>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

/// Configuration for adaptive-step solvers (Dormand-Prince RK45).
pub struct AdaptiveConfig {
    /// Absolute error tolerance.
    pub atol: f64,
    /// Relative error tolerance.
    pub rtol: f64,
    /// Minimum allowed step size.
    pub dt_min: f64,
    /// Maximum allowed step size.
    pub dt_max: f64,
    /// Initial step size.
    pub dt_init: f64,
    /// Optional output times for interpolated recording.
    pub saveat: Option<Vec<f64>>,
    /// Optional early termination predicate on time.
    pub terminate: Option<Box<dyn Fn(f64) -> bool>>,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            atol: 1e-6,
            rtol: 1e-3,
            dt_min: 1e-12,
            dt_max: 1.0,
            dt_init: 0.01,
            saveat: None,
            terminate: None,
        }
    }
}

impl AdaptiveConfig {
    /// Set the initial step size.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt_init = dt;
        self
    }

    /// Set absolute and relative tolerances.
    pub fn with_tol(mut self, atol: f64, rtol: f64) -> Self {
        self.atol = atol;
        self.rtol = rtol;
        self
    }

    /// Record solution only at specific output times.
    pub fn with_saveat(mut self, times: Vec<f64>) -> Self {
        self.saveat = Some(times);
        self
    }
}

/// Configuration for the BDF-1 (Backward Euler) implicit solver.
pub struct Bdf1Config {
    /// Fixed step size.
    pub dt: f64,
    /// Fixed-point iteration convergence tolerance.
    pub tol: f64,
    /// Maximum fixed-point iterations per step.
    pub max_iter: usize,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Default for Bdf1Config {
    fn default() -> Self {
        Self {
            dt: 1e-3,
            tol: 1e-8,
            max_iter: 50,
            saveat: None,
        }
    }
}

impl Bdf1Config {
    /// Set the fixed step size.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set the convergence tolerance and maximum iterations.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set specific output times for the solution.
    pub fn with_saveat(mut self, times: Vec<f64>) -> Self {
        self.saveat = Some(times);
        self
    }
}

/// Configuration for the BDF-2 (second-order BDF) implicit solver.
pub struct Bdf2Config {
    /// Fixed step size.
    pub dt: f64,
    /// Fixed-point iteration convergence tolerance.
    pub tol: f64,
    /// Maximum fixed-point iterations per step.
    pub max_iter: usize,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Default for Bdf2Config {
    fn default() -> Self {
        Self {
            dt: 1e-3,
            tol: 1e-8,
            max_iter: 50,
            saveat: None,
        }
    }
}

impl Bdf2Config {
    /// Set the fixed step size.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set specific output times for the solution.
    pub fn with_saveat(mut self, times: Vec<f64>) -> Self {
        self.saveat = Some(times);
        self
    }
}

/// Configuration for the DAE (differential-algebraic equation) solver.
#[derive(Clone)]
pub struct DaeConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Newton iteration convergence tolerance.
    pub tol: f64,
    /// Maximum Newton iterations per step.
    pub max_iter: usize,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Default for DaeConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            tol: 1e-8,
            max_iter: 50,
            saveat: None,
        }
    }
}

impl DaeConfig {
    /// Set the fixed step size.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }
    /// Set the Newton convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the maximum Newton iterations per step.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Record solution only at specific output times.
    pub fn with_saveat(mut self, times: Vec<f64>) -> Self {
        self.saveat = Some(times);
        self
    }
}

/// Configuration for the scalar integrating-factor Euler method.
pub struct IfEulerScalarConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Stiffness coefficient lambda (the linear coefficient in `y' = -lambda*y + N(t,y)`).
    pub stiffness: f64,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Default for IfEulerScalarConfig {
    fn default() -> Self {
        Self {
            dt: 1e-3,
            stiffness: 1.0,
            saveat: None,
        }
    }
}

/// Configuration for the matrix exponential time-differencing solver.
#[derive(Clone)]
pub struct MetdConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Number of terms in the phi_1 series expansion (default 8).
    pub order: usize,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Default for MetdConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            order: 8,
            saveat: None,
        }
    }
}

/// Configuration for the Stormer-Verlet symplectic integrator.
#[derive(Clone)]
pub struct StormerVerletConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Particle mass (default 1.0). Used as grad_T(p) = p / mass.
    pub mass: f64,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Default for StormerVerletConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            mass: 1.0,
            saveat: None,
        }
    }
}

/// Configuration for the Parareal parallel-in-time solver.
pub struct PararealConfig {
    /// Number of time sub-intervals (= rayon parallelism degree).
    pub n_intervals: usize,
    /// Maximum Parareal correction iterations (3-10 typical).
    pub max_iter: usize,
    /// Convergence tolerance: `max |Y^{k+1}[i] - Y^k[i]| < tol`.
    pub tol: f64,
    /// Optional output times (not used by parareal_solve, included for API consistency).
    pub saveat: Option<Vec<f64>>,
}

impl Default for PararealConfig {
    fn default() -> Self {
        Self {
            n_intervals: 8,
            max_iter: 10,
            tol: 1e-6,
            saveat: None,
        }
    }
}

#[inline]
fn require_positive(name: &str, v: f64) -> Result<()> {
    if v <= 0.0 {
        return Err(Error::invalid(format!("{name} must be positive")));
    }
    Ok(())
}

pub(crate) fn validate(t_span: (f64, f64), dt: f64) -> Result<()> {
    validate_span(t_span)?;
    require_positive("dt", dt)?;
    let span_len = (t_span.1 - t_span.0).abs();
    if dt > span_len {
        return Err(Error::invalid("dt exceeds t_span range"));
    }
    Ok(())
}

fn validate_adaptive(t_span: (f64, f64), config: &AdaptiveConfig) -> Result<()> {
    validate_span(t_span)?;
    require_positive("dt_init", config.dt_init)?;
    require_positive("dt_min", config.dt_min)?;
    require_positive("dt_max", config.dt_max)?;
    if config.dt_min > config.dt_max {
        return Err(Error::invalid("dt_min must be <= dt_max"));
    }
    Ok(())
}

fn validate_span(t_span: (f64, f64)) -> Result<()> {
    if (t_span.1 - t_span.0).abs() < 1e-15 {
        return Err(Error::invalid("t_span.0 and t_span.1 must differ"));
    }
    Ok(())
}

#[inline]
pub(crate) fn alloc_trajectory<T: Scalar, B: Backend>(
    t_span: (f64, f64),
    dt: f64,
    y0: &Tensor<T, B>,
) -> (Vec<f64>, Vec<Tensor<T, B>>) {
    let capacity = ((t_span.1 - t_span.0).abs() / dt).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity);
    times.push(t_span.0);
    states.push(y0.clone());
    (times, states)
}

#[inline]
pub(crate) fn time_direction(t_span: (f64, f64)) -> f64 {
    if t_span.1 >= t_span.0 { 1.0 } else { -1.0 }
}

#[inline]
pub(crate) fn sc<T: Scalar>(v: f64) -> T {
    T::from_f64(v)
}

macro_rules! lincomb {
    ($base:expr; $($c:expr, $k:expr);+ $(;)?) => {{
        let mut _acc = $base.clone();
        $( _acc = &_acc + &(&$k * sc::<T>($c)); )+
        _acc
    }};
}

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

#[cfg(feature = "cpu")]
#[inline]
pub(crate) fn phi1(z: f64) -> f64 {
    if z.abs() < 1e-8 { 1.0 } else { z.exp_m1() / z }
}

#[cfg(feature = "cpu")]
pub(crate) fn diff_inf_norm<T: Scalar>(a: &Tensor<T, Cpu>, b: &Tensor<T, Cpu>) -> f64 {
    let (rows, cols) = a.shape();
    let mut mx = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let d: f64 = nabla_core::scalar::math_utils::abs(&(a.get(i, j) - b.get(i, j))).into();
            if d > mx {
                mx = d;
            }
        }
    }
    mx
}

#[cfg(feature = "cpu")]
pub(crate) fn inf_norm_vec(v: &Tensor<f64, Cpu>) -> f64 {
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

#[allow(clippy::ref_option)]
pub(crate) fn apply_saveat<T: Scalar, B: Backend>(
    sol: OdeSolution<T, B>,
    saveat: &Option<Vec<f64>>,
) -> OdeSolution<T, B> {
    let requested = match saveat {
        Some(v) if !v.is_empty() => v,
        _ => return sol,
    };
    if sol.times.len() < 2 {
        return sol;
    }

    let mut out_times = Vec::with_capacity(requested.len());
    let mut out_states = Vec::with_capacity(requested.len());
    let mut idx = 0_usize; // cursor into sol.times

    for &t_req in requested {
        // Advance cursor so sol.times[idx] <= t_req < sol.times[idx+1]
        // (or idx is at the last segment).
        while idx + 1 < sol.times.len() - 1 && sol.times[idx + 1] < t_req {
            idx += 1;
        }
        let t0 = sol.times[idx];
        let t1 = sol.times[idx + 1];
        let dt_seg = t1 - t0;

        if dt_seg.abs() < 1e-15 {
            // Degenerate segment: just copy the left state.
            out_times.push(t_req);
            out_states.push(sol.states[idx].clone());
        } else {
            let alpha = ((t_req - t0) / dt_seg).clamp(0.0, 1.0);
            let alpha_t: T = sc(alpha);
            let one_minus: T = sc(1.0 - alpha);
            // y = (1 - alpha) * y0 + alpha * y1
            let y_interp = &(&sol.states[idx] * one_minus) + &(&sol.states[idx + 1] * alpha_t);
            out_times.push(t_req);
            out_states.push(y_interp);
        }
    }

    OdeSolution {
        times: out_times,
        states: out_states,
    }
}

/// Configuration for the explicit Euler method.
pub struct EulerConfig {
    /// Fixed step size.
    pub dt: f64,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl EulerConfig {
    /// Create a new Euler configuration with the given step size.
    #[must_use]
    pub fn new(dt: f64) -> Self {
        Self { dt, saveat: None }
    }

    /// Set specific output times for the solution.
    #[must_use]
    pub fn with_saveat(mut self, saveat: Vec<f64>) -> Self {
        self.saveat = Some(saveat);
        self
    }
}

/// Configuration for the classic RK4 method.
pub struct Rk4Config {
    /// Fixed step size.
    pub dt: f64,
    /// Optional output times. When `Some`, only record solution at these times
    /// (using linear interpolation); when `None`, record every accepted step.
    pub saveat: Option<Vec<f64>>,
}

impl Rk4Config {
    /// Create a new RK4 configuration with the given step size.
    #[must_use]
    pub fn new(dt: f64) -> Self {
        Self { dt, saveat: None }
    }

    /// Set specific output times for the solution.
    #[must_use]
    pub fn with_saveat(mut self, saveat: Vec<f64>) -> Self {
        self.saveat = Some(saveat);
        self
    }
}

/// Bundled ODE initial-value problem (RHS, initial state, time span).
pub struct OdeProblem<T: Scalar, B: Backend, F> {
    /// Right-hand side function `f(t, y) -> dy/dt`.
    pub f: F,
    /// Initial state at `t_span.0`.
    pub y0: Tensor<T, B>,
    /// Integration interval `(t_start, t_end)`.
    pub t_span: (f64, f64),
}

impl<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F> OdeProblem<T, B, F>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
{
    /// Create a new ODE problem.
    #[must_use]
    pub fn new(f: F, y0: Tensor<T, B>, t_span: (f64, f64)) -> Self {
        Self { f, y0, t_span }
    }

    /// Solve with the Euler method using a fixed step size.
    pub fn solve_euler(self, dt: f64) -> Result<OdeSolution<T, B>> {
        euler(self.f, &self.y0, self.t_span, dt)
    }

    /// Solve with the Euler method using a configuration struct.
    pub fn solve_euler_config(self, config: &EulerConfig) -> Result<OdeSolution<T, B>> {
        euler_with_config(self.f, &self.y0, self.t_span, config)
    }

    /// Solve with the classic RK4 method using a fixed step size.
    pub fn solve_rk4(self, dt: f64) -> Result<OdeSolution<T, B>> {
        rk4(self.f, &self.y0, self.t_span, dt)
    }

    /// Solve with the classic RK4 method using a configuration struct.
    pub fn solve_rk4_config(self, config: &Rk4Config) -> Result<OdeSolution<T, B>> {
        rk4_with_config(self.f, &self.y0, self.t_span, config)
    }
}

impl<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F> OdeProblem<T, B, F>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
    T::Real: Into<f64>,
{
    /// Solve with the adaptive Dormand-Prince RK45 method.
    pub fn solve_adaptive(self, config: &AdaptiveConfig) -> Result<OdeSolution<T, B>> {
        dormand_prince(self.f, &self.y0, self.t_span, config)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, R: IntoOdeRhs<T, Cpu>, F> OdeProblem<T, Cpu, F>
where
    F: Fn(f64, &Tensor<T, Cpu>) -> R,
{
    /// Solve with BDF-1 (Backward Euler) implicit method.
    pub fn solve_bdf1(self, config: &Bdf1Config) -> Result<OdeSolution<T, Cpu>> {
        bdf1(self.f, &self.y0, self.t_span, config)
    }

    /// Solve with BDF-2 (second-order BDF) implicit method.
    pub fn solve_bdf2(self, config: &Bdf2Config) -> Result<OdeSolution<T, Cpu>> {
        bdf2(self.f, &self.y0, self.t_span, config)
    }

    /// Solve with the scalar integrating-factor Euler method for stiff ODEs.
    pub fn solve_if_euler_scalar(
        self,
        config: &IfEulerScalarConfig,
    ) -> Result<OdeSolution<T, Cpu>> {
        if_euler_scalar(self.f, &self.y0, self.t_span, config)
    }
}

/// Explicit Euler method for initial value problems.
pub fn euler<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
{
    let f = wrap_rhs(f);
    validate(t_span, dt)?;
    let dir = time_direction(t_span);
    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();
    let remaining = |t: f64| (t_span.1 - t) * dir;

    while remaining(t) > 1e-14 {
        let h = dir * dt.min(remaining(t).abs());
        let k = f(t, &y)?;
        y = &y + &(&k * sc::<T>(h));
        t += h;
        times.push(t);
        states.push(y.clone());
    }

    Ok(OdeSolution { times, states })
}

/// Euler method with configuration struct and saveat support.
pub fn euler_with_config<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &EulerConfig,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
{
    let f = wrap_rhs(f);
    let sol = euler(f, y0, t_span, config.dt)?;
    Ok(apply_saveat(sol, &config.saveat))
}

/// Classic fourth-order Runge-Kutta method.
pub fn rk4<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
{
    let f = wrap_rhs(f);
    validate(t_span, dt)?;
    let dir = time_direction(t_span);
    let (mut times, mut states) = alloc_trajectory(t_span, dt, y0);
    let mut t = t_span.0;
    let mut y = y0.clone();
    let remaining = |t: f64| (t_span.1 - t) * dir;

    while remaining(t) > 1e-14 {
        let h = dir * dt.min(remaining(t).abs());
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

/// RK4 method with configuration struct and saveat support.
pub fn rk4_with_config<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &Rk4Config,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
{
    let f = wrap_rhs(f);
    let sol = rk4(f, y0, t_span, config.dt)?;
    Ok(apply_saveat(sol, &config.saveat))
}

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

const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1_113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2_187.0 / 6_784.0;
const B6: f64 = 11.0 / 84.0;

const BS1: f64 = 5_179.0 / 57_600.0;
const BS3: f64 = 7_571.0 / 16_695.0;
const BS4: f64 = 393.0 / 640.0;
const BS5: f64 = -92_097.0 / 339_200.0;
const BS6: f64 = 187.0 / 2_100.0;
const BS7: f64 = 1.0 / 40.0;

const E1: f64 = B1 - BS1;
const E3: f64 = B3 - BS3;
const E4: f64 = B4 - BS4;
const E5: f64 = B5 - BS5;
const E6: f64 = B6 - BS6;
const E7: f64 = -BS7; // B7 = 0

/// Adaptive Dormand-Prince RK45 method with FSAL and error control.
pub fn dormand_prince<T: Scalar, B: Backend, R: IntoOdeRhs<T, B>, F>(
    f: F,
    y0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &AdaptiveConfig,
) -> Result<OdeSolution<T, B>>
where
    F: Fn(f64, &Tensor<T, B>) -> R,
    T::Real: Into<f64>,
{
    let f = wrap_rhs(f);
    validate_adaptive(t_span, config)?;

    let dir = time_direction(t_span);
    let remaining = |t: f64| (t_span.1 - t) * dir;

    let mut times = vec![t_span.0];
    let mut states = vec![y0.clone()];

    let mut t = t_span.0;
    let mut y = y0.clone();
    let mut dt = config
        .dt_init
        .min(config.dt_max)
        .max(config.dt_min)
        .min((t_span.1 - t_span.0).abs());

    // FSAL: compute k1 once, then reuse k7 of accepted step as k1 of next step.
    let mut k1 = f(t, &y)?;

    while remaining(t) > 1e-14 {
        // Clamp step to not overshoot end of interval.
        let h = dir * dt.min(remaining(t));

        // Stage 2: t + h/5
        let k2 = f(t + h / 5.0, &lincomb!(y; h * A21, k1))?;

        // Stage 3: t + 3h/10
        let k3 = f(t + 3.0 * h / 10.0, &lincomb!(y; h * A31, k1; h * A32, k2))?;

        // Stage 4: t + 4h/5
        let k4 = f(
            t + 4.0 * h / 5.0,
            &lincomb!(y; h * A41, k1; h * A42, k2; h * A43, k3),
        )?;

        // Stage 5: t + 8h/9
        let k5 = f(
            t + 8.0 * h / 9.0,
            &lincomb!(y; h * A51, k1; h * A52, k2; h * A53, k3; h * A54, k4),
        )?;

        // Stage 6: t + h
        let k6 = f(
            t + h,
            &lincomb!(y; h * A61, k1; h * A62, k2; h * A63, k3; h * A64, k4; h * A65, k5),
        )?;

        // 5th-order solution.
        let y_new = lincomb!(y; h * B1, k1; h * B3, k3; h * B4, k4; h * B5, k5; h * B6, k6);

        // Stage 7 (FSAL): evaluate at t + h with the 5th-order solution.
        let k7 = f(t + h, &y_new)?;

        // Error estimate: h * sum_i e_i * k_i  (e_i = b_i - b*_i).
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

            // Check early termination after accepting the step.
            if let Some(ref terminate_fn) = config.terminate
                && terminate_fn(t)
            {
                break;
            }
        }

        // Adjust step size: PI controller with safety factor 0.9.
        // factor = 0.9 * err_norm^(-1/5), clamped to [0.2, 5.0].
        let factor = if err_norm == 0.0 {
            5.0_f64
        } else {
            (0.9 * err_norm.powf(-0.2)).clamp(0.2, 5.0)
        };

        dt = (h.abs() * factor).clamp(config.dt_min, config.dt_max);

        // Safety: avoid infinite loop if step cannot advance past dt_min.
        if h.abs() <= config.dt_min && err_norm > 1.0 {
            return Err(Error::invalid(
                "step size reached dt_min with error above tolerance; system may be stiff",
            ));
        }
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}
