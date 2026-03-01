// SDE solvers: Euler-Maruyama (strong order 0.5) and Milstein (strong order 1.0).
// All require feature = "cpu" (gated at the mod-level import in mod.rs).

use nabla_core::backend::Backend;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{alloc_trajectory, apply_saveat, sc, validate, OdeSolution};

// ---------------------------------------------------------------------------
// PRNG: xorshift64 + Box-Muller for N(0, 1) samples
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG (period 2^64 - 1).
struct Xorshift64(u64);

impl Xorshift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        // Avoid zero state (fixed point of xorshift).
        Self(if seed == 0 {
            0x5DEE_CE66_D1A4_F87D
        } else {
            seed
        })
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f64 in (0, 1) — open interval (Box-Muller requires non-zero).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Use upper 52 bits for mantissa, shift into (0, 1).
        let u = (self.next_u64() >> 12) | 0x3FF0_0000_0000_0000;
        // SAFETY: bit pattern is a valid f64 in [1.0, 2.0).
        let v = f64::from_bits(u) - 1.0;
        // Clamp away from exact 0.0 for Box-Muller log safety.
        if v < f64::EPSILON {
            f64::EPSILON
        } else {
            v
        }
    }

    /// Generate a pair of N(0, 1) samples via Box-Muller transform.
    #[inline]
    fn normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

/// Fill a vector with N(0, 1) samples.
fn fill_normal(rng: &mut Xorshift64, buf: &mut [f64]) {
    let mut i = 0;
    while i + 1 < buf.len() {
        let (a, b) = rng.normal_pair();
        buf[i] = a;
        buf[i + 1] = b;
        i += 2;
    }
    // Handle odd length.
    if i < buf.len() {
        let (a, _) = rng.normal_pair();
        buf[i] = a;
    }
}

// ---------------------------------------------------------------------------
// SDE config
// ---------------------------------------------------------------------------

/// Configuration for SDE solvers.
pub struct SdeConfig {
    /// Time step size.
    pub dt: f64,
    /// Random seed for Wiener process.
    pub seed: u64,
    /// Number of Wiener process dimensions (defaults to state dimension).
    pub noise_dims: Option<usize>,
    /// Save at specific times only.
    pub saveat: Option<Vec<f64>>,
}

impl Default for SdeConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            seed: 42,
            noise_dims: None,
            saveat: None,
        }
    }
}

impl SdeConfig {
    /// Set the time step size.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set the random seed for the Wiener process.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set specific output times for the solution.
    pub fn with_saveat(mut self, times: Vec<f64>) -> Self {
        self.saveat = Some(times);
        self
    }
}

// ---------------------------------------------------------------------------
// Euler-Maruyama — strong order 0.5
// ---------------------------------------------------------------------------

/// Euler-Maruyama method for stochastic differential equations.
///
/// Solves `dX = f(t, X) dt + g(t, X) dW` where W is a Wiener process.
///
/// Strong convergence order 0.5.  Each element of the state tensor receives
/// an independent Wiener increment `dW_i ~ N(0, dt)`.
///
/// # Arguments
/// * `drift` — `f(t, X)` deterministic drift term
/// * `diffusion` — `g(t, X)` stochastic diffusion coefficient
/// * `x0` — initial state
/// * `t_span` — `(t_start, t_end)`
/// * `config` — SDE solver configuration
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, or if `drift`/`diffusion`
/// return an error.
pub fn euler_maruyama<T, B, F, G>(
    drift: F,
    diffusion: G,
    x0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &SdeConfig,
) -> Result<OdeSolution<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    G: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
{
    validate(t_span, config.dt)?;

    let (rows, cols) = x0.shape();
    let n_noise = config.noise_dims.unwrap_or(rows * cols);
    let sqrt_dt = config.dt.sqrt();

    let mut rng = Xorshift64::new(config.seed);
    let mut noise_buf = vec![0.0_f64; n_noise];

    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, x0);
    let mut t = t_span.0;
    let mut x = x0.clone();

    while t < t_span.1 {
        let h = config.dt.min(t_span.1 - t);
        let h_sqrt = if (h - config.dt).abs() < 1e-14 {
            sqrt_dt
        } else {
            h.sqrt()
        };

        let f_val = drift(t, &x)?;
        let g_val = diffusion(t, &x)?;

        // Generate dW: each element ~ N(0, h) = N(0,1) * sqrt(h).
        fill_normal(&mut rng, &mut noise_buf);
        let dw = Tensor::<T, B>::from_fn(rows, cols, |i, j| {
            let idx = i * cols + j;
            let z = if idx < noise_buf.len() {
                noise_buf[idx]
            } else {
                0.0
            };
            T::from_f64(z * h_sqrt)
        });

        // X_{n+1} = X_n + f * dt + g * dW
        let h_t: T = sc(h);
        x = &(&x + &(&f_val * h_t)) + &g_val.emul(&dw);
        t += h;
        times.push(t);
        states.push(x.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}

// ---------------------------------------------------------------------------
// ensemble_euler_maruyama — multiple independent trajectories
// ---------------------------------------------------------------------------

/// Run `n_trajectories` independent Euler-Maruyama paths with different seeds.
///
/// Each trajectory uses a seed derived from `config.seed + trajectory_index`,
/// producing statistically independent Wiener paths.
///
/// # Arguments
/// * `drift` — `f(t, X)` deterministic drift term
/// * `diffusion` — `g(t, X)` stochastic diffusion coefficient
/// * `x0` — initial state (shared across all trajectories)
/// * `t_span` — `(t_start, t_end)`
/// * `config` — SDE solver configuration (seed is used as base)
/// * `n_trajectories` — number of independent paths to simulate
///
/// # Errors
///
/// Returns an error if any individual trajectory fails.
pub fn ensemble_euler_maruyama<T, B, F, G>(
    drift: &F,
    diffusion: &G,
    x0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &SdeConfig,
    n_trajectories: usize,
) -> Result<Vec<OdeSolution<T, B>>>
where
    T: Scalar,
    B: Backend,
    F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    G: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
{
    let mut results = Vec::with_capacity(n_trajectories);
    for i in 0..n_trajectories {
        let traj_config = SdeConfig {
            dt: config.dt,
            seed: config.seed.wrapping_add(i as u64),
            noise_dims: config.noise_dims,
            saveat: config.saveat.clone(),
        };
        let sol = euler_maruyama(
            |t, x| drift(t, x),
            |t, x| diffusion(t, x),
            x0,
            t_span,
            &traj_config,
        )?;
        results.push(sol);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Milstein — strong order 1.0 (scalar noise)
// ---------------------------------------------------------------------------

/// Milstein method for SDEs with scalar noise.
///
/// Solves `dX = f(t, X) dt + g(t, X) dW` with the correction term
/// `0.5 g(t, X) g'(t, X) (dW^2 - dt)`, achieving strong order 1.0
/// (vs Euler-Maruyama's 0.5).
///
/// Requires the derivative of the diffusion coefficient `dg/dx`.
///
/// # Arguments
/// * `drift` — `f(t, X)` deterministic drift term
/// * `diffusion` — `g(t, X)` stochastic diffusion coefficient
/// * `diffusion_deriv` — `dg/dx(t, X)` derivative of diffusion w.r.t. state
/// * `x0` — initial state
/// * `t_span` — `(t_start, t_end)`
/// * `config` — SDE solver configuration
///
/// # Errors
///
/// Returns an error if `t_span` or `dt` are invalid, or if any of the
/// user-supplied functions return an error.
pub fn milstein<T, B, F, G, DG>(
    drift: F,
    diffusion: G,
    diffusion_deriv: DG,
    x0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &SdeConfig,
) -> Result<OdeSolution<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    G: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    DG: Fn(f64, &Tensor<T, B>) -> Result<Tensor<T, B>>,
{
    validate(t_span, config.dt)?;

    let (rows, cols) = x0.shape();
    let n_noise = config.noise_dims.unwrap_or(rows * cols);
    let sqrt_dt = config.dt.sqrt();

    let mut rng = Xorshift64::new(config.seed);
    let mut noise_buf = vec![0.0_f64; n_noise];

    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, x0);
    let mut t = t_span.0;
    let mut x = x0.clone();

    let half: T = sc(0.5);

    while t < t_span.1 {
        let h = config.dt.min(t_span.1 - t);
        let h_sqrt = if (h - config.dt).abs() < 1e-14 {
            sqrt_dt
        } else {
            h.sqrt()
        };
        let h_t: T = sc(h);

        let f_val = drift(t, &x)?;
        let g_val = diffusion(t, &x)?;
        let gp_val = diffusion_deriv(t, &x)?;

        // Generate dW.
        fill_normal(&mut rng, &mut noise_buf);
        let dw = Tensor::<T, B>::from_fn(rows, cols, |i, j| {
            let idx = i * cols + j;
            let z = if idx < noise_buf.len() {
                noise_buf[idx]
            } else {
                0.0
            };
            T::from_f64(z * h_sqrt)
        });

        // dW^2 - dt (element-wise).
        let dw_sq_minus_dt = Tensor::<T, B>::from_fn(rows, cols, |i, j| {
            let w = dw.get(i, j);
            w * w - h_t
        });

        // X_{n+1} = X_n + f*dt + g*dW + 0.5 * g * g' * (dW^2 - dt)
        let drift_term = &f_val * h_t;
        let diffusion_term = g_val.emul(&dw);
        let milstein_correction = &g_val.emul(&gp_val).emul(&dw_sq_minus_dt) * half;

        x = &(&(&x + &drift_term) + &diffusion_term) + &milstein_correction;
        t += h;
        times.push(t);
        states.push(x.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}
