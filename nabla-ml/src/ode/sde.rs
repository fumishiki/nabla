
use nabla_core::backend::Backend;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{alloc_trajectory, apply_saveat, sc, time_direction, validate, wrap_rhs, IntoOdeRhs, OdeSolution};


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


/// Configuration for stochastic differential equation solvers.
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

    /// Set the number of independent Wiener process dimensions.
    ///
    /// When `None` (default), the noise dimension equals the total number of
    /// state elements (`rows * cols`).  Set this to decouple noise dimensionality
    /// from state size, e.g. for systems driven by fewer noise sources than
    /// state variables.
    pub fn with_noise_dims(mut self, dims: usize) -> Self {
        self.noise_dims = Some(dims);
        self
    }

    /// Set specific output times for the solution.
    pub fn with_saveat(mut self, times: Vec<f64>) -> Self {
        self.saveat = Some(times);
        self
    }
}


/// Euler-Maruyama method for stochastic differential equations.
pub fn euler_maruyama<T, B, RF, RG, F, G>(
    drift: F,
    diffusion: G,
    x0: &Tensor<T, B>,
    t_span: (f64, f64),
    config: &SdeConfig,
) -> Result<OdeSolution<T, B>>
where
    T: Scalar,
    B: Backend,
    RF: IntoOdeRhs<T, B>,
    RG: IntoOdeRhs<T, B>,
    F: Fn(f64, &Tensor<T, B>) -> RF,
    G: Fn(f64, &Tensor<T, B>) -> RG,
{
    let drift = wrap_rhs(drift);
    let diffusion = wrap_rhs(diffusion);
    validate(t_span, config.dt)?;

    let dir = time_direction(t_span);
    let remaining = |t: f64| (t_span.1 - t) * dir;

    let (rows, cols) = x0.shape();
    let n_noise = config.noise_dims.unwrap_or(rows * cols);
    let sqrt_dt = config.dt.sqrt();

    let mut rng = Xorshift64::new(config.seed);
    let mut noise_buf = vec![0.0_f64; n_noise];

    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, x0);
    let mut t = t_span.0;
    let mut x = x0.clone();

    while remaining(t) > 1e-14 {
        let h_abs = config.dt.min(remaining(t).abs());
        let h = dir * h_abs;
        let h_sqrt = if (h_abs - config.dt).abs() < 1e-14 {
            sqrt_dt
        } else {
            h_abs.sqrt()
        };

        let f_val = drift(t, &x)?;
        let g_val = diffusion(t, &x)?;

        // Generate dW: each element ~ N(0, |h|) = N(0,1) * sqrt(|h|).
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

        // X_{n+1} = X_n + f * h + g * dW  (h is signed for direction)
        let h_t: T = sc(h);
        x = &(&x + &(&f_val * h_t)) + &g_val.emul(&dw);
        t += h;
        times.push(t);
        states.push(x.clone());
    }

    let sol = OdeSolution { times, states };
    Ok(apply_saveat(sol, &config.saveat))
}


/// Parallel ensemble of Euler-Maruyama trajectories with independent seeds.
pub fn ensemble_euler_maruyama<T, B, RF, RG, F, G>(
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
    RF: IntoOdeRhs<T, B>,
    RG: IntoOdeRhs<T, B>,
    F: Fn(f64, &Tensor<T, B>) -> RF + Sync,
    G: Fn(f64, &Tensor<T, B>) -> RG + Sync,
{
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_trajectories)
            .map(|i| {
                let traj_config = SdeConfig {
                    dt: config.dt,
                    seed: config.seed.wrapping_add(i as u64),
                    noise_dims: config.noise_dims,
                    saveat: config.saveat.clone(),
                };
                s.spawn(move || {
                    euler_maruyama(
                        |t, x| drift(t, x),
                        |t, x| diffusion(t, x),
                        x0,
                        t_span,
                        &traj_config,
                    )
                })
            })
            .collect();

        let mut results = Vec::with_capacity(n_trajectories);
        for handle in handles {
            // Thread panics are propagated by scope; join returns the Result.
            match handle.join() {
                Ok(sol) => results.push(sol?),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(results)
    })
}


/// Milstein method for SDEs with strong order 1.0 convergence.
pub fn milstein<T, B, RF, RG, RDG, F, G, DG>(
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
    RF: IntoOdeRhs<T, B>,
    RG: IntoOdeRhs<T, B>,
    RDG: IntoOdeRhs<T, B>,
    F: Fn(f64, &Tensor<T, B>) -> RF,
    G: Fn(f64, &Tensor<T, B>) -> RG,
    DG: Fn(f64, &Tensor<T, B>) -> RDG,
{
    let drift = wrap_rhs(drift);
    let diffusion = wrap_rhs(diffusion);
    let diffusion_deriv = wrap_rhs(diffusion_deriv);
    validate(t_span, config.dt)?;

    let dir = time_direction(t_span);
    let remaining = |t: f64| (t_span.1 - t) * dir;

    let (rows, cols) = x0.shape();
    let n_noise = config.noise_dims.unwrap_or(rows * cols);
    let sqrt_dt = config.dt.sqrt();

    let mut rng = Xorshift64::new(config.seed);
    let mut noise_buf = vec![0.0_f64; n_noise];

    let (mut times, mut states) = alloc_trajectory(t_span, config.dt, x0);
    let mut t = t_span.0;
    let mut x = x0.clone();

    let half: T = sc(0.5);

    while remaining(t) > 1e-14 {
        let h_abs = config.dt.min(remaining(t).abs());
        let h = dir * h_abs;
        let h_sqrt = if (h_abs - config.dt).abs() < 1e-14 {
            sqrt_dt
        } else {
            h_abs.sqrt()
        };
        let h_t: T = sc(h);
        let h_abs_t: T = sc(h_abs);

        let f_val = drift(t, &x)?;
        let g_val = diffusion(t, &x)?;
        let gp_val = diffusion_deriv(t, &x)?;

        // Generate dW (magnitude scales with |h|, sign is independent).
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

        // dW^2 - |dt| (element-wise, Milstein correction uses absolute step).
        let dw_sq_minus_dt = Tensor::<T, B>::from_fn(rows, cols, |i, j| {
            let w = dw.get(i, j);
            w * w - h_abs_t
        });

        // X_{n+1} = X_n + f*h + g*dW + 0.5 * g * g' * (dW^2 - |h|)
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
