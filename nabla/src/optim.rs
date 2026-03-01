//! Optimizer utilities: AdamW, learning-rate schedules, and gradient scaling.
//!
//! Provides CPU-side optimizers and utilities for mixed-precision training.

use crate::{scalar, tensor};

#[inline]
fn warmup_lr(base_lr: f64, step: usize, warmup_steps: usize) -> f64 {
    base_lr * step as f64 / warmup_steps.max(1) as f64
}

#[inline]
fn decay_progress(step: usize, start: usize, total: usize) -> f64 {
    (step - start) as f64 / (total - start).max(1) as f64
}

#[inline]
fn ramp(start: f64, end: f64, step: usize, total_steps: usize) -> f64 {
    start + (end - start) * step as f64 / total_steps.max(1) as f64
}

#[inline]
fn cosine_decay(high: f64, low: f64, progress: f64) -> f64 {
    low + 0.5 * (high - low) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// In-place AdamW parameter update.
///
/// Updates `param`, `m` (1st moment), `v` (2nd moment) in-place.
/// Returns nothing; all modifications are in-place via mutable references.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step<T: scalar::Scalar>(
    param: &mut tensor::Tensor<T>,
    grad: &tensor::Tensor<T>,
    m: &mut tensor::Tensor<T>,
    v: &mut tensor::Tensor<T>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: usize,
) {
    let bias_corr1 = 1.0 - beta1.powi(step as i32);
    let bias_corr2 = 1.0 - beta2.powi(step as i32);
    let step_size = lr / bias_corr1;
    let b1 = T::from_f64(beta1);
    let one_minus_b1 = T::from_f64(1.0 - beta1);
    let b2 = T::from_f64(beta2);
    let one_minus_b2 = T::from_f64(1.0 - beta2);
    let wd_lr = T::from_f64(weight_decay * lr);
    let eps_t = T::from_f64(eps);
    let ss = T::from_f64(step_size);
    let inv_bias_corr2 = T::from_f64(1.0 / bias_corr2);
    let half = T::from_f64(0.5);

    // m = beta1 * m + (1 - beta1) * grad
    *m = &(&*m * b1) + &(grad * one_minus_b1);
    // v = beta2 * v + (1 - beta2) * grad^2
    *v = &(&*v * b2) + &(&grad.emul(grad) * one_minus_b2);
    // denom = sqrt(v / (1 - beta2^t)) + eps
    let denom = (&*v * inv_bias_corr2).powf(half).map(|x| x + eps_t);
    // update = step_size * m / denom
    let update = (&*m * ss).ediv(&denom);
    // param = param - update - weight_decay * lr * param
    *param = &(&*param - &update) - &(&*param * wd_lr);
}

/// Learning rate schedule types.
pub enum LrSchedule {
    /// Cosine annealing with linear warmup.
    Cosine {
        /// Number of linear warmup steps.
        warmup_steps: usize,
        /// Total training steps.
        total_steps: usize,
        /// Minimum learning rate after decay.
        min_lr: f64,
    },
    /// Linear warmup then linear decay.
    Linear {
        /// Number of linear warmup steps.
        warmup_steps: usize,
        /// Total training steps.
        total_steps: usize,
    },
    /// One-cycle policy (warmup then cosine decay).
    OneCycle {
        /// Peak learning rate.
        max_lr: f64,
        /// Total training steps.
        total_steps: usize,
        /// Fraction of steps spent on warmup.
        pct_start: f64,
    },
}

/// Compute learning rate at given step.
#[must_use]
pub fn lr_at_step(schedule: &LrSchedule, base_lr: f64, step: usize) -> f64 {
    match schedule {
        LrSchedule::Cosine {
            warmup_steps,
            total_steps,
            min_lr,
        } => {
            if step < *warmup_steps {
                warmup_lr(base_lr, step, *warmup_steps)
            } else {
                let progress = decay_progress(step, *warmup_steps, *total_steps);
                cosine_decay(base_lr, *min_lr, progress)
            }
        }
        LrSchedule::Linear {
            warmup_steps,
            total_steps,
        } => {
            if step < *warmup_steps {
                warmup_lr(base_lr, step, *warmup_steps)
            } else {
                let progress = decay_progress(step, *warmup_steps, *total_steps);
                base_lr * (1.0 - progress).max(0.0)
            }
        }
        LrSchedule::OneCycle {
            max_lr,
            total_steps,
            pct_start,
        } => {
            let up_steps = (*total_steps as f64 * pct_start) as usize;
            if step < up_steps {
                ramp(base_lr, *max_lr, step, up_steps)
            } else {
                let progress = decay_progress(step, up_steps, *total_steps);
                cosine_decay(*max_lr, base_lr, progress)
            }
        }
    }
}

/// Dynamic loss scaler for mixed-precision training.
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    consecutive_ok: usize,
}

impl GradScaler {
    /// Create a new `GradScaler` with default parameters.
    ///
    /// Defaults: scale=65536, growth_factor=2, backoff_factor=0.5,
    /// growth_interval=2000.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_ok: 0,
        }
    }

    /// Scale the loss before backward pass.
    #[must_use]
    pub fn scale_loss<T: scalar::Scalar>(&self, loss: &tensor::Tensor<T>) -> tensor::Tensor<T> {
        loss * T::from_f64(self.scale)
    }

    /// Unscale gradients and check for inf/nan.
    ///
    /// Returns `true` if gradients are valid (no inf/nan detected).
    /// On invalid gradients: backs off the scale and zeros out all grad tensors.
    pub fn unscale_and_update<T: scalar::Scalar>(
        &mut self,
        grads: &mut [tensor::Tensor<T>],
    ) -> bool {
        let inv_scale = T::from_f64(1.0 / self.scale);
        let mut has_inf_nan = false;
        'outer: for g in grads.iter() {
            let (m, n) = g.shape();
            for r in 0..m {
                for c in 0..n {
                    let v = g.get(r, c).to_f64();
                    if !v.is_finite() {
                        has_inf_nan = true;
                        break 'outer;
                    }
                }
            }
        }
        if has_inf_nan {
            self.scale *= self.backoff_factor;
            self.consecutive_ok = 0;
            for g in grads.iter_mut() {
                let (m, n) = g.shape();
                *g = tensor::Tensor::zeros(m, n);
            }
            false
        } else {
            for g in grads.iter_mut() {
                *g = &*g * inv_scale;
            }
            self.consecutive_ok += 1;
            if self.consecutive_ok >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_ok = 0;
            }
            true
        }
    }

    /// Current scale factor.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Current scale factor (alias for [`GradScaler::scale`]).
    ///
    /// Use with `Variable::scale()` for mixed-precision backward:
    /// ```ignore
    /// let scaler = GradScaler::new();
    /// let scaled_loss = loss_var.scale(T::from_f64(scaler.scale_factor()));
    /// scaled_loss.backward()?;
    /// ```
    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful learning rate scheduler that wraps an [`LrSchedule`].
///
/// Tracks the current step and computes the learning rate at each step.
pub struct LrScheduler {
    schedule: LrSchedule,
    base_lr: f64,
    current_step: usize,
}

impl LrScheduler {
    /// Create a new scheduler with the given schedule and base learning rate.
    #[must_use]
    pub fn new(schedule: LrSchedule, base_lr: f64) -> Self {
        Self {
            schedule,
            base_lr,
            current_step: 0,
        }
    }

    /// Advance by one step and return the current learning rate.
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        lr_at_step(&self.schedule, self.base_lr, self.current_step)
    }

    /// Get the learning rate at the current step without advancing.
    #[must_use]
    pub fn get_lr(&self) -> f64 {
        lr_at_step(&self.schedule, self.base_lr, self.current_step)
    }
}
