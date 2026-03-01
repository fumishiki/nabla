//! Optimizer trait and stateful implementations.
//!
//! The [`Optimizer`] trait provides a generic interface for parameter updates.
//! [`AdamW`] is the primary implementation, wrapping the same logic as
//! [`crate::adamw_step`] but with internal moment tracking.

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use crate::module::Module;

/// Generic optimizer interface.
pub trait Optimizer<T: Scalar, B: Backend> {
    /// Perform one optimization step: update `params` using `grads`.
    ///
    /// `params` and `grads` must have the same length and matching shapes.
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]);

    /// Simplified step: update owned parameter tensors using corresponding gradients.
    ///
    /// Convenience wrapper over [`Optimizer::step`] that accepts slices directly.
    fn step_slices(&mut self, params: &mut [Tensor<T, B>], grads: &[Tensor<T, B>]) {
        let mut param_refs: Vec<&mut Tensor<T, B>> = params.iter_mut().collect();
        let grad_refs: Vec<&Tensor<T, B>> = grads.iter().collect();
        self.step(&mut param_refs, &grad_refs);
    }

    /// Reset all internal state (moments, step counter, etc.).
    fn reset(&mut self);

    /// Set the learning rate dynamically.
    ///
    /// Default implementation is a no-op for optimizers that do not support
    /// dynamic learning rate changes.
    fn set_lr(&mut self, _lr: f64) {}
}

/// AdamW optimizer with decoupled weight decay and momentum tracking.
pub struct AdamW<T: Scalar, B: Backend> {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: usize,
    m: Vec<Tensor<T, B>>,
    v: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> AdamW<T, B> {
    /// Create a new `AdamW` optimizer.
    ///
    /// `param_shapes` provides `(nrows, ncols)` for each parameter tensor so
    /// moment buffers can be pre-allocated. Defaults: beta1=0.9, beta2=0.999,
    /// eps=1e-8, weight_decay=0.01.
    #[must_use]
    pub fn new(lr: f64, param_shapes: &[(usize, usize)]) -> Self {
        let m = param_shapes
            .iter()
            .map(|&(r, c)| Tensor::zeros(r, c))
            .collect();
        let v = param_shapes
            .iter()
            .map(|&(r, c)| Tensor::zeros(r, c))
            .collect();
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            m,
            v,
        }
    }

    /// Create a new `AdamW` from raw parameter tensors.
    ///
    /// Infers shapes from the provided tensors and delegates to [`AdamW::new`].
    /// Uses default hyperparameters (beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01).
    #[must_use]
    pub fn from_params(lr: f64, params: &[&Tensor<T, B>]) -> Self {
        let shapes: Vec<(usize, usize)> = params.iter().map(|p| p.shape()).collect();
        Self::new(lr, &shapes)
    }

    /// Create a new `AdamW` from a module's parameters.
    ///
    /// Moment buffers are initialized as zeros matching each parameter shape.
    /// Uses default hyperparameters (beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01).
    #[must_use]
    pub fn from_module<M: Module<T, B>>(module: &M, lr: f64) -> Self {
        let shapes: Vec<(usize, usize)> = module.parameters().iter().map(|p| p.shape()).collect();
        Self::new(lr, &shapes)
    }

    /// Override beta1 (first moment decay).
    #[must_use]
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Override beta2 (second moment decay).
    #[must_use]
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Override epsilon (numerical stability).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Override weight decay coefficient.
    #[must_use]
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set the learning rate dynamically.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl<T: Scalar, B: Backend> Optimizer<T, B> for AdamW<T, B> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "nabla: AdamW::step params/grads length mismatch"
        );
        assert_eq!(
            params.len(),
            self.m.len(),
            "nabla: AdamW::step params count differs from moment buffers"
        );
        self.step_count += 1;
        let step = self.step_count;
        let bias_corr1 = 1.0 - self.beta1.powi(step as i32);
        let bias_corr2 = 1.0 - self.beta2.powi(step as i32);
        let step_size = self.lr / bias_corr1;
        let b1 = T::from_f64(self.beta1);
        let one_minus_b1 = T::from_f64(1.0 - self.beta1);
        let b2 = T::from_f64(self.beta2);
        let one_minus_b2 = T::from_f64(1.0 - self.beta2);
        let wd_lr = T::from_f64(self.weight_decay * self.lr);
        let eps_t = T::from_f64(self.eps);
        let ss = T::from_f64(step_size);
        let inv_bias_corr2 = T::from_f64(1.0 / bias_corr2);
        let half = T::from_f64(0.5);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // m = beta1 * m + (1 - beta1) * grad
            self.m[i] = &(&self.m[i] * b1) + &(&(**grad) * one_minus_b1);
            // v = beta2 * v + (1 - beta2) * grad^2
            self.v[i] = &(&self.v[i] * b2) + &(&grad.emul(grad) * one_minus_b2);
            // v_hat = v / (1 - beta2^t), then sqrt(v_hat) + eps
            let denom = (&self.v[i] * inv_bias_corr2).powf(half).map(|x| x + eps_t);
            // update = step_size * m / denom
            let update = (&self.m[i] * ss).ediv(&denom);
            // param = param - update - weight_decay * lr * param
            **param = &(&(**param) - &update) - &(&(**param) * wd_lr);
        }
    }

    fn reset(&mut self) {
        self.step_count = 0;
        for m in &mut self.m {
            let (r, c) = m.shape();
            *m = Tensor::zeros(r, c);
        }
        for v in &mut self.v {
            let (r, c) = v.shape();
            *v = Tensor::zeros(r, c);
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// SGD optimizer with optional momentum and weight decay.
pub struct SGD<T: Scalar, B: Backend> {
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    velocity: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> SGD<T, B> {
    /// Create a new `SGD` optimizer.
    ///
    /// `param_shapes` provides `(nrows, ncols)` for each parameter tensor so
    /// velocity buffers can be pre-allocated.
    #[must_use]
    pub fn new(lr: f64, momentum: f64, weight_decay: f64, param_shapes: &[(usize, usize)]) -> Self {
        let velocity = param_shapes
            .iter()
            .map(|&(r, c)| Tensor::zeros(r, c))
            .collect();
        Self {
            lr,
            momentum,
            weight_decay,
            velocity,
        }
    }

    /// Set the learning rate dynamically.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl<T: Scalar, B: Backend> Optimizer<T, B> for SGD<T, B> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "nabla: SGD::step params/grads length mismatch"
        );
        assert_eq!(
            params.len(),
            self.velocity.len(),
            "nabla: SGD::step params count differs from velocity buffers"
        );
        let mom = T::from_f64(self.momentum);
        let lr_t = T::from_f64(self.lr);
        let wd_lr = T::from_f64(self.weight_decay * self.lr);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // v = momentum * v + grad
            self.velocity[i] = &(&self.velocity[i] * mom) + *grad;
            // param = param - lr * v - weight_decay * lr * param
            **param = &(&(**param) - &(&self.velocity[i] * lr_t)) - &(&(**param) * wd_lr);
        }
    }

    fn reset(&mut self) {
        for v in &mut self.velocity {
            let (r, c) = v.shape();
            *v = Tensor::zeros(r, c);
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}
