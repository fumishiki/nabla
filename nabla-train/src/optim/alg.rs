use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use crate::optim::core::{OptimKind, OptimMeta, OptimState, Optimizer};

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
    #[must_use]
    pub fn new(lr: f64, param_shapes: &[(usize, usize)]) -> Self {
        let m = param_shapes.iter().map(|&(r, c)| Tensor::zeros(r, c)).collect();
        let v = param_shapes.iter().map(|&(r, c)| Tensor::zeros(r, c)).collect();
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

    #[must_use]
    pub fn from_params(lr: f64, params: &[&Tensor<T, B>]) -> Self {
        let shapes: Vec<(usize, usize)> = params.iter().map(|p| p.shape()).collect();
        Self::new(lr, &shapes)
    }

    #[must_use]
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    #[must_use]
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    #[must_use]
    pub fn lr(&self) -> f64 { self.lr }

    #[must_use]
    pub fn step_count(&self) -> usize { self.step_count }

    #[must_use]
    pub fn moments(&self) -> (&[Tensor<T, B>], &[Tensor<T, B>]) { (&self.m, &self.v) }

    pub fn moments_mut(&mut self) -> (&mut [Tensor<T, B>], &mut [Tensor<T, B>]) {
        (&mut self.m, &mut self.v)
    }

    pub fn set_state(
        &mut self,
        step_count: usize,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        lr: f64,
    ) {
        self.step_count = step_count;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.weight_decay = weight_decay;
        self.lr = lr;
    }
}

impl<T: Scalar, B: Backend> Optimizer<T, B> for AdamW<T, B> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]) {
        assert_eq!(params.len(), grads.len(), "nabla-train: AdamW params/grads length mismatch");
        assert_eq!(params.len(), self.m.len(), "nabla-train: AdamW params count mismatch");
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

        let one_minus_wd_lr = T::one() - wd_lr;
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // m = beta1 * m + (1-beta1) * grad  (in-place: 0 alloc)
            self.m[i] *= b1;
            self.m[i].axpy_inplace(one_minus_b1, grad);
            // v = beta2 * v + (1-beta2) * grad^2  (1 alloc for grad_sq)
            self.v[i] *= b2;
            let grad_sq = grad.emul(grad);
            self.v[i].axpy_inplace(one_minus_b2, &grad_sq);
            let denom = (&self.v[i] * inv_bias_corr2).powf(half).map(|x| x + eps_t);
            let update = (&self.m[i] * ss).ediv(&denom);
            // param = param * (1 - wd*lr) - update  (in-place: 0 alloc)
            **param *= one_minus_wd_lr;
            param.axpy_inplace(-T::one(), &update);
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

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
}

pub struct Adam<T: Scalar, B: Backend> {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step_count: usize,
    m: Vec<Tensor<T, B>>,
    v: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> Adam<T, B> {
    #[must_use]
    pub fn new(lr: f64, param_shapes: &[(usize, usize)]) -> Self {
        let m = param_shapes.iter().map(|&(r, c)| Tensor::zeros(r, c)).collect();
        let v = param_shapes.iter().map(|&(r, c)| Tensor::zeros(r, c)).collect();
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_count: 0,
            m,
            v,
        }
    }

    #[must_use]
    pub fn from_params(lr: f64, params: &[&Tensor<T, B>]) -> Self {
        let shapes: Vec<(usize, usize)> = params.iter().map(|p| p.shape()).collect();
        Self::new(lr, &shapes)
    }

    #[must_use]
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    #[must_use]
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn lr(&self) -> f64 { self.lr }

    #[must_use]
    pub fn step_count(&self) -> usize { self.step_count }

    #[must_use]
    pub fn moments(&self) -> (&[Tensor<T, B>], &[Tensor<T, B>]) { (&self.m, &self.v) }

    pub fn moments_mut(&mut self) -> (&mut [Tensor<T, B>], &mut [Tensor<T, B>]) {
        (&mut self.m, &mut self.v)
    }

    pub fn set_state(
        &mut self,
        step_count: usize,
        beta1: f64,
        beta2: f64,
        eps: f64,
        lr: f64,
    ) {
        self.step_count = step_count;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.lr = lr;
    }
}

impl<T: Scalar, B: Backend> Optimizer<T, B> for Adam<T, B> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]) {
        assert_eq!(params.len(), grads.len(), "nabla-train: Adam params/grads length mismatch");
        assert_eq!(params.len(), self.m.len(), "nabla-train: Adam params count mismatch");
        self.step_count += 1;
        let step = self.step_count;
        let bias_corr1 = 1.0 - self.beta1.powi(step as i32);
        let bias_corr2 = 1.0 - self.beta2.powi(step as i32);
        let step_size = self.lr / bias_corr1;
        let b1 = T::from_f64(self.beta1);
        let one_minus_b1 = T::from_f64(1.0 - self.beta1);
        let b2 = T::from_f64(self.beta2);
        let one_minus_b2 = T::from_f64(1.0 - self.beta2);
        let eps_t = T::from_f64(self.eps);
        let ss = T::from_f64(step_size);
        let inv_bias_corr2 = T::from_f64(1.0 / bias_corr2);
        let half = T::from_f64(0.5);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // m = beta1 * m + (1-beta1) * grad  (in-place: 0 alloc)
            self.m[i] *= b1;
            self.m[i].axpy_inplace(one_minus_b1, grad);
            // v = beta2 * v + (1-beta2) * grad^2  (1 alloc for grad_sq)
            self.v[i] *= b2;
            let grad_sq = grad.emul(grad);
            self.v[i].axpy_inplace(one_minus_b2, &grad_sq);
            let denom = (&self.v[i] * inv_bias_corr2).powf(half).map(|x| x + eps_t);
            let update = (&self.m[i] * ss).ediv(&denom);
            param.axpy_inplace(-T::one(), &update);
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

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
}

pub struct Sgd<T: Scalar, B: Backend> {
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    velocity: Vec<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> Sgd<T, B> {
    #[must_use]
    pub fn new(lr: f64, param_shapes: &[(usize, usize)]) -> Self {
        let velocity = param_shapes.iter().map(|&(r, c)| Tensor::zeros(r, c)).collect();
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity,
        }
    }

    #[must_use]
    pub fn from_params(lr: f64, params: &[&Tensor<T, B>]) -> Self {
        let shapes: Vec<(usize, usize)> = params.iter().map(|p| p.shape()).collect();
        Self::new(lr, &shapes)
    }

    #[must_use]
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    #[must_use]
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    #[must_use]
    pub fn lr(&self) -> f64 { self.lr }

    #[must_use]
    pub fn momentum_value(&self) -> f64 { self.momentum }

    #[must_use]
    pub fn weight_decay_value(&self) -> f64 { self.weight_decay }

    #[must_use]
    pub fn velocity(&self) -> &[Tensor<T, B>] { &self.velocity }

    pub fn velocity_mut(&mut self) -> &mut [Tensor<T, B>] { &mut self.velocity }

    pub fn set_state(&mut self, lr: f64, momentum: f64, weight_decay: f64) {
        self.lr = lr;
        self.momentum = momentum;
        self.weight_decay = weight_decay;
    }
}

impl<T: Scalar, B: Backend> Optimizer<T, B> for Sgd<T, B> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]) {
        assert_eq!(params.len(), grads.len(), "nabla-train: SGD params/grads length mismatch");
        assert_eq!(params.len(), self.velocity.len(), "nabla-train: SGD params count mismatch");
        let lr_t = T::from_f64(self.lr);
        let momentum_t = T::from_f64(self.momentum);
        let wd_t = T::from_f64(self.weight_decay);
        let use_momentum = self.momentum != 0.0;
        let use_wd = self.weight_decay != 0.0;

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if !use_wd && !use_momentum {
                // Fast path: zero alloc, single kernel.
                param.axpy_inplace(-lr_t, grad);
            } else {
                let mut grad_eff = if use_wd { &(**grad) + &(&**param * wd_t) } else { (**grad).clone() };
                if use_momentum {
                    self.velocity[i] = &(&self.velocity[i] * momentum_t) + &grad_eff;
                    grad_eff = self.velocity[i].clone();
                }
                param.axpy_inplace(-lr_t, &grad_eff);
            }
        }
    }

    fn reset(&mut self) {
        for v in &mut self.velocity {
            let (r, c) = v.shape();
            *v = Tensor::zeros(r, c);
        }
    }

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_step<T: Scalar, B: Backend>(
    param: &mut Tensor<T, B>,
    grad: &Tensor<T, B>,
    m: &mut Tensor<T, B>,
    v: &mut Tensor<T, B>,
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

    // m = beta1 * m + (1-beta1) * grad  (in-place: 0 alloc)
    *m *= b1;
    m.axpy_inplace(one_minus_b1, grad);
    // v = beta2 * v + (1-beta2) * grad^2  (1 alloc for grad_sq)
    *v *= b2;
    let grad_sq = grad.emul(grad);
    v.axpy_inplace(one_minus_b2, &grad_sq);
    let denom = (&*v * inv_bias_corr2).powf(half).map(|x| x + eps_t);
    let update = (&*m * ss).ediv(&denom);
    // param = param * (1 - wd*lr) - update  (in-place: 0 alloc)
    let one_minus_wd_lr = T::one() - wd_lr;
    *param *= one_minus_wd_lr;
    param.axpy_inplace(-T::one(), &update);
}

impl<T: Scalar, B: Backend> OptimState<T, B> for AdamW<T, B> {
    fn kind(&self) -> OptimKind { OptimKind::AdamW }

    fn state_tensors(&self) -> Vec<(String, Tensor<T, B>)> {
        let mut out = Vec::with_capacity(self.m.len() * 2);
        for (i, t) in self.m.iter().enumerate() {
            out.push((format!("m{i}"), t.clone()));
        }
        for (i, t) in self.v.iter().enumerate() {
            out.push((format!("v{i}"), t.clone()));
        }
        out
    }

    fn load_state_tensors(&mut self, tensors: &[(String, Tensor<T, B>)]) -> Result<(), String> {
        let mut m = vec![None; self.m.len()];
        let mut v = vec![None; self.v.len()];
        for (name, t) in tensors {
            if let Some(idx) = name.strip_prefix('m') {
                let i = idx.parse::<usize>().map_err(|_| format!("bad tensor key: {name}"))?;
                if i >= m.len() { return Err(format!("m index out of range: {i}")); }
                m[i] = Some(t.clone());
            } else if let Some(idx) = name.strip_prefix('v') {
                let i = idx.parse::<usize>().map_err(|_| format!("bad tensor key: {name}"))?;
                if i >= v.len() { return Err(format!("v index out of range: {i}")); }
                v[i] = Some(t.clone());
            }
        }
        for (i, t) in m.into_iter().enumerate() {
            self.m[i] = t.ok_or_else(|| format!("missing m{i}"))?;
        }
        for (i, t) in v.into_iter().enumerate() {
            self.v[i] = t.ok_or_else(|| format!("missing v{i}"))?;
        }
        Ok(())
    }

    fn meta(&self) -> OptimMeta {
        OptimMeta {
            kind: OptimKind::AdamW,
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            momentum: 0.0,
            step_count: self.step_count,
        }
    }

    fn load_meta(&mut self, meta: &OptimMeta) -> Result<(), String> {
        if let OptimKind::AdamW = meta.kind {
            self.set_state(
                meta.step_count,
                meta.beta1,
                meta.beta2,
                meta.eps,
                meta.weight_decay,
                meta.lr,
            );
            Ok(())
        } else {
            Err("optim kind mismatch".to_owned())
        }
    }
}

impl<T: Scalar, B: Backend> OptimState<T, B> for Adam<T, B> {
    fn kind(&self) -> OptimKind { OptimKind::Adam }

    fn state_tensors(&self) -> Vec<(String, Tensor<T, B>)> {
        let mut out = Vec::with_capacity(self.m.len() * 2);
        for (i, t) in self.m.iter().enumerate() {
            out.push((format!("m{i}"), t.clone()));
        }
        for (i, t) in self.v.iter().enumerate() {
            out.push((format!("v{i}"), t.clone()));
        }
        out
    }

    fn load_state_tensors(&mut self, tensors: &[(String, Tensor<T, B>)]) -> Result<(), String> {
        let mut m = vec![None; self.m.len()];
        let mut v = vec![None; self.v.len()];
        for (name, t) in tensors {
            if let Some(idx) = name.strip_prefix('m') {
                let i = idx.parse::<usize>().map_err(|_| format!("bad tensor key: {name}"))?;
                if i >= m.len() { return Err(format!("m index out of range: {i}")); }
                m[i] = Some(t.clone());
            } else if let Some(idx) = name.strip_prefix('v') {
                let i = idx.parse::<usize>().map_err(|_| format!("bad tensor key: {name}"))?;
                if i >= v.len() { return Err(format!("v index out of range: {i}")); }
                v[i] = Some(t.clone());
            }
        }
        for (i, t) in m.into_iter().enumerate() {
            self.m[i] = t.ok_or_else(|| format!("missing m{i}"))?;
        }
        for (i, t) in v.into_iter().enumerate() {
            self.v[i] = t.ok_or_else(|| format!("missing v{i}"))?;
        }
        Ok(())
    }

    fn meta(&self) -> OptimMeta {
        OptimMeta {
            kind: OptimKind::Adam,
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: 0.0,
            momentum: 0.0,
            step_count: self.step_count,
        }
    }

    fn load_meta(&mut self, meta: &OptimMeta) -> Result<(), String> {
        if let OptimKind::Adam = meta.kind {
            self.set_state(
                meta.step_count,
                meta.beta1,
                meta.beta2,
                meta.eps,
                meta.lr,
            );
            Ok(())
        } else {
            Err("optim kind mismatch".to_owned())
        }
    }
}

impl<T: Scalar, B: Backend> OptimState<T, B> for Sgd<T, B> {
    fn kind(&self) -> OptimKind { OptimKind::Sgd }

    fn state_tensors(&self) -> Vec<(String, Tensor<T, B>)> {
        let mut out = Vec::with_capacity(self.velocity.len());
        for (i, t) in self.velocity.iter().enumerate() {
            out.push((format!("v{i}"), t.clone()));
        }
        out
    }

    fn load_state_tensors(&mut self, tensors: &[(String, Tensor<T, B>)]) -> Result<(), String> {
        let mut v = vec![None; self.velocity.len()];
        for (name, t) in tensors {
            if let Some(idx) = name.strip_prefix('v') {
                let i = idx.parse::<usize>().map_err(|_| format!("bad tensor key: {name}"))?;
                if i >= v.len() { return Err(format!("v index out of range: {i}")); }
                v[i] = Some(t.clone());
            }
        }
        for (i, t) in v.into_iter().enumerate() {
            self.velocity[i] = t.ok_or_else(|| format!("missing v{i}"))?;
        }
        Ok(())
    }

    fn meta(&self) -> OptimMeta {
        OptimMeta {
            kind: OptimKind::Sgd,
            lr: self.lr,
            beta1: 0.0,
            beta2: 0.0,
            eps: 0.0,
            weight_decay: self.weight_decay,
            momentum: self.momentum,
            step_count: 0,
        }
    }

    fn load_meta(&mut self, meta: &OptimMeta) -> Result<(), String> {
        if let OptimKind::Sgd = meta.kind {
            self.set_state(meta.lr, meta.momentum, meta.weight_decay);
            Ok(())
        } else {
            Err("optim kind mismatch".to_owned())
        }
    }
}
