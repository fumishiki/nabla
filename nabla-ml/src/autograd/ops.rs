use std::any::TypeId;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::constructors::seed_or_default;

use nabla_core::backend::Backend;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::core::{Tape, TapeEntry, Variable, accum_cell};

impl<T: Scalar, B: Backend> Variable<T, B> {
    /// Sum along `axis` (0 = column-wise → 1×n, 1 = row-wise → m×1).
    ///
    /// backward: broadcast grad back by expanding along the reduced axis.
    #[must_use]
    pub fn sum_axis_var(&self, axis: usize) -> Self {
        let out = self.data.sum_axis(axis);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let (in_rows, in_cols) = self.data.shape();
        let entry = TapeEntry::new(
            move |g| {
                // g has shape (1, ncols) or (nrows, 1); expand to input shape.
                Self::prop(&lr, &g.expand(in_rows, in_cols));
            },
            deps,
            "sum_axis",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Mean of all elements → scalar Variable of shape `(1, 1)`.
    ///
    /// backward: `grad / n_elements` broadcast to input shape.
    #[must_use]
    pub fn mean_var(&self) -> Self {
        let (nrows, ncols) = self.data.shape();
        let n = T::from_f64((nrows * ncols) as f64);
        let s = self.data.sum_all();
        let out = Tensor::fill(1, 1, s / n);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0) / n;
                Self::prop(&lr, &Tensor::fill(nrows, ncols, g_val));
            },
            deps,
            "mean",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Alias for [`Variable::sum_all_var`].
    #[must_use]
    pub fn sum(&self) -> Self {
        self.sum_all_var()
    }

    /// Alias for [`Variable::mean_var`].
    #[must_use]
    pub fn mean(&self) -> Self {
        self.mean_var()
    }

    /// Alias for [`Variable::sum_axis_var`].
    #[must_use]
    pub fn sum_axis(&self, axis: usize) -> Self {
        self.sum_axis_var(axis)
    }

    /// Mean along `axis` (0 = column-wise, 1 = row-wise).
    ///
    /// backward: broadcast `grad / axis_len` back by expanding along the reduced axis.
    #[must_use]
    pub fn mean_axis_var(&self, axis: usize) -> Self {
        let out = self.data.mean_axis(axis);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let (in_rows, in_cols) = self.data.shape();
        let axis_len = if axis == 0 { in_rows } else { in_cols };
        let inv_n = T::from_f64(1.0 / axis_len as f64);
        let entry = TapeEntry::new(
            move |g| {
                let scaled = g * inv_n;
                Self::prop(&lr, &scaled.expand(in_rows, in_cols));
            },
            deps,
            "mean_axis",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Alias for [`Variable::mean_axis_var`].
    #[must_use]
    pub fn mean_axis(&self, axis: usize) -> Self {
        self.mean_axis_var(axis)
    }

    /// Cross-entropy loss: fused log-softmax + NLL.
    ///
    /// Forward: `mean(-sum(targets * log_softmax(logits), axis=1))` per sample.
    /// Backward for logits: `(softmax(logits) - targets) / batch_size`.
    ///
    /// `targets` should be one-hot or probability tensors (not class indices).
    ///
    /// # Errors
    ///
    /// Returns `Err` if logits and targets shapes do not match.
    pub fn cross_entropy(&self, targets: &Self) -> Result<Self> {
        let targets_data = targets.data();
        let (batch, n) = self.data.shape();
        if targets_data.shape() != (batch, n) {
            return Err(nabla_core::error::Error::invalid(format!(
                "cross_entropy shape mismatch -- logits {}x{} vs targets {}x{}",
                batch,
                n,
                targets_data.nrows(),
                targets_data.ncols()
            )));
        }
        let log_sm = self.data.log_softmax(1);
        let loss_val = log_sm.cross_entropy_loss(targets_data);
        let out = Tensor::fill(1, 1, loss_val);

        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let logits_data = Rc::clone(&self.data);
        let tgt = targets_data.clone();
        let entry = TapeEntry::new(
            move |g| {
                let sm = logits_data.softmax(1);
                let inv_batch = T::from_f64(1.0 / batch as f64);
                let g_val = g.get(0, 0);
                // dL/d(logits) = (softmax - targets) / batch * upstream_grad
                let delta = &(&sm - &tgt) * (g_val * inv_batch);
                Self::prop(&lr, &delta);
            },
            deps,
            "cross_entropy",
        );
        Ok(Self::derived(&self.tape, out, entry))
    }

    /// Sum all elements → scalar Variable of shape `(1, 1)`.
    ///
    /// backward: broadcast `out_grad[0,0]` to fill the input shape.
    #[must_use]
    pub fn sum_all_var(&self) -> Self {
        let s = self.data.sum_all();
        let (nrows, ncols) = self.data.shape();
        let out = Tensor::fill(1, 1, s);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                Self::prop(&lr, &Tensor::fill(nrows, ncols, g_val));
            },
            deps,
            "sum",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Softmax along `axis`.
    ///
    /// backward: `grad * softmax - softmax * sum(grad * softmax, axis)`.
    #[must_use]
    pub fn softmax(&self, axis: usize) -> Self {
        let out = self.data.softmax(axis);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let sm = out.clone();
        let entry = TapeEntry::new(
            move |g| {
                let gs = g.emul(&sm);
                let sum_gs = gs.sum_axis(axis);
                let (m, n) = sm.shape();
                let delta = &gs - &sm.emul(&sum_gs.expand(m, n));
                Self::prop(&lr, &delta);
            },
            deps,
            "softmax",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Reshape to `(nrows, ncols)`.
    ///
    /// backward: reshape grad back to original shape.
    #[must_use]
    pub fn reshape(&self, nrows: usize, ncols: usize) -> Self {
        let (orig_r, orig_c) = self.data.shape();
        let out = self.data.reshape(nrows, ncols);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(
            move |g| {
                Self::prop(&lr, &g.reshape(orig_r, orig_c));
            },
            deps,
            "reshape",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Transpose.
    ///
    /// backward: transpose the gradient.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let out = self.data.t();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(
            move |g| {
                Self::prop(&lr, &g.t());
            },
            deps,
            "transpose",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Short alias for [`Variable::transpose`].
    #[must_use]
    pub fn t(&self) -> Self {
        self.transpose()
    }

    /// Linear forward: `x @ weight^T + bias` (all tracked).
    ///
    /// backward: `grad_x = g @ W`, `grad_w = g^T @ x`, `grad_b = sum(g, axis=0)`.
    #[must_use]
    pub fn linear_forward(&self, weight: &Self, bias: &Self) -> Self {
        let out = &(&*self.data * &(*weight.data).t()) + &*bias.data;
        let deps = Self::deps_of(&[self.entry_idx, weight.entry_idx, bias.entry_idx]);
        let (xr, wr, br) = (self.input_refs(), weight.input_refs(), bias.input_refs());
        let (x_data, w_data) = (Rc::clone(&self.data), Rc::clone(&weight.data));
        let entry = TapeEntry::new(
            move |g| {
                Self::prop(&xr, &(g * &*w_data));
                Self::prop(&wr, &(&g.t() * &*x_data));
                Self::prop(&br, &g.sum_axis(0));
            },
            deps,
            "linear",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Dropout with probability `p`. No-op when `training` is false.
    ///
    /// backward: `grad * mask * scale`.
    #[must_use]
    pub fn dropout(&self, p: f64, training: bool) -> Self {
        #[cfg(feature = "cuda")]
        assert!(
            TypeId::of::<B>() != TypeId::of::<nabla_core::backend::Cuda>(),
            "nabla: Variable::dropout is CPU-only on CUDA; GPU path must use a dedicated kernel"
        );
        #[cfg(feature = "hip")]
        assert!(
            TypeId::of::<B>() != TypeId::of::<nabla_core::backend::Hip>(),
            "nabla: Variable::dropout is CPU-only on HIP; GPU path must use a dedicated kernel"
        );
        #[cfg(feature = "gpu")]
        assert!(
            TypeId::of::<B>() != TypeId::of::<nabla_core::backend::Gpu>(),
            "nabla: Variable::dropout is CPU-only on WGPU; GPU path must use a dedicated kernel"
        );
        if !training || p <= 0.0 {
            return self.scale(T::one_impl()); // identity through tape
        }
        let (m, n) = self.data.shape();
        if p >= 1.0 {
            let out = Tensor::zeros(m, n);
            let deps = Self::deps_of(&[self.entry_idx]);
            let lr = self.input_refs();
            let entry = TapeEntry::new(
                move |_g| {
                    Self::prop(&lr, &Tensor::zeros(m, n));
                },
                deps,
                "dropout",
            );
            return Self::derived(&self.tape, out, entry);
        }
        let scale = T::from_f64(1.0 / (1.0 - p));
        let threshold = (p * (u64::MAX as f64)) as u64;
        let mut s = seed_or_default();
        // Build mask
        let mask = Tensor::from_fn(m, n, |_r, _c| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s < threshold {
                T::zero()
            } else {
                T::one_impl()
            }
        });
        let out = self.data.emul(&mask).emul(&Tensor::fill(m, n, scale));
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(
            move |g| {
                Self::prop(&lr, &g.emul(&mask).emul(&Tensor::fill(m, n, scale)));
            },
            deps,
            "dropout",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise clamp to `[lo, hi]`.
    ///
    /// backward: grad passes through where `lo <= x <= hi`, zero otherwise.
    #[must_use]
    pub fn clamp(&self, lo: T, hi: T) -> Self {
        let out = self.data.clamp(lo, hi);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(
            move |g| {
                let (m, n) = input.shape();
                let lo_t = Tensor::fill(1, 1, lo).expand(m, n);
                let hi_t = Tensor::fill(1, 1, hi).expand(m, n);
                let above_lo = &*input - &lo_t;
                let below_hi = &hi_t - &*input;
                let step_lo = (&above_lo + &above_lo.abs()).sign();
                let step_hi = (&below_hi + &below_hi.abs()).sign();
                let mask = step_lo.emul(&step_hi);
                Self::prop(&lr, &g.emul(&mask));
            },
            deps,
            "clamp",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Cross-entropy with integer class indices (as `T` values cast to usize).
    ///
    /// Converts indices to one-hot internally, then delegates to [`Variable::cross_entropy`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `targets` does not have exactly 1 column.
    pub fn cross_entropy_indices(&self, targets: &Tensor<T, B>) -> Result<Self> {
        if targets.ncols() != 1 {
            return Err(nabla_core::error::Error::invalid(
                "cross_entropy_indices: targets must be a column vector (ncols == 1)",
            ));
        }
        let (_batch, n) = self.data.shape();
        let one_hot = Tensor::from_storage(B::one_hot_from_indices(targets.storage(), n));
        let one_hot_var = self.tape.variable(one_hot)?;
        self.cross_entropy(&one_hot_var)
    }

    /// MSE loss: `mean((self - target)^2)` → scalar `(1,1)`.
    ///
    /// backward: `2 * (self - target) / n`.
    #[must_use]
    pub fn mse_loss(&self, target: &Self) -> Self {
        let diff = &*self.data - &*target.data;
        let (m, n) = diff.shape();
        let count = T::from_f64((m * n) as f64);
        let sq_sum = diff.emul(&diff).sum_all();
        let out = Tensor::fill(1, 1, sq_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let two_over_n = T::from_f64(2.0) / count;
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                let delta = &diff * (g_val * two_over_n);
                Self::prop(&lr, &delta);
                Self::prop(&rr, &(-&delta));
            },
            deps,
            "mse_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Fused MSE sum loss: `sum((self - target)^2)` -> scalar `(1,1)`.
    ///
    /// backward: `2 * (self - target) * upstream_grad`.
    #[must_use]
    pub fn mse_sum_loss(&self, target: &Self) -> Self {
        let pred_data = self.data.clone();
        let target_data = target.data.clone();
        let out = pred_data.mse_sum_loss_tensor(&target_data);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let entry = TapeEntry::new(
            move |g| {
                let grad_storage =
                    B::mse_sum_bwd(pred_data.storage(), target_data.storage(), g.storage());
                let delta = Tensor::from_storage(grad_storage);
                let neg_delta = -&delta;
                Self::prop_owned(&lr, delta);
                Self::prop_owned(&rr, neg_delta);
            },
            deps,
            "mse_sum_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// L1 loss: `mean(|self - target|)` -> scalar `(1,1)`.
    ///
    /// backward: `sign(self - target) / n` for self, negated for target.
    #[must_use]
    pub fn l1_loss(&self, target: &Self) -> Self {
        let diff = &*self.data - &*target.data;
        let (m, n) = diff.shape();
        let count = T::from_f64((m * n) as f64);
        let abs_sum = diff.abs().sum_all();
        let out = Tensor::fill(1, 1, abs_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let inv_n = T::one_impl() / count;
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                let sign = diff.sign();
                let delta = &sign * (g_val * inv_n);
                Self::prop(&lr, &delta);
                Self::prop(&rr, &(-&delta));
            },
            deps,
            "l1_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Huber loss: smooth L1 around `delta` threshold -> scalar `(1,1)`.
    ///
    /// Forward: if `|d| <= delta`, `0.5*d^2`; else `delta*(|d| - 0.5*delta)`. Mean reduction.
    /// backward: if `|d| <= delta`, `d/n`; else `delta*sign(d)/n`.
    #[must_use]
    pub fn huber_loss(&self, target: &Self, delta: T) -> Self {
        let diff = &*self.data - &*target.data;
        let (m, n) = diff.shape();
        let count = T::from_f64((m * n) as f64);
        let half = T::from_f64(0.5);
        let two = T::from_f64(2.0);
        let abs = diff.abs();
        let delta_t = Tensor::fill(1, 1, delta).expand(m, n);
        let min_abs = (&abs + &delta_t - (&abs - &delta_t).abs()) / two;
        let quad_base = min_abs.emul(&min_abs);
        let quad = &quad_base * half;
        let lin = delta_t.emul(&(&abs - &min_abs));
        let loss_sum = (quad + lin).sum_all();
        let out = Tensor::fill(1, 1, loss_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let inv_n = T::one_impl() / count;
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                let abs = diff.abs();
                let delta_t = Tensor::fill(1, 1, delta).expand(m, n);
                let min_abs = (&abs + &delta_t - (&abs - &delta_t).abs()) / two;
                let grad_base = diff.sign().emul(&min_abs);
                let grad_diff = &grad_base * (g_val * inv_n);
                Self::prop(&lr, &grad_diff);
                Self::prop(&rr, &(-&grad_diff));
            },
            deps,
            "huber_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Smooth L1 loss with transition point `beta` -> scalar `(1,1)`.
    ///
    /// Forward: if `|d| < beta`, `0.5*d^2/beta`; else `|d| - 0.5*beta`. Mean reduction.
    /// backward: if `|d| < beta`, `d/(beta*n)`; else `sign(d)/n`.
    #[must_use]
    pub fn smooth_l1_loss(&self, target: &Self, beta: T) -> Self {
        let diff = &*self.data - &*target.data;
        let (m, n) = diff.shape();
        let count = T::from_f64((m * n) as f64);
        let half = T::from_f64(0.5);
        let two = T::from_f64(2.0);
        let abs = diff.abs();
        let beta_t = Tensor::fill(1, 1, beta).expand(m, n);
        let min_abs = (&abs + &beta_t - (&abs - &beta_t).abs()) / two;
        let quad_base = min_abs.emul(&min_abs);
        let quad = &quad_base * (half / beta);
        let lin = &abs - &min_abs;
        let loss_sum = (quad + lin).sum_all();
        let out = Tensor::fill(1, 1, loss_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let inv_n = T::one_impl() / count;
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                let abs = diff.abs();
                let beta_t = Tensor::fill(1, 1, beta).expand(m, n);
                let min_abs = (&abs + &beta_t - (&abs - &beta_t).abs()) / two;
                let grad_base = diff.sign().emul(&min_abs);
                let grad_base = &grad_base * (T::one_impl() / beta);
                let grad_diff = &grad_base * (g_val * inv_n);
                Self::prop(&lr, &grad_diff);
                Self::prop(&rr, &(-&grad_diff));
            },
            deps,
            "smooth_l1_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Binary cross-entropy: `-mean(target*ln(self) + (1-target)*ln(1-self))` -> scalar `(1,1)`.
    ///
    /// backward for self: `(-target/self + (1-target)/(1-self)) / n`.
    #[must_use]
    pub fn binary_cross_entropy(&self, target: &Self) -> Self {
        let (m, n) = self.data.shape();
        let count = T::from_f64((m * n) as f64);
        let one = T::one_impl();
        let eps = T::from_f64(1e-12);
        let one_t = Tensor::fill(1, 1, one).expand(m, n);
        let p = self.data.clamp(eps, one - eps);
        let term1 = target.data.emul(&p.ln());
        let term2 = (&one_t - &*target.data).emul(&(&one_t - &p).ln());
        let loss_sum = (-&(term1 + term2)).sum_all();
        let out = Tensor::fill(1, 1, loss_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let p_saved = p.clone();
        let one_saved = one_t.clone();
        let tgt_data = Rc::clone(&target.data);
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                let inv_n = T::one_impl() / count;
                let one_minus_t = &one_saved - &*tgt_data;
                let one_minus_p = &one_saved - &p_saved;
                let grad_pred = -(tgt_data.ediv(&p_saved)) + one_minus_t.ediv(&one_minus_p);
                let grad_pred = &grad_pred * (g_val * inv_n);
                let grad_tgt = -(p_saved.ln()) + one_minus_p.ln();
                let grad_tgt = &grad_tgt * (g_val * inv_n);
                Self::prop(&lr, &grad_pred);
                Self::prop(&rr, &grad_tgt);
            },
            deps,
            "bce_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Binary cross-entropy with logits -> scalar `(1,1)`.
    ///
    /// backward for self: `(sigmoid(self) - target) / n`.
    #[must_use]
    pub fn bce_with_logits(&self, target: &Self) -> Self {
        let (m, n) = self.data.shape();
        let count = T::from_f64((m * n) as f64);
        let half = T::from_f64(0.5);
        let one = T::one_impl();
        let one_t = Tensor::fill(1, 1, one).expand(m, n);
        let x = &*self.data;
        let y = &*target.data;
        let abs_x = x.abs();
        let sum_x = x + &abs_x;
        let relu_x = &sum_x * half;
        let loss_sum =
            (relu_x - x.emul(y) + (&one_t + (-&abs_x).exp()).ln()).sum_all();
        let out = Tensor::fill(1, 1, loss_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let x_saved = Rc::clone(&self.data);
        let y_saved = Rc::clone(&target.data);
        let one_saved = one_t;
        let entry = TapeEntry::new(
            move |g| {
                let g_val = g.get(0, 0);
                let inv_n = T::one_impl() / count;
                let neg_x = -&*x_saved;
                let sig = one_saved.ediv(&(&one_saved + neg_x.exp()));
                let grad_pred = &(&sig - &*y_saved) * (g_val * inv_n);
                let grad_tgt = &(-&*x_saved) * (g_val * inv_n);
                Self::prop(&lr, &grad_pred);
                Self::prop(&rr, &grad_tgt);
            },
            deps,
            "bce_with_logits",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Negative log-likelihood loss from log-probabilities -> scalar `(1,1)`.
    ///
    /// `targets` should be a column vector of class indices.
    ///
    /// # Errors
    ///
    /// Returns `Err` if target shape mismatches or indices are out of bounds.
    pub fn nll_loss(&self, targets: &Tensor<T, B>) -> Result<Self> {
        let (batch, classes) = self.data.shape();
        if targets.ncols() != 1 || targets.nrows() != batch {
            return Err(nabla_core::error::Error::invalid(format!(
                "nll_loss: targets must be (batch,1), got {}x{}",
                targets.nrows(),
                targets.ncols()
            )));
        }
        let one_hot = Tensor::from_storage(B::one_hot_from_indices(targets.storage(), classes));
        let picked = self.data.emul(&one_hot).sum_axis(1);
        let loss_sum = (-&picked).sum_all();
        let out = Tensor::fill(1, 1, loss_sum / T::from_f64(batch as f64));
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let one_hot_saved = one_hot.clone();
        let entry = TapeEntry::new(
            move |g| {
                let g_val: T = g.get(0, 0);
                let inv_batch = T::from_f64(1.0 / batch as f64);
                let grad = &one_hot_saved * (-g_val * inv_batch);
                Self::prop(&lr, &grad);
            },
            deps,
            "nll_loss",
        );
        Ok(Self::derived(&self.tape, out, entry))
    }

    /// Cosine embedding loss for pairs `(self, other)` and label `y` in {1, -1}.
    #[must_use]
    pub fn cosine_embedding_loss(&self, other: &Self, y: T, margin: T) -> Self {
        let (m, n) = self.data.shape();
        assert_eq!(
            (m, n),
            other.data.shape(),
            "cosine_embedding_loss: shape mismatch"
        );
        let dot = self.data.emul(&other.data).sum_all();
        let n1 = self.data.norm();
        let n2 = other.data.norm();
        let eps = T::from_f64(1e-8);
        let denom = n1 * n2 + eps;
        let cos_sim = dot / denom;
        let (loss, gate) = if y.to_f64() > 0.0 {
            (T::one_impl() - cos_sim, T::from_f64(-1.0))
        } else {
            let v = cos_sim - margin;
            let two = T::from_f64(2.0);
            let loss = (v + v.math_abs()) / two;
            let gate = if v.to_f64() > 0.0 { T::one_impl() } else { T::zero() };
            (loss, gate)
        };
        let out = Tensor::fill(1, 1, loss);

        let deps = Self::deps_of(&[self.entry_idx, other.entry_idx]);
        let (lr, rr) = (self.input_refs(), other.input_refs());
        let x1 = Rc::clone(&self.data);
        let x2 = Rc::clone(&other.data);
        let entry = TapeEntry::new(
            move |g| {
                let g_val: T = g.get(0, 0);
                let x1_ref = &*x1;
                let x2_ref = &*x2;
                let dot = x1_ref.emul(x2_ref).sum_all();
                let n1 = x1_ref.norm();
                let n2 = x2_ref.norm();
                let eps = T::from_f64(1e-8);
                let denom = n1 * n2 + eps;
                let denom_sq = denom * denom;
                let n1_safe = if n1.to_f64() == 0.0 { eps } else { n1 };
                let n2_safe = if n2.to_f64() == 0.0 { eps } else { n2 };
                let coeff_x2 = T::one_impl() / denom;
                let coeff_x1 = dot * n2_safe / (n1_safe * denom_sq);
                let coeff_y1 = dot * n1_safe / (n2_safe * denom_sq);
                let grad_x1 = &(x2_ref * coeff_x2 - x1_ref * coeff_x1) * (g_val * gate);
                let grad_x2 = &(x1_ref * coeff_x2 - x2_ref * coeff_y1) * (g_val * gate);
                Self::prop(&lr, &grad_x1);
                Self::prop(&rr, &grad_x2);
            },
            deps,
            "cosine_embedding_loss",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise sign: -1, 0, or 1.
    ///
    /// backward: zero (subgradient convention).
    #[must_use]
    pub fn sign(&self) -> Self {
        let out = self.data.sign();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(
            move |g| {
                let (gm, gn) = g.shape();
                Self::prop(&lr, &Tensor::zeros(gm, gn));
            },
            deps,
            "sign",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Leaky ReLU: `max(alpha*x, x)`.
    ///
    /// backward: `grad * (1 if x > 0, alpha if x <= 0)`.
    #[must_use]
    pub fn leaky_relu(&self, alpha: f64) -> Self {
        let alpha_t = T::from_f64(alpha);
        let out = self.data.leaky_relu(alpha_t);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(
            move |g| {
                Self::prop_owned(&lr, g.leaky_relu_backward(&*input, alpha_t));
            },
            deps,
            "leaky_relu",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// ELU: `x if x > 0, alpha*(exp(x)-1) if x <= 0`.
    ///
    /// backward: `grad * (1 if x > 0, alpha*exp(x) if x <= 0)`.
    #[must_use]
    pub fn elu(&self, alpha: f64) -> Self {
        let alpha_t = T::from_f64(alpha);
        let out = self.data.elu(alpha_t);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(
            move |g| {
                Self::prop(&lr, &g.elu_backward(&*input, alpha_t));
            },
            deps,
            "elu",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Layer normalization over axis=1 (normalize each row).
    ///
    /// Forward: `(x - mean) / sqrt(var + eps)` per row.
    /// backward: full layer norm Jacobian.
    #[must_use]
    pub fn layer_norm(&self, eps: f64) -> Self {
        let eps_t = T::from_f64(eps);
        let out = self.data.layer_norm(1, eps_t);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(
            move |g| {
                let (m, n) = input.shape();
                let mean = input.mean_axis(1);
                let mean_exp = mean.expand(m, n);
                let diff = &*input - &mean_exp;
                let var = diff.emul(&diff).mean_axis(1);
                let eps_exp = Tensor::fill(1, 1, eps_t).expand(m, 1);
                let inv_std = (&var + &eps_exp).powf(T::from_f64(-0.5));
                let inv_exp = inv_std.expand(m, n);
                let x_hat = diff.emul(&inv_exp);
                let mean_g = g.mean_axis(1);
                let mean_gx = g.emul(&x_hat).mean_axis(1);
                let mean_g_exp = mean_g.expand(m, n);
                let mean_gx_exp = mean_gx.expand(m, n);
                let grad_out = (g - &mean_g_exp - x_hat.emul(&mean_gx_exp)).emul(&inv_exp);
                Self::prop(&lr, &grad_out);
            },
            deps,
            "layer_norm",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Group normalization over `num_groups` of channels (axis=1).
    ///
    /// Forward: per-row group norm, then affine `weight`/`bias`.
    #[must_use]
    pub fn group_norm(&self, num_groups: usize, weight: &Self, bias: &Self, eps: f64) -> Self {
        #[cfg(feature = "cuda")]
        assert!(
            TypeId::of::<B>() != TypeId::of::<nabla_core::backend::Cuda>(),
            "nabla: Variable::group_norm is CPU-only on CUDA; GPU path needs a dedicated backward kernel"
        );
        #[cfg(feature = "hip")]
        assert!(
            TypeId::of::<B>() != TypeId::of::<nabla_core::backend::Hip>(),
            "nabla: Variable::group_norm is CPU-only on HIP; GPU path needs a dedicated backward kernel"
        );
        #[cfg(feature = "gpu")]
        assert!(
            TypeId::of::<B>() != TypeId::of::<nabla_core::backend::Gpu>(),
            "nabla: Variable::group_norm is CPU-only on WGPU; GPU path needs a dedicated backward kernel"
        );
        let eps_t = T::from_f64(eps);
        let (m, n) = self.data.shape();
        assert!(
            n % num_groups == 0,
            "group_norm: channels {n} not divisible by groups {num_groups}"
        );
        let g_size = n / num_groups;
        let g_size_f = T::from_f64(g_size as f64);
        let out = Tensor::from_fn(m, n, |r, c| {
            let g = c / g_size;
            let g_start = g * g_size;
            let mean = (0..g_size).fold(T::zero(), |acc, j| acc + self.data.get(r, g_start + j))
                / g_size_f;
            let var = (0..g_size).fold(T::zero(), |acc, j| {
                let d = self.data.get(r, g_start + j) - mean;
                acc + d * d
            }) / g_size_f;
            let inv_std = T::one_impl() / (var + eps_t).math_sqrt();
            let x_hat = (self.data.get(r, c) - mean) * inv_std;
            x_hat * weight.data.get(0, c) + bias.data.get(0, c)
        });
        let deps = Self::deps_of(&[self.entry_idx, weight.entry_idx, bias.entry_idx]);
        let (xr, wr, br) = (self.input_refs(), weight.input_refs(), bias.input_refs());
        let input = Rc::clone(&self.data);
        let weight_data = Rc::clone(&weight.data);
        let entry = TapeEntry::new(
            move |g| {
                let (m, n) = input.shape();
                let g_size = n / num_groups;
                let g_size_f = T::from_f64(g_size as f64);
                let eps_t = T::from_f64(eps);

                let d_weight = Tensor::from_fn(1, n, |_, c| {
                    (0..m).fold(T::zero(), |acc, r| {
                        let g_idx = c / g_size;
                        let g_start = g_idx * g_size;
                        let mean = (0..g_size)
                            .fold(T::zero(), |acc2, j| acc2 + input.get(r, g_start + j))
                            / g_size_f;
                        let var = (0..g_size).fold(T::zero(), |acc2, j| {
                            let d = input.get(r, g_start + j) - mean;
                            acc2 + d * d
                        }) / g_size_f;
                        let inv_std = T::one_impl() / (var + eps_t).math_sqrt();
                        let x_hat = (input.get(r, c) - mean) * inv_std;
                        acc + g.get(r, c) * x_hat
                    })
                });
                Self::prop(&wr, &d_weight);

                let d_bias = Tensor::from_fn(1, n, |_, c| {
                    (0..m).fold(T::zero(), |acc, r| acc + g.get(r, c))
                });
                Self::prop(&br, &d_bias);

                let d_x = Tensor::from_fn(m, n, |r, c| {
                    let g_idx = c / g_size;
                    let g_start = g_idx * g_size;
                    let mean = (0..g_size)
                        .fold(T::zero(), |acc, j| acc + input.get(r, g_start + j))
                        / g_size_f;
                    let var = (0..g_size).fold(T::zero(), |acc, j| {
                        let d = input.get(r, g_start + j) - mean;
                        acc + d * d
                    }) / g_size_f;
                    let inv_std = T::one_impl() / (var + eps_t).math_sqrt();

                    let mean_gw = (0..g_size).fold(T::zero(), |acc, j| {
                        acc + g.get(r, g_start + j) * weight_data.get(0, g_start + j)
                    }) / g_size_f;
                    let mean_gw_xh = (0..g_size).fold(T::zero(), |acc, j| {
                        let xh = (input.get(r, g_start + j) - mean) * inv_std;
                        acc + g.get(r, g_start + j) * weight_data.get(0, g_start + j) * xh
                    }) / g_size_f;

                    let xh = (input.get(r, c) - mean) * inv_std;
                    let gw = g.get(r, c) * weight_data.get(0, c);
                    inv_std * (gw - mean_gw - xh * mean_gw_xh)
                });
                Self::prop(&xr, &d_x);
            },
            deps,
            "group_norm",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Lookup rows from an embedding table (self).
    ///
    /// `self` is `(vocab_size, embed_dim)`, indices selects rows.
    /// Returns `(indices.len(), embed_dim)`.
    /// backward: scatter-add grad rows back into embedding gradient.
    #[must_use]
    pub fn embedding_lookup(&self, indices: &[usize]) -> Self {
        let embed_dim = self.data.ncols();
        let idx_data: Vec<T> = indices
            .iter()
            .map(|&i| T::from_f64(i as f64))
            .collect();
        let idx_tensor = Tensor::from_vec(idx_data, indices.len(), 1);
        let out = Tensor::embedding(&idx_tensor, &self.data);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let (vocab, dim) = self.data.shape();
        let idx_saved = idx_tensor.clone();
        let entry = TapeEntry::new(
            move |g| {
                let grad_storage =
                    B::embedding_backward(idx_saved.storage(), g.storage(), vocab);
                let grad_w = Tensor::from_storage(grad_storage);
                Self::prop(&lr, &grad_w);
            },
            deps,
            "embedding",
        );
        Self::derived(&self.tape, out, entry)
    }

    /// Batch normalization across rows (batch dimension).
    ///
    /// Normalizes each column to zero mean and unit variance across the batch,
    /// then scales by `gamma` and shifts by `beta`.
    ///
    /// backward: full Jacobian with respect to x, gamma, and beta.
    #[must_use]
    pub fn batch_norm(&self, gamma: &Self, beta: &Self, eps: f64) -> Self {
        let eps_t = T::from_f64(eps);
        let (m, n) = self.data.shape();
        let m_f = T::from_f64(m as f64);
        let mean = self.data.mean_axis(0);
        let var = self.data.var_axis(0);
        let eps_tn = Tensor::fill(1, 1, eps_t).expand(1, n);
        let inv_std = (&var + &eps_tn).powf(T::from_f64(-0.5));
        let mean_exp = mean.expand(m, n);
        let inv_exp = inv_std.expand(m, n);
        let x_hat = (&*self.data - &mean_exp).emul(&inv_exp);
        let gamma_exp = gamma.data.expand(m, n);
        let beta_exp = beta.data.expand(m, n);
        let out = x_hat.emul(&gamma_exp) + &beta_exp;

        let deps = Self::deps_of(&[self.entry_idx, gamma.entry_idx, beta.entry_idx]);
        let (xr, gr, br) = (self.input_refs(), gamma.input_refs(), beta.input_refs());
        let gamma_data = Rc::clone(&gamma.data);
        let saved_x_hat = x_hat;
        let saved_inv_std = inv_std;
        let entry = TapeEntry::new(
            move |g| {
                let d_gamma = g.emul(&saved_x_hat).sum_axis(0);
                Self::prop(&gr, &d_gamma);

                let d_beta = g.sum_axis(0);
                Self::prop(&br, &d_beta);

                let gamma_exp = gamma_data.expand(m, n);
                let dx_hat = g.emul(&gamma_exp);
                let d_gamma_exp = d_gamma.expand(m, n);
                let d_beta_exp = d_beta.expand(m, n);
                let inv_std_exp = saved_inv_std.expand(m, n);
                let term = &dx_hat * m_f - &d_beta_exp - saved_x_hat.emul(&d_gamma_exp);
                let d_x_tmp = inv_std_exp.emul(&term);
                let d_x = &d_x_tmp * (T::one_impl() / m_f);
                Self::prop(&xr, &d_x);
            },
            deps,
            "batch_norm",
        );
        Self::derived(&self.tape, out, entry)
    }

    // -----------------------------------------------------------------------
    // Backward pass
    // -----------------------------------------------------------------------

    /// Run reverse-mode AD from this scalar `(1,1)` variable.
    ///
    /// Seeds this variable's gradient with `ones(1,1)`, then walks the tape
    /// in reverse order, calling each recorded backward closure.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the output is not scalar `(1,1)` or any gradient
    /// contains NaN or Inf. For non-scalar outputs, use [`Variable::backward_with`].
    pub fn backward(&self) -> Result<()> {
        let (nrows, ncols) = self.data.shape();
        if nrows != 1 || ncols != 1 {
            return Err(nabla_core::error::Error::invalid(
                "backward() requires scalar (1,1) output. For non-scalar, use backward_with(grad_output).",
            ));
        }
        self.backward_impl(&Tensor::fill(1, 1, T::one_impl()), true)
    }

    /// Run reverse-mode AD without NaN/Inf gradient validation.
    ///
    /// Skips the O(n) per-element check, useful in hot loops where inputs are
    /// known to be well-behaved. Prefer [`Variable::backward`] for safety.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the output is not scalar `(1,1)`.
    pub fn backward_unchecked(&self) -> Result<()> {
        let (nrows, ncols) = self.data.shape();
        if nrows != 1 || ncols != 1 {
            return Err(nabla_core::error::Error::invalid(
                "backward_unchecked() requires scalar (1,1) output. For non-scalar, use backward_with(grad_output).",
            ));
        }
        self.backward_impl(&Tensor::fill(1, 1, T::one_impl()), false)
    }

    /// Run reverse-mode AD with an explicit gradient seed.
    ///
    /// Use this for non-scalar outputs where `backward()` would error.
    /// `grad_output` must match this variable's shape.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `grad_output` shape mismatches or any gradient
    /// contains NaN or Inf.
    pub fn backward_with(&self, grad_output: &Tensor<T, B>) -> Result<()> {
        let (nrows, ncols) = self.data.shape();
        let (gr, gc) = grad_output.shape();
        if gr != nrows || gc != ncols {
            return Err(nabla_core::error::Error::invalid(format!(
                "backward_with: grad_output shape ({gr}, {gc}) must match output shape ({nrows}, {ncols})"
            )));
        }
        self.backward_impl(grad_output, true)
    }

    fn backward_impl(&self, seed: &Tensor<T, B>, check_nan: bool) -> Result<()> {
        if let Some(entry) = &self.tape_entry {
            entry.accum(seed);
        } else if let Some(slot) = &self.grad_slot {
            accum_cell(slot, seed);
        }

        let entries = self.tape.entries.borrow();
        let mut reachable = HashSet::new();
        if let Some(idx) = self.entry_idx {
            reachable.insert(idx);
        }
        for i in (0..entries.len()).rev() {
            if reachable.contains(&i) {
                for &dep in &entries[i].deps {
                    reachable.insert(dep);
                }
            }
        }

        for (i, entry) in entries.iter().enumerate().rev() {
            if !reachable.contains(&i) {
                continue;
            }
            let borrow = entry.grad.borrow();
            if let Some(ref g) = *borrow {
                if check_nan {
                    let (_, n) = g.shape();
                    let data = g.to_vec();
                    for (idx, v) in data.iter().enumerate() {
                        let f = v.to_f64();
                        if f.is_nan() || f.is_infinite() {
                            let r = idx / n;
                            let c = idx % n;
                            return Err(nabla_core::error::Error::eval(format!(
                                "NaN/Inf detected in gradient at ({r}, {c})"
                            )));
                        }
                    }
                }
                (entry.backward)(g);
            }
            drop(borrow);
        }

        Ok(())
    }
}

/// Generate 4 std::ops impls for a binary Variable op: &&, val-val, val-&, &-val.
macro_rules! impl_var_op {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl<T: Scalar, B: Backend> $trait for &Variable<T, B> {
            type Output = Variable<T, B>;
            fn $method(self, rhs: Self) -> Self::Output {
                self.$inner(rhs)
            }
        }
        impl<T: Scalar, B: Backend> $trait for Variable<T, B> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                self.$inner(&rhs)
            }
        }
        impl<T: Scalar, B: Backend> $trait<&Variable<T, B>> for Variable<T, B> {
            type Output = Self;
            fn $method(self, rhs: &Variable<T, B>) -> Self::Output {
                self.$inner(rhs)
            }
        }
        impl<T: Scalar, B: Backend> $trait<Variable<T, B>> for &Variable<T, B> {
            type Output = Variable<T, B>;
            fn $method(self, rhs: Variable<T, B>) -> Self::Output {
                self.$inner(&rhs)
            }
        }
    };
}

impl_var_op!(Add, add, add_var);
impl_var_op!(Sub, sub, sub_var);
impl_var_op!(Mul, mul, matmul);

impl<T: Scalar, B: Backend> Neg for &Variable<T, B> {
    type Output = Variable<T, B>;
    fn neg(self) -> Self::Output {
        self.neg_var()
    }
}

impl<T: Scalar, B: Backend> Neg for Variable<T, B> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.neg_var()
    }
}

impl<T: Scalar, B: Backend> Mul<T> for &Variable<T, B> {
    type Output = Variable<T, B>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Scalar, B: Backend> Mul<T> for Variable<T, B> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Scalar, B: Backend> Div<T> for &Variable<T, B> {
    type Output = Variable<T, B>;
    fn div(self, rhs: T) -> Self::Output {
        self.scale(T::one_impl() / rhs)
    }
}

impl<T: Scalar, B: Backend> Div<T> for Variable<T, B> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        (&self).div(rhs)
    }
}

/// Precomputed shape info for repeated gradient evaluation.
pub struct GradPrep<T: Scalar> {
    /// Expected input shape `(rows, cols)`.
    pub input_shape: (usize, usize),
    _phantom: PhantomData<T>,
}

/// Prepare a reusable gradient handle for `f` at point `x`.
pub fn gradient_prep<T, B, F>(_f: &F, x: &Tensor<T, B>) -> GradPrep<T>
where
    T: Scalar,
    B: Backend,
    F: Fn(&Variable<T, B>) -> Variable<T, B>,
{
    GradPrep {
        input_shape: x.shape(),
        _phantom: PhantomData,
    }
}

/// Compute the gradient of `f` at `x` using a prepared handle.
pub fn gradient<T, B, F>(f: &F, x: &Tensor<T, B>, prep: &GradPrep<T>) -> Result<Tensor<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(&Variable<T, B>) -> Variable<T, B>,
{
    if x.shape() != prep.input_shape {
        return Err(nabla_core::error::Error::invalid(format!(
            "gradient: input shape {:?} != prep shape {:?}",
            x.shape(),
            prep.input_shape
        )));
    }
    grad_impl(f, x)
}

/// Compute the gradient of scalar-valued `f` at `x` in one call.
pub fn grad<T, B, F>(f: F, x: &Tensor<T, B>) -> Result<Tensor<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(&Variable<T, B>) -> Variable<T, B>,
{
    grad_impl(&f, x)
}

fn grad_impl<T, B, F>(f: &F, x: &Tensor<T, B>) -> Result<Tensor<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(&Variable<T, B>) -> Variable<T, B>,
{
    let tape = Tape::new();
    let x_var = tape.variable(x.clone())?;
    let y_var = f(&x_var);
    y_var.backward()?;
    x_var.grad()
}

/// Clip gradient tensors by global L2 norm, returning the original norm.
pub fn clip_grad_norm<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>], max_norm: f64) -> f64 {
    let total_norm = grads
        .iter()
        .map(|g| g.norm().to_f64().powi(2))
        .sum::<f64>()
        .sqrt();
    if total_norm > max_norm {
        let scale = T::from_f64(max_norm / total_norm);
        for g in grads.iter_mut() {
            *g = &*g * scale;
        }
    }
    total_norm
}

/// Zero out all gradient tensors in-place.
pub fn zero_grad<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>]) {
    for g in grads.iter_mut() {
        let (r, c) = g.shape();
        *g = Tensor::zeros(r, c);
    }
}

/// Multiply all gradient tensors by a scalar factor.
pub fn scale_grad<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>], factor: T) {
    for g in grads.iter_mut() {
        *g = &*g * factor;
    }
}
