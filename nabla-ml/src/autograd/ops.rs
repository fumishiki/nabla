use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::constructors::seed_or_default;

use nabla_core::backend::Backend;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::core::{accum_cell, Tape, TapeEntry, Variable};

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
        let entry = TapeEntry::new(move |g| {
            // g has shape (1, ncols) or (nrows, 1); expand to input shape.
            Self::prop(&lr, &g.expand(in_rows, in_cols));
        }, deps, "sum_axis");
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
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0) / n;
            Self::prop(&lr, &Tensor::fill(nrows, ncols, g_val));
        }, deps, "mean");
        Self::derived(&self.tape, out, entry)
    }

    /// Alias for [`Variable::sum_all_var`].
    #[must_use]
    pub fn sum(&self) -> Self { self.sum_all_var() }

    /// Alias for [`Variable::mean_var`].
    #[must_use]
    pub fn mean(&self) -> Self { self.mean_var() }

    /// Alias for [`Variable::sum_axis_var`].
    #[must_use]
    pub fn sum_axis(&self, axis: usize) -> Self { self.sum_axis_var(axis) }

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
        let entry = TapeEntry::new(move |g| {
            let scaled = g * inv_n;
            Self::prop(&lr, &scaled.expand(in_rows, in_cols));
        }, deps, "mean_axis");
        Self::derived(&self.tape, out, entry)
    }

    /// Alias for [`Variable::mean_axis_var`].
    #[must_use]
    pub fn mean_axis(&self, axis: usize) -> Self { self.mean_axis_var(axis) }

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
                batch, n, targets_data.nrows(), targets_data.ncols()
            )));
        }
        let log_sm = self.data.log_softmax(1);
        let loss_val = log_sm.cross_entropy_loss(targets_data);
        let out = Tensor::fill(1, 1, loss_val);

        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let logits_data = Rc::clone(&self.data);
        let tgt = targets_data.clone();
        let entry = TapeEntry::new(move |g| {
            let sm = logits_data.softmax(1);
            let inv_batch = T::from_f64(1.0 / batch as f64);
            let g_val = g.get(0, 0);
            // dL/d(logits) = (softmax - targets) / batch * upstream_grad
            let delta = &(&sm - &tgt) * (g_val * inv_batch);
            Self::prop(&lr, &delta);
        }, deps, "cross_entropy");
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
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            Self::prop(&lr, &Tensor::fill(nrows, ncols, g_val));
        }, deps, "sum");
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
        let entry = TapeEntry::new(move |g| {
            let gs = g.emul(&sm);
            let sum_gs = gs.sum_axis(axis);
            let (m, n) = sm.shape();
            let delta = &gs - &sm.emul(&sum_gs.expand(m, n));
            Self::prop(&lr, &delta);
        }, deps, "softmax");
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
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.reshape(orig_r, orig_c));
        }, deps, "reshape");
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
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.t());
        }, deps, "transpose");
        Self::derived(&self.tape, out, entry)
    }

    /// Short alias for [`Variable::transpose`].
    #[must_use]
    pub fn t(&self) -> Self { self.transpose() }

    /// Linear forward: `x @ weight^T + bias` (all tracked).
    ///
    /// backward: `grad_x = g @ W`, `grad_w = g^T @ x`, `grad_b = sum(g, axis=0)`.
    #[must_use]
    pub fn linear_forward(&self, weight: &Self, bias: &Self) -> Self {
        let out = &(&*self.data * &(*weight.data).t()) + &*bias.data;
        let deps = Self::deps_of(&[self.entry_idx, weight.entry_idx, bias.entry_idx]);
        let (xr, wr, br) = (self.input_refs(), weight.input_refs(), bias.input_refs());
        let (x_data, w_data) = (Rc::clone(&self.data), Rc::clone(&weight.data));
        let entry = TapeEntry::new(move |g| {
            Self::prop(&xr, &(g * &*w_data));
            Self::prop(&wr, &(&g.t() * &*x_data));
            Self::prop(&br, &g.sum_axis(0));
        }, deps, "linear");
        Self::derived(&self.tape, out, entry)
    }

    /// Dropout with probability `p`. No-op when `training` is false.
    ///
    /// backward: `grad * mask * scale`.
    #[must_use]
    pub fn dropout(&self, p: f64, training: bool) -> Self {
        if !training || p <= 0.0 {
            return self.scale(T::one_impl()); // identity through tape
        }
        let (m, n) = self.data.shape();
        if p >= 1.0 {
            let out = Tensor::zeros(m, n);
            let deps = Self::deps_of(&[self.entry_idx]);
            let lr = self.input_refs();
            let entry = TapeEntry::new(move |_g| {
                Self::prop(&lr, &Tensor::zeros(m, n));
            }, deps, "dropout");
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
            if s < threshold { T::zero() } else { T::one_impl() }
        });
        let out = self.data.emul(&mask).emul(&Tensor::fill(m, n, scale));
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&mask).emul(&Tensor::fill(m, n, scale)));
        }, deps, "dropout");
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
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let lo_f = lo.to_f64();
            let hi_f = hi.to_f64();
            let mask = Tensor::from_fn(m, n, |r, c| {
                let v = input.get(r, c).to_f64();
                if v >= lo_f && v <= hi_f { T::one_impl() } else { T::zero() }
            });
            Self::prop(&lr, &g.emul(&mask));
        }, deps, "clamp");
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
                "cross_entropy_indices: targets must be a column vector (ncols == 1)"
            ));
        }
        let (batch, n) = self.data.shape();
        let one_hot = Tensor::from_fn(batch, n, |r, c| {
            let idx = targets.get(r, 0).to_f64() as usize;
            if c == idx { T::one_impl() } else { T::zero() }
        });
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
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let delta = &diff * (g_val * two_over_n);
            Self::prop(&lr, &delta);
            Self::prop(&rr, &(-&delta));
        }, deps, "mse_loss");
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
        let entry = TapeEntry::new(move |g| {
            let grad_storage = B::mse_sum_bwd(pred_data.storage(), target_data.storage(), g.storage());
            let delta = Tensor::from_storage(grad_storage);
            Self::prop(&lr, &delta);
            Self::prop(&rr, &(-&delta));
        }, deps, "mse_sum_loss");
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
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let (dm, dn) = diff.shape();
            let sign = Tensor::from_fn(dm, dn, |r, c| {
                let v = diff.get(r, c).to_f64();
                if v > 0.0 { T::one_impl() } else if v < 0.0 { T::zero() - T::one_impl() } else { T::zero() }
            });
            let delta = &sign * (g_val * inv_n);
            Self::prop(&lr, &delta);
            Self::prop(&rr, &(-&delta));
        }, deps, "l1_loss");
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
        let delta_f = delta.to_f64();
        let mut total = T::zero();
        for r in 0..m {
            for c in 0..n {
                let d = diff.get(r, c).to_f64().abs();
                total = total + if d <= delta_f {
                    T::from_f64(0.5 * d * d)
                } else {
                    delta * (T::from_f64(d) - half * delta)
                };
            }
        }
        let out = Tensor::fill(1, 1, total / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let inv_n = T::one_impl() / count;
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let (dm, dn) = diff.shape();
            let grad_diff = Tensor::from_fn(dm, dn, |r, c| {
                let d = diff.get(r, c);
                let ad = d.to_f64().abs();
                if ad <= delta_f { d * (g_val * inv_n) } else {
                    let s = if d.to_f64() > 0.0 { T::one_impl() } else { T::zero() - T::one_impl() };
                    delta * s * (g_val * inv_n)
                }
            });
            Self::prop(&lr, &grad_diff);
            Self::prop(&rr, &(-&grad_diff));
        }, deps, "huber_loss");
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
        let beta_f = beta.to_f64();
        let mut total = T::zero();
        for r in 0..m {
            for c in 0..n {
                let d = diff.get(r, c);
                let ad = d.to_f64().abs();
                total = total + if ad < beta_f {
                    half * d * d / beta
                } else {
                    T::from_f64(ad) - half * beta
                };
            }
        }
        let out = Tensor::fill(1, 1, total / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let inv_n = T::one_impl() / count;
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let (dm, dn) = diff.shape();
            let grad_diff = Tensor::from_fn(dm, dn, |r, c| {
                let d = diff.get(r, c);
                let ad = d.to_f64().abs();
                if ad < beta_f {
                    d * (g_val * inv_n) / beta
                } else if d.to_f64() > 0.0 {
                    T::one_impl() * (g_val * inv_n)
                } else if d.to_f64() < 0.0 {
                    (T::zero() - T::one_impl()) * (g_val * inv_n)
                } else {
                    T::zero()
                }
            });
            Self::prop(&lr, &grad_diff);
            Self::prop(&rr, &(-&grad_diff));
        }, deps, "smooth_l1_loss");
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
        let loss_sum = {
            let mut s = T::zero();
            for r in 0..m {
                for c in 0..n {
                    let p = self.data.get(r, c);
                    let t = target.data.get(r, c);
                    let p_clamped = if p.to_f64() < eps.to_f64() { eps } else if p.to_f64() > (one - eps).to_f64() { one - eps } else { p };
                    s = s - (t * T::from_f64(p_clamped.to_f64().ln()) + (one - t) * T::from_f64((one - p_clamped).to_f64().ln()));
                }
            }
            s
        };
        let out = Tensor::fill(1, 1, loss_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let pred_data = Rc::clone(&self.data);
        let tgt_data = Rc::clone(&target.data);
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let inv_n = T::one_impl() / count;
            let (pm, pn) = pred_data.shape();
            let grad_pred = Tensor::from_fn(pm, pn, |r, c| {
                let p = pred_data.get(r, c);
                let t = tgt_data.get(r, c);
                let p_c = if p.to_f64() < eps.to_f64() { eps } else if p.to_f64() > (one - eps).to_f64() { one - eps } else { p };
                (T::zero() - t / p_c + (one - t) / (one - p_c)) * (g_val * inv_n)
            });
            let grad_tgt = Tensor::from_fn(pm, pn, |r, c| {
                let p = pred_data.get(r, c);
                let p_c = if p.to_f64() < eps.to_f64() { eps } else if p.to_f64() > (one - eps).to_f64() { one - eps } else { p };
                (T::zero() - T::from_f64(p_c.to_f64().ln()) + T::from_f64((one - p_c).to_f64().ln())) * (g_val * inv_n)
            });
            Self::prop(&lr, &grad_pred);
            Self::prop(&rr, &grad_tgt);
        }, deps, "bce_loss");
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
        let loss_sum = {
            let mut s = T::zero();
            for r in 0..m {
                for c in 0..n {
                    let x = self.data.get(r, c);
                    let y = target.data.get(r, c);
                    let abs_x = x.math_abs();
                    let relu_x = (x + abs_x) * half;
                    s = s + relu_x - x * y
                        + (T::one_impl() + (T::zero() - abs_x).math_exp()).math_ln();
                }
            }
            s
        };
        let out = Tensor::fill(1, 1, loss_sum / count);
        let deps = Self::deps_of(&[self.entry_idx, target.entry_idx]);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let pred_data = Rc::clone(&self.data);
        let tgt_data = Rc::clone(&target.data);
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let inv_n = T::one_impl() / count;
            let (pm, pn) = pred_data.shape();
            let grad_pred = Tensor::from_fn(pm, pn, |r, c| {
                let x = pred_data.get(r, c);
                let y = tgt_data.get(r, c);
                let sig = T::one_impl() / (T::one_impl() + (T::zero() - x).math_exp());
                (sig - y) * (g_val * inv_n)
            });
            let grad_tgt = Tensor::from_fn(pm, pn, |r, c| {
                let x = pred_data.get(r, c);
                (T::zero() - x) * (g_val * inv_n)
            });
            Self::prop(&lr, &grad_pred);
            Self::prop(&rr, &grad_tgt);
        }, deps, "bce_with_logits");
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
        let mut sum = T::zero();
        for r in 0..batch {
            let idx_f = targets.get(r, 0).to_f64();
            if !idx_f.is_finite() || idx_f < 0.0 || idx_f >= classes as f64 {
                return Err(nabla_core::error::Error::invalid(format!(
                    "nll_loss: target index out of bounds at row {r}: {idx_f}"
                )));
            }
            let idx = idx_f as usize;
            sum = sum - self.data.get(r, idx);
        }
        let out = Tensor::fill(1, 1, sum / T::from_f64(batch as f64));
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let targets_data = targets.clone();
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let inv_batch = T::from_f64(1.0 / batch as f64);
            let mut grad = Tensor::zeros(batch, classes);
            for r in 0..batch {
                let idx = targets_data.get(r, 0).to_f64() as usize;
                let old = grad.get(r, idx);
                grad.set(r, idx, old - g_val * inv_batch);
            }
            Self::prop(&lr, &grad);
        }, deps, "nll_loss");
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
        let dot = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |a, c| a + self.data.get(r, c) * other.data.get(r, c))
        });
        let n1 = self.data.norm();
        let n2 = other.data.norm();
        let eps = T::from_f64(1e-8);
        let denom = n1 * n2 + eps;
        let cos_sim = dot / denom;
        let loss = if y.to_f64() > 0.0 {
            T::one_impl() - cos_sim
        } else {
            let v = cos_sim - margin;
            let two = T::from_f64(2.0);
            (v + v.math_abs()) / two
        };
        let out = Tensor::fill(1, 1, loss);

        let deps = Self::deps_of(&[self.entry_idx, other.entry_idx]);
        let (lr, rr) = (self.input_refs(), other.input_refs());
        let x1 = Rc::clone(&self.data);
        let x2 = Rc::clone(&other.data);
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let (m, n) = x1.shape();
            let dot = (0..m).fold(T::zero(), |acc, r| {
                (0..n).fold(acc, |a, c| a + x1.get(r, c) * x2.get(r, c))
            });
            let n1 = x1.norm();
            let n2 = x2.norm();
            let eps = T::from_f64(1e-8);
            let denom = n1 * n2 + eps;
            let cos_sim = dot / denom;
            let v = cos_sim - margin;
            let gate = if y.to_f64() > 0.0 {
                T::from_f64(-1.0)
            } else if v.to_f64() > 0.0 {
                T::one_impl()
            } else {
                T::zero()
            };
            let denom_sq = denom * denom;
            let n1_safe = if n1.to_f64() == 0.0 { eps } else { n1 };
            let n2_safe = if n2.to_f64() == 0.0 { eps } else { n2 };
            let coeff_x2 = T::one_impl() / denom;
            let coeff_x1 = dot * n2_safe / (n1_safe * denom_sq);
            let coeff_y1 = dot * n1_safe / (n2_safe * denom_sq);
            let grad_x1 = Tensor::from_fn(m, n, |r, c| {
                let term = x2.get(r, c) * coeff_x2 - x1.get(r, c) * coeff_x1;
                term * (g_val * gate)
            });
            let grad_x2 = Tensor::from_fn(m, n, |r, c| {
                let term = x1.get(r, c) * coeff_x2 - x2.get(r, c) * coeff_y1;
                term * (g_val * gate)
            });
            Self::prop(&lr, &grad_x1);
            Self::prop(&rr, &grad_x2);
        }, deps, "cosine_embedding_loss");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise sign: -1, 0, or 1.
    ///
    /// backward: zero (subgradient convention).
    #[must_use]
    pub fn sign(&self) -> Self {
        let (m, n) = self.data.shape();
        let out = Tensor::from_fn(m, n, |r, c| {
            let x = self.data.get(r, c).to_f64();
            if x > 0.0 {
                T::one_impl()
            } else if x < 0.0 {
                T::zero() - T::one_impl()
            } else {
                T::zero()
            }
        });
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            let (gm, gn) = g.shape();
            Self::prop(&lr, &Tensor::zeros(gm, gn));
        }, deps, "sign");
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
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.leaky_relu_backward(&*input, alpha_t));
        }, deps, "leaky_relu");
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
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.elu_backward(&*input, alpha_t));
        }, deps, "elu");
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
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let n_f = T::from_f64(n as f64);
            // Full layer norm backward per row:
            // dx = (1/std) * (dout - mean(dout) - x_hat * mean(dout * x_hat))
            let grad_out = Tensor::from_fn(m, n, |r, c| {
                // Compute row mean and variance
                let row_mean = (0..n).fold(T::zero(), |acc, j| acc + input.get(r, j)) / n_f;
                let row_var = (0..n).fold(T::zero(), |acc, j| {
                    let d = input.get(r, j) - row_mean;
                    acc + d * d
                }) / n_f;
                let inv_std = T::one_impl() / (row_var + eps_t).math_sqrt();
                let x_hat = (input.get(r, c) - row_mean) * inv_std;

                // mean(dout) and mean(dout * x_hat) for this row
                let mean_g = (0..n).fold(T::zero(), |acc, j| acc + g.get(r, j)) / n_f;
                let mean_gx = (0..n).fold(T::zero(), |acc, j| {
                    let xh = (input.get(r, j) - row_mean) * inv_std;
                    acc + g.get(r, j) * xh
                }) / n_f;

                inv_std * (g.get(r, c) - mean_g - x_hat * mean_gx)
            });
            Self::prop(&lr, &grad_out);
        }, deps, "layer_norm");
        Self::derived(&self.tape, out, entry)
    }

    /// Group normalization over `num_groups` of channels (axis=1).
    ///
    /// Forward: per-row group norm, then affine `weight`/`bias`.
    #[must_use]
    pub fn group_norm(&self, num_groups: usize, weight: &Self, bias: &Self, eps: f64) -> Self {
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
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let g_size = n / num_groups;
            let g_size_f = T::from_f64(g_size as f64);
            let eps_t = T::from_f64(eps);

            let d_weight = Tensor::from_fn(1, n, |_, c| {
                (0..m).fold(T::zero(), |acc, r| {
                    let g_idx = c / g_size;
                    let g_start = g_idx * g_size;
                    let mean = (0..g_size).fold(T::zero(), |acc2, j| {
                        acc2 + input.get(r, g_start + j)
                    }) / g_size_f;
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
                let mean = (0..g_size).fold(T::zero(), |acc, j| {
                    acc + input.get(r, g_start + j)
                }) / g_size_f;
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
        }, deps, "group_norm");
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
        let out = Tensor::from_fn(indices.len(), embed_dim, |r, c| {
            self.data.get(indices[r], c)
        });
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let (vocab, dim) = self.data.shape();
        let idx = indices.to_vec();
        let entry = TapeEntry::new(move |g| {
            // Scatter-add: accumulate grad rows back into embedding positions.
            let mut grad_w = Tensor::zeros(vocab, dim);
            for (i, &row_idx) in idx.iter().enumerate() {
                for c in 0..dim {
                    let old = grad_w.get(row_idx, c);
                    let delta = g.get(i, c);
                    grad_w.set(row_idx, c, old + delta);
                }
            }
            Self::prop(&lr, &grad_w);
        }, deps, "embedding");
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

        // Compute per-column mean and variance across batch (rows).
        let col_mean: Tensor<T, B> = Tensor::from_fn(1, n, |_, c| {
            (0..m).fold(T::zero(), |acc, r| acc + self.data.get(r, c)) / m_f
        });
        let col_var: Tensor<T, B> = Tensor::<T, B>::from_fn(1, n, |_, c| {
            let mu = col_mean.get(0, c);
            (0..m).fold(T::zero(), |acc, r| {
                let d = self.data.get(r, c) - mu;
                acc + d * d
            }) / m_f
        });

        // x_hat = (x - mean) / sqrt(var + eps)
        let x_hat: Tensor<T, B> = Tensor::from_fn(m, n, |r, c| {
            let mu = col_mean.get(0, c);
            let inv_std = T::one_impl() / (col_var.get(0, c) + eps_t).math_sqrt();
            (self.data.get(r, c) - mu) * inv_std
        });

        // out = gamma * x_hat + beta
        let out = Tensor::from_fn(m, n, |r, c| {
            gamma.data.get(0, c) * x_hat.get(r, c) + beta.data.get(0, c)
        });

        let deps = Self::deps_of(&[self.entry_idx, gamma.entry_idx, beta.entry_idx]);
        let (xr, gr, br) = (self.input_refs(), gamma.input_refs(), beta.input_refs());
        let input = Rc::clone(&self.data);
        let gamma_data = Rc::clone(&gamma.data);
        let saved_x_hat = x_hat;
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let m_f = T::from_f64(m as f64);
            let eps_t = T::from_f64(eps);

            // d_gamma = sum(g * x_hat, axis=0)  →  (1, n)
            let d_gamma = Tensor::from_fn(1, n, |_, c| {
                (0..m).fold(T::zero(), |acc, r| acc + g.get(r, c) * saved_x_hat.get(r, c))
            });
            Self::prop(&gr, &d_gamma);

            // d_beta = sum(g, axis=0)  →  (1, n)
            let d_beta = Tensor::from_fn(1, n, |_, c| {
                (0..m).fold(T::zero(), |acc, r| acc + g.get(r, c))
            });
            Self::prop(&br, &d_beta);

            // d_x_hat = g * gamma  →  (m, n)
            // d_x = (1/sqrt(var+eps)) * (d_x_hat - mean(d_x_hat) - x_hat * mean(d_x_hat * x_hat))
            let d_x = Tensor::from_fn(m, n, |r, c| {
                // Recompute per-column stats
                let mu = (0..m).fold(T::zero(), |acc, ri| acc + input.get(ri, c)) / m_f;
                let var = (0..m).fold(T::zero(), |acc, ri| {
                    let d = input.get(ri, c) - mu;
                    acc + d * d
                }) / m_f;
                let inv_std = T::one_impl() / (var + eps_t).math_sqrt();

                let gam = gamma_data.get(0, c);
                // d_x_hat_j = g_j * gamma for column c
                let mean_dxh = (0..m).fold(T::zero(), |acc, ri| acc + g.get(ri, c) * gam) / m_f;
                let mean_dxh_xh = (0..m).fold(T::zero(), |acc, ri| {
                    let xh = (input.get(ri, c) - mu) * inv_std;
                    acc + g.get(ri, c) * gam * xh
                }) / m_f;

                let xh = (input.get(r, c) - mu) * inv_std;
                let dxh = g.get(r, c) * gam;
                inv_std * (dxh - mean_dxh - xh * mean_dxh_xh)
            });
            Self::prop(&xr, &d_x);
        }, deps, "batch_norm");
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
                "backward() requires scalar (1,1) output. For non-scalar, use backward_with(grad_output)."
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
                "backward_unchecked() requires scalar (1,1) output. For non-scalar, use backward_with(grad_output)."
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
                            return Err(nabla_core::error::Error::eval(
                                format!("NaN/Inf detected in gradient at ({r}, {c})")
                            ));
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
            fn $method(self, rhs: Self) -> Self::Output { self.$inner(rhs) }
        }
        impl<T: Scalar, B: Backend> $trait for Variable<T, B> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output { self.$inner(&rhs) }
        }
        impl<T: Scalar, B: Backend> $trait<&Variable<T, B>> for Variable<T, B> {
            type Output = Self;
            fn $method(self, rhs: &Variable<T, B>) -> Self::Output { self.$inner(rhs) }
        }
        impl<T: Scalar, B: Backend> $trait<Variable<T, B>> for &Variable<T, B> {
            type Output = Variable<T, B>;
            fn $method(self, rhs: Variable<T, B>) -> Self::Output { self.$inner(&rhs) }
        }
    };
}

impl_var_op!(Add, add, add_var);
impl_var_op!(Sub, sub, sub_var);
impl_var_op!(Mul, mul, matmul);

impl<T: Scalar, B: Backend> Neg for &Variable<T, B> {
    type Output = Variable<T, B>;
    fn neg(self) -> Self::Output { self.neg_var() }
}

impl<T: Scalar, B: Backend> Neg for Variable<T, B> {
    type Output = Self;
    fn neg(self) -> Self::Output { self.neg_var() }
}

impl<T: Scalar, B: Backend> Mul<T> for &Variable<T, B> {
    type Output = Variable<T, B>;
    fn mul(self, rhs: T) -> Self::Output { self.scale(rhs) }
}

impl<T: Scalar, B: Backend> Mul<T> for Variable<T, B> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output { self.scale(rhs) }
}

impl<T: Scalar, B: Backend> Div<T> for &Variable<T, B> {
    type Output = Variable<T, B>;
    fn div(self, rhs: T) -> Self::Output { self.scale(T::one_impl() / rhs) }
}

impl<T: Scalar, B: Backend> Div<T> for Variable<T, B> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output { (&self).div(rhs) }
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
pub fn gradient<T, B, F>(
    f: &F,
    x: &Tensor<T, B>,
    prep: &GradPrep<T>,
) -> Result<Tensor<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(&Variable<T, B>) -> Variable<T, B>,
{
    if x.shape() != prep.input_shape {
        return Err(nabla_core::error::Error::invalid(format!(
            "gradient: input shape {:?} != prep shape {:?}",
            x.shape(), prep.input_shape
        )));
    }
    grad_impl(f, x)
}

/// Compute the gradient of scalar-valued `f` at `x` in one call.
pub fn grad<T, B, F>(
    f: F,
    x: &Tensor<T, B>,
) -> Result<Tensor<T, B>>
where
    T: Scalar,
    B: Backend,
    F: Fn(&Variable<T, B>) -> Variable<T, B>,
{
    grad_impl(&f, x)
}

fn grad_impl<T, B, F>(
    f: &F,
    x: &Tensor<T, B>,
) -> Result<Tensor<T, B>>
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
