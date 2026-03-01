// tensor/nn/norm.rs — Normalization and dropout.

use core::marker::PhantomData;

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::{two, Tensor};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    fn axis_is_rows(axis: usize, op: &str) -> bool {
        assert!(axis <= 1, "nabla: {op} axis must be 0 or 1, got {axis}");
        axis == 1
    }

    /// Softmax along given axis (0=columns, 1=rows).
    #[must_use]
    pub fn softmax(&self, axis: usize) -> Self {
        if Self::axis_is_rows(axis, "softmax") {
            Self::from_storage(B::softmax(&self.storage))
        } else {
            self.t().softmax(1).t()
        }
    }

    /// Numerically stable log-softmax along `axis` (0 = columns, 1 = rows).
    #[must_use]
    pub fn log_softmax(&self, axis: usize) -> Self {
        if !Self::axis_is_rows(axis, "log_softmax") {
            return self.t().log_softmax(1).t();
        }
        let (m, n) = self.shape();
        let two = two::<T>();
        Self::from_fn(m, n, |r, c| {
            let row_max = (0..n).fold(self.get(r, 0), |acc, j| {
                let v = self.get(r, j);
                (acc + v + (acc - v).math_abs()) / two
            });
            let log_sum_exp = {
                let s: T = (0..n)
                    .map(|j| (self.get(r, j) - row_max).math_exp())
                    .fold(T::zero(), |a, b| a + b);
                row_max + s.math_ln()
            };
            self.get(r, c) - log_sum_exp
        })
    }

    // ---- Normalization ----

    /// Layer normalization along `axis`: `(x - mean) / (std + eps)`.
    #[must_use]
    pub fn layer_norm(&self, axis: usize, eps: T) -> Self {
        let mean = self.mean_axis(axis);
        let std = self.std_axis(axis);
        let (sr, sc) = std.shape();
        let inv_std = Self::from_fn(sr, sc, |r, c| T::one() / (std.get(r, c) + eps));
        let neg_mean = -&mean;
        if Self::axis_is_rows(axis, "layer_norm") {
            let centered = self.broadcast_add_cols(&neg_mean);
            centered.broadcast_mul_cols(&inv_std)
        } else {
            let centered = self.broadcast_add_rows(&neg_mean);
            centered.broadcast_mul_rows(&inv_std)
        }
    }

    /// RMS normalization along `axis`: `x / rms(x) * weight`.
    #[must_use]
    pub fn rms_norm(&self, axis: usize, weight: &Self, eps: T) -> Self {
        let (m, n) = self.shape();
        if Self::axis_is_rows(axis, "rms_norm") {
            let rms = Self::from_fn(m, 1, |r, _| {
                let sq_sum = (0..n).fold(T::zero(), |acc, c| {
                    let v = self.get(r, c);
                    acc + v * v
                });
                (sq_sum / T::from_f64(n as f64) + eps).math_sqrt()
            });
            let normed =
                self.broadcast_mul_cols(&Self::from_fn(m, 1, |r, c| T::one() / rms.get(r, c)));
            normed.broadcast_mul_rows(weight)
        } else {
            let rms = Self::from_fn(1, n, |_, c| {
                let sq_sum = (0..m).fold(T::zero(), |acc, r| {
                    let v = self.get(r, c);
                    acc + v * v
                });
                (sq_sum / T::from_f64(m as f64) + eps).math_sqrt()
            });
            let normed =
                self.broadcast_mul_rows(&Self::from_fn(1, n, |r, c| T::one() / rms.get(r, c)));
            normed.broadcast_mul_cols(weight)
        }
    }

    /// Batch normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.
    #[must_use]
    pub fn batch_norm(
        &self,
        running_mean: &Self,
        running_var: &Self,
        weight: &Self,
        bias: &Self,
        eps: T,
    ) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            let mu = running_mean.get(0, c);
            let var = running_var.get(0, c);
            let w = weight.get(0, c);
            let b = bias.get(0, c);
            (x - mu) / (var + eps).math_sqrt() * w + b
        })
    }

    /// Batch normalization with running statistics update (training mode).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_train(
        &self,
        gamma: &Self,
        beta: &Self,
        running_mean: &mut Self,
        running_var: &mut Self,
        eps: T,
        momentum: T,
        training: bool,
    ) -> Self {
        Self {
            storage: B::batch_norm_train(
                &self.storage,
                &gamma.storage,
                &beta.storage,
                &mut running_mean.storage,
                &mut running_var.storage,
                eps,
                momentum,
                training,
            ),
            _axes: PhantomData,
        }
    }

    /// Group normalization: divide channels into groups, normalize each group.
    #[must_use]
    pub fn group_norm(&self, num_groups: usize, weight: &Self, bias: &Self, eps: T) -> Self {
        let (m, n) = self.shape();
        assert!(
            n % num_groups == 0,
            "nabla: group_norm C={n} not divisible by groups={num_groups}"
        );
        let g_size = n / num_groups;
        Self::from_fn(m, n, |r, c| {
            let g = c / g_size;
            let g_start = g * g_size;
            let mean = (0..g_size).fold(T::zero(), |acc, j| acc + self.get(r, g_start + j))
                / T::from_f64(g_size as f64);
            let var = (0..g_size).fold(T::zero(), |acc, j| {
                let d = self.get(r, g_start + j) - mean;
                acc + d * d
            }) / T::from_f64(g_size as f64);
            let x = self.get(r, c);
            (x - mean) / (var + eps).math_sqrt() * weight.get(0, c) + bias.get(0, c)
        })
    }
}

// ---- Dropout (cpu-gated) ----

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Dropout: randomly zeroes elements with probability `p` during training.
    #[must_use]
    pub fn dropout(&self, p: f64, training: bool, seed: u64) -> Self {
        if !training || p <= 0.0 {
            return self.clone();
        }
        if p >= 1.0 {
            let (m, n) = self.shape();
            return Self::zeros(m, n);
        }
        let scale = T::from_f64(1.0 / (1.0 - p));
        let threshold = (p * (u64::MAX as f64)) as u64;
        let (m, n) = self.shape();
        let mut s = if seed == 0 { 0xDEAD_BEEF_CAFE_1234_u64 } else { seed };
        let mut data = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let x = self.get(r, c);
                data.push(if s < threshold { T::zero() } else { x * scale });
            }
        }
        Self::from_storage(B::from_vec(m, n, data))
    }

    /// Dropout with automatic seed from system time.
    ///
    /// Convenience wrapper around [`dropout`](Self::dropout) that derives a seed
    /// from `SystemTime::now()` so callers need not supply one.
    #[must_use]
    pub fn dropout_auto(&self, p: f64, training: bool) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0xCAFE_BABE_1337_u64, |d| {
                let nanos = d.as_nanos() as u64;
                // Mix bits so sequential calls differ
                nanos ^ (nanos >> 17) ^ (nanos.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            });
        self.dropout(p, training, seed)
    }
}
