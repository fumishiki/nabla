// tensor/nn/losses.rs — Loss functions.

use core::marker::PhantomData;

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::{two, Tensor};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Loss functions ----

    /// Cross-entropy loss from log-softmax predictions and target probabilities.
    #[must_use]
    pub fn cross_entropy_loss(&self, targets: &Self) -> T {
        let (batch, n) = self.shape();
        assert_eq!(
            targets.shape(), (batch, n),
            "nabla: cross_entropy_loss shape mismatch -- self {}x{} vs targets {}x{}",
            batch, n, targets.nrows(), targets.ncols()
        );
        let sum: T = (0..batch)
            .map(|r| {
                (0..n)
                    .map(|c| targets.get(r, c) * self.get(r, c))
                    .fold(T::zero(), |a, b| a + b)
            })
            .fold(T::zero(), |a, b| a + b);
        -(sum / T::from_f64(batch as f64))
    }

    /// Cross-entropy loss: fused softmax + NLL.
    #[must_use]
    pub fn cross_entropy_fused(&self, target: &Self) -> Self {
        let (n, c) = self.shape();
        assert_eq!(
            target.nrows(), n,
            "nabla: cross_entropy_fused shape mismatch -- input {}x{} vs target {}x{}",
            n, c, target.nrows(), target.ncols()
        );
        Self {
            storage: B::cross_entropy_fused(&self.storage, &target.storage, n, c),
            _axes: PhantomData,
        }
    }

    /// MSE loss: `mean((pred - target)^2)`.
    #[must_use]
    pub fn mse_loss(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let d = self.get(r, c) - target.get(r, c);
                acc2 + d * d
            })
        });
        sum / total
    }

    /// L1 loss: `mean(|pred - target|)`.
    #[must_use]
    pub fn l1_loss(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                acc2 + (self.get(r, c) - target.get(r, c)).math_abs()
            })
        });
        sum / total
    }

    /// Smooth L1 (Huber) loss with transition point `beta`.
    #[must_use]
    pub fn smooth_l1_loss(&self, target: &Self, beta: T) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let half = T::from_f64(0.5);
        let two = two::<T>();
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let d = (self.get(r, c) - target.get(r, c)).math_abs();
                let pick = (d + beta - (d - beta).math_abs()) / two;
                acc2 + half * pick * pick / beta + (d - pick)
            })
        });
        sum / total
    }

    /// Binary cross-entropy with logits: `-[y * log(sigma(x)) + (1-y) * log(1-sigma(x))]`.
    #[must_use]
    pub fn bce_with_logits(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let x = self.get(r, c);
                let y = target.get(r, c);
                let abs_x = x.math_abs();
                let relu_x = (x + abs_x) / two::<T>();
                acc2 + relu_x - x * y + (T::one() + (T::zero() - abs_x).math_exp()).math_ln()
            })
        });
        sum / total
    }

    /// Negative log-likelihood loss.
    #[must_use]
    pub fn nll_loss(&self, targets: &Self) -> T {
        let m = self.nrows();
        let sum = (0..m).fold(T::zero(), |acc, r| {
            let cls = targets.get(r, 0).to_f64() as usize;
            acc - self.get(r, cls)
        });
        sum / T::from_f64(m as f64)
    }

    /// KL divergence: `sum(q * (log(q) - log_p))` (batchmean reduction).
    #[must_use]
    pub fn kl_div(&self, q: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64(m as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let qv = q.get(r, c);
                let log_p = self.get(r, c);
                acc2 + qv * (qv.math_ln() - log_p)
            })
        });
        sum / total
    }

    /// Cosine embedding loss for pairs `(x1, x2)` with label `y` in {1, -1}.
    #[must_use]
    pub fn cosine_embedding_loss(x1: &Self, x2: &Self, y: T, margin: T) -> T {
        let (m, n) = x1.shape();
        let dot = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |a, c| a + x1.get(r, c) * x2.get(r, c))
        });
        let n1 = x1.norm();
        let n2 = x2.norm();
        let eps = T::from_f64(1e-8);
        let cos_sim = dot / (n1 * n2 + eps);
        let two = two::<T>();
        if y.to_f64() > 0.0 {
            T::one() - cos_sim
        } else {
            let v = cos_sim - margin;
            (v + v.math_abs()) / two
        }
    }
}
