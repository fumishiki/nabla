use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::{Tensor, two};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- ML activation functions ----

    /// Element-wise ReLU: `max(x, 0)`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let two = two::<T>();
        let abs = self.abs();
        (self + &abs) / two
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        Self::from_storage(B::sigmoid(&self.storage))
    }

    /// Element-wise GELU (tanh approximation):
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    #[must_use]
    pub fn gelu(&self) -> Self {
        let half = T::from_f64(0.5);
        let k = T::from_f64(0.797_884_560_8); // sqrt(2/pi)
        let c = T::from_f64(0.044_715);
        let (m, n) = self.shape();
        let ones = Tensor::<T, B>::fill(1, 1, T::one()).expand(m, n);
        let x3 = self.emul(self).emul(self);
        let x3c = &x3 * c;
        let inner_sum = self + &x3c;
        let inner = &inner_sum * k;
        let tanh_inner = inner.tanh();
        let out = self.emul(&(&tanh_inner + &ones));
        &out * half
    }

    /// Element-wise SiLU (Swish): `x * sigmoid(x)`.
    #[must_use]
    pub fn silu(&self) -> Self {
        Self::from_storage(B::silu(&self.storage))
    }

    /// Element-wise Mish: `x * tanh(softplus(x))` where softplus(x) = ln(1 + exp(x)).
    #[must_use]
    pub fn mish(&self) -> Self {
        Self::from_storage(B::mish(&self.storage))
    }

    /// Element-wise Leaky ReLU: `max(alpha * x, x)`.
    #[must_use]
    pub fn leaky_relu(&self, alpha: T) -> Self {
        Self::from_storage(B::leaky_relu(&self.storage, alpha))
    }

    /// Element-wise ELU: `x if x > 0, alpha * (exp(x) - 1) otherwise`.
    #[must_use]
    pub fn elu(&self, alpha: T) -> Self {
        Self::from_storage(B::elu(&self.storage, alpha))
    }

    /// Element-wise HardSwish: `x * relu6(x + 3) / 6`.
    #[must_use]
    pub fn hardswish(&self) -> Self {
        Self::from_storage(B::hardswish(&self.storage))
    }

    // ---- Backward activations (GPU-native, no D2H) ----

    /// ReLU backward: `self * (input > 0 ? 1 : 0)` where self = grad.
    #[must_use]
    pub fn relu_backward(&self, input: &Self) -> Self {
        Self::from_storage(B::relu_backward(&self.storage, &input.storage))
    }
    /// Leaky ReLU backward: `self * (input > 0 ? 1 : alpha)` where self = grad.
    #[must_use]
    pub fn leaky_relu_backward(&self, input: &Self, alpha: T) -> Self {
        Self::from_storage(B::leaky_relu_backward(&self.storage, &input.storage, alpha))
    }
    /// ELU backward: `self * (input > 0 ? 1 : alpha * exp(input))` where self = grad.
    #[must_use]
    pub fn elu_backward(&self, input: &Self, alpha: T) -> Self {
        Self::from_storage(B::elu_backward(&self.storage, &input.storage, alpha))
    }
    /// GELU backward: `self * (cdf + x * pdf)` where self = grad.
    #[must_use]
    pub fn gelu_backward(&self, input: &Self) -> Self {
        Self::from_storage(B::gelu_backward(&self.storage, &input.storage))
    }
    /// Abs backward: `self * sign(input)` where self = grad.
    #[must_use]
    pub fn abs_backward(&self, input: &Self) -> Self {
        Self::from_storage(B::abs_backward(&self.storage, &input.storage))
    }
}

use core::marker::PhantomData;

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
        self.softmax(1).ln()
    }

    // ---- Normalization ----

    /// Layer normalization along `axis`: `(x - mean) / (std + eps)`.
    #[must_use]
    pub fn layer_norm(&self, axis: usize, eps: T) -> Self {
        if !Self::axis_is_rows(axis, "layer_norm") {
            return self.t().layer_norm(1, eps).t();
        }
        let (_, n) = self.shape();
        let gamma = Tensor::<T, B>::fill(1, n, T::one_impl());
        let beta = Tensor::<T, B>::fill(1, n, T::zero());
        Self::from_storage(B::layer_norm(
            &self.storage,
            &gamma.storage,
            &beta.storage,
            eps,
        ))
    }

    /// RMS normalization along `axis`: `x / rms(x) * weight`.
    #[must_use]
    pub fn rms_norm(&self, axis: usize, weight: &Self, eps: T) -> Self {
        if !Self::axis_is_rows(axis, "rms_norm") {
            return self.t().rms_norm(1, &weight.t(), eps).t();
        }
        Self::from_storage(B::rms_norm(&self.storage, &weight.storage, eps))
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
        let mut rm = running_mean.clone();
        let mut rv = running_var.clone();
        Self::from_storage(B::batch_norm_train(
            &self.storage,
            &weight.storage,
            &bias.storage,
            &mut rm.storage,
            &mut rv.storage,
            eps,
            T::from_f64(0.0),
            false,
        ))
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
        let (_, n) = self.shape();
        assert!(
            n % num_groups == 0,
            "nabla: group_norm C={n} not divisible by groups={num_groups}"
        );
        Self {
            storage: B::group_norm(
                &self.storage,
                &weight.storage,
                &bias.storage,
                num_groups,
                eps,
            ),
            _axes: PhantomData,
        }
    }
}

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
        let mut s = if seed == 0 {
            0xDEAD_BEEF_CAFE_1234_u64
        } else {
            seed
        };
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

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Loss functions ----

    /// Cross-entropy loss from log-softmax predictions and target probabilities.
    #[must_use]
    pub fn cross_entropy_loss(&self, targets: &Self) -> T {
        let (batch, n) = self.shape();
        assert_eq!(
            targets.shape(),
            (batch, n),
            "nabla: cross_entropy_loss shape mismatch -- self {}x{} vs targets {}x{}",
            batch,
            n,
            targets.nrows(),
            targets.ncols()
        );
        let sum = targets.emul(self).sum_all();
        -(sum / T::from_f64(batch as f64))
    }

    /// Cross-entropy loss: fused softmax + NLL.
    #[must_use]
    pub fn cross_entropy_fused(&self, target: &Self) -> Self {
        let (n, c) = self.shape();
        assert_eq!(
            target.nrows(),
            n,
            "nabla: cross_entropy_fused shape mismatch -- input {}x{} vs target {}x{}",
            n,
            c,
            target.nrows(),
            target.ncols()
        );
        Self {
            storage: B::cross_entropy_fused(&self.storage, &target.storage, n, c),
            _axes: PhantomData,
        }
    }

    /// Fused MSE sum loss: `sum((self - target)^2)` -> (1,1) tensor.
    #[must_use]
    pub fn mse_sum_loss_tensor(&self, target: &Self) -> Self {
        Self::from_storage(B::mse_sum_fwd(&self.storage, &target.storage))
    }

    /// MSE loss: `mean((pred - target)^2)`.
    #[must_use]
    pub fn mse_loss(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let diff = self - target;
        diff.emul(&diff).sum_all() / total
    }

    /// L1 loss: `mean(|pred - target|)`.
    #[must_use]
    pub fn l1_loss(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let diff = self - target;
        diff.abs().sum_all() / total
    }

    /// Smooth L1 (Huber) loss with transition point `beta`.
    #[must_use]
    pub fn smooth_l1_loss(&self, target: &Self, beta: T) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let half = T::from_f64(0.5);
        let two = two::<T>();
        let diff = self - target;
        let abs = diff.abs();
        let beta_t = Tensor::fill(1, 1, beta).expand(m, n);
        let pick = (&abs + &beta_t - (&abs - &beta_t).abs()) / two;
        let quad = &pick.emul(&pick) * (half / beta);
        let lin = &abs - &pick;
        (quad + lin).sum_all() / total
    }

    /// Binary cross-entropy with logits: `-[y * log(sigma(x)) + (1-y) * log(1-sigma(x))]`.
    #[must_use]
    pub fn bce_with_logits(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let abs_x = self.abs();
        let one = Tensor::fill(1, 1, T::one()).expand(m, n);
        let relu_x = (self + &abs_x) / two::<T>();
        let loss_sum = (relu_x - self.emul(target) + (&one + (-&abs_x).exp()).ln()).sum_all();
        loss_sum / total
    }

    /// Negative log-likelihood loss.
    #[must_use]
    pub fn nll_loss(&self, targets: &Self) -> T {
        let m = self.nrows();
        let one_hot = Tensor::from_storage(B::one_hot_from_indices(&targets.storage, self.ncols()));
        let picked = self.emul(&one_hot).sum_axis(1);
        (-&picked).sum_all() / T::from_f64(m as f64)
    }

    /// KL divergence: `sum(q * (log(q) - log_p))` (batchmean reduction).
    #[must_use]
    pub fn kl_div(&self, q: &Self) -> T {
        let (m, _n) = self.shape();
        let total = T::from_f64(m as f64);
        let sum = q.emul(&(q.ln() - self)).sum_all();
        sum / total
    }

    /// Cosine embedding loss for pairs `(x1, x2)` with label `y` in {1, -1}.
    #[must_use]
    pub fn cosine_embedding_loss(x1: &Self, x2: &Self, y: T, margin: T) -> T {
        let (_m, _n) = x1.shape();
        let dot = x1.emul(x2).sum_all();
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

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Attention / Transformer ----

    /// Embedding lookup: select rows from weight matrix by indices.
    #[must_use]
    pub fn embedding(indices: &Self, weight: &Self) -> Self {
        Self::from_storage(B::embedding(&indices.storage, &weight.storage))
    }

    /// Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`.
    #[must_use]
    pub fn scaled_dot_product_attention(q: &Self, k: &Self, v: &Self, mask: Option<&Self>) -> Self {
        let d_k = q.ncols();
        let scale = T::from_f64(1.0 / (d_k as f64).sqrt());
        let kt = k.t();
        let mut scores = &(&q.clone() * &kt) * scale;
        if let Some(m) = mask {
            let neg_inf = T::from_f64(f64::NEG_INFINITY);
            scores = scores.masked_fill(m, neg_inf);
        }
        let attn = scores.softmax(1);
        &attn * v
    }

    /// Multi-head attention.
    #[must_use]
    pub fn multi_head_attention(
        q: &Self,
        k: &Self,
        v: &Self,
        num_heads: usize,
        mask: Option<&Self>,
    ) -> Self {
        let d_model = q.ncols();
        assert!(
            d_model.is_multiple_of(num_heads),
            "nabla: d_model must be divisible by num_heads"
        );
        let d_head = d_model / num_heads;
        let seq_q = q.nrows();
        let seq_k = k.nrows();
        let head_outputs: Vec<Self> = (0..num_heads)
            .map(|h| {
                let q_h = q.submatrix(0, seq_q, h * d_head, (h + 1) * d_head);
                let k_h = k.submatrix(0, seq_k, h * d_head, (h + 1) * d_head);
                let v_h = v.submatrix(0, seq_k, h * d_head, (h + 1) * d_head);
                Self::scaled_dot_product_attention(&q_h, &k_h, &v_h, mask)
            })
            .collect();
        let refs: Vec<&Self> = head_outputs.iter().collect();
        Self::hcat(&refs)
    }

    /// Scaled dot-product attention with FlashAttention-2 on GPU backends.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        q: &Self,
        k: &Self,
        v: &Self,
        mask: Option<&Self>,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        batch_heads: usize,
    ) -> Self {
        assert!(
            head_dim <= 128,
            "nabla: sdpa head_dim must be <= 128 (FA_HEAD_DIM_MAX), got {head_dim}"
        );
        assert_eq!(
            q.nrows(),
            batch_heads * seq_q,
            "nabla: sdpa Q nrows must equal batch_heads*seq_q"
        );
        assert_eq!(
            q.ncols(),
            head_dim,
            "nabla: sdpa Q ncols must equal head_dim"
        );
        Self::from_storage(B::sdpa(
            &q.storage,
            &k.storage,
            &v.storage,
            mask.map(|m| &m.storage),
            seq_q,
            seq_k,
            head_dim,
            batch_heads,
        ))
    }

    // ---- Walsh-Hadamard Transform ----

    /// Walsh-Hadamard Transform on each row. Columns must be power-of-2.
    #[must_use]
    pub fn wht(&self) -> Self {
        assert!(
            self.ncols().is_power_of_two(),
            "WHT requires power-of-2 columns, got {}",
            self.ncols()
        );
        Self::from_storage(B::wht(&self.storage))
    }

    /// Inverse Walsh-Hadamard Transform (WHT / N) on each row. Columns must be power-of-2.
    #[must_use]
    pub fn wht_inverse(&self) -> Self {
        assert!(
            self.ncols().is_power_of_two(),
            "WHT requires power-of-2 columns, got {}",
            self.ncols()
        );
        Self::from_storage(B::wht_inverse(&self.storage))
    }

    // ---- Batched operations ----

    /// Batched matrix multiply.
    #[must_use]
    pub fn bmm(&self, other: &Self, batch: usize, m: usize, k: usize, n: usize) -> Self {
        assert_eq!(self.nrows(), batch * m);
        assert_eq!(self.ncols(), k);
        assert_eq!(other.nrows(), batch * k);
        assert_eq!(other.ncols(), n);
        Self::from_storage(B::bmm(&self.storage, &other.storage, batch, m, k, n))
    }

    /// Batched matrix multiply with B transposed: `C[b] = A[b] @ B[b]^T`.
    #[must_use]
    pub fn bmm_nt(&self, other: &Self, batch: usize, m: usize, k: usize, n: usize) -> Self {
        assert_eq!(self.nrows(), batch * m);
        assert_eq!(self.ncols(), k);
        assert_eq!(other.nrows(), batch * n);
        assert_eq!(other.ncols(), k);
        Self::from_storage(B::bmm_nt(&self.storage, &other.storage, batch, m, k, n))
    }

    /// `C = alpha * A @ B + beta * C` (addmm).
    #[must_use]
    pub fn addmm(&self, a: &Self, b: &Self, beta: T, alpha: T) -> Self {
        let (m, n) = self.shape();
        let k = a.ncols();
        assert_eq!(a.nrows(), m);
        assert_eq!(b.shape(), (k, n));
        Self::from_storage(B::addmm(&self.storage, &a.storage, &b.storage, beta, alpha))
    }

    /// Batched addmm.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn baddbmm(
        &self,
        a: &Self,
        b: &Self,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        beta: T,
        alpha: T,
    ) -> Self {
        assert_eq!(self.nrows(), batch * m);
        assert_eq!(self.ncols(), n);
        Self::from_storage(B::baddbmm(
            &self.storage,
            &a.storage,
            &b.storage,
            batch,
            m,
            k,
            n,
            beta,
            alpha,
        ))
    }
}

// ---- In-place CUDA ops ----

#[cfg(feature = "cuda")]
impl<T: Scalar> Tensor<T, crate::backend::Cuda> {
    /// In-place sigmoid: `self[i] = 1 / (1 + exp(-self[i]))`. No allocation.
    #[inline]
    pub fn sigmoid_inplace(&mut self) {
        crate::cuda_backend::launch_unary_inplace(&mut self.storage, "sigmoid");
    }

    /// In-place ReLU: `self[i] = max(0, self[i])`.
    /// Uses `relu(x) = (x + |x|) * 0.5` with 1 temp allocation for abs.
    pub fn relu_inplace(&mut self) {
        let abs_storage = crate::cuda_backend::launch_unary(&self.storage, "abs");
        <crate::backend::Cuda as crate::backend::BackendCore>::axpy_inplace(
            &mut self.storage,
            T::one_impl(),
            &abs_storage,
        );
        crate::cuda_backend::cuda_scale_inplace(&mut self.storage, T::from_f64(0.5));
    }
}
