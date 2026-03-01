// tensor/nn/attention.rs — Attention, embedding, and batched linear algebra.

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

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
        q: &Self, k: &Self, v: &Self, num_heads: usize, mask: Option<&Self>,
    ) -> Self {
        let d_model = q.ncols();
        assert!(d_model.is_multiple_of(num_heads), "nabla: d_model must be divisible by num_heads");
        let d_head = d_model / num_heads;
        let seq_q = q.nrows();
        let seq_k = k.nrows();
        let mut head_outputs: Vec<Self> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let q_h = q.submatrix(0, seq_q, h * d_head, (h + 1) * d_head);
            let k_h = k.submatrix(0, seq_k, h * d_head, (h + 1) * d_head);
            let v_h = v.submatrix(0, seq_k, h * d_head, (h + 1) * d_head);
            head_outputs.push(Self::scaled_dot_product_attention(&q_h, &k_h, &v_h, mask));
        }
        let refs: Vec<&Self> = head_outputs.iter().collect();
        Self::hcat(&refs)
    }

    /// Scaled dot-product attention with FlashAttention-2 on GPU backends.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        q: &Self, k: &Self, v: &Self, mask: Option<&Self>,
        seq_q: usize, seq_k: usize, head_dim: usize, batch_heads: usize,
    ) -> Self {
        assert!(head_dim <= 128, "nabla: sdpa head_dim must be <= 128 (FA_HEAD_DIM_MAX), got {head_dim}");
        assert_eq!(q.nrows(), batch_heads * seq_q, "nabla: sdpa Q nrows must equal batch_heads*seq_q");
        assert_eq!(q.ncols(), head_dim, "nabla: sdpa Q ncols must equal head_dim");
        Self::from_storage(B::sdpa(
            &q.storage, &k.storage, &v.storage,
            mask.map(|m| &m.storage),
            seq_q, seq_k, head_dim, batch_heads,
        ))
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
        &self, a: &Self, b: &Self, batch: usize, m: usize, k: usize, n: usize,
        beta: T, alpha: T,
    ) -> Self {
        assert_eq!(self.nrows(), batch * m);
        assert_eq!(self.ncols(), n);
        Self::from_storage(B::baddbmm(
            &self.storage, &a.storage, &b.storage, batch, m, k, n, beta, alpha,
        ))
    }
}
