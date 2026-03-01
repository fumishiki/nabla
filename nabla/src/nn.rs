//! Neural network utility functions.
//!
//! Initializers, positional encodings, KV-cache helpers, and linear solvers
//! for building neural network models with nabla tensors.

use crate::{scalar, tensor};
use crate::constructors::{rand, randn};

/// Xavier/Glorot uniform initializer for weight tensors of shape `(fan_out, fan_in)`.
///
/// Samples from a uniform distribution `[-limit, limit]` where
/// `limit = sqrt(6 / (fan_in + fan_out))`.
#[must_use]
pub fn xavier_uniform<T: scalar::Scalar>(fan_in: usize, fan_out: usize) -> tensor::Tensor<T> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let two_limit = T::from_f64(2.0 * limit);
    let offset = T::from_f64(limit);
    // rand generates [0, 1); scale to [-limit, limit]
    let r = rand::<T>(fan_out, fan_in);
    tensor::Tensor::from_fn(fan_out, fan_in, |i, j| r.get(i, j) * two_limit - offset)
}

/// He/Kaiming normal initializer for weight tensors of shape `(fan_out, fan_in)`.
///
/// Samples from N(0, sqrt(2 / fan_in)), suited for ReLU activations.
#[must_use]
pub fn kaiming_normal<T: scalar::Scalar>(fan_out: usize, fan_in: usize) -> tensor::Tensor<T> {
    let std = (2.0 / fan_in as f64).sqrt();
    let r = randn::<T>(fan_out, fan_in);
    &r * T::from_f64(std)
}

/// Apply Rotary Positional Embedding (RoPE) to a tensor.
///
/// `x`: `(seq_len, head_dim)`, `seq_offset`: starting position index,
/// `theta`: base frequency (typically `10000.0`).
#[must_use]
pub fn rotary_embedding<T: scalar::Scalar>(
    x: &tensor::Tensor<T>,
    head_dim: usize,
    seq_offset: usize,
    theta: f64,
) -> tensor::Tensor<T> {
    let (seq_len, d) = x.shape();
    assert_eq!(d, head_dim, "nabla: rotary_embedding head_dim mismatch");
    let half = head_dim / 2;
    tensor::Tensor::from_fn(seq_len, head_dim, |s, i| {
        let pos = (s + seq_offset) as f64;
        let (left, right) = if i < half { (i, i + half) } else { (i - half, i) };
        let freq = 1.0 / theta.powf(2.0 * (left as f64) / head_dim as f64);
        let cos_val = T::from_f64((pos * freq).cos());
        let sin_val = T::from_f64((pos * freq).sin());
        let x_left = x.get(s, left);
        let x_right = x.get(s, right);
        let (cos_mul, sin_mul) = (x_left * cos_val, x_left * sin_val);
        if i < half {
            cos_mul - x_right * sin_val
        } else {
            sin_mul + x_right * cos_val
        }
    })
}

/// Append new tokens to a KV cache, returning the updated full cache.
///
/// `cache`: `(current_seq_len, dim)`, `new_kv`: `(new_tokens, dim)`.
/// Returns a new tensor of shape `(current_seq_len + new_tokens, dim)`.
#[must_use]
pub fn kv_cache_append<T: scalar::Scalar>(
    cache: &tensor::Tensor<T>,
    new_kv: &tensor::Tensor<T>,
) -> tensor::Tensor<T> {
    tensor::Tensor::vcat(&[cache, new_kv])
}

/// Embedding lookup: select rows from weight matrix by indices.
///
/// Free-function wrapper around [`tensor::Tensor::embedding`] so callers
/// can write `embedding(&idx, &w)` instead of `Tensor::embedding(&idx, &w)`.
#[must_use]
pub fn embedding<T: scalar::Scalar, B: crate::backend::Backend>(
    indices: &tensor::Tensor<T, B>,
    weight: &tensor::Tensor<T, B>,
) -> tensor::Tensor<T, B> {
    tensor::Tensor::embedding(indices, weight)
}

/// Linear layer: `x @ weight^T + bias`.
///
/// `x`: `(batch, in_features)`, `weight`: `(out_features, in_features)`.
/// Optional `bias`: `(1, out_features)` or `(out_features,)` broadcast-added.
#[must_use]
pub fn linear<T: scalar::Scalar>(
    x: &tensor::Tensor<T>,
    weight: &tensor::Tensor<T>,
    bias: Option<&tensor::Tensor<T>>,
) -> tensor::Tensor<T> {
    // x @ weight^T
    let wt = weight.t();
    let out = x * &wt;
    match bias {
        Some(b) => &out + b,
        None => out,
    }
}

/// Solve `A·x = b`, automatically selecting the best method.
///
/// Square matrices use LU; overdetermined systems use QR least-squares.
#[cfg(feature = "cpu")]
pub fn backslash(
    a: &tensor::Tensor<f64, nabla_core::backend::Cpu>,
    b: &tensor::Tensor<f64, nabla_core::backend::Cpu>,
) -> nabla_core::error::Result<tensor::Tensor<f64, nabla_core::backend::Cpu>> {
    use crate::linalg::LinalgExt as _;
    let (m, n) = a.shape();
    if m == n {
        a.solve(b)
    } else {
        a.lstsq(b)
    }
}
