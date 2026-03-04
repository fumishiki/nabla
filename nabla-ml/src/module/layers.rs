use std::rc::Rc;

use nabla_core::backend::{Backend, DefaultBackend};
#[cfg(not(feature = "cpu"))]
use nabla_core::error::Error;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use crate::autograd::{Tape, TensorLike, Variable};
#[cfg(feature = "cpu")]
use crate::constructors::{rand, randn};
use crate::{scalar, tensor};

use super::{ForwardResult, Module};

/// Xavier/Glorot uniform weight initialization (CPU-only).
#[must_use]
#[cfg(feature = "cpu")]
pub fn xavier_uniform<T: scalar::Scalar>(fan_in: usize, fan_out: usize) -> tensor::Tensor<T> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let two_limit = T::from_f64(2.0 * limit);
    let offset = T::from_f64(limit);
    let r = rand::<T>(fan_out, fan_in);
    tensor::Tensor::from_fn(fan_out, fan_in, |i, j| r.get(i, j) * two_limit - offset)
}

/// Kaiming/He normal weight initialization for ReLU networks (CPU-only).
#[must_use]
#[cfg(feature = "cpu")]
pub fn kaiming_normal<T: scalar::Scalar>(fan_in: usize, fan_out: usize) -> tensor::Tensor<T> {
    let std = (2.0 / fan_in as f64).sqrt();
    let r = randn::<T>(fan_out, fan_in);
    &r * T::from_f64(std)
}

/// Apply rotary positional embedding (RoPE) to input tensor (CPU-only).
#[must_use]
#[cfg(feature = "cpu")]
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
        let (left, right) = if i < half {
            (i, i + half)
        } else {
            (i - half, i)
        };
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

/// Append new key-value rows to an existing KV cache.
#[must_use]
pub fn kv_cache_append<T: scalar::Scalar>(
    cache: &tensor::Tensor<T>,
    new_kv: &tensor::Tensor<T>,
) -> tensor::Tensor<T> {
    tensor::Tensor::vcat(&[cache, new_kv])
}

/// Look up rows from an embedding weight matrix by index.
#[must_use]
pub fn embedding<T: scalar::Scalar, B: crate::backend::Backend>(
    indices: &tensor::Tensor<T, B>,
    weight: &tensor::Tensor<T, B>,
) -> tensor::Tensor<T, B> {
    tensor::Tensor::embedding(indices, weight)
}

/// Functional linear transform: `x @ weight^T + bias`.
#[must_use]
pub fn linear<T: scalar::Scalar>(
    x: &tensor::Tensor<T>,
    weight: &tensor::Tensor<T>,
    bias: Option<&tensor::Tensor<T>>,
) -> tensor::Tensor<T> {
    let wt = weight.t();
    let out = x * &wt;
    match bias {
        Some(b) => &out + b,
        None => out,
    }
}

/// Solve `A \ b` via LU (square) or least-squares (rectangular).
#[cfg(feature = "cpu")]
pub fn backslash(
    a: &tensor::Tensor<f64, nabla_core::backend::Cpu>,
    b: &tensor::Tensor<f64, nabla_core::backend::Cpu>,
) -> nabla_core::error::Result<tensor::Tensor<f64, nabla_core::backend::Cpu>> {
    use crate::linalg::LinalgExt as _;
    let (m, n) = a.shape();
    if m == n { a.solve(b) } else { a.lstsq(b) }
}

/// Enumeration of supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationKind {
    /// Rectified linear unit.
    Relu,
    /// Gaussian error linear unit.
    Gelu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Sigmoid linear unit (Swish).
    Silu,
    /// Hyperbolic tangent.
    Tanh,
    /// Leaky ReLU with negative slope `alpha`.
    LeakyRelu(f64),
    /// Exponential linear unit with scale `alpha`.
    Elu(f64),
}

/// Activation function module wrapping an [`ActivationKind`].
pub struct Activation<T: Scalar, B: Backend> {
    kind: ActivationKind,
    training: bool,
    _marker: std::marker::PhantomData<fn(&Tensor<T, B>) -> Tensor<T, B>>,
}

macro_rules! activation_ctors {
    ($($name:ident => $kind:expr),* $(,)?) => {
        $(
            /// Create an activation module of this kind.
            #[must_use]
            pub fn $name() -> Self {
                Self { kind: $kind, training: true, _marker: std::marker::PhantomData }
            }
        )*
    };
}

impl<T: Scalar, B: Backend> Activation<T, B> {
    /// Return the activation function kind.
    #[must_use]
    pub fn kind(&self) -> ActivationKind {
        self.kind
    }

    activation_ctors!(
        relu => ActivationKind::Relu, gelu => ActivationKind::Gelu,
        sigmoid => ActivationKind::Sigmoid, silu => ActivationKind::Silu,
        tanh => ActivationKind::Tanh,
        leaky_relu => ActivationKind::LeakyRelu(0.01), elu => ActivationKind::Elu(1.0),
    );
}

fn apply_activation<T: Scalar, B: Backend, X: TensorLike<T, B>>(kind: ActivationKind, x: &X) -> X {
    match kind {
        ActivationKind::Relu => x.tl_relu(),
        ActivationKind::Gelu => x.tl_gelu(),
        ActivationKind::Sigmoid => x.tl_sigmoid(),
        ActivationKind::Silu => x.tl_silu(),
        ActivationKind::Tanh => x.tl_tanh(),
        ActivationKind::LeakyRelu(a) => x.tl_leaky_relu(a),
        ActivationKind::Elu(a) => x.tl_elu(a),
    }
}

impl<T: Scalar, B: Backend> Module<T, B> for Activation<T, B> {
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B> {
        apply_activation(self.kind, x)
    }

    fn forward_var(&self, x: &Variable<T, B>, _tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
        Ok(apply_activation(self.kind, x))
    }

    impl_module_params!();
}

/// Dropout regularization layer with inverted scaling.
pub struct DropoutLayer<T: Scalar, B: Backend> {
    p: f64,
    training: bool,
    _marker: std::marker::PhantomData<(T, B)>,
}

impl<T: Scalar, B: Backend> DropoutLayer<T, B> {
    /// Create a new dropout layer with drop probability `p` in `[0, 1)`.
    #[must_use]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "nabla: dropout probability must be in [0, 1), got {p}"
        );
        Self {
            p,
            training: true,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Module<T, B> for DropoutLayer<T, B> {
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B> {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }
        let scale = T::from_f64(1.0 / (1.0 - self.p));
        let (nrows, ncols) = x.shape();
        let mask_rand = crate::constructors::rand::<T>(nrows, ncols);
        let mask = Tensor::from_fn(nrows, ncols, |r, c| {
            if mask_rand.get(r, c).to_f64() > self.p {
                scale
            } else {
                T::zero()
            }
        });
        x.emul(&mask)
    }

    fn forward_var(&self, x: &Variable<T, B>, _tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
        Ok(x.dropout(self.p, self.training))
    }

    impl_module_params!();
}

/// Embedding lookup table module.
pub struct EmbeddingLayer<T: Scalar, B: Backend> {
    /// Embedding weight matrix `(num_embeddings, embedding_dim)`.
    pub weight: Tensor<T, B>,
    training: bool,
}

#[cfg(feature = "cpu")]
impl<T: Scalar> EmbeddingLayer<T, DefaultBackend> {
    /// Create a new embedding layer with `randn` initialization.
    #[must_use]
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = crate::constructors::randn::<T>(num_embeddings, embedding_dim);
        Self {
            weight,
            training: true,
        }
    }
}

impl<T: Scalar, B: Backend> Module<T, B> for EmbeddingLayer<T, B> {
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B> {
        Tensor::embedding(x, &self.weight)
    }

    fn forward_var_tracked(
        &self,
        x: &Variable<T, B>,
        tape: &Rc<Tape<T, B>>,
    ) -> Result<ForwardResult<T, B>> {
        #[cfg(not(feature = "cpu"))]
        {
            let _ = (x, tape);
            Err(Error::invalid(
                "nabla: EmbeddingLayer::forward_var_tracked is CPU-only",
            ))
        }
        #[cfg(feature = "cpu")]
        {
            let w_var = tape.variable(self.weight.clone())?;
            let (_, n) = x.data().shape();
            let num_embeddings = self.weight.nrows();
            let indices: Vec<usize> = (0..n)
            .map(|i| {
                let v = x.data().get(0, i).to_f64();
                assert!(
                    v >= 0.0,
                    "nabla: embedding index must be non-negative, got {v} at position {i}"
                );
                let idx = v as usize;
                assert!(
                    idx < num_embeddings,
                    "nabla: embedding index {idx} out of bounds [0, {num_embeddings}) at position {i}"
                );
                idx
            })
            .collect();
            let output = w_var.embedding_lookup(&indices);
            Ok(ForwardResult {
                output,
                param_vars: vec![w_var],
            })
        }
    }

    impl_module_params!(weight);
}

/// Layer normalization module with learnable affine parameters.
pub struct LayerNormModule<T: Scalar, B: Backend> {
    /// Scale parameter `(1, features)`.
    pub gamma: Tensor<T, B>,
    /// Shift parameter `(1, features)`.
    pub beta: Tensor<T, B>,
    eps: f64,
    training: bool,
}

impl<T: Scalar> LayerNormModule<T, DefaultBackend> {
    /// Create a new layer normalization module for `features` dimensions.
    #[must_use]
    pub fn new(features: usize) -> Self {
        let gamma = Tensor::ones(1, features);
        let beta = Tensor::zeros(1, features);
        Self {
            gamma,
            beta,
            eps: 1e-5,
            training: true,
        }
    }
}

impl<T: Scalar, B: Backend> Module<T, B> for LayerNormModule<T, B> {
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B> {
        let eps_t = T::from_f64(self.eps);
        let normed = x.layer_norm(1, eps_t);
        let (nrows, ncols) = normed.shape();
        let gamma = self.gamma.expand(nrows, ncols);
        let beta = self.beta.expand(nrows, ncols);
        normed.emul(&gamma) + &beta
    }

    fn forward_var_tracked(
        &self,
        x: &Variable<T, B>,
        tape: &Rc<Tape<T, B>>,
    ) -> Result<ForwardResult<T, B>> {
        let gamma_var = tape.variable(self.gamma.clone())?;
        let beta_var = tape.variable(self.beta.clone())?;
        let normed = x.layer_norm(self.eps);
        let nrows = x.data().nrows();
        let ones = tape.variable(Tensor::ones(nrows, 1))?;
        let gamma_exp = ones.matmul(&gamma_var);
        let scaled = normed.emul(&gamma_exp);
        let ones_b = tape.variable(Tensor::ones(nrows, 1))?;
        let beta_exp = ones_b.matmul(&beta_var);
        let output = scaled.add_var(&beta_exp);
        Ok(ForwardResult {
            output,
            param_vars: vec![gamma_var, beta_var],
        })
    }

    impl_module_params!(gamma, beta);
}
