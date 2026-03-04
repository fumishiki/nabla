//! nabla — Rust linear algebra DSL.
//!
//! Import [`prelude`] for the most common types and traits.

#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]
#![allow(
    clippy::return_self_not_must_use,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::type_complexity
)]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        clippy::approx_constant,
        clippy::assertions_on_constants
    )
)]

pub use nabla_core::{
    LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64, backend, error, layout, scalar,
    tensor,
};

#[cfg(feature = "gpu")]
pub use nabla_core::gpu;

#[cfg(feature = "gpu")]
pub use nabla_core::gpu::shaders as wgsl;

#[cfg(feature = "cuda")]
pub use nabla_core::{CudaError, CudaResult, cuda_synchronize};

#[cfg(feature = "cuda")]
pub use nabla_core::cuda_backend::{
    DoubleBuffer, Epilogue, KernelNodeState, NablaCudaGraph, PyGraph, TrainingGraph,
    cuda_copy_from_host, cuda_graph_capture, cuda_graph_capture_cached, cuda_matmul_epilogue,
    cuda_to_vec_async,
};

#[cfg(feature = "cuda")]
pub use nabla_core::cuda_backend::{ConditionalGraph, ConditionalKind};

#[cfg(feature = "cuda")]
pub use nabla_core::cuda_backend::{CondCmp, cuda_conditional_set_from_scalar, cuda_if_positive};

#[cfg(feature = "cpu")]
#[allow(missing_docs)]
pub mod linalg;

#[cfg(feature = "cpu")]
pub use linalg::backslash;

#[cfg(feature = "cpu")]
#[allow(missing_docs)]
#[path = "misc/sparse.rs"]
pub mod sparse;

/// Reverse-mode automatic differentiation (autograd).
pub mod autograd;

/// Computer algebra system for symbolic expressions.
pub mod cas;

/// ODE/SDE solvers for initial value problems.
#[cfg(feature = "cpu")]
pub mod ode;

/// Neural network modules and layers.
pub mod module;

#[path = "misc/surface.rs"]
mod surface;

pub use surface::constructors;

pub use surface::notation;

/// Convenient Result alias for examples and applications.
pub type NablaResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub use nabla_macros::nabla_main as main;

/// Batch-create tracked Variables on a tape.
///
/// `vars!(tape; w1 = expr1, w2 = expr2)` expands to
/// `let w1 = tape.variable(expr1)?; let w2 = tape.variable(expr2)?;`
#[macro_export]
macro_rules! vars {
    ($tape:expr; $($name:ident = $init:expr),+ $(,)?) => {
        $(let $name = $tape.variable($init)?;)+
    };
}

/// Differentiate a block with respect to tracked variables.
///
/// Creates a tape, wraps each initializer as a `Variable`, executes the body,
/// calls `backward()` on the result, and extracts gradients.
///
/// Returns `(loss_variable, grad1, grad2, ...)`.
///
/// The implicit `tape` binding is available inside the body for creating
/// additional tracked variables via `tape.variable(expr)?`.
#[macro_export]
macro_rules! ad {
    ($backend:ty; $($name:ident = $init:expr),+ => |$tape_name:ident| $body:block) => {{
        let $tape_name = $crate::autograd::Tape::<_, $backend>::new();
        $(let $name = $tape_name.variable($init)?;)+
        let __ad_loss = $body;
        __ad_loss.backward()?;
        (__ad_loss, $($name.grad()?),+)
    }};
    ($backend:ty; $($name:ident = $init:expr),+ => $body:block) => {{
        #[allow(unused_variables)]
        let tape = $crate::autograd::Tape::<_, $backend>::new();
        $(let $name = tape.variable($init)?;)+
        let __ad_loss = $body;
        __ad_loss.backward()?;
        (__ad_loss, $($name.grad()?),+)
    }};
}

/// Build a `Sequential` model from a list of layers.
///
/// Each expression must implement `Module<T, B>` and be `'static`.
/// The macro wraps each layer in `Box::new()` and calls `.push()`.
#[macro_export]
macro_rules! sequential {
    ($($layer:expr),+ $(,)?) => {{
        let mut __seq = $crate::nn::Sequential::new();
        $(__seq.push(::std::boxed::Box::new($layer));)+
        __seq
    }};
}

/// Create a variable binding map for CAS evaluation.
///
/// `cas_vars!{ x: 1.0, y: 2.0 }` expands to `HashMap::from([("x", 1.0), ("y", 2.0)])`.
#[macro_export]
macro_rules! cas_vars {
    ($($name:ident : $val:expr),+ $(,)?) => {
        std::collections::HashMap::from([$(( stringify!($name), $val )),+])
    };
}

/// Destructure a column vector into scalar variables.
#[macro_export]
macro_rules! vec_unpack {
    ($tensor:expr, $($var:ident),+ $(,)?) => {
        let __nabla_tmp = &$tensor;
        let mut __nabla_idx = 0usize;
        $(
            let $var = __nabla_tmp.get(__nabla_idx, 0);
            __nabla_idx += 1;
        )+
    };
}

/// Re-exports of neural network types and functions.
pub mod nn {
    pub use crate::module::{
        Activation, ActivationKind, DropoutLayer, EmbeddingLayer, ForwardResult, LayerNormModule,
        Linear, Module, Sequential, StateError, embedding, kv_cache_append, linear, load_tensors,
        save_tensors,
    };
    #[cfg(feature = "cpu")]
    pub use crate::module::{kaiming_normal, rotary_embedding, xavier_uniform};
}

pub use {constructors::*, nn::*};

/// Miscellaneous numeric utilities.
pub mod util {
    /// Create a 32-bit complex number.
    #[cfg(feature = "cpu")]
    #[inline]
    #[must_use]
    pub fn c32(re: f32, im: f32) -> crate::scalar::c32 {
        crate::scalar::c32::new(re, im)
    }

    /// Create a 64-bit complex number.
    #[cfg(feature = "cpu")]
    #[inline]
    #[must_use]
    pub fn c64(re: f64, im: f64) -> crate::scalar::c64 {
        crate::scalar::c64::new(re, im)
    }

    /// Linearly spaced vector from `start` to `stop` inclusive.
    #[must_use]
    pub fn linspace_vec(start: f64, stop: f64, n: usize) -> Vec<f64> {
        match n {
            0 => Vec::new(),
            1 => vec![start],
            _ => {
                #[allow(clippy::cast_precision_loss)]
                let delta = (stop - start) / (n as f64 - 1.0);
                #[allow(clippy::cast_precision_loss)]
                (0..n).map(|i| start + i as f64 * delta).collect()
            }
        }
    }
}

/// Convenience re-exports for common types and traits.
pub mod prelude {
    // Core types
    pub use nabla_core::backend::{Backend, DefaultBackend};
    pub use nabla_core::error::{Error, Result};
    pub use nabla_core::layout::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};
    pub use nabla_core::scalar::{Fp4E2M1, Fp8E4M3, Fp8E5M2, Scalar};
    #[cfg(feature = "cpu")]
    pub use nabla_core::tensor::Matrix;
    #[cfg(feature = "cpu")]
    pub use nabla_core::tensor::{Array, NdTensor, StaticMatrix};
    pub use nabla_core::tensor::{MatrixLike, Tensor, TensorView};

    // Macros (proc + decl)
    pub use crate::{NablaResult, ad, cas_vars, sequential, vars, vec_unpack};
    pub use nabla_macros::{
        Module, axis, block, einsum, fuse, generated, mat, math, mega_fuse, nabla_grad, named,
        named_zeros, sym,
    };

    // Autograd
    pub use crate::autograd::{Tape, TensorLike, Variable, clip_grad_norm, scale_grad, zero_grad};

    // Module / NN
    pub use crate::nn::{
        Activation, ActivationKind, DropoutLayer, EmbeddingLayer, ForwardResult, LayerNormModule,
        Linear, Module, Sequential, embedding, linear, load_tensors, save_tensors,
    };
    #[cfg(feature = "cpu")]
    pub use crate::nn::{kaiming_normal, xavier_uniform};

    // Constructors
    #[cfg(feature = "cpu")]
    pub use crate::constructors::{
        arange, clear_seed, eye, from_fn, linspace, nd_zeros, rand, randn, set_seed,
    };
    pub use crate::constructors::{fill, ones, zeros};

    // IO
    pub use crate::nn::StateError;

    // CAS (basic)
    pub use crate::cas::{diff, diff_simplify, eval, simplify};

    // ODE (basic)
    #[cfg(feature = "cpu")]
    pub use crate::ode::{AdaptiveConfig, OdeProblem, OdeSolution, dormand_prince, euler, rk4};

    // --- cpu-gated ---
    #[cfg(feature = "cpu")]
    pub use crate::{
        autograd::{GradPrep, grad, gradient, gradient_prep},
        linalg::{
            Diagonal, LinalgExt, Side, Symmetric, TriKind, Triangular, discrete_lyapunov,
            discrete_sylvester, expm, logm, lyapunov, schur, sqrtm, sylvester,
        },
        ode::{DaeConfig, IfEulerScalarConfig, dae_solve, if_euler_scalar},
        sparse::*,
    };
    #[cfg(feature = "cpu")]
    pub use half::{bf16, f16};
    #[cfg(all(feature = "cuda", not(feature = "cpu")))]
    pub use half::{bf16, f16};
    #[cfg(feature = "cpu")]
    pub use nabla_core::backend::Cpu;
    #[cfg(feature = "cpu")]
    pub use nabla_core::scalar::{Dual, MultiDual, c32, c64};
    #[cfg(feature = "cpu")]
    pub use nabla_macros::stencil;

    // --- gpu-gated ---
    #[cfg(feature = "gpu")]
    pub use nabla_core::backend::Gpu;

    // --- cuda-gated ---
    #[cfg(feature = "cuda")]
    pub use nabla_core::cuda_backend::{
        Epilogue, NablaCudaGraph, PyGraph, TrainingGraph, cuda_graph_capture, cuda_matmul_epilogue,
    };
    #[cfg(feature = "cuda")]
    pub use nabla_core::{CudaError, CudaResult, cuda_synchronize};
}
