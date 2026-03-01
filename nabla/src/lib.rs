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

// ---------------------------------------------------------------------------
// Re-export nabla-core modules for path-based access
// ---------------------------------------------------------------------------

pub use nabla_core::{
    LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64, backend, error, layout,
    scalar, tensor,
};

#[cfg(feature = "gpu")]
pub use nabla_core::gpu;

#[cfg(feature = "gpu")]
/// WGSL shader generators (wgpu backend only).
pub use nabla_core::gpu::shaders as wgsl;

/// CUDA Graph capture/replay for eliminating kernel launch overhead.
#[cfg(feature = "cuda")]
pub use nabla_core::{
    DoubleBuffer, KernelNodeState, NablaCudaGraph, PyGraph, TrainingGraph,
    cuda_copy_from_host, cuda_graph_capture, cuda_graph_capture_cached,
    cuda_synchronize, cuda_to_vec_async,
};

/// CUDA Conditional Nodes: IF/WHILE/SWITCH control flow inside graphs (requires CUDA 12.4+).
#[cfg(feature = "cuda")]
pub use nabla_core::{ConditionalGraph, ConditionalKind};

/// Conditional-set helpers: drive IF/WHILE nodes from scalar tensors (requires CUDA 12.4+).
#[cfg(feature = "cuda")]
pub use nabla_core::{CondCmp, cuda_conditional_set_from_scalar, cuda_if_positive};

/// cublasLt epilogue-fused GEMM (Relu/Gelu/Bias variants, f32 only).
#[cfg(feature = "cuda")]
pub use nabla_core::{CudaError, CudaResult, Epilogue, cuda_matmul_epilogue};

// ---------------------------------------------------------------------------
// Domain modules
// ---------------------------------------------------------------------------

/// Dense linear algebra factorizations and solvers.
#[cfg(feature = "cpu")]
pub mod linalg;

/// Sparse matrix support.
#[cfg(feature = "cpu")]
pub mod sparse;

/// Reverse-mode automatic differentiation.
pub mod autograd;

/// Symbolic computer algebra system.
pub mod cas;

/// ODE/SDE solvers: Euler, RK4, Dormand-Prince, Euler-Maruyama, Milstein.
pub mod ode;

/// Free constructor and math functions for tensors.
pub mod constructors;

/// Neural network utility functions (initializers, RoPE, KV cache, backslash).
pub mod nn;

/// Optimizer utilities: AdamW, learning-rate schedules, and gradient scaling.
pub mod optim;

/// Neural network module trait (`nn.Module` equivalent).
pub mod module;

/// Stateful optimizer trait and implementations (AdamW with moment tracking).
pub mod optimizer;

/// Tensor serialization: save and load named tensors in the NBLA binary format.
pub mod io;

// Declarative macros mirroring Julia math notation; `#[macro_export]` places them at crate root.
mod notation;

// ---------------------------------------------------------------------------
// Wildcard re-exports from split modules (preserves `nabla::zeros`, etc.)
// ---------------------------------------------------------------------------

pub use {constructors::*, io::*, nn::*, optim::*};

// Explicit re-export for macro hygiene: `$crate::approx_eq` in the `approx!`
// macro must resolve at crate root.
pub use constructors::approx_eq;

// ---------------------------------------------------------------------------
// Backward-compatible `util` module
// ---------------------------------------------------------------------------

/// Utility functions mirroring Julia math notation.
///
/// Macros (`between!`, `frange!`, `map!`, etc.) are also available directly at
/// the crate root via `#[macro_export]`; this module keeps the `nabla::util::`
/// path working for non-macro helpers such as [`util::c32`] and [`util::c64`].
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
    pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
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

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

/// Prelude for convenient imports.
pub mod prelude {
    pub use nabla_core::backend::{Backend, DefaultBackend};
    pub use nabla_core::error::{Error, Result};
    pub use nabla_core::layout::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};
    pub use nabla_core::scalar::Scalar;
    pub use nabla_core::tensor::{Array, NdTensor, StaticMatrix};
    #[cfg(feature = "cpu")]
    pub use nabla_core::tensor::{DynTensor, Matrix};
    pub use nabla_core::tensor::nn_conv::{Conv1dConfig, Conv2dConfig, Conv3dConfig};
    pub use nabla_core::tensor::MatrixLike;
    pub use nabla_core::tensor::{MatmulCompat, Tensor, TensorView};
    pub use nabla_macros::{
        axis, block, einsum, fuse, generated, mat, mega_fuse, nabla_grad, named, named_zeros, sym,
    };

    #[cfg(feature = "cpu")]
    pub use crate::{
        autograd::{GradPrep, grad, gradient, gradient_prep},
        linalg::{
            Diagonal, LinalgExt, Side, Symmetric, TriKind, Triangular, discrete_lyapunov,
            discrete_sylvester, expm, logm, lyapunov, schur, sqrtm, sylvester,
        },
        ode::{
            DaeConfig, IfEulerScalarConfig, bdf1, bdf2, dae_solve, ensemble_euler_maruyama,
            euler_maruyama, if_euler_scalar, metd_solve, milstein, parareal_solve,
            parareal_solve_tensor, stormer_verlet,
        },
        sparse::*,
    };
    pub use crate::ode::{
        euler, rk4, dormand_prince,
        euler_with_config, rk4_with_config,
        EulerConfig, Rk4Config, OdeProblem,
    };
    pub use crate::{
        autograd::{Tape, Variable, clip_grad_norm, scale_grad, zero_grad},
        cas::{Expr, ExprKind, diff, eval, eval_tensor, hessian, jacobian, simplify, substitute, var},
        constructors::{
            approx_eq, arange, cross, diagm, dot, eye, fill, from_fn, geomspace, kron, linspace,
            logspace, nd_zeros, norm, norm_ord, ones, rand, randn, tr, zeros,
        },
        io::{load_tensors, save_tensors},
        module::{Linear, Module},
        nn::{embedding, kaiming_normal, kv_cache_append, linear, rotary_embedding, xavier_uniform},
        optimizer::{AdamW, Optimizer, SGD},
        ode::{
            AdaptiveConfig, Bdf1Config, Bdf2Config, MetdConfig, OdeSolution, PararealConfig,
            SdeConfig, StormerVerletConfig, SymplecticSolution,
        },
        optim::{GradScaler, LrSchedule, LrScheduler, adamw_step, lr_at_step},
    };
    #[cfg(feature = "cpu")]
    pub use crate::nn::backslash;
    #[cfg(feature = "cpu")]
    pub use crate::linalg::{
        balance, care, circulant, continuous_riccati, frechet_deriv, hessenberg, polar,
        solve_tridiag, toeplitz, vandermonde, vandermonde_rect,
    };
    #[cfg(feature = "cpu")]
    pub use half::{bf16, f16};
    #[cfg(feature = "cpu")]
    pub use nabla_core::backend::Cpu;
    #[cfg(feature = "cpu")]
    pub use nabla_core::scalar::{Dual, MultiDual, c32, c64};
    #[cfg(feature = "cpu")]
    pub use nabla_macros::stencil;

    #[cfg(feature = "gpu")]
    pub use nabla_core::backend::Gpu;

    #[cfg(feature = "cuda")]
    pub use nabla_core::{
        CudaError, CudaResult, DoubleBuffer, Epilogue, KernelNodeState, NablaCudaGraph,
        PyGraph, TrainingGraph,
        cuda_copy_from_host, cuda_graph_capture, cuda_graph_capture_cached,
        cuda_matmul_epilogue, cuda_synchronize, cuda_to_vec_async,
    };

    #[cfg(feature = "cuda")]
    pub use nabla_core::{ConditionalGraph, ConditionalKind};
}
