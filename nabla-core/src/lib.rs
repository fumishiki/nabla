//! nabla-core — Core tensor, backend, and scalar types for the nabla DSL.

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
    clippy::cast_precision_loss
)]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        clippy::approx_constant,
        clippy::assertions_on_constants
    )
)]

// Require at least one backend. Multiple backends are allowed — DefaultBackend priority: cuda > hip > gpu > cpu.
#[cfg(not(any(feature = "cpu", feature = "gpu", feature = "cuda", feature = "hip")))]
compile_error!("nabla-core: enable at least one backend feature (cpu / wgpu / cuda / hip)");

// ── common ──────────────────────────────────────────────────────────────────

/// Error types for nabla operations.
#[path = "common/error.rs"]
pub mod error;

/// Scalar numeric types supported by nabla.
#[path = "common/scalar/mod.rs"]
pub mod scalar;

/// 2-D dense tensor type with operator overloads.
#[path = "common/tensor/mod.rs"]
pub mod tensor;

/// Compute backend trait + DefaultBackend alias.
#[path = "common/backend.rs"]
pub mod backend;

// ── cpu ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[path = "cpu/mod.rs"]
pub(crate) mod cpu;

// ── wgpu ────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[path = "wgpu/mod.rs"]
pub mod gpu;

/// F₂ binary matrix layout for GPU shared memory swizzling.
#[path = "wgpu/layout.rs"]
pub mod layout;

// ── cuda/hip ─────────────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", feature = "hip"))]
#[path = "cuda_hip/kernels.rs"]
pub(crate) mod kernels_cu;

#[cfg(any(feature = "cuda", feature = "hip"))]
#[path = "cuda_hip/mod.rs"]
pub mod gpu_common;

#[cfg(feature = "cuda")]
#[path = "cuda_hip/cuda.rs"]
pub mod cuda_backend;

#[cfg(feature = "hip")]
#[path = "cuda_hip/hip.rs"]
pub(crate) mod hip_backend;

// ── re-exports ───────────────────────────────────────────────────────────────

pub use backend::{Backend, DefaultBackend};
pub use layout::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};
pub use scalar::Scalar;
pub use tensor::{MatrixLike, Tensor, TensorView};

#[cfg(feature = "cuda")]
pub use cuda_backend::{
    CuBuffer, NablaCudaGraph, cuda_graph_capture, cuda_graph_capture_cached, cuda_synchronize,
    CudaError, CudaResult, DoubleBuffer, KernelNodeState, PyGraph, TrainingGraph,
    cuda_copy_from_host, cuda_to_vec_async,
};

#[cfg(feature = "cuda")]
pub use cuda_backend::{ConditionalGraph, ConditionalKind};

#[cfg(feature = "cuda")]
pub use cuda_backend::{CondCmp, cuda_conditional_set_from_scalar, cuda_if_positive};

#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu_common::RtcStorage;
