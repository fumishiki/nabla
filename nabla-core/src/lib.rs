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

/// Error types for nabla operations.
pub mod error;

/// Scalar numeric types supported by nabla.
pub mod scalar;

/// Compute backends (CPU + GPU + CUDA + HIP).
pub mod backend;

#[cfg(feature = "gpu")]
/// wgpu backend module.
pub mod gpu;

#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) mod kernels_cu;

#[cfg(any(feature = "cuda", feature = "hip"))]
pub mod gpu_common;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[cfg(feature = "cuda")]
pub use cuda_backend::{CuBuffer, NablaCudaGraph, cuda_graph_capture, cuda_graph_capture_cached, cuda_synchronize};

#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu_common::RtcStorage;

#[cfg(feature = "hip")]
pub(crate) mod hip_backend;

/// 2-D dense tensor type with operator overloads.
pub mod tensor;

/// F₂ binary matrix layout for GPU shared memory swizzling.
pub mod layout;

/// WGSL shader generators (pure string ops, always compiled).
pub mod wgsl;

pub use backend::{Backend, DefaultBackend};
pub use layout::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};
pub use scalar::Scalar;
pub use tensor::Tensor;
