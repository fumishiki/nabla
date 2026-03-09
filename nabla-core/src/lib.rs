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

#[cfg(not(any(feature = "cpu", feature = "gpu", feature = "cuda", feature = "hip")))]
compile_error!("nabla-core: enable at least one backend feature (cpu / wgpu / cuda / hip)");

/// Scalar trait and primitive numeric types (f32, f64, complex, dual).
#[path = "common/scalar/mod.rs"]
pub mod scalar;

/// Tensor types and all element-wise / linear-algebra operations.
#[path = "common/tensor/mod.rs"]
pub mod tensor;

/// Backend abstraction layer: CPU, wgpu, CUDA, HIP.
#[path = "common/backend/mod.rs"]
pub mod backend;

pub use backend::error;

#[allow(missing_docs)]
#[path = "common/layout.rs"]
pub mod layout;

#[cfg(feature = "cpu")]
#[allow(missing_docs)]
#[path = "cpu/mod.rs"]
pub(crate) mod cpu;

#[cfg(feature = "gpu")]
#[allow(missing_docs)]
#[path = "wgpu/mod.rs"]
pub mod gpu;

#[cfg(any(feature = "cuda", feature = "hip"))]
#[path = "cuda_hip/common/kernels/mod.rs"]
pub(crate) mod kernels_cu;

#[cfg(any(feature = "cuda", feature = "hip"))]
#[allow(missing_docs)]
#[path = "cuda_hip/mod.rs"]
pub mod gpu_common;

#[cfg(feature = "cuda")]
pub use gpu_common::cuda as cuda_backend;

#[cfg(feature = "hip")]
pub(crate) use gpu_common::hip as hip_backend;

pub use backend::{Backend, DefaultBackend};
pub use layout::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};
pub use scalar::Scalar;
pub use tensor::{MatrixLike, Tensor, TensorView};

#[cfg(feature = "cuda")]
pub use cuda_backend::{
    AllocationProfile, AnalyzedNode, CondCmp, ConditionalGraph, ConditionalKind, CuBuffer,
    CudaError, CudaResult, CudaStorage, DoubleBuffer, Epilogue, EpilogueCandidate, FusionCandidate,
    KernelClass, KernelNodeState, NablaCudaGraph, NablaGraph, OptimizationReport, PyGraph,
    PyGraphTrainingGraph, TrainingGraph, TransposeElimCandidate, analyze_graph, apply_all_fusions,
    apply_fusion, cuda_compute_stream, cuda_conditional_set_from_scalar, cuda_copy_from_host,
    cuda_graph_capture, cuda_graph_capture_cached, cuda_if_positive, cuda_launch_kernel_src,
    cuda_matmul_epilogue, cuda_matmul_epilogue_bf16, cuda_pool_diagnostics,
    cuda_pool_start_recording, cuda_pool_stop_recording_and_warm, cuda_pre_warm_pool,
    cuda_synchronize, cuda_to_vec_async, cuda_transfer_stats, cuda_transfer_stats_reset,
    cuda_upload_u32, extract_allocation_profile, optimize_with_cache,
};
