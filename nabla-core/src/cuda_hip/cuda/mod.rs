#![allow(unused_imports)]

mod backend;
mod blas_ops;
mod core;
mod fusion;
mod graph;
mod graph_compile;
mod indexing_ops;
mod norm_attn_ops;
mod ops;
mod reduce;
mod training;

pub(super) use backend::*;
pub(super) use blas_ops::*;
pub(super) use core::*;
pub(super) use fusion::*;
pub(super) use graph::*;
pub(super) use graph_compile::*;
pub(super) use indexing_ops::*;
pub(super) use norm_attn_ops::*;
pub(super) use ops::*;
pub(super) use reduce::*;
pub(super) use training::*;

pub(crate) use backend::{launch_unary, launch_unary_inplace};
pub use blas_ops::{cuda_matmul_epilogue, cuda_matmul_epilogue_bf16};
pub use core::{
    CuBuffer, CudaError, CudaResult, CudaStorage, Epilogue, cuda_compute_stream,
    cuda_pool_diagnostics, cuda_pool_start_recording, cuda_pool_stop_recording_and_warm,
    cuda_pre_warm_pool, cuda_synchronize, cuda_transfer_stats, cuda_transfer_stats_reset,
    cuda_upload_u32,
};
pub use fusion::cuda_launch_kernel_src;
pub use graph::{
    AllocationProfile, AnalyzedNode, CondCmp, ConditionalGraph, ConditionalKind, EpilogueCandidate,
    FusionCandidate, KernelClass, KernelNodeState, NablaCudaGraph, OptimizationReport, PyGraph,
    PyGraphTrainingGraph, TransposeElimCandidate, analyze_graph, cuda_conditional_set_from_scalar,
    cuda_copy_from_host, cuda_graph_capture, cuda_graph_capture_cached, cuda_if_positive,
    cuda_to_vec_async, extract_allocation_profile,
};
pub use graph_compile::{apply_all_fusions, apply_fusion, optimize_with_cache};
pub(crate) use ops::cuda_scale_inplace;
pub use training::{DoubleBuffer, NablaGraph, TrainingGraph};
pub(crate) use training::{GpuOp, GpuTape};
