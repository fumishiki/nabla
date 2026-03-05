//! WebGPU (wgpu) backend — f32 by default, optional f16.
//!
//! WGSL (WebGPU Shading Language) does not include f64 in its core specification;
//! all storage buffers and arithmetic are f32. A `shader-f16` extension exists but
//! f64 has no equivalent extension. This backend supports `Tensor<f32>` and, when the
//! `wgpu-f16` feature is enabled and the device supports shader-f16, `Tensor<f16>`.
//! For f64 workloads use the `cuda`, `hip`, or `cpu` feature instead.

pub(crate) mod ops;
pub mod shaders;
pub(crate) mod storage;

#[allow(unused_imports)]
pub(crate) use ops::*;
#[allow(unused_imports)]
pub(crate) use storage::*;
