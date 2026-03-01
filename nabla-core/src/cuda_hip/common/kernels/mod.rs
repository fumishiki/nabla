
#![allow(dead_code)]

pub(crate) const BLOCK_SIZE: u32 = 256;
pub(crate) const REDUCE_BLOCK: u32 = BLOCK_SIZE;
pub(crate) const REDUCE_GRID_CAP: u32 = 256;

pub(crate) const KERNELS: &str = concat!(
    include_str!("kernels_basic_ops.cuh"),
    include_str!("kernels_norm_pool.cuh"),
    include_str!("kernels_im2col_conv.cuh")
);
#[cfg(feature = "cuda")]
pub(crate) const WMMA_KERNELS: &str = include_str!("kernels_wmma_cuda.cuh");

#[cfg(feature = "hip")]
pub(crate) const WMMA_KERNELS: &str = include_str!("kernels_wmma_hip.cuh");

#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub(crate) const WMMA_KERNELS: &str = "";

#[cfg(feature = "cuda")]
pub(crate) const COND_SET_KERNELS: &str = include_str!("kernels_cond_set.cuh");
