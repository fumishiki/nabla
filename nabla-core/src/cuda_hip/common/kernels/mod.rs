#![allow(dead_code)]

pub(crate) const BLOCK_SIZE: u32 = 256;
pub(crate) const REDUCE_BLOCK: u32 = BLOCK_SIZE;
pub(crate) const REDUCE_GRID_CAP: u32 = 256;

pub(crate) const KERNELS: &str = concat!(
    include_str!("k_basic_core.cuh"),
    include_str!("k_basic_math.cuh"),
    include_str!("k_basic_red32.cuh"),
    include_str!("k_basic_red64.cuh"),
    include_str!("k_norm_softmax.cuh"),
    include_str!("k_norm_group.cuh"),
    include_str!("k_norm_reduce.cuh"),
    include_str!("k_norm_pool.cuh"),
    include_str!("k_conv_bn_loss.cuh"),
    include_str!("k_attn.cuh"),
    include_str!("k_conv_misc.cuh")
);
#[cfg(feature = "cuda")]
pub(crate) const WMMA_KERNELS: &str = include_str!("kernels_wmma_cuda.cuh");

#[cfg(feature = "hip")]
pub(crate) const WMMA_KERNELS: &str = include_str!("kernels_wmma_hip.cuh");

#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub(crate) const WMMA_KERNELS: &str = "";

#[cfg(feature = "cuda")]
pub(crate) const COND_SET_KERNELS: &str = include_str!("kernels_cond_set.cuh");
