use std::ffi::c_void;

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{self, grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

/// Check if kernel is cached; if not, compile `src_fn()` via NVRTC and register it.
fn ensure_kernel(
    ctx: &CudaCtx,
    kernel_name: &str,
    src_fn: impl FnOnce() -> String,
    _maxreg: Option<u32>,
) {
    {
        let map = ctx
            .kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.contains_key(kernel_name) {
            return;
        }
    }
    let src = src_fn();
    let _ = super::graph_compile::compile_and_cache_kernel(ctx, kernel_name, &src)
        .or_panic("NVRTC compile failed");
}

/// Compile `src` via NVRTC if not already cached, then register.
fn ensure_kernel_src(ctx: &CudaCtx, kernel_name: &str, src: &str) {
    {
        let map = ctx
            .kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.contains_key(kernel_name) {
            return;
        }
    }
    let _ = super::graph_compile::compile_and_cache_kernel(ctx, kernel_name, src)
        .or_panic("NVRTC compile failed");
}

/// SAFETY: each `*const u8` in `inputs` must be a valid `&CudaStorage<T>` cast.
unsafe fn extract_input_ptrs<T: Scalar>(inputs: &[*const u8]) -> Vec<CUdeviceptr> {
    inputs
        .iter()
        .map(|&p| {
            let storage = unsafe { &*(p as *const CudaStorage<T>) };
            storage.buf.ptr
        })
        .collect()
}

pub fn cuda_launch_kernel_src(
    kernel_name: &str,
    src: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    args: &mut [*mut c_void],
) {
    let ctx = get_ctx();
    ensure_kernel_src(ctx, kernel_name, src);
    let func = expect_ok(get_kernel(ctx, kernel_name), "CUDA kernel lookup");
    unsafe {
        result::launch_kernel(
            func,
            grid,
            block,
            shared_bytes,
            ctx.stream.cu_stream(),
            args,
        )
        .or_panic("CUDA launch kernel");
    }
}

pub(super) fn cuda_fuse_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
    reg_estimate: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_fused_{kernel_hash}_{tsuf}");

    ensure_kernel(
        ctx,
        &kernel_name,
        || {
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            gpu_common::fuse_kernel_source(
                gpu_expr,
                n_inputs,
                type_name,
                &kernel_name,
                reg_estimate,
                true,
            )
        },
        (reg_estimate > 80).then_some(120),
    );

    let func = expect_ok(get_kernel(ctx, &kernel_name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let n_u32 = n as u32;
    let grid = cuda_grid_1d::<T>(n);

    // SAFETY: input pointers are valid CudaStorage<T> cast by the macro.
    let input_ptrs = unsafe { extract_input_ptrs::<T>(inputs) };

    let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + 2);
    args.extend(
        input_ptrs
            .iter()
            .map(|ptr| ptr as *const CUdeviceptr as *mut c_void),
    );
    args.push(&out_buf.ptr as *const CUdeviceptr as *mut c_void);
    args.push(&n_u32 as *const u32 as *mut c_void);

    // SAFETY: launching fused kernel with correct argument layout.
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        )
        .or_panic("CUDA launch fused kernel");
    }
    CudaStorage::new(nrows, ncols, out_buf)
}

#[inline]
fn use_tiled_mega(n_ops: usize, n: usize) -> bool {
    n_ops >= 2 && n >= 65_536
}

fn ops_all_share_inputs(ops: &[MegaFuseOp]) -> bool {
    let Some(first) = ops.first() else {
        return false;
    };
    ops.iter()
        .all(|op| op.n_inputs == first.n_inputs && op.inputs == first.inputs)
}

pub(crate) struct MegaFuseOp {
    pub inputs: Vec<*const u8>,
    pub gpu_expr: String,
    pub n_inputs: usize,
    pub uses_prev: bool,
}

pub(crate) fn cuda_mega_fuse_launch<T: Scalar>(
    ops: &[MegaFuseOp],
    nrows: usize,
    ncols: usize,
    kernel_hash: &str,
) -> Vec<CudaStorage<T>> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let tsuf = type_suffix::<T>();
    let any_dag = ops.iter().any(|op| op.uses_prev);
    let use_tiled = !any_dag && use_tiled_mega(ops.len(), n) && ops_all_share_inputs(ops);
    let kernel_name = if use_tiled {
        format!("k_mega_tile_{kernel_hash}_{tsuf}")
    } else {
        format!("k_mega_{kernel_hash}_{tsuf}")
    };

    ensure_kernel(
        ctx,
        &kernel_name,
        || {
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let op_descs: Vec<(String, usize)> = ops
                .iter()
                .map(|op| (op.gpu_expr.clone(), op.n_inputs))
                .collect();
            let op_uses_prev: Vec<bool> = ops.iter().map(|op| op.uses_prev).collect();
            if use_tiled {
                gpu_common::mega_fuse_tiled_kernel_source(&op_descs, type_name, &kernel_name, true)
            } else {
                gpu_common::mega_fuse_kernel_source(
                    &op_descs,
                    &op_uses_prev,
                    type_name,
                    &kernel_name,
                    true,
                )
            }
        },
        None,
    );

    let func = expect_ok(get_kernel(ctx, &kernel_name), "CUDA kernel lookup");

    let out_bufs: Vec<CuBuffer> = (0..ops.len())
        .map(|_| {
            CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc")
        })
        .collect();

    let n_u32 = n as u32;

    if use_tiled {
        let n_inputs = ops[0].n_inputs;
        // SAFETY: each *const u8 was cast from a &CudaStorage<T> by the macro.
        let shared_input_ptrs = unsafe { extract_input_ptrs::<T>(&ops[0].inputs) };

        let tile_size: usize = if tsuf == "f32" { 1024 } else { 512 };
        let grid = grid_1d(n.div_ceil(tile_size));

        let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + ops.len() + 1);
        args.extend(
            shared_input_ptrs
                .iter()
                .map(|ptr| ptr as *const CUdeviceptr as *mut c_void),
        );
        args.extend(
            out_bufs
                .iter()
                .map(|buf| &buf.ptr as *const CUdeviceptr as *mut c_void),
        );
        args.push(&n_u32 as *const u32 as *mut c_void);

        // SAFETY: launching tiled mega-kernel; argument layout matches generated source.
        unsafe {
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut args,
            )
            .or_panic("CUDA launch mega-fused kernel");
        }
    } else {
        let grid = cuda_grid_1d::<T>(n);

        // SAFETY: raw pointer cast back to CudaStorage<T>; caller guarantees validity.
        let input_ptrs: Vec<Vec<CUdeviceptr>> = ops
            .iter()
            .map(|op| unsafe { extract_input_ptrs::<T>(&op.inputs) })
            .collect();

        let total_args = ops.iter().map(|op| op.n_inputs + 1).sum::<usize>() + 1;
        let mut args: Vec<*mut c_void> = Vec::with_capacity(total_args);
        for (op_idx, op) in ops.iter().enumerate() {
            args.extend(
                input_ptrs[op_idx][..op.n_inputs]
                    .iter()
                    .map(|ptr| ptr as *const CUdeviceptr as *mut c_void),
            );
            args.push(&out_bufs[op_idx].ptr as *const CUdeviceptr as *mut c_void);
        }
        args.push(&n_u32 as *const u32 as *mut c_void);

        // SAFETY: launching standard mega-fused kernel with correct argument layout.
        unsafe {
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut args,
            )
            .or_panic("CUDA launch mega-fused kernel");
        }
    }

    out_bufs
        .into_iter()
        .map(|buf| CudaStorage::new(nrows, ncols, buf))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_fuse_reduce_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
    reduce_op: u8,
    axis: u8,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_fuse_reduce_{kernel_hash}_{tsuf}");

    ensure_kernel(
        ctx,
        &kernel_name,
        || {
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            gpu_common::fuse_reduce_kernel_source(
                gpu_expr,
                n_inputs,
                type_name,
                &kernel_name,
                axis,
                true,
            )
        },
        None,
    );

    let (out_rows, out_cols, grid_dim) = if axis == 1 {
        (nrows, 1usize, nrows as u32)
    } else {
        (1usize, ncols, ncols as u32)
    };

    let func = expect_ok(get_kernel(ctx, &kernel_name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, out_rows * out_cols * std::mem::size_of::<T>())
            .or_panic("CUDA alloc");

    let rows_u32 = nrows as u32;
    let cols_u32 = ncols as u32;

    // SAFETY: each *const u8 in `inputs` is a valid &CudaStorage<T> cast.
    let input_ptrs = unsafe { extract_input_ptrs::<T>(inputs) };

    let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + 3);
    args.extend(
        input_ptrs
            .iter()
            .map(|ptr| ptr as *const CUdeviceptr as *mut c_void),
    );
    args.push(&out_buf.ptr as *const CUdeviceptr as *mut c_void);
    args.push(&rows_u32 as *const u32 as *mut c_void);
    args.push(&cols_u32 as *const u32 as *mut c_void);

    // SAFETY: launching fused map-reduce kernel with correct argument layout.
    unsafe {
        result::launch_kernel(
            func,
            (grid_dim, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        )
        .or_panic("CUDA launch fuse-reduce kernel");
    }

    let summed = CudaStorage::new(out_rows, out_cols, out_buf);

    if reduce_op == 3 {
        let count = if axis == 1 { ncols } else { nrows };
        let inv_n = T::from_f64(1.0 / count as f64);
        cuda_scale(&summed, inv_n)
    } else {
        summed
    }
}
