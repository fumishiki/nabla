use std::ffi::c_void;

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

/// Cast CUdeviceptr ref to kernel arg pointer.
macro_rules! dp { ($e:expr) => { &$e as *const CUdeviceptr as *mut c_void } }
/// Cast u32 ref to kernel arg pointer.
macro_rules! up { ($e:expr) => { &$e as *const u32 as *mut c_void } }

/// Push a typed f32-or-f64 scalar into a `Vec<*mut c_void>` kernel arg list.
/// For f32 kernels the value is narrowed to f32; otherwise kept as f64.
/// Returns the stored value (must remain alive until kernel launch).
macro_rules! push_typed_scalar {
    ($args:expr, $val:expr, $f32_slot:ident, $f64_slot:ident) => {
        {
            let _f = ($val).to_f64();
            if type_suffix::<T>() == "f32" {
                $f32_slot = _f as f32;
                $args.push(&$f32_slot as *const f32 as *mut c_void);
            } else {
                $f64_slot = _f;
                $args.push(&$f64_slot as *const f64 as *mut c_void);
            }
        }
    };
}

/// Launch kernel with a typed scalar as last arg (f32 for f32 kernels, f64 otherwise).
/// SAFETY: all pointers in `base_args` must be valid device/host pointers.
unsafe fn launch_with_typed_last<T: Scalar>(
    func: cudarc::driver::sys::CUfunction,
    grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32,
    stream: cudarc::driver::sys::CUstream,
    base_args: &mut Vec<*mut c_void>, val: T,
) {
    unsafe {
        let f = val.to_f64();
        if type_suffix::<T>() == "f32" {
            let v = f as f32;
            base_args.push(&v as *const f32 as *mut c_void);
            result::launch_kernel(func, grid, block, smem, stream, base_args).or_panic("CUDA launch");
        } else {
            base_args.push(&f as *const f64 as *mut c_void);
            result::launch_kernel(func, grid, block, smem, stream, base_args).or_panic("CUDA launch");
        }
    }
}

pub(super) fn cuda_layer_norm<T: Scalar>(
    a: &CudaStorage<T>, gamma: &CudaStorage<T>, beta: &CudaStorage<T>, eps: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let (rows, cols) = (a.nrows, a.ncols);
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, "layer_norm", type_suffix::<T>())), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, rows * cols);
    let (r, c) = (rows as u32, cols as u32);
    let mut args = vec![dp!(a.buf.ptr), dp!(gamma.buf.ptr), dp!(beta.buf.ptr), dp!(out_buf.ptr), up!(r), up!(c)];
    // SAFETY: all pointers are valid GPU buffers; eps is typed scalar last arg.
    unsafe { launch_with_typed_last::<T>(func, (r, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(), &mut args, eps); }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_rms_norm<T: Scalar>(
    a: &CudaStorage<T>, gamma: &CudaStorage<T>, eps: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let (rows, cols) = (a.nrows, a.ncols);
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, "rms_norm", type_suffix::<T>())), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, rows * cols);
    let (r, c) = (rows as u32, cols as u32);
    let mut args = vec![dp!(a.buf.ptr), dp!(gamma.buf.ptr), dp!(out_buf.ptr), up!(r), up!(c)];
    // SAFETY: all pointers are valid GPU buffers; eps is typed scalar last arg.
    unsafe { launch_with_typed_last::<T>(func, (r, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(), &mut args, eps); }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_group_norm<T: Scalar>(
    a: &CudaStorage<T>, gamma: &CudaStorage<T>, beta: &CudaStorage<T>, groups: usize, eps: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let (rows, cols) = (a.nrows, a.ncols);
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, "group_norm", type_suffix::<T>())), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, rows * cols);
    let (r, c, g) = (rows as u32, cols as u32, groups as u32);
    let grid_x = (rows * groups) as u32;
    let mut args = vec![dp!(a.buf.ptr), dp!(gamma.buf.ptr), dp!(beta.buf.ptr), dp!(out_buf.ptr), up!(r), up!(c), up!(g)];
    // SAFETY: all pointers are valid GPU buffers; eps is typed scalar last arg.
    unsafe { launch_with_typed_last::<T>(func, (grid_x, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(), &mut args, eps); }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_batch_norm_train<T: Scalar>(
    a: &CudaStorage<T>, gamma: &CudaStorage<T>, beta: &CudaStorage<T>,
    running_mean: &mut CudaStorage<T>, running_var: &mut CudaStorage<T>,
    eps: T, momentum: T, training: bool,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let (n, c) = (a.nrows, a.ncols);
    let total = n * c;
    let (total_u32, c_u32) = (total as u32, c as u32);
    let mut nbuf_fwd = [0u8; 64];
    let fwd_func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf_fwd, "batch_norm_fwd", type_suffix::<T>())), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, total);
    let fwd_grid = (total_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let (mean_s, var_s);
    let (mean_ptr, var_ptr) = if training {
        let mut nbuf_stats = [0u8; 64];
        let stats_func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf_stats, "batch_norm_stats", type_suffix::<T>())), "CUDA kernel lookup");
        let (mean_buf, var_buf) = (alloc_out::<T>(ctx, c), alloc_out::<T>(ctx, c));
        let n_u32 = n as u32;
        // SAFETY: all pointers are valid GPU buffers.
        unsafe {
            result::launch_kernel(
                stats_func, ((c_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                0, ctx.stream.cu_stream(),
                &mut [dp!(a.buf.ptr), dp!(mean_buf.ptr), dp!(var_buf.ptr), up!(n_u32), up!(c_u32)],
            ).or_panic("CUDA launch batch_norm_stats");
        }
        mean_s = Some(CudaStorage::<T>::new(1, c, mean_buf));
        var_s = Some(CudaStorage::<T>::new(1, c, var_buf));
        // SAFETY: both just assigned to Some.
        let ms = mean_s.as_ref().unwrap_or_else(|| unreachable!());
        let vs = var_s.as_ref().unwrap_or_else(|| unreachable!());
        let mut nbuf_upd = [0u8; 64];
        let upd_func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf_upd, "batch_norm_update_running", type_suffix::<T>())), "CUDA kernel lookup");
        let upd_grid = (c_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut upd_args = vec![dp!(running_mean.buf.ptr), dp!(running_var.buf.ptr), dp!(ms.buf.ptr), dp!(vs.buf.ptr)];
        // Momentum is typed scalar; c_u32 follows after it.
        let mut _mom_f32 = 0.0f32;
        let mut _mom_f64 = 0.0f64;
        push_typed_scalar!(upd_args, momentum, _mom_f32, _mom_f64);
        upd_args.push(up!(c_u32));
        // SAFETY: all pointers are valid GPU buffers; scalar args on stack.
        unsafe {
            result::launch_kernel(upd_func, (upd_grid, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(), &mut upd_args)
                .or_panic("CUDA launch batch_norm_update_running");
        }
        running_mean.invalidate_cache();
        running_var.invalidate_cache();
        (ms.buf.ptr, vs.buf.ptr)
    } else {
        mean_s = None;
        var_s = None;
        (running_mean.buf.ptr, running_var.buf.ptr)
    };

    // Fwd kernel: eps is typed scalar between ptrs and (total, c) u32 args.
    let mut fwd_args = vec![
        dp!(a.buf.ptr), dp!(gamma.buf.ptr), dp!(beta.buf.ptr),
        dp!(mean_ptr), dp!(var_ptr), dp!(out_buf.ptr),
    ];
    let mut _eps_f32 = 0.0f32;
    let mut _eps_f64 = 0.0f64;
    push_typed_scalar!(fwd_args, eps, _eps_f32, _eps_f64);
    fwd_args.push(up!(total_u32));
    fwd_args.push(up!(c_u32));
    // SAFETY: all pointers are valid GPU buffers; scalar args on stack.
    unsafe {
        result::launch_kernel(fwd_func, (fwd_grid, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(), &mut fwd_args)
            .or_panic("CUDA launch batch_norm_fwd");
    }
    drop((mean_s, var_s));
    CudaStorage::new(n, c, out_buf)
}

pub(super) fn cuda_cross_entropy_fused<T: Scalar>(
    input: &CudaStorage<T>, target: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let (n, c) = (input.nrows, input.ncols);
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, "cross_entropy", type_suffix::<T>())), "CUDA kernel lookup");
    let loss_buf = alloc_out::<T>(ctx, n);
    let (n_u32, c_u32) = (n as u32, c as u32);
    // SAFETY: all pointers are valid GPU buffers.
    unsafe {
        result::launch_kernel(
            func, ((n_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
            0, ctx.stream.cu_stream(),
            &mut [dp!(input.buf.ptr), dp!(target.buf.ptr), dp!(loss_buf.ptr), up!(n_u32), up!(c_u32)],
        ).or_panic("CUDA launch cross_entropy");
    }
    let loss_s = CudaStorage::new(n, 1, loss_buf);
    let mut sum_s = cuda_sum_all_1x1(&loss_s);
    cuda_scale_inplace(&mut sum_s, T::from_f64(1.0 / n as f64));
    sum_s
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_sdpa<T: Scalar>(
    q: &CudaStorage<T>, k: &CudaStorage<T>, v: &CudaStorage<T>,
    seq_q: usize, seq_k: usize, head_dim: usize, batch_heads: usize,
) -> CudaStorage<T> {
    const FA_BLOCK_M: u32 = 64;
    const FA_BLOCK_N: u32 = 64;
    let ctx = get_ctx();
    let out_n = batch_heads * seq_q * head_dim;
    let out_buf = alloc_out::<T>(ctx, out_n);
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, "sdpa", type_suffix::<T>())), "CUDA kernel lookup");
    let grid = batch_heads as u32 * seq_q.div_ceil(FA_BLOCK_M as usize) as u32;
    let smem_elem = if type_suffix::<T>() == "f64" { 8usize } else { 4 };
    let smem = (2 * FA_BLOCK_N as usize * head_dim * smem_elem) as u32;
    let (sq, sk, hd, bh) = (seq_q as u32, seq_k as u32, head_dim as u32, batch_heads as u32);
    let scale = T::from_f64(1.0 / (head_dim as f64).sqrt());
    let mut args = vec![dp!(q.buf.ptr), dp!(k.buf.ptr), dp!(v.buf.ptr), dp!(out_buf.ptr), up!(sq), up!(sk), up!(hd), up!(bh)];
    // SAFETY: all pointers are valid GPU buffers; scale is typed scalar last arg.
    unsafe { launch_with_typed_last::<T>(func, (grid, 1, 1), (FA_BLOCK_M, 1, 1), smem, ctx.stream.cu_stream(), &mut args, scale); }
    CudaStorage::new(batch_heads * seq_q, head_dim, out_buf)
}

pub(super) fn cuda_embedding<T: Scalar>(
    indices: &CudaStorage<T>, weight: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = weight.ncols;
    let total = n_tokens * embed_dim;
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, "embedding", type_suffix::<T>())), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, total);
    let (nt, ed) = (n_tokens as u32, embed_dim as u32);
    // SAFETY: all pointers are valid GPU buffers.
    unsafe {
        result::launch_kernel(
            func, ((total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
            0, ctx.stream.cu_stream(),
            &mut [dp!(indices.buf.ptr), dp!(weight.buf.ptr), dp!(out_buf.ptr), up!(nt), up!(ed)],
        ).or_panic("CUDA launch embedding");
    }
    CudaStorage::new(n_tokens, embed_dim, out_buf)
}

pub(super) fn cuda_embedding_backward<T: Scalar>(
    indices: &CudaStorage<T>, grad: &CudaStorage<T>, vocab: usize,
) -> CudaStorage<T> {
    let tsuf = type_suffix::<T>();
    let type_name = super::indexing_ops::cuda_type_name::<T>();
    let kernel_name = format!("k_embedding_bwd_{tsuf}");
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* indices, const {type_name}* grad, {type_name}* out, int n_tokens, int embed_dim) {{\n\
            int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n\
            int total = n_tokens * embed_dim;\n\
            if (i >= total) return;\n\
            int row = i / embed_dim;\n\
            int col = i - row * embed_dim;\n\
            int idx = (int)indices[row];\n\
            atomicAdd(&out[idx * embed_dim + col], grad[i]);\n\
        }}\n"
    );
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = grad.ncols;
    let total = n_tokens * embed_dim;
    let out = cuda_zeros(vocab, embed_dim);
    let (nt_i32, ed_i32) = (n_tokens as i32, embed_dim as i32);
    let mut args: Vec<*mut c_void> = vec![
        dp!(indices.buf.ptr), dp!(grad.buf.ptr), dp!(out.buf.ptr),
        &nt_i32 as *const i32 as *mut c_void,
        &ed_i32 as *const i32 as *mut c_void,
    ];
    cuda_launch_kernel_src(&kernel_name, &src, (grid_1d(total), 1, 1), (BLOCK_SIZE, 1, 1), 0, &mut args);
    out
}

fn cuda_wht_impl<T: Scalar>(a: &CudaStorage<T>, inverse: bool) -> CudaStorage<T> {
    let ctx = get_ctx();
    let (rows, cols) = (a.nrows, a.ncols);
    let op = if inverse { "wht_inverse" } else { "wht" };
    let mut nbuf = [0u8; 64];
    let func = expect_ok(get_kernel(ctx, kernel_name_buf(&mut nbuf, op, type_suffix::<T>())), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, rows * cols);
    let (r, c) = (rows as u32, cols as u32);
    let block = (cols as u32).min(BLOCK_SIZE);
    // bf16 computes in f32 shared memory
    let smem = if type_suffix::<T>() == "bf16" { cols * 4 } else { cols * std::mem::size_of::<T>() };
    // SAFETY: launching pre-compiled kernel with correct args.
    unsafe {
        result::launch_kernel(
            func, (rows as u32, 1, 1), (block, 1, 1), smem as u32, ctx.stream.cu_stream(),
            &mut [dp!(a.buf.ptr), dp!(out_buf.ptr), up!(r), up!(c)],
        ).or_panic("CUDA launch wht");
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_wht<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    cuda_wht_impl(a, false)
}

pub(super) fn cuda_wht_inverse<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    cuda_wht_impl(a, true)
}
