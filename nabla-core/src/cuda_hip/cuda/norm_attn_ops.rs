use std::ffi::c_void;

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

pub(super) fn cuda_layer_norm<T: Scalar>(
    a: &CudaStorage<T>,
    gamma: &CudaStorage<T>,
    beta: &CudaStorage<T>,
    eps: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let name = format!("k_layer_norm_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let eps_f = eps.to_f64();
    unsafe {
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            result::launch_kernel(
                func,
                (rows as u32, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u32 as *const u32 as *mut c_void,
                    &cols_u32 as *const u32 as *mut c_void,
                    &eps_val as *const f32 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        } else {
            result::launch_kernel(
                func,
                (rows as u32, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u32 as *const u32 as *mut c_void,
                    &cols_u32 as *const u32 as *mut c_void,
                    &eps_f as *const f64 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        }
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_rms_norm<T: Scalar>(
    a: &CudaStorage<T>,
    gamma: &CudaStorage<T>,
    eps: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let name = format!("k_rms_norm_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let eps_f = eps.to_f64();
    unsafe {
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            result::launch_kernel(
                func,
                (rows as u32, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u32 as *const u32 as *mut c_void,
                    &cols_u32 as *const u32 as *mut c_void,
                    &eps_val as *const f32 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        } else {
            result::launch_kernel(
                func,
                (rows as u32, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u32 as *const u32 as *mut c_void,
                    &cols_u32 as *const u32 as *mut c_void,
                    &eps_f as *const f64 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        }
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_group_norm<T: Scalar>(
    a: &CudaStorage<T>,
    gamma: &CudaStorage<T>,
    beta: &CudaStorage<T>,
    groups: usize,
    eps: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let name = format!("k_group_norm_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let groups_u32 = groups as u32;
    let eps_f = eps.to_f64();
    let grid_x = (rows * groups) as u32;
    unsafe {
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            result::launch_kernel(
                func,
                (grid_x, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u32 as *const u32 as *mut c_void,
                    &cols_u32 as *const u32 as *mut c_void,
                    &groups_u32 as *const u32 as *mut c_void,
                    &eps_val as *const f32 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        } else {
            result::launch_kernel(
                func,
                (grid_x, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u32 as *const u32 as *mut c_void,
                    &cols_u32 as *const u32 as *mut c_void,
                    &groups_u32 as *const u32 as *mut c_void,
                    &eps_f as *const f64 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        }
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_batch_norm_train<T: Scalar>(
    a: &CudaStorage<T>,
    gamma: &CudaStorage<T>,
    beta: &CudaStorage<T>,
    running_mean: &mut CudaStorage<T>,
    running_var: &mut CudaStorage<T>,
    eps: T,
    momentum: T,
    training: bool,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.nrows;
    let c = a.ncols;
    let total = n * c;
    let sz = std::mem::size_of::<T>();
    let eps_f = eps.to_f64();
    let total_u32 = total as u32;
    let c_u32 = c as u32;
    let fwd_name = format!("k_batch_norm_fwd_{}", type_suffix::<T>());
    let fwd_func = expect_ok(get_kernel(ctx, &fwd_name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, total * sz).or_panic("CUDA alloc batch_norm out");
    let fwd_grid = (total_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if training {
        let stats_name = format!("k_batch_norm_stats_{}", type_suffix::<T>());
        let stats_func = expect_ok(get_kernel(ctx, &stats_name), "CUDA kernel lookup");
        let mean_buf =
            CuBuffer::alloc_async(&ctx.stream, c * sz).or_panic("CUDA alloc batch_norm mean");
        let var_buf =
            CuBuffer::alloc_async(&ctx.stream, c * sz).or_panic("CUDA alloc batch_norm var");
        let stats_grid = (c_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n_u32 = n as u32;
        unsafe {
            result::launch_kernel(
                stats_func,
                (stats_grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &mean_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &var_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                    &c_u32 as *const u32 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {stats_name}");
        }
        let mean_s = CudaStorage::new(1, c, mean_buf);
        let var_s = CudaStorage::new(1, c, var_buf);
        let one_minus = T::from_f64(1.0) - momentum;
        for i in 0..c {
            let m = cuda_get(&mean_s, 0, i);
            let v = cuda_get(&var_s, 0, i);
            let rm = cuda_get(running_mean, 0, i);
            let rv = cuda_get(running_var, 0, i);
            cuda_set(running_mean, 0, i, one_minus * rm + momentum * m);
            cuda_set(running_var, 0, i, one_minus * rv + momentum * v);
        }
        unsafe {
            if type_suffix::<T>() == "f32" {
                let eps_val = eps_f as f32;
                result::launch_kernel(
                    fwd_func,
                    (fwd_grid, 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &mean_s.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &var_s.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &eps_val as *const f32 as *mut c_void,
                        &total_u32 as *const u32 as *mut c_void,
                        &c_u32 as *const u32 as *mut c_void,
                    ],
                )
                .or_panic("CUDA launch {fwd_name}");
            } else {
                result::launch_kernel(
                    fwd_func,
                    (fwd_grid, 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &mean_s.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &var_s.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &eps_f as *const f64 as *mut c_void,
                        &total_u32 as *const u32 as *mut c_void,
                        &c_u32 as *const u32 as *mut c_void,
                    ],
                )
                .or_panic("CUDA launch {fwd_name}");
            }
        }
    } else {
        unsafe {
            if type_suffix::<T>() == "f32" {
                let eps_val = eps_f as f32;
                result::launch_kernel(
                    fwd_func,
                    (fwd_grid, 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &running_mean.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &running_var.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &eps_val as *const f32 as *mut c_void,
                        &total_u32 as *const u32 as *mut c_void,
                        &c_u32 as *const u32 as *mut c_void,
                    ],
                )
                .or_panic("CUDA launch {fwd_name}");
            } else {
                result::launch_kernel(
                    fwd_func,
                    (fwd_grid, 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &gamma.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &beta.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &running_mean.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &running_var.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &eps_f as *const f64 as *mut c_void,
                        &total_u32 as *const u32 as *mut c_void,
                        &c_u32 as *const u32 as *mut c_void,
                    ],
                )
                .or_panic("CUDA launch {fwd_name}");
            }
        }
    }
    CudaStorage::new(n, c, out_buf)
}

pub(super) fn cuda_cross_entropy_fused<T: Scalar>(
    input: &CudaStorage<T>,
    target: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = input.nrows;
    let c = input.ncols;
    let sz = std::mem::size_of::<T>();
    let name = format!("k_cross_entropy_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let loss_buf =
        CuBuffer::alloc_async(&ctx.stream, n * sz).or_panic("CUDA alloc cross_entropy loss");
    let n_u32 = n as u32;
    let c_u32 = c as u32;
    let grid = (n_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &input.buf.ptr as *const CUdeviceptr as *mut c_void,
                &target.buf.ptr as *const CUdeviceptr as *mut c_void,
                &loss_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
                &c_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    let loss_s = CudaStorage::new(n, 1, loss_buf);
    let total = (0..n).fold(T::zero(), |acc, i| acc + cuda_get(&loss_s, i, 0));
    let mean = total / T::from_f64(n as f64);
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, sz).or_panic("CUDA alloc cross_entropy result");
    let mut out_s = CudaStorage::new(1, 1, out_buf);
    cuda_set(&mut out_s, 0, 0, mean);
    out_s
}

#[allow(clippy::too_many_arguments)]

pub(super) fn cuda_sdpa<T: Scalar>(
    q: &CudaStorage<T>,
    k: &CudaStorage<T>,
    v: &CudaStorage<T>,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    batch_heads: usize,
) -> CudaStorage<T> {
    const FA_BLOCK_M: u32 = 64;
    const FA_BLOCK_N: u32 = 64;
    let ctx = get_ctx();
    let sz = std::mem::size_of::<T>();
    let out_n = batch_heads * seq_q * head_dim;
    let out_buf = CuBuffer::alloc_async(&ctx.stream, out_n * sz).or_panic("CUDA alloc sdpa out");
    let name = format!("k_sdpa_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let num_q_tiles = seq_q.div_ceil(FA_BLOCK_M as usize) as u32;
    let grid = batch_heads as u32 * num_q_tiles;
    let smem = if type_suffix::<T>() == "f64" {
        2 * FA_BLOCK_N as usize * head_dim * std::mem::size_of::<f64>()
    } else {
        2 * FA_BLOCK_N as usize * head_dim * std::mem::size_of::<f32>()
    };
    let seq_q_u = seq_q as u32;
    let seq_k_u = seq_k as u32;
    let head_dim_u = head_dim as u32;
    let bh_u = batch_heads as u32;
    let scale_f64 = 1.0 / (head_dim as f64).sqrt();
    unsafe {
        if type_suffix::<T>() != "f64" {
            let scale = scale_f64 as f32;
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (FA_BLOCK_M, 1, 1),
                smem as u32,
                ctx.stream.cu_stream(),
                &mut [
                    &q.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &k.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &v.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &seq_q_u as *const u32 as *mut c_void,
                    &seq_k_u as *const u32 as *mut c_void,
                    &head_dim_u as *const u32 as *mut c_void,
                    &bh_u as *const u32 as *mut c_void,
                    &scale as *const f32 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        } else {
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (FA_BLOCK_M, 1, 1),
                smem as u32,
                ctx.stream.cu_stream(),
                &mut [
                    &q.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &k.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &v.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &seq_q_u as *const u32 as *mut c_void,
                    &seq_k_u as *const u32 as *mut c_void,
                    &head_dim_u as *const u32 as *mut c_void,
                    &bh_u as *const u32 as *mut c_void,
                    &scale_f64 as *const f64 as *mut c_void,
                ],
            )
            .or_panic("CUDA launch {name}");
        }
    }
    CudaStorage::new(batch_heads * seq_q, head_dim, out_buf)
}

pub(super) fn cuda_embedding<T: Scalar>(
    indices: &CudaStorage<T>,
    weight: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = weight.ncols;
    let total = n_tokens * embed_dim;
    let name = format!("k_embedding_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let n_tokens_u32 = n_tokens as u32;
    let embed_dim_u32 = embed_dim as u32;
    unsafe {
        result::launch_kernel(
            func,
            ((total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &indices.buf.ptr as *const CUdeviceptr as *mut c_void,
                &weight.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_tokens_u32 as *const u32 as *mut c_void,
                &embed_dim_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(n_tokens, embed_dim, out_buf)
}

pub(super) fn cuda_embedding_backward<T: Scalar>(
    indices: &CudaStorage<T>,
    grad: &CudaStorage<T>,
    vocab: usize,
) -> CudaStorage<T> {
    let tsuf = type_suffix::<T>();
    let type_name = if tsuf == "f32" {
        "float"
    } else if tsuf == "f64" {
        "double"
    } else {
        panic!("cuda_embedding_backward: only f32/f64 supported");
    };
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
    let out = cuda_zeros(vocab, embed_dim);
    let n_tokens_i32 = n_tokens as i32;
    let embed_dim_i32 = embed_dim as i32;
    let total = n_tokens * embed_dim;
    let mut args: Vec<*mut c_void> = vec![
        &indices.buf.ptr as *const CUdeviceptr as *mut c_void,
        &grad.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out.buf.ptr as *const CUdeviceptr as *mut c_void,
        &n_tokens_i32 as *const i32 as *mut c_void,
        &embed_dim_i32 as *const i32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(total), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    out
}
