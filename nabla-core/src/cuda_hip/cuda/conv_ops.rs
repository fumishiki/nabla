use std::ffi::c_void;

use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::driver::result;
use cudarc::driver::sys::{CUdeviceptr, CUfunction};

use crate::gpu_common;
use crate::gpu_common::type_suffix;
use crate::kernels_cu::{BLOCK_SIZE, REDUCE_BLOCK, REDUCE_GRID_CAP};
use crate::scalar::Scalar;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_im2col<T: Scalar>(
    input: &CudaStorage<T>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    out_h: usize,
    out_w: usize,
) -> CudaStorage<T> {
    let k_cols = c_in * kh * kw;
    let out_spatial = out_h * out_w;
    let ctx = get_ctx();
    let name = format!("k_im2col_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let col_buf = CuBuffer::alloc_async(
        &ctx.stream,
        n * k_cols * out_spatial * std::mem::size_of::<T>(),
    )
    .or_panic("CUDA alloc im2col");
    let col_elem = (k_cols * out_spatial) as u32;
    let grid_x = (col_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, dh_u, dw_u, out_h_u, out_w_u) = (
        c_in as u32,
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        dh as u32,
        dw as u32,
        out_h as u32,
        out_w as u32,
    );
    // SAFETY: all pointers are valid GPU buffers; scalar args are stack values.
    unsafe {
        result::launch_kernel(
            func,
            (grid_x, n as u32, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &input.buf.ptr as *const CUdeviceptr as *mut c_void,
                &col_buf.ptr as *const CUdeviceptr as *mut c_void,
                &c_in_u as *const u32 as *mut c_void,
                &h_u as *const u32 as *mut c_void,
                &w_u as *const u32 as *mut c_void,
                &kh_u as *const u32 as *mut c_void,
                &kw_u as *const u32 as *mut c_void,
                &sh_u as *const u32 as *mut c_void,
                &sw_u as *const u32 as *mut c_void,
                &ph_u as *const u32 as *mut c_void,
                &pw_u as *const u32 as *mut c_void,
                &dh_u as *const u32 as *mut c_void,
                &dw_u as *const u32 as *mut c_void,
                &out_h_u as *const u32 as *mut c_void,
                &out_w_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(n * k_cols, out_spatial, col_buf)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_conv2d<T: Scalar>(
    input: &CudaStorage<T>,
    weight: &CudaStorage<T>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> CudaStorage<T> {
    assert!(
        groups == 1,
        "GPU conv2d: groups > 1 not supported; use CPU backend"
    );
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp8E4M3>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp8E5M2>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp4E2M1>()
    {
        let in_f16 = cuda_cast::<T, half::f16>(input);
        let w_f16 = cuda_cast::<T, half::f16>(weight);
        let out_f16 = cuda_conv2d::<half::f16>(
            &in_f16, &w_f16, n, c_in, h, w, c_out, kh, kw, stride, padding, dilation, groups,
        );
        return cuda_cast::<half::f16, T>(&out_f16);
    }
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let out_spatial = out_h * out_w;
    let k_cols = c_in * kh * kw;

    let col = cuda_im2col(
        input, n, c_in, h, w, kh, kw, sh, sw, ph, pw, dh, dw, out_h, out_w,
    );

    let ctx = get_ctx();
    let out_buf = CuBuffer::alloc_async(
        &ctx.stream,
        n * c_out * out_spatial * std::mem::size_of::<T>(),
    )
    .or_panic("CUDA alloc conv2d out");
    let mut out = CudaStorage::new(n * c_out, out_spatial, out_buf);

    // SAFETY: pointers are valid GPU buffers; alpha/beta are stack scalars copied by cuBLAS.
    unsafe {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha_f = 1.0f32;
            let beta_f = 0.0f32;
            cublas_result::sgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_spatial as i32, // cuBLAS m = our n (out_spatial)
                c_out as i32,       // cuBLAS n = our m (C_out)
                k_cols as i32,      // k
                &alpha_f,
                col.buf.ptr as *const f32,
                out_spatial as i32,            // ldb = ncols of col
                (k_cols * out_spatial) as i64, // stride_B per batch
                weight.buf.ptr as *const f32,
                k_cols as i32, // lda = ncols of weight
                0_i64,         // stride_A = 0: broadcast weight
                &beta_f,
                out.buf.ptr as *mut f32,
                out_spatial as i32,           // ldc = ncols of out
                (c_out * out_spatial) as i64, // stride_C per batch
                n as i32,
            )
            .or_panic("cuBLAS sgemm_strided_batched conv2d");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let alpha_d = 1.0f64;
            let beta_d = 0.0f64;
            cublas_result::dgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_spatial as i32,
                c_out as i32,
                k_cols as i32,
                &alpha_d,
                col.buf.ptr as *const f64,
                out_spatial as i32,
                (k_cols * out_spatial) as i64,
                weight.buf.ptr as *const f64,
                k_cols as i32,
                0_i64,
                &beta_d,
                out.buf.ptr as *mut f64,
                out_spatial as i32,
                (c_out * out_spatial) as i64,
                n as i32,
            )
            .or_panic("cuBLAS dgemm_strided_batched conv2d");
        } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            let alpha_f = 1.0f32;
            let beta_f = 0.0f32;
            let status = cublas_sys::cublasGemmStridedBatchedEx(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_spatial as i32,
                c_out as i32,
                k_cols as i32,
                &alpha_f as *const f32 as *const std::ffi::c_void,
                col.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                out_spatial as i32,
                (k_cols * out_spatial) as i64,
                weight.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k_cols as i32,
                0_i64,
                &beta_f as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                out_spatial as i32,
                (c_out * out_spatial) as i64,
                n as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                panic!("cuBLAS gemm_strided_batched_ex conv2d f16: {status:?}");
            }
        } else {
            panic!("GPU conv2d: unsupported scalar type (f32/f64/f16 only)");
        }
    }
    out.invalidate_cache();
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_conv_transpose2d<T: Scalar>(
    input: &CudaStorage<T>,
    weight: &CudaStorage<T>,
    n_batch: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> CudaStorage<T> {
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let out_h = (h - 1) * sh + kh - 2 * ph + oph;
    let out_w = (w - 1) * sw + kw - 2 * pw + opw;
    let total = n_batch * c_out * out_h * out_w;
    let ctx = get_ctx();
    let name = format!("k_conv_transpose2d_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>())
        .or_panic("CUDA alloc conv_transpose2d");
    let (n_u, c_in_u, h_u, w_u, c_out_u, kh_u, kw_u, out_h_u, out_w_u, sh_u, sw_u, ph_u, pw_u) = (
        n_batch as i32,
        c_in as i32,
        h as i32,
        w as i32,
        c_out as i32,
        kh as i32,
        kw as i32,
        out_h as i32,
        out_w as i32,
        sh as i32,
        sw as i32,
        ph as i32,
        pw as i32,
    );
    // SAFETY: all pointers are valid GPU buffers; scalar args are stack values.
    unsafe {
        result::launch_kernel(
            func,
            ((total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &input.buf.ptr as *const CUdeviceptr as *mut c_void,
                &weight.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u as *const i32 as *mut c_void,
                &c_in_u as *const i32 as *mut c_void,
                &h_u as *const i32 as *mut c_void,
                &w_u as *const i32 as *mut c_void,
                &c_out_u as *const i32 as *mut c_void,
                &kh_u as *const i32 as *mut c_void,
                &kw_u as *const i32 as *mut c_void,
                &out_h_u as *const i32 as *mut c_void,
                &out_w_u as *const i32 as *mut c_void,
                &sh_u as *const i32 as *mut c_void,
                &sw_u as *const i32 as *mut c_void,
                &ph_u as *const i32 as *mut c_void,
                &pw_u as *const i32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(n_batch * c_out, out_h * out_w, out_buf)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_im1col<T: Scalar>(
    input: &CudaStorage<T>,
    n: usize,
    c_in: usize,
    l: usize,
    kl: usize,
    sl: usize,
    pl: usize,
    dl: usize,
    out_l: usize,
) -> CudaStorage<T> {
    let k_cols = c_in * kl;
    let ctx = get_ctx();
    let name = format!("k_im1col_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let col_buf = CuBuffer::alloc_async(&ctx.stream, n * k_cols * out_l * std::mem::size_of::<T>())
        .or_panic("CUDA alloc im1col");
    let col_elem = (k_cols * out_l) as u32;
    let grid_x = (col_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, l_u, kl_u, sl_u, pl_u, dl_u, out_l_u) = (
        c_in as u32,
        l as u32,
        kl as u32,
        sl as u32,
        pl as u32,
        dl as u32,
        out_l as u32,
    );
    // SAFETY: all pointers are valid GPU buffers; scalar args are stack values.
    unsafe {
        result::launch_kernel(
            func,
            (grid_x, n as u32, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &input.buf.ptr as *const CUdeviceptr as *mut c_void,
                &col_buf.ptr as *const CUdeviceptr as *mut c_void,
                &c_in_u as *const u32 as *mut c_void,
                &l_u as *const u32 as *mut c_void,
                &kl_u as *const u32 as *mut c_void,
                &sl_u as *const u32 as *mut c_void,
                &pl_u as *const u32 as *mut c_void,
                &dl_u as *const u32 as *mut c_void,
                &out_l_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(n * k_cols, out_l, col_buf)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_im3col<T: Scalar>(
    input: &CudaStorage<T>,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    out_d: usize,
    out_h: usize,
    out_w: usize,
) -> CudaStorage<T> {
    let k_vol = c_in * kd * kh * kw;
    let out_vol = out_d * out_h * out_w;
    let ctx = get_ctx();
    let name = format!("k_im3col_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let col_buf =
        CuBuffer::alloc_async(&ctx.stream, n * k_vol * out_vol * std::mem::size_of::<T>())
            .or_panic("CUDA alloc im3col");
    let col_elem = (k_vol * out_vol) as u32;
    let grid_x = (col_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, d_u, h_u, w_u) = (c_in as u32, d as u32, h as u32, w as u32);
    let (kd_u, kh_u, kw_u) = (kd as u32, kh as u32, kw as u32);
    let (sd_u, sh_u, sw_u) = (sd as u32, sh as u32, sw as u32);
    let (pd_u, ph_u, pw_u) = (pd as u32, ph as u32, pw as u32);
    let (dd_u, dh_u, dw_u) = (dd as u32, dh as u32, dw as u32);
    let (out_d_u, out_h_u, out_w_u) = (out_d as u32, out_h as u32, out_w as u32);
    // SAFETY: all pointers are valid GPU buffers; scalar args are stack values.
    unsafe {
        result::launch_kernel(
            func,
            (grid_x, n as u32, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &input.buf.ptr as *const CUdeviceptr as *mut c_void,
                &col_buf.ptr as *const CUdeviceptr as *mut c_void,
                &c_in_u as *const u32 as *mut c_void,
                &d_u as *const u32 as *mut c_void,
                &h_u as *const u32 as *mut c_void,
                &w_u as *const u32 as *mut c_void,
                &kd_u as *const u32 as *mut c_void,
                &kh_u as *const u32 as *mut c_void,
                &kw_u as *const u32 as *mut c_void,
                &sd_u as *const u32 as *mut c_void,
                &sh_u as *const u32 as *mut c_void,
                &sw_u as *const u32 as *mut c_void,
                &pd_u as *const u32 as *mut c_void,
                &ph_u as *const u32 as *mut c_void,
                &pw_u as *const u32 as *mut c_void,
                &dd_u as *const u32 as *mut c_void,
                &dh_u as *const u32 as *mut c_void,
                &dw_u as *const u32 as *mut c_void,
                &out_d_u as *const u32 as *mut c_void,
                &out_h_u as *const u32 as *mut c_void,
                &out_w_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(n * k_vol, out_vol, col_buf)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_conv1d<T: Scalar>(
    input: &CudaStorage<T>,
    weight: &CudaStorage<T>,
    n: usize,
    c_in: usize,
    l: usize,
    c_out: usize,
    kl: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> CudaStorage<T> {
    assert!(
        groups == 1,
        "GPU conv1d: groups > 1 not supported; use CPU backend"
    );
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp8E4M3>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp8E5M2>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp4E2M1>()
    {
        let in_f16 = cuda_cast::<T, half::f16>(input);
        let w_f16 = cuda_cast::<T, half::f16>(weight);
        let out_f16 = cuda_conv1d::<half::f16>(
            &in_f16, &w_f16, n, c_in, l, c_out, kl, stride, padding, dilation, groups,
        );
        return cuda_cast::<half::f16, T>(&out_f16);
    }
    let out_l = (l + 2 * padding - dilation * (kl - 1) - 1) / stride + 1;
    let k_cols = c_in * kl;

    let col = cuda_im1col(input, n, c_in, l, kl, stride, padding, dilation, out_l);

    let ctx = get_ctx();
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * c_out * out_l * std::mem::size_of::<T>())
        .or_panic("CUDA alloc conv1d out");
    let mut out = CudaStorage::new(n * c_out, out_l, out_buf);
    // SAFETY: pointers are valid GPU buffers; alpha/beta are stack scalars copied by cuBLAS.
    unsafe {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha_f = 1.0f32;
            let beta_f = 0.0f32;
            cublas_result::sgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_l as i32,
                c_out as i32,
                k_cols as i32,
                &alpha_f,
                col.buf.ptr as *const f32,
                out_l as i32,
                (k_cols * out_l) as i64,
                weight.buf.ptr as *const f32,
                k_cols as i32,
                0_i64,
                &beta_f,
                out.buf.ptr as *mut f32,
                out_l as i32,
                (c_out * out_l) as i64,
                n as i32,
            )
            .or_panic("cuBLAS sgemm_strided_batched conv1d");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let alpha_d = 1.0f64;
            let beta_d = 0.0f64;
            cublas_result::dgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_l as i32,
                c_out as i32,
                k_cols as i32,
                &alpha_d,
                col.buf.ptr as *const f64,
                out_l as i32,
                (k_cols * out_l) as i64,
                weight.buf.ptr as *const f64,
                k_cols as i32,
                0_i64,
                &beta_d,
                out.buf.ptr as *mut f64,
                out_l as i32,
                (c_out * out_l) as i64,
                n as i32,
            )
            .or_panic("cuBLAS dgemm_strided_batched conv1d");
        } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            let alpha_f = 1.0f32;
            let beta_f = 0.0f32;
            let status = cublas_sys::cublasGemmStridedBatchedEx(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_l as i32,
                c_out as i32,
                k_cols as i32,
                &alpha_f as *const f32 as *const std::ffi::c_void,
                col.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                out_l as i32,
                (k_cols * out_l) as i64,
                weight.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k_cols as i32,
                0_i64,
                &beta_f as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                out_l as i32,
                (c_out * out_l) as i64,
                n as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                panic!("cuBLAS gemm_strided_batched_ex conv1d f16: {status:?}");
            }
        } else {
            panic!("GPU conv1d: unsupported scalar type (f32/f64/f16 only)");
        }
    }
    out.invalidate_cache();
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_conv3d<T: Scalar>(
    input: &CudaStorage<T>,
    weight: &CudaStorage<T>,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> CudaStorage<T> {
    assert!(
        groups == 1,
        "GPU conv3d: groups > 1 not supported; use CPU backend"
    );
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp8E4M3>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp8E5M2>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::scalar::Fp4E2M1>()
    {
        let in_f16 = cuda_cast::<T, half::f16>(input);
        let w_f16 = cuda_cast::<T, half::f16>(weight);
        let out_f16 = cuda_conv3d::<half::f16>(
            &in_f16, &w_f16, n, c_in, d, h, w, c_out, kd, kh, kw, stride, padding, dilation,
            groups,
        );
        return cuda_cast::<half::f16, T>(&out_f16);
    }
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;
    let (dd, dh, dw) = dilation;
    let out_d = (d + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let out_vol = out_d * out_h * out_w;
    let k_vol = c_in * kd * kh * kw;

    let col = cuda_im3col(
        input, n, c_in, d, h, w, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, out_d, out_h,
        out_w,
    );

    let ctx = get_ctx();
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * c_out * out_vol * std::mem::size_of::<T>())
            .or_panic("CUDA alloc conv3d out");
    let mut out = CudaStorage::new(n * c_out, out_vol, out_buf);
    // SAFETY: pointers are valid GPU buffers; alpha/beta are stack scalars copied by cuBLAS.
    unsafe {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha_f = 1.0f32;
            let beta_f = 0.0f32;
            cublas_result::sgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_vol as i32,
                c_out as i32,
                k_vol as i32,
                &alpha_f,
                col.buf.ptr as *const f32,
                out_vol as i32,
                (k_vol * out_vol) as i64,
                weight.buf.ptr as *const f32,
                k_vol as i32,
                0_i64,
                &beta_f,
                out.buf.ptr as *mut f32,
                out_vol as i32,
                (c_out * out_vol) as i64,
                n as i32,
            )
            .or_panic("cuBLAS sgemm_strided_batched conv3d");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let alpha_d = 1.0f64;
            let beta_d = 0.0f64;
            cublas_result::dgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_vol as i32,
                c_out as i32,
                k_vol as i32,
                &alpha_d,
                col.buf.ptr as *const f64,
                out_vol as i32,
                (k_vol * out_vol) as i64,
                weight.buf.ptr as *const f64,
                k_vol as i32,
                0_i64,
                &beta_d,
                out.buf.ptr as *mut f64,
                out_vol as i32,
                (c_out * out_vol) as i64,
                n as i32,
            )
            .or_panic("cuBLAS dgemm_strided_batched conv3d");
        } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            let alpha_f = 1.0f32;
            let beta_f = 0.0f32;
            let status = cublas_sys::cublasGemmStridedBatchedEx(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                out_vol as i32,
                c_out as i32,
                k_vol as i32,
                &alpha_f as *const f32 as *const std::ffi::c_void,
                col.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                out_vol as i32,
                (k_vol * out_vol) as i64,
                weight.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k_vol as i32,
                0_i64,
                &beta_f as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                out_vol as i32,
                (c_out * out_vol) as i64,
                n as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                panic!("cuBLAS gemm_strided_batched_ex conv3d f16: {status:?}");
            }
        } else {
            panic!("GPU conv3d: unsupported scalar type (f32/f64/f16 only)");
        }
    }
    out.invalidate_cache();
    out
}

pub(super) fn reduce_func_idx<T: Scalar>(base: usize) -> usize {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        base
    } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
        base + 3
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        base + 6
    } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E4M3>() {
        base + 9
    } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E5M2>() {
        base + 12
    } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp4E2M1>() {
        base + 15
    } else {
        panic!("CUDA reduction: unsupported scalar type");
    }
}

#[inline]
unsafe fn launch_reduce(
    func: CUfunction,
    grid: u32,
    block: u32,
    stream: cudarc::driver::sys::CUstream,
    args: &mut [*mut c_void],
) {
    // SAFETY: caller guarantees func, stream, and args are valid CUDA objects.
    unsafe {
        cudarc::driver::sys::cuLaunchKernel(
            func,
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }
}

#[inline]
unsafe fn reduce_readback<T: Scalar>(ctx: &CudaCtx) -> T {
    // SAFETY: caller guarantees ctx.stream and reduce_host_ptr are valid.
    unsafe {
        cudarc::driver::sys::cuStreamSynchronize(ctx.stream.cu_stream());
        std::ptr::read_volatile(ctx.reduce_host_ptr.0 as *const T)
    }
}

pub fn cuda_synchronize() {
    let ctx = get_ctx();
    // SAFETY: cuStreamSynchronize is safe to call; stream is valid.
    unsafe {
        cudarc::driver::sys::cuStreamSynchronize(ctx.stream.cu_stream());
    }
}

pub(crate) fn cuda_sum_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    let ctx = get_ctx();
    let n = a.n();
    if n == 0 {
        return T::zero();
    }
    let func = ctx.reduce_funcs[reduce_func_idx::<T>(0)].0;
    let grid1 = REDUCE_GRID_CAP.min(((n as u32) + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
    let scratch = ctx.reduce_scratch;
    let out_dptr = ctx.reduce_host_dptr;

    let n_u32 = n as u32;
    unsafe {
        launch_reduce(
            func,
            grid1,
            REDUCE_BLOCK,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &scratch as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
                &out_dptr as *const CUdeviceptr as *mut c_void,
            ],
        );
        reduce_readback::<T>(ctx)
    }
}

pub(crate) fn cuda_max_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    cuda_reduce_minmax(a, 1) // func_base=1 → max
}
pub(crate) fn cuda_min_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    cuda_reduce_minmax(a, 2) // func_base=2 → min
}

pub(super) fn cuda_reduce_minmax<T: Scalar>(a: &CudaStorage<T>, func_base: usize) -> T {
    let ctx = get_ctx();
    let n = a.n();
    assert!(n > 0, "reduction on empty");
    let func = ctx.reduce_funcs[reduce_func_idx::<T>(func_base)].0;
    let grid1 = REDUCE_GRID_CAP.min(((n as u32) + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
    let scratch = ctx.reduce_scratch;
    let out_dptr = ctx.reduce_host_dptr;

    let n_u32 = n as u32;
    unsafe {
        launch_reduce(
            func,
            grid1,
            REDUCE_BLOCK,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &scratch as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
                &out_dptr as *const CUdeviceptr as *mut c_void,
            ],
        );
        reduce_readback::<T>(ctx)
    }
}
pub(crate) fn cuda_argmax_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) {
    gpu_common::rtc_argmax_all(a)
}
pub(crate) fn cuda_argmin_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) {
    gpu_common::rtc_argmin_all(a)
}
