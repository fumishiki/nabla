use std::ffi::c_void;
use std::sync::Mutex;

use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{RtcStorage, grid_1d, lock_or_recover, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

pub(crate) fn cuda_zeros<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
    let ctx = get_ctx();
    let buf = expect_ok(
        CuBuffer::alloc_zeros(&ctx.stream, nrows * ncols * std::mem::size_of::<T>()),
        "CUDA alloc",
    );
    CudaStorage::new(nrows, ncols, buf)
}

pub(crate) fn cuda_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let data = vec![val; n];
    let buf = expect_ok(CuBuffer::from_host(&ctx.stream, &data), "CUDA upload");
    CudaStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn cuda_from_fn<T: Scalar>(
    nrows: usize,
    ncols: usize,
    mut f: impl FnMut(usize, usize) -> T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let mut data = Vec::with_capacity(n);
    for r in 0..nrows {
        for c in 0..ncols {
            data.push(f(r, c));
        }
    }
    let buf = expect_ok(CuBuffer::from_host(&ctx.stream, &data), "CUDA upload");
    CudaStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn cuda_from_vec_async<T: Scalar>(
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let buf = expect_ok(
        CuBuffer::from_host_nonblocking(&ctx.stream, &ctx.copy_stream, &data),
        "CUDA async upload",
    );
    CudaStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn cuda_get<T: Scalar>(s: &CudaStorage<T>, r: usize, c: usize) -> T {
    {
        let guard = lock_or_recover(&s.host_cache);
        if let Some(cache) = guard.as_ref() {
            return cache[r * s.ncols + c];
        }
    }
    let ctx = get_ctx();
    let byte_offset = (r * s.ncols + c) * std::mem::size_of::<T>();
    expect_ok(
        s.buf.copy_element::<T>(&ctx.stream, byte_offset),
        "CUDA single-element readback",
    )
}

pub(crate) fn cuda_set<T: Scalar>(s: &mut CudaStorage<T>, r: usize, c: usize, v: T) {
    s.invalidate_cache();
    let ctx = get_ctx();
    let offset = (r * s.ncols + c) * std::mem::size_of::<T>();
    let src = std::slice::from_ref(&v);
    // SAFETY: uploading single element to correct offset in GPU buffer.
    unsafe {
        let _ = result::memcpy_htod_async(s.buf.ptr + offset as u64, src, ctx.stream.cu_stream());
    }
}
pub(crate) fn cuda_clone<T: Scalar>(s: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let bytes = s.n() * std::mem::size_of::<T>();
    let new_buf = expect_ok(CuBuffer::alloc_async(&ctx.stream, bytes), "CUDA alloc");
    if bytes > 0 {
        // SAFETY: device-to-device copy of same-sized buffers.
        unsafe {
            let _ =
                result::memcpy_dtod_async(new_buf.ptr, s.buf.ptr, bytes, ctx.stream.cu_stream());
        }
    }
    CudaStorage {
        nrows: s.nrows,
        ncols: s.ncols,
        buf: new_buf,
        host_cache: Mutex::new(None),
    }
}

pub(crate) fn cuda_transpose<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_transpose_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows as *const u32 as *mut c_void,
                    &cols as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch transpose",
        );
    }
    CudaStorage::new(a.ncols, a.nrows, out_buf)
}

pub(crate) fn cuda_scale<T: Scalar>(a: &CudaStorage<T>, s: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_scale_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    let grid = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &s as *const T as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch scale",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_cast<T: Scalar, U: Scalar>(a: &CudaStorage<T>) -> CudaStorage<U> {
    use std::any::TypeId;

    if TypeId::of::<T>() == TypeId::of::<U>() {
        let cloned = cuda_clone(a);
        // SAFETY: T == U, identical layout for CudaStorage.
        return unsafe { std::mem::transmute::<CudaStorage<T>, CudaStorage<U>>(cloned) };
    }

    fn cast_to_f32<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<f32> {
        use std::any::TypeId;
        let ctx = get_ctx();
        let n = a.n();
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let cloned = cuda_clone(a);
            // SAFETY: T is f32.
            return unsafe { std::mem::transmute::<CudaStorage<T>, CudaStorage<f32>>(cloned) };
        }
        let name = if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            "k_cast_f16_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            "k_cast_f64_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E4M3>() {
            "k_cast_fp8e4m3_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E5M2>() {
            "k_cast_fp8e5m2_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp4E2M1>() {
            "k_cast_fp4e2m1_to_f32"
        } else {
            panic!("CUDA cast: unsupported source type");
        };
        let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
        let out_buf = alloc_out::<f32>(ctx, n);
        let n_u32 = n as u32;
        unsafe {
            expect_ok(
                result::launch_kernel(
                    func,
                    (grid_1d(n), 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &n_u32 as *const u32 as *mut c_void,
                    ],
                ),
                "CUDA launch cast to f32",
            );
        }
        CudaStorage::new(a.nrows, a.ncols, out_buf)
    }

    fn cast_from_f32<U: Scalar>(a: &CudaStorage<f32>) -> CudaStorage<U> {
        use std::any::TypeId;
        let ctx = get_ctx();
        let n = a.n();
        if TypeId::of::<U>() == TypeId::of::<f32>() {
            let cloned = cuda_clone(a);
            // SAFETY: U is f32.
            return unsafe { std::mem::transmute::<CudaStorage<f32>, CudaStorage<U>>(cloned) };
        }
        let name = if TypeId::of::<U>() == TypeId::of::<half::f16>() {
            "k_cast_f32_to_f16"
        } else if TypeId::of::<U>() == TypeId::of::<f64>() {
            "k_cast_f32_to_f64"
        } else if TypeId::of::<U>() == TypeId::of::<crate::scalar::Fp8E4M3>() {
            "k_cast_f32_to_fp8e4m3"
        } else if TypeId::of::<U>() == TypeId::of::<crate::scalar::Fp8E5M2>() {
            "k_cast_f32_to_fp8e5m2"
        } else if TypeId::of::<U>() == TypeId::of::<crate::scalar::Fp4E2M1>() {
            "k_cast_f32_to_fp4e2m1"
        } else {
            panic!("CUDA cast: unsupported destination type");
        };
        let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
        let out_buf = alloc_out::<U>(ctx, n);
        let n_u32 = n as u32;
        unsafe {
            expect_ok(
                result::launch_kernel(
                    func,
                    (grid_1d(n), 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &n_u32 as *const u32 as *mut c_void,
                    ],
                ),
                "CUDA launch cast from f32",
            );
        }
        CudaStorage::new(a.nrows, a.ncols, out_buf)
    }

    let tmp = cast_to_f32(a);
    cast_from_f32::<U>(&tmp)
}

pub(crate) fn cuda_masked_fill<T: Scalar>(
    a: &CudaStorage<T>,
    mask: &CudaStorage<T>,
    value: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_masked_fill_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &mask.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &value as *const T as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch masked_fill",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_where<T: Scalar>(
    a: &CudaStorage<T>,
    cond: &CudaStorage<T>,
    b: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_where_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &cond.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch where",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_axpy_inplace<T: Scalar>(y: &mut CudaStorage<T>, alpha: T, x: &CudaStorage<T>) {
    let ctx = get_ctx();
    let n = y.n();
    let name = format!("k_axpy_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let n_u32 = n as u32;
    let grid = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };
    // SAFETY: kernel writes y in-place; x is read-only.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &y.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &alpha as *const T as *mut c_void,
                    &x.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch axpy",
        );
    }
    y.invalidate_cache();
}

pub(crate) fn cuda_powf<T: Scalar>(a: &CudaStorage<T>, p: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_powf_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    let grid = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &p as *const T as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch powf",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_expand<T: Scalar>(
    out: &mut CudaStorage<T>,
    src: &CudaStorage<T>,
    src_rows: usize,
    src_cols: usize,
) {
    let ctx = get_ctx();
    let dst_rows = out.nrows;
    let dst_cols = out.ncols;
    let n = dst_rows * dst_cols;
    let name = format!("k_expand_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let src_rows_u32 = src_rows as u32;
    let src_cols_u32 = src_cols as u32;
    let dst_rows_u32 = dst_rows as u32;
    let dst_cols_u32 = dst_cols as u32;
    // SAFETY: launching expand kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &out.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src_rows_u32 as *const u32 as *mut c_void,
                    &src_cols_u32 as *const u32 as *mut c_void,
                    &dst_rows_u32 as *const u32 as *mut c_void,
                    &dst_cols_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch expand",
        );
    }
    out.invalidate_cache();
}

pub(crate) fn cuda_has_wmma() -> bool {
    get_ctx().has_wmma
}

pub(super) fn cuda_matmul_tiled<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let name = format!("k_matmul_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let m = a.nrows as u32;
    let k = a.ncols as u32;
    let n = b.ncols as u32;
    let out_bytes = out.n() * std::mem::size_of::<T>();
    if out_bytes > 0 {
        // SAFETY: zeroing output buffer before matmul accumulation.
        unsafe {
            let _ = result::memset_d8_async(out.buf.ptr, 0, out_bytes, ctx.stream.cu_stream());
        }
    }
    let grid_x = n.div_ceil(16);
    let grid_y = m.div_ceil(16);
    // SAFETY: launching CUDA kernel with correct argument pointers.
    unsafe {
        result::launch_kernel(
            func,
            (grid_x, grid_y, 1),
            (16, 16, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out.buf.ptr as *const CUdeviceptr as *mut c_void,
                &m as *const u32 as *mut c_void,
                &k as *const u32 as *mut c_void,
                &n as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch matmul");
    }
}

pub(super) fn cublas_gemm<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    if out.n() == 0 {
        return;
    }
    let m = a.nrows as i32;
    let k = a.ncols as i32;
    let n = b.ncols as i32;
    use std::any::TypeId;
    // SAFETY: pointers are valid GPU buffers; alpha/beta are on host stack (cuBLAS copies them).
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                k,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex f32");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            cublas_result::dgemm(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &1.0f64,
                b.buf.ptr as *const f64,
                n,
                a.buf.ptr as *const f64,
                k,
                &0.0f64,
                out.buf.ptr as *mut f64,
                n,
            )
            .or_panic("cuBLAS dgemm");
        } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            // FP16 input/output, FP16 accumulation via Tensor Cores.
            // alpha/beta must be passed as fp16 bits when compute type is CUBLAS_COMPUTE_16F.
            let alpha = half::f16::ONE;
            let beta = half::f16::ZERO;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const half::f16 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k,
                &beta as *const half::f16 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex f16");
        } else if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
            // BF16 input/output, FP32 accumulation for numerical stability.
            let alpha = 1.0f32;
            let beta = 0.0f32;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex bf16");
        } else {
            cuda_matmul_tiled(ctx, out, a, b);
        }
    }
}

pub(crate) fn cuda_matmul<T: Scalar>(
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let ctx = get_ctx();
    out.invalidate_cache();
    cublas_gemm(ctx, out, a, b);
}

// Row-major C = A^T @ B via cuBLAS. A is [k,m], B is [k,n], C is [m,n].
// Col-major: C^T(n,m) = B_col(n,k) @ A_col(m,k)^T(k,m) => gemm(N, T, n, m, k, B, n, A, m, C, n)
pub(super) fn cublas_gemm_tn<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    if out.n() == 0 {
        return;
    }
    let m = a.ncols as i32;
    let k = a.nrows as i32;
    let n = b.ncols as i32;
    use std::any::TypeId;
    // SAFETY: pointers are valid GPU buffers; alpha/beta on host stack.
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let (alpha, beta) = (1.0f32, 0.0f32);
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                n,
                m,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                m,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex tn f32");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            cublas_result::dgemm(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                n,
                m,
                k,
                &1.0f64,
                b.buf.ptr as *const f64,
                n,
                a.buf.ptr as *const f64,
                m,
                &0.0f64,
                out.buf.ptr as *mut f64,
                n,
            )
            .or_panic("cuBLAS dgemm tn");
        } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            let alpha = half::f16::ONE;
            let beta = half::f16::ZERO;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                n,
                m,
                k,
                &alpha as *const half::f16 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                m,
                &beta as *const half::f16 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex tn f16");
        } else if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                n,
                m,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                m,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex tn bf16");
        } else {
            let a_t = cuda_transpose(a);
            cuda_matmul_tiled(ctx, out, &a_t, b);
        }
    }
}

// Row-major C = A @ B^T via cuBLAS. A is [m,k], B is [n,k], C is [m,n].
// Col-major: C^T(n,m) = B_col^T(n,k)(T,k) @ A_col(k,m)(N) => gemm(T, N, n, m, k, B, k, A, k, C, n)
pub(super) fn cublas_gemm_nt<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    if out.n() == 0 {
        return;
    }
    let m = a.nrows as i32;
    let k = a.ncols as i32;
    let n = b.nrows as i32;
    use std::any::TypeId;
    // SAFETY: pointers are valid GPU buffers; alpha/beta on host stack.
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let (alpha, beta) = (1.0f32, 0.0f32);
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                k,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                k,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex nt f32");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            cublas_result::dgemm(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &1.0f64,
                b.buf.ptr as *const f64,
                k,
                a.buf.ptr as *const f64,
                k,
                &0.0f64,
                out.buf.ptr as *mut f64,
                n,
            )
            .or_panic("cuBLAS dgemm nt");
        } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            let alpha = half::f16::ONE;
            let beta = half::f16::ZERO;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const half::f16 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k,
                &beta as *const half::f16 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex nt f16");
        } else if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .or_panic("cuBLAS gemm_ex nt bf16");
        } else {
            let b_t = cuda_transpose(b);
            cuda_matmul_tiled(ctx, out, a, &b_t);
        }
    }
}

pub(crate) fn cuda_matmul_tn<T: Scalar>(
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let ctx = get_ctx();
    out.invalidate_cache();
    cublas_gemm_tn(ctx, out, a, b);
}

pub(crate) fn cuda_matmul_nt<T: Scalar>(
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let ctx = get_ctx();
    out.invalidate_cache();
    cublas_gemm_nt(ctx, out, a, b);
}

const EPILOGUE_RELU_EXPR: &str = "(in0[i] + fabs(in0[i])) * 0.5";
const EPILOGUE_GELU_EXPR: &str =
    "0.5 * in0[i] * (1.0 + tanh(0.7978845608 * (in0[i] + 0.044715 * in0[i] * in0[i] * in0[i])))";
const EPILOGUE_RELU_HASH: &str = "nabla_epilogue_relu_v1";
const EPILOGUE_GELU_HASH: &str = "nabla_epilogue_gelu_v1";

pub(crate) fn cuda_matmul_epilogue_fallback<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    epilogue_id: u8,
) -> CudaStorage<T> {
    let mut gemm_out = cuda_zeros::<T>(a.nrows, b.ncols);
    cuda_matmul(&mut gemm_out, a, b);
    let m = gemm_out.nrows;
    let n = gemm_out.ncols;
    let in_ptr = &raw const gemm_out as *const CudaStorage<T> as *const u8;
    match epilogue_id {
        0 => cuda_fuse_launch::<T>(
            &[in_ptr],
            m,
            n,
            EPILOGUE_RELU_EXPR,
            EPILOGUE_RELU_HASH,
            1,
            4,
        ),
        1 => cuda_fuse_launch::<T>(
            &[in_ptr],
            m,
            n,
            EPILOGUE_GELU_EXPR,
            EPILOGUE_GELU_HASH,
            1,
            8,
        ),
        _ => gemm_out,
    }
}

pub(super) fn cublas_lt_gemm_f32(
    ctx: &CudaCtx,
    out: &mut CudaStorage<f32>,
    a: &CudaStorage<f32>,
    b: &CudaStorage<f32>,
    epilogue: Epilogue,
    bias: Option<CUdeviceptr>,
) -> Result<(), CudaError> {
    use super::core::{cublaslt_result as lt, cublaslt_sys as lts};
    use std::mem;

    if out.n() == 0 {
        return Ok(());
    }

    let m = a.nrows as u64;
    let k = a.ncols as u64;
    let n = b.ncols as u64;

    let layout_b = lt::create_matrix_layout(lts::cudaDataType_t::CUDA_R_32F, n, k, n as i64)?;
    let layout_a = lt::create_matrix_layout(lts::cudaDataType_t::CUDA_R_32F, k, m, k as i64)?;
    let layout_c = lt::create_matrix_layout(lts::cudaDataType_t::CUDA_R_32F, n, m, n as i64)?;

    let matmul_desc = lt::create_matmul_desc(
        lts::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        lts::cudaDataType_t::CUDA_R_32F,
    )?;

    let lt_epilogue: lts::cublasLtEpilogue_t = match epilogue {
        Epilogue::None => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        Epilogue::Relu => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        Epilogue::Gelu => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
        Epilogue::Bias => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        Epilogue::ReluBias => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
        Epilogue::GeluBias => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
    };

    // SAFETY: matmul_desc is valid; lt_epilogue is a pod value whose size matches the attribute.
    unsafe {
        lt::set_matmul_desc_attribute(
            matmul_desc,
            lts::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            &lt_epilogue as *const lts::cublasLtEpilogue_t as *const c_void,
            mem::size_of::<lts::cublasLtEpilogue_t>(),
        )?;
    }

    if let Some(bias_ptr) = bias {
        if matches!(
            epilogue,
            Epilogue::Bias | Epilogue::ReluBias | Epilogue::GeluBias
        ) {
            // SAFETY: bias_ptr is a valid device pointer supplied by the caller.
            unsafe {
                lt::set_matmul_desc_attribute(
                    matmul_desc,
                    lts::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &bias_ptr as *const CUdeviceptr as *const c_void,
                    mem::size_of::<CUdeviceptr>(),
                )?;
            }
        }
    }

    let matmul_pref = lt::create_matmul_pref()?;
    // SAFETY: matmul_pref is valid; size is a pod usize matching the preference attribute.
    unsafe {
        lt::set_matmul_pref_attribute(
            matmul_pref,
            lts::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ctx.blas_lt_workspace_size as *const usize as *const c_void,
            mem::size_of::<usize>(),
        )?;
    }

    // SAFETY: all descriptors/layouts are valid; heuristic output is written by the library.
    let heuristic = unsafe {
        lt::get_matmul_algo_heuristic(
            ctx.blas_lt.0,
            matmul_desc,
            layout_b, // A in col-major = B in row-major
            layout_a, // B in col-major = A in row-major
            layout_c,
            layout_c,
            matmul_pref,
        )?
    };

    let alpha = 1.0f32;
    let beta = 0.0f32;
    let stream_ptr = ctx.stream.cu_stream();

    // SAFETY: all handles/layouts/pointers are valid; alpha/beta are stack scalars.
    unsafe {
        lt::matmul(
            ctx.blas_lt.0,
            matmul_desc,
            &alpha as *const f32 as *const c_void,
            &beta as *const f32 as *const c_void,
            b.buf.ptr as *const c_void, // B row-major = A col-major
            layout_b,
            a.buf.ptr as *const c_void, // A row-major = B col-major
            layout_a,
            out.buf.ptr as *const c_void,
            layout_c,
            out.buf.ptr as *mut c_void,
            layout_c,
            &heuristic.algo,
            ctx.blas_lt_workspace as *mut c_void,
            ctx.blas_lt_workspace_size,
            stream_ptr as cublaslt_sys::cudaStream_t,
        )?;
    }

    // SAFETY: destroy calls are safe after the matmul has been enqueued on the stream.
    unsafe {
        let _ = lt::destroy_matmul_pref(matmul_pref);
        let _ = lt::destroy_matmul_desc(matmul_desc);
        let _ = lt::destroy_matrix_layout(layout_c);
        let _ = lt::destroy_matrix_layout(layout_a);
        let _ = lt::destroy_matrix_layout(layout_b);
    }

    Ok(())
}

pub fn cuda_matmul_epilogue(
    out: &mut CudaStorage<f32>,
    a: &CudaStorage<f32>,
    b: &CudaStorage<f32>,
    epilogue: Epilogue,
    bias: Option<CUdeviceptr>,
) -> Result<(), CudaError> {
    let ctx = get_ctx();
    out.invalidate_cache();
    cublas_lt_gemm_f32(ctx, out, a, b, epilogue, bias)
}

pub(super) fn cublas_gemm_strided_batched<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
) {
    if batch == 0 || m == 0 || n == 0 {
        return;
    }
    use std::any::TypeId;
    // SAFETY: pointers are valid GPU buffers; alpha/beta are stack scalars (cuBLAS copies them).
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha_f = alpha.to_f64() as f32;
            let beta_f = beta.to_f64() as f32;
            cublas_result::sgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha_f,
                b.buf.ptr as *const f32,
                n as i32,
                (k * n) as i64,
                a.buf.ptr as *const f32,
                k as i32,
                (m * k) as i64,
                &beta_f,
                out.buf.ptr as *mut f32,
                n as i32,
                (m * n) as i64,
                batch as i32,
            )
            .or_panic("cuBLAS sgemm_strided_batched");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let alpha_d = alpha.to_f64();
            let beta_d = beta.to_f64();
            cublas_result::dgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha_d,
                b.buf.ptr as *const f64,
                n as i32,
                (k * n) as i64,
                a.buf.ptr as *const f64,
                k as i32,
                (m * k) as i64,
                &beta_d,
                out.buf.ptr as *mut f64,
                n as i32,
                (m * n) as i64,
                batch as i32,
            )
            .or_panic("cuBLAS dgemm_strided_batched");
        } else {
            // SAFETY: borrow_ptr creates non-owning views into existing valid GPU buffers.
            for bi in 0..batch {
                let a_off = (bi * m * k * std::mem::size_of::<T>()) as u64;
                let b_off = (bi * k * n * std::mem::size_of::<T>()) as u64;
                let c_off = (bi * m * n * std::mem::size_of::<T>()) as u64;
                let a_slice = RtcStorage::new(
                    m,
                    k,
                    CuBuffer::borrow_ptr(a.buf.ptr + a_off, m * k * std::mem::size_of::<T>()),
                );
                let b_slice = RtcStorage::new(
                    k,
                    n,
                    CuBuffer::borrow_ptr(b.buf.ptr + b_off, k * n * std::mem::size_of::<T>()),
                );
                if beta == T::zero() && alpha == T::one() {
                    let mut tmp = cuda_zeros::<T>(m, n);
                    cuda_matmul_tiled(ctx, &mut tmp, &a_slice, &b_slice);
                    let bytes = m * n * std::mem::size_of::<T>();
                    result::memcpy_dtod_async(
                        out.buf.ptr + c_off,
                        tmp.buf.ptr,
                        bytes,
                        ctx.stream.cu_stream(),
                    )
                    .or_panic("CUDA memcpy batch tiled");
                } else {
                    let c_view = RtcStorage::new(
                        m,
                        n,
                        CuBuffer::borrow_ptr(out.buf.ptr + c_off, m * n * std::mem::size_of::<T>()),
                    );
                    let mut ab_tmp = cuda_zeros::<T>(m, n);
                    cuda_matmul_tiled(ctx, &mut ab_tmp, &a_slice, &b_slice);
                    let scaled_c = cuda_scale(&c_view, beta);
                    let scaled_ab = cuda_scale(&ab_tmp, alpha);
                    let res = launch_binary(&scaled_c, &scaled_ab, "add");
                    let bytes = m * n * std::mem::size_of::<T>();
                    result::memcpy_dtod_async(
                        out.buf.ptr + c_off,
                        res.buf.ptr,
                        bytes,
                        ctx.stream.cu_stream(),
                    )
                    .or_panic("CUDA memcpy batch fallback");
                }
            }
        }
    }
    out.invalidate_cache();
}

pub(crate) fn cuda_bmm<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let mut out = cuda_zeros::<T>(batch * m, n);
    cublas_gemm_strided_batched(ctx, &mut out, a, b, batch, m, k, n, T::one(), T::zero());
    out
}

pub(crate) fn cuda_addmm<T: Scalar>(
    c: &CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    beta: T,
    alpha: T,
) -> CudaStorage<T> {
    let m = c.nrows;
    let k = a.ncols;
    let n = c.ncols;
    let ctx = get_ctx();
    let mut out = cuda_clone(c);
    cublas_gemm_strided_batched(ctx, &mut out, a, b, 1, m, k, n, alpha, beta);
    out
}

pub(crate) fn cuda_baddbmm<T: Scalar>(
    c: &CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    beta: T,
    alpha: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let mut out = cuda_clone(c);
    cublas_gemm_strided_batched(ctx, &mut out, a, b, batch, m, k, n, alpha, beta);
    out
}

const TRSM_BASE: usize = 32;

pub(crate) fn gpu_trsm_lower<T: Scalar>(l: &CudaStorage<T>, b: &mut CudaStorage<T>) {
    let n = l.nrows;
    assert_eq!(l.nrows, l.ncols, "TRSM: L must be square");
    assert_eq!(b.nrows, n, "TRSM: B row count must match L");
    if n == 0 {
        return;
    }

    if n <= TRSM_BASE {
        trsm_base_host(l, b);
        return;
    }

    let half = n / 2;
    let nrhs = b.ncols;

    let l11 = cuda_submatrix(l, 0, 0, half, half);
    let mut b1 = cuda_submatrix(b, 0, 0, half, nrhs);
    gpu_trsm_lower(&l11, &mut b1);
    cuda_write_submatrix(b, 0, 0, &b1);

    let l21 = cuda_submatrix(l, half, 0, n - half, half);
    let b2 = cuda_submatrix(b, half, 0, n - half, nrhs);
    let mut tmp = cuda_zeros::<T>(n - half, nrhs);
    cuda_matmul(&mut tmp, &l21, &b1);
    let b2_updated = launch_binary(&b2, &tmp, "sub");
    cuda_write_submatrix(b, half, 0, &b2_updated);

    let l22 = cuda_submatrix(l, half, half, n - half, n - half);
    let mut b2_final = cuda_submatrix(b, half, 0, n - half, nrhs);
    gpu_trsm_lower(&l22, &mut b2_final);
    cuda_write_submatrix(b, half, 0, &b2_final);
}

fn trsm_base_host<T: Scalar>(l: &CudaStorage<T>, b: &mut CudaStorage<T>) {
    let n = l.nrows;
    let nrhs = b.ncols;
    let ctx = get_ctx();

    let mut l_host = vec![T::zero(); n * n];
    let mut b_host = vec![T::zero(); n * nrhs];
    l.buf
        .copy_to_host(&ctx.stream, &mut l_host)
        .or_panic("TRSM readback L");
    b.buf
        .copy_to_host(&ctx.stream, &mut b_host)
        .or_panic("TRSM readback B");

    for i in 0..n {
        let l_ii = l_host[i * n + i];
        for j in 0..nrhs {
            let mut sum = b_host[i * nrhs + j];
            for k in 0..i {
                sum = sum - l_host[i * n + k] * b_host[k * nrhs + j];
            }
            b_host[i * nrhs + j] = sum / l_ii;
        }
    }

    b.invalidate_cache();
    // SAFETY: uploading solved result back to the same-sized GPU buffer.
    unsafe {
        let _ = result::memcpy_htod_async(b.buf.ptr, &b_host, ctx.stream.cu_stream());
    }
}

fn cuda_submatrix<T: Scalar>(
    src: &CudaStorage<T>,
    row_off: usize,
    col_off: usize,
    nrows: usize,
    ncols: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let src_cols = src.ncols;

    let mut host = vec![T::zero(); src.n()];
    src.buf
        .copy_to_host(&ctx.stream, &mut host)
        .or_panic("submatrix readback");

    let mut sub = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            sub.push(host[(row_off + r) * src_cols + (col_off + c)]);
        }
    }

    let buf = CuBuffer::from_host(&ctx.stream, &sub).or_panic("submatrix upload");
    CudaStorage::new_cached(nrows, ncols, buf, sub)
}

fn cuda_write_submatrix<T: Scalar>(
    dst: &mut CudaStorage<T>,
    row_off: usize,
    col_off: usize,
    src: &CudaStorage<T>,
) {
    let ctx = get_ctx();
    let dst_cols = dst.ncols;
    let src_rows = src.nrows;
    let src_cols = src.ncols;

    let mut dst_host = vec![T::zero(); dst.n()];
    dst.buf
        .copy_to_host(&ctx.stream, &mut dst_host)
        .or_panic("write_submatrix dst readback");

    let mut src_host = vec![T::zero(); src.n()];
    src.buf
        .copy_to_host(&ctx.stream, &mut src_host)
        .or_panic("write_submatrix src readback");

    for r in 0..src_rows {
        for c in 0..src_cols {
            dst_host[(row_off + r) * dst_cols + (col_off + c)] = src_host[r * src_cols + c];
        }
    }

    dst.invalidate_cache();
    // SAFETY: uploading patched host data back to the same-sized GPU buffer.
    unsafe {
        let _ = result::memcpy_htod_async(dst.buf.ptr, &dst_host, ctx.stream.cu_stream());
    }
}

pub(crate) fn cuda_mse_sum_fwd<T: Scalar>(
    pred: &CudaStorage<T>,
    target: &CudaStorage<T>,
) -> CudaStorage<T> {
    use crate::kernels_cu::{REDUCE_BLOCK, REDUCE_GRID_CAP};
    let ctx = get_ctx();
    let n = pred.n();
    assert_eq!(n, target.n(), "mse_sum_fwd: pred/target size mismatch");
    if n == 0 {
        return cuda_zeros(1, 1);
    }
    let name = format!("k_mse_sum_fwd_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA mse_sum_fwd kernel");
    let grid1 = REDUCE_GRID_CAP.min(((n as u32) + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
    let scratch = ctx.reduce_scratch;
    let out_buf = alloc_out::<T>(ctx, 1);
    let n_u32 = n as u32;
    // SAFETY: launching reduction kernel; scratch is pre-allocated, out_buf is freshly allocated.
    unsafe {
        cudarc::driver::sys::cuLaunchKernel(
            func,
            grid1,
            1,
            1,
            REDUCE_BLOCK,
            1,
            1,
            0,
            ctx.stream.cu_stream(),
            [
                &pred.buf.ptr as *const CUdeviceptr as *mut c_void,
                &target.buf.ptr as *const CUdeviceptr as *mut c_void,
                &scratch as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
            ]
            .as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }
    CudaStorage::new(1, 1, out_buf)
}

pub(crate) fn cuda_mse_sum_bwd<T: Scalar>(
    pred: &CudaStorage<T>,
    target: &CudaStorage<T>,
    grad: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = pred.n();
    let name = format!("k_mse_sum_bwd_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA mse_sum_bwd kernel");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    let grid = cuda_grid_1d::<T>(n);
    // SAFETY: launching elementwise kernel with valid buffer pointers.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &pred.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &target.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &grad.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch mse_sum_bwd",
        );
    }
    CudaStorage::new(pred.nrows, pred.ncols, out_buf)
}

pub(crate) fn cuda_multi_axpy3_inplace<T: Scalar>(
    y: [&mut CudaStorage<T>; 3],
    x: [&CudaStorage<T>; 3],
    alpha: T,
) {
    let ctx = get_ctx();
    let n0 = y[0].n() as u32;
    let n1 = y[1].n() as u32;
    let n2 = y[2].n() as u32;
    let total = n0.max(n1).max(n2) as usize;
    if total == 0 {
        return;
    }
    let name = format!("k_multi_axpy3_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA multi_axpy3 kernel");
    let grid = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((total + 3) / 4)
    } else {
        grid_1d(total)
    };
    // SAFETY: launching kernel that updates y arrays in-place; x arrays are read-only.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &y[0].buf.ptr as *const CUdeviceptr as *mut c_void,
                    &x[0].buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n0 as *const u32 as *mut c_void,
                    &y[1].buf.ptr as *const CUdeviceptr as *mut c_void,
                    &x[1].buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n1 as *const u32 as *mut c_void,
                    &y[2].buf.ptr as *const CUdeviceptr as *mut c_void,
                    &x[2].buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n2 as *const u32 as *mut c_void,
                    &alpha as *const T as *mut c_void,
                ],
            ),
            "CUDA launch multi_axpy3",
        );
    }
    y[0].invalidate_cache();
    y[1].invalidate_cache();
    y[2].invalidate_cache();
}

/// Broadcast-add a bias row vector `(1, n)` to each row of `a (m, n)`.
/// Expands bias then element-wise adds using the standard `k_add` kernel.
pub(crate) fn cuda_add_bias_row<T: Scalar>(a: &CudaStorage<T>, bias: &CudaStorage<T>) -> CudaStorage<T> {
    let mut bias_expanded = cuda_zeros::<T>(a.nrows, a.ncols);
    cuda_expand(&mut bias_expanded, bias, 1, bias.ncols);
    launch_binary(a, &bias_expanded, "add")
}
