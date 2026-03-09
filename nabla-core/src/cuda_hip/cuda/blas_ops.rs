use std::ffi::c_void;

use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{RtcStorage, type_suffix};
use crate::scalar::Scalar;

use super::*;

// ---------------------------------------------------------------------------
// cuBLAS type dispatch helpers
// ---------------------------------------------------------------------------

/// Per-type cuBLAS parameters for gemm_ex dispatch.
struct GemmTypeParams {
    data_type: cublas_sys::cudaDataType_t,
    compute_type: cublas_sys::cublasComputeType_t,
    /// Alpha/beta as raw bytes (f32, f64, or f16 on stack).
    alpha_ptr: *const c_void,
    beta_ptr: *const c_void,
}

/// Dispatch gemm_ex for a single type via TypeId. Returns false if fallback needed.
///
/// `op_b`/`op_a` are the cuBLAS ops for the B and A matrices (col-major convention).
/// `ldb`/`lda`/`ldc` are the leading dimensions.
///
/// SAFETY: caller must ensure all pointers are valid GPU buffers.
unsafe fn gemm_ex_typed<T: Scalar>(
    ctx: &CudaCtx,
    op_b: cublas_sys::cublasOperation_t,
    op_a: cublas_sys::cublasOperation_t,
    n: i32,
    m: i32,
    k: i32,
    b_ptr: CUdeviceptr,
    ldb: i32,
    a_ptr: CUdeviceptr,
    lda: i32,
    out_ptr: CUdeviceptr,
    ldc: i32,
    label: &str,
) -> bool {
    use std::any::TypeId;

    let alpha_f32 = 1.0f32;
    let beta_f32 = 0.0f32;
    let alpha_f64 = 1.0f64;
    let beta_f64 = 0.0f64;
    let alpha_f16 = half::f16::ONE;
    let beta_f16 = half::f16::ZERO;

    let params = if TypeId::of::<T>() == TypeId::of::<f32>() {
        GemmTypeParams {
            data_type: cublas_sys::cudaDataType_t::CUDA_R_32F,
            compute_type: cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            alpha_ptr: &alpha_f32 as *const f32 as *const c_void,
            beta_ptr: &beta_f32 as *const f32 as *const c_void,
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublas_result::dgemm(
                ctx.blas.0,
                op_b,
                op_a,
                n,
                m,
                k,
                &alpha_f64,
                b_ptr as *const f64,
                ldb,
                a_ptr as *const f64,
                lda,
                &beta_f64,
                out_ptr as *mut f64,
                ldc,
            )
            .or_panic(label);
        }
        return true;
    } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
        GemmTypeParams {
            data_type: cublas_sys::cudaDataType_t::CUDA_R_16F,
            compute_type: cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            alpha_ptr: &alpha_f16 as *const half::f16 as *const c_void,
            beta_ptr: &beta_f16 as *const half::f16 as *const c_void,
        }
    } else if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
        GemmTypeParams {
            data_type: cublas_sys::cudaDataType_t::CUDA_R_16BF,
            compute_type: cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            alpha_ptr: &alpha_f32 as *const f32 as *const c_void,
            beta_ptr: &beta_f32 as *const f32 as *const c_void,
        }
    } else {
        return false;
    };

    unsafe {
        cublas_result::gemm_ex(
            ctx.blas.0,
            op_b,
            op_a,
            n,
            m,
            k,
            params.alpha_ptr,
            b_ptr as *const c_void,
            params.data_type,
            ldb,
            a_ptr as *const c_void,
            params.data_type,
            lda,
            params.beta_ptr,
            out_ptr as *mut c_void,
            params.data_type,
            ldc,
            params.compute_type,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .or_panic(label);
    }
    true
}

// ---------------------------------------------------------------------------
// cuBLAS GEMV f32 fast-path
// ---------------------------------------------------------------------------

/// GEMV f32 fast-path: C(1,n) = alpha * op(B) * A + beta * C.
///
/// SAFETY: all pointers must be valid GPU buffers; alpha/beta on stack.
#[inline]
unsafe fn sgemv_fast(
    ctx: &CudaCtx,
    op: cublas_sys::cublasOperation_t,
    rows: i32,
    cols: i32,
    mat_ptr: CUdeviceptr,
    ld: i32,
    vec_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    label: &str,
) {
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: pointers are valid GPU buffers; alpha/beta on host stack.
    unsafe {
        cublas_result::sgemv(
            ctx.blas.0,
            op,
            rows,
            cols,
            &alpha,
            mat_ptr as *const f32,
            ld,
            vec_ptr as *const f32,
            1,
            &beta,
            out_ptr as *mut f32,
            1,
        )
        .or_panic(label);
    }
}

// ---------------------------------------------------------------------------
// Tiled kernel matmul (generic fallback)
// ---------------------------------------------------------------------------

pub(super) fn cuda_matmul_tiled<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "matmul", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
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

// ---------------------------------------------------------------------------
// cuBLAS GEMM variants (NN, TN, NT)
// ---------------------------------------------------------------------------

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

    // f32 GEMV fast path
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && m == 1 {
        // SAFETY: TypeId guarantees T == f32; pointers are valid GPU buffers.
        unsafe {
            sgemv_fast(
                ctx,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                k,
                b.buf.ptr,
                n,
                a.buf.ptr,
                out.buf.ptr,
                "cuBLAS sgemv",
            )
        };
        return;
    }

    // SAFETY: pointers are valid GPU buffers; alpha/beta on host stack.
    let dispatched = unsafe {
        gemm_ex_typed::<T>(
            ctx,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            b.buf.ptr,
            n,
            a.buf.ptr,
            k,
            out.buf.ptr,
            n,
            "cuBLAS gemm_ex",
        )
    };
    if !dispatched {
        cuda_matmul_tiled(ctx, out, a, b);
    }
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

    // f32 GEMV fast path
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && m == 1 {
        // SAFETY: TypeId guarantees T == f32; pointers are valid GPU buffers.
        unsafe {
            sgemv_fast(
                ctx,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n,
                k,
                b.buf.ptr,
                n,
                a.buf.ptr,
                out.buf.ptr,
                "cuBLAS sgemv tn",
            )
        };
        return;
    }

    // SAFETY: pointers are valid GPU buffers; alpha/beta on host stack.
    let dispatched = unsafe {
        gemm_ex_typed::<T>(
            ctx,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            n,
            m,
            k,
            b.buf.ptr,
            n,
            a.buf.ptr,
            m,
            out.buf.ptr,
            n,
            "cuBLAS gemm_ex tn",
        )
    };
    if !dispatched {
        let a_t = cuda_transpose(a);
        cuda_matmul_tiled(ctx, out, &a_t, b);
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

    // f32 GEMV fast path
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && m == 1 {
        // SAFETY: TypeId guarantees T == f32; pointers are valid GPU buffers.
        unsafe {
            sgemv_fast(
                ctx,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                k,
                n,
                b.buf.ptr,
                k,
                a.buf.ptr,
                out.buf.ptr,
                "cuBLAS sgemv nt",
            )
        };
        return;
    }

    // SAFETY: pointers are valid GPU buffers; alpha/beta on host stack.
    let dispatched = unsafe {
        gemm_ex_typed::<T>(
            ctx,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            b.buf.ptr,
            k,
            a.buf.ptr,
            k,
            out.buf.ptr,
            n,
            "cuBLAS gemm_ex nt",
        )
    };
    if !dispatched {
        let b_t = cuda_transpose(b);
        cuda_matmul_tiled(ctx, out, a, &b_t);
    }
}

macro_rules! impl_cuda_matmul {
    ($name:ident, $gemm:ident) => {
        pub(crate) fn $name<T: Scalar>(
            out: &mut CudaStorage<T>,
            a: &CudaStorage<T>,
            b: &CudaStorage<T>,
        ) {
            let ctx = get_ctx();
            out.invalidate_cache();
            $gemm(ctx, out, a, b);
        }
    };
}

impl_cuda_matmul!(cuda_matmul, cublas_gemm);
impl_cuda_matmul!(cuda_matmul_tn, cublas_gemm_tn);
impl_cuda_matmul!(cuda_matmul_nt, cublas_gemm_nt);

// ---------------------------------------------------------------------------
// Epilogue fallback (fuse-based)
// ---------------------------------------------------------------------------

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
    let mut gemm_out = cuda_empty::<T>(a.nrows, b.ncols);
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

// ---------------------------------------------------------------------------
// cublasLt helpers
// ---------------------------------------------------------------------------

use super::core::{cublaslt_result as lt, cublaslt_sys as lts};

/// Common cublasLt GEMM setup: create layouts, matmul desc, pref, heuristic, execute, destroy.
///
/// SAFETY: all device pointers must be valid. Caller ensures correct types.
unsafe fn lt_gemm_core(
    ctx: &CudaCtx,
    ab_type: lts::cudaDataType_t,
    c_type: lts::cudaDataType_t,
    compute_type: lts::cublasComputeType_t,
    scale_type: lts::cudaDataType_t,
    m: u64,
    k: u64,
    n: u64,
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    epilogue: Option<(&Epilogue, Option<CUdeviceptr>, Option<lts::cudaDataType_t>)>,
) -> Result<(), CudaError> {
    unsafe {
        let layout_b = lt::create_matrix_layout(ab_type, n, k, n as i64)?;
        let layout_a = lt::create_matrix_layout(ab_type, k, m, k as i64)?;
        let layout_c = lt::create_matrix_layout(c_type, n, m, n as i64)?;

        let matmul_desc = lt::create_matmul_desc(compute_type, scale_type)?;

        if let Some((epi, bias, bias_type)) = epilogue {
            let lt_epi: lts::cublasLtEpilogue_t = match epi {
                Epilogue::None => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
                Epilogue::Relu => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
                Epilogue::Gelu => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
                Epilogue::Bias => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
                Epilogue::ReluBias => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
                Epilogue::GeluBias => lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
            };

            // SAFETY: matmul_desc is valid; lt_epi is a pod value matching the attribute size.
            lt::set_matmul_desc_attribute(
                matmul_desc,
                lts::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &lt_epi as *const lts::cublasLtEpilogue_t as *const c_void,
                std::mem::size_of::<lts::cublasLtEpilogue_t>(),
            )?;

            if let Some(bias_ptr) = bias {
                if matches!(
                    epi,
                    Epilogue::Bias | Epilogue::ReluBias | Epilogue::GeluBias
                ) {
                    // SAFETY: bias_ptr is a valid device pointer supplied by the caller.
                    lt::set_matmul_desc_attribute(
                        matmul_desc,
                        lts::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                        &bias_ptr as *const CUdeviceptr as *const c_void,
                        std::mem::size_of::<CUdeviceptr>(),
                    )?;
                    if let Some(bt) = bias_type {
                        lt::set_matmul_desc_attribute(
                        matmul_desc,
                        lts::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                        &bt as *const lts::cudaDataType_t as *const c_void,
                        std::mem::size_of::<lts::cudaDataType_t>(),
                    )?;
                    }
                }
            }
        }

        let matmul_pref = lt::create_matmul_pref()?;
        // SAFETY: matmul_pref is valid; size is a pod usize matching the preference attribute.
        lt::set_matmul_pref_attribute(
            matmul_pref,
            lts::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ctx.blas_lt_workspace_size as *const usize as *const c_void,
            std::mem::size_of::<usize>(),
        )?;

        // SAFETY: all descriptors/layouts are valid; heuristic output is written by the library.
        let heuristic = lt::get_matmul_algo_heuristic(
            ctx.blas_lt.0,
            matmul_desc,
            layout_b,
            layout_a,
            layout_c,
            layout_c,
            matmul_pref,
        )?;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // SAFETY: all handles/layouts/pointers are valid; alpha/beta are f32 stack scalars.
        lt::matmul(
            ctx.blas_lt.0,
            matmul_desc,
            &alpha as *const f32 as *const c_void,
            &beta as *const f32 as *const c_void,
            b_ptr as *const c_void,
            layout_b,
            a_ptr as *const c_void,
            layout_a,
            out_ptr as *const c_void,
            layout_c,
            out_ptr as *mut c_void,
            layout_c,
            &heuristic.algo,
            ctx.blas_lt_workspace as *mut c_void,
            ctx.blas_lt_workspace_size,
            ctx.stream.cu_stream() as lts::cudaStream_t,
        )?;

        // SAFETY: destroy calls are safe after the matmul has been enqueued on the stream.
        let _ = lt::destroy_matmul_pref(matmul_pref);
        let _ = lt::destroy_matmul_desc(matmul_desc);
        let _ = lt::destroy_matrix_layout(layout_c);
        let _ = lt::destroy_matrix_layout(layout_a);
        let _ = lt::destroy_matrix_layout(layout_b);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// cublasLt GEMM: typed epilogue dispatch (f32 / bf16)
// ---------------------------------------------------------------------------

/// Shared cublasLt GEMM with epilogue for f32 and bf16.
///
/// SAFETY: all device pointers in `out`, `a`, `b` must be valid GPU buffers.
unsafe fn cublas_lt_gemm_epilogue(
    ctx: &CudaCtx,
    ab_type: lts::cudaDataType_t,
    compute_type: lts::cublasComputeType_t,
    bias_data_type: Option<lts::cudaDataType_t>,
    m: u64,
    k: u64,
    n: u64,
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    out_ptr: CUdeviceptr,
    epilogue: Epilogue,
    bias: Option<CUdeviceptr>,
) -> Result<(), CudaError> {
    // SAFETY: forwarded from caller.
    unsafe {
        lt_gemm_core(
            ctx,
            ab_type,
            ab_type,
            compute_type,
            lts::cudaDataType_t::CUDA_R_32F,
            m,
            k,
            n,
            a_ptr,
            b_ptr,
            out_ptr,
            Some((&epilogue, bias, bias_data_type)),
        )
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
    if out.n() == 0 {
        return Ok(());
    }
    let (m, k, n) = (a.nrows as u64, a.ncols as u64, b.ncols as u64);
    // SAFETY: all pointers are valid GPU buffers from CudaStorage.
    unsafe {
        cublas_lt_gemm_epilogue(
            ctx,
            lts::cudaDataType_t::CUDA_R_32F,
            lts::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            None,
            m,
            k,
            n,
            a.buf.ptr,
            b.buf.ptr,
            out.buf.ptr,
            epilogue,
            bias,
        )
    }
}

pub(super) fn cublas_lt_gemm_bf16(
    ctx: &CudaCtx,
    out: &mut CudaStorage<half::bf16>,
    a: &CudaStorage<half::bf16>,
    b: &CudaStorage<half::bf16>,
    epilogue: Epilogue,
    bias: Option<CUdeviceptr>,
) -> Result<(), CudaError> {
    if out.n() == 0 {
        return Ok(());
    }
    let (m, k, n) = (a.nrows as u64, a.ncols as u64, b.ncols as u64);
    // SAFETY: all pointers are valid GPU buffers from CudaStorage.
    unsafe {
        cublas_lt_gemm_epilogue(
            ctx,
            lts::cudaDataType_t::CUDA_R_16BF,
            lts::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Some(lts::cudaDataType_t::CUDA_R_16BF),
            m,
            k,
            n,
            a.buf.ptr,
            b.buf.ptr,
            out.buf.ptr,
            epilogue,
            bias,
        )
    }
}

// ---------------------------------------------------------------------------
// cublasLt GEMM: FP8 (shared helper)
// ---------------------------------------------------------------------------

/// Shared FP8 cublasLt GEMM: output is always bf16, compute is f32.
fn cublas_lt_gemm_fp8(
    ctx: &CudaCtx,
    out: &mut CudaStorage<half::bf16>,
    ab_type: lts::cudaDataType_t,
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    m: u64,
    k: u64,
    n: u64,
) -> Result<(), CudaError> {
    if out.n() == 0 {
        return Ok(());
    }
    // SAFETY: all pointers are valid GPU buffers from CudaStorage.
    unsafe {
        lt_gemm_core(
            ctx,
            ab_type,
            lts::cudaDataType_t::CUDA_R_16BF,
            lts::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lts::cudaDataType_t::CUDA_R_32F,
            m,
            k,
            n,
            a_ptr,
            b_ptr,
            out.buf.ptr,
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Public FP8 matmul wrappers
// ---------------------------------------------------------------------------

fn cuda_fp8_matmul_impl(
    a_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    ab_type: lts::cudaDataType_t,
    m: usize,
    k: usize,
    n: usize,
    label: &str,
) -> CudaStorage<half::bf16> {
    let ctx = get_ctx();
    let mut out = cuda_zeros::<half::bf16>(m, n);
    cublas_lt_gemm_fp8(
        ctx, &mut out, ab_type, a_ptr, b_ptr, m as u64, k as u64, n as u64,
    )
    .or_panic(label);
    out
}

pub(crate) fn cuda_fp8_matmul_e4m3(
    a: &CudaStorage<crate::scalar::Fp8E4M3>,
    b: &CudaStorage<crate::scalar::Fp8E4M3>,
) -> CudaStorage<half::bf16> {
    cuda_fp8_matmul_impl(
        a.buf.ptr,
        b.buf.ptr,
        lts::cudaDataType_t::CUDA_R_8F_E4M3,
        a.nrows,
        a.ncols,
        b.ncols,
        "cublasLt FP8 E4M3 GEMM",
    )
}

pub(crate) fn cuda_fp8_matmul_e5m2(
    a: &CudaStorage<crate::scalar::Fp8E5M2>,
    b: &CudaStorage<crate::scalar::Fp8E5M2>,
) -> CudaStorage<half::bf16> {
    cuda_fp8_matmul_impl(
        a.buf.ptr,
        b.buf.ptr,
        lts::cudaDataType_t::CUDA_R_8F_E5M2,
        a.nrows,
        a.ncols,
        b.ncols,
        "cublasLt FP8 E5M2 GEMM",
    )
}

// ---------------------------------------------------------------------------
// Public epilogue matmul wrappers
// ---------------------------------------------------------------------------

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

pub fn cuda_matmul_epilogue_bf16(
    out: &mut CudaStorage<half::bf16>,
    a: &CudaStorage<half::bf16>,
    b: &CudaStorage<half::bf16>,
    epilogue: Epilogue,
    bias: Option<CUdeviceptr>,
) -> Result<(), CudaError> {
    let ctx = get_ctx();
    out.invalidate_cache();
    cublas_lt_gemm_bf16(ctx, out, a, b, epilogue, bias)
}

// ---------------------------------------------------------------------------
// Batched GEMM
// ---------------------------------------------------------------------------

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
    let mut out = cuda_empty::<T>(batch * m, n);
    cublas_gemm_strided_batched(ctx, &mut out, a, b, batch, m, k, n, T::one(), T::zero());
    out
}

pub(crate) fn cuda_bmm_nt<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let mut out = cuda_empty::<T>(batch * m, n);
    if batch == 0 || m == 0 || n == 0 {
        return out;
    }
    use std::any::TypeId;
    // SAFETY: pointers are valid GPU buffers; alpha/beta are stack scalars.
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let (alpha_f, beta_f) = (1.0f32, 0.0f32);
            cublas_result::sgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha_f,
                b.buf.ptr as *const f32,
                k as i32,
                (n * k) as i64,
                a.buf.ptr as *const f32,
                k as i32,
                (m * k) as i64,
                &beta_f,
                out.buf.ptr as *mut f32,
                n as i32,
                (m * n) as i64,
                batch as i32,
            )
            .or_panic("cuBLAS sgemm_strided_batched bmm_nt");
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let (alpha_d, beta_d) = (1.0f64, 0.0f64);
            cublas_result::dgemm_strided_batched(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha_d,
                b.buf.ptr as *const f64,
                k as i32,
                (n * k) as i64,
                a.buf.ptr as *const f64,
                k as i32,
                (m * k) as i64,
                &beta_d,
                out.buf.ptr as *mut f64,
                n as i32,
                (m * n) as i64,
                batch as i32,
            )
            .or_panic("cuBLAS dgemm_strided_batched bmm_nt");
        } else {
            bmm_nt_tiled_fallback(ctx, &mut out, a, b, batch, m, k, n);
        }
    }
    out.invalidate_cache();
    out
}

/// Tiled fallback for bmm_nt when cuBLAS strided batched is unavailable.
fn bmm_nt_tiled_fallback<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let elem = std::mem::size_of::<T>();
    for bi in 0..batch {
        // SAFETY: borrow_ptr creates non-owning views into existing valid GPU buffers.
        let a_slice = RtcStorage::new(m, k, unsafe {
            CuBuffer::borrow_ptr(a.buf.ptr + (bi * m * k * elem) as u64, m * k * elem)
        });
        let b_slice = RtcStorage::new(n, k, unsafe {
            CuBuffer::borrow_ptr(b.buf.ptr + (bi * n * k * elem) as u64, n * k * elem)
        });
        let b_t = cuda_transpose(&b_slice);
        let mut tmp = cuda_zeros::<T>(m, n);
        cuda_matmul_tiled(ctx, &mut tmp, &a_slice, &b_t);
        let bytes = m * n * elem;
        // SAFETY: copying result to correct offset in output buffer.
        unsafe {
            result::memcpy_dtod_async(
                out.buf.ptr + (bi * m * n * elem) as u64,
                tmp.buf.ptr,
                bytes,
                ctx.stream.cu_stream(),
            )
            .or_panic("CUDA memcpy batch tiled bmm_nt");
        }
    }
}

pub(crate) fn cuda_addmm<T: Scalar>(
    c: &CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    beta: T,
    alpha: T,
) -> CudaStorage<T> {
    let (m, k, n) = (c.nrows, a.ncols, c.ncols);
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
