use std::ffi::c_void;

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::common::rtc::EnsureCache;
use crate::gpu_common::{self, grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

#[inline]
pub(crate) fn kernel_name_buf<'a>(buf: &'a mut [u8; 64], op: &str, suffix: &str) -> &'a str {
    let mut pos = 0;
    buf[pos..pos + 2].copy_from_slice(b"k_");
    pos += 2;
    buf[pos..pos + op.len()].copy_from_slice(op.as_bytes());
    pos += op.len();
    buf[pos] = b'_';
    pos += 1;
    buf[pos..pos + suffix.len()].copy_from_slice(suffix.as_bytes());
    pos += suffix.len();
    // SAFETY: all kernel name components are ASCII
    unsafe { std::str::from_utf8_unchecked(&buf[..pos]) }
}

#[inline]
pub(super) fn cuda_grid_1d<T: Scalar>(n: usize) -> u32 {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    }
}

#[inline]
pub(crate) fn launch_unary<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, op, type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (cuda_grid_1d::<T>(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA kernel launch",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

#[inline]
pub(crate) fn launch_unary_inplace<T: Scalar>(a: &mut CudaStorage<T>, op: &str) {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, op, type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let n_u32 = n as u32;
    // SAFETY: kernel is elementwise; reading and writing same buffer is safe.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (cuda_grid_1d::<T>(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA kernel launch inplace",
        );
    }
    a.invalidate_cache();
}

#[inline]
pub(super) fn launch_binary<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    op: &str,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, op, type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (cuda_grid_1d::<T>(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA kernel launch",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

#[inline]
pub(super) fn launch_binary_alpha<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    alpha: T,
    op: &str,
) -> CudaStorage<T> {
    use std::any::TypeId;
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, op, type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let alpha_f64 = alpha.to_f64();
        unsafe {
            expect_ok(
                result::launch_kernel(
                    func,
                    (cuda_grid_1d::<T>(n), 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &alpha_f64 as *const f64 as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &n_u32 as *const u32 as *mut c_void,
                    ],
                ),
                "CUDA kernel launch",
            );
        }
    } else {
        let alpha_f32 = alpha.to_f64() as f32;
        unsafe {
            expect_ok(
                result::launch_kernel(
                    func,
                    (cuda_grid_1d::<T>(n), 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &alpha_f32 as *const f32 as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &n_u32 as *const u32 as *mut c_void,
                    ],
                ),
                "CUDA kernel launch",
            );
        }
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

impl crate::backend::private::Sealed for crate::backend::Cuda {}

impl crate::backend::BackendCore for crate::backend::Cuda {
    type Storage<T: Scalar> = CudaStorage<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
        cuda_zeros(nrows, ncols)
    }
    fn empty<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
        cuda_empty(nrows, ncols)
    }

    gpu_common::rtc_core_impl! {
        CudaStorage; fill=cuda_fill, from_fn=cuda_from_fn, from_vec_async=cuda_from_vec_async,
        get=cuda_get, set=cuda_set, transpose=cuda_transpose, scale=cuda_scale,
        clone_storage=cuda_clone,
    }

    #[inline]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> CudaStorage<T> {
        let ctx = get_ctx();
        let buf = CuBuffer::from_host(&ctx.stream, &data).or_panic("CUDA upload");
        CudaStorage::new_cached(nrows, ncols, buf, data)
    }

    fn one_hot_from_indices<T: Scalar>(
        indices: &CudaStorage<T>,
        n_classes: usize,
    ) -> CudaStorage<T> {
        cuda_one_hot_from_indices(indices, n_classes)
    }

    #[inline]
    fn to_vec_async<T: Scalar>(a: &CudaStorage<T>) -> Vec<T> {
        cuda_to_vec_async(a).or_panic("CUDA D2H async")
    }

    #[inline]
    fn cast<T: Scalar, U: Scalar>(a: &CudaStorage<T>) -> CudaStorage<U> {
        cuda_cast(a)
    }

    #[inline]
    fn prefetch<T: Scalar>(storage: &CudaStorage<T>) {
        storage.ensure_cache();
    }

    #[inline]
    fn sync<T: Scalar>(_s: &CudaStorage<T>) {
        let ctx = get_ctx();
        ctx.stream.synchronize().or_panic("CUDA sync");
    }

    #[inline]
    fn device_ptr<T: Scalar>(storage: &CudaStorage<T>) -> u64 {
        storage.buf.ptr
    }

    gpu_common::gpu_binary_ops!(CudaStorage; add, sub);

    #[inline]
    fn axpy_inplace<T: Scalar>(y: &mut CudaStorage<T>, alpha: T, x: &CudaStorage<T>) {
        cuda_axpy_inplace(y, alpha, x);
    }

    #[inline]
    fn expand_into<T: Scalar>(
        out: &mut CudaStorage<T>,
        src: &CudaStorage<T>,
        src_rows: usize,
        src_cols: usize,
    ) {
        cuda_expand(out, src, src_rows, src_cols);
    }
}

impl crate::backend::BackendMath for crate::backend::Cuda {
    gpu_common::gpu_unary_ops!(CudaStorage; exp, ln, log1p, sin, cos, tan, tanh, sqrt, abs, recip, erf, ceil, floor, round, asin, acos, atan, sinh, cosh, asinh, acosh, atanh, log2, log10);
    gpu_common::gpu_binary_ops!(CudaStorage; emul, ediv, atan2);
    gpu_common::rtc_math_impl!(CudaStorage; powf=cuda_powf,);

    #[inline]
    fn masked_fill<T: Scalar>(
        a: &CudaStorage<T>,
        mask: &CudaStorage<T>,
        value: T,
    ) -> CudaStorage<T> {
        cuda_masked_fill(a, mask, value)
    }

    #[inline]
    fn where_cond<T: Scalar>(
        a: &CudaStorage<T>,
        cond: &CudaStorage<T>,
        b: &CudaStorage<T>,
    ) -> CudaStorage<T> {
        cuda_where(a, cond, b)
    }
}

impl crate::backend::BackendReduce for crate::backend::Cuda {
    gpu_common::rtc_reduce_impl! {
        CudaStorage; sum_all=cuda_sum_all, max_all=cuda_max_all, min_all=cuda_min_all,
        argmax_all=cuda_argmax_all, argmin_all=cuda_argmin_all, axis_reduce=cuda_axis_reduce,
        cumsum_cumprod=cuda_cumsum_cumprod, prod_all=cuda_prod_all,
    }

    #[inline]
    fn sum_all_1x1<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_sum_all_1x1(a)
    }
    #[inline]
    fn max_all_1x1<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_max_all_1x1(a)
    }
    #[inline]
    fn min_all_1x1<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_min_all_1x1(a)
    }

    #[inline]
    fn diag<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_diag(a)
    }

    #[inline]
    fn trace<T: Scalar>(a: &CudaStorage<T>) -> T {
        let d = cuda_diag(a);
        cuda_sum_all(&d)
    }

    #[inline]
    fn mse_sum_fwd<T: Scalar>(pred: &CudaStorage<T>, target: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_mse_sum_fwd(pred, target)
    }

    #[inline]
    fn mse_sum_bwd<T: Scalar>(
        pred: &CudaStorage<T>,
        target: &CudaStorage<T>,
        grad: &CudaStorage<T>,
    ) -> CudaStorage<T> {
        cuda_mse_sum_bwd(pred, target, grad)
    }
}

impl crate::backend::BackendShape for crate::backend::Cuda {
    #[inline]
    fn reshape_metadata<T: Scalar>(a: &mut CudaStorage<T>, new_rows: usize, new_cols: usize) {
        a.reshape_metadata(new_rows, new_cols);
    }

    #[inline]
    fn reshape_copy<T: Scalar>(
        a: &CudaStorage<T>,
        out_rows: usize,
        out_cols: usize,
    ) -> CudaStorage<T> {
        cuda_copy_reshape(a, out_rows, out_cols)
    }

    #[inline]
    fn submatrix<T: Scalar>(
        a: &CudaStorage<T>,
        row_start: usize,
        col_start: usize,
        out_rows: usize,
        out_cols: usize,
    ) -> CudaStorage<T> {
        cuda_submatrix(a, row_start, col_start, out_rows, out_cols)
    }

    #[inline]
    fn slice_set<T: Scalar>(
        dst: &mut CudaStorage<T>,
        row_start: usize,
        col_start: usize,
        src: &CudaStorage<T>,
    ) {
        cuda_slice_set(dst, row_start, col_start, src);
    }

    #[inline]
    fn repeat<T: Scalar>(a: &CudaStorage<T>, row_reps: usize, col_reps: usize) -> CudaStorage<T> {
        cuda_repeat(a, row_reps, col_reps)
    }

    #[inline]
    fn pad<T: Scalar>(
        a: &CudaStorage<T>,
        left: usize,
        right: usize,
        top: usize,
        bottom: usize,
        value: T,
    ) -> CudaStorage<T> {
        cuda_pad(a, left, right, top, bottom, value)
    }

    #[inline]
    fn triu<T: Scalar>(a: &CudaStorage<T>, diagonal: isize) -> CudaStorage<T> {
        cuda_triu(a, diagonal)
    }

    #[inline]
    fn tril<T: Scalar>(a: &CudaStorage<T>, diagonal: isize) -> CudaStorage<T> {
        cuda_tril(a, diagonal)
    }

    #[inline]
    fn roll<T: Scalar>(a: &CudaStorage<T>, shift: isize, axis: usize) -> CudaStorage<T> {
        cuda_roll(a, shift, axis)
    }

    #[inline]
    fn flip<T: Scalar>(a: &CudaStorage<T>, axis: usize) -> CudaStorage<T> {
        cuda_flip(a, axis)
    }

    #[inline]
    fn from_diag<T: Scalar>(v: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_from_diag(v)
    }

    #[inline]
    fn gather_rows<T: Scalar>(a: &CudaStorage<T>, indices: &[usize]) -> CudaStorage<T> {
        cuda_gather_rows(a, indices)
    }

    #[inline]
    fn gather<T: Scalar>(
        a: &CudaStorage<T>,
        axis: usize,
        index: &CudaStorage<T>,
    ) -> CudaStorage<T> {
        cuda_gather(a, axis, index)
    }

    #[inline]
    fn scatter<T: Scalar>(
        a: &CudaStorage<T>,
        axis: usize,
        index: &CudaStorage<T>,
        src: &CudaStorage<T>,
    ) -> CudaStorage<T> {
        cuda_scatter(a, axis, index, src)
    }

    #[inline]
    fn index_select<T: Scalar>(
        a: &CudaStorage<T>,
        axis: usize,
        index: &CudaStorage<T>,
    ) -> CudaStorage<T> {
        cuda_index_select(a, axis, index)
    }

    #[inline]
    fn sort_rows<T: Scalar>(
        a: &CudaStorage<T>,
        descending: bool,
    ) -> (CudaStorage<T>, CudaStorage<T>) {
        cuda_sort_rows(a, descending)
    }

    #[inline]
    fn topk_rows<T: Scalar>(a: &CudaStorage<T>, k: usize) -> (CudaStorage<T>, CudaStorage<T>) {
        cuda_topk_rows(a, k)
    }

    #[inline]
    fn meshgrid<T: Scalar>(
        x: &CudaStorage<T>,
        y: &CudaStorage<T>,
    ) -> (CudaStorage<T>, CudaStorage<T>) {
        cuda_meshgrid(x, y)
    }

    #[inline]
    fn scatter_add_dim0<T: Scalar>(
        dst: &mut CudaStorage<T>,
        indices: &[usize],
        src: &CudaStorage<T>,
    ) {
        cuda_scatter_add_dim0(dst, indices, src);
    }

    #[inline]
    fn scatter_add<T: Scalar>(
        dst: &mut CudaStorage<T>,
        axis: usize,
        indices: &[usize],
        src: &CudaStorage<T>,
    ) {
        cuda_scatter_add(dst, axis, indices, src);
    }

    #[inline]
    fn kron<T: Scalar>(
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
        m: usize,
        n: usize,
        p: usize,
        q: usize,
    ) -> CudaStorage<T> {
        cuda_kron(a, b, m, n, p, q)
    }
}

/// Try cublasLt epilogue for concrete type `U`; fall back on error.
///
/// SAFETY: caller must verify `TypeId::of::<T>() == TypeId::of::<U>()`.
fn try_lt_epilogue<T: Scalar, U: Scalar>(
    a: &CudaStorage<T>, b: &CudaStorage<T>,
    epilogue: Epilogue, bias: Option<CUdeviceptr>,
    matmul_fn: fn(&mut CudaStorage<U>, &CudaStorage<U>, &CudaStorage<U>, Epilogue, Option<CUdeviceptr>) -> CudaResult<()>,
    fallback: impl FnOnce() -> CudaStorage<T>,
) -> CudaStorage<T> {
    // SAFETY: TypeId check by caller guarantees T == U; same layout.
    let (a_u, b_u) = unsafe {
        (&*(a as *const CudaStorage<T> as *const CudaStorage<U>),
         &*(b as *const CudaStorage<T> as *const CudaStorage<U>))
    };
    let mut out_u = cuda_zeros::<U>(a_u.nrows, b_u.ncols);
    match matmul_fn(&mut out_u, a_u, b_u, epilogue, bias) {
        // SAFETY: T == U verified by caller.
        Ok(()) => unsafe { std::mem::transmute::<CudaStorage<U>, CudaStorage<T>>(out_u) },
        Err(_) => fallback(),
    }
}

impl crate::backend::BackendBlas for crate::backend::Cuda {
    #[inline]
    fn matmul_into<T: Scalar>(out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) {
        cuda_matmul(out, a, b);
    }

    #[inline]
    fn matmul_tn_into<T: Scalar>(out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) {
        cuda_matmul_tn(out, a, b);
    }

    #[inline]
    fn matmul_nt_into<T: Scalar>(out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) {
        cuda_matmul_nt(out, a, b);
    }

    fn matmul_epilogue<T: Scalar>(
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
        epilogue_id: u8,
    ) -> CudaStorage<T> {
        use std::any::TypeId;
        let epilogue = if epilogue_id == 0 { Epilogue::Relu } else { Epilogue::Gelu };
        let fallback = || cuda_matmul_epilogue_fallback(a, b, epilogue_id);
        if epilogue_id <= 1 {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                // SAFETY: TypeId guarantees T == f32.
                return try_lt_epilogue(a, b, epilogue, None, cuda_matmul_epilogue, fallback);
            }
            if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
                // SAFETY: TypeId guarantees T == half::bf16.
                return try_lt_epilogue(a, b, epilogue, None, cuda_matmul_epilogue_bf16, fallback);
            }
        }
        fallback()
    }

    fn matmul_bias<T: Scalar>(
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
        bias: &CudaStorage<T>,
    ) -> CudaStorage<T> {
        use std::any::TypeId;
        let bias_ptr = Some(bias.buf.ptr);
        let fallback = || {
            let mut out = cuda_zeros::<T>(a.nrows, b.ncols);
            cuda_matmul(&mut out, a, b);
            cuda_add_bias_row(&out, bias)
        };
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: TypeId guarantees T == f32.
            return try_lt_epilogue(a, b, Epilogue::Bias, bias_ptr, cuda_matmul_epilogue, fallback);
        }
        if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
            // SAFETY: TypeId guarantees T == half::bf16.
            return try_lt_epilogue(a, b, Epilogue::Bias, bias_ptr, cuda_matmul_epilogue_bf16, fallback);
        }
        fallback()
    }

    #[inline]
    fn bmm<T: Scalar>(
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> CudaStorage<T> {
        cuda_bmm(a, b, batch, m, k, n)
    }

    #[inline]
    fn bmm_nt<T: Scalar>(
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> CudaStorage<T> {
        cuda_bmm_nt(a, b, batch, m, k, n)
    }

    #[inline]
    fn addmm<T: Scalar>(
        c: &CudaStorage<T>,
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
        beta: T,
        alpha: T,
    ) -> CudaStorage<T> {
        cuda_addmm(c, a, b, beta, alpha)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn baddbmm<T: Scalar>(
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
        cuda_baddbmm(c, a, b, batch, m, k, n, beta, alpha)
    }

    #[inline]
    fn fp8_matmul_e4m3(
        a: &CudaStorage<crate::scalar::Fp8E4M3>,
        b: &CudaStorage<crate::scalar::Fp8E4M3>,
    ) -> CudaStorage<half::bf16> {
        cuda_fp8_matmul_e4m3(a, b)
    }

    #[inline]
    fn fp8_matmul_e5m2(
        a: &CudaStorage<crate::scalar::Fp8E5M2>,
        b: &CudaStorage<crate::scalar::Fp8E5M2>,
    ) -> CudaStorage<half::bf16> {
        cuda_fp8_matmul_e5m2(a, b)
    }
}

impl crate::backend::BackendNN for crate::backend::Cuda {
    gpu_common::gpu_unary_ops!(CudaStorage; sigmoid, silu, mish, hardswish);
    #[inline]
    fn relu_backward<T: Scalar>(g: &CudaStorage<T>, x: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(g, x, "relu_bwd")
    }
    #[inline]
    fn leaky_relu_backward<T: Scalar>(
        g: &CudaStorage<T>,
        x: &CudaStorage<T>,
        alpha: T,
    ) -> CudaStorage<T> {
        launch_binary_alpha(g, x, alpha, "leaky_relu_bwd")
    }
    #[inline]
    fn elu_backward<T: Scalar>(g: &CudaStorage<T>, x: &CudaStorage<T>, alpha: T) -> CudaStorage<T> {
        launch_binary_alpha(g, x, alpha, "elu_bwd")
    }
    #[inline]
    fn gelu_backward<T: Scalar>(g: &CudaStorage<T>, x: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(g, x, "gelu_bwd")
    }
    #[inline]
    fn abs_backward<T: Scalar>(g: &CudaStorage<T>, x: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(g, x, "abs_bwd")
    }
    gpu_common::rtc_nn_impl! {
        CudaStorage; softmax=cuda_softmax, layer_norm=cuda_layer_norm, rms_norm=cuda_rms_norm,
        group_norm=cuda_group_norm,
        batch_norm_train=cuda_batch_norm_train, cross_entropy_fused=cuda_cross_entropy_fused,
        sdpa=cuda_sdpa, embedding=cuda_embedding, max_pool2d=cuda_max_pool2d,
        max_pool2d_with_idx=cuda_max_pool2d_with_idx, avg_pool2d=cuda_avg_pool2d,
        adaptive_avg_pool2d=cuda_adaptive_avg_pool2d, conv2d=cuda_conv2d, conv1d=cuda_conv1d,
        conv3d=cuda_conv3d, conv_transpose2d=cuda_conv_transpose2d,
    }

    #[inline]
    fn embedding_backward<T: Scalar>(
        indices: &CudaStorage<T>,
        grad: &CudaStorage<T>,
        vocab: usize,
    ) -> CudaStorage<T> {
        cuda_embedding_backward(indices, grad, vocab)
    }
    #[inline]
    fn wht<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_wht(a)
    }
    #[inline]
    fn wht_inverse<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
        cuda_wht_inverse(a)
    }
}

impl crate::backend::BackendFusion for crate::backend::Cuda {
    fn fuse_launch<T: Scalar>(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        _cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reg_estimate: usize,
    ) -> CudaStorage<T> {
        cuda_fuse_launch::<T>(
            inputs,
            nrows,
            ncols,
            gpu_expr,
            kernel_hash,
            n_inputs,
            reg_estimate,
        )
    }

    fn mega_fuse_launch<'a, T: Scalar>(
        ops: &[(Vec<*const u8>, String, usize, bool)],
        nrows: usize,
        ncols: usize,
        _cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T + 'a>>,
        kernel_hash: &str,
    ) -> Vec<CudaStorage<T>> {
        let mega_ops: Vec<MegaFuseOp> = ops
            .iter()
            .map(|(inputs, gpu_expr, n_inputs, uses_prev)| MegaFuseOp {
                inputs: inputs.clone(),
                gpu_expr: gpu_expr.clone(),
                n_inputs: *n_inputs,
                uses_prev: *uses_prev,
            })
            .collect();
        cuda_mega_fuse_launch::<T>(&mega_ops, nrows, ncols, kernel_hash)
    }

    #[allow(clippy::too_many_arguments)]
    fn fuse_reduce_launch<T: Scalar>(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        _cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reduce_op: u8,
        axis: u8,
    ) -> CudaStorage<T> {
        cuda_fuse_reduce_launch::<T>(
            inputs,
            nrows,
            ncols,
            gpu_expr,
            kernel_hash,
            n_inputs,
            reduce_op,
            axis,
        )
    }
}
