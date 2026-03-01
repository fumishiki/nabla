use std::ffi::{CString, c_void};

use cudarc::driver::result;
use cudarc::driver::sys::{CUdeviceptr, CUfunction};
use cudarc::nvrtc;

use crate::gpu_common::common::rtc::EnsureCache;
use crate::gpu_common::{self, grid_1d, type_suffix};
use crate::kernels_cu::{self, BLOCK_SIZE};
use crate::scalar::Scalar;

use super::*;

const KERNEL_NAMES: &[&str] = &[
    "k_neg_f32",
    "k_recip_f32",
    "k_exp_f32",
    "k_ln_f32",
    "k_log1p_f32",
    "k_sin_f32",
    "k_cos_f32",
    "k_tanh_f32",
    "k_sqrt_f32",
    "k_abs_f32",
    "k_ceil_f32",
    "k_floor_f32",
    "k_round_f32",
    "k_erf_f32",
    "k_asin_f32",
    "k_acos_f32",
    "k_atan_f32",
    "k_atan2_f32",
    "k_sinh_f32",
    "k_cosh_f32",
    "k_asinh_f32",
    "k_acosh_f32",
    "k_atanh_f32",
    "k_log2_f32",
    "k_log10_f32",
    "k_sigmoid_f32",
    "k_silu_f32",
    "k_mish_f32",
    "k_leaky_relu_f32",
    "k_elu_f32",
    "k_hardswish_f32",
    "k_add_f32",
    "k_sub_f32",
    "k_emul_f32",
    "k_ediv_f32",
    "k_scale_f32",
    "k_powf_f32",
    "k_fill_f32",
    "k_transpose_f32",
    "k_matmul_f32",
    "k_sum_f32",
    "k_max_f32",
    "k_min_f32",
    "k_softmax_f32",
    "k_layer_norm_f32",
    "k_rms_norm_f32",
    "k_sum_axis1_f32",
    "k_max_axis1_f32",
    "k_embedding_f32",
    "k_cumsum_axis1_f32",
    "k_cumprod_axis1_f32",
    "k_neg_f64",
    "k_recip_f64",
    "k_exp_f64",
    "k_ln_f64",
    "k_log1p_f64",
    "k_sin_f64",
    "k_cos_f64",
    "k_tanh_f64",
    "k_sqrt_f64",
    "k_abs_f64",
    "k_ceil_f64",
    "k_floor_f64",
    "k_round_f64",
    "k_erf_f64",
    "k_asin_f64",
    "k_acos_f64",
    "k_atan_f64",
    "k_atan2_f64",
    "k_sinh_f64",
    "k_cosh_f64",
    "k_asinh_f64",
    "k_acosh_f64",
    "k_atanh_f64",
    "k_log2_f64",
    "k_log10_f64",
    "k_sigmoid_f64",
    "k_silu_f64",
    "k_mish_f64",
    "k_leaky_relu_f64",
    "k_elu_f64",
    "k_hardswish_f64",
    "k_add_f64",
    "k_sub_f64",
    "k_emul_f64",
    "k_ediv_f64",
    "k_scale_f64",
    "k_powf_f64",
    "k_fill_f64",
    "k_transpose_f64",
    "k_matmul_f64",
    "k_sum_f64",
    "k_max_f64",
    "k_min_f64",
    "k_softmax_f64",
    "k_layer_norm_f64",
    "k_rms_norm_f64",
    "k_sum_axis1_f64",
    "k_max_axis1_f64",
    "k_embedding_f64",
    "k_cumsum_axis1_f64",
    "k_cumprod_axis1_f64",
    "k_prod_partial_f32",
    "k_prod_partial_f64",
    "k_max_pool2d_f32",
    "k_max_pool2d_with_idx_f32",
    "k_avg_pool2d_f32",
    "k_adaptive_avg_pool2d_f32",
    "k_max_pool2d_f64",
    "k_max_pool2d_with_idx_f64",
    "k_avg_pool2d_f64",
    "k_adaptive_avg_pool2d_f64",
    "k_im2col_f32",
    "k_im2col_f64",
    "k_im1col_f32",
    "k_im1col_f64",
    "k_im3col_f32",
    "k_im3col_f64",
    "k_batch_norm_stats_f32",
    "k_batch_norm_fwd_f32",
    "k_batch_norm_stats_f64",
    "k_batch_norm_fwd_f64",
    "k_cross_entropy_f32",
    "k_cross_entropy_f64",
    "k_sdpa_f32",
    "k_sdpa_f64",
    "k_conv_transpose2d_f32",
    "k_conv_transpose2d_f64",
    "k_axpy_f32",
    "k_axpy_f64",
    "k_relu_bwd_f32",
    "k_relu_bwd_f64",
    "k_leaky_relu_bwd_f32",
    "k_leaky_relu_bwd_f64",
    "k_elu_bwd_f32",
    "k_elu_bwd_f64",
    "k_gelu_bwd_f32",
    "k_gelu_bwd_f64",
    "k_abs_bwd_f32",
    "k_abs_bwd_f64",
    "k_expand_f32",
    "k_expand_f64",
    "k_mse_sum_fwd_f32",
    "k_mse_sum_fwd_f64",
    "k_mse_sum_bwd_f32",
    "k_mse_sum_bwd_f64",
    "k_multi_axpy3_f32",
    "k_multi_axpy3_f64",
];

pub(super) fn compile_all_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
    let ptx = nvrtc::compile_ptx_with_opts(
        kernels_cu::KERNELS,
        nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        },
    )?;

    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled PTX data as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };

    let mut map = ctx
        .kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &name in KERNEL_NAMES {
        let c_fn = CString::new(name).map_err(|_| CudaError::NullPtr)?;
        // SAFETY: getting function handle from loaded module.
        let func = unsafe { result::module::get_function(module, c_fn)? };
        map.insert(
            name.to_owned(),
            KernelEntry {
                func,
                _module: module,
            },
        );
    }
    Ok(())
}

const WMMA_KERNEL_NAMES: &[&str] = &["k_matmul_wmma_f16"];

pub(super) fn compile_wmma_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
    let src = kernels_cu::WMMA_KERNELS;
    if src.is_empty() {
        return Ok(());
    }

    let ptx = nvrtc::compile_ptx_with_opts(
        src,
        nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        },
    )?;

    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled WMMA PTX as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };

    let mut map = ctx
        .kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &name in WMMA_KERNEL_NAMES {
        let c_fn = CString::new(name).map_err(|_| CudaError::NullPtr)?;
        // SAFETY: getting function handle from loaded WMMA module.
        let func = unsafe { result::module::get_function(module, c_fn)? };
        map.insert(
            name.to_owned(),
            KernelEntry {
                func,
                _module: module,
            },
        );
    }
    Ok(())
}

pub(super) fn get_kernel(ctx: &CudaCtx, name: &str) -> CudaResult<CUfunction> {
    let map = ctx
        .kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(name)
        .map(|e| e.func)
        .ok_or_else(|| CudaError::KernelNotFound(name.to_owned()))
}

pub(super) fn cuda_grid_1d<T: Scalar>(n: usize) -> u32 {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    }
}

pub(super) fn launch_unary<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
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
            &format!("CUDA launch {name}"),
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(super) fn launch_binary<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    op: &str,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
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
            &format!("CUDA launch {name}"),
        );
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

    #[inline]
    fn to_vec_async<T: Scalar>(a: &CudaStorage<T>) -> Vec<T> {
        cuda_to_vec_async(a).or_panic("CUDA D2H async")
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
}

impl crate::backend::BackendReduce for crate::backend::Cuda {
    gpu_common::rtc_reduce_impl! {
        CudaStorage; sum_all=cuda_sum_all, max_all=cuda_max_all, min_all=cuda_min_all,
        argmax_all=cuda_argmax_all, argmin_all=cuda_argmin_all, axis_reduce=cuda_axis_reduce,
        cumsum_cumprod=cuda_cumsum_cumprod, prod_all=cuda_prod_all,
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
        if TypeId::of::<T>() == TypeId::of::<f32>() && epilogue_id <= 1 {
            let epilogue = if epilogue_id == 0 {
                Epilogue::Relu
            } else {
                Epilogue::Gelu
            };
            // SAFETY: TypeId check above guarantees T == f32 at runtime.
            let (a_f32, b_f32) = unsafe {
                (
                    &*(a as *const CudaStorage<T> as *const CudaStorage<f32>),
                    &*(b as *const CudaStorage<T> as *const CudaStorage<f32>),
                )
            };
            let mut out_f32 = cuda_zeros::<f32>(a_f32.nrows, b_f32.ncols);
            match cuda_matmul_epilogue(&mut out_f32, a_f32, b_f32, epilogue, None) {
                Ok(()) => {
                    // SAFETY: T == f32 is verified by TypeId above.
                    unsafe { std::mem::transmute::<CudaStorage<f32>, CudaStorage<T>>(out_f32) }
                }
                Err(_) => cuda_matmul_epilogue_fallback(a, b, epilogue_id),
            }
        } else {
            cuda_matmul_epilogue_fallback(a, b, epilogue_id)
        }
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
}

impl crate::backend::BackendNN for crate::backend::Cuda {
    gpu_common::gpu_unary_ops!(CudaStorage; silu, mish, hardswish);
    #[inline]
    fn relu_backward<T: Scalar>(g: &CudaStorage<T>, x: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(g, x, "relu_bwd")
    }
    #[inline]
    fn leaky_relu_backward<T: Scalar>(
        g: &CudaStorage<T>,
        x: &CudaStorage<T>,
        _alpha: T,
    ) -> CudaStorage<T> {
        launch_binary(g, x, "leaky_relu_bwd")
    }
    #[inline]
    fn elu_backward<T: Scalar>(
        g: &CudaStorage<T>,
        x: &CudaStorage<T>,
        _alpha: T,
    ) -> CudaStorage<T> {
        launch_binary(g, x, "elu_bwd")
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
        batch_norm_train=cuda_batch_norm_train, cross_entropy_fused=cuda_cross_entropy_fused,
        sdpa=cuda_sdpa, embedding=cuda_embedding, max_pool2d=cuda_max_pool2d,
        max_pool2d_with_idx=cuda_max_pool2d_with_idx, avg_pool2d=cuda_avg_pool2d,
        adaptive_avg_pool2d=cuda_adaptive_avg_pool2d, conv2d=cuda_conv2d, conv1d=cuda_conv1d,
        conv3d=cuda_conv3d, conv_transpose2d=cuda_conv_transpose2d,
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
            .map(|(inputs, expr, n_in, up)| MegaFuseOp {
                inputs: inputs.clone(),
                gpu_expr: expr.clone(),
                n_inputs: *n_in,
                uses_prev: *up,
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
