use crate::gpu_common;
use crate::gpu_common::common::rtc::EnsureCache;

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
    "k_group_norm_f32",
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
    "k_group_norm_f64",
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
];

pub(super) fn compile_all_kernels(ctx: &HipCtx) -> HipResult<()> {
    let src = CString::new(kernels_cu::KERNELS).map_err(|_| HipError::NullPtr)?;

    // hiprtc compilation
    let mut prog: hip::hiprtcProgram = core::ptr::null_mut();
    let name = CString::new("nabla_kernels").map_err(|_| HipError::NullPtr)?;
    // SAFETY: creating hiprtc program from source string.
    let err = unsafe {
        hip::hiprtcCreateProgram(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    if err != hip::hiprtcResult::HIPRTC_SUCCESS {
        return Err(HipError::Rtc(format!("hiprtcCreateProgram: {err:?}")));
    }

    // SAFETY: compiling the hiprtc program with no extra options.
    let err = unsafe { hip::hiprtcCompileProgram(prog, 0, core::ptr::null_mut()) };
    if err != hip::hiprtcResult::HIPRTC_SUCCESS {
        return Err(HipError::Rtc(format!("hiprtcCompileProgram: {err:?}")));
    }

    let mut code_size: usize = 0;
    // SAFETY: querying compiled code size.
    let err = unsafe { hip::hiprtcGetCodeSize(prog, &mut code_size) };
    if err != hip::hiprtcResult::HIPRTC_SUCCESS {
        return Err(HipError::Rtc(format!("hiprtcGetCodeSize: {err:?}")));
    }

    let mut code = vec![0u8; code_size];
    // SAFETY: retrieving compiled code into properly-sized buffer.
    let err = unsafe { hip::hiprtcGetCode(prog, code.as_mut_ptr().cast()) };
    if err != hip::hiprtcResult::HIPRTC_SUCCESS {
        return Err(HipError::Rtc(format!("hiprtcGetCode: {err:?}")));
    }

    // SAFETY: destroying the hiprtc program after extracting code.
    unsafe { hip::hiprtcDestroyProgram(&mut prog) };

    // Load module from compiled code
    let mut module: hip::hipModule_t = core::ptr::null_mut();
    // SAFETY: loading compiled GPU code as a HIP module.
    check(unsafe { hip::hipModuleLoadData(&mut module, code.as_ptr().cast()) })?;

    let mut map = ctx
        .dyn_kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: we are inside OnceLock init — single-threaded, so mutating kernel_funcs is safe.
    let ctx_ptr = ctx as *const HipCtx as *mut HipCtx;
    for &kname in KERNEL_NAMES {
        let c_fn = CString::new(kname).map_err(|_| HipError::NullPtr)?;
        let mut func: hip::hipFunction_t = core::ptr::null_mut();
        // SAFETY: getting function handle from loaded module.
        check(unsafe { hip::hipModuleGetFunction(&mut func, module, c_fn.as_ptr()) })?;
        // Populate flat array for O(1) hot-path lookup
        if let Some(kid) = KernelId::from_name(kname) {
            // SAFETY: ctx_ptr is valid and we are the only writer during init.
            unsafe {
                (*ctx_ptr).kernel_funcs[kid as usize] = SyncFn(func);
            }
        }
        // Also store in dyn_kernels HashMap for fuse/mega dynamic kernel lookup
        map.insert(
            kname.to_owned(),
            KernelEntry {
                func,
                _module: module,
            },
        );
    }
    Ok(())
}

#[inline(always)]
pub(super) fn get_kernel_by_id(ctx: &HipCtx, id: KernelId) -> hip::hipFunction_t {
    ctx.kernel_funcs[id as usize].0
}

pub(super) fn get_kernel(ctx: &HipCtx, name: &str) -> HipResult<hip::hipFunction_t> {
    let map = ctx
        .dyn_kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(name)
        .map(|e| e.func)
        .ok_or_else(|| HipError::KernelNotFound(name.to_owned()))
}

pub(super) fn hip_launch(
    func: hip::hipFunction_t,
    grid: [u32; 3],
    block: [u32; 3],
    args: &mut [*mut c_void],
) {
    // SAFETY: launching HIP kernel with caller-provided valid arguments.
    let err = unsafe {
        hip::hipModuleLaunchKernel(
            func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            0,
            core::ptr::null_mut(), // default stream
            args.as_mut_ptr(),
            core::ptr::null_mut(),
        )
    };
    if err != hip::hipError_t::hipSuccess {
        panic!("HIP launch failed: {err:?}");
    }
    // No sync here — let kernels run asynchronously on the default stream.
    // Ordering is guaranteed within the same stream; host readback (hipMemcpy D2H)
    // implicitly synchronizes.
}

pub(super) fn hip_launch_smem(
    func: hip::hipFunction_t,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    args: &mut [*mut c_void],
) {
    let err = unsafe {
        hip::hipModuleLaunchKernel(
            func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            core::ptr::null_mut(),
            args.as_mut_ptr(),
            core::ptr::null_mut(),
        )
    };
    if err != hip::hipError_t::hipSuccess {
        panic!("HIP launch (smem) failed: {err:?}");
    }
}

pub(super) fn hip_prepare_launch<T: Scalar>(
    n: usize,
    op: &str,
) -> (hip::hipFunction_t, HipBuffer, u32) {
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = hip_or_panic(
        HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()),
        "HIP alloc",
    );
    (func, out_buf, n as u32)
}

pub(super) fn launch_unary<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let n = a.n();
    let (func, out_buf, n_u32) = hip_prepare_launch::<T>(n, op);
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

pub(super) fn launch_binary<T: Scalar>(
    a: &HipStorage<T>,
    b: &HipStorage<T>,
    op: &str,
) -> HipStorage<T> {
    let n = a.n();
    let (func, out_buf, n_u32) = hip_prepare_launch::<T>(n, op);
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&b.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

impl crate::backend::private::Sealed for crate::backend::Hip {}

impl crate::backend::BackendCore for crate::backend::Hip {
    type Storage<T: Scalar> = HipStorage<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
        hip_zeros(nrows, ncols)
    }
    fn empty<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
        hip_empty(nrows, ncols)
    }

    gpu_common::rtc_core_impl! {
        HipStorage; fill=hip_fill, from_fn=hip_from_fn, from_vec_async=hip_from_vec_async,
        get=hip_get, set=hip_set, transpose=hip_transpose, scale=hip_scale,
        clone_storage=hip_clone,
    }

    #[inline]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> HipStorage<T> {
        let buf = hip_or_panic(HipBuffer::from_host(&data), "HIP upload");
        HipStorage::new_cached(nrows, ncols, buf, data)
    }

    #[inline]
    fn prefetch<T: Scalar>(storage: &HipStorage<T>) {
        storage.ensure_cache();
    }

    gpu_common::gpu_binary_ops!(HipStorage; add, sub);

    #[inline]
    fn axpy_inplace<T: Scalar>(y: &mut HipStorage<T>, alpha: T, x: &HipStorage<T>) {
        hip_axpy_inplace(y, alpha, x);
    }

    #[inline]
    fn expand_into<T: Scalar>(
        out: &mut HipStorage<T>,
        src: &HipStorage<T>,
        src_rows: usize,
        src_cols: usize,
    ) {
        hip_expand(out, src, src_rows, src_cols);
    }
}

impl crate::backend::BackendMath for crate::backend::Hip {
    gpu_common::gpu_unary_ops!(HipStorage; exp, ln, log1p, sin, cos, tan, tanh, sqrt, abs, recip, erf, ceil, floor, round, asin, acos, atan, sinh, cosh, asinh, acosh, atanh, log2, log10);
    gpu_common::gpu_binary_ops!(HipStorage; emul, ediv, atan2);
    gpu_common::rtc_math_impl!(HipStorage; powf=hip_powf,);
}

impl crate::backend::BackendReduce for crate::backend::Hip {
    gpu_common::rtc_reduce_impl! {
        HipStorage; sum_all=hip_sum_all, max_all=hip_max_all, min_all=hip_min_all,
        argmax_all=hip_argmax_all, argmin_all=hip_argmin_all, axis_reduce=hip_axis_reduce,
        cumsum_cumprod=hip_cumsum_cumprod, prod_all=hip_prod_all,
    }
}

impl crate::backend::BackendShape for crate::backend::Hip {}

impl crate::backend::BackendBlas for crate::backend::Hip {
    #[inline]
    fn matmul_into<T: Scalar>(out: &mut HipStorage<T>, a: &HipStorage<T>, b: &HipStorage<T>) {
        hip_matmul(out, a, b);
    }
}

impl crate::backend::BackendNN for crate::backend::Hip {
    gpu_common::gpu_unary_ops!(HipStorage; silu, mish, hardswish);
    #[inline]
    fn relu_backward<T: Scalar>(g: &HipStorage<T>, x: &HipStorage<T>) -> HipStorage<T> {
        launch_binary(g, x, "relu_bwd")
    }
    #[inline]
    fn leaky_relu_backward<T: Scalar>(
        g: &HipStorage<T>,
        x: &HipStorage<T>,
        _alpha: T,
    ) -> HipStorage<T> {
        launch_binary(g, x, "leaky_relu_bwd")
    }
    #[inline]
    fn elu_backward<T: Scalar>(g: &HipStorage<T>, x: &HipStorage<T>, _alpha: T) -> HipStorage<T> {
        launch_binary(g, x, "elu_bwd")
    }
    #[inline]
    fn gelu_backward<T: Scalar>(g: &HipStorage<T>, x: &HipStorage<T>) -> HipStorage<T> {
        launch_binary(g, x, "gelu_bwd")
    }
    #[inline]
    fn abs_backward<T: Scalar>(g: &HipStorage<T>, x: &HipStorage<T>) -> HipStorage<T> {
        launch_binary(g, x, "abs_bwd")
    }
    gpu_common::rtc_nn_impl! {
        HipStorage; softmax=hip_softmax, layer_norm=hip_layer_norm, rms_norm=hip_rms_norm,
        group_norm=hip_group_norm, batch_norm_train=hip_batch_norm_train,
        cross_entropy_fused=hip_cross_entropy_fused, sdpa=hip_sdpa, embedding=hip_embedding,
        max_pool2d=hip_max_pool2d, max_pool2d_with_idx=hip_max_pool2d_with_idx,
        avg_pool2d=hip_avg_pool2d, adaptive_avg_pool2d=hip_adaptive_avg_pool2d,
        conv2d=hip_conv2d, conv1d=hip_conv1d, conv3d=hip_conv3d,
        conv_transpose2d=hip_conv_transpose2d,
    }
}

impl crate::backend::BackendFusion for crate::backend::Hip {
    fn fuse_launch<T: Scalar>(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        _cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reg_estimate: usize,
    ) -> HipStorage<T> {
        hip_fuse_launch::<T>(
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
    ) -> Vec<HipStorage<T>> {
        let mega_ops: Vec<MegaFuseOp> = ops
            .iter()
            .map(|(inputs, expr, n_in, up)| MegaFuseOp {
                inputs: inputs.clone(),
                gpu_expr: expr.clone(),
                n_inputs: *n_in,
                uses_prev: *up,
            })
            .collect();
        hip_mega_fuse_launch::<T>(&mega_ops, nrows, ncols, kernel_hash)
    }
}

pub(super) fn hip_fuse_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
    reg_estimate: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_fused_{kernel_hash}_{tsuf}");

    // Check cache, compile if missing
    {
        let map = ctx
            .dyn_kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let src_str = gpu_common::fuse_kernel_source(
                gpu_expr,
                n_inputs,
                type_name,
                &kernel_name,
                reg_estimate,
                false,
            );
            let c_src = CString::new(src_str).unwrap_or_else(|_| panic!("null in source"));
            let prog_name = CString::new("nabla_fuse").unwrap_or_else(|_| panic!("null"));

            let mut prog: hip::hiprtcProgram = core::ptr::null_mut();
            let err = unsafe {
                hip::hiprtcCreateProgram(
                    &mut prog,
                    c_src.as_ptr(),
                    prog_name.as_ptr(),
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcCreateProgram for fuse: {err:?}");
            }
            let err = unsafe { hip::hiprtcCompileProgram(prog, 0, core::ptr::null_mut()) };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcCompileProgram for fuse: {err:?}");
            }
            let mut code_size: usize = 0;
            let err = unsafe { hip::hiprtcGetCodeSize(prog, &mut code_size) };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcGetCodeSize: {err:?}");
            }
            let mut code = vec![0u8; code_size];
            let err = unsafe { hip::hiprtcGetCode(prog, code.as_mut_ptr().cast()) };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcGetCode: {err:?}");
            }
            unsafe { hip::hiprtcDestroyProgram(&mut prog) };

            let mut module: hip::hipModule_t = core::ptr::null_mut();
            check(unsafe { hip::hipModuleLoadData(&mut module, code.as_ptr().cast()) })
                .unwrap_or_else(|e| panic!("HIP module load: {e}"));
            let c_fn = CString::new(kernel_name.as_str()).unwrap_or_else(|_| panic!("null"));
            let mut func: hip::hipFunction_t = core::ptr::null_mut();
            check(unsafe { hip::hipModuleGetFunction(&mut func, module, c_fn.as_ptr()) })
                .unwrap_or_else(|e| panic!("HIP get_function: {e}"));

            let mut map = ctx
                .dyn_kernels
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(
                kernel_name.clone(),
                KernelEntry {
                    func,
                    _module: module,
                },
            );
        }
    }

    let func = get_kernel(ctx, &kernel_name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };

    // SAFETY: input pointers are valid HipStorage<T>
    let input_ptrs: Vec<*mut c_void> = inputs
        .iter()
        .map(|&p| {
            let storage = unsafe { &*(p as *const HipStorage<T>) };
            storage.buf.ptr
        })
        .collect();

    let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + 2);
    for ptr in &input_ptrs {
        args.push(ptr as *const *mut c_void as *mut c_void);
    }
    args.push((&out_buf.ptr as *const *mut c_void).cast_mut().cast());
    args.push((&n_u32 as *const u32).cast_mut().cast());

    hip_launch(func, [grid, 1, 1], [BLOCK_SIZE, 1, 1], &mut args);
    HipStorage::new(nrows, ncols, out_buf)
}

pub(crate) struct MegaFuseOp {
    /// Raw pointers to input HipStorage buffers (as `*const u8`).
    pub inputs: Vec<*const u8>,
    /// GPU C expression, using `opK_inN[i]` or `__NABLA_PREV__` placeholders.
    pub gpu_expr: String,
    /// Number of logical inputs for this operation (includes the `prev` slot when `uses_prev`).
    pub n_inputs: usize,
    /// When `true`, the first logical input (`in0`) is the previous op's output register.
    /// No global-memory pointer is passed for that slot; the kernel reads `op{k-1}_r` directly.
    pub uses_prev: bool,
}

pub(crate) fn hip_mega_fuse_launch<T: Scalar>(
    ops: &[MegaFuseOp],
    nrows: usize,
    ncols: usize,
    kernel_hash: &str,
) -> Vec<HipStorage<T>> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_mega_{kernel_hash}_{tsuf}");

    // Compile mega-kernel (JIT + cache)
    {
        let map = ctx
            .dyn_kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let op_descs: Vec<(String, usize)> = ops
                .iter()
                .map(|op| (op.gpu_expr.clone(), op.n_inputs))
                .collect();
            let op_uses_prev: Vec<bool> = ops.iter().map(|op| op.uses_prev).collect();
            let src_str = gpu_common::mega_fuse_kernel_source(
                &op_descs,
                &op_uses_prev,
                type_name,
                &kernel_name,
                false,
            );
            let c_src = CString::new(src_str).unwrap_or_else(|_| panic!("null in source"));
            let prog_name = CString::new("nabla_mega_fuse").unwrap_or_else(|_| panic!("null"));

            let mut prog: hip::hiprtcProgram = core::ptr::null_mut();
            let err = unsafe {
                hip::hiprtcCreateProgram(
                    &mut prog,
                    c_src.as_ptr(),
                    prog_name.as_ptr(),
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcCreateProgram for mega-fuse: {err:?}");
            }
            let err = unsafe { hip::hiprtcCompileProgram(prog, 0, core::ptr::null_mut()) };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcCompileProgram for mega-fuse: {err:?}");
            }
            let mut code_size: usize = 0;
            let err = unsafe { hip::hiprtcGetCodeSize(prog, &mut code_size) };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcGetCodeSize: {err:?}");
            }
            let mut code = vec![0u8; code_size];
            let err = unsafe { hip::hiprtcGetCode(prog, code.as_mut_ptr().cast()) };
            if err != hip::hiprtcResult::HIPRTC_SUCCESS {
                panic!("hiprtcGetCode: {err:?}");
            }
            unsafe { hip::hiprtcDestroyProgram(&mut prog) };

            let mut module: hip::hipModule_t = core::ptr::null_mut();
            check(unsafe { hip::hipModuleLoadData(&mut module, code.as_ptr().cast()) })
                .unwrap_or_else(|e| panic!("HIP module load: {e}"));
            let c_fn = CString::new(kernel_name.as_str()).unwrap_or_else(|_| panic!("null"));
            let mut func: hip::hipFunction_t = core::ptr::null_mut();
            check(unsafe { hip::hipModuleGetFunction(&mut func, module, c_fn.as_ptr()) })
                .unwrap_or_else(|e| panic!("HIP get_function: {e}"));

            let mut map = ctx
                .dyn_kernels
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(
                kernel_name.clone(),
                KernelEntry {
                    func,
                    _module: module,
                },
            );
        }
    }

    let func = get_kernel(ctx, &kernel_name).unwrap_or_else(|e| panic!("{e}"));

    // Allocate output buffers
    let out_bufs: Vec<HipBuffer> = (0..ops.len())
        .map(|_| {
            HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
                .unwrap_or_else(|e| panic!("HIP alloc: {e}"))
        })
        .collect();

    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };

    // Collect input device pointers
    let input_ptrs: Vec<Vec<*mut c_void>> = ops
        .iter()
        .map(|op| {
            op.inputs
                .iter()
                .map(|&p| {
                    let storage = unsafe { &*(p as *const HipStorage<T>) };
                    storage.buf.ptr
                })
                .collect()
        })
        .collect();

    // Build kernel argument array.  All input pointers are passed for every op;
    // uses_prev ops receive the same n_inputs pointers as non-prev ops since the
    // __NABLA_PREV__ sentinel resolves to a register, not an inN pointer.
    let total_args = ops.iter().map(|op| op.n_inputs + 1).sum::<usize>() + 1;
    let mut args: Vec<*mut c_void> = Vec::with_capacity(total_args);
    for (op_idx, op) in ops.iter().enumerate() {
        for j in 0..op.n_inputs {
            args.push(&input_ptrs[op_idx][j] as *const *mut c_void as *mut c_void);
        }
        args.push(&out_bufs[op_idx].ptr as *const *mut c_void as *mut c_void);
    }
    args.push((&n_u32 as *const u32).cast_mut().cast());

    hip_launch(func, [grid, 1, 1], [BLOCK_SIZE, 1, 1], &mut args);

    out_bufs
        .into_iter()
        .map(|buf| HipStorage::new(nrows, ncols, buf))
        .collect()
}
