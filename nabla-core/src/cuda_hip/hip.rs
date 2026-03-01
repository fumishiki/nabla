// hip_backend.rs — HIP backend via hip-runtime-sys bindings + hiprtc JIT compilation.
//
// Design mirrors cuda_backend.rs:
//   - HipCtx (OnceLock singleton): device init + hiprtc module cache.
//   - HipStorage<T> = RtcStorage<HipBuffer, T>: shared GPU storage with lazy host_cache.
//   - Same CUDA C kernel source (CUDA/HIP source-compatible for compute kernels).

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::{Mutex, OnceLock};

use hip_runtime_sys as hip;

use crate::gpu_common::{
    self, EnsureCache, KERNEL_COUNT, KernelId, LARGE_ALLOC_SIZE, MemoryPool, RtcStorage,
    SMALL_ALLOC_SIZE, SMALL_LARGE_BOUNDARY, grid_1d, kernel_id, lock_or_recover, round_size,
    type_suffix,
};
use crate::kernels_cu::{self, BLOCK_SIZE};
use crate::scalar::Scalar;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) enum HipError {
    Runtime(hip::hipError_t),
    Rtc(String),
    KernelNotFound(String),
    NullPtr,
}

impl core::fmt::Display for HipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Runtime(e) => write!(f, "HIP runtime: {e:?}"),
            Self::Rtc(s) => write!(f, "hiprtc: {s}"),
            Self::KernelNotFound(s) => write!(f, "kernel not found: {s}"),
            Self::NullPtr => write!(f, "null pointer"),
        }
    }
}

type HipResult<T> = Result<T, HipError>;

fn check(err: hip::hipError_t) -> HipResult<()> {
    if err == hip::hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(HipError::Runtime(err))
    }
}

#[inline]
fn hip_or_panic<T>(result: HipResult<T>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

// ── GPU buffer (RAII) ────────────────────────────────────────────────────────

type HipPool = MemoryPool<*mut c_void>;

fn hip_free(ptr: *mut c_void, _size: usize) {
    unsafe {
        let _ = hip::hipFree(ptr);
    }
}

pub struct HipBuffer {
    pub(crate) ptr: *mut c_void,
    size: usize,
    alloc_size: usize,
    pooled: bool,
}

// SAFETY: HipBuffer wraps a raw GPU device pointer. The pointer is not
// dereferenced on the host — it is only passed to HIP API calls. Access
// to the buffer is single-owner (no aliasing). Send+Sync is safe because
// HIP API calls that use device pointers are thread-safe.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl HipBuffer {
    #[inline]
    fn empty(size_bytes: usize) -> Option<Self> {
        (size_bytes == 0).then(|| Self {
            ptr: core::ptr::null_mut(),
            size: 0,
            alloc_size: 0,
            pooled: false,
        })
    }

    fn alloc_from_pool(size_bytes: usize) -> HipResult<(*mut c_void, usize)> {
        if size_bytes == 0 {
            return Ok((core::ptr::null_mut(), 0));
        }
        let ctx = get_ctx();
        let mut pool = ctx
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((ptr, size_class)) = pool.try_alloc(size_bytes) {
            pool.allocated_bytes += size_class;
            return Ok((ptr, size_class));
        }
        let rounded = round_size(size_bytes);
        let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
            rounded.max(SMALL_ALLOC_SIZE)
        } else {
            rounded.max(LARGE_ALLOC_SIZE)
        };
        drop(pool);
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check(unsafe { hip::hipMalloc(&mut ptr, alloc_sz) })?;
        let mut pool = ctx
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if alloc_sz > rounded {
            let split_ptr = unsafe { ptr.byte_add(rounded) };
            pool.release(split_ptr, alloc_sz - rounded);
            pool.allocated_bytes += rounded;
            return Ok((ptr, rounded));
        }
        pool.allocated_bytes += alloc_sz;
        Ok((ptr, alloc_sz))
    }

    fn alloc_zeros(size_bytes: usize) -> HipResult<Self> {
        if let Some(buf) = Self::empty(size_bytes) {
            return Ok(buf);
        }
        let (ptr, alloc_size) = Self::alloc_from_pool(size_bytes)?;
        check(unsafe { hip::hipMemset(ptr, 0, alloc_size) })?;
        Ok(Self {
            ptr,
            size: size_bytes,
            alloc_size,
            pooled: true,
        })
    }

    fn from_host<T: Scalar>(data: &[T]) -> HipResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if let Some(buf) = Self::empty(bytes) {
            return Ok(buf);
        }
        let (ptr, alloc_size) = Self::alloc_from_pool(bytes)?;
        check(unsafe {
            hip::hipMemcpy(
                ptr,
                data.as_ptr().cast(),
                bytes,
                hip::hipMemcpyKind::hipMemcpyHostToDevice,
            )
        })?;
        Ok(Self {
            ptr,
            size: bytes,
            alloc_size,
            pooled: true,
        })
    }

    fn copy_to_host<T: Scalar>(&self, out: &mut [T]) -> HipResult<()> {
        let bytes = core::mem::size_of_val(out);
        if bytes > 0 {
            // hipMemcpy D2H is synchronous — implicitly waits for all prior ops.
            // SAFETY: out is properly sized and T is POD.
            check(unsafe {
                hip::hipMemcpy(
                    out.as_mut_ptr().cast(),
                    self.ptr,
                    bytes,
                    hip::hipMemcpyKind::hipMemcpyDeviceToHost,
                )
            })?;
        }
        Ok(())
    }

    /// Copy a single element at byte offset from device to host.
    fn copy_element<T: Scalar>(&self, byte_offset: usize) -> HipResult<T> {
        let mut val = T::zero();
        // SAFETY: reading sizeof::<T> bytes from device at ptr+offset.
        let src = unsafe { self.ptr.byte_add(byte_offset) };
        check(unsafe {
            hip::hipMemcpy(
                (&mut val as *mut T).cast(),
                src,
                core::mem::size_of::<T>(),
                hip::hipMemcpyKind::hipMemcpyDeviceToHost,
            )
        })?;
        Ok(val)
    }

    /// Wrap an external device pointer as a **borrowed** (non-freeing) `HipBuffer`.
    ///
    /// # Safety
    /// - `ptr` must be a valid HIP device pointer with at least `size_bytes` allocated.
    /// - The pointer must outlive this buffer; `hipFree` is NOT called on drop.
    pub(crate) unsafe fn borrow_ptr(ptr: *mut c_void, size_bytes: usize) -> Self {
        // alloc_size = 0 signals "borrowed" — Drop will skip freeing (see Drop impl).
        Self {
            ptr,
            size: size_bytes,
            alloc_size: 0,
            pooled: false,
        }
    }

    /// Non-blocking H2D: allocate normally, copy on separate copy stream,
    /// synchronize via HIP event so the default stream waits for the transfer.
    fn from_host_nonblocking<T: Scalar>(
        copy_stream: hip::hipStream_t,
        data: &[T],
    ) -> HipResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                size: 0,
                alloc_size: 0,
                pooled: false,
            });
        }
        let (ptr, alloc_size) = Self::alloc_from_pool(bytes)?;
        // Copy on the copy stream (non-blocking w.r.t. default stream)
        check(unsafe {
            hip::hipMemcpyAsync(
                ptr,
                data.as_ptr().cast(),
                bytes,
                hip::hipMemcpyKind::hipMemcpyHostToDevice,
                copy_stream,
            )
        })?;
        // Record event on copy stream, make default stream wait
        let mut event: hip::hipEvent_t = core::ptr::null_mut();
        check(unsafe {
            hip::hipEventCreateWithFlags(&mut event, 0x02 /* hipEventDisableTiming */)
        })?;
        check(unsafe { hip::hipEventRecord(event, copy_stream) })?;
        // Default stream (null) waits on the event
        check(unsafe { hip::hipStreamWaitEvent(core::ptr::null_mut(), event, 0) })?;
        // Destroy event — safe because hipStreamWaitEvent captures the dependency
        unsafe {
            let _ = hip::hipEventDestroy(event);
        }
        Ok(Self {
            ptr,
            size: bytes,
            alloc_size,
            pooled: true,
        })
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.alloc_size > 0 {
            // alloc_size == 0 → borrowed pointer (borrow_ptr), do NOT free.
            if self.pooled {
                let ctx = get_ctx();
                let mut pool = ctx
                    .pool
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes = pool.allocated_bytes.saturating_sub(self.alloc_size);
                pool.release(self.ptr, self.alloc_size);
                pool.maybe_gc(hip_free);
            } else {
                unsafe {
                    let _ = hip::hipFree(self.ptr);
                }
            }
        }
    }
}

// ── Storage type alias ───────────────────────────────────────────────────────

/// Row-major HIP-backed matrix storage.
pub type HipStorage<T> = RtcStorage<HipBuffer, T>;

// SAFETY: HipBuffer is Send+Sync (raw GPU pointer). Mutex<Option<Vec<T>>>
// is Send+Sync when T: Send+Sync, which Scalar guarantees.
unsafe impl<T: Scalar> Send for HipStorage<T> {}
unsafe impl<T: Scalar> Sync for HipStorage<T> {}

impl<T: Scalar> EnsureCache for HipStorage<T> {
    fn ensure_cache(&self) {
        let mut guard = lock_or_recover(&self.host_cache);
        if guard.is_none() {
            let mut host = vec![T::zero(); self.n()];
            if let Err(e) = self.buf.copy_to_host(&mut host) {
                panic!("HIP readback failed: {e}");
            }
            *guard = Some(host);
        }
    }
}

// ── Kernel cache ─────────────────────────────────────────────────────────────

struct KernelEntry {
    func: hip::hipFunction_t,
    _module: hip::hipModule_t,
}

// SAFETY: hipFunction_t and hipModule_t are opaque handles (pointers) to
// GPU driver objects. They are thread-safe to store and pass to HIP API
// calls from any thread — the HIP runtime serializes access internally.
unsafe impl Send for KernelEntry {}
unsafe impl Sync for KernelEntry {}

// Wrapper to make hipFunction_t Send+Sync (HIP function handles are thread-safe)
#[derive(Clone, Copy)]
struct SyncFn(hip::hipFunction_t);
// SAFETY: hipFunction_t is an opaque handle — thread-safe to store/use.
unsafe impl Send for SyncFn {}
unsafe impl Sync for SyncFn {}

// ── HipCtx singleton ─────────────────────────────────────────────────────────

struct HipCtx {
    /// Pre-compiled kernel functions indexed by `KernelId`. Lock-free O(1) access.
    kernel_funcs: [SyncFn; KERNEL_COUNT],
    /// Dynamic kernels (fuse/mega) still use HashMap for JIT-compiled kernels.
    dyn_kernels: Mutex<HashMap<String, KernelEntry>>,
    pool: Mutex<HipPool>,
    /// Separate stream for H2D/D2H transfers (multi-stream pipeline).
    copy_stream: hip::hipStream_t,
}

// SAFETY: hipStream_t is an opaque handle to a HIP stream. HIP API calls
// are thread-safe, and the stream handle is only passed to HIP functions.
unsafe impl Send for HipCtx {}
unsafe impl Sync for HipCtx {}

fn get_ctx() -> &'static HipCtx {
    static CTX: OnceLock<HipCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        // SAFETY: initializing HIP runtime on device 0.
        let err = unsafe { hip::hipSetDevice(0) };
        if err != hip::hipError_t::hipSuccess {
            panic!("HIP device 0 init failed: {err:?}");
        }
        let mut copy_stream: hip::hipStream_t = core::ptr::null_mut();
        let err = unsafe { hip::hipStreamCreate(&mut copy_stream) };
        if err != hip::hipError_t::hipSuccess {
            panic!("HIP copy stream creation failed: {err:?}");
        }
        let hip_ctx = HipCtx {
            kernel_funcs: [SyncFn(core::ptr::null_mut()); KERNEL_COUNT],
            dyn_kernels: Mutex::new(HashMap::new()),
            pool: Mutex::new(HipPool::new()),
            copy_stream,
        };
        if let Err(e) = compile_all_kernels(&hip_ctx) {
            panic!("HIP kernel compilation failed: {e}");
        }
        hip_ctx
    })
}

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
    // activations f32
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
    // row-wise f32
    "k_softmax_f32",
    "k_layer_norm_f32",
    "k_rms_norm_f32",
    "k_sum_axis1_f32",
    "k_max_axis1_f32",
    "k_embedding_f32",
    // cumulative ops f32
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
    // activations f64
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
    // row-wise f64
    "k_softmax_f64",
    "k_layer_norm_f64",
    "k_rms_norm_f64",
    "k_sum_axis1_f64",
    "k_max_axis1_f64",
    "k_embedding_f64",
    // cumulative ops f64
    "k_cumsum_axis1_f64",
    "k_cumprod_axis1_f64",
    // product reduction
    "k_prod_partial_f32",
    "k_prod_partial_f64",
    // pooling f32
    "k_max_pool2d_f32",
    "k_max_pool2d_with_idx_f32",
    "k_avg_pool2d_f32",
    "k_adaptive_avg_pool2d_f32",
    // pooling f64
    "k_max_pool2d_f64",
    "k_max_pool2d_with_idx_f64",
    "k_avg_pool2d_f64",
    "k_adaptive_avg_pool2d_f64",
    // im2col f32 + f64
    "k_im2col_f32",
    "k_im2col_f64",
    // batch-norm f32 + f64
    "k_batch_norm_stats_f32",
    "k_batch_norm_fwd_f32",
    "k_batch_norm_stats_f64",
    "k_batch_norm_fwd_f64",
    // cross-entropy f32 + f64
    "k_cross_entropy_f32",
    "k_cross_entropy_f64",
    // FlashAttention-2 f32 + f64
    "k_sdpa_f32",
    "k_sdpa_f64",
    // conv_transpose2d f32 + f64
    "k_conv_transpose2d_f32",
    "k_conv_transpose2d_f64",
];

fn compile_all_kernels(ctx: &HipCtx) -> HipResult<()> {
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
            unsafe { (*ctx_ptr).kernel_funcs[kid as usize] = SyncFn(func); }
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

/// O(1) kernel lookup — no Mutex, no HashMap, no allocation.
#[inline(always)]
fn get_kernel_by_id(ctx: &HipCtx, id: KernelId) -> hip::hipFunction_t {
    ctx.kernel_funcs[id as usize].0
}

/// HashMap-based kernel lookup — used only for fuse/mega dynamic JIT kernels.
fn get_kernel(ctx: &HipCtx, name: &str) -> HipResult<hip::hipFunction_t> {
    let map = ctx
        .dyn_kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(name)
        .map(|e| e.func)
        .ok_or_else(|| HipError::KernelNotFound(name.to_owned()))
}

// ── Launch helper ────────────────────────────────────────────────────────────

fn hip_launch(func: hip::hipFunction_t, grid: [u32; 3], block: [u32; 3], args: &mut [*mut c_void]) {
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

/// Like hip_launch but with explicit shared memory bytes (for Blelloch scan etc.).
fn hip_launch_smem(
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

// ── Launch helpers ───────────────────────────────────────────────────────────

fn hip_prepare_launch<T: Scalar>(n: usize, op: &str) -> (hip::hipFunction_t, HipBuffer, u32) {
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = hip_or_panic(HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()), "HIP alloc");
    (func, out_buf, n as u32)
}

fn launch_unary<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
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

fn launch_binary<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>, op: &str) -> HipStorage<T> {
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

// ── Public operations ────────────────────────────────────────────────────────

pub(crate) fn hip_zeros<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
    let buf = hip_or_panic(
        HipBuffer::alloc_zeros(nrows * ncols * core::mem::size_of::<T>()),
        "HIP alloc",
    );
    HipStorage::new(nrows, ncols, buf)
}

pub(crate) fn hip_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> HipStorage<T> {
    let n = nrows * ncols;
    let data = vec![val; n];
    let buf = hip_or_panic(HipBuffer::from_host(&data), "HIP upload");
    HipStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn hip_from_fn<T: Scalar>(
    nrows: usize,
    ncols: usize,
    mut f: impl FnMut(usize, usize) -> T,
) -> HipStorage<T> {
    let n = nrows * ncols;
    let mut data = Vec::with_capacity(n);
    for r in 0..nrows {
        for c in 0..ncols {
            data.push(f(r, c));
        }
    }
    let buf = hip_or_panic(HipBuffer::from_host(&data), "HIP upload");
    HipStorage::new_cached(nrows, ncols, buf, data)
}

/// Non-blocking H2D upload: data transfer on copy stream overlaps with compute.
pub(crate) fn hip_from_vec_async<T: Scalar>(
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let buf = hip_or_panic(
        HipBuffer::from_host_nonblocking(ctx.copy_stream, &data),
        "HIP async upload",
    );
    HipStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn hip_get<T: Scalar>(s: &HipStorage<T>, r: usize, c: usize) -> T {
    // Fast path: if host cache exists, read from it.
    {
        let guard = lock_or_recover(&s.host_cache);
        if let Some(cache) = guard.as_ref() {
            return cache[r * s.ncols + c];
        }
    }
    // Slow path: single-element D2H (avoid copying entire tensor).
    let byte_offset = (r * s.ncols + c) * core::mem::size_of::<T>();
    hip_or_panic(
        s.buf.copy_element::<T>(byte_offset),
        "HIP single-element readback",
    )
}

pub(crate) fn hip_set<T: Scalar>(s: &mut HipStorage<T>, r: usize, c: usize, v: T) {
    s.invalidate_cache();
    let offset = (r * s.ncols + c) * core::mem::size_of::<T>();
    let src = core::slice::from_ref(&v);
    // SAFETY: uploading single element to correct offset in GPU buffer.
    let dst = unsafe { s.buf.ptr.byte_add(offset) };
    let err = unsafe {
        hip::hipMemcpy(
            dst,
            src.as_ptr().cast(),
            core::mem::size_of::<T>(),
            hip::hipMemcpyKind::hipMemcpyHostToDevice,
        )
    };
    if err != hip::hipError_t::hipSuccess {
        panic!("HIP memcpy failed: {err:?}");
    }
}

pub(crate) fn hip_clone<T: Scalar>(s: &HipStorage<T>) -> HipStorage<T> {
    let bytes = s.n() * core::mem::size_of::<T>();
    let new_buf = hip_or_panic(HipBuffer::alloc_zeros(bytes), "HIP alloc");
    if bytes > 0 {
        // SAFETY: device-to-device copy of same-sized buffers.
        let err = unsafe {
            hip::hipMemcpy(
                new_buf.ptr,
                s.buf.ptr,
                bytes,
                hip::hipMemcpyKind::hipMemcpyDeviceToDevice,
            )
        };
        if err != hip::hipError_t::hipSuccess {
            panic!("HIP d2d copy failed: {err:?}");
        }
    }
    // Don't copy the host cache: the cloned buffer has its own GPU data.
    // ensure_cache() will repopulate lazily if a CPU readback is later needed.
    RtcStorage {
        nrows: s.nrows,
        ncols: s.ncols,
        buf: new_buf,
        host_cache: Mutex::new(None),
    }
}

pub(crate) fn hip_transpose<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("transpose"));
    let out_buf = hip_or_panic(HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()), "HIP alloc");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows as *const u32).cast_mut().cast(),
            (&cols as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(a.ncols, a.nrows, out_buf)
}

pub(crate) fn hip_scale<T: Scalar>(a: &HipStorage<T>, s: T) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("scale"));
    let out_buf = hip_or_panic(HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()), "HIP alloc");
    let n_u32 = n as u32;
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&s as *const T).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn hip_powf<T: Scalar>(a: &HipStorage<T>, p: T) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("powf"));
    let out_buf = hip_or_panic(HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()), "HIP alloc");
    let n_u32 = n as u32;
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&p as *const T).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn hip_matmul<T: Scalar>(out: &mut HipStorage<T>, a: &HipStorage<T>, b: &HipStorage<T>) {
    let ctx = get_ctx();
    out.invalidate_cache();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("matmul"));
    let m = a.nrows as u32;
    let k = a.ncols as u32;
    let n = b.ncols as u32;
    let out_bytes = out.n() * core::mem::size_of::<T>();
    if out_bytes > 0 {
        // SAFETY: zeroing output buffer before matmul accumulation.
        unsafe {
            let _ = hip::hipMemset(out.buf.ptr, 0, out_bytes);
        }
    }
    let grid_x = n.div_ceil(16);
    let grid_y = m.div_ceil(16);
    hip_launch(
        func,
        [grid_x, grid_y, 1],
        [16, 16, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&b.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&m as *const u32).cast_mut().cast(),
            (&k as *const u32).cast_mut().cast(),
            (&n as *const u32).cast_mut().cast(),
        ],
    );
}

// Reductions — delegated to shared gpu_common implementation

pub(crate) fn hip_sum_all<T: Scalar>(a: &HipStorage<T>) -> T {
    gpu_common::rtc_sum_all(a)
}
pub(crate) fn hip_max_all<T: Scalar>(a: &HipStorage<T>) -> T {
    gpu_common::rtc_max_all(a)
}
pub(crate) fn hip_min_all<T: Scalar>(a: &HipStorage<T>) -> T {
    gpu_common::rtc_min_all(a)
}
pub(crate) fn hip_argmax_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) {
    gpu_common::rtc_argmax_all(a)
}
pub(crate) fn hip_argmin_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) {
    gpu_common::rtc_argmin_all(a)
}

// ── Row-wise kernel launch helpers ──────────────────────────────────────────

fn hip_max_pool2d<T: Scalar>(
    a: &HipStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("max_pool2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let (h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, out_h_u, out_w_u, nc_u) = (
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(nc, out_h * out_w, out_buf)
}

fn hip_avg_pool2d<T: Scalar>(
    a: &HipStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("avg_pool2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let (h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, out_h_u, out_w_u, nc_u) = (
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(nc, out_h * out_w, out_buf)
}

fn hip_adaptive_avg_pool2d<T: Scalar>(
    a: &HipStorage<T>,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("adaptive_avg_pool2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let (in_h_u, in_w_u, out_h_u, out_w_u, nc_u) = (
        in_h as u32,
        in_w as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&in_h_u as *const u32).cast_mut().cast(),
            (&in_w_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(nc, out_h * out_w, out_buf)
}

#[allow(clippy::too_many_arguments)]
fn hip_im2col<T: Scalar>(
    input: &HipStorage<T>,
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
) -> HipStorage<T> {
    let k_cols = c_in * kh * kw;
    let out_spatial = out_h * out_w;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("im2col"));
    let col_buf = HipBuffer::alloc_zeros(n * k_cols * out_spatial * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc im2col: {e}"));
    let col_elem = k_cols * out_spatial;
    let grid_x = ((col_elem as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
    hip_launch(
        func,
        [grid_x, n as u32, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&col_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&c_in_u as *const u32).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&dh_u as *const u32).cast_mut().cast(),
            (&dw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n * k_cols, out_spatial, col_buf)
}

#[allow(clippy::too_many_arguments)]
fn hip_im1col<T: Scalar>(
    input: &HipStorage<T>,
    n: usize,
    c_in: usize,
    l: usize,
    kl: usize,
    sl: usize,
    pl: usize,
    dl: usize,
    out_l: usize,
) -> HipStorage<T> {
    let k_cols = c_in * kl;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("im1col"));
    let col_buf = HipBuffer::alloc_zeros(n * k_cols * out_l * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc im1col: {e}"));
    let col_elem = k_cols * out_l;
    let grid_x = ((col_elem as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, l_u, kl_u, sl_u, pl_u, dl_u, out_l_u) = (
        c_in as u32,
        l as u32,
        kl as u32,
        sl as u32,
        pl as u32,
        dl as u32,
        out_l as u32,
    );
    hip_launch(
        func,
        [grid_x, n as u32, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&col_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&c_in_u as *const u32).cast_mut().cast(),
            (&l_u as *const u32).cast_mut().cast(),
            (&kl_u as *const u32).cast_mut().cast(),
            (&sl_u as *const u32).cast_mut().cast(),
            (&pl_u as *const u32).cast_mut().cast(),
            (&dl_u as *const u32).cast_mut().cast(),
            (&out_l_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n * k_cols, out_l, col_buf)
}

#[allow(clippy::too_many_arguments)]
fn hip_im3col<T: Scalar>(
    input: &HipStorage<T>,
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
) -> HipStorage<T> {
    let k_vol = c_in * kd * kh * kw;
    let out_vol = out_d * out_h * out_w;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("im3col"));
    let col_buf = HipBuffer::alloc_zeros(n * k_vol * out_vol * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc im3col: {e}"));
    let col_elem = k_vol * out_vol;
    let grid_x = ((col_elem as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, d_u, h_u, w_u) = (c_in as u32, d as u32, h as u32, w as u32);
    let (kd_u, kh_u, kw_u) = (kd as u32, kh as u32, kw as u32);
    let (sd_u, sh_u, sw_u) = (sd as u32, sh as u32, sw as u32);
    let (pd_u, ph_u, pw_u) = (pd as u32, ph as u32, pw as u32);
    let (dd_u, dh_u, dw_u) = (dd as u32, dh as u32, dw as u32);
    let (out_d_u, out_h_u, out_w_u) = (out_d as u32, out_h as u32, out_w as u32);
    hip_launch(
        func,
        [grid_x, n as u32, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&col_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&c_in_u as *const u32).cast_mut().cast(),
            (&d_u as *const u32).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kd_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sd_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&pd_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&dd_u as *const u32).cast_mut().cast(),
            (&dh_u as *const u32).cast_mut().cast(),
            (&dw_u as *const u32).cast_mut().cast(),
            (&out_d_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n * k_vol, out_vol, col_buf)
}

/// im1col + per-sample matmul conv1d (groups == 1 only).
#[allow(clippy::too_many_arguments)]
fn hip_conv1d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
    n: usize,
    c_in: usize,
    l: usize,
    c_out: usize,
    kl: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> HipStorage<T> {
    assert!(
        groups == 1,
        "GPU conv1d: groups > 1 not supported; use CPU backend"
    );
    let out_l = (l + 2 * padding - dilation * (kl - 1) - 1) / stride + 1;
    let k_cols = c_in * kl;

    let col = hip_im1col(input, n, c_in, l, kl, stride, padding, dilation, out_l);

    let out_buf = HipBuffer::alloc_zeros(n * c_out * out_l * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv1d out: {e}"));
    let mut out = HipStorage::new(n * c_out, out_l, out_buf);
    for bi in 0..n {
        let col_off = bi * k_cols * out_l * core::mem::size_of::<T>();
        let out_off = bi * c_out * out_l * core::mem::size_of::<T>();
        // SAFETY: offsets are within allocated buffers; borrow_ptr creates non-owning views.
        let col_ptr = unsafe { col.buf.ptr.byte_add(col_off) };
        let out_ptr = unsafe { out.buf.ptr.byte_add(out_off) };
        let col_slice = HipStorage::new(k_cols, out_l, unsafe {
            HipBuffer::borrow_ptr(col_ptr, k_cols * out_l * core::mem::size_of::<T>())
        });
        let mut out_slice = HipStorage::new(c_out, out_l, unsafe {
            HipBuffer::borrow_ptr(out_ptr, c_out * out_l * core::mem::size_of::<T>())
        });
        hip_matmul(&mut out_slice, weight, &col_slice);
        core::mem::forget(col_slice);
        core::mem::forget(out_slice);
    }
    out.invalidate_cache();
    out
}

/// im3col + per-sample matmul conv3d (groups == 1 only).
#[allow(clippy::too_many_arguments)]
fn hip_conv3d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
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
) -> HipStorage<T> {
    assert!(
        groups == 1,
        "GPU conv3d: groups > 1 not supported; use CPU backend"
    );
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;
    let (dd, dh, dw) = dilation;
    let out_d = (d + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let out_vol = out_d * out_h * out_w;
    let k_vol = c_in * kd * kh * kw;

    let col = hip_im3col(
        input, n, c_in, d, h, w, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, out_d, out_h,
        out_w,
    );

    let out_buf = HipBuffer::alloc_zeros(n * c_out * out_vol * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv3d out: {e}"));
    let mut out = HipStorage::new(n * c_out, out_vol, out_buf);
    for bi in 0..n {
        let col_off = bi * k_vol * out_vol * core::mem::size_of::<T>();
        let out_off = bi * c_out * out_vol * core::mem::size_of::<T>();
        // SAFETY: offsets are within allocated buffers; borrow_ptr creates non-owning views.
        let col_ptr = unsafe { col.buf.ptr.byte_add(col_off) };
        let out_ptr = unsafe { out.buf.ptr.byte_add(out_off) };
        let col_slice = HipStorage::new(k_vol, out_vol, unsafe {
            HipBuffer::borrow_ptr(col_ptr, k_vol * out_vol * core::mem::size_of::<T>())
        });
        let mut out_slice = HipStorage::new(c_out, out_vol, unsafe {
            HipBuffer::borrow_ptr(out_ptr, c_out * out_vol * core::mem::size_of::<T>())
        });
        hip_matmul(&mut out_slice, weight, &col_slice);
        core::mem::forget(col_slice);
        core::mem::forget(out_slice);
    }
    out.invalidate_cache();
    out
}

/// im2col + per-sample tiled-matmul conv2d (groups == 1 only).
#[allow(clippy::too_many_arguments)]
fn hip_conv2d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
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
) -> HipStorage<T> {
    assert!(
        groups == 1,
        "GPU conv2d: groups > 1 not supported; use CPU backend"
    );
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let out_spatial = out_h * out_w;
    let k_cols = c_in * kh * kw;

    // Step 1: im2col → col: (N*k_cols, out_spatial)
    let col = hip_im2col(
        input, n, c_in, h, w, kh, kw, sh, sw, ph, pw, dh, dw, out_h, out_w,
    );

    // Step 2: for each sample, GEMM weight (c_out x k_cols) @ col[b] (k_cols x out_spatial).
    let out_buf = HipBuffer::alloc_zeros(n * c_out * out_spatial * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv2d out: {e}"));
    let mut out = HipStorage::new(n * c_out, out_spatial, out_buf);
    for bi in 0..n {
        let col_off = bi * k_cols * out_spatial * core::mem::size_of::<T>();
        let out_off = bi * c_out * out_spatial * core::mem::size_of::<T>();
        // SAFETY: offsets are within allocated buffers; borrow_ptr creates non-owning views.
        let col_ptr = unsafe { col.buf.ptr.byte_add(col_off) };
        let out_ptr = unsafe { out.buf.ptr.byte_add(out_off) };
        let col_slice = HipStorage::new(k_cols, out_spatial, unsafe {
            HipBuffer::borrow_ptr(col_ptr, k_cols * out_spatial * core::mem::size_of::<T>())
        });
        let mut out_slice = HipStorage::new(c_out, out_spatial, unsafe {
            HipBuffer::borrow_ptr(out_ptr, c_out * out_spatial * core::mem::size_of::<T>())
        });
        hip_matmul(&mut out_slice, weight, &col_slice);
        // Prevent borrowed buffers from being freed on drop.
        core::mem::forget(col_slice);
        core::mem::forget(out_slice);
    }
    out.invalidate_cache();
    out
}

#[allow(clippy::too_many_arguments)]
fn hip_conv_transpose2d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
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
) -> HipStorage<T> {
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let out_h = (h - 1) * sh + kh - 2 * ph + oph;
    let out_w = (w - 1) * sw + kw - 2 * pw + opw;
    let total = n_batch * c_out * out_h * out_w;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("conv_transpose2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv_transpose2d: {e}"));
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
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&weight.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u as *const i32).cast_mut().cast(),
            (&c_in_u as *const i32).cast_mut().cast(),
            (&h_u as *const i32).cast_mut().cast(),
            (&w_u as *const i32).cast_mut().cast(),
            (&c_out_u as *const i32).cast_mut().cast(),
            (&kh_u as *const i32).cast_mut().cast(),
            (&kw_u as *const i32).cast_mut().cast(),
            (&out_h_u as *const i32).cast_mut().cast(),
            (&out_w_u as *const i32).cast_mut().cast(),
            (&sh_u as *const i32).cast_mut().cast(),
            (&sw_u as *const i32).cast_mut().cast(),
            (&ph_u as *const i32).cast_mut().cast(),
            (&pw_u as *const i32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n_batch * c_out, out_h * out_w, out_buf)
}

fn hip_softmax<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("softmax"));
    let out_buf = HipBuffer::alloc_zeros(rows * cols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    hip_launch(
        func,
        [rows as u32, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, cols, out_buf)
}

fn hip_layer_norm<T: Scalar>(
    a: &HipStorage<T>,
    gamma: &HipStorage<T>,
    beta: &HipStorage<T>,
    eps: T,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("layer_norm"));
    let out_buf = HipBuffer::alloc_zeros(rows * cols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let eps_f = eps.to_f64();
    if type_suffix::<T>() == "f32" {
        let eps_val = eps_f as f32;
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_val as *const f32).cast_mut().cast(),
            ],
        );
    } else {
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_f as *const f64).cast_mut().cast(),
            ],
        );
    }
    HipStorage::new(rows, cols, out_buf)
}

fn hip_rms_norm<T: Scalar>(a: &HipStorage<T>, gamma: &HipStorage<T>, eps: T) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("rms_norm"));
    let out_buf = HipBuffer::alloc_zeros(rows * cols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let eps_f = eps.to_f64();
    if type_suffix::<T>() == "f32" {
        let eps_val = eps_f as f32;
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_val as *const f32).cast_mut().cast(),
            ],
        );
    } else {
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_f as *const f64).cast_mut().cast(),
            ],
        );
    }
    HipStorage::new(rows, cols, out_buf)
}

fn hip_batch_norm_train<T: Scalar>(
    a: &HipStorage<T>,
    gamma: &HipStorage<T>,
    beta: &HipStorage<T>,
    running_mean: &mut HipStorage<T>,
    running_var: &mut HipStorage<T>,
    eps: T,
    momentum: T,
    training: bool,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.nrows;
    let c = a.ncols;
    let total = n * c;
    let sz = core::mem::size_of::<T>();
    let eps_f = eps.to_f64();
    let total_u32 = total as u32;
    let c_u32 = c as u32;
    let fwd_func = get_kernel_by_id(ctx, kernel_id::<T>("batch_norm_fwd"));
    let out_buf = HipBuffer::alloc_zeros(total * sz)
        .unwrap_or_else(|e| panic!("HIP alloc batch_norm out: {e}"));
    let fwd_grid = (total_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if training {
        let stats_func = get_kernel_by_id(ctx, kernel_id::<T>("batch_norm_stats"));
        let mean_buf = HipBuffer::alloc_zeros(c * sz)
            .unwrap_or_else(|e| panic!("HIP alloc batch_norm mean: {e}"));
        let var_buf = HipBuffer::alloc_zeros(c * sz)
            .unwrap_or_else(|e| panic!("HIP alloc batch_norm var: {e}"));
        let stats_grid = (c_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n_u32 = n as u32;
        hip_launch(
            stats_func,
            [stats_grid, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&mean_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&var_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&n_u32 as *const u32).cast_mut().cast(),
                (&c_u32 as *const u32).cast_mut().cast(),
            ],
        );
        let mean_s = HipStorage::new(1, c, mean_buf);
        let var_s = HipStorage::new(1, c, var_buf);
        let one_minus = T::from_f64(1.0) - momentum;
        for i in 0..c {
            let m = hip_get(&mean_s, 0, i);
            let v = hip_get(&var_s, 0, i);
            let rm = hip_get(running_mean, 0, i);
            let rv = hip_get(running_var, 0, i);
            hip_set(running_mean, 0, i, one_minus * rm + momentum * m);
            hip_set(running_var, 0, i, one_minus * rv + momentum * v);
        }
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&mean_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&var_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_val as *const f32).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        } else {
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&mean_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&var_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_f as *const f64).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        }
    } else {
        // Eval mode: use running_mean/running_var directly.
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&running_mean.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&running_var.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_val as *const f32).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        } else {
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&running_mean.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&running_var.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_f as *const f64).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        }
    }
    HipStorage::new(n, c, out_buf)
}

fn hip_cross_entropy_fused<T: Scalar>(
    input: &HipStorage<T>,
    target: &HipStorage<T>,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = input.nrows;
    let c = input.ncols;
    let sz = core::mem::size_of::<T>();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("cross_entropy"));
    let loss_buf = HipBuffer::alloc_zeros(n * sz)
        .unwrap_or_else(|e| panic!("HIP alloc cross_entropy loss: {e}"));
    let n_u32 = n as u32;
    let c_u32 = c as u32;
    let grid = (n_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hip_launch(
        func,
        [grid, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&target.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&loss_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
            (&c_u32 as *const u32).cast_mut().cast(),
        ],
    );
    let loss_s = HipStorage::new(n, 1, loss_buf);
    let total = (0..n).fold(T::zero(), |acc, i| acc + hip_get(&loss_s, i, 0));
    let mean = total / T::from_f64(n as f64);
    let out_buf = HipBuffer::alloc_zeros(sz)
        .unwrap_or_else(|e| panic!("HIP alloc cross_entropy result: {e}"));
    let mut out_s = HipStorage::new(1, 1, out_buf);
    hip_set(&mut out_s, 0, 0, mean);
    out_s
}

/// FlashAttention-2 SDPA on HIP. Q/K/V: (BH*seq, head_dim) row-major.
#[allow(clippy::too_many_arguments)]
fn hip_sdpa<T: Scalar>(
    q: &HipStorage<T>,
    k: &HipStorage<T>,
    v: &HipStorage<T>,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    batch_heads: usize,
) -> HipStorage<T> {
    const FA_BLOCK_M: u32 = 64;
    const FA_BLOCK_N: u32 = 64;
    let sz = core::mem::size_of::<T>();
    let out_buf = HipBuffer::alloc_zeros(batch_heads * seq_q * head_dim * sz)
        .unwrap_or_else(|e| panic!("HIP alloc sdpa: {e}"));
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("sdpa"));
    let num_q_tiles = seq_q.div_ceil(FA_BLOCK_M as usize) as u32;
    let grid = batch_heads as u32 * num_q_tiles;
    let smem = 2 * FA_BLOCK_N as usize * head_dim * sz;
    let seq_q_u = seq_q as u32;
    let seq_k_u = seq_k as u32;
    let head_dim_u = head_dim as u32;
    let bh_u = batch_heads as u32;
    let scale_f64 = 1.0_f64 / (head_dim as f64).sqrt();
    if type_suffix::<T>() == "f32" {
        let scale = scale_f64 as f32;
        hip_launch_smem(
            func,
            [grid, 1, 1],
            [FA_BLOCK_M, 1, 1],
            smem as u32,
            &mut [
                (&q.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&k.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&v.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&seq_q_u as *const u32).cast_mut().cast(),
                (&seq_k_u as *const u32).cast_mut().cast(),
                (&head_dim_u as *const u32).cast_mut().cast(),
                (&bh_u as *const u32).cast_mut().cast(),
                (&scale as *const f32).cast_mut().cast(),
            ],
        );
    } else {
        hip_launch_smem(
            func,
            [grid, 1, 1],
            [FA_BLOCK_M, 1, 1],
            smem as u32,
            &mut [
                (&q.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&k.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&v.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&seq_q_u as *const u32).cast_mut().cast(),
                (&seq_k_u as *const u32).cast_mut().cast(),
                (&head_dim_u as *const u32).cast_mut().cast(),
                (&bh_u as *const u32).cast_mut().cast(),
                (&scale_f64 as *const f64).cast_mut().cast(),
            ],
        );
    }
    HipStorage::new(batch_heads * seq_q, head_dim, out_buf)
}

fn hip_axis_reduce<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = HipBuffer::alloc_zeros(rows * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    hip_launch(
        func,
        [rows as u32, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, 1, out_buf)
}

fn hip_embedding<T: Scalar>(indices: &HipStorage<T>, weight: &HipStorage<T>) -> HipStorage<T> {
    let ctx = get_ctx();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = weight.ncols;
    let total = n_tokens * embed_dim;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("embedding"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let n_tokens_u32 = n_tokens as u32;
    let embed_dim_u32 = embed_dim as u32;
    hip_launch(
        func,
        [((total as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&indices.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&weight.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_tokens_u32 as *const u32).cast_mut().cast(),
            (&embed_dim_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n_tokens, embed_dim, out_buf)
}

fn hip_axis_same_shape<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    hip_launch(
        func,
        [grid_1d(rows), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, cols, out_buf)
}

/// Blelloch-scan dispatch for cumsum/cumprod: one block per row with shared memory.
fn hip_cumsum_cumprod<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let shared_mem = (2 * BLOCK_SIZE as usize * core::mem::size_of::<T>()) as u32;
    hip_launch_smem(
        func,
        [rows as u32, 1, 1],
        [BLOCK_SIZE, 1, 1],
        shared_mem,
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, cols, out_buf)
}

/// GPU product reduction for HIP via host-side fold (matches rtc_fold_first pattern).
pub(crate) fn hip_prod_all<T: Scalar>(a: &HipStorage<T>) -> T {
    gpu_common::rtc_fold_first_prod(a)
}

/// Max pooling with argmax flat-index output.
fn hip_max_pool2d_with_idx<T: Scalar>(
    a: &HipStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> (HipStorage<T>, HipStorage<T>) {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("max_pool2d_with_idx"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let idx_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc idx: {e}"));
    let (h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, out_h_u, out_w_u, nc_u) = (
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&idx_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    (
        HipStorage::new(nc, out_h * out_w, out_buf),
        HipStorage::new(nc, out_h * out_w, idx_buf),
    )
}

fn hip_fuse_launch<T: Scalar>(
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

// ── Mega-fused element-wise kernel (multi-op single launch) ──────────────────

/// Descriptor for one operation in a mega-fused kernel launch.
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

/// Launch a mega-kernel that executes multiple fused element-wise operations
/// in a single GPU kernel launch (HIP backend).
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

// ── Backend impl ─────────────────────────────────────────────────────────────

impl crate::backend::private::Sealed for crate::backend::Hip {}

impl crate::backend::Backend for crate::backend::Hip {
    type Storage<T: Scalar> = HipStorage<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
        hip_zeros(nrows, ncols)
    }

    gpu_common::rtc_backend_impl! {
        HipStorage;
        fill = hip_fill,
        from_fn = hip_from_fn,
        from_vec_async = hip_from_vec_async,
        get = hip_get,
        set = hip_set,
        transpose = hip_transpose,
        scale = hip_scale,
        clone_storage = hip_clone,
        powf = hip_powf,
        sum_all = hip_sum_all,
        max_all = hip_max_all,
        min_all = hip_min_all,
        argmax_all = hip_argmax_all,
        argmin_all = hip_argmin_all,
        softmax = hip_softmax,
        layer_norm = hip_layer_norm,
        rms_norm = hip_rms_norm,
        batch_norm_train = hip_batch_norm_train,
        cross_entropy_fused = hip_cross_entropy_fused,
        sdpa = hip_sdpa,
        axis_reduce = hip_axis_reduce,
        embedding = hip_embedding,
        cumsum_cumprod = hip_cumsum_cumprod,
        prod_all = hip_prod_all,
        max_pool2d = hip_max_pool2d,
        max_pool2d_with_idx = hip_max_pool2d_with_idx,
        avg_pool2d = hip_avg_pool2d,
        adaptive_avg_pool2d = hip_adaptive_avg_pool2d,
        conv2d = hip_conv2d,
        conv1d = hip_conv1d,
        conv3d = hip_conv3d,
        conv_transpose2d = hip_conv_transpose2d,
    }

    #[inline]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> HipStorage<T> {
        let buf = hip_or_panic(HipBuffer::from_host(&data), "HIP upload");
        HipStorage::new_cached(nrows, ncols, buf, data)
    }

    #[inline]
    fn matmul_into<T: Scalar>(out: &mut HipStorage<T>, a: &HipStorage<T>, b: &HipStorage<T>) {
        hip_matmul(out, a, b);
    }

    gpu_common::gpu_unary_ops!(HipStorage; exp, ln, log1p, sin, cos, tan, tanh, sqrt, abs, recip, erf, ceil, floor, round, asin, acos, atan, sinh, cosh, asinh, acosh, atanh, log2, log10);
    gpu_common::gpu_binary_ops!(HipStorage; add, sub, emul, ediv, atan2);


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

    gpu_common::gpu_unary_ops!(HipStorage; silu, mish, hardswish);
}
