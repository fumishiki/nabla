// hip_backend.rs — HIP backend via hip-runtime-sys bindings + hiprtc JIT compilation.
//
// Design mirrors cuda_backend.rs:
//   - HipCtx (OnceLock singleton): device init + hiprtc module cache.
//   - HipStorage<T> = RtcStorage<HipBuffer, T>: shared GPU storage with lazy host_cache.
//   - Same CUDA C kernel source (CUDA/HIP source-compatible for compute kernels).

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, OnceLock};

use hip_runtime_sys as hip;

use crate::gpu_common::{self, EnsureCache, MemoryPool, RtcStorage, grid_1d, lock_or_recover, round_size, type_suffix,
    SMALL_LARGE_BOUNDARY, SMALL_ALLOC_SIZE, LARGE_ALLOC_SIZE};
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

// ── GPU buffer (RAII) ────────────────────────────────────────────────────────

type HipPool = MemoryPool<*mut c_void>;

fn hip_free(ptr: *mut c_void, _size: usize) {
    unsafe { let _ = hip::hipFree(ptr); }
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
    fn alloc_zeros(size_bytes: usize) -> HipResult<Self> {
        if size_bytes == 0 {
            return Ok(Self { ptr: core::ptr::null_mut(), size: 0, alloc_size: 0, pooled: false });
        }
        let ctx = get_ctx();
        let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (ptr, alloc_size) = if let Some((p, sc)) = pool.try_alloc(size_bytes) {
            (p, sc)
        } else {
            let rounded = round_size(size_bytes);
            let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
                rounded.max(SMALL_ALLOC_SIZE)
            } else {
                rounded.max(LARGE_ALLOC_SIZE)
            };
            drop(pool);
            let mut ptr: *mut c_void = core::ptr::null_mut();
            check(unsafe { hip::hipMalloc(&mut ptr, alloc_sz) })?;
            if alloc_sz > rounded {
                let split_ptr = unsafe { ptr.byte_add(rounded) };
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.release(split_ptr, alloc_sz - rounded);
                pool.allocated_bytes += rounded;
            } else {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes += alloc_sz;
            }
            (ptr, rounded)
        };
        check(unsafe { hip::hipMemset(ptr, 0, alloc_size) })?;
        Ok(Self { ptr, size: size_bytes, alloc_size, pooled: true })
    }

    fn from_host<T: Scalar>(data: &[T]) -> HipResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self { ptr: core::ptr::null_mut(), size: 0, alloc_size: 0, pooled: false });
        }
        let ctx = get_ctx();
        let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (ptr, alloc_size) = if let Some((p, sc)) = pool.try_alloc(bytes) {
            (p, sc)
        } else {
            let rounded = round_size(bytes);
            let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
                rounded.max(SMALL_ALLOC_SIZE)
            } else {
                rounded.max(LARGE_ALLOC_SIZE)
            };
            drop(pool);
            let mut ptr: *mut c_void = core::ptr::null_mut();
            check(unsafe { hip::hipMalloc(&mut ptr, alloc_sz) })?;
            if alloc_sz > rounded {
                let split_ptr = unsafe { ptr.byte_add(rounded) };
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.release(split_ptr, alloc_sz - rounded);
                pool.allocated_bytes += rounded;
            } else {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes += alloc_sz;
            }
            (ptr, rounded)
        };
        check(unsafe {
            hip::hipMemcpy(ptr, data.as_ptr().cast(), bytes, hip::hipMemcpyKind::hipMemcpyHostToDevice)
        })?;
        Ok(Self { ptr, size: bytes, alloc_size, pooled: true })
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

    /// Non-blocking H2D: allocate normally, copy on separate copy stream,
    /// synchronize via HIP event so the default stream waits for the transfer.
    fn from_host_nonblocking<T: Scalar>(copy_stream: hip::hipStream_t, data: &[T]) -> HipResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self { ptr: core::ptr::null_mut(), size: 0, alloc_size: 0, pooled: false });
        }
        // Allocate buffer from pool (same as from_host)
        let ctx = get_ctx();
        let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (ptr, alloc_size) = if let Some((p, sc)) = pool.try_alloc(bytes) {
            (p, sc)
        } else {
            let rounded = round_size(bytes);
            let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
                rounded.max(SMALL_ALLOC_SIZE)
            } else {
                rounded.max(LARGE_ALLOC_SIZE)
            };
            drop(pool);
            let mut ptr: *mut c_void = core::ptr::null_mut();
            check(unsafe { hip::hipMalloc(&mut ptr, alloc_sz) })?;
            if alloc_sz > rounded {
                let split_ptr = unsafe { ptr.byte_add(rounded) };
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.release(split_ptr, alloc_sz - rounded);
                pool.allocated_bytes += rounded;
            } else {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes += alloc_sz;
            }
            (ptr, rounded)
        };
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
        check(unsafe { hip::hipEventCreateWithFlags(&mut event, 0x02 /* hipEventDisableTiming */) })?;
        check(unsafe { hip::hipEventRecord(event, copy_stream) })?;
        // Default stream (null) waits on the event
        check(unsafe { hip::hipStreamWaitEvent(core::ptr::null_mut(), event, 0) })?;
        // Destroy event — safe because hipStreamWaitEvent captures the dependency
        unsafe { let _ = hip::hipEventDestroy(event); }
        Ok(Self { ptr, size: bytes, alloc_size, pooled: true })
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if self.pooled {
                let ctx = get_ctx();
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes = pool.allocated_bytes.saturating_sub(self.alloc_size);
                pool.release(self.ptr, self.alloc_size);
                pool.maybe_gc(hip_free);
            } else {
                unsafe { let _ = hip::hipFree(self.ptr); }
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

// ── HipCtx singleton ─────────────────────────────────────────────────────────

struct HipCtx {
    kernels: Mutex<HashMap<String, KernelEntry>>,
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
            kernels: Mutex::new(HashMap::new()),
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
    "k_neg_f32", "k_recip_f32", "k_exp_f32", "k_ln_f32", "k_log1p_f32",
    "k_sin_f32", "k_cos_f32", "k_tanh_f32", "k_sqrt_f32", "k_abs_f32",
    "k_ceil_f32", "k_floor_f32", "k_round_f32", "k_erf_f32",
    "k_add_f32", "k_sub_f32", "k_emul_f32", "k_ediv_f32",
    "k_scale_f32", "k_powf_f32", "k_fill_f32",
    "k_transpose_f32", "k_matmul_f32", "k_sum_f32", "k_max_f32", "k_min_f32",
    "k_neg_f64", "k_recip_f64", "k_exp_f64", "k_ln_f64", "k_log1p_f64",
    "k_sin_f64", "k_cos_f64", "k_tanh_f64", "k_sqrt_f64", "k_abs_f64",
    "k_ceil_f64", "k_floor_f64", "k_round_f64", "k_erf_f64",
    "k_add_f64", "k_sub_f64", "k_emul_f64", "k_ediv_f64",
    "k_scale_f64", "k_powf_f64", "k_fill_f64",
    "k_transpose_f64", "k_matmul_f64", "k_sum_f64", "k_max_f64", "k_min_f64",
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

    let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    for &kname in KERNEL_NAMES {
        let c_fn = CString::new(kname).map_err(|_| HipError::NullPtr)?;
        let mut func: hip::hipFunction_t = core::ptr::null_mut();
        // SAFETY: getting function handle from loaded module.
        check(unsafe { hip::hipModuleGetFunction(&mut func, module, c_fn.as_ptr()) })?;
        map.insert(kname.to_owned(), KernelEntry { func, _module: module });
    }
    Ok(())
}

fn get_kernel(ctx: &HipCtx, name: &str) -> HipResult<hip::hipFunction_t> {
    let map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(name)
        .map(|e| e.func)
        .ok_or_else(|| HipError::KernelNotFound(name.to_owned()))
}

// ── Launch helper ────────────────────────────────────────────────────────────

fn hip_launch(
    func: hip::hipFunction_t,
    grid: [u32; 3],
    block: [u32; 3],
    args: &mut [*mut c_void],
) {
    // SAFETY: launching HIP kernel with caller-provided valid arguments.
    let err = unsafe {
        hip::hipModuleLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
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

// ── Launch helpers ───────────────────────────────────────────────────────────

fn hip_prepare_launch<T: Scalar>(
    n: usize, op: &str,
) -> (hip::hipFunction_t, HipBuffer, u32) {
    let ctx = get_ctx();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    (func, out_buf, n as u32)
}

fn launch_unary<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let n = a.n();
    let (func, out_buf, n_u32) = hip_prepare_launch::<T>(n, op);
    hip_launch(func, [grid_1d(n), 1, 1], [BLOCK_SIZE, 1, 1], &mut [
        (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
        (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
        (&n_u32 as *const u32).cast_mut().cast(),
    ]);
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

fn launch_binary<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let n = a.n();
    let (func, out_buf, n_u32) = hip_prepare_launch::<T>(n, op);
    hip_launch(func, [grid_1d(n), 1, 1], [BLOCK_SIZE, 1, 1], &mut [
        (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
        (&b.buf.ptr as *const *mut c_void).cast_mut().cast(),
        (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
        (&n_u32 as *const u32).cast_mut().cast(),
    ]);
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

// ── Public operations ────────────────────────────────────────────────────────

pub(crate) fn hip_zeros<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
    let buf = HipBuffer::alloc_zeros(nrows * ncols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    HipStorage::new(nrows, ncols, buf)
}

pub(crate) fn hip_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> HipStorage<T> {
    let n = nrows * ncols;
    let data = vec![val; n];
    let buf = HipBuffer::from_host(&data).unwrap_or_else(|e| panic!("HIP upload: {e}"));
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
    let buf = HipBuffer::from_host(&data).unwrap_or_else(|e| panic!("HIP upload: {e}"));
    HipStorage::new_cached(nrows, ncols, buf, data)
}

/// Non-blocking H2D upload: data transfer on copy stream overlaps with compute.
pub(crate) fn hip_from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> HipStorage<T> {
    let ctx = get_ctx();
    let buf = HipBuffer::from_host_nonblocking(ctx.copy_stream, &data)
        .unwrap_or_else(|e| panic!("HIP async upload: {e}"));
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
    s.buf.copy_element::<T>(byte_offset)
        .unwrap_or_else(|e| panic!("HIP single-element readback: {e}"))
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
    let new_buf = HipBuffer::alloc_zeros(bytes).unwrap_or_else(|e| panic!("HIP alloc: {e}"));
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
    let cache = lock_or_recover(&s.host_cache).clone();
    RtcStorage { nrows: s.nrows, ncols: s.ncols, buf: new_buf, host_cache: Mutex::new(cache) }
}

pub(crate) fn hip_transpose<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_transpose_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
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
    let name = format!("k_scale_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
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
    let name = format!("k_powf_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
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

pub(crate) fn hip_matmul<T: Scalar>(
    out: &mut HipStorage<T>,
    a: &HipStorage<T>,
    b: &HipStorage<T>,
) {
    let ctx = get_ctx();
    out.invalidate_cache();
    let name = format!("k_matmul_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let m = a.nrows as u32;
    let k = a.ncols as u32;
    let n = b.ncols as u32;
    let out_bytes = out.n() * core::mem::size_of::<T>();
    if out_bytes > 0 {
        // SAFETY: zeroing output buffer before matmul accumulation.
        unsafe { let _ = hip::hipMemset(out.buf.ptr, 0, out_bytes); }
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

pub(crate) fn hip_sum_all<T: Scalar>(a: &HipStorage<T>) -> T { gpu_common::rtc_sum_all(a) }
pub(crate) fn hip_max_all<T: Scalar>(a: &HipStorage<T>) -> T { gpu_common::rtc_max_all(a) }
pub(crate) fn hip_min_all<T: Scalar>(a: &HipStorage<T>) -> T { gpu_common::rtc_min_all(a) }
pub(crate) fn hip_argmax_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) { gpu_common::rtc_argmax_all(a) }
pub(crate) fn hip_argmin_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) { gpu_common::rtc_argmin_all(a) }

// ── Fused element-wise kernel launch ────────────────────────────────────────



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
        let map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let src_str = gpu_common::fuse_kernel_source(gpu_expr, n_inputs, type_name, &kernel_name, reg_estimate, false);
            let c_src = CString::new(src_str).unwrap_or_else(|_| panic!("null in source"));
            let prog_name = CString::new("nabla_fuse").unwrap_or_else(|_| panic!("null"));

            let mut prog: hip::hiprtcProgram = core::ptr::null_mut();
            let err = unsafe {
                hip::hiprtcCreateProgram(
                    &mut prog, c_src.as_ptr(), prog_name.as_ptr(),
                    0, core::ptr::null_mut(), core::ptr::null_mut(),
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

            let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(kernel_name.clone(), KernelEntry { func, _module: module });
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
    let input_ptrs: Vec<*mut c_void> = inputs.iter().map(|&p| {
        let storage = unsafe { &*(p as *const HipStorage<T>) };
        storage.buf.ptr
    }).collect();

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
    /// GPU C expression, using `inN[i]` placeholders.
    pub gpu_expr: String,
    /// Number of inputs for this operation.
    pub n_inputs: usize,
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
        let map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let op_descs: Vec<(String, usize)> = ops.iter()
                .map(|op| (op.gpu_expr.clone(), op.n_inputs))
                .collect();
            let src_str = gpu_common::mega_fuse_kernel_source(&op_descs, type_name, &kernel_name, false);
            let c_src = CString::new(src_str).unwrap_or_else(|_| panic!("null in source"));
            let prog_name = CString::new("nabla_mega_fuse").unwrap_or_else(|_| panic!("null"));

            let mut prog: hip::hiprtcProgram = core::ptr::null_mut();
            let err = unsafe {
                hip::hiprtcCreateProgram(
                    &mut prog, c_src.as_ptr(), prog_name.as_ptr(),
                    0, core::ptr::null_mut(), core::ptr::null_mut(),
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

            let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(kernel_name.clone(), KernelEntry { func, _module: module });
        }
    }

    let func = get_kernel(ctx, &kernel_name).unwrap_or_else(|e| panic!("{e}"));

    // Allocate output buffers
    let out_bufs: Vec<HipBuffer> = (0..ops.len())
        .map(|_| HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
            .unwrap_or_else(|e| panic!("HIP alloc: {e}")))
        .collect();

    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };

    // Collect input device pointers
    let input_ptrs: Vec<Vec<*mut c_void>> = ops.iter().map(|op| {
        op.inputs.iter().map(|&p| {
            let storage = unsafe { &*(p as *const HipStorage<T>) };
            storage.buf.ptr
        }).collect()
    }).collect();

    // Build kernel argument array
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

    out_bufs.into_iter()
        .map(|buf| HipStorage::new(nrows, ncols, buf))
        .collect()
}

// ── Backend impl ─────────────────────────────────────────────────────────────

impl crate::backend::private::Sealed for crate::backend::Hip {}

impl crate::backend::Backend for crate::backend::Hip {
    type Storage<T: Scalar> = HipStorage<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> { hip_zeros(nrows, ncols) }

    #[inline]
    fn fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> HipStorage<T> { hip_fill(nrows, ncols, val) }

    #[inline]
    fn identity<T: Scalar>(n: usize) -> HipStorage<T> {
        hip_from_fn(n, n, |r, c| if r == c { T::one() } else { T::zero() })
    }

    #[inline]
    fn from_fn<T: Scalar>(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> HipStorage<T> {
        hip_from_fn(nrows, ncols, f)
    }

    #[inline]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> HipStorage<T> {
        let buf = HipBuffer::from_host(&data).unwrap_or_else(|e| panic!("HIP upload: {e}"));
        HipStorage::new_cached(nrows, ncols, buf, data)
    }

    #[inline]
    fn from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> HipStorage<T> {
        hip_from_vec_async(nrows, ncols, data)
    }

    #[inline]
    fn nrows<T: Scalar>(s: &HipStorage<T>) -> usize { s.nrows }
    #[inline]
    fn ncols<T: Scalar>(s: &HipStorage<T>) -> usize { s.ncols }
    #[inline]
    fn get<T: Scalar>(s: &HipStorage<T>, r: usize, c: usize) -> T { hip_get(s, r, c) }
    #[inline]
    fn set<T: Scalar>(s: &mut HipStorage<T>, r: usize, c: usize, v: T) { hip_set(s, r, c, v) }

    #[inline]
    fn matmul_into<T: Scalar>(out: &mut HipStorage<T>, a: &HipStorage<T>, b: &HipStorage<T>) {
        hip_matmul(out, a, b);
    }

    #[inline]
    fn neg<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "neg") }
    #[inline]
    fn transpose<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { hip_transpose(a) }
    #[inline]
    fn scale<T: Scalar>(a: &HipStorage<T>, s: T) -> HipStorage<T> { hip_scale(a, s) }
    #[inline]
    fn clone_storage<T: Scalar>(s: &HipStorage<T>) -> HipStorage<T> { hip_clone(s) }

    gpu_common::gpu_unary_ops!(HipStorage; exp, ln, log1p, sin, cos, tanh, sqrt, abs, recip, erf, ceil, floor, round);
    gpu_common::gpu_binary_ops!(HipStorage; add, sub, emul, ediv);

    #[inline]
    fn powf<T: Scalar>(a: &HipStorage<T>, p: T) -> HipStorage<T> { hip_powf(a, p) }

    #[inline]
    fn sum_all<T: Scalar>(a: &HipStorage<T>) -> T { hip_sum_all(a) }
    #[inline]
    fn max_all<T: Scalar>(a: &HipStorage<T>) -> T { hip_max_all(a) }
    #[inline]
    fn min_all<T: Scalar>(a: &HipStorage<T>) -> T { hip_min_all(a) }
    #[inline]
    fn argmax_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) { hip_argmax_all(a) }
    #[inline]
    fn argmin_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) { hip_argmin_all(a) }

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
        hip_fuse_launch::<T>(inputs, nrows, ncols, gpu_expr, kernel_hash, n_inputs, reg_estimate)
    }

    fn mega_fuse_launch<T: Scalar>(
        ops: &[(Vec<*const u8>, String, usize)],
        nrows: usize,
        ncols: usize,
        _cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T>>,
        kernel_hash: &str,
    ) -> Vec<HipStorage<T>> {
        let mega_ops: Vec<MegaFuseOp> = ops.iter().map(|(inputs, expr, n_in)| {
            MegaFuseOp { inputs: inputs.clone(), gpu_expr: expr.clone(), n_inputs: *n_in }
        }).collect();
        hip_mega_fuse_launch::<T>(&mega_ops, nrows, ncols, kernel_hash)
    }

    fn silu<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "silu") }
    fn mish<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "mish") }
    fn leaky_relu<T: Scalar>(_a: &HipStorage<T>, _s: T) -> HipStorage<T> { unimplemented!("leaky_relu: HIP kernel not yet implemented") }
    fn elu<T: Scalar>(_a: &HipStorage<T>, _alpha: T) -> HipStorage<T> { unimplemented!("elu: HIP kernel not yet implemented") }
    fn hardswish<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "hardswish") }
    fn softmax<T: Scalar>(_a: &HipStorage<T>) -> HipStorage<T> { unimplemented!("softmax: HIP kernel not yet implemented") }
    fn layer_norm<T: Scalar>(_a: &HipStorage<T>, _g: &HipStorage<T>, _b: &HipStorage<T>, _eps: T) -> HipStorage<T> { unimplemented!("layer_norm: HIP kernel not yet implemented") }
    fn rms_norm<T: Scalar>(_a: &HipStorage<T>, _g: &HipStorage<T>, _eps: T) -> HipStorage<T> { unimplemented!("rms_norm: HIP kernel not yet implemented") }
    fn sum_axis1<T: Scalar>(_a: &HipStorage<T>) -> HipStorage<T> { unimplemented!("sum_axis1: HIP kernel not yet implemented") }
    fn max_axis1<T: Scalar>(_a: &HipStorage<T>) -> HipStorage<T> { unimplemented!("max_axis1: HIP kernel not yet implemented") }
    fn embedding<T: Scalar>(_i: &HipStorage<T>, _w: &HipStorage<T>) -> HipStorage<T> { unimplemented!("embedding: HIP kernel not yet implemented") }
}
