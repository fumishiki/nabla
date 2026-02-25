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

use crate::gpu_common::{self, EnsureCache, RtcStorage, grid_1d, lock_or_recover, type_suffix};
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

// Round up to next power-of-2 (minimum 256 bytes).
fn hip_size_class(size: usize) -> usize {
    let min = 256;
    if size <= min { return min; }
    size.next_power_of_two()
}

/// Block-based caching memory pool for HIP (mirrors CUDA MemoryPool).
struct HipMemoryPool {
    bins: HashMap<usize, Vec<*mut c_void>>,
    cached_bytes: usize,
}

impl HipMemoryPool {
    fn new() -> Self {
        Self { bins: HashMap::new(), cached_bytes: 0 }
    }

    fn try_alloc(&mut self, size_bytes: usize) -> Option<(*mut c_void, usize)> {
        let sc = hip_size_class(size_bytes);
        if let Some(bin) = self.bins.get_mut(&sc) {
            if let Some(ptr) = bin.pop() {
                self.cached_bytes -= sc;
                return Some((ptr, sc));
            }
        }
        None
    }

    fn release(&mut self, ptr: *mut c_void, allocated_size: usize) {
        let sc = hip_size_class(allocated_size);
        self.bins.entry(sc).or_default().push(ptr);
        self.cached_bytes += sc;
    }
}

impl Drop for HipMemoryPool {
    fn drop(&mut self) {
        for (_, ptrs) in self.bins.drain() {
            for ptr in ptrs {
                unsafe { let _ = hip::hipFree(ptr); }
            }
        }
    }
}

// SAFETY: HipMemoryPool stores raw GPU device pointers that are not
// dereferenced on the host. HIP API calls are thread-safe.
unsafe impl Send for HipMemoryPool {}

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
            drop(pool);
            let sc = hip_size_class(size_bytes);
            let mut ptr: *mut c_void = core::ptr::null_mut();
            check(unsafe { hip::hipMalloc(&mut ptr, sc) })?;
            (ptr, sc)
        };
        // SAFETY: zeroing allocated device memory.
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
            drop(pool);
            let sc = hip_size_class(bytes);
            let mut ptr: *mut c_void = core::ptr::null_mut();
            check(unsafe { hip::hipMalloc(&mut ptr, sc) })?;
            (ptr, sc)
        };
        // SAFETY: T is POD (Scalar: Copy); uploading raw bytes to GPU.
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
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if self.pooled {
                // Return buffer to pool for reuse instead of freeing.
                let ctx = get_ctx();
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.release(self.ptr, self.alloc_size);
            } else {
                // SAFETY: freeing device memory not managed by pool.
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
    pool: Mutex<HipMemoryPool>,
}

fn get_ctx() -> &'static HipCtx {
    static CTX: OnceLock<HipCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        // SAFETY: initializing HIP runtime on device 0.
        let err = unsafe { hip::hipSetDevice(0) };
        if err != hip::hipError_t::hipSuccess {
            panic!("HIP device 0 init failed: {err:?}");
        }
        let hip_ctx = HipCtx {
            kernels: Mutex::new(HashMap::new()),
            pool: Mutex::new(HipMemoryPool::new()),
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

fn launch_unary<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
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
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(a.nrows, a.ncols, out_buf)
}

fn launch_binary<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
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
            (&b.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
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

fn fuse_kernel_source(gpu_expr: &str, n_inputs: usize, type_name: &str, kernel_name: &str) -> String {
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(if is_f32 { 1536 } else { 512 });

    if is_f32 {
        let scalar_expr = gpu_expr.to_string();
        src.push_str("extern \"C\" __global__ void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const float* in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str("float* out, unsigned n) {\n");
        src.push_str("    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i4 * 4;\n");
        src.push_str("    if (i + 3 < n) {\n");
        for j in 0..n_inputs {
            src.push_str(&format!(
                "        float4 v{j} = reinterpret_cast<const float4*>(in{j})[i4];\n"
            ));
        }
        src.push_str("        float4 r;\n");
        for comp in &["x", "y", "z", "w"] {
            let mut comp_expr = scalar_expr.clone();
            for j in (0..n_inputs).rev() {
                comp_expr = comp_expr.replace(
                    &format!("in{j}[i]"),
                    &format!("v{j}.{comp}"),
                );
            }
            src.push_str(&format!("        r.{comp} = {comp_expr};\n"));
        }
        src.push_str("        reinterpret_cast<float4*>(out)[i4] = r;\n");
        src.push_str("    } else {\n");
        src.push_str("        for (unsigned j = i; j < n && j < i + 4; j++) {\n");
        let mut tail_expr = scalar_expr;
        for j in (0..n_inputs).rev() {
            tail_expr = tail_expr.replace(
                &format!("in{j}[i]"),
                &format!("in{j}[j]"),
            );
        }
        src.push_str(&format!("            out[j] = {tail_expr};\n"));
        src.push_str("        }\n");
        src.push_str("    }\n}\n");
    } else {
        src.push_str("extern \"C\" __global__ void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const ");
            src.push_str(type_name);
            src.push_str("* in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str(type_name);
        src.push_str("* out, unsigned n) {\n");
        src.push_str("    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    if (i < n) {\n");
        src.push_str("        out[i] = ");
        src.push_str(gpu_expr);
        src.push_str(";\n");
        src.push_str("    }\n}\n");
    }
    src
}

fn hip_fuse_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
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
            let src_str = fuse_kernel_source(gpu_expr, n_inputs, type_name, &kernel_name);
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
    fn add<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>) -> HipStorage<T> { launch_binary(a, b, "add") }
    #[inline]
    fn sub<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>) -> HipStorage<T> { launch_binary(a, b, "sub") }
    #[inline]
    fn neg<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "neg") }
    #[inline]
    fn transpose<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { hip_transpose(a) }
    #[inline]
    fn scale<T: Scalar>(a: &HipStorage<T>, s: T) -> HipStorage<T> { hip_scale(a, s) }
    #[inline]
    fn clone_storage<T: Scalar>(s: &HipStorage<T>) -> HipStorage<T> { hip_clone(s) }

    #[inline]
    fn exp<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "exp") }
    #[inline]
    fn ln<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "ln") }
    #[inline]
    fn log1p<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "log1p") }
    #[inline]
    fn sin<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "sin") }
    #[inline]
    fn cos<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "cos") }
    #[inline]
    fn tanh<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "tanh") }
    #[inline]
    fn sqrt<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "sqrt") }
    #[inline]
    fn abs<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "abs") }
    #[inline]
    fn recip<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "recip") }
    #[inline]
    fn erf<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "erf") }
    #[inline]
    fn ceil<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "ceil") }
    #[inline]
    fn floor<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "floor") }
    #[inline]
    fn round<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> { launch_unary(a, "round") }
    #[inline]
    fn powf<T: Scalar>(a: &HipStorage<T>, p: T) -> HipStorage<T> { hip_powf(a, p) }
    #[inline]
    fn emul<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>) -> HipStorage<T> { launch_binary(a, b, "emul") }
    #[inline]
    fn ediv<T: Scalar>(a: &HipStorage<T>, b: &HipStorage<T>) -> HipStorage<T> { launch_binary(a, b, "ediv") }

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
    ) -> HipStorage<T> {
        hip_fuse_launch::<T>(inputs, nrows, ncols, gpu_expr, kernel_hash, n_inputs)
    }
}
