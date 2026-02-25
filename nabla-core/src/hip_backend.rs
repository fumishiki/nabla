// hip_backend.rs — HIP backend via hip-runtime-sys bindings + hiprtc JIT compilation.
//
// Design mirrors cuda_backend.rs:
//   - HipCtx (OnceLock singleton): device init + hiprtc module cache.
//   - HipStorage<T>: RAII GPU memory + lazy host_cache.
//   - Same CUDA C kernel source (CUDA/HIP source-compatible for compute kernels).

use std::any::TypeId;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, OnceLock};

use hip_runtime_sys as hip;

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

struct HipBuffer {
    ptr: *mut c_void,
    size: usize,
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
            return Ok(Self { ptr: core::ptr::null_mut(), size: 0 });
        }
        let mut ptr: *mut c_void = core::ptr::null_mut();
        // SAFETY: hipMalloc allocates device memory; ptr is output parameter.
        check(unsafe { hip::hipMalloc(&mut ptr, size_bytes) })?;
        // SAFETY: zeroing allocated device memory.
        check(unsafe { hip::hipMemset(ptr, 0, size_bytes) })?;
        Ok(Self { ptr, size: size_bytes })
    }

    fn from_host<T: Scalar>(data: &[T]) -> HipResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self { ptr: core::ptr::null_mut(), size: 0 });
        }
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check(unsafe { hip::hipMalloc(&mut ptr, bytes) })?;
        // SAFETY: T is POD (Scalar: Copy); uploading raw bytes to GPU.
        check(unsafe {
            hip::hipMemcpy(ptr, data.as_ptr().cast(), bytes, hip::hipMemcpyKind::hipMemcpyHostToDevice)
        })?;
        Ok(Self { ptr, size: bytes })
    }

    fn copy_to_host<T: Scalar>(&self, out: &mut [T]) -> HipResult<()> {
        let bytes = core::mem::size_of_val(out);
        if bytes > 0 {
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
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: freeing device memory allocated by hipMalloc.
            unsafe { let _ = hip::hipFree(self.ptr); }
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

// ── Type + grid helpers ──────────────────────────────────────────────────────

fn type_suffix<T: Scalar>() -> &'static str {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        "f32"
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        "f64"
    } else {
        panic!("HIP backend supports f32/f64 only")
    }
}

fn grid_1d(n: usize) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    { n.div_ceil(BLOCK_SIZE as usize) as u32 }
}

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
    // SAFETY: synchronizing the default stream.
    let sync = unsafe { hip::hipDeviceSynchronize() };
    if sync != hip::hipError_t::hipSuccess {
        panic!("HIP sync failed: {sync:?}");
    }
}

// ── HipStorage ───────────────────────────────────────────────────────────────

/// Row-major HIP-backed matrix storage.
pub struct HipStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    buf: HipBuffer,
    host_cache: Mutex<Option<Vec<T>>>,
}

// SAFETY: HipBuffer is Send+Sync (raw GPU pointer). Mutex<Option<Vec<T>>>
// is Send+Sync when T: Send+Sync, which Scalar guarantees.
unsafe impl<T: Scalar> Send for HipStorage<T> {}
unsafe impl<T: Scalar> Sync for HipStorage<T> {}

fn lock_or_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<T: Scalar> HipStorage<T> {
    fn new(nrows: usize, ncols: usize, buf: HipBuffer) -> Self {
        Self { nrows, ncols, buf, host_cache: Mutex::new(None) }
    }

    fn new_cached(nrows: usize, ncols: usize, buf: HipBuffer, cache: Vec<T>) -> Self {
        Self { nrows, ncols, buf, host_cache: Mutex::new(Some(cache)) }
    }

    fn n(&self) -> usize { self.nrows * self.ncols }

    fn invalidate_cache(&mut self) {
        *lock_or_recover(&self.host_cache) = None;
    }

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

    fn cached_get(&self, idx: usize) -> T {
        self.ensure_cache();
        let guard = lock_or_recover(&self.host_cache);
        guard.as_ref().expect("cache populated")[idx]
    }
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
    s.cached_get(r * s.ncols + c)
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
    HipStorage { nrows: s.nrows, ncols: s.ncols, buf: new_buf, host_cache: Mutex::new(cache) }
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

// Reductions — readback to host (simple, correct)

pub(crate) fn hip_sum_all<T: Scalar>(a: &HipStorage<T>) -> T {
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    data.iter().fold(T::zero(), |acc, &x| acc + x)
}

pub(crate) fn hip_max_all<T: Scalar>(a: &HipStorage<T>) -> T {
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    assert!(!data.is_empty(), "max_all: matrix must be non-empty");
    let mut it = data.iter();
    let init = *it.next().expect("non-empty");
    it.fold(init, |acc, &x| acc.reduction_max(x))
}

pub(crate) fn hip_min_all<T: Scalar>(a: &HipStorage<T>) -> T {
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    assert!(!data.is_empty(), "min_all: matrix must be non-empty");
    let mut it = data.iter();
    let init = *it.next().expect("non-empty");
    it.fold(init, |acc, &x| acc.reduction_min(x))
}

pub(crate) fn hip_argmax_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) {
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    let cols = a.ncols;
    let mut best = 0usize;
    for i in 1..data.len() {
        if data[i].reduction_gt(data[best]) { best = i; }
    }
    (best / cols, best % cols)
}

pub(crate) fn hip_argmin_all<T: Scalar>(a: &HipStorage<T>) -> (usize, usize) {
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    let cols = a.ncols;
    let mut best = 0usize;
    for i in 1..data.len() {
        if data[best].reduction_gt(data[i]) { best = i; }
    }
    (best / cols, best % cols)
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
}
