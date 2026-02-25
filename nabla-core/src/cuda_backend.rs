// cuda_backend.rs — CUDA backend via cudarc 0.19 + NVRTC JIT compilation.
//
// Design:
//   - CudaCtx (OnceLock singleton): cudarc CudaContext + default stream + JIT module cache.
//   - CudaStorage<T> = RtcStorage<CuBuffer, T>: shared GPU storage with lazy host_cache.
//   - Kernels compiled once from kernels_cu::KERNELS, cached per-function.
//   - TypeId dispatch: f32/f64 → type-suffixed kernel name.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUmodule};
use cudarc::driver::{result, CudaContext as CudarcContext, CudaStream};
use cudarc::nvrtc;

use crate::gpu_common::{self, EnsureCache, RtcStorage, grid_1d, lock_or_recover, type_suffix};
use crate::kernels_cu::{self, BLOCK_SIZE};
use crate::scalar::Scalar;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) enum CudaError {
    Driver(cudarc::driver::DriverError),
    Nvrtc(nvrtc::CompileError),
    KernelNotFound(String),
    NullPtr,
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<nvrtc::CompileError> for CudaError {
    fn from(e: nvrtc::CompileError) -> Self {
        Self::Nvrtc(e)
    }
}

impl core::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver: {e}"),
            Self::Nvrtc(e) => write!(f, "NVRTC: {e}"),
            Self::KernelNotFound(s) => write!(f, "kernel not found: {s}"),
            Self::NullPtr => write!(f, "null pointer"),
        }
    }
}

type CudaResult<T> = Result<T, CudaError>;

// ── GPU buffer (RAII) ────────────────────────────────────────────────────────

pub(crate) struct CuBuffer {
    pub(crate) ptr: CUdeviceptr,
    size: usize,
}

impl CuBuffer {
    fn alloc_zeros(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if size_bytes == 0 {
            return Ok(Self { ptr: 0, size: 0 });
        }
        // SAFETY: cudarc result-level API for raw memory alloc + memset.
        let dptr = unsafe { result::malloc_sync(size_bytes)? };
        unsafe { result::memset_d8_async(dptr, 0, size_bytes, stream.cu_stream())? };
        Ok(Self { ptr: dptr, size: size_bytes })
    }

    fn from_host<T: Scalar>(stream: &Arc<CudaStream>, data: &[T]) -> CudaResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self { ptr: 0, size: 0 });
        }
        let dptr = unsafe { result::malloc_sync(bytes)? };
        // SAFETY: T is POD (Scalar: Copy + Send + Sync); uploading raw bytes to GPU.
        unsafe { result::memcpy_htod_async(dptr, data, stream.cu_stream())? };
        Ok(Self { ptr: dptr, size: bytes })
    }

    fn copy_to_host<T: Scalar>(&self, stream: &Arc<CudaStream>, out: &mut [T]) -> CudaResult<()> {
        let bytes = core::mem::size_of_val(out);
        if bytes > 0 {
            stream.synchronize()?;
            // SAFETY: out is properly sized and T is POD.
            unsafe { result::memcpy_dtoh_sync(out, self.ptr)? };
        }
        Ok(())
    }
}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: freeing GPU memory allocated by us.
            unsafe { let _ = result::free_sync(self.ptr); }
        }
    }
}

// ── Storage type alias ───────────────────────────────────────────────────────

/// Row-major CUDA-backed matrix storage.
pub type CudaStorage<T> = RtcStorage<CuBuffer, T>;

// SAFETY: CuBuffer is a raw GPU pointer (u64) + usize — trivially Send+Sync.
// Mutex<Option<Vec<T>>> is Send+Sync when T: Send+Sync (Scalar guarantees this).
unsafe impl<T: Scalar> Send for CudaStorage<T> {}
unsafe impl<T: Scalar> Sync for CudaStorage<T> {}

impl<T: Scalar> EnsureCache for CudaStorage<T> {
    fn ensure_cache(&self) {
        let mut guard = lock_or_recover(&self.host_cache);
        if guard.is_none() {
            let ctx = get_ctx();
            let mut host = vec![T::zero(); self.n()];
            if let Err(e) = self.buf.copy_to_host(&ctx.stream, &mut host) {
                panic!("CUDA readback failed: {e}");
            }
            *guard = Some(host);
        }
    }
}

// ── Kernel cache entry ───────────────────────────────────────────────────────

struct KernelEntry {
    func: CUfunction,
    _module: CUmodule,
}

// ── CudaCtx singleton ────────────────────────────────────────────────────────

struct CudaCtx {
    stream: Arc<CudaStream>,
    kernels: Mutex<HashMap<String, KernelEntry>>,
    has_wmma: bool,
}

// Volta+ (compute >= 7.0) supports WMMA tensor cores
fn query_wmma_support() -> bool {
    // SAFETY: querying device attribute via cudarc driver-level API.
    let major = unsafe {
        result::device::get_attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            0,
        )
    };
    major.map_or(false, |v| v >= 7)
}

fn get_ctx() -> &'static CudaCtx {
    static CTX: OnceLock<CudaCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = CudarcContext::new(0).expect("CUDA device 0 init failed");
        let stream = ctx.default_stream();
        let has_wmma = query_wmma_support();
        let cuda_ctx = CudaCtx {
            stream,
            kernels: Mutex::new(HashMap::new()),
            has_wmma,
        };
        // Pre-compile all kernels from the combined source
        if let Err(e) = compile_all_kernels(&cuda_ctx) {
            panic!("CUDA kernel compilation failed: {e}");
        }
        // Compile WMMA kernels if tensor cores available
        if has_wmma {
            if let Err(e) = compile_wmma_kernels(&cuda_ctx) {
                // Non-fatal: fall back to tiled matmul
                eprintln!("WMMA kernel compilation failed (falling back to tiled): {e}");
            }
        }
        cuda_ctx
    })
}

// All kernel names we compile from KERNELS source
const KERNEL_NAMES: &[&str] = &[
    // unary f32
    "k_neg_f32", "k_recip_f32", "k_exp_f32", "k_ln_f32", "k_log1p_f32",
    "k_sin_f32", "k_cos_f32", "k_tanh_f32", "k_sqrt_f32", "k_abs_f32",
    "k_ceil_f32", "k_floor_f32", "k_round_f32", "k_erf_f32",
    // binary f32
    "k_add_f32", "k_sub_f32", "k_emul_f32", "k_ediv_f32",
    // scalar f32
    "k_scale_f32", "k_powf_f32", "k_fill_f32",
    // transpose+matmul+reduction f32
    "k_transpose_f32", "k_matmul_f32", "k_sum_f32", "k_max_f32", "k_min_f32",
    // unary f64
    "k_neg_f64", "k_recip_f64", "k_exp_f64", "k_ln_f64", "k_log1p_f64",
    "k_sin_f64", "k_cos_f64", "k_tanh_f64", "k_sqrt_f64", "k_abs_f64",
    "k_ceil_f64", "k_floor_f64", "k_round_f64", "k_erf_f64",
    // binary f64
    "k_add_f64", "k_sub_f64", "k_emul_f64", "k_ediv_f64",
    // scalar f64
    "k_scale_f64", "k_powf_f64", "k_fill_f64",
    // transpose+matmul+reduction f64
    "k_transpose_f64", "k_matmul_f64", "k_sum_f64", "k_max_f64", "k_min_f64",
];

fn compile_all_kernels(ctx: &CudaCtx) -> CudaResult<()> {
    let ptx = nvrtc::compile_ptx_with_opts(
        kernels_cu::KERNELS,
        nvrtc::CompileOptions {
            arch: Some("compute_70"),
            ..Default::default()
        },
    )?;

    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled PTX data as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };

    let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    for &name in KERNEL_NAMES {
        let c_fn = CString::new(name).map_err(|_| CudaError::NullPtr)?;
        // SAFETY: getting function handle from loaded module.
        let func = unsafe { result::module::get_function(module, c_fn)? };
        map.insert(name.to_owned(), KernelEntry { func, _module: module });
    }
    Ok(())
}

const WMMA_KERNEL_NAMES: &[&str] = &["k_matmul_wmma_f16"];

fn compile_wmma_kernels(ctx: &CudaCtx) -> CudaResult<()> {
    let src = kernels_cu::WMMA_KERNELS;
    if src.is_empty() {
        return Ok(());
    }

    let ptx = nvrtc::compile_ptx_with_opts(
        src,
        nvrtc::CompileOptions {
            arch: Some("compute_70"),
            ..Default::default()
        },
    )?;

    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled WMMA PTX as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };

    let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    for &name in WMMA_KERNEL_NAMES {
        let c_fn = CString::new(name).map_err(|_| CudaError::NullPtr)?;
        // SAFETY: getting function handle from loaded WMMA module.
        let func = unsafe { result::module::get_function(module, c_fn)? };
        map.insert(name.to_owned(), KernelEntry { func, _module: module });
    }
    Ok(())
}

fn get_kernel(ctx: &CudaCtx, name: &str) -> CudaResult<CUfunction> {
    let map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(name)
        .map(|e| e.func)
        .ok_or_else(|| CudaError::KernelNotFound(name.to_owned()))
}

// ── Launch helpers ───────────────────────────────────────────────────────────

// SAFETY for all launch_* functions: kernel arguments are raw pointers to GPU
// memory or scalar values. Caller guarantees buffers are valid and sized correctly.

fn launch_unary<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_zeros(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    // SAFETY: launching CUDA kernel with correct argument pointers.
    unsafe {
        result::launch_kernel(
            func,
            [grid_1d(n), 1, 1],
            [BLOCK_SIZE, 1, 1],
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch {name}: {e}"));
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

fn launch_binary<T: Scalar>(a: &CudaStorage<T>, b: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_zeros(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    unsafe {
        result::launch_kernel(
            func,
            [grid_1d(n), 1, 1],
            [BLOCK_SIZE, 1, 1],
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch {name}: {e}"));
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

// ── Public GPU operations (called by Backend impl) ───────────────────────────

pub(crate) fn cuda_zeros<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
    let ctx = get_ctx();
    let buf = CuBuffer::alloc_zeros(&ctx.stream, nrows * ncols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    CudaStorage::new(nrows, ncols, buf)
}

pub(crate) fn cuda_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let data = vec![val; n];
    let buf = CuBuffer::from_host(&ctx.stream, &data)
        .unwrap_or_else(|e| panic!("CUDA upload: {e}"));
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
    let buf = CuBuffer::from_host(&ctx.stream, &data)
        .unwrap_or_else(|e| panic!("CUDA upload: {e}"));
    CudaStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn cuda_get<T: Scalar>(s: &CudaStorage<T>, r: usize, c: usize) -> T {
    s.cached_get(r * s.ncols + c)
}

pub(crate) fn cuda_set<T: Scalar>(s: &mut CudaStorage<T>, r: usize, c: usize, v: T) {
    s.invalidate_cache();
    let ctx = get_ctx();
    let offset = (r * s.ncols + c) * core::mem::size_of::<T>();
    let src = core::slice::from_ref(&v);
    // SAFETY: uploading single element to correct offset in GPU buffer.
    unsafe {
        let _ = result::memcpy_htod_async(
            s.buf.ptr + offset as u64,
            src,
            ctx.stream.cu_stream(),
        );
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
}

pub(crate) fn cuda_clone<T: Scalar>(s: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let bytes = s.n() * core::mem::size_of::<T>();
    let new_buf = CuBuffer::alloc_zeros(&ctx.stream, bytes)
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    if bytes > 0 {
        // SAFETY: device-to-device copy of same-sized buffers.
        unsafe {
            let _ = result::memcpy_dtod_async(new_buf.ptr, s.buf.ptr, bytes, ctx.stream.cu_stream());
        }
        ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    }
    let cache = lock_or_recover(&s.host_cache).clone();
    CudaStorage { nrows: s.nrows, ncols: s.ncols, buf: new_buf, host_cache: Mutex::new(cache) }
}

pub(crate) fn cuda_transpose<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_transpose_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_zeros(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    unsafe {
        result::launch_kernel(
            func,
            [grid_1d(n), 1, 1],
            [BLOCK_SIZE, 1, 1],
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &rows as *const u32 as *mut c_void,
                &cols as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch transpose: {e}"));
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    CudaStorage::new(a.ncols, a.nrows, out_buf)
}

pub(crate) fn cuda_scale<T: Scalar>(a: &CudaStorage<T>, s: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_scale_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_zeros(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    unsafe {
        result::launch_kernel(
            func,
            [grid_1d(n), 1, 1],
            [BLOCK_SIZE, 1, 1],
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &s as *const T as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch scale: {e}"));
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_powf<T: Scalar>(a: &CudaStorage<T>, p: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_powf_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_zeros(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    unsafe {
        result::launch_kernel(
            func,
            [grid_1d(n), 1, 1],
            [BLOCK_SIZE, 1, 1],
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &p as *const T as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch powf: {e}"));
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_has_wmma() -> bool {
    get_ctx().has_wmma
}

fn cuda_matmul_tiled<T: Scalar>(
    ctx: &CudaCtx,
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let name = format!("k_matmul_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let m = a.nrows as u32;
    let k = a.ncols as u32;
    let n = b.ncols as u32;
    let out_bytes = out.n() * core::mem::size_of::<T>();
    if out_bytes > 0 {
        // SAFETY: zeroing output buffer before matmul accumulation.
        unsafe {
            let _ = result::memset_d8_async(out.buf.ptr, 0, out_bytes, ctx.stream.cu_stream());
        }
    }
    // Tiled matmul: 2D grid with TILE=16
    let grid_x = n.div_ceil(16);
    let grid_y = m.div_ceil(16);
    // SAFETY: launching CUDA kernel with correct argument pointers.
    unsafe {
        result::launch_kernel(
            func,
            [grid_x, grid_y, 1],
            [16, 16, 1],
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
        ).unwrap_or_else(|e| panic!("CUDA launch matmul: {e}"));
    }
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
}

pub(crate) fn cuda_matmul<T: Scalar>(
    out: &mut CudaStorage<T>,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
) {
    let ctx = get_ctx();
    out.invalidate_cache();
    // WMMA requires f16 input conversion — only use tiled path for f32/f64 native types.
    // WMMA dispatch is available via cuda_matmul_wmma_f16 for pre-converted f16 data.
    cuda_matmul_tiled(ctx, out, a, b);
}

// Reductions — delegated to shared gpu_common implementation

pub(crate) fn cuda_sum_all<T: Scalar>(a: &CudaStorage<T>) -> T { gpu_common::rtc_sum_all(a) }
pub(crate) fn cuda_max_all<T: Scalar>(a: &CudaStorage<T>) -> T { gpu_common::rtc_max_all(a) }
pub(crate) fn cuda_min_all<T: Scalar>(a: &CudaStorage<T>) -> T { gpu_common::rtc_min_all(a) }
pub(crate) fn cuda_argmax_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) { gpu_common::rtc_argmax_all(a) }
pub(crate) fn cuda_argmin_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) { gpu_common::rtc_argmin_all(a) }

// ── Backend impl ─────────────────────────────────────────────────────────────

impl crate::backend::private::Sealed for crate::backend::Cuda {}

impl crate::backend::Backend for crate::backend::Cuda {
    type Storage<T: Scalar> = CudaStorage<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
        cuda_zeros(nrows, ncols)
    }

    #[inline]
    fn fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> CudaStorage<T> {
        cuda_fill(nrows, ncols, val)
    }

    #[inline]
    fn identity<T: Scalar>(n: usize) -> CudaStorage<T> {
        cuda_from_fn(n, n, |r, c| if r == c { T::one() } else { T::zero() })
    }

    #[inline]
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> CudaStorage<T> {
        cuda_from_fn(nrows, ncols, f)
    }

    #[inline]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> CudaStorage<T> {
        let ctx = get_ctx();
        let buf = CuBuffer::from_host(&ctx.stream, &data)
            .unwrap_or_else(|e| panic!("CUDA upload: {e}"));
        CudaStorage::new_cached(nrows, ncols, buf, data)
    }

    #[inline]
    fn nrows<T: Scalar>(s: &CudaStorage<T>) -> usize { s.nrows }

    #[inline]
    fn ncols<T: Scalar>(s: &CudaStorage<T>) -> usize { s.ncols }

    #[inline]
    fn get<T: Scalar>(s: &CudaStorage<T>, r: usize, c: usize) -> T { cuda_get(s, r, c) }

    #[inline]
    fn set<T: Scalar>(s: &mut CudaStorage<T>, r: usize, c: usize, v: T) { cuda_set(s, r, c, v) }

    #[inline]
    fn matmul_into<T: Scalar>(out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) {
        cuda_matmul(out, a, b);
    }

    #[inline]
    fn add<T: Scalar>(a: &CudaStorage<T>, b: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(a, b, "add")
    }

    #[inline]
    fn sub<T: Scalar>(a: &CudaStorage<T>, b: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(a, b, "sub")
    }

    #[inline]
    fn neg<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "neg") }

    #[inline]
    fn transpose<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { cuda_transpose(a) }

    #[inline]
    fn scale<T: Scalar>(a: &CudaStorage<T>, s: T) -> CudaStorage<T> { cuda_scale(a, s) }

    #[inline]
    fn clone_storage<T: Scalar>(s: &CudaStorage<T>) -> CudaStorage<T> { cuda_clone(s) }

    #[inline]
    fn exp<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "exp") }

    #[inline]
    fn ln<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "ln") }

    #[inline]
    fn log1p<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "log1p") }

    #[inline]
    fn sin<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "sin") }

    #[inline]
    fn cos<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "cos") }

    #[inline]
    fn tanh<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "tanh") }

    #[inline]
    fn sqrt<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "sqrt") }

    #[inline]
    fn abs<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "abs") }

    #[inline]
    fn recip<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "recip") }

    #[inline]
    fn erf<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "erf") }

    #[inline]
    fn ceil<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "ceil") }

    #[inline]
    fn floor<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "floor") }

    #[inline]
    fn round<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "round") }

    #[inline]
    fn powf<T: Scalar>(a: &CudaStorage<T>, p: T) -> CudaStorage<T> { cuda_powf(a, p) }

    #[inline]
    fn emul<T: Scalar>(a: &CudaStorage<T>, b: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(a, b, "emul")
    }

    #[inline]
    fn ediv<T: Scalar>(a: &CudaStorage<T>, b: &CudaStorage<T>) -> CudaStorage<T> {
        launch_binary(a, b, "ediv")
    }

    #[inline]
    fn sum_all<T: Scalar>(a: &CudaStorage<T>) -> T { cuda_sum_all(a) }

    #[inline]
    fn max_all<T: Scalar>(a: &CudaStorage<T>) -> T { cuda_max_all(a) }

    #[inline]
    fn min_all<T: Scalar>(a: &CudaStorage<T>) -> T { cuda_min_all(a) }

    #[inline]
    fn argmax_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) { cuda_argmax_all(a) }

    #[inline]
    fn argmin_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) { cuda_argmin_all(a) }
}

// ── Recursive GEMM-based TRSM (GPU-resident triangular solve) ─────────────

const TRSM_BASE: usize = 32;

/// Solve L*X = B for lower-triangular L, overwriting B with X.
///
/// Recursive GEMM decomposition [2504.13821]: split L/B at midpoint,
/// solve top block via recursion, GEMM-update bottom, recurse bottom.
/// Base case (n <= 32): host forward substitution.
pub(crate) fn gpu_trsm_lower<T: Scalar>(
    l: &CudaStorage<T>,
    b: &mut CudaStorage<T>,
) {
    let n = l.nrows;
    assert_eq!(l.nrows, l.ncols, "TRSM: L must be square");
    assert_eq!(b.nrows, n, "TRSM: B row count must match L");
    if n == 0 { return; }

    if n <= TRSM_BASE {
        trsm_base_host(l, b);
        return;
    }

    let half = n / 2;
    let nrhs = b.ncols;

    // L = [[L11, 0], [L21, L22]], B = [B1; B2]
    // Step 1: X1 = TRSM(L11, B1)
    let l11 = cuda_submatrix(l, 0, 0, half, half);
    let mut b1 = cuda_submatrix(b, 0, 0, half, nrhs);
    gpu_trsm_lower(&l11, &mut b1);
    cuda_write_submatrix(b, 0, 0, &b1);

    // Step 2: B2 -= L21 * X1
    let l21 = cuda_submatrix(l, half, 0, n - half, half);
    let b2 = cuda_submatrix(b, half, 0, n - half, nrhs);
    let mut tmp = cuda_zeros::<T>(n - half, nrhs);
    cuda_matmul(&mut tmp, &l21, &b1);
    let b2_updated = launch_binary(&b2, &tmp, "sub");
    cuda_write_submatrix(b, half, 0, &b2_updated);

    // Step 3: X2 = TRSM(L22, B2_updated)
    let l22 = cuda_submatrix(l, half, half, n - half, n - half);
    let mut b2_final = cuda_submatrix(b, half, 0, n - half, nrhs);
    gpu_trsm_lower(&l22, &mut b2_final);
    cuda_write_submatrix(b, half, 0, &b2_final);
}

/// Base case: readback to host, forward-substitute, upload result.
fn trsm_base_host<T: Scalar>(l: &CudaStorage<T>, b: &mut CudaStorage<T>) {
    let n = l.nrows;
    let nrhs = b.ncols;
    let ctx = get_ctx();

    let mut l_host = vec![T::zero(); n * n];
    let mut b_host = vec![T::zero(); n * nrhs];
    l.buf.copy_to_host(&ctx.stream, &mut l_host)
        .unwrap_or_else(|e| panic!("TRSM readback L: {e}"));
    b.buf.copy_to_host(&ctx.stream, &mut b_host)
        .unwrap_or_else(|e| panic!("TRSM readback B: {e}"));

    // Forward substitution: x[i,j] = (b[i,j] - sum_{k<i} l[i,k]*x[k,j]) / l[i,i]
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
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("TRSM upload: {e}"));
}

/// Extract a sub-matrix from GPU storage (host round-trip).
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
    src.buf.copy_to_host(&ctx.stream, &mut host)
        .unwrap_or_else(|e| panic!("submatrix readback: {e}"));

    let mut sub = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            sub.push(host[(row_off + r) * src_cols + (col_off + c)]);
        }
    }

    let buf = CuBuffer::from_host(&ctx.stream, &sub)
        .unwrap_or_else(|e| panic!("submatrix upload: {e}"));
    CudaStorage::new_cached(nrows, ncols, buf, sub)
}

/// Write a sub-matrix back into a larger GPU storage at (row_off, col_off).
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
    dst.buf.copy_to_host(&ctx.stream, &mut dst_host)
        .unwrap_or_else(|e| panic!("write_submatrix dst readback: {e}"));

    let mut src_host = vec![T::zero(); src.n()];
    src.buf.copy_to_host(&ctx.stream, &mut src_host)
        .unwrap_or_else(|e| panic!("write_submatrix src readback: {e}"));

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
    ctx.stream.synchronize().unwrap_or_else(|e| panic!("write_submatrix upload: {e}"));
}

// ── GPU-resident AD tape ─────────────────────────────────────────────────────

/// Operation recorded on the GPU tape for reverse-mode AD.
#[derive(Clone)]
pub(crate) enum GpuOp {
    Add { a_id: usize, b_id: usize, out_id: usize },
    Sub { a_id: usize, b_id: usize, out_id: usize },
    Neg { a_id: usize, out_id: usize },
    Scale { a_id: usize, s_idx: usize, out_id: usize },
    Emul { a_id: usize, b_id: usize, out_id: usize },
    Matmul { a_id: usize, b_id: usize, out_id: usize, m: usize, k: usize, n: usize },
    Exp { a_id: usize, out_id: usize },
    Ln { a_id: usize, out_id: usize },
    Sin { a_id: usize, out_id: usize },
    Cos { a_id: usize, out_id: usize },
    Tanh { a_id: usize, out_id: usize },
    SumAll { a_id: usize, out_id: usize, rows: usize, cols: usize },
}

/// GPU-resident AD tape: records forward ops, replays backward on device.
///
/// Buffers keyed by integer id. `backward()` walks ops in reverse,
/// accumulating gradients into per-buffer GPU storage via existing kernels.
pub(crate) struct GpuTape<T: Scalar> {
    ops: Vec<GpuOp>,
    buffers: HashMap<usize, CudaStorage<T>>,
    grads: HashMap<usize, CudaStorage<T>>,
    next_id: usize,
}

impl<T: Scalar> GpuTape<T> {
    pub(crate) fn new() -> Self {
        Self { ops: Vec::new(), buffers: HashMap::new(), grads: HashMap::new(), next_id: 0 }
    }

    /// Register a buffer and return its id.
    pub(crate) fn register(&mut self, storage: CudaStorage<T>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.buffers.insert(id, storage);
        id
    }

    /// Record an op with a placeholder out_id, register the output, return actual out_id.
    pub(crate) fn record(&mut self, op: GpuOp, out: CudaStorage<T>) -> usize {
        let out_id = self.register(out);
        let patched = match op {
            GpuOp::Add { a_id, b_id, .. } => GpuOp::Add { a_id, b_id, out_id },
            GpuOp::Sub { a_id, b_id, .. } => GpuOp::Sub { a_id, b_id, out_id },
            GpuOp::Neg { a_id, .. } => GpuOp::Neg { a_id, out_id },
            GpuOp::Scale { a_id, s_idx, .. } => GpuOp::Scale { a_id, s_idx, out_id },
            GpuOp::Emul { a_id, b_id, .. } => GpuOp::Emul { a_id, b_id, out_id },
            GpuOp::Matmul { a_id, b_id, m, k, n, .. } =>
                GpuOp::Matmul { a_id, b_id, out_id, m, k, n },
            GpuOp::Exp { a_id, .. } => GpuOp::Exp { a_id, out_id },
            GpuOp::Ln { a_id, .. } => GpuOp::Ln { a_id, out_id },
            GpuOp::Sin { a_id, .. } => GpuOp::Sin { a_id, out_id },
            GpuOp::Cos { a_id, .. } => GpuOp::Cos { a_id, out_id },
            GpuOp::Tanh { a_id, .. } => GpuOp::Tanh { a_id, out_id },
            GpuOp::SumAll { a_id, rows, cols, .. } => GpuOp::SumAll { a_id, out_id, rows, cols },
        };
        self.ops.push(patched);
        out_id
    }

    fn accum_grad(&mut self, id: usize, delta: CudaStorage<T>) {
        if let Some(existing) = self.grads.get(&id) {
            let sum = launch_binary(existing, &delta, "add");
            self.grads.insert(id, sum);
        } else {
            self.grads.insert(id, delta);
        }
    }

    /// Reverse-mode AD from `loss_id`, seeding gradient with ones.
    pub(crate) fn backward(&mut self, loss_id: usize) {
        let loss_buf = self.buffers.get(&loss_id)
            .unwrap_or_else(|| panic!("GpuTape::backward: loss_id {loss_id} not found"));
        let seed = cuda_fill(loss_buf.nrows, loss_buf.ncols, T::one_impl());
        self.grads.insert(loss_id, seed);

        for i in (0..self.ops.len()).rev() {
            let op = self.ops[i].clone();
            match op {
                GpuOp::Add { a_id, b_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let ga = cuda_clone(g);
                        let gb = cuda_clone(g);
                        self.accum_grad(a_id, ga);
                        self.accum_grad(b_id, gb);
                    }
                }
                GpuOp::Sub { a_id, b_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let ga = cuda_clone(g);
                        let neg_g = launch_unary(g, "neg");
                        self.accum_grad(a_id, ga);
                        self.accum_grad(b_id, neg_g);
                    }
                }
                GpuOp::Neg { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let neg_g = launch_unary(g, "neg");
                        self.accum_grad(a_id, neg_g);
                    }
                }
                GpuOp::Scale { a_id, s_idx, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let s_val = cuda_get(
                            self.buffers.get(&s_idx)
                                .unwrap_or_else(|| panic!("GpuTape: scalar {s_idx} missing")),
                            0, 0,
                        );
                        let da = cuda_scale(g, s_val);
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Emul { a_id, b_id, out_id } => {
                    // grad_a += g .* b, grad_b += g .* a
                    if let Some(g) = self.grads.get(&out_id) {
                        let b_buf = self.buffers.get(&b_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {b_id} missing"));
                        let a_buf = self.buffers.get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let da = launch_binary(g, b_buf, "emul");
                        let db = launch_binary(g, a_buf, "emul");
                        self.accum_grad(a_id, da);
                        self.accum_grad(b_id, db);
                    }
                }
                GpuOp::Matmul { a_id, b_id, out_id, m, k: _, n } => {
                    // grad_a += g @ b^T, grad_b += a^T @ g
                    if let Some(g) = self.grads.get(&out_id) {
                        let b_buf = self.buffers.get(&b_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {b_id} missing"));
                        let a_buf = self.buffers.get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let bt = cuda_transpose(b_buf);
                        let at = cuda_transpose(a_buf);
                        let mut da = cuda_zeros::<T>(m, bt.ncols);
                        cuda_matmul(&mut da, g, &bt);
                        let mut db = cuda_zeros::<T>(at.nrows, n);
                        cuda_matmul(&mut db, &at, g);
                        self.accum_grad(a_id, da);
                        self.accum_grad(b_id, db);
                    }
                }
                GpuOp::Exp { a_id, out_id } => {
                    // grad_a += g .* out (exp(a) = out)
                    if let Some(g) = self.grads.get(&out_id) {
                        let out_buf = self.buffers.get(&out_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {out_id} missing"));
                        let da = launch_binary(g, out_buf, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Ln { a_id, out_id } => {
                    // grad_a += g / a
                    if let Some(g) = self.grads.get(&out_id) {
                        let a_buf = self.buffers.get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let da = launch_binary(g, a_buf, "ediv");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Sin { a_id, out_id } => {
                    // grad_a += g .* cos(a)
                    if let Some(g) = self.grads.get(&out_id) {
                        let a_buf = self.buffers.get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let cos_a = launch_unary(a_buf, "cos");
                        let da = launch_binary(g, &cos_a, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Cos { a_id, out_id } => {
                    // grad_a += g .* (-sin(a))
                    if let Some(g) = self.grads.get(&out_id) {
                        let a_buf = self.buffers.get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let sin_a = launch_unary(a_buf, "sin");
                        let neg_sin = launch_unary(&sin_a, "neg");
                        let da = launch_binary(g, &neg_sin, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Tanh { a_id, out_id } => {
                    // grad_a += g .* (1 - tanh^2) = g .* (1 - out^2)
                    if let Some(g) = self.grads.get(&out_id) {
                        let out_buf = self.buffers.get(&out_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {out_id} missing"));
                        let out_sq = launch_binary(out_buf, out_buf, "emul");
                        let ones = cuda_fill(out_sq.nrows, out_sq.ncols, T::one_impl());
                        let sech2 = launch_binary(&ones, &out_sq, "sub");
                        let da = launch_binary(g, &sech2, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::SumAll { a_id, out_id, rows, cols } => {
                    // grad_a += fill(g[0,0], rows, cols)
                    if let Some(g) = self.grads.get(&out_id) {
                        let g_val = cuda_get(g, 0, 0);
                        let da = cuda_fill(rows, cols, g_val);
                        self.accum_grad(a_id, da);
                    }
                }
            }
        }
    }

    /// Retrieve accumulated gradient for a buffer id.
    pub(crate) fn grad(&self, id: usize) -> Option<&CudaStorage<T>> {
        self.grads.get(&id)
    }
}
