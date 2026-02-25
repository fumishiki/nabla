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

use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::driver::sys::{CUdeviceptr, CUevent, CUfunction, CUmodule, CUstreamCaptureMode};
use cudarc::driver::{result, CudaContext as CudarcContext, CudaGraph as CudarcCudaGraph, CudaStream};
use cudarc::nvrtc;

use crate::gpu_common::{
    self, EnsureCache, FreeBlock, GpuPtr, MemoryPool, RtcStorage,
    grid_1d, lock_or_recover, type_suffix, round_size,
    SMALL_LARGE_BOUNDARY, SMALL_ALLOC_SIZE, LARGE_ALLOC_SIZE,
};
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

type CudaPool = MemoryPool<CUdeviceptr>;

pub struct CuBuffer {
    pub(crate) ptr: CUdeviceptr,
    size: usize,       // requested size
    alloc_size: usize,  // actual allocated size (size_class rounded)
    pooled: bool,       // true = return to pool on Drop; false = direct free
}

impl CuBuffer {
    /// Wrap an externally-owned `CUdeviceptr` as a non-owning `CuBuffer`.
    ///
    /// # Safety
    /// - `ptr` must be a valid device pointer with at least `size_bytes` allocated.
    /// - The caller must ensure the pointer outlives this `CuBuffer`.
    /// - The returned buffer is **non-pooled** and its `Drop` will call `cuMemFree`.
    ///   To create a truly borrowed (non-freeing) buffer, use [`Self::borrow_ptr`].
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, size_bytes: usize) -> Self {
        Self { ptr, size: size_bytes, alloc_size: size_bytes, pooled: false }
    }

    /// Wrap an externally-owned `CUdeviceptr` as a **borrowed** (non-freeing) `CuBuffer`.
    ///
    /// Unlike [`from_raw_parts`], `Drop` will **NOT** free the pointer.
    /// Use this when the GPU memory is managed by an external allocator (e.g. `GpuTensor`).
    ///
    /// # Safety
    /// - `ptr` must be valid for at least `size_bytes` and must outlive this buffer.
    pub unsafe fn borrow_ptr(ptr: CUdeviceptr, size_bytes: usize) -> Self {
        // pooled=false + ptr=0 trick won't work because ptr != 0.
        // Instead we use a sentinel: alloc_size = 0 means "borrowed, don't free".
        Self { ptr, size: size_bytes, alloc_size: 0, pooled: false }
    }

    /// Returns the raw `CUdeviceptr`.
    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    fn alloc_zeros(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if size_bytes == 0 {
            return Ok(Self { ptr: 0, size: 0, alloc_size: 0, pooled: false });
        }
        let ctx = get_ctx();
        let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (dptr, alloc_size) = if let Some((ptr, sc)) = pool.try_alloc(size_bytes) {
            (ptr, sc)
        } else {
            // Over-allocate to batch cudaMalloc calls (PyTorch strategy)
            let rounded = round_size(size_bytes);
            let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
                rounded.max(SMALL_ALLOC_SIZE)
            } else {
                rounded.max(LARGE_ALLOC_SIZE)
            };
            drop(pool);
            let dptr = unsafe { result::malloc_async(stream.cu_stream(), alloc_sz)? };
            // If over-allocated, split remainder back into pool
            if alloc_sz > rounded {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.release(dptr + rounded as u64, alloc_sz - rounded);
                pool.allocated_bytes += rounded;
            } else {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes += alloc_sz;
            }
            (dptr, rounded)
        };
        // SAFETY: zeroing allocated device memory.
        unsafe { result::memset_d8_async(dptr, 0, alloc_size, stream.cu_stream())? };
        Ok(Self { ptr: dptr, size: size_bytes, alloc_size, pooled: true })
    }

    fn alloc_async(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if size_bytes == 0 {
            return Ok(Self { ptr: 0, size: 0, alloc_size: 0, pooled: false });
        }
        let ctx = get_ctx();
        let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((ptr, sc)) = pool.try_alloc(size_bytes) {
            return Ok(Self { ptr, size: size_bytes, alloc_size: sc, pooled: true });
        }
        let rounded = round_size(size_bytes);
        let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
            rounded.max(SMALL_ALLOC_SIZE)
        } else {
            rounded.max(LARGE_ALLOC_SIZE)
        };
        drop(pool);
        let dptr = unsafe { result::malloc_async(stream.cu_stream(), alloc_sz)? };
        if alloc_sz > rounded {
            let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            pool.release(dptr + rounded as u64, alloc_sz - rounded);
            pool.allocated_bytes += rounded;
        } else {
            let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            pool.allocated_bytes += alloc_sz;
        }
        Ok(Self { ptr: dptr, size: size_bytes, alloc_size: rounded, pooled: true })
    }

    fn from_host<T: Scalar>(stream: &Arc<CudaStream>, data: &[T]) -> CudaResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self { ptr: 0, size: 0, alloc_size: 0, pooled: false });
        }
        let ctx = get_ctx();
        let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (dptr, alloc_size) = if let Some((ptr, sc)) = pool.try_alloc(bytes) {
            (ptr, sc)
        } else {
            let rounded = round_size(bytes);
            let alloc_sz = if rounded < SMALL_LARGE_BOUNDARY {
                rounded.max(SMALL_ALLOC_SIZE)
            } else {
                rounded.max(LARGE_ALLOC_SIZE)
            };
            drop(pool);
            let dptr = unsafe { result::malloc_async(stream.cu_stream(), alloc_sz)? };
            if alloc_sz > rounded {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.release(dptr + rounded as u64, alloc_sz - rounded);
                pool.allocated_bytes += rounded;
            } else {
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes += alloc_sz;
            }
            (dptr, rounded)
        };
        // SAFETY: T is POD (Scalar: Copy + Send + Sync); uploading raw bytes to GPU.
        unsafe { result::memcpy_htod_async(dptr, data, stream.cu_stream())? };
        Ok(Self { ptr: dptr, size: bytes, alloc_size, pooled: true })
    }

    fn copy_to_host<T: Scalar>(&self, stream: &Arc<CudaStream>, out: &mut [T]) -> CudaResult<()> {
        let bytes = core::mem::size_of_val(out);
        if bytes > 0 {
            // memcpy_dtoh_sync is synchronous and waits for all prior stream ops.
            // SAFETY: out is properly sized and T is POD.
            unsafe { result::memcpy_dtoh_sync(out, self.ptr)? };
        }
        Ok(())
    }

    /// Copy a single element at byte offset from device to host.
    fn copy_element<T: Scalar>(&self, stream: &Arc<CudaStream>, byte_offset: usize) -> CudaResult<T> {
        let mut val = T::zero();
        // Sync stream first so the GPU data is ready, then copy 1 element.
        stream.synchronize()?;
        // SAFETY: reading sizeof::<T> bytes from device at ptr+offset into &mut val.
        unsafe {
            result::memcpy_dtoh_sync(
                core::slice::from_mut(&mut val),
                self.ptr + byte_offset as u64,
            )?;
        }
        Ok(val)
    }

    /// Non-blocking H2D: allocate on compute stream, copy on separate copy stream,
    /// synchronize via CUDA event so compute stream waits for the transfer to finish.
    /// This allows the compute stream to run kernels on other data while the copy is in flight.
    fn from_host_nonblocking<T: Scalar>(
        compute_stream: &Arc<CudaStream>,
        copy_stream: &Arc<CudaStream>,
        data: &[T],
    ) -> CudaResult<Self> {
        let bytes = core::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self { ptr: 0, size: 0, alloc_size: 0, pooled: false });
        }
        // Allocate buffer (from pool or cudaMallocAsync on compute stream)
        let buf = Self::alloc_async(compute_stream, bytes)?;
        // Copy on the copy stream (non-blocking w.r.t. compute stream)
        unsafe { result::memcpy_htod_async(buf.ptr, data, copy_stream.cu_stream())? };
        // Record event on copy stream, make compute stream wait
        let event = result::event::create(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        )?;
        unsafe { result::event::record(event, copy_stream.cu_stream())? };
        unsafe {
            result::stream::wait_event(
                compute_stream.cu_stream(),
                event,
                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )?;
        }
        // Destroy the event — safe because cuStreamWaitEvent captures the dependency,
        // and cuEventDestroy is valid even before the event completes.
        unsafe { result::event::destroy(event)? };
        Ok(buf)
    }
}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 && self.alloc_size > 0 {
            if self.pooled {
                let ctx = get_ctx();
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes = pool.allocated_bytes.saturating_sub(self.alloc_size);
                pool.release(self.ptr, self.alloc_size);
                pool.maybe_gc(|ptr, _| unsafe { let _ = result::free_sync(ptr); });
            } else {
                unsafe { let _ = result::free_sync(self.ptr); }
            }
        }
        // alloc_size == 0 → borrowed pointer, do NOT free
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

// SAFETY: CUfunction/*mut CUfunc_st and CUmodule/*mut CUmod_st are opaque CUDA handles.
// CUDA driver guarantees these are valid across threads when the context is current.
unsafe impl Send for KernelEntry {}
unsafe impl Sync for KernelEntry {}

// ── cuBLAS handle ────────────────────────────────────────────────────────────

struct CublasHandle(cublas_sys::cublasHandle_t);

// SAFETY: cublasHandle_t is bound to the CUDA context, used from OnceLock singleton only.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

// ── CudaCtx singleton ────────────────────────────────────────────────────────

struct CudaCtx {
    stream: Arc<CudaStream>,
    /// Separate stream for H2D/D2H transfers (multi-stream pipeline).
    copy_stream: Arc<CudaStream>,
    kernels: Mutex<HashMap<String, KernelEntry>>,
    pool: Mutex<CudaPool>,
    has_wmma: bool,
    blas: CublasHandle,
    /// Cached CUDA graphs keyed by user-provided name for deduplication.
    graphs: Mutex<HashMap<String, Arc<NablaCudaGraph>>>,
}

// Returns (major, minor) compute capability of device 0.
fn query_compute_capability() -> (i32, i32) {
    // SAFETY: querying device attributes via cudarc driver-level API.
    let major = unsafe {
        result::device::get_attribute(
            0,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }.unwrap_or(7);
    let minor = unsafe {
        result::device::get_attribute(
            0,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }.unwrap_or(0);
    (major, minor)
}

// Volta+ (compute >= 7.0) supports WMMA tensor cores
fn query_wmma_support() -> bool {
    query_compute_capability().0 >= 7
}

// NVRTC arch string matching the device (e.g. "compute_90" for GH200 sm_90a).
// Returns &'static str to satisfy nvrtc::CompileOptions lifetime requirement.
fn nvrtc_arch(major: i32, minor: i32) -> &'static str {
    match (major, minor) {
        (7, 0) => "compute_70",
        (7, 5) => "compute_75",
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 7) => "compute_87",
        (8, 9) => "compute_89",
        (9, 0) => "compute_90",
        _ if major >= 9 => "compute_90",
        _ => "compute_70",
    }
}

fn get_ctx() -> &'static CudaCtx {
    static CTX: OnceLock<CudaCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = CudarcContext::new(0).expect("CUDA device 0 init failed");
        let stream = ctx.default_stream();
        let copy_stream = ctx.new_stream().expect("CUDA copy stream creation failed");
        let (major, minor) = query_compute_capability();
        let arch: &'static str = nvrtc_arch(major, minor);
        let has_wmma = major >= 7;
        // SAFETY: initializing cuBLAS handle and binding it to the default stream.
        let blas_raw = unsafe { cublas_result::create_handle().expect("cuBLAS init failed") };
        unsafe { cublas_result::set_stream(blas_raw, stream.cu_stream() as cublas_sys::cudaStream_t).expect("cuBLAS set_stream failed") };
        // TF32 tensor cores on Ampere+ (sm_80+, including GH200 sm_90a) for ~15x FP32 speedup.
        unsafe { let _ = cublas_sys::cublasSetMathMode(blas_raw, cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH); }
        let cuda_ctx = CudaCtx {
            stream,
            copy_stream,
            kernels: Mutex::new(HashMap::new()),
            pool: Mutex::new(CudaPool::new()),
            has_wmma,
            blas: CublasHandle(blas_raw),
            graphs: Mutex::new(HashMap::new()),
        };
        // Pre-compile all kernels from the combined source
        if let Err(e) = compile_all_kernels(&cuda_ctx, &arch) {
            panic!("CUDA kernel compilation failed: {e}");
        }
        // Compile WMMA kernels if tensor cores available
        if has_wmma {
            if let Err(e) = compile_wmma_kernels(&cuda_ctx, &arch) {
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

fn compile_all_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
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

fn compile_wmma_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
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

fn cuda_grid_1d<T: Scalar>(n: usize) -> u32 {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    }
}

fn launch_unary<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    unsafe {
        result::launch_kernel(
            func, (cuda_grid_1d::<T>(n), 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch {name}: {e}"));
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

fn launch_binary<T: Scalar>(a: &CudaStorage<T>, b: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    unsafe {
        result::launch_kernel(
            func, (cuda_grid_1d::<T>(n), 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch {name}: {e}"));
    }
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

/// Non-blocking H2D upload: data transfer on copy stream overlaps with compute.
pub(crate) fn cuda_from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let buf = CuBuffer::from_host_nonblocking(&ctx.stream, &ctx.copy_stream, &data)
        .unwrap_or_else(|e| panic!("CUDA async upload: {e}"));
    CudaStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn cuda_get<T: Scalar>(s: &CudaStorage<T>, r: usize, c: usize) -> T {
    // Fast path: if host cache exists, read from it.
    {
        let guard = lock_or_recover(&s.host_cache);
        if let Some(cache) = guard.as_ref() {
            return cache[r * s.ncols + c];
        }
    }
    // Slow path: single-element D2H (avoid copying entire tensor).
    let ctx = get_ctx();
    let byte_offset = (r * s.ncols + c) * core::mem::size_of::<T>();
    s.buf.copy_element::<T>(&ctx.stream, byte_offset)
        .unwrap_or_else(|e| panic!("CUDA single-element readback: {e}"))
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
    // Async upload — stream ordering guarantees correctness for subsequent GPU ops.
}pub(crate) fn cuda_clone<T: Scalar>(s: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let bytes = s.n() * core::mem::size_of::<T>();
    let new_buf = CuBuffer::alloc_async(&ctx.stream, bytes)
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    if bytes > 0 {
        // SAFETY: device-to-device copy of same-sized buffers.
        unsafe {
            let _ = result::memcpy_dtod_async(new_buf.ptr, s.buf.ptr, bytes, ctx.stream.cu_stream());
        }
    }
    let cache = lock_or_recover(&s.host_cache).clone();
    CudaStorage { nrows: s.nrows, ncols: s.ncols, buf: new_buf, host_cache: Mutex::new(cache) }
}

pub(crate) fn cuda_transpose<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_transpose_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    unsafe {
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
        ).unwrap_or_else(|e| panic!("CUDA launch transpose: {e}"));
    }
    CudaStorage::new(a.ncols, a.nrows, out_buf)
}

pub(crate) fn cuda_scale<T: Scalar>(a: &CudaStorage<T>, s: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_scale_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };
    unsafe {
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
        ).unwrap_or_else(|e| panic!("CUDA launch scale: {e}"));
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_powf<T: Scalar>(a: &CudaStorage<T>, p: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_powf_{}", type_suffix::<T>());
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };
    unsafe {
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
        ).unwrap_or_else(|e| panic!("CUDA launch powf: {e}"));
    }
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
        ).unwrap_or_else(|e| panic!("CUDA launch matmul: {e}"));
    }
}

// cuBLAS GEMM: row-major C = A * B
//
// cuBLAS is column-major. Row-major C = A * B is equivalent to col-major:
//   C^T = B^T * A^T
// Since row-major A stored as flat array = col-major A^T, we call:
//   sgemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 1, B_ptr, ldb=n, A_ptr, lda=k, 0, C_ptr, ldc=n)
fn cublas_gemm<T: Scalar>(ctx: &CudaCtx, out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) {
    if out.n() == 0 { return; }
    let m = a.nrows as i32;
    let k = a.ncols as i32;
    let n = b.ncols as i32;
    use std::any::TypeId;
    // SAFETY: pointers are valid GPU buffers; alpha/beta are on host stack (cuBLAS copies them).
    // gemm_ex with CUBLAS_COMPUTE_32F_FAST_TF32 + CUBLAS_GEMM_DEFAULT_TENSOR_OP forces
    // TF32 tensor core paths for all matrix sizes (sgemm uses heuristics that skip tensor cores
    // for medium-sized matrices like 1024×1024 and 2048×2048).
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            cublas_result::gemm_ex(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n, m, k,
                &alpha as *const f32 as *const std::ffi::c_void,
                b.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F, n,
                a.buf.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F, k,
                &beta as *const f32 as *const std::ffi::c_void,
                out.buf.ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F, n,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ).unwrap_or_else(|e| panic!("cuBLAS gemm_ex f32: {e}"));
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            cublas_result::dgemm(
                ctx.blas.0,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n, m, k,
                &1.0f64,
                b.buf.ptr as *const f64, n,
                a.buf.ptr as *const f64, k,
                &0.0f64,
                out.buf.ptr as *mut f64, n,
            ).unwrap_or_else(|e| panic!("cuBLAS dgemm: {e}"));
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

// Reductions — GPU-side kernels (warp-shuffle + atomicAdd/per-block)

pub(crate) fn cuda_sum_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    let ctx = get_ctx();
    let n = a.n();
    if n == 0 { return T::zero(); }
    let suffix = type_suffix::<T>();
    let name = format!("k_sum_{suffix}");
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    // k_sum uses atomicAdd to a single output scalar (initialized to 0)
    let out_buf = CuBuffer::alloc_zeros(&ctx.stream, core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    let grid = grid_1d(n);
    unsafe {
        result::launch_kernel(
            func, (grid, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        ).unwrap_or_else(|e| panic!("CUDA launch {name}: {e}"));
    }
    // Read single scalar back
    let mut result = [T::zero()];
    out_buf.copy_to_host::<T>(&ctx.stream, &mut result)
        .unwrap_or_else(|e| panic!("CUDA D2H: {e}"));
    result[0]
}

pub(crate) fn cuda_max_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    cuda_reduce_extremum(a, "max")
}
pub(crate) fn cuda_min_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    cuda_reduce_extremum(a, "min")
}

fn cuda_reduce_extremum<T: Scalar>(a: &CudaStorage<T>, op: &str) -> T {
    let ctx = get_ctx();
    let n = a.n();
    assert!(n > 0, "reduction on empty");
    let suffix = type_suffix::<T>();
    let name = format!("k_{op}_{suffix}");
    let func = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    // k_max/k_min write per-block results; need multi-pass until 1 element remains
    // Init value = first element
    let init_buf = CuBuffer::alloc_async(&ctx.stream, core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    unsafe {
        let _ = result::memcpy_dtod_async(init_buf.ptr, a.buf.ptr, core::mem::size_of::<T>(), ctx.stream.cu_stream());
    }
    let mut in_ptr = a.buf.ptr;
    let mut cur_n = n as u32;
    let mut temp_bufs: Vec<CuBuffer> = Vec::new();
    loop {
        let grid = grid_1d(cur_n as usize);
        let out_buf = CuBuffer::alloc_async(&ctx.stream, (grid as usize) * core::mem::size_of::<T>())
            .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
        unsafe {
            result::launch_kernel(
                func, (grid, 1, 1), (BLOCK_SIZE, 1, 1), 0, ctx.stream.cu_stream(),
                &mut [
                    &in_ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &cur_n as *const u32 as *mut c_void,
                    &init_buf.ptr as *const CUdeviceptr as *mut c_void,
                ],
            ).unwrap_or_else(|e| panic!("CUDA launch {name}: {e}"));
        }
        in_ptr = out_buf.ptr;
        temp_bufs.push(out_buf);
        if grid == 1 { break; }
        cur_n = grid;
    }
    let mut result = [T::zero()];
    temp_bufs.last().unwrap().copy_to_host::<T>(&ctx.stream, &mut result)
        .unwrap_or_else(|e| panic!("CUDA D2H: {e}"));
    result[0]
}
pub(crate) fn cuda_argmax_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) { gpu_common::rtc_argmax_all(a) }
pub(crate) fn cuda_argmin_all<T: Scalar>(a: &CudaStorage<T>) -> (usize, usize) { gpu_common::rtc_argmin_all(a) }

// ── CUDA Graph capture/replay ───────────────────────────────────────────────
//
// Wraps cudarc's CudaGraph API to capture a sequence of kernel launches into
// a graph, then replay with a single CPU dispatch. Eliminates 0.5–2μs
// per-kernel launch overhead; 90–95% launch overhead reduction.
// Break-even at ~50 replays. Critical for training loops.

/// A captured CUDA graph ready for replay.
/// Wraps cudarc's `CudaGraph` with nabla-level caching/deduplication.
pub struct NablaCudaGraph {
    inner: CudarcCudaGraph,
}

// SAFETY: CudarcCudaGraph holds Arc<CudaStream> which is Send+Sync.
// We serialize access via the CudaCtx::graphs Mutex for cache operations.
unsafe impl Send for NablaCudaGraph {}
unsafe impl Sync for NablaCudaGraph {}

impl NablaCudaGraph {
    /// Begin capturing kernel launches on the default stream.
    fn begin_capture() -> CudaResult<()> {
        let ctx = get_ctx();
        ctx.stream.begin_capture(
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        )?;
        Ok(())
    }

    /// End capture and instantiate the graph for replay.
    fn end_capture() -> CudaResult<Self> {
        let ctx = get_ctx();
        // 0 = no special instantiation flags
        let flags: cudarc::driver::sys::CUgraphInstantiate_flags =
            unsafe { core::mem::transmute(0u32) };
        let graph = ctx.stream.end_capture(flags)?
            .ok_or(CudaError::NullPtr)?;
        Ok(Self { inner: graph })
    }

    /// Replay the captured graph with a single CPU dispatch.
    pub fn launch(&self) -> CudaResult<()> {
        self.inner.launch()?;
        Ok(())
    }
}

/// Capture a sequence of kernel launches as a CUDA Graph for fast replay.
///
/// The closure `f` is executed once to record all kernel launches. The
/// resulting graph can then be replayed many times with [`NablaCudaGraph::launch()`],
/// eliminating per-kernel launch overhead (0.5–2μs each).
///
/// Break-even at ~50 replays. Ideal for training loops where the same
/// computation repeats every iteration.
///
/// # Example
/// ```ignore
/// let graph = cuda_graph_capture(|| {
///     // kernel launches recorded here
/// })?;
/// for _ in 0..1000 {
///     graph.launch()?; // single CPU dispatch replays all kernels
/// }
/// ```
pub fn cuda_graph_capture<F: FnOnce()>(f: F) -> CudaResult<NablaCudaGraph> {
    NablaCudaGraph::begin_capture()?;
    f();
    NablaCudaGraph::end_capture()
}

/// Capture or retrieve a cached CUDA Graph by name.
///
/// If a graph with the given `name` already exists in the cache, returns
/// the cached version without re-capturing. Otherwise, captures via `f`
/// and stores for future reuse.
pub fn cuda_graph_capture_cached<F: FnOnce()>(
    name: &str,
    f: F,
) -> CudaResult<Arc<NablaCudaGraph>> {
    let ctx = get_ctx();
    {
        let cache = lock_or_recover(&ctx.graphs);
        if let Some(g) = cache.get(name) {
            return Ok(Arc::clone(g));
        }
    }
    let graph = cuda_graph_capture(f)?;
    let graph = Arc::new(graph);
    let mut cache = lock_or_recover(&ctx.graphs);
    cache.insert(name.to_string(), Arc::clone(&graph));
    Ok(graph)
}

// ── Fused element-wise kernel launch ────────────────────────────────────────

fn cuda_fuse_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
    reg_estimate: usize,
) -> CudaStorage<T> {
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
            let src = gpu_common::fuse_kernel_source(gpu_expr, n_inputs, type_name, &kernel_name, reg_estimate, true);
            let (major, minor) = query_compute_capability();
            let arch: &'static str = nvrtc_arch(major, minor);
            // Limit register usage when pressure is high to allow more warps per SM
            let maxreg = if reg_estimate > 80 { Some(120) } else { None };
            let ptx = nvrtc::compile_ptx_with_opts(
                &src,
                nvrtc::CompileOptions { arch: Some(arch), maxrregcount: maxreg, ..Default::default() },
            ).unwrap_or_else(|e| panic!("NVRTC fuse compile failed: {e}"));
            let ptx_src = ptx.to_src();
            let c_ptx = CString::new(ptx_src).unwrap_or_else(|_| panic!("null in PTX"));
            // SAFETY: loading compiled PTX as a CUDA module.
            let module = unsafe {
                result::module::load_data(c_ptx.as_ptr().cast::<c_void>())
            }.unwrap_or_else(|e| panic!("CUDA module load: {e}"));
            let c_fn = CString::new(kernel_name.as_str()).unwrap_or_else(|_| panic!("null in kernel name"));
            // SAFETY: getting function handle from loaded module.
            let func = unsafe {
                result::module::get_function(module, c_fn)
            }.unwrap_or_else(|e| panic!("CUDA get_function: {e}"));
            let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(kernel_name.clone(), KernelEntry { func, _module: module });
        }
    }

    let func = get_kernel(ctx, &kernel_name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    // f32 fused kernels use float4: 4 elements per thread
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };

    // Build kernel argument array: in0_ptr, in1_ptr, ..., out_ptr, n
    // SAFETY: input pointers are valid CudaStorage<T> — cast back to extract .buf.ptr
    let input_ptrs: Vec<CUdeviceptr> = inputs.iter().map(|&p| {
        let storage = unsafe { &*(p as *const CudaStorage<T>) };
        storage.buf.ptr
    }).collect();

    let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + 2);
    for ptr in &input_ptrs {
        args.push(ptr as *const CUdeviceptr as *mut c_void);
    }
    args.push(&out_buf.ptr as *const CUdeviceptr as *mut c_void);
    args.push(&n_u32 as *const u32 as *mut c_void);

    // SAFETY: launching fused kernel with correct argument layout.
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        ).unwrap_or_else(|e| panic!("CUDA launch {kernel_name}: {e}"));
    }
    CudaStorage::new(nrows, ncols, out_buf)
}

// ── Mega-fused element-wise kernel (multi-op single launch) ──────────────────

/// Descriptor for one operation in a mega-fused kernel launch.
pub(crate) struct MegaFuseOp {
    /// Raw pointers to input CudaStorage buffers (as `*const u8`).
    pub inputs: Vec<*const u8>,
    /// GPU C expression, using `opK_inN[i]` placeholders.
    pub gpu_expr: String,
    /// Number of inputs for this operation.
    pub n_inputs: usize,
}

/// Launch a mega-kernel that executes multiple fused element-wise operations
/// in a single GPU kernel launch, eliminating inter-op launch overhead.
///
/// All operations must have the same tensor dimensions (nrows × ncols).
pub(crate) fn cuda_mega_fuse_launch<T: Scalar>(
    ops: &[MegaFuseOp],
    nrows: usize,
    ncols: usize,
    kernel_hash: &str,
) -> Vec<CudaStorage<T>> {
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
            let src = gpu_common::mega_fuse_kernel_source(&op_descs, type_name, &kernel_name, true);
            let (major, minor) = query_compute_capability();
            let arch: &'static str = nvrtc_arch(major, minor);
            let ptx = nvrtc::compile_ptx_with_opts(
                &src,
                nvrtc::CompileOptions { arch: Some(arch), ..Default::default() },
            ).unwrap_or_else(|e| panic!("NVRTC mega-fuse compile failed: {e}"));
            let ptx_src = ptx.to_src();
            let c_ptx = CString::new(ptx_src).unwrap_or_else(|_| panic!("null in PTX"));
            let module = unsafe {
                result::module::load_data(c_ptx.as_ptr().cast::<c_void>())
            }.unwrap_or_else(|e| panic!("CUDA module load: {e}"));
            let c_fn = CString::new(kernel_name.as_str()).unwrap_or_else(|_| panic!("null in kernel name"));
            let func = unsafe {
                result::module::get_function(module, c_fn)
            }.unwrap_or_else(|e| panic!("CUDA get_function: {e}"));
            let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(kernel_name.clone(), KernelEntry { func, _module: module });
        }
    }

    let func = get_kernel(ctx, &kernel_name).unwrap_or_else(|e| panic!("{e}"));

    // Allocate output buffers
    let out_bufs: Vec<CuBuffer> = (0..ops.len())
        .map(|_| CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
            .unwrap_or_else(|e| panic!("CUDA alloc: {e}")))
        .collect();

    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };

    // Collect input device pointers and output device pointers
    let input_ptrs: Vec<Vec<CUdeviceptr>> = ops.iter().map(|op| {
        op.inputs.iter().map(|&p| {
            let storage = unsafe { &*(p as *const CudaStorage<T>) };
            storage.buf.ptr
        }).collect()
    }).collect();

    // Build kernel argument array: op0_in0, op0_in1, ..., op0_out, op1_in0, ..., op1_out, ..., n
    let total_args = ops.iter().map(|op| op.n_inputs + 1).sum::<usize>() + 1;
    let mut args: Vec<*mut c_void> = Vec::with_capacity(total_args);
    for (op_idx, op) in ops.iter().enumerate() {
        for j in 0..op.n_inputs {
            args.push(&input_ptrs[op_idx][j] as *const CUdeviceptr as *mut c_void);
        }
        args.push(&out_bufs[op_idx].ptr as *const CUdeviceptr as *mut c_void);
    }
    args.push(&n_u32 as *const u32 as *mut c_void);

    // SAFETY: launching mega-fused kernel with correct argument layout.
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        ).unwrap_or_else(|e| panic!("CUDA launch {kernel_name}: {e}"));
    }

    out_bufs.into_iter()
        .map(|buf| CudaStorage::new(nrows, ncols, buf))
        .collect()
}

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
    fn from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> CudaStorage<T> {
        cuda_from_vec_async(nrows, ncols, data)
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
    fn sync<T: Scalar>(_s: &CudaStorage<T>) {
        let ctx = get_ctx();
        ctx.stream.synchronize().unwrap_or_else(|e| panic!("CUDA sync: {e}"));
    }

    #[inline]
    fn matmul_into<T: Scalar>(out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) {
        cuda_matmul(out, a, b);
    }

    #[inline]
    fn neg<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { launch_unary(a, "neg") }

    #[inline]
    fn transpose<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> { cuda_transpose(a) }

    #[inline]
    fn scale<T: Scalar>(a: &CudaStorage<T>, s: T) -> CudaStorage<T> { cuda_scale(a, s) }

    #[inline]
    fn clone_storage<T: Scalar>(s: &CudaStorage<T>) -> CudaStorage<T> { cuda_clone(s) }

    gpu_common::gpu_unary_ops!(CudaStorage; exp, ln, log1p, sin, cos, tanh, sqrt, abs, recip, erf, ceil, floor, round);
    gpu_common::gpu_binary_ops!(CudaStorage; add, sub, emul, ediv);

    #[inline]
    fn powf<T: Scalar>(a: &CudaStorage<T>, p: T) -> CudaStorage<T> { cuda_powf(a, p) }

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
        cuda_fuse_launch::<T>(inputs, nrows, ncols, gpu_expr, kernel_hash, n_inputs, reg_estimate)
    }

    fn mega_fuse_launch<T: Scalar>(
        ops: &[(Vec<*const u8>, String, usize)],
        nrows: usize,
        ncols: usize,
        _cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T>>,
        kernel_hash: &str,
    ) -> Vec<CudaStorage<T>> {
        let mega_ops: Vec<MegaFuseOp> = ops.iter().map(|(inputs, expr, n_in)| {
            MegaFuseOp { inputs: inputs.clone(), gpu_expr: expr.clone(), n_inputs: *n_in }
        }).collect();
        cuda_mega_fuse_launch::<T>(&mega_ops, nrows, ncols, kernel_hash)
    }
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
    // Async upload — stream ordering handles correctness.
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
    // Async upload — stream ordering handles correctness.
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
