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
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUmodule, CUstreamCaptureMode};
use cudarc::driver::{result, CudaContext as CudarcContext, CudaGraph as CudarcCudaGraph, CudaStream};
use cudarc::nvrtc;

use crate::gpu_common::{self, EnsureCache, RtcStorage, lock_or_recover, type_suffix};
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

/// Round up to 512-byte alignment (PyTorch-style, much less waste than power-of-2).
fn round_size(size: usize) -> usize {
    const ALIGN: usize = 512;
    if size == 0 { return ALIGN; }
    (size + ALIGN - 1) & !(ALIGN - 1)
}

/// Boundary between small pool (<1MB) and large pool (≥1MB).
const SMALL_LARGE_BOUNDARY: usize = 1 << 20; // 1MB
/// Minimum split remainder for small pool blocks.
const SMALL_SPLIT_MIN: usize = 512;
/// Minimum split remainder for large pool blocks.
const LARGE_SPLIT_MIN: usize = 1 << 20; // 1MB
/// Over-allocate size for small allocs (batch cudaMalloc calls).
const SMALL_ALLOC_SIZE: usize = 2 << 20; // 2MB
/// Over-allocate size for large allocs.
const LARGE_ALLOC_SIZE: usize = 20 << 20; // 20MB
/// GC threshold: free cached blocks when usage exceeds this fraction.
const GC_THRESHOLD: f64 = 0.9;

/// A free block in the pool, tracked for best-fit + coalescing.
struct FreeBlock {
    ptr: CUdeviceptr,
    size: usize,
}

/// Best-fit caching memory pool with block splitting and coalescing.
/// Mirrors PyTorch's CUDACachingAllocator design:
/// - 512B-aligned sizes (not power-of-2)
/// - Dual pools: small (<1MB) and large (≥1MB)
/// - Block splitting when remainder ≥ threshold
/// - Best-fit search (sorted by size)
/// - GC threshold to avoid OOM
struct MemoryPool {
    small_free: Vec<FreeBlock>, // sorted by size ascending
    large_free: Vec<FreeBlock>, // sorted by size ascending
    allocated_bytes: usize,
    cached_bytes: usize,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            small_free: Vec::new(),
            large_free: Vec::new(),
            allocated_bytes: 0,
            cached_bytes: 0,
        }
    }

    /// Best-fit: find smallest block ≥ requested size. Returns index if found.
    fn best_fit(pool: &[FreeBlock], size: usize) -> Option<usize> {
        // Binary search for first block with size >= requested
        let pos = pool.partition_point(|b| b.size < size);
        if pos < pool.len() { Some(pos) } else { None }
    }

    fn free_list(&mut self, size: usize) -> &mut Vec<FreeBlock> {
        if size < SMALL_LARGE_BOUNDARY { &mut self.small_free } else { &mut self.large_free }
    }

    fn split_min(size: usize) -> usize {
        if size < SMALL_LARGE_BOUNDARY { SMALL_SPLIT_MIN } else { LARGE_SPLIT_MIN }
    }

    /// Try to allocate from pool. Splits oversized blocks.
    /// Returns (ptr, actual_alloc_size) or None.
    fn try_alloc(&mut self, size: usize) -> Option<(CUdeviceptr, usize)> {
        let rounded = round_size(size);
        let pool = if rounded < SMALL_LARGE_BOUNDARY {
            &mut self.small_free
        } else {
            &mut self.large_free
        };
        let idx = Self::best_fit(pool, rounded)?;
        let block = pool.remove(idx);
        self.cached_bytes -= block.size;

        let remainder = block.size - rounded;
        let split_threshold = Self::split_min(rounded);
        if remainder >= split_threshold {
            // Split: return requested portion, keep remainder in pool
            let split_block = FreeBlock {
                ptr: block.ptr + rounded as u64,
                size: remainder,
            };
            let target = if remainder < SMALL_LARGE_BOUNDARY {
                &mut self.small_free
            } else {
                &mut self.large_free
            };
            let pos = target.partition_point(|b| b.size < remainder);
            target.insert(pos, split_block);
            self.cached_bytes += remainder;
            Some((block.ptr, rounded))
        } else {
            // Use entire block (avoid tiny fragments)
            Some((block.ptr, block.size))
        }
    }

    /// Return a block to the pool, inserting sorted by size.
    fn release(&mut self, ptr: CUdeviceptr, size: usize) {
        let pool = if size < SMALL_LARGE_BOUNDARY {
            &mut self.small_free
        } else {
            &mut self.large_free
        };
        let pos = pool.partition_point(|b| b.size < size);
        pool.insert(pos, FreeBlock { ptr, size });
        self.cached_bytes += size;
    }

    /// GC: free cached blocks if allocated exceeds threshold.
    fn maybe_gc(&mut self) {
        let total = self.allocated_bytes + self.cached_bytes;
        if total == 0 { return; }
        let usage_ratio = self.allocated_bytes as f64 / total as f64;
        if usage_ratio > GC_THRESHOLD && self.cached_bytes > 0 {
            self.trim(0);
        }
    }

    /// Free cached blocks until pool size ≤ target_bytes. Returns bytes freed.
    fn trim(&mut self, target_bytes: usize) -> usize {
        let mut freed = 0usize;
        // Free large blocks first (bigger impact)
        while self.cached_bytes > target_bytes {
            if let Some(block) = self.large_free.pop() {
                unsafe { let _ = result::free_sync(block.ptr); }
                self.cached_bytes -= block.size;
                freed += block.size;
            } else if let Some(block) = self.small_free.pop() {
                unsafe { let _ = result::free_sync(block.ptr); }
                self.cached_bytes -= block.size;
                freed += block.size;
            } else {
                break;
            }
        }
        freed
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        for block in self.small_free.drain(..) {
            unsafe { let _ = result::free_sync(block.ptr); }
        }
        for block in self.large_free.drain(..) {
            unsafe { let _ = result::free_sync(block.ptr); }
        }
    }
}

pub struct CuBuffer {
    pub(crate) ptr: CUdeviceptr,
    size: usize,       // requested size
    alloc_size: usize,  // actual allocated size (size_class rounded)
    pooled: bool,       // true = return to pool on Drop; false = direct free
}

impl CuBuffer {
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
}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if self.pooled {
                let ctx = get_ctx();
                let mut pool = ctx.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes = pool.allocated_bytes.saturating_sub(self.alloc_size);
                pool.release(self.ptr, self.alloc_size);
                pool.maybe_gc();
            } else {
                unsafe { let _ = result::free_sync(self.ptr); }
            }
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
    optimal_block: u32, // occupancy-based block size
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

/// Query optimal block size for a kernel using CUDA occupancy API.
/// Falls back to BLOCK_SIZE (256) if query fails.
fn optimal_block_size(func: CUfunction) -> u32 {
    let mut min_grid_size: i32 = 0;
    let mut block_size: i32 = 0;
    // SAFETY: querying occupancy for a valid CUfunction handle.
    let status = unsafe {
        cudarc::driver::sys::cuOccupancyMaxPotentialBlockSize(
            &mut min_grid_size,
            &mut block_size,
            func,
            None,  // no dynamic shared mem callback
            0,     // dynamic shared mem per block
            0,     // block size limit (0 = no limit)
        )
    };
    if status == cudarc::driver::sys::CUresult::CUDA_SUCCESS && block_size > 0 {
        let _ = min_grid_size; // suppress unused warning
        block_size as u32
    } else {
        BLOCK_SIZE
    }
}

// ── CudaCtx singleton ────────────────────────────────────────────────────────

struct CudaCtx {
    stream: Arc<CudaStream>,
    kernels: Mutex<HashMap<String, KernelEntry>>,
    pool: Mutex<MemoryPool>,
    has_wmma: bool,
    blas: CublasHandle,
    /// Cached CUDA graphs keyed by user-provided name for deduplication.
    graphs: Mutex<HashMap<String, Arc<NablaCudaGraph>>>,
    /// SM count for persistent grid-stride kernel launches.
    sm_count: u32,
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

// Query SM (multiprocessor) count for persistent grid-stride launches.
fn query_sm_count() -> u32 {
    let sm = unsafe {
        result::device::get_attribute(
            0,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }.unwrap_or(80);
    sm as u32
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
            kernels: Mutex::new(HashMap::new()),
            pool: Mutex::new(MemoryPool::new()),
            has_wmma,
            blas: CublasHandle(blas_raw),
            graphs: Mutex::new(HashMap::new()),
            sm_count: query_sm_count(),
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
        let ob = optimal_block_size(func);
        map.insert(name.to_owned(), KernelEntry { func, _module: module, optimal_block: ob });
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
        let ob = optimal_block_size(func);
        map.insert(name.to_owned(), KernelEntry { func, _module: module, optimal_block: ob });
    }
    Ok(())
}

fn get_kernel(ctx: &CudaCtx, name: &str) -> CudaResult<(CUfunction, u32)> {
    let map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(name)
        .map(|e| (e.func, e.optimal_block))
        .ok_or_else(|| CudaError::KernelNotFound(name.to_owned()))
}

// ── Launch helpers ───────────────────────────────────────────────────────────

/// Persistent grid for f32 float4 kernels: cap to SM_COUNT * 4 blocks.
fn persistent_grid_f32(ctx: &CudaCtx, n: usize, block: u32) -> u32 {
    let full = ((n + 3) / 4).div_ceil(block as usize) as u32;
    core::cmp::min(ctx.sm_count * 4, full)
}

// SAFETY for all launch_* functions: kernel arguments are raw pointers to GPU
// memory or scalar values. Caller guarantees buffers are valid and sized correctly.

fn launch_unary<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let (func, ob) = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    // f32 kernels use persistent grid-stride with float4
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        persistent_grid_f32(ctx, n, ob)
    } else {
        n.div_ceil(ob as usize) as u32
    };
    // SAFETY: launching CUDA kernel with correct argument pointers.
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (ob, 1, 1),
            0,
            ctx.stream.cu_stream(),
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
    let (func, ob) = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        persistent_grid_f32(ctx, n, ob)
    } else {
        n.div_ceil(ob as usize) as u32
    };
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (ob, 1, 1),
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
    let (func, ob) = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    unsafe {
        result::launch_kernel(
            func,
            (n.div_ceil(ob as usize) as u32, 1, 1),
            (ob, 1, 1),
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
    let (func, ob) = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        persistent_grid_f32(ctx, n, ob)
    } else {
        n.div_ceil(ob as usize) as u32
    };
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (ob, 1, 1),
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
    let (func, ob) = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        persistent_grid_f32(ctx, n, ob)
    } else {
        n.div_ceil(ob as usize) as u32
    };
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (ob, 1, 1),
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
    let (func, _ob) = get_kernel(ctx, &name).unwrap_or_else(|e| panic!("{e}"));
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

// Reductions — delegated to shared gpu_common implementation

pub(crate) fn cuda_sum_all<T: Scalar>(a: &CudaStorage<T>) -> T { gpu_common::rtc_sum_all(a) }
pub(crate) fn cuda_max_all<T: Scalar>(a: &CudaStorage<T>) -> T { gpu_common::rtc_max_all(a) }
pub(crate) fn cuda_min_all<T: Scalar>(a: &CudaStorage<T>) -> T { gpu_common::rtc_min_all(a) }
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

/// Generate full CUDA C kernel source from a fused expression body.
/// For f32: generates float4-vectorized kernel with scalar tail.
/// For f64: generates scalar kernel (double2 shows minimal benefit).
fn fuse_kernel_source(gpu_expr: &str, n_inputs: usize, type_name: &str, kernel_name: &str, reg_estimate: usize) -> String {
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(if is_f32 { 1536 } else { 512 });

    // Annotate kernel with register pressure estimate
    src.push_str(&format!("// estimated registers: {reg_estimate}\n"));

    if is_f32 {
        // float4-vectorized kernel with persistent grid-stride loop
        let scalar_expr = gpu_expr.to_string(); // uses inN[i] pattern

        src.push_str("extern \"C\" __global__ void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const float* in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str("float* out, unsigned n) {\n");
        src.push_str("    unsigned stride = gridDim.x * blockDim.x;\n");
        src.push_str("    unsigned n4 = n >> 2;\n");
        src.push_str("    for (unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += stride) {\n");
        // Load float4 for each input
        for j in 0..n_inputs {
            src.push_str(&format!(
                "        float4 v{j} = __ldg(reinterpret_cast<const float4*>(in{j}) + i4);\n"
            ));
        }
        // Apply expression to each component (.x, .y, .z, .w)
        src.push_str("        float4 r;\n");
        for comp in &["x", "y", "z", "w"] {
            // Replace inN[i] with vN.comp
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
        src.push_str("    }\n");
        // Scalar tail with grid-stride
        src.push_str("    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {\n");
        let mut tail_expr = scalar_expr;
        for j in (0..n_inputs).rev() {
            tail_expr = tail_expr.replace(
                &format!("in{j}[i]"),
                &format!("__ldg(&in{j}[j])"),
            );
        }
        src.push_str(&format!("        out[j] = {tail_expr};\n"));
        src.push_str("    }\n");
        src.push_str("}\n");
    } else {
        // f64 scalar kernel with __ldg prefetch
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
        // Replace inN[i] with __ldg(&inN[i]) for read-only cache hint
        let mut ldg_expr = gpu_expr.to_string();
        for j in (0..n_inputs).rev() {
            ldg_expr = ldg_expr.replace(
                &format!("in{j}[i]"),
                &format!("__ldg(&in{j}[i])"),
            );
        }
        src.push_str("        out[i] = ");
        src.push_str(&ldg_expr);
        src.push_str(";\n");
        src.push_str("    }\n}\n");
    }
    src
}

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
            let src = fuse_kernel_source(gpu_expr, n_inputs, type_name, &kernel_name, reg_estimate);
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
            let ob = optimal_block_size(func);
            let mut map = ctx.kernels.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(kernel_name.clone(), KernelEntry { func, _module: module, optimal_block: ob });
        }
    }

    let (func, ob) = get_kernel(ctx, &kernel_name).unwrap_or_else(|e| panic!("{e}"));
    let out_buf = CuBuffer::alloc_async(&ctx.stream, n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("CUDA alloc: {e}"));
    let n_u32 = n as u32;
    // f32 fused kernels use persistent grid-stride with float4
    let grid = if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        persistent_grid_f32(ctx, n, ob)
    } else {
        n.div_ceil(ob as usize) as u32
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
            (ob, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        ).unwrap_or_else(|e| panic!("CUDA launch {kernel_name}: {e}"));
    }
    CudaStorage::new(nrows, ncols, out_buf)
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
