use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::cublaslt::result::CublasError;
pub(super) use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUmodule};
use cudarc::driver::{CudaContext as CudarcContext, CudaStream, result};
use cudarc::nvrtc;

use crate::cuda_backend::NablaCudaGraph;
use crate::gpu_common::{
    EnsureCache, MemoryPool, RtcStorage, bucket_size, grid_1d, lock_or_recover, type_suffix,
};
use crate::kernels_cu::{self, BLOCK_SIZE, REDUCE_GRID_CAP};
use crate::scalar::Scalar;

#[inline]
fn wmma_jit_enabled() -> bool {
    std::env::var("NABLA_DISABLE_WMMA_JIT").map_or(true, |v| {
        !matches!(v.as_str(), "1" | "true" | "TRUE" | "True")
    })
}

static TRANSFER_H2D: AtomicU64 = AtomicU64::new(0);
static TRANSFER_D2H: AtomicU64 = AtomicU64::new(0);

#[inline]
fn transfer_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NABLA_TRANSFER_DEBUG").as_deref(),
            Ok("1" | "true" | "TRUE" | "True")
        )
    })
}

#[inline]
pub(super) fn record_h2d() {
    if transfer_debug_enabled() {
        TRANSFER_H2D.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(super) fn record_d2h() {
    if transfer_debug_enabled() {
        TRANSFER_D2H.fetch_add(1, Ordering::Relaxed);
    }
}

/// Returns `(h2d_count, d2h_count)` since last reset. Only active when `NABLA_TRANSFER_DEBUG=1`.
pub fn cuda_transfer_stats() -> (u64, u64) {
    (
        TRANSFER_H2D.load(Ordering::Relaxed),
        TRANSFER_D2H.load(Ordering::Relaxed),
    )
}

pub fn cuda_transfer_stats_reset() {
    TRANSFER_H2D.store(0, Ordering::Relaxed);
    TRANSFER_D2H.store(0, Ordering::Relaxed);
}

#[derive(Debug)]
pub enum CudaError {
    Driver(cudarc::driver::DriverError),
    Nvrtc(nvrtc::CompileError),
    CublasLt(CublasError),
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

impl From<CublasError> for CudaError {
    fn from(e: CublasError) -> Self {
        Self::CublasLt(e)
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver: {e}"),
            Self::Nvrtc(e) => write!(f, "NVRTC: {e}"),
            Self::CublasLt(e) => write!(f, "cuBLASLt: {e}"),
            Self::KernelNotFound(s) => write!(f, "kernel not found: {s}"),
            Self::NullPtr => write!(f, "null pointer"),
        }
    }
}

pub type CudaResult<T> = Result<T, CudaError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epilogue {
    None,
    Relu,
    Gelu,
    Bias,
    ReluBias,
    GeluBias,
}

#[inline]
pub(super) fn expect_ok<T, E: std::fmt::Display>(result: Result<T, E>, message: &str) -> T {
    result.unwrap_or_else(|e| panic!("{message}: {e}"))
}

pub(super) trait ResultExt<T> {
    fn or_panic(self, context: &str) -> T;
}

impl<T, E: std::fmt::Display> ResultExt<T> for Result<T, E> {
    #[inline]
    fn or_panic(self, context: &str) -> T {
        self.unwrap_or_else(|e| {
            if context.is_empty() {
                panic!("{e}")
            } else {
                panic!("{context}: {e}")
            }
        })
    }
}

#[inline]
pub(super) fn alloc_out<T: Scalar>(ctx: &CudaCtx, n: usize) -> CuBuffer {
    expect_ok(
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()),
        "CUDA alloc",
    )
}

pub(super) type CudaPool = MemoryPool<CUdeviceptr>;

/// Allocate a bucketed GPU block for the pool. Used by pre_warm and auto-warm.
#[inline]
pub(super) fn pool_alloc_block(
    stream: cudarc::driver::sys::CUstream,
    size: usize,
) -> Option<(CUdeviceptr, usize)> {
    let bsz = bucket_size(size);
    // SAFETY: allocating device memory via cuMemAllocAsync.
    let dptr = unsafe { result::malloc_async(stream, bsz) }.ok()?;
    Some((dptr, bsz))
}

pub struct CuBuffer {
    pub(crate) ptr: CUdeviceptr,
    alloc_size: usize,
    pooled: bool,
}

impl CuBuffer {
    #[inline]
    fn empty(size_bytes: usize) -> Option<Self> {
        (size_bytes == 0).then(|| Self {
            ptr: 0,
            alloc_size: 0,
            pooled: false,
        })
    }

    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, size_bytes: usize) -> Self {
        Self {
            ptr,
            alloc_size: size_bytes,
            pooled: false,
        }
    }

    pub unsafe fn borrow_ptr(ptr: CUdeviceptr, _size_bytes: usize) -> Self {
        Self {
            ptr,
            alloc_size: 0,
            pooled: false,
        }
    }

    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    fn maybe_auto_warm(ctx: &CudaCtx) {
        let warm_sizes = {
            let mut pool = ctx.pool_lock();
            if !pool.should_auto_warm() {
                return;
            }
            let sizes = pool.stop_recording();
            if sizes.is_empty() {
                pool.set_warmed();
                return;
            }
            sizes
        };
        let stream = ctx.stream.cu_stream();
        let mut pool = ctx.pool_lock();
        let added = pool.pre_warm(&warm_sizes, |sz| pool_alloc_block(stream, sz));
        pool.set_warmed();
        if crate::gpu_common::pool_debug_enabled() {
            eprintln!(
                "[nabla pool] auto-warm: recorded {} allocs, added {} blocks",
                warm_sizes.len(),
                added
            );
            pool.print_diagnostics();
        }
    }

    fn alloc_from_pool(
        stream: &Arc<CudaStream>,
        size_bytes: usize,
    ) -> CudaResult<(CUdeviceptr, usize)> {
        if size_bytes == 0 {
            return Ok((0, 0));
        }
        let capturing = super::cuda_graph_is_capturing();
        let ctx = get_ctx();
        let alloc_size = bucket_size(size_bytes);
        if !capturing {
            let (hit, should_warm) = {
                let mut pool = ctx.pool_lock();
                let warm = pool.should_auto_warm();
                let hit = pool.try_alloc(size_bytes).map(|(ptr, size_class)| {
                    pool.allocated_bytes += size_class;
                    (ptr, size_class)
                });
                (hit, warm)
            };
            if should_warm {
                Self::maybe_auto_warm(ctx);
            }
            if let Some(r) = hit {
                return Ok(r);
            }
        }
        let dptr = match unsafe { result::malloc_async(stream.cu_stream(), alloc_size) } {
            Ok(ptr) => ptr,
            Err(_) if !capturing => {
                let cu = stream.cu_stream();
                let mut pool = ctx.pool_lock();
                pool.trim(0, |ptr, _| unsafe {
                    // SAFETY: ptr was allocated via cuMemAllocAsync; must free with cuMemFreeAsync.
                    let _ = result::free_async(ptr, cu);
                });
                drop(pool);
                let _ = stream.synchronize();
                unsafe { result::malloc_async(cu, alloc_size)? }
            }
            Err(e) => return Err(e.into()),
        };
        if !capturing {
            let mut pool = ctx.pool_lock();
            pool.allocated_bytes += alloc_size;
        }
        Ok((dptr, alloc_size))
    }

    fn alloc_pooled(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        let pooled = !super::cuda_graph_is_capturing();
        let (dptr, alloc_size) = Self::alloc_from_pool(stream, size_bytes)?;
        Ok(Self {
            ptr: dptr,
            alloc_size,
            pooled,
        })
    }

    pub(super) fn alloc_zeros(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if let Some(buf) = Self::empty(size_bytes) {
            return Ok(buf);
        }
        let buf = Self::alloc_pooled(stream, size_bytes)?;
        // SAFETY: zeroing allocated device memory.
        unsafe { result::memset_d8_async(buf.ptr, 0, buf.alloc_size, stream.cu_stream())? };
        Ok(buf)
    }

    pub(super) fn alloc_async(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if let Some(buf) = Self::empty(size_bytes) {
            return Ok(buf);
        }
        Self::alloc_pooled(stream, size_bytes)
    }

    /// Upload a POD slice to GPU. Caller guarantees `T: Copy`.
    fn upload_pod<T: Copy>(stream: &Arc<CudaStream>, data: &[T]) -> CudaResult<Self> {
        let bytes = std::mem::size_of_val(data);
        if let Some(buf) = Self::empty(bytes) {
            return Ok(buf);
        }
        record_h2d();
        let pooled = !super::cuda_graph_is_capturing();
        let (dptr, alloc_size) = Self::alloc_from_pool(stream, bytes)?;
        // SAFETY: T is POD (Copy); uploading raw bytes to GPU.
        unsafe { result::memcpy_htod_async(dptr, data, stream.cu_stream())? };
        Ok(Self {
            ptr: dptr,
            alloc_size,
            pooled,
        })
    }

    pub(super) fn from_host<T: Scalar>(stream: &Arc<CudaStream>, data: &[T]) -> CudaResult<Self> {
        Self::upload_pod(stream, data)
    }

    pub(super) fn from_host_u32(stream: &Arc<CudaStream>, data: &[u32]) -> CudaResult<Self> {
        Self::upload_pod(stream, data)
    }

    pub(super) fn copy_to_host<T: Scalar>(
        &self,
        _stream: &Arc<CudaStream>,
        out: &mut [T],
    ) -> CudaResult<()> {
        if super::cuda_graph_is_capturing() {
            panic!("CUDA Graph capture forbids D2H readback; call prefetch() before capture.");
        }
        let bytes = std::mem::size_of_val(out);
        if bytes > 0 {
            record_d2h();
            // SAFETY: out is properly sized and T is POD.
            unsafe { result::memcpy_dtoh_sync(out, self.ptr)? };
        }
        Ok(())
    }

    pub(super) fn from_host_nonblocking<T: Scalar>(
        compute_stream: &Arc<CudaStream>,
        copy_stream: &Arc<CudaStream>,
        data: &[T],
    ) -> CudaResult<Self> {
        let bytes = std::mem::size_of_val(data);
        if let Some(buf) = Self::empty(bytes) {
            return Ok(buf);
        }
        record_h2d();
        let buf = Self::alloc_async(compute_stream, bytes)?;
        unsafe { result::memcpy_htod_async(buf.ptr, data, copy_stream.cu_stream())? };
        let event =
            result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
        unsafe { result::event::record(event, copy_stream.cu_stream())? };
        unsafe {
            result::stream::wait_event(
                compute_stream.cu_stream(),
                event,
                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )?;
        }
        unsafe { result::event::destroy(event)? };
        Ok(buf)
    }
}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 && self.alloc_size > 0 {
            if self.pooled {
                let ctx = get_ctx();
                let mut pool = ctx.pool_lock();
                pool.allocated_bytes = pool.allocated_bytes.saturating_sub(self.alloc_size);
                if super::cuda_graph_is_capturing() {
                    unsafe {
                        // SAFETY: ptr was allocated via cuMemAllocAsync; free during capture to avoid double-free on replay.
                        let stream = get_ctx().stream.cu_stream();
                        let _ = result::free_async(self.ptr, stream);
                    }
                } else {
                    pool.release(self.ptr, self.alloc_size);
                    pool.maybe_gc(|ptr, _| unsafe {
                        // SAFETY: ptr was allocated via cuMemAllocAsync; must be freed with cuMemFreeAsync.
                        let stream = get_ctx().stream.cu_stream();
                        let _ = result::free_async(ptr, stream);
                    });
                }
            } else {
                unsafe {
                    // SAFETY: ptr was allocated via cuMemAllocAsync.
                    let _ = result::free_async(self.ptr, get_ctx().stream.cu_stream());
                }
            }
        }
    }
}

pub type CudaStorage<T> = RtcStorage<CuBuffer, T>;

// SAFETY: CuBuffer is a raw GPU pointer (u64) + usize — trivially Send+Sync.
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

pub(super) struct KernelEntry {
    pub(super) func: CUfunction,
    pub(super) _module: CUmodule,
}

// SAFETY: CUfunction/*mut CUfunc_st and CUmodule/*mut CUmod_st are opaque CUDA handles.
unsafe impl Send for KernelEntry {}
unsafe impl Sync for KernelEntry {}

pub(super) struct CublasHandle(pub(super) cublas_sys::cublasHandle_t);

// SAFETY: cublasHandle_t is bound to the CUDA context, used from OnceLock singleton only.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

#[derive(Clone, Copy)]
pub(super) struct SyncFn(pub(super) CUfunction);
unsafe impl Send for SyncFn {}
unsafe impl Sync for SyncFn {}

pub(super) struct CublasLtHandle(pub(super) cublaslt_sys::cublasLtHandle_t);
// SAFETY: cublasLtHandle_t is bound to the CUDA context, used from OnceLock singleton only.
unsafe impl Send for CublasLtHandle {}
unsafe impl Sync for CublasLtHandle {}

pub(super) struct CudaCtx {
    pub(super) stream: Arc<CudaStream>,
    pub(super) copy_stream: Arc<CudaStream>,
    pub(super) kernels: RwLock<HashMap<String, KernelEntry>>,
    pub(super) pool: Mutex<CudaPool>,
    pub(super) blas: CublasHandle,
    pub(super) blas_lt: CublasLtHandle,
    pub(super) blas_lt_workspace: CUdeviceptr,
    pub(super) blas_lt_workspace_size: usize,
    pub(super) graphs: Mutex<HashMap<String, Arc<NablaCudaGraph>>>,
    pub(super) reduce_scratch: CUdeviceptr,
    pub(super) reduce_funcs: [SyncFn; 18],
    pub(super) d2h_mutex: Mutex<()>,
}

impl CudaCtx {
    #[inline]
    pub(super) fn pool_lock(&self) -> std::sync::MutexGuard<'_, CudaPool> {
        self.pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[inline]
    pub(super) fn kernels_read(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, HashMap<String, KernelEntry>> {
        self.kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[inline]
    pub(super) fn kernels_write(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, HashMap<String, KernelEntry>> {
        self.kernels
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub(super) fn query_compute_capability() -> (i32, i32) {
    // SAFETY: querying device attributes via cudarc driver-level API.
    let major = unsafe {
        result::device::get_attribute(
            0,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(7);
    let minor = unsafe {
        result::device::get_attribute(
            0,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    (major, minor)
}

pub(super) fn nvrtc_arch(major: i32, minor: i32) -> &'static str {
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

pub(super) fn get_ctx() -> &'static CudaCtx {
    static CTX: OnceLock<CudaCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = expect_ok(CudarcContext::new(0), "CUDA device 0 init failed");
        let stream = expect_ok(ctx.new_stream(), "CUDA compute stream creation failed");
        let copy_stream = expect_ok(ctx.new_stream(), "CUDA copy stream creation failed");
        let (major, minor) = query_compute_capability();
        let arch: &'static str = nvrtc_arch(major, minor);
        let has_wmma = major >= 7;
        // SAFETY: initializing cuBLAS handle and binding it to the default stream.
        let blas_raw = expect_ok(cublas_result::create_handle(), "cuBLAS init failed");
        unsafe {
            expect_ok(
                cublas_result::set_stream(blas_raw, stream.cu_stream() as cublas_sys::cudaStream_t),
                "cuBLAS set_stream failed",
            )
        };
        unsafe {
            let _ = cublas_sys::cublasSetMathMode(
                blas_raw,
                cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
            );
        }
        // SAFETY: initializing cublasLt handle.
        let blas_lt_raw = expect_ok(cublaslt_result::create_handle(), "cublasLt init failed");
        let blas_lt_workspace_size = if major >= 9 {
            32 * 1024 * 1024
        } else {
            4 * 1024 * 1024
        };
        let blas_lt_workspace = unsafe {
            expect_ok(
                result::malloc_async(stream.cu_stream(), blas_lt_workspace_size),
                "cublasLt workspace alloc failed",
            )
        };

        let reduce_scratch_size = REDUCE_GRID_CAP as usize * 8 + 16;
        let reduce_scratch = unsafe {
            expect_ok(
                result::malloc_async(stream.cu_stream(), reduce_scratch_size),
                "CUDA reduce scratch alloc failed",
            )
        };
        let cuda_ctx = CudaCtx {
            stream,
            copy_stream,
            kernels: RwLock::new(HashMap::new()),
            pool: Mutex::new(CudaPool::new()),
            blas: CublasHandle(blas_raw),
            blas_lt: CublasLtHandle(blas_lt_raw),
            blas_lt_workspace,
            blas_lt_workspace_size,
            graphs: Mutex::new(HashMap::new()),
            reduce_scratch,
            reduce_funcs: [SyncFn(std::ptr::null_mut()); 18],
            d2h_mutex: Mutex::new(()),
        };
        if let Err(e) = compile_all_kernels(&cuda_ctx, arch) {
            panic!("CUDA kernel compilation failed: {e}");
        }
        if has_wmma && wmma_jit_enabled() {
            if let Err(e) = compile_wmma_kernels(&cuda_ctx, arch) {
                eprintln!("WMMA kernel compilation failed (falling back to tiled): {e}");
            }
        }
        let rf = |name: &str| expect_ok(get_kernel(&cuda_ctx, name), "reduce kernel missing");
        // SAFETY: we're still constructing the OnceLock value, so mutating is fine.
        let ctx_ptr = &cuda_ctx as *const CudaCtx as *mut CudaCtx;
        unsafe {
            (*ctx_ptr).reduce_funcs = [
                SyncFn(rf("k_sum_f32")),
                SyncFn(rf("k_max_f32")),
                SyncFn(rf("k_min_f32")),
                SyncFn(rf("k_sum_f16")),
                SyncFn(rf("k_max_f16")),
                SyncFn(rf("k_min_f16")),
                SyncFn(rf("k_sum_f64")),
                SyncFn(rf("k_max_f64")),
                SyncFn(rf("k_min_f64")),
                SyncFn(rf("k_sum_fp8e4m3")),
                SyncFn(rf("k_max_fp8e4m3")),
                SyncFn(rf("k_min_fp8e4m3")),
                SyncFn(rf("k_sum_fp8e5m2")),
                SyncFn(rf("k_max_fp8e5m2")),
                SyncFn(rf("k_min_fp8e5m2")),
                SyncFn(rf("k_sum_fp4e2m1")),
                SyncFn(rf("k_max_fp4e2m1")),
                SyncFn(rf("k_min_fp4e2m1")),
            ];
        }
        cuda_ctx
    })
}

pub fn cuda_compute_stream() -> Arc<CudaStream> {
    get_ctx().stream.clone()
}

pub fn cuda_pre_warm_pool(sizes: &[usize]) -> CudaResult<usize> {
    let ctx = get_ctx();
    let stream = ctx.stream.cu_stream();
    let mut pool = ctx.pool_lock();
    let added = pool.pre_warm(sizes, |sz| pool_alloc_block(stream, sz));
    Ok(added)
}

pub fn cuda_pool_start_recording() {
    let ctx = get_ctx();
    let mut pool = ctx.pool_lock();
    pool.start_recording();
}

pub fn cuda_pool_stop_recording_and_warm() -> CudaResult<usize> {
    let ctx = get_ctx();
    let mut pool = ctx.pool_lock();
    if pool.is_warmed() {
        return Ok(0);
    }
    let sizes = pool.stop_recording();
    if sizes.is_empty() {
        return Ok(0);
    }
    let stream = ctx.stream.cu_stream();
    let added = pool.pre_warm(&sizes, |sz| pool_alloc_block(stream, sz));
    pool.set_warmed();
    if crate::gpu_common::pool_debug_enabled() {
        eprintln!(
            "[nabla pool] auto pre-warm: recorded {} allocs, added {} blocks",
            sizes.len(),
            added
        );
        pool.print_diagnostics();
    }
    Ok(added)
}

pub fn cuda_pool_diagnostics() {
    let ctx = get_ctx();
    let pool = ctx.pool_lock();
    pool.print_diagnostics();
}

pub fn cuda_synchronize() {
    // SAFETY: stream is valid; cuStreamSynchronize has no side effects beyond waiting.
    unsafe { cudarc::driver::sys::cuStreamSynchronize(get_ctx().stream.cu_stream()) };
}

pub fn cuda_upload_u32(data: &[u32]) -> CuBuffer {
    let ctx = get_ctx();
    CuBuffer::from_host_u32(&ctx.stream, data).or_panic("CUDA upload u32")
}

pub(super) const KERNEL_NAMES: &[&str] = &[
    "k_cast_f32_to_f16",
    "k_cast_f16_to_f32",
    "k_cast_f64_to_f32",
    "k_cast_f32_to_f64",
    "k_cast_f32_to_fp8e4m3",
    "k_cast_fp8e4m3_to_f32",
    "k_cast_f32_to_fp8e5m2",
    "k_cast_fp8e5m2_to_f32",
    "k_cast_f32_to_fp4e2m1",
    "k_cast_fp4e2m1_to_f32",
    "k_masked_fill_f32",
    "k_masked_fill_f64",
    "k_masked_fill_f16",
    "k_masked_fill_fp8e4m3",
    "k_masked_fill_fp8e5m2",
    "k_masked_fill_fp4e2m1",
    "k_where_f32",
    "k_where_f64",
    "k_where_f16",
    "k_where_fp8e4m3",
    "k_where_fp8e5m2",
    "k_where_fp4e2m1",
    "k_neg_f32",
    "k_recip_f32",
    "k_exp_f32",
    "k_ln_f32",
    "k_log1p_f32",
    "k_sin_f32",
    "k_cos_f32",
    "k_tan_f32",
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
    "k_neg_f16",
    "k_recip_f16",
    "k_exp_f16",
    "k_ln_f16",
    "k_log1p_f16",
    "k_sin_f16",
    "k_cos_f16",
    "k_tan_f16",
    "k_tanh_f16",
    "k_sqrt_f16",
    "k_abs_f16",
    "k_ceil_f16",
    "k_floor_f16",
    "k_round_f16",
    "k_erf_f16",
    "k_asin_f16",
    "k_acos_f16",
    "k_atan_f16",
    "k_atan2_f16",
    "k_atan2_fp8e4m3",
    "k_atan2_fp8e5m2",
    "k_atan2_fp4e2m1",
    "k_sinh_f16",
    "k_cosh_f16",
    "k_asinh_f16",
    "k_acosh_f16",
    "k_atanh_f16",
    "k_log2_f16",
    "k_log10_f16",
    "k_sigmoid_f16",
    "k_silu_f16",
    "k_mish_f16",
    "k_leaky_relu_f16",
    "k_elu_f16",
    "k_hardswish_f16",
    "k_add_f16",
    "k_sub_f16",
    "k_emul_f16",
    "k_ediv_f16",
    "k_scale_f16",
    "k_powf_f16",
    "k_fill_f16",
    "k_transpose_f16",
    "k_matmul_f16",
    "k_sum_f16",
    "k_max_f16",
    "k_min_f16",
    "k_relu_bwd_f16",
    "k_leaky_relu_bwd_f16",
    "k_elu_bwd_f16",
    "k_gelu_bwd_f16",
    "k_abs_bwd_f16",
    "k_expand_f16",
    "k_mse_sum_fwd_f16",
    "k_mse_sum_bwd_f16",
    "k_multi_axpy3_f16",
    "k_axpy_f16",
    "k_softmax_f16",
    "k_layer_norm_f16",
    "k_rms_norm_f16",
    "k_group_norm_f16",
    "k_sum_axis1_f16",
    "k_max_axis1_f16",
    "k_embedding_f16",
    "k_cumsum_axis1_f16",
    "k_cumprod_axis1_f16",
    "k_prod_partial_f16",
    "k_max_pool2d_with_idx_f16",
    "k_max_pool2d_f16",
    "k_avg_pool2d_f16",
    "k_adaptive_avg_pool2d_f16",
    "k_im2col_f16",
    "k_im1col_f16",
    "k_im3col_f16",
    "k_conv_transpose2d_f16",
    "k_batch_norm_stats_f16",
    "k_batch_norm_fwd_f16",
    "k_cross_entropy_f16",
    "k_sdpa_f16",
    "k_masked_fill_bf16",
    "k_where_bf16",
    "k_neg_bf16",
    "k_recip_bf16",
    "k_exp_bf16",
    "k_ln_bf16",
    "k_log1p_bf16",
    "k_sin_bf16",
    "k_cos_bf16",
    "k_tan_bf16",
    "k_tanh_bf16",
    "k_sqrt_bf16",
    "k_abs_bf16",
    "k_ceil_bf16",
    "k_floor_bf16",
    "k_round_bf16",
    "k_erf_bf16",
    "k_asin_bf16",
    "k_acos_bf16",
    "k_atan_bf16",
    "k_atan2_bf16",
    "k_sinh_bf16",
    "k_cosh_bf16",
    "k_asinh_bf16",
    "k_acosh_bf16",
    "k_atanh_bf16",
    "k_log2_bf16",
    "k_log10_bf16",
    "k_sigmoid_bf16",
    "k_silu_bf16",
    "k_mish_bf16",
    "k_leaky_relu_bf16",
    "k_elu_bf16",
    "k_hardswish_bf16",
    "k_add_bf16",
    "k_sub_bf16",
    "k_emul_bf16",
    "k_ediv_bf16",
    "k_scale_bf16",
    "k_powf_bf16",
    "k_fill_bf16",
    "k_transpose_bf16",
    "k_matmul_bf16",
    "k_sum_bf16",
    "k_max_bf16",
    "k_min_bf16",
    "k_relu_bwd_bf16",
    "k_leaky_relu_bwd_bf16",
    "k_elu_bwd_bf16",
    "k_gelu_bwd_bf16",
    "k_abs_bwd_bf16",
    "k_expand_bf16",
    "k_mse_sum_fwd_bf16",
    "k_mse_sum_bwd_bf16",
    "k_multi_axpy3_bf16",
    "k_axpy_bf16",
    "k_softmax_bf16",
    "k_layer_norm_bf16",
    "k_rms_norm_bf16",
    "k_group_norm_bf16",
    "k_sum_axis1_bf16",
    "k_max_axis1_bf16",
    "k_embedding_bf16",
    "k_cumsum_axis1_bf16",
    "k_cumprod_axis1_bf16",
    "k_prod_partial_bf16",
    "k_max_pool2d_with_idx_bf16",
    "k_max_pool2d_bf16",
    "k_avg_pool2d_bf16",
    "k_adaptive_avg_pool2d_bf16",
    "k_im2col_bf16",
    "k_im1col_bf16",
    "k_im3col_bf16",
    "k_conv_transpose2d_bf16",
    "k_batch_norm_stats_bf16",
    "k_batch_norm_fwd_bf16",
    "k_cross_entropy_bf16",
    "k_sdpa_bf16",
    "k_cast_f32_to_bf16",
    "k_cast_bf16_to_f32",
    "k_neg_f64",
    "k_recip_f64",
    "k_exp_f64",
    "k_ln_f64",
    "k_log1p_f64",
    "k_sin_f64",
    "k_cos_f64",
    "k_tan_f64",
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
    "k_wht_f32",
    "k_wht_f64",
    "k_wht_bf16",
    "k_wht_inverse_f32",
    "k_wht_inverse_f64",
    "k_wht_inverse_bf16",
    // Indexing kernels (k_indexing.cuh) — f32/f64 only
    "k_submatrix_f32",
    "k_submatrix_f64",
    "k_slice_set_f32",
    "k_slice_set_f64",
    "k_gather_rows_u32idx_f32",
    "k_gather_rows_u32idx_f64",
    "k_gather_f32",
    "k_gather_f64",
    "k_scatter_f32",
    "k_scatter_f64",
    "k_index_select_f32",
    "k_index_select_f64",
    "k_scatter_add_dim0_u32idx_f32",
    "k_scatter_add_dim0_u32idx_f64",
    "k_scatter_add_dim1_u32idx_f32",
    "k_scatter_add_dim1_u32idx_f64",
    "k_sort_rows_f32",
    "k_sort_rows_f64",
];

pub(super) const FP8_SUFFIXES: &[&str] = &["fp8e4m3", "fp8e5m2", "fp4e2m1"];
pub(super) const FP8_UNARY_OPS: &[&str] = &[
    "neg",
    "recip",
    "exp",
    "ln",
    "log1p",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sqrt",
    "abs",
    "ceil",
    "floor",
    "round",
    "erf",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "log2",
    "log10",
    "sigmoid",
    "silu",
    "mish",
    "leaky_relu",
    "elu",
    "hardswish",
];
pub(super) const FP8_BINARY_OPS: &[&str] = &["add", "sub", "emul", "ediv"];
pub(super) const FP8_EXTRA_OPS: &[&str] = &[
    "scale",
    "powf",
    "fill",
    "transpose",
    "matmul",
    "sum",
    "max",
    "min",
    "softmax",
    "layer_norm",
    "rms_norm",
    "group_norm",
    "sum_axis1",
    "max_axis1",
    "embedding",
    "cumsum_axis1",
    "cumprod_axis1",
    "prod_partial",
    "max_pool2d_with_idx",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "im2col",
    "im1col",
    "im3col",
    "conv_transpose2d",
    "batch_norm_stats",
    "batch_norm_fwd",
    "cross_entropy",
    "sdpa",
    "axpy",
    "expand",
    "mse_sum_fwd",
    "mse_sum_bwd",
];

pub(super) const WMMA_KERNEL_NAMES: &[&str] = &["k_matmul_wmma_f16", "k_matmul_wmma_bf16"];

pub(super) fn nvrtc_include_paths() -> Vec<String> {
    [
        "/usr/include",
        "/usr/include/aarch64-linux-gnu",
        "/usr/include/x86_64-linux-gnu",
    ]
    .into_iter()
    .filter(|p| std::path::Path::new(p).is_dir())
    .map(ToString::to_string)
    .collect()
}

pub(super) fn compile_all_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
    let ptx = nvrtc::compile_ptx_with_opts(
        kernels_cu::KERNELS,
        nvrtc::CompileOptions {
            arch: Some(arch),
            include_paths: nvrtc_include_paths(),
            ..Default::default()
        },
    )?;
    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled PTX data as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };
    let mut map = ctx.kernels_write();
    let load_kernel = |map: &mut HashMap<String, KernelEntry>, name: String| -> CudaResult<()> {
        let c_fn = CString::new(name.as_str()).map_err(|_| CudaError::NullPtr)?;
        // SAFETY: getting function handle from loaded module.
        let func = unsafe { result::module::get_function(module, c_fn) }.map_err(|e| {
            eprintln!("[nabla] CUDA kernel not found in PTX: {name}");
            CudaError::from(e)
        })?;
        map.insert(
            name,
            KernelEntry {
                func,
                _module: module,
            },
        );
        Ok(())
    };
    for &name in KERNEL_NAMES {
        load_kernel(&mut map, name.to_owned())?;
    }
    for &suffix in FP8_SUFFIXES {
        for &op in FP8_UNARY_OPS
            .iter()
            .chain(FP8_BINARY_OPS)
            .chain(FP8_EXTRA_OPS)
        {
            load_kernel(&mut map, format!("k_{op}_{suffix}"))?;
        }
    }
    Ok(())
}

pub(super) fn compile_wmma_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
    let src = kernels_cu::WMMA_KERNELS;
    if src.is_empty() {
        return Ok(());
    }
    let ptx = nvrtc::compile_ptx_with_opts(
        src,
        nvrtc::CompileOptions {
            arch: Some(arch),
            include_paths: nvrtc_include_paths(),
            ..Default::default()
        },
    )?;
    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled WMMA PTX as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };
    let mut map = ctx.kernels_write();
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

#[inline]
pub(super) fn get_kernel(ctx: &CudaCtx, name: &str) -> CudaResult<CUfunction> {
    ctx.kernels_read()
        .get(name)
        .map(|e| e.func)
        .ok_or_else(|| CudaError::KernelNotFound(name.to_owned()))
}

#[inline]
fn cuda_alloc_storage<T: Scalar>(nrows: usize, ncols: usize, zero: bool) -> CudaStorage<T> {
    let ctx = get_ctx();
    let bytes = nrows * ncols * std::mem::size_of::<T>();
    let buf = if zero {
        expect_ok(CuBuffer::alloc_zeros(&ctx.stream, bytes), "CUDA alloc")
    } else {
        expect_ok(CuBuffer::alloc_async(&ctx.stream, bytes), "CUDA alloc")
    };
    CudaStorage::new(nrows, ncols, buf)
}

pub(crate) fn cuda_zeros<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
    cuda_alloc_storage(nrows, ncols, true)
}

pub(crate) fn cuda_empty<T: Scalar>(nrows: usize, ncols: usize) -> CudaStorage<T> {
    cuda_alloc_storage(nrows, ncols, false)
}

pub(crate) fn cuda_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let buf = expect_ok(
            CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()),
            "CUDA alloc",
        );
        // SAFETY: T is f32 (verified by TypeId check); transmute to read bits.
        let bits: u32 = unsafe { std::mem::transmute_copy::<T, f32>(&val) }.to_bits();
        // SAFETY: buf.ptr is a valid device pointer; n is the element count.
        unsafe {
            let res =
                cudarc::driver::sys::cuMemsetD32Async(buf.ptr, bits, n, ctx.stream.cu_stream());
            assert_eq!(
                res,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "cuMemsetD32Async failed: {res:?}"
            );
        }
        return CudaStorage::new(nrows, ncols, buf);
    }

    if std::mem::size_of::<T>() == 2
        && (std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>())
    {
        let buf = expect_ok(CuBuffer::alloc_async(&ctx.stream, n * 2), "CUDA alloc");
        // SAFETY: T is bf16 or f16 (verified above); both are 2 bytes, read raw bits as u16.
        let bits: u16 = unsafe { std::mem::transmute_copy::<T, u16>(&val) };
        // SAFETY: buf.ptr is a valid device pointer; n is the element count of 16-bit values.
        unsafe {
            let res =
                cudarc::driver::sys::cuMemsetD16Async(buf.ptr, bits, n, ctx.stream.cu_stream());
            assert_eq!(
                res,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "cuMemsetD16Async failed: {res:?}"
            );
        }
        return CudaStorage::new(nrows, ncols, buf);
    }

    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = super::kernel_name_buf(&mut nbuf, "fill", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA fill kernel lookup");
    let n_u32 = n as u32;
    // SAFETY: val is passed by-value on the stack; kernel reads it as the matching GPU type.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &val as *const T as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch fill",
        );
    }
    CudaStorage::new(nrows, ncols, out_buf)
}

pub(crate) fn cuda_from_fn<T: Scalar>(
    nrows: usize,
    ncols: usize,
    _f: impl FnMut(usize, usize) -> T,
) -> CudaStorage<T> {
    if matches!(
        std::env::var("NABLA_FAST_FROM_FN").as_deref(),
        Ok("1" | "true" | "TRUE" | "True")
    ) {
        return cuda_zeros(nrows, ncols);
    }
    panic!("nabla: Tensor::from_fn is CPU-only; CUDA fallback is forbidden");
}

pub(crate) fn cuda_from_vec_async<T: Scalar>(
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let buf = expect_ok(
        CuBuffer::from_host_nonblocking(&ctx.stream, &ctx.copy_stream, &data),
        "CUDA async upload",
    );
    CudaStorage::new_cached(nrows, ncols, buf, data)
}

pub(crate) fn cuda_one_hot_from_indices<T: Scalar>(
    indices: &CudaStorage<T>,
    n_classes: usize,
) -> CudaStorage<T> {
    cuda_zeros(indices.nrows, n_classes)
}

pub(crate) fn cuda_get<T: Scalar>(s: &CudaStorage<T>, r: usize, c: usize) -> T {
    s.ensure_cache();
    let guard = lock_or_recover(&s.host_cache);
    guard.as_ref().map_or_else(
        || panic!("cuda_get: cache missing after ensure_cache"),
        |cache| cache[r * s.ncols + c],
    )
}

pub(crate) fn cuda_set<T: Scalar>(s: &mut CudaStorage<T>, r: usize, c: usize, v: T) {
    s.invalidate_cache();
    let ctx = get_ctx();
    let offset = (r * s.ncols + c) * std::mem::size_of::<T>();
    let src = std::slice::from_ref(&v);
    // SAFETY: uploading single element to correct offset in GPU buffer.
    unsafe {
        let _ = result::memcpy_htod_async(s.buf.ptr + offset as u64, src, ctx.stream.cu_stream());
    }
}

pub(crate) fn cuda_clone<T: Scalar>(s: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let bytes = s.n() * std::mem::size_of::<T>();
    let new_buf = expect_ok(CuBuffer::alloc_async(&ctx.stream, bytes), "CUDA alloc");
    if bytes > 0 {
        // SAFETY: device-to-device copy of same-sized buffers.
        unsafe {
            let _ =
                result::memcpy_dtod_async(new_buf.ptr, s.buf.ptr, bytes, ctx.stream.cu_stream());
        }
    }
    CudaStorage {
        nrows: s.nrows,
        ncols: s.ncols,
        buf: new_buf,
        host_cache: Mutex::new(None),
    }
}

pub(crate) fn cuda_transpose<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = super::kernel_name_buf(&mut nbuf, "transpose", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    unsafe {
        expect_ok(
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
            ),
            "CUDA launch transpose",
        );
    }
    CudaStorage::new(a.ncols, a.nrows, out_buf)
}
