use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::cublaslt::result::CublasError;
pub(super) use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUmodule};
use cudarc::driver::{CudaContext as CudarcContext, CudaStream, result};
use cudarc::nvrtc;

use crate::cuda_backend::NablaCudaGraph;
use crate::gpu_common::{
    EnsureCache, LARGE_ALLOC_SIZE, MemoryPool, RtcStorage, SMALL_ALLOC_SIZE, SMALL_LARGE_BOUNDARY,
    lock_or_recover, round_size,
};
use crate::kernels_cu::REDUCE_GRID_CAP;
use crate::scalar::Scalar;

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
    match result {
        Ok(value) => value,
        Err(error) => panic!("{message}: {error}"),
    }
}

pub(super) trait ResultExt<T> {
    fn or_panic(self, context: &str) -> T;
}

impl<T, E: std::fmt::Display> ResultExt<T> for Result<T, E> {
    #[inline]
    fn or_panic(self, context: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) if context.is_empty() => panic!("{error}"),
            Err(error) => panic!("{context}: {error}"),
        }
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

pub struct CuBuffer {
    pub(crate) ptr: CUdeviceptr,
    size: usize,       // requested size
    alloc_size: usize, // actual allocated size (size_class rounded)
    pooled: bool,      // true = return to pool on Drop; false = direct free
}

impl CuBuffer {
    #[inline]
    fn empty(size_bytes: usize) -> Option<Self> {
        (size_bytes == 0).then(|| Self {
            ptr: 0,
            size: 0,
            alloc_size: 0,
            pooled: false,
        })
    }

    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, size_bytes: usize) -> Self {
        Self {
            ptr,
            size: size_bytes,
            alloc_size: size_bytes,
            pooled: false,
        }
    }

    pub unsafe fn borrow_ptr(ptr: CUdeviceptr, size_bytes: usize) -> Self {
        Self {
            ptr,
            size: size_bytes,
            alloc_size: 0,
            pooled: false,
        }
    }

    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
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
        // During CUDA Graph capture, bypass the pool entirely so every
        // malloc_async is recorded as a graph allocation node.  Pool hits
        // would reuse CPU-side cached pointers that the graph runtime knows
        // nothing about, causing CUDA_ERROR_INVALID_VALUE on replay.
        if !capturing {
            let mut pool = ctx
                .pool
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((ptr, size_class)) = pool.try_alloc(size_bytes) {
                pool.allocated_bytes += size_class;
                return Ok((ptr, size_class));
            }
        }
        let rounded = round_size(size_bytes);
        let alloc_size = if rounded < SMALL_LARGE_BOUNDARY {
            rounded.max(SMALL_ALLOC_SIZE)
        } else {
            rounded.max(LARGE_ALLOC_SIZE)
        };
        let dptr = unsafe { result::malloc_async(stream.cu_stream(), alloc_size)? };
        if !capturing {
            let mut pool = ctx
                .pool
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            pool.allocated_bytes += alloc_size;
        }
        Ok((dptr, alloc_size))
    }

    pub(super) fn alloc_zeros(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if let Some(buf) = Self::empty(size_bytes) {
            return Ok(buf);
        }
        let pooled = !super::cuda_graph_is_capturing();
        let (dptr, alloc_size) = Self::alloc_from_pool(stream, size_bytes)?;
        // SAFETY: zeroing allocated device memory.
        unsafe { result::memset_d8_async(dptr, 0, alloc_size, stream.cu_stream())? };
        Ok(Self {
            ptr: dptr,
            size: size_bytes,
            alloc_size,
            pooled,
        })
    }

    pub(super) fn alloc_async(stream: &Arc<CudaStream>, size_bytes: usize) -> CudaResult<Self> {
        if let Some(buf) = Self::empty(size_bytes) {
            return Ok(buf);
        }
        let pooled = !super::cuda_graph_is_capturing();
        let (dptr, alloc_size) = Self::alloc_from_pool(stream, size_bytes)?;
        Ok(Self {
            ptr: dptr,
            size: size_bytes,
            alloc_size,
            pooled,
        })
    }

    pub(super) fn from_host<T: Scalar>(stream: &Arc<CudaStream>, data: &[T]) -> CudaResult<Self> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self {
                ptr: 0,
                size: 0,
                alloc_size: 0,
                pooled: false,
            });
        }
        let pooled = !super::cuda_graph_is_capturing();
        let (dptr, alloc_size) = Self::alloc_from_pool(stream, bytes)?;
        // SAFETY: T is POD (Scalar: Copy + Send + Sync); uploading raw bytes to GPU.
        unsafe { result::memcpy_htod_async(dptr, data, stream.cu_stream())? };
        Ok(Self {
            ptr: dptr,
            size: bytes,
            alloc_size,
            pooled,
        })
    }

    pub(super) fn from_host_u32(stream: &Arc<CudaStream>, data: &[u32]) -> CudaResult<Self> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self {
                ptr: 0,
                size: 0,
                alloc_size: 0,
                pooled: false,
            });
        }
        let pooled = !super::cuda_graph_is_capturing();
        let (dptr, alloc_size) = Self::alloc_from_pool(stream, bytes)?;
        // SAFETY: u32 is POD; uploading raw bytes to GPU.
        unsafe { result::memcpy_htod_async(dptr, data, stream.cu_stream())? };
        Ok(Self {
            ptr: dptr,
            size: bytes,
            alloc_size,
            pooled,
        })
    }

    pub(super) fn copy_to_host<T: Scalar>(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut [T],
    ) -> CudaResult<()> {
        if super::cuda_graph_is_capturing() {
            panic!("CUDA Graph capture forbids D2H readback; call prefetch() before capture.");
        }
        let bytes = std::mem::size_of_val(out);
        if bytes > 0 {
            // SAFETY: out is properly sized and T is POD.
            unsafe { result::memcpy_dtoh_sync(out, self.ptr)? };
        }
        Ok(())
    }

    pub(super) fn copy_element<T: Scalar>(
        &self,
        stream: &Arc<CudaStream>,
        byte_offset: usize,
    ) -> CudaResult<T> {
        if super::cuda_graph_is_capturing() {
            panic!("CUDA Graph capture forbids D2H readback; avoid get() during capture.");
        }
        let mut val = T::zero();
        stream.synchronize()?;
        // SAFETY: reading sizeof::<T> bytes from device at ptr+offset into &mut val.
        unsafe {
            result::memcpy_dtoh_sync(
                std::slice::from_mut(&mut val),
                self.ptr + byte_offset as u64,
            )?;
        }
        Ok(val)
    }

    pub(super) fn from_host_nonblocking<T: Scalar>(
        compute_stream: &Arc<CudaStream>,
        copy_stream: &Arc<CudaStream>,
        data: &[T],
    ) -> CudaResult<Self> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(Self {
                ptr: 0,
                size: 0,
                alloc_size: 0,
                pooled: false,
            });
        }
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
                let mut pool = ctx
                    .pool
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.allocated_bytes = pool.allocated_bytes.saturating_sub(self.alloc_size);
                if super::cuda_graph_is_capturing() {
                    unsafe {
                        // SAFETY: ptr was allocated via cuMemAllocAsync; free_async is required.
                        // During CUDA Graph capture, returning to the pool would retain a pointer
                        // that will be freed by the captured graph, causing double-free on replay.
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
                    // During capture: use async free so it's recorded as a graph node.
                    // Outside capture: sync free is fine for non-pooled buffers.
                    if super::cuda_graph_is_capturing() {
                        let stream = get_ctx().stream.cu_stream();
                        let _ = result::free_async(self.ptr, stream);
                    } else {
                        let _ = result::free_sync(self.ptr);
                    }
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

#[derive(Clone, Copy)]
pub(super) struct SyncHostPtr(pub(super) *mut u8);
unsafe impl Send for SyncHostPtr {}
unsafe impl Sync for SyncHostPtr {}

pub(super) struct CublasLtHandle(pub(super) cublaslt_sys::cublasLtHandle_t);
// SAFETY: cublasLtHandle_t is bound to the CUDA context, used from OnceLock singleton only.
unsafe impl Send for CublasLtHandle {}
unsafe impl Sync for CublasLtHandle {}

pub(super) struct CudaCtx {
    pub(super) stream: Arc<CudaStream>,
    pub(super) copy_stream: Arc<CudaStream>,
    pub(super) kernels: Mutex<HashMap<String, KernelEntry>>,
    pub(super) pool: Mutex<CudaPool>,
    pub(super) has_wmma: bool,
    pub(super) blas: CublasHandle,
    pub(super) blas_lt: CublasLtHandle,
    pub(super) blas_lt_workspace: CUdeviceptr,
    pub(super) blas_lt_workspace_size: usize,
    pub(super) graphs: Mutex<HashMap<String, Arc<NablaCudaGraph>>>,
    pub(super) reduce_scratch: CUdeviceptr,
    pub(super) reduce_host_ptr: SyncHostPtr,
    pub(super) reduce_host_dptr: CUdeviceptr,
    pub(super) reduce_funcs: [SyncFn; 18],
    pub(super) d2h_mutex: Mutex<()>,
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

pub(super) fn query_wmma_support() -> bool {
    query_compute_capability().0 >= 7
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
        let blas_raw = expect_ok(
            unsafe { cublas_result::create_handle() },
            "cuBLAS init failed",
        );
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
        let blas_lt_raw = expect_ok(
            unsafe { cublaslt_result::create_handle() },
            "cublasLt init failed",
        );
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
        let reduce_host_ptr = unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r = cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, 8);
            assert_eq!(
                r,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "cuMemAllocHost failed"
            );
            ptr as *mut u8
        };
        let reduce_host_dptr = unsafe {
            let mut dptr: CUdeviceptr = 0;
            let r = cudarc::driver::sys::cuMemHostGetDevicePointer_v2(
                &mut dptr,
                reduce_host_ptr as *mut c_void,
                0,
            );
            assert_eq!(
                r,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "cuMemHostGetDevicePointer failed"
            );
            dptr
        };
        let cuda_ctx = CudaCtx {
            stream,
            copy_stream,
            kernels: Mutex::new(HashMap::new()),
            pool: Mutex::new(CudaPool::new()),
            has_wmma,
            blas: CublasHandle(blas_raw),
            blas_lt: CublasLtHandle(blas_lt_raw),
            blas_lt_workspace,
            blas_lt_workspace_size,
            graphs: Mutex::new(HashMap::new()),
            reduce_scratch,
            reduce_host_ptr: SyncHostPtr(reduce_host_ptr),
            reduce_host_dptr,
            reduce_funcs: [SyncFn(std::ptr::null_mut()); 18],
            d2h_mutex: Mutex::new(()),
        };
        if let Err(e) = super::compile_all_kernels(&cuda_ctx, &arch) {
            panic!("CUDA kernel compilation failed: {e}");
        }
        if has_wmma {
            if let Err(e) = super::compile_wmma_kernels(&cuda_ctx, &arch) {
                eprintln!("WMMA kernel compilation failed (falling back to tiled): {e}");
            }
        }
        let rf =
            |name: &str| expect_ok(super::get_kernel(&cuda_ctx, name), "reduce kernel missing");
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

pub fn cuda_upload_u32(data: &[u32]) -> CuBuffer {
    let ctx = get_ctx();
    CuBuffer::from_host_u32(&ctx.stream, data).or_panic("CUDA upload u32")
}
