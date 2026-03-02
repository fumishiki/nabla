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
        pool.allocated_bytes += alloc_sz;
        Ok((ptr, alloc_sz))
    }

    fn alloc_no_zero(size_bytes: usize) -> HipResult<Self> {
        if let Some(buf) = Self::empty(size_bytes) {
            return Ok(buf);
        }
        let (ptr, alloc_size) = Self::alloc_from_pool(size_bytes)?;
        Ok(Self {
            ptr,
            size: size_bytes,
            alloc_size,
            pooled: true,
        })
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

pub type HipStorage<T> = RtcStorage<HipBuffer, T>;

// SAFETY: HipBuffer is Send+Sync (raw GPU pointer). Mutex<Option<Vec<T>>>
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

struct KernelEntry {
    func: hip::hipFunction_t,
    _module: hip::hipModule_t,
}

// SAFETY: hipFunction_t and hipModule_t are opaque handles (pointers) to
unsafe impl Send for KernelEntry {}
unsafe impl Sync for KernelEntry {}

#[derive(Clone, Copy)]
struct SyncFn(hip::hipFunction_t);
// SAFETY: hipFunction_t is an opaque handle — thread-safe to store/use.
unsafe impl Send for SyncFn {}
unsafe impl Sync for SyncFn {}

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
        if let Err(e) = super::compile_all_kernels(&hip_ctx) {
            panic!("HIP kernel compilation failed: {e}");
        }
        hip_ctx
    })
}

pub(crate) fn hip_zeros<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
    let buf = hip_or_panic(
        HipBuffer::alloc_zeros(nrows * ncols * core::mem::size_of::<T>()),
        "HIP alloc",
    );
    HipStorage::new(nrows, ncols, buf)
}

pub(crate) fn hip_empty<T: Scalar>(nrows: usize, ncols: usize) -> HipStorage<T> {
    let buf = hip_or_panic(
        HipBuffer::alloc_no_zero(nrows * ncols * core::mem::size_of::<T>()),
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
    let out_buf = hip_or_panic(
        HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()),
        "HIP alloc",
    );
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
    let out_buf = hip_or_panic(
        HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()),
        "HIP alloc",
    );
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
    let out_buf = hip_or_panic(
        HipBuffer::alloc_zeros(n * core::mem::size_of::<T>()),
        "HIP alloc",
    );
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

pub(crate) fn hip_expand<T: Scalar>(
    out: &mut HipStorage<T>,
    src: &HipStorage<T>,
    src_rows: usize,
    src_cols: usize,
) {
    let ctx = get_ctx();
    let dst_rows = out.nrows;
    let dst_cols = out.ncols;
    let n = dst_rows * dst_cols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("expand"));
    let src_rows_u32 = src_rows as u32;
    let src_cols_u32 = src_cols as u32;
    let dst_rows_u32 = dst_rows as u32;
    let dst_cols_u32 = dst_cols as u32;
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&out.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&src.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&src_rows_u32 as *const u32).cast_mut().cast(),
            (&src_cols_u32 as *const u32).cast_mut().cast(),
            (&dst_rows_u32 as *const u32).cast_mut().cast(),
            (&dst_cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    out.invalidate_cache();
}

pub(crate) fn hip_axpy_inplace<T: Scalar>(y: &mut HipStorage<T>, alpha: T, x: &HipStorage<T>) {
    let ctx = get_ctx();
    let n = y.n();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("axpy"));
    let n_u32 = n as u32;
    hip_launch(
        func,
        [grid_1d(n), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&y.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&alpha as *const T).cast_mut().cast(),
            (&x.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
        ],
    );
    y.invalidate_cache();
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
