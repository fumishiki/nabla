// gpu_common.rs — Shared abstraction for CUDA and HIP GPU backends.
//
// RtcStorage<B, T> unifies CudaStorage<T> and HipStorage<T>:
//   - Row-major layout with lazy host_cache for readback.
//   - Reduction ops (sum/max/min/argmax/argmin) on cached host data.
//   - type_suffix / grid_1d helpers shared across backends.

use std::sync::Mutex;

use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

// ── Shared helpers ──────────────────────────────────────────────────────────

pub(crate) fn lock_or_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn type_suffix<T: Scalar>() -> &'static str {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        "f32"
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        "f64"
    } else {
        panic!("GPU backend supports f32/f64 only")
    }
}

pub(crate) fn grid_1d(n: usize) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    { n.div_ceil(BLOCK_SIZE as usize) as u32 }
}

// ── RtcStorage ──────────────────────────────────────────────────────────────

/// Row-major GPU-backed matrix storage with lazy host cache.
///
/// Generic over `B` (the raw GPU buffer type) — instantiated as
/// `RtcStorage<CuBuffer, T>` for CUDA, `RtcStorage<HipBuffer, T>` for HIP.
pub(crate) struct RtcStorage<B, T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) buf: B,
    pub(crate) host_cache: Mutex<Option<Vec<T>>>,
}

impl<B, T: Scalar> RtcStorage<B, T> {
    pub(crate) fn new(nrows: usize, ncols: usize, buf: B) -> Self {
        Self { nrows, ncols, buf, host_cache: Mutex::new(None) }
    }

    pub(crate) fn new_cached(nrows: usize, ncols: usize, buf: B, cache: Vec<T>) -> Self {
        Self { nrows, ncols, buf, host_cache: Mutex::new(Some(cache)) }
    }

    pub(crate) fn n(&self) -> usize {
        self.nrows * self.ncols
    }

    pub(crate) fn invalidate_cache(&mut self) {
        *lock_or_recover(&self.host_cache) = None;
    }

    pub(crate) fn cached_get(&self, idx: usize) -> T
    where
        Self: EnsureCache,
    {
        self.ensure_cache();
        let guard = lock_or_recover(&self.host_cache);
        guard.as_ref().expect("cache populated")[idx]
    }
}

/// Backend-specific cache fill — implemented per backend since CUDA needs
/// stream synchronization while HIP uses direct memcpy.
pub(crate) trait EnsureCache {
    fn ensure_cache(&self);
}

// ── Reduction ops (host-side, shared) ───────────────────────────────────────

pub(crate) fn rtc_sum_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    data.iter().fold(T::zero(), |acc, &x| acc + x)
}

pub(crate) fn rtc_max_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    assert!(!data.is_empty(), "max_all: matrix must be non-empty");
    let mut it = data.iter();
    let init = *it.next().expect("non-empty");
    it.fold(init, |acc, &x| acc.reduction_max(x))
}

pub(crate) fn rtc_min_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    assert!(!data.is_empty(), "min_all: matrix must be non-empty");
    let mut it = data.iter();
    let init = *it.next().expect("non-empty");
    it.fold(init, |acc, &x| acc.reduction_min(x))
}

pub(crate) fn rtc_argmax_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
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

pub(crate) fn rtc_argmin_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
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
