// gpu_common.rs — Shared abstraction for CUDA and HIP GPU backends.
//
// RtcStorage<B, T> unifies CudaStorage<T> and HipStorage<T>:
//   - Row-major layout with lazy host_cache for readback.
//   - Reduction ops (sum/max/min/argmax/argmin) on cached host data.
//   - type_suffix / grid_1d helpers shared across backends.
//   - MemoryPool<P>: generic best-fit caching allocator (CUDA/HIP).
//   - fuse_kernel_source / mega_fuse_kernel_source: shared kernel codegen.

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
    {
        n.div_ceil(BLOCK_SIZE as usize) as u32
    }
}

// ── Memory pool constants & helpers ─────────────────────────────────────────

/// Round up to 512-byte alignment (PyTorch-style, much less waste than power-of-2).
pub(crate) fn round_size(size: usize) -> usize {
    const ALIGN: usize = 512;
    if size == 0 {
        return ALIGN;
    }
    (size + ALIGN - 1) & !(ALIGN - 1)
}

/// Boundary between small pool (<1MB) and large pool (≥1MB).
pub(crate) const SMALL_LARGE_BOUNDARY: usize = 1 << 20; // 1MB
/// Minimum split remainder for small pool blocks.
pub(crate) const SMALL_SPLIT_MIN: usize = 512;
/// Minimum split remainder for large pool blocks.
pub(crate) const LARGE_SPLIT_MIN: usize = 1 << 20; // 1MB
/// Over-allocate size for small allocs (batch malloc calls).
pub(crate) const SMALL_ALLOC_SIZE: usize = 2 << 20; // 2MB
/// Over-allocate size for large allocs.
pub(crate) const LARGE_ALLOC_SIZE: usize = 20 << 20; // 20MB
/// GC threshold: free cached blocks when usage exceeds this fraction.
pub(crate) const GC_THRESHOLD: f64 = 0.9;

// ── Generic GPU pointer trait ───────────────────────────────────────────────

/// Null value and byte-offset for a GPU pointer type.
pub(crate) trait GpuPtr: Copy + Send + Eq {
    fn null() -> Self;
    fn offset(self, bytes: usize) -> Self;
}

#[cfg(feature = "cuda")]
impl GpuPtr for u64 {
    fn null() -> Self {
        0
    }
    fn offset(self, bytes: usize) -> Self {
        self + bytes as u64
    }
}

#[cfg(feature = "hip")]
impl GpuPtr for *mut std::ffi::c_void {
    fn null() -> Self {
        std::ptr::null_mut()
    }
    fn offset(self, bytes: usize) -> Self {
        unsafe { self.byte_add(bytes) }
    }
}

// ── Generic memory pool ─────────────────────────────────────────────────────

/// A free block in the pool, tracked for best-fit + coalescing.
pub(crate) struct FreeBlock<P: GpuPtr> {
    pub ptr: P,
    pub size: usize,
}

/// Best-fit caching memory pool with block splitting and coalescing.
/// Mirrors PyTorch's CUDACachingAllocator design:
/// - 512B-aligned sizes (not power-of-2)
/// - Dual pools: small (<1MB) and large (≥1MB)
/// - Block splitting when remainder ≥ threshold
/// - Best-fit search (sorted by size)
/// - GC threshold to avoid OOM
pub(crate) struct MemoryPool<P: GpuPtr> {
    pub small_free: Vec<FreeBlock<P>>,
    pub large_free: Vec<FreeBlock<P>>,
    pub allocated_bytes: usize,
    pub cached_bytes: usize,
}

impl<P: GpuPtr> MemoryPool<P> {
    pub fn new() -> Self {
        Self {
            small_free: Vec::new(),
            large_free: Vec::new(),
            allocated_bytes: 0,
            cached_bytes: 0,
        }
    }

    /// Best-fit: find smallest block ≥ requested size. Returns index if found.
    pub fn best_fit(pool: &[FreeBlock<P>], size: usize) -> Option<usize> {
        let pos = pool.partition_point(|b| b.size < size);
        if pos < pool.len() { Some(pos) } else { None }
    }

    pub fn split_min(size: usize) -> usize {
        if size < SMALL_LARGE_BOUNDARY {
            SMALL_SPLIT_MIN
        } else {
            LARGE_SPLIT_MIN
        }
    }

    /// Try to allocate from pool. Splits oversized blocks.
    /// Returns (ptr, actual_alloc_size) or None.
    pub fn try_alloc(&mut self, size: usize) -> Option<(P, usize)> {
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
            let split_block = FreeBlock {
                ptr: block.ptr.offset(rounded),
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
            Some((block.ptr, block.size))
        }
    }

    /// Return a block to the pool, coalescing adjacent free blocks.
    pub fn release(&mut self, ptr: P, size: usize) {
        let (mut merged_ptr, mut merged_size) = (ptr, size);

        // Try coalescing with adjacent blocks in BOTH pools.
        for pool in [
            &mut self.small_free as &mut Vec<FreeBlock<P>>,
            &mut self.large_free,
        ] {
            let mut i = 0;
            while i < pool.len() {
                let bp = pool[i].ptr;
                let bs = pool[i].size;
                // Check if pool[i] is immediately before or after our block.
                if bp.offset(bs) == merged_ptr {
                    // pool[i] directly precedes us → extend left.
                    merged_ptr = bp;
                    merged_size += bs;
                    self.cached_bytes -= bs;
                    pool.remove(i);
                } else if merged_ptr.offset(merged_size) == bp {
                    // pool[i] directly follows us → extend right.
                    merged_size += bs;
                    self.cached_bytes -= bs;
                    pool.remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // Insert merged block into the appropriate pool, sorted by size.
        let pool = if merged_size < SMALL_LARGE_BOUNDARY {
            &mut self.small_free
        } else {
            &mut self.large_free
        };
        let pos = pool.partition_point(|b| b.size < merged_size);
        pool.insert(
            pos,
            FreeBlock {
                ptr: merged_ptr,
                size: merged_size,
            },
        );
        self.cached_bytes += merged_size;
    }

    /// GC: free cached blocks if allocated exceeds threshold.
    /// Calls `free_fn` to actually free device memory.
    pub fn maybe_gc<F: FnMut(P, usize)>(&mut self, free_fn: F) {
        let total = self.allocated_bytes + self.cached_bytes;
        if total == 0 {
            return;
        }
        let usage_ratio = self.allocated_bytes as f64 / total as f64;
        if usage_ratio > GC_THRESHOLD && self.cached_bytes > 0 {
            self.trim(0, free_fn);
        }
    }

    /// Free cached blocks until pool size ≤ target_bytes. Returns bytes freed.
    pub fn trim<F: FnMut(P, usize)>(&mut self, target_bytes: usize, mut free_fn: F) -> usize {
        let mut freed = 0usize;
        while self.cached_bytes > target_bytes {
            if let Some(block) = self.large_free.pop() {
                free_fn(block.ptr, block.size);
                self.cached_bytes -= block.size;
                freed += block.size;
            } else if let Some(block) = self.small_free.pop() {
                free_fn(block.ptr, block.size);
                self.cached_bytes -= block.size;
                freed += block.size;
            } else {
                break;
            }
        }
        freed
    }

    /// Drain all cached blocks, calling `free_fn` for each.
    pub fn drain_all<F: FnMut(P, usize)>(&mut self, mut free_fn: F) {
        for block in self.small_free.drain(..) {
            free_fn(block.ptr, block.size);
        }
        for block in self.large_free.drain(..) {
            free_fn(block.ptr, block.size);
        }
    }
}

// ── Fused kernel source generation ──────────────────────────────────────────

/// Generate a fused element-wise kernel in CUDA C.
///
/// When `use_ldg` is true (CUDA), read-only loads use `__ldg()` cache hints.
/// When false (HIP), direct loads are used instead.
pub(crate) fn fuse_kernel_source(
    gpu_expr: &str,
    n_inputs: usize,
    type_name: &str,
    kernel_name: &str,
    reg_estimate: usize,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(if is_f32 { 1536 } else { 512 });

    src.push_str(&format!("// estimated registers: {reg_estimate}\n"));

    if is_f32 {
        let scalar_expr = gpu_expr.to_string();

        src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const float* __restrict__ in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str("float* __restrict__ out, unsigned n) {\n");
        src.push_str("    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i4 * 4;\n");
        src.push_str("    if (i + 3 < n) {\n");
        for j in 0..n_inputs {
            if use_ldg {
                src.push_str(&format!(
                    "        float4 v{j} = __ldg(reinterpret_cast<const float4*>(in{j}) + i4);\n"
                ));
            } else {
                src.push_str(&format!(
                    "        float4 v{j} = reinterpret_cast<const float4*>(in{j})[i4];\n"
                ));
            }
        }
        src.push_str("        float4 r;\n");
        for comp in &["x", "y", "z", "w"] {
            let mut comp_expr = scalar_expr.clone();
            for j in (0..n_inputs).rev() {
                comp_expr = comp_expr.replace(&format!("in{j}[i]"), &format!("v{j}.{comp}"));
            }
            src.push_str(&format!("        r.{comp} = {comp_expr};\n"));
        }
        src.push_str("        reinterpret_cast<float4*>(out)[i4] = r;\n");
        src.push_str("    } else {\n");
        src.push_str("        for (unsigned j = i; j < n && j < i + 4; j++) {\n");
        let mut tail_expr = scalar_expr;
        for j in (0..n_inputs).rev() {
            if use_ldg {
                tail_expr = tail_expr.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[j])"));
            } else {
                tail_expr = tail_expr.replace(&format!("in{j}[i]"), &format!("in{j}[j]"));
            }
        }
        src.push_str(&format!("            out[j] = {tail_expr};\n"));
        src.push_str("        }\n");
        src.push_str("    }\n}\n");
    } else {
        src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const ");
            src.push_str(type_name);
            src.push_str("* __restrict__ in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str(type_name);
        src.push_str("* __restrict__ out, unsigned n) {\n");
        src.push_str("    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    if (i < n) {\n");
        if use_ldg {
            let mut ldg_expr = gpu_expr.to_string();
            for j in (0..n_inputs).rev() {
                ldg_expr = ldg_expr.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            src.push_str("        out[i] = ");
            src.push_str(&ldg_expr);
            src.push_str(";\n");
        } else {
            src.push_str("        out[i] = ");
            src.push_str(gpu_expr);
            src.push_str(";\n");
        }
        src.push_str("    }\n}\n");
    }
    src
}

/// Generate a mega-kernel that fuses multiple element-wise operations into a
/// single launch. Each op reads from its own input buffers and writes to its
/// own output buffer.
///
/// When `use_ldg` is true (CUDA), read-only loads use `__ldg()` cache hints.
/// When false (HIP), direct loads are used instead.
pub(crate) fn mega_fuse_kernel_source(
    ops: &[(String, usize)], // (gpu_expr, n_inputs)
    type_name: &str,
    kernel_name: &str,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(2048);

    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');
    let mut first = true;
    for (op_idx, (_expr, n_in)) in ops.iter().enumerate() {
        for j in 0..*n_in {
            if !first {
                src.push_str(", ");
            }
            first = false;
            src.push_str(&format!("const {type_name}* __restrict__ op{op_idx}_in{j}"));
        }
        if !first {
            src.push_str(", ");
        }
        first = false;
        src.push_str(&format!("{type_name}* __restrict__ op{op_idx}_out"));
    }
    src.push_str(", unsigned n) {\n");

    if is_f32 {
        src.push_str("    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i4 * 4;\n");
        src.push_str("    if (i + 3 < n) {\n");

        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            src.push_str(&format!("        // Op {op_idx}\n"));
            for j in 0..*n_in {
                if use_ldg {
                    src.push_str(&format!(
                        "        float4 op{op_idx}_v{j} = __ldg(reinterpret_cast<const float4*>(op{op_idx}_in{j}) + i4);\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "        float4 op{op_idx}_v{j} = reinterpret_cast<const float4*>(op{op_idx}_in{j})[i4];\n"
                    ));
                }
            }
            src.push_str(&format!("        float4 op{op_idx}_r;\n"));
            for comp in &["x", "y", "z", "w"] {
                let mut comp_expr = gpu_expr.clone();
                for j in (0..*n_in).rev() {
                    comp_expr =
                        comp_expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_v{j}.{comp}"));
                }
                src.push_str(&format!("        op{op_idx}_r.{comp} = {comp_expr};\n"));
            }
            src.push_str(&format!(
                "        reinterpret_cast<float4*>(op{op_idx}_out)[i4] = op{op_idx}_r;\n"
            ));
        }

        src.push_str("    } else {\n");
        src.push_str("        for (unsigned j = i; j < n && j < i + 4; j++) {\n");
        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let mut tail_expr = gpu_expr.clone();
            for j in (0..*n_in).rev() {
                if use_ldg {
                    tail_expr = tail_expr.replace(
                        &format!("in{j}[i]"),
                        &format!("__ldg(&op{op_idx}_in{j}[j])"),
                    );
                } else {
                    tail_expr =
                        tail_expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_in{j}[j]"));
                }
            }
            src.push_str(&format!("            op{op_idx}_out[j] = {tail_expr};\n"));
        }
        src.push_str("        }\n");
        src.push_str("    }\n}\n");
    } else {
        src.push_str("    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    if (i < n) {\n");
        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            if use_ldg {
                let mut ldg_expr = gpu_expr.clone();
                for j in (0..*n_in).rev() {
                    ldg_expr = ldg_expr.replace(
                        &format!("in{j}[i]"),
                        &format!("__ldg(&op{op_idx}_in{j}[i])"),
                    );
                }
                src.push_str(&format!("        op{op_idx}_out[i] = {ldg_expr};\n"));
            } else {
                let mut expr = gpu_expr.clone();
                for j in (0..*n_in).rev() {
                    expr = expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_in{j}[i]"));
                }
                src.push_str(&format!("        op{op_idx}_out[i] = {expr};\n"));
            }
        }
        src.push_str("    }\n}\n");
    }
    src
}

// ── RtcStorage ──────────────────────────────────────────────────────────────

/// Row-major GPU-backed matrix storage with lazy host cache.
///
/// Generic over `B` (the raw GPU buffer type) — instantiated as
/// `RtcStorage<CuBuffer, T>` for CUDA, `RtcStorage<HipBuffer, T>` for HIP.
pub struct RtcStorage<B, T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) buf: B,
    pub(crate) host_cache: Mutex<Option<Vec<T>>>,
}

impl<B, T: Scalar> RtcStorage<B, T> {
    pub(crate) fn new(nrows: usize, ncols: usize, buf: B) -> Self {
        Self {
            nrows,
            ncols,
            buf,
            host_cache: Mutex::new(None),
        }
    }

    /// Public constructor for external buffer wrapping (e.g. GpuTensor → nabla bridge).
    pub fn from_parts(nrows: usize, ncols: usize, buf: B) -> Self {
        Self {
            nrows,
            ncols,
            buf,
            host_cache: Mutex::new(None),
        }
    }

    /// Returns a reference to the raw GPU buffer.
    pub fn buffer(&self) -> &B {
        &self.buf
    }

    pub(crate) fn new_cached(nrows: usize, ncols: usize, buf: B, cache: Vec<T>) -> Self {
        Self {
            nrows,
            ncols,
            buf,
            host_cache: Mutex::new(Some(cache)),
        }
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

// Shared helper: fold-first reduction on cached host data.
fn rtc_fold_first<B, T: Scalar>(a: &RtcStorage<B, T>, f: impl Fn(T, T) -> T) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    assert!(!data.is_empty(), "reduction on empty matrix");
    let mut it = data.iter();
    let init = *it.next().unwrap();
    it.fold(init, |acc, &x| f(acc, x))
}

// Shared helper: argext on cached host data.
fn rtc_argext<B, T: Scalar>(
    a: &RtcStorage<B, T>,
    is_better: impl Fn(T, T) -> bool,
) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    let mut best = 0usize;
    for i in 1..data.len() {
        if is_better(data[i], data[best]) {
            best = i;
        }
    }
    (best / a.ncols, best % a.ncols)
}

pub(crate) fn rtc_sum_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    data.iter().fold(T::zero(), |acc, &x| acc + x)
}

pub(crate) fn rtc_fold_first_prod<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = guard.as_ref().expect("cache populated");
    data.iter().fold(T::one(), |acc, &x| acc * x)
}

pub(crate) fn rtc_max_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_fold_first(a, |acc, x| acc.reduction_max(x))
}

pub(crate) fn rtc_min_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_fold_first(a, |acc, x| acc.reduction_min(x))
}

pub(crate) fn rtc_argmax_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_argext(a, |cur, best| cur.reduction_gt(best))
}

pub(crate) fn rtc_argmin_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_argext(a, |cur, best| best.reduction_gt(cur))
}

// ── GPU Backend trait method generators ─────────────────────────────────────

/// Generate `fn name<T: Scalar>(a: &$Stor<T>) -> $Stor<T> { launch_unary(a, "name") }`
/// for each unary op name listed.
macro_rules! gpu_unary_ops {
    ($Stor:ident; $($name:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $name<T: Scalar>(a: &$Stor<T>) -> $Stor<T> { launch_unary(a, stringify!($name)) }
        )+
    };
}
pub(crate) use gpu_unary_ops;

/// Generate `fn name<T: Scalar>(a: &$Stor<T>, b: &$Stor<T>) -> $Stor<T> { launch_binary(a, b, "name") }`
/// for each binary op name listed.
macro_rules! gpu_binary_ops {
    ($Stor:ident; $($name:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $name<T: Scalar>(a: &$Stor<T>, b: &$Stor<T>) -> $Stor<T> { launch_binary(a, b, stringify!($name)) }
        )+
    };
}
pub(crate) use gpu_binary_ops;
