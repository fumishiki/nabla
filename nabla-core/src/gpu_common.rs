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
    n.div_ceil(BLOCK_SIZE as usize) as u32
}

// ── Static kernel name lookup ──────────────────────────────────────────────

/// Pre-computed (op, (f32_name, f64_name)) pairs for all pre-compiled kernels.
/// Avoids `format!()` allocation on every kernel dispatch.
pub(crate) const KERNEL_PAIRS: &[(&str, (&str, &str))] = &[
    // unary math
    ("neg", ("k_neg_f32", "k_neg_f64")),
    ("recip", ("k_recip_f32", "k_recip_f64")),
    ("exp", ("k_exp_f32", "k_exp_f64")),
    ("ln", ("k_ln_f32", "k_ln_f64")),
    ("log1p", ("k_log1p_f32", "k_log1p_f64")),
    ("sin", ("k_sin_f32", "k_sin_f64")),
    ("cos", ("k_cos_f32", "k_cos_f64")),
    ("tanh", ("k_tanh_f32", "k_tanh_f64")),
    ("sqrt", ("k_sqrt_f32", "k_sqrt_f64")),
    ("abs", ("k_abs_f32", "k_abs_f64")),
    ("ceil", ("k_ceil_f32", "k_ceil_f64")),
    ("floor", ("k_floor_f32", "k_floor_f64")),
    ("round", ("k_round_f32", "k_round_f64")),
    ("erf", ("k_erf_f32", "k_erf_f64")),
    ("asin", ("k_asin_f32", "k_asin_f64")),
    ("acos", ("k_acos_f32", "k_acos_f64")),
    ("atan", ("k_atan_f32", "k_atan_f64")),
    ("atan2", ("k_atan2_f32", "k_atan2_f64")),
    ("sinh", ("k_sinh_f32", "k_sinh_f64")),
    ("cosh", ("k_cosh_f32", "k_cosh_f64")),
    ("asinh", ("k_asinh_f32", "k_asinh_f64")),
    ("acosh", ("k_acosh_f32", "k_acosh_f64")),
    ("atanh", ("k_atanh_f32", "k_atanh_f64")),
    ("log2", ("k_log2_f32", "k_log2_f64")),
    ("log10", ("k_log10_f32", "k_log10_f64")),
    // activations
    ("sigmoid", ("k_sigmoid_f32", "k_sigmoid_f64")),
    ("silu", ("k_silu_f32", "k_silu_f64")),
    ("mish", ("k_mish_f32", "k_mish_f64")),
    ("leaky_relu", ("k_leaky_relu_f32", "k_leaky_relu_f64")),
    ("elu", ("k_elu_f32", "k_elu_f64")),
    ("hardswish", ("k_hardswish_f32", "k_hardswish_f64")),
    // binary
    ("add", ("k_add_f32", "k_add_f64")),
    ("sub", ("k_sub_f32", "k_sub_f64")),
    ("emul", ("k_emul_f32", "k_emul_f64")),
    ("ediv", ("k_ediv_f32", "k_ediv_f64")),
    // scalar ops
    ("scale", ("k_scale_f32", "k_scale_f64")),
    ("powf", ("k_powf_f32", "k_powf_f64")),
    ("fill", ("k_fill_f32", "k_fill_f64")),
    // matrix ops
    ("transpose", ("k_transpose_f32", "k_transpose_f64")),
    ("matmul", ("k_matmul_f32", "k_matmul_f64")),
    // reductions
    ("sum", ("k_sum_f32", "k_sum_f64")),
    ("max", ("k_max_f32", "k_max_f64")),
    ("min", ("k_min_f32", "k_min_f64")),
    ("prod_partial", ("k_prod_partial_f32", "k_prod_partial_f64")),
    // row-wise
    ("softmax", ("k_softmax_f32", "k_softmax_f64")),
    ("layer_norm", ("k_layer_norm_f32", "k_layer_norm_f64")),
    ("rms_norm", ("k_rms_norm_f32", "k_rms_norm_f64")),
    ("sum_axis1", ("k_sum_axis1_f32", "k_sum_axis1_f64")),
    ("max_axis1", ("k_max_axis1_f32", "k_max_axis1_f64")),
    ("embedding", ("k_embedding_f32", "k_embedding_f64")),
    // cumulative
    ("cumsum_axis1", ("k_cumsum_axis1_f32", "k_cumsum_axis1_f64")),
    ("cumprod_axis1", ("k_cumprod_axis1_f32", "k_cumprod_axis1_f64")),
    // pooling
    ("max_pool2d", ("k_max_pool2d_f32", "k_max_pool2d_f64")),
    ("max_pool2d_with_idx", ("k_max_pool2d_with_idx_f32", "k_max_pool2d_with_idx_f64")),
    ("avg_pool2d", ("k_avg_pool2d_f32", "k_avg_pool2d_f64")),
    ("adaptive_avg_pool2d", ("k_adaptive_avg_pool2d_f32", "k_adaptive_avg_pool2d_f64")),
    // im2col
    ("im2col", ("k_im2col_f32", "k_im2col_f64")),
    ("im1col", ("k_im1col_f32", "k_im1col_f64")),
    ("im3col", ("k_im3col_f32", "k_im3col_f64")),
    // batch norm
    ("batch_norm_stats", ("k_batch_norm_stats_f32", "k_batch_norm_stats_f64")),
    ("batch_norm_fwd", ("k_batch_norm_fwd_f32", "k_batch_norm_fwd_f64")),
    // loss / attention
    ("cross_entropy", ("k_cross_entropy_f32", "k_cross_entropy_f64")),
    ("sdpa", ("k_sdpa_f32", "k_sdpa_f64")),
    // conv
    ("conv_transpose2d", ("k_conv_transpose2d_f32", "k_conv_transpose2d_f64")),
];

/// Look up the static kernel name for a given op and scalar type.
/// Returns `None` if the op is not in the pre-compiled table (e.g. fused/mega kernels).
pub(crate) fn static_kernel_name<T: Scalar>(op: &str) -> Option<&'static str> {
    let suffix = type_suffix::<T>();
    KERNEL_PAIRS.iter().find(|(o, _)| *o == op).map(|(_, names)| {
        if suffix == "f32" { names.0 } else { names.1 }
    })
}

// ── KernelId: O(1) kernel lookup by array index ─────────────────────────────

/// Numeric ID for every pre-compiled kernel. Used as array index for O(1) lookup.
/// Each base op has _F32 and _F64 variants. Order matches KERNEL_PAIRS.
#[derive(Clone, Copy, Debug)]
#[repr(u16)]
pub(crate) enum KernelId {
    // unary math
    NegF32 = 0, NegF64,
    RecipF32, RecipF64,
    ExpF32, ExpF64,
    LnF32, LnF64,
    Log1pF32, Log1pF64,
    SinF32, SinF64,
    CosF32, CosF64,
    TanhF32, TanhF64,
    SqrtF32, SqrtF64,
    AbsF32, AbsF64,
    CeilF32, CeilF64,
    FloorF32, FloorF64,
    RoundF32, RoundF64,
    ErfF32, ErfF64,
    AsinF32, AsinF64,
    AcosF32, AcosF64,
    AtanF32, AtanF64,
    Atan2F32, Atan2F64,
    SinhF32, SinhF64,
    CoshF32, CoshF64,
    AsinhF32, AsinhF64,
    AcoshF32, AcoshF64,
    AtanhF32, AtanhF64,
    Log2F32, Log2F64,
    Log10F32, Log10F64,
    // activations
    SigmoidF32, SigmoidF64,
    SiluF32, SiluF64,
    MishF32, MishF64,
    LeakyReluF32, LeakyReluF64,
    EluF32, EluF64,
    HardswishF32, HardswishF64,
    // binary
    AddF32, AddF64,
    SubF32, SubF64,
    EmulF32, EmulF64,
    EdivF32, EdivF64,
    // scalar
    ScaleF32, ScaleF64,
    PowfF32, PowfF64,
    FillF32, FillF64,
    // matrix
    TransposeF32, TransposeF64,
    MatmulF32, MatmulF64,
    // reduction
    SumF32, SumF64,
    MaxF32, MaxF64,
    MinF32, MinF64,
    ProdPartialF32, ProdPartialF64,
    // row-wise
    SoftmaxF32, SoftmaxF64,
    LayerNormF32, LayerNormF64,
    RmsNormF32, RmsNormF64,
    SumAxis1F32, SumAxis1F64,
    MaxAxis1F32, MaxAxis1F64,
    EmbeddingF32, EmbeddingF64,
    // cumulative
    CumsumAxis1F32, CumsumAxis1F64,
    CumprodAxis1F32, CumprodAxis1F64,
    // pooling
    MaxPool2dF32, MaxPool2dF64,
    MaxPool2dWithIdxF32, MaxPool2dWithIdxF64,
    AvgPool2dF32, AvgPool2dF64,
    AdaptiveAvgPool2dF32, AdaptiveAvgPool2dF64,
    // im*col
    Im2colF32, Im2colF64,
    Im1colF32, Im1colF64,
    Im3colF32, Im3colF64,
    // batch norm
    BatchNormStatsF32, BatchNormStatsF64,
    BatchNormFwdF32, BatchNormFwdF64,
    // loss / attention
    CrossEntropyF32, CrossEntropyF64,
    SdpaF32, SdpaF64,
    // conv
    ConvTranspose2dF32, ConvTranspose2dF64,
    /// Sentinel — MUST be last. Value equals the total number of kernel IDs.
    _Count,
}

/// Total number of pre-compiled kernel slots (used to size the flat array).
pub(crate) const KERNEL_COUNT: usize = KernelId::_Count as usize;

/// (kernel_name_str, KernelId) — used at init time only to populate the flat array.
pub(crate) const KERNEL_ID_MAP: &[(&str, KernelId)] = &[
    // unary math f32
    ("k_neg_f32", KernelId::NegF32),
    ("k_neg_f64", KernelId::NegF64),
    ("k_recip_f32", KernelId::RecipF32),
    ("k_recip_f64", KernelId::RecipF64),
    ("k_exp_f32", KernelId::ExpF32),
    ("k_exp_f64", KernelId::ExpF64),
    ("k_ln_f32", KernelId::LnF32),
    ("k_ln_f64", KernelId::LnF64),
    ("k_log1p_f32", KernelId::Log1pF32),
    ("k_log1p_f64", KernelId::Log1pF64),
    ("k_sin_f32", KernelId::SinF32),
    ("k_sin_f64", KernelId::SinF64),
    ("k_cos_f32", KernelId::CosF32),
    ("k_cos_f64", KernelId::CosF64),
    ("k_tanh_f32", KernelId::TanhF32),
    ("k_tanh_f64", KernelId::TanhF64),
    ("k_sqrt_f32", KernelId::SqrtF32),
    ("k_sqrt_f64", KernelId::SqrtF64),
    ("k_abs_f32", KernelId::AbsF32),
    ("k_abs_f64", KernelId::AbsF64),
    ("k_ceil_f32", KernelId::CeilF32),
    ("k_ceil_f64", KernelId::CeilF64),
    ("k_floor_f32", KernelId::FloorF32),
    ("k_floor_f64", KernelId::FloorF64),
    ("k_round_f32", KernelId::RoundF32),
    ("k_round_f64", KernelId::RoundF64),
    ("k_erf_f32", KernelId::ErfF32),
    ("k_erf_f64", KernelId::ErfF64),
    ("k_asin_f32", KernelId::AsinF32),
    ("k_asin_f64", KernelId::AsinF64),
    ("k_acos_f32", KernelId::AcosF32),
    ("k_acos_f64", KernelId::AcosF64),
    ("k_atan_f32", KernelId::AtanF32),
    ("k_atan_f64", KernelId::AtanF64),
    ("k_atan2_f32", KernelId::Atan2F32),
    ("k_atan2_f64", KernelId::Atan2F64),
    ("k_sinh_f32", KernelId::SinhF32),
    ("k_sinh_f64", KernelId::SinhF64),
    ("k_cosh_f32", KernelId::CoshF32),
    ("k_cosh_f64", KernelId::CoshF64),
    ("k_asinh_f32", KernelId::AsinhF32),
    ("k_asinh_f64", KernelId::AsinhF64),
    ("k_acosh_f32", KernelId::AcoshF32),
    ("k_acosh_f64", KernelId::AcoshF64),
    ("k_atanh_f32", KernelId::AtanhF32),
    ("k_atanh_f64", KernelId::AtanhF64),
    ("k_log2_f32", KernelId::Log2F32),
    ("k_log2_f64", KernelId::Log2F64),
    ("k_log10_f32", KernelId::Log10F32),
    ("k_log10_f64", KernelId::Log10F64),
    // activations
    ("k_sigmoid_f32", KernelId::SigmoidF32),
    ("k_sigmoid_f64", KernelId::SigmoidF64),
    ("k_silu_f32", KernelId::SiluF32),
    ("k_silu_f64", KernelId::SiluF64),
    ("k_mish_f32", KernelId::MishF32),
    ("k_mish_f64", KernelId::MishF64),
    ("k_leaky_relu_f32", KernelId::LeakyReluF32),
    ("k_leaky_relu_f64", KernelId::LeakyReluF64),
    ("k_elu_f32", KernelId::EluF32),
    ("k_elu_f64", KernelId::EluF64),
    ("k_hardswish_f32", KernelId::HardswishF32),
    ("k_hardswish_f64", KernelId::HardswishF64),
    // binary
    ("k_add_f32", KernelId::AddF32),
    ("k_add_f64", KernelId::AddF64),
    ("k_sub_f32", KernelId::SubF32),
    ("k_sub_f64", KernelId::SubF64),
    ("k_emul_f32", KernelId::EmulF32),
    ("k_emul_f64", KernelId::EmulF64),
    ("k_ediv_f32", KernelId::EdivF32),
    ("k_ediv_f64", KernelId::EdivF64),
    // scalar
    ("k_scale_f32", KernelId::ScaleF32),
    ("k_scale_f64", KernelId::ScaleF64),
    ("k_powf_f32", KernelId::PowfF32),
    ("k_powf_f64", KernelId::PowfF64),
    ("k_fill_f32", KernelId::FillF32),
    ("k_fill_f64", KernelId::FillF64),
    // matrix
    ("k_transpose_f32", KernelId::TransposeF32),
    ("k_transpose_f64", KernelId::TransposeF64),
    ("k_matmul_f32", KernelId::MatmulF32),
    ("k_matmul_f64", KernelId::MatmulF64),
    // reduction
    ("k_sum_f32", KernelId::SumF32),
    ("k_sum_f64", KernelId::SumF64),
    ("k_max_f32", KernelId::MaxF32),
    ("k_max_f64", KernelId::MaxF64),
    ("k_min_f32", KernelId::MinF32),
    ("k_min_f64", KernelId::MinF64),
    ("k_prod_partial_f32", KernelId::ProdPartialF32),
    ("k_prod_partial_f64", KernelId::ProdPartialF64),
    // row-wise
    ("k_softmax_f32", KernelId::SoftmaxF32),
    ("k_softmax_f64", KernelId::SoftmaxF64),
    ("k_layer_norm_f32", KernelId::LayerNormF32),
    ("k_layer_norm_f64", KernelId::LayerNormF64),
    ("k_rms_norm_f32", KernelId::RmsNormF32),
    ("k_rms_norm_f64", KernelId::RmsNormF64),
    ("k_sum_axis1_f32", KernelId::SumAxis1F32),
    ("k_sum_axis1_f64", KernelId::SumAxis1F64),
    ("k_max_axis1_f32", KernelId::MaxAxis1F32),
    ("k_max_axis1_f64", KernelId::MaxAxis1F64),
    ("k_embedding_f32", KernelId::EmbeddingF32),
    ("k_embedding_f64", KernelId::EmbeddingF64),
    // cumulative
    ("k_cumsum_axis1_f32", KernelId::CumsumAxis1F32),
    ("k_cumsum_axis1_f64", KernelId::CumsumAxis1F64),
    ("k_cumprod_axis1_f32", KernelId::CumprodAxis1F32),
    ("k_cumprod_axis1_f64", KernelId::CumprodAxis1F64),
    // pooling
    ("k_max_pool2d_f32", KernelId::MaxPool2dF32),
    ("k_max_pool2d_f64", KernelId::MaxPool2dF64),
    ("k_max_pool2d_with_idx_f32", KernelId::MaxPool2dWithIdxF32),
    ("k_max_pool2d_with_idx_f64", KernelId::MaxPool2dWithIdxF64),
    ("k_avg_pool2d_f32", KernelId::AvgPool2dF32),
    ("k_avg_pool2d_f64", KernelId::AvgPool2dF64),
    ("k_adaptive_avg_pool2d_f32", KernelId::AdaptiveAvgPool2dF32),
    ("k_adaptive_avg_pool2d_f64", KernelId::AdaptiveAvgPool2dF64),
    // im*col
    ("k_im2col_f32", KernelId::Im2colF32),
    ("k_im2col_f64", KernelId::Im2colF64),
    ("k_im1col_f32", KernelId::Im1colF32),
    ("k_im1col_f64", KernelId::Im1colF64),
    ("k_im3col_f32", KernelId::Im3colF32),
    ("k_im3col_f64", KernelId::Im3colF64),
    // batch norm
    ("k_batch_norm_stats_f32", KernelId::BatchNormStatsF32),
    ("k_batch_norm_stats_f64", KernelId::BatchNormStatsF64),
    ("k_batch_norm_fwd_f32", KernelId::BatchNormFwdF32),
    ("k_batch_norm_fwd_f64", KernelId::BatchNormFwdF64),
    // loss / attention
    ("k_cross_entropy_f32", KernelId::CrossEntropyF32),
    ("k_cross_entropy_f64", KernelId::CrossEntropyF64),
    ("k_sdpa_f32", KernelId::SdpaF32),
    ("k_sdpa_f64", KernelId::SdpaF64),
    // conv
    ("k_conv_transpose2d_f32", KernelId::ConvTranspose2dF32),
    ("k_conv_transpose2d_f64", KernelId::ConvTranspose2dF64),
];

impl KernelId {
    /// Map kernel name string to `KernelId`. Used during init only.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        KERNEL_ID_MAP.iter().find(|(n, _)| *n == name).map(|(_, id)| *id)
    }
}

/// Resolve (op_name, scalar_type) to `KernelId` for the hot path.
/// Panics if the op is not in the pre-compiled table.
pub(crate) fn kernel_id<T: Scalar>(op: &str) -> KernelId {
    let full_name = static_kernel_name::<T>(op)
        .unwrap_or_else(|| panic!("unknown kernel op: {op}"));
    KernelId::from_name(full_name)
        .unwrap_or_else(|| panic!("no KernelId for: {full_name}"))
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
pub(crate) const GC_THRESHOLD: f64 = 0.97;

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
        (pos < pool.len()).then_some(pos)
    }

    pub fn split_min(size: usize) -> usize {
        if size < SMALL_LARGE_BOUNDARY {
            SMALL_SPLIT_MIN
        } else {
            LARGE_SPLIT_MIN
        }
    }

    /// Try to allocate `rounded` bytes from `pool`.
    /// `cached_bytes` is updated in-place.
    /// Returns (ptr, actual_alloc_size) or None.
    fn try_alloc_from(
        pool: &mut Vec<FreeBlock<P>>,
        rounded: usize,
        cached_bytes: &mut usize,
    ) -> Option<(P, usize)> {
        let idx = Self::best_fit(pool, rounded)?;
        let block = pool.remove(idx);
        *cached_bytes -= block.size;
        // Return the whole block without splitting.
        // Each pool block = one cuMemAllocAsync allocation.
        // Splitting would create sub-blocks whose pointers are invalid for cuMemFreeAsync.
        Some((block.ptr, block.size))
    }

    /// Try to allocate from pool. Splits oversized blocks.
    /// Searches the primary pool first; falls back to the other pool so that
    /// small requests can reuse blocks coalesced into large_free (and vice
    /// versa). This is critical during CUDA Graph capture where cuMemAlloc is
    /// forbidden.
    /// Returns (ptr, actual_alloc_size) or None.
    pub fn try_alloc(&mut self, size: usize) -> Option<(P, usize)> {
        let rounded = round_size(size);
        if rounded < SMALL_LARGE_BOUNDARY {
            // Primary: small pool. Fallback: large pool (large block serves
            // small request via splitting — remainder stays in large_free).
            if let result @ Some(_) =
                Self::try_alloc_from(&mut self.small_free, rounded, &mut self.cached_bytes)
            {
                return result;
            }
            Self::try_alloc_from(&mut self.large_free, rounded, &mut self.cached_bytes)
        } else {
            // Primary: large pool. Fallback: small pool (unlikely to satisfy a
            // large request, but included for symmetry).
            if let result @ Some(_) =
                Self::try_alloc_from(&mut self.large_free, rounded, &mut self.cached_bytes)
            {
                return result;
            }
            Self::try_alloc_from(&mut self.small_free, rounded, &mut self.cached_bytes)
        }
    }

    /// Return a block to the pool. No coalescing — each block stays as an
    /// independent allocation unit so that GC can safely `cuMemFreeAsync` it
    /// with the original pointer from `cuMemAllocAsync`.
    pub fn release(&mut self, ptr: P, size: usize) {
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
        // double2 vectorized path for f64 (2 elements per thread)
        let scalar_expr = gpu_expr.to_string();

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
        src.push_str("    unsigned i2 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i2 * 2;\n");
        src.push_str("    if (i + 1 < n) {\n");
        for j in 0..n_inputs {
            if use_ldg {
                src.push_str(&format!(
                    "        double2 v{j} = __ldg(reinterpret_cast<const double2*>(in{j}) + i2);\n"
                ));
            } else {
                src.push_str(&format!(
                    "        double2 v{j} = reinterpret_cast<const double2*>(in{j})[i2];\n"
                ));
            }
        }
        src.push_str("        double2 r;\n");
        for comp in &["x", "y"] {
            let mut comp_expr = scalar_expr.clone();
            for j in (0..n_inputs).rev() {
                comp_expr = comp_expr.replace(&format!("in{j}[i]"), &format!("v{j}.{comp}"));
            }
            src.push_str(&format!("        r.{comp} = {comp_expr};\n"));
        }
        src.push_str("        reinterpret_cast<double2*>(out)[i2] = r;\n");
        src.push_str("    } else if (i < n) {\n");
        // Scalar tail for odd element count
        let tail_expr = if use_ldg {
            let mut e = scalar_expr;
            for j in (0..n_inputs).rev() {
                e = e.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            e
        } else {
            scalar_expr
        };
        src.push_str(&format!("        out[i] = {tail_expr};\n"));
        src.push_str("    }\n}\n");
    }
    src
}

/// Generate a fused map-reduce kernel source.
///
/// For `axis=1`: one block per row, threads cooperatively reduce columns.
/// For `axis=0`: one block per column, threads cooperatively reduce rows.
///
/// `reduce_op`: 0=sum, 3=mean (caller divides by count for mean if needed,
/// but this kernel always emits a sum; the Backend wrapper divides for mean).
///
/// `use_ldg`: use `__ldg()` for read-only loads (CUDA only).
pub(crate) fn fuse_reduce_kernel_source(
    gpu_expr: &str,
    n_inputs: usize,
    type_name: &str,
    kernel_name: &str,
    axis: u8,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    let zero_init = if is_f32 { "0.0f" } else { "0.0" };
    let t = type_name;
    let mut src = String::with_capacity(2048);

    // Parameter list: const T* in0, ..., T* out, unsigned rows, unsigned cols
    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');
    for i in 0..n_inputs {
        src.push_str(&format!("const {t}* __restrict__ in{i}, "));
    }
    src.push_str(&format!("{t}* __restrict__ out, unsigned rows, unsigned cols) {{\n"));

    if axis == 1 {
        // axis=1: each block handles one row → output shape (rows, 1).
        src.push_str("    unsigned row = blockIdx.x;\n");
        src.push_str("    if (row >= rows) return;\n");
        src.push_str(&format!("    {t} acc = {zero_init};\n"));
        src.push_str("    for (unsigned col = threadIdx.x; col < cols; col += blockDim.x) {\n");
        src.push_str("        unsigned i = row * cols + col;\n");
        // Apply pointwise expression using `in0[i], in1[i], ...` placeholders.
        let load_expr = gpu_expr.to_string();
        let load_str = if use_ldg {
            let mut e = load_expr.clone();
            for j in (0..n_inputs).rev() {
                e = e.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            e
        } else {
            load_expr
        };
        src.push_str(&format!("        {t} v = {load_str};\n"));
        src.push_str("        acc += v;\n");
        src.push_str("    }\n");
    } else {
        // axis=0: each block handles one column → output shape (1, cols).
        src.push_str("    unsigned col = blockIdx.x;\n");
        src.push_str("    if (col >= cols) return;\n");
        src.push_str(&format!("    {t} acc = {zero_init};\n"));
        src.push_str("    for (unsigned row = threadIdx.x; row < rows; row += blockDim.x) {\n");
        src.push_str("        unsigned i = row * cols + col;\n");
        let load_expr = gpu_expr.to_string();
        let load_str = if use_ldg {
            let mut e = load_expr.clone();
            for j in (0..n_inputs).rev() {
                e = e.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            e
        } else {
            load_expr
        };
        src.push_str(&format!("        {t} v = {load_str};\n"));
        src.push_str("        acc += v;\n");
        src.push_str("    }\n");
    }

    // Warp shuffle reduction
    src.push_str("    // warp-level reduction\n");
    src.push_str("    for (int offset = 16; offset > 0; offset >>= 1)\n");
    src.push_str("        acc += __shfl_down_sync(0xffffffff, acc, offset);\n");

    // Block-level reduction via shared memory (handles blockDim > 32)
    src.push_str("    __shared__ ");
    src.push_str(t);
    src.push_str(" smem[32];\n");
    src.push_str("    unsigned lane = threadIdx.x & 31u;\n");
    src.push_str("    unsigned wid  = threadIdx.x >> 5u;\n");
    src.push_str("    if (lane == 0) smem[wid] = acc;\n");
    src.push_str("    __syncthreads();\n");
    src.push_str("    if (threadIdx.x < (blockDim.x >> 5u)) {\n");
    src.push_str("        acc = smem[threadIdx.x];\n");
    src.push_str("        for (int offset = 16; offset > 0; offset >>= 1)\n");
    src.push_str("            acc += __shfl_down_sync(0xffffffff, acc, offset);\n");
    src.push_str("    }\n");

    if axis == 1 {
        src.push_str("    if (threadIdx.x == 0) out[row] = acc;\n");
    } else {
        src.push_str("    if (threadIdx.x == 0) out[col] = acc;\n");
    }

    src.push_str("}\n");
    src
}

/// Generate a mega-kernel that fuses multiple element-wise operations into a
/// single launch. Each op reads from its own input buffers and writes to its
/// own output buffer.
///
/// When `use_ldg` is true (CUDA), read-only loads use `__ldg()` cache hints.
/// When false (HIP), direct loads are used instead.
/// Generate CUDA/HIP C source for a mega-fused element-wise kernel.
///
/// # DAG fusion (`uses_prev`)
///
/// When `uses_prev[k]` is `true`, op k reads the result of op k-1 from a
/// register (`op{k-1}_r`) instead of from a global-memory input buffer.
/// The macro emits `"__NABLA_PREV__"` as a sentinel in the `gpu_expr`; this
/// function replaces it with the appropriate register reference per path:
///
/// - float4 main path:  `op{k-1}_r.{comp}`
/// - double2 main path: `op{k-1}_r.{comp}`
/// - f32 scalar tail:   `op{k-1}_out[j]` (already written earlier in loop)
/// - f64 scalar tail:   `op{k-1}_out[i]`
///
/// The kernel signature for a `uses_prev` op omits the `in0` pointer; any
/// other tensor references in the expression still get their own `inN` params.
///
/// # Parameters
/// * `ops`        — per-op `(gpu_expr, n_inputs)` tuples.
/// * `uses_prev`  — one bool per op; must have the same length as `ops`.
/// * `type_name`  — `"float"` or `"double"`.
/// * `kernel_name`— CUDA/HIP function name.
/// * `use_ldg`    — emit `__ldg(...)` read-only cache hint (CUDA only).
pub(crate) fn mega_fuse_kernel_source(
    ops: &[(String, usize)], // (gpu_expr, n_inputs)
    uses_prev: &[bool],
    type_name: &str,
    kernel_name: &str,
    use_ldg: bool,
) -> String {
    debug_assert_eq!(
        ops.len(),
        uses_prev.len(),
        "mega_fuse_kernel_source: ops and uses_prev must have equal length"
    );
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(2048);

    // ── Kernel signature ──────────────────────────────────────────────────────
    // All input pointers are emitted regardless of uses_prev.
    // The `__NABLA_PREV__` sentinel in the GPU expr maps to a register reference,
    // not to any inN pointer — it is completely orthogonal to the inN[i] mapping.
    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');
    let mut first_param = true;
    for (op_idx, (_expr, n_in)) in ops.iter().enumerate() {
        for j in 0..*n_in {
            if !first_param {
                src.push_str(", ");
            }
            first_param = false;
            src.push_str(&format!("const {type_name}* __restrict__ op{op_idx}_in{j}"));
        }
        if !first_param {
            src.push_str(", ");
        }
        first_param = false;
        src.push_str(&format!("{type_name}* __restrict__ op{op_idx}_out"));
    }
    src.push_str(", unsigned n) {\n");

    if is_f32 {
        // ── float4 vectorized main path ───────────────────────────────────────
        src.push_str("    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i4 * 4;\n");
        src.push_str("    if (i + 3 < n) {\n");

        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            src.push_str(&format!("        // Op {op_idx}\n"));
            // Load all global inputs.  For uses_prev ops the expr may also reference
            // __NABLA_PREV__ (replaced below) alongside normal inN[i] references.
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
                // Replace DAG sentinel with the previous op's register component.
                if op_uses_prev {
                    let prev_reg = format!("op{}_r.{}", op_idx - 1, comp);
                    comp_expr = comp_expr.replace("__NABLA_PREV__", &prev_reg);
                }
                // Replace inN[i] placeholders with loaded float4 components.
                for j in (0..*n_in).rev() {
                    comp_expr = comp_expr
                        .replace(&format!("in{j}[i]"), &format!("op{op_idx}_v{j}.{comp}"));
                }
                src.push_str(&format!("        op{op_idx}_r.{comp} = {comp_expr};\n"));
            }
            src.push_str(&format!(
                "        reinterpret_cast<float4*>(op{op_idx}_out)[i4] = op{op_idx}_r;\n"
            ));
        }

        // ── f32 scalar tail (remainder < 4 elements) ─────────────────────────
        src.push_str("    } else {\n");
        src.push_str("        for (unsigned j = i; j < n && j < i + 4; j++) {\n");
        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            let mut tail_expr = gpu_expr.clone();
            // DAG sentinel → previous op's output element (already written in this loop iteration).
            if op_uses_prev {
                tail_expr = tail_expr
                    .replace("__NABLA_PREV__", &format!("op{}_out[j]", op_idx - 1));
            }
            for j in (0..*n_in).rev() {
                if use_ldg {
                    tail_expr = tail_expr.replace(
                        &format!("in{j}[i]"),
                        &format!("__ldg(&op{op_idx}_in{j}[j])"),
                    );
                } else {
                    tail_expr = tail_expr
                        .replace(&format!("in{j}[i]"), &format!("op{op_idx}_in{j}[j]"));
                }
            }
            src.push_str(&format!("            op{op_idx}_out[j] = {tail_expr};\n"));
        }
        src.push_str("        }\n");
        src.push_str("    }\n}\n");
    } else {
        // ── double2 vectorized main path ──────────────────────────────────────
        src.push_str("    unsigned i2 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i2 * 2;\n");
        src.push_str("    if (i + 1 < n) {\n");

        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            src.push_str(&format!("        // Op {op_idx}\n"));
            for j in 0..*n_in {
                if use_ldg {
                    src.push_str(&format!(
                        "        double2 op{op_idx}_v{j} = __ldg(reinterpret_cast<const double2*>(op{op_idx}_in{j}) + i2);\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "        double2 op{op_idx}_v{j} = reinterpret_cast<const double2*>(op{op_idx}_in{j})[i2];\n"
                    ));
                }
            }
            src.push_str(&format!("        double2 op{op_idx}_r;\n"));
            for comp in &["x", "y"] {
                let mut comp_expr = gpu_expr.clone();
                // Replace DAG sentinel with the previous op's register component.
                if op_uses_prev {
                    let prev_reg = format!("op{}_r.{}", op_idx - 1, comp);
                    comp_expr = comp_expr.replace("__NABLA_PREV__", &prev_reg);
                }
                for j in (0..*n_in).rev() {
                    comp_expr = comp_expr
                        .replace(&format!("in{j}[i]"), &format!("op{op_idx}_v{j}.{comp}"));
                }
                src.push_str(&format!("        op{op_idx}_r.{comp} = {comp_expr};\n"));
            }
            src.push_str(&format!(
                "        reinterpret_cast<double2*>(op{op_idx}_out)[i2] = op{op_idx}_r;\n"
            ));
        }

        // ── f64 scalar tail (odd element count) ──────────────────────────────
        src.push_str("    } else if (i < n) {\n");
        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            let mut tail_expr = gpu_expr.clone();
            // DAG sentinel → previous op's output element (already written).
            if op_uses_prev {
                tail_expr = tail_expr
                    .replace("__NABLA_PREV__", &format!("op{}_out[i]", op_idx - 1));
            }
            for j in (0..*n_in).rev() {
                if use_ldg {
                    tail_expr = tail_expr.replace(
                        &format!("in{j}[i]"),
                        &format!("__ldg(&op{op_idx}_in{j}[i])"),
                    );
                } else {
                    tail_expr = tail_expr
                        .replace(&format!("in{j}[i]"), &format!("op{op_idx}_in{j}[i]"));
                }
            }
            src.push_str(&format!("        op{op_idx}_out[i] = {tail_expr};\n"));
        }
        src.push_str("    }\n}\n");
    }
    src
}

/// Generate a tiled mega-kernel that fuses multiple element-wise operations
/// using shared memory to amortize global-memory bandwidth when all ops share
/// the same input buffers.
///
/// Each thread block loads one tile of every shared input into `__shared__`
/// memory, then all ops read from that tile. This avoids redundant L2/HBM
/// traffic when the same input is consumed by multiple operations.
///
/// Grid = `ceil(n / tile_size)` (standard, no atomic work-stealing).
/// Tile size: 1024 elements for f32 (256 threads × 4), 512 for f64 (256 × 2).
///
/// Only beneficial when:
/// - All ops use the same set of inputs (same `n_inputs` and same pointers).
/// - `n >= 65536` (enough work to fill shared-memory pipelines).
/// - `ops.len() >= 2` (shared memory load amortises across ≥2 consumers).
///
/// `use_ldg`: use `__ldg()` for the global-memory load phase (CUDA only).
pub(crate) fn mega_fuse_tiled_kernel_source(
    ops: &[(String, usize)], // (gpu_expr, n_inputs) — all must have equal n_inputs
    type_name: &str,
    kernel_name: &str,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    // Tile sizes chosen so smem[tile_size] fits in typical 48 KiB L1/smem.
    // f32: 256 threads × 4 elems = 1024 floats  = 4 KiB per input.
    // f64: 256 threads × 2 elems = 512  doubles = 4 KiB per input.
    let tile_size: usize = if is_f32 { 1024 } else { 512 };
    let elems_per_thread: usize = if is_f32 { 4 } else { 2 };
    let n_inputs = ops.first().map(|(_, n)| *n).unwrap_or(1);

    let mut src = String::with_capacity(4096);

    // Kernel signature: all shared inputs once, then one output per op, then n.
    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');

    // Shared inputs appear once at the front (they are the same across all ops).
    for j in 0..n_inputs {
        src.push_str(&format!("const {type_name}* __restrict__ in{j}, "));
    }
    // One output per op.
    for (op_idx, _) in ops.iter().enumerate() {
        src.push_str(&format!("{type_name}* __restrict__ op{op_idx}_out, "));
    }
    src.push_str("unsigned n) {\n");

    // Shared memory: one smem slot per input, each of tile_size elements.
    for j in 0..n_inputs {
        src.push_str(&format!(
            "    __shared__ {type_name} s_in{j}[{tile_size}];\n"
        ));
    }

    src.push_str(&format!(
        "    unsigned tile_base = blockIdx.x * {tile_size}u;\n"
    ));
    src.push_str("    unsigned tid = threadIdx.x;\n\n");

    // Cooperative load: 256 threads load tile_size elements (elems_per_thread per thread).
    src.push_str("    // Phase 1: cooperative load of shared inputs into smem\n");
    src.push_str(&format!(
        "    #pragma unroll\n    for (unsigned k = 0; k < {elems_per_thread}u; k++) {{\n"
    ));
    src.push_str(&format!(
        "        unsigned smem_idx = tid * {elems_per_thread}u + k;\n"
    ));
    src.push_str("        unsigned glob_idx = tile_base + smem_idx;\n");
    src.push_str("        if (glob_idx < n) {\n");
    for j in 0..n_inputs {
        if use_ldg {
            src.push_str(&format!(
                "            s_in{j}[smem_idx] = __ldg(&in{j}[glob_idx]);\n"
            ));
        } else {
            src.push_str(&format!(
                "            s_in{j}[smem_idx] = in{j}[glob_idx];\n"
            ));
        }
    }
    src.push_str("        }\n    }\n");
    src.push_str("    __syncthreads();\n\n");

    // Phase 2: each thread processes elems_per_thread elements from smem.
    src.push_str("    // Phase 2: apply all ops reading from smem\n");
    src.push_str(&format!(
        "    #pragma unroll\n    for (unsigned k = 0; k < {elems_per_thread}u; k++) {{\n"
    ));
    src.push_str(&format!(
        "        unsigned smem_idx = tid * {elems_per_thread}u + k;\n"
    ));
    src.push_str("        unsigned glob_idx = tile_base + smem_idx;\n");
    src.push_str("        if (glob_idx < n) {\n");

    // Emit each op: replace `in{j}[i]` placeholders with smem references.
    for (op_idx, (gpu_expr, _n_in)) in ops.iter().enumerate() {
        let mut expr = gpu_expr.clone();
        // Replace placeholders in reverse order to avoid partial matches.
        for j in (0..n_inputs).rev() {
            expr = expr.replace(&format!("in{j}[i]"), &format!("s_in{j}[smem_idx]"));
        }
        src.push_str(&format!("            op{op_idx}_out[glob_idx] = {expr};\n"));
    }

    src.push_str("        }\n    }\n}\n");
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
        with_cached_data(self, |data| data[idx])
    }
}

/// Backend-specific cache fill — implemented per backend since CUDA needs
/// stream synchronization while HIP uses direct memcpy.
pub(crate) trait EnsureCache {
    fn ensure_cache(&self);
}

// ── Reduction ops (host-side, shared) ───────────────────────────────────────

fn with_cached_data<B, T: Scalar, R>(
    a: &RtcStorage<B, T>,
    f: impl FnOnce(&[T]) -> R,
) -> R
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = match guard.as_ref() {
        Some(data) => data,
        None => panic!("cache populated"),
    };
    f(data)
}

// Shared helper: fold-first reduction on cached host data.
fn rtc_fold_first<B, T: Scalar>(a: &RtcStorage<B, T>, f: impl Fn(T, T) -> T) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| {
        let (first, rest) = match data.split_first() {
            Some((first, rest)) => (first, rest),
            None => panic!("reduction on empty matrix"),
        };
        rest.iter().fold(*first, |acc, &x| f(acc, x))
    })
}

// Shared helper: argext on cached host data.
fn rtc_argext<B, T: Scalar>(
    a: &RtcStorage<B, T>,
    is_better: impl Fn(T, T) -> bool,
) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| {
        let mut best = 0usize;
        for i in 1..data.len() {
            if is_better(data[i], data[best]) {
                best = i;
            }
        }
        (best / a.ncols, best % a.ncols)
    })
}

pub(crate) fn rtc_sum_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| data.iter().fold(T::zero(), |acc, &x| acc + x))
}

pub(crate) fn rtc_fold_first_prod<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| data.iter().fold(T::one(), |acc, &x| acc * x))
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

/// Generate all trivially-delegating `Backend` trait methods shared by CUDA and HIP.
///
/// Methods already handled by `gpu_unary_ops!` / `gpu_binary_ops!` and
/// backend-specific methods (`zeros`, `from_vec`, `sync`, `matmul_into`,
/// `matmul_epilogue`, `bmm`, `addmm`, `baddbmm`, `fuse_*`) are NOT included.
macro_rules! rtc_backend_impl {
    (
        $Stor:ident;
        fill = $fill:ident,
        from_fn = $from_fn:ident,
        from_vec_async = $fva:ident,
        get = $get:ident,
        set = $set:ident,
        transpose = $transpose:ident,
        scale = $scale:ident,
        clone_storage = $clone:ident,
        powf = $powf:ident,
        sum_all = $sum_all:ident,
        max_all = $max_all:ident,
        min_all = $min_all:ident,
        argmax_all = $argmax:ident,
        argmin_all = $argmin:ident,
        softmax = $softmax:ident,
        layer_norm = $ln:ident,
        rms_norm = $rms:ident,
        batch_norm_train = $bn:ident,
        cross_entropy_fused = $ce:ident,
        sdpa = $sdpa:ident,
        axis_reduce = $ar:ident,
        embedding = $emb:ident,
        cumsum_cumprod = $csc:ident,
        prod_all = $pa:ident,
        max_pool2d = $mp2:ident,
        max_pool2d_with_idx = $mpi2:ident,
        avg_pool2d = $ap2:ident,
        adaptive_avg_pool2d = $aap2:ident,
        conv2d = $c2:ident,
        conv1d = $c1:ident,
        conv3d = $c3:ident,
        conv_transpose2d = $ct2:ident $(,)?
    ) => {
        #[inline]
        fn fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> $Stor<T> {
            $fill(nrows, ncols, val)
        }

        #[inline]
        fn identity<T: Scalar>(n: usize) -> $Stor<T> {
            $from_fn(n, n, |r, c| if r == c { T::one() } else { T::zero() })
        }

        #[inline]
        fn from_fn<T: Scalar>(
            nrows: usize,
            ncols: usize,
            f: impl FnMut(usize, usize) -> T,
        ) -> $Stor<T> {
            $from_fn(nrows, ncols, f)
        }

        #[inline]
        fn from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> $Stor<T> {
            $fva(nrows, ncols, data)
        }

        #[inline]
        fn nrows<T: Scalar>(s: &$Stor<T>) -> usize { s.nrows }

        #[inline]
        fn ncols<T: Scalar>(s: &$Stor<T>) -> usize { s.ncols }

        #[inline]
        fn get<T: Scalar>(s: &$Stor<T>, r: usize, c: usize) -> T {
            $get(s, r, c)
        }

        #[inline]
        fn set<T: Scalar>(s: &mut $Stor<T>, r: usize, c: usize, v: T) {
            $set(s, r, c, v)
        }

        #[inline]
        fn neg<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            launch_unary(a, "neg")
        }

        #[inline]
        fn transpose<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $transpose(a)
        }

        #[inline]
        fn scale<T: Scalar>(a: &$Stor<T>, s: T) -> $Stor<T> {
            $scale(a, s)
        }

        #[inline]
        fn clone_storage<T: Scalar>(s: &$Stor<T>) -> $Stor<T> {
            $clone(s)
        }

        #[inline]
        fn leaky_relu<T: Scalar>(a: &$Stor<T>, _negative_slope: T) -> $Stor<T> {
            launch_unary(a, "leaky_relu")
        }

        #[inline]
        fn elu<T: Scalar>(a: &$Stor<T>, _alpha: T) -> $Stor<T> {
            launch_unary(a, "elu")
        }

        #[inline]
        fn powf<T: Scalar>(a: &$Stor<T>, p: T) -> $Stor<T> {
            $powf(a, p)
        }

        #[inline]
        fn sum_all<T: Scalar>(a: &$Stor<T>) -> T {
            $sum_all(a)
        }

        #[inline]
        fn max_all<T: Scalar>(a: &$Stor<T>) -> T {
            $max_all(a)
        }

        #[inline]
        fn min_all<T: Scalar>(a: &$Stor<T>) -> T {
            $min_all(a)
        }

        #[inline]
        fn argmax_all<T: Scalar>(a: &$Stor<T>) -> (usize, usize) {
            $argmax(a)
        }

        #[inline]
        fn argmin_all<T: Scalar>(a: &$Stor<T>) -> (usize, usize) {
            $argmin(a)
        }

        fn softmax<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $softmax(a)
        }

        fn layer_norm<T: Scalar>(
            a: &$Stor<T>,
            gamma: &$Stor<T>,
            beta: &$Stor<T>,
            eps: T,
        ) -> $Stor<T> {
            $ln(a, gamma, beta, eps)
        }

        fn rms_norm<T: Scalar>(a: &$Stor<T>, gamma: &$Stor<T>, eps: T) -> $Stor<T> {
            $rms(a, gamma, eps)
        }

        #[allow(clippy::too_many_arguments)]
        fn batch_norm_train<T: Scalar>(
            a: &$Stor<T>,
            gamma: &$Stor<T>,
            beta: &$Stor<T>,
            running_mean: &mut $Stor<T>,
            running_var: &mut $Stor<T>,
            eps: T,
            momentum: T,
            training: bool,
        ) -> $Stor<T> {
            $bn(a, gamma, beta, running_mean, running_var, eps, momentum, training)
        }

        fn cross_entropy_fused<T: Scalar>(
            input: &$Stor<T>,
            target: &$Stor<T>,
            _n: usize,
            _c: usize,
        ) -> $Stor<T> {
            $ce(input, target)
        }

        #[allow(clippy::too_many_arguments)]
        fn sdpa<T: Scalar>(
            q: &$Stor<T>,
            k: &$Stor<T>,
            v: &$Stor<T>,
            _mask: Option<&$Stor<T>>,
            seq_q: usize,
            seq_k: usize,
            head_dim: usize,
            batch_heads: usize,
        ) -> $Stor<T> {
            $sdpa(q, k, v, seq_q, seq_k, head_dim, batch_heads)
        }

        fn sum_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $ar(a, "sum_axis1")
        }

        fn max_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $ar(a, "max_axis1")
        }

        fn embedding<T: Scalar>(indices: &$Stor<T>, weight: &$Stor<T>) -> $Stor<T> {
            $emb(indices, weight)
        }

        fn cumsum_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $csc(a, "cumsum_axis1")
        }

        fn cumprod_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $csc(a, "cumprod_axis1")
        }

        #[inline]
        fn prod_all<T: Scalar>(a: &$Stor<T>) -> T {
            $pa(a)
        }

        #[allow(clippy::too_many_arguments)]
        fn max_pool2d<T: Scalar>(
            a: &$Stor<T>,
            h: usize, w: usize,
            kh: usize, kw: usize,
            sh: usize, sw: usize,
            ph: usize, pw: usize,
        ) -> $Stor<T> {
            $mp2(a, h, w, kh, kw, sh, sw, ph, pw)
        }

        #[allow(clippy::too_many_arguments)]
        fn max_pool2d_with_indices<T: Scalar>(
            a: &$Stor<T>,
            h: usize, w: usize,
            kh: usize, kw: usize,
            sh: usize, sw: usize,
            ph: usize, pw: usize,
        ) -> ($Stor<T>, $Stor<T>) {
            $mpi2(a, h, w, kh, kw, sh, sw, ph, pw)
        }

        #[allow(clippy::too_many_arguments)]
        fn avg_pool2d<T: Scalar>(
            a: &$Stor<T>,
            h: usize, w: usize,
            kh: usize, kw: usize,
            sh: usize, sw: usize,
            ph: usize, pw: usize,
        ) -> $Stor<T> {
            $ap2(a, h, w, kh, kw, sh, sw, ph, pw)
        }

        fn adaptive_avg_pool2d<T: Scalar>(
            a: &$Stor<T>,
            in_h: usize, in_w: usize,
            out_h: usize, out_w: usize,
        ) -> $Stor<T> {
            $aap2(a, in_h, in_w, out_h, out_w)
        }

        #[allow(clippy::too_many_arguments)]
        fn conv2d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n: usize, c_in: usize, h: usize, w: usize,
            c_out: usize, kh: usize, kw: usize,
            stride: (usize, usize),
            padding: (usize, usize),
            dilation: (usize, usize),
            groups: usize,
        ) -> $Stor<T> {
            $c2(input, weight, n, c_in, h, w, c_out, kh, kw, stride, padding, dilation, groups)
        }

        #[allow(clippy::too_many_arguments)]
        fn conv1d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n_batch: usize, c_in: usize, length: usize,
            c_out: usize, kernel_size: usize,
            stride: usize, padding: usize,
            dilation: usize, groups: usize,
        ) -> $Stor<T> {
            $c1(
                input, weight, n_batch, c_in, length, c_out, kernel_size,
                stride, padding, dilation, groups,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn conv3d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n_batch: usize, c_in: usize,
            d: usize, h: usize, w: usize,
            c_out: usize, kd: usize, kh: usize, kw: usize,
            stride: (usize, usize, usize),
            padding: (usize, usize, usize),
            dilation: (usize, usize, usize),
            groups: usize,
        ) -> $Stor<T> {
            $c3(
                input, weight, n_batch, c_in, d, h, w, c_out, kd, kh, kw,
                stride, padding, dilation, groups,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn conv_transpose2d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n_batch: usize, c_in: usize, h: usize, w: usize,
            c_out: usize, kh: usize, kw: usize,
            stride: (usize, usize),
            padding: (usize, usize),
            output_padding: (usize, usize),
        ) -> $Stor<T> {
            $ct2(
                input, weight, n_batch, c_in, h, w, c_out, kh, kw,
                stride, padding, output_padding,
            )
        }
    };
}
pub(crate) use rtc_backend_impl;
