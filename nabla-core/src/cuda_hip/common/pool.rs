use std::sync::Mutex;

use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

pub(crate) fn lock_or_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn type_suffix<T: Scalar>() -> &'static str {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        "f32"
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        "f64"
    } else if TypeId::of::<T>() == TypeId::of::<half::f16>() {
        "f16"
    } else if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
        "bf16"
    } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E4M3>() {
        "fp8e4m3"
    } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E5M2>() {
        "fp8e5m2"
    } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp4E2M1>() {
        "fp4e2m1"
    } else {
        panic!("GPU backend: unsupported scalar type")
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn grid_1d(n: usize) -> u32 {
    n.div_ceil(BLOCK_SIZE as usize) as u32
}

#[cfg(feature = "hip")]
pub(crate) const KERNEL_PAIRS: &[(&str, (&str, &str))] = &[
    ("neg", ("k_neg_f32", "k_neg_f64")),
    ("recip", ("k_recip_f32", "k_recip_f64")),
    ("exp", ("k_exp_f32", "k_exp_f64")),
    ("ln", ("k_ln_f32", "k_ln_f64")),
    ("log1p", ("k_log1p_f32", "k_log1p_f64")),
    ("sin", ("k_sin_f32", "k_sin_f64")),
    ("cos", ("k_cos_f32", "k_cos_f64")),
    ("tan", ("k_tan_f32", "k_tan_f64")),
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
    ("sigmoid", ("k_sigmoid_f32", "k_sigmoid_f64")),
    ("silu", ("k_silu_f32", "k_silu_f64")),
    ("mish", ("k_mish_f32", "k_mish_f64")),
    ("leaky_relu", ("k_leaky_relu_f32", "k_leaky_relu_f64")),
    ("elu", ("k_elu_f32", "k_elu_f64")),
    ("hardswish", ("k_hardswish_f32", "k_hardswish_f64")),
    ("add", ("k_add_f32", "k_add_f64")),
    ("sub", ("k_sub_f32", "k_sub_f64")),
    ("emul", ("k_emul_f32", "k_emul_f64")),
    ("ediv", ("k_ediv_f32", "k_ediv_f64")),
    ("scale", ("k_scale_f32", "k_scale_f64")),
    ("powf", ("k_powf_f32", "k_powf_f64")),
    ("fill", ("k_fill_f32", "k_fill_f64")),
    ("transpose", ("k_transpose_f32", "k_transpose_f64")),
    ("matmul", ("k_matmul_f32", "k_matmul_f64")),
    ("sum", ("k_sum_f32", "k_sum_f64")),
    ("max", ("k_max_f32", "k_max_f64")),
    ("min", ("k_min_f32", "k_min_f64")),
    ("prod_partial", ("k_prod_partial_f32", "k_prod_partial_f64")),
    ("softmax", ("k_softmax_f32", "k_softmax_f64")),
    ("layer_norm", ("k_layer_norm_f32", "k_layer_norm_f64")),
    ("rms_norm", ("k_rms_norm_f32", "k_rms_norm_f64")),
    ("sum_axis1", ("k_sum_axis1_f32", "k_sum_axis1_f64")),
    ("max_axis1", ("k_max_axis1_f32", "k_max_axis1_f64")),
    ("embedding", ("k_embedding_f32", "k_embedding_f64")),
    ("cumsum_axis1", ("k_cumsum_axis1_f32", "k_cumsum_axis1_f64")),
    (
        "cumprod_axis1",
        ("k_cumprod_axis1_f32", "k_cumprod_axis1_f64"),
    ),
    ("max_pool2d", ("k_max_pool2d_f32", "k_max_pool2d_f64")),
    (
        "max_pool2d_with_idx",
        ("k_max_pool2d_with_idx_f32", "k_max_pool2d_with_idx_f64"),
    ),
    ("avg_pool2d", ("k_avg_pool2d_f32", "k_avg_pool2d_f64")),
    (
        "adaptive_avg_pool2d",
        ("k_adaptive_avg_pool2d_f32", "k_adaptive_avg_pool2d_f64"),
    ),
    ("im2col", ("k_im2col_f32", "k_im2col_f64")),
    ("im1col", ("k_im1col_f32", "k_im1col_f64")),
    ("im3col", ("k_im3col_f32", "k_im3col_f64")),
    (
        "batch_norm_stats",
        ("k_batch_norm_stats_f32", "k_batch_norm_stats_f64"),
    ),
    (
        "batch_norm_fwd",
        ("k_batch_norm_fwd_f32", "k_batch_norm_fwd_f64"),
    ),
    (
        "cross_entropy",
        ("k_cross_entropy_f32", "k_cross_entropy_f64"),
    ),
    ("sdpa", ("k_sdpa_f32", "k_sdpa_f64")),
    (
        "conv_transpose2d",
        ("k_conv_transpose2d_f32", "k_conv_transpose2d_f64"),
    ),
    ("axpy", ("k_axpy_f32", "k_axpy_f64")),
    ("expand", ("k_expand_f32", "k_expand_f64")),
    ("mse_sum_fwd", ("k_mse_sum_fwd_f32", "k_mse_sum_fwd_f64")),
    ("mse_sum_bwd", ("k_mse_sum_bwd_f32", "k_mse_sum_bwd_f64")),
    ("multi_axpy3", ("k_multi_axpy3_f32", "k_multi_axpy3_f64")),
];

#[cfg(feature = "hip")]
pub(crate) fn static_kernel_name<T: Scalar>(op: &str) -> Option<&'static str> {
    let suffix = type_suffix::<T>();
    KERNEL_PAIRS.iter().find(|(o, _)| *o == op).map(
        |(_, names)| {
            if suffix == "f32" { names.0 } else { names.1 }
        },
    )
}

#[derive(Clone, Copy, Debug)]
#[repr(u16)]
#[cfg(feature = "hip")]
pub(crate) enum KernelId {
    NegF32,
    NegF64,
    RecipF32,
    RecipF64,
    ExpF32,
    ExpF64,
    LnF32,
    LnF64,
    Log1pF32,
    Log1pF64,
    SinF32,
    SinF64,
    CosF32,
    CosF64,
    TanF32,
    TanF64,
    TanhF32,
    TanhF64,
    SqrtF32,
    SqrtF64,
    AbsF32,
    AbsF64,
    CeilF32,
    CeilF64,
    FloorF32,
    FloorF64,
    RoundF32,
    RoundF64,
    ErfF32,
    ErfF64,
    AsinF32,
    AsinF64,
    AcosF32,
    AcosF64,
    AtanF32,
    AtanF64,
    Atan2F32,
    Atan2F64,
    SinhF32,
    SinhF64,
    CoshF32,
    CoshF64,
    AsinhF32,
    AsinhF64,
    AcoshF32,
    AcoshF64,
    AtanhF32,
    AtanhF64,
    Log2F32,
    Log2F64,
    Log10F32,
    Log10F64,
    SigmoidF32,
    SigmoidF64,
    SiluF32,
    SiluF64,
    MishF32,
    MishF64,
    LeakyReluF32,
    LeakyReluF64,
    EluF32,
    EluF64,
    HardswishF32,
    HardswishF64,
    AddF32,
    AddF64,
    SubF32,
    SubF64,
    EmulF32,
    EmulF64,
    EdivF32,
    EdivF64,
    ScaleF32,
    ScaleF64,
    PowfF32,
    PowfF64,
    FillF32,
    FillF64,
    TransposeF32,
    TransposeF64,
    MatmulF32,
    MatmulF64,
    SumF32,
    SumF64,
    MaxF32,
    MaxF64,
    MinF32,
    MinF64,
    ProdPartialF32,
    ProdPartialF64,
    SoftmaxF32,
    SoftmaxF64,
    LayerNormF32,
    LayerNormF64,
    RmsNormF32,
    RmsNormF64,
    GroupNormF32,
    GroupNormF64,
    SumAxis1F32,
    SumAxis1F64,
    MaxAxis1F32,
    MaxAxis1F64,
    EmbeddingF32,
    EmbeddingF64,
    CumsumAxis1F32,
    CumsumAxis1F64,
    CumprodAxis1F32,
    CumprodAxis1F64,
    MaxPool2dF32,
    MaxPool2dF64,
    MaxPool2dWithIdxF32,
    MaxPool2dWithIdxF64,
    AvgPool2dF32,
    AvgPool2dF64,
    AdaptiveAvgPool2dF32,
    AdaptiveAvgPool2dF64,
    Im2colF32,
    Im2colF64,
    Im1colF32,
    Im1colF64,
    Im3colF32,
    Im3colF64,
    BatchNormStatsF32,
    BatchNormStatsF64,
    BatchNormFwdF32,
    BatchNormFwdF64,
    CrossEntropyF32,
    CrossEntropyF64,
    SdpaF32,
    SdpaF64,
    ConvTranspose2dF32,
    ConvTranspose2dF64,
    AxpyF32,
    AxpyF64,
    ExpandF32,
    ExpandF64,
    MseSumFwdF32,
    MseSumFwdF64,
    MseSumBwdF32,
    MseSumBwdF64,
    MultiAxpy3F32,
    MultiAxpy3F64,
    _Count,
}

#[cfg(feature = "hip")]
pub(crate) const KERNEL_COUNT: usize = KernelId::_Count as usize;

#[cfg(feature = "hip")]
pub(crate) const KERNEL_ID_MAP: &[(&str, KernelId)] = &[
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
    ("k_tan_f32", KernelId::TanF32),
    ("k_tan_f64", KernelId::TanF64),
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
    ("k_add_f32", KernelId::AddF32),
    ("k_add_f64", KernelId::AddF64),
    ("k_sub_f32", KernelId::SubF32),
    ("k_sub_f64", KernelId::SubF64),
    ("k_emul_f32", KernelId::EmulF32),
    ("k_emul_f64", KernelId::EmulF64),
    ("k_ediv_f32", KernelId::EdivF32),
    ("k_ediv_f64", KernelId::EdivF64),
    ("k_scale_f32", KernelId::ScaleF32),
    ("k_scale_f64", KernelId::ScaleF64),
    ("k_powf_f32", KernelId::PowfF32),
    ("k_powf_f64", KernelId::PowfF64),
    ("k_fill_f32", KernelId::FillF32),
    ("k_fill_f64", KernelId::FillF64),
    ("k_transpose_f32", KernelId::TransposeF32),
    ("k_transpose_f64", KernelId::TransposeF64),
    ("k_matmul_f32", KernelId::MatmulF32),
    ("k_matmul_f64", KernelId::MatmulF64),
    ("k_sum_f32", KernelId::SumF32),
    ("k_sum_f64", KernelId::SumF64),
    ("k_max_f32", KernelId::MaxF32),
    ("k_max_f64", KernelId::MaxF64),
    ("k_min_f32", KernelId::MinF32),
    ("k_min_f64", KernelId::MinF64),
    ("k_prod_partial_f32", KernelId::ProdPartialF32),
    ("k_prod_partial_f64", KernelId::ProdPartialF64),
    ("k_softmax_f32", KernelId::SoftmaxF32),
    ("k_softmax_f64", KernelId::SoftmaxF64),
    ("k_layer_norm_f32", KernelId::LayerNormF32),
    ("k_layer_norm_f64", KernelId::LayerNormF64),
    ("k_rms_norm_f32", KernelId::RmsNormF32),
    ("k_rms_norm_f64", KernelId::RmsNormF64),
    ("k_group_norm_f32", KernelId::GroupNormF32),
    ("k_group_norm_f64", KernelId::GroupNormF64),
    ("k_sum_axis1_f32", KernelId::SumAxis1F32),
    ("k_sum_axis1_f64", KernelId::SumAxis1F64),
    ("k_max_axis1_f32", KernelId::MaxAxis1F32),
    ("k_max_axis1_f64", KernelId::MaxAxis1F64),
    ("k_embedding_f32", KernelId::EmbeddingF32),
    ("k_embedding_f64", KernelId::EmbeddingF64),
    ("k_cumsum_axis1_f32", KernelId::CumsumAxis1F32),
    ("k_cumsum_axis1_f64", KernelId::CumsumAxis1F64),
    ("k_cumprod_axis1_f32", KernelId::CumprodAxis1F32),
    ("k_cumprod_axis1_f64", KernelId::CumprodAxis1F64),
    ("k_max_pool2d_f32", KernelId::MaxPool2dF32),
    ("k_max_pool2d_f64", KernelId::MaxPool2dF64),
    ("k_max_pool2d_with_idx_f32", KernelId::MaxPool2dWithIdxF32),
    ("k_max_pool2d_with_idx_f64", KernelId::MaxPool2dWithIdxF64),
    ("k_avg_pool2d_f32", KernelId::AvgPool2dF32),
    ("k_avg_pool2d_f64", KernelId::AvgPool2dF64),
    ("k_adaptive_avg_pool2d_f32", KernelId::AdaptiveAvgPool2dF32),
    ("k_adaptive_avg_pool2d_f64", KernelId::AdaptiveAvgPool2dF64),
    ("k_im2col_f32", KernelId::Im2colF32),
    ("k_im2col_f64", KernelId::Im2colF64),
    ("k_im1col_f32", KernelId::Im1colF32),
    ("k_im1col_f64", KernelId::Im1colF64),
    ("k_im3col_f32", KernelId::Im3colF32),
    ("k_im3col_f64", KernelId::Im3colF64),
    ("k_batch_norm_stats_f32", KernelId::BatchNormStatsF32),
    ("k_batch_norm_stats_f64", KernelId::BatchNormStatsF64),
    ("k_batch_norm_fwd_f32", KernelId::BatchNormFwdF32),
    ("k_batch_norm_fwd_f64", KernelId::BatchNormFwdF64),
    ("k_cross_entropy_f32", KernelId::CrossEntropyF32),
    ("k_cross_entropy_f64", KernelId::CrossEntropyF64),
    ("k_sdpa_f32", KernelId::SdpaF32),
    ("k_sdpa_f64", KernelId::SdpaF64),
    ("k_conv_transpose2d_f32", KernelId::ConvTranspose2dF32),
    ("k_conv_transpose2d_f64", KernelId::ConvTranspose2dF64),
    ("k_axpy_f32", KernelId::AxpyF32),
    ("k_axpy_f64", KernelId::AxpyF64),
    ("k_expand_f32", KernelId::ExpandF32),
    ("k_expand_f64", KernelId::ExpandF64),
    ("k_mse_sum_fwd_f32", KernelId::MseSumFwdF32),
    ("k_mse_sum_fwd_f64", KernelId::MseSumFwdF64),
    ("k_mse_sum_bwd_f32", KernelId::MseSumBwdF32),
    ("k_mse_sum_bwd_f64", KernelId::MseSumBwdF64),
    ("k_multi_axpy3_f32", KernelId::MultiAxpy3F32),
    ("k_multi_axpy3_f64", KernelId::MultiAxpy3F64),
];

#[cfg(feature = "hip")]
impl KernelId {
    /// Map kernel name string to `KernelId`. Used during init only.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        KERNEL_ID_MAP
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, id)| *id)
    }
}

#[cfg(feature = "hip")]
pub(crate) fn kernel_id<T: Scalar>(op: &str) -> KernelId {
    let full_name =
        static_kernel_name::<T>(op).unwrap_or_else(|| panic!("unknown kernel op: {op}"));
    KernelId::from_name(full_name).unwrap_or_else(|| panic!("no KernelId for: {full_name}"))
}

pub(crate) fn round_size(size: usize) -> usize {
    const ALIGN: usize = 512;
    if size == 0 {
        return ALIGN;
    }
    (size + ALIGN - 1) & !(ALIGN - 1)
}

pub(crate) const SMALL_LARGE_BOUNDARY: usize = 1 << 20; // 1MB
pub(crate) const SMALL_SPLIT_MIN: usize = 512;
pub(crate) const LARGE_SPLIT_MIN: usize = 1 << 20; // 1MB
pub(crate) const SMALL_ALLOC_SIZE: usize = 2 << 20; // 2MB
pub(crate) const LARGE_ALLOC_SIZE: usize = 20 << 20; // 20MB
pub(crate) const GC_THRESHOLD: f64 = 0.97;

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

pub(crate) struct FreeBlock<P: GpuPtr> {
    pub ptr: P,
    pub size: usize,
}

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
