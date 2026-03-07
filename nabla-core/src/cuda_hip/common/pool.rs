use std::sync::{Mutex, atomic::{AtomicBool, Ordering}};

use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

static POOL_DEBUG: AtomicBool = AtomicBool::new(false);
static POOL_DEBUG_CHECKED: AtomicBool = AtomicBool::new(false);

#[inline]
pub(crate) fn pool_debug_enabled() -> bool {
    if !POOL_DEBUG_CHECKED.load(Ordering::Relaxed) {
        let enabled = std::env::var("NABLA_POOL_DEBUG").is_ok_and(|v| matches!(v.as_str(), "1" | "true"));
        POOL_DEBUG.store(enabled, Ordering::Relaxed);
        POOL_DEBUG_CHECKED.store(true, Ordering::Release);
    }
    POOL_DEBUG.load(Ordering::Relaxed)
}

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
        "batch_norm_update_running",
        ("k_batch_norm_update_running_f32", "k_batch_norm_update_running_f64"),
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
    ("wht", ("k_wht_f32", "k_wht_f64")),
    ("wht_inverse", ("k_wht_inverse_f32", "k_wht_inverse_f64")),
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
    BatchNormUpdateRunningF32,
    BatchNormUpdateRunningF64,
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
    WhtF32,
    WhtF64,
    WhtInverseF32,
    WhtInverseF64,
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
    ("k_batch_norm_update_running_f32", KernelId::BatchNormUpdateRunningF32),
    ("k_batch_norm_update_running_f64", KernelId::BatchNormUpdateRunningF64),
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
    ("k_wht_f32", KernelId::WhtF32),
    ("k_wht_f64", KernelId::WhtF64),
    ("k_wht_inverse_f32", KernelId::WhtInverseF32),
    ("k_wht_inverse_f64", KernelId::WhtInverseF64),
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

/// GC triggers when allocated/(allocated+cached) exceeds this ratio.
pub(crate) const GC_THRESHOLD: f64 = 0.97;

// Power-of-2 bucket range: 512B (2^9) .. 256MB (2^28) → 20 buckets.
const MIN_BUCKET_EXP: u32 = 9; // 512B
const MAX_BUCKET_EXP: u32 = 28; // 256MB
const BUCKET_COUNT: usize = (MAX_BUCKET_EXP - MIN_BUCKET_EXP + 1) as usize; // 20

#[allow(dead_code)]
pub(crate) trait GpuPtr: Copy + Send + Eq {
    fn null() -> Self;
}

#[cfg(feature = "cuda")]
impl GpuPtr for u64 {
    fn null() -> Self {
        0
    }
}

#[cfg(feature = "hip")]
impl GpuPtr for *mut std::ffi::c_void {
    fn null() -> Self {
        std::ptr::null_mut()
    }
}

/// Round `size` up to the next power of 2, clamped to [512, 256MB].
/// Sizes > 256MB are rounded to 512B alignment via `round_size`.
#[inline]
pub(crate) fn bucket_size(size: usize) -> usize {
    let min = 1usize << MIN_BUCKET_EXP;
    let max = 1usize << MAX_BUCKET_EXP;
    if size <= min {
        return min;
    }
    if size > max {
        return round_size(size);
    }
    size.next_power_of_two()
}

/// Map a bucket size to its index. Returns `None` for oversized allocations.
#[inline]
fn bucket_index(bucket_sz: usize) -> Option<usize> {
    if !bucket_sz.is_power_of_two() {
        return None;
    }
    let exp = bucket_sz.trailing_zeros();
    if !(MIN_BUCKET_EXP..=MAX_BUCKET_EXP).contains(&exp) {
        return None;
    }
    Some((exp - MIN_BUCKET_EXP) as usize)
}

/// Default number of allocs before auto-warm triggers.
const AUTO_WARM_THRESHOLD: u64 = 2000;

pub(crate) struct MemoryPool<P: GpuPtr> {
    /// Per-bucket free lists. Index 0 = 512B, index 19 = 256MB.
    buckets: [Vec<P>; BUCKET_COUNT],
    /// Oversized blocks (> 256MB) stored as (ptr, size).
    oversized: Vec<(P, usize)>,
    pub allocated_bytes: usize,
    pub cached_bytes: usize,
    // Diagnostics
    hits: u64,
    misses: u64,
    gc_count: u64,
    // Allocation recording for eager pre-warm (first iteration)
    recording: bool,
    recorded_sizes: Vec<usize>,
    warmed: bool,
    /// Alloc counter for auto-warm: starts recording at first alloc,
    /// triggers pre-warm when count reaches threshold.
    alloc_count: u64,
    auto_warm_threshold: u64,
}

impl<P: GpuPtr> MemoryPool<P> {
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Vec::new()),
            oversized: Vec::new(),
            allocated_bytes: 0,
            cached_bytes: 0,
            hits: 0,
            misses: 0,
            gc_count: 0,
            recording: true,
            recorded_sizes: Vec::new(),
            warmed: false,
            alloc_count: 0,
            auto_warm_threshold: AUTO_WARM_THRESHOLD,
        }
    }

    /// Pop a block from the bucket matching `size`. O(1).
    /// Returns (ptr, bucket_size) or None.
    pub fn try_alloc(&mut self, size: usize) -> Option<(P, usize)> {
        self.alloc_count += 1;
        if self.recording {
            self.recorded_sizes.push(size);
        }
        let bsz = bucket_size(size);
        if let Some(idx) = bucket_index(bsz) {
            if let Some(ptr) = self.buckets[idx].pop() {
                self.cached_bytes -= bsz;
                self.hits += 1;
                return Some((ptr, bsz));
            }
            self.misses += 1;
            if pool_debug_enabled() {
                eprintln!("[nabla pool] MISS bucket={bsz} req={size} avail=0");
            }
            return None;
        }
        // Oversized: exact-match search (rare path)
        let rounded = round_size(size);
        let pos = self.oversized.iter().position(|(_, s)| *s >= rounded);
        if let Some(pos) = pos {
            let (ptr, actual) = self.oversized.swap_remove(pos);
            self.cached_bytes -= actual;
            self.hits += 1;
            Some((ptr, actual))
        } else {
            self.misses += 1;
            if pool_debug_enabled() {
                eprintln!("[nabla pool] MISS oversized req={size} rounded={rounded}");
            }
            None
        }
    }

    /// Push a block back to the appropriate bucket. O(1).
    pub fn release(&mut self, ptr: P, size: usize) {
        if let Some(idx) = bucket_index(size) {
            self.buckets[idx].push(ptr);
        } else {
            self.oversized.push((ptr, size));
        }
        self.cached_bytes += size;
    }

    /// GC: free cached blocks if allocated/(allocated+cached) exceeds threshold.
    /// Keeps 25% of cached bytes as headroom to avoid immediate re-allocation.
    pub fn maybe_gc<F: FnMut(P, usize)>(&mut self, free_fn: F) {
        let total = self.allocated_bytes + self.cached_bytes;
        if total == 0 {
            return;
        }
        let usage_ratio = self.allocated_bytes as f64 / total as f64;
        if usage_ratio > GC_THRESHOLD && self.cached_bytes > 0 {
            self.gc_count += 1;
            // Keep 25% of cached bytes as headroom instead of trimming to 0.
            // This avoids the pattern: GC frees all → next allocs all miss → re-alloc spike.
            let keep = self.cached_bytes / 4;
            if pool_debug_enabled() {
                eprintln!(
                    "[nabla pool] GC #{} ratio={:.3} cached={}KB keep={}KB",
                    self.gc_count, usage_ratio,
                    self.cached_bytes / 1024, keep / 1024,
                );
            }
            self.trim(keep, free_fn);
        }
    }

    /// Free cached blocks until cached_bytes ≤ target_bytes.
    /// Frees from largest buckets first. Returns total bytes freed.
    pub fn trim<F: FnMut(P, usize)>(&mut self, target_bytes: usize, mut free_fn: F) -> usize {
        let mut freed = 0usize;
        // Oversized first (largest)
        while self.cached_bytes > target_bytes {
            if let Some((ptr, sz)) = self.oversized.pop() {
                free_fn(ptr, sz);
                self.cached_bytes -= sz;
                freed += sz;
            } else {
                break;
            }
        }
        // Then buckets from largest to smallest
        for idx in (0..BUCKET_COUNT).rev() {
            while self.cached_bytes > target_bytes {
                if let Some(ptr) = self.buckets[idx].pop() {
                    let sz = 1usize << (idx as u32 + MIN_BUCKET_EXP);
                    free_fn(ptr, sz);
                    self.cached_bytes -= sz;
                    freed += sz;
                } else {
                    break;
                }
            }
            if self.cached_bytes <= target_bytes {
                break;
            }
        }
        freed
    }

    /// Pre-warm: for each requested size, ensure the bucket has enough blocks.
    /// Returns the number of new blocks allocated.
    pub fn pre_warm<F>(&mut self, sizes: &[usize], mut alloc_fn: F) -> usize
    where
        F: FnMut(usize) -> Option<(P, usize)>,
    {
        // Count how many blocks of each bucket size are needed
        let mut needed = [0usize; BUCKET_COUNT];
        for &s in sizes {
            let bsz = bucket_size(s);
            if let Some(idx) = bucket_index(bsz) {
                needed[idx] += 1;
            }
        }
        let mut added = 0usize;
        for (idx, &need) in needed.iter().enumerate() {
            let have = self.buckets[idx].len();
            let deficit = need.saturating_sub(have);
            let bsz = 1usize << (idx as u32 + MIN_BUCKET_EXP);
            for _ in 0..deficit {
                if let Some((ptr, actual)) = alloc_fn(bsz) {
                    self.release(ptr, actual);
                    added += 1;
                }
            }
        }
        added
    }

    /// Start recording allocation sizes for later pre-warm.
    pub fn start_recording(&mut self) {
        self.recorded_sizes.clear();
        self.recording = true;
    }

    /// Stop recording and return the captured sizes.
    /// After this call, `pre_warm` can be called with the returned sizes.
    pub fn stop_recording(&mut self) -> Vec<usize> {
        self.recording = false;
        std::mem::take(&mut self.recorded_sizes)
    }

    /// Whether the pool has been pre-warmed from recorded allocations.
    pub fn is_warmed(&self) -> bool { self.warmed }

    /// Mark the pool as warmed (called after pre_warm with recorded sizes).
    pub fn set_warmed(&mut self) { self.warmed = true; }

    /// Returns true exactly once: when alloc_count reaches the auto-warm threshold
    /// and the pool hasn't been warmed yet. Caller should stop recording and pre-warm.
    pub fn should_auto_warm(&self) -> bool {
        !self.warmed && self.recording && self.alloc_count >= self.auto_warm_threshold
    }


    /// Print diagnostic summary to stderr (NABLA_POOL_DEBUG=1).
    pub fn print_diagnostics(&self) {
        if !pool_debug_enabled() {
            return;
        }
        let total = self.hits + self.misses;
        let rate = if total > 0 { self.hits as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!(
            "[nabla pool] hits={} misses={} rate={:.1}% gc={} alloc={}MB cached={}MB buckets=[{}]",
            self.hits, self.misses, rate, self.gc_count,
            self.allocated_bytes / (1024 * 1024),
            self.cached_bytes / (1024 * 1024),
            (0..BUCKET_COUNT)
                .filter(|i| !self.buckets[*i].is_empty())
                .map(|i| format!("{}:{}", 1usize << (i as u32 + MIN_BUCKET_EXP), self.buckets[i].len()))
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    /// Drain all cached blocks, calling `free_fn` for each.
    #[allow(dead_code)]
    pub fn drain_all<F: FnMut(P, usize)>(&mut self, mut free_fn: F) {
        for (idx, bucket) in self.buckets.iter_mut().enumerate() {
            let sz = 1usize << (idx as u32 + MIN_BUCKET_EXP);
            for ptr in bucket.drain(..) {
                free_fn(ptr, sz);
            }
        }
        for (ptr, sz) in self.oversized.drain(..) {
            free_fn(ptr, sz);
        }
        self.cached_bytes = 0;
    }
}
