// scalar/half_impl.rs — half-precision types (f16, bf16) — CPU-only, f32-promoted compute.

use super::{erf_approx, MathOps, ReductionOps, Scalar};

/// Helper: generate `fn math_X(self) -> Self { <$ty>::from_f32(f32::from(self).X()) }` for half types.
macro_rules! delegate_half_math {
    ($ty:ty; $($math_fn:ident => $std_fn:ident),+ $(,)?) => {
        $(#[inline] fn $math_fn(self) -> Self { <$ty>::from_f32(f32::from(self).$std_fn()) })+
    };
}

macro_rules! impl_half_mathops {
    ($ty:ty) => {
        impl MathOps for $ty {
            delegate_half_math!($ty;
                math_exp => exp, math_ln => ln, math_log1p => ln_1p,
                math_sin => sin, math_cos => cos, math_tanh => tanh,
                math_sqrt => sqrt, math_abs => abs, math_recip => recip,
                math_ceil => ceil, math_floor => floor, math_round => round,
                math_asin => asin, math_acos => acos, math_atan => atan,
                math_sinh => sinh, math_cosh => cosh,
                math_asinh => asinh, math_acosh => acosh, math_atanh => atanh,
                math_log2 => log2, math_log10 => log10,
            );
            #[inline] #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self { <$ty>::from_f32(erf_approx(f64::from(f32::from(self))) as f32) }
            #[inline] fn math_powf(self, p: Self) -> Self { <$ty>::from_f32(f32::from(self).powf(f32::from(p))) }
            #[inline] fn math_mul(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self) * f32::from(other)) }
            #[inline] fn math_div(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self) / f32::from(other)) }
            #[inline] fn math_atan2(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self).atan2(f32::from(other))) }
        }
    };
}

impl_half_mathops!(half::f16);
impl_half_mathops!(half::bf16);

macro_rules! impl_half_reduction {
    ($ty:ty) => {
        impl ReductionOps for $ty {
            #[inline]
            fn reduction_add(self, other: Self) -> Self {
                <$ty>::from_f32(f32::from(self) + f32::from(other))
            }
            #[inline]
            fn reduction_max(self, other: Self) -> Self {
                <$ty>::from_f32(f32::from(self).max(f32::from(other)))
            }
            #[inline]
            fn reduction_min(self, other: Self) -> Self {
                <$ty>::from_f32(f32::from(self).min(f32::from(other)))
            }
            #[inline]
            fn reduction_gt(self, other: Self) -> bool {
                f32::from(self) > f32::from(other)
            }
        }
    };
}

impl_half_reduction!(half::f16);
impl_half_reduction!(half::bf16);

macro_rules! impl_half_scalar {
    ($ty:ty) => {
        impl Scalar for $ty {
            type Real = f32;
            const IS_REAL: bool = true;
            #[inline]
            fn zero() -> Self {
                <$ty>::ZERO
            }
            #[inline]
            fn one() -> Self {
                <$ty>::ONE
            }
            #[inline]
            fn conj(self) -> Self {
                self
            }
            #[inline]
            fn abs_val(self) -> Self::Real {
                f32::from(self).abs()
            }
            #[allow(clippy::cast_possible_truncation)]
            #[inline]
            fn from_f64(v: f64) -> Self {
                <$ty>::from_f32(v as f32)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                f64::from(f32::from(self))
            }
        }
    };
}

impl_half_scalar!(half::f16);
impl_half_scalar!(half::bf16);
