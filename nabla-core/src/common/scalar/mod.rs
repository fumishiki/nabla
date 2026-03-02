#![allow(clippy::many_single_char_names)]

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

#[inline]
pub(crate) fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = 1.0 - poly * (-x * x).exp();
    result.copysign(x)
}

/// Element-wise math operations dispatched per scalar type.
#[allow(missing_docs)]
pub trait MathOps: Sized + Copy {
    fn math_exp(self) -> Self;
    fn math_ln(self) -> Self;
    fn math_log1p(self) -> Self;
    fn math_sin(self) -> Self;
    fn math_cos(self) -> Self;
    fn math_tan(self) -> Self;
    fn math_tanh(self) -> Self;
    fn math_sqrt(self) -> Self;
    fn math_abs(self) -> Self;
    fn math_recip(self) -> Self;
    fn math_erf(self) -> Self;
    fn math_ceil(self) -> Self;
    fn math_floor(self) -> Self;
    fn math_round(self) -> Self;
    fn math_powf(self, p: Self) -> Self;
    fn math_mul(self, other: Self) -> Self;
    fn math_div(self, other: Self) -> Self;
    fn math_asin(self) -> Self;
    fn math_acos(self) -> Self;
    fn math_atan(self) -> Self;
    fn math_atan2(self, other: Self) -> Self;
    fn math_sinh(self) -> Self;
    fn math_cosh(self) -> Self;
    fn math_asinh(self) -> Self;
    fn math_acosh(self) -> Self;
    fn math_atanh(self) -> Self;
    fn math_log2(self) -> Self;
    fn math_log10(self) -> Self;
}

macro_rules! delegate_math {
    ($($math_fn:ident => $std_fn:ident),+ $(,)?) => {
        $(#[inline] fn $math_fn(self) -> Self { self.$std_fn() })+
    };
}

macro_rules! impl_real_mathops {
    ($ty:ty, $erf_conv:expr) => {
        impl MathOps for $ty {
            delegate_math!(
                math_exp => exp, math_ln => ln, math_log1p => ln_1p,
                math_sin => sin, math_cos => cos, math_tan => tan, math_tanh => tanh,
                math_sqrt => sqrt, math_abs => abs, math_recip => recip,
                math_ceil => ceil, math_floor => floor, math_round => round,
                math_asin => asin, math_acos => acos, math_atan => atan,
                math_sinh => sinh, math_cosh => cosh,
                math_asinh => asinh, math_acosh => acosh, math_atanh => atanh,
                math_log2 => log2, math_log10 => log10,
            );
            #[inline] #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self { $erf_conv(self) }
            #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p) }
            #[inline] fn math_mul(self, other: Self) -> Self { self * other }
            #[inline] fn math_div(self, other: Self) -> Self { self / other }
            #[inline] fn math_atan2(self, other: Self) -> Self { self.atan2(other) }
        }
    };
}

impl_real_mathops!(f32, |x: f32| erf_approx(f64::from(x)) as f32);
impl_real_mathops!(f64, erf_approx);

pub(crate) trait ReductionOps: Sized + Copy {
    /// Accumulate `self + other` for sum reduction.
    fn reduction_add(self, other: Self) -> Self;
    /// Return the element with the larger magnitude (or value for real types).
    fn reduction_max(self, other: Self) -> Self;
    /// Return the element with the smaller magnitude (or value for real types).
    fn reduction_min(self, other: Self) -> Self;
    /// Returns `true` if `self` is strictly "greater than" `other`.
    ///
    /// For real types (`f32`/`f64`): standard `>` comparison.
    /// For complex types (`c32`/`c64`): compares magnitudes `|self|^2 > |other|^2`.
    fn reduction_gt(self, other: Self) -> bool;
}

macro_rules! impl_real_reduction {
    ($ty:ty) => {
        impl ReductionOps for $ty {
            #[inline]
            fn reduction_add(self, other: Self) -> Self {
                self + other
            }
            #[inline]
            fn reduction_max(self, other: Self) -> Self {
                self.max(other)
            }
            #[inline]
            fn reduction_min(self, other: Self) -> Self {
                self.min(other)
            }
            #[inline]
            fn reduction_gt(self, other: Self) -> bool {
                self > other
            }
        }
    };
}

impl_real_reduction!(f32);
impl_real_reduction!(f64);

/// Scalar element type for tensors (f32, f64, complex, dual numbers).
#[allow(private_bounds)]
pub trait Scalar:
    Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + fmt::Debug
    + fmt::Display
    + MathOps
    + ReductionOps
{
    /// The real (non-imaginary) component type.
    /// For real scalars, `Real = Self`. For complex scalars, `Real` is the
    /// underlying float type.
    type Real: RealScalar;

    /// `true` for real scalars (`f32`, `f64`); `false` for complex (`c32`, `c64`).
    const IS_REAL: bool;

    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Complex conjugate. No-op for real types.
    #[must_use]
    fn conj(self) -> Self;

    /// Absolute value (magnitude). Returns a real scalar.
    fn abs_val(self) -> Self::Real;

    /// Lossless-where-possible conversion from `f64`.
    fn from_f64(v: f64) -> Self;

    /// Convert to `f64` (real part only for complex types).
    fn to_f64(self) -> f64;

    // --- Backward compatibility shims ---

    /// Alias for [`zero`](Self::zero).
    #[must_use]
    #[inline]
    fn zero_impl() -> Self {
        Self::zero()
    }

    /// Alias for [`one`](Self::one).
    #[must_use]
    #[inline]
    fn one_impl() -> Self {
        Self::one()
    }
}

macro_rules! impl_real_scalar {
    ($T:ty, $zero:expr, $one:expr, $from_f64:expr, $to_f64:expr) => {
        impl Scalar for $T {
            type Real = $T;
            const IS_REAL: bool = true;
            #[inline]
            fn zero() -> Self {
                $zero
            }
            #[inline]
            fn one() -> Self {
                $one
            }
            #[inline]
            fn conj(self) -> Self {
                self
            }
            #[inline]
            fn abs_val(self) -> Self::Real {
                self.abs()
            }
            #[allow(clippy::cast_possible_truncation)]
            #[inline]
            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                $to_f64(self)
            }
        }
    };
}

impl_real_scalar!(f32, 0.0, 1.0, |v: f64| v as f32, f64::from);
impl_real_scalar!(f64, 0.0, 1.0, |v: f64| v, |x: f64| x);

/// Marker trait for real-valued scalars (f32, f64) with total ordering.
pub trait RealScalar: Scalar<Real = Self> + PartialOrd + Into<f64> {}

impl RealScalar for f32 {}
impl RealScalar for f64 {}

#[cfg(feature = "cpu")]
mod complex;
#[cfg(feature = "cpu")]
mod dual;
#[cfg(feature = "cpu")]
mod multi_dual;

#[cfg(any(feature = "cpu", feature = "cuda", feature = "hip"))]
mod half_impl {
    use super::{MathOps, ReductionOps, Scalar, erf_approx};

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
                    math_sin => sin, math_cos => cos, math_tan => tan, math_tanh => tanh,
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
}

mod lowp;

pub use lowp::{Fp4E2M1, Fp8E4M3, Fp8E5M2};

/// Free-function wrappers for common scalar operations.
pub mod math_utils {
    use super::Scalar;

    /// Additive identity for `T`.
    #[must_use]
    #[inline]
    pub fn zero<T: Scalar>() -> T {
        T::zero()
    }

    /// Element-wise addition.
    #[inline]
    pub fn add<T: Scalar>(a: &T, b: &T) -> T {
        *a + *b
    }

    /// Element-wise multiplication.
    #[inline]
    pub fn mul<T: Scalar>(a: &T, b: &T) -> T {
        *a * *b
    }

    /// Complex conjugate (no-op for real types).
    #[inline]
    pub fn conj<T: Scalar>(a: &T) -> T {
        a.conj()
    }

    /// Absolute value (magnitude for complex types), returning the real component type.
    #[inline]
    pub fn abs<T: Scalar>(a: &T) -> T::Real {
        a.abs_val()
    }

    /// Convert an `f64` literal into any Scalar type.
    #[must_use]
    #[inline]
    pub fn from_f64<T: Scalar>(v: f64) -> T {
        T::from_f64(v)
    }
}

#[cfg(feature = "cpu")]
pub use complex::{Complex, c32, c64};
#[cfg(feature = "cpu")]
pub use dual::Dual;
#[cfg(feature = "cpu")]
pub use multi_dual::MultiDual;
