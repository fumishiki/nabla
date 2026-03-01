// scalar — Scalar trait + Complex<T> struct + MathOps + ReductionOps + math_utils.
//
// Scalar numeric types, complex numbers, and math utilities for the
// sole numeric abstraction layer for nabla.  All four supported element types
// (f32, f64, c32, c64) implement the `Scalar` trait defined here.
//
// Design notes:
//   - `MathOps` and `ReductionOps` are `pub(crate)` — they are sealed impl
//     details that back element-wise and reduction operations in the backend.
//   - `Scalar`, `RealScalar`, `Complex<T>`, `c32`, `c64`, and `math_utils`
//     are `pub` — they form the public numeric surface of nabla.
//   - Complex types are `#[cfg(feature = "cpu")]` gated because GPU backends
//     currently support f32/f64 only.
//   - `erf_approx` lives here because it is shared by MathOps impls for all
//     four types.

#![allow(clippy::many_single_char_names)]

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// erf approximation (Abramowitz & Stegun, max error ~1.5e-7)
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun polynomial approximation for erf (max error ~1.5e-7).
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

// ---------------------------------------------------------------------------
// MathOps — element-wise math dispatch (pub(crate), sealed impl detail)
// ---------------------------------------------------------------------------

/// Element-wise math operations for all four Scalar types.
///
/// `pub(crate)` — not part of the public API; used as a supertrait bound on
/// [`Scalar`] so that backends can call these methods generically.
#[allow(missing_docs)]
pub trait MathOps: Sized + Copy {
    fn math_exp(self) -> Self;
    fn math_ln(self) -> Self;
    fn math_log1p(self) -> Self;
    fn math_sin(self) -> Self;
    fn math_cos(self) -> Self;
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

/// Helper: generate `#[inline] fn math_X(self) -> Self { self.X() }` for simple delegations.
macro_rules! delegate_math {
    ($($math_fn:ident => $std_fn:ident),+ $(,)?) => {
        $(#[inline] fn $math_fn(self) -> Self { self.$std_fn() })+
    };
}

/// Implement `MathOps` for a real float type (`f32` / `f64`).
macro_rules! impl_real_mathops {
    ($ty:ty, $erf_conv:expr) => {
        impl MathOps for $ty {
            delegate_math!(
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

// ---------------------------------------------------------------------------
// ReductionOps — reduction helpers (pub(crate), sealed impl detail)
// ---------------------------------------------------------------------------

/// Reduction helpers: sum identity and ordered comparison for all four Scalar types.
///
/// `pub(crate)` — used exclusively by `sum_all`/`max_all`/`min_all`/
/// `argmax_all`/`argmin_all` in the backend; not part of the public API.
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

/// Implement `ReductionOps` for a real type (`f32` / `f64`).
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

// ---------------------------------------------------------------------------
// Scalar trait
// ---------------------------------------------------------------------------

/// Marker + capability trait for numeric types supported by nabla.
///
/// Implemented for `f32`, `f64`, `c32`, and `c64`.  All backends are generic
/// over `T: Scalar` and dispatch element-wise operations through [`MathOps`]
/// and `ReductionOps`.
// MathOps and ReductionOps are private supertraits (sealed impl details).
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

/// Implement `Scalar` for a real float type (`f32` / `f64`).
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

// ---------------------------------------------------------------------------
// RealScalar subtrait
// ---------------------------------------------------------------------------

/// Subtrait for real-valued scalars (`f32`, `f64`).
///
/// Adds `PartialOrd` and `Into<f64>` (for use in Complex arithmetic).
pub trait RealScalar: Scalar<Real = Self> + PartialOrd + Into<f64> {}

impl RealScalar for f32 {}
impl RealScalar for f64 {}

// ---------------------------------------------------------------------------
// Sub-modules (macros above are visible to child modules by textual scoping)
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu")]
mod complex;
#[cfg(feature = "cpu")]
mod dual;
#[cfg(feature = "cpu")]
mod multi_dual;
#[cfg(feature = "cpu")]
mod half_impl;

/// Utility functions for use in einsum-generated code.
#[path = "utils.rs"]
pub mod math_utils;

// Re-exports
#[cfg(feature = "cpu")]
pub use complex::{Complex, c32, c64};
#[cfg(feature = "cpu")]
pub use dual::Dual;
#[cfg(feature = "cpu")]
pub use multi_dual::MultiDual;
