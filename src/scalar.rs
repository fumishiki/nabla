// scalar.rs — Scalar trait + Complex<T> struct + MathOps + ReductionOps + math_utils.
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
    if x >= 0.0 { result } else { -result }
}

// ---------------------------------------------------------------------------
// MathOps — element-wise math dispatch (pub(crate), sealed impl detail)
// ---------------------------------------------------------------------------

/// Element-wise math operations for all four Scalar types.
///
/// `pub(crate)` — not part of the public API; used as a supertrait bound on
/// [`Scalar`] so that backends can call these methods generically.
pub(crate) trait MathOps: Sized + Copy {
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
}

/// Implement `MathOps` for a real float type (`f32` / `f64`).
macro_rules! impl_real_mathops {
    ($ty:ty, $erf_conv:expr) => {
        impl MathOps for $ty {
            #[inline]
            fn math_exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn math_ln(self) -> Self {
                self.ln()
            }
            #[inline]
            fn math_log1p(self) -> Self {
                self.ln_1p()
            }
            #[inline]
            fn math_sin(self) -> Self {
                self.sin()
            }
            #[inline]
            fn math_cos(self) -> Self {
                self.cos()
            }
            #[inline]
            fn math_tanh(self) -> Self {
                self.tanh()
            }
            #[inline]
            fn math_sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline]
            fn math_abs(self) -> Self {
                self.abs()
            }
            #[inline]
            fn math_recip(self) -> Self {
                self.recip()
            }
            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self {
                $erf_conv(self)
            }
            #[inline]
            fn math_ceil(self) -> Self {
                self.ceil()
            }
            #[inline]
            fn math_floor(self) -> Self {
                self.floor()
            }
            #[inline]
            fn math_round(self) -> Self {
                self.round()
            }
            #[inline]
            fn math_powf(self, p: Self) -> Self {
                self.powf(p)
            }
            #[inline]
            fn math_mul(self, other: Self) -> Self {
                self * other
            }
            #[inline]
            fn math_div(self, other: Self) -> Self {
                self / other
            }
        }
    };
}

impl_real_mathops!(f32, |x: f32| erf_approx(f64::from(x)) as f32);
impl_real_mathops!(f64, erf_approx);

/// Implement `MathOps` for `Complex<T>`.
#[cfg(feature = "cpu")]
macro_rules! impl_complex_mathops {
    ($ty:ty, $real:ty) => {
        impl MathOps for $ty {
            #[inline]
            fn math_exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn math_ln(self) -> Self {
                self.ln()
            }
            #[inline]
            fn math_log1p(self) -> Self {
                Complex::new(1.0 as $real + self.re, self.im).ln()
            }
            #[inline]
            fn math_sin(self) -> Self {
                self.sin()
            }
            #[inline]
            fn math_cos(self) -> Self {
                self.cos()
            }
            #[inline]
            fn math_tanh(self) -> Self {
                self.tanh()
            }
            #[inline]
            fn math_sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline]
            fn math_abs(self) -> Self {
                Complex::new(self.norm(), 0.0)
            }
            #[inline]
            fn math_recip(self) -> Self {
                self.inv()
            }
            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self {
                Complex::new(
                    erf_approx(f64::from(self.re)) as $real,
                    erf_approx(f64::from(self.im)) as $real,
                )
            }
            #[inline]
            fn math_ceil(self) -> Self {
                Complex::new(self.re.ceil(), self.im.ceil())
            }
            #[inline]
            fn math_floor(self) -> Self {
                Complex::new(self.re.floor(), self.im.floor())
            }
            #[inline]
            fn math_round(self) -> Self {
                Complex::new(self.re.round(), self.im.round())
            }
            #[inline]
            fn math_powf(self, p: Self) -> Self {
                self.powf(p.re)
            }
            #[inline]
            fn math_mul(self, other: Self) -> Self {
                self * other
            }
            #[inline]
            fn math_div(self, other: Self) -> Self {
                self / other
            }
        }
    };
}

#[cfg(feature = "cpu")]
impl_complex_mathops!(c32, f32);
#[cfg(feature = "cpu")]
impl_complex_mathops!(c64, f64);

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
    /// For complex types (`c32`/`c64`): compares magnitudes `|self|² > |other|²`.
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

/// Magnitude-squared for complex comparison: `re² + im²`.
macro_rules! mag2 {
    ($z:expr) => {
        $z.re * $z.re + $z.im * $z.im
    };
}

/// Implement `ReductionOps` for `Complex<T>`.
#[cfg(feature = "cpu")]
macro_rules! impl_complex_reduction {
    ($ty:ty) => {
        impl ReductionOps for $ty {
            #[inline]
            fn reduction_add(self, other: Self) -> Self {
                self + other
            }
            #[inline]
            fn reduction_max(self, other: Self) -> Self {
                if mag2!(self) >= mag2!(other) {
                    self
                } else {
                    other
                }
            }
            #[inline]
            fn reduction_min(self, other: Self) -> Self {
                if mag2!(self) <= mag2!(other) {
                    self
                } else {
                    other
                }
            }
            #[inline]
            fn reduction_gt(self, other: Self) -> bool {
                mag2!(self) > mag2!(other)
            }
        }
    };
}

#[cfg(feature = "cpu")]
impl_complex_reduction!(c32);
#[cfg(feature = "cpu")]
impl_complex_reduction!(c64);

// ---------------------------------------------------------------------------
// Complex<T> struct
// ---------------------------------------------------------------------------

/// A generic complex number with real part `re` and imaginary part `im`.
///
/// `c32 = Complex<f32>` and `c64 = Complex<f64>` are the two concrete types
/// exposed by nabla. Complex types are CPU-only (GPU backends support `f32`/`f64`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg(feature = "cpu")]
pub struct Complex<T: RealScalar> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

#[cfg(feature = "cpu")]
#[allow(non_camel_case_types)]
/// 32-bit complex number (`Complex<f32>`).
pub type c32 = Complex<f32>;
#[cfg(feature = "cpu")]
#[allow(non_camel_case_types)]
/// 64-bit complex number (`Complex<f64>`).
pub type c64 = Complex<f64>;

#[cfg(feature = "cpu")]
impl<T: RealScalar> Complex<T> {
    /// Construct a new complex number from real and imaginary parts.
    #[inline]
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Magnitude (absolute value): `sqrt(re² + im²)`.
    #[inline]
    pub fn norm(self) -> T {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        T::from_f64((r * r + i * i).sqrt())
    }

    /// Complex conjugate: `re - im·i`.
    #[must_use]
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, T::from_f64(-self.im.into()))
    }

    /// Multiplicative inverse: `1 / z`.
    #[must_use]
    #[inline]
    pub fn inv(self) -> Self {
        let denom = self.re.into() * self.re.into() + self.im.into() * self.im.into();
        Self::new(
            T::from_f64(self.re.into() / denom),
            T::from_f64(-self.im.into() / denom),
        )
    }

    /// Complex exponential: `e^(re) * (cos(im) + i·sin(im))`.
    #[must_use]
    #[inline]
    pub fn exp(self) -> Self {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        let scale = r.exp();
        Self::new(T::from_f64(scale * i.cos()), T::from_f64(scale * i.sin()))
    }

    /// Complex natural logarithm: `ln|z| + i·arg(z)`.
    #[must_use]
    #[inline]
    pub fn ln(self) -> Self {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        Self::new(
            T::from_f64((r * r + i * i).sqrt().ln()),
            T::from_f64(i.atan2(r)),
        )
    }

    /// Complex sine: `sin(re)·cosh(im) + i·cos(re)·sinh(im)`.
    #[must_use]
    #[inline]
    pub fn sin(self) -> Self {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        Self::new(
            T::from_f64(r.sin() * i.cosh()),
            T::from_f64(r.cos() * i.sinh()),
        )
    }

    /// Complex cosine: `cos(re)·cosh(im) - i·sin(re)·sinh(im)`.
    #[must_use]
    #[inline]
    pub fn cos(self) -> Self {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        Self::new(
            T::from_f64(r.cos() * i.cosh()),
            T::from_f64(-r.sin() * i.sinh()),
        )
    }

    /// Complex hyperbolic tangent.
    #[must_use]
    #[inline]
    pub fn tanh(self) -> Self {
        // tanh(z) = (e^z - e^-z) / (e^z + e^-z)
        let e_pos = self.exp();
        let e_neg = Self::new(T::from_f64(-self.re.into()), T::from_f64(-self.im.into())).exp();
        let num = e_pos - e_neg;
        let den = e_pos + e_neg;
        num / den
    }

    /// Complex square root (principal branch).
    #[must_use]
    #[inline]
    pub fn sqrt(self) -> Self {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        let modulus = (r * r + i * i).sqrt();
        let sqrt_mod = modulus.sqrt();
        if sqrt_mod == 0.0 {
            return Self::new(T::from_f64(0.0), T::from_f64(0.0));
        }
        let sign_im: f64 = if i < 0.0 { -1.0 } else { 1.0 };
        let re_part = f64::midpoint(modulus, r).sqrt();
        let im_part = sign_im * ((modulus - r) / 2.0).sqrt();
        Self::new(T::from_f64(re_part), T::from_f64(im_part))
    }

    /// Complex power: `z^p` using principal value `exp(p·ln(z))`.
    ///
    /// Takes the real exponent `p` only (matches Julia `z^p` semantics).
    #[must_use]
    #[inline]
    pub fn powf(self, p: T) -> Self {
        // z^p = exp(p * ln(z))
        let ln_z = self.ln();
        let p_f64: f64 = p.into();
        let scaled = Self::new(
            T::from_f64(ln_z.re.into() * p_f64),
            T::from_f64(ln_z.im.into() * p_f64),
        );
        scaled.exp()
    }
}

// Arithmetic ops for Complex<T>

/// Implement `Add` or `Sub` for `Complex<T>` (component-wise real arithmetic).
#[cfg(feature = "cpu")]
macro_rules! impl_complex_binop {
    ($Op:ident, $fn_name:ident, $op:tt) => {
        impl<T: RealScalar> $Op for Complex<T> {
            type Output = Self;
            #[inline]
            fn $fn_name(self, rhs: Self) -> Self {
                Self::new(
                    T::from_f64(self.re.into() $op rhs.re.into()),
                    T::from_f64(self.im.into() $op rhs.im.into()),
                )
            }
        }
    };
}

#[cfg(feature = "cpu")]
impl_complex_binop!(Add, add, +);
#[cfg(feature = "cpu")]
impl_complex_binop!(Sub, sub, -);

#[cfg(feature = "cpu")]
impl<T: RealScalar> Mul for Complex<T> {
    type Output = Self;
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a: f64 = self.re.into();
        let b: f64 = self.im.into();
        let c: f64 = rhs.re.into();
        let d: f64 = rhs.im.into();
        Self::new(T::from_f64(a * c - b * d), T::from_f64(a * d + b * c))
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Div for Complex<T> {
    type Output = Self;
    // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let a: f64 = self.re.into();
        let b: f64 = self.im.into();
        let c: f64 = rhs.re.into();
        let d: f64 = rhs.im.into();
        let denom = c * c + d * d;
        Self::new(
            T::from_f64((a * c + b * d) / denom),
            T::from_f64((b * c - a * d) / denom),
        )
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(T::from_f64(-self.re.into()), T::from_f64(-self.im.into()))
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}i", self.re, self.im)
    }
}

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
// Scalar trait
// ---------------------------------------------------------------------------

/// Marker + capability trait for numeric types supported by nabla.
///
/// Implemented for `f32`, `f64`, `c32`, and `c64`.  All backends are generic
/// over `T: Scalar` and dispatch element-wise operations through [`MathOps`]
/// and [`ReductionOps`].
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

    /// Alias for [`zero`](Self::zero). Allows existing code using
    /// `T::zero_impl()` (legacy convention) to compile unchanged.
    #[must_use]
    #[inline]
    fn zero_impl() -> Self {
        Self::zero()
    }

    /// Alias for [`one`](Self::one). Allows existing code using
    /// `T::one_impl()` (legacy convention) to compile unchanged.
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

/// Implement `Scalar` for a complex type (`c32` / `c64`).
#[cfg(feature = "cpu")]
macro_rules! impl_complex_scalar {
    ($T:ty, $F:ty, $from_f64:expr, $to_f64:expr) => {
        impl Scalar for $T {
            type Real = $F;
            const IS_REAL: bool = false;
            #[inline]
            fn zero() -> Self {
                Complex::new(0.0, 0.0)
            }
            #[inline]
            fn one() -> Self {
                Complex::new(1.0, 0.0)
            }
            #[inline]
            fn conj(self) -> Self {
                Complex::new(self.re, -self.im)
            }
            #[inline]
            fn abs_val(self) -> Self::Real {
                self.norm()
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

#[cfg(feature = "cpu")]
impl_complex_scalar!(c32, f32, |v: f64| Complex::new(v as f32, 0.0), |z: c32| {
    f64::from(z.re)
});
#[cfg(feature = "cpu")]
impl_complex_scalar!(c64, f64, |v: f64| Complex::new(v, 0.0), |z: c64| z.re);

// ---------------------------------------------------------------------------
// math_utils — public helpers used by einsum codegen
// ---------------------------------------------------------------------------

/// Utility functions for use in einsum-generated code.
///
/// These are thin wrappers over [`Scalar`] trait methods and are kept `pub`
/// so that code generated by the `einsum!` macro can call them without
/// importing the trait itself.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_scalar_basics() {
        assert_eq!(f32::zero(), 0.0_f32);
        assert_eq!(f32::one(), 1.0_f32);
        assert_eq!(f64::zero(), 0.0_f64);
        assert_eq!(f64::one(), 1.0_f64);
        assert!(<f32 as Scalar>::IS_REAL);
        assert!(<f64 as Scalar>::IS_REAL);
    }

    #[test]
    fn real_mathops() {
        let x: f64 = 1.0;
        assert!((x.math_exp() - std::f64::consts::E).abs() < 1e-10);
        assert!((x.math_ln() - 0.0).abs() < 1e-10);
        assert!((x.math_sqrt() - 1.0).abs() < 1e-10);
        assert!((x.math_erf() - erf_approx(1.0)).abs() < 1e-14);
    }

    #[test]
    fn real_reduction() {
        let a: f64 = 3.0;
        let b: f64 = 5.0;
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(a.reduction_add(b), 8.0);
        assert_eq!(a.reduction_max(b), 5.0);
        assert_eq!(a.reduction_min(b), 3.0);
        assert!(b.reduction_gt(a));
    }

    #[test]
    fn zero_one_impl_compat() {
        // Backward compat shims must equal the canonical methods.
        assert_eq!(f64::zero_impl(), f64::zero());
        assert_eq!(f64::one_impl(), f64::one());
    }

    #[test]
    fn math_utils_fns() {
        let a: f64 = 2.0;
        let b: f64 = 3.0;
        assert_eq!(math_utils::zero::<f64>(), 0.0);
        assert_eq!(math_utils::add(&a, &b), 5.0);
        assert_eq!(math_utils::mul(&a, &b), 6.0);
        assert_eq!(math_utils::conj(&a), 2.0);
        assert_eq!(math_utils::abs(&a), 2.0);
        assert_eq!(math_utils::from_f64::<f64>(3.14), 3.14);
    }

    #[cfg(feature = "cpu")]
    mod complex_tests {
        use super::*;

        #[test]
        fn complex_arithmetic() {
            let z1 = c64::new(1.0, 2.0);
            let z2 = c64::new(3.0, 4.0);
            let sum = z1 + z2;
            assert!((sum.re - 4.0).abs() < 1e-12);
            assert!((sum.im - 6.0).abs() < 1e-12);

            let diff = z2 - z1;
            assert!((diff.re - 2.0).abs() < 1e-12);
            assert!((diff.im - 2.0).abs() < 1e-12);

            // (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
            let prod = z1 * z2;
            assert!((prod.re - (-5.0)).abs() < 1e-12);
            assert!((prod.im - 10.0).abs() < 1e-12);
        }

        #[test]
        fn complex_scalar_trait() {
            assert!(!<c64 as Scalar>::IS_REAL);
            let z = c64::zero();
            assert_eq!(z.re, 0.0);
            assert_eq!(z.im, 0.0);
            let o = c64::one();
            assert_eq!(o.re, 1.0);
            assert_eq!(o.im, 0.0);
            let w = c64::new(3.0, 4.0);
            assert!((w.abs_val() - 5.0).abs() < 1e-12);
            let wc = w.conj();
            assert_eq!(wc.re, 3.0);
            assert!((wc.im - (-4.0)).abs() < 1e-12);
        }

        #[test]
        fn complex_norm_and_inv() {
            let z = c64::new(3.0, 4.0);
            assert!((z.norm() - 5.0).abs() < 1e-12);
            let zi = z.inv();
            let one = z * zi;
            assert!((one.re - 1.0).abs() < 1e-12);
            assert!(one.im.abs() < 1e-12);
        }

        #[test]
        fn complex_exp_ln() {
            // e^(i*pi) ≈ -1 + 0i
            let z = c64::new(0.0, std::f64::consts::PI);
            let e = z.exp();
            assert!((e.re - (-1.0)).abs() < 1e-12);
            assert!(e.im.abs() < 1e-12);
        }

        #[test]
        fn complex_display() {
            let z = c64::new(1.0, 2.0);
            let s = format!("{z}");
            assert_eq!(s, "1+2i");
        }

        #[test]
        fn complex_reduction() {
            let z1 = c64::new(3.0, 4.0); // |z| = 5
            let z2 = c64::new(1.0, 1.0); // |z| ≈ 1.41
            assert_eq!(c64::zero(), c64::new(0.0, 0.0));
            assert!(z1.reduction_gt(z2));
            assert_eq!(z1.reduction_max(z2), z1);
            assert_eq!(z1.reduction_min(z2), z2);
        }
    }
}
