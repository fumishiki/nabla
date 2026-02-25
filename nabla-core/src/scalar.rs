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
            );
            #[inline] #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self { $erf_conv(self) }
            #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p) }
            #[inline] fn math_mul(self, other: Self) -> Self { self * other }
            #[inline] fn math_div(self, other: Self) -> Self { self / other }
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
            delegate_math!(math_exp => exp, math_ln => ln, math_sin => sin,
                           math_cos => cos, math_tanh => tanh, math_sqrt => sqrt);
            #[inline] fn math_log1p(self) -> Self { Complex::new(1.0 as $real + self.re, self.im).ln() }
            #[inline] fn math_abs(self) -> Self { Complex::new(self.norm(), 0.0) }
            #[inline] fn math_recip(self) -> Self { self.inv() }
            #[inline] #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self {
                Complex::new(erf_approx(f64::from(self.re)) as $real, erf_approx(f64::from(self.im)) as $real)
            }
            #[inline] fn math_ceil(self) -> Self { Complex::new(self.re.ceil(), self.im.ceil()) }
            #[inline] fn math_floor(self) -> Self { Complex::new(self.re.floor(), self.im.floor()) }
            #[inline] fn math_round(self) -> Self { Complex::new(self.re.round(), self.im.round()) }
            #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p.re) }
            #[inline] fn math_mul(self, other: Self) -> Self { self * other }
            #[inline] fn math_div(self, other: Self) -> Self { self / other }
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

/// Magnitude-squared for complex comparison: `re^2 + im^2`.
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

    /// Magnitude (absolute value): `sqrt(re^2 + im^2)`.
    #[inline]
    pub fn norm(self) -> T {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        T::from_f64((r * r + i * i).sqrt())
    }

    /// Complex conjugate: `re - im*i`.
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

    /// Complex exponential: `e^(re) * (cos(im) + i*sin(im))`.
    #[must_use]
    #[inline]
    pub fn exp(self) -> Self {
        let r: f64 = self.re.into();
        let i: f64 = self.im.into();
        let scale = r.exp();
        Self::new(T::from_f64(scale * i.cos()), T::from_f64(scale * i.sin()))
    }

    /// Complex natural logarithm: `ln|z| + i*arg(z)`.
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

    /// Complex sine: `sin(re)*cosh(im) + i*cos(re)*sinh(im)`.
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

    /// Complex cosine: `cos(re)*cosh(im) - i*sin(re)*sinh(im)`.
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

    /// Complex power: `z^p` using principal value `exp(p*ln(z))`.
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
    // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
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
// Dual<T> — forward-mode automatic differentiation
// ---------------------------------------------------------------------------

/// Dual number for forward-mode automatic differentiation: `value + deriv·ε`.
///
/// Propagates first-order derivatives through arithmetic and transcendental
/// functions via the chain rule.  CPU-only (`#[cfg(feature = "cpu")]`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "cpu")]
pub struct Dual<T: RealScalar> {
    /// Primal value.
    pub value: T,
    /// Tangent (derivative) component.
    pub deriv: T,
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Dual<T> {
    /// Construct a dual number with explicit value and derivative.
    #[inline]
    pub fn new(value: T, deriv: T) -> Self {
        Self { value, deriv }
    }

    /// Construct a constant (derivative = 0).
    #[inline]
    pub fn constant(value: T) -> Self {
        Self { value, deriv: T::zero() }
    }

    // Convenience wrappers mirroring f64 standard methods so that
    // `#[nabla_grad]`-annotated functions using `.exp()`, `.sin()`, etc.
    // compile transparently over `Dual<T>`.

    /// Dual exponential: delegates to `MathOps::math_exp`.
    #[inline]
    pub fn exp(self) -> Self { self.math_exp() }

    /// Dual natural log: delegates to `MathOps::math_ln`.
    #[inline]
    pub fn ln(self) -> Self { self.math_ln() }

    /// Dual sin: delegates to `MathOps::math_sin`.
    #[inline]
    pub fn sin(self) -> Self { self.math_sin() }

    /// Dual cos: delegates to `MathOps::math_cos`.
    #[inline]
    pub fn cos(self) -> Self { self.math_cos() }

    /// Dual tanh: delegates to `MathOps::math_tanh`.
    #[inline]
    pub fn tanh(self) -> Self { self.math_tanh() }

    /// Dual sqrt: delegates to `MathOps::math_sqrt`.
    #[inline]
    pub fn sqrt(self) -> Self { self.math_sqrt() }

    /// Dual abs: delegates to `MathOps::math_abs`.
    #[inline]
    pub fn abs(self) -> Self { self.math_abs() }

    /// Dual reciprocal: delegates to `MathOps::math_recip`.
    #[inline]
    pub fn recip(self) -> Self { self.math_recip() }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}ε", self.value, self.deriv)
    }
}

// Arithmetic ops for Dual<T>

#[cfg(feature = "cpu")]
impl<T: RealScalar> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            T::from_f64(self.value.into() + rhs.value.into()),
            T::from_f64(self.deriv.into() + rhs.deriv.into()),
        )
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            T::from_f64(self.value.into() - rhs.value.into()),
            T::from_f64(self.deriv.into() - rhs.deriv.into()),
        )
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Mul for Dual<T> {
    type Output = Self;
    // (a+bε)(c+dε) = ac + (ad+bc)ε
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let c: f64 = rhs.value.into();
        let d: f64 = rhs.deriv.into();
        Self::new(T::from_f64(a * c), T::from_f64(a * d + b * c))
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Div for Dual<T> {
    type Output = Self;
    // (a+bε)/(c+dε) = a/c + (bc-ad)/c²·ε
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let c: f64 = rhs.value.into();
        let d: f64 = rhs.deriv.into();
        Self::new(
            T::from_f64(a / c),
            T::from_f64((b * c - a * d) / (c * c)),
        )
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(T::from_f64(-self.value.into()), T::from_f64(-self.deriv.into()))
    }
}

// Mixed scalar-Dual arithmetic: enables natural code like `1.0 + dual` in #[nabla_grad].

macro_rules! impl_dual_scalar_ops {
    ($scalar:ty) => {
        #[cfg(feature = "cpu")]
        impl Add<Dual<$scalar>> for $scalar {
            type Output = Dual<$scalar>;
            #[inline]
            fn add(self, rhs: Dual<$scalar>) -> Dual<$scalar> {
                Dual::new(self + rhs.value, rhs.deriv)
            }
        }

        #[cfg(feature = "cpu")]
        impl Add<$scalar> for Dual<$scalar> {
            type Output = Dual<$scalar>;
            #[inline]
            fn add(self, rhs: $scalar) -> Dual<$scalar> {
                Dual::new(self.value + rhs, self.deriv)
            }
        }

        #[cfg(feature = "cpu")]
        impl Sub<Dual<$scalar>> for $scalar {
            type Output = Dual<$scalar>;
            #[inline]
            fn sub(self, rhs: Dual<$scalar>) -> Dual<$scalar> {
                Dual::new(self - rhs.value, <$scalar>::from_f64(-Into::<f64>::into(rhs.deriv)))
            }
        }

        #[cfg(feature = "cpu")]
        impl Sub<$scalar> for Dual<$scalar> {
            type Output = Dual<$scalar>;
            #[inline]
            fn sub(self, rhs: $scalar) -> Dual<$scalar> {
                Dual::new(self.value - rhs, self.deriv)
            }
        }

        #[cfg(feature = "cpu")]
        impl Mul<Dual<$scalar>> for $scalar {
            type Output = Dual<$scalar>;
            #[inline]
            fn mul(self, rhs: Dual<$scalar>) -> Dual<$scalar> {
                Dual::new(
                    <$scalar>::from_f64(f64::from(self) * f64::from(rhs.value)),
                    <$scalar>::from_f64(f64::from(self) * f64::from(rhs.deriv)),
                )
            }
        }

        #[cfg(feature = "cpu")]
        impl Mul<$scalar> for Dual<$scalar> {
            type Output = Dual<$scalar>;
            #[inline]
            fn mul(self, rhs: $scalar) -> Dual<$scalar> {
                Dual::new(
                    <$scalar>::from_f64(f64::from(self.value) * f64::from(rhs)),
                    <$scalar>::from_f64(f64::from(self.deriv) * f64::from(rhs)),
                )
            }
        }

        #[cfg(feature = "cpu")]
        impl Div<Dual<$scalar>> for $scalar {
            type Output = Dual<$scalar>;
            #[inline]
            fn div(self, rhs: Dual<$scalar>) -> Dual<$scalar> {
                // a / (c+dε) = a/c + (-a*d/c²)ε
                let a = f64::from(self);
                let c = f64::from(rhs.value);
                let d = f64::from(rhs.deriv);
                Dual::new(
                    <$scalar>::from_f64(a / c),
                    <$scalar>::from_f64(-a * d / (c * c)),
                )
            }
        }

        #[cfg(feature = "cpu")]
        impl Div<$scalar> for Dual<$scalar> {
            type Output = Dual<$scalar>;
            #[inline]
            fn div(self, rhs: $scalar) -> Dual<$scalar> {
                let c = f64::from(rhs);
                Dual::new(
                    <$scalar>::from_f64(f64::from(self.value) / c),
                    <$scalar>::from_f64(f64::from(self.deriv) / c),
                )
            }
        }
    };
}

impl_dual_scalar_ops!(f32);
impl_dual_scalar_ops!(f64);

// MathOps for Dual<T> — chain rule propagation

#[cfg(feature = "cpu")]
impl<T: RealScalar> MathOps for Dual<T> {
    // exp(a+bε) = exp(a) + exp(a)*b·ε
    #[inline]
    fn math_exp(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let ea = a.exp();
        Self::new(T::from_f64(ea), T::from_f64(ea * b))
    }
    // ln(a+bε) = ln(a) + b/a·ε
    #[inline]
    fn math_ln(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.ln()), T::from_f64(b / a))
    }
    // log1p(a+bε) = ln(1+a) + b/(1+a)·ε
    #[inline]
    fn math_log1p(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64((1.0 + a).ln()), T::from_f64(b / (1.0 + a)))
    }
    // sin(a+bε) = sin(a) + cos(a)*b·ε
    #[inline]
    fn math_sin(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.sin()), T::from_f64(a.cos() * b))
    }
    // cos(a+bε) = cos(a) - sin(a)*b·ε
    #[inline]
    fn math_cos(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.cos()), T::from_f64(-a.sin() * b))
    }
    // tanh(a+bε) = tanh(a) + (1-tanh²(a))*b·ε
    #[inline]
    fn math_tanh(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let t = a.tanh();
        Self::new(T::from_f64(t), T::from_f64((1.0 - t * t) * b))
    }
    // sqrt(a+bε) = sqrt(a) + b/(2*sqrt(a))·ε
    #[inline]
    fn math_sqrt(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let sa = a.sqrt();
        Self::new(T::from_f64(sa), T::from_f64(b / (2.0 * sa)))
    }
    // |a+bε| = |a| + sign(a)*b·ε
    #[inline]
    fn math_abs(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let s = if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 };
        Self::new(T::from_f64(a.abs()), T::from_f64(s * b))
    }
    // recip(a+bε) = 1/a - b/a²·ε
    #[inline]
    fn math_recip(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(1.0 / a), T::from_f64(-b / (a * a)))
    }
    // erf(a+bε) = erf(a) + 2/√π·exp(-a²)·b·ε
    #[inline]
    fn math_erf(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let d = 2.0 / core::f64::consts::PI.sqrt() * (-a * a).exp() * b;
        Self::new(T::from_f64(erf_approx(a)), T::from_f64(d))
    }
    #[inline]
    fn math_ceil(self) -> Self {
        let a: f64 = self.value.into();
        Self::new(T::from_f64(a.ceil()), T::zero())
    }
    #[inline]
    fn math_floor(self) -> Self {
        let a: f64 = self.value.into();
        Self::new(T::from_f64(a.floor()), T::zero())
    }
    #[inline]
    fn math_round(self) -> Self {
        let a: f64 = self.value.into();
        Self::new(T::from_f64(a.round()), T::zero())
    }
    // a^p = a^p.value + p.value * a^(p.value-1) * b·ε
    #[inline]
    fn math_powf(self, p: Self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let pv: f64 = p.value.into();
        Self::new(
            T::from_f64(a.powf(pv)),
            T::from_f64(pv * a.powf(pv - 1.0) * b),
        )
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

// ReductionOps for Dual<T>

#[cfg(feature = "cpu")]
impl<T: RealScalar> ReductionOps for Dual<T> {
    #[inline]
    fn reduction_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn reduction_max(self, other: Self) -> Self {
        if self.value > other.value { self } else { other }
    }
    #[inline]
    fn reduction_min(self, other: Self) -> Self {
        if self.value < other.value { self } else { other }
    }
    #[inline]
    fn reduction_gt(self, other: Self) -> bool {
        self.value > other.value
    }
}

// Scalar for Dual<T>

#[cfg(feature = "cpu")]
impl<T: RealScalar> Scalar for Dual<T> {
    type Real = T;
    const IS_REAL: bool = true;
    #[inline]
    fn zero() -> Self {
        Dual::new(T::zero(), T::zero())
    }
    #[inline]
    fn one() -> Self {
        Dual::new(T::one(), T::zero())
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn abs_val(self) -> Self::Real {
        self.value.abs_val()
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        Dual::new(T::from_f64(v), T::zero())
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.value.to_f64()
    }
}

// ---------------------------------------------------------------------------
// MultiDual<T, N> — N-lane batch dual number for Jacobian columns
// ---------------------------------------------------------------------------

/// N-lane batch dual number. Computes N partial derivatives simultaneously.
/// `derivs[i]` tracks the partial derivative along lane `i`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(feature = "cpu")]
pub struct MultiDual<T: RealScalar, const N: usize> {
    /// Primal value.
    pub value: T,
    /// Tangent (derivative) components — one per lane.
    pub derivs: [T; N],
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> MultiDual<T, N> {
    /// Construct with explicit value and derivative array.
    #[inline]
    pub fn new(value: T, derivs: [T; N]) -> Self {
        Self { value, derivs }
    }

    /// Construct a constant (all derivatives zero).
    #[inline]
    pub fn constant(value: T) -> Self {
        Self { value, derivs: [T::zero(); N] }
    }

    /// Seed lane `lane` with derivative 1, all others 0.
    #[inline]
    pub fn seed(value: T, lane: usize) -> Self {
        let mut derivs = [T::zero(); N];
        derivs[lane] = T::one();
        Self { value, derivs }
    }

    /// Apply a chain rule: given f(a) and f'(a), produce the MultiDual result.
    #[inline]
    fn chain(self, fval: f64, fprime: f64) -> Self {
        let mut out = [T::zero(); N];
        let mut i = 0;
        while i < N {
            let d: f64 = self.derivs[i].into();
            out[i] = T::from_f64(fprime * d);
            i += 1;
        }
        Self { value: T::from_f64(fval), derivs: out }
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> fmt::Display for MultiDual<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + [", self.value)?;
        for (i, d) in self.derivs.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{d}")?;
        }
        write!(f, "]·ε")
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> PartialOrd for MultiDual<T, N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

// Arithmetic ops for MultiDual<T, N>

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> Add for MultiDual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut d = [T::zero(); N];
        let mut i = 0;
        while i < N {
            d[i] = T::from_f64(self.derivs[i].into() + rhs.derivs[i].into());
            i += 1;
        }
        Self::new(T::from_f64(self.value.into() + rhs.value.into()), d)
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> Sub for MultiDual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut d = [T::zero(); N];
        let mut i = 0;
        while i < N {
            d[i] = T::from_f64(self.derivs[i].into() - rhs.derivs[i].into());
            i += 1;
        }
        Self::new(T::from_f64(self.value.into() - rhs.value.into()), d)
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> Mul for MultiDual<T, N> {
    type Output = Self;
    // (f*g)' = f'*g + f*g'  per lane
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a: f64 = self.value.into();
        let c: f64 = rhs.value.into();
        let mut d = [T::zero(); N];
        let mut i = 0;
        while i < N {
            let b: f64 = self.derivs[i].into();
            let e: f64 = rhs.derivs[i].into();
            d[i] = T::from_f64(a * e + b * c);
            i += 1;
        }
        Self::new(T::from_f64(a * c), d)
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> Div for MultiDual<T, N> {
    type Output = Self;
    // (f/g)' = (f'*g - f*g') / g^2  per lane
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let a: f64 = self.value.into();
        let c: f64 = rhs.value.into();
        let c2 = c * c;
        let mut d = [T::zero(); N];
        let mut i = 0;
        while i < N {
            let b: f64 = self.derivs[i].into();
            let e: f64 = rhs.derivs[i].into();
            d[i] = T::from_f64((b * c - a * e) / c2);
            i += 1;
        }
        Self::new(T::from_f64(a / c), d)
    }
}

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> Neg for MultiDual<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut d = [T::zero(); N];
        let mut i = 0;
        while i < N {
            d[i] = T::from_f64(-self.derivs[i].into());
            i += 1;
        }
        Self::new(T::from_f64(-self.value.into()), d)
    }
}

// MathOps for MultiDual<T, N> — chain rule propagation per lane

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> MathOps for MultiDual<T, N> {
    #[inline]
    fn math_exp(self) -> Self {
        let a: f64 = self.value.into();
        let ea = a.exp();
        self.chain(ea, ea)
    }
    #[inline]
    fn math_ln(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.ln(), 1.0 / a)
    }
    #[inline]
    fn math_log1p(self) -> Self {
        let a: f64 = self.value.into();
        self.chain((1.0 + a).ln(), 1.0 / (1.0 + a))
    }
    #[inline]
    fn math_sin(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.sin(), a.cos())
    }
    #[inline]
    fn math_cos(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.cos(), -a.sin())
    }
    #[inline]
    fn math_tanh(self) -> Self {
        let a: f64 = self.value.into();
        let t = a.tanh();
        self.chain(t, 1.0 - t * t)
    }
    #[inline]
    fn math_sqrt(self) -> Self {
        let a: f64 = self.value.into();
        let sa = a.sqrt();
        self.chain(sa, 1.0 / (2.0 * sa))
    }
    #[inline]
    fn math_abs(self) -> Self {
        let a: f64 = self.value.into();
        let s = if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 };
        self.chain(a.abs(), s)
    }
    #[inline]
    fn math_recip(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(1.0 / a, -1.0 / (a * a))
    }
    #[inline]
    fn math_erf(self) -> Self {
        let a: f64 = self.value.into();
        let fprime = 2.0 / core::f64::consts::PI.sqrt() * (-a * a).exp();
        self.chain(erf_approx(a), fprime)
    }
    #[inline]
    fn math_ceil(self) -> Self {
        let a: f64 = self.value.into();
        Self { value: T::from_f64(a.ceil()), derivs: [T::zero(); N] }
    }
    #[inline]
    fn math_floor(self) -> Self {
        let a: f64 = self.value.into();
        Self { value: T::from_f64(a.floor()), derivs: [T::zero(); N] }
    }
    #[inline]
    fn math_round(self) -> Self {
        let a: f64 = self.value.into();
        Self { value: T::from_f64(a.round()), derivs: [T::zero(); N] }
    }
    #[inline]
    fn math_powf(self, p: Self) -> Self {
        let a: f64 = self.value.into();
        let pv: f64 = p.value.into();
        self.chain(a.powf(pv), pv * a.powf(pv - 1.0))
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

// ReductionOps for MultiDual<T, N>

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> ReductionOps for MultiDual<T, N> {
    #[inline]
    fn reduction_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn reduction_max(self, other: Self) -> Self {
        if self.value > other.value { self } else { other }
    }
    #[inline]
    fn reduction_min(self, other: Self) -> Self {
        if self.value < other.value { self } else { other }
    }
    #[inline]
    fn reduction_gt(self, other: Self) -> bool {
        self.value > other.value
    }
}

// Scalar for MultiDual<T, N>

#[cfg(feature = "cpu")]
impl<T: RealScalar, const N: usize> Scalar for MultiDual<T, N> {
    type Real = T;
    const IS_REAL: bool = false;
    #[inline]
    fn zero() -> Self {
        MultiDual { value: T::zero(), derivs: [T::zero(); N] }
    }
    #[inline]
    fn one() -> Self {
        MultiDual { value: T::one(), derivs: [T::zero(); N] }
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn abs_val(self) -> Self::Real {
        self.value.abs_val()
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        MultiDual { value: T::from_f64(v), derivs: [T::zero(); N] }
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.value.to_f64()
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
// half-precision types (f16, bf16) — CPU-only, f32-promoted compute
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu")]
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
            );
            #[inline] #[allow(clippy::cast_possible_truncation)]
            fn math_erf(self) -> Self { <$ty>::from_f32(erf_approx(f64::from(f32::from(self))) as f32) }
            #[inline] fn math_powf(self, p: Self) -> Self { <$ty>::from_f32(f32::from(self).powf(f32::from(p))) }
            #[inline] fn math_mul(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self) * f32::from(other)) }
            #[inline] fn math_div(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self) / f32::from(other)) }
        }
    };
}

#[cfg(feature = "cpu")]
impl_half_mathops!(half::f16);
#[cfg(feature = "cpu")]
impl_half_mathops!(half::bf16);

#[cfg(feature = "cpu")]
macro_rules! impl_half_reduction {
    ($ty:ty) => {
        impl ReductionOps for $ty {
            #[inline] fn reduction_add(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self) + f32::from(other)) }
            #[inline] fn reduction_max(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self).max(f32::from(other))) }
            #[inline] fn reduction_min(self, other: Self) -> Self { <$ty>::from_f32(f32::from(self).min(f32::from(other))) }
            #[inline] fn reduction_gt(self, other: Self) -> bool { f32::from(self) > f32::from(other) }
        }
    };
}

#[cfg(feature = "cpu")]
impl_half_reduction!(half::f16);
#[cfg(feature = "cpu")]
impl_half_reduction!(half::bf16);

#[cfg(feature = "cpu")]
macro_rules! impl_half_scalar {
    ($ty:ty) => {
        impl Scalar for $ty {
            type Real = f32;
            const IS_REAL: bool = true;
            #[inline] fn zero() -> Self { <$ty>::ZERO }
            #[inline] fn one() -> Self { <$ty>::ONE }
            #[inline] fn conj(self) -> Self { self }
            #[inline] fn abs_val(self) -> Self::Real { f32::from(self).abs() }
            #[allow(clippy::cast_possible_truncation)]
            #[inline] fn from_f64(v: f64) -> Self { <$ty>::from_f32(v as f32) }
            #[inline] fn to_f64(self) -> f64 { f64::from(f32::from(self)) }
        }
    };
}

#[cfg(feature = "cpu")]
impl_half_scalar!(half::f16);
#[cfg(feature = "cpu")]
impl_half_scalar!(half::bf16);

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
