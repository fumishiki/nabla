
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{MathOps, RealScalar, ReductionOps, Scalar, erf_approx};


#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Complex<T: RealScalar> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

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

impl_complex_binop!(Add, add, +);
impl_complex_binop!(Sub, sub, -);

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

impl<T: RealScalar> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(T::from_f64(-self.re.into()), T::from_f64(-self.im.into()))
    }
}

impl<T: RealScalar> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}i", self.re, self.im)
    }
}


macro_rules! impl_complex_mathops {
    ($ty:ty, $real:ty) => {
        impl MathOps for $ty {
            delegate_math!(math_exp => exp, math_ln => ln, math_sin => sin,
                           math_cos => cos, math_tanh => tanh, math_sqrt => sqrt);
            // tan(z) = sin(z) / cos(z)
            #[inline] fn math_tan(self) -> Self { self.sin() / self.cos() }
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
            // asin(z) = -i * ln(iz + sqrt(1 - z^2))
            #[inline] fn math_asin(self) -> Self {
                let i = Complex::new(0.0 as $real, 1.0 as $real);
                let one = Complex::new(1.0 as $real, 0.0 as $real);
                let neg_i = Complex::new(0.0 as $real, -1.0 as $real);
                let z2 = self * self;
                let inner = (i * self) + (one - z2).sqrt();
                neg_i * inner.ln()
            }
            // acos(z) = -i * ln(z + i*sqrt(1 - z^2))
            #[inline] fn math_acos(self) -> Self {
                let i = Complex::new(0.0 as $real, 1.0 as $real);
                let one = Complex::new(1.0 as $real, 0.0 as $real);
                let neg_i = Complex::new(0.0 as $real, -1.0 as $real);
                let z2 = self * self;
                let inner = self + i * (one - z2).sqrt();
                neg_i * inner.ln()
            }
            // atan(z) = (i/2) * ln((1-iz)/(1+iz))
            #[inline] fn math_atan(self) -> Self {
                let i = Complex::new(0.0 as $real, 1.0 as $real);
                let one = Complex::new(1.0 as $real, 0.0 as $real);
                let half_i = Complex::new(0.0 as $real, 0.5 as $real);
                let iz = i * self;
                let ratio = (one - iz) / (one + iz);
                half_i * ratio.ln()
            }
            #[inline] fn math_atan2(self, _other: Self) -> Self {
                panic!("atan2 is not defined for complex numbers")
            }
            // sinh(z) = (exp(z) - exp(-z)) / 2
            #[inline] fn math_sinh(self) -> Self {
                let half = Complex::new(0.5 as $real, 0.0 as $real);
                half * (self.exp() - (-self).exp())
            }
            // cosh(z) = (exp(z) + exp(-z)) / 2
            #[inline] fn math_cosh(self) -> Self {
                let half = Complex::new(0.5 as $real, 0.0 as $real);
                half * (self.exp() + (-self).exp())
            }
            // asinh(z) = ln(z + sqrt(z^2 + 1))
            #[inline] fn math_asinh(self) -> Self {
                let one = Complex::new(1.0 as $real, 0.0 as $real);
                (self + (self * self + one).sqrt()).ln()
            }
            // acosh(z) = ln(z + sqrt(z^2 - 1))
            #[inline] fn math_acosh(self) -> Self {
                let one = Complex::new(1.0 as $real, 0.0 as $real);
                (self + (self * self - one).sqrt()).ln()
            }
            // atanh(z) = (1/2) * ln((1+z)/(1-z))
            #[inline] fn math_atanh(self) -> Self {
                let one = Complex::new(1.0 as $real, 0.0 as $real);
                let half = Complex::new(0.5 as $real, 0.0 as $real);
                half * ((one + self) / (one - self)).ln()
            }
            // log2(z) = ln(z) / ln(2)
            #[inline] fn math_log2(self) -> Self {
                let ln2 = Complex::new(core::f64::consts::LN_2 as $real, 0.0 as $real);
                self.ln() / ln2
            }
            // log10(z) = ln(z) / ln(10)
            #[inline] fn math_log10(self) -> Self {
                let ln10 = Complex::new(core::f64::consts::LN_10 as $real, 0.0 as $real);
                self.ln() / ln10
            }
        }
    };
}

impl_complex_mathops!(c32, f32);
impl_complex_mathops!(c64, f64);


macro_rules! mag2 {
    ($z:expr) => {
        $z.re * $z.re + $z.im * $z.im
    };
}

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

impl_complex_reduction!(c32);
impl_complex_reduction!(c64);


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

impl_complex_scalar!(c32, f32, |v: f64| Complex::new(v as f32, 0.0), |z: c32| {
    f64::from(z.re)
});
impl_complex_scalar!(c64, f64, |v: f64| Complex::new(v, 0.0), |z: c64| z.re);
