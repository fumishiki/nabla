
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{MathOps, RealScalar, ReductionOps, Scalar, erf_approx};


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<T: RealScalar> {
    /// Primal value.
    pub value: T,
    /// Tangent (derivative) component.
    pub deriv: T,
}

impl<T: RealScalar> Dual<T> {
    /// Construct a dual number with explicit value and derivative.
    #[inline]
    pub fn new(value: T, deriv: T) -> Self {
        Self { value, deriv }
    }

    /// Construct a constant (derivative = 0).
    #[inline]
    pub fn constant(value: T) -> Self {
        Self {
            value,
            deriv: T::zero(),
        }
    }

    // Convenience wrappers mirroring f64 standard methods so that
    // `#[nabla_grad]`-annotated functions using `.exp()`, `.sin()`, etc.
    // compile transparently over `Dual<T>`.

    /// Dual exponential: delegates to `MathOps::math_exp`.
    #[inline]
    pub fn exp(self) -> Self {
        self.math_exp()
    }

    /// Dual natural log: delegates to `MathOps::math_ln`.
    #[inline]
    pub fn ln(self) -> Self {
        self.math_ln()
    }

    /// Dual sin: delegates to `MathOps::math_sin`.
    #[inline]
    pub fn sin(self) -> Self {
        self.math_sin()
    }

    /// Dual cos: delegates to `MathOps::math_cos`.
    #[inline]
    pub fn cos(self) -> Self {
        self.math_cos()
    }

    /// Dual tan: delegates to `MathOps::math_tan`.
    #[inline]
    pub fn tan(self) -> Self {
        self.math_tan()
    }

    /// Dual tanh: delegates to `MathOps::math_tanh`.
    #[inline]
    pub fn tanh(self) -> Self {
        self.math_tanh()
    }

    /// Dual sqrt: delegates to `MathOps::math_sqrt`.
    #[inline]
    pub fn sqrt(self) -> Self {
        self.math_sqrt()
    }

    /// Dual abs: delegates to `MathOps::math_abs`.
    #[inline]
    pub fn abs(self) -> Self {
        self.math_abs()
    }

    /// Dual reciprocal: delegates to `MathOps::math_recip`.
    #[inline]
    pub fn recip(self) -> Self {
        self.math_recip()
    }

    /// Dual asin: delegates to `MathOps::math_asin`.
    #[inline]
    pub fn asin(self) -> Self {
        self.math_asin()
    }

    /// Dual acos: delegates to `MathOps::math_acos`.
    #[inline]
    pub fn acos(self) -> Self {
        self.math_acos()
    }

    /// Dual atan: delegates to `MathOps::math_atan`.
    #[inline]
    pub fn atan(self) -> Self {
        self.math_atan()
    }

    /// Dual atan2: delegates to `MathOps::math_atan2`.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        self.math_atan2(other)
    }

    /// Dual sinh: delegates to `MathOps::math_sinh`.
    #[inline]
    pub fn sinh(self) -> Self {
        self.math_sinh()
    }

    /// Dual cosh: delegates to `MathOps::math_cosh`.
    #[inline]
    pub fn cosh(self) -> Self {
        self.math_cosh()
    }

    /// Dual asinh: delegates to `MathOps::math_asinh`.
    #[inline]
    pub fn asinh(self) -> Self {
        self.math_asinh()
    }

    /// Dual acosh: delegates to `MathOps::math_acosh`.
    #[inline]
    pub fn acosh(self) -> Self {
        self.math_acosh()
    }

    /// Dual atanh: delegates to `MathOps::math_atanh`.
    #[inline]
    pub fn atanh(self) -> Self {
        self.math_atanh()
    }

    /// Dual log2: delegates to `MathOps::math_log2`.
    #[inline]
    pub fn log2(self) -> Self {
        self.math_log2()
    }

    /// Dual log10: delegates to `MathOps::math_log10`.
    #[inline]
    pub fn log10(self) -> Self {
        self.math_log10()
    }
}

impl<T: RealScalar> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}ε", self.value, self.deriv)
    }
}


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

impl<T: RealScalar> Div for Dual<T> {
    type Output = Self;
    // (a+bε)/(c+dε) = a/c + (bc-ad)/c²·ε
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let c: f64 = rhs.value.into();
        let d: f64 = rhs.deriv.into();
        Self::new(T::from_f64(a / c), T::from_f64((b * c - a * d) / (c * c)))
    }
}

impl<T: RealScalar> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(
            T::from_f64(-self.value.into()),
            T::from_f64(-self.deriv.into()),
        )
    }
}


macro_rules! impl_dual_scalar_ops {
    ($scalar:ty) => {
        impl Add<Dual<$scalar>> for $scalar {
            type Output = Dual<$scalar>;
            #[inline]
            fn add(self, rhs: Dual<$scalar>) -> Dual<$scalar> {
                Dual::new(self + rhs.value, rhs.deriv)
            }
        }

        impl Add<$scalar> for Dual<$scalar> {
            type Output = Dual<$scalar>;
            #[inline]
            fn add(self, rhs: $scalar) -> Dual<$scalar> {
                Dual::new(self.value + rhs, self.deriv)
            }
        }

        impl Sub<Dual<$scalar>> for $scalar {
            type Output = Dual<$scalar>;
            #[inline]
            fn sub(self, rhs: Dual<$scalar>) -> Dual<$scalar> {
                Dual::new(
                    self - rhs.value,
                    <$scalar>::from_f64(-Into::<f64>::into(rhs.deriv)),
                )
            }
        }

        impl Sub<$scalar> for Dual<$scalar> {
            type Output = Dual<$scalar>;
            #[inline]
            fn sub(self, rhs: $scalar) -> Dual<$scalar> {
                Dual::new(self.value - rhs, self.deriv)
            }
        }

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
    // tan(a+bε) = tan(a) + sec²(a)*b·ε
    #[inline]
    fn math_tan(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        let c = a.cos();
        Self::new(T::from_f64(a.tan()), T::from_f64(b / (c * c)))
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
        let s = if a > 0.0 {
            1.0
        } else if a < 0.0 {
            -1.0
        } else {
            0.0
        };
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
    // asin(a+bε) = asin(a) + b/sqrt(1-a²)·ε
    #[inline]
    fn math_asin(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.asin()), T::from_f64(b / (1.0 - a * a).sqrt()))
    }
    // acos(a+bε) = acos(a) - b/sqrt(1-a²)·ε
    #[inline]
    fn math_acos(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(
            T::from_f64(a.acos()),
            T::from_f64(-b / (1.0 - a * a).sqrt()),
        )
    }
    // atan(a+bε) = atan(a) + b/(1+a²)·ε
    #[inline]
    fn math_atan(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.atan()), T::from_f64(b / (1.0 + a * a)))
    }
    // atan2(y+byε, x+bxε): d/dy = x/(x²+y²), only propagate self (y) deriv
    #[inline]
    fn math_atan2(self, other: Self) -> Self {
        let y: f64 = self.value.into();
        let x: f64 = other.value.into();
        let by: f64 = self.deriv.into();
        let bx: f64 = other.deriv.into();
        let d = x * x + y * y;
        Self::new(T::from_f64(y.atan2(x)), T::from_f64((x * by - y * bx) / d))
    }
    // sinh(a+bε) = sinh(a) + cosh(a)*b·ε
    #[inline]
    fn math_sinh(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.sinh()), T::from_f64(a.cosh() * b))
    }
    // cosh(a+bε) = cosh(a) + sinh(a)*b·ε
    #[inline]
    fn math_cosh(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.cosh()), T::from_f64(a.sinh() * b))
    }
    // asinh(a+bε) = asinh(a) + b/sqrt(a²+1)·ε
    #[inline]
    fn math_asinh(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(
            T::from_f64(a.asinh()),
            T::from_f64(b / (a * a + 1.0).sqrt()),
        )
    }
    // acosh(a+bε) = acosh(a) + b/sqrt(a²-1)·ε
    #[inline]
    fn math_acosh(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(
            T::from_f64(a.acosh()),
            T::from_f64(b / (a * a - 1.0).sqrt()),
        )
    }
    // atanh(a+bε) = atanh(a) + b/(1-a²)·ε
    #[inline]
    fn math_atanh(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(T::from_f64(a.atanh()), T::from_f64(b / (1.0 - a * a)))
    }
    // log2(a+bε) = log2(a) + b/(a*ln2)·ε
    #[inline]
    fn math_log2(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(
            T::from_f64(a.log2()),
            T::from_f64(b / (a * core::f64::consts::LN_2)),
        )
    }
    // log10(a+bε) = log10(a) + b/(a*ln10)·ε
    #[inline]
    fn math_log10(self) -> Self {
        let a: f64 = self.value.into();
        let b: f64 = self.deriv.into();
        Self::new(
            T::from_f64(a.log10()),
            T::from_f64(b / (a * core::f64::consts::LN_10)),
        )
    }
}


impl<T: RealScalar> ReductionOps for Dual<T> {
    #[inline]
    fn reduction_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn reduction_max(self, other: Self) -> Self {
        if self.value > other.value {
            self
        } else {
            other
        }
    }
    #[inline]
    fn reduction_min(self, other: Self) -> Self {
        if self.value < other.value {
            self
        } else {
            other
        }
    }
    #[inline]
    fn reduction_gt(self, other: Self) -> bool {
        self.value > other.value
    }
}


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
