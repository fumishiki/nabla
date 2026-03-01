
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{MathOps, RealScalar, ReductionOps, Scalar, erf_approx};


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiDual<T: RealScalar, const N: usize> {
    /// Primal value.
    pub value: T,
    /// Tangent (derivative) components — one per lane.
    pub derivs: [T; N],
}

impl<T: RealScalar, const N: usize> MultiDual<T, N> {
    /// Construct with explicit value and derivative array.
    #[inline]
    pub fn new(value: T, derivs: [T; N]) -> Self {
        Self { value, derivs }
    }

    /// Construct a constant (all derivatives zero).
    #[inline]
    pub fn constant(value: T) -> Self {
        Self {
            value,
            derivs: [T::zero(); N],
        }
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
        Self {
            value: T::from_f64(fval),
            derivs: out,
        }
    }
}

impl<T: RealScalar, const N: usize> fmt::Display for MultiDual<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + [", self.value)?;
        for (i, d) in self.derivs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]·ε")
    }
}

impl<T: RealScalar, const N: usize> PartialOrd for MultiDual<T, N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}


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
    // tan'(a) = sec²(a) = 1/cos²(a)
    #[inline]
    fn math_tan(self) -> Self {
        let a: f64 = self.value.into();
        let c = a.cos();
        self.chain(a.tan(), 1.0 / (c * c))
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
        let s = if a > 0.0 {
            1.0
        } else if a < 0.0 {
            -1.0
        } else {
            0.0
        };
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
        Self {
            value: T::from_f64(a.ceil()),
            derivs: [T::zero(); N],
        }
    }
    #[inline]
    fn math_floor(self) -> Self {
        let a: f64 = self.value.into();
        Self {
            value: T::from_f64(a.floor()),
            derivs: [T::zero(); N],
        }
    }
    #[inline]
    fn math_round(self) -> Self {
        let a: f64 = self.value.into();
        Self {
            value: T::from_f64(a.round()),
            derivs: [T::zero(); N],
        }
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
    // asin'(a) = 1/sqrt(1-a²)
    #[inline]
    fn math_asin(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.asin(), 1.0 / (1.0 - a * a).sqrt())
    }
    // acos'(a) = -1/sqrt(1-a²)
    #[inline]
    fn math_acos(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.acos(), -1.0 / (1.0 - a * a).sqrt())
    }
    // atan'(a) = 1/(1+a²)
    #[inline]
    fn math_atan(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.atan(), 1.0 / (1.0 + a * a))
    }
    // atan2: propagate both y and x derivatives
    #[inline]
    fn math_atan2(self, other: Self) -> Self {
        let y: f64 = self.value.into();
        let x: f64 = other.value.into();
        let d = x * x + y * y;
        let mut out = [T::zero(); N];
        let mut i = 0;
        while i < N {
            let dy: f64 = self.derivs[i].into();
            let dx: f64 = other.derivs[i].into();
            out[i] = T::from_f64((x * dy - y * dx) / d);
            i += 1;
        }
        Self::new(T::from_f64(y.atan2(x)), out)
    }
    // sinh'(a) = cosh(a)
    #[inline]
    fn math_sinh(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.sinh(), a.cosh())
    }
    // cosh'(a) = sinh(a)
    #[inline]
    fn math_cosh(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.cosh(), a.sinh())
    }
    // asinh'(a) = 1/sqrt(a²+1)
    #[inline]
    fn math_asinh(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.asinh(), 1.0 / (a * a + 1.0).sqrt())
    }
    // acosh'(a) = 1/sqrt(a²-1)
    #[inline]
    fn math_acosh(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.acosh(), 1.0 / (a * a - 1.0).sqrt())
    }
    // atanh'(a) = 1/(1-a²)
    #[inline]
    fn math_atanh(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.atanh(), 1.0 / (1.0 - a * a))
    }
    // log2'(a) = 1/(a*ln2)
    #[inline]
    fn math_log2(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.log2(), 1.0 / (a * core::f64::consts::LN_2))
    }
    // log10'(a) = 1/(a*ln10)
    #[inline]
    fn math_log10(self) -> Self {
        let a: f64 = self.value.into();
        self.chain(a.log10(), 1.0 / (a * core::f64::consts::LN_10))
    }
}


impl<T: RealScalar, const N: usize> ReductionOps for MultiDual<T, N> {
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


impl<T: RealScalar, const N: usize> Scalar for MultiDual<T, N> {
    type Real = T;
    const IS_REAL: bool = false;
    #[inline]
    fn zero() -> Self {
        MultiDual {
            value: T::zero(),
            derivs: [T::zero(); N],
        }
    }
    #[inline]
    fn one() -> Self {
        MultiDual {
            value: T::one(),
            derivs: [T::zero(); N],
        }
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
        MultiDual {
            value: T::from_f64(v),
            derivs: [T::zero(); N],
        }
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.value.to_f64()
    }
}
