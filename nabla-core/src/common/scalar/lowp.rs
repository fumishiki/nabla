use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{MathOps, ReductionOps, Scalar};

#[inline]
fn quantize_fp8_e4m3(v: f32) -> u8 {
    const M: i32 = 3;
    const BIAS: i32 = 7;
    const EMIN: i32 = -6;
    const EMAX: i32 = 7;
    if v == 0.0 {
        return 0;
    }
    let sign = v.is_sign_negative() as u8;
    let av = v.abs();
    if !av.is_finite() {
        let exp_bits = (EMAX + BIAS) as u8;
        let mant = (1u8 << M) - 1;
        return (sign << 7) | (exp_bits << M) | mant;
    }
    let mut exp = av.log2().floor() as i32;
    if exp < EMIN {
        return 0;
    }
    let mant_bits: i32;
    if exp > EMAX {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        let base = f32::from_bits(((exp + 127) as u32) << 23);
        let mut mant = av / base - 1.0;
        if mant < 0.0 {
            mant = 0.0;
        }
        let mut mant_i = (mant * (1 << M) as f32).round() as i32;
        if mant_i >= (1 << M) {
            mant_i = 0;
            exp += 1;
            if exp > EMAX {
                exp = EMAX;
                mant_i = (1 << M) - 1;
            }
        }
        mant_bits = mant_i;
    }
    let exp_bits = (exp + BIAS) as u8;
    (sign << 7) | (exp_bits << M) | (mant_bits as u8)
}

#[inline]
fn dequantize_fp8_e4m3(bits: u8) -> f32 {
    const M: i32 = 3;
    const BIAS: i32 = 7;
    if bits == 0 {
        return 0.0;
    }
    let sign = (bits >> 7) & 1;
    let exp_bits = ((bits >> M) & 0x0f) as i32;
    let mant_bits = (bits & ((1 << M) - 1) as u8) as i32;
    let exp = exp_bits - BIAS;
    let mant = 1.0 + mant_bits as f32 / (1 << M) as f32;
    let v = mant * f32::from_bits(((exp + 127) as u32) << 23);
    if sign == 1 { -v } else { v }
}

#[inline]
fn quantize_fp8_e5m2(v: f32) -> u8 {
    const M: i32 = 2;
    const BIAS: i32 = 15;
    const EMIN: i32 = -14;
    const EMAX: i32 = 15;
    if v == 0.0 {
        return 0;
    }
    let sign = v.is_sign_negative() as u8;
    let av = v.abs();
    if !av.is_finite() {
        let exp_bits = (EMAX + BIAS) as u8;
        let mant = (1u8 << M) - 1;
        return (sign << 7) | (exp_bits << M) | mant;
    }
    let mut exp = av.log2().floor() as i32;
    if exp < EMIN {
        return 0;
    }
    let mant_bits: i32;
    if exp > EMAX {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        let base = f32::from_bits(((exp + 127) as u32) << 23);
        let mut mant = av / base - 1.0;
        if mant < 0.0 {
            mant = 0.0;
        }
        let mut mant_i = (mant * (1 << M) as f32).round() as i32;
        if mant_i >= (1 << M) {
            mant_i = 0;
            exp += 1;
            if exp > EMAX {
                exp = EMAX;
                mant_i = (1 << M) - 1;
            }
        }
        mant_bits = mant_i;
    }
    let exp_bits = (exp + BIAS) as u8;
    (sign << 7) | (exp_bits << M) | (mant_bits as u8)
}

#[inline]
fn dequantize_fp8_e5m2(bits: u8) -> f32 {
    const M: i32 = 2;
    const BIAS: i32 = 15;
    if bits == 0 {
        return 0.0;
    }
    let sign = (bits >> 7) & 1;
    let exp_bits = ((bits >> M) & 0x1f) as i32;
    let mant_bits = (bits & ((1 << M) - 1) as u8) as i32;
    let exp = exp_bits - BIAS;
    let mant = 1.0 + mant_bits as f32 / (1 << M) as f32;
    let v = mant * f32::from_bits(((exp + 127) as u32) << 23);
    if sign == 1 { -v } else { v }
}

#[inline]
fn quantize_fp4_e2m1(v: f32) -> u8 {
    const M: i32 = 1;
    const BIAS: i32 = 1;
    const EMIN: i32 = -1;
    const EMAX: i32 = 2;
    if v == 0.0 {
        return 0;
    }
    let sign = v.is_sign_negative() as u8;
    let av = v.abs();
    if !av.is_finite() {
        let exp_bits = (EMAX + BIAS) as u8;
        let mant = (1u8 << M) - 1;
        return (sign << 3) | (exp_bits << M) | mant;
    }
    let mut exp = av.log2().floor() as i32;
    if exp < EMIN {
        return 0;
    }
    let mant_bits: i32;
    if exp > EMAX {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        let base = f32::from_bits(((exp + 127) as u32) << 23);
        let mut mant = av / base - 1.0;
        if mant < 0.0 {
            mant = 0.0;
        }
        let mut mant_i = (mant * (1 << M) as f32).round() as i32;
        if mant_i >= (1 << M) {
            mant_i = 0;
            exp += 1;
            if exp > EMAX {
                exp = EMAX;
                mant_i = (1 << M) - 1;
            }
        }
        mant_bits = mant_i;
    }
    let exp_bits = (exp + BIAS) as u8;
    ((sign << 3) | (exp_bits << M) | (mant_bits as u8)) & 0x0f
}

#[inline]
fn dequantize_fp4_e2m1(bits: u8) -> f32 {
    const M: i32 = 1;
    const BIAS: i32 = 1;
    let b = bits & 0x0f;
    if b == 0 {
        return 0.0;
    }
    let sign = (b >> 3) & 1;
    let exp_bits = ((b >> M) & 0x03) as i32;
    let mant_bits = (b & ((1 << M) - 1) as u8) as i32;
    let exp = exp_bits - BIAS;
    let mant = 1.0 + mant_bits as f32 / (1 << M) as f32;
    let v = mant * f32::from_bits(((exp + 127) as u32) << 23);
    if sign == 1 { -v } else { v }
}

#[repr(transparent)]
#[derive(Copy, Clone, Default, PartialEq, Eq)]
/// 8-bit floating-point scalar with 4 exponent and 3 mantissa bits (E4M3).
pub struct Fp8E4M3(pub u8);

#[repr(transparent)]
#[derive(Copy, Clone, Default, PartialEq, Eq)]
/// 8-bit floating-point scalar with 5 exponent and 2 mantissa bits (E5M2).
pub struct Fp8E5M2(pub u8);

#[repr(transparent)]
#[derive(Copy, Clone, Default, PartialEq, Eq)]
/// 4-bit floating-point scalar with 2 exponent and 1 mantissa bit (E2M1).
pub struct Fp4E2M1(pub u8);

macro_rules! impl_lowp_type {
    ($ty:ident, $quant:ident, $dequant:ident) => {
        impl $ty {
            /// Convert from `f32`.
            #[inline]
            pub fn from_f32(v: f32) -> Self {
                Self($quant(v))
            }
            /// Convert to `f32`.
            #[inline]
            pub fn to_f32(self) -> f32 {
                $dequant(self.0)
            }
        }

        impl From<f32> for $ty {
            #[inline]
            fn from(v: f32) -> Self {
                Self::from_f32(v)
            }
        }

        impl From<$ty> for f32 {
            #[inline]
            fn from(v: $ty) -> f32 {
                v.to_f32()
            }
        }

        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_f32())
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_f32())
            }
        }

        impl Add for $ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                Self::from_f32(self.to_f32() + rhs.to_f32())
            }
        }

        impl Sub for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::from_f32(self.to_f32() - rhs.to_f32())
            }
        }

        impl Mul for $ty {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                Self::from_f32(self.to_f32() * rhs.to_f32())
            }
        }

        impl Div for $ty {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                Self::from_f32(self.to_f32() / rhs.to_f32())
            }
        }

        impl Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self::from_f32(-self.to_f32())
            }
        }

        impl MathOps for $ty {
            #[inline]
            fn math_exp(self) -> Self {
                Self::from_f32(self.to_f32().exp())
            }
            #[inline]
            fn math_ln(self) -> Self {
                Self::from_f32(self.to_f32().ln())
            }
            #[inline]
            fn math_log1p(self) -> Self {
                Self::from_f32(self.to_f32().ln_1p())
            }
            #[inline]
            fn math_sin(self) -> Self {
                Self::from_f32(self.to_f32().sin())
            }
            #[inline]
            fn math_cos(self) -> Self {
                Self::from_f32(self.to_f32().cos())
            }
            #[inline]
            fn math_tan(self) -> Self {
                Self::from_f32(self.to_f32().tan())
            }
            #[inline]
            fn math_tanh(self) -> Self {
                Self::from_f32(self.to_f32().tanh())
            }
            #[inline]
            fn math_sqrt(self) -> Self {
                Self::from_f32(self.to_f32().sqrt())
            }
            #[inline]
            fn math_abs(self) -> Self {
                Self::from_f32(self.to_f32().abs())
            }
            #[inline]
            fn math_recip(self) -> Self {
                Self::from_f32(self.to_f32().recip())
            }
            #[inline]
            fn math_erf(self) -> Self {
                let v = self.to_f32() as f64;
                let t = 1.0 / (1.0 + 0.327_591_1 * v.abs());
                let poly = t
                    * (0.254_829_592
                        + t * (-0.284_496_736
                            + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
                let res = 1.0 - poly * (-v * v).exp();
                Self::from_f32((res.copysign(v)) as f32)
            }
            #[inline]
            fn math_ceil(self) -> Self {
                Self::from_f32(self.to_f32().ceil())
            }
            #[inline]
            fn math_floor(self) -> Self {
                Self::from_f32(self.to_f32().floor())
            }
            #[inline]
            fn math_round(self) -> Self {
                Self::from_f32(self.to_f32().round())
            }
            #[inline]
            fn math_powf(self, p: Self) -> Self {
                Self::from_f32(self.to_f32().powf(p.to_f32()))
            }
            #[inline]
            fn math_mul(self, other: Self) -> Self {
                Self::from_f32(self.to_f32() * other.to_f32())
            }
            #[inline]
            fn math_div(self, other: Self) -> Self {
                Self::from_f32(self.to_f32() / other.to_f32())
            }
            #[inline]
            fn math_asin(self) -> Self {
                Self::from_f32(self.to_f32().asin())
            }
            #[inline]
            fn math_acos(self) -> Self {
                Self::from_f32(self.to_f32().acos())
            }
            #[inline]
            fn math_atan(self) -> Self {
                Self::from_f32(self.to_f32().atan())
            }
            #[inline]
            fn math_atan2(self, other: Self) -> Self {
                Self::from_f32(self.to_f32().atan2(other.to_f32()))
            }
            #[inline]
            fn math_sinh(self) -> Self {
                Self::from_f32(self.to_f32().sinh())
            }
            #[inline]
            fn math_cosh(self) -> Self {
                Self::from_f32(self.to_f32().cosh())
            }
            #[inline]
            fn math_asinh(self) -> Self {
                Self::from_f32(self.to_f32().asinh())
            }
            #[inline]
            fn math_acosh(self) -> Self {
                Self::from_f32(self.to_f32().acosh())
            }
            #[inline]
            fn math_atanh(self) -> Self {
                Self::from_f32(self.to_f32().atanh())
            }
            #[inline]
            fn math_log2(self) -> Self {
                Self::from_f32(self.to_f32().log2())
            }
            #[inline]
            fn math_log10(self) -> Self {
                Self::from_f32(self.to_f32().log10())
            }
        }

        impl ReductionOps for $ty {
            #[inline]
            fn reduction_add(self, other: Self) -> Self {
                Self::from_f32(self.to_f32() + other.to_f32())
            }
            #[inline]
            fn reduction_max(self, other: Self) -> Self {
                let a = self.to_f32();
                let b = other.to_f32();
                if a >= b { self } else { other }
            }
            #[inline]
            fn reduction_min(self, other: Self) -> Self {
                let a = self.to_f32();
                let b = other.to_f32();
                if a <= b { self } else { other }
            }
            #[inline]
            fn reduction_gt(self, other: Self) -> bool {
                self.to_f32() > other.to_f32()
            }
        }

        impl Scalar for $ty {
            type Real = f32;
            const IS_REAL: bool = true;
            #[inline]
            fn zero() -> Self {
                Self(0)
            }
            #[inline]
            fn one() -> Self {
                Self::from_f32(1.0)
            }
            #[inline]
            fn conj(self) -> Self {
                self
            }
            #[inline]
            fn abs_val(self) -> Self::Real {
                self.to_f32().abs()
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                Self::from_f32(v as f32)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                f64::from(self.to_f32())
            }
        }
    };
}

impl_lowp_type!(Fp8E4M3, quantize_fp8_e4m3, dequantize_fp8_e4m3);
impl_lowp_type!(Fp8E5M2, quantize_fp8_e5m2, dequantize_fp8_e5m2);
impl_lowp_type!(Fp4E2M1, quantize_fp4_e2m1, dequantize_fp4_e2m1);
