use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{MathOps, ReductionOps, Scalar, erf_approx};

/// Quantize `f32` to low-precision: `M` mantissa bits, `BIAS`, `EMIN`, `EMAX`, `SIGN_BIT` position.
#[inline]
fn quantize_lowp<
    const M: i32,
    const BIAS: i32,
    const EMIN: i32,
    const EMAX: i32,
    const SIGN_BIT: u8,
>(
    v: f32,
    mask: u8,
) -> u8 {
    if v == 0.0 {
        return 0;
    }
    let sign = u8::from(v.is_sign_negative());
    let av = v.abs();
    if !av.is_finite() {
        let exp_bits = (EMAX + BIAS) as u8;
        return ((sign << SIGN_BIT) | (exp_bits << M) | ((1u8 << M) - 1)) & mask;
    }
    let mut exp = av.log2().floor() as i32;
    if exp < EMIN {
        return 0;
    }
    let mant_bits;
    if exp > EMAX {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        let base = f32::from_bits(((exp + 127) as u32) << 23);
        let mut mant_i = ((av / base - 1.0).max(0.0) * (1 << M) as f32).round() as i32;
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
    ((sign << SIGN_BIT) | ((exp + BIAS) as u8) << M | (mant_bits as u8)) & mask
}

/// Dequantize low-precision bits to `f32`: `M` mantissa bits, `BIAS`, `SIGN_BIT`, `EXP_MASK`.
#[inline]
fn dequantize_lowp<const M: i32, const BIAS: i32, const SIGN_BIT: u8, const EXP_MASK: u8>(
    bits: u8,
    premask: u8,
) -> f32 {
    let b = bits & premask;
    if b == 0 {
        return 0.0;
    }
    let sign = (b >> SIGN_BIT) & 1;
    let exp = i32::from((b >> M) & EXP_MASK) - BIAS;
    let mant = 1.0 + i32::from(b & ((1 << M) - 1) as u8) as f32 / (1 << M) as f32;
    let v = mant * f32::from_bits(((exp + 127) as u32) << 23);
    if sign == 1 { -v } else { v }
}

#[inline]
fn quantize_fp8_e4m3(v: f32) -> u8 {
    quantize_lowp::<3, 7, -6, 7, 7>(v, 0xff)
}
#[inline]
fn dequantize_fp8_e4m3(b: u8) -> f32 {
    dequantize_lowp::<3, 7, 7, 0x0f>(b, 0xff)
}
#[inline]
fn quantize_fp8_e5m2(v: f32) -> u8 {
    quantize_lowp::<2, 15, -14, 15, 7>(v, 0xff)
}
#[inline]
fn dequantize_fp8_e5m2(b: u8) -> f32 {
    dequantize_lowp::<2, 15, 7, 0x1f>(b, 0xff)
}
#[inline]
fn quantize_fp4_e2m1(v: f32) -> u8 {
    quantize_lowp::<1, 1, -1, 2, 3>(v, 0x0f)
}
#[inline]
fn dequantize_fp4_e2m1(b: u8) -> f32 {
    dequantize_lowp::<1, 1, 3, 0x03>(b, 0x0f)
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

macro_rules! impl_lowp_arith {
    ($ty:ident, $Op:ident, $fn_name:ident, $op:tt) => {
        impl $Op for $ty {
            type Output = Self;
            #[inline] fn $fn_name(self, rhs: Self) -> Self { Self::from_f32(self.to_f32() $op rhs.to_f32()) }
        }
    };
}

macro_rules! delegate_lowp_math {
    ($($math_fn:ident => $std_fn:ident),+ $(,)?) => {
        $(#[inline] fn $math_fn(self) -> Self { Self::from_f32(self.to_f32().$std_fn()) })+
    };
}

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

        impl From<f32> for $ty { #[inline] fn from(v: f32) -> Self { Self::from_f32(v) } }
        impl From<$ty> for f32 { #[inline] fn from(v: $ty) -> f32 { v.to_f32() } }
        impl fmt::Debug for $ty { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.to_f32()) } }
        impl fmt::Display for $ty { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.to_f32()) } }

        impl_lowp_arith!($ty, Add, add, +);
        impl_lowp_arith!($ty, Sub, sub, -);
        impl_lowp_arith!($ty, Mul, mul, *);
        impl_lowp_arith!($ty, Div, div, /);

        impl Neg for $ty {
            type Output = Self;
            #[inline] fn neg(self) -> Self { Self::from_f32(-self.to_f32()) }
        }

        impl MathOps for $ty {
            delegate_lowp_math!(
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
            fn math_erf(self) -> Self { Self::from_f32(erf_approx(f64::from(self.to_f32())) as f32) }
            #[inline] fn math_powf(self, p: Self) -> Self { Self::from_f32(self.to_f32().powf(p.to_f32())) }
            #[inline] fn math_mul(self, other: Self) -> Self { Self::from_f32(self.to_f32() * other.to_f32()) }
            #[inline] fn math_div(self, other: Self) -> Self { Self::from_f32(self.to_f32() / other.to_f32()) }
            #[inline] fn math_atan2(self, other: Self) -> Self { Self::from_f32(self.to_f32().atan2(other.to_f32())) }
        }

        impl ReductionOps for $ty {
            #[inline] fn reduction_add(self, other: Self) -> Self { Self::from_f32(self.to_f32() + other.to_f32()) }
            #[inline] fn reduction_max(self, other: Self) -> Self { if self.to_f32() >= other.to_f32() { self } else { other } }
            #[inline] fn reduction_min(self, other: Self) -> Self { if self.to_f32() <= other.to_f32() { self } else { other } }
            #[inline] fn reduction_gt(self, other: Self) -> bool { self.to_f32() > other.to_f32() }
        }

        impl Scalar for $ty {
            type Real = f32;
            const IS_REAL: bool = true;
            #[inline] fn zero() -> Self { Self(0) }
            #[inline] fn one() -> Self { Self::from_f32(1.0) }
            #[inline] fn conj(self) -> Self { self }
            #[inline] fn abs_val(self) -> Self::Real { self.to_f32().abs() }
            #[allow(clippy::cast_possible_truncation)]
            #[inline] fn from_f64(v: f64) -> Self { Self::from_f32(v as f32) }
            #[inline] fn to_f64(self) -> f64 { f64::from(self.to_f32()) }
        }
    };
}

impl_lowp_type!(Fp8E4M3, quantize_fp8_e4m3, dequantize_fp8_e4m3);
impl_lowp_type!(Fp8E5M2, quantize_fp8_e5m2, dequantize_fp8_e5m2);
impl_lowp_type!(Fp4E2M1, quantize_fp4_e2m1, dequantize_fp4_e2m1);
