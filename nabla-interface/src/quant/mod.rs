//! GGUF quantization: enum, metadata, packing/unpacking dispatch.
//!
//! Type ID reference (ggml):
//! `F32`=0 `F16`=1 `Q4_0`=2 `Q4_1`=3 `Q5_0`=6 `Q5_1`=7 `Q8_0`=8 `Q8_1`=9
//! `Q2_K`=10 `Q3_K_S`=11 `Q3_K_M`=12 `Q3_K_L`=13 `Q4_K_S`=14 `Q4_K_M`=15
//! `Q5_K_S`=16 `Q5_K_M`=17 `Q6_K`=18 `IQ2_XXS`=19 `IQ2_XS`=20 `IQ3_XXS`=21
//! `IQ1_S`=22 `IQ4_NL`=23 `IQ3_S`=24 `IQ2_S`=25 `IQ4_XS`=26 `I8`=27 `I16`=28
//! `I32`=29 `I64`=30 `F64`=31 `IQ1_M`=32 `BF16`=33 `TQ1_0`=34 `TQ2_0`=35

mod kquant;
mod legacy;

pub use kquant::*;
pub use legacy::*;

use crate::{Error, Result};
use half::f16;

const QK: usize = 32;
const QK_K: usize = 256;

/// All 36 GGUF quantization types recognized by llama.cpp / GGML.
/// Discriminants match ggml type codes exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types, missing_docs)]
pub enum GgufQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K_S = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
    IQ2_XXS = 19,
    IQ2_XS = 20,
    IQ3_XXS = 21,
    IQ1_S = 22,
    IQ4_NL = 23,
    IQ3_S = 24,
    IQ2_S = 25,
    IQ4_XS = 26,
    I8 = 27,
    I16 = 28,
    I32 = 29,
    I64 = 30,
    F64 = 31,
    IQ1_M = 32,
    BF16 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
}

impl GgufQuantType {
    /// Return the ggml type code for GGUF `tensor_info`.
    #[must_use]
    pub const fn type_id(self) -> u32 {
        self as u32
    }

    /// Number of floats per quantization block.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL => QK,
            Self::Q2_K
            | Self::Q3_K_S
            | Self::Q3_K_M
            | Self::Q3_K_L
            | Self::Q4_K_S
            | Self::Q4_K_M
            | Self::Q5_K_S
            | Self::Q5_K_M
            | Self::Q6_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ4_XS
            | Self::IQ1_S
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => QK_K,
        }
    }

    /// Byte size of one quantized block.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::I8 => 1,
            Self::Q4_0 | Self::IQ4_NL => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L | Self::IQ3_S => 110,
            Self::Q4_K_S | Self::Q4_K_M => 144,
            Self::Q5_K_S | Self::Q5_K_M => 176,
            Self::Q6_K => 210,
            Self::IQ1_S => 50,
            Self::IQ1_M => 56,
            Self::IQ2_XXS | Self::TQ2_0 => 66,
            Self::IQ2_XS => 74,
            Self::IQ2_S => 82,
            Self::IQ3_XXS => 98,
            Self::IQ4_XS => 136,
            Self::TQ1_0 => 58,
        }
    }

    /// Bits per weight x1000 (integer for display: 4500 = 4.5 bpw).
    #[must_use]
    pub const fn bpw_x1000(self) -> u32 {
        match self {
            Self::F32 | Self::I32 => 32_000,
            Self::F16 | Self::BF16 | Self::I16 => 16_000,
            Self::F64 | Self::I64 => 64_000,
            Self::I8 => 8_000,
            Self::Q4_0 | Self::IQ4_NL | Self::Q4_K_S | Self::Q4_K_M => 4_500,
            Self::Q4_1 => 5_000,
            Self::Q5_0 | Self::Q5_K_S | Self::Q5_K_M => 5_500,
            Self::Q5_1 => 6_000,
            Self::Q8_0 => 8_500,
            Self::Q8_1 => 9_000,
            Self::Q2_K => 2_625,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L | Self::IQ3_S => 3_438,
            Self::Q6_K => 6_563,
            Self::IQ1_S => 1_563,
            Self::IQ1_M => 1_750,
            Self::IQ2_XXS | Self::TQ2_0 => 2_063,
            Self::IQ2_XS => 2_313,
            Self::IQ2_S => 2_563,
            Self::IQ3_XXS => 3_063,
            Self::IQ4_XS => 4_250,
            Self::TQ1_0 => 1_688,
        }
    }

    /// Whether this type supports direct quantize/dequantize (no importance matrix needed).
    #[must_use]
    pub const fn is_quantizable(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2_K
                | Self::Q3_K_S
                | Self::Q3_K_M
                | Self::Q3_K_L
                | Self::Q4_K_S
                | Self::Q4_K_M
                | Self::Q5_K_S
                | Self::Q5_K_M
                | Self::Q6_K
        )
    }
}

/// Quantize f32 slice to the specified GGUF type.
///
/// # Errors
/// Returns `Error::Quant` if length is invalid or type is not quantizable.
pub fn quantize(data: &[f32], qtype: GgufQuantType) -> Result<Vec<u8>> {
    match qtype {
        GgufQuantType::F32 => {
            let mut out = Vec::with_capacity(data.len() * 4);
            for &v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        GgufQuantType::F16 => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for &v in data {
                out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            Ok(out)
        }
        GgufQuantType::BF16 => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for &v in data {
                #[allow(clippy::cast_possible_truncation)]
                let hi = (v.to_bits() >> 16) as u16;
                out.extend_from_slice(&hi.to_le_bytes());
            }
            Ok(out)
        }
        GgufQuantType::F64 => {
            let mut out = Vec::with_capacity(data.len() * 8);
            for &v in data {
                out.extend_from_slice(&f64::from(v).to_le_bytes());
            }
            Ok(out)
        }
        GgufQuantType::Q4_0 => quantize_q4_0(data),
        GgufQuantType::Q4_1 => quantize_q4_1(data),
        GgufQuantType::Q5_0 => quantize_q5_0(data),
        GgufQuantType::Q5_1 => quantize_q5_1(data),
        GgufQuantType::Q8_0 => quantize_q8_0(data),
        GgufQuantType::Q8_1 => quantize_q8_1(data),
        GgufQuantType::Q2_K => quantize_q2_k(data),
        GgufQuantType::Q3_K_S | GgufQuantType::Q3_K_M | GgufQuantType::Q3_K_L => {
            quantize_q3_k(data)
        }
        GgufQuantType::Q4_K_S | GgufQuantType::Q4_K_M => quantize_q4_k(data),
        GgufQuantType::Q5_K_S | GgufQuantType::Q5_K_M => quantize_q5_k(data),
        GgufQuantType::Q6_K => quantize_q6_k(data),
        _ => Err(Error::Quant(format!(
            "{qtype:?} requires importance matrix or is not quantizable"
        ))),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn f64_bytes_to_f32(b: &[u8]) -> f32 {
    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
}

/// Dequantize raw block bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if byte length is invalid or type is not supported.
pub fn dequantize(data: &[u8], qtype: GgufQuantType) -> Result<Vec<f32>> {
    match qtype {
        GgufQuantType::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(Error::Quant(format!(
                    "F32: byte len {} not divisible by 4",
                    data.len()
                )));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        GgufQuantType::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(Error::Quant(format!(
                    "F16: byte len {} not divisible by 2",
                    data.len()
                )));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect())
        }
        GgufQuantType::BF16 => {
            if !data.len().is_multiple_of(2) {
                return Err(Error::Quant(format!(
                    "BF16: byte len {} not divisible by 2",
                    data.len()
                )));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect())
        }
        GgufQuantType::F64 => {
            if !data.len().is_multiple_of(8) {
                return Err(Error::Quant(format!(
                    "F64: byte len {} not divisible by 8",
                    data.len()
                )));
            }
            Ok(data.chunks_exact(8).map(f64_bytes_to_f32).collect())
        }
        GgufQuantType::Q4_0 => dequantize_q4_0(data),
        GgufQuantType::Q4_1 => dequantize_q4_1(data),
        GgufQuantType::Q5_0 => dequantize_q5_0(data),
        GgufQuantType::Q5_1 => dequantize_q5_1(data),
        GgufQuantType::Q8_0 => dequantize_q8_0(data),
        GgufQuantType::Q8_1 => dequantize_q8_1(data),
        GgufQuantType::Q2_K => dequantize_q2_k(data),
        GgufQuantType::Q3_K_S | GgufQuantType::Q3_K_M | GgufQuantType::Q3_K_L => {
            dequantize_q3_k(data)
        }
        GgufQuantType::Q4_K_S | GgufQuantType::Q4_K_M => dequantize_q4_k(data),
        GgufQuantType::Q5_K_S | GgufQuantType::Q5_K_M => dequantize_q5_k(data),
        GgufQuantType::Q6_K => dequantize_q6_k(data),
        _ => Err(Error::Quant(format!(
            "{qtype:?} dequantization not supported"
        ))),
    }
}

/// Recommended quant type per layer (attention/norm=`Q6_K`, FFN=`Q4_K_M`).
#[must_use]
pub fn recommended_quant_for_layer(name: &str) -> GgufQuantType {
    if name.contains("embed") || name.contains("token_embd") {
        return GgufQuantType::F16;
    }
    if name.contains("output.weight") || name.contains("output_norm") {
        return GgufQuantType::Q6_K;
    }
    if name.contains("attn") || name.contains("attention") || name.contains("norm") {
        GgufQuantType::Q6_K
    } else {
        GgufQuantType::Q4_K_M
    }
}
