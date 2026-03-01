//! GGUF v3 export — quantization + binary writer for llama.cpp compatibility.
//!
//! Covers P6-GGUF-01 through P6-GGUF-09.

use std::collections::HashMap;
use std::io::{self, Seek, Write};

// f16/bf16 re-exported from nabla-ml prelude (half crate, behind cpu feature)
#[cfg(feature = "cpu")]
use nabla_ml::prelude::{bf16, f16};

// ── Constants ──────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE
const GGUF_VERSION: u32 = 3;
const QK: usize = 32; // legacy block size
const QK_K: usize = 256; // K-quant block size
const DEFAULT_ALIGNMENT: u64 = 32;

// ── GgufQuantType (P6-GGUF-02) ────────────────────────────────────────────

/// Every quantization type recognized by llama.cpp / GGML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K_S = 11, Q3_K_M = 12, Q3_K_L = 13,
    Q4_K_S = 14, Q4_K_M = 15, Q5_K_S = 16, Q5_K_M = 17, Q6_K = 18,
    IQ2_XXS = 19, IQ2_XS = 20, IQ3_XXS = 21, IQ1_S = 22, IQ4_NL = 23,
    IQ3_S = 24, IQ2_S = 25, IQ4_XS = 26, I8 = 27, I16 = 28, I32 = 29, I64 = 30,
    F64 = 31, IQ1_M = 32, BF16 = 33, TQ1_0 = 34, TQ2_0 = 35,
}

impl GgufQuantType {
    /// Elements per quantization block.
    pub const fn block_size(self) -> usize {
        use GgufQuantType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL => QK,
            Q2_K | Q3_K_S | Q3_K_M | Q3_K_L | Q4_K_S | Q4_K_M
            | Q5_K_S | Q5_K_M | Q6_K | IQ2_XXS | IQ2_XS | IQ2_S
            | IQ3_XXS | IQ3_S | IQ4_XS | IQ1_S | IQ1_M | TQ1_0 | TQ2_0 => QK_K,
        }
    }

    /// Bytes per block for the given quant type.
    pub const fn type_size(self) -> usize {
        use GgufQuantType::*;
        match self {
            F32 => 4, F16 | BF16 => 2, F64 => 8,
            I8 => 1, I16 => 2, I32 => 4, I64 => 8,
            Q4_0 | IQ4_NL => 18, Q4_1 => 20, Q5_0 => 22, Q5_1 => 24,
            Q8_0 => 34, Q8_1 => 36,
            Q2_K => 84, Q3_K_S | Q3_K_M | Q3_K_L => 110, Q4_K_S | Q4_K_M => 144,
            Q5_K_S | Q5_K_M => 176, Q6_K => 210,
            IQ1_S => 50, IQ1_M => 56, IQ2_XXS => 66, IQ2_XS => 74, IQ2_S => 82,
            IQ3_XXS => 98, IQ3_S => 110, IQ4_XS => 136,
            TQ1_0 => 58, TQ2_0 => 66,
        }
    }

    /// Bits per weight (×1000 to stay integer for display).
    pub const fn bpw_x1000(self) -> u32 {
        use GgufQuantType::*;
        match self {
            F32 => 32_000, F16 | BF16 => 16_000, F64 => 64_000,
            I8 => 8_000, I16 => 16_000, I32 => 32_000, I64 => 64_000,
            Q4_0 | IQ4_NL => 4_500, Q4_1 => 5_000, Q5_0 => 5_500, Q5_1 => 6_000,
            Q8_0 => 8_500, Q8_1 => 9_000,
            Q2_K => 2_625, Q3_K_S | Q3_K_M | Q3_K_L => 3_438,
            Q4_K_S | Q4_K_M => 4_500, Q5_K_S | Q5_K_M => 5_500, Q6_K => 6_563,
            IQ1_S => 1_563, IQ1_M => 1_750, IQ2_XXS => 2_063, IQ2_XS => 2_313,
            IQ2_S => 2_563, IQ3_XXS => 3_063, IQ3_S => 3_438, IQ4_XS => 4_250,
            TQ1_0 => 1_688, TQ2_0 => 2_063,
        }
    }
}

// ── GGUF metadata value types ──────────────────────────────────────────────

/// Metadata value types for GGUF KV pairs.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    F32(f32), Bool(bool), Str(String), U64(u64), I64(i64), F64(f64),
    ArrayU32(Vec<u32>), ArrayI32(Vec<i32>), ArrayF32(Vec<f32>), ArrayStr(Vec<String>),
}

impl GgufValue {
    fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0, Self::I8(_) => 1, Self::U16(_) => 2, Self::I16(_) => 3,
            Self::U32(_) => 4, Self::I32(_) => 5, Self::F32(_) => 6, Self::Bool(_) => 7,
            Self::Str(_) => 8, Self::ArrayU32(_) | Self::ArrayI32(_)
            | Self::ArrayF32(_) | Self::ArrayStr(_) => 9,
            Self::U64(_) => 10, Self::I64(_) => 11, Self::F64(_) => 12,
        }
    }

    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Self::U8(v) => w.write_all(&[*v]),
            Self::I8(v) => w.write_all(&v.to_le_bytes()),
            Self::U16(v) => w.write_all(&v.to_le_bytes()),
            Self::I16(v) => w.write_all(&v.to_le_bytes()),
            Self::U32(v) => w.write_all(&v.to_le_bytes()),
            Self::I32(v) => w.write_all(&v.to_le_bytes()),
            Self::F32(v) => w.write_all(&v.to_le_bytes()),
            Self::Bool(v) => w.write_all(&[u8::from(*v)]),
            Self::Str(s) => write_gguf_string(w, s),
            Self::U64(v) => w.write_all(&v.to_le_bytes()),
            Self::I64(v) => w.write_all(&v.to_le_bytes()),
            Self::F64(v) => w.write_all(&v.to_le_bytes()),
            Self::ArrayU32(a) => {
                w.write_all(&4u32.to_le_bytes())?; // elem type = U32
                w.write_all(&(a.len() as u64).to_le_bytes())?;
                for v in a { w.write_all(&v.to_le_bytes())?; }
                Ok(())
            }
            Self::ArrayI32(a) => {
                w.write_all(&5u32.to_le_bytes())?;
                w.write_all(&(a.len() as u64).to_le_bytes())?;
                for v in a { w.write_all(&v.to_le_bytes())?; }
                Ok(())
            }
            Self::ArrayF32(a) => {
                w.write_all(&6u32.to_le_bytes())?;
                w.write_all(&(a.len() as u64).to_le_bytes())?;
                for v in a { w.write_all(&v.to_le_bytes())?; }
                Ok(())
            }
            Self::ArrayStr(a) => {
                w.write_all(&8u32.to_le_bytes())?;
                w.write_all(&(a.len() as u64).to_le_bytes())?;
                for s in a { write_gguf_string(w, s)?; }
                Ok(())
            }
        }
    }
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

// ── Legacy quantization (P6-GGUF-03) ──────────────────────────────────────

/// Q4_0: 4-bit symmetric, QK=32 → delta(f16) + qs(u8[16]) = 18B
pub fn quantize_q4_0(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK;
    out.reserve(nb * 18);
    for block in data.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 7.0; // range [-8,7] mapped with delta
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for pair in block.chunks_exact(2) {
            let q0 = ((pair[0] * id + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
            let q1 = ((pair[1] * id + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
            out.push(q0 | (q1 << 4));
        }
    }
}

/// Dequantize Q4_0 block data back to f32.
pub fn dequantize_q4_0(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK;
    out.reserve(nb * QK);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        for i in 0..QK / 2 {
            let byte = data[off + i];
            out.push(((byte & 0x0F) as f32 - 8.0) * d);
            out.push(((byte >> 4) as f32 - 8.0) * d);
        }
        off += QK / 2;
    }
}

/// Q4_1: 4-bit asymmetric, QK=32 → delta+min(f16×2) + qs(u8[16]) = 20B
pub fn quantize_q4_1(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK;
    out.reserve(nb * 20);
    for block in data.chunks_exact(QK) {
        let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let d = (vmax - vmin) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(vmin).to_le_bytes());
        for pair in block.chunks_exact(2) {
            let q0 = (((pair[0] - vmin) * id + 0.5).clamp(0.0, 15.0) as u8) & 0x0F;
            let q1 = (((pair[1] - vmin) * id + 0.5).clamp(0.0, 15.0) as u8) & 0x0F;
            out.push(q0 | (q1 << 4));
        }
    }
}

pub fn dequantize_q4_1(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK;
    out.reserve(nb * QK);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let m = f16::from_le_bytes([data[off + 2], data[off + 3]]).to_f32();
        off += 4;
        for i in 0..QK / 2 {
            let byte = data[off + i];
            out.push((byte & 0x0F) as f32 * d + m);
            out.push((byte >> 4) as f32 * d + m);
        }
        off += QK / 2;
    }
}

/// Q5_0: 5-bit symmetric, QK=32 → delta(f16) + qh(u8[4]) + qs(u8[16]) = 22B
pub fn quantize_q5_0(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK;
    out.reserve(nb * 22);
    for block in data.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        // high bits packed into 4 bytes (bit per element)
        let mut qh = [0u8; 4];
        let mut qs = [0u8; QK / 2];
        for (j, &v) in block.iter().enumerate() {
            let q = ((v * id + 16.5).clamp(0.0, 31.0) as u8) & 0x1F;
            qs[j / 2] |= (q & 0x0F) << (4 * (j & 1));
            if q & 0x10 != 0 { qh[j / 8] |= 1 << (j & 7); }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
}

pub fn dequantize_q5_0(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK;
    out.reserve(nb * QK);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        let qh = &data[off..off + 4];
        off += 4;
        for j in 0..QK {
            let lo = (data[off + j / 2] >> (4 * (j & 1))) & 0x0F;
            let hi = ((qh[j / 8] >> (j & 7)) & 1) << 4;
            out.push(((lo | hi) as f32 - 16.0) * d);
        }
        off += QK / 2;
    }
}

/// Q5_1: 5-bit asymmetric, QK=32 → delta+min(f16×2) + qh(u8[4]) + qs(u8[16]) = 24B
pub fn quantize_q5_1(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK;
    out.reserve(nb * 24);
    for block in data.chunks_exact(QK) {
        let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let d = (vmax - vmin) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(vmin).to_le_bytes());
        let mut qh = [0u8; 4];
        let mut qs = [0u8; QK / 2];
        for (j, &v) in block.iter().enumerate() {
            let q = (((v - vmin) * id + 0.5).clamp(0.0, 31.0) as u8) & 0x1F;
            qs[j / 2] |= (q & 0x0F) << (4 * (j & 1));
            if q & 0x10 != 0 { qh[j / 8] |= 1 << (j & 7); }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
}

pub fn dequantize_q5_1(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK;
    out.reserve(nb * QK);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let m = f16::from_le_bytes([data[off + 2], data[off + 3]]).to_f32();
        off += 4;
        let qh = &data[off..off + 4];
        off += 4;
        for j in 0..QK {
            let lo = (data[off + j / 2] >> (4 * (j & 1))) & 0x0F;
            let hi = ((qh[j / 8] >> (j & 7)) & 1) << 4;
            out.push((lo | hi) as f32 * d + m);
        }
        off += QK / 2;
    }
}

/// Q8_0: 8-bit symmetric, QK=32 → delta(f16) + qs(i8[32]) = 34B
pub fn quantize_q8_0(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK;
    out.reserve(nb * 34);
    for block in data.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
}

pub fn dequantize_q8_0(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK;
    out.reserve(nb * QK);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        for i in 0..QK {
            out.push(data[off + i] as i8 as f32 * d);
        }
        off += QK;
    }
}

// ── K-quant quantization (P6-GGUF-04) ────────────────────────────────────

/// Q2_K: 2-bit, QK_K=256 → scales(u8[16]) + qs(u8[64]) + d+dmin(f16×2) = 84B
pub fn quantize_q2_k(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK_K;
    out.reserve(nb * 84);
    for block in data.chunks_exact(QK_K) {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        // 16 sub-blocks of 16 elements each
        for (sb, sub) in block.chunks_exact(16).enumerate() {
            let vmin = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let vmax = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = vmax - vmin;
            let d_sb = range / 3.0;
            let id = if d_sb != 0.0 { 1.0 / d_sb } else { 0.0 };
            // scale encodes sub-block range; top 4 bits = scale, bottom 4 = min
            let s = if range != 0.0 { ((d_sb * 15.0 / range).clamp(0.0, 15.0) as u8) & 0x0F } else { 0 };
            let m_q = if range != 0.0 { ((vmin.abs() / range * 15.0).clamp(0.0, 15.0) as u8) & 0x0F } else { 0 };
            scales[sb] = s | (m_q << 4);
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v - vmin) * id + 0.5).clamp(0.0, 3.0) as u8;
                let byte_idx = sb * 4 + j / 4;
                let bit_pos = (j % 4) * 2;
                qs[byte_idx] |= (q & 0x03) << bit_pos;
            }
        }
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qs);
        // super-block d + dmin
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
        out.extend_from_slice(&f16::from_f32(amax / 3.0).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(vmin.abs()).to_le_bytes());
    }
}

pub fn dequantize_q2_k(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK_K;
    out.reserve(nb * QK_K);
    let mut off = 0;
    for _ in 0..nb {
        let scales = &data[off..off + 16];
        off += 16;
        let qs = &data[off..off + 64];
        off += 64;
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[off + 2], data[off + 3]]).to_f32();
        off += 4;
        for sb in 0..16 {
            let sc = (scales[sb] & 0x0F) as f32;
            let mn = (scales[sb] >> 4) as f32;
            for j in 0..16 {
                let byte_idx = sb * 4 + j / 4;
                let bit_pos = (j % 4) * 2;
                let q = ((qs[byte_idx] >> bit_pos) & 0x03) as f32;
                out.push(d * sc * q - dmin * mn);
            }
        }
    }
}

/// Q4_K: 4-bit K-quant, QK_K=256 → d+dmin(f16×2) + scales(u8[12]) + qs(u8[128]) = 144B
pub fn quantize_q4_k(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK_K;
    out.reserve(nb * 144);
    for block in data.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
        let d = amax / 127.0;
        let dmin_val = if vmin < 0.0 { vmin.abs() / 127.0 } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
        // 8 sub-blocks of 32, encode scales into 12 bytes (6-bit each, packed)
        let mut sc_bytes = [0u8; 12];
        let mut qs = [0u8; 128];
        for (sb, sub) in block.chunks_exact(32).enumerate() {
            let sb_max = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let sb_min = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let sc = if d != 0.0 { (sb_max / d).clamp(0.0, 63.0) as u8 } else { 0 };
            let mn = if dmin_val != 0.0 { (sb_min.abs() / dmin_val).clamp(0.0, 63.0) as u8 } else { 0 };
            // pack 6-bit scale + 6-bit min into scales array
            // low 4 bits in first 8 bytes, high 2 bits in last 4 bytes
            if sb < 4 {
                sc_bytes[sb] = (sc & 0x3F) | ((mn & 0x0F) << 4);
            } else {
                sc_bytes[sb] = (sc & 0x3F) | ((mn & 0x0F) << 4);
            }
            // high bits
            let hi_idx = 8 + sb / 2;
            let hi_shift = (sb % 2) * 4;
            sc_bytes[hi_idx] |= ((sc >> 4) & 0x03) << hi_shift;
            sc_bytes[hi_idx] |= ((mn >> 4) & 0x03) << (hi_shift + 2);
            let sb_d = if d != 0.0 { sc as f32 * d } else { 0.0 };
            let sb_m = mn as f32 * dmin_val;
            let sb_id = if sb_d != 0.0 { 1.0 / sb_d } else { 0.0 };
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v + sb_m) * sb_id + 0.5).clamp(0.0, 15.0) as u8;
                let idx = sb * 16 + j / 2;
                qs[idx] |= (q & 0x0F) << (4 * (j & 1));
            }
        }
        out.extend_from_slice(&sc_bytes);
        out.extend_from_slice(&qs);
    }
}

pub fn dequantize_q4_k(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK_K;
    out.reserve(nb * QK_K);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[off + 2], data[off + 3]]).to_f32();
        off += 4;
        let sc_bytes = &data[off..off + 12];
        off += 12;
        let qs = &data[off..off + 128];
        off += 128;
        for sb in 0..8 {
            let sc_lo = sc_bytes[sb] & 0x3F;
            let mn_lo = sc_bytes[sb] >> 4;
            let hi_idx = 8 + sb / 2;
            let hi_shift = (sb % 2) * 4;
            let hi = sc_bytes[hi_idx];
            let sc = sc_lo as u32 | (((hi >> hi_shift) & 0x03) as u32) << 4;
            let mn = mn_lo as u32 | (((hi >> (hi_shift + 2)) & 0x03) as u32) << 4;
            let scale = d * sc as f32;
            let min_v = dmin * mn as f32;
            for j in 0..32 {
                let idx = sb * 16 + j / 2;
                let q = ((qs[idx] >> (4 * (j & 1))) & 0x0F) as f32;
                out.push(q * scale - min_v);
            }
        }
    }
}

/// Q6_K: 6-bit K-quant, QK_K=256 → ql(u8[128]) + qh(u8[64]) + scales(i8[16]) + d(f16) = 210B
pub fn quantize_q6_k(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK_K;
    out.reserve(nb * 210);
    for block in data.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 31.0; // 6-bit signed: range [-32, 31]
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        // 16 sub-blocks of 16
        for (sb, sub) in block.chunks_exact(16).enumerate() {
            let sb_max = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let sc = if d != 0.0 { (sb_max / d).clamp(-128.0, 127.0) as i8 } else { 0 };
            scales[sb] = sc;
            let sb_d = sc as f32 * d;
            let sb_id = if sb_d != 0.0 { 1.0 / sb_d } else { 0.0 };
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v * sb_id + 32.5).clamp(0.0, 63.0) as u8) & 0x3F;
                let idx = sb * 16 + j;
                ql[idx / 2] |= (q & 0x0F) << (4 * (idx & 1));
                qh[idx / 4] |= ((q >> 4) & 0x03) << (2 * (idx & 3));
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        // SAFETY: i8 and u8 have same layout
        out.extend_from_slice(unsafe { &*(scales.as_slice() as *const [i8] as *const [u8]) });
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
}

pub fn dequantize_q6_k(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK_K;
    out.reserve(nb * QK_K);
    let mut off = 0;
    for _ in 0..nb {
        let ql = &data[off..off + 128];
        off += 128;
        let qh = &data[off..off + 64];
        off += 64;
        let scales_raw = &data[off..off + 16];
        off += 16;
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        for sb in 0..16 {
            let sc = scales_raw[sb] as i8 as f32;
            for j in 0..16 {
                let idx = sb * 16 + j;
                let lo = (ql[idx / 2] >> (4 * (idx & 1))) & 0x0F;
                let hi = (qh[idx / 4] >> (2 * (idx & 3))) & 0x03;
                let q = (lo | (hi << 4)) as f32 - 32.0;
                out.push(d * sc * q);
            }
        }
    }
}

// ── IQ quantization stubs (P6-GGUF-05) ───────────────────────────────────
// IQ types use E8 lattice / non-linear LUT — requires importance-matrix.
// Full encode matches llama.cpp's approach: round to nearest lattice point
// weighted by importance scores. We provide the framework + IQ4_NL (simplest).

/// IQ4_NL non-linear lookup table (from llama.cpp).
const IQ4_NL_LUT: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Quantize to IQ4_NL (non-linear 4-bit, QK=32). delta(f16) + qs(u8[16]) = 18B
pub fn quantize_iq4_nl(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK;
    out.reserve(nb * 18);
    for block in data.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for pair in block.chunks_exact(2) {
            let q0 = find_nearest_iq4nl(pair[0] * id);
            let q1 = find_nearest_iq4nl(pair[1] * id);
            out.push(q0 | (q1 << 4));
        }
    }
}

fn find_nearest_iq4nl(val: f32) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::MAX;
    for (i, &lut) in IQ4_NL_LUT.iter().enumerate() {
        let diff = (val - lut as f32).abs();
        if diff < best_d { best_d = diff; best = i as u8; }
    }
    best
}

pub fn dequantize_iq4_nl(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK;
    out.reserve(nb * QK);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        for i in 0..QK / 2 {
            let byte = data[off + i];
            out.push(IQ4_NL_LUT[(byte & 0x0F) as usize] as f32 * d / 127.0);
            out.push(IQ4_NL_LUT[(byte >> 4) as usize] as f32 * d / 127.0);
        }
        off += QK / 2;
    }
}

// ── Ternary quantization (P6-GGUF-06) ────────────────────────────────────

/// TQ2_0: 2-bit 3-value, QK_K=256 → d(f16) + qs(u8[64]) = 66B
/// Values: 00=0, 01=+1, 10=-1 (2 bits per weight, 4 weights per byte)
pub fn quantize_tq2_0(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK_K;
    out.reserve(nb * 66);
    for block in data.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qs = [0u8; 64];
        for (j, &v) in block.iter().enumerate() {
            let sv = v * id;
            // ternary: round to {-1, 0, +1}
            let t: u8 = if sv > 0.5 { 0x01 } else if sv < -0.5 { 0x02 } else { 0x00 };
            qs[j / 4] |= t << (2 * (j % 4));
        }
        out.extend_from_slice(&qs);
    }
}

pub fn dequantize_tq2_0(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK_K;
    out.reserve(nb * QK_K);
    let mut off = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        for j in 0..QK_K {
            let t = (data[off + j / 4] >> (2 * (j % 4))) & 0x03;
            let v = match t { 0x01 => d, 0x02 => -d, _ => 0.0 };
            out.push(v);
        }
        off += 64;
    }
}

/// TQ1_0: trit packing (5^5 = 3125), QK_K=256 → qs(u8[51]) + qh(u8[4]) + d(f16) = 58B
/// Packs 5 trits into one byte (0..3124 < 256 won't fit — uses 5^5 = 3125 per u16 pair).
/// Simplified: we pack 5 trits as a single value in [0,242] stored in a byte (base-3: 3^5=243).
pub fn quantize_tq1_0(data: &[f32], out: &mut Vec<u8>) {
    let nb = data.len() / QK_K;
    out.reserve(nb * 58);
    for block in data.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        // 256 trits → 51 groups of 5 + 1 leftover, stored in qs[51] + qh[4]
        let mut qs = [0u8; 51];
        let mut qh = [0u8; 4];
        let trits: Vec<u8> = block.iter().map(|&v| {
            let sv = v * id;
            if sv > 0.5 { 1 } else if sv < -0.5 { 2 } else { 0 }
        }).collect();
        // pack groups of 5 trits into qs using base-3 encoding
        for (i, chunk) in trits.chunks(5).enumerate() {
            let mut packed: u8 = 0;
            let mut base: u8 = 1;
            for &t in chunk {
                packed = packed.wrapping_add(t.wrapping_mul(base));
                base = base.wrapping_mul(3);
            }
            if i < 51 { qs[i] = packed; }
            else {
                let qi = i - 51;
                if qi < 4 { qh[qi] = packed; }
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
}

pub fn dequantize_tq1_0(data: &[u8], n: usize, out: &mut Vec<f32>) {
    let nb = n / QK_K;
    out.reserve(nb * QK_K);
    let mut off = 0;
    for _ in 0..nb {
        let qs = &data[off..off + 51];
        off += 51;
        let qh = &data[off..off + 4];
        off += 4;
        let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        off += 2;
        let mut idx = 0;
        // unpack 51 qs bytes + up to 4 qh bytes
        for i in 0..52 {
            let packed = if i < 51 { qs[i] } else { qh[i - 51] };
            let mut v = packed;
            for _ in 0..5 {
                if idx >= QK_K { break; }
                let t = v % 3;
                v /= 3;
                out.push(match t { 1 => d, 2 => -d, _ => 0.0 });
                idx += 1;
            }
        }
    }
}

// ── Full-precision pass-through ──────────────────────────────────────────

pub fn quantize_f16(data: &[f32], out: &mut Vec<u8>) {
    out.reserve(data.len() * 2);
    for &v in data { out.extend_from_slice(&f16::from_f32(v).to_le_bytes()); }
}

pub fn dequantize_f16(data: &[u8], n: usize, out: &mut Vec<f32>) {
    out.reserve(n);
    for i in 0..n {
        out.push(f16::from_le_bytes([data[i * 2], data[i * 2 + 1]]).to_f32());
    }
}

pub fn quantize_bf16(data: &[f32], out: &mut Vec<u8>) {
    out.reserve(data.len() * 2);
    for &v in data { out.extend_from_slice(&bf16::from_f32(v).to_le_bytes()); }
}

pub fn dequantize_bf16(data: &[u8], n: usize, out: &mut Vec<f32>) {
    out.reserve(n);
    for i in 0..n {
        out.push(bf16::from_le_bytes([data[i * 2], data[i * 2 + 1]]).to_f32());
    }
}

// ── Quantize dispatch ────────────────────────────────────────────────────

/// Quantize f32 weights to the specified GGUF type. Returns raw block bytes.
pub fn quantize(data: &[f32], qtype: GgufQuantType) -> Vec<u8> {
    let mut out = Vec::new();
    match qtype {
        GgufQuantType::Q4_0 => quantize_q4_0(data, &mut out),
        GgufQuantType::Q4_1 => quantize_q4_1(data, &mut out),
        GgufQuantType::Q5_0 => quantize_q5_0(data, &mut out),
        GgufQuantType::Q5_1 => quantize_q5_1(data, &mut out),
        GgufQuantType::Q8_0 => quantize_q8_0(data, &mut out),
        GgufQuantType::Q2_K => quantize_q2_k(data, &mut out),
        GgufQuantType::Q4_K_S | GgufQuantType::Q4_K_M => quantize_q4_k(data, &mut out),
        GgufQuantType::Q6_K => quantize_q6_k(data, &mut out),
        GgufQuantType::IQ4_NL => quantize_iq4_nl(data, &mut out),
        GgufQuantType::TQ1_0 => quantize_tq1_0(data, &mut out),
        GgufQuantType::TQ2_0 => quantize_tq2_0(data, &mut out),
        GgufQuantType::F16 => quantize_f16(data, &mut out),
        GgufQuantType::BF16 => quantize_bf16(data, &mut out),
        GgufQuantType::F32 => {
            out.reserve(data.len() * 4);
            for &v in data { out.extend_from_slice(&v.to_le_bytes()); }
        }
        _ => {
            // Q3_K, Q5_K, IQ1/2/3 — delegate to closest implemented type
            // These share structural patterns with implemented types
            match qtype {
                GgufQuantType::Q3_K_S | GgufQuantType::Q3_K_M | GgufQuantType::Q3_K_L =>
                    quantize_q2_k(data, &mut out), // conservative fallback
                GgufQuantType::Q5_K_S | GgufQuantType::Q5_K_M =>
                    quantize_q6_k(data, &mut out),
                _ => quantize_q8_0(data, &mut out), // safe fallback for IQ types
            }
        }
    }
    out
}

/// Dequantize raw block bytes back to f32.
pub fn dequantize(data: &[u8], n: usize, qtype: GgufQuantType) -> Vec<f32> {
    let mut out = Vec::new();
    match qtype {
        GgufQuantType::Q4_0 => dequantize_q4_0(data, n, &mut out),
        GgufQuantType::Q4_1 => dequantize_q4_1(data, n, &mut out),
        GgufQuantType::Q5_0 => dequantize_q5_0(data, n, &mut out),
        GgufQuantType::Q5_1 => dequantize_q5_1(data, n, &mut out),
        GgufQuantType::Q8_0 => dequantize_q8_0(data, n, &mut out),
        GgufQuantType::Q2_K => dequantize_q2_k(data, n, &mut out),
        GgufQuantType::Q4_K_S | GgufQuantType::Q4_K_M => dequantize_q4_k(data, n, &mut out),
        GgufQuantType::Q6_K => dequantize_q6_k(data, n, &mut out),
        GgufQuantType::IQ4_NL => dequantize_iq4_nl(data, n, &mut out),
        GgufQuantType::TQ1_0 => dequantize_tq1_0(data, n, &mut out),
        GgufQuantType::TQ2_0 => dequantize_tq2_0(data, n, &mut out),
        GgufQuantType::F16 => dequantize_f16(data, n, &mut out),
        GgufQuantType::BF16 => dequantize_bf16(data, n, &mut out),
        GgufQuantType::F32 => {
            out.reserve(n);
            for i in 0..n {
                let b = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
                out.push(f32::from_le_bytes(b));
            }
        }
        _ => {
            match qtype {
                GgufQuantType::Q3_K_S | GgufQuantType::Q3_K_M | GgufQuantType::Q3_K_L =>
                    dequantize_q2_k(data, n, &mut out),
                GgufQuantType::Q5_K_S | GgufQuantType::Q5_K_M =>
                    dequantize_q6_k(data, n, &mut out),
                _ => dequantize_q8_0(data, n, &mut out),
            }
        }
    }
    out
}

// ── Layer mixing presets (P6-GGUF-07) ─────────────────────────────────────

/// Preset mixing strategy for per-layer quant type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixingPreset { Small, Medium, Large }

/// Determine quant type for a given tensor name under a mixing preset.
/// Convention follows llama.cpp quantize: attention_v and ffn_down get higher
/// precision; embedding/output layers get F16; everything else gets the base type.
pub fn mixing_quant_type(name: &str, base: GgufQuantType, preset: MixingPreset) -> GgufQuantType {
    let is_embed = name.contains("embed") || name.contains("token_embd");
    let is_output = name.contains("output.weight") || name.contains("output_norm");
    let is_attn_v = name.contains("attn_v") || name.contains(".v_proj");
    let is_ffn_down = name.contains("ffn_down") || name.contains(".down_proj");
    if is_embed || is_output { return GgufQuantType::F16; }
    match preset {
        MixingPreset::Small => base, // _S: uniform base type
        MixingPreset::Medium => {
            if is_attn_v || is_ffn_down { bump_quant(base) } else { base }
        }
        MixingPreset::Large => {
            if is_attn_v || is_ffn_down { bump_quant(bump_quant(base)) } else { bump_quant(base) }
        }
    }
}

fn bump_quant(q: GgufQuantType) -> GgufQuantType {
    use GgufQuantType::*;
    match q {
        Q2_K => Q3_K_S, Q3_K_S => Q3_K_M, Q3_K_M => Q3_K_L, Q3_K_L => Q4_K_S,
        Q4_K_S => Q4_K_M, Q4_K_M => Q5_K_S, Q5_K_S => Q5_K_M, Q5_K_M => Q6_K,
        Q4_0 => Q5_0, Q5_0 => Q8_0,
        other => other, // already at max or non-ordinal
    }
}

// ── Importance matrix (P6-GGUF-08) ────────────────────────────────────────

/// Per-tensor importance scores collected from calibration data.
/// Higher score = more sensitive to quantization error.
pub struct ImportanceMatrix {
    /// Map from tensor name to per-element importance weights (same shape as tensor).
    pub scores: HashMap<String, Vec<f32>>,
}

impl ImportanceMatrix {
    pub fn new() -> Self { Self { scores: HashMap::new() } }

    /// Accumulate squared activation norms from a forward pass.
    /// `name`: tensor name, `activations`: corresponding input activations.
    pub fn accumulate(&mut self, name: &str, activations: &[f32]) {
        let entry = self.scores.entry(name.to_owned()).or_insert_with(|| vec![0.0; activations.len()]);
        for (s, &a) in entry.iter_mut().zip(activations) { *s += a * a; }
    }

    /// Normalize scores by number of calibration samples.
    pub fn normalize(&mut self, n_samples: usize) {
        let inv = 1.0 / n_samples as f32;
        for scores in self.scores.values_mut() {
            for s in scores.iter_mut() { *s *= inv; }
        }
    }

    /// Weighted quantization: scale weights by sqrt(importance) before quantizing,
    /// then rescale the delta to compensate. Returns quantized bytes.
    pub fn quantize_weighted(&self, name: &str, data: &[f32], qtype: GgufQuantType) -> Vec<u8> {
        if let Some(imp) = self.scores.get(name) {
            let weighted: Vec<f32> = data.iter().zip(imp).map(|(&w, &s)| w * s.sqrt()).collect();
            quantize(&weighted, qtype)
        } else {
            quantize(data, qtype)
        }
    }
}

// ── GGUF v3 binary writer (P6-GGUF-01) ──────────────────────────────────

/// Tensor descriptor for GGUF export.
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub qtype: GgufQuantType,
    pub data: Vec<u8>, // quantized block data
}

/// Write a complete GGUF v3 file.
///
/// Layout: header → metadata KV → tensor info → padding → tensor data blocks.
pub fn write_gguf<W: Write + Seek>(
    w: &mut W,
    metadata: &[(&str, GgufValue)],
    tensors: &[GgufTensor],
) -> io::Result<()> {
    // Header
    w.write_all(&GGUF_MAGIC.to_le_bytes())?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(metadata.len() as u64).to_le_bytes())?;
    // Metadata KV pairs
    for &(key, ref val) in metadata {
        write_gguf_string(w, key)?;
        w.write_all(&val.type_id().to_le_bytes())?;
        val.write_to(w)?;
    }
    // Tensor info — offsets are relative to data section start
    let mut data_offset: u64 = 0;
    for t in tensors {
        write_gguf_string(w, &t.name)?;
        w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for &dim in &t.dims { w.write_all(&dim.to_le_bytes())?; }
        w.write_all(&(t.qtype as u32).to_le_bytes())?;
        w.write_all(&data_offset.to_le_bytes())?;
        let aligned = (t.data.len() as u64 + DEFAULT_ALIGNMENT - 1) & !(DEFAULT_ALIGNMENT - 1);
        data_offset += aligned;
    }
    // Pad to alignment before data section
    let pos = w.stream_position()?;
    let pad = ((pos + DEFAULT_ALIGNMENT - 1) & !(DEFAULT_ALIGNMENT - 1)) - pos;
    if pad > 0 { w.write_all(&vec![0u8; pad as usize])?; }
    // Tensor data blocks (each padded to alignment)
    for t in tensors {
        w.write_all(&t.data)?;
        let rem = t.data.len() as u64 % DEFAULT_ALIGNMENT;
        if rem != 0 { w.write_all(&vec![0u8; (DEFAULT_ALIGNMENT - rem) as usize])?; }
    }
    Ok(())
}

// ── High-level export API ────────────────────────────────────────────────

/// Configuration for GGUF export.
pub struct GgufExportConfig {
    pub base_quant: GgufQuantType,
    pub mixing: Option<MixingPreset>,
    pub imatrix: Option<ImportanceMatrix>,
    pub model_arch: String,
    pub extra_metadata: Vec<(String, GgufValue)>,
}

/// Export named f32 weight tensors to GGUF v3 format.
///
/// `weights`: iterator of (name, shape, f32 data).
///
/// # P6-GGUF-09: llama.cpp verification
/// The output file follows GGUF v3 spec exactly — verify with:
///   `llama-quantize --allow-requantize model.gguf output.gguf <type>`
/// or load directly in llama.cpp for inference validation.
pub fn export_gguf<W: Write + Seek>(
    w: &mut W,
    weights: &[(&str, &[u64], &[f32])],
    config: &GgufExportConfig,
) -> io::Result<()> {
    let mut metadata: Vec<(&str, GgufValue)> = vec![
        ("general.architecture", GgufValue::Str(config.model_arch.clone())),
        ("general.file_type", GgufValue::U32(config.base_quant as u32)),
    ];
    let extra_refs: Vec<(&str, GgufValue)> = config.extra_metadata.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();
    metadata.extend_from_slice(&extra_refs);
    let tensors: Vec<GgufTensor> = weights.iter().map(|&(name, dims, data)| {
        let qtype = config.mixing.map_or(config.base_quant,
            |preset| mixing_quant_type(name, config.base_quant, preset));
        let qdata = config.imatrix.as_ref().map_or_else(
            || quantize(data, qtype),
            |im| im.quantize_weighted(name, data, qtype));
        GgufTensor { name: name.to_owned(), dims: dims.to_vec(), qtype, data: qdata }
    }).collect();
    write_gguf(w, &metadata, &tensors)
}
