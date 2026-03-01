//! Legacy quantization: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`.

use crate::{Error, Result};
use half::f16;

const QK: usize = 32;
const BQ4_0: usize = 18; // f16 delta + 16 nibble bytes
const BQ4_1: usize = 20; // f16 delta + f16 min + 16 nibble bytes
const BQ5_0: usize = 22; // f16 delta + 4 high-bit mask + 16 nibble bytes
const BQ5_1: usize = 24; // f16 delta + f16 min + 4 high-bit mask + 16 nibble bytes
const BQ8_0: usize = 34; // f16 delta + 32 i8
const BQ8_1: usize = 36; // f16 delta + f16 sum + 32 i8

fn check_len(name: &str, len: usize, divisor: usize) -> Result<()> {
    if !len.is_multiple_of(divisor) {
        return Err(Error::Quant(format!(
            "{name}: len {len} not divisible by {divisor}"
        )));
    }
    Ok(())
}

/// Quantize f32 slice to `Q4_0` format (18 bytes / 32 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 32.
pub fn quantize_q4_0(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q4_0", data.len(), QK)?;
    let nb = data.len() / QK;
    let mut out = Vec::with_capacity(nb * BQ4_0);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 7.0;
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for pair in blk.chunks_exact(2) {
            let q0 = ((pair[0] * id) + 8.5).clamp(0.0, 15.0) as u8;
            let q1 = ((pair[1] * id) + 8.5).clamp(0.0, 15.0) as u8;
            out.push(q0 | (q1 << 4));
        }
    }
    Ok(out)
}

/// Dequantize `Q4_0` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 18.
pub fn dequantize_q4_0(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q4_0 bytes", data.len(), BQ4_0)?;
    let nb = data.len() / BQ4_0;
    let mut out = Vec::with_capacity(nb * QK);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        p += 2;
        for j in 0..16 {
            let b = data[p + j];
            out.push((f32::from(b & 0x0F) - 8.0) * d);
            out.push((f32::from((b >> 4) & 0x0F) - 8.0) * d);
        }
        p += 16;
    }
    Ok(out)
}

/// Quantize f32 slice to `Q4_1` (asymmetric 4-bit with min, 20 bytes / 32 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 32.
pub fn quantize_q4_1(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q4_1", data.len(), QK)?;
    let nb = data.len() / QK;
    let mut out = Vec::with_capacity(nb * BQ4_1);
    for blk in data.chunks_exact(QK) {
        let vmin = blk.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = blk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let d = (vmax - vmin) / 15.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(vmin).to_le_bytes());
        for pair in blk.chunks_exact(2) {
            let q0 = (((pair[0] - vmin) * id + 0.5).clamp(0.0, 15.0) as u8) & 0x0F;
            let q1 = (((pair[1] - vmin) * id + 0.5).clamp(0.0, 15.0) as u8) & 0x0F;
            out.push(q0 | (q1 << 4));
        }
    }
    Ok(out)
}

/// Dequantize `Q4_1` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 20.
pub fn dequantize_q4_1(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q4_1 bytes", data.len(), BQ4_1)?;
    let nb = data.len() / BQ4_1;
    let mut out = Vec::with_capacity(nb * QK);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        let m = f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
        p += 4;
        for j in 0..16 {
            let b = data[p + j];
            out.push(f32::from(b & 0x0F) * d + m);
            out.push(f32::from(b >> 4) * d + m);
        }
        p += 16;
    }
    Ok(out)
}

/// Quantize f32 slice to `Q5_0` (symmetric 5-bit, 22 bytes / 32 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 32.
pub fn quantize_q5_0(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q5_0", data.len(), QK)?;
    let nb = data.len() / QK;
    let mut out = Vec::with_capacity(nb * BQ5_0);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 15.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qh = [0u8; 4];
        let mut qs = [0u8; 16];
        for (j, &v) in blk.iter().enumerate() {
            let q = ((v * id + 16.5).clamp(0.0, 31.0) as u8) & 0x1F;
            qs[j / 2] |= (q & 0x0F) << (4 * (j & 1));
            if q & 0x10 != 0 {
                qh[j / 8] |= 1 << (j & 7);
            }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Dequantize `Q5_0` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 22.
pub fn dequantize_q5_0(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q5_0 bytes", data.len(), BQ5_0)?;
    let nb = data.len() / BQ5_0;
    let mut out = Vec::with_capacity(nb * QK);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        p += 2;
        let qh = &data[p..p + 4];
        p += 4;
        for j in 0..QK {
            let lo = (data[p + j / 2] >> (4 * (j & 1))) & 0x0F;
            let hi = ((qh[j / 8] >> (j & 7)) & 1) << 4;
            out.push((f32::from(lo | hi) - 16.0) * d);
        }
        p += 16;
    }
    Ok(out)
}

/// Quantize f32 slice to `Q5_1` (asymmetric 5-bit with min, 24 bytes / 32 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 32.
pub fn quantize_q5_1(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q5_1", data.len(), QK)?;
    let nb = data.len() / QK;
    let mut out = Vec::with_capacity(nb * BQ5_1);
    for blk in data.chunks_exact(QK) {
        let vmin = blk.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = blk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let d = (vmax - vmin) / 31.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(vmin).to_le_bytes());
        let mut qh = [0u8; 4];
        let mut qs = [0u8; 16];
        for (j, &v) in blk.iter().enumerate() {
            let q = (((v - vmin) * id + 0.5).clamp(0.0, 31.0) as u8) & 0x1F;
            qs[j / 2] |= (q & 0x0F) << (4 * (j & 1));
            if q & 0x10 != 0 {
                qh[j / 8] |= 1 << (j & 7);
            }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Dequantize `Q5_1` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 24.
pub fn dequantize_q5_1(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q5_1 bytes", data.len(), BQ5_1)?;
    let nb = data.len() / BQ5_1;
    let mut out = Vec::with_capacity(nb * QK);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        let m = f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
        p += 4;
        let qh = &data[p..p + 4];
        p += 4;
        for j in 0..QK {
            let lo = (data[p + j / 2] >> (4 * (j & 1))) & 0x0F;
            let hi = ((qh[j / 8] >> (j & 7)) & 1) << 4;
            out.push(f32::from(lo | hi) * d + m);
        }
        p += 16;
    }
    Ok(out)
}

/// Quantize f32 slice to `Q8_0` (symmetric 8-bit, 34 bytes / 32 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 32.
pub fn quantize_q8_0(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q8_0", data.len(), QK)?;
    let nb = data.len() / QK;
    let mut out = Vec::with_capacity(nb * BQ8_0);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for &v in blk {
            out.push((v * id).round().clamp(-128.0, 127.0) as i8 as u8);
        }
    }
    Ok(out)
}

/// Dequantize `Q8_0` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 34.
pub fn dequantize_q8_0(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q8_0 bytes", data.len(), BQ8_0)?;
    let nb = data.len() / BQ8_0;
    let mut out = Vec::with_capacity(nb * QK);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        p += 2;
        for j in 0..QK {
            out.push(f32::from(data[p + j] as i8) * d);
        }
        p += QK;
    }
    Ok(out)
}

/// Quantize f32 slice to `Q8_1` (8-bit with sum, 36 bytes / 32 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 32.
pub fn quantize_q8_1(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q8_1", data.len(), QK)?;
    let nb = data.len() / QK;
    let mut out = Vec::with_capacity(nb * BQ8_1);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut sum = 0.0f32;
        let mut qs = [0i8; 32];
        for (j, &v) in blk.iter().enumerate() {
            let q = (v * id).round().clamp(-128.0, 127.0);
            qs[j] = q as i8;
            sum += q;
        }
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(sum).to_le_bytes());
        // SAFETY: i8 and u8 have identical layout
        let qs_u8: &[u8; 32] = unsafe { &*std::ptr::from_ref(&qs).cast() };
        out.extend_from_slice(qs_u8);
    }
    Ok(out)
}

/// Dequantize `Q8_1` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 36.
pub fn dequantize_q8_1(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q8_1 bytes", data.len(), BQ8_1)?;
    let nb = data.len() / BQ8_1;
    let mut out = Vec::with_capacity(nb * QK);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        p += 4; // skip d(2) + sum(2)
        for j in 0..QK {
            out.push(f32::from(data[p + j] as i8) * d);
        }
        p += QK;
    }
    Ok(out)
}
