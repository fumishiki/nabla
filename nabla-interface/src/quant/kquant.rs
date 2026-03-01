//! K-quant quantization: `Q2_K`, `Q3_K`, `Q4_K` (S/M), `Q5_K` (S/M), `Q6_K`.
//! `Q*_K_S` and `Q*_K_M` share the same block format; S/M difference is per-layer mixing only.

use crate::{Error, Result};
use half::f16;

const QK_K: usize = 256;
const BQ2_K: usize = 84; // scales(16) + qs(64) + d+dmin(4)
const BQ3_K: usize = 110; // hmask(32) + qs(64) + scales(12) + d(2)
const BQ4_K: usize = 144; // d+dmin(4) + scales(12) + qs(128)
const BQ5_K: usize = 176; // d+dmin(4) + scales(12) + qh(32) + qs(128)
const BQ6_K: usize = 210; // ql(128) + qh(64) + scales(16) + d(2)

fn check_len(name: &str, len: usize, divisor: usize) -> Result<()> {
    if !len.is_multiple_of(divisor) {
        return Err(Error::Quant(format!(
            "{name}: len {len} not divisible by {divisor}"
        )));
    }
    Ok(())
}

/// Quantize f32 slice to `Q2_K` (2-bit K-quant, 84 bytes / 256 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 256.
pub fn quantize_q2_k(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q2_K", data.len(), QK_K)?;
    let nb = data.len() / QK_K;
    let mut out = Vec::with_capacity(nb * BQ2_K);
    for blk in data.chunks_exact(QK_K) {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (sb, sub) in blk.chunks_exact(16).enumerate() {
            let vmin = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let vmax = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = vmax - vmin;
            let inv_d = if range == 0.0 { 0.0 } else { 3.0 / range };
            let sc = if range == 0.0 { 0 } else { 15u8 };
            let mq = if range == 0.0 {
                0
            } else {
                ((vmin.abs() / range * 15.0).clamp(0.0, 15.0) as u8) & 0x0F
            };
            scales[sb] = sc | (mq << 4);
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v - vmin) * inv_d + 0.5).clamp(0.0, 3.0) as u8;
                qs[sb * 4 + j / 4] |= (q & 0x03) << ((j % 4) * 2);
            }
        }
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let vmin = blk.iter().copied().fold(f32::INFINITY, f32::min);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&f16::from_f32(amax / 3.0).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(vmin.abs()).to_le_bytes());
    }
    Ok(out)
}

/// Dequantize `Q2_K` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 84.
pub fn dequantize_q2_k(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q2_K bytes", data.len(), BQ2_K)?;
    let nb = data.len() / BQ2_K;
    let mut out = Vec::with_capacity(nb * QK_K);
    let mut p = 0;
    for _ in 0..nb {
        let scales = &data[p..p + 16];
        p += 16;
        let qs = &data[p..p + 64];
        p += 64;
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
        p += 4;
        for (sb, sc_byte) in scales.iter().enumerate() {
            let sc = f32::from(sc_byte & 0x0F);
            let mn = f32::from(sc_byte >> 4);
            for j in 0..16 {
                let q = f32::from((qs[sb * 4 + j / 4] >> ((j % 4) * 2)) & 0x03);
                out.push(d * sc * q - dmin * mn);
            }
        }
    }
    Ok(out)
}

/// Quantize f32 slice to `Q3_K` (3-bit K-quant, 110 bytes / 256 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 256.
pub fn quantize_q3_k(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q3_K", data.len(), QK_K)?;
    let nb = data.len() / QK_K;
    let mut out = Vec::with_capacity(nb * BQ3_K);
    for blk in data.chunks_exact(QK_K) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 3.0;
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let mut sc_raw = [0i8; 16];
        for (sb, sub) in blk.chunks_exact(16).enumerate() {
            let sb_max = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let sc = if d == 0.0 {
                0
            } else {
                (sb_max / d).clamp(-128.0, 127.0) as i8
            };
            sc_raw[sb] = sc;
            let sb_d = f32::from(sc) * d;
            let inv_sb_d = if sb_d == 0.0 { 0.0 } else { 1.0 / sb_d };
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v * inv_sb_d + 4.5).clamp(0.0, 7.0) as u8) & 0x07;
                let idx = sb * 16 + j;
                qs[idx / 4] |= (q & 0x03) << ((idx % 4) * 2);
                if q & 0x04 != 0 {
                    hmask[idx / 8] |= 1 << (idx & 7);
                }
            }
        }
        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        // Pack 16 6-bit scales into 12 bytes
        let mut sp = [0u8; 12];
        for i in 0..8 {
            sp[i] = (sc_raw[i] as u8) & 0x3F;
        }
        for i in 8..16 {
            let s = (sc_raw[i] as u8) & 0x3F;
            sp[i - 8] |= (s & 0x03) << 6;
            if i < 12 {
                sp[4 + (i - 8)] = s >> 2;
            }
        }
        out.extend_from_slice(&sp);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    Ok(out)
}

/// Dequantize `Q3_K` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 110.
pub fn dequantize_q3_k(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q3_K bytes", data.len(), BQ3_K)?;
    let nb = data.len() / BQ3_K;
    let mut out = Vec::with_capacity(nb * QK_K);
    let mut p = 0;
    for _ in 0..nb {
        let hmask = &data[p..p + 32];
        p += 32;
        let qs = &data[p..p + 64];
        p += 64;
        let sp = &data[p..p + 12];
        p += 12;
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        p += 2;
        let mut sc_vals = [0i8; 16];
        for i in 0..8 {
            sc_vals[i] = (sp[i] & 0x3F) as i8;
        }
        for i in 8..16 {
            let lo = (sp[i - 8] >> 6) & 0x03;
            let hi = if i < 12 {
                (sp[4 + (i - 8)] & 0x0F) << 2
            } else {
                0
            };
            sc_vals[i] = (lo | hi) as i8;
        }
        for (sb, &sc) in sc_vals.iter().enumerate() {
            let s = f32::from(sc);
            for j in 0..16 {
                let idx = sb * 16 + j;
                let lo = (qs[idx / 4] >> ((idx % 4) * 2)) & 0x03;
                let hi = ((hmask[idx / 8] >> (idx & 7)) & 1) << 2;
                let q = f32::from(lo | hi) - 4.0;
                out.push(d * s * q);
            }
        }
    }
    Ok(out)
}

/// Quantize f32 slice to `Q4_K` (shared by `Q4_K_S` and `Q4_K_M`, 144 bytes / 256 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 256.
pub fn quantize_q4_k(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q4_K", data.len(), QK_K)?;
    let nb = data.len() / QK_K;
    let mut out = Vec::with_capacity(nb * BQ4_K);
    for blk in data.chunks_exact(QK_K) {
        let mut s_scales = [0.0f32; 8];
        let mut s_mins = [0.0f32; 8];
        for (si, sub) in blk.chunks_exact(32).enumerate() {
            let fmin = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let fmax = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            s_mins[si] = (-fmin).max(0.0);
            s_scales[si] = fmax.max(0.0) + s_mins[si];
        }
        let max_sc = s_scales.iter().copied().fold(0.0f32, f32::max);
        let max_mn = s_mins.iter().copied().fold(0.0f32, f32::max);
        let inv_sc = if max_sc > 0.0 { 63.0 / max_sc } else { 0.0 };
        let inv_mn = if max_mn > 0.0 { 63.0 / max_mn } else { 0.0 };
        let scale_d = f16::from_f32(max_sc / 63.0);
        let min_d = f16::from_f32(max_mn / 63.0);
        out.extend_from_slice(&scale_d.to_le_bytes());
        out.extend_from_slice(&min_d.to_le_bytes());
        let mut qs6 = [0u8; 8];
        let mut qm6 = [0u8; 8];
        for i in 0..8 {
            qs6[i] = (s_scales[i] * inv_sc + 0.5).min(63.0) as u8;
            qm6[i] = (s_mins[i] * inv_mn + 0.5).min(63.0) as u8;
        }
        let mut sp = [0u8; 12];
        for i in 0..4 {
            sp[i] = (qs6[2 * i] & 0xF) | ((qs6[2 * i + 1] & 0xF) << 4);
            sp[4 + i] = (qm6[2 * i] & 0xF) | ((qm6[2 * i + 1] & 0xF) << 4);
            sp[8 + i] = ((qs6[2 * i] >> 4) & 3)
                | (((qs6[2 * i + 1] >> 4) & 3) << 2)
                | (((qm6[2 * i] >> 4) & 3) << 4)
                | (((qm6[2 * i + 1] >> 4) & 3) << 6);
        }
        out.extend_from_slice(&sp);
        let scale_f = scale_d.to_f32();
        let min_f = min_d.to_f32();
        let mut packed_qs = [0u8; 128];
        for (si, sub) in blk.chunks_exact(32).enumerate() {
            let sc = f32::from(qs6[si]) * scale_f;
            let mn = f32::from(qm6[si]) * min_f;
            let inv = if sc > 0.0 { 15.0 / sc } else { 0.0 };
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v + mn) * inv + 0.5).clamp(0.0, 15.0) as u8;
                let idx = si * 16 + j / 2;
                if j % 2 == 0 {
                    packed_qs[idx] = q;
                } else {
                    packed_qs[idx] |= q << 4;
                }
            }
        }
        out.extend_from_slice(&packed_qs);
    }
    Ok(out)
}

/// Dequantize `Q4_K` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 144.
pub fn dequantize_q4_k(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q4_K bytes", data.len(), BQ4_K)?;
    let nb = data.len() / BQ4_K;
    let mut out = Vec::with_capacity(nb * QK_K);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        let dm = f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
        p += 4;
        let sr = &data[p..p + 12];
        p += 12;
        let qs = &data[p..p + 128];
        p += 128;
        let mut qs6 = [0u8; 8];
        let mut qm6 = [0u8; 8];
        for i in 0..4 {
            qs6[2 * i] = (sr[i] & 0xF) | ((sr[8 + i] & 3) << 4);
            qs6[2 * i + 1] = (sr[i] >> 4) | (((sr[8 + i] >> 2) & 3) << 4);
            qm6[2 * i] = (sr[4 + i] & 0xF) | (((sr[8 + i] >> 4) & 3) << 4);
            qm6[2 * i + 1] = (sr[4 + i] >> 4) | (((sr[8 + i] >> 6) & 3) << 4);
        }
        for si in 0..8 {
            let sc = f32::from(qs6[si]) * d;
            let mn = f32::from(qm6[si]) * dm;
            for j in 0..32 {
                let idx = si * 16 + j / 2;
                let q = if j % 2 == 0 {
                    qs[idx] & 0xF
                } else {
                    qs[idx] >> 4
                };
                out.push(f32::from(q) * sc / 15.0 - mn);
            }
        }
    }
    Ok(out)
}

/// Quantize f32 slice to `Q5_K` (shared by `Q5_K_S` and `Q5_K_M`, 176 bytes / 256 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 256.
pub fn quantize_q5_k(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q5_K", data.len(), QK_K)?;
    let nb = data.len() / QK_K;
    let mut out = Vec::with_capacity(nb * BQ5_K);
    for blk in data.chunks_exact(QK_K) {
        let mut s_scales = [0.0f32; 8];
        let mut s_mins = [0.0f32; 8];
        for (si, sub) in blk.chunks_exact(32).enumerate() {
            let fmin = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let fmax = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            s_mins[si] = (-fmin).max(0.0);
            s_scales[si] = fmax.max(0.0) + s_mins[si];
        }
        let max_sc = s_scales.iter().copied().fold(0.0f32, f32::max);
        let max_mn = s_mins.iter().copied().fold(0.0f32, f32::max);
        let inv_sc = if max_sc > 0.0 { 63.0 / max_sc } else { 0.0 };
        let inv_mn = if max_mn > 0.0 { 63.0 / max_mn } else { 0.0 };
        let scale_d = f16::from_f32(max_sc / 63.0);
        let min_d = f16::from_f32(max_mn / 63.0);
        out.extend_from_slice(&scale_d.to_le_bytes());
        out.extend_from_slice(&min_d.to_le_bytes());
        let mut qs6 = [0u8; 8];
        let mut qm6 = [0u8; 8];
        for i in 0..8 {
            qs6[i] = (s_scales[i] * inv_sc + 0.5).min(63.0) as u8;
            qm6[i] = (s_mins[i] * inv_mn + 0.5).min(63.0) as u8;
        }
        let mut sp = [0u8; 12];
        for i in 0..4 {
            sp[i] = (qs6[2 * i] & 0xF) | ((qs6[2 * i + 1] & 0xF) << 4);
            sp[4 + i] = (qm6[2 * i] & 0xF) | ((qm6[2 * i + 1] & 0xF) << 4);
            sp[8 + i] = ((qs6[2 * i] >> 4) & 3)
                | (((qs6[2 * i + 1] >> 4) & 3) << 2)
                | (((qm6[2 * i] >> 4) & 3) << 4)
                | (((qm6[2 * i + 1] >> 4) & 3) << 6);
        }
        out.extend_from_slice(&sp);
        let scale_f = scale_d.to_f32();
        let min_f = min_d.to_f32();
        let mut qh = [0u8; 32];
        let mut packed_qs = [0u8; 128];
        for (si, sub) in blk.chunks_exact(32).enumerate() {
            let sc = f32::from(qs6[si]) * scale_f;
            let mn = f32::from(qm6[si]) * min_f;
            let inv = if sc > 0.0 { 31.0 / sc } else { 0.0 };
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v + mn) * inv + 0.5).clamp(0.0, 31.0) as u8;
                let idx = si * 32 + j;
                packed_qs[idx / 2] |= (q & 0x0F) << (4 * (idx & 1));
                if q & 0x10 != 0 {
                    qh[idx / 8] |= 1 << (idx & 7);
                }
            }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&packed_qs);
    }
    Ok(out)
}

/// Dequantize `Q5_K` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 176.
pub fn dequantize_q5_k(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q5_K bytes", data.len(), BQ5_K)?;
    let nb = data.len() / BQ5_K;
    let mut out = Vec::with_capacity(nb * QK_K);
    let mut p = 0;
    for _ in 0..nb {
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        let dm = f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
        p += 4;
        let sr = &data[p..p + 12];
        p += 12;
        let qh = &data[p..p + 32];
        p += 32;
        let qs = &data[p..p + 128];
        p += 128;
        let mut qs6 = [0u8; 8];
        let mut qm6 = [0u8; 8];
        for i in 0..4 {
            qs6[2 * i] = (sr[i] & 0xF) | ((sr[8 + i] & 3) << 4);
            qs6[2 * i + 1] = (sr[i] >> 4) | (((sr[8 + i] >> 2) & 3) << 4);
            qm6[2 * i] = (sr[4 + i] & 0xF) | (((sr[8 + i] >> 4) & 3) << 4);
            qm6[2 * i + 1] = (sr[4 + i] >> 4) | (((sr[8 + i] >> 6) & 3) << 4);
        }
        for si in 0..8 {
            let sc = f32::from(qs6[si]) * d;
            let mn = f32::from(qm6[si]) * dm;
            for j in 0..32 {
                let idx = si * 32 + j;
                let lo = (qs[idx / 2] >> (4 * (idx & 1))) & 0x0F;
                let hi = ((qh[idx / 8] >> (idx & 7)) & 1) << 4;
                let q = f32::from(lo | hi);
                out.push(q * sc / 31.0 - mn);
            }
        }
    }
    Ok(out)
}

/// Quantize f32 slice to `Q6_K` (6-bit K-quant, 210 bytes / 256 elements).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 256.
pub fn quantize_q6_k(data: &[f32]) -> Result<Vec<u8>> {
    check_len("Q6_K", data.len(), QK_K)?;
    let nb = data.len() / QK_K;
    let mut out = Vec::with_capacity(nb * BQ6_K);
    for blk in data.chunks_exact(QK_K) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 31.0;
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for (sb, sub) in blk.chunks_exact(16).enumerate() {
            let sb_max = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let sc = if d == 0.0 {
                0
            } else {
                (sb_max / d).clamp(-128.0, 127.0) as i8
            };
            scales[sb] = sc;
            let sb_d = f32::from(sc) * d;
            let inv_sb_d = if sb_d == 0.0 { 0.0 } else { 1.0 / sb_d };
            for (j, &v) in sub.iter().enumerate() {
                let q = ((v * inv_sb_d + 32.5).clamp(0.0, 63.0) as u8) & 0x3F;
                let idx = sb * 16 + j;
                ql[idx / 2] |= (q & 0x0F) << (4 * (idx & 1));
                qh[idx / 4] |= ((q >> 4) & 0x03) << (2 * (idx & 3));
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        // SAFETY: i8 and u8 have identical layout
        let scales_u8: &[u8; 16] = unsafe { &*std::ptr::from_ref(&scales).cast() };
        out.extend_from_slice(scales_u8);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    Ok(out)
}

/// Dequantize `Q6_K` bytes back to f32.
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 210.
pub fn dequantize_q6_k(data: &[u8]) -> Result<Vec<f32>> {
    check_len("Q6_K bytes", data.len(), BQ6_K)?;
    let nb = data.len() / BQ6_K;
    let mut out = Vec::with_capacity(nb * QK_K);
    let mut p = 0;
    for _ in 0..nb {
        let ql = &data[p..p + 128];
        p += 128;
        let qh = &data[p..p + 64];
        p += 64;
        let scales_raw = &data[p..p + 16];
        p += 16;
        let d = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
        p += 2;
        for (sb, &sc_byte) in scales_raw.iter().enumerate() {
            let sc = f32::from(sc_byte as i8);
            for j in 0..16 {
                let idx = sb * 16 + j;
                let lo = (ql[idx / 2] >> (4 * (idx & 1))) & 0x0F;
                let hi = (qh[idx / 4] >> (2 * (idx & 3))) & 0x03;
                let q = f32::from(lo | (hi << 4)) - 32.0;
                out.push(d * sc * q);
            }
        }
    }
    Ok(out)
}

/// Alias for [`quantize_q4_k`] (`Q4_K_M` uses the same block format as `Q4_K_S`).
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 256.
pub fn quantize_q4_k_m(data: &[f32]) -> Result<Vec<u8>> {
    quantize_q4_k(data)
}

/// Alias for [`dequantize_q4_k`].
///
/// # Errors
/// Returns `Error::Quant` if `data.len()` is not a multiple of 144.
pub fn dequantize_q4_k_m(data: &[u8]) -> Result<Vec<f32>> {
    dequantize_q4_k(data)
}
