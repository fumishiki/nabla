//! AWQ INT4 weight-only quantization (P6-AWQ-01..05).
//!
//! Activation-aware weight quantization: calibrate per-channel activation
//! magnitudes, grid-search optimal scales, pack weights into INT4 u32 words.

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

// ── P6-AWQ-05: group_size default ──────────────────────────────────────
const DEFAULT_GROUP_SIZE: usize = 128;
const GRID_STEPS: usize = 20;
const INT4_MIN: i8 = -8;
const INT4_MAX: i8 = 7;

// ── Calibration statistics (P6-AWQ-01) ─────────────────────────────────

/// Per-channel activation statistics collected during calibration.
pub struct CalibrationStats<T: Scalar, B: Backend> {
    pub channel_means: Tensor<T, B>,
    pub channel_absmax: Tensor<T, B>,
    pub num_samples: usize,
    accum_sum: Tensor<T, B>,
    accum_max: Tensor<T, B>,
}

impl<T: Scalar, B: Backend> CalibrationStats<T, B> {
    /// Start calibration for `num_channels` input channels.
    #[must_use]
    pub fn new(num_channels: usize) -> Self {
        Self {
            channel_means: Tensor::zeros(1, num_channels),
            channel_absmax: Tensor::zeros(1, num_channels),
            num_samples: 0,
            accum_sum: Tensor::zeros(1, num_channels),
            accum_max: Tensor::zeros(1, num_channels),
        }
    }

    /// Feed one activation batch `(batch_rows, channels)` into the statistics.
    pub fn update(&mut self, activations: &Tensor<T, B>) {
        let (_rows, cols) = activations.shape();
        assert_eq!(
            cols,
            self.accum_sum.ncols(),
            "nabla-train: calibration channel mismatch"
        );
        let abs_act = activations.map(T::math_abs);
        let batch_sum = Tensor::from_fn(1, cols, |_, c| {
            let mut s = T::zero();
            for r in 0..activations.nrows() {
                s = s + abs_act.get(r, c);
            }
            s
        });
        let batch_max: Tensor<T, B> = Tensor::from_fn(1, cols, |_, c| {
            let mut m = T::zero();
            for r in 0..activations.nrows() {
                let v = abs_act.get(r, c);
                if v.to_f64() > m.to_f64() {
                    m = v;
                }
            }
            m
        });
        let new_sum = &self.accum_sum + &batch_sum;
        self.accum_sum = new_sum;
        let prev_max = self.accum_max.clone();
        self.accum_max = Tensor::from_fn(1, cols, |_, c| {
            let a = prev_max.get(0, c);
            let b = batch_max.get(0, c);
            if a.to_f64() > b.to_f64() { a } else { b }
        });
        self.num_samples += activations.nrows();
        let count = T::from_f64(self.num_samples as f64);
        self.channel_means = self.accum_sum.map(|v| v / count);
        self.channel_absmax = self.accum_max.clone();
    }
}

// ── Quantized weight representation ────────────────────────────────────

/// AWQ-quantized weight matrix (INT4 packed).
pub struct QuantizedWeight<T: Scalar, B: Backend> {
    /// Packed INT4 data: each u32 holds 8 weights. Shape: `(out_features, ceil(in_features/8))`.
    pub packed: Vec<u32>,
    /// Per-group scales. Shape: `(out_features, num_groups)`.
    pub scales: Tensor<T, B>,
    /// Per-group zero-points. Shape: `(out_features, num_groups)`.
    pub zeros: Tensor<T, B>,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

// ── P6-AWQ-02: per-channel scale optimization ──────────────────────────

/// Compute optimal per-channel AWQ scales via activation-aware grid search.
fn compute_awq_scales<T: Scalar, B: Backend>(
    weight: &Tensor<T, B>,
    stats: &CalibrationStats<T, B>,
    group_size: usize,
) -> Tensor<T, B> {
    let (out_f, in_f) = weight.shape();
    let num_groups = in_f.div_ceil(group_size);
    // O(out_f, num_groups) optimal scales via grid search
    Tensor::from_fn(out_f, num_groups, |row, g| {
        let g_start = g * group_size;
        let g_end = (g_start + group_size).min(in_f);
        // W_max in group
        let mut w_max = T::zero();
        for c in g_start..g_end {
            let v = weight.get(row, c).math_abs();
            if v.to_f64() > w_max.to_f64() {
                w_max = v;
            }
        }
        // activation importance for this group
        let mut act_importance = T::zero();
        for c in g_start..g_end {
            let a = stats.channel_absmax.get(0, c);
            if a.to_f64() > act_importance.to_f64() {
                act_importance = a;
            }
        }
        // grid search: scale ∈ [0.1, 1.0], minimize MSE(quant(w*s) / s, w)
        let mut best_scale = T::one();
        let mut best_mse = f64::MAX;
        for step in 0..GRID_STEPS {
            let alpha = 0.1 + 0.9 * (step as f64) / ((GRID_STEPS - 1) as f64);
            // scale = act_importance^alpha
            let s = T::from_f64(act_importance.to_f64().powf(alpha));
            if s.to_f64() <= 0.0 {
                continue;
            }
            let mut mse = 0.0;
            for c in g_start..g_end {
                let w = weight.get(row, c);
                let ws = w * s;
                // quantize ws to INT4 range, then dequant
                let q_scale = T::from_f64(w_max.to_f64() * s.to_f64() / f64::from(INT4_MAX));
                if q_scale.to_f64() <= 0.0 {
                    continue;
                }
                let q = (ws.to_f64() / q_scale.to_f64())
                    .round()
                    .clamp(f64::from(INT4_MIN), f64::from(INT4_MAX));
                let deq = q * q_scale.to_f64() / s.to_f64();
                let diff = w.to_f64() - deq;
                mse += diff * diff;
            }
            if mse < best_mse {
                best_mse = mse;
                best_scale = s;
            }
        }
        best_scale
    })
}

/// Quantize a group of weights given a scale, returning (packed_nibbles, group_scale, group_zero).
fn quantize_group<T: Scalar>(weights: &[T], act_scale: f64) -> (Vec<i8>, f64, f64) {
    // scale weights by activation scale
    let scaled: Vec<f64> = weights.iter().map(|w| w.to_f64() * act_scale).collect();
    let w_max = scaled.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let q_scale = if w_max > 0.0 {
        w_max / f64::from(INT4_MAX)
    } else {
        1.0
    };
    let quantized: Vec<i8> = scaled
        .iter()
        .map(|&v| {
            (v / q_scale)
                .round()
                .clamp(f64::from(INT4_MIN), f64::from(INT4_MAX)) as i8
        })
        .collect();
    (quantized, q_scale / act_scale, 0.0)
}

// ── P6-AWQ-03: INT4 packing (8 weights → 1 u32, little-endian) ────────

/// Pack 8 signed INT4 values into one u32 (little-endian nibble order).
#[must_use]
pub fn pack_int4(nibbles: &[i8]) -> u32 {
    debug_assert!(nibbles.len() <= 8);
    let mut packed = 0u32;
    for (i, &n) in nibbles.iter().enumerate() {
        let bits = (n as u8 & 0x0F) as u32;
        packed |= bits << (i * 4);
    }
    packed
}

/// Unpack one u32 into 8 signed INT4 values.
#[must_use]
pub fn unpack_int4(packed: u32) -> [i8; 8] {
    let mut out = [0i8; 8];
    for i in 0..8 {
        let bits = ((packed >> (i * 4)) & 0x0F) as u8;
        // sign-extend 4-bit to i8
        out[i] = if bits & 0x08 != 0 {
            (bits | 0xF0) as i8
        } else {
            bits as i8
        };
    }
    out
}

// ── P6-AWQ-01..05: full quantization pipeline ──────────────────────────

/// Quantize a weight matrix using AWQ with calibration stats.
pub fn quantize_awq<T: Scalar, B: Backend>(
    weight: &Tensor<T, B>,
    stats: &CalibrationStats<T, B>,
    group_size: usize,
) -> QuantizedWeight<T, B> {
    let gs = if group_size == 0 {
        DEFAULT_GROUP_SIZE
    } else {
        group_size
    };
    let (out_f, in_f) = weight.shape();
    let num_groups = (in_f + gs - 1) / gs;
    let packed_cols = (in_f + 7) / 8;
    let awq_scales = compute_awq_scales(weight, stats, gs);
    let mut all_packed = Vec::with_capacity(out_f * packed_cols);
    let mut scale_data = Vec::with_capacity(out_f * num_groups);
    let mut zero_data = Vec::with_capacity(out_f * num_groups);
    for row in 0..out_f {
        let mut row_nibbles = Vec::with_capacity(in_f);
        for g in 0..num_groups {
            let g_start = g * gs;
            let g_end = (g_start + gs).min(in_f);
            let group_weights: Vec<T> = (g_start..g_end).map(|c| weight.get(row, c)).collect();
            let act_s = awq_scales.get(row, g).to_f64();
            let (nibbles, s, z) = quantize_group(&group_weights, act_s);
            row_nibbles.extend_from_slice(&nibbles);
            scale_data.push(T::from_f64(s));
            zero_data.push(T::from_f64(z));
        }
        // pad to multiple of 8
        while row_nibbles.len() % 8 != 0 {
            row_nibbles.push(0);
        }
        for chunk in row_nibbles.chunks(8) {
            all_packed.push(pack_int4(chunk));
        }
    }
    QuantizedWeight {
        packed: all_packed,
        scales: Tensor::from_vec(scale_data, out_f, num_groups),
        zeros: Tensor::from_vec(zero_data, out_f, num_groups),
        out_features: out_f,
        in_features: in_f,
        group_size: gs,
    }
}

/// Quantize with default group_size=128.
pub fn quantize_awq_default<T: Scalar, B: Backend>(
    weight: &Tensor<T, B>,
    stats: &CalibrationStats<T, B>,
) -> QuantizedWeight<T, B> {
    quantize_awq(weight, stats, DEFAULT_GROUP_SIZE)
}

// ── Dequantization (CPU reference) ─────────────────────────────────────

/// Dequantize packed INT4 weights back to a float tensor (CPU reference).
pub fn dequantize<T: Scalar, B: Backend>(qw: &QuantizedWeight<T, B>) -> Tensor<T, B> {
    let (out_f, in_f, gs) = (qw.out_features, qw.in_features, qw.group_size);
    let packed_cols = (in_f + 7) / 8;
    Tensor::from_fn(out_f, in_f, |r, c| {
        let group_idx = c / gs;
        let scale = qw.scales.get(r, group_idx);
        let zero = qw.zeros.get(r, group_idx);
        let pack_idx = r * packed_cols + c / 8;
        let nibble_idx = c % 8;
        let nibbles = unpack_int4(qw.packed[pack_idx]);
        let q = T::from_f64(f64::from(nibbles[nibble_idx]));
        (q - zero) * scale
    })
}

/// Dequantize and multiply: out = input @ dequant(qw)^T. Shape: `(M, out_features)`.
pub fn dequant_matmul<T: Scalar, B: Backend>(
    input: &Tensor<T, B>,
    qw: &QuantizedWeight<T, B>,
) -> Tensor<T, B> {
    let deq = dequantize(qw);
    // input (M, in_f) @ deq^T (in_f, out_f) = (M, out_f)
    input.matmul_nt(&deq)
}

// ── P6-AWQ-04: CUDA dequant-matmul kernel ──────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_dequant {
    use super::*;
    use std::ffi::c_void;

    use nabla_core::backend::Cuda;
    use nabla_core::cuda_backend::{cuda_launch_kernel_src, cuda_upload_u32};

    /// CUDA INT4 dequant-matmul kernel source (NVRTC JIT).
    const DEQUANT_MATMUL_SRC: &str = r#"
extern "C" __global__ void dequant_matmul_f32(
    const float* __restrict__ input,   // (M, K)
    const unsigned int* __restrict__ packed, // (N, packed_K)
    const float* __restrict__ scales,  // (N, num_groups)
    const float* __restrict__ zeros,   // (N, num_groups)
    float* __restrict__ output,        // (M, N)
    int M, int N, int K, int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dim
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dim
    if (row >= M || col >= N) return;
    int packed_K = (K + 7) / 8;
    int num_groups = (K + group_size - 1) / group_size;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        int g = k / group_size;
        float s = scales[col * num_groups + g];
        float z = zeros[col * num_groups + g];
        unsigned int pack = packed[col * packed_K + k / 8];
        int nibble = (pack >> ((k % 8) * 4)) & 0xF;
        int ival = (nibble & 0x8) ? (nibble | 0xFFFFFFF0) : nibble;
        float w = ((float)ival - z) * s;
        acc += input[row * K + k] * w;
    }
    output[row * N + col] = acc;
}
    "#;

    /// Launch CUDA dequant-matmul kernel.
    pub fn cuda_dequant_matmul(
        input: &Tensor<f32, Cuda>,
        qw: &QuantizedWeight<f32, Cuda>,
    ) -> Tensor<f32, Cuda> {
        let (m, k) = input.shape();
        let n = qw.out_features;
        assert_eq!(k, qw.in_features, "cuda_dequant_matmul: K mismatch");
        let num_groups = (k + qw.group_size - 1) / qw.group_size;
        assert_eq!(
            qw.scales.shape(),
            (n, num_groups),
            "cuda_dequant_matmul: scales shape mismatch"
        );
        assert_eq!(
            qw.zeros.shape(),
            (n, num_groups),
            "cuda_dequant_matmul: zeros shape mismatch"
        );

        let packed_buf = cuda_upload_u32(&qw.packed);
        let out = Tensor::<f32, Cuda>::empty(m, n);

        let input_ptr = input.storage().buffer().as_ptr();
        let packed_ptr = packed_buf.as_ptr();
        let scales_ptr = qw.scales.storage().buffer().as_ptr();
        let zeros_ptr = qw.zeros.storage().buffer().as_ptr();
        let out_ptr = out.storage().buffer().as_ptr();
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let group_i32 = qw.group_size as i32;

        let block_x = 16u32;
        let block_y = 16u32;
        let grid_x = ((n as u32) + block_x - 1) / block_x;
        let grid_y = ((m as u32) + block_y - 1) / block_y;

        let mut args: Vec<*mut c_void> = vec![
            &input_ptr as *const _ as *mut c_void,
            &packed_ptr as *const _ as *mut c_void,
            &scales_ptr as *const _ as *mut c_void,
            &zeros_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &n_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &group_i32 as *const _ as *mut c_void,
        ];

        let kernel_name = "k_dequant_matmul_f32";
        cuda_launch_kernel_src(
            kernel_name,
            DEQUANT_MATMUL_SRC,
            (grid_x, grid_y, 1),
            (block_x, block_y, 1),
            0,
            &mut args,
        );

        out
    }
}

#[cfg(feature = "cuda")]
pub use cuda_dequant::cuda_dequant_matmul;
