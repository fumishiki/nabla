use crate::backend::Backend;
use crate::scalar::{Fp4E2M1, Fp8E4M3, Fp8E5M2};

use super::Tensor;

impl<B: Backend> Tensor<f32, B> {
    /// Quantize f32 tensor to Fp8E4M3 (element-wise cast).
    #[must_use]
    pub fn quantize_fp8_e4m3(&self) -> Tensor<Fp8E4M3, B> {
        self.cast::<Fp8E4M3>()
    }

    /// Quantize f32 tensor to Fp8E5M2 (element-wise cast).
    #[must_use]
    pub fn quantize_fp8_e5m2(&self) -> Tensor<Fp8E5M2, B> {
        self.cast::<Fp8E5M2>()
    }

    /// Blockwise quantize to Fp4E2M1: returns (quantized, per-block scales).
    /// Each block is independently scaled to max_abs / 6.0 before quantization.
    #[must_use]
    pub fn quantize_fp4_blockwise(&self, block_size: usize) -> (Tensor<Fp4E2M1, B>, Vec<f32>) {
        let vals = self.to_vec();
        let (m, nc) = self.shape();
        let n = vals.len();
        let num_blocks = n.div_ceil(block_size);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut scaled = Vec::with_capacity(n);
        for b in 0..num_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let block = &vals[start..end];
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
            scales.push(scale);
            for &v in block {
                scaled.push(v / scale);
            }
        }
        (
            Tensor::<f32, B>::from_vec(scaled, m, nc).cast::<Fp4E2M1>(),
            scales,
        )
    }
}

impl<B: Backend> Tensor<Fp8E4M3, B> {
    /// Dequantize Fp8E4M3 tensor to f32 (element-wise cast).
    #[must_use]
    pub fn dequantize_fp8_e4m3(&self) -> Tensor<f32, B> {
        self.cast::<f32>()
    }
}

impl<B: Backend> Tensor<Fp8E5M2, B> {
    /// Dequantize Fp8E5M2 tensor to f32 (element-wise cast).
    #[must_use]
    pub fn dequantize_fp8_e5m2(&self) -> Tensor<f32, B> {
        self.cast::<f32>()
    }
}

impl<B: Backend> Tensor<Fp4E2M1, B> {
    /// Dequantize blockwise Fp4E2M1 tensor to f32 using the given per-block scales.
    #[must_use]
    pub fn dequantize_fp4_blockwise(&self, scales: &[f32], block_size: usize) -> Tensor<f32, B> {
        let fp4_f32 = self.cast::<f32>().to_vec();
        let (m, nc) = self.shape();
        let result: Vec<f32> = fp4_f32
            .chunks(block_size)
            .enumerate()
            .flat_map(|(b, chunk)| {
                let scale = scales[b];
                chunk.iter().map(move |&v| v * scale).collect::<Vec<_>>()
            })
            .collect();
        Tensor::<f32, B>::from_vec(result, m, nc)
    }
}
