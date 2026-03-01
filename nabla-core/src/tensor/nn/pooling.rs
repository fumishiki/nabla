// tensor/nn/pooling.rs — Pooling and interpolation operations.

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::{two, Tensor};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Pooling ----

    /// 2-D max pooling.
    #[must_use]
    pub fn max_pool2d(
        &self, h: usize, w: usize, kh: usize, kw: usize,
        stride: (usize, usize), padding: (usize, usize),
    ) -> Self {
        Self::from_storage(B::max_pool2d(
            &self.storage, h, w, kh, kw, stride.0, stride.1, padding.0, padding.1,
        ))
    }

    /// 2-D max pooling with argmax flat indices.
    #[must_use]
    pub fn max_pool2d_with_indices(
        &self, h: usize, w: usize, kh: usize, kw: usize,
        stride: (usize, usize), padding: (usize, usize),
    ) -> (Self, Self) {
        let (v, idx) = B::max_pool2d_with_indices(
            &self.storage, h, w, kh, kw, stride.0, stride.1, padding.0, padding.1,
        );
        (Self::from_storage(v), Self::from_storage(idx))
    }

    /// 2-D average pooling.
    #[must_use]
    pub fn avg_pool2d(
        &self, h: usize, w: usize, kh: usize, kw: usize,
        stride: (usize, usize), padding: (usize, usize),
    ) -> Self {
        Self::from_storage(B::avg_pool2d(
            &self.storage, h, w, kh, kw, stride.0, stride.1, padding.0, padding.1,
        ))
    }

    /// Adaptive average pool 2-D.
    #[must_use]
    pub fn adaptive_avg_pool2d(&self, h: usize, w: usize, out_h: usize, out_w: usize) -> Self {
        Self::from_storage(B::adaptive_avg_pool2d(&self.storage, h, w, out_h, out_w))
    }

    /// 1-D max pooling.
    #[must_use]
    pub fn max_pool1d(&self, length: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;
        let nc = self.nrows();
        let two = two::<T>();
        Self::from_fn(nc, out_len, |ch, col| {
            let mut best = T::zero();
            let mut first = true;
            for k in 0..kernel_size {
                let il = col * stride + k;
                if il >= padding && il < length + padding {
                    let v = self.get(ch, il - padding);
                    if first {
                        best = v;
                        first = false;
                    } else {
                        best = (best + v + (best - v).math_abs()) / two;
                    }
                }
            }
            best
        })
    }

    /// 1-D average pooling.
    #[must_use]
    pub fn avg_pool1d(&self, length: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;
        let nc = self.nrows();
        let ks = T::from_f64(kernel_size as f64);
        Self::from_fn(nc, out_len, |ch, col| {
            let mut sum = T::zero();
            for k in 0..kernel_size {
                let il = col * stride + k;
                if il >= padding && il < length + padding {
                    sum = sum + self.get(ch, il - padding);
                }
            }
            sum / ks
        })
    }
}

// ---- Interpolation (cpu-gated) ----

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Nearest-neighbor interpolation (upsample/downsample).
    #[must_use]
    pub fn interpolate_nearest(&self, h: usize, w: usize, out_h: usize, out_w: usize) -> Self {
        let nc = self.nrows();
        Self::from_fn(nc, out_h * out_w, |row, col| {
            let oh = col / out_w;
            let ow = col % out_w;
            let ih = oh * h / out_h;
            let iw = ow * w / out_w;
            self.get(row, ih * w + iw)
        })
    }

    /// Bilinear interpolation (upsample/downsample).
    #[must_use]
    pub fn interpolate_bilinear(&self, h: usize, w: usize, out_h: usize, out_w: usize) -> Self {
        let nc = self.nrows();
        Self::from_fn(nc, out_h * out_w, |row, col| {
            let oh = col / out_w;
            let ow = col % out_w;
            let scale_h = h as f64 / out_h as f64;
            let scale_w = w as f64 / out_w as f64;
            let src_h = (oh as f64 + 0.5) * scale_h - 0.5;
            let src_w = (ow as f64 + 0.5) * scale_w - 0.5;
            let h0 = src_h.floor().max(0.0) as usize;
            let w0 = src_w.floor().max(0.0) as usize;
            let h1 = (h0 + 1).min(h - 1);
            let w1 = (w0 + 1).min(w - 1);
            let fh = T::from_f64((src_h - h0 as f64).clamp(0.0, 1.0));
            let fw = T::from_f64((src_w - w0 as f64).clamp(0.0, 1.0));
            let one = T::one();
            let v00 = self.get(row, h0 * w + w0);
            let v01 = self.get(row, h0 * w + w1);
            let v10 = self.get(row, h1 * w + w0);
            let v11 = self.get(row, h1 * w + w1);
            let top = v00 * (one - fw) + v01 * fw;
            let bot = v10 * (one - fw) + v11 * fw;
            top * (one - fh) + bot * fh
        })
    }
}
