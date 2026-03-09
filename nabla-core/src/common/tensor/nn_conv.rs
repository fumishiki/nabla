use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
#[cfg(feature = "cpu")]
use crate::tensor::two;

macro_rules! conv_config {
    ($name:ident, $doc:literal, $spatial:ty, $stride_def:expr, $pad_def:expr, $dil_def:expr) => {
        #[doc = $doc]
        pub struct $name {
            /// Stride.
            pub stride: $spatial,
            /// Zero-padding.
            pub padding: $spatial,
            /// Dilation.
            pub dilation: $spatial,
            /// Number of groups for grouped convolution.
            pub groups: usize,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    stride: $stride_def,
                    padding: $pad_def,
                    dilation: $dil_def,
                    groups: 1,
                }
            }
        }

        impl $name {
            /// Set stride.
            #[must_use]
            pub fn stride(mut self, s: $spatial) -> Self {
                self.stride = s;
                self
            }
            /// Set padding.
            #[must_use]
            pub fn padding(mut self, p: $spatial) -> Self {
                self.padding = p;
                self
            }
            /// Set dilation.
            #[must_use]
            pub fn dilation(mut self, d: $spatial) -> Self {
                self.dilation = d;
                self
            }
            /// Set groups.
            #[must_use]
            pub fn groups(mut self, g: usize) -> Self {
                self.groups = g;
                self
            }
        }
    };
}

conv_config!(
    Conv1dConfig,
    "Configuration for 1-D convolution.",
    usize,
    1,
    0,
    1
);
conv_config!(
    Conv2dConfig,
    "Configuration for 2-D convolution.",
    (usize, usize),
    (1, 1),
    (0, 0),
    (1, 1)
);
conv_config!(
    Conv3dConfig,
    "Configuration for 3-D convolution.",
    (usize, usize, usize),
    (1, 1, 1),
    (0, 0, 0),
    (1, 1, 1)
);

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Convolution ----

    /// im2col: unfold input patches for convolution.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cpu")]
    fn im2col(
        x: &Self,
        c_in: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Self {
        let out_h = (h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
        let out_w = (w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
        let col_rows = c_in * kh * kw;
        let col_cols = out_h * out_w;
        Self::from_fn(col_rows, col_cols, |row, col| {
            let ow = col % out_w;
            let oh = col / out_w;
            let kw_idx = row % kw;
            let kh_idx = (row / kw) % kh;
            let c = row / (kh * kw);
            let ih = oh * stride.0 + kh_idx * dilation.0;
            let iw = ow * stride.1 + kw_idx * dilation.1;
            if ih >= padding.0 && ih < h + padding.0 && iw >= padding.1 && iw < w + padding.1 {
                x.get(c, (ih - padding.0) * w + (iw - padding.1))
            } else {
                T::zero()
            }
        })
    }

    /// 2-D convolution.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        assert!(c_in.is_multiple_of(groups) && c_out.is_multiple_of(groups));
        let out = Self::from_storage(B::conv2d(
            &self.storage,
            &weight.storage,
            n_batch,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            stride,
            padding,
            dilation,
            groups,
        ));
        if let Some(bi) = bias {
            super::assert_cpu_only::<B>("conv2d");
            #[cfg(feature = "cpu")]
            {
                let out_h = (h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
                let out_w = (w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
                let out_spatial = out_h * out_w;
                let bias_exp = B::from_fn(n_batch * c_out, out_spatial, |row, _col| {
                    B::get(&bi.storage, 0, row % c_out)
                });
                Self::from_storage(B::add(&out.storage, &bias_exp))
            }
            #[cfg(not(feature = "cpu"))]
            {
                let _ = &bi;
                panic!("nabla: conv2d bias is CPU-only; GPU path needs a dedicated bias kernel");
            }
        } else {
            out
        }
    }

    /// 1-D convolution.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        length: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        assert!(c_in.is_multiple_of(groups) && c_out.is_multiple_of(groups));
        let out = Self::from_storage(B::conv1d(
            &self.storage,
            &weight.storage,
            n_batch,
            c_in,
            length,
            c_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        ));
        if let Some(bi) = bias {
            super::assert_cpu_only::<B>("conv1d");
            #[cfg(feature = "cpu")]
            {
                let out_len =
                    (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
                let bias_exp = B::from_fn(n_batch * c_out, out_len, |row, _col| {
                    B::get(&bi.storage, 0, row % c_out)
                });
                Self::from_storage(B::add(&out.storage, &bias_exp))
            }
            #[cfg(not(feature = "cpu"))]
            {
                let _ = &bi;
                panic!("nabla: conv1d bias is CPU-only; GPU path needs a dedicated bias kernel");
            }
        } else {
            out
        }
    }

    /// 2-D convolution with [`Conv2dConfig`] builder (fewer positional args).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_with(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        config: &Conv2dConfig,
    ) -> Self {
        self.conv2d(
            weight,
            bias,
            n_batch,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            config.stride,
            config.padding,
            config.dilation,
            config.groups,
        )
    }

    /// 1-D convolution with [`Conv1dConfig`] builder (fewer positional args).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_with(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        length: usize,
        c_out: usize,
        kernel_size: usize,
        config: &Conv1dConfig,
    ) -> Self {
        self.conv1d(
            weight,
            bias,
            n_batch,
            c_in,
            length,
            c_out,
            kernel_size,
            config.stride,
            config.padding,
            config.dilation,
            config.groups,
        )
    }

    /// Transposed 2-D convolution.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> Self {
        let out = Self::from_storage(B::conv_transpose2d(
            &self.storage,
            &weight.storage,
            n_batch,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            stride,
            padding,
            output_padding,
        ));
        if let Some(bi) = bias {
            super::assert_cpu_only::<B>("conv_transpose2d");
            #[cfg(feature = "cpu")]
            {
                let out_h = (h - 1) * stride.0 - 2 * padding.0 + kh + output_padding.0;
                let out_w = (w - 1) * stride.1 - 2 * padding.1 + kw + output_padding.1;
                let bias_exp = B::from_fn(n_batch * c_out, out_h * out_w, |row, _col| {
                    B::get(&bi.storage, 0, row % c_out)
                });
                Self::from_storage(B::add(&out.storage, &bias_exp))
            }
            #[cfg(not(feature = "cpu"))]
            {
                let _ = &bi;
                panic!(
                    "nabla: conv_transpose2d bias is CPU-only; GPU path needs a dedicated bias kernel"
                );
            }
        } else {
            out
        }
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// 3-D convolution.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        d: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        assert!(c_in.is_multiple_of(groups) && c_out.is_multiple_of(groups));
        let out = Self::from_storage(B::conv3d(
            &self.storage,
            &weight.storage,
            n_batch,
            c_in,
            d,
            h,
            w,
            c_out,
            kd,
            kh,
            kw,
            stride,
            padding,
            dilation,
            groups,
        ));
        if let Some(bi) = bias {
            super::assert_cpu_only::<B>("conv3d");
            #[cfg(feature = "cpu")]
            {
                let out_d = (d + 2 * padding.0 - dilation.0 * (kd - 1) - 1) / stride.0 + 1;
                let out_h = (h + 2 * padding.1 - dilation.1 * (kh - 1) - 1) / stride.1 + 1;
                let out_w = (w + 2 * padding.2 - dilation.2 * (kw - 1) - 1) / stride.2 + 1;
                let out_spatial = out_d * out_h * out_w;
                let bias_exp = B::from_fn(n_batch * c_out, out_spatial, |row, _col| {
                    B::get(&bi.storage, 0, row % c_out)
                });
                Self::from_storage(B::add(&out.storage, &bias_exp))
            }
            #[cfg(not(feature = "cpu"))]
            {
                let _ = &bi;
                panic!("nabla: conv3d bias is CPU-only; GPU path needs a dedicated bias kernel");
            }
        } else {
            out
        }
    }

    /// 3-D convolution with [`Conv3dConfig`] builder (fewer positional args).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_with(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        n_batch: usize,
        c_in: usize,
        d: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        config: &Conv3dConfig,
    ) -> Self {
        self.conv3d(
            weight,
            bias,
            n_batch,
            c_in,
            d,
            h,
            w,
            c_out,
            kd,
            kh,
            kw,
            config.stride,
            config.padding,
            config.dilation,
            config.groups,
        )
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Pooling ----

    /// 2-D max pooling.
    #[must_use]
    pub fn max_pool2d(
        &self,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self::from_storage(B::max_pool2d(
            &self.storage,
            h,
            w,
            kh,
            kw,
            stride.0,
            stride.1,
            padding.0,
            padding.1,
        ))
    }

    /// 2-D max pooling with argmax flat indices.
    #[must_use]
    pub fn max_pool2d_with_indices(
        &self,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> (Self, Self) {
        let (v, idx) = B::max_pool2d_with_indices(
            &self.storage,
            h,
            w,
            kh,
            kw,
            stride.0,
            stride.1,
            padding.0,
            padding.1,
        );
        (Self::from_storage(v), Self::from_storage(idx))
    }

    /// 2-D average pooling.
    #[must_use]
    pub fn avg_pool2d(
        &self,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self::from_storage(B::avg_pool2d(
            &self.storage,
            h,
            w,
            kh,
            kw,
            stride.0,
            stride.1,
            padding.0,
            padding.1,
        ))
    }

    /// Adaptive average pool 2-D.
    #[must_use]
    pub fn adaptive_avg_pool2d(&self, h: usize, w: usize, out_h: usize, out_w: usize) -> Self {
        Self::from_storage(B::adaptive_avg_pool2d(&self.storage, h, w, out_h, out_w))
    }

    /// 1-D max pooling.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn max_pool1d(
        &self,
        length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
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
    #[cfg(feature = "cpu")]
    pub fn avg_pool1d(
        &self,
        length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
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
