// tensor/nn/conv.rs — Convolution operations and config structs.

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

// ---- Convolution configuration builders ----

/// Configuration for 2-D convolution, replacing positional stride/padding/dilation/groups args.
pub struct Conv2dConfig {
    /// Stride in (height, width).
    pub stride: (usize, usize),
    /// Zero-padding in (height, width).
    pub padding: (usize, usize),
    /// Dilation in (height, width).
    pub dilation: (usize, usize),
    /// Number of groups for grouped convolution.
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

impl Conv2dConfig {
    /// Set stride.
    #[must_use]
    pub fn stride(mut self, s: (usize, usize)) -> Self {
        self.stride = s;
        self
    }
    /// Set padding.
    #[must_use]
    pub fn padding(mut self, p: (usize, usize)) -> Self {
        self.padding = p;
        self
    }
    /// Set dilation.
    #[must_use]
    pub fn dilation(mut self, d: (usize, usize)) -> Self {
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

/// Configuration for 1-D convolution.
pub struct Conv1dConfig {
    /// Stride.
    pub stride: usize,
    /// Zero-padding.
    pub padding: usize,
    /// Dilation.
    pub dilation: usize,
    /// Number of groups for grouped convolution.
    pub groups: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        }
    }
}

impl Conv1dConfig {
    /// Set stride.
    #[must_use]
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }
    /// Set padding.
    #[must_use]
    pub fn padding(mut self, p: usize) -> Self {
        self.padding = p;
        self
    }
    /// Set dilation.
    #[must_use]
    pub fn dilation(mut self, d: usize) -> Self {
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

/// Configuration for 3-D convolution.
pub struct Conv3dConfig {
    /// Stride in (depth, height, width).
    pub stride: (usize, usize, usize),
    /// Zero-padding in (depth, height, width).
    pub padding: (usize, usize, usize),
    /// Dilation in (depth, height, width).
    pub dilation: (usize, usize, usize),
    /// Number of groups for grouped convolution.
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
        }
    }
}

impl Conv3dConfig {
    /// Set stride.
    #[must_use]
    pub fn stride(mut self, s: (usize, usize, usize)) -> Self {
        self.stride = s;
        self
    }
    /// Set padding.
    #[must_use]
    pub fn padding(mut self, p: (usize, usize, usize)) -> Self {
        self.padding = p;
        self
    }
    /// Set dilation.
    #[must_use]
    pub fn dilation(mut self, d: (usize, usize, usize)) -> Self {
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

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Convolution ----

    /// im2col: unfold input patches for convolution.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
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
            &self.storage, &weight.storage,
            n_batch, c_in, h, w, c_out, kh, kw, stride, padding, dilation, groups,
        ));
        if let Some(bi) = bias {
            let out_h = (h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
            let out_w = (w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
            let out_spatial = out_h * out_w;
            let bias_exp = B::from_fn(n_batch * c_out, out_spatial, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
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
            &self.storage, &weight.storage,
            n_batch, c_in, length, c_out, kernel_size, stride, padding, dilation, groups,
        ));
        if let Some(bi) = bias {
            let out_len = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            let bias_exp = B::from_fn(n_batch * c_out, out_len, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
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
            weight, bias, n_batch, c_in, h, w, c_out, kh, kw,
            config.stride, config.padding, config.dilation, config.groups,
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
            weight, bias, n_batch, c_in, length, c_out, kernel_size,
            config.stride, config.padding, config.dilation, config.groups,
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
        let out_h = (h - 1) * stride.0 - 2 * padding.0 + kh + output_padding.0;
        let out_w = (w - 1) * stride.1 - 2 * padding.1 + kw + output_padding.1;
        let out = Self::from_storage(B::conv_transpose2d(
            &self.storage, &weight.storage,
            n_batch, c_in, h, w, c_out, kh, kw, stride, padding, output_padding,
        ));
        if let Some(bi) = bias {
            let bias_exp = B::from_fn(n_batch * c_out, out_h * out_w, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
        } else {
            out
        }
    }
}

// ---- 3-D convolution (cpu-gated) ----

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// 3-D convolution.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d(
        &self, weight: &Self, bias: Option<&Self>,
        n_batch: usize, c_in: usize, d: usize, h: usize, w: usize,
        c_out: usize, kd: usize, kh: usize, kw: usize,
        stride: (usize, usize, usize), padding: (usize, usize, usize),
        dilation: (usize, usize, usize), groups: usize,
    ) -> Self {
        assert!(c_in.is_multiple_of(groups) && c_out.is_multiple_of(groups));
        let out_d = (d + 2 * padding.0 - dilation.0 * (kd - 1) - 1) / stride.0 + 1;
        let out_h = (h + 2 * padding.1 - dilation.1 * (kh - 1) - 1) / stride.1 + 1;
        let out_w = (w + 2 * padding.2 - dilation.2 * (kw - 1) - 1) / stride.2 + 1;
        let out_spatial = out_d * out_h * out_w;
        let out = Self::from_storage(B::conv3d(
            &self.storage, &weight.storage,
            n_batch, c_in, d, h, w, c_out, kd, kh, kw, stride, padding, dilation, groups,
        ));
        if let Some(bi) = bias {
            let bias_exp = B::from_fn(n_batch * c_out, out_spatial, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
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
            weight, bias, n_batch, c_in, d, h, w, c_out, kd, kh, kw,
            config.stride, config.padding, config.dilation, config.groups,
        )
    }
}
