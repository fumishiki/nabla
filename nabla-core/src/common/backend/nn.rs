use super::BackendCore;
use crate::scalar::Scalar;

/// Shared WHT butterfly implementation (forward + inverse).
fn wht_impl<T: Scalar, B: BackendCore + ?Sized>(a: &B::Storage<T>, inverse: bool) -> B::Storage<T> {
    let (rows, cols) = (B::nrows(a), B::ncols(a));
    let mut data: Vec<T> = (0..rows * cols)
        .map(|idx| B::get(a, idx / cols, idx % cols))
        .collect();
    let inv_n = T::from_f64(1.0 / cols as f64);
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let mut half = 1;
        while half < cols {
            let mut i = 0;
            while i < cols {
                for j in i..i + half {
                    let (u, v) = (row[j], row[j + half]);
                    row[j] = u + v;
                    row[j + half] = u - v;
                }
                i += half << 1;
            }
            half <<= 1;
        }
        if inverse {
            for v in row.iter_mut() {
                *v = *v * inv_n;
            }
        }
    }
    B::from_fn(rows, cols, |r, c| data[r * cols + c])
}

/// Neural network ops: activations, norms, convolutions, pooling, and attention.
pub trait BackendNN: BackendCore {
    // --- Activations ---
    /// SiLU activation: x * sigmoid(x)
    fn silu<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Mish activation: x * tanh(softplus(x))
    fn mish<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Leaky ReLU: max(0, x) + negative_slope * min(0, x)
    fn leaky_relu<T: Scalar>(a: &Self::Storage<T>, negative_slope: T) -> Self::Storage<T>;
    /// ELU: x if x > 0, alpha*(exp(x)-1) otherwise
    fn elu<T: Scalar>(a: &Self::Storage<T>, alpha: T) -> Self::Storage<T>;
    /// Sigmoid: 1 / (1 + exp(-x))
    fn sigmoid<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// HardSwish: x * min(max(x+3, 0), 6) / 6
    fn hardswish<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    // --- Norms ---
    /// Row-wise softmax.
    fn softmax<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Fused layer normalization.
    fn layer_norm<T: Scalar>(
        a: &Self::Storage<T>,
        gamma: &Self::Storage<T>,
        beta: &Self::Storage<T>,
        eps: T,
    ) -> Self::Storage<T>;
    /// Fused RMS normalization.
    fn rms_norm<T: Scalar>(
        a: &Self::Storage<T>,
        gamma: &Self::Storage<T>,
        eps: T,
    ) -> Self::Storage<T>;
    /// Group normalization: normalize within channel groups.
    fn group_norm<T: Scalar>(
        a: &Self::Storage<T>,
        gamma: &Self::Storage<T>,
        beta: &Self::Storage<T>,
        groups: usize,
        eps: T,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let g_size = cols / groups;
        Self::from_fn(rows, cols, |r, c| {
            let g_start = (c / g_size) * g_size;
            let mean = (0..g_size).fold(T::zero(), |acc, j| acc + Self::get(a, r, g_start + j))
                / T::from_f64(g_size as f64);
            let var = (0..g_size).fold(T::zero(), |acc, j| {
                let d = Self::get(a, r, g_start + j) - mean;
                acc + d * d
            }) / T::from_f64(g_size as f64);
            let x = Self::get(a, r, c);
            (x - mean) / (var + eps).math_sqrt() * Self::get(gamma, 0, c) + Self::get(beta, 0, c)
        })
    }

    // --- Training ---
    /// Batch normalization (training mode).
    #[allow(clippy::too_many_arguments)]
    fn batch_norm_train<T: Scalar>(
        a: &Self::Storage<T>,
        gamma: &Self::Storage<T>,
        beta: &Self::Storage<T>,
        running_mean: &mut Self::Storage<T>,
        running_var: &mut Self::Storage<T>,
        eps: T,
        momentum: T,
        training: bool,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let n_f = T::from_f64(rows as f64);
        let (mean, var): (Vec<T>, Vec<T>) = if training {
            (0..cols)
                .map(|c| {
                    let m = (0..rows).fold(T::zero(), |acc, r| acc + Self::get(a, r, c)) / n_f;
                    let v = (0..rows).fold(T::zero(), |acc, r| {
                        let d = Self::get(a, r, c) - m;
                        acc + d * d
                    }) / n_f;
                    (m, v)
                })
                .unzip()
        } else {
            (
                (0..cols).map(|c| Self::get(running_mean, 0, c)).collect(),
                (0..cols).map(|c| Self::get(running_var, 0, c)).collect(),
            )
        };
        if training {
            let one_minus = T::from_f64(1.0) - momentum;
            for c in 0..cols {
                Self::set(
                    running_mean,
                    0,
                    c,
                    one_minus * Self::get(running_mean, 0, c) + momentum * mean[c],
                );
                Self::set(
                    running_var,
                    0,
                    c,
                    one_minus * Self::get(running_var, 0, c) + momentum * var[c],
                );
            }
        }
        Self::from_fn(rows, cols, |r, c| {
            (Self::get(a, r, c) - mean[c]) / (var[c] + eps).math_sqrt() * Self::get(gamma, 0, c)
                + Self::get(beta, 0, c)
        })
    }

    /// Cross-entropy loss: fused softmax + NLL.
    fn cross_entropy_fused<T: Scalar>(
        input: &Self::Storage<T>,
        target: &Self::Storage<T>,
        n: usize,
        c: usize,
    ) -> Self::Storage<T> {
        let total = (0..n).fold(T::zero(), |acc, i| {
            let row_max = (0..c).fold(T::from_f64(f64::NEG_INFINITY), |mx, j| {
                let v = Self::get(input, i, j);
                if v.to_f64() > mx.to_f64() { v } else { mx }
            });
            let sum_exp = (0..c).fold(T::zero(), |s, j| {
                s + (Self::get(input, i, j) - row_max).math_exp()
            });
            let t = Self::get(target, i, 0).to_f64() as usize;
            acc + -(Self::get(input, i, t) - row_max - sum_exp.math_ln())
        });
        Self::from_fn(1, 1, |_, _| total / T::from_f64(n as f64))
    }

    // --- Embedding ---
    /// Embedding gather: indices (n_tokens, 1) float-encoded, weight (vocab, embed_dim).
    fn embedding<T: Scalar>(
        indices: &Self::Storage<T>,
        weight: &Self::Storage<T>,
    ) -> Self::Storage<T>;

    /// Embedding backward: scatter-add grad into weight rows.
    fn embedding_backward<T: Scalar>(
        indices: &Self::Storage<T>,
        grad: &Self::Storage<T>,
        vocab: usize,
    ) -> Self::Storage<T> {
        let n_tokens = Self::nrows(indices) * Self::ncols(indices);
        let embed_dim = Self::ncols(grad);
        let mut out = Self::zeros(vocab, embed_dim);
        for r in 0..n_tokens {
            let idx = Self::get(indices, r, 0).to_f64() as usize;
            for c in 0..embed_dim {
                let v = Self::get(&out, idx, c) + Self::get(grad, r, c);
                Self::set(&mut out, idx, c, v);
            }
        }
        out
    }

    // --- Pooling ---
    /// 2-D max pooling.
    #[allow(clippy::too_many_arguments)]
    fn max_pool2d<T: Scalar>(
        a: &Self::Storage<T>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> Self::Storage<T> {
        let nc = Self::nrows(a);
        let (out_h, out_w) = ((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1);
        Self::from_fn(nc, out_h * out_w, |n, op| {
            let (oh, ow) = (op / out_w, op % out_w);
            let mut best = T::from_f64(f64::NEG_INFINITY);
            let mut found = false;
            for khr in 0..kh {
                for kwc in 0..kw {
                    let (ih, iw) = (oh * sh + khr, ow * sw + kwc);
                    if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                        let v = Self::get(a, n, (ih - ph) * w + (iw - pw));
                        best = if found {
                            crate::scalar::ReductionOps::reduction_max(best, v)
                        } else {
                            v
                        };
                        found = true;
                    }
                }
            }
            if found { best } else { T::zero() }
        })
    }

    /// 2-D max pooling with argmax indices.
    #[allow(clippy::too_many_arguments)]
    fn max_pool2d_with_indices<T: Scalar>(
        a: &Self::Storage<T>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> (Self::Storage<T>, Self::Storage<T>) {
        let nc = Self::nrows(a);
        let (out_h, out_w) = ((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1);
        let cap = nc * out_h * out_w;
        let mut vals = Vec::with_capacity(cap);
        let mut idxs = Vec::with_capacity(cap);
        for n in 0..nc {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let (mut best, mut best_flat, mut found) =
                        (T::from_f64(f64::NEG_INFINITY), 0usize, false);
                    for khr in 0..kh {
                        for kwc in 0..kw {
                            let (ih, iw) = (oh * sh + khr, ow * sw + kwc);
                            if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                                let fi = n * h * w + (ih - ph) * w + (iw - pw);
                                let v = Self::get(a, n, (ih - ph) * w + (iw - pw));
                                if !found || crate::scalar::ReductionOps::reduction_gt(v, best) {
                                    best = v;
                                    best_flat = fi;
                                    found = true;
                                }
                            }
                        }
                    }
                    vals.push(if found { best } else { T::zero() });
                    idxs.push(T::from_f64(best_flat as f64));
                }
            }
        }
        (
            Self::from_vec(nc, out_h * out_w, vals),
            Self::from_vec(nc, out_h * out_w, idxs),
        )
    }

    /// 2-D average pooling.
    #[allow(clippy::too_many_arguments)]
    fn avg_pool2d<T: Scalar>(
        a: &Self::Storage<T>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> Self::Storage<T> {
        let nc = Self::nrows(a);
        let (out_h, out_w) = ((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1);
        Self::from_fn(nc, out_h * out_w, |n, op| {
            let (oh, ow) = (op / out_w, op % out_w);
            let (mut sum, mut cnt) = (T::zero(), 0usize);
            for khr in 0..kh {
                for kwc in 0..kw {
                    let (ih, iw) = (oh * sh + khr, ow * sw + kwc);
                    if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                        sum = sum + Self::get(a, n, (ih - ph) * w + (iw - pw));
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                T::zero()
            } else {
                sum / T::from_f64(cnt as f64)
            }
        })
    }

    /// Adaptive average pooling: pools to fixed (out_h, out_w).
    fn adaptive_avg_pool2d<T: Scalar>(
        a: &Self::Storage<T>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Self::Storage<T> {
        let nc = Self::nrows(a);
        Self::from_fn(nc, out_h * out_w, |n, op| {
            let (oh, ow) = (op / out_w, op % out_w);
            let (ih_s, ih_e) = (oh * in_h / out_h, (oh + 1) * in_h / out_h);
            let (iw_s, iw_e) = (ow * in_w / out_w, (ow + 1) * in_w / out_w);
            let (mut sum, mut cnt) = (T::zero(), 0usize);
            for ih in ih_s..ih_e {
                for iw in iw_s..iw_e {
                    sum = sum + Self::get(a, n, ih * in_w + iw);
                    cnt += 1;
                }
            }
            if cnt == 0 {
                T::zero()
            } else {
                sum / T::from_f64(cnt as f64)
            }
        })
    }

    // --- Conv ---
    /// 2-D convolution (without bias).
    #[allow(clippy::too_many_arguments)]
    fn conv2d<T: Scalar>(
        input: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        n: usize,
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
    ) -> Self::Storage<T> {
        let (c_in_g, c_out_g) = (c_in / groups, c_out / groups);
        let out_h = (h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
        let out_w = (w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
        Self::from_fn(n * c_out, out_h * out_w, |row, col| {
            let (b, oc) = (row / c_out, row % c_out);
            let g = oc / c_out_g;
            let (oh, ow) = (col / out_w, col % out_w);
            let mut acc = T::zero();
            for ic in 0..c_in_g {
                for khr in 0..kh {
                    for kwc in 0..kw {
                        let (ih, iw) = (
                            oh * stride.0 + khr * dilation.0,
                            ow * stride.1 + kwc * dilation.1,
                        );
                        if ih >= padding.0
                            && ih < h + padding.0
                            && iw >= padding.1
                            && iw < w + padding.1
                        {
                            acc = acc
                                + Self::get(
                                    input,
                                    b * c_in + g * c_in_g + ic,
                                    (ih - padding.0) * w + (iw - padding.1),
                                ) * Self::get(weight, oc, ic * kh * kw + khr * kw + kwc);
                        }
                    }
                }
            }
            acc
        })
    }

    /// 1-D convolution (im1col-based default).
    #[allow(clippy::too_many_arguments)]
    fn conv1d<T: Scalar>(
        input: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        n_batch: usize,
        c_in: usize,
        length: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Self::Storage<T> {
        assert!(c_in.is_multiple_of(groups) && c_out.is_multiple_of(groups));
        let c_in_g = c_in / groups;
        let out_len = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        Self::from_fn(n_batch * c_out, out_len, |row, col| {
            let (b, oc) = (row / c_out, row % c_out);
            let g = oc / (c_out / groups);
            let mut acc = T::zero();
            for ic in 0..c_in_g {
                for k in 0..kernel_size {
                    let il = col * stride + k * dilation;
                    if il >= padding && il < length + padding {
                        acc = acc
                            + Self::get(input, b * c_in + g * c_in_g + ic, il - padding)
                                * Self::get(weight, oc, ic * kernel_size + k);
                    }
                }
            }
            acc
        })
    }

    /// 3-D convolution (default loop implementation).
    #[allow(clippy::too_many_arguments)]
    fn conv3d<T: Scalar>(
        input: &Self::Storage<T>,
        weight: &Self::Storage<T>,
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
    ) -> Self::Storage<T> {
        assert!(c_in.is_multiple_of(groups) && c_out.is_multiple_of(groups));
        let (c_in_g, c_out_g) = (c_in / groups, c_out / groups);
        let out_d = (d + 2 * padding.0 - dilation.0 * (kd - 1) - 1) / stride.0 + 1;
        let out_h = (h + 2 * padding.1 - dilation.1 * (kh - 1) - 1) / stride.1 + 1;
        let out_w = (w + 2 * padding.2 - dilation.2 * (kw - 1) - 1) / stride.2 + 1;
        Self::from_fn(n_batch * c_out, out_d * out_h * out_w, |row, col| {
            let (b, oc) = (row / c_out, row % c_out);
            let g = oc / c_out_g;
            let od = col / (out_h * out_w);
            let (oh, ow) = ((col / out_w) % out_h, col % out_w);
            let mut acc = T::zero();
            for ic in 0..c_in_g {
                for kdr in 0..kd {
                    for khr in 0..kh {
                        for kwc in 0..kw {
                            let (id, ih, iw) = (
                                od * stride.0 + kdr * dilation.0,
                                oh * stride.1 + khr * dilation.1,
                                ow * stride.2 + kwc * dilation.2,
                            );
                            if id >= padding.0
                                && id < d + padding.0
                                && ih >= padding.1
                                && ih < h + padding.1
                                && iw >= padding.2
                                && iw < w + padding.2
                            {
                                acc = acc
                                    + Self::get(
                                        input,
                                        b * c_in + g * c_in_g + ic,
                                        (id - padding.0) * h * w
                                            + (ih - padding.1) * w
                                            + (iw - padding.2),
                                    ) * Self::get(
                                        weight,
                                        oc,
                                        ic * kd * kh * kw + kdr * kh * kw + khr * kw + kwc,
                                    );
                            }
                        }
                    }
                }
            }
            acc
        })
    }

    /// Transposed 2-D convolution (default loop implementation).
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose2d<T: Scalar>(
        input: &Self::Storage<T>,
        weight: &Self::Storage<T>,
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
    ) -> Self::Storage<T> {
        let out_h = (h - 1) * stride.0 - 2 * padding.0 + kh + output_padding.0;
        let out_w = (w - 1) * stride.1 - 2 * padding.1 + kw + output_padding.1;
        Self::from_fn(n_batch * c_out, out_h * out_w, |row, col| {
            let (b, oc) = (row / c_out, row % c_out);
            let (oh, ow) = (col / out_w, col % out_w);
            let mut acc = T::zero();
            for ic in 0..c_in {
                for khr in 0..kh {
                    for kwc in 0..kw {
                        let (ih_pad, iw_pad) = (oh + padding.0, ow + padding.1);
                        if ih_pad >= khr
                            && iw_pad >= kwc
                            && (ih_pad - khr).is_multiple_of(stride.0)
                            && (iw_pad - kwc).is_multiple_of(stride.1)
                        {
                            let (ih, iw) = ((ih_pad - khr) / stride.0, (iw_pad - kwc) / stride.1);
                            if ih < h && iw < w {
                                acc = acc
                                    + Self::get(input, b * c_in + ic, ih * w + iw)
                                        * Self::get(weight, ic, oc * kh * kw + khr * kw + kwc);
                            }
                        }
                    }
                }
            }
            acc
        })
    }

    // --- Backward activations (GPU-native, no D2H during CUDA Graph capture) ---
    /// ReLU backward: `grad * (input > 0 ? 1 : 0)`.
    fn relu_backward<T: Scalar>(
        grad: &Self::Storage<T>,
        input: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(grad), Self::ncols(grad));
        Self::from_fn(m, n, |r, c| {
            if Self::get(input, r, c).to_f64() > 0.0 {
                Self::get(grad, r, c)
            } else {
                T::zero()
            }
        })
    }
    /// Leaky ReLU backward: `grad * (input > 0 ? 1 : alpha)`.
    fn leaky_relu_backward<T: Scalar>(
        grad: &Self::Storage<T>,
        input: &Self::Storage<T>,
        alpha: T,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(grad), Self::ncols(grad));
        Self::from_fn(m, n, |r, c| {
            let g = Self::get(grad, r, c);
            if Self::get(input, r, c).to_f64() > 0.0 {
                g
            } else {
                alpha * g
            }
        })
    }
    /// ELU backward: `grad * (input > 0 ? 1 : alpha * exp(input))`.
    fn elu_backward<T: Scalar>(
        grad: &Self::Storage<T>,
        input: &Self::Storage<T>,
        alpha: T,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(grad), Self::ncols(grad));
        Self::from_fn(m, n, |r, c| {
            let (g, x) = (Self::get(grad, r, c), Self::get(input, r, c));
            if x.to_f64() > 0.0 {
                g
            } else {
                g * alpha * x.math_exp()
            }
        })
    }
    /// GELU backward.
    fn gelu_backward<T: Scalar>(
        grad: &Self::Storage<T>,
        input: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(grad), Self::ncols(grad));
        let (inv_sqrt2, inv_sqrt_2pi) = (
            T::from_f64(std::f64::consts::FRAC_1_SQRT_2),
            T::from_f64(1.0 / (2.0 * std::f64::consts::PI).sqrt()),
        );
        let (half, neg_half) = (T::from_f64(0.5), T::from_f64(-0.5));
        Self::from_fn(m, n, |r, c| {
            let (g, x) = (Self::get(grad, r, c), Self::get(input, r, c));
            let cdf = half * (T::one() + (x * inv_sqrt2).math_erf());
            let pdf = (neg_half * x * x).math_exp() * inv_sqrt_2pi;
            g * (cdf + x * pdf)
        })
    }
    /// Abs backward: `grad * sign(input)`.
    fn abs_backward<T: Scalar>(
        grad: &Self::Storage<T>,
        input: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(grad), Self::ncols(grad));
        Self::from_fn(m, n, |r, c| {
            let g = Self::get(grad, r, c);
            let x = Self::get(input, r, c).to_f64();
            if x > 0.0 {
                g
            } else if x < 0.0 {
                T::zero() - g
            } else {
                T::zero()
            }
        })
    }

    // --- Transform ---
    /// Walsh-Hadamard Transform: in-place iterative butterfly on each row.
    fn wht<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        wht_impl::<T, Self>(a, false)
    }
    /// Inverse Walsh-Hadamard Transform: butterfly + normalize by 1/cols.
    fn wht_inverse<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        wht_impl::<T, Self>(a, true)
    }

    // --- Attention ---
    /// Scaled dot-product attention.
    #[allow(clippy::too_many_arguments)]
    fn sdpa<T: Scalar>(
        q: &Self::Storage<T>,
        k: &Self::Storage<T>,
        v: &Self::Storage<T>,
        _mask: Option<&Self::Storage<T>>,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        batch_heads: usize,
    ) -> Self::Storage<T> {
        let scale = T::from_f64(1.0 / (head_dim as f64).sqrt());
        let mut out = Self::from_fn(batch_heads * seq_q, head_dim, |_, _| T::zero());
        for bh in 0..batch_heads {
            for i in 0..seq_q {
                let scores: Vec<T> = (0..seq_k)
                    .map(|j| {
                        (0..head_dim).fold(T::zero(), |dot, d| {
                            dot + Self::get(q, bh * seq_q + i, d) * Self::get(k, bh * seq_k + j, d)
                        }) * scale
                    })
                    .collect();
                let max_s = scores
                    .iter()
                    .fold(T::from_f64(f64::NEG_INFINITY), |acc, &x| {
                        if x.to_f64() > acc.to_f64() { x } else { acc }
                    });
                let sum_exp = scores
                    .iter()
                    .fold(T::zero(), |acc, &x| acc + (x - max_s).math_exp());
                let inv = T::one() / sum_exp;
                let weights: Vec<T> = scores
                    .iter()
                    .map(|&x| (x - max_s).math_exp() * inv)
                    .collect();
                for d in 0..head_dim {
                    let val = weights.iter().enumerate().fold(T::zero(), |acc, (j, &w)| {
                        acc + w * Self::get(v, bh * seq_k + j, d)
                    });
                    Self::set(&mut out, bh * seq_q + i, d, val);
                }
            }
        }
        out
    }
}
