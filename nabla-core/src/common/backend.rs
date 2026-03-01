use crate::scalar::Scalar;

/// Error types for nabla backend operations.
pub mod error {
    use core::{fmt, result};

    /// Convenience `Result` alias for nabla operations.
    pub type Result<T> = result::Result<T, Error>;

    /// Errors that can occur in nabla operations.
    #[derive(Debug)]
    pub enum Error {
        /// Matrix shape was not compatible with the operation.
        ShapeMismatch {
            /// Expected shape `(rows, cols)`.
            expected: (usize, usize),
            /// Actual shape `(rows, cols)`.
            got: (usize, usize),
        },
        /// A dimension value was invalid for the given context.
        InvalidDimension(String),
        /// GPU kernel launch or execution failed.
        GpuKernelFailed(String),
        /// Expression evaluation failed (unbound variable, empty context, etc.).
        EvalError(String),
        /// No gradient was computed for this variable.
        NoGradient,
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::ShapeMismatch { expected, got } => write!(
                    f,
                    "shape mismatch: expected ({}, {}), got ({}, {})",
                    expected.0, expected.1, got.0, got.1
                ),
                Self::InvalidDimension(msg) => write!(f, "invalid dimension: {msg}"),
                Self::GpuKernelFailed(msg) => write!(f, "GPU kernel failed: {msg}"),
                Self::EvalError(msg) => write!(f, "eval error: {msg}"),
                Self::NoGradient => write!(f, "no gradient computed for this variable"),
            }
        }
    }

    impl Error {
        /// Create a shape mismatch error.
        #[inline]
        pub fn mismatch(expected: (usize, usize), got: (usize, usize)) -> Self {
            Self::ShapeMismatch { expected, got }
        }

        /// Create an invalid dimension error.
        #[inline]
        pub fn invalid<T: core::fmt::Display>(msg: T) -> Self {
            Self::InvalidDimension(msg.to_string())
        }

        /// Create an eval error.
        #[inline]
        pub fn eval<T: core::fmt::Display>(msg: T) -> Self {
            Self::EvalError(msg.to_string())
        }
    }

    impl std::error::Error for Error {}
}
pub use error::{Error, Result};

#[cfg(feature = "cpu")]
pub use crate::cpu::{Cpu, CpuStorage};

pub(crate) mod private {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------------
// Sub-trait 1: BackendCore — Storage + arithmetic fundamentals (18 methods)
// ---------------------------------------------------------------------------

/// Core storage and arithmetic operations for all backends.
pub trait BackendCore: private::Sealed + Send + Sync + 'static {
    /// Owned storage for a 2-D matrix of element type `T`.
    type Storage<T: Scalar>: Send + Sync;

    /// Allocate a zero-filled matrix.
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> Self::Storage<T>;

    /// Allocate a matrix filled with a constant scalar value.
    fn fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> Self::Storage<T> {
        Self::from_fn(nrows, ncols, |_, _| val)
    }

    /// Allocate an n x n identity matrix.
    #[must_use]
    fn identity<T: Scalar>(n: usize) -> Self::Storage<T> {
        Self::from_fn(n, n, |r, c| if r == c { T::one() } else { T::zero() })
    }

    /// Allocate a matrix and fill it by calling `f(row, col)`.
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> Self::Storage<T>;

    /// Build storage from a pre-allocated row-major `Vec<T>` (zero-copy when possible).
    #[must_use]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> Self::Storage<T> {
        let expected = nrows * ncols;
        assert_eq!(
            data.len(),
            expected,
            "from_vec: data length must equal nrows * ncols"
        );
        let mut i = 0usize;
        Self::from_fn(nrows, ncols, move |_, _| {
            let v = data[i];
            i += 1;
            v
        })
    }

    /// Non-blocking H2D upload: data transfer on a separate copy stream overlaps with compute.
    /// Default falls back to synchronous `from_vec`. GPU backends override for overlap.
    #[must_use]
    fn from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> Self::Storage<T> {
        Self::from_vec(nrows, ncols, data)
    }

    /// Non-blocking D2H transfer: copies tensor data to a `Vec<T>` using the copy stream.
    fn to_vec_async<T: Scalar>(a: &Self::Storage<T>) -> Vec<T> {
        let rows = Self::nrows(a);
        let cols = Self::ncols(a);
        let n = rows * cols;
        let mut out = Vec::with_capacity(n);
        for r in 0..rows {
            for c in 0..cols {
                out.push(Self::get(a, r, c));
            }
        }
        out
    }

    /// Row count of `storage`.
    fn nrows<T: Scalar>(storage: &Self::Storage<T>) -> usize;
    /// Column count of `storage`.
    fn ncols<T: Scalar>(storage: &Self::Storage<T>) -> usize;
    /// Read element at `(row, col)`.
    fn get<T: Scalar>(storage: &Self::Storage<T>, row: usize, col: usize) -> T;
    /// Write element at `(row, col)`.
    fn set<T: Scalar>(storage: &mut Self::Storage<T>, row: usize, col: usize, val: T);

    /// Pre-fetch storage data to host cache (GPU→host bulk transfer).
    /// On CPU backends this is a no-op.
    fn prefetch<T: Scalar>(_storage: &Self::Storage<T>) {}

    /// Block until all pending operations on this backend's device/stream have completed.
    fn sync<T: Scalar>(_storage: &Self::Storage<T>) {}

    /// Clone storage.
    fn clone_storage<T: Scalar>(storage: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise addition.
    fn add<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise subtraction.
    fn sub<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise negation.
    fn neg<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Transpose: result has shape `(ncols(a), nrows(a))`.
    fn transpose<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Scalar multiply: every element of `a` multiplied by `s`.
    fn scale<T: Scalar>(a: &Self::Storage<T>, s: T) -> Self::Storage<T>;

    /// In-place axpy: `y[i] += alpha * x[i]`. Zero allocation, single kernel.
    fn axpy_inplace<T: Scalar>(y: &mut Self::Storage<T>, alpha: T, x: &Self::Storage<T>);

    /// GPU-native broadcast expand: write `src` (src_rows x src_cols) into `out` (dst_rows x dst_cols).
    fn expand_into<T: Scalar>(
        out: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_rows: usize,
        src_cols: usize,
    ) {
        let (dst_rows, dst_cols) = (Self::nrows(out), Self::ncols(out));
        *out = Self::from_fn(dst_rows, dst_cols, |r, c| {
            Self::get(
                src,
                if src_rows == 1 { 0 } else { r },
                if src_cols == 1 { 0 } else { c },
            )
        });
    }
}

// ---------------------------------------------------------------------------
// Sub-trait 2: BackendMath — Element-wise math (28 methods)
// ---------------------------------------------------------------------------

/// Element-wise math operations (exp, sin, cos, etc.).
pub trait BackendMath: BackendCore {
    /// Element-wise exponential.
    fn exp<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise natural logarithm.
    fn ln<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise `ln(1 + x)`.
    fn log1p<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise sine.
    fn sin<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise cosine.
    fn cos<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise tangent.
    fn tan<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise hyperbolic tangent.
    fn tanh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise square root.
    fn sqrt<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise absolute value.
    fn abs<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise reciprocal `1/x`.
    fn recip<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise Gauss error function.
    fn erf<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise ceiling.
    fn ceil<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise floor.
    fn floor<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise round to nearest integer.
    fn round<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise arc sine.
    fn asin<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise arc cosine.
    fn acos<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise arc tangent.
    fn atan<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise two-argument arc tangent.
    fn atan2<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise hyperbolic sine.
    fn sinh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise hyperbolic cosine.
    fn cosh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise inverse hyperbolic sine.
    fn asinh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise inverse hyperbolic cosine.
    fn acosh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise inverse hyperbolic tangent.
    fn atanh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise base-2 logarithm.
    fn log2<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise base-10 logarithm.
    fn log10<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise power `a[i,j]^p`.
    fn powf<T: Scalar>(a: &Self::Storage<T>, p: T) -> Self::Storage<T>;
    /// Element-wise multiplication `a[i,j] * b[i,j]`.
    fn emul<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;
    /// Element-wise division `a[i,j] / b[i,j]`.
    fn ediv<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;
}

// ---------------------------------------------------------------------------
// Sub-trait 3: BackendReduce — Reductions (12 methods)
// ---------------------------------------------------------------------------

/// Reduction operations (sum, max, min, norm, etc.).
pub trait BackendReduce: BackendCore {
    /// Sum all elements of the matrix.
    fn sum_all<T: Scalar>(a: &Self::Storage<T>) -> T;
    /// Element with the maximum value.
    fn max_all<T: Scalar>(a: &Self::Storage<T>) -> T;
    /// Element with the minimum value.
    fn min_all<T: Scalar>(a: &Self::Storage<T>) -> T;
    /// `(row, col)` of the element with the maximum value.
    fn argmax_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);
    /// `(row, col)` of the element with the minimum value.
    fn argmin_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);

    /// Product of all elements.
    fn prod_all<T: Scalar>(a: &Self::Storage<T>) -> T {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let mut acc = T::one();
        for r in 0..rows {
            for c in 0..cols {
                acc = acc * Self::get(a, r, c);
            }
        }
        acc
    }

    /// Count of elements not equal to zero.
    fn count_nonzero<T: Scalar>(a: &Self::Storage<T>) -> usize {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let mut cnt = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                if Self::get(a, r, c).to_f64() != 0.0 {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    /// Sum along axis=1 (columns): (nrows, ncols) -> (nrows, 1).
    fn sum_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Max along axis=1: (nrows, ncols) -> (nrows, 1).
    fn max_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Cumulative sum along axis 1 (row-wise prefix sum). Same shape output.
    fn cumsum_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let mut data = vec![T::zero(); rows * cols];
        for r in 0..rows {
            let mut acc = T::zero();
            for c in 0..cols {
                acc = acc + Self::get(a, r, c);
                data[r * cols + c] = acc;
            }
        }
        Self::from_vec(rows, cols, data)
    }

    /// Cumulative product along axis 1 (row-wise prefix product). Same shape output.
    fn cumprod_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let mut data = vec![T::zero(); rows * cols];
        for r in 0..rows {
            let mut acc = T::one();
            for c in 0..cols {
                acc = acc * Self::get(a, r, c);
                data[r * cols + c] = acc;
            }
        }
        Self::from_vec(rows, cols, data)
    }

    /// Fused MSE sum forward: `sum((pred-target)^2)` → (1,1) storage.
    fn mse_sum_fwd<T: Scalar>(
        pred: &Self::Storage<T>,
        target: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(pred), Self::ncols(pred));
        let mut sum = T::zero();
        for r in 0..rows {
            for c in 0..cols {
                let d = Self::get(pred, r, c) - Self::get(target, r, c);
                sum = sum + d * d;
            }
        }
        Self::fill(1, 1, sum)
    }

    /// Fused MSE sum backward: `out[i] = 2*(pred[i]-target[i])*grad`.
    fn mse_sum_bwd<T: Scalar>(
        pred: &Self::Storage<T>,
        target: &Self::Storage<T>,
        grad: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(pred), Self::ncols(pred));
        let two_g = T::from_f64(2.0) * Self::get(grad, 0, 0);
        Self::from_fn(rows, cols, |r, c| {
            (Self::get(pred, r, c) - Self::get(target, r, c)) * two_g
        })
    }

    /// Lp norm: `(sum |x_i|^p)^(1/p)`, or `max|x_i|` for p=inf.
    fn norm_lp<T: Scalar>(a: &Self::Storage<T>, p: T) -> T {
        let p_f64 = p.to_f64();
        if p_f64.is_infinite() && p_f64 > 0.0 {
            let (rows, cols) = (Self::nrows(a), Self::ncols(a));
            let mut max_val = T::zero();
            for r in 0..rows {
                for c in 0..cols {
                    let v = Self::get(a, r, c).math_abs();
                    if crate::scalar::ReductionOps::reduction_gt(v, max_val) {
                        max_val = v;
                    }
                }
            }
            return max_val;
        }
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let mut sum = T::zero();
        for r in 0..rows {
            for c in 0..cols {
                sum = sum + Self::get(a, r, c).math_abs().math_powf(p);
            }
        }
        sum.math_powf(T::one() / p)
    }
}

// ---------------------------------------------------------------------------
// Sub-trait 4: BackendBlas — Matrix multiply + batched (5 methods)
// ---------------------------------------------------------------------------

/// Matrix multiplication and batched BLAS operations.
pub trait BackendBlas: BackendCore {
    /// Compute `out = a * b`, overwriting `out`.
    fn matmul_into<T: Scalar>(
        out: &mut Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    );

    /// Compute `out = a^T * b` (transpose first operand). Default: transpose + matmul.
    fn matmul_tn_into<T: Scalar>(
        out: &mut Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    ) {
        let a_t = Self::transpose(a);
        Self::matmul_into(out, &a_t, b);
    }

    /// Compute `out = a * b^T` (transpose second operand). Default: transpose + matmul.
    fn matmul_nt_into<T: Scalar>(
        out: &mut Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    ) {
        let b_t = Self::transpose(b);
        Self::matmul_into(out, a, &b_t);
    }

    /// Fused GEMM + epilogue activation in a single dispatch.
    fn matmul_epilogue<T: Scalar>(
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
        epilogue_id: u8,
    ) -> Self::Storage<T>
    where
        Self: BackendMath + BackendNN,
    {
        let m = Self::nrows(a);
        let n = Self::ncols(b);
        let mut out = Self::zeros(m, n);
        Self::matmul_into(&mut out, a, b);
        let two = T::one() + T::one();
        match epilogue_id {
            // ReLU: (x + |x|) / 2
            0 => Self::from_fn(m, n, |r, c| {
                let x = Self::get(&out, r, c);
                (x + x.math_abs()) / two
            }),
            // GELU (tanh approximation)
            1 => {
                let half = T::from_f64(0.5);
                let k = T::from_f64(0.797_884_560_8);
                let c = T::from_f64(0.044_715);
                Self::from_fn(m, n, |r, col| {
                    let x = Self::get(&out, r, col);
                    let inner = k * (x + c * x * x * x);
                    half * x * (T::one() + inner.math_tanh())
                })
            }
            _ => out,
        }
    }

    /// Batched matrix multiply: `C[b] = A[b] @ B[b]`.
    fn bmm<T: Scalar>(
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage<T> {
        Self::from_fn(batch * m, n, |r, c| {
            let bi = r / m;
            let i = r % m;
            (0..k).fold(T::zero(), |acc, j| {
                acc + Self::get(a, bi * m + i, j) * Self::get(b, bi * k + j, c)
            })
        })
    }

    /// `C = beta * self + alpha * (A @ B)`.
    fn addmm<T: Scalar>(
        c: &Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
        beta: T,
        alpha: T,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(c), Self::ncols(c));
        let k = Self::ncols(a);
        Self::from_fn(m, n, |r, col| {
            let ab = (0..k).fold(T::zero(), |acc, j| {
                acc + Self::get(a, r, j) * Self::get(b, j, col)
            });
            beta * Self::get(c, r, col) + alpha * ab
        })
    }

    /// Batched addmm: `C = beta * C + alpha * (A[b] @ B[b])`.
    #[allow(clippy::too_many_arguments)]
    fn baddbmm<T: Scalar>(
        c: &Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        beta: T,
        alpha: T,
    ) -> Self::Storage<T> {
        Self::from_fn(batch * m, n, |r, col| {
            let bi = r / m;
            let i = r % m;
            let ab = (0..k).fold(T::zero(), |acc, j| {
                acc + Self::get(a, bi * m + i, j) * Self::get(b, bi * k + j, col)
            });
            beta * Self::get(c, r, col) + alpha * ab
        })
    }
}

// ---------------------------------------------------------------------------
// Sub-trait 5: BackendNN — NN ops + conv + pooling + attention (20 methods)
// ---------------------------------------------------------------------------

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
        let rows = Self::nrows(a);
        let cols = Self::ncols(a);
        let n_f = T::from_f64(rows as f64);
        let one = T::from_f64(1.0);
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
            let m: Vec<T> = (0..cols).map(|c| Self::get(running_mean, 0, c)).collect();
            let v: Vec<T> = (0..cols).map(|c| Self::get(running_var, 0, c)).collect();
            (m, v)
        };
        if training {
            let one_minus = one - momentum;
            for c in 0..cols {
                let rm = Self::get(running_mean, 0, c);
                let rv = Self::get(running_var, 0, c);
                Self::set(running_mean, 0, c, one_minus * rm + momentum * mean[c]);
                Self::set(running_var, 0, c, one_minus * rv + momentum * var[c]);
            }
        }
        Self::from_fn(rows, cols, |r, c| {
            let x = Self::get(a, r, c);
            let w = Self::get(gamma, 0, c);
            let b = Self::get(beta, 0, c);
            (x - mean[c]) / (var[c] + eps).math_sqrt() * w + b
        })
    }

    /// Cross-entropy loss: fused softmax + NLL.
    fn cross_entropy_fused<T: Scalar>(
        input: &Self::Storage<T>,
        target: &Self::Storage<T>,
        n: usize,
        c: usize,
    ) -> Self::Storage<T> {
        let mut total = T::zero();
        for i in 0..n {
            let row_max = (0..c).fold(T::from_f64(f64::NEG_INFINITY), |acc, j| {
                let v = Self::get(input, i, j);
                if v.to_f64() > acc.to_f64() { v } else { acc }
            });
            let sum_exp = (0..c).fold(T::zero(), |acc, j| {
                acc + (Self::get(input, i, j) - row_max).math_exp()
            });
            let t = Self::get(target, i, 0).to_f64() as usize;
            total = total + -(Self::get(input, i, t) - row_max - sum_exp.math_ln());
        }
        let mean = total / T::from_f64(n as f64);
        Self::from_fn(1, 1, |_, _| mean)
    }

    // --- Embedding ---
    /// Embedding gather: indices (n_tokens, 1) float-encoded, weight (vocab, embed_dim).
    fn embedding<T: Scalar>(
        indices: &Self::Storage<T>,
        weight: &Self::Storage<T>,
    ) -> Self::Storage<T>;

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
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        Self::from_fn(nc, out_h * out_w, |n, op| {
            let oh = op / out_w;
            let ow = op % out_w;
            let mut best = T::from_f64(f64::NEG_INFINITY);
            let mut found = false;
            for khr in 0..kh {
                for kwc in 0..kw {
                    let ih = oh * sh + khr;
                    let iw = ow * sw + kwc;
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
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        let mut vals = Vec::with_capacity(nc * out_h * out_w);
        let mut idxs = Vec::with_capacity(nc * out_h * out_w);
        for n in 0..nc {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut best = T::from_f64(f64::NEG_INFINITY);
                    let mut best_flat = 0usize;
                    let mut found = false;
                    for khr in 0..kh {
                        for kwc in 0..kw {
                            let ih = oh * sh + khr;
                            let iw = ow * sw + kwc;
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
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        Self::from_fn(nc, out_h * out_w, |n, op| {
            let oh = op / out_w;
            let ow = op % out_w;
            let mut sum = T::zero();
            let mut cnt = 0usize;
            for khr in 0..kh {
                for kwc in 0..kw {
                    let ih = oh * sh + khr;
                    let iw = ow * sw + kwc;
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
            let oh = op / out_w;
            let ow = op % out_w;
            let ih_start = oh * in_h / out_h;
            let ih_end = (oh + 1) * in_h / out_h;
            let iw_start = ow * in_w / out_w;
            let iw_end = (ow + 1) * in_w / out_w;
            let mut sum = T::zero();
            let mut cnt = 0usize;
            for ih in ih_start..ih_end {
                for iw in iw_start..iw_end {
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
        let c_in_g = c_in / groups;
        let c_out_g = c_out / groups;
        let out_h = (h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
        let out_w = (w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
        let out_spatial = out_h * out_w;
        Self::from_fn(n * c_out, out_spatial, |row, col| {
            let b = row / c_out;
            let oc = row % c_out;
            let g = oc / c_out_g;
            let oh = col / out_w;
            let ow = col % out_w;
            let mut acc = T::zero();
            for ic in 0..c_in_g {
                for khr in 0..kh {
                    for kwc in 0..kw {
                        let ih = oh * stride.0 + khr * dilation.0;
                        let iw = ow * stride.1 + kwc * dilation.1;
                        if ih >= padding.0
                            && ih < h + padding.0
                            && iw >= padding.1
                            && iw < w + padding.1
                        {
                            let x = Self::get(
                                input,
                                b * c_in + g * c_in_g + ic,
                                (ih - padding.0) * w + (iw - padding.1),
                            );
                            let wt = Self::get(weight, oc, ic * kh * kw + khr * kw + kwc);
                            acc = acc + x * wt;
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
            let b = row / c_out;
            let oc = row % c_out;
            let g = oc / (c_out / groups);
            let mut acc = T::zero();
            for ic in 0..c_in_g {
                for k in 0..kernel_size {
                    let il = col * stride + k * dilation;
                    if il >= padding && il < length + padding {
                        let x_val = Self::get(input, b * c_in + g * c_in_g + ic, il - padding);
                        let w_val = Self::get(weight, oc, ic * kernel_size + k);
                        acc = acc + x_val * w_val;
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
        let c_in_g = c_in / groups;
        let c_out_g = c_out / groups;
        let out_d = (d + 2 * padding.0 - dilation.0 * (kd - 1) - 1) / stride.0 + 1;
        let out_h = (h + 2 * padding.1 - dilation.1 * (kh - 1) - 1) / stride.1 + 1;
        let out_w = (w + 2 * padding.2 - dilation.2 * (kw - 1) - 1) / stride.2 + 1;
        let out_spatial = out_d * out_h * out_w;
        Self::from_fn(n_batch * c_out, out_spatial, |row, col| {
            let b = row / c_out;
            let oc = row % c_out;
            let g = oc / c_out_g;
            let od = col / (out_h * out_w);
            let oh = (col / out_w) % out_h;
            let ow = col % out_w;
            let mut acc = T::zero();
            for ic in 0..c_in_g {
                for kdr in 0..kd {
                    for khr in 0..kh {
                        for kwc in 0..kw {
                            let id = od * stride.0 + kdr * dilation.0;
                            let ih = oh * stride.1 + khr * dilation.1;
                            let iw = ow * stride.2 + kwc * dilation.2;
                            if id >= padding.0
                                && id < d + padding.0
                                && ih >= padding.1
                                && ih < h + padding.1
                                && iw >= padding.2
                                && iw < w + padding.2
                            {
                                let x_val = Self::get(
                                    input,
                                    b * c_in + g * c_in_g + ic,
                                    (id - padding.0) * h * w
                                        + (ih - padding.1) * w
                                        + (iw - padding.2),
                                );
                                let w_val = Self::get(
                                    weight,
                                    oc,
                                    ic * kd * kh * kw + kdr * kh * kw + khr * kw + kwc,
                                );
                                acc = acc + x_val * w_val;
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
            let b = row / c_out;
            let oc = row % c_out;
            let oh = col / out_w;
            let ow = col % out_w;
            let mut acc = T::zero();
            for ic in 0..c_in {
                for khr in 0..kh {
                    for kwc in 0..kw {
                        let ih_pad = oh + padding.0;
                        let iw_pad = ow + padding.1;
                        if ih_pad >= khr
                            && iw_pad >= kwc
                            && (ih_pad - khr).is_multiple_of(stride.0)
                            && (iw_pad - kwc).is_multiple_of(stride.1)
                        {
                            let ih = (ih_pad - khr) / stride.0;
                            let iw = (iw_pad - kwc) / stride.1;
                            if ih < h && iw < w {
                                let x_val = Self::get(input, b * c_in + ic, ih * w + iw);
                                let w_val = Self::get(weight, ic, oc * kh * kw + khr * kw + kwc);
                                acc = acc + x_val * w_val;
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
            let g = Self::get(grad, r, c);
            let x = Self::get(input, r, c);
            if x.to_f64() > 0.0 {
                g
            } else {
                g * alpha * x.math_exp()
            }
        })
    }
    /// GELU backward: `grad * (cdf + x * pdf)` where cdf = 0.5*(1+erf(x/sqrt(2))), pdf = exp(-x^2/2)/sqrt(2*pi).
    fn gelu_backward<T: Scalar>(
        grad: &Self::Storage<T>,
        input: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (m, n) = (Self::nrows(grad), Self::ncols(grad));
        let inv_sqrt2 = T::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        let inv_sqrt_2pi = T::from_f64(1.0 / (2.0 * std::f64::consts::PI).sqrt());
        let half = T::from_f64(0.5);
        let neg_half = T::from_f64(-0.5);
        Self::from_fn(m, n, |r, c| {
            let g = Self::get(grad, r, c);
            let x = Self::get(input, r, c);
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
                let mut scores = vec![T::zero(); seq_k];
                for (j, score_slot) in scores.iter_mut().enumerate() {
                    let mut dot = T::zero();
                    for d in 0..head_dim {
                        dot =
                            dot + Self::get(q, bh * seq_q + i, d) * Self::get(k, bh * seq_k + j, d);
                    }
                    *score_slot = dot * scale;
                }
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

// ---------------------------------------------------------------------------
// Sub-trait 6: BackendFusion — Kernel fusion (3 methods)
// ---------------------------------------------------------------------------

/// Kernel fusion: element-wise fuse, mega-fuse, and map-reduce fusion.
pub trait BackendFusion: BackendCore {
    /// Launch a fused element-wise kernel.
    #[allow(clippy::too_many_arguments)]
    fn fuse_launch<T: Scalar>(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reg_estimate: usize,
    ) -> Self::Storage<T>;

    /// Launch a mega-fused kernel.
    fn mega_fuse_launch<'a, T: Scalar>(
        ops: &[(Vec<*const u8>, String, usize, bool)],
        nrows: usize,
        ncols: usize,
        cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T + 'a>>,
        kernel_hash: &str,
    ) -> Vec<Self::Storage<T>>;

    /// Fused map-reduce: element-wise expression + axis reduction in a single pass.
    #[allow(clippy::too_many_arguments)]
    fn fuse_reduce_launch<T: Scalar>(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reduce_op: u8,
        axis: u8,
    ) -> Self::Storage<T>
    where
        Self: BackendReduce,
    {
        let intermediate = Self::fuse_launch::<T>(
            inputs,
            nrows,
            ncols,
            cpu_fn,
            gpu_expr,
            kernel_hash,
            n_inputs,
            0,
        );
        let summed = match axis {
            0 => {
                let t = Self::transpose(&intermediate);
                let s = Self::sum_axis1(&t);
                Self::transpose(&s)
            }
            _ => Self::sum_axis1(&intermediate),
        };
        match reduce_op {
            3 => {
                let count = match axis {
                    0 => nrows,
                    _ => ncols,
                };
                let inv_n = T::from_f64(1.0 / count as f64);
                Self::scale(&summed, inv_n)
            }
            _ => summed,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend = supertrait of all 6 sub-traits + blanket impl
// ---------------------------------------------------------------------------

/// Unified backend supertrait combining all six sub-traits.
pub trait Backend:
    BackendCore + BackendMath + BackendReduce + BackendBlas + BackendNN + BackendFusion
{
}

impl<B: BackendCore + BackendMath + BackendReduce + BackendBlas + BackendNN + BackendFusion> Backend
    for B
{
}

// ---------------------------------------------------------------------------
// Struct definitions + Sealed impls (unchanged)
// ---------------------------------------------------------------------------

/// WebGPU backend using wgpu compute shaders.
#[cfg(feature = "gpu")]
pub struct Gpu;

#[cfg(feature = "gpu")]
impl private::Sealed for Gpu {}

/// CUDA backend using cudarc and cuBLAS.
#[cfg(feature = "cuda")]
pub struct Cuda;

/// AMD HIP backend using ROCm.
#[cfg(feature = "hip")]
pub struct Hip;

/// Default backend selected at compile time based on enabled features.
#[cfg(feature = "cuda")]
pub type DefaultBackend = Cuda;

/// Default backend selected at compile time based on enabled features.
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub type DefaultBackend = Hip;

/// Default backend selected at compile time based on enabled features.
#[cfg(all(feature = "gpu", not(feature = "cuda"), not(feature = "hip")))]
pub type DefaultBackend = Gpu;

/// Default backend selected at compile time based on enabled features.
#[cfg(all(
    feature = "cpu",
    not(feature = "cuda"),
    not(feature = "hip"),
    not(feature = "gpu")
))]
pub type DefaultBackend = Cpu;
