use crate::scalar::{Fp8E4M3, Fp8E5M2, Scalar};

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

    /// Allocate an uninitialized matrix — caller **must** overwrite every element
    /// before reading (e.g. cuBLAS beta=0). Default falls back to `zeros` (safe).
    fn empty<T: Scalar>(nrows: usize, ncols: usize) -> Self::Storage<T> {
        Self::zeros(nrows, ncols)
    }

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

    /// Build a one-hot matrix from class indices stored in a `(nrows x 1)` column.
    fn one_hot_from_indices<T: Scalar>(
        indices: &Self::Storage<T>,
        n_classes: usize,
    ) -> Self::Storage<T> {
        let rows = Self::nrows(indices);
        Self::from_fn(rows, n_classes, |r, c| {
            let idx = Self::get(indices, r, 0).to_f64() as usize;
            if c == idx { T::one() } else { T::zero() }
        })
    }

    /// Build storage from a pre-allocated row-major `Vec<T>` (zero-copy when possible).
    #[must_use]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> Self::Storage<T> {
        assert_eq!(
            data.len(),
            nrows * ncols,
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
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| Self::get(a, r, c)))
            .collect()
    }

    /// Cast storage element type using `f64` as an intermediate (default: host loop).
    fn cast<T: Scalar, U: Scalar>(a: &Self::Storage<T>) -> Self::Storage<U> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        Self::from_fn(rows, cols, |r, c| U::from_f64(Self::get(a, r, c).to_f64()))
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

    /// GPU device pointer for this storage (0 for CPU backends).
    fn device_ptr<T: Scalar>(_storage: &Self::Storage<T>) -> u64 {
        0
    }

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

    /// Replace elements where `mask` is non-zero with `value`.
    fn masked_fill<T: Scalar>(
        a: &Self::Storage<T>,
        mask: &Self::Storage<T>,
        value: T,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        Self::from_fn(rows, cols, |r, c| {
            if Self::get(mask, r, c).to_f64() == 0.0 {
                Self::get(a, r, c)
            } else {
                value
            }
        })
    }

    /// Element-wise conditional: `where cond != 0, pick a, else pick b`.
    fn where_cond<T: Scalar>(
        a: &Self::Storage<T>,
        cond: &Self::Storage<T>,
        b: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        Self::from_fn(rows, cols, |r, c| {
            if Self::get(cond, r, c).to_f64() != 0.0 {
                Self::get(a, r, c)
            } else {
                Self::get(b, r, c)
            }
        })
    }
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
    /// GPU-resident sum: returns a 1x1 Storage instead of scalar (avoids DtoH sync).
    /// Default: falls back to `sum_all` + `from_vec`.
    fn sum_all_1x1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        Self::from_vec(1, 1, vec![Self::sum_all(a)])
    }
    /// GPU-resident max: returns a 1x1 Storage instead of scalar (avoids DtoH sync).
    fn max_all_1x1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        Self::from_vec(1, 1, vec![Self::max_all(a)])
    }
    /// GPU-resident min: returns a 1x1 Storage instead of scalar (avoids DtoH sync).
    fn min_all_1x1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        Self::from_vec(1, 1, vec![Self::min_all(a)])
    }

    /// `(row, col)` of the element with the maximum value.
    fn argmax_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);
    /// `(row, col)` of the element with the minimum value.
    fn argmin_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);

    /// Extract diagonal as an `nx1` column vector.
    fn diag<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let n = Self::nrows(a).min(Self::ncols(a));
        Self::from_fn(n, 1, |i, _| Self::get(a, i, i))
    }

    /// Sum of diagonal elements.
    fn trace<T: Scalar>(a: &Self::Storage<T>) -> T {
        let n = Self::nrows(a).min(Self::ncols(a));
        (0..n).fold(T::zero(), |acc, i| acc + Self::get(a, i, i))
    }

    /// Product of all elements.
    fn prod_all<T: Scalar>(a: &Self::Storage<T>) -> T {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .fold(T::one(), |acc, (r, c)| acc * Self::get(a, r, c))
    }

    /// Count of elements not equal to zero.
    fn count_nonzero<T: Scalar>(a: &Self::Storage<T>) -> usize {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .filter(|&(r, c)| Self::get(a, r, c).to_f64() != 0.0)
            .count()
    }

    /// Sum along axis=1 (columns): (nrows, ncols) -> (nrows, 1).
    fn sum_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;
    /// Max along axis=1: (nrows, ncols) -> (nrows, 1).
    fn max_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Cumulative sum along axis 1 (row-wise prefix sum). Same shape output.
    fn cumsum_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        cum_scan::<T, Self>(a, T::zero(), |a, b| a + b)
    }

    /// Cumulative product along axis 1 (row-wise prefix product). Same shape output.
    fn cumprod_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        cum_scan::<T, Self>(a, T::one(), |a, b| a * b)
    }

    /// Fused MSE sum forward: `sum((pred-target)^2)` -> (1,1) storage.
    fn mse_sum_fwd<T: Scalar>(
        pred: &Self::Storage<T>,
        target: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(pred), Self::ncols(pred));
        let sum = (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c))).fold(
            T::zero(),
            |acc, (r, c)| {
                let d = Self::get(pred, r, c) - Self::get(target, r, c);
                acc + d * d
            },
        );
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
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        let elems = (0..rows).flat_map(|r| (0..cols).map(move |c| Self::get(a, r, c).math_abs()));
        let p_f64 = p.to_f64();
        if p_f64.is_infinite() && p_f64 > 0.0 {
            return elems.fold(T::zero(), crate::scalar::ReductionOps::reduction_max);
        }
        elems
            .fold(T::zero(), |acc, v| acc + v.math_powf(p))
            .math_powf(T::one() / p)
    }
}

// ---------------------------------------------------------------------------
// Sub-trait 4: BackendShape — Shape/indexing/sort ops (18 methods)
// ---------------------------------------------------------------------------

/// Shape manipulation and indexing ops. GPU backends must provide kernels.
pub trait BackendShape: BackendCore {
    /// Zero-copy reshape: change metadata only, no allocation or copy.
    /// Valid for contiguous row-major storage (always true in nabla).
    fn reshape_metadata<T: Scalar>(_a: &mut Self::Storage<T>, _new_rows: usize, _new_cols: usize) {
        panic!("nabla: reshape_metadata not implemented for this backend");
    }

    /// Copy-reshape to `(out_rows, out_cols)` preserving row-major order.
    fn reshape_copy<T: Scalar>(
        _a: &Self::Storage<T>,
        _out_rows: usize,
        _out_cols: usize,
    ) -> Self::Storage<T> {
        panic!("nabla: reshape_copy not implemented for this backend");
    }

    /// Return a contiguous copy (same shape).
    fn contiguous<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let (rows, cols) = (Self::nrows(a), Self::ncols(a));
        Self::reshape_copy(a, rows, cols)
    }

    /// Extract submatrix `[row_start, row_start+out_rows) x [col_start, col_start+out_cols)`.
    fn submatrix<T: Scalar>(
        _a: &Self::Storage<T>,
        _row_start: usize,
        _col_start: usize,
        _out_rows: usize,
        _out_cols: usize,
    ) -> Self::Storage<T> {
        panic!("nabla: submatrix not implemented for this backend");
    }

    /// Write `src` into `dst` at `(row_start, col_start)`.
    fn slice_set<T: Scalar>(
        _dst: &mut Self::Storage<T>,
        _row_start: usize,
        _col_start: usize,
        _src: &Self::Storage<T>,
    ) {
        panic!("nabla: slice_set not implemented for this backend");
    }

    /// Repeat rows/cols by the given factors.
    fn repeat<T: Scalar>(
        _a: &Self::Storage<T>,
        _row_reps: usize,
        _col_reps: usize,
    ) -> Self::Storage<T> {
        panic!("nabla: repeat not implemented for this backend");
    }

    /// Pad with constant value.
    fn pad<T: Scalar>(
        _a: &Self::Storage<T>,
        _left: usize,
        _right: usize,
        _top: usize,
        _bottom: usize,
        _value: T,
    ) -> Self::Storage<T> {
        panic!("nabla: pad not implemented for this backend");
    }

    /// Upper-triangular mask.
    fn triu<T: Scalar>(_a: &Self::Storage<T>, _diagonal: isize) -> Self::Storage<T> {
        panic!("nabla: triu not implemented for this backend");
    }

    /// Lower-triangular mask.
    fn tril<T: Scalar>(_a: &Self::Storage<T>, _diagonal: isize) -> Self::Storage<T> {
        panic!("nabla: tril not implemented for this backend");
    }

    /// Roll along axis.
    fn roll<T: Scalar>(_a: &Self::Storage<T>, _shift: isize, _axis: usize) -> Self::Storage<T> {
        panic!("nabla: roll not implemented for this backend");
    }

    /// Flip along axis.
    fn flip<T: Scalar>(_a: &Self::Storage<T>, _axis: usize) -> Self::Storage<T> {
        panic!("nabla: flip not implemented for this backend");
    }

    /// Diagonal matrix from vector.
    fn from_diag<T: Scalar>(_v: &Self::Storage<T>) -> Self::Storage<T> {
        panic!("nabla: from_diag not implemented for this backend");
    }

    /// Gather rows by index vector.
    fn gather_rows<T: Scalar>(_a: &Self::Storage<T>, _indices: &[usize]) -> Self::Storage<T> {
        panic!("nabla: gather_rows not implemented for this backend");
    }

    /// Gather along axis using index tensor.
    fn gather<T: Scalar>(
        _a: &Self::Storage<T>,
        _axis: usize,
        _index: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        panic!("nabla: gather not implemented for this backend");
    }

    /// Scatter along axis using index tensor.
    fn scatter<T: Scalar>(
        _a: &Self::Storage<T>,
        _axis: usize,
        _index: &Self::Storage<T>,
        _src: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        panic!("nabla: scatter not implemented for this backend");
    }

    /// Select along axis using 1-D index tensor.
    fn index_select<T: Scalar>(
        _a: &Self::Storage<T>,
        _axis: usize,
        _index: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        panic!("nabla: index_select not implemented for this backend");
    }

    /// Sort rows (axis=1). Returns `(values, indices)` with indices stored as `T`.
    fn sort_rows<T: Scalar>(
        _a: &Self::Storage<T>,
        _descending: bool,
    ) -> (Self::Storage<T>, Self::Storage<T>) {
        panic!("nabla: sort_rows not implemented for this backend");
    }

    /// Top-k per row (axis=1). Returns `(values[rows×k], indices[rows×k])` descending.
    /// O(n·k) on GPU vs O(n²) full sort. Default falls back to sort_rows + slice.
    fn topk_rows<T: Scalar>(
        _a: &Self::Storage<T>,
        _k: usize,
    ) -> (Self::Storage<T>, Self::Storage<T>) {
        panic!("nabla: topk_rows not implemented for this backend");
    }

    /// Meshgrid for 1-D `x` and `y`. Returns `(grid_x, grid_y)`.
    fn meshgrid<T: Scalar>(
        _x: &Self::Storage<T>,
        _y: &Self::Storage<T>,
    ) -> (Self::Storage<T>, Self::Storage<T>) {
        panic!("nabla: meshgrid not implemented for this backend");
    }

    /// Scatter-add along dimension 0.
    fn scatter_add_dim0<T: Scalar>(
        _dst: &mut Self::Storage<T>,
        _indices: &[usize],
        _src: &Self::Storage<T>,
    ) {
        panic!("nabla: scatter_add_dim0 not implemented for this backend");
    }

    /// Scatter-add along arbitrary axis (0=rows, 1=cols).
    fn scatter_add<T: Scalar>(
        dst: &mut Self::Storage<T>,
        axis: usize,
        indices: &[usize],
        src: &Self::Storage<T>,
    ) {
        if axis == 0 {
            Self::scatter_add_dim0(dst, indices, src);
        } else {
            let src_rows = Self::nrows(src);
            let src_cols = Self::ncols(src);
            for r in 0..src_rows {
                for (c, &dst_c) in indices.iter().enumerate().take(src_cols) {
                    let val = Self::get(dst, r, dst_c) + Self::get(src, r, c);
                    Self::set(dst, r, dst_c, val);
                }
            }
        }
    }

    /// Kronecker product: `(m,n) x (p,q)` -> `(m*p, n*q)`.
    fn kron<T: Scalar>(
        _a: &Self::Storage<T>,
        _b: &Self::Storage<T>,
        _m: usize,
        _n: usize,
        _p: usize,
        _q: usize,
    ) -> Self::Storage<T> {
        panic!("nabla: kron not implemented for this backend");
    }
}

// ---------------------------------------------------------------------------
// Sub-trait 5: BackendBlas — Matrix multiply + batched (5 methods)
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

    /// Fused GEMM + bias add: `out[i,j] = (a * b)[i,j] + bias[0,j]`.
    /// Default: matmul then element-wise row broadcast.
    /// CUDA backend overrides with a single cublasLt `CUBLASLT_EPILOGUE_BIAS` call.
    fn matmul_bias<T: Scalar>(
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
        bias: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let m = Self::nrows(a);
        let n = Self::ncols(b);
        let mut out = Self::zeros(m, n);
        Self::matmul_into(&mut out, a, b);
        Self::from_fn(m, n, |r, c| Self::get(&out, r, c) + Self::get(bias, 0, c))
    }

    /// FP8 E4M3 matmul: inputs in Fp8E4M3, output in bf16. Hardware-accelerated on Hopper+.
    #[cfg(any(
        feature = "cpu",
        feature = "cuda",
        feature = "hip",
        feature = "wgpu-f16"
    ))]
    fn fp8_matmul_e4m3(
        a: &Self::Storage<Fp8E4M3>,
        b: &Self::Storage<Fp8E4M3>,
    ) -> Self::Storage<half::bf16> {
        fp8_matmul_default::<Fp8E4M3, Self>(a, b)
    }

    /// FP8 E5M2 matmul: inputs in Fp8E5M2, output in bf16. Hardware-accelerated on Hopper+.
    #[cfg(any(
        feature = "cpu",
        feature = "cuda",
        feature = "hip",
        feature = "wgpu-f16"
    ))]
    fn fp8_matmul_e5m2(
        a: &Self::Storage<Fp8E5M2>,
        b: &Self::Storage<Fp8E5M2>,
    ) -> Self::Storage<half::bf16> {
        fp8_matmul_default::<Fp8E5M2, Self>(a, b)
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

    /// Batched matrix multiply with B transposed: `C[b] = A[b] @ B[b]^T`.
    fn bmm_nt<T: Scalar>(
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
                acc + Self::get(a, bi * m + i, j) * Self::get(b, bi * n + c, j)
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

mod nn;
pub use nn::BackendNN;

/// Default FP8 matmul: cast to f32, matmul, cast to bf16.
#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "hip",
    feature = "wgpu-f16"
))]
fn fp8_matmul_default<F: Scalar, B: BackendBlas + ?Sized>(
    a: &B::Storage<F>,
    b: &B::Storage<F>,
) -> B::Storage<half::bf16> {
    let (m, k) = (B::nrows(a), B::ncols(a));
    let n = B::ncols(b);
    let a_f32: B::Storage<f32> = B::from_fn(m, k, |r, c| B::get(a, r, c).to_f64() as f32);
    let b_f32: B::Storage<f32> = B::from_fn(k, n, |r, c| B::get(b, r, c).to_f64() as f32);
    let mut out_f32 = B::zeros::<f32>(m, n);
    B::matmul_into(&mut out_f32, &a_f32, &b_f32);
    B::from_fn(m, n, |r, c| half::bf16::from_f32(B::get(&out_f32, r, c)))
}

/// Row-wise cumulative scan with given identity and binary op.
fn cum_scan<T: Scalar, B: BackendCore + ?Sized>(
    a: &B::Storage<T>,
    identity: T,
    op: fn(T, T) -> T,
) -> B::Storage<T> {
    let (rows, cols) = (B::nrows(a), B::ncols(a));
    let mut data = vec![T::zero(); rows * cols];
    for r in 0..rows {
        let mut acc = identity;
        for c in 0..cols {
            acc = op(acc, B::get(a, r, c));
            data[r * cols + c] = acc;
        }
    }
    B::from_vec(rows, cols, data)
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
    BackendCore + BackendMath + BackendReduce + BackendShape + BackendBlas + BackendNN + BackendFusion
{
}

impl<
    B: BackendCore
        + BackendMath
        + BackendReduce
        + BackendShape
        + BackendBlas
        + BackendNN
        + BackendFusion,
> Backend for B
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
