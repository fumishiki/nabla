// tensor.rs — Tensor<T, B> struct, constructors, accessors, ops, and Display.
//
// Design notes:
// - Operator overloads are defined on references to avoid moves (Julia semantics).
// - Shape mismatches in Add/Sub/Mul panic with a descriptive message (Option A).
// - Adjoint uses `T::IS_REAL` const to choose between transpose and conj+transpose.
// - `adjoint` delegates element-wise conjugation via `scalar::math_utils::conj`.

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Bound, Index, IndexMut, Mul, Neg, RangeBounds, Sub};

#[cfg(feature = "cpu")]
use rayon::prelude::*;

#[cfg(feature = "cpu")]
use crate::backend::Cpu;
use crate::backend::{Backend, DefaultBackend};
use crate::scalar::Scalar;

/// A 2-D dense matrix backed by a pluggable [`Backend`].
///
/// The default backend is [`crate::backend::Cpu`], which uses nabla's CPU kernels.
/// The optional `Axes` parameter carries named axis types at zero cost.
pub struct Tensor<T: Scalar, B: Backend = DefaultBackend, Axes = ()> {
    storage: B::Storage<T>,
    _axes: PhantomData<fn() -> Axes>,
}

impl<T: Scalar, B: Backend, Axes> Clone for Tensor<T, B, Axes> {
    fn clone(&self) -> Self {
        Self {
            storage: B::clone_storage(&self.storage),
            _axes: PhantomData,
        }
    }
}

// Named axes: zero-cost axis reinterpretation + accessors for any Axes.
impl<T: Scalar, B: Backend, Axes> Tensor<T, B, Axes> {
    /// Reinterpret the phantom axis type (zero-cost, compile-time only).
    #[inline]
    #[must_use]
    pub fn with_axes<NewAxes>(self) -> Tensor<T, B, NewAxes> {
        Tensor { storage: self.storage, _axes: PhantomData }
    }

    /// Erase axis types back to untyped `()`.
    #[inline]
    #[must_use]
    pub fn erase_axes(self) -> Tensor<T, B> {
        Tensor { storage: self.storage, _axes: PhantomData }
    }

    /// Number of rows.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        B::nrows(&self.storage)
    }

    /// Number of columns.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        B::ncols(&self.storage)
    }

    /// Shape as `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Read element at `(row, col)`.
    #[must_use]
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        B::get(&self.storage, row, col)
    }

    /// Block until all pending backend operations (e.g. GPU kernels) have completed.
    /// On GPU backends this flushes the command stream; on CPU this is a no-op.
    /// Use this for accurate throughput benchmarks instead of `get()` to avoid
    /// including the device-to-host transfer cost in measurements.
    pub fn sync(&self) {
        B::sync(&self.storage);
    }

    /// Dimension along a given axis: 0 -> nrows, 1 -> ncols.
    #[must_use]
    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        match axis {
            0 => self.nrows(),
            1 => self.ncols(),
            _ => panic!("Tensor is 2-D: axis must be 0 or 1, got {axis}"),
        }
    }

    /// Opaque pointer to internal storage (for fuse! macro GPU codegen).
    #[doc(hidden)]
    #[inline]
    pub fn __storage_ptr(&self) -> *const u8 {
        &self.storage as *const B::Storage<T> as *const u8
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Fused element-wise kernel launch (for fuse! macro codegen).
    ///
    /// GPU backends JIT-compile the expression; CPU backends use the closure.
    #[doc(hidden)]
    #[inline]
    pub fn __fuse_elementwise(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reg_estimate: usize,
    ) -> Self {
        Self::from_storage(B::fuse_launch(inputs, nrows, ncols, cpu_fn, gpu_expr, kernel_hash, n_inputs, reg_estimate))
    }

    /// Execute multiple fused element-wise operations as a **single** GPU
    /// kernel launch (mega-kernel fusion).
    ///
    /// All operations must have the same tensor dimensions and element type.
    /// On GPU backends this emits one mega-kernel that processes every op
    /// sequentially within a single launch, eliminating all inter-op kernel
    /// launch overhead.  On the CPU backend each operation is executed
    /// independently via `from_fn`.
    ///
    /// # Arguments
    ///
    /// * `ops` — per-op descriptors: `(input_ptrs, gpu_expr, n_inputs)`.
    /// * `nrows`, `ncols` — shared dimensions for all ops.
    /// * `cpu_fns` — per-op CPU closures (used only on CPU backend).
    /// * `kernel_hash` — cache key for the compiled mega-kernel.
    ///
    /// # Example (internal codegen helper)
    ///
    /// ```ignore
    /// let results = Tensor::<f32>::__mega_fuse_elementwise(
    ///     &ops, nrows, ncols, cpu_fns, "hash123",
    /// );
    /// ```
    #[doc(hidden)]
    #[inline]
    pub fn __mega_fuse_elementwise(
        ops: &[(Vec<*const u8>, String, usize)],
        nrows: usize,
        ncols: usize,
        cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T>>,
        kernel_hash: &str,
    ) -> Vec<Self> {
        B::mega_fuse_launch::<T>(ops, nrows, ncols, cpu_fns, kernel_hash)
            .into_iter()
            .map(Self::from_storage)
            .collect()
    }
}

/// Sealed module for `MatmulCompat`.
mod matmul_compat_seal {
    pub trait Sealed {}
    impl Sealed for () {}
    impl<A, K> Sealed for (A, K) {}
}

/// Compile-time axis compatibility for matrix multiplication.
///
/// `(A, K) * (K, B) -> (A, B)` when axes are named.
/// Untyped `()` is compatible with everything.
pub trait MatmulCompat<Rhs, Out>: matmul_compat_seal::Sealed {}

impl MatmulCompat<(), ()> for () {}
impl<A, K> MatmulCompat<(), ()> for (A, K) {}
impl<K, B> MatmulCompat<(K, B), ()> for () {}
impl<A, K, B> MatmulCompat<(K, B), (A, B)> for (A, K) {}

fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    (start, end)
}

/// Write a matrix in `[[a, b], [c, d]]` style.
///
/// `prefix` is written before the outer `[`; when `None` a space is inserted
/// between rows (Display style), otherwise `, ` (Debug style).
pub(crate) fn fmt_matrix(
    rows: usize,
    cols: usize,
    mut elem: impl FnMut(usize, usize, &mut fmt::Formatter<'_>) -> fmt::Result,
    f: &mut fmt::Formatter<'_>,
    prefix: Option<&str>,
) -> fmt::Result {
    if let Some(p) = prefix {
        write!(f, "{p}")?;
    }
    write!(f, "[")?;
    for r in 0..rows {
        if prefix.is_none() && r > 0 {
            write!(f, " ")?;
        }
        write!(f, "[")?;
        for c in 0..cols {
            if c > 0 {
                write!(f, ", ")?;
            }
            elem(r, c, f)?;
        }
        write!(f, "]")?;
        if r + 1 < rows {
            if prefix.is_some() {
                write!(f, ", ")?;
            } else {
                writeln!(f)?;
            }
        }
    }
    write!(f, "]")
}

/// Generate element-wise math forwarding methods on Tensor.
macro_rules! impl_tensor_math {
    ($($(#[$meta:meta])* $name:ident);+ $(;)?) => {
        impl<T: Scalar, B: Backend> Tensor<T, B> {
            $(
                $(#[$meta])*
                #[must_use]
                #[inline]
                pub fn $name(&self) -> Self {
                    Self::from_storage(B::$name(&self.storage))
                }
            )+
        }
    };
}

impl_tensor_math! {
    /// Element-wise `e^x`.
    exp;
    /// Element-wise natural logarithm `ln(x)`.
    ln;
    /// Element-wise `ln(1 + x)`.
    log1p;
    /// Element-wise `sin(x)`.
    sin;
    /// Element-wise `cos(x)`.
    cos;
    /// Element-wise `tanh(x)`.
    tanh;
    /// Element-wise `sqrt(x)`.
    sqrt;
    /// Element-wise absolute value.
    abs;
    /// Element-wise reciprocal `1/x`.
    recip;
    /// Element-wise error function.
    erf;
    /// Element-wise `ceil(x)`.
    ceil;
    /// Element-wise `floor(x)`.
    floor;
    /// Element-wise `round(x)`.
    round;
}

/// Helper: `T::one() + T::one()` (avoids repeating the pattern).
#[inline]
fn two<T: Scalar>() -> T { T::one() + T::one() }

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Validate two tensors have the same shape.
    fn assert_same_shape(&self, other: &Self, op: &str) {
        assert!(
            self.nrows() == other.nrows() && self.ncols() == other.ncols(),
            "{op}: shape mismatch ({},{}) vs ({},{})",
            self.nrows(), self.ncols(), other.nrows(), other.ncols()
        );
    }

    /// Validate vector matches row/column dimension for broadcast.
    fn assert_broadcast_shape(&self, vec: &Self, axis: usize, op: &str) {
        if axis == 0 {
            assert!(vec.nrows() == 1 && vec.ncols() == self.ncols(),
                "{op}: expected (1,{}) got ({},{})", self.ncols(), vec.nrows(), vec.ncols());
        } else {
            assert!(vec.ncols() == 1 && vec.nrows() == self.nrows(),
                "{op}: expected ({},1) got ({},{})", self.nrows(), vec.nrows(), vec.ncols());
        }
    }

    /// Reduce along axis with a custom fold function and initial value from first element.
    fn reduce_axis<F: Fn(T, T) -> T>(&self, axis: usize, f: F) -> Self {
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                (0..self.nrows()).map(|r| self.get(r, c))
                    .fold(self.get(0, c), &f)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                (0..self.ncols()).map(|c| self.get(r, c))
                    .fold(self.get(r, 0), &f)
            }),
            _ => panic!("nabla: reduce_axis axis must be 0 or 1, got {axis}"),
        }
    }

    /// Apply a binary op between self and a broadcast vector along the given axis.
    fn apply_broadcast_op<F: Fn(T, T) -> T>(&self, vec: &Self, axis: usize, op: &str, f: F) -> Self {
        self.assert_broadcast_shape(vec, axis, op);
        let (m, n) = self.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| f(self.get(r, c), vec.get(0, c))),
            _ => Self::from_fn(m, n, |r, c| f(self.get(r, c), vec.get(r, 0))),
        }
    }

    #[inline]
    fn check_range(name: &str, index: usize, bound: usize) {
        assert!(
            index < bound,
            "{name} index {index} out of bounds for {bound}"
        );
    }

    #[inline]
    fn check_half_open(name: &str, start: usize, end: usize, bound: usize) {
        assert!(
            start <= end,
            "{name} start ({start}) must be <= end ({end})"
        );
        assert!(end <= bound, "{name} end ({end}) must be <= {bound}");
    }

    /// Create a tensor from a preallocated storage.
    #[inline]
    /// Construct from raw backend storage.
    pub fn from_storage(storage: B::Storage<T>) -> Self {
        Self { storage, _axes: PhantomData }
    }

    /// Allocate a zero-filled matrix of shape `(nrows, ncols)`.
    #[must_use]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::from_storage(B::zeros(nrows, ncols))
    }

    /// Allocate an `nrows x ncols` matrix filled with `val`.
    #[must_use]
    pub fn fill(nrows: usize, ncols: usize, val: T) -> Self {
        Self::from_storage(B::fill(nrows, ncols, val))
    }

    /// Allocate a matrix whose `(i, j)` element is `f(i, j)`.
    #[must_use]
    pub fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> Self {
        Self::from_storage(B::from_fn(nrows, ncols, f))
    }

    /// Allocate an `n x n` identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_storage(B::identity(n))
    }

    /// Convert class index slice to one-hot matrix `(n_samples × n_classes)`.
    #[must_use]
    pub fn one_hot(indices: &[usize], n_classes: usize) -> Self {
        Self::from_fn(indices.len(), n_classes, |r, c| {
            if c == indices[r] { T::one() } else { T::zero() }
        })
    }

    /// Create tensor from slice with non-blocking H2D transfer.
    /// The transfer happens on a separate copy stream and can overlap with compute.
    /// On CPU backend this is identical to the synchronous path.
    #[must_use]
    pub fn from_slice_async(data: &[T], nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols, "from_slice_async: data.len() must equal nrows * ncols");
        Self::from_storage(B::from_vec_async(nrows, ncols, data.to_vec()))
    }

    /// Copy a submatrix into a fresh tensor.
    #[must_use]
    pub fn submatrix(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        Self::check_half_open("submatrix row", row_start, row_end, self.nrows());
        Self::check_half_open("submatrix col", col_start, col_end, self.ncols());
        let nrows = row_end - row_start;
        let ncols = col_end - col_start;
        Self::from_storage(B::from_fn(nrows, ncols, |r, c| self.get(row_start + r, col_start + c)))
    }

    /// Set element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        Self::check_range("set row", row, self.nrows());
        Self::check_range("set col", col, self.ncols());
        B::set(&mut self.storage, row, col, val);
    }

    /// Slice by row and column ranges, returning a new `Tensor` (copy).
    #[must_use]
    pub fn slice(&self, rows: impl RangeBounds<usize>, cols: impl RangeBounds<usize>) -> Self {
        let (rs, re) = resolve_range(rows, self.nrows());
        let (cs, ce) = resolve_range(cols, self.ncols());
        self.submatrix(rs, re, cs, ce)
    }

    /// Slice rows, all columns.
    #[must_use]
    pub fn slice_rows(&self, rows: impl RangeBounds<usize>) -> Self {
        self.slice(rows, ..)
    }

    /// Slice columns, all rows.
    #[must_use]
    pub fn slice_cols(&self, cols: impl RangeBounds<usize>) -> Self {
        self.slice(.., cols)
    }

    /// Extract a row vector tensor with shape `1 x ncols`.
    #[must_use]
    pub fn row(&self, row: usize) -> Self {
        Self::check_range("row", row, self.nrows());
        Self::from_storage(B::from_fn(1, self.ncols(), |_, c| self.get(row, c)))
    }

    /// Extract a column vector tensor with shape `nrows x 1`.
    #[must_use]
    pub fn col(&self, col: usize) -> Self {
        Self::check_range("col", col, self.ncols());
        Self::from_storage(B::from_fn(self.nrows(), 1, |r, _| self.get(r, col)))
    }

    /// Split into four quadrants: `(top-left, top-right, bottom-left, bottom-right)`.
    #[must_use]
    pub fn split_at(&self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        Self::check_half_open("split row", 0, row, self.nrows());
        Self::check_half_open("split col", 0, col, self.ncols());
        let top_left = self.submatrix(0, row, 0, col);
        let top_right = self.submatrix(0, row, col, self.ncols());
        let bottom_left = self.submatrix(row, self.nrows(), 0, col);
        let bottom_right = self.submatrix(row, self.nrows(), col, self.ncols());
        (top_left, top_right, bottom_left, bottom_right)
    }

    /// Element-wise `x^p` for scalar exponent `p`.
    #[must_use]
    #[inline]
    pub fn powf(&self, p: T) -> Self {
        Self::from_storage(B::powf(&self.storage, p))
    }

    /// Element-wise multiplication `self[i,j] * other[i,j]`.
    #[must_use]
    pub fn emul(&self, other: &Self) -> Self {
        self.assert_same_shape(other, "nabla: emul");
        Self::from_storage(B::emul(&self.storage, &other.storage))
    }

    /// Element-wise division `self[i,j] / other[i,j]`.
    #[must_use]
    pub fn ediv(&self, other: &Self) -> Self {
        self.assert_same_shape(other, "nabla: ediv");
        Self::from_storage(B::ediv(&self.storage, &other.storage))
    }

    /// Add a row vector `(1×n)` to every row of `self (m×n)`.
    #[must_use]
    pub fn broadcast_add_rows(&self, row: &Self) -> Self {
        self.apply_broadcast_op(row, 0, "nabla: broadcast_add_rows", |a, b| a + b)
    }

    /// Add a column vector `(m×1)` to every column of `self (m×n)`.
    #[must_use]
    pub fn broadcast_add_cols(&self, col: &Self) -> Self {
        self.apply_broadcast_op(col, 1, "nabla: broadcast_add_cols", |a, b| a + b)
    }

    /// Element-wise multiply each row by a row vector `(1×n)`.
    #[must_use]
    pub fn broadcast_mul_rows(&self, row: &Self) -> Self {
        self.apply_broadcast_op(row, 0, "nabla: broadcast_mul_rows", |a, b| a * b)
    }

    /// Element-wise multiply each column by a column vector `(m×1)`.
    #[must_use]
    pub fn broadcast_mul_cols(&self, col: &Self) -> Self {
        self.apply_broadcast_op(col, 1, "nabla: broadcast_mul_cols", |a, b| a * b)
    }

    /// In-place add a row vector `(1×n)` to every row.
    pub fn broadcast_add_rows_(&mut self, row: &Self) {
        self.assert_broadcast_shape(row, 0, "nabla: broadcast_add_rows_");
        let (m, n) = self.shape();
        for r in 0..m {
            for c in 0..n {
                self.set(r, c, self.get(r, c) + row.get(0, c));
            }
        }
    }

    /// In-place add a column vector `(m×1)` to every column.
    pub fn broadcast_add_cols_(&mut self, col: &Self) {
        self.assert_broadcast_shape(col, 1, "nabla: broadcast_add_cols_");
        let (m, n) = self.shape();
        for r in 0..m {
            for c in 0..n {
                self.set(r, c, self.get(r, c) + col.get(r, 0));
            }
        }
    }

    /// Numerically stable log-softmax along `axis` (0 = columns, 1 = rows).
    #[must_use]
    pub fn log_softmax(&self, axis: usize) -> Self {
        match axis {
            1 => {
                let (m, n) = self.shape();
                let two = two::<T>();
                Self::from_fn(m, n, |r, c| {
                    // max via (a + b + |a - b|) / 2
                    let row_max = (0..n).fold(self.get(r, 0), |acc, j| {
                        let v = self.get(r, j);
                        (acc + v + (acc - v).math_abs()) / two
                    });
                    let log_sum_exp = {
                        let s: T = (0..n)
                            .map(|j| (self.get(r, j) - row_max).math_exp())
                            .fold(T::zero(), |a, b| a + b);
                        row_max + s.math_ln()
                    };
                    self.get(r, c) - log_sum_exp
                })
            }
            0 => self.t().log_softmax(1).t(),
            _ => panic!("nabla: log_softmax axis must be 0 or 1, got {axis}"),
        }
    }

    /// Cross-entropy loss from log-softmax predictions and target probabilities.
    ///
    /// `self` = log-softmax output `(batch × classes)`,
    /// `targets` = one-hot or probability distribution `(batch × classes)`.
    /// Returns `-mean(sum(targets * log_probs, axis=1))`.
    #[must_use]
    pub fn cross_entropy_loss(&self, targets: &Self) -> T {
        let (batch, n) = self.shape();
        assert_eq!(targets.shape(), (batch, n),
            "nabla: cross_entropy_loss shape mismatch — self {}×{} vs targets {}×{}",
            batch, n, targets.nrows(), targets.ncols());
        let sum: T = (0..batch)
            .map(|r| {
                (0..n)
                    .map(|c| targets.get(r, c) * self.get(r, c))
                    .fold(T::zero(), |a, b| a + b)
            })
            .fold(T::zero(), |a, b| a + b);
        -(sum / T::from_f64(batch as f64))
    }

    /// Extract the diagonal as an `n×1` column vector, where `n = min(rows, cols)`.
    #[must_use]
    pub fn diag(&self) -> Self {
        let n = self.nrows().min(self.ncols());
        Self::from_fn(n, 1, |i, _| self.get(i, i))
    }

    /// Sum of all elements.
    #[must_use]
    #[inline]
    pub fn sum_all(&self) -> T {
        B::sum_all(&self.storage)
    }

    /// Frobenius norm squared: sum of squares of all elements.
    #[must_use]
    pub fn norm_sq(&self) -> T {
        self.emul(self).sum_all()
    }

    /// Frobenius norm: `sqrt(sum of squares)`. For vectors this is the L2 norm.
    #[must_use]
    pub fn norm(&self) -> T {
        self.norm_sq().math_sqrt()
    }

    /// L1 norm: sum of absolute values.
    #[must_use]
    pub fn l1_norm(&self) -> T {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| self.get(r, c).math_abs()).sum_all()
    }

    /// L-infinity norm: maximum absolute value.
    #[must_use]
    pub fn linf_norm(&self) -> T {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| self.get(r, c).math_abs()).max_all()
    }

    /// Normalize to unit vector (L2 norm = 1). Returns clone if norm is zero.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == T::zero() {
            self.clone()
        } else {
            self * (T::one() / n)
        }
    }

    /// Same-shape tensor filled with zeros.
    #[must_use]
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.nrows(), self.ncols())
    }

    /// Same-shape tensor filled with ones.
    #[must_use]
    pub fn ones_like(&self) -> Self {
        Self::fill(self.nrows(), self.ncols(), T::one())
    }

    /// Same-shape tensor filled with `val`.
    #[must_use]
    pub fn fill_like(&self, val: T) -> Self {
        Self::fill(self.nrows(), self.ncols(), val)
    }

    /// Select rows by index. Duplicates allowed.
    #[must_use]
    pub fn gather_rows(&self, indices: &[usize]) -> Self {
        let nc = self.ncols();
        Self::from_fn(indices.len(), nc, |r, c| self.get(indices[r], c))
    }

    // ── ML activation functions ─────────────────────────────────────

    /// Element-wise ReLU: `max(x, 0)`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let two = two::<T>();
        let (m, n) = self.shape();
        // relu(x) = (x + |x|) / 2  — avoids PartialOrd requirement
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            (x + x.math_abs()) / two
        })
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            T::one() / (T::one() + (T::zero() - x).math_exp())
        })
    }

    /// Element-wise GELU (tanh approximation):
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    #[must_use]
    pub fn gelu(&self) -> Self {
        let half = T::from_f64(0.5);
        let k = T::from_f64(0.797_884_560_8); // sqrt(2/pi)
        let c = T::from_f64(0.044_715);
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, col| {
            let x = self.get(r, col);
            let inner = k * (x + c * x * x * x);
            half * x * (T::one() + inner.math_tanh())
        })
    }

    /// Softmax along given axis (0=columns, 1=rows).
    ///
    /// Uses the log-sum-exp trick for numerical stability: subtract max before exp.
    #[must_use]
    pub fn softmax(&self, axis: usize) -> Self {
        match axis {
            1 => {
                let (m, n) = self.shape();
                Self::from_fn(m, n, |r, c| {
                    let row_max = (0..n).fold(self.get(r, 0), |acc, j| {
                        let v = self.get(r, j);
                        // max via (a + b + |a - b|) / 2
                        let two = two::<T>();
                        (acc + v + (acc - v).math_abs()) / two
                    });
                    let exp_sum = (0..n).fold(T::zero(), |acc, j| {
                        acc + (self.get(r, j) - row_max).math_exp()
                    });
                    (self.get(r, c) - row_max).math_exp() / exp_sum
                })
            }
            0 => self.t().softmax(1).t(),
            _ => panic!("nabla: softmax axis must be 0 or 1, got {axis}"),
        }
    }

    /// Element-wise clamp: values below `lo` become `lo`, above `hi` become `hi`.
    ///
    /// Uses `(a + b + |a - b|) / 2` for min/max to avoid `PartialOrd` bound.
    #[must_use]
    pub fn clamp(&self, lo: T, hi: T) -> Self {
        let two = two::<T>();
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            // max(x, lo) = (x + lo + |x - lo|) / 2
            let clamped_lo = (x + lo + (x - lo).math_abs()) / two;
            // min(clamped_lo, hi) = (clamped_lo + hi - |clamped_lo - hi|) / 2
            (clamped_lo + hi - (clamped_lo - hi).math_abs()) / two
        })
    }

    /// Create an n x n diagonal matrix from a vector (n x 1 or 1 x n).
    #[must_use]
    pub fn from_diag(v: &Self) -> Self {
        let n = v.nrows().max(v.ncols());
        let is_col = v.nrows() >= v.ncols();
        Self::from_fn(n, n, |r, c| {
            if r == c {
                if is_col { v.get(r, 0) } else { v.get(0, c) }
            } else {
                T::zero()
            }
        })
    }

    /// Sum of diagonal elements.
    #[must_use]
    pub fn trace(&self) -> T {
        let n = self.nrows().min(self.ncols());
        (0..n).map(|i| self.get(i, i)).fold(T::zero(), |a, b| a + b)
    }

    /// Element with the maximum value (or maximum magnitude for complex types).
    #[must_use]
    #[inline]
    pub fn max_all(&self) -> T {
        B::max_all(&self.storage)
    }

    /// Element with the minimum value (or minimum magnitude for complex types).
    #[must_use]
    #[inline]
    pub fn min_all(&self) -> T {
        B::min_all(&self.storage)
    }

    /// `(row, col)` of the element with the maximum value (or magnitude for complex types).
    #[must_use]
    #[inline]
    pub fn argmax(&self) -> (usize, usize) {
        B::argmax_all(&self.storage)
    }

    /// `(row, col)` of the element with the minimum value (or magnitude for complex types).
    #[must_use]
    #[inline]
    pub fn argmin(&self) -> (usize, usize) {
        B::argmin_all(&self.storage)
    }

    /// Return the transpose: a new `Tensor` of shape `(ncols, nrows)`.
    #[must_use]
    pub fn t(&self) -> Self {
        Self::from_storage(B::transpose(&self.storage))
    }

    /// Return the conjugate transpose (adjoint / Hermitian transpose).
    #[must_use]
    pub fn adjoint(&self) -> Self {
        let (r, c) = self.shape();
        Self::from_storage(if T::IS_REAL {
            B::from_fn(c, r, |i, j| self.get(j, i))
        } else {
            B::from_fn(c, r, |i, j| {
                crate::scalar::math_utils::conj(&self.get(j, i))
            })
        })
    }

    /// Permute axes of a 2-D tensor.
    ///
    /// `axes` must be a permutation of `[0, 1]`:
    /// - `[0, 1]` -> identity (clone)
    /// - `[1, 0]` -> transpose
    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), 2, "nabla: Tensor is 2-D — permute axes must have length 2");
        assert!(
            axes[0] < 2 && axes[1] < 2 && axes[0] != axes[1],
            "nabla: permute axes must be a permutation of {{0, 1}}, got [{}, {}]",
            axes[0], axes[1]
        );
        match (axes[0], axes[1]) {
            (0, 1) => self.clone(),
            _ => self.t(),
        }
    }

    /// Compute `out = a * b` (matrix multiply), overwriting `out`.
    pub fn matmul_into(out: &mut Self, a: &Self, b: &Self) {
        let (m, k_a) = a.shape();
        let (k_b, n) = b.shape();
        assert_eq!(
            k_a, k_b,
            "matmul inner dimension mismatch: a is ({m}, {k_a}), b is ({k_b}, {n})"
        );
        let (out_r, out_c) = out.shape();
        assert!(
            out_r == m && out_c == n,
            "matmul output shape mismatch: expected ({m}, {n}), got ({out_r}, {out_c})"
        );
        B::matmul_into(&mut out.storage, &a.storage, &b.storage);
    }

    /// Apply a closure element-wise, returning a new tensor.
    #[must_use]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| f(self.get(r, c)))
    }

    /// Fused matmul + element-wise activation (2-pass on CPU, single kernel on GPU TODO).
    #[must_use]
    pub fn matmul_fused<F>(a: &Self, b: &Self, act: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let c = a * b;
        c.map(act)
    }

    /// Reshape to `(m, n)`, preserving row-major element order.
    #[must_use]
    pub fn reshape(&self, m: usize, n: usize) -> Self {
        let (rows, cols) = self.shape();
        assert_eq!(
            m * n,
            rows * cols,
            "reshape: {rows}x{cols} cannot reshape to {m}x{n}"
        );
        Self::from_fn(m, n, |r, c| {
            let flat = r * n + c;
            self.get(flat / cols, flat % cols)
        })
    }

    /// Flatten to `(1, nrows*ncols)`.
    #[must_use]
    pub fn flatten(&self) -> Self {
        let n = self.nrows() * self.ncols();
        self.reshape(1, n)
    }

    /// Sum along axis. axis=0 -> (1, ncols), axis=1 -> (nrows, 1).
    #[must_use]
    pub fn sum_axis(&self, axis: usize) -> Self {
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                (0..self.nrows()).fold(T::zero(), |acc, r| acc + self.get(r, c))
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                (0..self.ncols()).fold(T::zero(), |acc, c| acc + self.get(r, c))
            }),
            _ => panic!("sum_axis: axis {axis} out of bounds"),
        }
    }

    /// Mean along axis.
    #[must_use]
    pub fn mean_axis(&self, axis: usize) -> Self {
        let n = match axis {
            0 => self.nrows(),
            1 => self.ncols(),
            _ => panic!("mean_axis: axis {axis} out of bounds"),
        };
        let sum = self.sum_axis(axis);
        let inv_n = T::from_f64(1.0 / n as f64);
        &sum * inv_n
    }

    /// Cumulative sum along axis (0 = column-wise, 1 = row-wise).
    #[must_use]
    pub fn cumsum(&self, axis: usize) -> Self {
        match axis {
            1 => Self::from_fn(self.nrows(), self.ncols(), |r, c| {
                (0..=c).map(|j| self.get(r, j)).fold(T::zero(), |a, b| a + b)
            }),
            0 => Self::from_fn(self.nrows(), self.ncols(), |r, c| {
                (0..=r).map(|i| self.get(i, c)).fold(T::zero(), |a, b| a + b)
            }),
            _ => panic!("nabla: cumsum axis must be 0 or 1, got {axis}"),
        }
    }

    /// Vertical concat: stack `tensors` row-by-row. All must have same ncols.
    #[must_use]
    pub fn vcat(tensors: &[&Self]) -> Self {
        assert!(!tensors.is_empty(), "nabla: vcat on empty slice");
        let ncols = tensors[0].ncols();
        for (i, t) in tensors.iter().enumerate() {
            assert_eq!(
                t.ncols(),
                ncols,
                "nabla: vcat tensor[{i}] has {c} cols, expected {ncols}",
                c = t.ncols()
            );
        }
        let offsets: Vec<usize> = tensors
            .iter()
            .scan(0usize, |acc, t| {
                let s = *acc;
                *acc += t.nrows();
                Some(s)
            })
            .collect();
        let total = offsets.last().map_or(0, |&o| o) + tensors.last().map_or(0, |t| t.nrows());
        Self::from_fn(total, ncols, |r, c| {
            let ti = offsets.partition_point(|&o| o <= r) - 1;
            tensors[ti].get(r - offsets[ti], c)
        })
    }

    /// Horizontal concat: stack `tensors` column-by-column. All must have same nrows.
    #[must_use]
    pub fn hcat(tensors: &[&Self]) -> Self {
        assert!(!tensors.is_empty(), "nabla: hcat on empty slice");
        let nrows = tensors[0].nrows();
        for (i, t) in tensors.iter().enumerate() {
            assert_eq!(
                t.nrows(),
                nrows,
                "nabla: hcat tensor[{i}] has {r} rows, expected {nrows}",
                r = t.nrows()
            );
        }
        let offsets: Vec<usize> = tensors
            .iter()
            .scan(0usize, |acc, t| {
                let s = *acc;
                *acc += t.ncols();
                Some(s)
            })
            .collect();
        let total = offsets.last().map_or(0, |&o| o) + tensors.last().map_or(0, |t| t.ncols());
        Self::from_fn(nrows, total, |r, c| {
            let ti = offsets.partition_point(|&o| o <= c) - 1;
            tensors[ti].get(r, c - offsets[ti])
        })
    }

    /// Concatenate tensors along `axis` (0=row/vcat, 1=col/hcat).
    #[must_use]
    pub fn cat(tensors: &[&Self], axis: usize) -> Self {
        match axis {
            0 => Self::vcat(tensors),
            1 => Self::hcat(tensors),
            _ => panic!("nabla: cat axis {axis} out of bounds for 2-D tensor"),
        }
    }

    /// Stack tensors along a new axis, producing an `NdTensor`.
    /// All must have the same shape. axis=0 -> (n, nrows, ncols).
    #[must_use]
    pub fn stack(tensors: &[&Self], axis: usize) -> NdTensor<T> {
        assert!(!tensors.is_empty(), "nabla: stack on empty slice");
        assert!(
            axis <= 2,
            "nabla: stack axis {axis} out of bounds (max 2 for 2-D inputs)"
        );
        let (r, c) = tensors[0].shape();
        for (i, t) in tensors.iter().enumerate() {
            let (tr, tc) = t.shape();
            assert!(
                tr == r && tc == c,
                "nabla: stack tensor[{i}] shape ({tr}x{tc}) differs from tensor[0] ({r}x{c})"
            );
        }
        let n = tensors.len();
        let new_shape: [usize; 3] = match axis {
            0 => [n, r, c],
            1 => [r, n, c],
            _ => [r, c, n],
        };
        NdTensor::from_fn(&new_shape, |idx| {
            let (ti, row, col) = match axis {
                0 => (idx[0], idx[1], idx[2]),
                1 => (idx[1], idx[0], idx[2]),
                _ => (idx[2], idx[0], idx[1]),
            };
            tensors[ti].get(row, col)
        })
    }

    /// Split into `n` equal chunks along `axis`. Panics if not evenly divisible.
    #[must_use]
    pub fn chunk(&self, n: usize, axis: usize) -> Vec<Self> {
        match axis {
            0 => {
                let r = self.nrows();
                assert_eq!(
                    r % n,
                    0,
                    "nabla: chunk nrows ({r}) not divisible by {n}"
                );
                let chunk_size = r / n;
                (0..n)
                    .map(|i| self.slice_rows(i * chunk_size..(i + 1) * chunk_size))
                    .collect()
            }
            1 => {
                let c = self.ncols();
                assert_eq!(
                    c % n,
                    0,
                    "nabla: chunk ncols ({c}) not divisible by {n}"
                );
                let chunk_size = c / n;
                (0..n)
                    .map(|i| self.slice_cols(i * chunk_size..(i + 1) * chunk_size))
                    .collect()
            }
            _ => panic!("nabla: chunk axis {axis} out of bounds for 2-D tensor"),
        }
    }

    /// Remove a size-1 dimension. For 2-D tensors this validates the axis has size 1
    /// and returns a clone (still 2-D — use `NdTensor` for true rank reduction).
    #[must_use]
    pub fn squeeze(&self, axis: usize) -> Self {
        let d = self.dim(axis);
        assert_eq!(
            d, 1,
            "nabla: squeeze({axis}) — dim is {d}, not 1"
        );
        self.clone()
    }

    /// Insert a size-1 dimension at `axis`, producing an `NdTensor`.
    /// tensor (m, n).unsqueeze(0) -> (1, m, n)
    /// tensor (m, n).unsqueeze(1) -> (m, 1, n)
    /// tensor (m, n).unsqueeze(2) -> (m, n, 1)
    #[must_use]
    pub fn unsqueeze(&self, axis: usize) -> NdTensor<T> {
        assert!(
            axis <= 2,
            "nabla: unsqueeze axis {axis} out of bounds (max 2 for 2-D tensor)"
        );
        let (r, c) = self.shape();
        let mut new_shape = vec![r, c];
        new_shape.insert(axis, 1);
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig = idx.to_vec();
            orig.remove(axis);
            self.get(orig[0], orig[1])
        })
    }

    /// Reshape alias (PyTorch-style naming). Same as `reshape()`.
    #[must_use]
    pub fn view(&self, m: usize, n: usize) -> Self {
        self.reshape(m, n)
    }

    /// Sum along axis with keepdim semantics (for 2-D, identical to `sum_axis`).
    #[must_use]
    pub fn sum_axis_keepdim(&self, axis: usize) -> Self {
        self.sum_axis(axis)
    }

    /// Mean along axis with keepdim semantics (for 2-D, identical to `mean_axis`).
    #[must_use]
    pub fn mean_axis_keepdim(&self, axis: usize) -> Self {
        self.mean_axis(axis)
    }

    /// Maximum along axis: axis 0 → 1×n (column-wise), axis 1 → m×1 (row-wise).
    #[must_use]
    pub fn max_axis(&self, axis: usize) -> Self {
        self.reduce_axis(axis, |a, b| if b.reduction_gt(a) { b } else { a })
    }

    /// Minimum along axis: axis 0 → 1×n (column-wise), axis 1 → m×1 (row-wise).
    #[must_use]
    pub fn min_axis(&self, axis: usize) -> Self {
        self.reduce_axis(axis, |a, b| if a.reduction_gt(b) { b } else { a })
    }

    /// Maximum along axis with keepdim semantics (for 2-D, identical to `max_axis`).
    #[must_use]
    pub fn max_axis_keepdim(&self, axis: usize) -> Self {
        self.max_axis(axis)
    }

    /// Minimum along axis with keepdim semantics (for 2-D, identical to `min_axis`).
    #[must_use]
    pub fn min_axis_keepdim(&self, axis: usize) -> Self {
        self.min_axis(axis)
    }

    /// Population variance along axis: `E[X²] - E[X]²`.
    #[must_use]
    pub fn var_axis(&self, axis: usize) -> Self {
        let mean = self.mean_axis(axis);
        let (mr, mc) = mean.shape();
        // E[X²]
        let sq = Self::from_fn(self.nrows(), self.ncols(), |r, c| {
            let x = self.get(r, c);
            x * x
        });
        let mean_sq = sq.mean_axis(axis);
        // E[X²] - E[X]²
        Self::from_fn(mr, mc, |r, c| {
            let m = mean.get(r, c);
            mean_sq.get(r, c) - m * m
        })
    }

    /// Population standard deviation along axis: `sqrt(var_axis)`.
    #[must_use]
    pub fn std_axis(&self, axis: usize) -> Self {
        let v = self.var_axis(axis);
        Self::from_fn(v.nrows(), v.ncols(), |r, c| v.get(r, c).math_sqrt())
    }

    /// Variance along axis with keepdim semantics (for 2-D, identical to `var_axis`).
    #[must_use]
    pub fn var_axis_keepdim(&self, axis: usize) -> Self {
        self.var_axis(axis)
    }

    /// Std deviation along axis with keepdim semantics (for 2-D, identical to `std_axis`).
    #[must_use]
    pub fn std_axis_keepdim(&self, axis: usize) -> Self {
        self.std_axis(axis)
    }

    /// Layer normalization along `axis`: `(x - mean) / (std + eps)`.
    ///
    /// Normalizes each slice to zero-mean unit-variance. Axis 1 normalizes
    /// each row independently (standard transformer usage).
    #[must_use]
    pub fn layer_norm(&self, axis: usize, eps: T) -> Self {
        let mean = self.mean_axis(axis);
        let std = self.std_axis(axis);
        let (sr, sc) = std.shape();
        let inv_std = Self::from_fn(sr, sc, |r, c| T::one() / (std.get(r, c) + eps));
        let neg_mean = -&mean;
        match axis {
            1 => {
                // mean/std shape: (m, 1) — broadcast as column vectors
                let centered = self.broadcast_add_cols(&neg_mean);
                centered.broadcast_mul_cols(&inv_std)
            }
            0 => {
                // mean/std shape: (1, n) — broadcast as row vectors
                let centered = self.broadcast_add_rows(&neg_mean);
                centered.broadcast_mul_rows(&inv_std)
            }
            _ => panic!("nabla: layer_norm axis must be 0 or 1, got {axis}"),
        }
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Parallel `from_fn` -- construct a matrix using rayon.
    #[must_use]
    pub fn par_from_fn(
        nrows: usize,
        ncols: usize,
        f: impl Fn(usize, usize) -> T + Send + Sync,
    ) -> Self {
        let data: Vec<T> = (0..nrows * ncols)
            .into_par_iter()
            .map(|idx| f(idx / ncols, idx % ncols))
            .collect();
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }

    /// Parallel element-wise transform -- applies `f` to every element using rayon.
    #[must_use]
    pub fn par_map(&self, f: impl Fn(T) -> T + Send + Sync) -> Self {
        let (nrows, ncols) = self.shape();
        let data: Vec<T> = (0..nrows * ncols)
            .into_par_iter()
            .map(|idx| f(B::get(&self.storage, idx / ncols, idx % ncols)))
            .collect();
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar> Tensor<T, Cpu> {
    /// Borrow the underlying row-major data slice (zero-copy).
    #[inline]
    /// View underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.data_slice()
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar> Index<(usize, usize)> for Tensor<T, Cpu> {
    type Output = T;

    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &T {
        let (nrows, ncols) = self.shape();
        assert!(
            r < nrows && c < ncols,
            "nabla: index ({r},{c}) out of bounds for {nrows}×{ncols} tensor"
        );
        self.storage.get_ref(r, c)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar> IndexMut<(usize, usize)> for Tensor<T, Cpu> {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        let (nrows, ncols) = self.shape();
        assert!(
            r < nrows && c < ncols,
            "nabla: index ({r},{c}) out of bounds for {nrows}×{ncols} tensor"
        );
        self.storage.get_mut(r, c)
    }
}

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $backend_fn:ident, $op:literal) => {
        impl<T: Scalar, B: Backend> $trait for &Tensor<T, B> {
            type Output = Tensor<T, B>;

            fn $method(self, rhs: Self) -> Self::Output {
                let (m, n) = self.shape();
                let (p, q) = rhs.shape();
                assert!(
                    m == p && n == q,
                    concat!("nabla: ", $op, " ({}×{}) vs ({}×{}) — shapes must match"),
                    m, n, p, q,
                );
                Tensor::from_storage(B::$backend_fn(&self.storage, &rhs.storage))
            }
        }
    };
}

impl_tensor_binop!(Add, add, add, "add");
impl_tensor_binop!(Sub, sub, sub, "sub");

impl<T: Scalar, B: Backend> Neg for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn neg(self) -> Self::Output {
        Tensor::from_storage(B::neg(&self.storage))
    }
}

/// Matrix multiply (`*`). Panics if inner dimensions do not match.
impl<T: Scalar, B: Backend> Mul for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (m, k_a) = self.shape();
        let (k_b, n) = rhs.shape();
        assert_eq!(
            k_a, k_b,
            "nabla: matmul ({m}×{k_a}) × ({k_b}×{n}) — inner dims {k_a} ≠ {k_b}"
        );
        let mut out = Tensor::<T, B>::zeros(m, n);
        B::matmul_into(&mut out.storage, &self.storage, &rhs.storage);
        out
    }
}

/// Scalar multiply: `&tensor * scalar`.
impl<T: Scalar, B: Backend> Mul<T> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_storage(B::scale(&self.storage, rhs))
    }
}

impl<T: Scalar + fmt::Display, B: Backend> fmt::Display for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        fmt_matrix(
            rows,
            cols,
            |r, c, f| write!(f, "{}", self.get(r, c)),
            f,
            None,
        )
    }
}

impl<T: Scalar + fmt::Debug, B: Backend> fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        let prefix = format!("Tensor({rows}x{cols})");
        fmt_matrix(
            rows,
            cols,
            |r, c, f| write!(f, "{:?}", self.get(r, c)),
            f,
            Some(&prefix),
        )
    }
}

// ---------------------------------------------------------------------------
// NdTensor<T> — N-dimensional tensor for higher-order einsum operations.
// ---------------------------------------------------------------------------

/// N-dimensional tensor stored as a flat `Vec<T>` in row-major (C-order) layout.
pub struct NdTensor<T: Scalar> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: Scalar> Clone for NdTensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T: Scalar> NdTensor<T> {
    /// Compute row-major strides from shape via reverse cumulative product.
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    /// Convert N-D indices to a flat linear index.
    fn linear_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(
            indices.len(),
            self.shape.len(),
            "NdTensor: expected {} indices, got {}",
            self.shape.len(),
            indices.len()
        );
        indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }

    /// Allocate a zero-filled tensor of the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![T::zero(); n],
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        }
    }

    /// Allocate a tensor whose element at `indices` is `f(indices)`.
    #[must_use]
    pub fn from_fn(shape: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Self {
        let n: usize = shape.iter().product();
        let strides = Self::compute_strides(shape);
        let ndim = shape.len();
        let mut data = Vec::with_capacity(n);
        let mut indices = vec![0usize; ndim];
        for _ in 0..n {
            data.push(f(&indices));
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Number of dimensions.
    #[must_use]
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of dimension `axis`.
    #[must_use]
    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    /// Shape as a slice.
    #[must_use]
    #[inline]
    pub fn shape_vec(&self) -> &[usize] {
        &self.shape
    }

    /// Read element at the given N-D indices.
    #[must_use]
    #[inline]
    pub fn get_nd(&self, indices: &[usize]) -> T {
        let idx = self.linear_index(indices);
        self.data[idx]
    }

    /// Write element at the given N-D indices.
    #[inline]
    pub fn set_nd(&mut self, indices: &[usize], val: T) {
        let idx = self.linear_index(indices);
        self.data[idx] = val;
    }

    /// Total number of elements.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty (any dimension is 0).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Extract a 2-D [`Tensor`] from the last two dimensions, fixing
    /// all preceding batch dimensions.
    #[must_use]
    pub fn slice_2d(&self, batch_indices: &[usize]) -> Tensor<T> {
        assert_eq!(
            batch_indices.len() + 2,
            self.ndim(),
            "NdTensor::slice_2d: expected {} batch indices, got {}",
            self.ndim() - 2,
            batch_indices.len()
        );
        let nrows = self.shape[self.ndim() - 2];
        let ncols = self.shape[self.ndim() - 1];
        let base_offset: usize = batch_indices
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum();
        let row_stride = self.strides[self.ndim() - 2];
        let col_stride = self.strides[self.ndim() - 1];
        Tensor::from_fn(nrows, ncols, |r, c| {
            self.data[base_offset + r * row_stride + c * col_stride]
        })
    }

    /// Set a 2-D slice in the last two dimensions, fixing all
    /// preceding batch dimensions.
    pub fn set_slice_2d(&mut self, batch_indices: &[usize], tensor: &Tensor<T>) {
        assert_eq!(
            batch_indices.len() + 2,
            self.ndim(),
            "NdTensor::set_slice_2d: expected {} batch indices, got {}",
            self.ndim() - 2,
            batch_indices.len()
        );
        let nrows = self.shape[self.ndim() - 2];
        let ncols = self.shape[self.ndim() - 1];
        let base_offset: usize = batch_indices
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum();
        let row_stride = self.strides[self.ndim() - 2];
        let col_stride = self.strides[self.ndim() - 1];
        for r in 0..nrows {
            for c in 0..ncols {
                self.data[base_offset + r * row_stride + c * col_stride] = tensor.get(r, c);
            }
        }
    }

    /// Convenience: number of rows for the last-2 dimension.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        assert!(self.ndim() >= 2, "NdTensor::nrows requires ndim >= 2");
        self.shape[self.ndim() - 2]
    }

    /// Convenience: number of cols for the last dimension.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        assert!(self.ndim() >= 1, "NdTensor::ncols requires ndim >= 1");
        self.shape[self.ndim() - 1]
    }

    /// Reorder axes. `axes` must be a permutation of `0..ndim()`.
    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        let nd = self.ndim();
        assert_eq!(
            axes.len(),
            nd,
            "nabla: permute axes length ({}) != ndim ({nd})",
            axes.len()
        );
        let mut seen = vec![false; nd];
        for &a in axes {
            assert!(a < nd, "nabla: permute axis {a} >= ndim {nd}");
            assert!(!seen[a], "nabla: permute duplicate axis {a}");
            seen[a] = true;
        }
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.dim(a)).collect();
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig_idx = vec![0usize; nd];
            for (new_axis, &orig_axis) in axes.iter().enumerate() {
                orig_idx[orig_axis] = idx[new_axis];
            }
            self.get_nd(&orig_idx)
        })
    }

    /// Remove a size-1 axis.
    #[must_use]
    pub fn squeeze(&self, axis: usize) -> Self {
        let nd = self.ndim();
        assert!(axis < nd, "nabla: squeeze axis {axis} >= ndim {nd}");
        let d = self.dim(axis);
        assert_eq!(d, 1, "nabla: squeeze dim({axis}) is {d}, not 1");
        let new_shape: Vec<usize> = (0..nd)
            .filter(|&i| i != axis)
            .map(|i| self.dim(i))
            .collect();
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig = idx.to_vec();
            orig.insert(axis, 0);
            self.get_nd(&orig)
        })
    }

    /// Insert a size-1 axis at `axis`.
    #[must_use]
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let nd = self.ndim();
        assert!(
            axis <= nd,
            "nabla: unsqueeze axis {axis} > ndim {nd}"
        );
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig = idx.to_vec();
            orig.remove(axis);
            self.get_nd(&orig)
        })
    }
}

impl<T: Scalar + fmt::Debug> fmt::Debug for NdTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdTensor(shape={:?}, data={:?})", self.shape, self.data)
    }
}

impl<T: Scalar> Index<&[usize]> for NdTensor<T> {
    type Output = T;

    #[inline]
    fn index(&self, indices: &[usize]) -> &T {
        assert!(
            indices.len() == self.shape.len(),
            "nabla: NdTensor expected {} indices, got {}",
            self.shape.len(),
            indices.len()
        );
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            assert!(
                idx < dim,
                "nabla: NdTensor index[{i}]={idx} out of bounds for dim {dim}"
            );
        }
        let flat = self.linear_index(indices);
        &self.data[flat]
    }
}

// Static matrices and abstract type hierarchy.

/// A stack-allocated `R x C` matrix with element type `T`.
#[derive(Clone, Copy)]
pub struct StaticMatrix<T: Scalar, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T: Scalar, const R: usize, const C: usize> StaticMatrix<T, R, C> {
    /// Zero-filled `R x C` matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [[T::zero_impl(); C]; R],
        }
    }

    /// Matrix whose `(r, c)` element is `f(r, c)`.
    #[must_use]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = [[T::zero_impl(); C]; R];
        for (r, row) in data.iter_mut().enumerate() {
            for (c, elem) in row.iter_mut().enumerate() {
                *elem = f(r, c);
            }
        }
        Self { data }
    }

    /// Identity-like matrix: `1` on the diagonal, `0` elsewhere.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_fn(|r, c| {
            if r == c {
                T::one_impl()
            } else {
                T::zero_impl()
            }
        })
    }

    /// Number of rows (always `R`).
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        R
    }

    /// Number of columns (always `C`).
    #[must_use]
    #[inline]
    pub const fn ncols(&self) -> usize {
        C
    }

    /// Shape `(R, C)`.
    #[must_use]
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Read element at `(row, col)`.
    #[must_use]
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row][col]
    }

    /// Write element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[row][col] = val;
    }

    /// Transpose: returns a new `C x R` static matrix (stack-only).
    #[must_use]
    pub fn t(&self) -> StaticMatrix<T, C, R> {
        StaticMatrix::<T, C, R>::from_fn(|r, c| self.data[c][r])
    }

    /// Conjugate-transpose (adjoint).
    #[must_use]
    pub fn adjoint(&self) -> StaticMatrix<T, C, R> {
        if T::IS_REAL {
            return self.t();
        }
        StaticMatrix::<T, C, R>::from_fn(|r, c| crate::scalar::math_utils::conj(&self.data[c][r]))
    }

    /// Matrix multiply `self (RxK) * rhs (KxN)` -> `StaticMatrix<T, R, N>`.
    #[must_use]
    pub fn matmul<const N: usize>(&self, rhs: &StaticMatrix<T, C, N>) -> StaticMatrix<T, R, N> {
        StaticMatrix::<T, R, N>::from_fn(|r, c| {
            (0..C).fold(T::zero_impl(), |acc, k| {
                acc + self.data[r][k] * rhs.data[k][c]
            })
        })
    }

    /// Outer product of two vectors: `u[i] * v[j]` -> `R x C` matrix.
    ///
    /// `out[i][j] = u[i] * v[j]`
    #[must_use]
    pub fn outer(u: &[T; R], v: &[T; C]) -> Self {
        Self::from_fn(|r, c| u[r] * v[c])
    }

    /// Read-only access to the raw `[[T; C]; R]` backing array.
    #[must_use]
    #[inline]
    pub fn data(&self) -> &[[T; C]; R] {
        &self.data
    }

    /// Convert to a heap-allocated [`Tensor`] (copies all elements).
    #[must_use]
    pub fn to_tensor(&self) -> Tensor<T, DefaultBackend> {
        Tensor::from_fn(R, C, |r, c| self.data[r][c])
    }

    /// Build from a [`Tensor`] (copies all elements).
    #[must_use]
    pub fn from_tensor(tensor: &Tensor<T, DefaultBackend>) -> Self {
        assert_eq!(
            tensor.shape(),
            (R, C),
            "StaticMatrix::from_tensor shape mismatch: expected ({R}, {C}), got {:?}",
            tensor.shape()
        );
        Self::from_fn(|r, c| tensor.get(r, c))
    }
}

macro_rules! impl_static_binop {
    (binary: $trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar, const R: usize, const C: usize> $trait for StaticMatrix<T, R, C> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                Self::from_fn(|r, c| self.data[r][c] $op rhs.data[r][c])
            }
        }
    };
    (unary: $trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar, const R: usize, const C: usize> $trait for StaticMatrix<T, R, C> {
            type Output = Self;
            fn $method(self) -> Self {
                Self::from_fn(|r, c| $op self.data[r][c])
            }
        }
    };
    (scalar: $trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar, const R: usize, const C: usize> $trait<T> for StaticMatrix<T, R, C> {
            type Output = Self;
            fn $method(self, rhs: T) -> Self {
                Self::from_fn(|r, c| self.data[r][c] $op rhs)
            }
        }
    };
}

impl_static_binop!(binary: Add, add, +);
impl_static_binop!(binary: Sub, sub, -);
impl_static_binop!(unary: Neg, neg, -);
impl_static_binop!(scalar: Mul, mul, *);

/// Multiply `StaticMatrix<T, R, K>` by `StaticMatrix<T, K, C>`.
/// Compile-time error if inner dimensions don't match.
impl<T: Scalar, const R: usize, const K: usize, const N: usize> Mul<StaticMatrix<T, K, N>>
    for StaticMatrix<T, R, K>
{
    type Output = StaticMatrix<T, R, N>;
    fn mul(self, rhs: StaticMatrix<T, K, N>) -> Self::Output {
        self.matmul(&rhs)
    }
}

// Reference-based operators for StaticMatrix (match nabla convention: &a op &b).

impl<T: Scalar, const R: usize, const C: usize> Add<&StaticMatrix<T, R, C>>
    for &StaticMatrix<T, R, C>
{
    type Output = StaticMatrix<T, R, C>;
    fn add(self, rhs: &StaticMatrix<T, R, C>) -> Self::Output {
        StaticMatrix::from_fn(|r, c| self.data[r][c] + rhs.data[r][c])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Sub<&StaticMatrix<T, R, C>>
    for &StaticMatrix<T, R, C>
{
    type Output = StaticMatrix<T, R, C>;
    fn sub(self, rhs: &StaticMatrix<T, R, C>) -> Self::Output {
        StaticMatrix::from_fn(|r, c| self.data[r][c] - rhs.data[r][c])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Neg for &StaticMatrix<T, R, C> {
    type Output = StaticMatrix<T, R, C>;
    fn neg(self) -> Self::Output {
        StaticMatrix::from_fn(|r, c| -self.data[r][c])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Mul<T> for &StaticMatrix<T, R, C> {
    type Output = StaticMatrix<T, R, C>;
    fn mul(self, rhs: T) -> Self::Output {
        StaticMatrix::from_fn(|r, c| self.data[r][c] * rhs)
    }
}

impl<T: Scalar, const R: usize, const K: usize, const N: usize> Mul<&StaticMatrix<T, K, N>>
    for &StaticMatrix<T, R, K>
{
    type Output = StaticMatrix<T, R, N>;
    fn mul(self, rhs: &StaticMatrix<T, K, N>) -> Self::Output {
        self.matmul(rhs)
    }
}

impl<T: Scalar, const R: usize, const C: usize> Index<(usize, usize)>
    for StaticMatrix<T, R, C>
{
    type Output = T;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row][col]
    }
}

impl<T: Scalar, const R: usize, const C: usize> IndexMut<(usize, usize)>
    for StaticMatrix<T, R, C>
{
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[row][col]
    }
}

impl<T: Scalar + fmt::Display, const R: usize, const C: usize> fmt::Display
    for StaticMatrix<T, R, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(R, C, |r, c, f| write!(f, "{}", self.data[r][c]), f, None)
    }
}

impl<T: Scalar + fmt::Debug, const R: usize, const C: usize> fmt::Debug for StaticMatrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = format!("StaticMatrix({R}x{C})");
        fmt_matrix(
            R,
            C,
            |r, c, f| write!(f, "{:?}", self.data[r][c]),
            f,
            Some(&prefix),
        )
    }
}

/// Base trait for any 2-D array-like type.
pub trait Array<T: Scalar> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Shape as `(nrows, ncols)`.
    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    /// Read element at `(row, col)`.
    fn get(&self, row: usize, col: usize) -> T;
}

#[cfg(feature = "cpu")]
/// Matrix algebra trait extending [`Array`].
pub trait Matrix<T: Scalar>: Array<T> {
    /// Transpose into a CPU-backed [`Tensor`].
    fn t_dyn(&self) -> Tensor<T, Cpu>;
    /// Matrix multiply `self x rhs` into a CPU-backed [`Tensor`].
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu>;
}

#[cfg(feature = "cpu")]
fn dyn_matmul<T: Scalar>(lhs: &dyn Array<T>, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
    let (m, k) = lhs.shape();
    let (k2, n) = rhs.shape();
    assert_eq!(
        k, k2,
        "matmul_dyn inner dimension mismatch: lhs k={k}, rhs k={k2}"
    );
    Tensor::from_fn(m, n, |r, c| {
        (0..k).fold(T::zero_impl(), |acc, i| acc + lhs.get(r, i) * rhs.get(i, c))
    })
}

impl<T: Scalar, B: Backend> Array<T> for Tensor<T, B> {
    #[inline]
    fn nrows(&self) -> usize {
        Tensor::nrows(self)
    }
    #[inline]
    fn ncols(&self) -> usize {
        Tensor::ncols(self)
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        Tensor::get(self, row, col)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Matrix<T> for Tensor<T, B> {
    fn t_dyn(&self) -> Tensor<T, Cpu> {
        let (r, c) = self.shape();
        Tensor::from_fn(c, r, |i, j| self.get(j, i))
    }
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
        dyn_matmul(self, rhs)
    }
}

impl<T: Scalar, const R: usize, const C: usize> Array<T> for StaticMatrix<T, R, C> {
    #[inline]
    fn nrows(&self) -> usize {
        R
    }
    #[inline]
    fn ncols(&self) -> usize {
        C
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        StaticMatrix::get(self, row, col)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, const R: usize, const C: usize> Matrix<T> for StaticMatrix<T, R, C> {
    fn t_dyn(&self) -> Tensor<T, Cpu> {
        Tensor::from_fn(C, R, |r, c| self.get(c, r))
    }
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
        dyn_matmul(self, rhs)
    }
}

// -- DynTensor: enum-based closed multiple dispatch --

#[cfg(feature = "cpu")]
/// A type-erased 2-D dense matrix that can live on any enabled backend.
pub enum DynTensor {
    /// CPU backend (always available).
    Cpu(Tensor<f32, Cpu>),
}

#[cfg(feature = "cpu")]
macro_rules! dyn_dispatch {
    (ref $self:expr, $method:ident) => {
        match $self { DynTensor::Cpu(t) => t.$method() }
    };
    (binop $self:expr, $rhs:expr, $op:tt) => {
        match ($self, $rhs) { (DynTensor::Cpu(a), DynTensor::Cpu(b)) => DynTensor::Cpu(a $op b) }
    };
}

#[cfg(feature = "cpu")]
impl DynTensor {
    /// Construct a `DynTensor` on the CPU backend.
    pub fn cpu_f32(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> f32) -> Self {
        Self::Cpu(Tensor::from_fn(nrows, ncols, f))
    }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        dyn_dispatch!(ref self, nrows)
    }
    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        dyn_dispatch!(ref self, ncols)
    }
    /// Shape as `(nrows, ncols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Read element at `(row, col)`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        match self {
            Self::Cpu(t) => t.get(row, col),
        }
    }

    /// Element-wise addition. Panics if shapes differ.
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        dyn_dispatch!(binop self, rhs, +)
    }
    /// Element-wise subtraction. Panics if shapes differ.
    #[must_use]
    pub fn sub(&self, rhs: &Self) -> Self {
        dyn_dispatch!(binop self, rhs, -)
    }

    /// Matrix multiplication (`self x rhs`). Panics if shapes are incompatible.
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Self::Cpu(a), Self::Cpu(b)) => {
                let mut out = Tensor::<f32, Cpu>::zeros(a.nrows(), b.ncols());
                Tensor::matmul_into(&mut out, a, b);
                Self::Cpu(out)
            }
        }
    }

    /// Move to CPU-backed tensor (always a clone for the Cpu variant).
    #[must_use]
    pub fn to_cpu(&self) -> Tensor<f32, Cpu> {
        match self {
            Self::Cpu(t) => t.clone(),
        }
    }
}
