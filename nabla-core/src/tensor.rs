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
        Tensor {
            storage: self.storage,
            _axes: PhantomData,
        }
    }

    /// Erase axis types back to untyped `()`.
    #[inline]
    #[must_use]
    pub fn erase_axes(self) -> Tensor<T, B> {
        Tensor {
            storage: self.storage,
            _axes: PhantomData,
        }
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
        (&raw const self.storage).cast::<u8>()
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Fused element-wise kernel launch (for fuse! macro codegen).
    ///
    /// GPU backends JIT-compile the expression; CPU backends use the closure.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
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
        Self::from_storage(B::fuse_launch(
            inputs,
            nrows,
            ncols,
            cpu_fn,
            gpu_expr,
            kernel_hash,
            n_inputs,
            reg_estimate,
        ))
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
fn two<T: Scalar>() -> T {
    T::one() + T::one()
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Validate two tensors have the same shape.
    fn assert_same_shape(&self, other: &Self, op: &str) {
        assert!(
            self.nrows() == other.nrows() && self.ncols() == other.ncols(),
            "{op}: shape mismatch ({},{}) vs ({},{})",
            self.nrows(),
            self.ncols(),
            other.nrows(),
            other.ncols()
        );
    }

    /// Validate vector matches row/column dimension for broadcast.
    fn assert_broadcast_shape(&self, vec: &Self, axis: usize, op: &str) {
        if axis == 0 {
            assert!(
                vec.nrows() == 1 && vec.ncols() == self.ncols(),
                "{op}: expected (1,{}) got ({},{})",
                self.ncols(),
                vec.nrows(),
                vec.ncols()
            );
        } else {
            assert!(
                vec.ncols() == 1 && vec.nrows() == self.nrows(),
                "{op}: expected ({},1) got ({},{})",
                self.nrows(),
                vec.nrows(),
                vec.ncols()
            );
        }
    }

    /// Reduce along axis with a custom fold function and initial value from first element.
    fn reduce_axis<F: Fn(T, T) -> T>(&self, axis: usize, f: F) -> Self {
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                (0..self.nrows())
                    .map(|r| self.get(r, c))
                    .fold(self.get(0, c), &f)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                (0..self.ncols())
                    .map(|c| self.get(r, c))
                    .fold(self.get(r, 0), &f)
            }),
            _ => panic!("nabla: reduce_axis axis must be 0 or 1, got {axis}"),
        }
    }

    /// Apply a binary op between self and a broadcast vector along the given axis.
    fn apply_broadcast_op<F: Fn(T, T) -> T>(
        &self,
        vec: &Self,
        axis: usize,
        op: &str,
        f: F,
    ) -> Self {
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
        Self {
            storage,
            _axes: PhantomData,
        }
    }

    /// Consume this tensor and return its backing storage.
    pub fn into_storage(self) -> B::Storage<T> {
        self.storage
    }

    /// Borrow the backing storage.
    pub fn storage(&self) -> &B::Storage<T> {
        &self.storage
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
        assert_eq!(
            data.len(),
            nrows * ncols,
            "from_slice_async: data.len() must equal nrows * ncols"
        );
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
        Self::from_storage(B::from_fn(nrows, ncols, |r, c| {
            self.get(row_start + r, col_start + c)
        }))
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
        assert_eq!(
            targets.shape(),
            (batch, n),
            "nabla: cross_entropy_loss shape mismatch — self {}×{} vs targets {}×{}",
            batch,
            n,
            targets.nrows(),
            targets.ncols()
        );
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

    /// Lp norm: `(sum |x_i|^p)^(1/p)`, or `max|x_i|` for p=∞.
    /// Dispatches to `B::norm_lp` which uses GPU-accelerated operations.
    #[must_use]
    pub fn norm_lp(&self, p: T) -> T {
        B::norm_lp(&self.storage, p)
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

    /// Element-wise SiLU (Swish): `x * sigmoid(x)`.
    #[must_use]
    pub fn silu(&self) -> Self {
        Self::from_storage(B::silu(&self.storage))
    }

    /// Element-wise Mish: `x * tanh(softplus(x))` where softplus(x) = ln(1 + exp(x)).
    #[must_use]
    pub fn mish(&self) -> Self {
        Self::from_storage(B::mish(&self.storage))
    }

    /// Element-wise Leaky ReLU: `max(alpha * x, x)`.
    #[must_use]
    pub fn leaky_relu(&self, alpha: T) -> Self {
        Self::from_storage(B::leaky_relu(&self.storage, alpha))
    }

    /// Element-wise ELU: `x if x > 0, alpha * (exp(x) - 1) otherwise`.
    #[must_use]
    pub fn elu(&self, alpha: T) -> Self {
        Self::from_storage(B::elu(&self.storage, alpha))
    }

    /// Element-wise HardSwish: `x * relu6(x + 3) / 6`.
    #[must_use]
    pub fn hardswish(&self) -> Self {
        Self::from_storage(B::hardswish(&self.storage))
    }

    /// Softmax along given axis (0=columns, 1=rows).
    ///
    /// Uses the log-sum-exp trick for numerical stability: subtract max before exp.
    #[must_use]
    pub fn softmax(&self, axis: usize) -> Self {
        match axis {
            1 => Self::from_storage(B::softmax(&self.storage)),
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

    /// Short alias for conjugate transpose (`adjoint` / Hermitian transpose).
    #[must_use]
    #[inline]
    pub fn h(&self) -> Self {
        self.adjoint()
    }

    /// Permute axes of a 2-D tensor.
    ///
    /// `axes` must be a permutation of `[0, 1]`:
    /// - `[0, 1]` -> identity (clone)
    /// - `[1, 0]` -> transpose
    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            2,
            "nabla: Tensor is 2-D — permute axes must have length 2"
        );
        assert!(
            axes[0] < 2 && axes[1] < 2 && axes[0] != axes[1],
            "nabla: permute axes must be a permutation of {{0, 1}}, got [{}, {}]",
            axes[0],
            axes[1]
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

    /// Unflatten a dimension: reshape `axis` dim by splitting it into `sizes`.
    ///
    /// For a 2-D tensor: axis=0 splits rows, axis=1 splits columns.
    /// `sizes` is `(new_axis_size, other_size)` such that `new_axis_size * other_size = dim_size`.
    /// Result: axis=0 → `(sizes.0, sizes.1 * ncols)`, axis=1 → `(nrows * sizes.0, sizes.1)`.
    #[must_use]
    pub fn unflatten(&self, axis: usize, sizes: (usize, usize)) -> Self {
        let (m, n) = self.shape();
        match axis {
            0 => {
                assert_eq!(
                    sizes.0 * sizes.1,
                    m,
                    "unflatten: {m} != {}*{}",
                    sizes.0,
                    sizes.1
                );
                self.reshape(sizes.0, sizes.1 * n)
            }
            1 => {
                assert_eq!(
                    sizes.0 * sizes.1,
                    n,
                    "unflatten: {n} != {}*{}",
                    sizes.0,
                    sizes.1
                );
                self.reshape(m * sizes.0, sizes.1)
            }
            _ => panic!("unflatten: axis {axis} out of bounds for 2-D tensor"),
        }
    }

    /// Return a contiguous copy of the tensor (forces row-major layout).
    ///
    /// After `transpose()` or `permute()`, the logical layout may differ from
    /// physical memory. `contiguous()` materializes into a fresh, dense buffer.
    #[must_use]
    pub fn contiguous(&self) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| self.get(r, c))
    }

    /// Detach from the computation graph: returns a clone with no gradient tracking.
    ///
    /// For tensors without AD, this is equivalent to `clone()`.
    #[must_use]
    pub fn detach(&self) -> Self {
        self.clone()
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
            1 => Self::from_storage(B::cumsum_axis1(&self.storage)),
            0 => {
                // axis=0: transpose → cumsum_axis1 → transpose
                let t = Self::from_storage(B::transpose(&self.storage));
                let cs = Self::from_storage(B::cumsum_axis1(&t.storage));
                Self::from_storage(B::transpose(&cs.storage))
            }
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
                assert_eq!(r % n, 0, "nabla: chunk nrows ({r}) not divisible by {n}");
                let chunk_size = r / n;
                (0..n)
                    .map(|i| self.slice_rows(i * chunk_size..(i + 1) * chunk_size))
                    .collect()
            }
            1 => {
                let c = self.ncols();
                assert_eq!(c % n, 0, "nabla: chunk ncols ({c}) not divisible by {n}");
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
        assert_eq!(d, 1, "nabla: squeeze({axis}) — dim is {d}, not 1");
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
                let centered = self.broadcast_add_cols(&neg_mean);
                centered.broadcast_mul_cols(&inv_std)
            }
            0 => {
                let centered = self.broadcast_add_rows(&neg_mean);
                centered.broadcast_mul_rows(&inv_std)
            }
            _ => panic!("nabla: layer_norm axis must be 0 or 1, got {axis}"),
        }
    }

    // ── RMS normalization ───────────────────────────────────────────

    /// RMS normalization along `axis`: `x / rms(x) * weight`.
    ///
    /// `weight` shape must match the normalized dimension (broadcast).
    #[must_use]
    pub fn rms_norm(&self, axis: usize, weight: &Self, eps: T) -> Self {
        let (m, n) = self.shape();
        match axis {
            1 => {
                let rms = Self::from_fn(m, 1, |r, _| {
                    let sq_sum = (0..n).fold(T::zero(), |acc, c| {
                        let v = self.get(r, c);
                        acc + v * v
                    });
                    (sq_sum / T::from_f64(n as f64) + eps).math_sqrt()
                });
                let normed =
                    self.broadcast_mul_cols(&Self::from_fn(m, 1, |r, c| T::one() / rms.get(r, c)));
                normed.broadcast_mul_rows(weight)
            }
            0 => {
                let rms = Self::from_fn(1, n, |_, c| {
                    let sq_sum = (0..m).fold(T::zero(), |acc, r| {
                        let v = self.get(r, c);
                        acc + v * v
                    });
                    (sq_sum / T::from_f64(m as f64) + eps).math_sqrt()
                });
                let normed =
                    self.broadcast_mul_rows(&Self::from_fn(1, n, |r, c| T::one() / rms.get(r, c)));
                normed.broadcast_mul_cols(weight)
            }
            _ => panic!("nabla: rms_norm axis must be 0 or 1, got {axis}"),
        }
    }

    /// Batch normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.
    ///
    /// `x` is (N, C). `weight` and `bias` are (1, C). In eval mode, `running_mean`
    /// and `running_var` are (1, C).
    #[must_use]
    pub fn batch_norm(
        &self,
        running_mean: &Self,
        running_var: &Self,
        weight: &Self,
        bias: &Self,
        eps: T,
    ) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            let mu = running_mean.get(0, c);
            let var = running_var.get(0, c);
            let w = weight.get(0, c);
            let b = bias.get(0, c);
            (x - mu) / (var + eps).math_sqrt() * w + b
        })
    }

    /// Batch normalization with running statistics update (training mode).
    ///
    /// `self` is (N, C). `gamma`/`beta` are (1, C). `running_mean`/`running_var` are (1, C)
    /// updated in-place when `training=true`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_train(
        &self,
        gamma: &Self,
        beta: &Self,
        running_mean: &mut Self,
        running_var: &mut Self,
        eps: T,
        momentum: T,
        training: bool,
    ) -> Self {
        Self {
            storage: B::batch_norm_train(
                &self.storage,
                &gamma.storage,
                &beta.storage,
                &mut running_mean.storage,
                &mut running_var.storage,
                eps,
                momentum,
                training,
            ),
            _axes: PhantomData,
        }
    }

    /// Cross-entropy loss: fused softmax + NLL. `self` = (N, C) logits,
    /// `target` = (N, 1) class indices stored as T. Returns (1, 1) scalar tensor.
    #[must_use]
    pub fn cross_entropy_fused(&self, target: &Self) -> Self {
        let (n, c) = self.shape();
        assert_eq!(
            target.nrows(),
            n,
            "nabla: cross_entropy_fused shape mismatch — input {}×{} vs target {}×{}",
            n,
            c,
            target.nrows(),
            target.ncols()
        );
        Self {
            storage: B::cross_entropy_fused(&self.storage, &target.storage, n, c),
            _axes: PhantomData,
        }
    }

    /// Group normalization: divide channels into groups, normalize each group.
    ///
    /// `x` is (N, C). `num_groups` divides C. `weight` and `bias` are (1, C).
    #[must_use]
    pub fn group_norm(&self, num_groups: usize, weight: &Self, bias: &Self, eps: T) -> Self {
        let (m, n) = self.shape();
        assert!(
            n % num_groups == 0,
            "nabla: group_norm C={n} not divisible by groups={num_groups}"
        );
        let g_size = n / num_groups;
        Self::from_fn(m, n, |r, c| {
            let g = c / g_size;
            let g_start = g * g_size;
            let mean = (0..g_size).fold(T::zero(), |acc, j| acc + self.get(r, g_start + j))
                / T::from_f64(g_size as f64);
            let var = (0..g_size).fold(T::zero(), |acc, j| {
                let d = self.get(r, g_start + j) - mean;
                acc + d * d
            }) / T::from_f64(g_size as f64);
            let x = self.get(r, c);
            (x - mean) / (var + eps).math_sqrt() * weight.get(0, c) + bias.get(0, c)
        })
    }

    // ── Loss functions ──────────────────────────────────────────────

    /// MSE loss: `mean((pred - target)^2)`.
    #[must_use]
    pub fn mse_loss(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let d = self.get(r, c) - target.get(r, c);
                acc2 + d * d
            })
        });
        sum / total
    }

    /// L1 loss: `mean(|pred - target|)`.
    #[must_use]
    pub fn l1_loss(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                acc2 + (self.get(r, c) - target.get(r, c)).math_abs()
            })
        });
        sum / total
    }

    /// Smooth L1 (Huber) loss with transition point `beta`.
    #[must_use]
    pub fn smooth_l1_loss(&self, target: &Self, beta: T) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let half = T::from_f64(0.5);
        let two = two::<T>();
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let d = (self.get(r, c) - target.get(r, c)).math_abs();
                // if d < beta: 0.5 * d^2 / beta, else: d - 0.5 * beta
                // Use branchless: pick = min(d, beta)
                let pick = (d + beta - (d - beta).math_abs()) / two; // min(d, beta)
                // When d < beta: pick = d, cost = 0.5 * d * d / beta
                // When d >= beta: pick = beta, cost = d - 0.5 * beta
                // Blend: 0.5 * pick * pick / beta + (d - pick) (which is 0 when d<beta)
                acc2 + half * pick * pick / beta + (d - pick)
            })
        });
        sum / total
    }

    /// Binary cross-entropy with logits: `-[y * log(σ(x)) + (1-y) * log(1-σ(x))]`.
    #[must_use]
    pub fn bce_with_logits(&self, target: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64((m * n) as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let x = self.get(r, c);
                let y = target.get(r, c);
                // Numerically stable: max(x, 0) - x*y + log(1 + exp(-|x|))
                let abs_x = x.math_abs();
                let relu_x = (x + abs_x) / two::<T>();
                acc2 + relu_x - x * y + (T::one() + (T::zero() - abs_x).math_exp()).math_ln()
            })
        });
        sum / total
    }

    /// Negative log-likelihood loss. `self` is log-probabilities (N, C), `targets` is class indices as (N, 1).
    #[must_use]
    pub fn nll_loss(&self, targets: &Self) -> T {
        let m = self.nrows();
        let sum = (0..m).fold(T::zero(), |acc, r| {
            let cls = targets.get(r, 0).to_f64() as usize;
            acc - self.get(r, cls)
        });
        sum / T::from_f64(m as f64)
    }

    /// KL divergence: `sum(q * (log(q) - log_p))` (batchmean reduction).
    ///
    /// `self` is log_p (log-probabilities), `q` is target distribution.
    #[must_use]
    pub fn kl_div(&self, q: &Self) -> T {
        let (m, n) = self.shape();
        let total = T::from_f64(m as f64);
        let sum = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |acc2, c| {
                let qv = q.get(r, c);
                let log_p = self.get(r, c);
                acc2 + qv * (qv.math_ln() - log_p)
            })
        });
        sum / total
    }

    /// Cosine embedding loss for pairs `(x1, x2)` with label `y` ∈ {1, -1}.
    ///
    /// When y=1: `1 - cos(x1, x2)`. When y=-1: `max(0, cos(x1, x2) - margin)`.
    #[must_use]
    pub fn cosine_embedding_loss(x1: &Self, x2: &Self, y: T, margin: T) -> T {
        let (m, n) = x1.shape();
        let dot = (0..m).fold(T::zero(), |acc, r| {
            (0..n).fold(acc, |a, c| a + x1.get(r, c) * x2.get(r, c))
        });
        let n1 = x1.norm();
        let n2 = x2.norm();
        let eps = T::from_f64(1e-8);
        let cos_sim = dot / (n1 * n2 + eps);
        let two = two::<T>();
        if y.to_f64() > 0.0 {
            T::one() - cos_sim
        } else {
            let v = cos_sim - margin;
            (v + v.math_abs()) / two // max(0, v)
        }
    }

    // ── Reduction extensions ────────────────────────────────────────

    /// Product of all elements.
    #[must_use]
    pub fn prod_all(&self) -> T {
        B::prod_all(&self.storage)
    }

    /// Count of non-zero elements.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        B::count_nonzero(&self.storage)
    }

    /// Argmax along axis. Returns indices tensor.
    #[must_use]
    pub fn argmax_axis(&self, axis: usize) -> Self {
        let two = two::<T>();
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                let mut best_idx = 0usize;
                let mut best_val = self.get(0, c);
                for r in 1..self.nrows() {
                    let v = self.get(r, c);
                    let diff = v - best_val;
                    let is_gt = (diff + diff.math_abs()) / two; // > 0 when v > best
                    if is_gt.to_f64() > 0.0 {
                        best_val = v;
                        best_idx = r;
                    }
                }
                T::from_f64(best_idx as f64)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                let mut best_idx = 0usize;
                let mut best_val = self.get(r, 0);
                for c in 1..self.ncols() {
                    let v = self.get(r, c);
                    let diff = v - best_val;
                    let is_gt = (diff + diff.math_abs()) / two;
                    if is_gt.to_f64() > 0.0 {
                        best_val = v;
                        best_idx = c;
                    }
                }
                T::from_f64(best_idx as f64)
            }),
            _ => panic!("nabla: argmax_axis axis must be 0 or 1, got {axis}"),
        }
    }

    /// Argmin along axis. Returns indices tensor.
    #[must_use]
    pub fn argmin_axis(&self, axis: usize) -> Self {
        let neg = -self;
        neg.argmax_axis(axis)
    }

    /// Cumulative product along axis.
    #[must_use]
    pub fn cumprod(&self, axis: usize) -> Self {
        match axis {
            1 => Self::from_storage(B::cumprod_axis1(&self.storage)),
            0 => {
                // axis=0: transpose → cumprod_axis1 → transpose
                let t = Self::from_storage(B::transpose(&self.storage));
                let cp = Self::from_storage(B::cumprod_axis1(&t.storage));
                Self::from_storage(B::transpose(&cp.storage))
            }
            _ => panic!("nabla: cumprod axis must be 0 or 1, got {axis}"),
        }
    }

    /// Lp-norm along axis.
    #[must_use]
    pub fn norm_axis(&self, p: T, axis: usize) -> Self {
        let pf = p.to_f64();
        let inv_p = T::from_f64(1.0 / pf);
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                let sum = (0..self.nrows()).fold(T::zero(), |acc, r| {
                    acc + self.get(r, c).math_abs().math_powf(p)
                });
                sum.math_powf(inv_p)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                let sum = (0..self.ncols()).fold(T::zero(), |acc, c| {
                    acc + self.get(r, c).math_abs().math_powf(p)
                });
                sum.math_powf(inv_p)
            }),
            _ => panic!("nabla: norm_axis axis must be 0 or 1, got {axis}"),
        }
    }

    // ── Construction / utility ──────────────────────────────────────

    /// Uninitialized tensor (actually zeroed — Rust safety).
    #[must_use]
    pub fn empty(nrows: usize, ncols: usize) -> Self {
        Self::zeros(nrows, ncols)
    }

    /// Generate a 1-D tensor: `[start, start+step, start+2*step, ...]` with length `n`.
    #[must_use]
    pub fn arange(start: T, step: T, n: usize) -> Self {
        Self::from_fn(1, n, |_, c| start + step * T::from_f64(c as f64))
    }

    /// Generate a 1-D tensor of `n` evenly spaced values from `start` to `end` (inclusive).
    #[must_use]
    pub fn linspace(start: T, end: T, n: usize) -> Self {
        assert!(n >= 2, "nabla: linspace needs n >= 2");
        let denom = T::from_f64((n - 1) as f64);
        Self::from_fn(1, n, |_, c| {
            let t = T::from_f64(c as f64) / denom;
            start + (end - start) * t
        })
    }

    /// Same-shape tensor filled with `val`.
    #[must_use]
    pub fn full_like(&self, val: T) -> Self {
        Self::fill(self.nrows(), self.ncols(), val)
    }

    // ── Tensor manipulation ─────────────────────────────────────────

    /// Split into chunks of given sizes along axis.
    #[must_use]
    pub fn split(&self, sizes: &[usize], axis: usize) -> Vec<Self> {
        match axis {
            0 => {
                let mut offset = 0;
                sizes
                    .iter()
                    .map(|&s| {
                        let part = self.submatrix(offset, offset + s, 0, self.ncols());
                        offset += s;
                        part
                    })
                    .collect()
            }
            1 => {
                let mut offset = 0;
                sizes
                    .iter()
                    .map(|&s| {
                        let part = self.submatrix(0, self.nrows(), offset, offset + s);
                        offset += s;
                        part
                    })
                    .collect()
            }
            _ => panic!("nabla: split axis must be 0 or 1, got {axis}"),
        }
    }

    /// Repeat tensor along each axis. `reps = (row_repeats, col_repeats)`.
    #[must_use]
    pub fn repeat(&self, row_reps: usize, col_reps: usize) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m * row_reps, n * col_reps, |r, c| self.get(r % m, c % n))
    }

    /// Expand (broadcast view) — repeats without copying for broadcast dimensions.
    /// For a 2-D tensor, if a dimension is 1 it can be expanded to `target_dim`.
    #[must_use]
    pub fn expand(&self, target_rows: usize, target_cols: usize) -> Self {
        let (m, n) = self.shape();
        assert!(
            (m == 1 || m == target_rows) && (n == 1 || n == target_cols),
            "nabla: expand ({m},{n}) → ({target_rows},{target_cols}) invalid"
        );
        Self::from_fn(target_rows, target_cols, |r, c| {
            self.get(if m == 1 { 0 } else { r }, if n == 1 { 0 } else { c })
        })
    }

    /// Pad tensor. `padding = [left, right, top, bottom]`. Fill with `value`.
    #[must_use]
    pub fn pad(&self, padding: [usize; 4], value: T) -> Self {
        let [left, right, top, bottom] = padding;
        let (m, n) = self.shape();
        Self::from_fn(m + top + bottom, n + left + right, |r, c| {
            if r >= top && r < m + top && c >= left && c < n + left {
                self.get(r - top, c - left)
            } else {
                value
            }
        })
    }

    /// General gather along dimension.
    ///
    /// `index` shape determines output shape. For axis=1:
    /// `out[i][j] = self[i][index[i][j]]`
    #[must_use]
    pub fn gather(&self, axis: usize, index: &Self) -> Self {
        let (m, n) = index.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| self.get(index.get(r, c).to_f64() as usize, c)),
            1 => Self::from_fn(m, n, |r, c| self.get(r, index.get(r, c).to_f64() as usize)),
            _ => panic!("nabla: gather axis must be 0 or 1, got {axis}"),
        }
    }

    /// Scatter: write `src` values into self at positions given by `index` along `axis`.
    ///
    /// Returns a new tensor. For axis=1: `out[i][index[i][j]] = src[i][j]`.
    #[must_use]
    pub fn scatter(&self, axis: usize, index: &Self, src: &Self) -> Self {
        let mut out = self.clone();
        let (si, sj) = index.shape();
        for r in 0..si {
            for c in 0..sj {
                let idx = index.get(r, c).to_f64() as usize;
                let val = src.get(r, c);
                match axis {
                    0 => out.set(idx, c, val),
                    1 => out.set(r, idx, val),
                    _ => panic!("nabla: scatter axis must be 0 or 1"),
                }
            }
        }
        out
    }

    /// Select elements along axis by index vector. `index` is 1-D (1×K or K×1).
    #[must_use]
    pub fn index_select(&self, axis: usize, index: &Self) -> Self {
        let k = index.nrows() * index.ncols();
        let get_idx = |i: usize| -> usize {
            if index.nrows() == 1 {
                index.get(0, i).to_f64() as usize
            } else {
                index.get(i, 0).to_f64() as usize
            }
        };
        match axis {
            0 => Self::from_fn(k, self.ncols(), |r, c| self.get(get_idx(r), c)),
            1 => Self::from_fn(self.nrows(), k, |r, c| self.get(r, get_idx(c))),
            _ => panic!("nabla: index_select axis must be 0 or 1, got {axis}"),
        }
    }

    /// Replace elements where `mask` is non-zero with `value`.
    #[must_use]
    pub fn masked_fill(&self, mask: &Self, value: T) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if mask.get(r, c).to_f64() == 0.0 {
                self.get(r, c)
            } else {
                value
            }
        })
    }

    /// Element-wise conditional: `where cond != 0, pick self, else pick other`.
    #[must_use]
    pub fn where_cond(&self, cond: &Self, other: &Self) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if cond.get(r, c).to_f64() == 0.0 {
                other.get(r, c)
            } else {
                self.get(r, c)
            }
        })
    }

    /// Upper triangular matrix (zero below diagonal + offset).
    #[must_use]
    pub fn triu(&self, diagonal: isize) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if (c as isize) >= (r as isize) + diagonal {
                self.get(r, c)
            } else {
                T::zero()
            }
        })
    }

    /// Lower triangular matrix (zero above diagonal + offset).
    #[must_use]
    pub fn tril(&self, diagonal: isize) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if (c as isize) <= (r as isize) + diagonal {
                self.get(r, c)
            } else {
                T::zero()
            }
        })
    }

    /// Roll elements along axis by `shift` positions (circular shift).
    #[must_use]
    pub fn roll(&self, shift: isize, axis: usize) -> Self {
        let (m, n) = self.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| {
                self.get(((r as isize - shift).rem_euclid(m as isize)) as usize, c)
            }),
            1 => Self::from_fn(m, n, |r, c| {
                self.get(r, ((c as isize - shift).rem_euclid(n as isize)) as usize)
            }),
            _ => panic!("nabla: roll axis must be 0 or 1, got {axis}"),
        }
    }

    /// Flip (reverse) along axis.
    #[must_use]
    pub fn flip(&self, axis: usize) -> Self {
        let (m, n) = self.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| self.get(m - 1 - r, c)),
            1 => Self::from_fn(m, n, |r, c| self.get(r, n - 1 - c)),
            _ => panic!("nabla: flip axis must be 0 or 1, got {axis}"),
        }
    }

    /// Top-k values and indices along axis=1 (rows). Returns `(values, indices)`.
    #[must_use]
    pub fn topk(&self, k: usize, axis: usize) -> (Self, Self) {
        assert!(axis == 1, "nabla: topk currently supports axis=1 only");
        let (m, n) = self.shape();
        assert!(k <= n, "nabla: topk k={k} > ncols={n}");
        // For each row, find top-k by partial sort
        let mut all_vals = vec![T::zero(); m * k];
        let mut all_idxs = vec![T::zero(); m * k];
        for r in 0..m {
            let mut pairs: Vec<(T, usize)> = (0..n).map(|c| (self.get(r, c), c)).collect();
            // Sort descending by value
            pairs.sort_by(|a, b| b.0.to_f64().total_cmp(&a.0.to_f64()));
            for j in 0..k {
                all_vals[r * k + j] = pairs[j].0;
                all_idxs[r * k + j] = T::from_f64(pairs[j].1 as f64);
            }
        }
        let vals = Self::from_fn(m, k, |r, c| all_vals[r * k + c]);
        let idxs = Self::from_fn(m, k, |r, c| all_idxs[r * k + c]);
        (vals, idxs)
    }

    /// Sort along axis=1 (rows). Returns `(sorted_values, indices)`.
    #[must_use]
    pub fn sort(&self, axis: usize, descending: bool) -> (Self, Self) {
        assert!(axis == 1, "nabla: sort currently supports axis=1 only");
        let (m, n) = self.shape();
        let mut all_vals = vec![T::zero(); m * n];
        let mut all_idxs = vec![T::zero(); m * n];
        for r in 0..m {
            let mut pairs: Vec<(T, usize)> = (0..n).map(|c| (self.get(r, c), c)).collect();
            if descending {
                pairs.sort_by(|a, b| b.0.to_f64().total_cmp(&a.0.to_f64()));
            } else {
                pairs.sort_by(|a, b| a.0.to_f64().total_cmp(&b.0.to_f64()));
            }
            for c in 0..n {
                all_vals[r * n + c] = pairs[c].0;
                all_idxs[r * n + c] = T::from_f64(pairs[c].1 as f64);
            }
        }
        let vals = Self::from_fn(m, n, |r, c| all_vals[r * n + c]);
        let idxs = Self::from_fn(m, n, |r, c| all_idxs[r * n + c]);
        (vals, idxs)
    }

    /// Create 2-D meshgrid from two 1-D tensors. Returns `(grid_x, grid_y)`.
    #[must_use]
    pub fn meshgrid(x: &Self, y: &Self) -> (Self, Self) {
        let nx = x.nrows() * x.ncols();
        let ny = y.nrows() * y.ncols();
        let get_x = |i: usize| {
            if x.nrows() == 1 {
                x.get(0, i)
            } else {
                x.get(i, 0)
            }
        };
        let get_y = |i: usize| {
            if y.nrows() == 1 {
                y.get(0, i)
            } else {
                y.get(i, 0)
            }
        };
        let gx = Self::from_fn(ny, nx, |_, c| get_x(c));
        let gy = Self::from_fn(ny, nx, |r, _| get_y(r));
        (gx, gy)
    }

    // ── Batched operations ──────────────────────────────────────────
    // 2-D Tensor treated as batch of row-vectors or reshaped to batch of matrices.

    /// Batched matrix multiply: treat self as (B, M, K) and other as (B, K, N) packed row-major.
    ///
    /// Self is (B*M, K), other is (B*K, N). Output is (B*M, N).
    #[must_use]
    pub fn bmm(&self, other: &Self, batch: usize, m: usize, k: usize, n: usize) -> Self {
        assert_eq!(self.nrows(), batch * m);
        assert_eq!(self.ncols(), k);
        assert_eq!(other.nrows(), batch * k);
        assert_eq!(other.ncols(), n);
        Self::from_storage(B::bmm(&self.storage, &other.storage, batch, m, k, n))
    }

    /// `C = alpha * A @ B + beta * C` (addmm). Self is C, returns new tensor.
    #[must_use]
    pub fn addmm(&self, a: &Self, b: &Self, beta: T, alpha: T) -> Self {
        let (m, n) = self.shape();
        let k = a.ncols();
        assert_eq!(a.nrows(), m);
        assert_eq!(b.shape(), (k, n));
        Self::from_storage(B::addmm(&self.storage, &a.storage, &b.storage, beta, alpha))
    }

    /// Batched addmm: `C = beta * C + alpha * A @ B` for each batch.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn baddbmm(
        &self,
        a: &Self,
        b: &Self,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        beta: T,
        alpha: T,
    ) -> Self {
        assert_eq!(self.nrows(), batch * m);
        assert_eq!(self.ncols(), n);
        Self::from_storage(B::baddbmm(
            &self.storage,
            &a.storage,
            &b.storage,
            batch,
            m,
            k,
            n,
            beta,
            alpha,
        ))
    }

    // ── Convolution ─────────────────────────────────────────────────

    /// im2col: unfold input patches for convolution.
    ///
    /// Input `x` is (C_in, H*W) flattened. Returns (C_in*kH*kW, out_H*out_W).
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

    /// 2-D convolution: `conv2d(input, weight, bias, stride, padding, dilation, groups)`.
    ///
    /// Input: (N*C_in, H*W), Weight: (C_out, C_in/groups * kH * kW), Bias: (1, C_out) or empty.
    /// Output: (N*C_out, out_H * out_W).
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
    ///
    /// Input: (N*C_in, L), Weight: (C_out, C_in/groups * K), Bias: (1, C_out).
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
            let out_len = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            let bias_exp = B::from_fn(n_batch * c_out, out_len, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
        } else {
            out
        }
    }

    /// Transposed 2-D convolution (deconvolution / fractionally-strided convolution).
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
            let bias_exp = B::from_fn(n_batch * c_out, out_h * out_w, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
        } else {
            out
        }
    }

    // ── Pooling ─────────────────────────────────────────────────────

    /// 2-D max pooling.
    ///
    /// Input: (N*C, H*W). Output: (N*C, out_H*out_W).
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
    ///
    /// Returns `(values, indices)` where `indices[i]` is the flat index in the
    /// input tensor of the maximum element for output position `i`.
    /// Indices are stored as `T` (float-cast), matching `argmax_axis` convention.
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
    ///
    /// Input: (N*C, H*W). Output: (N*C, out_H*out_W).
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

    /// Adaptive average pool 2-D: output fixed size regardless of input.
    ///
    /// Input: (N*C, H*W). Output: (N*C, out_H*out_W).
    #[must_use]
    pub fn adaptive_avg_pool2d(&self, h: usize, w: usize, out_h: usize, out_w: usize) -> Self {
        Self::from_storage(B::adaptive_avg_pool2d(&self.storage, h, w, out_h, out_w))
    }

    /// 1-D max pooling. Input: (N*C, L). Output: (N*C, out_L).
    #[must_use]
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

    /// 1-D average pooling. Input: (N*C, L). Output: (N*C, out_L).
    #[must_use]
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

    // ── Attention / Transformer ─────────────────────────────────────

    /// Embedding lookup: select rows from weight matrix by indices.
    ///
    /// `indices` is (N, seq_len) containing integer indices.
    /// `weight` is (vocab_size, embed_dim).
    /// Output: (N * seq_len, embed_dim).
    #[must_use]
    pub fn embedding(indices: &Self, weight: &Self) -> Self {
        Self::from_storage(B::embedding(&indices.storage, &weight.storage))
    }

    /// Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`.
    ///
    /// Q: (seq_q, d_k), K: (seq_k, d_k), V: (seq_k, d_v).
    /// Optional `mask`: (seq_q, seq_k) — positions with non-zero values are masked (set to -inf).
    /// Returns: (seq_q, d_v).
    #[must_use]
    pub fn scaled_dot_product_attention(q: &Self, k: &Self, v: &Self, mask: Option<&Self>) -> Self {
        let d_k = q.ncols();
        let scale = T::from_f64(1.0 / (d_k as f64).sqrt());

        // scores = Q @ K^T * scale
        let kt = k.t();
        let mut scores = &(&q.clone() * &kt) * scale;

        // Apply mask: set masked positions to -inf
        if let Some(m) = mask {
            let neg_inf = T::from_f64(f64::NEG_INFINITY);
            scores = scores.masked_fill(m, neg_inf);
        }

        // softmax along axis=1 (each row)
        let attn = scores.softmax(1);

        // output = attn @ V
        &attn * v
    }

    /// Multi-head attention.
    ///
    /// Q, K, V: (seq, d_model). Splits into `num_heads` heads, applies SDPA, concatenates.
    /// Returns: (seq, d_model).
    #[must_use]
    pub fn multi_head_attention(
        q: &Self,
        k: &Self,
        v: &Self,
        num_heads: usize,
        mask: Option<&Self>,
    ) -> Self {
        let d_model = q.ncols();
        assert!(
            d_model.is_multiple_of(num_heads),
            "nabla: d_model must be divisible by num_heads"
        );
        let d_head = d_model / num_heads;
        let seq_q = q.nrows();
        let seq_k = k.nrows();

        // Split into heads and compute attention for each
        let mut head_outputs: Vec<Self> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let q_h = q.submatrix(0, seq_q, h * d_head, (h + 1) * d_head);
            let k_h = k.submatrix(0, seq_k, h * d_head, (h + 1) * d_head);
            let v_h = v.submatrix(0, seq_k, h * d_head, (h + 1) * d_head);
            head_outputs.push(Self::scaled_dot_product_attention(&q_h, &k_h, &v_h, mask));
        }

        // Concatenate heads along columns
        let refs: Vec<&Self> = head_outputs.iter().collect();
        Self::hcat(&refs)
    }

    /// Scaled dot-product attention with FlashAttention-2 on GPU backends.
    ///
    /// Q, K, V layout: `(batch_heads * seq_{q,k}, head_dim)` — 2-D row-major.
    /// Returns `(batch_heads * seq_q, head_dim)`.
    ///
    /// `head_dim` must be ≤ 128 (compile-enforced per REQ-G-ATTN spec).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        q: &Self,
        k: &Self,
        v: &Self,
        mask: Option<&Self>,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        batch_heads: usize,
    ) -> Self {
        assert!(
            head_dim <= 128,
            "nabla: sdpa head_dim must be ≤ 128 (FA_HEAD_DIM_MAX), got {head_dim}"
        );
        assert_eq!(
            q.nrows(),
            batch_heads * seq_q,
            "nabla: sdpa Q nrows must equal batch_heads*seq_q"
        );
        assert_eq!(
            q.ncols(),
            head_dim,
            "nabla: sdpa Q ncols must equal head_dim"
        );
        Self::from_storage(B::sdpa(
            &q.storage,
            &k.storage,
            &v.storage,
            mask.map(|m| &m.storage),
            seq_q,
            seq_k,
            head_dim,
            batch_heads,
        ))
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

    // ── Conv3D ──────────────────────────────────────────────────────

    /// 3-D convolution.
    ///
    /// Input: (N*C_in, D*H*W), Weight: (C_out, C_in/groups * kD * kH * kW).
    /// Output: (N*C_out, out_D * out_H * out_W).
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
        let out_d = (d + 2 * padding.0 - dilation.0 * (kd - 1) - 1) / stride.0 + 1;
        let out_h = (h + 2 * padding.1 - dilation.1 * (kh - 1) - 1) / stride.1 + 1;
        let out_w = (w + 2 * padding.2 - dilation.2 * (kw - 1) - 1) / stride.2 + 1;
        let out_spatial = out_d * out_h * out_w;
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
            let bias_exp = B::from_fn(n_batch * c_out, out_spatial, |row, _col| {
                B::get(&bi.storage, 0, row % c_out)
            });
            Self::from_storage(B::add(&out.storage, &bias_exp))
        } else {
            out
        }
    }

    // ── Random constructors ─────────────────────────────────────────

    /// Uniform random tensor in [0, 1). Uses xorshift64 seeded from `seed`.
    #[must_use]
    pub fn rand(nrows: usize, ncols: usize, seed: u64) -> Self {
        let s0 = if seed == 0 {
            0x1234_5678_9ABC_DEF0_u64
        } else {
            seed
        };
        let n = nrows * ncols;
        let mut data = Vec::with_capacity(n);
        let mut s = s0;
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            data.push(T::from_f64((s as f64) / (u64::MAX as f64)));
        }
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }

    /// Normal-distributed random tensor (mean=0, std=1) via Box-Muller. Uses xorshift64 seeded from `seed`.
    #[must_use]
    pub fn randn(nrows: usize, ncols: usize, seed: u64) -> Self {
        let mut s = if seed == 0 {
            0x1234_5678_9ABC_DEF0_u64
        } else {
            seed
        };
        let mut xorshift = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64) / (u64::MAX as f64)
        };
        let n = nrows * ncols;
        let mut data = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let u1 = xorshift().max(1e-300);
            let u2 = xorshift();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data.push(T::from_f64(r * theta.cos()));
            if i + 1 < n {
                data.push(T::from_f64(r * theta.sin()));
            }
            i += 2;
        }
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }

    // ── Dropout ─────────────────────────────────────────────────────

    /// Dropout: randomly zeroes elements with probability `p` during training.
    /// When `training` is false, returns a clone. `seed` controls the random mask.
    #[must_use]
    pub fn dropout(&self, p: f64, training: bool, seed: u64) -> Self {
        if !training || p <= 0.0 {
            return self.clone();
        }
        if p >= 1.0 {
            let (m, n) = self.shape();
            return Self::zeros(m, n);
        }
        let scale = T::from_f64(1.0 / (1.0 - p));
        let threshold = (p * (u64::MAX as f64)) as u64;
        let (m, n) = self.shape();
        let mut s = if seed == 0 {
            0xDEAD_BEEF_CAFE_1234_u64
        } else {
            seed
        };
        let mut data = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let x = self.get(r, c);
                data.push(if s < threshold { T::zero() } else { x * scale });
            }
        }
        Self::from_storage(B::from_vec(m, n, data))
    }

    // ── Interpolate ─────────────────────────────────────────────────

    /// Nearest-neighbor interpolation (upsample/downsample).
    ///
    /// Input: (N*C, H*W). Output: (N*C, out_H * out_W).
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
    ///
    /// Input: (N*C, H*W). Output: (N*C, out_H * out_W).
    #[must_use]
    pub fn interpolate_bilinear(&self, h: usize, w: usize, out_h: usize, out_w: usize) -> Self {
        let nc = self.nrows();
        Self::from_fn(nc, out_h * out_w, |row, col| {
            let oh = col / out_w;
            let ow = col % out_w;
            // Map output coords to input coords (align_corners=false)
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
            // bilinear blend
            let top = v00 * (one - fw) + v01 * fw;
            let bot = v10 * (one - fw) + v11 * fw;
            top * (one - fh) + bot * fh
        })
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
                    m,
                    n,
                    p,
                    q,
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
        assert!(axis <= nd, "nabla: unsqueeze axis {axis} > ndim {nd}");
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

    /// Short alias for conjugate-transpose (`adjoint`).
    #[must_use]
    #[inline]
    pub fn h(&self) -> StaticMatrix<T, C, R> {
        self.adjoint()
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

impl<T: Scalar, const R: usize, const C: usize> Index<(usize, usize)> for StaticMatrix<T, R, C> {
    type Output = T;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row][col]
    }
}

impl<T: Scalar, const R: usize, const C: usize> IndexMut<(usize, usize)> for StaticMatrix<T, R, C> {
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
