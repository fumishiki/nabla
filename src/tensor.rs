// tensor.rs — Tensor<T, B> struct, constructors, accessors, ops, and Display.
//
// Design notes:
// - Operator overloads are defined on references to avoid moves (Julia semantics).
// - Shape mismatches in Add/Sub/Mul panic with a descriptive message (Option A).
// - Adjoint uses `T::IS_REAL` const to choose between transpose and conj+transpose.
// - `adjoint` delegates element-wise conjugation via `faer_traits::conj`.

use core::fmt;
use core::ops::{Add, Bound, Mul, Neg, RangeBounds, Sub};

use crate::backend::{Backend, Cpu, DefaultBackend};
use crate::scalar::Scalar;

/// A 2-D dense matrix backed by a pluggable [`Backend`].
///
/// The default backend is [`crate::backend::Cpu`], which uses faer's SIMD kernels.
pub struct Tensor<T: Scalar, B: Backend = DefaultBackend> {
    storage: B::Storage<T>,
}

impl<T: Scalar, B: Backend> Clone for Tensor<T, B> {
    fn clone(&self) -> Self {
        Self {
            storage: B::clone_storage(&self.storage),
        }
    }
}

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

impl<T: Scalar, B: Backend> Tensor<T, B> {
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
    pub(crate) fn from_storage(storage: B::Storage<T>) -> Self {
        Self { storage }
    }

    /// Borrow the underlying storage.
    #[inline]
    pub(crate) fn storage_ref(&self) -> &B::Storage<T> {
        &self.storage
    }

    /// Mutably borrow the underlying storage.
    #[inline]
    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage<T> {
        &mut self.storage
    }

    /// Allocate a zero-filled matrix of shape `(nrows, ncols)`.
    #[must_use]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            storage: B::zeros(nrows, ncols),
        }
    }

    /// Allocate a matrix whose `(i, j)` element is `f(i, j)`.
    #[must_use]
    pub fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> Self {
        Self {
            storage: B::from_fn(nrows, ncols, f),
        }
    }

    /// Allocate an `n × n` identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |r, c| {
            if r == c {
                T::one_impl()
            } else {
                T::zero_impl()
            }
        })
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

    /// Copy a submatrix into a fresh tensor.
    ///
    /// `row_end` and `col_end` are exclusive upper bounds.
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
        Self {
            storage: B::from_fn(nrows, ncols, |r, c| self.get(row_start + r, col_start + c)),
        }
    }

    /// Set element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        Self::check_range("set row", row, self.nrows());
        Self::check_range("set col", col, self.ncols());
        B::set(&mut self.storage, row, col, val);
    }

    /// Slice by row and column ranges, returning a new `Tensor` (copy).
    ///
    /// `A[row_range, col_range]` — copy of the selected submatrix.
    #[must_use]
    pub fn slice(&self, rows: impl RangeBounds<usize>, cols: impl RangeBounds<usize>) -> Self {
        let (rs, re) = resolve_range(rows, self.nrows());
        let (cs, ce) = resolve_range(cols, self.ncols());
        self.submatrix(rs, re, cs, ce)
    }

    /// Slice rows, all columns.
    ///
    /// `A[rows, :]` — slice rows, all columns.
    #[must_use]
    pub fn slice_rows(&self, rows: impl RangeBounds<usize>) -> Self {
        self.slice(rows, ..)
    }

    /// Slice columns, all rows.
    ///
    /// `A[:, cols]` — slice columns, all rows.
    #[must_use]
    pub fn slice_cols(&self, cols: impl RangeBounds<usize>) -> Self {
        self.slice(.., cols)
    }

    /// Extract a row vector tensor with shape `1 × ncols`.
    #[must_use]
    pub fn row(&self, row: usize) -> Self {
        Self::check_range("row", row, self.nrows());
        Self {
            storage: B::from_fn(1, self.ncols(), |_, c| self.get(row, c)),
        }
    }

    /// Extract a column vector tensor with shape `nrows × 1`.
    #[must_use]
    pub fn col(&self, col: usize) -> Self {
        Self::check_range("col", col, self.ncols());
        Self {
            storage: B::from_fn(self.nrows(), 1, |r, _| self.get(r, col)),
        }
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

    /// Element-wise `e^x`.
    #[must_use]
    #[inline]
    pub fn exp(&self) -> Self {
        Self::from_storage(B::exp(&self.storage))
    }

    /// Element-wise natural logarithm `ln(x)`.
    #[must_use]
    #[inline]
    pub fn ln(&self) -> Self {
        Self::from_storage(B::ln(&self.storage))
    }

    /// Element-wise `ln(1 + x)`.
    #[must_use]
    #[inline]
    pub fn log1p(&self) -> Self {
        Self::from_storage(B::log1p(&self.storage))
    }

    /// Element-wise `sin(x)`.
    #[must_use]
    #[inline]
    pub fn sin(&self) -> Self {
        Self::from_storage(B::sin(&self.storage))
    }

    /// Element-wise `cos(x)`.
    #[must_use]
    #[inline]
    pub fn cos(&self) -> Self {
        Self::from_storage(B::cos(&self.storage))
    }

    /// Element-wise `tanh(x)`.
    #[must_use]
    #[inline]
    pub fn tanh(&self) -> Self {
        Self::from_storage(B::tanh(&self.storage))
    }

    /// Element-wise `sqrt(x)`.
    #[must_use]
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self::from_storage(B::sqrt(&self.storage))
    }

    /// Element-wise absolute value.
    ///
    /// For complex types, returns the magnitude as the real part with zero imaginary part.
    #[must_use]
    #[inline]
    pub fn abs(&self) -> Self {
        Self::from_storage(B::abs(&self.storage))
    }

    /// Element-wise reciprocal `1/x`.
    #[must_use]
    #[inline]
    pub fn recip(&self) -> Self {
        Self::from_storage(B::recip(&self.storage))
    }

    /// Element-wise error function.
    ///
    /// Uses the Abramowitz & Stegun polynomial approximation (max error ~1.5e-7).
    #[must_use]
    #[inline]
    pub fn erf(&self) -> Self {
        Self::from_storage(B::erf(&self.storage))
    }

    /// Element-wise `ceil(x)`.
    #[must_use]
    #[inline]
    pub fn ceil(&self) -> Self {
        Self::from_storage(B::ceil(&self.storage))
    }

    /// Element-wise `floor(x)`.
    #[must_use]
    #[inline]
    pub fn floor(&self) -> Self {
        Self::from_storage(B::floor(&self.storage))
    }

    /// Element-wise `round(x)`.
    #[must_use]
    #[inline]
    pub fn round(&self) -> Self {
        Self::from_storage(B::round(&self.storage))
    }

    /// Element-wise `x^p` for scalar exponent `p`.
    #[must_use]
    #[inline]
    pub fn powf(&self, p: T) -> Self {
        Self::from_storage(B::powf(&self.storage, p))
    }

    /// Element-wise multiplication `self[i,j] * other[i,j]`.
    ///
    /// # Panics
    ///
    /// Panics if `self.shape() != other.shape()`.
    #[must_use]
    pub fn mul_elem(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "mul_elem shape mismatch");
        Self::from_storage(B::mul_elem(&self.storage, &other.storage))
    }

    /// Element-wise division `self[i,j] / other[i,j]`.
    ///
    /// # Panics
    ///
    /// Panics if `self.shape() != other.shape()`.
    #[must_use]
    pub fn div_elem(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "div_elem shape mismatch");
        Self::from_storage(B::div_elem(&self.storage, &other.storage))
    }

    /// Sum of all elements.
    ///
    /// Returns the additive identity for an empty matrix.
    #[must_use]
    #[inline]
    pub fn sum_all(&self) -> T {
        B::sum_all(&self.storage)
    }

    /// Element with the maximum value (or maximum magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    #[must_use]
    #[inline]
    pub fn max_all(&self) -> T {
        B::max_all(&self.storage)
    }

    /// Element with the minimum value (or minimum magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    #[must_use]
    #[inline]
    pub fn min_all(&self) -> T {
        B::min_all(&self.storage)
    }

    /// `(row, col)` of the element with the maximum value (or magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    #[must_use]
    #[inline]
    pub fn argmax(&self) -> (usize, usize) {
        B::argmax_all(&self.storage)
    }

    /// `(row, col)` of the element with the minimum value (or magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    #[must_use]
    #[inline]
    pub fn argmin(&self) -> (usize, usize) {
        B::argmin_all(&self.storage)
    }

    /// Return the transpose: a new `Tensor` of shape `(ncols, nrows)`.
    #[must_use]
    pub fn t(&self) -> Self {
        Self {
            storage: B::transpose(&self.storage),
        }
    }

    /// Return the conjugate transpose (adjoint / Hermitian transpose).
    ///
    /// For real types (`f32`, `f64`) this is identical to [`Tensor::t`].
    /// For complex types (`c32`, `c64`) each element is conjugated before
    /// transposing.  Both cases are computed in a single `from_fn` pass —
    /// no intermediate allocation.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        let (r, c) = self.shape();
        Self::from_storage(if T::IS_REAL {
            B::from_fn(c, r, |i, j| self.get(j, i))
        } else {
            B::from_fn(c, r, |i, j| faer_traits::math_utils::conj(&self.get(j, i)))
        })
    }

    /// Compute `out = a * b` (matrix multiply), overwriting `out`.
    ///
    /// # Panics
    ///
    /// Panics if inner dimensions do not match, or if `out` does not have
    /// shape `(a.nrows(), b.ncols())`.
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

    /// Copy all elements into a tensor on a (possibly different) backend.
    ///
    /// This is the general backend-conversion primitive.  Internally it calls
    /// `get(r, c)` on every element and constructs the target tensor via
    /// `B2::from_fn`, so it works across any pair of backends.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nabla::prelude::*;
    /// let cpu_t: Tensor<f32, Cpu> = Tensor::from_fn(2, 2, |r, c| (r + c) as f32);
    /// // round-trip through CPU (always available)
    /// let cpu2 = cpu_t.to_backend::<Cpu>();
    /// assert_eq!(cpu2.get(1, 1), 2.0_f32);
    /// ```
    #[must_use]
    pub fn to_backend<B2: Backend>(&self) -> Tensor<T, B2> {
        let (rows, cols) = self.shape();
        Tensor {
            storage: B2::from_fn(rows, cols, |r, c| self.get(r, c)),
        }
    }

    /// Copy into a CPU-backed tensor.
    ///
    /// Equivalent to `self.to_backend::<Cpu>()`.  Always available regardless
    /// of which backend features are enabled.
    #[must_use]
    #[inline]
    pub fn to_cpu(&self) -> Tensor<T, crate::backend::Cpu> {
        self.to_backend::<crate::backend::Cpu>()
    }
}

#[cfg(feature = "cuda")]
impl<T: Scalar, B: crate::backend::Backend> Tensor<T, B> {
    /// Copy into a CUDA-backed tensor (requires `cuda` feature).
    #[must_use]
    #[inline]
    pub fn to_cuda(&self) -> Tensor<T, crate::backend::Cuda> {
        self.to_backend::<crate::backend::Cuda>()
    }
}

#[cfg(feature = "wgpu")]
impl<T: Scalar, B: crate::backend::Backend> Tensor<T, B> {
    /// Copy into a wgpu-backed tensor (requires `wgpu` feature).
    #[must_use]
    #[inline]
    pub fn to_wgpu(&self) -> Tensor<T, crate::backend::Wgpu> {
        self.to_backend::<crate::backend::Wgpu>()
    }
}

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $backend_fn:ident, $op:literal) => {
        impl<T: Scalar, B: Backend> $trait for &Tensor<T, B> {
            type Output = Tensor<T, B>;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    concat!($op, " shape mismatch: lhs {:?} vs rhs {:?}"),
                    self.shape(),
                    rhs.shape()
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
            "matmul inner dimension mismatch: lhs ({m}, {k_a}), rhs ({k_b}, {n})"
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

// Static matrices and abstract type hierarchy.
//
// Combines:
//   - StaticMatrix<T, R, C>  — stack-allocated const-generic matrix
//   - Array<T> / Matrix<T>   — abstract trait hierarchy

/// A stack-allocated `R × C` matrix with element type `T`.
///
/// Arithmetic (`+`, `-`, `*`) is stack-only and allocation-free.
#[derive(Clone, Copy)]
pub struct StaticMatrix<T: Scalar, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T: Scalar, const R: usize, const C: usize> StaticMatrix<T, R, C> {
    /// Zero-filled `R × C` matrix.
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

    /// Transpose: returns a new `C × R` static matrix (stack-only).
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
        StaticMatrix::<T, C, R>::from_fn(|r, c| faer_traits::math_utils::conj(&self.data[c][r]))
    }

    /// Matrix multiply `self (R×K) * rhs (K×N)` → `StaticMatrix<T, R, N>`.
    #[must_use]
    pub fn matmul<const N: usize>(&self, rhs: &StaticMatrix<T, C, N>) -> StaticMatrix<T, R, N> {
        StaticMatrix::<T, R, N>::from_fn(|r, c| {
            (0..C).fold(T::zero_impl(), |acc, k| {
                acc + self.data[r][k] * rhs.data[k][c]
            })
        })
    }

    /// Convert to a heap-allocated [`Tensor`] (copies all elements).
    #[must_use]
    pub fn to_tensor(&self) -> Tensor<T, DefaultBackend> {
        Tensor::from_fn(R, C, |r, c| self.data[r][c])
    }

    /// Build from a [`Tensor`] (copies all elements).
    ///
    /// # Panics
    /// Panics if `tensor.shape() != (R, C)`.
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

impl<T: Scalar, const R: usize, const K: usize, const N: usize> Mul<StaticMatrix<T, K, N>>
    for StaticMatrix<T, R, K>
{
    type Output = StaticMatrix<T, R, N>;
    fn mul(self, rhs: StaticMatrix<T, K, N>) -> Self::Output {
        self.matmul(&rhs)
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
///
/// Implemented by both dynamic [`Tensor`] and static [`StaticMatrix`].
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

/// Matrix algebra trait extending [`Array`].
///
/// `t_dyn` and `matmul_dyn` always return a CPU-backed [`Tensor`] so the
/// return type is independent of which backend `self` uses.
pub trait Matrix<T: Scalar>: Array<T> {
    /// Transpose into a CPU-backed [`Tensor`].
    fn t_dyn(&self) -> Tensor<T, Cpu>;
    /// Matrix multiply `self × rhs` into a CPU-backed [`Tensor`].
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu>;
}

/// Shared matmul logic for any pair of `Array<T>` implementors.
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

impl<T: Scalar, B: Backend> Matrix<T> for Tensor<T, B> {
    fn t_dyn(&self) -> Tensor<T, Cpu> {
        self.to_cpu().t()
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

impl<T: Scalar, const R: usize, const C: usize> Matrix<T> for StaticMatrix<T, R, C> {
    fn t_dyn(&self) -> Tensor<T, Cpu> {
        Tensor::from_fn(C, R, |r, c| self.get(c, r))
    }
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
        dyn_matmul(self, rhs)
    }
}
