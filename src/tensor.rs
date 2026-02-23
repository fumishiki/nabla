// tensor.rs — Tensor<T, B> struct, constructors, accessors, ops, and Display.
//
// Design notes:
// - Operator overloads are defined on references to avoid moves (Julia semantics).
// - Shape mismatches in Add/Sub/Mul panic with a descriptive message (Option A).
// - Adjoint uses `T::IS_REAL` const to choose between transpose and conj+transpose.
// - `adjoint` delegates element-wise conjugation via `faer_traits::conj`.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use crate::backend::{Backend, DefaultBackend};
use crate::scalar::Scalar;

/// A 2-D dense matrix backed by a pluggable [`Backend`].
///
/// The default backend is [`crate::backend::Cpu`], which uses faer's SIMD kernels.
pub struct Tensor<T: Scalar, B: Backend = DefaultBackend> {
    storage: B::Storage<T>,
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
    /// transposing.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        if T::IS_REAL {
            return self.t();
        }
        let (r, c) = self.shape();
        let conj = B::from_fn(r, c, |i, j| {
            faer_traits::math_utils::conj(&B::get(&self.storage, i, j))
        });
        Self { storage: conj }.t()
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
}

impl<T: Scalar, B: Backend> Add for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "add shape mismatch: lhs {:?} vs rhs {:?}",
            self.shape(),
            rhs.shape()
        );
        Tensor {
            storage: B::add(&self.storage, &rhs.storage),
        }
    }
}

impl<T: Scalar, B: Backend> Sub for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "sub shape mismatch: lhs {:?} vs rhs {:?}",
            self.shape(),
            rhs.shape()
        );
        Tensor {
            storage: B::sub(&self.storage, &rhs.storage),
        }
    }
}

impl<T: Scalar, B: Backend> Neg for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn neg(self) -> Self::Output {
        Tensor {
            storage: B::neg(&self.storage),
        }
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
        Tensor {
            storage: B::scale(&self.storage, rhs),
        }
    }
}

impl<T: Scalar + fmt::Display, B: Backend> fmt::Display for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        write!(f, "[")?;
        for r in 0..rows {
            if r > 0 {
                write!(f, " ")?;
            }
            write!(f, "[")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", B::get(&self.storage, r, c))?;
            }
            write!(f, "]")?;
            if r + 1 < rows {
                writeln!(f)?;
            }
        }
        write!(f, "]")
    }
}

impl<T: Scalar + fmt::Debug, B: Backend> fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        write!(f, "Tensor({rows}x{cols})[")?;
        for r in 0..rows {
            write!(f, "[")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", B::get(&self.storage, r, c))?;
            }
            write!(f, "]")?;
            if r + 1 < rows {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}
