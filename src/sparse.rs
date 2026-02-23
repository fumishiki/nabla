// sparse.rs — Sparse matrix support (Wave 18 stub).
//
// This module is scheduled for full rewrite in Wave 18. Until then it exposes
// a minimal compilable surface so that linalg.rs and other modules can build.

use core::fmt;

use crate::backend::Cpu;
use crate::error::{Error, Result};
use crate::linalg::Side;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

#[inline]
fn sparse_error<T: fmt::Display>(op: &'static str, shape: (usize, usize), err: T) -> Error {
    Error::invalid(format!("{op} failed for sparse matrix {shape:?}: {err}"))
}

#[inline]
fn check_rhs_rows<T: Scalar>(expected_rows: usize, rhs: &Tensor<T, Cpu>) -> Result<()> {
    if expected_rows == rhs.nrows() {
        Ok(())
    } else {
        Err(Error::mismatch((expected_rows, rhs.ncols()), rhs.shape()))
    }
}

/// COO triplet entry for constructing sparse matrices.
#[derive(Clone, Copy, Debug)]
pub struct Triplet<T: Scalar> {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
    /// Value.
    pub val: T,
}

impl<T: Scalar> Triplet<T> {
    /// Construct a new triplet.
    #[must_use]
    #[inline]
    pub fn new(row: usize, col: usize, val: T) -> Self {
        Self { row, col, val }
    }
}

/// CSC sparse matrix stored as sorted column arrays.
///
/// Wave 18 will replace this with a self-contained CSC implementation. For now
/// the struct holds COO data converted to CSC via simple sort.
#[derive(Clone)]
pub struct SparseMatrix<T: Scalar> {
    nrows: usize,
    ncols: usize,
    /// Column pointers (length ncols+1).
    col_ptr: Vec<usize>,
    /// Row indices (length nnz).
    row_idx: Vec<usize>,
    /// Values (length nnz).
    values: Vec<T>,
}

impl<T: Scalar> SparseMatrix<T> {
    /// Build from COO triplets.
    ///
    /// # Errors
    /// Returns `Err` when indices are out of bounds.
    pub fn try_new_from_triplets(
        nrows: usize,
        ncols: usize,
        entries: &[Triplet<T>],
    ) -> Result<Self> {
        Self::build_csc(nrows, ncols, entries)
    }

    /// Build from COO triplets (nonnegative indices, same as `try_new_from_triplets`).
    ///
    /// # Errors
    /// Returns `Err` when indices are out of bounds.
    pub fn try_new_from_nonnegative_triplets(
        nrows: usize,
        ncols: usize,
        entries: &[Triplet<T>],
    ) -> Result<Self> {
        Self::build_csc(nrows, ncols, entries)
    }

    fn build_csc(nrows: usize, ncols: usize, entries: &[Triplet<T>]) -> Result<Self> {
        // Note: only accessible through concrete type impls
        for e in entries {
            if e.row >= nrows {
                return Err(sparse_error(
                    "build_csc",
                    (nrows, ncols),
                    format!("row index {} out of bounds", e.row),
                ));
            }
            if e.col >= ncols {
                return Err(sparse_error(
                    "build_csc",
                    (nrows, ncols),
                    format!("col index {} out of bounds", e.col),
                ));
            }
        }

        // Count entries per column
        let mut col_counts = vec![0usize; ncols];
        for e in entries {
            col_counts[e.col] += 1;
        }

        // Build col_ptr (prefix sum)
        let mut col_ptr = vec![0usize; ncols + 1];
        for j in 0..ncols {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }

        // Fill row_idx and values (stable sort within each column by row)
        let nnz = entries.len();
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut pos = col_ptr.clone();

        for e in entries {
            let p = pos[e.col];
            row_idx[p] = e.row;
            values[p] = e.val;
            pos[e.col] += 1;
        }

        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Number of rows.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Matrix shape as `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored (non-zero) entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Multiply `self × dense_rhs` into a dense tensor.
    ///
    /// # Errors
    /// Returns `Err` when shapes are incompatible.
    pub fn matmul_dense(&self, rhs: &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>> {
        if self.ncols != rhs.nrows() {
            return Err(Error::mismatch((self.nrows, rhs.ncols()), rhs.shape()));
        }
        let m = self.nrows;
        let n = rhs.ncols();
        let mut out = Tensor::zeros(m, n);
        for j in 0..self.ncols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                let a_ij = self.values[p];
                for k in 0..n {
                    let old = out.get(i, k);
                    out.set(i, k, old + a_ij * rhs.get(j, k));
                }
            }
        }
        Ok(out)
    }

    fn to_dense(&self) -> Tensor<T, Cpu> {
        let mut out = Tensor::zeros(self.nrows, self.ncols);
        for j in 0..self.ncols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                out.set(i, j, self.values[p]);
            }
        }
        out
    }
}

impl SparseMatrix<f64> {
    /// Solve `A·x = b` via sparse Cholesky (positive-definite symmetric).
    ///
    /// # Errors
    /// Returns `Err` when factorization or solve fails.
    pub fn cholesky_solve(&self, side: Side, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        check_rhs_rows(self.nrows, rhs)?;
        let dense = self.to_dense();
        let llt = dense.llt(side)?;
        Ok(llt.solve(rhs))
    }

    /// Solve `A·x = b` via sparse LU.
    ///
    /// # Errors
    /// Returns `Err` when factorization or solve fails.
    pub fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        check_rhs_rows(self.nrows, rhs)?;
        let dense = self.to_dense();
        dense.solve(rhs)
    }

    /// Least-squares solve `A·x ≈ b` via sparse QR.
    ///
    /// # Errors
    /// Returns `Err` when shapes are incompatible or solve fails.
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        check_rhs_rows(self.nrows, rhs)?;
        let dense = self.to_dense();
        dense.solve_lstsq(rhs)
    }
}

/// Alias for `SparseMatrix` (backward compat).
pub type SparseColMat<T> = SparseMatrix<T>;
