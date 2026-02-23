use core::fmt;
use std::convert::TryFrom;

use faer::{
    Side, prelude::*, sparse as faer_sparse,
    sparse::linalg::matmul::sparse_dense_matmul as faer_sparse_dense_matmul,
};

use crate::error::{Error, Result};
use crate::scalar::Scalar;
use crate::tensor::Tensor;

type TripletEntriesNonNegative<T> = faer_sparse::Triplet<isize, isize, T>;
type TripletEntries<T> = faer_sparse::Triplet<usize, usize, T>;
type SparseStorage<T> = faer_sparse::SparseColMat<usize, T>;
type SparseStorageRef<'a, T> = faer_sparse::SparseColMatRef<'a, usize, T>;
type SymbolicLlt = faer_sparse::linalg::solvers::SymbolicLlt<usize>;
type SymbolicLu = faer_sparse::linalg::solvers::SymbolicLu<usize>;
type SymbolicQr = faer_sparse::linalg::solvers::SymbolicQr<usize>;
type NumericLlt<T> = faer_sparse::linalg::solvers::Llt<usize, T>;
type NumericLu<T> = faer_sparse::linalg::solvers::Lu<usize, T>;
type NumericQr<T> = faer_sparse::linalg::solvers::Qr<usize, T>;

#[inline]
fn sparse_error<T: fmt::Display>(op: &'static str, shape: (usize, usize), err: T) -> Error {
    Error::invalid(format!("{op} failed for sparse matrix {shape:?}: {err}"))
}

#[inline]
fn check_rhs_rows<T: Scalar>(expected_rows: usize, rhs: &Tensor<T>) -> Result<()> {
    if expected_rows == rhs.nrows() {
        Ok(())
    } else {
        Err(Error::mismatch((expected_rows, rhs.ncols()), rhs.shape()))
    }
}

/// Owned CSC sparse matrix with `usize` indices.
#[derive(Clone)]
pub struct SparseMatrix<T: Scalar> {
    storage: SparseStorage<T>,
}

impl<T: Scalar> SparseMatrix<T> {
    /// Build from COO triplets.
    ///
    /// # Errors
    /// Returns `Err` when COO data is invalid or matrix shape is unsupported.
    pub fn try_new_from_triplets(
        nrows: usize,
        ncols: usize,
        entries: &[Triplet<T>],
    ) -> Result<Self> {
        let storage = SparseStorage::try_new_from_triplets(nrows, ncols, entries)
            .map_err(|err| sparse_error("try_new_from_triplets", (nrows, ncols), err))?;
        Ok(Self { storage })
    }

    /// Build from COO triplets allowing zero-based nonnegative structure assumptions.
    ///
    /// # Errors
    /// Returns `Err` when COO indices or matrix shape are invalid.
    pub fn try_new_from_nonnegative_triplets(
        nrows: usize,
        ncols: usize,
        entries: &[Triplet<T>],
    ) -> Result<Self> {
        let entries = entries
            .iter()
            .map(
                |entry| -> core::result::Result<TripletEntriesNonNegative<T>, Error> {
                    let row = isize::try_from(entry.row)
                        .map_err(|_| Error::invalid("triplet row index does not fit in isize"))?;
                    let col = isize::try_from(entry.col)
                        .map_err(|_| Error::invalid("triplet col index does not fit in isize"))?;
                    Ok(TripletEntriesNonNegative {
                        row,
                        col,
                        val: entry.val,
                    })
                },
            )
            .collect::<core::result::Result<Vec<_>, _>>()?;
        let storage = SparseStorage::try_new_from_nonnegative_triplets(nrows, ncols, &entries)
            .map_err(|err| {
                sparse_error("try_new_from_nonnegative_triplets", (nrows, ncols), err)
            })?;
        Ok(Self { storage })
    }

    /// Borrow as a faer sparse view.
    #[inline]
    #[must_use]
    pub fn as_ref(&self) -> SparseStorageRef<'_, T> {
        self.storage.as_ref()
    }

    /// Consume and return the underlying faer storage.
    #[inline]
    #[must_use]
    pub fn into_storage(self) -> SparseStorage<T> {
        self.storage
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.storage.nrows()
    }

    /// Number of columns.
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.storage.ncols()
    }

    /// Matrix shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Number of non-zero entries.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.storage.as_ref().parts().1.len()
    }

    /// Return sparse symbolic and value parts.
    #[inline]
    #[must_use]
    pub fn parts(&self) -> (faer_sparse::SymbolicSparseColMatRef<'_, usize>, &'_ [T]) {
        self.storage.as_ref().parts()
    }

    /// Sparse Cholesky symbolic factorization.
    ///
    /// # Errors
    /// Returns `Err` when symbolic factorization fails.
    pub fn symbolic_llt(&self, side: Side) -> Result<SymbolicLlt> {
        SymbolicLlt::try_new(self.as_ref().symbolic(), side)
            .map_err(|err| sparse_error("symbolic_llt", self.shape(), err))
    }

    /// Sparse LU symbolic factorization.
    ///
    /// # Errors
    /// Returns `Err` when symbolic factorization fails.
    pub fn symbolic_lu(&self) -> Result<SymbolicLu> {
        SymbolicLu::try_new(self.as_ref().symbolic())
            .map_err(|err| sparse_error("symbolic_lu", self.shape(), err))
    }

    /// Sparse QR symbolic factorization.
    ///
    /// # Errors
    /// Returns `Err` when symbolic factorization fails.
    pub fn symbolic_qr(&self) -> Result<SymbolicQr> {
        SymbolicQr::try_new(self.as_ref().symbolic())
            .map_err(|err| sparse_error("symbolic_qr", self.shape(), err))
    }

    /// Cholesky factorization with pre-computed symbolic data.
    ///
    /// # Errors
    /// Returns `Err` when numeric factorization fails.
    pub fn llt_with_symbolic(&self, symbolic: SymbolicLlt, side: Side) -> Result<NumericLlt<T>> {
        NumericLlt::try_new_with_symbolic(symbolic, self.as_ref(), side)
            .map_err(|err| sparse_error("llt_with_symbolic", self.shape(), err))
    }

    /// LU factorization with pre-computed symbolic data.
    ///
    /// # Errors
    /// Returns `Err` when numeric factorization fails.
    pub fn lu_with_symbolic(&self, symbolic: SymbolicLu) -> Result<NumericLu<T>> {
        NumericLu::try_new_with_symbolic(symbolic, self.as_ref())
            .map_err(|err| sparse_error("lu_with_symbolic", self.shape(), err))
    }

    /// QR factorization with pre-computed symbolic data.
    ///
    /// # Errors
    /// Returns `Err` when numeric factorization fails.
    pub fn qr_with_symbolic(&self, symbolic: SymbolicQr) -> Result<NumericQr<T>> {
        NumericQr::try_new_with_symbolic(symbolic, self.as_ref())
            .map_err(|err| sparse_error("qr_with_symbolic", self.shape(), err))
    }

    /// Compute sparse Cholesky factors.
    ///
    /// # Errors
    /// Returns `Err` when Cholesky factorization fails.
    pub fn cholesky(&self, side: Side) -> Result<NumericLlt<T>> {
        self.storage
            .sp_cholesky(side)
            .map_err(|err| sparse_error("cholesky", self.shape(), err))
    }

    /// Solve `A x = b` via sparse Cholesky.
    ///
    /// # Errors
    /// Returns `Err` when RHS shape is incompatible or solver fails.
    pub fn cholesky_solve(&self, side: Side, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        check_rhs_rows(self.nrows(), rhs)?;
        let llt = self.cholesky(side)?;
        Ok(Tensor::from_storage(llt.solve(rhs.as_mat_ref())))
    }

    /// Solve `A x = b` via sparse Cholesky with symbolic prepass.
    ///
    /// # Errors
    /// Returns `Err` when RHS shape is incompatible or solver fails.
    pub fn cholesky_solve_with_symbolic(
        &self,
        symbolic: SymbolicLlt,
        side: Side,
        rhs: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        check_rhs_rows(self.nrows(), rhs)?;
        let llt = self.llt_with_symbolic(symbolic, side)?;
        Ok(Tensor::from_storage(llt.solve(rhs.as_mat_ref())))
    }

    /// Compute sparse LU factors.
    ///
    /// # Errors
    /// Returns `Err` when LU factorization fails.
    pub fn lu(&self) -> Result<NumericLu<T>> {
        self.storage
            .sp_lu()
            .map_err(|err| sparse_error("lu", self.shape(), err))
    }

    /// Solve `A x = b` via sparse LU.
    ///
    /// # Errors
    /// Returns `Err` when RHS shape is incompatible or solver fails.
    pub fn lu_solve(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        check_rhs_rows(self.nrows(), rhs)?;
        let lu = self.lu()?;
        Ok(Tensor::from_storage(lu.solve(rhs.as_mat_ref())))
    }

    /// Solve `A x = b` via sparse LU with symbolic prepass.
    ///
    /// # Errors
    /// Returns `Err` when RHS shape is incompatible or solver fails.
    pub fn lu_solve_with_symbolic(
        &self,
        symbolic: SymbolicLu,
        rhs: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        check_rhs_rows(self.nrows(), rhs)?;
        let lu = self.lu_with_symbolic(symbolic)?;
        Ok(Tensor::from_storage(lu.solve(rhs.as_mat_ref())))
    }

    /// Compute sparse QR factors.
    ///
    /// # Errors
    /// Returns `Err` when QR factorization fails.
    pub fn qr(&self) -> Result<NumericQr<T>> {
        self.storage
            .sp_qr()
            .map_err(|err| sparse_error("qr", self.shape(), err))
    }

    /// Least-squares solve `A x ≈ b` via sparse QR.
    ///
    /// # Errors
    /// Returns `Err` when RHS shape is incompatible or solver fails.
    pub fn qr_solve_lstsq(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        check_rhs_rows(self.nrows(), rhs)?;
        let qr = self.qr()?;
        Ok(Tensor::from_storage(qr.solve_lstsq(rhs.as_mat_ref())))
    }

    /// Least-squares solve `A x ≈ b` via sparse QR with symbolic prepass.
    ///
    /// # Errors
    /// Returns `Err` when RHS shape is incompatible or solver fails.
    pub fn qr_solve_lstsq_with_symbolic(
        &self,
        symbolic: SymbolicQr,
        rhs: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        check_rhs_rows(self.nrows(), rhs)?;
        let qr = self.qr_with_symbolic(symbolic)?;
        Ok(Tensor::from_storage(qr.solve_lstsq(rhs.as_mat_ref())))
    }

    /// Multiply sparse × dense into dense output.
    pub fn sparse_dense_matmul(&self, right: &Tensor<T>, alpha: T) -> Tensor<T> {
        let mut out = Tensor::zeros(self.nrows(), right.ncols());
        faer_sparse_dense_matmul(
            out.as_mat_mut(),
            faer::Accum::Replace,
            self.storage.as_ref(),
            right.as_mat_ref(),
            alpha,
            faer::Par::Seq,
        );
        out
    }
}

/// Convenience alias for sparse triplet entries.
pub type Triplet<T> = TripletEntries<T>;

/// Sparse CSC matrix alias.
pub type SparseColMat<T> = SparseStorage<T>;

/// Sparse CSC matrix reference alias.
pub type SparseColMatRef<'a, T> = SparseStorageRef<'a, T>;

/// Common sparse-factorization aliases.
pub mod linalg {
    /// Sparse-matrix × dense-matrix multiplication helpers.
    pub mod matmul {
        pub use faer::sparse::linalg::matmul::sparse_dense_matmul;
    }

    /// Sparse decomposition and solver aliases.
    pub mod solvers {
        pub use faer::sparse::linalg::solvers::{Llt, Lu, Qr, SymbolicLlt, SymbolicLu, SymbolicQr};
    }
}
