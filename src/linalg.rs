use core::fmt;

use faer::{
    linalg::solvers::{self, DenseSolveCore},
    prelude::*,
    sparse::linalg::matmul::sparse_dense_matmul,
    Accum,
    Par,
    Side,
};

use crate::backend::Cpu;
use crate::error::{Error, Result};
use crate::scalar::Scalar;
use crate::tensor::Tensor;

#[inline]
fn shape_mismatch(expected: (usize, usize), got: (usize, usize)) -> Error {
    Error::mismatch(expected, got)
}

#[inline]
    fn factorization_failed<T: fmt::Debug>(
    op: &'static str,
    shape: (usize, usize),
    err: T,
) -> Error {
    Error::invalid(format!("{op} failed for matrix {shape:?}: {err:?}"))
}

#[inline]
fn check_shape(expected: (usize, usize), got: (usize, usize)) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(shape_mismatch(expected, got))
    }
}

#[inline]
fn require_square(shape: (usize, usize), op: &'static str) -> Result<()> {
    if shape.0 == shape.1 {
        Ok(())
    } else {
        Err(Error::invalid(format!("{op} requires square input: {shape:?}")))
    }
}

impl<T: Scalar> Tensor<T, Cpu> {
    /// Borrow the underlying matrix as a faer `MatRef`.
    #[inline]
    #[must_use]
    pub fn as_mat_ref(&self) -> faer::MatRef<'_, T> {
        self.storage_ref().as_ref()
    }

    /// Borrow the underlying matrix as a mutable faer `MatMut`.
    #[inline]
    #[must_use]
    pub fn as_mat_mut(&mut self) -> faer::MatMut<'_, T> {
        self.storage_mut().as_mut()
    }

    /// LU decomposition with partial pivoting (`PA = LU`).
    #[must_use]
    #[inline]
    pub fn partial_piv_lu(&self) -> solvers::PartialPivLu<T> {
        self.as_mat_ref().partial_piv_lu()
    }

    /// LU decomposition with full pivoting (`PAQ^T = LU`).
    #[must_use]
    #[inline]
    pub fn full_piv_lu(&self) -> solvers::FullPivLu<T> {
        self.as_mat_ref().full_piv_lu()
    }

    /// QR decomposition (`A = QR`).
    #[must_use]
    #[inline]
    pub fn qr(&self) -> solvers::Qr<T> {
        self.as_mat_ref().qr()
    }

    /// Column-pivoted QR decomposition (`A P^T = Q R`).
    #[must_use]
    #[inline]
    pub fn col_piv_qr(&self) -> solvers::ColPivQr<T> {
        self.as_mat_ref().col_piv_qr()
    }

    /// Cholesky factorization (`A = L L^T`).
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or factorization fails.
    pub fn llt(&self, side: Side) -> Result<solvers::Llt<T>> {
        require_square(self.shape(), "llt")?;
        self.as_mat_ref().llt(side)
            .map_err(|e| factorization_failed("llt", self.shape(), e))
    }

    /// LDL^T factorization.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or factorization fails.
    pub fn ldlt(&self, side: Side) -> Result<solvers::Ldlt<T>> {
        require_square(self.shape(), "ldlt")?;
        self.as_mat_ref()
            .ldlt(side)
            .map_err(|e| factorization_failed("ldlt", self.shape(), e))
    }

    /// Bunch-Kaufman `LBL^T` decomposition.
    #[must_use]
    #[inline]
    pub fn lblt(&self, side: Side) -> solvers::Lblt<T> {
        self.as_mat_ref().lblt(side)
    }

    /// Singular value decomposition (`A = U S V^H`).
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    pub fn svd(&self) -> Result<solvers::Svd<T>> {
        self.as_mat_ref().svd()
            .map_err(|e| factorization_failed("svd", self.shape(), e))
    }

    /// Thin SVD (`A = U S V^H` with compact `U` and `V`).
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    pub fn thin_svd(&self) -> Result<solvers::Svd<T>> {
        self.as_mat_ref().thin_svd()
            .map_err(|e| factorization_failed("thin_svd", self.shape(), e))
    }

    /// Singular values of `A`.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    pub fn singular_values(&self) -> Result<Vec<T::Real>> {
        self.as_mat_ref().singular_values()
            .map_err(|e| factorization_failed("singular_values", self.shape(), e))
    }

    /// Self-adjoint eigen decomposition.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or decomposition fails.
    pub fn self_adjoint_eigen(&self, side: Side) -> Result<solvers::SelfAdjointEigen<T>> {
        require_square(self.shape(), "self_adjoint_eigen")?;
        self.as_mat_ref()
            .self_adjoint_eigen(side)
            .map_err(|e| factorization_failed("self_adjoint_eigen", self.shape(), e))
    }

    /// Eigenvalues of a self-adjoint matrix.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or eigensolver fails.
    pub fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<T::Real>> {
        require_square(self.shape(), "self_adjoint_eigenvalues")?;
        self.as_mat_ref().self_adjoint_eigenvalues(side).map_err(|e| {
            factorization_failed("self_adjoint_eigenvalues", self.shape(), e)
        })
    }

    /// Solve `A x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.ncols(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        Ok(Tensor::from_storage(lu.solve(rhs.as_mat_ref())))
    }

    /// Solve `A^T x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_transpose(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        Ok(Tensor::from_storage(lu.solve_transpose(rhs.as_mat_ref())))
    }

    /// Solve `A^H x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_adjoint(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        Ok(Tensor::from_storage(lu.solve_adjoint(rhs.as_mat_ref())))
    }

    /// Solve in place: `A x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((self.ncols(), rhs.ncols()), rhs.shape())?;
        self.partial_piv_lu().solve_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve in place: `A^T x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_transpose_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        self.partial_piv_lu().solve_transpose_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve in place: `A^H x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_adjoint_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        self.partial_piv_lu().solve_adjoint_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve `x A = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn rsolve(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        Ok(Tensor::from_storage(lu.rsolve(rhs.as_mat_ref())))
    }

    /// Solve `x A^T = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn rsolve_transpose(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.nrows()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        Ok(Tensor::from_storage(lu.rsolve_transpose(rhs.as_mat_ref())))
    }

    /// Solve `x A^H = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn rsolve_adjoint(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.nrows()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        Ok(Tensor::from_storage(lu.rsolve_adjoint(rhs.as_mat_ref())))
    }

    /// Solve in place: `x A = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn rsolve_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((rhs.nrows(), self.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu();
        lu.rsolve_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Least-squares solve by QR: `A x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or least-squares solve fails.
    pub fn solve_lstsq(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let qr = self.qr();
        Ok(Tensor::from_storage(qr.solve_lstsq(rhs.as_mat_ref())))
    }

    /// In-place least-squares solve by QR: `A x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or least-squares solve fails.
    pub fn solve_lstsq_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let qr = self.qr();
        qr.solve_lstsq_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve `L x = b` for a lower-triangular matrix `L` in place.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_lower_triangular_in_place(&self, rhs: &mut Self) -> Result<()> {
        require_square(self.shape(), "solve_lower_triangular_in_place")?;
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        self.as_mat_ref().solve_lower_triangular_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve `U x = b` for an upper-triangular matrix `U` in place.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_upper_triangular_in_place(&self, rhs: &mut Self) -> Result<()> {
        require_square(self.shape(), "solve_upper_triangular_in_place")?;
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        self.as_mat_ref().solve_upper_triangular_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve `L x = b` for a unit lower-triangular matrix `L` in place.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_unit_lower_triangular_in_place(
        &self,
        rhs: &mut Self,
    ) -> Result<()> {
        require_square(self.shape(), "solve_unit_lower_triangular_in_place")?;
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        self.as_mat_ref()
            .solve_unit_lower_triangular_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Solve `U x = b` for a unit upper-triangular matrix `U` in place.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    pub fn solve_unit_upper_triangular_in_place(
        &self,
        rhs: &mut Self,
    ) -> Result<()> {
        require_square(self.shape(), "solve_unit_upper_triangular_in_place")?;
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        self.as_mat_ref()
            .solve_unit_upper_triangular_in_place(rhs.as_mat_mut());
        Ok(())
    }

    /// Reconstruct from partial-pivot LU factors.
    #[must_use]
    pub fn partial_piv_lu_reconstruct(&self) -> Self {
        Tensor::from_storage(self.partial_piv_lu().reconstruct())
    }

    /// Inverse from partial-pivot LU factors.
    #[must_use]
    pub fn partial_piv_lu_inverse(&self) -> Self {
        Tensor::from_storage(self.partial_piv_lu().inverse())
    }

    /// Reconstruct from Cholesky factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization is not available.
    pub fn llt_reconstruct(&self, side: Side) -> Result<Self> {
        Ok(Tensor::from_storage(self.llt(side)?.reconstruct()))
    }

    /// Inverse from Cholesky factors.
    ///
    /// # Errors
    /// Returns `Err` when inversion fails.
    pub fn llt_inverse(&self, side: Side) -> Result<Self> {
        Ok(Tensor::from_storage(self.llt(side)?.inverse()))
    }

    /// Reconstruct from LDLT factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization is not available.
    pub fn ldlt_reconstruct(&self, side: Side) -> Result<Self> {
        Ok(Tensor::from_storage(self.ldlt(side)?.reconstruct()))
    }

    /// Inverse from LDLT factors.
    ///
    /// # Errors
    /// Returns `Err` when inversion fails.
    pub fn ldlt_inverse(&self, side: Side) -> Result<Self> {
        Ok(Tensor::from_storage(self.ldlt(side)?.inverse()))
    }

    /// Reconstruct from Bunch-Kaufman factors.
    #[must_use]
    pub fn lblt_reconstruct(&self, side: Side) -> Self {
        Tensor::from_storage(self.lblt(side).reconstruct())
    }

    /// Multiply sparse column matrix and dense matrix into dense result.
    ///
    /// This is a convenience wrapper around
    /// `faer::sparse::linalg::matmul::sparse_dense_matmul`.
    pub fn sparse_dense_matmul(
        left: &faer::sparse::SparseColMat<usize, T>,
        right: &Self,
        alpha: T,
    ) -> Self {
        let ncols = right.ncols();
        let nrows = left.nrows();
        let mut out = Tensor::zeros(nrows, ncols);
        sparse_dense_matmul(
            out.as_mat_mut(),
            Accum::Replace,
            left.as_ref(),
            right.as_mat_ref(),
            alpha,
            Par::Seq,
        );
        out
    }
}
