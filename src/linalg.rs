use core::fmt;

use faer::{
    linalg::solvers::{self, DenseSolveCore},
    prelude::*,
    sparse::linalg::matmul::sparse_dense_matmul,
    Accum, Par, Side,
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
fn factorization_failed<T: fmt::Debug>(op: &'static str, shape: (usize, usize), err: T) -> Error {
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
        Err(Error::invalid(format!(
            "{op} requires square input: {shape:?}"
        )))
    }
}

/// Generate `solve`-family methods that delegate to partial-pivot LU.
///
/// Variants produced:
/// - returning `Result<Self>`: `solve`, `solve_transpose`, `solve_adjoint`,
///   `rsolve`, `rsolve_transpose`, `rsolve_adjoint`
/// - in-place `Result<()>`: `solve_in_place`, `solve_transpose_in_place`,
///   `solve_adjoint_in_place`, `rsolve_in_place`
macro_rules! lu_solve {
    // returning variant: fn $name(&self, rhs: &Self) -> Result<Self>
    (ret $name:ident, $doc:literal, $expected:expr, $lu_method:ident) => {
        #[doc = $doc]
        ///
        /// # Errors
        /// Returns `Err` when dimensions mismatch or solve fails.
        pub fn $name(&self, rhs: &Self) -> Result<Self> {
            check_shape($expected(self, rhs), rhs.shape())?;
            let lu = self.partial_piv_lu();
            Ok(Tensor::from_storage(lu.$lu_method(rhs.as_mat_ref())))
        }
    };
    // in-place variant: fn $name(&self, rhs: &mut Self) -> Result<()>
    (inplace $name:ident, $doc:literal, $expected:expr, $lu_method:ident) => {
        #[doc = $doc]
        ///
        /// # Errors
        /// Returns `Err` when dimensions mismatch or solve fails.
        pub fn $name(&self, rhs: &mut Self) -> Result<()> {
            check_shape($expected(self, rhs), rhs.shape())?;
            self.partial_piv_lu().$lu_method(rhs.as_mat_mut());
            Ok(())
        }
    };
}

/// Generate triangular in-place solve methods.
///
/// Pattern: `require_square` + `check_shape` + `self.as_mat_ref().METHOD(rhs.as_mat_mut())`.
macro_rules! tri_solve {
    ($name:ident, $doc:literal, $mat_method:ident) => {
        #[doc = $doc]
        ///
        /// # Errors
        /// Returns `Err` when dimensions mismatch or solve fails.
        pub fn $name(&self, rhs: &mut Self) -> Result<()> {
            require_square(self.shape(), stringify!($name))?;
            check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
            self.as_mat_ref().$mat_method(rhs.as_mat_mut());
            Ok(())
        }
    };
}

/// Generate reconstruct/inverse methods for fallible factorizations (those
/// that take a `Side` argument and return `Result`).
macro_rules! factored_op {
    // fallible factorization (takes Side, returns Result)
    (fallible $name:ident, $doc:literal, $factorize:ident, $op:ident) => {
        #[doc = $doc]
        ///
        /// # Errors
        /// Returns `Err` when factorization fails.
        pub fn $name(&self, side: Side) -> Result<Self> {
            Ok(Tensor::from_storage(self.$factorize(side)?.$op()))
        }
    };
    // infallible factorization (no Side, returns Self directly)
    (infallible $name:ident, $doc:literal, $factorize:ident, $op:ident) => {
        #[doc = $doc]
        #[must_use]
        pub fn $name(&self) -> Self {
            Tensor::from_storage(self.$factorize().$op())
        }
    };
    // infallible factorization with Side argument
    (infallible_side $name:ident, $doc:literal, $factorize:ident, $op:ident) => {
        #[doc = $doc]
        #[must_use]
        pub fn $name(&self, side: Side) -> Self {
            Tensor::from_storage(self.$factorize(side).$op())
        }
    };
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
        self.as_mat_ref()
            .llt(side)
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
        self.as_mat_ref()
            .svd()
            .map_err(|e| factorization_failed("svd", self.shape(), e))
    }

    /// Thin SVD (`A = U S V^H` with compact `U` and `V`).
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    pub fn thin_svd(&self) -> Result<solvers::Svd<T>> {
        self.as_mat_ref()
            .thin_svd()
            .map_err(|e| factorization_failed("thin_svd", self.shape(), e))
    }

    /// Singular values of `A`.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    pub fn singular_values(&self) -> Result<Vec<T::Real>> {
        self.as_mat_ref()
            .singular_values()
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
        self.as_mat_ref()
            .self_adjoint_eigenvalues(side)
            .map_err(|e| factorization_failed("self_adjoint_eigenvalues", self.shape(), e))
    }

    lu_solve!(ret solve,             "Solve `A x = b`.",           |s: &Self, r: &Self| (s.ncols(), r.ncols()), solve);
    lu_solve!(ret solve_transpose,   "Solve `A^T x = b`.",         |s: &Self, r: &Self| (s.nrows(), r.ncols()), solve_transpose);
    lu_solve!(ret solve_adjoint,     "Solve `A^H x = b`.",         |s: &Self, r: &Self| (s.nrows(), r.ncols()), solve_adjoint);
    lu_solve!(ret rsolve,            "Solve `x A = b`.",           |s: &Self, r: &Self| (r.nrows(), s.ncols()), rsolve);
    lu_solve!(ret rsolve_transpose,  "Solve `x A^T = b`.",         |s: &Self, r: &Self| (r.nrows(), s.nrows()), rsolve_transpose);
    lu_solve!(ret rsolve_adjoint,    "Solve `x A^H = b`.",         |s: &Self, r: &Self| (r.nrows(), s.nrows()), rsolve_adjoint);

    lu_solve!(inplace solve_in_place,           "Solve in place: `A x = b`.",   |s: &Self, r: &Self| (s.ncols(), r.ncols()), solve_in_place);
    lu_solve!(inplace solve_transpose_in_place, "Solve in place: `A^T x = b`.", |s: &Self, r: &Self| (s.nrows(), r.ncols()), solve_transpose_in_place);
    lu_solve!(inplace solve_adjoint_in_place,   "Solve in place: `A^H x = b`.", |s: &Self, r: &Self| (s.nrows(), r.ncols()), solve_adjoint_in_place);
    lu_solve!(inplace rsolve_in_place,          "Solve in place: `x A = b`.",   |s: &Self, r: &Self| (r.nrows(), s.ncols()), rsolve_in_place);

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

    tri_solve!(
        solve_lower_triangular_in_place,
        "Solve `L x = b` for a lower-triangular matrix `L` in place.",
        solve_lower_triangular_in_place
    );
    tri_solve!(
        solve_upper_triangular_in_place,
        "Solve `U x = b` for an upper-triangular matrix `U` in place.",
        solve_upper_triangular_in_place
    );
    tri_solve!(
        solve_unit_lower_triangular_in_place,
        "Solve `L x = b` for a unit lower-triangular matrix `L` in place.",
        solve_unit_lower_triangular_in_place
    );
    tri_solve!(
        solve_unit_upper_triangular_in_place,
        "Solve `U x = b` for a unit upper-triangular matrix `U` in place.",
        solve_unit_upper_triangular_in_place
    );

    factored_op!(infallible partial_piv_lu_reconstruct, "Reconstruct from partial-pivot LU factors.", partial_piv_lu, reconstruct);
    factored_op!(infallible partial_piv_lu_inverse,     "Inverse from partial-pivot LU factors.",     partial_piv_lu, inverse);

    factored_op!(fallible llt_reconstruct,  "Reconstruct from Cholesky factors.",  llt,  reconstruct);
    factored_op!(fallible llt_inverse,      "Inverse from Cholesky factors.",      llt,  inverse);
    factored_op!(fallible ldlt_reconstruct, "Reconstruct from LDLT factors.",      ldlt, reconstruct);
    factored_op!(fallible ldlt_inverse,     "Inverse from LDLT factors.",          ldlt, inverse);

    factored_op!(infallible_side lblt_reconstruct, "Reconstruct from Bunch-Kaufman factors.", lblt, reconstruct);

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

/// Diagonal matrix that stores only the `n` diagonal elements.
///
pub struct Diagonal<T: Scalar> {
    diag: Vec<T>,
}

impl<T: Scalar> Diagonal<T> {
    /// Create from a vector of diagonal elements.  The matrix is implicitly
    /// `n × n` where `n = diag.len()`.
    #[must_use]
    pub fn new(diag: Vec<T>) -> Self {
        Self { diag }
    }

    /// Side length (the matrix is square: `size × size`).
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.diag.len()
    }

    /// Diagonal element at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.size()`.
    #[must_use]
    #[inline]
    pub fn get(&self, i: usize) -> T {
        self.diag[i]
    }

    /// Convert to a dense `n × n` [`Tensor`] with zeros off the diagonal.
    #[must_use]
    pub fn to_tensor(&self) -> Tensor<T> {
        let n = self.size();
        Tensor::from_fn(
            n,
            n,
            |r, c| {
                if r == c {
                    self.diag[r]
                } else {
                    T::zero_impl()
                }
            },
        )
    }

    /// Efficient diagonal-times-dense multiplication: `D * rhs`.
    ///
    /// Each row `i` of the result is `self.diag[i] * rhs.row(i)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] when `rhs.nrows() != self.size()`.
    pub fn mul_dense(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        let n = self.size();
        if rhs.nrows() != n {
            return Err(Error::mismatch((n, rhs.ncols()), rhs.shape()));
        }
        Ok(Tensor::from_fn(n, rhs.ncols(), |r, c| {
            self.diag[r] * rhs.get(r, c)
        }))
    }
}

/// Symmetric matrix view — tags a [`Tensor`] for symmetric operations.
///
/// Only the triangle indicated by `side` is referenced by solvers.
pub struct Symmetric<T: Scalar> {
    tensor: Tensor<T, Cpu>,
    side: Side,
}

impl<T: Scalar> Symmetric<T> {
    /// Wrap `tensor` as symmetric, reading from `side`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimension`] when `tensor` is not square.
    pub fn new(tensor: Tensor<T, Cpu>, side: Side) -> Result<Self> {
        require_square(tensor.shape(), "Symmetric::new")?;
        Ok(Self { tensor, side })
    }

    /// Borrow the underlying tensor.
    #[must_use]
    #[inline]
    pub fn as_tensor(&self) -> &Tensor<T, Cpu> {
        &self.tensor
    }

    /// Which triangle is the authoritative source for solver routines.
    #[must_use]
    #[inline]
    pub fn side(&self) -> Side {
        self.side
    }

    /// Cholesky factorization `A = L Lᴴ` (or `A = Uᴴ U`).
    ///
    /// Delegates to [`Tensor::llt`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the matrix is not positive-definite.
    pub fn llt(&self) -> Result<solvers::Llt<T>> {
        self.tensor.llt(self.side)
    }

    /// Full self-adjoint eigendecomposition `A = V Λ Vᴴ`.
    ///
    /// Delegates to [`Tensor::self_adjoint_eigen`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the eigensolver fails to converge.
    pub fn eigen(&self) -> Result<solvers::SelfAdjointEigen<T>> {
        self.tensor.self_adjoint_eigen(self.side)
    }

    /// Eigenvalues only (ascending order).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the eigensolver fails to converge.
    pub fn eigenvalues(&self) -> Result<Vec<T::Real>> {
        self.tensor.self_adjoint_eigenvalues(self.side)
    }
}

/// Selects which kind of triangular structure a [`Triangular`] wrapper represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriKind {
    /// Lower-triangular (diagonal and below).
    Lower,
    /// Upper-triangular (diagonal and above).
    Upper,
    /// Unit lower-triangular (lower with implicit 1 on diagonal).
    UnitLower,
    /// Unit upper-triangular (upper with implicit 1 on diagonal).
    UnitUpper,
}

/// Triangular matrix view.
///
/// The underlying storage is a full square tensor; the structural tag
/// directs solver routines to use only the relevant triangle.
pub struct Triangular<T: Scalar> {
    tensor: Tensor<T, Cpu>,
    kind: TriKind,
}

impl<T: Scalar> Triangular<T> {
    /// Wrap `tensor` as triangular with the given `kind`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimension`] when `tensor` is not square.
    pub fn new(tensor: Tensor<T, Cpu>, kind: TriKind) -> Result<Self> {
        require_square(tensor.shape(), "Triangular::new")?;
        Ok(Self { tensor, kind })
    }

    /// Borrow the underlying tensor.
    #[must_use]
    #[inline]
    pub fn as_tensor(&self) -> &Tensor<T, Cpu> {
        &self.tensor
    }

    /// The triangular kind of this wrapper.
    #[must_use]
    #[inline]
    pub fn kind(&self) -> &TriKind {
        &self.kind
    }

    /// Solve `T x = b` in place, where `T` is this triangular matrix.
    ///
    /// Delegates to the appropriate [`Tensor`] triangular solve based on
    /// [`TriKind`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if dimensions mismatch.
    pub fn solve_in_place(&self, rhs: &mut Tensor<T, Cpu>) -> Result<()> {
        match self.kind {
            TriKind::Lower => self.tensor.solve_lower_triangular_in_place(rhs),
            TriKind::Upper => self.tensor.solve_upper_triangular_in_place(rhs),
            TriKind::UnitLower => self.tensor.solve_unit_lower_triangular_in_place(rhs),
            TriKind::UnitUpper => self.tensor.solve_unit_upper_triangular_in_place(rhs),
        }
    }
}
