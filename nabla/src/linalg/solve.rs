// LinalgExt trait + impl for Tensor<f64, Cpu>, convenience wrappers,
// and wrapper types: Diagonal, Symmetric, Triangular, TriKind.

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    Side, bwd_sub, check_shape, from_f64_buf, fwd_sub, require_square, to_f64_buf,
    impl_factorization_methods, impl_solve_in_place, impl_triangular_solve_ip,
};
use super::{
    chol::{Lblt, Ldlt, Llt},
    eigen::SelfAdjointEigen,
    francis::francis_qr_schur,
    lu::{FullPivLu, PartialPivLu},
    qr::{ColPivQr, Qr},
    svd::Svd,
};

// ===========================================================================
// LinalgExt trait
// ===========================================================================

/// Extension trait adding dense linear algebra methods to `Tensor<f64, Cpu>`.
pub trait LinalgExt {
    /// LU decomposition with partial pivoting.
    fn partial_piv_lu(&self) -> Result<PartialPivLu<f64>>;
    /// Alias for [`partial_piv_lu`](Self::partial_piv_lu).
    fn lu(&self) -> Result<PartialPivLu<f64>> {
        self.partial_piv_lu()
    }
    /// LU decomposition with full pivoting.
    fn full_piv_lu(&self) -> Result<FullPivLu<f64>>;
    /// QR decomposition.
    fn qr(&self) -> Qr<f64>;
    /// Column-pivoted QR decomposition.
    fn col_piv_qr(&self) -> ColPivQr<f64>;
    /// Cholesky factorization.
    fn llt(&self, side: Side) -> Result<Llt<f64>>;
    /// Alias for lower-triangle Cholesky.
    fn chol(&self) -> Result<Llt<f64>> {
        self.llt(Side::Lower)
    }
    /// LDL^T factorization.
    fn ldlt(&self, side: Side) -> Result<Ldlt<f64>>;
    /// Alias for lower-triangle LDL^T.
    fn ldl(&self) -> Result<Ldlt<f64>> {
        self.ldlt(Side::Lower)
    }
    /// Bunch-Kaufman LBL^T decomposition.
    fn lblt(&self, side: Side) -> Lblt<f64>;
    /// Singular value decomposition.
    fn svd(&self) -> Result<Svd<f64>>;
    /// Thin SVD.
    fn thin_svd(&self) -> Result<Svd<f64>>;
    /// Singular values in descending order.
    fn singular_values(&self) -> Result<Vec<f64>>;
    /// Alias for [`singular_values`](Self::singular_values).
    fn svdvals(&self) -> Result<Vec<f64>> {
        self.singular_values()
    }
    /// Self-adjoint eigendecomposition.
    fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<f64>>;
    /// Eigenvalues of a self-adjoint matrix.
    fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<f64>>;
    /// Wrap as a symmetric view.
    fn sym(&self, side: Side) -> Result<Symmetric<f64>>;
    /// Solve A*x = b.
    fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Solve A^T*x = b.
    fn solve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Solve A^H*x = b.
    fn solve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Solve x*A = b.
    fn rsolve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Solve x*A^T = b.
    fn rsolve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Solve x*A^H = b.
    fn rsolve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Solve in place A*x = b.
    fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve in place A^T*x = b.
    fn solve_transpose_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve in place A^H*x = b.
    fn solve_adjoint_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve in place x*A = b.
    fn rsolve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Least-squares solve.
    fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>;
    /// Alias for [`solve_lstsq`](Self::solve_lstsq).
    fn lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        self.solve_lstsq(rhs)
    }
    /// Least-squares solve in place.
    fn solve_lstsq_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve lower triangular in place.
    fn solve_lower_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve upper triangular in place.
    fn solve_upper_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve unit lower triangular in place.
    fn solve_unit_lower_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Solve unit upper triangular in place.
    fn solve_unit_upper_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()>;
    /// Reconstruct from partial-pivot LU.
    fn partial_piv_lu_reconstruct(&self) -> Result<Tensor<f64, Cpu>>;
    /// Inverse from partial-pivot LU.
    fn partial_piv_lu_inverse(&self) -> Result<Tensor<f64, Cpu>>;
    /// Reconstruct from Cholesky.
    fn llt_reconstruct(&self, side: Side) -> Result<Tensor<f64, Cpu>>;
    /// Inverse from Cholesky.
    fn llt_inverse(&self, side: Side) -> Result<Tensor<f64, Cpu>>;
    /// Reconstruct from LDL^T.
    fn ldlt_reconstruct(&self, side: Side) -> Result<Tensor<f64, Cpu>>;
    /// Inverse from LDL^T.
    fn ldlt_inverse(&self, side: Side) -> Result<Tensor<f64, Cpu>>;
    /// Reconstruct from Bunch-Kaufman.
    fn lblt_reconstruct(&self, side: Side) -> Tensor<f64, Cpu>;
    /// Determinant via LU decomposition.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn det(&self) -> Result<f64>;
    /// Log absolute determinant (numerically stable).
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn logdet(&self) -> Result<f64>;
    /// SVD returning `(U, singular_values, V^T)` tuple.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn svd_into(&self) -> Result<(Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>)>;
    /// QR returning `(Q, R)` tuple.
    fn qr_into(&self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>);
    /// Condition number σ_max / σ_min via SVD.
    ///
    /// Returns `0.0` for empty matrices and `f64::INFINITY` for rank-deficient matrices.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn cond(&self) -> Result<f64>;
    /// Numerical rank: count of singular values greater than `tol`.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn rank(&self, tol: f64) -> Result<usize>;
    /// Moore-Penrose pseudo-inverse `A⁺ = V · diag(1/σᵢ) · U^T`.
    ///
    /// Singular values below `max(m,n) * ε * σ_max` are treated as zero.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn pinv(&self) -> Result<Tensor<f64, Cpu>>;
    /// Integer matrix power via binary exponentiation.
    ///
    /// - `n = 0` → identity
    /// - `n > 0` → repeated squaring
    /// - `n < 0` → inverse then repeated squaring
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square, or `n < 0` and the matrix is singular.
    fn matrix_power(&self, n: i32) -> Result<Tensor<f64, Cpu>>;
    /// Non-symmetric eigendecomposition via Francis implicit double-shift QR.
    ///
    /// Returns `(eigenvalues, Schur-form T)`.
    /// Each eigenvalue is a `(real, imag)` pair.
    /// Complex eigenvalues appear as conjugate pairs read from 2×2 diagonal blocks.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square or the QR iteration fails to converge.
    fn eig_into(&self) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>)>;

    /// Generalized eigenvalue problem: `A x = λ B x`.
    ///
    /// Reduces to standard form via Cholesky of B: `L⁻¹ A L⁻ᵀ z = λ z`.
    /// Returns eigenvalues as `(real, imaginary)` pairs.
    ///
    /// # Errors
    /// Returns `Err` when B is not positive-definite, shapes mismatch,
    /// or eigenvalue computation fails.
    fn geig(&self, b: &Self) -> Result<Vec<(f64, f64)>>;

    /// 1-norm condition number: `‖A‖₁ · ‖A⁻¹‖₁`.
    ///
    /// Uses LU factorization for the inverse. Returns `f64::INFINITY` for
    /// singular matrices.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square or LU factorization fails.
    fn cond1(&self) -> Result<f64>;

    /// Infinity-norm condition number: `‖A‖_∞ · ‖A⁻¹‖_∞`.
    ///
    /// Uses LU factorization for the inverse. Returns `f64::INFINITY` for
    /// singular matrices.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square or LU factorization fails.
    fn cond_inf(&self) -> Result<f64>;

    /// General p-norm condition number.
    ///
    /// Dispatches based on `p`:
    /// - `p = 1.0` → [`cond1`](Self::cond1) (1-norm via LU)
    /// - `p = 2.0` → [`cond`](Self::cond) (2-norm via SVD)
    /// - `p = f64::INFINITY` → [`cond_inf`](Self::cond_inf) (infinity-norm via LU)
    ///
    /// # Errors
    /// Returns `Err` for unsupported `p` values, or when the underlying method fails.
    fn cond_p(&self, p: f64) -> Result<f64>;

    /// Matrix inverse via LU factorization.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn inv(&self) -> Result<Tensor<f64, Cpu>>;

    /// Null space: right singular vectors corresponding to singular values < `tol`.
    ///
    /// Returns a matrix whose columns span the null space. If no null space
    /// exists, returns an empty tensor with 0 columns.
    fn null_space(&self, tol: f64) -> Result<Tensor<f64, Cpu>>;

    /// Orthogonal basis for the column space via SVD.
    ///
    /// Returns left singular vectors corresponding to singular values >= `tol`.
    fn orth(&self, tol: f64) -> Result<Tensor<f64, Cpu>>;

    /// Sign and log absolute determinant.
    ///
    /// Returns `(sign, log_abs_det)` where sign is `1.0` or `-1.0`.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn slogdet(&self) -> Result<(f64, f64)>;
}

impl LinalgExt for Tensor<f64, Cpu> {
    /// LU decomposition with partial pivoting (`PA = LU`).
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn partial_piv_lu(&self) -> Result<PartialPivLu<f64>> {
        PartialPivLu::factorize(self)
    }

    /// LU decomposition with full pivoting (`PAQ^T = LU`).
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn full_piv_lu(&self) -> Result<FullPivLu<f64>> {
        FullPivLu::factorize(self)
    }

    /// QR decomposition (`A = QR`).
    fn qr(&self) -> Qr<f64> {
        Qr::factorize(self)
    }

    /// Column-pivoted QR decomposition (`A·P^T = QR`).
    fn col_piv_qr(&self) -> ColPivQr<f64> {
        ColPivQr::factorize(self)
    }

    /// Cholesky factorization (`A = L·L^T`).
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or not positive-definite.
    fn llt(&self, side: Side) -> Result<Llt<f64>> {
        Llt::factorize(self, side)
    }

    /// LDL^T factorization.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or factorization fails.
    fn ldlt(&self, side: Side) -> Result<Ldlt<f64>> {
        Ldlt::factorize(self, side)
    }

    /// Bunch-Kaufman `LBL^T` decomposition.
    fn lblt(&self, side: Side) -> Lblt<f64> {
        Lblt::factorize(self, side)
    }

    /// Singular value decomposition (`A = U·Σ·V^H`).
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn svd(&self) -> Result<Svd<f64>> {
        Svd::factorize(self)
    }

    /// Thin SVD (same as full SVD for f64 in this implementation).
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn thin_svd(&self) -> Result<Svd<f64>> {
        Svd::factorize(self)
    }

    /// Singular values of `A` in descending order.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn singular_values(&self) -> Result<Vec<f64>> {
        Ok(Svd::factorize(self)?.s().to_vec())
    }

    /// Self-adjoint eigendecomposition.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or eigensolver fails.
    fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<f64>> {
        SelfAdjointEigen::factorize(self, side)
    }

    /// Eigenvalues of a self-adjoint matrix (ascending order).
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square or eigensolver fails.
    fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<f64>> {
        Ok(SelfAdjointEigen::factorize(self, side)?.eigenvalues().to_vec())
    }

    /// Symmetric matrix view.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not square.
    fn sym(&self, side: Side) -> Result<Symmetric<f64>> {
        Symmetric::new(self.clone(), side)
    }

    // --- Solve methods (delegate to PartialPivLu) ---

    /// Solve `A·x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn solve(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.ncols(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu()?;
        Ok(lu.solve_impl(rhs))
    }

    /// Solve `A^T·x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn solve_transpose(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let at = self.t();
        let lu = at.partial_piv_lu()?;
        Ok(lu.solve_impl(rhs))
    }

    /// Solve `A^H·x = b` (same as `A^T` for real scalars).
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn solve_adjoint(&self, rhs: &Self) -> Result<Self> {
        self.solve_transpose(rhs)
    }

    /// Solve `x·A = b` (i.e., `A^T·x^T = b^T`).
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn rsolve(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.ncols()), rhs.shape())?;
        // x*A = b => A^T*x^T = b^T
        let at = self.t();
        let bt = rhs.t();
        let lu = at.partial_piv_lu()?;
        let xt = lu.solve_impl(&bt);
        Ok(xt.t())
    }

    /// Solve `x·A^T = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn rsolve_transpose(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.nrows()), rhs.shape())?;
        let bt = rhs.t();
        let lu = self.partial_piv_lu()?;
        let xt = lu.solve_impl(&bt);
        Ok(xt.t())
    }

    /// Solve `x·A^H = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn rsolve_adjoint(&self, rhs: &Self) -> Result<Self> {
        self.rsolve_transpose(rhs)
    }

    /// Solve in place: `A·x = b` (overwrites `rhs` with `x`).
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch or solve fails.
    fn solve_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((self.ncols(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu()?;
        *rhs = lu.solve_impl(rhs);
        Ok(())
    }

    impl_solve_in_place!(solve_transpose_in_place, solve_transpose);
    impl_solve_in_place!(solve_adjoint_in_place, solve_adjoint);
    impl_solve_in_place!(rsolve_in_place, rsolve);

    /// Least-squares solve by QR: `A·x = b`.
    ///
    /// # Errors
    /// Returns `Err` when dimensions mismatch.
    fn solve_lstsq(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let qr = self.qr();
        Ok(qr.solve_lstsq_impl(rhs))
    }

    impl_solve_in_place!(solve_lstsq_in_place, solve_lstsq);

    // --- Triangular solves ---

    impl_triangular_solve_ip!(
        solve_lower_triangular_in_place,
        "solve_lower_triangular_in_place",
        fwd_sub,
        false
    );
    impl_triangular_solve_ip!(
        solve_upper_triangular_in_place,
        "solve_upper_triangular_in_place",
        bwd_sub,
        false
    );
    impl_triangular_solve_ip!(
        solve_unit_lower_triangular_in_place,
        "solve_unit_lower_triangular_in_place",
        fwd_sub,
        true
    );
    impl_triangular_solve_ip!(
        solve_unit_upper_triangular_in_place,
        "solve_unit_upper_triangular_in_place",
        bwd_sub,
        true
    );

    // --- Factored operations ---

    /// Reconstruct from partial-pivot LU factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization fails.
    fn partial_piv_lu_reconstruct(&self) -> Result<Self> {
        Ok(PartialPivLu::factorize(self)?.reconstruct_impl())
    }

    /// Inverse from partial-pivot LU factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization fails.
    fn partial_piv_lu_inverse(&self) -> Result<Self> {
        Ok(PartialPivLu::factorize(self)?.inverse_impl())
    }

    /// Reconstruct from Cholesky factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization fails.
    fn llt_reconstruct(&self, side: Side) -> Result<Self> {
        Ok(Llt::factorize(self, side)?.reconstruct_impl())
    }

    /// Inverse from Cholesky factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization fails.
    fn llt_inverse(&self, side: Side) -> Result<Self> {
        Ok(Llt::factorize(self, side)?.inverse_impl())
    }

    /// Reconstruct from LDL^T factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization fails.
    fn ldlt_reconstruct(&self, side: Side) -> Result<Self> {
        Ok(Ldlt::factorize(self, side)?.reconstruct_impl())
    }

    /// Inverse from LDL^T factors.
    ///
    /// # Errors
    /// Returns `Err` when factorization fails.
    fn ldlt_inverse(&self, side: Side) -> Result<Self> {
        Ok(Ldlt::factorize(self, side)?.inverse_impl())
    }

    /// Reconstruct from Bunch-Kaufman factors.
    fn lblt_reconstruct(&self, side: Side) -> Self {
        Lblt::factorize(self, side).reconstruct_impl()
    }

    /// Determinant via LU decomposition.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn det(&self) -> Result<f64> {
        let lu = self.partial_piv_lu()?;
        Ok(lu.det())
    }

    /// Log absolute determinant (numerically stable).
    ///
    /// Convenience wrapper around [`slogdet`](Self::slogdet) returning only the
    /// log-absolute-determinant component.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn logdet(&self) -> Result<f64> {
        self.slogdet().map(|(_, lad)| lad)
    }

    /// SVD returning `(U, singular_values, V^T)` tuple.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn svd_into(&self) -> Result<(Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>)> {
        self.svd().map(Svd::into_parts)
    }

    /// QR returning `(Q, R)` tuple.
    fn qr_into(&self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>) {
        self.qr().into_parts()
    }

    /// Condition number σ_max / σ_min via SVD.
    ///
    /// Returns `0.0` for empty matrices and `f64::INFINITY` for rank-deficient matrices.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn cond(&self) -> Result<f64> {
        let svd = Svd::factorize(self)?;
        let s = svd.s();
        if s.is_empty() {
            return Ok(0.0);
        }
        let max = s[0];
        let min = s[s.len() - 1];
        if min.abs() < 1e-15 {
            return Ok(f64::INFINITY);
        }
        Ok(max / min)
    }

    /// Numerical rank: count of singular values greater than `tol`.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn rank(&self, tol: f64) -> Result<usize> {
        let svd = Svd::factorize(self)?;
        Ok(svd.s().iter().filter(|&&sv| sv > tol).count())
    }

    /// Moore-Penrose pseudo-inverse `A⁺ = V · diag(1/σᵢ) · U^T`.
    ///
    /// Singular values below `max(m,n) * ε * σ_max` are treated as zero.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn pinv(&self) -> Result<Tensor<f64, Cpu>> {
        let (m, n) = self.shape();
        let svd = Svd::factorize(self)?;
        let s = svd.s();
        let sigma_max = s.first().copied().unwrap_or(0.0);
        let tol = (m.max(n) as f64) * f64::EPSILON * sigma_max;

        // A+ = V * diag(1/s_i if s_i > tol else 0) * U^T
        // vt is (n x n), u is (m x m); we need V = vt^T (n x n)
        // pinv shape: n x m
        let u = svd.u();
        let vt = svd.vt();
        let k = s.len();

        let pinv_buf: Vec<f64> = (0..n)
            .flat_map(|i| {
                (0..m).map(move |j| {
                    // pinv[i,j] = sum_r  V[i,r] * (1/s[r]) * U[j,r]
                    //           = sum_r  vt[r,i] * (1/s[r]) * u[j,r]
                    (0..k)
                        .map(|r| {
                            if s[r] > tol {
                                vt.get(r, i) * (1.0 / s[r]) * u.get(j, r)
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                })
            })
            .collect();
        Ok(from_f64_buf(pinv_buf, n, m))
    }

    /// Integer matrix power via binary exponentiation.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square, or `n < 0` and the matrix is singular.
    fn matrix_power(&self, n: i32) -> Result<Tensor<f64, Cpu>> {
        let (rows, cols) = self.shape();
        require_square((rows, cols), "matrix_power")?;

        if n == 0 {
            return Ok(Tensor::<f64, Cpu>::identity(rows));
        }

        // For negative powers compute inverse first
        let base = if n < 0 {
            self.partial_piv_lu_inverse()?
        } else {
            self.clone()
        };

        let exp = n.unsigned_abs();
        Ok(binary_matpow(&base, exp, rows))
    }

    /// Non-symmetric eigendecomposition via Francis implicit double-shift QR.
    ///
    /// Returns `(eigenvalues, Schur-form T)`.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square or QR iteration fails to converge.
    fn eig_into(&self) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>)> {
        require_square(self.shape(), "eig_into")?;
        let n = self.nrows();
        if n == 0 {
            return Ok((Vec::new(), Tensor::<f64, Cpu>::zeros(0, 0)));
        }
        francis_qr_schur(self, n)
    }

    /// Generalized eigenvalue problem: `A x = λ B x`.
    ///
    /// # Errors
    /// Returns `Err` when B is not positive-definite, shapes mismatch,
    /// or eigenvalue computation fails.
    fn geig(&self, b: &Self) -> Result<Vec<(f64, f64)>> {
        require_square(self.shape(), "geig")?;
        require_square(b.shape(), "geig")?;
        check_shape(self.shape(), b.shape())?;
        let n = self.nrows();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Cholesky: B = L L^T
        let llt = b.llt(Side::Lower)?;
        let l = &llt.l;
        // C = L^{-1} A L^{-T}
        // Step 1: solve L Y = A for Y = L^{-1} A  (column by column via triangular solve)
        let mut y = self.clone();
        l.solve_lower_triangular_in_place(&mut y)?;
        // Step 2: C = Y L^{-T} = (L^{-1} Y^T)^T
        let mut yt = y.t();
        l.solve_lower_triangular_in_place(&mut yt)?;
        let c = yt.t();
        // Standard eigenvalue problem on C
        let (eigs, _) = c.eig_into()?;
        Ok(eigs)
    }

    /// 1-norm condition number: `‖A‖₁ · ‖A⁻¹‖₁`.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square or LU factorization fails.
    fn cond1(&self) -> Result<f64> {
        require_square(self.shape(), "cond1")?;
        let n = self.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        // ||A||_1 = max column sum of absolute values
        let norm_a = norm1_impl(self, n);
        // Inverse via LU
        let lu = self.partial_piv_lu()?;
        let inv = lu.inverse_impl();
        let norm_inv = norm1_impl(&inv, n);
        Ok(norm_a * norm_inv)
    }

    /// Infinity-norm condition number: `‖A‖_∞ · ‖A⁻¹‖_∞`.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is not square or LU factorization fails.
    fn cond_inf(&self) -> Result<f64> {
        require_square(self.shape(), "cond_inf")?;
        let n = self.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        // ||A||_inf = max row sum of absolute values
        let norm_a = norm_inf_impl(self, n);
        let lu = self.partial_piv_lu()?;
        let inv = lu.inverse_impl();
        let norm_inv = norm_inf_impl(&inv, n);
        Ok(norm_a * norm_inv)
    }

    /// General p-norm condition number.
    ///
    /// # Errors
    /// Returns `Err` for unsupported `p` values, or when the underlying method fails.
    fn cond_p(&self, p: f64) -> Result<f64> {
        if (p - 1.0).abs() < f64::EPSILON {
            self.cond1()
        } else if (p - 2.0).abs() < f64::EPSILON {
            self.cond()
        } else if p.is_infinite() && p > 0.0 {
            self.cond_inf()
        } else {
            Err(Error::invalid(format!(
                "cond_p: unsupported p={p}; only 1.0, 2.0, and inf are supported"
            )))
        }
    }

    /// Matrix inverse via LU factorization.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn inv(&self) -> Result<Tensor<f64, Cpu>> {
        Ok(self.partial_piv_lu()?.inverse_impl())
    }

    /// Null space: right singular vectors corresponding to singular values < `tol`.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn null_space(&self, tol: f64) -> Result<Tensor<f64, Cpu>> {
        let n = self.ncols();
        let svd = Svd::factorize(self)?;
        let s = svd.s();
        let vt = svd.vt();
        // Columns of V (rows of V^T) where sigma < tol
        let null_indices: Vec<usize> = s.iter().enumerate()
            .filter(|&(_, &sv)| sv < tol)
            .map(|(i, _)| i)
            .collect();
        if null_indices.is_empty() {
            return Ok(Tensor::<f64, Cpu>::zeros(n, 0));
        }
        let k = null_indices.len();
        let buf: Vec<f64> = (0..n)
            .flat_map(|row| {
                null_indices.iter().map(move |&idx| vt.get(idx, row))
            })
            .collect();
        Ok(from_f64_buf(buf, n, k))
    }

    /// Orthogonal basis for the column space via SVD.
    ///
    /// # Errors
    /// Returns `Err` when SVD fails to converge.
    fn orth(&self, tol: f64) -> Result<Tensor<f64, Cpu>> {
        let m = self.nrows();
        let svd = Svd::factorize(self)?;
        let s = svd.s();
        let u = svd.u();
        let rank = s.iter().filter(|&&sv| sv >= tol).count();
        if rank == 0 {
            return Ok(Tensor::<f64, Cpu>::zeros(m, 0));
        }
        let buf: Vec<f64> = (0..m)
            .flat_map(|row| {
                (0..rank).map(move |col| u.get(row, col))
            })
            .collect();
        Ok(from_f64_buf(buf, m, rank))
    }

    /// Sign and log absolute determinant.
    ///
    /// # Errors
    /// Returns `Err` when the matrix is singular or not square.
    fn slogdet(&self) -> Result<(f64, f64)> {
        let lu = self.partial_piv_lu()?;
        let n = lu.n;
        let lu_buf = to_f64_buf(&lu.lu);
        // Compute sign from permutation parity and diagonal signs
        let mut visited = vec![false; n];
        let mut n_cycles = 0usize;
        for start in 0..n {
            if !visited[start] {
                n_cycles += 1;
                let mut cur = start;
                while !visited[cur] {
                    visited[cur] = true;
                    cur = lu.piv[cur];
                }
            }
        }
        let perm_sign: f64 = if (n - n_cycles).is_multiple_of(2) { 1.0 } else { -1.0 };
        let mut sign = perm_sign;
        let mut log_abs_det = 0.0f64;
        for i in 0..n {
            let d = super::buf_get(&lu_buf, n, i, i);
            if d < 0.0 {
                sign = -sign;
            }
            log_abs_det += d.abs().ln();
        }
        Ok((sign, log_abs_det))
    }
}

// ===========================================================================
// Solve methods on factorization types (via macro)
// ===========================================================================

impl_factorization_methods!(general PartialPivLu);
impl_factorization_methods!(symmetric Llt);
impl_factorization_methods!(symmetric Ldlt);

impl Lblt<f64> {
    /// Reconstruct the stored matrix.
    #[must_use]
    pub fn reconstruct(&self) -> Tensor<f64, Cpu> {
        self.reconstruct_impl()
    }
}

impl Qr<f64> {
    /// Least-squares solve `A·x ≈ b`.
    #[must_use]
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        self.solve_lstsq_impl(rhs)
    }

    /// In-place least-squares solve.
    pub fn solve_lstsq_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
        *rhs = self.solve_lstsq_impl(rhs);
    }

    /// Extract Q matrix (`m × m` orthogonal) from the stored Householder form.
    ///
    /// Computes `Q = H_0 · H_1 · … · H_{k-1}` by applying `Q^T` to the m×m
    /// identity and transposing the result.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn q_matrix(&self) -> Tensor<f64, Cpu> {
        let m = self.m;
        // Build m×m identity as row-major flat buffer (n_rhs = m)
        let mut eye = vec![0.0f64; m * m];
        for i in 0..m {
            eye[i * m + i] = 1.0;
        }
        // apply_qt computes Q^T * rhs; feed identity to get Q^T
        let qt = self.apply_qt(&eye, m);
        // Transpose Q^T to get Q
        let mut q = vec![0.0f64; m * m];
        for r in 0..m {
            for c in 0..m {
                q[r * m + c] = qt[c * m + r];
            }
        }
        from_f64_buf(q, m, m)
    }

    /// Extract R matrix (`min(m,n) × n` upper triangular) from the stored QR buffer.
    ///
    /// Returns the upper triangular part of the first `k = min(m,n)` rows of the
    /// internal QR buffer, which contains R in its upper triangle.
    #[must_use]
    pub fn r_matrix(&self) -> Tensor<f64, Cpu> {
        let (m, n) = (self.m, self.n);
        let k = m.min(n);
        let qr_buf = to_f64_buf(&self.qr);
        let mut r = vec![0.0f64; k * n];
        for i in 0..k {
            for j in i..n {
                r[i * n + j] = super::buf_get(&qr_buf, n, i, j);
            }
        }
        from_f64_buf(r, k, n)
    }

    /// Unpack into `(Q, R)` matrices.
    ///
    /// Returns the `m × m` orthogonal matrix Q and the `min(m,n) × n` upper
    /// triangular matrix R such that `A = Q · R` (with Q trimmed to `m × min(m,n)`
    /// for rectangular A via the leading columns of the full Q).
    #[must_use]
    pub fn into_parts(self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>) {
        let q = self.q_matrix();
        let r = self.r_matrix();
        (q, r)
    }
}

impl ColPivQr<f64> {
    /// Least-squares solve `A·x ≈ b`.
    #[must_use]
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        self.solve_lstsq_impl(rhs)
    }
}

// ---------------------------------------------------------------------------
// binary_matpow — integer matrix power by repeated squaring
// ---------------------------------------------------------------------------

/// Compute the matrix 1-norm: max over columns of the absolute column sum.
fn norm1_impl(a: &Tensor<f64, Cpu>, n: usize) -> f64 {
    let mut max_col_sum = 0.0f64;
    for j in 0..n {
        let col_sum: f64 = (0..n).map(|i| a.get(i, j).abs()).sum();
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

/// Compute the matrix infinity-norm: max over rows of the absolute row sum.
fn norm_inf_impl(a: &Tensor<f64, Cpu>, n: usize) -> f64 {
    let mut max_row_sum = 0.0f64;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| a.get(i, j).abs()).sum();
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    max_row_sum
}

/// Compute `base^exp` for a square matrix via binary (repeated) squaring.
fn binary_matpow(base: &Tensor<f64, Cpu>, mut exp: u32, n: usize) -> Tensor<f64, Cpu> {
    let mut result = Tensor::<f64, Cpu>::identity(n);
    let mut current = base.clone();
    while exp > 0 {
        if exp & 1 == 1 {
            result = &result * &current;
        }
        current = &current * &current;
        exp >>= 1;
    }
    result
}

// ===========================================================================
// Diagonal
// ===========================================================================

/// Diagonal matrix that stores only the `n` diagonal elements.
pub struct Diagonal<T: Scalar> {
    diag: Vec<T>,
}

impl<T: Scalar> Diagonal<T> {
    /// Create from a vector of diagonal elements. Matrix is implicitly `n × n`.
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
                if r == c { self.diag[r] } else { T::zero_impl() }
            },
        )
    }

    /// Efficient diagonal-times-dense multiplication: `D * rhs`.
    ///
    /// # Errors
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

// ===========================================================================
// Symmetric
// ===========================================================================

/// Symmetric matrix view — tags a [`Tensor`] for symmetric operations.
pub struct Symmetric<T: Scalar> {
    tensor: Tensor<T, Cpu>,
    side: Side,
}

impl<T: Scalar> Symmetric<T> {
    /// Wrap `tensor` as symmetric, reading from `side`.
    ///
    /// # Errors
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
}

impl Symmetric<f64> {
    /// Cholesky factorization `A = L·L^T`.
    ///
    /// # Errors
    /// Returns `Err` if the matrix is not positive-definite.
    pub fn llt(&self) -> Result<Llt<f64>> {
        self.tensor.llt(self.side)
    }

    /// Full self-adjoint eigendecomposition `A = V·Λ·V^T`.
    ///
    /// # Errors
    /// Returns `Err` if the eigensolver fails to converge.
    pub fn eigen(&self) -> Result<SelfAdjointEigen<f64>> {
        self.tensor.self_adjoint_eigen(self.side)
    }

    /// Alias for [`eigen`](Self::eigen).
    ///
    /// # Errors
    /// Returns `Err` if the eigensolver fails to converge.
    pub fn eigh(&self) -> Result<SelfAdjointEigen<f64>> {
        self.eigen()
    }

    /// Eigenvalues only (ascending order).
    ///
    /// # Errors
    /// Returns `Err` if the eigensolver fails to converge.
    pub fn eigenvalues(&self) -> Result<Vec<f64>> {
        self.tensor.self_adjoint_eigenvalues(self.side)
    }
}

// ===========================================================================
// TriKind / Triangular
// ===========================================================================

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
pub struct Triangular<T: Scalar> {
    tensor: Tensor<T, Cpu>,
    kind: TriKind,
}

impl<T: Scalar> Triangular<T> {
    /// Wrap `tensor` as triangular with the given `kind`.
    ///
    /// # Errors
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
}

impl Triangular<f64> {
    /// Solve `T·x = b` in place, where `T` is this triangular matrix.
    ///
    /// # Errors
    /// Returns `Err` if dimensions mismatch.
    pub fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        match self.kind {
            TriKind::Lower => self.tensor.solve_lower_triangular_in_place(rhs),
            TriKind::Upper => self.tensor.solve_upper_triangular_in_place(rhs),
            TriKind::UnitLower => self.tensor.solve_unit_lower_triangular_in_place(rhs),
            TriKind::UnitUpper => self.tensor.solve_unit_upper_triangular_in_place(rhs),
        }
    }
}
