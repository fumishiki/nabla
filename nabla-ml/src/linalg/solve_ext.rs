use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::{
    Side, bwd_sub, check_shape, from_f64_buf, fwd_sub, require_square, to_f64_buf,
    impl_solve_in_place, impl_triangular_solve_ip,
};
use super::chol::{Lblt, Ldlt, Llt};
use super::eigen::SelfAdjointEigen;
use super::lu::{FullPivLu, PartialPivLu};
use super::matrix_fn;
use super::qr::{ColPivQr, Qr};
use super::structured;
use super::svd::Svd;
use super::solve_types::Symmetric;

pub trait LinalgExt {
    /// LU decomposition with partial pivoting.
    fn partial_piv_lu(&self) -> Result<PartialPivLu<f64>>;
    /// Alias for `partial_piv_lu`.
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
    /// Alias for `singular_values`.
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
    /// Alias for `solve_lstsq`.
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
    fn det(&self) -> Result<f64>;
    /// Log absolute determinant.
    fn logdet(&self) -> Result<f64>;
    /// SVD returning `(U, singular_values, V^T)` tuple.
    fn svd_into(&self) -> Result<(Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>)>;
    /// QR returning `(Q, R)` tuple.
    fn qr_into(&self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>);
    /// Condition number σ_max / σ_min via SVD.
    fn cond(&self) -> Result<f64>;
    /// Numerical rank: count of singular values greater than `tol`.
    fn rank(&self, tol: f64) -> Result<usize>;
    /// Moore-Penrose pseudo-inverse.
    fn pinv(&self) -> Result<Tensor<f64, Cpu>>;
    /// Integer matrix power via binary exponentiation.
    fn matrix_power(&self, n: i32) -> Result<Tensor<f64, Cpu>>;
    /// Non-symmetric eigendecomposition via real Schur.
    fn eig_into(&self) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>, Tensor<f64, Cpu>)>;
    /// Alias for `eig_into`.
    fn eig(&self) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        self.eig_into()
    }
    /// Generalized eigenvalue problem `A x = λ B x`.
    fn geig(&self, b: &Self) -> Result<Vec<(f64, f64)>>;
    /// 1-norm condition number.
    fn cond1(&self) -> Result<f64>;
    /// Infinity-norm condition number.
    fn cond_inf(&self) -> Result<f64>;
    /// General p-norm condition number.
    fn cond_p(&self, p: f64) -> Result<f64>;
    /// Matrix inverse via LU factorization.
    fn inv(&self) -> Result<Tensor<f64, Cpu>>;
    /// Null space via SVD with tolerance.
    fn null_space(&self, tol: f64) -> Result<Tensor<f64, Cpu>>;
    /// Orthogonal basis for the column space via SVD.
    fn orth(&self, tol: f64) -> Result<Tensor<f64, Cpu>>;
    /// Sign and log absolute determinant.
    fn slogdet(&self) -> Result<(f64, f64)>;
    /// Matrix exponential `exp(A)`.
    fn expm(&self) -> Result<Tensor<f64, Cpu>>;
    /// Matrix logarithm `log(A)`.
    fn logm(&self) -> Result<Tensor<f64, Cpu>>;
    /// Matrix square root `A^{1/2}`.
    fn sqrtm(&self) -> Result<Tensor<f64, Cpu>>;
    /// Real Schur decomposition `A = Q T Q^T`.
    fn schur_decomp(&self) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)>;
    /// Polar decomposition `A = U H`.
    fn polar_decomp(&self) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)>;
}

impl LinalgExt for Tensor<f64, Cpu> {
    fn partial_piv_lu(&self) -> Result<PartialPivLu<f64>> {
        PartialPivLu::factorize(self)
    }

    fn full_piv_lu(&self) -> Result<FullPivLu<f64>> {
        FullPivLu::factorize(self)
    }

    fn qr(&self) -> Qr<f64> {
        Qr::factorize(self)
    }

    fn col_piv_qr(&self) -> ColPivQr<f64> {
        ColPivQr::factorize(self)
    }

    fn llt(&self, side: Side) -> Result<Llt<f64>> {
        Llt::factorize(self, side)
    }

    fn ldlt(&self, side: Side) -> Result<Ldlt<f64>> {
        Ldlt::factorize(self, side)
    }

    fn lblt(&self, side: Side) -> Lblt<f64> {
        Lblt::factorize(self, side)
    }

    fn svd(&self) -> Result<Svd<f64>> {
        Svd::factorize(self)
    }

    fn thin_svd(&self) -> Result<Svd<f64>> {
        Svd::factorize(self)
    }

    fn singular_values(&self) -> Result<Vec<f64>> {
        Svd::singular_values(self)
    }

    fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<f64>> {
        SelfAdjointEigen::factorize(self, side)
    }

    fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<f64>> {
        Ok(SelfAdjointEigen::factorize(self, side)?.eigenvalues().to_vec())
    }

    fn sym(&self, side: Side) -> Result<Symmetric<f64>> {
        Symmetric::new(self.clone(), side)
    }

    fn solve(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.ncols(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu()?;
        Ok(lu.solve_impl(rhs))
    }

    fn solve_transpose(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let at = self.t();
        let lu = at.partial_piv_lu()?;
        Ok(lu.solve_impl(rhs))
    }

    fn solve_adjoint(&self, rhs: &Self) -> Result<Self> {
        self.solve_transpose(rhs)
    }

    fn rsolve(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.ncols()), rhs.shape())?;
        let at = self.t();
        let bt = rhs.t();
        let lu = at.partial_piv_lu()?;
        let xt = lu.solve_impl(&bt);
        Ok(xt.t())
    }

    fn rsolve_transpose(&self, rhs: &Self) -> Result<Self> {
        check_shape((rhs.nrows(), self.nrows()), rhs.shape())?;
        let bt = rhs.t();
        let lu = self.partial_piv_lu()?;
        let xt = lu.solve_impl(&bt);
        Ok(xt.t())
    }

    fn rsolve_adjoint(&self, rhs: &Self) -> Result<Self> {
        self.rsolve_transpose(rhs)
    }

    fn solve_in_place(&self, rhs: &mut Self) -> Result<()> {
        check_shape((self.ncols(), rhs.ncols()), rhs.shape())?;
        let lu = self.partial_piv_lu()?;
        *rhs = lu.solve_impl(rhs);
        Ok(())
    }

    impl_solve_in_place!(solve_transpose_in_place, solve_transpose);
    impl_solve_in_place!(solve_adjoint_in_place, solve_adjoint);
    impl_solve_in_place!(rsolve_in_place, rsolve);

    fn solve_lstsq(&self, rhs: &Self) -> Result<Self> {
        check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
        let qr = self.qr();
        Ok(qr.solve_lstsq_impl(rhs))
    }

    impl_solve_in_place!(solve_lstsq_in_place, solve_lstsq);

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

    fn partial_piv_lu_reconstruct(&self) -> Result<Self> {
        Ok(PartialPivLu::factorize(self)?.reconstruct_impl())
    }

    fn partial_piv_lu_inverse(&self) -> Result<Self> {
        Ok(PartialPivLu::factorize(self)?.inverse_impl())
    }

    fn llt_reconstruct(&self, side: Side) -> Result<Self> {
        Ok(Llt::factorize(self, side)?.reconstruct_impl())
    }

    fn llt_inverse(&self, side: Side) -> Result<Self> {
        Ok(Llt::factorize(self, side)?.inverse_impl())
    }

    fn ldlt_reconstruct(&self, side: Side) -> Result<Self> {
        Ok(Ldlt::factorize(self, side)?.reconstruct_impl())
    }

    fn ldlt_inverse(&self, side: Side) -> Result<Self> {
        Ok(Ldlt::factorize(self, side)?.inverse_impl())
    }

    fn lblt_reconstruct(&self, side: Side) -> Self {
        Lblt::factorize(self, side).reconstruct_impl()
    }

    fn det(&self) -> Result<f64> {
        let lu = self.partial_piv_lu()?;
        Ok(lu.det())
    }

    fn logdet(&self) -> Result<f64> {
        self.slogdet().map(|(_, lad)| lad)
    }

    fn svd_into(&self) -> Result<(Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>)> {
        self.svd().map(Svd::into_parts)
    }

    fn qr_into(&self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>) {
        let (m, n) = self.shape();
        if n > 0 && m >= 2 * n {
            let block_rows = (n * 4).max(n);
            return crate::linalg::qr::tsqr(self, block_rows);
        }
        self.qr().into_parts()
    }

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

    fn rank(&self, tol: f64) -> Result<usize> {
        let svd = Svd::factorize(self)?;
        Ok(svd.s().iter().filter(|&&sv| sv > tol).count())
    }

    fn pinv(&self) -> Result<Tensor<f64, Cpu>> {
        let (m, n) = self.shape();
        let svd = Svd::factorize(self)?;
        let s = svd.s();
        let sigma_max = s.first().copied().unwrap_or(0.0);
        let tol = (m.max(n) as f64) * f64::EPSILON * sigma_max;
        let u = svd.u();
        let vt = svd.vt();
        let k = s.len();

        let pinv_buf: Vec<f64> = (0..n)
            .flat_map(|i| {
                (0..m).map(move |j| {
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

    fn matrix_power(&self, n: i32) -> Result<Tensor<f64, Cpu>> {
        let (rows, cols) = self.shape();
        require_square((rows, cols), "matrix_power")?;

        if n == 0 {
            return Ok(Tensor::<f64, Cpu>::identity(rows));
        }

        let base = if n < 0 {
            self.partial_piv_lu_inverse()?
        } else {
            self.clone()
        };

        let mut exp = n.unsigned_abs();
        let mut result = Tensor::<f64, Cpu>::identity(rows);
        let mut current = base;
        while exp > 0 {
            if exp & 1 == 1 {
                result = &result * &current;
            }
            current = &current * &current;
            exp >>= 1;
        }
        Ok(result)
    }

    fn eig_into(&self) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        let n = self.nrows();
        require_square(self.shape(), "eig_into")?;
        let (t, q) = matrix_fn::schur(self)?;
        let mut eigs = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            if i + 1 < n && t.get(i + 1, i).abs() > f64::EPSILON * 100.0 {
                let a = t.get(i, i);
                let b = t.get(i, i + 1);
                let c = t.get(i + 1, i);
                let d = t.get(i + 1, i + 1);
                let tr = a + d;
                let det = a * d - b * c;
                let disc = tr * tr - 4.0 * det;
                if disc < 0.0 {
                    let re = tr / 2.0;
                    let im = (-disc).sqrt() / 2.0;
                    eigs.push((re, im));
                    eigs.push((re, -im));
                } else {
                    let sq = disc.sqrt();
                    eigs.push(((tr + sq) / 2.0, 0.0));
                    eigs.push(((tr - sq) / 2.0, 0.0));
                }
                i += 2;
            } else {
                eigs.push((t.get(i, i), 0.0));
                i += 1;
            }
        }
        Ok((eigs, t, q))
    }

    fn geig(&self, b: &Self) -> Result<Vec<(f64, f64)>> {
        require_square(self.shape(), "geig")?;
        require_square(b.shape(), "geig")?;
        check_shape(self.shape(), b.shape())?;
        let n = self.nrows();
        if n == 0 {
            return Ok(Vec::new());
        }
        let llt = b.llt(Side::Lower)?;
        let l = &llt.l;
        let mut y = self.clone();
        l.solve_lower_triangular_in_place(&mut y)?;
        let mut yt = y.t();
        l.solve_lower_triangular_in_place(&mut yt)?;
        let c = yt.t();
        let (eigs, _, _) = c.eig_into()?;
        Ok(eigs)
    }

    fn cond1(&self) -> Result<f64> {
        require_square(self.shape(), "cond1")?;
        let n = self.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        let mut norm_a = 0.0f64;
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| self.get(i, j).abs()).sum();
            if col_sum > norm_a {
                norm_a = col_sum;
            }
        }
        let lu = self.partial_piv_lu()?;
        let inv = lu.inverse_impl();
        let mut norm_inv = 0.0f64;
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| inv.get(i, j).abs()).sum();
            if col_sum > norm_inv {
                norm_inv = col_sum;
            }
        }
        Ok(norm_a * norm_inv)
    }

    fn cond_inf(&self) -> Result<f64> {
        require_square(self.shape(), "cond_inf")?;
        let n = self.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        let mut norm_a = 0.0f64;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| self.get(i, j).abs()).sum();
            if row_sum > norm_a {
                norm_a = row_sum;
            }
        }
        let lu = self.partial_piv_lu()?;
        let inv = lu.inverse_impl();
        let mut norm_inv = 0.0f64;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| inv.get(i, j).abs()).sum();
            if row_sum > norm_inv {
                norm_inv = row_sum;
            }
        }
        Ok(norm_a * norm_inv)
    }

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

    fn inv(&self) -> Result<Tensor<f64, Cpu>> {
        Ok(self.partial_piv_lu()?.inverse_impl())
    }

    fn null_space(&self, tol: f64) -> Result<Tensor<f64, Cpu>> {
        let n = self.ncols();
        let svd = Svd::factorize(self)?;
        let s = svd.s();
        let vt = svd.vt();
        let null_indices: Vec<usize> = s
            .iter()
            .enumerate()
            .filter(|&(_, &sv)| sv < tol)
            .map(|(i, _)| i)
            .collect();
        if null_indices.is_empty() {
            return Ok(Tensor::<f64, Cpu>::zeros(n, 0));
        }
        let k = null_indices.len();
        let buf: Vec<f64> = (0..n)
            .flat_map(|row| null_indices.iter().map(move |&idx| vt.get(idx, row)))
            .collect();
        Ok(from_f64_buf(buf, n, k))
    }

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
            .flat_map(|row| (0..rank).map(move |col| u.get(row, col)))
            .collect();
        Ok(from_f64_buf(buf, m, rank))
    }

    fn slogdet(&self) -> Result<(f64, f64)> {
        let lu = self.partial_piv_lu()?;
        let n = lu.n;
        let lu_buf = to_f64_buf(&lu.lu);
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

    fn expm(&self) -> Result<Tensor<f64, Cpu>> {
        matrix_fn::expm(self)
    }

    fn logm(&self) -> Result<Tensor<f64, Cpu>> {
        matrix_fn::logm(self)
    }

    fn sqrtm(&self) -> Result<Tensor<f64, Cpu>> {
        matrix_fn::sqrtm(self)
    }

    fn schur_decomp(&self) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        matrix_fn::schur(self)
    }

    fn polar_decomp(&self) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        structured::polar(self)
    }
}
