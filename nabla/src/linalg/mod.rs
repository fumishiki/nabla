// nabla-linalg — Self-contained dense linear algebra factorizations for Tensor<T, Cpu>.
//
// All algorithms operate on Tensor<T, Cpu> via .get()/.set()/.from_fn()/.zeros().
// No external BLAS/LAPACK dependencies.

use core::fmt;

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

// Submodules — macros must be declared before the submodules that use them.
/// LU decompositions (partial-pivot and full-pivot).
pub mod lu;
/// QR decompositions (Householder and column-pivoted).
pub mod qr;
/// Cholesky-family factorizations (LLT, LDLT, Bunch-Kaufman LBLT).
pub mod chol;
/// Singular value decomposition (full and thin SVD).
pub mod svd;
/// Self-adjoint eigendecomposition (Jacobi iteration).
pub mod eigen;
mod francis;
/// Matrix equations: Sylvester, Lyapunov, continuous Riccati, tridiagonal.
pub mod equation;
/// Matrix functions: expm, logm, sqrtm, schur, polar, balance.
pub mod matrix_fn;
/// Linear solvers and the [`LinalgExt`] trait.
pub mod solve;
/// Structured matrix constructors and Fréchet derivative.
pub mod structured;

// Re-export all public items so `nabla::linalg::*` paths are preserved.
pub use {
    chol::{Lblt, Ldlt, Llt},
    eigen::SelfAdjointEigen,
    equation::{discrete_lyapunov, discrete_sylvester, lyapunov, solve_tridiag, sylvester},
    lu::{FullPivLu, PartialPivLu},
    matrix_fn::{expm, logm, schur, sqrtm},
    qr::{ColPivQr, Qr},
    solve::{Diagonal, LinalgExt, Symmetric, TriKind, Triangular},
    structured::{
        balance, care, circulant, continuous_riccati, frechet_deriv, hessenberg, polar, toeplitz,
        vandermonde, vandermonde_rect,
    },
    svd::Svd,
};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn shape_mismatch(expected: (usize, usize), got: (usize, usize)) -> Error {
    Error::mismatch(expected, got)
}

#[inline]
pub(crate) fn factorization_failed<T: fmt::Debug>(
    op: &'static str,
    shape: (usize, usize),
    err: T,
) -> Error {
    Error::invalid(format!("{op} failed for matrix {shape:?}: {err:?}"))
}

#[inline]
pub(crate) fn check_shape(expected: (usize, usize), got: (usize, usize)) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(shape_mismatch(expected, got))
    }
}

#[inline]
pub(crate) fn require_square(shape: (usize, usize), op: &'static str) -> Result<()> {
    if shape.0 == shape.1 {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "{op} requires square input: {shape:?}"
        )))
    }
}

// ---------------------------------------------------------------------------
// Side enum
// ---------------------------------------------------------------------------

/// Which triangle of a symmetric matrix to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Lower triangle.
    Lower,
    /// Upper triangle.
    Upper,
}

// ---------------------------------------------------------------------------
// Internal helpers — work directly on Tensor<T, Cpu>
// ---------------------------------------------------------------------------

/// Copy `a` into a fresh `Vec<f64>` row-major (for algorithm scratch space).
pub(crate) fn to_f64_buf(a: &Tensor<f64, Cpu>) -> Vec<f64> {
    a.as_slice().to_vec()
}

/// Build a `Tensor<f64, Cpu>` from a row-major flat buffer `(m x n)`.
pub(crate) fn from_f64_buf(buf: Vec<f64>, m: usize, n: usize) -> Tensor<f64, Cpu> {
    use nabla_core::backend::Backend;
    Tensor::from_storage(Cpu::from_vec(m, n, buf))
}

/// Read element from row-major flat buffer.
#[inline]
pub(crate) fn buf_get(buf: &[f64], cols: usize, r: usize, c: usize) -> f64 {
    buf[r * cols + c]
}

/// Write element to row-major flat buffer.
#[inline]
pub(crate) fn buf_set(buf: &mut [f64], cols: usize, r: usize, c: usize, v: f64) {
    buf[r * cols + c] = v;
}

// ---------------------------------------------------------------------------
// Shared helper functions
// ---------------------------------------------------------------------------

/// Read element from symmetric matrix `a` (row-major flat, n×n) respecting `side`.
#[inline]
pub(crate) fn get_sym(a: &Tensor<f64, Cpu>, i: usize, j: usize, side: Side) -> f64 {
    match side {
        Side::Lower => a.get(i, j),
        Side::Upper => a.get(j, i),
    }
}

/// Build full n×n symmetric buffer from `a`, filling both triangles.
pub(crate) fn symmetrize_to_buf(a: &Tensor<f64, Cpu>, n: usize, side: Side) -> Vec<f64> {
    let mut buf = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = match side {
                Side::Lower => {
                    if i >= j {
                        a.get(i, j)
                    } else {
                        a.get(j, i)
                    }
                }
                Side::Upper => {
                    if i <= j {
                        a.get(i, j)
                    } else {
                        a.get(j, i)
                    }
                }
            };
            buf[i * n + j] = v;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Householder helpers
// ---------------------------------------------------------------------------

/// Compute a Householder reflector in-place.
///
/// On entry `v` is a column (or row) vector. On exit `v` is modified so that
/// `H = I - tau * v * v^T` maps it to `[-sign(v[0])*||v||, 0, …, 0]`.
///
/// Returns `Some(tau)` (with `tau = 2 / ||v||²`) or `None` when the vector is
/// already zero (no reflection needed).
#[inline]
pub(crate) fn householder_vec(v: &mut [f64]) -> Option<f64> {
    let sigma: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if sigma < f64::EPSILON {
        return None;
    }
    let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
    v[0] += sign * sigma;
    let norm_sq: f64 = v.iter().map(|&x| x * x).sum();
    if norm_sq < f64::EPSILON {
        return None;
    }
    Some(2.0 / norm_sq)
}

/// Apply `H = I - tau * v * v^T` from the **left** to rows `row_off..row_off+len`
/// of `buf` (row-major, `ncols` columns), for columns `col_start..col_end`.
#[inline]
pub(crate) fn householder_apply_left(
    buf: &mut [f64],
    ncols: usize,
    row_off: usize,
    col_start: usize,
    col_end: usize,
    v: &[f64],
    tau: f64,
) {
    let len = v.len();
    for jj in col_start..col_end {
        let dot: f64 = (0..len)
            .map(|i| v[i] * buf[(i + row_off) * ncols + jj])
            .sum();
        let scale = tau * dot;
        for (i, &vi) in v.iter().enumerate().take(len) {
            buf[(i + row_off) * ncols + jj] -= scale * vi;
        }
    }
}

/// Apply `H = I - tau * v * v^T` from the **right** to columns `col_off..col_off+len`
/// of `buf` (row-major, `ncols` columns), for rows `row_start..row_end`.
#[inline]
pub(crate) fn householder_apply_right(
    buf: &mut [f64],
    ncols: usize,
    row_start: usize,
    row_end: usize,
    col_off: usize,
    v: &[f64],
    tau: f64,
) {
    let len = v.len();
    for ii in row_start..row_end {
        let dot: f64 = (0..len).map(|c| v[c] * buf[ii * ncols + c + col_off]).sum();
        let scale = tau * dot;
        for (c, &vc) in v.iter().enumerate().take(len) {
            buf[ii * ncols + c + col_off] -= scale * vc;
        }
    }
}

/// Split combined LU buffer into separate L (unit lower) and U (upper) buffers.
pub(crate) fn extract_lu(lu_buf: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    use core::cmp::Ordering;
    let mut l = vec![0.0f64; n * n];
    let mut u = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            match i.cmp(&j) {
                Ordering::Greater => l[i * n + j] = lu_buf[i * n + j],
                Ordering::Equal => {
                    l[i * n + j] = 1.0;
                    u[i * n + j] = lu_buf[i * n + j];
                }
                Ordering::Less => u[i * n + j] = lu_buf[i * n + j],
            }
        }
    }
    (l, u)
}

/// Compute L·U matrix product into a flat n×n buffer.
pub(crate) fn matmul_buf(l: &[f64], u: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let l_ik = l[i * n + k];
            if l_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += l_ik * u[k * n + j];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Macros for boilerplate reduction
// ---------------------------------------------------------------------------

/// Generate pub solve/reconstruct/inverse wrappers for factorization types.
///
/// `symmetric` variant: solve_transpose/solve_adjoint delegate to solve_impl (A = A^T).
/// `general` variant: solve_transpose rebuilds via reconstruct+refactor.
macro_rules! impl_factorization_methods {
    // Common methods shared by both symmetric and general factorizations.
    (@common $Type:ident) => {
        /// Solve `A·x = b`.
        #[must_use]
        pub fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
            self.solve_impl(rhs)
        }
        /// Solve in place.
        pub fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
            *rhs = self.solve_impl(rhs);
        }
        /// Reconstruct original matrix.
        #[must_use]
        pub fn reconstruct(&self) -> Tensor<f64, Cpu> {
            self.reconstruct_impl()
        }
        /// Compute matrix inverse.
        #[must_use]
        pub fn inverse(&self) -> Tensor<f64, Cpu> {
            self.inverse_impl()
        }
    };
    // Symmetric factorizations: A^T = A, so transpose/adjoint solve = regular solve.
    (symmetric $Type:ident) => {
        impl $Type<f64> {
            impl_factorization_methods!(@common $Type);
            /// Solve `A^T·x = b` (same as solve for symmetric A).
            #[must_use]
            pub fn solve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                self.solve_impl(rhs)
            }
            /// Solve `A^H·x = b` (same as solve for symmetric A).
            #[must_use]
            pub fn solve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                self.solve_impl(rhs)
            }
        }
    };
    // General (non-symmetric) factorizations with full solve variants.
    (general $Type:ident) => {
        impl $Type<f64> {
            impl_factorization_methods!(@common $Type);
            /// Solve `A^T·x = b`.
            #[must_use]
            pub fn solve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                let a = self.reconstruct_impl();
                let at = a.t();
                match $Type::factorize(&at) {
                    Ok(f) => f.solve_impl(rhs),
                    Err(_) => rhs.clone(),
                }
            }
            /// Solve in place `A^T·x = b`.
            pub fn solve_transpose_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
                *rhs = self.solve_transpose(rhs);
            }
            /// Solve `A^H·x = b` (same as transpose for real).
            #[must_use]
            pub fn solve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                self.solve_transpose(rhs)
            }
            /// Solve in place `A^H·x = b`.
            pub fn solve_adjoint_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
                *rhs = self.solve_adjoint(rhs);
            }
            /// Solve `x·A = b`.
            #[must_use]
            pub fn rsolve(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                let bt = rhs.t();
                let a = self.reconstruct_impl();
                let at = a.t();
                match $Type::factorize(&at) {
                    Ok(f) => f.solve_impl(&bt).t(),
                    Err(_) => rhs.clone(),
                }
            }
            /// Solve in place `x·A = b`.
            pub fn rsolve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
                *rhs = self.rsolve(rhs);
            }
        }
    };
}

/// Generate pub in-place solve wrappers for `Tensor<f64, Cpu>`.
/// Maps `$fn_name` to `$solve_method` with `?` propagation.
macro_rules! impl_solve_in_place {
    ($fn_name:ident, $solve_method:ident) => {
        /// # Errors
        /// Returns `Err` when dimensions mismatch or solve fails.
        fn $fn_name(&self, rhs: &mut Self) -> Result<()> {
            *rhs = self.$solve_method(rhs)?;
            Ok(())
        }
    };
}

/// Generate triangular in-place solve on `Tensor<f64, Cpu>`.
/// `$sub_fn` is `fwd_sub` or `bwd_sub`, `$unit` is `true`/`false`.
macro_rules! impl_triangular_solve_ip {
    ($fn_name:ident, $op:literal, $sub_fn:ident, $unit:literal) => {
        /// # Errors
        /// Returns `Err` when dimensions mismatch.
        fn $fn_name(&self, rhs: &mut Self) -> Result<()> {
            require_square(self.shape(), $op)?;
            check_shape((self.nrows(), rhs.ncols()), rhs.shape())?;
            let n = self.nrows();
            let n_rhs = rhs.ncols();
            let tri_buf = to_f64_buf(self);
            let mut x = to_f64_buf(rhs);
            for col in 0..n_rhs {
                $sub_fn(&tri_buf, n, &mut x, n_rhs, col, $unit);
            }
            *rhs = from_f64_buf(x, n, n_rhs);
            Ok(())
        }
    };
}

// ---------------------------------------------------------------------------
// Triangular solve helpers operating on flat f64 buffers
// ---------------------------------------------------------------------------

/// Forward substitution: solve L·x = b where L is lower triangular (unit or not).
/// b is column `rhs_col` of rhs matrix (`m_rhs` rows, `n_rhs` cols), result written back.
pub(crate) fn fwd_sub(
    l: &[f64],
    n: usize,
    rhs: &mut [f64],
    n_rhs: usize,
    rhs_col: usize,
    unit: bool,
) {
    for i in 0..n {
        let mut sum = buf_get(rhs, n_rhs, i, rhs_col);
        for j in 0..i {
            sum -= buf_get(l, n, i, j) * buf_get(rhs, n_rhs, j, rhs_col);
        }
        let diag = if unit { 1.0 } else { buf_get(l, n, i, i) };
        buf_set(rhs, n_rhs, i, rhs_col, sum / diag);
    }
}

/// Backward substitution: solve U·x = b where U is upper triangular (unit or not).
pub(crate) fn bwd_sub(
    u: &[f64],
    n: usize,
    rhs: &mut [f64],
    n_rhs: usize,
    rhs_col: usize,
    unit: bool,
) {
    for i in (0..n).rev() {
        let mut sum = buf_get(rhs, n_rhs, i, rhs_col);
        for j in (i + 1)..n {
            sum -= buf_get(u, n, i, j) * buf_get(rhs, n_rhs, j, rhs_col);
        }
        let diag = if unit { 1.0 } else { buf_get(u, n, i, i) };
        buf_set(rhs, n_rhs, i, rhs_col, sum / diag);
    }
}

// Make macros available to submodules.
pub(crate) use impl_factorization_methods;
pub(crate) use impl_solve_in_place;
pub(crate) use impl_triangular_solve_ip;
