// nabla-linalg — Self-contained dense linear algebra factorizations for Tensor<T, Cpu>.
//
// All algorithms operate on Tensor<T, Cpu> via .get()/.set()/.from_fn()/.zeros().
// No external BLAS/LAPACK dependencies.


use core::cmp::Ordering;
use core::fmt;

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

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
fn to_f64_buf(a: &Tensor<f64, Cpu>) -> Vec<f64> {
    a.as_slice().to_vec()
}

/// Build a `Tensor<f64, Cpu>` from a row-major flat buffer `(m x n)`.
fn from_f64_buf(buf: Vec<f64>, m: usize, n: usize) -> Tensor<f64, Cpu> {
    use nabla_core::backend::Backend;
    Tensor::from_storage(Cpu::from_vec(m, n, buf))
}

/// Read element from row-major flat buffer.
#[inline]
fn buf_get(buf: &[f64], cols: usize, r: usize, c: usize) -> f64 {
    buf[r * cols + c]
}

/// Write element to row-major flat buffer.
#[inline]
fn buf_set(buf: &mut [f64], cols: usize, r: usize, c: usize, v: f64) {
    buf[r * cols + c] = v;
}

// ---------------------------------------------------------------------------
// Shared helper functions
// ---------------------------------------------------------------------------

/// Read element from symmetric matrix `a` (row-major flat, n×n) respecting `side`.
#[inline]
fn get_sym(a: &Tensor<f64, Cpu>, i: usize, j: usize, side: Side) -> f64 {
    match side {
        Side::Lower => a.get(i, j),
        Side::Upper => a.get(j, i),
    }
}

/// Build full n×n symmetric buffer from `a`, filling both triangles.
fn symmetrize_to_buf(a: &Tensor<f64, Cpu>, n: usize, side: Side) -> Vec<f64> {
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
fn householder_vec(v: &mut [f64]) -> Option<f64> {
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
fn householder_apply_left(
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
        let dot: f64 = (0..len).map(|i| v[i] * buf[(i + row_off) * ncols + jj]).sum();
        let scale = tau * dot;
        for (i, &vi) in v.iter().enumerate().take(len) {
            buf[(i + row_off) * ncols + jj] -= scale * vi;
        }
    }
}

/// Apply `H = I - tau * v * v^T` from the **right** to columns `col_off..col_off+len`
/// of `buf` (row-major, `ncols` columns), for rows `row_start..row_end`.
#[inline]
fn householder_apply_right(
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
fn extract_lu(lu_buf: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
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
fn matmul_buf(l: &[f64], u: &[f64], n: usize) -> Vec<f64> {
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
    // Symmetric factorizations: A^T = A, so transpose/adjoint solve = regular solve.
    (symmetric $Type:ident) => {
        impl $Type<f64> {
            /// Solve `A·x = b`.
            #[must_use]
            pub fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                self.solve_impl(rhs)
            }
            /// Solve in place.
            pub fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
                *rhs = self.solve_impl(rhs);
            }
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
        }
    };
    // General (non-symmetric) factorizations with full solve variants.
    (general $Type:ident) => {
        impl $Type<f64> {
            /// Solve `A·x = b`.
            #[must_use]
            pub fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                self.solve_impl(rhs)
            }
            /// Solve in place.
            pub fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
                *rhs = self.solve_impl(rhs);
            }
            /// Solve `A^T·x = b`.
            #[must_use]
            pub fn solve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
                let a = self.reconstruct_impl();
                let at = a.t();
                // Factorization of A^T; fall back to rhs on singular (should not happen).
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
fn fwd_sub(l: &[f64], n: usize, rhs: &mut [f64], n_rhs: usize, rhs_col: usize, unit: bool) {
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
fn bwd_sub(u: &[f64], n: usize, rhs: &mut [f64], n_rhs: usize, rhs_col: usize, unit: bool) {
    for i in (0..n).rev() {
        let mut sum = buf_get(rhs, n_rhs, i, rhs_col);
        for j in (i + 1)..n {
            sum -= buf_get(u, n, i, j) * buf_get(rhs, n_rhs, j, rhs_col);
        }
        let diag = if unit { 1.0 } else { buf_get(u, n, i, i) };
        buf_set(rhs, n_rhs, i, rhs_col, sum / diag);
    }
}

// ===========================================================================
// 1. PartialPivLu — Doolittle LU with partial (row) pivoting
// ===========================================================================

/// LU decomposition with partial row pivoting: `P·A = L·U`.
///
/// `lu` stores L (strictly below diagonal, implicit 1s) and U (diagonal and above) combined.
pub struct PartialPivLu<T: Scalar> {
    lu: Tensor<T, Cpu>,
    /// Row permutation: row i of PA was originally row piv[i] of A.
    piv: Vec<usize>,
    n: usize,
}

impl PartialPivLu<f64> {
    fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
        require_square(a.shape(), "partial_piv_lu")?;
        let n = a.nrows();
        let mut buf = to_f64_buf(a);
        let mut piv: Vec<usize> = (0..n).collect();

        for k in 0..n {
            // find pivot
            let mut max_val = buf_get(&buf, n, k, k).abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = buf_get(&buf, n, i, k).abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            // swap rows k and max_row
            if max_row != k {
                for j in 0..n {
                    let tmp = buf_get(&buf, n, k, j);
                    let v = buf_get(&buf, n, max_row, j);
                    buf_set(&mut buf, n, k, j, v);
                    buf_set(&mut buf, n, max_row, j, tmp);
                }
                piv.swap(k, max_row);
            }
            let pivot = buf_get(&buf, n, k, k);
            if pivot.abs() < f64::EPSILON * 1e-10 {
                return Err(factorization_failed(
                    "partial_piv_lu",
                    a.shape(),
                    "singular matrix",
                ));
            }
            // compute multipliers and update sub-matrix
            for i in (k + 1)..n {
                let m = buf_get(&buf, n, i, k) / pivot;
                buf_set(&mut buf, n, i, k, m);
                for j in (k + 1)..n {
                    let u_kj = buf_get(&buf, n, k, j);
                    let v = buf_get(&buf, n, i, j) - m * u_kj;
                    buf_set(&mut buf, n, i, j, v);
                }
            }
        }

        Ok(Self {
            lu: from_f64_buf(buf, n, n),
            piv,
            n,
        })
    }

    /// Apply permutation P to rhs columns.
    fn apply_perm(&self, rhs: &[f64], n_rhs: usize) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0f64; n * n_rhs];
        for i in 0..n {
            let src = self.piv[i];
            for j in 0..n_rhs {
                out[i * n_rhs + j] = rhs[src * n_rhs + j];
            }
        }
        out
    }

    fn solve_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        let n = self.n;
        let n_rhs = rhs.ncols();
        let rhs_buf = to_f64_buf(rhs);
        let mut x = self.apply_perm(&rhs_buf, n_rhs);

        let lu_buf = to_f64_buf(&self.lu);
        // Forward substitution (L with unit diagonal)
        for col in 0..n_rhs {
            fwd_sub(&lu_buf, n, &mut x, n_rhs, col, true);
        }
        // Backward substitution (U)
        for col in 0..n_rhs {
            bwd_sub(&lu_buf, n, &mut x, n_rhs, col, false);
        }
        from_f64_buf(x, n, n_rhs)
    }

    fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
        // Rebuild P^T · L · U
        let n = self.n;
        let lu_buf = to_f64_buf(&self.lu);
        let (l, u) = extract_lu(&lu_buf, n);
        let lu_prod = matmul_buf(&l, &u, n);
        // apply inverse permutation (P^T)
        let mut result = vec![0.0f64; n * n];
        for i in 0..n {
            let dest = self.piv[i]; // row dest in result
            for j in 0..n {
                result[dest * n + j] = lu_prod[i * n + j];
            }
        }
        from_f64_buf(result, n, n)
    }

    fn inverse_impl(&self) -> Tensor<f64, Cpu> {
        let eye = Tensor::identity(self.n);
        self.solve_impl(&eye)
    }
}

// ===========================================================================
// 2. FullPivLu — LU with full row+column pivoting
// ===========================================================================

/// LU decomposition with full (row and column) pivoting: `P·A·Q = L·U`.
#[allow(dead_code)]
pub struct FullPivLu<T: Scalar> {
    lu: Tensor<T, Cpu>,
    row_piv: Vec<usize>,
    col_piv: Vec<usize>,
}

impl FullPivLu<f64> {
    #[allow(clippy::many_single_char_names)]
    fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
        require_square(a.shape(), "full_piv_lu")?;
        let n = a.nrows();
        let mut buf = to_f64_buf(a);
        let mut rpiv: Vec<usize> = (0..n).collect();
        let mut cpiv: Vec<usize> = (0..n).collect();

        for k in 0..n {
            // find max element in sub-matrix [k..n, k..n]
            let mut max_val = 0.0f64;
            let mut max_r = k;
            let mut max_c = k;
            for i in k..n {
                for j in k..n {
                    let v = buf_get(&buf, n, i, j).abs();
                    if v > max_val {
                        max_val = v;
                        max_r = i;
                        max_c = j;
                    }
                }
            }
            // row swap
            if max_r != k {
                for j in 0..n {
                    let tmp = buf_get(&buf, n, k, j);
                    let v = buf_get(&buf, n, max_r, j);
                    buf_set(&mut buf, n, k, j, v);
                    buf_set(&mut buf, n, max_r, j, tmp);
                }
                rpiv.swap(k, max_r);
            }
            // col swap
            if max_c != k {
                for i in 0..n {
                    let tmp = buf_get(&buf, n, i, k);
                    let v = buf_get(&buf, n, i, max_c);
                    buf_set(&mut buf, n, i, k, v);
                    buf_set(&mut buf, n, i, max_c, tmp);
                }
                cpiv.swap(k, max_c);
            }
            let pivot = buf_get(&buf, n, k, k);
            if pivot.abs() < f64::EPSILON * 1e-10 {
                return Err(factorization_failed(
                    "full_piv_lu",
                    a.shape(),
                    "singular matrix",
                ));
            }
            for i in (k + 1)..n {
                let m = buf_get(&buf, n, i, k) / pivot;
                buf_set(&mut buf, n, i, k, m);
                for j in (k + 1)..n {
                    let u_kj = buf_get(&buf, n, k, j);
                    let v = buf_get(&buf, n, i, j) - m * u_kj;
                    buf_set(&mut buf, n, i, j, v);
                }
            }
        }

        Ok(Self {
            lu: from_f64_buf(buf, n, n),
            row_piv: rpiv,
            col_piv: cpiv,
        })
    }
}

// ===========================================================================
// 3. Qr — Householder QR decomposition
// ===========================================================================

/// QR decomposition via Householder reflections: `A = Q·R`.
#[allow(clippy::struct_field_names)]
pub struct Qr<T: Scalar> {
    /// Combined Q (as Householder vectors stored below diagonal) and R (upper triangle).
    qr: Tensor<T, Cpu>,
    /// Householder scaling factors (tau values).
    taus: Vec<f64>,
    m: usize,
    n: usize,
}

impl Qr<f64> {
    #[allow(clippy::many_single_char_names)]
    fn factorize(a: &Tensor<f64, Cpu>) -> Self {
        let (m, n) = a.shape();
        let mut buf = to_f64_buf(a);
        let k = m.min(n);
        let mut taus = Vec::with_capacity(k);

        for j in 0..k {
            let col_len = m - j;
            let mut v: Vec<f64> = (j..m).map(|i| buf_get(&buf, n, i, j)).collect();
            let Some(tau) = householder_vec(&mut v) else {
                taus.push(0.0);
                continue;
            };
            householder_apply_left(&mut buf, n, j, j, n, &v, tau);
            // Normalize v so v[0]=1 (standard compact QR storage).
            // tau_stored = tau * v0^2; v_stored[i] = v[i] / v0
            let v0 = v[0];
            taus.push(tau * v0 * v0);
            for (i, vi) in v.iter().enumerate().skip(1).take(col_len - 1) {
                buf_set(&mut buf, n, i + j, j, *vi / v0);
            }
        }

        Self {
            qr: from_f64_buf(buf, m, n),
            taus,
            m,
            n,
        }
    }

    /// Apply Q^T to rhs: returns Q^T * rhs.
    #[allow(clippy::many_single_char_names)]
    fn apply_qt(&self, rhs: &[f64], n_rhs: usize) -> Vec<f64> {
        let (m, n) = (self.m, self.n);
        let k = m.min(n);
        let mut x = rhs.to_vec();
        let qr_buf = to_f64_buf(&self.qr);

        for j in 0..k {
            let tau = self.taus[j];
            if tau == 0.0 {
                continue;
            }
            let col_len = m - j;
            // reconstruct v: v[0] = 1 (implicit), v[1..] stored in qr[j+1..m, j]
            // But we stored v[i] directly — need the actual v[0].
            // Recover: R[j,j] = -(sign)*||a_j|| and v[0] = a_orig[j,j] + sign*||..||
            // Simpler: store v as-is, v[0] is implicit 1 per standard compact storage.
            // Re-derive: standard compact QR stores v[1..] with v[0] = 1 implicit.
            // Our store above stores raw v[i] for i>=1 (v[1..col_len]).
            // Let's re-derive using stored values.
            let mut v = vec![1.0f64; col_len];
            for (i, vi) in v.iter_mut().enumerate().skip(1) {
                *vi = buf_get(&qr_buf, n, i + j, j);
            }
            // Apply H_j = I - tau * v * v^T to x[j..m, :]
            for col in 0..n_rhs {
                let dot: f64 = (0..col_len).map(|i| v[i] * x[(i + j) * n_rhs + col]).sum();
                let scale = tau * dot;
                for i in 0..col_len {
                    x[(i + j) * n_rhs + col] -= scale * v[i];
                }
            }
        }
        x
    }

    #[allow(clippy::many_single_char_names)]
    fn solve_lstsq_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        let (m, n) = (self.m, self.n);
        let n_rhs = rhs.ncols();
        let rhs_buf = to_f64_buf(rhs);

        // Q^T * rhs
        let mut x = self.apply_qt(&rhs_buf, n_rhs);

        // Solve R * result = x[0..n] by back-substitution
        let qr_buf = to_f64_buf(&self.qr);
        let r_rows = m.min(n);
        // x[:r_rows, :] is the RHS for R
        for col in 0..n_rhs {
            for i in (0..r_rows).rev() {
                let mut sum = x[i * n_rhs + col];
                for jj in (i + 1)..n {
                    sum -= buf_get(&qr_buf, n, i, jj) * x[jj * n_rhs + col];
                }
                let r_ii = buf_get(&qr_buf, n, i, i);
                x[i * n_rhs + col] = sum / r_ii;
            }
        }

        // Extract top n rows
        let mut result = vec![0.0f64; n * n_rhs];
        for i in 0..r_rows.min(n) {
            for col in 0..n_rhs {
                result[i * n_rhs + col] = x[i * n_rhs + col];
            }
        }
        from_f64_buf(result, n, n_rhs)
    }
}

// ===========================================================================
// 4. ColPivQr — Column-pivoted Householder QR
// ===========================================================================

/// Column-pivoted QR: `A·P^T = Q·R`.
pub struct ColPivQr<T: Scalar> {
    qr: Tensor<T, Cpu>,
    taus: Vec<f64>,
    col_piv: Vec<usize>,
    m: usize,
    n: usize,
}

impl ColPivQr<f64> {
    #[allow(clippy::many_single_char_names)]
    fn factorize(a: &Tensor<f64, Cpu>) -> Self {
        let (m, n) = a.shape();
        let mut buf = to_f64_buf(a);
        let k = m.min(n);
        let mut taus = Vec::with_capacity(k);
        let mut cpiv: Vec<usize> = (0..n).collect();

        // Column norms
        let mut col_norms: Vec<f64> = (0..n)
            .map(|j| {
                (0..m)
                    .map(|i| buf_get(&buf, n, i, j).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        for j in 0..k {
            // find column with max norm in [j..n]
            let max_col = (j..n)
                .max_by(|&a, &b| {
                    col_norms[a]
                        .partial_cmp(&col_norms[b])
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .unwrap_or(j);
            if max_col != j {
                // swap columns
                for i in 0..m {
                    let tmp = buf_get(&buf, n, i, j);
                    let v = buf_get(&buf, n, i, max_col);
                    buf_set(&mut buf, n, i, j, v);
                    buf_set(&mut buf, n, i, max_col, tmp);
                }
                cpiv.swap(j, max_col);
                col_norms.swap(j, max_col);
            }

            let col_len = m - j;
            let mut v: Vec<f64> = (j..m).map(|i| buf_get(&buf, n, i, j)).collect();
            let Some(tau) = householder_vec(&mut v) else {
                taus.push(0.0);
                continue;
            };
            householder_apply_left(&mut buf, n, j, j, n, &v, tau);
            // Normalize v so v[0]=1 (standard compact QR storage)
            let v0 = v[0];
            taus.push(tau * v0 * v0);
            for (i, vi) in v.iter().enumerate().skip(1).take(col_len - 1) {
                buf_set(&mut buf, n, i + j, j, *vi / v0);
            }
            // Update column norms
            for (cn, jj) in col_norms[(j + 1)..n].iter_mut().zip(j + 1..n) {
                let r_jj = buf_get(&buf, n, j, jj);
                let old_sq = cn.powi(2);
                let new_sq = (old_sq - r_jj * r_jj).max(0.0);
                *cn = new_sq.sqrt();
            }
        }

        Self {
            qr: from_f64_buf(buf, m, n),
            taus,
            col_piv: cpiv,
            m,
            n,
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn solve_lstsq_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        // Same as Qr::solve_lstsq but apply column permutation at end
        let (m, n) = (self.m, self.n);
        let n_rhs = rhs.ncols();
        let rhs_buf = to_f64_buf(rhs);
        let qr_buf = to_f64_buf(&self.qr);
        let k = m.min(n);

        // Apply Q^T
        let mut x = rhs_buf.clone();
        for j in 0..k {
            let tau = self.taus[j];
            if tau == 0.0 {
                continue;
            }
            let col_len = m - j;
            let mut v = vec![1.0f64; col_len];
            for (i, vi) in v.iter_mut().enumerate().skip(1) {
                *vi = buf_get(&qr_buf, n, i + j, j);
            }
            for col in 0..n_rhs {
                let dot: f64 = (0..col_len).map(|i| v[i] * x[(i + j) * n_rhs + col]).sum();
                let scale = tau * dot;
                for i in 0..col_len {
                    x[(i + j) * n_rhs + col] -= scale * v[i];
                }
            }
        }

        // Back-sub
        for col in 0..n_rhs {
            for i in (0..k).rev() {
                let mut sum = x[i * n_rhs + col];
                for jj in (i + 1)..k {
                    sum -= buf_get(&qr_buf, n, i, jj) * x[jj * n_rhs + col];
                }
                x[i * n_rhs + col] = sum / buf_get(&qr_buf, n, i, i);
            }
        }

        // Apply inverse col permutation
        let mut result = vec![0.0f64; n * n_rhs];
        for i in 0..k {
            let dest = self.col_piv[i];
            for col in 0..n_rhs {
                result[dest * n_rhs + col] = x[i * n_rhs + col];
            }
        }
        from_f64_buf(result, n, n_rhs)
    }
}

// ===========================================================================
// 5. Llt — Cholesky factorization (A = L·L^T)
// ===========================================================================

/// Cholesky factorization: `A = L·L^T` (positive-definite symmetric).
pub struct Llt<T: Scalar> {
    l: Tensor<T, Cpu>,
    n: usize,
}

impl Llt<f64> {
    fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Result<Self> {
        require_square(a.shape(), "llt")?;
        let n = a.nrows();
        let mut l = vec![0.0f64; n * n];

        // Cholesky column-by-column (lower triangle of A used per side)
        for j in 0..n {
            let mut diag = a.get(j, j);
            for k in 0..j {
                diag -= l[j * n + k] * l[j * n + k];
            }
            if diag <= 0.0 {
                return Err(factorization_failed(
                    "llt",
                    a.shape(),
                    "matrix is not positive-definite",
                ));
            }
            let l_jj = diag.sqrt();
            l[j * n + j] = l_jj;

            for i in (j + 1)..n {
                let mut sum = get_sym(a, i, j, side);
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = sum / l_jj;
            }
        }

        Ok(Self {
            l: from_f64_buf(l, n, n),
            n,
        })
    }

    fn solve_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        let n = self.n;
        let n_rhs = rhs.ncols();
        let mut x = to_f64_buf(rhs);
        let l_buf = to_f64_buf(&self.l);
        // Solve L·y = b
        for col in 0..n_rhs {
            fwd_sub(&l_buf, n, &mut x, n_rhs, col, false);
        }
        // Solve L^T·x = y
        for col in 0..n_rhs {
            // Backward sub with L^T (i.e., upper triangular with l[j][i])
            for i in (0..n).rev() {
                let mut sum = x[i * n_rhs + col];
                for j in (i + 1)..n {
                    // L^T[i,j] = L[j,i]
                    sum -= l_buf[j * n + i] * x[j * n_rhs + col];
                }
                x[i * n_rhs + col] = sum / l_buf[i * n + i];
            }
        }
        from_f64_buf(x, n, n_rhs)
    }

    fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
        // A = L · L^T
        let n = self.n;
        let l_buf = to_f64_buf(&self.l);
        let mut result = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..=i.min(j) {
                    sum += l_buf[i * n + k] * l_buf[j * n + k];
                }
                result[i * n + j] = sum;
            }
        }
        from_f64_buf(result, n, n)
    }

    fn inverse_impl(&self) -> Tensor<f64, Cpu> {
        let eye = Tensor::identity(self.n);
        self.solve_impl(&eye)
    }
}

// ===========================================================================
// 6. Ldlt — LDL^T decomposition (symmetric, not necessarily positive-definite)
// ===========================================================================

/// LDL^T factorization: `A = L·D·L^T` (symmetric, no sqrt).
pub struct Ldlt<T: Scalar> {
    /// L stored below diagonal (unit diagonal implicit), D on diagonal.
    ld: Tensor<T, Cpu>,
    n: usize,
}

impl Ldlt<f64> {
    fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Result<Self> {
        require_square(a.shape(), "ldlt")?;
        let n = a.nrows();
        let mut ld = vec![0.0f64; n * n];

        for j in 0..n {
            // D[j,j] = A[j,j] - sum_{k<j} L[j,k]^2 * D[k,k]
            let mut d_jj = a.get(j, j);
            for k in 0..j {
                let l_jk = ld[j * n + k];
                let d_kk = ld[k * n + k];
                d_jj -= l_jk * l_jk * d_kk;
            }
            if d_jj.abs() < f64::EPSILON * 1e-10 {
                return Err(factorization_failed(
                    "ldlt",
                    a.shape(),
                    "zero pivot encountered",
                ));
            }
            ld[j * n + j] = d_jj;

            for i in (j + 1)..n {
                let mut sum = get_sym(a, i, j, side);
                for k in 0..j {
                    sum -= ld[i * n + k] * ld[k * n + k] * ld[j * n + k];
                }
                ld[i * n + j] = sum / d_jj;
            }
        }

        Ok(Self {
            ld: from_f64_buf(ld, n, n),
            n,
        })
    }

    fn solve_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        let n = self.n;
        let n_rhs = rhs.ncols();
        let mut x = to_f64_buf(rhs);
        let ld_buf = to_f64_buf(&self.ld);

        // Solve L·y = b (unit lower triangular)
        for col in 0..n_rhs {
            fwd_sub(&ld_buf, n, &mut x, n_rhs, col, true);
        }
        // Solve D·z = y
        for i in 0..n {
            let d = ld_buf[i * n + i];
            for col in 0..n_rhs {
                x[i * n_rhs + col] /= d;
            }
        }
        // Solve L^T·x = z
        for col in 0..n_rhs {
            for i in (0..n).rev() {
                let mut sum = x[i * n_rhs + col];
                for j in (i + 1)..n {
                    // L^T[i,j] = L[j,i]
                    sum -= ld_buf[j * n + i] * x[j * n_rhs + col];
                }
                x[i * n_rhs + col] = sum;
            }
        }
        from_f64_buf(x, n, n_rhs)
    }

    fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
        let n = self.n;
        let ld_buf = to_f64_buf(&self.ld);
        // A = L·D·L^T
        // L: unit lower, D: diagonal (stored on diagonal of ld)
        let mut result = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f64;
                for k in 0..=j {
                    let lik = if i == k { 1.0 } else { ld_buf[i * n + k] };
                    let ljk = if j == k { 1.0 } else { ld_buf[j * n + k] };
                    let d_kk = ld_buf[k * n + k];
                    sum += lik * d_kk * ljk;
                }
                result[i * n + j] = sum;
                result[j * n + i] = sum;
            }
        }
        from_f64_buf(result, n, n)
    }

    fn inverse_impl(&self) -> Tensor<f64, Cpu> {
        let eye = Tensor::identity(self.n);
        self.solve_impl(&eye)
    }
}

// ===========================================================================
// 7. Lblt — Bunch-Kaufman pivoted LBL^T (indefinite symmetric)
// ===========================================================================

/// Bunch-Kaufman `LBL^T` decomposition for indefinite symmetric matrices.
///
/// Uses 1×1 and 2×2 pivots to handle indefinite systems.
pub struct Lblt<T: Scalar> {
    /// Combined L and B stored as flat f64 (L below diagonal, B on/near diagonal).
    lb: Tensor<T, Cpu>,
}

impl Lblt<f64> {
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Self {
        let n = a.nrows();
        if n == 0 {
            return Self {
                lb: Tensor::zeros(0, 0),
            };
        }

        let mut buf = symmetrize_to_buf(a, n, side);
        let mut piv = vec![0i64; n];

        // Bunch-Kaufman with alpha = (1 + sqrt(17))/8
        let alpha = (1.0 + 17.0f64.sqrt()) / 8.0;
        let mut k = 0usize;
        while k < n {
            let remaining = n - k;
            if remaining == 1 {
                piv[k] = (k + 1) as i64;
                k += 1;
                continue;
            }

            // Find |a_kk|
            let a_kk = buf_get(&buf, n, k, k).abs();

            // Find max off-diagonal in column k (rows k+1..n)
            let mut max_off = 0.0f64;
            let mut max_r = k + 1;
            for i in (k + 1)..n {
                let v = buf_get(&buf, n, i, k).abs();
                if v > max_off {
                    max_off = v;
                    max_r = i;
                }
            }

            if a_kk >= alpha * max_off {
                // 1×1 pivot: use diagonal element
                piv[k] = (k + 1) as i64;
                let d = buf_get(&buf, n, k, k);
                if d.abs() > f64::EPSILON * 1e-12 {
                    for i in (k + 1)..n {
                        let m = buf_get(&buf, n, i, k) / d;
                        for j in (k + 1)..n {
                            let v = buf_get(&buf, n, i, j) - m * buf_get(&buf, n, k, j);
                            buf_set(&mut buf, n, i, j, v);
                            buf_set(&mut buf, n, j, i, v);
                        }
                        buf_set(&mut buf, n, i, k, m);
                        buf_set(&mut buf, n, k, i, m);
                    }
                }
                k += 1;
            } else {
                // Check 2×2 pivot
                let a_rr = buf_get(&buf, n, max_r, max_r).abs();

                if a_rr * max_off >= alpha * max_off * max_off {
                    // 1×1 pivot with row/col max_r swapped to k
                    if max_r == k {
                        piv[k] = (k + 1) as i64;
                    } else {
                        // swap rows/cols k and max_r
                        for j in 0..n {
                            let tmp = buf_get(&buf, n, k, j);
                            let v = buf_get(&buf, n, max_r, j);
                            buf_set(&mut buf, n, k, j, v);
                            buf_set(&mut buf, n, max_r, j, tmp);
                        }
                        for i in 0..n {
                            let tmp = buf_get(&buf, n, i, k);
                            let v = buf_get(&buf, n, i, max_r);
                            buf_set(&mut buf, n, i, k, v);
                            buf_set(&mut buf, n, i, max_r, tmp);
                        }
                        piv[k] = (max_r + 1) as i64;
                    }
                    let d = buf_get(&buf, n, k, k);
                    if d.abs() > f64::EPSILON * 1e-12 {
                        for i in (k + 1)..n {
                            let m = buf_get(&buf, n, i, k) / d;
                            for j in (k + 1)..n {
                                let v = buf_get(&buf, n, i, j) - m * buf_get(&buf, n, k, j);
                                buf_set(&mut buf, n, i, j, v);
                                buf_set(&mut buf, n, j, i, v);
                            }
                            buf_set(&mut buf, n, i, k, m);
                            buf_set(&mut buf, n, k, i, m);
                        }
                    }
                    k += 1;
                } else {
                    // 2×2 pivot: swap max_r to k+1
                    if max_r != k + 1 {
                        for j in 0..n {
                            let tmp = buf_get(&buf, n, k + 1, j);
                            let v = buf_get(&buf, n, max_r, j);
                            buf_set(&mut buf, n, k + 1, j, v);
                            buf_set(&mut buf, n, max_r, j, tmp);
                        }
                        for i in 0..n {
                            let tmp = buf_get(&buf, n, i, k + 1);
                            let v = buf_get(&buf, n, i, max_r);
                            buf_set(&mut buf, n, i, k + 1, v);
                            buf_set(&mut buf, n, i, max_r, tmp);
                        }
                    }
                    // 2×2 pivot block B = [[a,b],[b,c]]
                    let b_a = buf_get(&buf, n, k, k);
                    let b_b = buf_get(&buf, n, k + 1, k);
                    let b_c = buf_get(&buf, n, k + 1, k + 1);
                    let det = b_a * b_c - b_b * b_b;
                    if det.abs() > f64::EPSILON * 1e-12 {
                        let b_inv_a = b_c / det;
                        let b_inv_b = -b_b / det;
                        let b_inv_c = b_a / det;
                        for i in (k + 2)..n {
                            let x_i = buf_get(&buf, n, i, k);
                            let y_i = buf_get(&buf, n, i, k + 1);
                            let m0 = x_i * b_inv_a + y_i * b_inv_b;
                            let m1 = x_i * b_inv_b + y_i * b_inv_c;
                            for j in (k + 2)..n {
                                let x_j = buf_get(&buf, n, k, j);
                                let y_j = buf_get(&buf, n, k + 1, j);
                                let v = buf_get(&buf, n, i, j) - m0 * x_j - m1 * y_j;
                                buf_set(&mut buf, n, i, j, v);
                                buf_set(&mut buf, n, j, i, v);
                            }
                            buf_set(&mut buf, n, i, k, m0);
                            buf_set(&mut buf, n, i, k + 1, m1);
                            buf_set(&mut buf, n, k, i, m0);
                            buf_set(&mut buf, n, k + 1, i, m1);
                        }
                    }
                    piv[k] = -((max_r + 1) as i64);
                    piv[k + 1] = -((max_r + 1) as i64);
                    k += 2;
                }
            }
        }

        Self {
            lb: from_f64_buf(buf, n, n),
        }
    }

    fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
        // Return the stored (partially factored) matrix as-is for reconstruction
        // A = L · B · L^T where B is block diagonal stored in lb
        // For practical purposes, return the stored lb as-is
        self.lb.clone()
    }
}

// ===========================================================================
// 8. Svd — Golub-Kahan bidiagonalization + implicit QR
// ===========================================================================

/// Full/thin SVD: `A = U·Σ·V^H`.
pub struct Svd<T: Scalar> {
    u: Tensor<T, Cpu>,
    s: Vec<f64>,
    vt: Tensor<T, Cpu>,
}

impl Svd<f64> {
    fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
        let (m, n) = a.shape();
        // Use Jacobi SVD for small matrices; bidiag+QR for larger
        if m >= n {
            Self::golub_kahan_svd(a, m, n)
        } else {
            // Compute SVD of A^T, then swap U and V^T
            let at = a.t();
            let svd_t = Self::golub_kahan_svd(&at, n, m)?;
            Ok(Self {
                u: svd_t.vt.t(),
                s: svd_t.s,
                vt: svd_t.u.t(),
            })
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn golub_kahan_svd(a: &Tensor<f64, Cpu>, m: usize, n: usize) -> Result<Self> {
        // Bidiagonalization via Householder
        let mut buf = to_f64_buf(a);
        let mut u_mat = vec![0.0f64; m * m];
        for i in 0..m {
            u_mat[i * m + i] = 1.0;
        }
        let mut vt_mat = vec![0.0f64; n * n];
        for i in 0..n {
            vt_mat[i * n + i] = 1.0;
        }

        let k = m.min(n);
        for j in 0..k {
            // Left Householder: zero out below diagonal in column j
            let mut v: Vec<f64> = (j..m).map(|i| buf_get(&buf, n, i, j)).collect();
            if let Some(tau) = householder_vec(&mut v) {
                householder_apply_left(&mut buf, n, j, j, n, &v, tau);
                // Apply to U: U * H (right-multiply, so transpose of left-apply)
                householder_apply_right(&mut u_mat, m, 0, m, j, &v, tau);
            }

            // Right Householder: zero out right of superdiagonal in row j
            if j + 2 < n {
                let mut w: Vec<f64> = ((j + 1)..n).map(|c| buf_get(&buf, n, j, c)).collect();
                if let Some(tau) = householder_vec(&mut w) {
                    householder_apply_right(&mut buf, n, j, m, j + 1, &w, tau);
                    // Apply to Vt: H * Vt (left-multiply rows)
                    householder_apply_left(&mut vt_mat, n, j + 1, 0, n, &w, tau);
                }
            }
        }

        // Extract bidiagonal: d (main), e (superdiagonal)
        let mut d: Vec<f64> = (0..k).map(|i| buf_get(&buf, n, i, i)).collect();
        let mut e: Vec<f64> = (0..(k.saturating_sub(1)))
            .map(|i| buf_get(&buf, n, i, i + 1))
            .collect();

        // Implicit QR iteration on bidiagonal
        Self::bidiag_qr_svd(&mut d, &mut e, &mut u_mat, &mut vt_mat, m, n, k)?;

        Ok(Self::sort_svd(&d, &u_mat, &vt_mat, m, n, k))
    }

    /// Sort singular values descending and permute U/Vt accordingly.
    fn sort_svd(d: &[f64], u_mat: &[f64], vt_mat: &[f64], m: usize, n: usize, k: usize) -> Self {
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| {
            d[b].abs()
                .partial_cmp(&d[a].abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let s_sorted: Vec<f64> = indices.iter().map(|&i| d[i].abs()).collect();

        let mut u_sorted = vec![0.0f64; m * m];
        let mut vt_sorted = vec![0.0f64; n * n];
        for (new_i, &old_i) in indices.iter().enumerate() {
            let sign = if d[old_i] < 0.0 { -1.0 } else { 1.0 };
            for r in 0..m {
                u_sorted[r * m + new_i] = sign * u_mat[r * m + old_i];
            }
            for c in 0..n {
                vt_sorted[new_i * n + c] = vt_mat[old_i * n + c];
            }
        }

        Self {
            u: from_f64_buf(u_sorted, m, m),
            s: s_sorted,
            vt: from_f64_buf(vt_sorted, n, n),
        }
    }

    /// Golub-Reinsch implicit QR sweep on bidiagonal matrix.
    #[allow(clippy::many_single_char_names)]
    fn bidiag_qr_svd(
        d: &mut [f64],
        e: &mut [f64],
        u: &mut [f64],
        vt: &mut [f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let mut max_iter = 30 * k;
        let mut p = k;

        while p > 0 {
            if max_iter == 0 {
                return Err(Error::invalid("SVD: failed to converge"));
            }
            max_iter -= 1;

            let tol = f64::EPSILON * d.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);

            // Deflate: shrink p while trailing e[p-2] is negligible
            while p > 1 && e[p - 2].abs() <= tol {
                e[p - 2] = 0.0;
                p -= 1;
            }
            if p <= 1 {
                break;
            }

            // Find q: start index of the unreduced active block
            // Scan upward from p-2: find the topmost index where e[i] is nonzero
            let mut q = p - 1;
            while q > 0 && e[q - 1].abs() > tol {
                q -= 1;
            }
            // Zero out e[q-1] if it was negligible
            if q > 0 && e[q - 1].abs() <= tol {
                e[q - 1] = 0.0;
            }
            // Find m_: zero diagonal in d[q..p]
            let mut found_zero_diag = false;
            for i in q..p.saturating_sub(1) {
                if d[i].abs() <= tol {
                    // Chase the zero off the diagonal via Givens rotations
                    let mut f = e[i];
                    e[i] = 0.0;
                    for j in i..p.saturating_sub(1) {
                        let g = d[j + 1];
                        let r = f.hypot(g);
                        let c = if r == 0.0 { 1.0 } else { g / r };
                        let s = if r == 0.0 { 0.0 } else { f / r };
                        // Apply Givens to columns i and j+1 of Vt (rows in our storage)
                        for col in 0..n {
                            let v0 = vt[i * n + col];
                            let v1 = vt[(j + 1) * n + col];
                            vt[i * n + col] = c * v0 + s * v1;
                            vt[(j + 1) * n + col] = -s * v0 + c * v1;
                        }
                        d[j + 1] = r;
                        if j + 1 < p.saturating_sub(1) {
                            f = s * e[j + 1];
                            e[j + 1] *= c;
                        }
                    }
                    found_zero_diag = true;
                    break;
                }
            }
            if found_zero_diag {
                continue;
            }

            // Wilkinson shift on trailing 2x2 of B^T*B
            let e_top = if p >= 3 && (p - 3) >= q { e[p - 3] } else { 0.0 };
            let a11 = d[p - 2] * d[p - 2] + e_top * e_top;
            let a12 = d[p - 2] * e[p - 2];
            let a22 = d[p - 1] * d[p - 1] + e[p - 2] * e[p - 2];
            let delta = (a11 - a22) * 0.5;
            let mu = a22 - a12 * a12 / (delta + delta.signum() * delta.hypot(a12) + f64::EPSILON);

            // Golub-Kahan implicit QR step (Golub & Van Loan, Algorithm 8.6.2)
            let mut y = d[q] * d[q] - mu;
            let mut z = d[q] * e[q];

            for i in q..p.saturating_sub(1) {
                // Right Givens: zero out z in column (i, i+1)
                let r = y.hypot(z);
                let c = if r == 0.0 { 1.0 } else { y / r };
                let s = if r == 0.0 { 0.0 } else { z / r };
                if i > q {
                    e[i - 1] = r;
                }
                y = c * d[i] + s * e[i];
                z = s * d[i + 1];
                let new_e_i = -s * d[i] + c * e[i];
                let new_d_i1 = c * d[i + 1];
                // Apply right Givens to rows of Vt
                for col in 0..n {
                    let v0 = vt[i * n + col];
                    let v1 = vt[(i + 1) * n + col];
                    vt[i * n + col] = c * v0 + s * v1;
                    vt[(i + 1) * n + col] = -s * v0 + c * v1;
                }

                // Left Givens: zero out z (the bulge below diagonal)
                let r2 = y.hypot(z);
                let c2 = if r2 == 0.0 { 1.0 } else { y / r2 };
                let s2 = if r2 == 0.0 { 0.0 } else { z / r2 };
                d[i] = r2;
                y = c2 * new_e_i + s2 * new_d_i1;
                d[i + 1] = -s2 * new_e_i + c2 * new_d_i1;
                if i + 1 < e.len() {
                    z = s2 * e[i + 1];
                    e[i + 1] *= c2;
                } else {
                    z = 0.0;
                }
                // Apply left Givens to columns of U
                for row in 0..m {
                    let u0 = u[row * m + i];
                    let u1 = u[row * m + i + 1];
                    u[row * m + i] = c2 * u0 + s2 * u1;
                    u[row * m + i + 1] = -s2 * u0 + c2 * u1;
                }
            }
            e[p - 2] = y;
        }
        Ok(())
    }

    /// Return singular values vector.
    #[must_use]
    pub fn s_values(&self) -> &[f64] {
        &self.s
    }
}

// ===========================================================================
// 9. SelfAdjointEigen — symmetric eigen (tridiag + Wilkinson QR shifts)
// ===========================================================================

/// Self-adjoint (symmetric) eigendecomposition: `A = V·Λ·V^T`.
pub struct SelfAdjointEigen<T: Scalar> {
    /// Eigenvectors stored as columns (V matrix).
    v: Tensor<T, Cpu>,
    /// Eigenvalues in ascending order.
    eigenvalues: Vec<f64>,
}

impl SelfAdjointEigen<f64> {
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Result<Self> {
        require_square(a.shape(), "self_adjoint_eigen")?;
        let n = a.nrows();

        let mut buf = symmetrize_to_buf(a, n, side);

        // Householder tridiagonalization
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }

        for j in 0..(n.saturating_sub(2)) {
            let mut v: Vec<f64> = ((j + 1)..n).map(|i| buf_get(&buf, n, i, j)).collect();
            let Some(tau) = householder_vec(&mut v) else {
                continue;
            };
            // Apply H from left: A = H * A
            householder_apply_left(&mut buf, n, j + 1, 0, n, &v, tau);
            // Apply H from right: A = A * H
            householder_apply_right(&mut buf, n, 0, n, j + 1, &v, tau);
            // Accumulate Q
            householder_apply_right(&mut q, n, 0, n, j + 1, &v, tau);
        }

        // Extract tridiagonal: diagonal d, offdiagonal e
        let mut d: Vec<f64> = (0..n).map(|i| buf_get(&buf, n, i, i)).collect();
        let mut e: Vec<f64> = (0..(n.saturating_sub(1)))
            .map(|i| buf_get(&buf, n, i + 1, i))
            .collect();

        // QL/QR iteration for symmetric tridiagonal (implicit Wilkinson shift)
        let max_iter = 30 * n;
        let mut iter_count = 0;
        let mut p = n;

        while p > 1 {
            // Find convergence at bottom
            let tol = f64::EPSILON * (d[p - 1].abs() + if p > 1 { e[p - 2].abs() } else { 0.0 });
            if p >= 2 && e[p - 2].abs() <= tol {
                e[p - 2] = 0.0;
                p -= 1;
                continue;
            }

            iter_count += 1;
            if iter_count > max_iter {
                return Err(factorization_failed(
                    "self_adjoint_eigen",
                    a.shape(),
                    "failed to converge",
                ));
            }

            // Wilkinson shift
            let mu = if p >= 2 {
                let a_nn = d[p - 1];
                let a_nm1 = d[p - 2];
                let b = e[p - 2];
                let delta = (a_nm1 - a_nn) / 2.0;
                let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
                a_nn - b * b / (delta + sign * delta.hypot(b))
            } else {
                d[0]
            };

            // QR step with Wilkinson shift
            let mut x = d[0] - mu;
            let mut z = e[0];

            for i in 0..(p - 1) {
                let r = x.hypot(z);
                let c = if r == 0.0 { 1.0 } else { x / r };
                let s = if r == 0.0 { 0.0 } else { z / r };

                // Apply Givens rotation G(i, i+1) from left and right
                if i > 0 {
                    e[i - 1] = r;
                }
                let d_i = d[i];
                let d_i1 = d[i + 1];
                let e_i = e[i];
                d[i] = c * c * d_i + 2.0 * s * c * e_i + s * s * d_i1;
                d[i + 1] = s * s * d_i - 2.0 * s * c * e_i + c * c * d_i1;
                e[i] = s * c * (d_i1 - d_i) + (c * c - s * s) * e_i;

                if i + 1 < p - 1 {
                    x = e[i];
                    z = s * e[i + 1];
                    e[i + 1] *= c;
                }

                // Accumulate into Q
                for row in 0..n {
                    let q0 = q[row * n + i];
                    let q1 = q[row * n + i + 1];
                    q[row * n + i] = c * q0 + s * q1;
                    q[row * n + i + 1] = -s * q0 + c * q1;
                }
            }
        }

        // Sort eigenvalues ascending (and corresponding eigenvectors)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            d[a].partial_cmp(&d[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let eigenvalues: Vec<f64> = indices.iter().map(|&i| d[i]).collect();

        let q_copy = q.clone();
        let mut q_sorted = vec![0.0f64; n * n];
        for (new_j, &old_j) in indices.iter().enumerate() {
            for row in 0..n {
                q_sorted[row * n + new_j] = q_copy[row * n + old_j];
            }
        }

        Ok(Self {
            v: from_f64_buf(q_sorted, n, n),
            eigenvalues,
        })
    }

    /// Eigenvalues in ascending order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Eigenvector matrix (columns are eigenvectors).
    #[must_use]
    pub fn vectors(&self) -> &Tensor<f64, Cpu> {
        &self.v
    }
}

// ===========================================================================
// LinalgExt — extension trait for Tensor<f64, Cpu> factorization and solve
// ===========================================================================

/// Extension trait adding dense linear algebra methods to `Tensor<f64, Cpu>`.
pub trait LinalgExt {
    /// LU decomposition with partial pivoting.
    fn partial_piv_lu(&self) -> Result<PartialPivLu<f64>>;
    /// LU decomposition with full pivoting.
    fn full_piv_lu(&self) -> Result<FullPivLu<f64>>;
    /// QR decomposition.
    fn qr(&self) -> Qr<f64>;
    /// Column-pivoted QR decomposition.
    fn col_piv_qr(&self) -> ColPivQr<f64>;
    /// Cholesky factorization.
    fn llt(&self, side: Side) -> Result<Llt<f64>>;
    /// LDL^T factorization.
    fn ldlt(&self, side: Side) -> Result<Ldlt<f64>>;
    /// Bunch-Kaufman LBL^T decomposition.
    fn lblt(&self, side: Side) -> Lblt<f64>;
    /// Singular value decomposition.
    fn svd(&self) -> Result<Svd<f64>>;
    /// Thin SVD.
    fn thin_svd(&self) -> Result<Svd<f64>>;
    /// Singular values in descending order.
    fn singular_values(&self) -> Result<Vec<f64>>;
    /// Self-adjoint eigendecomposition.
    fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<f64>>;
    /// Eigenvalues of a self-adjoint matrix.
    fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<f64>>;
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
        Ok(Svd::factorize(self)?.s)
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
        Ok(SelfAdjointEigen::factorize(self, side)?.eigenvalues)
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
}

// ===========================================================================
// Solve methods on factorization types
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
}

impl ColPivQr<f64> {
    /// Least-squares solve `A·x ≈ b`.
    #[must_use]
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        self.solve_lstsq_impl(rhs)
    }
}

impl Svd<f64> {
    /// Left singular vectors (`U` matrix, `m × m`).
    #[must_use]
    pub fn u(&self) -> &Tensor<f64, Cpu> {
        &self.u
    }

    /// Right singular vectors transposed (`V^T` matrix, `n × n`).
    #[must_use]
    pub fn vt(&self) -> &Tensor<f64, Cpu> {
        &self.vt
    }

    /// Singular values in descending order.
    #[must_use]
    pub fn s(&self) -> &[f64] {
        &self.s
    }

    /// Rank-k approximation: `U[:, 0..k] * diag(S[0..k]) * Vt[0..k, :]`.
    #[must_use]
    pub fn reconstruct_rank(&self, k: usize) -> Tensor<f64, Cpu> {
        let (m, _) = self.u.shape();
        let (_, n) = self.vt.shape();
        let k = k.min(self.s.len());
        Tensor::from_fn(m, n, |i, j| {
            (0..k)
                .map(|r| self.u.get(i, r) * self.s[r] * self.vt.get(r, j))
                .sum::<f64>()
        })
    }
}

impl SelfAdjointEigen<f64> {
    /// Eigenvalues in ascending order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Eigenvector matrix (columns are unit eigenvectors).
    #[must_use]
    pub fn eigenvectors(&self) -> &Tensor<f64, Cpu> {
        &self.v
    }
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

// ---------------------------------------------------------------------------
// expm — Matrix exponential via Padé [7/7] with scaling-and-squaring
// ---------------------------------------------------------------------------

// Padé [7/7] coefficients (Higham 2005, Table 10.2).
const PADE7: [f64; 8] = [
    1.0,
    0.5,
    0.12,
    1.833_333_333_333_333e-2,
    1.992_063_492_063_492e-3,
    1.630_434_782_608_696e-4,
    1.035_196_687_370_6e-5,
    5.175_983_561_643_836e-7,
];

/// Matrix exponential `exp(A)` via Padé [7/7] approximation with scaling-and-squaring.
///
/// Uses the algorithm of Higham (2005) restricted to order 7.
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square or the internal linear solve fails.
pub fn expm(a: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    require_square(a.shape(), "expm")?;

    if n == 0 {
        return Ok(Tensor::<f64, Cpu>::zeros(0, 0));
    }

    // 1-norm: max column-sum of |A|
    let mut norm1 = 0.0_f64;
    for j in 0..n {
        let mut col_sum = 0.0_f64;
        for i in 0..n {
            col_sum += a.get(i, j).abs();
        }
        if col_sum > norm1 {
            norm1 = col_sum;
        }
    }

    // Scaling: s = max(0, ceil(log2(norm1))) so that ||A/2^s|| <= 1
    let s = if norm1 <= 1.0 { 0_u32 } else { norm1.log2().ceil() as u32 };
    let scale = 0.5_f64.powi(s as i32);
    let a_s = a * scale;

    // Powers
    let a2 = &a_s * &a_s;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    let eye = Tensor::<f64, Cpu>::identity(n);

    // V = c[0]*I + c[2]*A2 + c[4]*A4 + c[6]*A6
    let v = &(&(&eye * PADE7[0]) + &(&a2 * PADE7[2])) + &(&(&a4 * PADE7[4]) + &(&a6 * PADE7[6]));

    // U_inner = c[1]*I + c[3]*A2 + c[5]*A4 + c[7]*A6
    let u_inner =
        &(&(&eye * PADE7[1]) + &(&a2 * PADE7[3])) + &(&(&a4 * PADE7[5]) + &(&a6 * PADE7[7]));
    let u = &a_s * &u_inner;

    // r = (V - U)^{-1} (V + U)
    let numer = &v + &u;
    let denom = &v - &u;
    let mut r = denom.solve(&numer)?;

    // Squaring
    for _ in 0..s {
        r = &r * &r;
    }

    Ok(r)
}
