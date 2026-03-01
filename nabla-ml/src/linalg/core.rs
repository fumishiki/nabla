use core::fmt;

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;
use rayon::prelude::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Lower triangle.
    Lower,
    /// Upper triangle.
    Upper,
}

pub(crate) fn to_f64_buf(a: &Tensor<f64, Cpu>) -> Vec<f64> {
    a.as_slice().to_vec()
}

pub(crate) fn from_f64_buf(buf: Vec<f64>, m: usize, n: usize) -> Tensor<f64, Cpu> {
    use nabla_core::backend::BackendCore;
    Tensor::from_storage(Cpu::from_vec(m, n, buf))
}

#[inline]
pub(crate) fn buf_get(buf: &[f64], cols: usize, r: usize, c: usize) -> f64 {
    buf[r * cols + c]
}

#[inline]
pub(crate) fn buf_set(buf: &mut [f64], cols: usize, r: usize, c: usize, v: f64) {
    buf[r * cols + c] = v;
}

#[inline]
pub(crate) fn get_sym(a: &Tensor<f64, Cpu>, i: usize, j: usize, side: Side) -> f64 {
    match side {
        Side::Lower => a.get(i, j),
        Side::Upper => a.get(j, i),
    }
}

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
    const BLOCK: usize = 32;
    let len = v.len();
    let base = buf.as_mut_ptr() as usize;
    let blocks: Vec<usize> = (col_start..col_end).step_by(BLOCK).collect();
    #[allow(clippy::needless_range_loop)]
    let apply_block = |jb: usize| {
        let jend = (jb + BLOCK).min(col_end);
        let width = jend - jb;
        let mut dots = vec![0.0f64; width];
        // SAFETY: each column block updates disjoint memory locations; ptr is valid.
        unsafe {
            for i in 0..len {
                let row = (i + row_off) * ncols;
                let vi = v[i];
                for (k, jj) in (jb..jend).enumerate() {
                    let ptr = (base as *mut f64).add(row + jj);
                    dots[k] += vi * *ptr;
                }
            }
            for i in 0..len {
                let row = (i + row_off) * ncols;
                let vi = v[i] * tau;
                for (k, jj) in (jb..jend).enumerate() {
                    let ptr = (base as *mut f64).add(row + jj);
                    *ptr -= vi * dots[k];
                }
            }
        }
    };
    if col_end.saturating_sub(col_start) >= 128 {
        blocks.into_par_iter().for_each(apply_block);
    } else {
        for jb in blocks {
            apply_block(jb);
        }
    }
}

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
    let base = buf.as_mut_ptr() as usize;
    let rows = row_end.saturating_sub(row_start);
    #[allow(clippy::needless_range_loop)]
    let apply_row = |ii: usize| {
        let mut dot = 0.0f64;
        // SAFETY: each row updates disjoint memory locations; ptr is valid.
        unsafe {
            for c in 0..len {
                let ptr = (base as *mut f64).add(ii * ncols + c + col_off);
                dot += v[c] * *ptr;
            }
            let scale = tau * dot;
            for (c, &vc) in v.iter().enumerate().take(len) {
                let idx = ii * ncols + c + col_off;
                let ptr = (base as *mut f64).add(idx);
                *ptr -= scale * vc;
            }
        }
    };
    if rows >= 128 {
        (row_start..row_end).into_par_iter().for_each(apply_row);
    } else {
        for ii in row_start..row_end {
            apply_row(ii);
        }
    }
}

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
            pub fn solve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
                let at = self.reconstruct_impl().t();
                let f = $Type::factorize(&at)?;
                Ok(f.solve_impl(rhs))
            }
            /// Solve in place `A^T·x = b`.
            pub fn solve_transpose_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
                *rhs = self.solve_transpose(rhs)?;
                Ok(())
            }
            /// Solve `A^H·x = b` (same as transpose for real).
            pub fn solve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
                self.solve_transpose(rhs)
            }
            /// Solve in place `A^H·x = b`.
            pub fn solve_adjoint_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
                *rhs = self.solve_adjoint(rhs)?;
                Ok(())
            }
            /// Solve `x·A = b`.
            pub fn rsolve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
                let bt = rhs.t();
                let at = self.reconstruct_impl().t();
                let f = $Type::factorize(&at)?;
                Ok(f.solve_impl(&bt).t())
            }
            /// Solve in place `x·A = b`.
            pub fn rsolve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
                *rhs = self.rsolve(rhs)?;
                Ok(())
            }
        }
    };
}

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

pub(crate) use impl_factorization_methods;
pub(crate) use impl_solve_in_place;
pub(crate) use impl_triangular_solve_ip;
