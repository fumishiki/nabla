// LU decomposition types: PartialPivLu (partial pivoting), FullPivLu (full pivoting).

use nabla_core::backend::Cpu;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    buf_get, buf_set, extract_lu, factorization_failed, from_f64_buf, fwd_sub, matmul_buf,
    require_square, to_f64_buf,
};

const PIVOT_EPS: f64 = f64::EPSILON * 1e-10;

fn swap_rows(buf: &mut [f64], n: usize, r1: usize, r2: usize) {
    for j in 0..n {
        let tmp = buf_get(buf, n, r1, j);
        let v = buf_get(buf, n, r2, j);
        buf_set(buf, n, r1, j, v);
        buf_set(buf, n, r2, j, tmp);
    }
}

fn swap_cols(buf: &mut [f64], n: usize, c1: usize, c2: usize) {
    for i in 0..n {
        let tmp = buf_get(buf, n, i, c1);
        let v = buf_get(buf, n, i, c2);
        buf_set(buf, n, i, c1, v);
        buf_set(buf, n, i, c2, tmp);
    }
}

// ===========================================================================
// 1. PartialPivLu — Doolittle LU with partial (row) pivoting
// ===========================================================================

/// LU decomposition with partial row pivoting: `P·A = L·U`.
///
/// `lu` stores L (strictly below diagonal, implicit 1s) and U (diagonal and above) combined.
pub struct PartialPivLu<T: Scalar> {
    pub(super) lu: Tensor<T, Cpu>,
    /// Row permutation: row i of PA was originally row piv[i] of A.
    pub(super) piv: Vec<usize>,
    pub(super) n: usize,
}

impl PartialPivLu<f64> {
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
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
                swap_rows(&mut buf, n, k, max_row);
                piv.swap(k, max_row);
            }
            let pivot = buf_get(&buf, n, k, k);
            if pivot.abs() < PIVOT_EPS {
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

    pub(crate) fn solve_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
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
            super::bwd_sub(&lu_buf, n, &mut x, n_rhs, col, false);
        }
        from_f64_buf(x, n, n_rhs)
    }

    pub(crate) fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
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

    pub(crate) fn inverse_impl(&self) -> Tensor<f64, Cpu> {
        let eye = Tensor::identity(self.n);
        self.solve_impl(&eye)
    }

    /// Determinant of the original matrix.
    ///
    /// Computed as `sign(P) * prod(diag(U))` where `sign(P)` is the permutation
    /// sign derived from the row-pivot vector via cycle decomposition.
    #[must_use]
    pub fn det(&self) -> f64 {
        let n = self.n;
        let lu_buf = to_f64_buf(&self.lu);
        // Product of U diagonal
        let u_prod: f64 = (0..n).map(|i| buf_get(&lu_buf, n, i, i)).product();
        // Permutation sign via cycle decomposition: sign = (-1)^(n - #cycles)
        let mut visited = vec![false; n];
        let mut n_cycles = 0usize;
        for start in 0..n {
            if !visited[start] {
                n_cycles += 1;
                let mut cur = start;
                while !visited[cur] {
                    visited[cur] = true;
                    cur = self.piv[cur];
                }
            }
        }
        let perm_sign = if (n - n_cycles).is_multiple_of(2) { 1.0 } else { -1.0 };
        perm_sign * u_prod
    }

    /// Log absolute determinant of the original matrix.
    ///
    /// Computed as `sum(log(|diag(U)|))`, numerically stable for matrices where
    /// the determinant would overflow or underflow as a raw `f64`.
    #[must_use]
    pub fn logdet(&self) -> f64 {
        let n = self.n;
        let lu_buf = to_f64_buf(&self.lu);
        (0..n)
            .map(|i| buf_get(&lu_buf, n, i, i).abs().ln())
            .sum()
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
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
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
                swap_rows(&mut buf, n, k, max_r);
                rpiv.swap(k, max_r);
            }
            // col swap
            if max_c != k {
                swap_cols(&mut buf, n, k, max_c);
                cpiv.swap(k, max_c);
            }
            let pivot = buf_get(&buf, n, k, k);
            if pivot.abs() < PIVOT_EPS {
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
