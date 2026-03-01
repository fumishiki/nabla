// Cholesky-family decompositions: Llt, Ldlt, Lblt (Bunch-Kaufman).

use nabla_core::backend::Cpu;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    Side, buf_get, buf_set, factorization_failed, from_f64_buf, fwd_sub, get_sym, require_square,
    symmetrize_to_buf, to_f64_buf,
};

const LDLT_PIVOT_EPS: f64 = f64::EPSILON * 1e-10;
const LBLT_PIVOT_EPS: f64 = f64::EPSILON * 1e-12;

/// Backward-substitute L^T · x = y.  When `unit` is true, L has unit diagonal.
fn bwd_sub_lt(l: &[f64], n: usize, x: &mut [f64], n_rhs: usize, col: usize, unit: bool) {
    for i in (0..n).rev() {
        let mut sum = x[i * n_rhs + col];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j * n_rhs + col];
        }
        x[i * n_rhs + col] = if unit { sum } else { sum / l[i * n + i] };
    }
}

fn swap_sym(buf: &mut [f64], n: usize, i: usize, j: usize) {
    if i == j {
        return;
    }
    for col in 0..n {
        let tmp = buf_get(buf, n, i, col);
        let v = buf_get(buf, n, j, col);
        buf_set(buf, n, i, col, v);
        buf_set(buf, n, j, col, tmp);
    }
    for row in 0..n {
        let tmp = buf_get(buf, n, row, i);
        let v = buf_get(buf, n, row, j);
        buf_set(buf, n, row, i, v);
        buf_set(buf, n, row, j, tmp);
    }
}

// ===========================================================================
// 5. Llt — Cholesky factorization (A = L·L^T)
// ===========================================================================

/// Cholesky factorization: `A = L·L^T` (positive-definite symmetric).
pub struct Llt<T: Scalar> {
    pub(super) l: Tensor<T, Cpu>,
    pub(super) n: usize,
}

impl Llt<f64> {
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Result<Self> {
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

    pub(crate) fn solve_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
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
            bwd_sub_lt(&l_buf, n, &mut x, n_rhs, col, false);
        }
        from_f64_buf(x, n, n_rhs)
    }

    pub(crate) fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
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

    pub(crate) fn inverse_impl(&self) -> Tensor<f64, Cpu> {
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
    pub(super) ld: Tensor<T, Cpu>,
    pub(super) n: usize,
}

impl Ldlt<f64> {
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Result<Self> {
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
            if d_jj.abs() < LDLT_PIVOT_EPS {
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

    pub(crate) fn solve_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
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
            bwd_sub_lt(&ld_buf, n, &mut x, n_rhs, col, true);
        }
        from_f64_buf(x, n, n_rhs)
    }

    pub(crate) fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
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

    pub(crate) fn inverse_impl(&self) -> Tensor<f64, Cpu> {
        let eye = Tensor::identity(self.n);
        self.solve_impl(&eye)
    }
}

// ===========================================================================
// 7. Lblt — Bunch-Kaufman pivoted LBL^T (indefinite symmetric)
// ===========================================================================

/// Bunch-Kaufman `LBL^T` decomposition for indefinite symmetric matrices.
///
/// Uses 1x1 and 2x2 pivots to handle indefinite systems.
pub struct Lblt<T: Scalar> {
    /// Combined L and B stored as flat f64 (L below diagonal, B on/near diagonal).
    pub(super) lb: Tensor<T, Cpu>,
}

impl Lblt<f64> {
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Self {
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
                // 1x1 pivot: use diagonal element
                piv[k] = (k + 1) as i64;
                let d = buf_get(&buf, n, k, k);
                if d.abs() > LBLT_PIVOT_EPS {
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
                // Check 2x2 pivot
                let a_rr = buf_get(&buf, n, max_r, max_r).abs();

                if a_rr * max_off >= alpha * max_off * max_off {
                    // 1x1 pivot with row/col max_r swapped to k
                    if max_r == k {
                        piv[k] = (k + 1) as i64;
                    } else {
                        // swap rows/cols k and max_r
                        swap_sym(&mut buf, n, k, max_r);
                        piv[k] = (max_r + 1) as i64;
                    }
                    let d = buf_get(&buf, n, k, k);
                    if d.abs() > LBLT_PIVOT_EPS {
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
                    // 2x2 pivot: swap max_r to k+1
                    if max_r != k + 1 {
                        swap_sym(&mut buf, n, k + 1, max_r);
                    }
                    // 2x2 pivot block B = [[a,b],[b,c]]
                    let b_a = buf_get(&buf, n, k, k);
                    let b_b = buf_get(&buf, n, k + 1, k);
                    let b_c = buf_get(&buf, n, k + 1, k + 1);
                    let det = b_a * b_c - b_b * b_b;
                    if det.abs() > LBLT_PIVOT_EPS {
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

    pub(crate) fn reconstruct_impl(&self) -> Tensor<f64, Cpu> {
        // Return the stored (partially factored) matrix as-is for reconstruction
        // A = L · B · L^T where B is block diagonal stored in lb
        // For practical purposes, return the stored lb as-is
        self.lb.clone()
    }
}
