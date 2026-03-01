// QR decomposition types: Qr (Householder), ColPivQr (column-pivoted Householder).

use nabla_core::backend::Cpu;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{buf_get, buf_set, from_f64_buf, householder_apply_left, householder_vec, to_f64_buf};

#[allow(clippy::many_single_char_names)]
fn apply_qt_in_place(
    qr_buf: &[f64],
    taus: &[f64],
    m: usize,
    n: usize,
    x: &mut [f64],
    n_rhs: usize,
) {
    let k = m.min(n);
    for j in 0..k {
        let tau = taus[j];
        if tau == 0.0 {
            continue;
        }
        let col_len = m - j;
        let mut v = vec![1.0f64; col_len];
        for (i, vi) in v.iter_mut().enumerate().skip(1) {
            *vi = buf_get(qr_buf, n, i + j, j);
        }
        for col in 0..n_rhs {
            let dot: f64 = (0..col_len).map(|i| v[i] * x[(i + j) * n_rhs + col]).sum();
            let scale = tau * dot;
            for i in 0..col_len {
                x[(i + j) * n_rhs + col] -= scale * v[i];
            }
        }
    }
}

// ===========================================================================
// 3. Qr — Householder QR decomposition
// ===========================================================================

/// QR decomposition via Householder reflections: `A = Q·R`.
#[allow(clippy::struct_field_names)]
pub struct Qr<T: Scalar> {
    /// Combined Q (as Householder vectors stored below diagonal) and R (upper triangle).
    pub(super) qr: Tensor<T, Cpu>,
    /// Householder scaling factors (tau values).
    pub(super) taus: Vec<f64>,
    pub(super) m: usize,
    pub(super) n: usize,
}

impl Qr<f64> {
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>) -> Self {
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
    pub(super) fn apply_qt(&self, rhs: &[f64], n_rhs: usize) -> Vec<f64> {
        let (m, n) = (self.m, self.n);
        let mut x = rhs.to_vec();
        let qr_buf = to_f64_buf(&self.qr);
        apply_qt_in_place(&qr_buf, &self.taus, m, n, &mut x, n_rhs);
        x
    }

    #[allow(clippy::many_single_char_names)]
    pub(crate) fn solve_lstsq_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
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
    pub(super) qr: Tensor<T, Cpu>,
    pub(super) taus: Vec<f64>,
    pub(super) col_piv: Vec<usize>,
    pub(super) m: usize,
    pub(super) n: usize,
}

impl ColPivQr<f64> {
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>) -> Self {
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
    pub(crate) fn solve_lstsq_impl(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        // Same as Qr::solve_lstsq but apply column permutation at end
        let (m, n) = (self.m, self.n);
        let n_rhs = rhs.ncols();
        let rhs_buf = to_f64_buf(rhs);
        let qr_buf = to_f64_buf(&self.qr);
        let k = m.min(n);

        // Apply Q^T
        let mut x = rhs_buf.clone();
        apply_qt_in_place(&qr_buf, &self.taus, m, n, &mut x, n_rhs);

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
