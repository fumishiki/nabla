// SVD — Golub-Kahan bidiagonalization + implicit QR iteration.

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    buf_get, from_f64_buf, householder_apply_left, householder_apply_right, householder_vec,
    to_f64_buf,
};

// ===========================================================================
// 8. Svd — Golub-Kahan bidiagonalization + implicit QR
// ===========================================================================

/// Full/thin SVD: `A = U·Sigma·V^H`.
pub struct Svd<T: Scalar> {
    pub(super) u: Tensor<T, Cpu>,
    pub(super) s: Vec<f64>,
    pub(super) vt: Tensor<T, Cpu>,
}

impl Svd<f64> {
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
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
    #[allow(clippy::too_many_lines)]
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
            let e_top = if p >= 3 && (p - 3) >= q {
                e[p - 3]
            } else {
                0.0
            };
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

    /// Unpack into `(U, singular_values, V^T)`.
    ///
    /// Consumes `self` and returns the three components of the decomposition
    /// `A = U·Sigma·V^T` as owned values.
    #[must_use]
    pub fn into_parts(self) -> (Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>) {
        (self.u, self.s, self.vt)
    }

    /// Left singular vectors (`U` matrix, `m x m`).
    #[must_use]
    pub fn u(&self) -> &Tensor<f64, Cpu> {
        &self.u
    }

    /// Right singular vectors transposed (`V^T` matrix, `n x n`).
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
