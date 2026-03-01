use nabla_core::backend::Cpu;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    Side, buf_get, factorization_failed, from_f64_buf, householder_apply_left,
    householder_apply_right, householder_vec, require_square, symmetrize_to_buf,
};

pub struct SelfAdjointEigen<T: Scalar> {
    /// Eigenvectors stored as columns (V matrix).
    pub(super) v: Tensor<T, Cpu>,
    /// Eigenvalues in ascending order.
    pub(super) eigenvalues: Vec<f64>,
}

impl SelfAdjointEigen<f64> {
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    pub(crate) fn factorize(a: &Tensor<f64, Cpu>, side: Side) -> Result<Self> {
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
        self.eigenvalues()
    }

    /// Eigenvector matrix (columns are eigenvectors).
    #[must_use]
    pub fn vectors(&self) -> &Tensor<f64, Cpu> {
        self.eigenvectors()
    }

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

#[allow(dead_code)]
mod francis {
    use nabla_core::backend::Cpu;
    use nabla_core::error::{Error, Result};
    use nabla_core::tensor::Tensor;

    use super::super::{
        buf_get, buf_set, from_f64_buf, householder_apply_left, householder_apply_right,
        householder_vec, to_f64_buf,
    };

    #[allow(clippy::many_single_char_names)]
    fn hessenberg_reduce(buf: &mut [f64], n: usize) {
        for j in 0..(n.saturating_sub(2)) {
            let mut v: Vec<f64> = ((j + 1)..n).map(|i| buf_get(buf, n, i, j)).collect();
            let Some(tau) = householder_vec(&mut v) else {
                continue;
            };
            householder_apply_left(buf, n, j + 1, 0, n, &v, tau);
            householder_apply_right(buf, n, 0, n, j + 1, &v, tau);
        }
    }

    #[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
    fn francis_double_shift_step(h: &mut [f64], n: usize, lo: usize, hi: usize) {
        let h11 = buf_get(h, n, hi - 1, hi - 1);
        let h12 = buf_get(h, n, hi - 1, hi);
        let h21 = buf_get(h, n, hi, hi - 1);
        let h22 = buf_get(h, n, hi, hi);
        let s = h11 + h22;
        let t = h11 * h22 - h12 * h21;

        let h00 = buf_get(h, n, lo, lo);
        let h10 = buf_get(h, n, lo + 1, lo);
        let h20 = if lo + 2 <= hi {
            buf_get(h, n, lo + 2, lo)
        } else {
            0.0
        };
        let h01 = buf_get(h, n, lo, lo + 1);
        let h11_lo = buf_get(h, n, lo + 1, lo + 1);

        let x = h00 * h00 + h01 * h10 - s * h00 + t;
        let y = h10 * (h00 + h11_lo - s);
        let z = h10 * h20;

        let len = hi - lo + 1;
        for k in 0..len.saturating_sub(2) {
            let col = lo + k;
            let row_end = (col + 3).min(hi + 1);
            let bulge_len = row_end - col;
            let mut pv = if k == 0 {
                vec![x, y, z]
            } else {
                (col..row_end)
                    .map(|r| buf_get(h, n, r, col - 1))
                    .collect::<Vec<_>>()
            };
            pv.truncate(bulge_len);

            let Some(tau) = householder_vec(&mut pv) else {
                continue;
            };

            let apply_col_start = if k == 0 { col } else { col - 1 };
            householder_apply_left(h, n, col, apply_col_start, n, &pv, tau);
            householder_apply_right(h, n, 0, hi + 1, col, &pv, tau);
        }

        let second_last = hi - 1;
        let a = buf_get(h, n, second_last, second_last - 1);
        let b_val = buf_get(h, n, hi, second_last - 1);
        let r = a.hypot(b_val);
        if r > f64::EPSILON {
            let c = a / r;
            let s_g = b_val / r;
            for col in (second_last - 1)..n {
                let v0 = buf_get(h, n, second_last, col);
                let v1 = buf_get(h, n, hi, col);
                buf_set(h, n, second_last, col, c * v0 + s_g * v1);
                buf_set(h, n, hi, col, -s_g * v0 + c * v1);
            }
            for row in 0..=hi {
                let v0 = buf_get(h, n, row, second_last);
                let v1 = buf_get(h, n, row, hi);
                buf_set(h, n, row, second_last, c * v0 + s_g * v1);
                buf_set(h, n, row, hi, -s_g * v0 + c * v1);
            }
            buf_set(h, n, hi, second_last - 1, 0.0);
        }
    }

    fn read_eigenvalues_from_schur(h: &[f64], n: usize) -> Vec<(f64, f64)> {
        let mut eigs = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            if i + 1 < n && buf_get(h, n, i + 1, i).abs() > f64::EPSILON * 10.0 {
                let a = buf_get(h, n, i, i);
                let b = buf_get(h, n, i, i + 1);
                let c = buf_get(h, n, i + 1, i);
                let d = buf_get(h, n, i + 1, i + 1);
                let tr = (a + d) * 0.5;
                let disc = (a - d) * (a - d) * 0.25 + b * c;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    eigs.push((tr + sq, 0.0));
                    eigs.push((tr - sq, 0.0));
                } else {
                    let im = (-disc).sqrt();
                    eigs.push((tr, im));
                    eigs.push((tr, -im));
                }
                i += 2;
            } else {
                eigs.push((buf_get(h, n, i, i), 0.0));
                i += 1;
            }
        }
        eigs
    }

    fn francis_qr_iterate(h: &mut [f64], n: usize) -> Result<()> {
        if n <= 1 {
            return Ok(());
        }

        let max_iter = 30 * n;
        let mut iter = 0;
        let mut hi = n - 1;

        while hi > 0 {
            if iter > max_iter {
                return Err(Error::invalid("eig_into: Francis QR failed to converge"));
            }

            let tol =
                f64::EPSILON * (buf_get(h, n, hi - 1, hi - 1).abs() + buf_get(h, n, hi, hi).abs());
            if buf_get(h, n, hi, hi - 1).abs() <= tol {
                buf_set(h, n, hi, hi - 1, 0.0);
                hi = hi.saturating_sub(1);
                continue;
            }

            let mut lo = hi - 1;
            while lo > 0 {
                let sub = buf_get(h, n, lo, lo - 1).abs();
                let local_tol = f64::EPSILON
                    * (buf_get(h, n, lo - 1, lo - 1).abs() + buf_get(h, n, lo, lo).abs());
                if sub <= local_tol {
                    break;
                }
                lo -= 1;
            }

            francis_double_shift_step(h, n, lo, hi);
            iter += 1;
        }
        Ok(())
    }

    pub(crate) fn francis_qr_schur(
        a: &Tensor<f64, Cpu>,
    ) -> Result<(Tensor<f64, Cpu>, Vec<(f64, f64)>)> {
        let (m, n) = a.shape();
        if m != n {
            return Err(Error::invalid("francis_qr_schur: input must be square"));
        }
        if n == 0 {
            return Ok((Tensor::<f64, Cpu>::zeros(0, 0), Vec::new()));
        }
        let mut h = to_f64_buf(a);
        hessenberg_reduce(&mut h, n);
        francis_qr_iterate(&mut h, n)?;
        let eigs = read_eigenvalues_from_schur(&h, n);
        Ok((from_f64_buf(h, n, n), eigs))
    }
}
