// SelfAdjointEigen — symmetric eigendecomposition via tridiagonalization + Wilkinson QR shifts.

use nabla_core::backend::Cpu;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::{
    Side, buf_get, factorization_failed, from_f64_buf, householder_apply_left,
    householder_apply_right, householder_vec, require_square, symmetrize_to_buf,
};

// ===========================================================================
// 9. SelfAdjointEigen — symmetric eigen (tridiag + Wilkinson QR shifts)
// ===========================================================================

/// Self-adjoint (symmetric) eigendecomposition: `A = V·Lambda·V^T`.
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
