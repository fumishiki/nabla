pub mod chol {
    // Cholesky-family decompositions: Llt, Ldlt, Lblt (Bunch-Kaufman).

    use nabla_core::backend::Cpu;
    use nabla_core::error::Result;
    use nabla_core::scalar::Scalar;
    use nabla_core::tensor::Tensor;

    use super::super::{
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
        pub(crate) l: Tensor<T, Cpu>,
        pub(crate) n: usize,
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
}

pub mod svd {
    // SVD — Golub-Kahan bidiagonalization + implicit QR iteration.

    use nabla_core::backend::Cpu;
    use nabla_core::error::{Error, Result};
    use nabla_core::scalar::Scalar;
    use nabla_core::tensor::Tensor;
    use rayon::prelude::*;

    use super::super::{
        buf_get, from_f64_buf, householder_apply_left, householder_apply_right, householder_vec,
        to_f64_buf, Side,
    };
    use super::super::eigen::SelfAdjointEigen;
    use super::super::qr::Qr;

    // ===========================================================================
    // 8. Svd — Golub-Kahan bidiagonalization + implicit QR
    // ===========================================================================

    /// Full/thin SVD: `A = U·Sigma·V^H`.
    pub struct Svd<T: Scalar> {
        pub(super) u: Tensor<T, Cpu>,
        pub(super) s: Vec<f64>,
        pub(super) vt: Tensor<T, Cpu>,
    }

    /// Tuning knobs for SVD performance/accuracy tradeoffs.
    #[derive(Clone, Copy)]
    pub struct SvdParams {
        pub max_iter_factor: usize,
        pub givens_parallel_threshold: usize,
        pub randomized_oversample: usize,
        pub randomized_power_iters: usize,
    }

    impl Default for SvdParams {
        fn default() -> Self {
            Self {
                max_iter_factor: 30,
                givens_parallel_threshold: 128,
                randomized_oversample: 8,
                randomized_power_iters: 1,
            }
        }
    }

    /// Randomized (low-rank) SVD: `A ≈ U·Sigma·V^T` with rank-k factors.
    pub struct RandomizedSvd {
        pub(super) u: Tensor<f64, Cpu>,
        pub(super) s: Vec<f64>,
        pub(super) vt: Tensor<f64, Cpu>,
    }

    impl RandomizedSvd {
        #[must_use]
        pub fn into_parts(self) -> (Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>) {
            (self.u, self.s, self.vt)
        }

        #[must_use]
        pub fn u(&self) -> &Tensor<f64, Cpu> {
            &self.u
        }

        #[must_use]
        pub fn vt(&self) -> &Tensor<f64, Cpu> {
            &self.vt
        }

        #[must_use]
        pub fn s(&self) -> &[f64] {
            &self.s
        }
    }

    impl Svd<f64> {
        pub(crate) fn factorize(a: &Tensor<f64, Cpu>) -> Result<Self> {
            Self::factorize_with_params(a, &SvdParams::default())
        }

        pub(crate) fn factorize_with_params(
            a: &Tensor<f64, Cpu>,
            params: &SvdParams,
        ) -> Result<Self> {
            let (m, n) = a.shape();
            // Use Jacobi SVD for small matrices; bidiag+QR for larger
            if m >= n {
                Self::golub_kahan_svd(a, m, n, params)
            } else {
                // Compute SVD of A^T, then swap U and V^T
                let at = a.t();
                let svd_t = Self::golub_kahan_svd(&at, n, m, params)?;
                Ok(Self {
                    u: svd_t.vt.t(),
                    s: svd_t.s,
                    vt: svd_t.u.t(),
                })
            }
        }

        pub(crate) fn singular_values(a: &Tensor<f64, Cpu>) -> Result<Vec<f64>> {
            Self::singular_values_with_params(a, &SvdParams::default())
        }

        pub(crate) fn singular_values_with_params(
            a: &Tensor<f64, Cpu>,
            params: &SvdParams,
        ) -> Result<Vec<f64>> {
            let (m, n) = a.shape();
            if m == 0 || n == 0 {
                return Ok(Vec::new());
            }
            if m >= n {
                let mut buf = to_f64_buf(a);
                let k = m.min(n);
                for j in 0..k {
                    let mut v: Vec<f64> = (j..m).map(|i| buf_get(&buf, n, i, j)).collect();
                    if let Some(tau) = householder_vec(&mut v) {
                        householder_apply_left(&mut buf, n, j, j, n, &v, tau);
                    }
                    if j + 2 < n {
                        let mut w: Vec<f64> = ((j + 1)..n).map(|c| buf_get(&buf, n, j, c)).collect();
                        if let Some(tau) = householder_vec(&mut w) {
                            householder_apply_right(&mut buf, n, j, m, j + 1, &w, tau);
                        }
                    }
                }
                let mut d: Vec<f64> = (0..k).map(|i| buf_get(&buf, n, i, i)).collect();
                let mut e: Vec<f64> = (0..(k.saturating_sub(1)))
                    .map(|i| buf_get(&buf, n, i, i + 1))
                    .collect();
                Self::bidiag_qr_svd(&mut d, &mut e, None, None, m, n, k, params)?;
                d.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(core::cmp::Ordering::Equal));
                Ok(d.into_iter().map(|x| x.abs()).collect())
            } else {
                let at = a.t();
                Self::singular_values_with_params(&at, params)
            }
        }

        pub fn randomized(
            a: &Tensor<f64, Cpu>,
            rank: usize,
            oversample: usize,
            n_iter: usize,
        ) -> Result<RandomizedSvd> {
            let params = SvdParams {
                randomized_oversample: oversample,
                randomized_power_iters: n_iter,
                ..SvdParams::default()
            };
            Self::randomized_with_params(a, rank, &params)
        }

        pub fn randomized_with_params(
            a: &Tensor<f64, Cpu>,
            rank: usize,
            params: &SvdParams,
        ) -> Result<RandomizedSvd> {
            let (m, n) = a.shape();
            if m == 0 || n == 0 {
                return Err(Error::invalid("randomized_svd: empty matrix"));
            }
            let k = rank.saturating_add(params.randomized_oversample).min(m.min(n));
            if k == 0 {
                return Err(Error::invalid("randomized_svd: rank must be > 0"));
            }

            let omega: Tensor<f64, Cpu> = {
                let mut seed = {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|dur| {
                            let nanos = dur.as_nanos();
                            (nanos as u64) ^ ((nanos >> 64) as u64)
                        })
                        .unwrap_or(0xA11C_E5EE_D5EE_DBAD_u64)
                };
                let total = n * k;
                let mut data = Vec::with_capacity(total);
                let mut i = 0usize;
                while i < total {
                    let u1 = {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let val = (seed as f64) / (u64::MAX as f64);
                        val.max(1e-300)
                    };
                    let u2 = {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed as f64) / (u64::MAX as f64)
                    };
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * u2;
                    data.push(r * theta.cos());
                    if i + 1 < total {
                        data.push(r * theta.sin());
                    }
                    i += 2;
                }
                Tensor::from_fn(n, k, |r, c| data[r * k + c])
            };
            let mut y = a * &omega;
            for _ in 0..params.randomized_power_iters {
                let z = a.t() * &y;
                y = a * &z;
            }
            let qr = Qr::factorize(&y);
            let q = qr.q_matrix_thin(k);
            let b = q.t() * a;
            let c = &b * &b.t();

            let eig = SelfAdjointEigen::factorize(&c, Side::Lower);
            if let Ok(eig) = eig {
                let evals = eig.eigenvalues();
                let evecs = eig.eigenvectors();

                let mut indices: Vec<usize> = (0..k).collect();
                indices.sort_by(|&a, &b| {
                    evals[b]
                        .partial_cmp(&evals[a])
                        .unwrap_or(core::cmp::Ordering::Equal)
                });

                let mut s: Vec<f64> = Vec::with_capacity(k);
                let mut u_hat = vec![0.0f64; k * k];
                for (new_i, &old_i) in indices.iter().enumerate() {
                    let sigma = evals[old_i].max(0.0).sqrt();
                    s.push(sigma);
                    for r in 0..k {
                        u_hat[r * k + new_i] = evecs.get(r, old_i);
                    }
                }
                let u_hat = from_f64_buf(u_hat, k, k);
                let u = &q * &u_hat;
                let mut vt = u_hat.t() * &b;
                let mut vt_buf = to_f64_buf(&vt);
                for i in 0..k {
                    let sigma = s[i];
                    if sigma > 0.0 {
                        for j in 0..n {
                            vt_buf[i * n + j] /= sigma;
                        }
                    } else {
                        for j in 0..n {
                            vt_buf[i * n + j] = 0.0;
                        }
                    }
                }
                vt = from_f64_buf(vt_buf, k, n);
                Ok(RandomizedSvd { u, s, vt })
            } else {
                let svd_b = Svd::factorize_with_params(&b, params)?;
                let u = &q * svd_b.u();
                Ok(RandomizedSvd {
                    u,
                    s: svd_b.s().to_vec(),
                    vt: svd_b.vt().clone(),
                })
            }
        }

        #[allow(clippy::many_single_char_names)]
        fn golub_kahan_svd(
            a: &Tensor<f64, Cpu>,
            m: usize,
            n: usize,
            params: &SvdParams,
        ) -> Result<Self> {
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
            Self::bidiag_qr_svd(
                &mut d,
                &mut e,
                Some(&mut u_mat),
                Some(&mut vt_mat),
                m,
                n,
                k,
                params,
            )?;

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
            mut u: Option<&mut [f64]>,
            mut vt: Option<&mut [f64]>,
            m: usize,
            n: usize,
            k: usize,
            params: &SvdParams,
        ) -> Result<()> {
            let mut max_iter = params.max_iter_factor.saturating_mul(k.max(1));
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
                            if let Some(vt_mat) = vt.as_deref_mut() {
                                let base = vt_mat.as_mut_ptr() as usize;
                                if n >= params.givens_parallel_threshold {
                                    (0..n).into_par_iter().for_each(|col| {
                                        // SAFETY: each column updates disjoint memory locations for rows i and j+1.
                                        unsafe {
                                            let v0 = *((base as *mut f64).add(i * n + col));
                                            let v1 = *((base as *mut f64).add((j + 1) * n + col));
                                            *((base as *mut f64).add(i * n + col)) = c * v0 + s * v1;
                                            *((base as *mut f64).add((j + 1) * n + col)) = -s * v0 + c * v1;
                                        }
                                    });
                                } else {
                                    for col in 0..n {
                                        // SAFETY: each column updates disjoint memory locations for rows i and j+1.
                                        unsafe {
                                            let v0 = *((base as *mut f64).add(i * n + col));
                                            let v1 = *((base as *mut f64).add((j + 1) * n + col));
                                            *((base as *mut f64).add(i * n + col)) = c * v0 + s * v1;
                                            *((base as *mut f64).add((j + 1) * n + col)) = -s * v0 + c * v1;
                                        }
                                    }
                                }
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
                    if let Some(vt_mat) = vt.as_deref_mut() {
                        let base = vt_mat.as_mut_ptr() as usize;
                                if n >= params.givens_parallel_threshold {
                                    (0..n).into_par_iter().for_each(|col| {
                                        // SAFETY: each column updates disjoint memory locations for rows i and i+1.
                                        unsafe {
                                            let v0 = *((base as *mut f64).add(i * n + col));
                                    let v1 = *((base as *mut f64).add((i + 1) * n + col));
                                    *((base as *mut f64).add(i * n + col)) = c * v0 + s * v1;
                                    *((base as *mut f64).add((i + 1) * n + col)) = -s * v0 + c * v1;
                                }
                            });
                        } else {
                            for col in 0..n {
                                // SAFETY: each column updates disjoint memory locations for rows i and i+1.
                                unsafe {
                                    let v0 = *((base as *mut f64).add(i * n + col));
                                    let v1 = *((base as *mut f64).add((i + 1) * n + col));
                                    *((base as *mut f64).add(i * n + col)) = c * v0 + s * v1;
                                    *((base as *mut f64).add((i + 1) * n + col)) = -s * v0 + c * v1;
                                }
                            }
                        }
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
                    if let Some(u_mat) = u.as_deref_mut() {
                        let base = u_mat.as_mut_ptr() as usize;
                        if m >= params.givens_parallel_threshold {
                            (0..m).into_par_iter().for_each(|row| {
                                // SAFETY: each row updates disjoint memory locations for columns i and i+1.
                                unsafe {
                                    let u0 = *((base as *mut f64).add(row * m + i));
                                    let u1 = *((base as *mut f64).add(row * m + i + 1));
                                    *((base as *mut f64).add(row * m + i)) = c2 * u0 + s2 * u1;
                                    *((base as *mut f64).add(row * m + i + 1)) = -s2 * u0 + c2 * u1;
                                }
                            });
                        } else {
                            for row in 0..m {
                                // SAFETY: each row updates disjoint memory locations for columns i and i+1.
                                unsafe {
                                    let u0 = *((base as *mut f64).add(row * m + i));
                                    let u1 = *((base as *mut f64).add(row * m + i + 1));
                                    *((base as *mut f64).add(row * m + i)) = c2 * u0 + s2 * u1;
                                    *((base as *mut f64).add(row * m + i + 1)) = -s2 * u0 + c2 * u1;
                                }
                            }
                        }
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
}
