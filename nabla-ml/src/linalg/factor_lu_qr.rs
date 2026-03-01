pub mod lu {
    // LU decomposition types: PartialPivLu (partial pivoting), FullPivLu (full pivoting).

    use nabla_core::backend::Cpu;
    use nabla_core::error::Result;
    use nabla_core::scalar::Scalar;
    use nabla_core::tensor::Tensor;

    use super::super::{
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
        pub(crate) lu: Tensor<T, Cpu>,
        /// Row permutation: row i of PA was originally row piv[i] of A.
        pub(crate) piv: Vec<usize>,
        pub(crate) n: usize,
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
                super::super::bwd_sub(&lu_buf, n, &mut x, n_rhs, col, false);
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
            let perm_sign = if (n - n_cycles).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
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
            (0..n).map(|i| buf_get(&lu_buf, n, i, i).abs().ln()).sum()
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
}

pub mod qr {
    // QR decomposition types: Qr (Householder), ColPivQr (column-pivoted Householder).

    use nabla_core::backend::Cpu;
    use nabla_core::scalar::Scalar;
    use nabla_core::tensor::Tensor;
    use rayon::prelude::*;

    use super::super::{
        buf_get, buf_set, from_f64_buf, householder_apply_left, householder_vec, to_f64_buf,
    };

    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
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
            let base = x.as_mut_ptr() as usize;
            (0..n_rhs).into_par_iter().for_each(|col| {
                let mut dot = 0.0f64;
                // SAFETY: each column updates disjoint memory locations; ptr is valid.
                unsafe {
                    for i in 0..col_len {
                        let ptr = (base as *mut f64).add((i + j) * n_rhs + col);
                        dot += v[i] * *ptr;
                    }
                    let scale = tau * dot;
                    for i in 0..col_len {
                        let idx = (i + j) * n_rhs + col;
                        let ptr = (base as *mut f64).add(idx);
                        *ptr -= scale * v[i];
                    }
                }
            });
        }
    }

    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    fn apply_q_in_place(
        qr_buf: &[f64],
        taus: &[f64],
        m: usize,
        n: usize,
        x: &mut [f64],
        n_rhs: usize,
    ) {
        let k = m.min(n);
        for j in (0..k).rev() {
            let tau = taus[j];
            if tau == 0.0 {
                continue;
            }
            let col_len = m - j;
            let mut v = vec![1.0f64; col_len];
            for (i, vi) in v.iter_mut().enumerate().skip(1) {
                *vi = buf_get(qr_buf, n, i + j, j);
            }
            let base = x.as_mut_ptr() as usize;
            (0..n_rhs).into_par_iter().for_each(|col| {
                let mut dot = 0.0f64;
                // SAFETY: each column updates disjoint memory locations; ptr is valid.
                unsafe {
                    for i in 0..col_len {
                        let ptr = (base as *mut f64).add((i + j) * n_rhs + col);
                        dot += v[i] * *ptr;
                    }
                    let scale = tau * dot;
                    for i in 0..col_len {
                        let idx = (i + j) * n_rhs + col;
                        let ptr = (base as *mut f64).add(idx);
                        *ptr -= scale * v[i];
                    }
                }
            });
        }
    }

    // ===========================================================================
    // 3. Qr — Householder QR decomposition
    // ===========================================================================

    /// TSQR for tall-skinny matrices: returns thin Q (m×n) and R (n×n).
    #[must_use]
    pub fn tsqr(a: &Tensor<f64, Cpu>, block_rows: usize) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>) {
        let (m, n) = a.shape();
        if m == 0 || n == 0 || m < 2 * n {
            let qr = Qr::factorize(a);
            let q = qr.q_matrix_thin(n);
            let r = qr.r_matrix();
            return (q, r);
        }
        let block_rows = block_rows.max(n);
        let mut q_blocks: Vec<Tensor<f64, Cpu>> = Vec::new();
        let mut r_blocks: Vec<Tensor<f64, Cpu>> = Vec::new();
        let mut row = 0usize;
        while row < m {
            let mut end = (row + block_rows).min(m);
            if end < m && m - end < n {
                end = m - n;
            }
            if end <= row {
                break;
            }
            let block = a.slice_rows(row..end);
            let qr = Qr::factorize(&block);
            q_blocks.push(qr.q_matrix_thin(n));
            r_blocks.push(qr.r_matrix());
            row = end;
        }
        let r_refs: Vec<&Tensor<f64, Cpu>> = r_blocks.iter().collect();
        let r_stack = Tensor::cat(&r_refs, 0);
        let qr2 = Qr::factorize(&r_stack);
        let q2 = qr2.q_matrix_thin(n);
        let r_final = qr2.r_matrix();

        let mut q_out_blocks: Vec<Tensor<f64, Cpu>> = Vec::with_capacity(q_blocks.len());
        for (i, q_i) in q_blocks.iter().enumerate() {
            let q2_block = q2.slice_rows((i * n)..((i + 1) * n));
            q_out_blocks.push(q_i * &q2_block);
        }
        let q_refs: Vec<&Tensor<f64, Cpu>> = q_out_blocks.iter().collect();
        let q_final = Tensor::cat(&q_refs, 0);
        (q_final, r_final)
    }

    /// QR decomposition via Householder reflections: `A = Q·R`.
    #[allow(clippy::struct_field_names)]
    pub struct Qr<T: Scalar> {
        /// Combined Q (as Householder vectors stored below diagonal) and R (upper triangle).
        pub(crate) qr: Tensor<T, Cpu>,
        /// Householder scaling factors (tau values).
        pub(crate) taus: Vec<f64>,
        pub(crate) m: usize,
        pub(crate) n: usize,
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
        pub(crate) fn apply_qt(&self, rhs: &[f64], n_rhs: usize) -> Vec<f64> {
            let (m, n) = (self.m, self.n);
            let mut x = rhs.to_vec();
            let qr_buf = to_f64_buf(&self.qr);
            apply_qt_in_place(&qr_buf, &self.taus, m, n, &mut x, n_rhs);
            x
        }

        #[allow(clippy::many_single_char_names)]
        pub(crate) fn apply_q(&self, rhs: &[f64], n_rhs: usize) -> Vec<f64> {
            let (m, n) = (self.m, self.n);
            let mut x = rhs.to_vec();
            let qr_buf = to_f64_buf(&self.qr);
            apply_q_in_place(&qr_buf, &self.taus, m, n, &mut x, n_rhs);
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
}
