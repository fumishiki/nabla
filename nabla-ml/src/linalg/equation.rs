use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::matrix_fn::schur;
use super::{from_f64_buf, require_square, to_f64_buf};

#[inline]
fn ensure_nonzero_pivot(val: f64, idx: usize) -> Result<()> {
    if val.abs() < f64::MIN_POSITIVE {
        return Err(Error::invalid(format!(
            "solve_tridiag: zero pivot at index {idx}"
        )));
    }
    Ok(())
}

#[inline]
fn ensure_shape(rows: usize, cols: usize, tensor: &Tensor<f64, Cpu>, ctx: &str) -> Result<()> {
    if tensor.shape() != (rows, cols) {
        return Err(Error::invalid(format!(
            "{ctx}: shape must be {rows}×{cols}, got {:?}",
            tensor.shape()
        )));
    }
    Ok(())
}

fn detect_block_sizes(t: &[f64], n: usize) -> Vec<usize> {
    let mut blocks = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n
            && t[(i + 1) * n + i].abs()
                > f64::EPSILON
                    * 4.0
                    * (t[i * n + i].abs() + t[(i + 1) * n + i + 1].abs()).max(f64::EPSILON)
        {
            blocks.push(2);
            i += 2;
        } else {
            blocks.push(1);
            i += 1;
        }
    }
    blocks
}

#[inline]
fn solve_2x2(a: f64, b: f64, c: f64, d: f64, r0: f64, r1: f64) -> Result<(f64, f64)> {
    let det = a * d - b * c;
    if det.abs() < f64::EPSILON * 1e6 {
        return Err(Error::invalid("sylvester: singular 2x2 block"));
    }
    Ok(((r0 * d - b * r1) / det, (a * r1 - r0 * c) / det))
}

#[inline]
fn solve_4x4(mat: &[f64; 16], rhs: &[f64; 4]) -> Result<[f64; 4]> {
    let mut a = *mat;
    let mut b = *rhs;
    for col in 0..4 {
        let pivot_row = (col..4)
            .max_by(|&r1, &r2| {
                a[r1 * 4 + col]
                    .abs()
                    .partial_cmp(&a[r2 * 4 + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        if pivot_row != col {
            for k in 0..4 {
                a.swap(col * 4 + k, pivot_row * 4 + k);
            }
            b.swap(col, pivot_row);
        }
        let piv = a[col * 4 + col];
        if piv.abs() < f64::EPSILON * 1e6 {
            return Err(Error::invalid("sylvester: singular 4x4 block"));
        }
        for row in (col + 1)..4 {
            let factor = a[row * 4 + col] / piv;
            for k in (col + 1)..4 {
                a[row * 4 + k] -= factor * a[col * 4 + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        let mut s = b[i];
        for k in (i + 1)..4 {
            s -= a[i * 4 + k] * x[k];
        }
        x[i] = s / a[i * 4 + i];
    }
    Ok(x)
}

fn solve_triangular_sylvester(
    ta: &[f64],
    tb: &[f64],
    f: &mut [f64],
    n: usize,
    m: usize,
) -> Result<()> {
    let ta_blocks = detect_block_sizes(ta, n);
    let tb_blocks = detect_block_sizes(tb, m);

    let mut jj = 0;
    for &bj in &tb_blocks {
        // Subtract contributions from previously solved column-blocks
        for jc in jj..(jj + bj) {
            for k in 0..jj {
                let tb_kj = tb[k * m + jc];
                if tb_kj.abs() < f64::EPSILON {
                    continue;
                }
                for i in 0..n {
                    f[i * m + jc] -= tb_kj * f[i * m + k];
                }
            }
        }

        // Back-substitute row-blocks of T_A (bottom to top)
        let mut ii = n;
        for &bi in ta_blocks.iter().rev() {
            ii -= bi;
            // Subtract contributions from rows below ii+bi
            for ic in ii..(ii + bi) {
                for jc in jj..(jj + bj) {
                    let mut s = 0.0;
                    for k in (ii + bi)..n {
                        s += ta[ic * n + k] * f[k * m + jc];
                    }
                    f[ic * m + jc] -= s;
                }
            }

            if bi == 1 && bj == 1 {
                let diag = ta[ii * n + ii] + tb[jj * m + jj];
                if diag.abs() < f64::EPSILON * 1e6 {
                    return Err(Error::invalid(format!(
                        "sylvester: near-singular at ({ii},{jj}); A and -B may share an eigenvalue"
                    )));
                }
                f[ii * m + jj] /= diag;
            } else if bi == 2 && bj == 1 {
                // (T_A_2x2 + shift*I) * [y0; y1] = [rhs0; rhs1]
                let shift = tb[jj * m + jj];
                let (y0, y1) = solve_2x2(
                    ta[ii * n + ii] + shift,
                    ta[ii * n + ii + 1],
                    ta[(ii + 1) * n + ii],
                    ta[(ii + 1) * n + ii + 1] + shift,
                    f[ii * m + jj],
                    f[(ii + 1) * m + jj],
                )?;
                f[ii * m + jj] = y0;
                f[(ii + 1) * m + jj] = y1;
            } else if bi == 1 && bj == 2 {
                // y(ii,jj) and y(ii,jj+1) coupled by T_B 2x2 block
                let d = ta[ii * n + ii];
                let (y0, y1) = solve_2x2(
                    d + tb[jj * m + jj],
                    tb[(jj + 1) * m + jj],
                    tb[jj * m + jj + 1],
                    d + tb[(jj + 1) * m + jj + 1],
                    f[ii * m + jj],
                    f[ii * m + jj + 1],
                )?;
                f[ii * m + jj] = y0;
                f[ii * m + jj + 1] = y1;
            } else {
                // 4x4: both T_A and T_B have 2x2 blocks
                let (i0, i1) = (ii, ii + 1);
                let (j0, j1) = (jj, jj + 1);
                // Unknowns: [y(i0,j0), y(i0,j1), y(i1,j0), y(i1,j1)]
                // Equation: T_A_block * Y_block + Y_block * T_B_block = F_block
                // Kronecker: (I_bj kron T_A_block + T_B_block^T kron I_bi) vec(Y) = vec(F)
                let (a00, a01) = (ta[i0 * n + i0], ta[i0 * n + i1]);
                let (a10, a11) = (ta[i1 * n + i0], ta[i1 * n + i1]);
                let (b00, b01) = (tb[j0 * m + j0], tb[j0 * m + j1]);
                let (b10, b11) = (tb[j1 * m + j0], tb[j1 * m + j1]);
                // (I kron A + B^T kron I) for vec order [y00, y10, y01, y11]
                let mat = [
                    a00 + b00,
                    a01,
                    b10,
                    0.0,
                    a10,
                    a11 + b00,
                    0.0,
                    b10,
                    b01,
                    0.0,
                    a00 + b11,
                    a01,
                    0.0,
                    b01,
                    a10,
                    a11 + b11,
                ];
                let rhs_vec = [
                    f[i0 * m + j0],
                    f[i1 * m + j0],
                    f[i0 * m + j1],
                    f[i1 * m + j1],
                ];
                let x = solve_4x4(&mat, &rhs_vec)?;
                f[i0 * m + j0] = x[0];
                f[i1 * m + j0] = x[1];
                f[i0 * m + j1] = x[2];
                f[i1 * m + j1] = x[3];
            }
        }
        jj += bj;
    }
    Ok(())
}

pub fn sylvester(
    a: &Tensor<f64, Cpu>,
    b: &Tensor<f64, Cpu>,
    c: &Tensor<f64, Cpu>,
) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    let m = b.nrows();
    require_square(a.shape(), "sylvester(A)")?;
    require_square(b.shape(), "sylvester(B)")?;
    ensure_shape(n, m, c, "sylvester(C)")?;

    if n == 0 || m == 0 {
        return Ok(Tensor::<f64, Cpu>::zeros(n, m));
    }

    let (ta_tensor, qa_tensor) = schur(a)?;
    let (tb_tensor, qb_tensor) = schur(b)?;

    let ta = to_f64_buf(&ta_tensor);
    let tb = to_f64_buf(&tb_tensor);

    // F = Q_A^T · C · Q_B
    let f_tensor = &(&qa_tensor.t() * c) * &qb_tensor;
    let mut f_buf = to_f64_buf(&f_tensor);

    solve_triangular_sylvester(&ta, &tb, &mut f_buf, n, m)?;

    // Y is now in f_buf; recover X = Q_A · Y · Q_B^T
    let y_tensor = from_f64_buf(f_buf, n, m);
    Ok(&(&qa_tensor * &y_tensor) * &qb_tensor.t())
}

pub fn lyapunov(a: &Tensor<f64, Cpu>, q: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    require_square(a.shape(), "lyapunov")?;
    let n = a.nrows();
    ensure_shape(n, n, q, "lyapunov(Q)")?;
    let at = a.t();
    sylvester(a, &at, q)
}

pub fn solve_tridiag(lower: &[f64], main: &[f64], upper: &[f64], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = main.len();
    if lower.len() != n.saturating_sub(1) || upper.len() != n.saturating_sub(1) || rhs.len() != n {
        return Err(Error::invalid(format!(
            "solve_tridiag: inconsistent lengths: lower={}, main={n}, upper={}, rhs={}",
            lower.len(),
            upper.len(),
            rhs.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Work buffers: modified main diagonal and rhs.
    let mut m = main.to_vec();
    let mut d = rhs.to_vec();

    // Forward sweep: eliminate sub-diagonal.
    for i in 1..n {
        ensure_nonzero_pivot(m[i - 1], i - 1)?;
        let factor = lower[i - 1] / m[i - 1];
        m[i] -= factor * upper[i - 1];
        d[i] -= factor * d[i - 1];
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    ensure_nonzero_pivot(m[n - 1], n - 1)?;
    x[n - 1] = d[n - 1] / m[n - 1];
    for i in (0..n - 1).rev() {
        ensure_nonzero_pivot(m[i], i)?;
        x[i] = (d[i] - upper[i] * x[i + 1]) / m[i];
    }
    Ok(x)
}

#[allow(clippy::too_many_lines)]
fn solve_triangular_discrete_sylvester(
    ta: &[f64],
    tb: &[f64],
    f: &mut [f64],
    n: usize,
    m: usize,
) -> Result<()> {
    let ta_blocks = detect_block_sizes(ta, n);
    let tb_blocks = detect_block_sizes(tb, m);

    let mut jj = 0;
    for &bj in &tb_blocks {
        // Add T_A * Y[:,k] * T_B[k,jc] contributions from solved columns k < jj
        for jc in jj..(jj + bj) {
            for k in 0..jj {
                let tb_kj = tb[k * m + jc];
                if tb_kj.abs() < f64::EPSILON {
                    continue;
                }
                for i in 0..n {
                    let mut ta_y = 0.0;
                    for r in 0..n {
                        ta_y += ta[i * n + r] * f[r * m + k];
                    }
                    f[i * m + jc] += tb_kj * ta_y;
                }
            }
        }

        // Back-substitute row-blocks of T_A (bottom to top)
        let mut ii = n;
        for &bi in ta_blocks.iter().rev() {
            ii -= bi;
            // Add T_A * (already-solved rows below) * T_B block contribution
            for ic in ii..(ii + bi) {
                for jc in jj..(jj + bj) {
                    let mut acc = 0.0;
                    for jc2 in jj..(jj + bj) {
                        let mut s = 0.0;
                        for k in (ii + bi)..n {
                            s += ta[ic * n + k] * f[k * m + jc2];
                        }
                        acc += s * tb[jc2 * m + jc];
                    }
                    f[ic * m + jc] += acc;
                }
            }

            if bi == 1 && bj == 1 {
                let diag = 1.0 - tb[jj * m + jj] * ta[ii * n + ii];
                if diag.abs() < f64::EPSILON * 1e6 {
                    return Err(Error::invalid(format!(
                        "discrete_sylvester: near-singular at ({ii},{jj})"
                    )));
                }
                f[ii * m + jj] /= diag;
            } else if bi == 2 && bj == 1 {
                let shift = tb[jj * m + jj];
                let (y0, y1) = solve_2x2(
                    1.0 - shift * ta[ii * n + ii],
                    -shift * ta[ii * n + ii + 1],
                    -shift * ta[(ii + 1) * n + ii],
                    1.0 - shift * ta[(ii + 1) * n + ii + 1],
                    f[ii * m + jj],
                    f[(ii + 1) * m + jj],
                )?;
                f[ii * m + jj] = y0;
                f[(ii + 1) * m + jj] = y1;
            } else if bi == 1 && bj == 2 {
                let a_ii = ta[ii * n + ii];
                let (b00, b01) = (tb[jj * m + jj], tb[jj * m + jj + 1]);
                let (b10, b11) = (tb[(jj + 1) * m + jj], tb[(jj + 1) * m + jj + 1]);
                let (y0, y1) = solve_2x2(
                    1.0 - b00 * a_ii,
                    -b10 * a_ii,
                    -b01 * a_ii,
                    1.0 - b11 * a_ii,
                    f[ii * m + jj],
                    f[ii * m + jj + 1],
                )?;
                f[ii * m + jj] = y0;
                f[ii * m + jj + 1] = y1;
            } else {
                let (i0, i1) = (ii, ii + 1);
                let (j0, j1) = (jj, jj + 1);
                let (a00, a01) = (ta[i0 * n + i0], ta[i0 * n + i1]);
                let (a10, a11) = (ta[i1 * n + i0], ta[i1 * n + i1]);
                let (b00, b01) = (tb[j0 * m + j0], tb[j0 * m + j1]);
                let (b10, b11) = (tb[j1 * m + j0], tb[j1 * m + j1]);
                // (I - B^T kron A) vec(Y) = vec(F), vec order [y00, y10, y01, y11]
                let mat = [
                    1.0 - b00 * a00,
                    -b00 * a01,
                    -b10 * a00,
                    -b10 * a01,
                    -b00 * a10,
                    1.0 - b00 * a11,
                    -b10 * a10,
                    -b10 * a11,
                    -b01 * a00,
                    -b01 * a01,
                    1.0 - b11 * a00,
                    -b11 * a01,
                    -b01 * a10,
                    -b01 * a11,
                    -b11 * a10,
                    1.0 - b11 * a11,
                ];
                let rhs_vec = [
                    f[i0 * m + j0],
                    f[i1 * m + j0],
                    f[i0 * m + j1],
                    f[i1 * m + j1],
                ];
                let x = solve_4x4(&mat, &rhs_vec)?;
                f[i0 * m + j0] = x[0];
                f[i1 * m + j0] = x[1];
                f[i0 * m + j1] = x[2];
                f[i1 * m + j1] = x[3];
            }
        }
        jj += bj;
    }
    Ok(())
}

pub fn discrete_sylvester(
    a: &Tensor<f64, Cpu>,
    b: &Tensor<f64, Cpu>,
    c: &Tensor<f64, Cpu>,
) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    let m = b.nrows();
    require_square(a.shape(), "discrete_sylvester(A)")?;
    require_square(b.shape(), "discrete_sylvester(B)")?;
    ensure_shape(n, m, c, "discrete_sylvester(C)")?;

    if n == 0 || m == 0 {
        return Ok(Tensor::<f64, Cpu>::zeros(n, m));
    }

    let (ta_tensor, qa_tensor) = schur(a)?;
    let (tb_tensor, qb_tensor) = schur(b)?;

    let ta = to_f64_buf(&ta_tensor);
    let tb = to_f64_buf(&tb_tensor);

    // F = Q_A^T · C · Q_B
    let f_tensor = &(&qa_tensor.t() * c) * &qb_tensor;
    let mut f_buf = to_f64_buf(&f_tensor);

    solve_triangular_discrete_sylvester(&ta, &tb, &mut f_buf, n, m)?;

    // Y is now in f_buf; recover X = Q_A · Y · Q_B^T
    let y_tensor = from_f64_buf(f_buf, n, m);
    Ok(&(&qa_tensor * &y_tensor) * &qb_tensor.t())
}

pub fn discrete_lyapunov(a: &Tensor<f64, Cpu>, q: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    require_square(a.shape(), "discrete_lyapunov")?;
    let n = a.nrows();
    ensure_shape(n, n, q, "discrete_lyapunov(Q)")?;
    let at = a.t();
    discrete_sylvester(a, &at, q)
}
