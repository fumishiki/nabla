use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::LinalgExt as _;
use super::{from_f64_buf, householder_vec, require_square, to_f64_buf};

const PADE7: [f64; 8] = [
    1.0,
    0.5,
    0.12,
    1.833_333_333_333_333e-2,
    1.992_063_492_063_492e-3,
    1.630_434_782_608_696e-4,
    1.035_196_687_370_6e-5,
    5.175_983_561_643_836e-7,
];

pub fn expm(a: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    require_square(a.shape(), "expm")?;

    if n == 0 {
        return Ok(Tensor::<f64, Cpu>::zeros(0, 0));
    }

    // 1-norm: max column-sum of |A|
    let mut norm1 = 0.0_f64;
    for j in 0..n {
        let mut col_sum = 0.0_f64;
        for i in 0..n {
            col_sum += a.get(i, j).abs();
        }
        if col_sum > norm1 {
            norm1 = col_sum;
        }
    }

    // Scaling: s chosen so ||A/2^s|| <= theta_7 = 3.93 (Higham 2005, Table 10.2)
    let s = if norm1 <= 3.93 {
        0_u32
    } else {
        (norm1 / 3.93).log2().ceil() as u32
    };
    let scale = 0.5_f64.powi(s as i32);
    let a_s = a * scale;

    // Powers
    let a2 = &a_s * &a_s;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    let eye = Tensor::<f64, Cpu>::identity(n);

    // V = c[0]*I + c[2]*A2 + c[4]*A4 + c[6]*A6
    let v = &(&(&eye * PADE7[0]) + &(&a2 * PADE7[2])) + &(&(&a4 * PADE7[4]) + &(&a6 * PADE7[6]));

    // U_inner = c[1]*I + c[3]*A2 + c[5]*A4 + c[7]*A6
    let u_inner =
        &(&(&eye * PADE7[1]) + &(&a2 * PADE7[3])) + &(&(&a4 * PADE7[5]) + &(&a6 * PADE7[7]));
    let u = &a_s * &u_inner;

    // r = (V - U)^{-1} (V + U)
    let numer = &v + &u;
    let denom = &v - &u;
    let mut r = denom.solve(&numer)?;

    // Squaring
    for _ in 0..s {
        r = &r * &r;
    }

    Ok(r)
}

pub(crate) fn schur_hessenberg(a_buf: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut h = a_buf.to_vec();
    // Q starts as identity
    let mut q = {
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        v
    };

    let mut work = vec![0.0f64; n];

    for k in 0..n.saturating_sub(2) {
        // Householder reflector for column k, rows k+1..n
        let len = n - k - 1;
        for i in 0..len {
            work[i] = h[(k + 1 + i) * n + k];
        }
        let Some(tau) = householder_vec(&mut work[..len]) else {
            continue;
        };
        // Apply H from left: H[k+1..n, k..n]
        for j in k..n {
            let dot: f64 = (0..len).map(|i| work[i] * h[(k + 1 + i) * n + j]).sum();
            let s = tau * dot;
            for i in 0..len {
                h[(k + 1 + i) * n + j] -= s * work[i];
            }
        }
        // Apply H from right: H[0..n, k+1..n]
        for i in 0..n {
            let dot: f64 = (0..len).map(|j| work[j] * h[i * n + k + 1 + j]).sum();
            let s = tau * dot;
            for j in 0..len {
                h[i * n + k + 1 + j] -= s * work[j];
            }
        }
        // Accumulate into Q: Q[0..n, k+1..n]
        for i in 0..n {
            let dot: f64 = (0..len).map(|j| work[j] * q[i * n + k + 1 + j]).sum();
            let s = tau * dot;
            for j in 0..len {
                q[i * n + k + 1 + j] -= s * work[j];
            }
        }
    }
    (h, q)
}

fn schur_francis_step(h: &mut [f64], q: &mut [f64], n: usize, lo: usize, hi: usize) {
    // Trailing 2×2 block for shift
    let a11 = h[hi * n + hi];
    let a12 = h[(hi - 1) * n + hi];
    let a21 = h[hi * n + hi - 1];
    let a22 = h[(hi - 1) * n + hi - 1];
    let s = a22 + a11;
    let p = a22 * a11 - a12 * a21;

    // First column of (H^2 - s*H + p*I)
    let mut x = h[lo * n + lo] * h[lo * n + lo] - s * h[lo * n + lo]
        + p
        + h[lo * n + lo + 1] * h[(lo + 1) * n + lo];
    let mut y = h[(lo + 1) * n + lo] * (h[lo * n + lo] + h[(lo + 1) * n + lo + 1] - s);
    let mut z = if lo + 2 <= hi {
        h[(lo + 1) * n + lo] * h[(lo + 2) * n + lo + 1]
    } else {
        0.0
    };

    for k in lo..hi {
        let len = if k + 2 <= hi { 3usize } else { 2usize };
        let mut v = [x, y, z];
        let Some(tau) = householder_vec(&mut v[..len]) else {
            break;
        };
        // Apply from left: rows k..k+len, cols k-1..n (but only k.. on first)
        let col_start = if k == lo { k } else { k - 1 };
        for j in col_start..n {
            let dot: f64 = (0..len).map(|i| v[i] * h[(k + i) * n + j]).sum();
            let sc = tau * dot;
            for i in 0..len {
                h[(k + i) * n + j] -= sc * v[i];
            }
        }
        // Apply from right: rows 0..hi+1, cols k..k+len
        for i in 0..=hi {
            let dot: f64 = (0..len).map(|j| v[j] * h[i * n + k + j]).sum();
            let sc = tau * dot;
            for j in 0..len {
                h[i * n + k + j] -= sc * v[j];
            }
        }
        // Accumulate into Q
        for i in 0..n {
            let dot: f64 = (0..len).map(|j| v[j] * q[i * n + k + j]).sum();
            let sc = tau * dot;
            for j in 0..len {
                q[i * n + k + j] -= sc * v[j];
            }
        }
        // Update bulge for next iteration
        if k + 1 < hi {
            x = h[(k + 1) * n + k];
            y = h[(k + 2) * n + k];
            z = if k + 3 <= hi { h[(k + 3) * n + k] } else { 0.0 };
        }
    }
}

pub fn schur(a: &Tensor<f64, Cpu>) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
    let n = a.nrows();
    require_square(a.shape(), "schur")?;

    match n {
        0 => {
            return Ok((
                Tensor::<f64, Cpu>::zeros(0, 0),
                Tensor::<f64, Cpu>::zeros(0, 0),
            ));
        }
        1 => return Ok((a.clone(), Tensor::<f64, Cpu>::identity(1))),
        _ => {}
    }

    let a_buf = to_f64_buf(a);
    let (mut h, mut q) = schur_hessenberg(&a_buf, n);

    let max_iter = 30 * n * n;
    let eps = f64::EPSILON * 4.0;
    let mut total_iter = 0usize;
    let mut hi = n - 1;

    'outer: loop {
        if hi == 0 {
            break;
        }
        // Deflation: check sub-diagonal
        let mut lo = hi;
        while lo > 0 {
            let sub = h[lo * n + lo - 1].abs();
            let scale = h[(lo - 1) * n + lo - 1].abs() + h[lo * n + lo].abs();
            if sub <= eps * scale.max(f64::EPSILON) {
                h[lo * n + lo - 1] = 0.0;
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            // 1×1 block converged
            if hi == 0 {
                break 'outer;
            }
            hi -= 1;
        } else if lo + 1 == hi {
            // 2×2 block converged
            if hi <= 1 {
                break 'outer;
            }
            hi -= 2;
        } else {
            // Run Francis step
            if total_iter >= max_iter {
                return Err(Error::invalid(format!(
                    "schur: QR iteration did not converge for {n}x{n} matrix after {max_iter} steps"
                )));
            }
            schur_francis_step(&mut h, &mut q, n, lo, hi);
            total_iter += 1;
        }
    }

    Ok((from_f64_buf(h, n, n), from_f64_buf(q, n, n)))
}

fn parlett_recurrence(
    t: &[f64],
    n: usize,
    block_sizes: &[usize],
    f_blocks: &[Vec<f64>],
    f_deriv: Option<&dyn Fn(f64) -> f64>,
) -> Vec<f64> {
    let mut f = vec![0.0f64; n * n];
    // Place diagonal blocks into f
    let mut pos = 0usize;
    for (bi, &bs) in block_sizes.iter().enumerate() {
        for r in 0..bs {
            for c in 0..bs {
                f[(pos + r) * n + pos + c] = f_blocks[bi][r * bs + c];
            }
        }
        pos += bs;
    }
    // Build block start indices
    let nb = block_sizes.len();
    let mut bstart = Vec::with_capacity(nb);
    let mut s = 0usize;
    for &bs in block_sizes {
        bstart.push(s);
        s += bs;
    }
    // Fill off-diagonal blocks using divided-difference recurrence
    let tol = f64::EPSILON * 1e8;
    for d in 1..nb {
        for bi in (0..nb - d).rev() {
            let bj = bi + d;
            let si = bstart[bi];
            let sj = bstart[bj];
            let bsi = block_sizes[bi];
            let bsj = block_sizes[bj];
            // Compute cross terms: sum_{k=bi+1..bj-1} F[i,k]*T[k,j] - T[i,k]*F[k,j]
            let mut rhs = vec![0.0f64; bsi * bsj];
            for bk in (bi + 1)..bj {
                let sk = bstart[bk];
                let bsk = block_sizes[bk];
                for r in 0..bsi {
                    for c in 0..bsj {
                        let mut val = 0.0;
                        for p in 0..bsk {
                            val += f[(si + r) * n + sk + p] * t[(sk + p) * n + sj + c]
                                - t[(si + r) * n + sk + p] * f[(sk + p) * n + sj + c];
                        }
                        rhs[r * bsj + c] += val;
                    }
                }
            }
            // Add (F_jj - F_ii) * T_ij contribution via Sylvester equation
            // Solve F_ii * X - X * F_jj = -(rhs + F_ii*T_ij - T_ij*F_jj) per Parlett
            // For scalar (1x1,1x1): x = (rhs + (f_j - f_i)*t_ij) / (lam_j - lam_i)
            if bsi == 1 && bsj == 1 {
                let lam_i = t[si * n + si];
                let lam_j = t[sj * n + sj];
                let fi = f[si * n + si];
                let fj = f[sj * n + sj];
                let t_ij = t[si * n + sj];
                let num = rhs[0] + (fj - fi) * t_ij;
                let denom = lam_j - lam_i;
                f[si * n + sj] = if denom.abs() < tol {
                    // L'Hopital: f'(lam_i) * t_ij + cross/regularized
                    match f_deriv {
                        Some(fd) => fd(lam_i) * t_ij + rhs[0] / (denom.abs() + f64::EPSILON),
                        None => num / (denom + f64::EPSILON.sqrt()),
                    }
                } else {
                    num / denom
                };
            } else {
                // General Sylvester: F_ii * X - X * F_jj = -RHS' where RHS' includes T_ij terms
                // Build right-hand side: -(rhs + F_ii * T_ij - T_ij * F_jj)
                let mut neg_rhs = vec![0.0f64; bsi * bsj];
                for r in 0..bsi {
                    for c in 0..bsj {
                        let mut fi_tij = 0.0;
                        for p in 0..bsi {
                            fi_tij += f[(si + r) * n + si + p] * t[(si + p) * n + sj + c];
                        }
                        let mut tij_fj = 0.0;
                        for p in 0..bsj {
                            tij_fj += t[(si + r) * n + sj + p] * f[(sj + p) * n + sj + c];
                        }
                        neg_rhs[r * bsj + c] = -(rhs[r * bsj + c] + fi_tij - tij_fj);
                    }
                }
                // Solve via vectorized Kronecker: (I_j kron F_ii - F_jj^T kron I_i) vec(X) = vec(-RHS)
                let dim = bsi * bsj;
                let mut mat = vec![0.0f64; dim * dim];
                for rr in 0..bsi {
                    for cc in 0..bsj {
                        let row = rr * bsj + cc;
                        // I_j kron F_ii contribution
                        for p in 0..bsi {
                            mat[row * dim + p * bsj + cc] += f[(si + rr) * n + si + p];
                        }
                        // -F_jj^T kron I_i contribution
                        for p in 0..bsj {
                            mat[row * dim + rr * bsj + p] -= f[(sj + cc) * n + sj + p];
                        }
                    }
                }
                // Solve small system (2x2 to 4x4) via Gaussian elimination
                let sol = solve_small_system(&mut mat, &neg_rhs, dim);
                for r in 0..bsi {
                    for c in 0..bsj {
                        f[(si + r) * n + sj + c] = sol[r * bsj + c];
                    }
                }
            }
        }
    }
    f
}

fn solve_small_system(a: &mut [f64], b: &[f64], dim: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    let mut piv: Vec<usize> = (0..dim).collect();
    for col in 0..dim {
        // Partial pivot
        let mut max_val = a[piv[col] * dim + col].abs();
        let mut max_row = col;
        for row in (col + 1)..dim {
            let v = a[piv[row] * dim + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        piv.swap(col, max_row);
        let pivot = a[piv[col] * dim + col];
        if pivot.abs() < f64::EPSILON * 1e4 {
            continue;
        }
        for row in (col + 1)..dim {
            let factor = a[piv[row] * dim + col] / pivot;
            for k in (col + 1)..dim {
                a[piv[row] * dim + k] -= factor * a[piv[col] * dim + k];
            }
            x[piv[row]] -= factor * x[piv[col]];
        }
    }
    // Back substitution
    let mut result = vec![0.0f64; dim];
    for col in (0..dim).rev() {
        let mut s = x[piv[col]];
        for k in (col + 1)..dim {
            s -= a[piv[col] * dim + k] * result[k];
        }
        let pivot = a[piv[col] * dim + col];
        result[col] = if pivot.abs() < f64::EPSILON * 1e4 {
            0.0
        } else {
            s / pivot
        };
    }
    result
}

#[allow(clippy::too_many_lines)]
pub fn logm(a: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    require_square(a.shape(), "logm")?;

    match n {
        0 => return Ok(Tensor::<f64, Cpu>::zeros(0, 0)),
        1 => {
            let d = a.get(0, 0);
            if d <= 0.0 {
                return Err(Error::invalid(format!(
                    "logm: eigenvalue {d} <= 0; logarithm undefined"
                )));
            }
            return Ok(from_f64_buf(vec![d.ln()], 1, 1));
        }
        _ => {}
    }

    let (t_tensor, q_tensor) = schur(a)?;
    let t = to_f64_buf(&t_tensor);

    // Detect block structure from quasi-upper-triangular Schur form
    let eps = f64::EPSILON * 4.0;
    let mut block_sizes = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n
            && t[(i + 1) * n + i].abs()
                > eps * (t[i * n + i].abs() + t[(i + 1) * n + i + 1].abs()).max(eps)
        {
            block_sizes.push(2);
            i += 2;
        } else {
            block_sizes.push(1);
            i += 1;
        }
    }

    // Compute f(block) for each diagonal block
    let mut f_blocks = Vec::with_capacity(block_sizes.len());
    let mut pos = 0usize;
    for &bs in &block_sizes {
        if bs == 1 {
            let d = t[pos * n + pos];
            if d <= 0.0 {
                return Err(Error::invalid(format!(
                    "logm: eigenvalue {d} <= 0 at index {pos}; logarithm undefined"
                )));
            }
            f_blocks.push(vec![d.ln()]);
        } else {
            // 2x2 block [[a,b],[c,d]]: eigenvalues = alpha +/- sqrt(disc)
            let a00 = t[pos * n + pos];
            let a01 = t[pos * n + pos + 1];
            let a10 = t[(pos + 1) * n + pos];
            let a11 = t[(pos + 1) * n + pos + 1];
            let alpha = (a00 + a11) * 0.5;
            let half_diff = (a00 - a11) * 0.5;
            let disc = half_diff * half_diff + a01 * a10;
            if disc < 0.0 {
                // Complex conjugate pair: mu = alpha +/- i*beta
                let beta = (-disc).sqrt();
                let modulus = (alpha * alpha + beta * beta).sqrt();
                if modulus <= 0.0 {
                    return Err(Error::invalid(format!(
                        "logm: zero eigenvalue at 2x2 block index {pos}"
                    )));
                }
                let ln_mod = modulus.ln();
                let theta = beta.atan2(alpha);
                // log(M) = ln|mu|*I + (theta/beta)*(M - alpha*I)
                let s = theta / beta;
                f_blocks.push(vec![
                    ln_mod + s * (a00 - alpha),
                    s * a01,
                    s * a10,
                    ln_mod + s * (a11 - alpha),
                ]);
            } else {
                // Real distinct eigenvalues
                let sq = disc.sqrt();
                let lam1 = alpha + sq;
                let lam2 = alpha - sq;
                if lam1 <= 0.0 || lam2 <= 0.0 {
                    return Err(Error::invalid(format!(
                        "logm: non-positive eigenvalue in 2x2 block at index {pos}"
                    )));
                }
                let ln1 = lam1.ln();
                let ln2 = lam2.ln();
                if sq.abs() < eps {
                    // Nearly equal: log(M) ~ ln(alpha)*I + (1/alpha)*(M - alpha*I)
                    let inv_a = 1.0 / alpha;
                    f_blocks.push(vec![
                        alpha.ln() + inv_a * (a00 - alpha),
                        inv_a * a01,
                        inv_a * a10,
                        alpha.ln() + inv_a * (a11 - alpha),
                    ]);
                } else {
                    // Sylvester formula: f(M) = ((M - lam2*I)*f(lam1) - (M - lam1*I)*f(lam2)) / (lam1 - lam2)
                    let d = lam1 - lam2;
                    f_blocks.push(vec![
                        ((a00 - lam2) * ln1 - (a00 - lam1) * ln2) / d,
                        (a01 * ln1 - a01 * ln2) / d,
                        (a10 * ln1 - a10 * ln2) / d,
                        ((a11 - lam2) * ln1 - (a11 - lam1) * ln2) / d,
                    ]);
                }
            }
        }
        pos += bs;
    }

    let f_deriv = |x: f64| -> f64 { 1.0 / x };
    let log_t_buf = parlett_recurrence(&t, n, &block_sizes, &f_blocks, Some(&f_deriv));
    let log_t = from_f64_buf(log_t_buf, n, n);

    // Reconstruct: Q · log(T) · Q^T
    Ok(&(&q_tensor * &log_t) * &q_tensor.t())
}

pub fn sqrtm(a: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    require_square(a.shape(), "sqrtm")?;

    match n {
        0 => return Ok(Tensor::<f64, Cpu>::zeros(0, 0)),
        1 => {
            let d = a.get(0, 0);
            if d < 0.0 {
                return Err(Error::invalid(format!(
                    "sqrtm: negative eigenvalue {d}; square root undefined over the reals"
                )));
            }
            return Ok(from_f64_buf(vec![d.sqrt()], 1, 1));
        }
        _ => {}
    }

    let eye = Tensor::<f64, Cpu>::identity(n);
    let mut y = a.clone();
    let mut z = eye.clone();

    let tol = (n as f64) * f64::EPSILON;
    let max_iter = 50usize;

    for _ in 0..max_iter {
        let z_inv = z
            .solve(&eye)
            .map_err(|e| Error::invalid(format!("sqrtm: Z inversion failed — {e}")))?;
        let y_inv = y
            .solve(&eye)
            .map_err(|e| Error::invalid(format!("sqrtm: Y inversion failed — {e}")))?;

        let y_new = &(&y + &z_inv) * 0.5_f64;
        let z_new = &(&z + &y_inv) * 0.5_f64;

        // Convergence: ||Y_{k+1} - Y_k||_F / max(||Y_k||_F, 1)
        let diff_buf = to_f64_buf(&(&y_new - &y));
        let y_buf = to_f64_buf(&y);
        let diff_norm: f64 = diff_buf.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let y_norm: f64 = y_buf.iter().map(|&x| x * x).sum::<f64>().sqrt();

        y = y_new;
        z = z_new;

        if diff_norm <= tol * y_norm.max(1.0) {
            return Ok(y);
        }
    }

    Err(Error::invalid(format!(
        "sqrtm: Denman-Beavers iteration did not converge for {n}×{n} matrix after {max_iter} steps"
    )))
}
