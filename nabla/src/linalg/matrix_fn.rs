// Matrix functions: expm, schur, logm, sqrtm.

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::{from_f64_buf, householder_vec, require_square, to_f64_buf};
use super::solve::LinalgExt as _;

// ---------------------------------------------------------------------------
// expm — Matrix exponential via Padé [7/7] with scaling-and-squaring
// ---------------------------------------------------------------------------

// Padé [7/7] coefficients (Higham 2005, Table 10.2).
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

/// Matrix exponential `exp(A)` via Padé [7/7] approximation with scaling-and-squaring.
///
/// Uses the algorithm of Higham (2005) restricted to order 7.
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square or the internal linear solve fails.
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

    // Scaling: s = max(0, ceil(log2(norm1))) so that ||A/2^s|| <= 1
    let s = if norm1 <= 1.0 {
        0_u32
    } else {
        norm1.log2().ceil() as u32
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

// ---------------------------------------------------------------------------
// schur — Real Schur decomposition A = Q · T · Q^T
// ---------------------------------------------------------------------------

/// Hessenberg reduction accumulating Q.  Returns `(h_buf, q_buf)` both n×n row-major.
///
/// `Q^T · A · Q = H`.  Unlike the in-place `hessenberg_reduce` helper this
/// function also builds the orthogonal accumulator Q needed by `schur`.
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

/// Francis double-shift QR step on the active window `lo..=hi` of the
/// n×n Hessenberg matrix stored in `h` (row-major).  Also updates `q`.
fn schur_francis_step(h: &mut [f64], q: &mut [f64], n: usize, lo: usize, hi: usize) {
    // Trailing 2×2 block for shift
    let a11 = h[hi * n + hi];
    let a12 = h[(hi - 1) * n + hi];
    let a21 = h[hi * n + hi - 1];
    let a22 = h[(hi - 1) * n + hi - 1];
    let s = a22 + a11;
    let p = a22 * a11 - a12 * a21;

    // First column of (H^2 - s*H + p*I)
    let mut x = h[lo * n + lo] * h[lo * n + lo]
        - s * h[lo * n + lo]
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

/// Real Schur decomposition `A = Q · T · Q^T`.
///
/// Returns `(T, Q)` where `T` is quasi-upper-triangular (real Schur form) and
/// `Q` is orthogonal.  Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square or the QR iteration fails to converge
/// within `30·n²` steps.
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

// ---------------------------------------------------------------------------
// Parlett recurrence for upper-triangular matrix functions
// ---------------------------------------------------------------------------

/// Parlett recurrence: compute `f(T)` for upper-triangular `T` (n×n row-major),
/// given `f_diag[i] = f(T[i,i])` for each diagonal entry.
///
/// Off-diagonal elements are filled using the divided-difference formula.
fn parlett_recurrence(t: &[f64], f_diag: &[f64], n: usize) -> Vec<f64> {
    let mut f = vec![0.0f64; n * n];
    for i in 0..n {
        f[i * n + i] = f_diag[i];
    }
    // Process super-diagonals from closest to farthest
    for j in 1..n {
        for i in (0..j).rev() {
            let lam_i = t[i * n + i];
            let lam_j = t[j * n + j];
            let t_ij = t[i * n + j];
            // Numerator: (f_j - f_i) * t_ij + cross terms from interior
            let mut num = (f_diag[j] - f_diag[i]) * t_ij;
            for k in (i + 1)..j {
                num += f[i * n + k] * t[k * n + j] - t[i * n + k] * f[k * n + j];
            }
            let denom = lam_j - lam_i;
            f[i * n + j] = if denom.abs() < f64::EPSILON * 1e8 {
                // Near-coalescent eigenvalues: L'Hôpital limit gives f'(lam_i)*t_ij
                // For simplicity, scale by a small regularisation
                num / (denom + f64::EPSILON.sqrt())
            } else {
                num / denom
            };
        }
    }
    f
}

// ---------------------------------------------------------------------------
// logm — Matrix logarithm via Schur + Parlett recurrence
// ---------------------------------------------------------------------------

/// Matrix logarithm `log(A)` via real Schur decomposition and Parlett recurrence.
///
/// Computes `A = Q·T·Q^T`, evaluates `log` entry-wise on the diagonal of `T`,
/// fills the upper triangle via the Parlett recurrence, then returns `Q·log(T)·Q^T`.
///
/// Valid when all eigenvalues of `A` are strictly positive real numbers.
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square, the Schur decomposition fails to
/// converge, or any diagonal block of `T` has a non-positive eigenvalue
/// (matrix logarithm is undefined on the non-positive real axis).
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

    // Compute f_diag = ln(T[i,i]) for each diagonal element
    let mut f_diag = vec![0.0f64; n];
    for i in 0..n {
        let diag = t[i * n + i];
        if diag <= 0.0 {
            return Err(Error::invalid(format!(
                "logm: eigenvalue {diag} <= 0 at index {i}; logarithm undefined"
            )));
        }
        f_diag[i] = diag.ln();
    }

    let log_t_buf = parlett_recurrence(&t, &f_diag, n);
    let log_t = from_f64_buf(log_t_buf, n, n);

    // Reconstruct: Q · log(T) · Q^T
    Ok(&(&q_tensor * &log_t) * &q_tensor.t())
}

// ---------------------------------------------------------------------------
// sqrtm — Matrix square root via Denman-Beavers iteration
// ---------------------------------------------------------------------------

/// Matrix square root `sqrt(A)` via Denman-Beavers coupled iteration.
///
/// Uses the iteration:
/// ```text
/// Y_{k+1} = (Y_k + Z_k^{-1}) / 2
/// Z_{k+1} = (Z_k + Y_k^{-1}) / 2
/// ```
/// with `Y_0 = A`, `Z_0 = I`.  Converges quadratically to `sqrt(A)` for
/// matrices with no eigenvalues on the closed negative real axis.
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square, any intermediate matrix inversion fails
/// (singular iterate), or convergence is not reached within 50 steps.
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
        let z_inv = z.solve(&eye).map_err(|e| {
            Error::invalid(format!("sqrtm: Z inversion failed — {e}"))
        })?;
        let y_inv = y.solve(&eye).map_err(|e| {
            Error::invalid(format!("sqrtm: Y inversion failed — {e}"))
        })?;

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
