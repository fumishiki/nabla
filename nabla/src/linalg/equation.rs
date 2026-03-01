// Matrix equations: sylvester, lyapunov, solve_tridiag.

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::{from_f64_buf, require_square, to_f64_buf};
use super::matrix_fn::schur;

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

// ---------------------------------------------------------------------------
// sylvester — Solve AX + XB = C via Bartels-Stewart algorithm
// ---------------------------------------------------------------------------

/// Back-substitution for the triangular Sylvester equation `T_A · Y + Y · T_B = F`.
///
/// `T_A` (n×n) and `T_B` (m×m) are upper triangular; `f` (n×m) holds the
/// right-hand side on entry and the solution `Y` on exit.
fn solve_triangular_sylvester(
    ta: &[f64],
    tb: &[f64],
    f: &mut [f64],
    n: usize,
    m: usize,
) -> Result<()> {
    // Process columns of Y left to right
    for j in 0..m {
        // Subtract contributions of already-solved columns: sum_{k<j} T_B[k,j] * Y_col_k
        for k in 0..j {
            let tb_kj = tb[k * m + j];
            if tb_kj.abs() < f64::EPSILON {
                continue;
            }
            for i in 0..n {
                f[i * m + j] -= tb_kj * f[i * m + k];
            }
        }
        // Solve (T_A + T_B[j,j] * I) * y_j = f_col_j  (upper triangular back-sub)
        let shift = tb[j * m + j];
        for i in (0..n).rev() {
            let mut rhs = f[i * m + j];
            for k in (i + 1)..n {
                rhs -= ta[i * n + k] * f[k * m + j];
            }
            let diag = ta[i * n + i] + shift;
            if diag.abs() < f64::EPSILON * 1e6 {
                return Err(Error::invalid(format!(
                    "sylvester: near-singular diagonal {diag:.3e} at position ({i},{j}); \
                     A and -B may share an eigenvalue"
                )));
            }
            f[i * m + j] = rhs / diag;
        }
    }
    Ok(())
}

/// Solve the Sylvester equation `AX + XB = C` via the Bartels-Stewart algorithm.
///
/// Steps:
/// 1. Schur-decompose `A = Q_A·T_A·Q_A^T` and `B = Q_B·T_B·Q_B^T`.
/// 2. Reduce to the triangular problem `T_A·Y + Y·T_B = Q_A^T·C·Q_B`.
/// 3. Solve column-by-column via back-substitution.
/// 4. Recover `X = Q_A·Y·Q_B^T`.
///
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if dimensions are inconsistent, Schur decompositions fail to
/// converge, or `A` and `-B` share an eigenvalue (singular coefficient operator).
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

// ---------------------------------------------------------------------------
// lyapunov — Solve AX + XAᵀ = Q (special case of Sylvester)
// ---------------------------------------------------------------------------

/// Solve the continuous Lyapunov equation `AX + XA^T = Q` (SciPy convention).
///
/// **Convention note**: MATLAB's `lyap(A, Q)` solves `AX + XA^T + Q = 0`,
/// which is equivalent to passing `-Q` to this function.
///
/// This is the Sylvester equation `AX + XB = Q` with `B = A^T`.
/// Delegates to [`sylvester`].  Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square, `Q` has the wrong shape, the Schur
/// decomposition fails to converge, or `A` has a purely imaginary eigenvalue
/// pair (which makes the coefficient operator singular).
pub fn lyapunov(a: &Tensor<f64, Cpu>, q: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
    require_square(a.shape(), "lyapunov")?;
    let n = a.nrows();
    ensure_shape(n, n, q, "lyapunov(Q)")?;
    let at = a.t();
    sylvester(a, &at, q)
}

// ---------------------------------------------------------------------------
// solve_tridiag — Thomas algorithm  O(n) tridiagonal solver
// ---------------------------------------------------------------------------

/// Solve a tridiagonal system `T·x = rhs` using the Thomas algorithm (O(n)).
///
/// * `lower`  — sub-diagonal, length `n - 1` (elements `T[i, i-1]` for `i = 1..n`).
/// * `main`   — main diagonal, length `n`.
/// * `upper`  — super-diagonal, length `n - 1` (elements `T[i, i+1]` for `i = 0..n-1`).
/// * `rhs`    — right-hand side, length `n`.
///
/// Returns the solution vector `x` of length `n`.
///
/// # Errors
///
/// Returns `Err` if the lengths are inconsistent or a zero pivot is encountered
/// during the forward sweep (singular or near-singular tridiagonal matrix).
pub fn solve_tridiag(
    lower: &[f64],
    main: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>> {
    let n = main.len();
    if lower.len() != n.saturating_sub(1)
        || upper.len() != n.saturating_sub(1)
        || rhs.len() != n
    {
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

// ---------------------------------------------------------------------------
// discrete_sylvester — Solve AXB - X + C = 0 via Bartels-Stewart
// ---------------------------------------------------------------------------

/// Back-substitution for the triangular discrete Sylvester equation
/// `T_A · Y · T_B - Y + F = 0`, i.e. `Y = T_A · Y · T_B + F`.
///
/// `T_A` (n x n) and `T_B` (m x m) are upper triangular (real Schur form).
/// `f` (n x m) holds the transformed RHS on entry and the solution `Y` on exit.
///
/// Solves column-by-column from left to right, within each column from bottom to top.
fn solve_triangular_discrete_sylvester(
    ta: &[f64],
    tb: &[f64],
    f: &mut [f64],
    n: usize,
    m: usize,
) -> Result<()> {
    // For each column j of Y, the equation for Y[:,j] is:
    //   Y[:,j] = T_A * (sum_{k<=j} T_B[k,j] * Y[:,k]) + F[:,j]
    //
    // Rearranging for column j with known columns k<j:
    //   Y[:,j] - T_B[j,j] * T_A * Y[:,j] = T_A * (sum_{k<j} T_B[k,j] * Y[:,k]) + F[:,j]
    //   (I - T_B[j,j] * T_A) * Y[:,j] = RHS
    //
    // Since T_A is upper triangular, (I - T_B[j,j]*T_A) is also upper triangular.
    // Solve via back-substitution from row n-1 down to 0.

    for j in 0..m {
        // Subtract contributions of already-solved columns k < j.
        for k in 0..j {
            let tb_kj = tb[k * m + j];
            if tb_kj.abs() < f64::EPSILON {
                continue;
            }
            // Add T_A * Y[:,k] * T_B[k,j] to f[:,j]
            for i in 0..n {
                // (T_A * Y[:,k])[i] = sum_r T_A[i,r] * Y[r,k]
                let mut ta_y_k_i = 0.0;
                for r in i..n {
                    ta_y_k_i += ta[i * n + r] * f[r * m + k];
                }
                f[i * m + j] += tb_kj * ta_y_k_i;
            }
        }

        // Solve (I - T_B[j,j] * T_A) * y_j = f_col_j via back-sub.
        let shift = tb[j * m + j];
        for i in (0..n).rev() {
            let mut rhs = f[i * m + j];
            // Subtract known entries from rows below: (I - shift*T_A)[i,k] * y[k]
            for k in (i + 1)..n {
                rhs -= (-shift) * ta[i * n + k] * f[k * m + j];
            }
            let diag = 1.0 - shift * ta[i * n + i];
            if diag.abs() < f64::EPSILON * 1e6 {
                return Err(Error::invalid(format!(
                    "discrete_sylvester: near-singular at ({i},{j}); \
                     spectral radii may violate solvability"
                )));
            }
            f[i * m + j] = rhs / diag;
        }
    }
    Ok(())
}

/// Solve the discrete Sylvester equation `AXB - X + C = 0` via Bartels-Stewart.
///
/// Steps:
/// 1. Schur-decompose `A = Q_A·T_A·Q_A^T` and `B = Q_B·T_B·Q_B^T`.
/// 2. Transform to `T_A·Y·T_B - Y + F = 0` where `Y = Q_A^T·X·Q_B`, `F = Q_A^T·C·Q_B`.
/// 3. Solve the triangular system column-by-column.
/// 4. Recover `X = Q_A·Y·Q_B^T`.
///
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if dimensions are inconsistent, Schur decompositions fail, or
/// the equation is singular (product of eigenvalues of A and B equals 1).
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

// ---------------------------------------------------------------------------
// discrete_lyapunov — Solve AXA^T - X + Q = 0
// ---------------------------------------------------------------------------

/// Solve the discrete Lyapunov equation `AXA^T - X + Q = 0`.
///
/// This is the discrete Sylvester equation `AXB - X + C = 0` with `B = A^T`
/// and `C = Q`. Delegates to [`discrete_sylvester`].
///
/// Only `f64` / `Cpu` backend.
///
/// # Errors
///
/// Returns `Err` if `A` is not square, `Q` has the wrong shape, Schur
/// decomposition fails, or the equation is singular.
pub fn discrete_lyapunov(
    a: &Tensor<f64, Cpu>,
    q: &Tensor<f64, Cpu>,
) -> Result<Tensor<f64, Cpu>> {
    require_square(a.shape(), "discrete_lyapunov")?;
    let n = a.nrows();
    ensure_shape(n, n, q, "discrete_lyapunov(Q)")?;
    let at = a.t();
    discrete_sylvester(a, &at, q)
}
