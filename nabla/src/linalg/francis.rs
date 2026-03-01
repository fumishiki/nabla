// Francis implicit double-shift QR algorithm for non-symmetric eigenvalues.
//
// Pipeline: Hessenberg reduction -> Francis QR iteration -> Schur form eigenvalue extraction.

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::{
    buf_get, buf_set, from_f64_buf, householder_apply_left, householder_apply_right,
    householder_vec, to_f64_buf,
};

/// Reduce `a` to upper Hessenberg form H in-place (row-major flat buffer, n x n).
///
/// Applies Householder reflectors to columns below the subdiagonal and
/// accumulates the similarity transform Q so that `Q^T * A * Q = H`.
#[allow(clippy::many_single_char_names)]
fn hessenberg_reduce(buf: &mut [f64], n: usize) {
    for j in 0..(n.saturating_sub(2)) {
        // Column vector below sub-diagonal: rows j+1..n, column j
        let mut v: Vec<f64> = ((j + 1)..n).map(|i| buf_get(buf, n, i, j)).collect();
        let Some(tau) = householder_vec(&mut v) else {
            continue;
        };
        // H from left: rows j+1..n, all columns
        householder_apply_left(buf, n, j + 1, 0, n, &v, tau);
        // H from right: all rows, cols j+1..n
        householder_apply_right(buf, n, 0, n, j + 1, &v, tau);
    }
}

/// Perform one Francis implicit double-shift QR step on the active submatrix
/// `h[lo..hi+1, lo..hi+1]` of the Hessenberg matrix (row-major, size n x n).
///
/// Uses the double-shift (mu1, mu2) equal to the eigenvalues of the trailing 2x2
/// block and chases the resulting bulge from column `lo` to column `hi-1`.
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn francis_double_shift_step(h: &mut [f64], n: usize, lo: usize, hi: usize) {
    // Trailing 2x2 block eigenvalues give the double shift
    let h11 = buf_get(h, n, hi - 1, hi - 1);
    let h12 = buf_get(h, n, hi - 1, hi);
    let h21 = buf_get(h, n, hi, hi - 1);
    let h22 = buf_get(h, n, hi, hi);
    let s = h11 + h22; // trace
    let t = h11 * h22 - h12 * h21; // determinant

    // Bulge initialization: first column of (H - mu1*I)(H - mu2*I) = H^2 - s*H + t*I
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

    // Chase the bulge from column lo to hi-1
    let len = hi - lo + 1;
    for k in 0..len.saturating_sub(2) {
        let col = lo + k;
        // Build Householder reflector for [x, y, z] (or [x, y] at last step)
        let row_end = (col + 3).min(hi + 1);
        let bulge_len = row_end - col;
        let mut pv = if k == 0 {
            vec![x, y, z]
        } else {
            (col..row_end)
                .map(|r| buf_get(h, n, r, col - 1))
                .collect::<Vec<_>>()
        };
        // Trim to actual bulge size
        pv.truncate(bulge_len);

        let Some(tau) = householder_vec(&mut pv) else {
            continue;
        };

        // Apply H from left: rows col..row_end, cols col-1..n (but col-1 only for k>0)
        let apply_col_start = if k == 0 { col } else { col - 1 };
        householder_apply_left(h, n, col, apply_col_start, n, &pv, tau);
        // Apply H from right: all rows 0..hi+1, cols col..row_end
        householder_apply_right(h, n, 0, hi + 1, col, &pv, tau);

    }

    // Final 2x2 Givens cleanup: restore upper Hessenberg at bottom
    let second_last = hi - 1;
    let a = buf_get(h, n, second_last, second_last - 1);
    let b_val = buf_get(h, n, hi, second_last - 1);
    let r = a.hypot(b_val);
    if r > f64::EPSILON {
        let c = a / r;
        let s_g = b_val / r;
        // Apply Givens from left (rows second_last and hi)
        for col in (second_last - 1)..n {
            let v0 = buf_get(h, n, second_last, col);
            let v1 = buf_get(h, n, hi, col);
            buf_set(h, n, second_last, col, c * v0 + s_g * v1);
            buf_set(h, n, hi, col, -s_g * v0 + c * v1);
        }
        // Apply Givens from right (cols second_last and hi)
        for row in 0..=hi {
            let v0 = buf_get(h, n, row, second_last);
            let v1 = buf_get(h, n, row, hi);
            buf_set(h, n, row, second_last, c * v0 + s_g * v1);
            buf_set(h, n, row, hi, -s_g * v0 + c * v1);
        }
        buf_set(h, n, hi, second_last - 1, 0.0);
    }
}

/// Extract eigenvalues from Schur form (upper quasi-triangular, 1x1 and 2x2 blocks).
fn read_eigenvalues_from_schur(h: &[f64], n: usize) -> Vec<(f64, f64)> {
    let mut eigs = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && buf_get(h, n, i + 1, i).abs() > f64::EPSILON * 10.0 {
            // 2x2 block -> complex conjugate pair
            let a = buf_get(h, n, i, i);
            let b = buf_get(h, n, i, i + 1);
            let c = buf_get(h, n, i + 1, i);
            let d = buf_get(h, n, i + 1, i + 1);
            let tr = (a + d) * 0.5;
            let disc = (a - d) * (a - d) * 0.25 + b * c;
            if disc >= 0.0 {
                // Real pair from near-diagonal block
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
            // 1x1 block -> real eigenvalue
            eigs.push((buf_get(h, n, i, i), 0.0));
            i += 1;
        }
    }
    eigs
}

/// Run Francis implicit double-shift QR on Hessenberg matrix `h` (n x n, flat row-major).
///
/// Returns when all eigenvalues have converged or max iterations exceeded.
/// The matrix is modified in-place to Schur quasi-triangular form.
fn francis_qr_iterate(h: &mut [f64], n: usize) -> Result<()> {
    if n <= 1 {
        return Ok(());
    }

    let max_iter = 30 * n;
    let mut iter = 0;
    let mut hi = n - 1;

    while hi > 0 {
        if iter > max_iter {
            return Err(Error::invalid(
                "eig_into: Francis QR failed to converge",
            ));
        }

        // Deflate: check if sub-diagonal entry at (hi, hi-1) is negligible
        let tol = f64::EPSILON
            * (buf_get(h, n, hi - 1, hi - 1).abs() + buf_get(h, n, hi, hi).abs());
        if buf_get(h, n, hi, hi - 1).abs() <= tol {
            buf_set(h, n, hi, hi - 1, 0.0);
            hi = hi.saturating_sub(1);
            continue;
        }

        // Find lo: start of active unreduced Hessenberg block
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

        // 2x2 active block still uses the same Francis step.
        francis_double_shift_step(h, n, lo, hi);
        iter += 1;
    }

    Ok(())
}

/// Full pipeline: Hessenberg reduction -> Francis QR -> eigenvalue extraction.
pub(crate) fn francis_qr_schur(
    a: &Tensor<f64, Cpu>,
    n: usize,
) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>)> {
    let mut h = to_f64_buf(a);
    hessenberg_reduce(&mut h, n);
    francis_qr_iterate(&mut h, n)?;
    let eigs = read_eigenvalues_from_schur(&h, n);
    let schur = from_f64_buf(h, n, n);
    Ok((eigs, schur))
}
