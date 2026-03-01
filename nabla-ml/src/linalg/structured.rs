use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::tensor::Tensor;

use super::lu::PartialPivLu;
use super::matrix_fn::{schur, schur_hessenberg};
use super::svd::Svd;
use super::{buf_get, from_f64_buf, require_square, to_f64_buf};

pub fn hessenberg(a: &Tensor<f64, Cpu>) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
    let n = a.nrows();
    require_square(a.shape(), "hessenberg")?;

    match n {
        0 => {
            return Ok((
                Tensor::<f64, Cpu>::zeros(0, 0),
                Tensor::<f64, Cpu>::zeros(0, 0),
            ));
        }
        1 => return Ok((Tensor::<f64, Cpu>::identity(1), a.clone())),
        _ => {}
    }

    // Delegate to the internal helper which builds both H and Q.
    let a_buf = to_f64_buf(a);
    let (h_buf, q_buf) = schur_hessenberg(&a_buf, n);

    Ok((from_f64_buf(q_buf, n, n), from_f64_buf(h_buf, n, n)))
}

pub fn polar(a: &Tensor<f64, Cpu>) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
    let (m, n) = a.shape();
    require_square(a.shape(), "polar")?;

    let svd = Svd::factorize(a)?;
    let u_svd = svd.u();
    let vt = svd.vt();
    let s = svd.s();

    // U_polar = U_svd · V^T
    let u_polar = u_svd * vt;

    // Build Σ as diagonal tensor (n×n)
    let sigma = Tensor::from_fn(m, n, |i, j| if i == j { s[i] } else { 0.0 });

    // H = V · Σ · V^T  (V = Vt^T)
    let v = vt.t();
    let h = &(&v * &sigma) * vt;

    Ok((u_polar, h))
}

#[must_use]
pub fn toeplitz(col: &[f64], row: &[f64]) -> Tensor<f64, Cpu> {
    let m = col.len();
    let n = row.len();
    if m == 0 || n == 0 {
        return Tensor::<f64, Cpu>::zeros(m, n);
    }
    Tensor::from_fn(m, n, |i, j| if i >= j { col[i - j] } else { row[j - i] })
}

#[must_use]
pub fn circulant(c: &[f64]) -> Tensor<f64, Cpu> {
    let n = c.len();
    if n == 0 {
        return Tensor::<f64, Cpu>::zeros(0, 0);
    }
    Tensor::from_fn(n, n, |i, j| {
        let idx = (i as isize - j as isize).rem_euclid(n as isize) as usize;
        c[idx]
    })
}

#[must_use]
pub fn vandermonde(nodes: &[f64]) -> Tensor<f64, Cpu> {
    let m = nodes.len();
    if m == 0 {
        return Tensor::<f64, Cpu>::zeros(0, 0);
    }
    Tensor::from_fn(m, m, |i, j| nodes[i].powi(j as i32))
}

pub fn vandermonde_rect(nodes: &[f64], ncols: usize) -> Result<Tensor<f64, Cpu>> {
    let m = nodes.len();
    if m == 0 {
        return Ok(Tensor::<f64, Cpu>::zeros(0, ncols));
    }
    if ncols == 0 {
        return Err(Error::invalid("vandermonde_rect: ncols must be > 0"));
    }
    Ok(Tensor::from_fn(m, ncols, |i, j| nodes[i].powi(j as i32)))
}

pub fn balance(a: &Tensor<f64, Cpu>) -> Result<(Tensor<f64, Cpu>, Vec<f64>)> {
    let n = a.nrows();
    require_square(a.shape(), "balance")?;

    let mut buf = to_f64_buf(a);
    let mut scale = vec![1.0f64; n];

    let beta = 2.0f64; // scaling base (power of 2)
    let eps = f64::EPSILON;

    loop {
        let mut converged = true;

        for i in 0..n {
            // Row norm (excluding diagonal)
            let row_norm: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| buf_get(&buf, n, i, j).abs())
                .sum();
            // Col norm (excluding diagonal)
            let col_norm: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| buf_get(&buf, n, j, i).abs())
                .sum();

            if row_norm < eps || col_norm < eps {
                continue;
            }

            // Find power of beta that brings row_norm ≈ col_norm.
            let mut s = 1.0f64;
            let mut r = row_norm;
            let mut c = col_norm;

            // Scale up: while row >> col, multiply column i by beta.
            while c < r / beta {
                s *= beta;
                c *= beta;
                r /= beta;
            }
            // Scale down: while col >> row, divide column i by beta.
            while c > r * beta {
                s /= beta;
                c /= beta;
                r *= beta;
            }

            // Accept if meaningful change.
            if (row_norm + col_norm) * 0.95 > r + c {
                converged = false;
                scale[i] *= s;

                // Apply: row i /= s, col i *= s.
                for j in 0..n {
                    buf[i * n + j] /= s;
                }
                for j in 0..n {
                    buf[j * n + i] *= s;
                }
            }
        }

        if converged {
            break;
        }
    }

    Ok((from_f64_buf(buf, n, n), scale))
}

pub fn frechet_deriv<F>(
    f: F,
    a: &Tensor<f64, Cpu>,
    e: &Tensor<f64, Cpu>,
) -> Result<Tensor<f64, Cpu>>
where
    F: Fn(&Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>>,
{
    let n = a.nrows();
    require_square(a.shape(), "frechet_deriv(A)")?;
    require_square(e.shape(), "frechet_deriv(E)")?;
    if a.shape() != e.shape() {
        return Err(Error::invalid(format!(
            "frechet_deriv: A and E must have the same shape, got {:?} vs {:?}",
            a.shape(),
            e.shape()
        )));
    }

    // Build 2n × 2n block matrix [[A, E], [0, A]].
    let two_n = 2 * n;
    let block = Tensor::from_fn(two_n, two_n, |i, j| {
        if i < n && j < n {
            a.get(i, j) // top-left: A
        } else if i < n && j >= n {
            e.get(i, j - n) // top-right: E
        } else if i >= n && j >= n {
            a.get(i - n, j - n) // bottom-right: A
        } else {
            0.0 // bottom-left: 0
        }
    });

    // Apply f to the block matrix.
    let fb = f(&block)?;

    // Extract upper-right n × n block.
    Ok(Tensor::from_fn(n, n, |i, j| fb.get(i, j + n)))
}

pub fn continuous_riccati(
    a: &Tensor<f64, Cpu>,
    b: &Tensor<f64, Cpu>,
    q: &Tensor<f64, Cpu>,
    r: &Tensor<f64, Cpu>,
) -> Result<Tensor<f64, Cpu>> {
    let n = a.nrows();
    require_square(a.shape(), "continuous_riccati(A)")?;
    require_square(q.shape(), "continuous_riccati(Q)")?;
    require_square(r.shape(), "continuous_riccati(R)")?;

    let nb = b.nrows();
    let mb = b.ncols();
    if nb != n {
        return Err(Error::invalid(format!(
            "continuous_riccati: B must have {n} rows, got {nb}"
        )));
    }
    if q.nrows() != n {
        return Err(Error::invalid(format!(
            "continuous_riccati: Q must be {n}×{n}, got {:?}",
            q.shape()
        )));
    }
    if r.nrows() != mb {
        return Err(Error::invalid(format!(
            "continuous_riccati: R must be {mb}×{mb}, got {:?}",
            r.shape()
        )));
    }

    // Compute R^{-1} via partial-pivot LU.
    let r_inv = {
        let lu = PartialPivLu::factorize(r)
            .map_err(|e| Error::invalid(format!("continuous_riccati: R singular: {e:?}")))?;
        let eye_m = Tensor::<f64, Cpu>::identity(mb);
        lu.solve_impl(&eye_m)
    };

    // S = B · R^{-1} · B^T  (n × n, positive semi-definite)
    let br = b * &r_inv;
    let s = &br * &b.t();

    // Hamiltonian H = [[A, -S], [-Q, -A^T]]  (2n × 2n)
    let two_n = 2 * n;
    let ham = Tensor::from_fn(two_n, two_n, |i, j| {
        if i < n && j < n {
            a.get(i, j) // top-left: A
        } else if i < n && j >= n {
            -s.get(i, j - n) // top-right: -S
        } else if i >= n && j < n {
            -q.get(i - n, j) // bottom-left: -Q
        } else {
            -a.get(j - n, i - n) // bottom-right: -A^T
        }
    });

    // Schur decompose the Hamiltonian: H = U · T · U^T.
    let (t_tensor, u_tensor) = schur(&ham)?;
    let t_buf = to_f64_buf(&t_tensor);
    let u_buf = to_f64_buf(&u_tensor);

    // Identify stable eigenvalues (negative real part).
    // Diagonal of T gives the real part of eigenvalues for 1×1 blocks;
    // for 2×2 blocks the real part is the average of the two diagonals.
    // We collect column indices of the stable Schur vectors.
    let mut stable_cols: Vec<usize> = Vec::with_capacity(n);
    let mut k = 0usize;
    while k < two_n {
        // Check if this is a 2×2 block (sub-diagonal nonzero).
        if k + 1 < two_n && t_buf[(k + 1) * two_n + k].abs() > f64::EPSILON {
            // 2×2 block: real part = (T[k,k] + T[k+1,k+1]) / 2
            let re = f64::midpoint(t_buf[k * two_n + k], t_buf[(k + 1) * two_n + k + 1]);
            if re < 0.0 {
                stable_cols.push(k);
                stable_cols.push(k + 1);
            }
            k += 2;
        } else {
            let re = t_buf[k * two_n + k];
            if re < 0.0 {
                stable_cols.push(k);
            }
            k += 1;
        }
    }

    if stable_cols.len() != n {
        return Err(Error::invalid(format!(
            "continuous_riccati: expected {n} stable eigenvalues, found {}; \
             Hamiltonian may have eigenvalues on the imaginary axis",
            stable_cols.len()
        )));
    }

    // Extract U11 (rows 0..n) and U21 (rows n..2n) from stable columns.
    let mut u11_buf = vec![0.0f64; n * n];
    let mut u21_buf = vec![0.0f64; n * n];
    for (col_out, &col_in) in stable_cols.iter().enumerate() {
        for row in 0..n {
            u11_buf[row * n + col_out] = u_buf[row * two_n + col_in];
            u21_buf[row * n + col_out] = u_buf[(row + n) * two_n + col_in];
        }
    }

    let u11 = from_f64_buf(u11_buf, n, n);
    let u21 = from_f64_buf(u21_buf, n, n);

    // X = U21 · U11^{-1}
    // Solve U11^T · X^T = U21^T  ⟺  X · U11 = U21  ⟺  X = U21 · inv(U11).
    let lu11 = PartialPivLu::factorize(&u11)
        .map_err(|e| Error::invalid(format!("continuous_riccati: U11 singular: {e:?}")))?;
    let eye_n = Tensor::<f64, Cpu>::identity(n);
    let u11_inv = lu11.solve_impl(&eye_n);
    Ok(&u21 * &u11_inv)
}

#[inline]
pub fn care(
    a: &Tensor<f64, Cpu>,
    b: &Tensor<f64, Cpu>,
    q: &Tensor<f64, Cpu>,
    r: &Tensor<f64, Cpu>,
) -> Result<Tensor<f64, Cpu>> {
    continuous_riccati(a, b, q, r)
}
