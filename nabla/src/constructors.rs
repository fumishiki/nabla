//! Free constructor and math functions for tensors.
//!
//! This module provides convenient free functions for creating tensors and
//! performing basic mathematical operations. All functions are re-exported
//! from the crate root via `pub use constructors::*;`.

use crate::{scalar, tensor};

#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| {
            let nanos = dur.as_nanos();
            (nanos as u64) ^ ((nanos >> 64) as u64)
        })
        .unwrap_or(0xA11C_E5EE_D5EE_DBAD_u64)
}

#[inline]
fn seed_or_default() -> u64 {
    let seed = default_seed();
    if seed == 0 {
        0x1234_5678_9ABC_DEF0_u64
    } else {
        seed
    }
}

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Allocate a zero-filled tensor of shape `(nrows, ncols)`.
#[must_use]
#[inline]
pub fn zeros<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    tensor::Tensor::zeros(nrows, ncols)
}

/// Allocate a one-filled tensor of shape `(nrows, ncols)`.
#[must_use]
#[inline]
pub fn ones<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    tensor::Tensor::fill(nrows, ncols, T::one())
}

/// Allocate a tensor of shape `(nrows, ncols)` filled with `value`.
#[must_use]
#[inline]
pub fn fill<T: scalar::Scalar>(nrows: usize, ncols: usize, value: T) -> tensor::Tensor<T> {
    tensor::Tensor::fill(nrows, ncols, value)
}

/// Allocate an identity matrix of size `n x n`.
#[must_use]
#[inline]
pub fn eye<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
    tensor::Tensor::identity(n)
}

/// Allocate a tensor whose element `(r, c)` is `f(r, c)`.
#[must_use]
#[inline]
pub fn from_fn<T: scalar::Scalar>(
    nrows: usize,
    ncols: usize,
    f: impl FnMut(usize, usize) -> T,
) -> tensor::Tensor<T> {
    tensor::Tensor::from_fn(nrows, ncols, f)
}

/// Allocate a zero-filled N-D tensor.
#[must_use]
#[inline]
pub fn nd_zeros<T: scalar::Scalar>(shape: &[usize]) -> tensor::NdTensor<T> {
    tensor::NdTensor::zeros(shape)
}

/// Uniform random tensor in `[0, 1)`.
#[must_use]
pub fn rand<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    let mut s = seed_or_default();
    tensor::Tensor::from_fn(nrows, ncols, |_, _| {
        let val = xorshift64(&mut s);
        T::from_f64((val as f64) / (u64::MAX as f64))
    })
}

/// Standard normal random tensor (`mean=0`, `std=1`).
#[must_use]
pub fn randn<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    let mut s = seed_or_default();
    let mut xorshift = || {
        let val = xorshift64(&mut s);
        (val as f64) / (u64::MAX as f64)
    };
    let n = nrows * ncols;
    let mut data = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let u1 = xorshift().max(1e-300);
        let u2 = xorshift();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(T::from_f64(r * theta.cos()));
        if i + 1 < n {
            data.push(T::from_f64(r * theta.sin()));
        }
        i += 2;
    }
    tensor::Tensor::from_fn(nrows, ncols, |r, c| data[r * ncols + c])
}

/// Column vector of zeros with shape `(n, 1)`.
#[must_use]
#[inline]
pub fn zeros_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
    zeros(n, 1)
}

/// Column vector of ones with shape `(n, 1)`.
#[must_use]
#[inline]
pub fn ones_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
    ones(n, 1)
}

/// Uniform random column vector in `[0, 1)` with shape `(n, 1)`.
#[must_use]
#[inline]
pub fn rand_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
    rand(n, 1)
}

/// Standard normal random column vector with shape `(n, 1)`.
#[must_use]
#[inline]
pub fn randn_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
    randn(n, 1)
}

/// 1-D half-open range tensor: `[start, start+step, ..., < stop]`.
#[must_use]
pub fn arange<T: scalar::Scalar>(start: T, stop: T, step: T) -> tensor::Tensor<T> {
    let step_f = step.to_f64();
    assert!(
        step_f.is_finite() && step_f != 0.0,
        "nabla: arange step must be non-zero finite, got {step_f}"
    );
    let stop_f = stop.to_f64();
    let is_forward = step_f > 0.0;
    let mut cur = start.to_f64();
    let mut n = 0usize;

    while (is_forward && cur < stop_f) || (!is_forward && cur > stop_f) {
        n += 1;
        cur += step_f;
    }

    tensor::Tensor::arange(start, step, n)
}

/// 1-D tensor of `n` evenly spaced values from `start` to `stop` (inclusive).
#[must_use]
#[inline]
pub fn linspace<T: scalar::Scalar>(start: T, stop: T, n: usize) -> tensor::Tensor<T> {
    tensor::Tensor::linspace(start, stop, n)
}

/// 1-D tensor of `n` points logarithmically spaced from `10^start` to `10^stop`.
///
/// Equivalent to NumPy's `np.logspace(start, stop, n)`.
#[must_use]
pub fn logspace<T: scalar::Scalar>(start: f64, stop: f64, n: usize) -> tensor::Tensor<T> {
    match n {
        0 => tensor::Tensor::zeros(1, 0),
        1 => tensor::Tensor::from_fn(1, 1, |_, _| T::from_f64(10.0_f64.powf(start))),
        _ => {
            #[allow(clippy::cast_precision_loss)]
            let step = (stop - start) / (n as f64 - 1.0);
            tensor::Tensor::from_fn(1, n, |_, j| {
                #[allow(clippy::cast_precision_loss)]
                let exponent = start + j as f64 * step;
                T::from_f64(10.0_f64.powf(exponent))
            })
        }
    }
}

/// 1-D tensor of `n` points geometrically spaced from `start` to `stop`.
///
/// Both `start` and `stop` must be non-zero and have the same sign.
/// Equivalent to NumPy's `np.geomspace(start, stop, n)`.
///
/// # Panics
///
/// Panics if `start` or `stop` is zero, or if they have different signs.
#[must_use]
pub fn geomspace<T: scalar::Scalar>(start: f64, stop: f64, n: usize) -> tensor::Tensor<T> {
    assert!(
        start != 0.0 && stop != 0.0,
        "nabla: geomspace requires non-zero start and stop"
    );
    assert!(
        start.signum() == stop.signum(),
        "nabla: geomspace requires start and stop to have the same sign"
    );
    match n {
        0 => tensor::Tensor::zeros(1, 0),
        1 => tensor::Tensor::from_fn(1, 1, |_, _| T::from_f64(start)),
        _ => {
            let log_start = start.abs().ln();
            let log_stop = stop.abs().ln();
            #[allow(clippy::cast_precision_loss)]
            let step = (log_stop - log_start) / (n as f64 - 1.0);
            let sign = start.signum();
            tensor::Tensor::from_fn(1, n, |_, j| {
                #[allow(clippy::cast_precision_loss)]
                let val = sign * (log_start + j as f64 * step).exp();
                T::from_f64(val)
            })
        }
    }
}

/// Inner product (dot product), returning a scalar.
#[must_use]
#[inline]
pub fn dot<T: scalar::Scalar>(a: &tensor::Tensor<T>, b: &tensor::Tensor<T>) -> T {
    a.dot(b)
}

/// Kronecker product: `A ⊗ B`.
#[must_use]
#[inline]
pub fn kron<T: scalar::Scalar>(
    a: &tensor::Tensor<T>,
    b: &tensor::Tensor<T>,
) -> tensor::Tensor<T> {
    a.kron(b)
}

/// Create diagonal matrix from a vector tensor (1×n or n×1).
#[must_use]
#[inline]
pub fn diagm<T: scalar::Scalar>(v: &tensor::Tensor<T>) -> tensor::Tensor<T> {
    tensor::Tensor::from_diag(v)
}

/// 3D cross product of two 3-element vectors.
/// Both must be (3,1) or (1,3) tensors.
#[must_use]
pub fn cross<T: scalar::Scalar>(
    a: &tensor::Tensor<T>,
    b: &tensor::Tensor<T>,
) -> tensor::Tensor<T> {
    let na = a.nrows() * a.ncols();
    let nb = b.nrows() * b.ncols();
    assert!(
        na == 3 && nb == 3,
        "nabla: cross requires 3-element vectors, got {na} and {nb}"
    );
    let a_is_row = a.ncols() == 3;
    let b_is_row = b.ncols() == 3;
    let comp_a = |row: usize, col: usize| {
        if a_is_row {
            a.get(row, col)
        } else {
            a.get(col, row)
        }
    };
    let comp_b = |row: usize, col: usize| {
        if b_is_row {
            b.get(row, col)
        } else {
            b.get(col, row)
        }
    };

    let a0 = comp_a(0, 0);
    let a1 = comp_a(1, 0);
    let a2 = comp_a(2, 0);
    let b0 = comp_b(0, 0);
    let b1 = comp_b(1, 0);
    let b2 = comp_b(2, 0);
    // Cross product: (a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0)
    tensor::Tensor::from_fn(3, 1, |i, _| match i {
        0 => a1 * b2 - a2 * b1,
        1 => a2 * b0 - a0 * b2,
        _ => a0 * b1 - a1 * b0,
    })
}

/// Frobenius/L2 norm of a tensor.
#[must_use]
#[inline]
pub fn norm<T: scalar::Scalar>(a: &tensor::Tensor<T>) -> T {
    a.norm()
}

/// Lp norm of a tensor with specified order.
#[must_use]
#[inline]
pub fn norm_ord<T: scalar::Scalar>(a: &tensor::Tensor<T>, p: f64) -> T {
    a.norm_ord(p)
}

/// Trace of a matrix (sum of diagonal elements).
#[must_use]
#[inline]
pub fn tr<T: scalar::Scalar>(a: &tensor::Tensor<T>) -> T {
    a.tr()
}

/// Check approximate equality of two tensors within absolute tolerance `atol`.
///
/// Returns `false` if shapes differ or any element pair exceeds `atol`.
#[must_use]
pub fn approx_eq<T: scalar::Scalar>(
    a: &tensor::Tensor<T>,
    b: &tensor::Tensor<T>,
    atol: f64,
) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let (m, n) = a.shape();
    for r in 0..m {
        for c in 0..n {
            if (a.get(r, c).to_f64() - b.get(r, c).to_f64()).abs() > atol {
                return false;
            }
        }
    }
    true
}
