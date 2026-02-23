// backend.rs — Sealed Backend trait + Cpu implementation backed by faer 0.24.
//
// API adaptations vs. spec (faer 0.24 differences):
//   - No `faer::Entity`; `faer::ComplexField` is the sole trait bound.
//   - `Mat::from_fn` closure receives `usize` indices for dynamic matrices.
//   - Element access uses `mat.get(row, col)` returning `&T`.
//   - matmul: `faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, one, Par::Seq)`.
//   - Scalar multiply: `mat * faer::Scale(s)`.
//   - Transpose collects via `Mat::from_fn` over the transposed view.

use faer::{Accum, Mat, Par, Scale, linalg::matmul::matmul};

use crate::scalar::Scalar;

// Abramowitz & Stegun polynomial approximation for erf (max error ~1.5e-7).
#[inline]
fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly =
        t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

// Reduction helpers: sum identity and ordered comparison for all four Scalar types.
// Used exclusively by sum_all/max_all/min_all/argmax_all/argmin_all — not part of the public API.
pub(crate) trait ReductionOps: Sized + Copy {
    /// Additive identity (zero) for folding sum.
    fn reduction_zero() -> Self;
    /// Accumulate `self + other` for sum reduction.
    fn reduction_add(self, other: Self) -> Self;
    /// Return the element with the larger magnitude (or value for real types).
    fn reduction_max(self, other: Self) -> Self;
    /// Return the element with the smaller magnitude (or value for real types).
    fn reduction_min(self, other: Self) -> Self;
    /// Returns `true` if `self` is strictly "greater than" `other`.
    ///
    /// For real types (`f32`/`f64`): standard `>` comparison.
    /// For complex types (`c32`/`c64`): compares magnitudes `|self|² > |other|²`.
    fn reduction_gt(self, other: Self) -> bool;
}

impl ReductionOps for f32 {
    #[inline] fn reduction_zero() -> Self { 0.0 }
    #[inline] fn reduction_add(self, other: Self) -> Self { self + other }
    #[inline] fn reduction_max(self, other: Self) -> Self { self.max(other) }
    #[inline] fn reduction_min(self, other: Self) -> Self { self.min(other) }
    #[inline] fn reduction_gt(self, other: Self) -> bool { self > other }
}

impl ReductionOps for f64 {
    #[inline] fn reduction_zero() -> Self { 0.0 }
    #[inline] fn reduction_add(self, other: Self) -> Self { self + other }
    #[inline] fn reduction_max(self, other: Self) -> Self { self.max(other) }
    #[inline] fn reduction_min(self, other: Self) -> Self { self.min(other) }
    #[inline] fn reduction_gt(self, other: Self) -> bool { self > other }
}

impl ReductionOps for faer::c32 {
    #[inline] fn reduction_zero() -> Self { Self::new(0.0, 0.0) }
    #[inline] fn reduction_add(self, other: Self) -> Self { self + other }
    // Compare by magnitude squared: re^2 + im^2
    #[inline] fn reduction_max(self, other: Self) -> Self {
        let self_mag2 = self.re * self.re + self.im * self.im;
        let other_mag2 = other.re * other.re + other.im * other.im;
        if self_mag2 >= other_mag2 { self } else { other }
    }
    #[inline] fn reduction_min(self, other: Self) -> Self {
        let self_mag2 = self.re * self.re + self.im * self.im;
        let other_mag2 = other.re * other.re + other.im * other.im;
        if self_mag2 <= other_mag2 { self } else { other }
    }
    #[inline] fn reduction_gt(self, other: Self) -> bool {
        let self_mag2 = self.re * self.re + self.im * self.im;
        let other_mag2 = other.re * other.re + other.im * other.im;
        self_mag2 > other_mag2
    }
}

impl ReductionOps for faer::c64 {
    #[inline] fn reduction_zero() -> Self { Self::new(0.0, 0.0) }
    #[inline] fn reduction_add(self, other: Self) -> Self { self + other }
    #[inline] fn reduction_max(self, other: Self) -> Self {
        let self_mag2 = self.re * self.re + self.im * self.im;
        let other_mag2 = other.re * other.re + other.im * other.im;
        if self_mag2 >= other_mag2 { self } else { other }
    }
    #[inline] fn reduction_min(self, other: Self) -> Self {
        let self_mag2 = self.re * self.re + self.im * self.im;
        let other_mag2 = other.re * other.re + other.im * other.im;
        if self_mag2 <= other_mag2 { self } else { other }
    }
    #[inline] fn reduction_gt(self, other: Self) -> bool {
        let self_mag2 = self.re * self.re + self.im * self.im;
        let other_mag2 = other.re * other.re + other.im * other.im;
        self_mag2 > other_mag2
    }
}

// Elementwise math dispatch for all four Scalar types.
// pub(crate) so that scalar::Scalar can use it as a supertrait bound.
pub(crate) trait MathOps: Sized + Copy {
    fn math_exp(self) -> Self;
    fn math_ln(self) -> Self;
    fn math_log1p(self) -> Self;
    fn math_sin(self) -> Self;
    fn math_cos(self) -> Self;
    fn math_tanh(self) -> Self;
    fn math_sqrt(self) -> Self;
    fn math_abs(self) -> Self;
    fn math_recip(self) -> Self;
    fn math_erf(self) -> Self;
    fn math_ceil(self) -> Self;
    fn math_floor(self) -> Self;
    fn math_round(self) -> Self;
    fn math_powf(self, p: Self) -> Self;
    fn math_mul(self, other: Self) -> Self;
    fn math_div(self, other: Self) -> Self;
}

impl MathOps for f32 {
    #[inline] fn math_exp(self) -> Self { self.exp() }
    #[inline] fn math_ln(self) -> Self { self.ln() }
    #[inline] fn math_log1p(self) -> Self { self.ln_1p() }
    #[inline] fn math_sin(self) -> Self { self.sin() }
    #[inline] fn math_cos(self) -> Self { self.cos() }
    #[inline] fn math_tanh(self) -> Self { self.tanh() }
    #[inline] fn math_sqrt(self) -> Self { self.sqrt() }
    #[inline] fn math_abs(self) -> Self { self.abs() }
    #[inline] fn math_recip(self) -> Self { self.recip() }
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn math_erf(self) -> Self { erf_approx(f64::from(self)) as f32 }
    #[inline] fn math_ceil(self) -> Self { self.ceil() }
    #[inline] fn math_floor(self) -> Self { self.floor() }
    #[inline] fn math_round(self) -> Self { self.round() }
    #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p) }
    #[inline] fn math_mul(self, other: Self) -> Self { self * other }
    #[inline] fn math_div(self, other: Self) -> Self { self / other }
}

impl MathOps for f64 {
    #[inline] fn math_exp(self) -> Self { self.exp() }
    #[inline] fn math_ln(self) -> Self { self.ln() }
    #[inline] fn math_log1p(self) -> Self { self.ln_1p() }
    #[inline] fn math_sin(self) -> Self { self.sin() }
    #[inline] fn math_cos(self) -> Self { self.cos() }
    #[inline] fn math_tanh(self) -> Self { self.tanh() }
    #[inline] fn math_sqrt(self) -> Self { self.sqrt() }
    #[inline] fn math_abs(self) -> Self { self.abs() }
    #[inline] fn math_recip(self) -> Self { self.recip() }
    #[inline] fn math_erf(self) -> Self { erf_approx(self) }
    #[inline] fn math_ceil(self) -> Self { self.ceil() }
    #[inline] fn math_floor(self) -> Self { self.floor() }
    #[inline] fn math_round(self) -> Self { self.round() }
    #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p) }
    #[inline] fn math_mul(self, other: Self) -> Self { self * other }
    #[inline] fn math_div(self, other: Self) -> Self { self / other }
}

impl MathOps for faer::c32 {
    #[inline] fn math_exp(self) -> Self { self.exp() }
    #[inline] fn math_ln(self) -> Self { self.ln() }
    // log1p(z) = ln(1 + z)
    #[inline] fn math_log1p(self) -> Self { (Self::new(1.0 + self.re, self.im)).ln() }
    #[inline] fn math_sin(self) -> Self { self.sin() }
    #[inline] fn math_cos(self) -> Self { self.cos() }
    #[inline] fn math_tanh(self) -> Self { self.tanh() }
    #[inline] fn math_sqrt(self) -> Self { self.sqrt() }
    // abs: magnitude as real part, 0 as imaginary
    #[inline] fn math_abs(self) -> Self { Self::new(self.norm(), 0.0) }
    // recip: 1/z = conj(z) / |z|^2
    #[inline] fn math_recip(self) -> Self { self.inv() }
    // erf: apply component-wise (approximate via real erf on re, im components)
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn math_erf(self) -> Self {
        Self::new(erf_approx(f64::from(self.re)) as f32, erf_approx(f64::from(self.im)) as f32)
    }
    // ceil/floor/round: component-wise on re/im
    #[inline] fn math_ceil(self) -> Self { Self::new(self.re.ceil(), self.im.ceil()) }
    #[inline] fn math_floor(self) -> Self { Self::new(self.re.floor(), self.im.floor()) }
    #[inline] fn math_round(self) -> Self { Self::new(self.re.round(), self.im.round()) }
    #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p.re) }
    #[inline] fn math_mul(self, other: Self) -> Self { self * other }
    #[inline] fn math_div(self, other: Self) -> Self { self / other }
}

impl MathOps for faer::c64 {
    #[inline] fn math_exp(self) -> Self { self.exp() }
    #[inline] fn math_ln(self) -> Self { self.ln() }
    #[inline] fn math_log1p(self) -> Self { (Self::new(1.0 + self.re, self.im)).ln() }
    #[inline] fn math_sin(self) -> Self { self.sin() }
    #[inline] fn math_cos(self) -> Self { self.cos() }
    #[inline] fn math_tanh(self) -> Self { self.tanh() }
    #[inline] fn math_sqrt(self) -> Self { self.sqrt() }
    #[inline] fn math_abs(self) -> Self { Self::new(self.norm(), 0.0) }
    #[inline] fn math_recip(self) -> Self { self.inv() }
    #[inline] fn math_erf(self) -> Self {
        Self::new(erf_approx(self.re), erf_approx(self.im))
    }
    #[inline] fn math_ceil(self) -> Self { Self::new(self.re.ceil(), self.im.ceil()) }
    #[inline] fn math_floor(self) -> Self { Self::new(self.re.floor(), self.im.floor()) }
    #[inline] fn math_round(self) -> Self { Self::new(self.re.round(), self.im.round()) }
    #[inline] fn math_powf(self, p: Self) -> Self { self.powf(p.re) }
    #[inline] fn math_mul(self, other: Self) -> Self { self * other }
    #[inline] fn math_div(self, other: Self) -> Self { self / other }
}

// Macro: apply a MathOps method element-wise via Mat::from_fn.
macro_rules! mat_elemwise_unary {
    ($a:expr, $method:ident) => {{
        let (r, c) = ($a.nrows(), $a.ncols());
        Mat::from_fn(r, c, |i, j| (*$a.get(i, j)).$method())
    }};
}

macro_rules! mat_elemwise_binary {
    ($a:expr, $b:expr, $method:ident) => {{
        let (r, c) = ($a.nrows(), $a.ncols());
        Mat::from_fn(r, c, |i, j| (*$a.get(i, j)).$method(*$b.get(i, j)))
    }};
}

mod private {
    pub trait Sealed {}
}

/// Compute backend abstraction (sealed — not implementable outside this crate).
pub trait Backend: private::Sealed + Send + Sync + 'static {
    /// Owned storage for a 2-D matrix of element type `T`.
    type Storage<T: Scalar>: Send + Sync;

    /// Allocate a zero-filled matrix.
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> Self::Storage<T>;

    /// Allocate a matrix and fill it by calling `f(row, col)`.
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> Self::Storage<T>;

    /// Row count of `storage`.
    fn nrows<T: Scalar>(storage: &Self::Storage<T>) -> usize;

    /// Column count of `storage`.
    fn ncols<T: Scalar>(storage: &Self::Storage<T>) -> usize;

    /// Read element at `(row, col)`.
    fn get<T: Scalar>(storage: &Self::Storage<T>, row: usize, col: usize) -> T;

    /// Write element at `(row, col)`.
    fn set<T: Scalar>(storage: &mut Self::Storage<T>, row: usize, col: usize, val: T);

    /// Compute `out = a * b`, overwriting `out`.
    fn matmul_into<T: Scalar>(
        out: &mut Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    );

    /// Element-wise addition.
    fn add<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise subtraction.
    fn sub<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise negation.
    fn neg<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Transpose: result has shape `(ncols(a), nrows(a))`.
    fn transpose<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Scalar multiply: every element of `a` multiplied by `s`.
    fn scale<T: Scalar>(a: &Self::Storage<T>, s: T) -> Self::Storage<T>;

    /// Clone storage.
    fn clone_storage<T: Scalar>(storage: &Self::Storage<T>) -> Self::Storage<T>;

    // --- Elementwise math operations ---

    /// Element-wise `e^x`.
    fn exp<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise natural logarithm `ln(x)`.
    fn ln<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `ln(1 + x)`.
    fn log1p<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `sin(x)`.
    fn sin<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `cos(x)`.
    fn cos<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `tanh(x)`.
    fn tanh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `sqrt(x)`.
    fn sqrt<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise absolute value.
    ///
    /// For complex types, returns the magnitude as the real part with zero imaginary part.
    fn abs<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise reciprocal `1/x`.
    fn recip<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise error function.
    ///
    /// Uses the Abramowitz & Stegun polynomial approximation (max error ~1.5e-7).
    /// For complex types, applies component-wise to re and im parts.
    fn erf<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `ceil(x)`.
    ///
    /// For complex types, applies component-wise to re and im parts.
    fn ceil<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `floor(x)`.
    ///
    /// For complex types, applies component-wise to re and im parts.
    fn floor<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `round(x)`.
    ///
    /// For complex types, applies component-wise to re and im parts.
    fn round<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `x^p` for scalar exponent `p`.
    ///
    /// For complex types, uses the real part of `p` as the exponent.
    fn powf<T: Scalar>(a: &Self::Storage<T>, p: T) -> Self::Storage<T>;

    /// Element-wise multiplication `a[i,j] * b[i,j]`.
    fn mul_elem<T: Scalar>(
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    ) -> Self::Storage<T>;

    /// Element-wise division `a[i,j] / b[i,j]`.
    fn div_elem<T: Scalar>(
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    ) -> Self::Storage<T>;

    // --- Reduction operations (whole-matrix → scalar) ---

    /// Sum all elements of the matrix.
    ///
    /// Returns the additive identity `T::zero` for an empty matrix.
    fn sum_all<T: Scalar>(a: &Self::Storage<T>) -> T;

    /// Element with the maximum value (or maximum magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    fn max_all<T: Scalar>(a: &Self::Storage<T>) -> T;

    /// Element with the minimum value (or minimum magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    fn min_all<T: Scalar>(a: &Self::Storage<T>) -> T;

    /// `(row, col)` of the element with the maximum value (or magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    fn argmax_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);

    /// `(row, col)` of the element with the minimum value (or magnitude for complex types).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    fn argmin_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);
}

/// CPU backend using faer's native SIMD kernels.
pub struct Cpu;

impl private::Sealed for Cpu {}

impl Backend for Cpu {
    type Storage<T: Scalar> = Mat<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> Mat<T> {
        Mat::zeros(nrows, ncols)
    }

    #[inline]
    fn from_fn<T: Scalar>(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> Mat<T> {
        Mat::from_fn(nrows, ncols, f)
    }

    #[inline]
    fn nrows<T: Scalar>(storage: &Mat<T>) -> usize {
        storage.nrows()
    }

    #[inline]
    fn ncols<T: Scalar>(storage: &Mat<T>) -> usize {
        storage.ncols()
    }

    #[inline]
    fn get<T: Scalar>(storage: &Mat<T>, row: usize, col: usize) -> T {
        *storage.get(row, col)
    }

    #[inline]
    fn set<T: Scalar>(storage: &mut Mat<T>, row: usize, col: usize, val: T) {
        *storage.get_mut(row, col) = val;
    }

    #[inline]
    fn matmul_into<T: Scalar>(out: &mut Mat<T>, a: &Mat<T>, b: &Mat<T>) {
        matmul(out, Accum::Replace, a, b, T::one_impl(), Par::Seq);
    }

    #[inline]
    fn add<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
        a + b
    }

    #[inline]
    fn sub<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
        a - b
    }

    #[inline]
    fn neg<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        -a
    }

    #[inline]
    fn transpose<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        let t = a.as_ref().transpose();
        Mat::from_fn(t.nrows(), t.ncols(), |r, c| *t.get(r, c))
    }

    #[inline]
    fn scale<T: Scalar>(a: &Mat<T>, s: T) -> Mat<T> {
        a * Scale(s)
    }

    #[inline]
    fn clone_storage<T: Scalar>(storage: &Mat<T>) -> Mat<T> {
        storage.clone()
    }

    #[inline]
    fn exp<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_exp)
    }

    #[inline]
    fn ln<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_ln)
    }

    #[inline]
    fn log1p<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_log1p)
    }

    #[inline]
    fn sin<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_sin)
    }

    #[inline]
    fn cos<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_cos)
    }

    #[inline]
    fn tanh<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_tanh)
    }

    #[inline]
    fn sqrt<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_sqrt)
    }

    #[inline]
    fn abs<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_abs)
    }

    #[inline]
    fn recip<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_recip)
    }

    #[inline]
    fn erf<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_erf)
    }

    #[inline]
    fn ceil<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_ceil)
    }

    #[inline]
    fn floor<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_floor)
    }

    #[inline]
    fn round<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        mat_elemwise_unary!(a, math_round)
    }

    #[inline]
    fn powf<T: Scalar>(a: &Mat<T>, p: T) -> Mat<T> {
        let (r, c) = (a.nrows(), a.ncols());
        Mat::from_fn(r, c, |i, j| (*a.get(i, j)).math_powf(p))
    }

    #[inline]
    fn mul_elem<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
        mat_elemwise_binary!(a, b, math_mul)
    }

    #[inline]
    fn div_elem<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
        mat_elemwise_binary!(a, b, math_div)
    }

    #[inline]
    fn sum_all<T: Scalar>(a: &Mat<T>) -> T {
        let (r, c) = (a.nrows(), a.ncols());
        (0..r).flat_map(|i| (0..c).map(move |j| (i, j)))
            .fold(T::reduction_zero(), |acc, (i, j)| acc.reduction_add(*a.get(i, j)))
    }

    #[inline]
    fn max_all<T: Scalar>(a: &Mat<T>) -> T {
        assert!(a.nrows() > 0 && a.ncols() > 0, "max_all: matrix must be non-empty");
        let (r, c) = (a.nrows(), a.ncols());
        let init = *a.get(0, 0);
        (0..r).flat_map(|i| (0..c).map(move |j| (i, j)))
            .skip(1)
            .fold(init, |acc, (i, j)| acc.reduction_max(*a.get(i, j)))
    }

    #[inline]
    fn min_all<T: Scalar>(a: &Mat<T>) -> T {
        assert!(a.nrows() > 0 && a.ncols() > 0, "min_all: matrix must be non-empty");
        let (r, c) = (a.nrows(), a.ncols());
        let init = *a.get(0, 0);
        (0..r).flat_map(|i| (0..c).map(move |j| (i, j)))
            .skip(1)
            .fold(init, |acc, (i, j)| acc.reduction_min(*a.get(i, j)))
    }

    #[inline]
    fn argmax_all<T: Scalar>(a: &Mat<T>) -> (usize, usize) {
        assert!(a.nrows() > 0 && a.ncols() > 0, "argmax_all: matrix must be non-empty");
        let (r, c) = (a.nrows(), a.ncols());
        let mut best = (0usize, 0usize);
        for i in 0..r {
            for j in 0..c {
                if i == 0 && j == 0 { continue; }
                if (*a.get(i, j)).reduction_gt(*a.get(best.0, best.1)) {
                    best = (i, j);
                }
            }
        }
        best
    }

    #[inline]
    fn argmin_all<T: Scalar>(a: &Mat<T>) -> (usize, usize) {
        assert!(a.nrows() > 0 && a.ncols() > 0, "argmin_all: matrix must be non-empty");
        let (r, c) = (a.nrows(), a.ncols());
        let mut best = (0usize, 0usize);
        for i in 0..r {
            for j in 0..c {
                if i == 0 && j == 0 { continue; }
                if (*a.get(best.0, best.1)).reduction_gt(*a.get(i, j)) {
                    best = (i, j);
                }
            }
        }
        best
    }
}

// Hip still delegates to Cpu; Cuda and Wgpu use real GPU storage via gpu.rs.
#[cfg(feature = "hip")]
type MatStorage<T> = Mat<T>;

#[cfg(feature = "cuda")]
/// CUDA backend — uses cubecl-cuda kernels for f32/f64, CPU fallback for c32/c64.
pub struct Cuda;

#[cfg(feature = "cuda")]
impl private::Sealed for Cuda {}

#[cfg(feature = "wgpu")]
/// wgpu backend — uses cubecl-wgpu kernels for f32/f64, CPU fallback for c32/c64.
pub struct Wgpu;

#[cfg(feature = "wgpu")]
impl private::Sealed for Wgpu {}

#[cfg(feature = "hip")]
/// HIP backend stub — currently delegates all operations to Cpu.
pub struct Hip;

#[cfg(feature = "hip")]
impl private::Sealed for Hip {}

// Macro is only used for Hip (delegates everything to Cpu).
#[cfg(feature = "hip")]
macro_rules! delegate_backend {
    ($backend:ty) => {
        impl Backend for $backend {
            type Storage<T: Scalar> = MatStorage<T>;

            #[inline]
            fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> MatStorage<T> {
                Cpu::zeros(nrows, ncols)
            }

            #[inline]
            fn from_fn<T: Scalar>(
                nrows: usize,
                ncols: usize,
                f: impl FnMut(usize, usize) -> T,
            ) -> MatStorage<T> {
                Cpu::from_fn(nrows, ncols, f)
            }

            #[inline]
            fn nrows<T: Scalar>(storage: &MatStorage<T>) -> usize {
                Cpu::nrows(storage)
            }

            #[inline]
            fn ncols<T: Scalar>(storage: &MatStorage<T>) -> usize {
                Cpu::ncols(storage)
            }

            #[inline]
            fn get<T: Scalar>(storage: &MatStorage<T>, row: usize, col: usize) -> T {
                Cpu::get(storage, row, col)
            }

            #[inline]
            fn set<T: Scalar>(storage: &mut MatStorage<T>, row: usize, col: usize, val: T) {
                Cpu::set(storage, row, col, val)
            }

            #[inline]
            fn matmul_into<T: Scalar>(
                out: &mut MatStorage<T>,
                a: &MatStorage<T>,
                b: &MatStorage<T>,
            ) {
                Cpu::matmul_into(out, a, b)
            }

            #[inline]
            fn add<T: Scalar>(a: &MatStorage<T>, b: &MatStorage<T>) -> MatStorage<T> {
                Cpu::add(a, b)
            }

            #[inline]
            fn sub<T: Scalar>(a: &MatStorage<T>, b: &MatStorage<T>) -> MatStorage<T> {
                Cpu::sub(a, b)
            }

            #[inline]
            fn neg<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::neg(a)
            }

            #[inline]
            fn transpose<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::transpose(a)
            }

            #[inline]
            fn scale<T: Scalar>(a: &MatStorage<T>, s: T) -> MatStorage<T> {
                Cpu::scale(a, s)
            }

            #[inline]
            fn clone_storage<T: Scalar>(storage: &MatStorage<T>) -> MatStorage<T> {
                Cpu::clone_storage(storage)
            }

            #[inline]
            fn exp<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::exp(a)
            }

            #[inline]
            fn ln<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::ln(a)
            }

            #[inline]
            fn log1p<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::log1p(a)
            }

            #[inline]
            fn sin<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::sin(a)
            }

            #[inline]
            fn cos<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::cos(a)
            }

            #[inline]
            fn tanh<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::tanh(a)
            }

            #[inline]
            fn sqrt<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::sqrt(a)
            }

            #[inline]
            fn abs<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::abs(a)
            }

            #[inline]
            fn recip<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::recip(a)
            }

            #[inline]
            fn erf<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::erf(a)
            }

            #[inline]
            fn ceil<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::ceil(a)
            }

            #[inline]
            fn floor<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::floor(a)
            }

            #[inline]
            fn round<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::round(a)
            }

            #[inline]
            fn powf<T: Scalar>(a: &MatStorage<T>, p: T) -> MatStorage<T> {
                Cpu::powf(a, p)
            }

            #[inline]
            fn mul_elem<T: Scalar>(
                a: &MatStorage<T>,
                b: &MatStorage<T>,
            ) -> MatStorage<T> {
                Cpu::mul_elem(a, b)
            }

            #[inline]
            fn div_elem<T: Scalar>(
                a: &MatStorage<T>,
                b: &MatStorage<T>,
            ) -> MatStorage<T> {
                Cpu::div_elem(a, b)
            }

            #[inline]
            fn sum_all<T: Scalar>(a: &MatStorage<T>) -> T {
                Cpu::sum_all(a)
            }

            #[inline]
            fn max_all<T: Scalar>(a: &MatStorage<T>) -> T {
                Cpu::max_all(a)
            }

            #[inline]
            fn min_all<T: Scalar>(a: &MatStorage<T>) -> T {
                Cpu::min_all(a)
            }

            #[inline]
            fn argmax_all<T: Scalar>(a: &MatStorage<T>) -> (usize, usize) {
                Cpu::argmax_all(a)
            }

            #[inline]
            fn argmin_all<T: Scalar>(a: &MatStorage<T>) -> (usize, usize) {
                Cpu::argmin_all(a)
            }
        }
    };
}

#[cfg(feature = "hip")]
delegate_backend!(Hip);

#[cfg(feature = "cuda")]
/// Default backend: CUDA (highest priority when enabled).
pub type DefaultBackend = Cuda;

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "hip")))]
/// Default backend: wgpu (used when cuda is not enabled).
pub type DefaultBackend = Wgpu;

#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
/// Default backend: HIP (used when cuda and wgpu are not enabled).
pub type DefaultBackend = Hip;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "hip")))]
/// Default backend: CPU (fallback when no GPU feature is enabled).
pub type DefaultBackend = Cpu;
