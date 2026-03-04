use rayon::prelude::*;

use core::ops::Range;

use crate::backend::Backend;
use crate::scalar::Scalar;

use super::{MatrixLike, Tensor};

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x1234_5678_9ABC_DEF0_u64
        } else {
            seed
        };
        Self { state }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Create a tensor from a flat `Vec<T>` with the given shape.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    #[must_use]
    pub fn from_vec(data: Vec<T>, nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "nabla: from_vec data.len() {} != nrows*ncols {}",
            data.len(),
            nrows * ncols,
        );
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }

    /// Allocate a zero-filled matrix of shape `(nrows, ncols)`.
    #[must_use]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::from_storage(B::zeros(nrows, ncols))
    }

    /// Tensor filled with ones.
    #[must_use]
    pub fn ones(nrows: usize, ncols: usize) -> Self {
        Self::fill(nrows, ncols, T::one())
    }

    /// Allocate an `nrows x ncols` matrix filled with `val`.
    #[must_use]
    pub fn fill(nrows: usize, ncols: usize, val: T) -> Self {
        Self::from_storage(B::fill(nrows, ncols, val))
    }

    /// Allocate a matrix whose `(i, j)` element is `f(i, j)`.
    #[must_use]
    pub fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> Self {
        Self::from_storage(B::from_fn(nrows, ncols, f))
    }

    /// Allocate an `n x n` identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_storage(B::identity(n))
    }

    /// Convert class index slice to one-hot matrix `(n_samples x n_classes)`.
    #[must_use]
    pub fn one_hot(indices: &[usize], n_classes: usize) -> Self {
        Self::from_fn(indices.len(), n_classes, |r, c| {
            if c == indices[r] { T::one() } else { T::zero() }
        })
    }

    /// Create tensor from slice with non-blocking H2D transfer.
    /// The transfer happens on a separate copy stream and can overlap with compute.
    /// On CPU backend this is identical to the synchronous path.
    #[must_use]
    pub fn from_slice_async(data: &[T], nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "from_slice_async: data.len() must equal nrows * ncols"
        );
        Self::from_storage(B::from_vec_async(nrows, ncols, data.to_vec()))
    }

    /// Identity matrix of same size as `self` (must be square).
    #[must_use]
    pub fn eye_like(&self) -> Self {
        let (m, n) = self.shape();
        assert_eq!(m, n, "nabla: eye_like requires square tensor, got {m}x{n}");
        Self::identity(n)
    }

    /// Same-shape tensor filled with zeros.
    #[must_use]
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.nrows(), self.ncols())
    }

    /// Same-shape tensor filled with ones.
    #[must_use]
    pub fn ones_like(&self) -> Self {
        self.fill_like(T::one())
    }

    /// Same-shape tensor filled with `val`.
    #[must_use]
    pub fn fill_like(&self, val: T) -> Self {
        Self::fill(self.nrows(), self.ncols(), val)
    }

    /// Same-shape tensor filled with `val`.
    #[must_use]
    pub fn full_like(&self, val: T) -> Self {
        self.fill_like(val)
    }

    /// Uninitialized tensor (actually zeroed -- Rust safety).
    #[must_use]
    pub fn empty(nrows: usize, ncols: usize) -> Self {
        Self::from_storage(B::empty(nrows, ncols))
    }

    /// Generate a 1-D tensor: `[start, start+step, start+2*step, ...]` with length `n`.
    #[must_use]
    pub fn arange(start: T, step: T, n: usize) -> Self {
        Self::from_fn(1, n, |_, c| start + step * T::from_f64(c as f64))
    }

    /// Generate a 1-D tensor of `n` evenly spaced values from `start` to `end` (inclusive).
    #[must_use]
    pub fn linspace(start: T, end: T, n: usize) -> Self {
        assert!(n >= 2, "nabla: linspace needs n >= 2");
        let denom = T::from_f64((n - 1) as f64);
        Self::from_fn(1, n, |_, c| {
            let t = T::from_f64(c as f64) / denom;
            start + (end - start) * t
        })
    }

    /// Allocate a zero-filled tensor of the same shape and backend type.
    #[must_use]
    pub fn similar(&self) -> Self {
        self.zeros_like()
    }

    /// Allocate a zero-filled tensor with a different shape but same backend type.
    #[must_use]
    pub fn similar_shape(&self, nrows: usize, ncols: usize) -> Self {
        Self::zeros(nrows, ncols)
    }

    /// Allocate a zero-filled tensor with the same shape but a different scalar type.
    #[must_use]
    pub fn similar_zeros<U: Scalar>(&self) -> Tensor<U, B> {
        Tensor::<U, B>::zeros(self.nrows(), self.ncols())
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Uniform random tensor in [0, 1). Uses xorshift64 seeded from `seed`.
    #[must_use]
    pub fn rand(nrows: usize, ncols: usize, seed: u64) -> Self {
        let n = nrows * ncols;
        let mut data = Vec::with_capacity(n);
        let mut rng = XorShift64::new(seed);
        data.extend((0..n).map(|_| T::from_f64(rng.next_f64())));
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }

    /// Normal-distributed random tensor (mean=0, std=1) via Box-Muller.
    /// Uses xorshift64 seeded from `seed`.
    #[must_use]
    pub fn randn(nrows: usize, ncols: usize, seed: u64) -> Self {
        let n = nrows * ncols;
        let mut data = Vec::with_capacity(n);
        let mut rng = XorShift64::new(seed);
        let mut i = 0;
        while i < n {
            let u1 = rng.next_f64().max(1e-300);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data.push(T::from_f64(r * theta.cos()));
            if i + 1 < n {
                data.push(T::from_f64(r * theta.sin()));
            }
            i += 2;
        }
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }

    /// Parallel `from_fn` -- construct a matrix using rayon.
    #[must_use]
    pub fn par_from_fn(
        nrows: usize,
        ncols: usize,
        f: impl Fn(usize, usize) -> T + Send + Sync,
    ) -> Self {
        let data: Vec<T> = (0..nrows * ncols)
            .into_par_iter()
            .map(|idx| f(idx / ncols, idx % ncols))
            .collect();
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }
}

/// Zero-copy read-only view into a sub-region of a [`Tensor`].
pub struct TensorView<'a, T: Scalar, B: Backend> {
    source: &'a Tensor<T, B>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
}

impl<T: Scalar, B: Backend> TensorView<'_, T, B> {
    /// Number of rows in the view.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        self.row_end - self.row_start
    }

    /// Number of columns in the view.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        self.col_end - self.col_start
    }

    /// Shape as `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Read element at `(row, col)` relative to the view origin.
    #[must_use]
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(
            row < self.nrows() && col < self.ncols(),
            "TensorView::get({row}, {col}) out of bounds for view of shape {:?}",
            self.shape(),
        );
        self.source.get(self.row_start + row, self.col_start + col)
    }

    /// Create an owned [`Tensor`] by copying the viewed region.
    #[must_use]
    pub fn to_owned_tensor(&self) -> Tensor<T, B> {
        let rs = self.row_start;
        let cs = self.col_start;
        let source = self.source;
        Tensor::from_fn(self.nrows(), self.ncols(), |r, c| {
            source.get(rs + r, cs + c)
        })
    }
}

impl<T: Scalar, B: Backend> MatrixLike<T> for TensorView<'_, T, B> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        self.get(row, col)
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Create a zero-copy read-only view of a submatrix.
    ///
    /// Unlike [`.slice()`](Self::slice) which copies data, this borrows the
    /// original tensor.
    ///
    /// # Panics
    ///
    /// Panics if the row or column range is out of bounds.
    #[must_use]
    pub fn view_slice(&self, rows: Range<usize>, cols: Range<usize>) -> TensorView<'_, T, B> {
        assert!(
            rows.end <= self.nrows(),
            "view_slice: row range {}..{} out of bounds for {} rows",
            rows.start,
            rows.end,
            self.nrows(),
        );
        assert!(
            cols.end <= self.ncols(),
            "view_slice: col range {}..{} out of bounds for {} cols",
            cols.start,
            cols.end,
            self.ncols(),
        );
        assert!(
            rows.start <= rows.end,
            "view_slice: row range {}..{} is inverted",
            rows.start,
            rows.end,
        );
        assert!(
            cols.start <= cols.end,
            "view_slice: col range {}..{} is inverted",
            cols.start,
            cols.end,
        );
        TensorView {
            source: self,
            row_start: rows.start,
            row_end: rows.end,
            col_start: cols.start,
            col_end: cols.end,
        }
    }
}

#[inline]
fn next_index(idx: &mut usize, limit: usize) -> Option<usize> {
    if *idx >= limit {
        return None;
    }
    let out = *idx;
    *idx += 1;
    Some(out)
}

/// Iterator over tensor rows, yielding `1 x ncols` tensors.
pub struct RowIter<'a, T: Scalar, B: Backend> {
    pub(super) tensor: &'a Tensor<T, B>,
    pub(super) idx: usize,
}

impl<T: Scalar, B: Backend> Iterator for RowIter<'_, T, B> {
    type Item = Tensor<T, B>;

    fn next(&mut self) -> Option<Self::Item> {
        let r = next_index(&mut self.idx, self.tensor.nrows())?;
        let nc = self.tensor.ncols();
        Some(Tensor::from_fn(1, nc, |_, c| self.tensor.get(r, c)))
    }
}

/// Iterator over tensor columns, yielding `nrows x 1` tensors.
pub struct ColIter<'a, T: Scalar, B: Backend> {
    pub(super) tensor: &'a Tensor<T, B>,
    pub(super) idx: usize,
}

impl<T: Scalar, B: Backend> Iterator for ColIter<'_, T, B> {
    type Item = Tensor<T, B>;

    fn next(&mut self) -> Option<Self::Item> {
        let c = next_index(&mut self.idx, self.tensor.ncols())?;
        let nr = self.tensor.nrows();
        Some(Tensor::from_fn(nr, 1, |r, _| self.tensor.get(r, c)))
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Iterator over rows, yielding `1 x ncols` tensors.
    pub fn eachrow(&self) -> RowIter<'_, T, B> {
        RowIter {
            tensor: self,
            idx: 0,
        }
    }

    /// Iterator over columns, yielding `nrows x 1` tensors.
    pub fn eachcol(&self) -> ColIter<'_, T, B> {
        ColIter {
            tensor: self,
            idx: 0,
        }
    }

    /// Iterate over all elements in row-major order.
    ///
    /// Yields each scalar value by visiting row 0 left-to-right, then row 1, etc.
    pub fn elements(&self) -> impl Iterator<Item = T> + '_ {
        let nc = self.ncols();
        let total = self.nrows() * nc;
        (0..total).map(move |idx| self.get(idx / nc, idx % nc))
    }

    /// Iterate over all elements with their `(row, col)` indices.
    ///
    /// Yields `(row, col, value)` tuples in row-major order.
    pub fn indexed_iter(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        let nc = self.ncols();
        let total = self.nrows() * nc;
        (0..total).map(move |idx| {
            let r = idx / nc;
            let c = idx % nc;
            (r, c, self.get(r, c))
        })
    }
}
