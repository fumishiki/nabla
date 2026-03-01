// view.rs — TensorView: zero-copy read-only view into a region of a Tensor.

use core::ops::Range;

use crate::backend::Backend;
use crate::matrix_like::MatrixLike;
use crate::scalar::Scalar;
use super::Tensor;

/// Zero-copy read-only view into a region of a [`Tensor`].
///
/// Created by [`Tensor::view_slice`]. Unlike [`.slice()`](Tensor::slice) which
/// copies, `TensorView` borrows the original data.
///
/// # Example
/// ```ignore
/// let a = zeros(100, 100);
/// let v = a.view_slice(0..10, 0..10);  // zero-copy
/// assert_eq!(v.get(0, 0), a.get(0, 0));
/// let owned = v.to_owned_tensor();     // copy only when needed
/// ```
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

// -- Tensor::view_slice constructor --

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
