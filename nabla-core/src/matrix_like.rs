// matrix_like.rs — Common read-only matrix interface (Julia `AbstractMatrix` equivalent).

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::{StaticMatrix, Tensor};

/// Common read-only matrix interface (Julia `AbstractMatrix` equivalent).
///
/// Implemented by [`Tensor<T,B>`], [`StaticMatrix<T,R,C>`], and
/// [`TensorView<T,B>`](crate::tensor::TensorView).
pub trait MatrixLike<T: Scalar> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Shape as `(rows, cols)`.
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    /// Element access (read-only).
    fn get(&self, row: usize, col: usize) -> T;
    /// Total number of elements.
    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }
    /// Whether the matrix is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// -- impl for Tensor<T, B, Axes> --

impl<T: Scalar, B: Backend, Axes> MatrixLike<T> for Tensor<T, B, Axes> {
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

// -- impl for StaticMatrix<T, R, C> --

impl<T: Scalar, const R: usize, const C: usize> MatrixLike<T> for StaticMatrix<T, R, C> {
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
