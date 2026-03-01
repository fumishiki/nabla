// tensor/iter.rs — RowIter, ColIter, eachrow, eachcol iterators.

use crate::backend::Backend;
use crate::scalar::Scalar;

use super::Tensor;

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
        RowIter { tensor: self, idx: 0 }
    }

    /// Iterator over columns, yielding `nrows x 1` tensors.
    pub fn eachcol(&self) -> ColIter<'_, T, B> {
        ColIter { tensor: self, idx: 0 }
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
