// ndtensor.rs — NdTensor<T>: N-dimensional tensor stored as a flat Vec<T>.

use core::fmt;
use core::ops::Index;

use super::Tensor;
use crate::backend::DefaultBackend;
use crate::scalar::Scalar;

/// N-dimensional tensor stored as a flat `Vec<T>` in row-major (C-order) layout.
#[derive(Clone)]
pub struct NdTensor<T: Scalar> {
    pub(super) data: Vec<T>,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
}

impl<T: Scalar> NdTensor<T> {
    /// Compute row-major strides from shape via reverse cumulative product.
    pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    /// Convert N-D indices to a flat linear index.
    fn linear_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(
            indices.len(),
            self.shape.len(),
            "NdTensor: expected {} indices, got {}",
            self.shape.len(),
            indices.len()
        );
        indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }

    /// Allocate a zero-filled tensor of the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![T::zero(); n],
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        }
    }

    /// Allocate a tensor whose element at `indices` is `f(indices)`.
    #[must_use]
    pub fn from_fn(shape: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Self {
        let n: usize = shape.iter().product();
        let strides = Self::compute_strides(shape);
        let ndim = shape.len();
        let mut data = Vec::with_capacity(n);
        let mut indices = vec![0usize; ndim];
        for _ in 0..n {
            data.push(f(&indices));
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Number of dimensions.
    #[must_use]
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of dimension `axis`.
    #[must_use]
    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    /// Shape as a slice.
    #[must_use]
    #[inline]
    pub fn shape_vec(&self) -> &[usize] {
        &self.shape
    }

    /// Read element at the given N-D indices.
    #[must_use]
    #[inline]
    pub fn get_nd(&self, indices: &[usize]) -> T {
        let idx = self.linear_index(indices);
        self.data[idx]
    }

    /// Write element at the given N-D indices.
    #[inline]
    pub fn set_nd(&mut self, indices: &[usize], val: T) {
        let idx = self.linear_index(indices);
        self.data[idx] = val;
    }

    /// Total number of elements.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty (any dimension is 0).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn slice_2d_meta(
        &self,
        batch_indices: &[usize],
        who: &str,
    ) -> (usize, usize, usize, usize, usize) {
        let ndim = self.ndim();
        assert_eq!(
            batch_indices.len() + 2,
            ndim,
            "NdTensor::{who}: expected {} batch indices, got {}",
            ndim - 2,
            batch_indices.len()
        );
        let nrows = self.shape[ndim - 2];
        let ncols = self.shape[ndim - 1];
        let base_offset: usize = batch_indices
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum();
        let row_stride = self.strides[ndim - 2];
        let col_stride = self.strides[ndim - 1];
        (base_offset, nrows, ncols, row_stride, col_stride)
    }

    /// Extract a 2-D [`Tensor`] from the last two dimensions, fixing
    /// all preceding batch dimensions.
    #[must_use]
    pub fn slice_2d(&self, batch_indices: &[usize]) -> Tensor<T> {
        let (base_offset, nrows, ncols, row_stride, col_stride) =
            self.slice_2d_meta(batch_indices, "slice_2d");
        Tensor::from_fn(nrows, ncols, |r, c| {
            self.data[base_offset + r * row_stride + c * col_stride]
        })
    }

    /// Set a 2-D slice in the last two dimensions, fixing all
    /// preceding batch dimensions.
    pub fn set_slice_2d(&mut self, batch_indices: &[usize], tensor: &Tensor<T>) {
        let (base_offset, nrows, ncols, row_stride, col_stride) =
            self.slice_2d_meta(batch_indices, "set_slice_2d");
        for r in 0..nrows {
            for c in 0..ncols {
                self.data[base_offset + r * row_stride + c * col_stride] = tensor.get(r, c);
            }
        }
    }

    /// Convenience: number of rows for the last-2 dimension.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        assert!(self.ndim() >= 2, "NdTensor::nrows requires ndim >= 2");
        self.shape[self.ndim() - 2]
    }

    /// Convenience: number of cols for the last dimension.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        assert!(self.ndim() >= 1, "NdTensor::ncols requires ndim >= 1");
        self.shape[self.ndim() - 1]
    }

    /// Reorder axes. `axes` must be a permutation of `0..ndim()`.
    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        let nd = self.ndim();
        assert_eq!(
            axes.len(),
            nd,
            "nabla: permute axes length ({}) != ndim ({nd})",
            axes.len()
        );
        let mut seen = vec![false; nd];
        for &a in axes {
            assert!(a < nd, "nabla: permute axis {a} >= ndim {nd}");
            assert!(!seen[a], "nabla: permute duplicate axis {a}");
            seen[a] = true;
        }
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.dim(a)).collect();
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig_idx = vec![0usize; nd];
            for (new_axis, &orig_axis) in axes.iter().enumerate() {
                orig_idx[orig_axis] = idx[new_axis];
            }
            self.get_nd(&orig_idx)
        })
    }

    /// Remove a size-1 axis.
    #[must_use]
    pub fn squeeze(&self, axis: usize) -> Self {
        let nd = self.ndim();
        assert!(axis < nd, "nabla: squeeze axis {axis} >= ndim {nd}");
        let d = self.dim(axis);
        assert_eq!(d, 1, "nabla: squeeze dim({axis}) is {d}, not 1");
        let new_shape: Vec<usize> = (0..nd)
            .filter(|&i| i != axis)
            .map(|i| self.dim(i))
            .collect();
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig = idx.to_vec();
            orig.insert(axis, 0);
            self.get_nd(&orig)
        })
    }

    /// Insert a size-1 axis at `axis`.
    #[must_use]
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let nd = self.ndim();
        assert!(axis <= nd, "nabla: unsqueeze axis {axis} > ndim {nd}");
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig = idx.to_vec();
            orig.remove(axis);
            self.get_nd(&orig)
        })
    }

    /// Create from shape and flat data.
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total,
            "nabla: NdTensor::from_vec data length {} ≠ shape product {}",
            data.len(),
            total
        );
        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Reshape to a new shape. Total elements must match.
    #[must_use]
    pub fn reshape_nd(&self, new_shape: &[usize]) -> Self {
        let total: usize = new_shape.iter().product();
        let old_total: usize = self.shape.iter().product();
        assert_eq!(
            total,
            old_total,
            "nabla: reshape_nd total mismatch: {total} ≠ {old_total}"
        );
        let strides = Self::compute_strides(new_shape);
        Self { data: self.data.clone(), shape: new_shape.to_vec(), strides }
    }

    /// Convert N-D tensor to 2D Tensor by treating the last dimension as columns
    /// and all preceding dimensions combined as rows.
    pub fn into_2d(self) -> Tensor<T, DefaultBackend> {
        let total: usize = self.shape.iter().product();
        let cols = *self.shape.last().unwrap_or(&1);
        let rows = if cols == 0 { 0 } else { total / cols };
        Tensor::from_fn(rows, cols, |r, c| self.data[r * cols + c])
    }
}

impl<T: Scalar + fmt::Debug> fmt::Debug for NdTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdTensor(shape={:?}, data={:?})", self.shape, self.data)
    }
}

impl<T: Scalar> Index<&[usize]> for NdTensor<T> {
    type Output = T;

    #[inline]
    fn index(&self, indices: &[usize]) -> &T {
        assert!(
            indices.len() == self.shape.len(),
            "nabla: NdTensor expected {} indices, got {}",
            self.shape.len(),
            indices.len()
        );
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            assert!(
                idx < dim,
                "nabla: NdTensor index[{i}]={idx} out of bounds for dim {dim}"
            );
        }
        let flat = self.linear_index(indices);
        &self.data[flat]
    }
}
