
use core::fmt;
use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::Tensor;
use super::fmt_matrix;
use crate::backend::{Backend, DefaultBackend};
use crate::scalar::Scalar;

#[cfg(feature = "cpu")]
use crate::backend::Cpu;


#[derive(Clone)]
/// N-dimensional tensor backed by a flat `Vec<T>`.
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
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Reshape to a new shape. Total elements must match.
    #[must_use]
    pub fn reshape_nd(&self, new_shape: &[usize]) -> Self {
        let total: usize = new_shape.iter().product();
        let old_total: usize = self.shape.iter().product();
        assert_eq!(
            total, old_total,
            "nabla: reshape_nd total mismatch: {total} ≠ {old_total}"
        );
        let strides = Self::compute_strides(new_shape);
        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides,
        }
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


#[derive(Clone, Copy)]
/// Stack-allocated `R x C` matrix with const-generic dimensions.
pub struct StaticMatrix<T: Scalar, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T: Scalar, const R: usize, const C: usize> StaticMatrix<T, R, C> {
    /// Zero-filled `R x C` matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            data: [[T::zero_impl(); C]; R],
        }
    }

    /// Matrix whose `(r, c)` element is `f(r, c)`.
    #[must_use]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = [[T::zero_impl(); C]; R];
        for (r, row) in data.iter_mut().enumerate() {
            for (c, elem) in row.iter_mut().enumerate() {
                *elem = f(r, c);
            }
        }
        Self { data }
    }

    /// Identity-like matrix: `1` on the diagonal, `0` elsewhere.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_fn(|r, c| {
            if r == c {
                T::one_impl()
            } else {
                T::zero_impl()
            }
        })
    }

    /// Number of rows (always `R`).
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        R
    }

    /// Number of columns (always `C`).
    #[must_use]
    #[inline]
    pub const fn ncols(&self) -> usize {
        C
    }

    /// Shape `(R, C)`.
    #[must_use]
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Read element at `(row, col)`.
    #[must_use]
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row][col]
    }

    /// Write element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[row][col] = val;
    }

    /// Transpose: returns a new `C x R` static matrix (stack-only).
    #[must_use]
    pub fn t(&self) -> StaticMatrix<T, C, R> {
        StaticMatrix::<T, C, R>::from_fn(|r, c| self.data[c][r])
    }

    /// Conjugate-transpose (adjoint).
    #[must_use]
    pub fn adjoint(&self) -> StaticMatrix<T, C, R> {
        if T::IS_REAL {
            self.t()
        } else {
            StaticMatrix::<T, C, R>::from_fn(|r, c| {
                crate::scalar::math_utils::conj(&self.data[c][r])
            })
        }
    }

    /// Short alias for conjugate-transpose (`adjoint`).
    #[must_use]
    #[inline]
    pub fn h(&self) -> StaticMatrix<T, C, R> {
        self.adjoint()
    }

    /// Matrix multiply `self (RxK) * rhs (KxN)` -> `StaticMatrix<T, R, N>`.
    #[must_use]
    pub fn matmul<const N: usize>(&self, rhs: &StaticMatrix<T, C, N>) -> StaticMatrix<T, R, N> {
        StaticMatrix::<T, R, N>::from_fn(|r, c| {
            (0..C).fold(T::zero_impl(), |acc, k| {
                acc + self.data[r][k] * rhs.data[k][c]
            })
        })
    }

    /// Outer product of two vectors: `u[i] * v[j]` -> `R x C` matrix.
    ///
    /// `out[i][j] = u[i] * v[j]`
    #[must_use]
    pub fn outer(u: &[T; R], v: &[T; C]) -> Self {
        Self::from_fn(|r, c| u[r] * v[c])
    }

    /// Read-only access to the raw `[[T; C]; R]` backing array.
    #[must_use]
    #[inline]
    pub fn data(&self) -> &[[T; C]; R] {
        &self.data
    }

    /// Convert to a heap-allocated [`Tensor`] (copies all elements).
    #[must_use]
    pub fn to_tensor(&self) -> Tensor<T, DefaultBackend> {
        Tensor::from_fn(R, C, |r, c| self.data[r][c])
    }

    /// Build from a [`Tensor`] (copies all elements).
    #[must_use]
    pub fn from_tensor(tensor: &Tensor<T, DefaultBackend>) -> Self {
        assert_eq!(
            tensor.shape(),
            (R, C),
            "StaticMatrix::from_tensor shape mismatch: expected ({R}, {C}), got {:?}",
            tensor.shape()
        );
        Self::from_fn(|r, c| tensor.get(r, c))
    }
}

macro_rules! impl_static_binop {
    (binary: $trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar, const R: usize, const C: usize> $trait for StaticMatrix<T, R, C> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                Self::from_fn(|r, c| self.data[r][c] $op rhs.data[r][c])
            }
        }
    };
    (unary: $trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar, const R: usize, const C: usize> $trait for StaticMatrix<T, R, C> {
            type Output = Self;
            fn $method(self) -> Self {
                Self::from_fn(|r, c| $op self.data[r][c])
            }
        }
    };
    (scalar: $trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar, const R: usize, const C: usize> $trait<T> for StaticMatrix<T, R, C> {
            type Output = Self;
            fn $method(self, rhs: T) -> Self {
                Self::from_fn(|r, c| self.data[r][c] $op rhs)
            }
        }
    };
}

impl_static_binop!(binary: Add, add, +);
impl_static_binop!(binary: Sub, sub, -);
impl_static_binop!(unary: Neg, neg, -);
impl_static_binop!(scalar: Mul, mul, *);

impl<T: Scalar, const R: usize, const K: usize, const N: usize> Mul<StaticMatrix<T, K, N>>
    for StaticMatrix<T, R, K>
{
    type Output = StaticMatrix<T, R, N>;
    fn mul(self, rhs: StaticMatrix<T, K, N>) -> Self::Output {
        self.matmul(&rhs)
    }
}


impl<T: Scalar, const R: usize, const C: usize> Add<&StaticMatrix<T, R, C>>
    for &StaticMatrix<T, R, C>
{
    type Output = StaticMatrix<T, R, C>;
    fn add(self, rhs: &StaticMatrix<T, R, C>) -> Self::Output {
        StaticMatrix::from_fn(|r, c| self.data[r][c] + rhs.data[r][c])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Sub<&StaticMatrix<T, R, C>>
    for &StaticMatrix<T, R, C>
{
    type Output = StaticMatrix<T, R, C>;
    fn sub(self, rhs: &StaticMatrix<T, R, C>) -> Self::Output {
        StaticMatrix::from_fn(|r, c| self.data[r][c] - rhs.data[r][c])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Neg for &StaticMatrix<T, R, C> {
    type Output = StaticMatrix<T, R, C>;
    fn neg(self) -> Self::Output {
        StaticMatrix::from_fn(|r, c| -self.data[r][c])
    }
}

impl<T: Scalar, const R: usize, const C: usize> Mul<T> for &StaticMatrix<T, R, C> {
    type Output = StaticMatrix<T, R, C>;
    fn mul(self, rhs: T) -> Self::Output {
        StaticMatrix::from_fn(|r, c| self.data[r][c] * rhs)
    }
}

impl<T: Scalar, const R: usize, const K: usize, const N: usize> Mul<&StaticMatrix<T, K, N>>
    for &StaticMatrix<T, R, K>
{
    type Output = StaticMatrix<T, R, N>;
    fn mul(self, rhs: &StaticMatrix<T, K, N>) -> Self::Output {
        self.matmul(rhs)
    }
}

impl<T: Scalar, const R: usize, const C: usize> Index<(usize, usize)> for StaticMatrix<T, R, C> {
    type Output = T;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row][col]
    }
}

impl<T: Scalar, const R: usize, const C: usize> IndexMut<(usize, usize)> for StaticMatrix<T, R, C> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[row][col]
    }
}

impl<T: Scalar + fmt::Display, const R: usize, const C: usize> fmt::Display
    for StaticMatrix<T, R, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(R, C, |r, c, f| write!(f, "{}", self.data[r][c]), f, None)
    }
}

impl<T: Scalar + fmt::Debug, const R: usize, const C: usize> fmt::Debug for StaticMatrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = format!("StaticMatrix({R}x{C})");
        fmt_matrix(
            R,
            C,
            |r, c, f| write!(f, "{:?}", self.data[r][c]),
            f,
            Some(&prefix),
        )
    }
}


/// Dynamic dispatch trait for 2-D array read access.
pub trait Array<T: Scalar> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Shape as `(nrows, ncols)`.
    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    /// Read element at `(row, col)`.
    fn get(&self, row: usize, col: usize) -> T;
}

#[cfg(feature = "cpu")]
/// CPU-backed 2-D matrix with dynamic dispatch for transpose and matmul.
pub trait Matrix<T: Scalar>: Array<T> {
    /// Transpose into a CPU-backed [`Tensor`].
    fn t_dyn(&self) -> Tensor<T, Cpu>;
    /// Matrix multiply `self x rhs` into a CPU-backed [`Tensor`].
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu>;
}

#[cfg(feature = "cpu")]
pub(super) fn dyn_matmul<T: Scalar>(lhs: &dyn Array<T>, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
    let (m, k) = lhs.shape();
    let (k2, n) = rhs.shape();
    assert_eq!(
        k, k2,
        "matmul_dyn inner dimension mismatch: lhs k={k}, rhs k={k2}"
    );
    Tensor::from_fn(m, n, |r, c| {
        (0..k).fold(T::zero_impl(), |acc, i| acc + lhs.get(r, i) * rhs.get(i, c))
    })
}

impl<T: Scalar, B: Backend> Array<T> for Tensor<T, B> {
    #[inline]
    fn nrows(&self) -> usize {
        Tensor::nrows(self)
    }
    #[inline]
    fn ncols(&self) -> usize {
        Tensor::ncols(self)
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        Tensor::get(self, row, col)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Matrix<T> for Tensor<T, B> {
    fn t_dyn(&self) -> Tensor<T, Cpu> {
        let (r, c) = self.shape();
        Tensor::from_fn(c, r, |i, j| self.get(j, i))
    }
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
        dyn_matmul(self, rhs)
    }
}

impl<T: Scalar, const R: usize, const C: usize> Array<T> for StaticMatrix<T, R, C> {
    #[inline]
    fn nrows(&self) -> usize {
        R
    }
    #[inline]
    fn ncols(&self) -> usize {
        C
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        StaticMatrix::get(self, row, col)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, const R: usize, const C: usize> Matrix<T> for StaticMatrix<T, R, C> {
    fn t_dyn(&self) -> Tensor<T, Cpu> {
        Tensor::from_fn(C, R, |r, c| self.get(c, r))
    }
    fn matmul_dyn(&self, rhs: &dyn Array<T>) -> Tensor<T, Cpu> {
        dyn_matmul(self, rhs)
    }
}


#[cfg(feature = "cpu")]
/// Dynamically-typed tensor for runtime backend dispatch.
pub enum DynTensor {
    /// CPU backend (always available).
    Cpu(Tensor<f32, Cpu>),
}

#[cfg(feature = "cpu")]
macro_rules! dyn_dispatch {
    (ref $self:expr, $method:ident) => {
        match $self { DynTensor::Cpu(t) => t.$method() }
    };
    (binop $self:expr, $rhs:expr, $op:tt) => {
        match ($self, $rhs) { (DynTensor::Cpu(a), DynTensor::Cpu(b)) => DynTensor::Cpu(a $op b) }
    };
}

#[cfg(feature = "cpu")]
impl DynTensor {
    #[inline]
    fn cpu(&self) -> &Tensor<f32, Cpu> {
        match self {
            Self::Cpu(t) => t,
        }
    }

    /// Construct a `DynTensor` on the CPU backend.
    pub fn cpu_f32(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> f32) -> Self {
        Self::Cpu(Tensor::from_fn(nrows, ncols, f))
    }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.cpu().nrows()
    }
    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cpu().ncols()
    }
    /// Shape as `(nrows, ncols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Read element at `(row, col)`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.cpu().get(row, col)
    }

    /// Element-wise addition. Panics if shapes differ.
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        dyn_dispatch!(binop self, rhs, +)
    }
    /// Element-wise subtraction. Panics if shapes differ.
    #[must_use]
    pub fn sub(&self, rhs: &Self) -> Self {
        dyn_dispatch!(binop self, rhs, -)
    }

    /// Matrix multiplication (`self x rhs`). Panics if shapes are incompatible.
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Self::Cpu(a), Self::Cpu(b)) => {
                let mut out = Tensor::<f32, Cpu>::zeros(a.nrows(), b.ncols());
                Tensor::matmul_into(&mut out, a, b);
                Self::Cpu(out)
            }
        }
    }

    /// Move to CPU-backed tensor (always a clone for the Cpu variant).
    #[must_use]
    pub fn to_cpu(&self) -> Tensor<f32, Cpu> {
        self.cpu().clone()
    }
}
