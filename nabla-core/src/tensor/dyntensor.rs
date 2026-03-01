// dyntensor.rs — Array/Matrix traits and DynTensor enum.

use super::static_matrix::StaticMatrix;
use super::Tensor;
use crate::backend::Backend;
use crate::scalar::Scalar;

#[cfg(feature = "cpu")]
use crate::backend::Cpu;

/// Base trait for any 2-D array-like type.
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
/// Matrix algebra trait extending [`Array`].
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

// -- DynTensor: enum-based closed multiple dispatch --

#[cfg(feature = "cpu")]
/// A type-erased 2-D dense matrix that can live on any enabled backend.
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
