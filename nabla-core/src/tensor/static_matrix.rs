// static_matrix.rs — StaticMatrix<T, R, C>: stack-allocated fixed-size matrix.

use core::fmt;
use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::display::fmt_matrix;
use crate::backend::DefaultBackend;
use crate::scalar::Scalar;
use super::Tensor;

/// A stack-allocated `R x C` matrix with element type `T`.
#[derive(Clone, Copy)]
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

/// Multiply `StaticMatrix<T, R, K>` by `StaticMatrix<T, K, C>`.
/// Compile-time error if inner dimensions don't match.
impl<T: Scalar, const R: usize, const K: usize, const N: usize> Mul<StaticMatrix<T, K, N>>
    for StaticMatrix<T, R, K>
{
    type Output = StaticMatrix<T, R, N>;
    fn mul(self, rhs: StaticMatrix<T, K, N>) -> Self::Output {
        self.matmul(&rhs)
    }
}

// Reference-based operators for StaticMatrix (match nabla convention: &a op &b).

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

impl<T: Scalar, const R: usize, const C: usize> IndexMut<(usize, usize)>
    for StaticMatrix<T, R, C>
{
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
