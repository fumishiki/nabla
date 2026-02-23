// backend.rs — Sealed Backend trait + Cpu implementation backed by faer 0.24.
//
// API adaptations vs. spec (faer 0.24 differences):
//   - No `faer::Entity`; `faer::ComplexField` is the sole trait bound.
//   - `Mat::from_fn` closure receives `usize` indices for dynamic matrices.
//   - Element access uses `mat.get(row, col)` returning `&T`.
//   - matmul: `faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, one, Par::Seq)`.
//   - Scalar multiply: `mat * faer::Scale(s)`.
//   - Transpose collects via `Mat::from_fn` over the transposed view.

use faer::{linalg::matmul::matmul, Accum, Mat, Par, Scale};

use crate::scalar::Scalar;

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
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> Mat<T> {
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
        let (rows, cols) = (t.nrows(), t.ncols());
        Mat::from_fn(rows, cols, |r, c| *t.get(r, c))
    }

    #[inline]
    fn scale<T: Scalar>(a: &Mat<T>, s: T) -> Mat<T> {
        a * Scale(s)
    }
}

// ---- GPU stubs (Wave 4) ------------------------------------------------------

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]
type MatStorage<T> = Mat<T>;

#[cfg(feature = "cuda")]
/// CUDA backend stub (not yet implemented — placeholder for future GPU support).
pub struct Cuda;

#[cfg(feature = "cuda")]
impl private::Sealed for Cuda {}

#[cfg(feature = "wgpu")]
/// wgpu backend stub (not yet implemented — placeholder for future GPU support).
pub struct Wgpu;

#[cfg(feature = "wgpu")]
impl private::Sealed for Wgpu {}

#[cfg(feature = "hip")]
/// HIP backend stub (not yet implemented — placeholder for future GPU support).
pub struct Hip;

#[cfg(feature = "hip")]
impl private::Sealed for Hip {}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]
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
        }
    };
}

#[cfg(feature = "cuda")]
delegate_backend!(Cuda);

#[cfg(feature = "wgpu")]
delegate_backend!(Wgpu);

#[cfg(feature = "hip")]
delegate_backend!(Hip);

// ---- DefaultBackend (cfg-gated priority: cuda > wgpu > hip > cpu) --------------

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
