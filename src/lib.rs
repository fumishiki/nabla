#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]

//! nabla — Rust linear algebra DSL backed by faer.
//!
//! Import [`prelude`] for the most common types and traits.

/// Error types for nabla operations.
pub mod error {
    /// Convenience alias for `Result<T, Error>`.
    pub type Result<T> = core::result::Result<T, Error>;

    /// Errors that can occur in nabla operations.
    #[derive(Debug)]
    pub enum Error {
        /// Matrix shape was not compatible with the operation.
        ShapeMismatch {
            /// Expected shape `(rows, cols)`.
            expected: (usize, usize),
            /// Actual shape `(rows, cols)`.
            got: (usize, usize),
        },
        /// A dimension value was invalid for the given context.
        InvalidDimension(String),
    }

    impl core::fmt::Display for Error {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::ShapeMismatch { expected, got } => write!(
                    f,
                    "shape mismatch: expected ({}, {}), got ({}, {})",
                    expected.0, expected.1, got.0, got.1
                ),
                Self::InvalidDimension(msg) => write!(f, "invalid dimension: {msg}"),
            }
        }
    }

    impl Error {
        #[inline]
        pub(crate) fn mismatch(expected: (usize, usize), got: (usize, usize)) -> Self {
            Self::ShapeMismatch { expected, got }
        }

        #[inline]
        pub(crate) fn invalid<T: core::fmt::Display>(msg: T) -> Self {
            Self::InvalidDimension(msg.to_string())
        }
    }

    impl std::error::Error for Error {}
}

/// Scalar numeric types supported by nabla.
pub mod scalar {
    use faer_traits::ComplexField;

    /// Marker trait for numeric types supported by nabla.
    ///
    /// Implemented for `f32`, `f64`, `c32`, and `c64` — the four types that faer
    /// provides native SIMD kernels for.
    pub trait Scalar: ComplexField + Copy + Send + Sync + 'static {}

    macro_rules! impl_scalar {
        ($($ty:ty),* $(,)?) => {
            $(impl Scalar for $ty {})*
        };
    }

    impl_scalar!(f32, f64, faer::c32, faer::c64);
}

/// Compute backends (CPU + future GPU stubs).
pub mod backend;

/// 2-D dense tensor type with operator overloads.
pub mod tensor;

/// Dense and sparse linear algebra helpers (includes structural wrappers).
pub mod linalg;

/// Sparse matrix support.
pub mod sparse;

/// Utility macros and functions mirroring Julia math notation (includes broadcast macros).
pub mod util {
    /// Chain comparison: `lo <= x < hi`.
    ///
    /// # Examples
    /// ```
    /// use nabla::between;
    /// assert!(between!(0.0_f64, 0.5, 1.0));
    /// assert!(!between!(0.0_f64, 1.0, 1.0)); // upper bound is exclusive
    /// ```
    #[macro_export]
    macro_rules! between {
        ($lo:expr, $x:expr, $hi:expr) => {
            $lo <= $x && $x < $hi
        };
    }

    /// Float range with step.
    ///
    /// Generates a `Vec<f64>` from `start` up to and including `stop` when the
    /// endpoint lands within half a ULP of a step boundary.
    ///
    /// # Examples
    /// ```
    /// use nabla::frange;
    /// let v = frange!(0.0_f64, 0.25, 1.0);
    /// assert_eq!(v.len(), 5); // 0.0, 0.25, 0.5, 0.75, 1.0
    /// ```
    #[macro_export]
    macro_rules! frange {
        ($start:expr, $step:expr, $stop:expr) => {{
            let start: f64 = $start;
            let step: f64 = $step;
            let stop: f64 = $stop;
            let mut v: Vec<f64> = Vec::new();
            if step != 0.0 {
                let n = ((stop - start) / step).floor() as usize;
                for i in 0..=n {
                    let val = start + i as f64 * step;
                    if (step > 0.0 && val <= stop + step.abs() * f64::EPSILON * 2.0)
                        || (step < 0.0 && val >= stop - step.abs() * f64::EPSILON * 2.0)
                    {
                        v.push(val);
                    }
                }
            }
            v
        }};
    }

    /// Create a 32-bit complex number.
    ///
    /// # Examples
    /// ```
    /// let z = nabla::util::c32(1.0, 2.0);
    /// assert_eq!(z.re, 1.0_f32);
    /// assert_eq!(z.im, 2.0_f32);
    /// ```
    #[inline]
    #[must_use]
    pub fn c32(re: f32, im: f32) -> faer::c32 {
        faer::c32::new(re, im)
    }

    /// Create a 64-bit complex number.
    ///
    /// # Examples
    /// ```
    /// let z = nabla::util::c64(1.0, 2.0);
    /// assert_eq!(z.re, 1.0_f64);
    /// assert_eq!(z.im, 2.0_f64);
    /// ```
    #[inline]
    #[must_use]
    pub fn c64(re: f64, im: f64) -> faer::c64 {
        faer::c64::new(re, im)
    }

    /// Linearly spaced vector.
    ///
    /// Returns `n` evenly spaced points from `start` to `stop` inclusive.
    ///
    /// - `n = 0` → empty `Vec`
    /// - `n = 1` → `vec![start]`
    ///
    /// # Examples
    /// ```
    /// let v = nabla::util::linspace(0.0, 1.0, 5);
    /// assert_eq!(v.len(), 5);
    /// assert!((v[0] - 0.0).abs() < 1e-10);
    /// assert!((v[4] - 1.0).abs() < 1e-10);
    /// assert!((v[2] - 0.5).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
        match n {
            0 => Vec::new(),
            1 => vec![start],
            _ => {
                #[allow(clippy::cast_precision_loss)]
                let delta = (stop - start) / (n as f64 - 1.0);
                #[allow(clippy::cast_precision_loss)]
                (0..n).map(|i| start + i as f64 * delta).collect()
            }
        }
    }

    /// Element-wise broadcast over same-shape tensors. Returns a new `Tensor`.
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::bcast;
    ///
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    /// // Unary: double every element.
    /// let doubled: Tensor<f64> = bcast!(|x| x * 2.0, &a);
    /// assert_eq!(doubled.get(0, 0), 2.0);
    ///
    /// let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j) as f64);
    /// // Binary: element-wise sum.
    /// let s: Tensor<f64> = bcast!(|x, y| x + y, &a, &b);
    /// assert_eq!(s.get(1, 1), 6.0); // a[1,1]=4, b[1,1]=2 → 6
    /// ```
    #[macro_export]
    macro_rules! bcast {
        ($f:expr, $a:expr) => {{
            let __a = $a;
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| $f(__a.get(__i, __j)))
        }};
        ($f:expr, $a:expr, $b:expr) => {{
            let __a = $a;
            let __b = $b;
            assert_eq!(__a.shape(), __b.shape(), "bcast! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j))
            })
        }};
        ($f:expr, $a:expr, $b:expr, $c:expr) => {{
            let __a = $a;
            let __b = $b;
            let __c_t = $c;
            assert_eq!(__a.shape(), __b.shape(), "bcast! shape mismatch");
            assert_eq!(__a.shape(), __c_t.shape(), "bcast! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j), __c_t.get(__i, __j))
            })
        }};
    }

    /// In-place broadcast: mutate `$out` element-wise.
    ///
    /// Requires that `$out` and every source tensor share the same shape.
    /// Uses [`crate::tensor::Tensor::set`] for each element; no temporary allocation is needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::zip_map;
    ///
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    /// let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    /// let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    /// zip_map!(out, |x, y| x * y, &a, &b);
    /// assert_eq!(out.get(0, 0), 2.0);
    /// ```
    #[macro_export]
    macro_rules! zip_map {
        ($out:expr, $f:expr, $a:expr) => {{
            let __a = $a;
            let (__r, __c) = $out.shape();
            assert_eq!((__r, __c), __a.shape(), "zip_map! shape mismatch");
            for __i in 0..__r {
                for __j in 0..__c {
                    $out.set(__i, __j, $f(__a.get(__i, __j)));
                }
            }
        }};
        ($out:expr, $f:expr, $a:expr, $b:expr) => {{
            let __a = $a;
            let __b = $b;
            let (__r, __c) = $out.shape();
            assert_eq!((__r, __c), __a.shape(), "zip_map! shape mismatch");
            assert_eq!((__r, __c), __b.shape(), "zip_map! shape mismatch");
            for __i in 0..__r {
                for __j in 0..__c {
                    $out.set(__i, __j, $f(__a.get(__i, __j), __b.get(__i, __j)));
                }
            }
        }};
    }
}

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::backend::{Backend, Cpu, DefaultBackend};
    pub use crate::error::{Error, Result};
    pub use crate::linalg::{Diagonal, Symmetric, TriKind, Triangular};
    pub use crate::scalar::Scalar;
    pub use crate::sparse::*;
    pub use crate::tensor::Tensor;
    pub use crate::tensor::{Array, Matrix, StaticMatrix};
    pub use crate::util::{c32, c64, linspace};
    pub use nabla_macros::{einsum, mat};

    #[cfg(feature = "cuda")]
    pub use crate::backend::Cuda;
    #[cfg(feature = "wgpu")]
    pub use crate::backend::Wgpu;
}
