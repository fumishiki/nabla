//! nabla — Rust linear algebra DSL.
//!
//! Import [`prelude`] for the most common types and traits.

#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        clippy::approx_constant,
        clippy::assertions_on_constants
    )
)]

#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!("nabla: exactly one backend feature must be enabled (cpu / gpu)");

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
        /// GPU kernel launch or execution failed.
        GpuKernelFailed(String),
        /// Expression evaluation failed (unbound variable, empty context, etc.).
        EvalError(String),
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
                Self::GpuKernelFailed(msg) => write!(f, "GPU kernel failed: {msg}"),
                Self::EvalError(msg) => write!(f, "eval error: {msg}"),
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

        #[inline]
        pub(crate) fn eval<T: core::fmt::Display>(msg: T) -> Self {
            Self::EvalError(msg.to_string())
        }
    }

    impl std::error::Error for Error {}
}

/// Scalar numeric types supported by nabla.
pub mod scalar;

/// Compute backends (CPU + future GPU stubs).
pub mod backend;

#[cfg(feature = "gpu")]
pub(crate) mod gpu;

/// 2-D dense tensor type with operator overloads.
pub mod tensor;

/// Dense and sparse linear algebra helpers (includes structural wrappers).
#[cfg(feature = "cpu")]
pub mod linalg;

/// Sparse matrix support.
#[cfg(feature = "cpu")]
pub mod sparse;

/// Symbolic Computer Algebra System (CAS): expression trees, differentiation, simplification.
pub mod cas;

/// ODE solvers: Euler, RK4, Dormand-Prince (adaptive).
pub mod ode;

/// Reverse-mode automatic differentiation (tape-based).
pub mod autograd;

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
    #[cfg(feature = "cpu")]
    #[inline]
    #[must_use]
    pub fn c32(re: f32, im: f32) -> crate::scalar::c32 {
        crate::scalar::c32::new(re, im)
    }

    /// Create a 64-bit complex number.
    ///
    /// # Examples
    /// ```
    /// let z = nabla::util::c64(1.0, 2.0);
    /// assert_eq!(z.re, 1.0_f64);
    /// assert_eq!(z.im, 2.0_f64);
    /// ```
    #[cfg(feature = "cpu")]
    #[inline]
    #[must_use]
    pub fn c64(re: f64, im: f64) -> crate::scalar::c64 {
        crate::scalar::c64::new(re, im)
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
            #[cfg(feature = "gpu")]
            compile_error!("bcast! is CPU-only. Use bcast_all! for GPU dispatch.");
            let __a = $a;
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| $f(__a.get(__i, __j)))
        }};
        ($f:expr, $a:expr, $b:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("bcast! is CPU-only. Use bcast_all! for GPU dispatch.");
            let __a = $a;
            let __b = $b;
            assert_eq!(__a.shape(), __b.shape(), "bcast! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j))
            })
        }};
        ($f:expr, $a:expr, $b:expr, $c:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("bcast! is CPU-only. Use bcast_all! for GPU dispatch.");
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

    /// Parallel broadcast: apply `$f` element-wise using rayon.
    ///
    /// Same semantics as [`bcast!`] but parallelises the computation across
    /// rayon's thread pool. Use for expensive per-element closures on large matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::par_bcast;
    ///
    /// let a: Tensor<f64> = Tensor::from_fn(4, 4, |i, j| (i * 4 + j) as f64);
    /// let b: Tensor<f64> = par_bcast!(|x| x * 2.0, &a);
    /// assert!((b.get(0, 1) - 2.0).abs() < 1e-12);
    /// ```
    #[macro_export]
    macro_rules! par_bcast {
        ($f:expr, $a:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("par_bcast! is CPU-only.");
            let __a = $a;
            __a.par_map($f)
        }};
        ($f:expr, $a:expr, $b:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("par_bcast! is CPU-only.");
            let __a = $a;
            let __b = $b;
            assert_eq!(__a.shape(), __b.shape(), "par_bcast! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::par_from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j))
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
            #[cfg(feature = "gpu")]
            compile_error!("zip_map! is CPU-only. Use bcast_all! for GPU dispatch.");
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
            #[cfg(feature = "gpu")]
            compile_error!("zip_map! is CPU-only. Use bcast_all! for GPU dispatch.");
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

    /// Julia-style pipe operator: `pipe!(val, f, g)` expands to `g(f(val))`.
    ///
    /// Supports an arbitrary chain of function references or closures.
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::pipe;
    /// let result = pipe!(2.0_f64, f64::sqrt, f64::ln);
    /// assert!((result - f64::ln(f64::sqrt(2.0_f64))).abs() < 1e-10);
    /// ```
    #[macro_export]
    macro_rules! pipe {
        ($val:expr) => {
            $val
        };
        ($val:expr, $f:expr) => {
            $f($val)
        };
        ($val:expr, $f:expr $(, $rest:expr)*) => {
            pipe!($f($val) $(, $rest)*)
        };
    }

    /// Julia-style splatting: unpack a tuple as function arguments.
    ///
    /// Supports tuples of arity 1–8, or an explicit list of expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::splat;
    /// fn add3(a: f64, b: f64, c: f64) -> f64 { a + b + c }
    /// assert!((splat!(add3, (1.0_f64, 2.0, 3.0)) - 6.0).abs() < 1e-12);
    /// ```
    #[macro_export]
    macro_rules! splat {
        // Tuple literal: splat!(f, (a, b, c, ...))
        ($f:expr, ($($args:expr),+ $(,)?)) => {
            $f($($args),+)
        };
    }
}

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::backend::{Backend, DefaultBackend};
    pub use crate::error::{Error, Result};
    pub use crate::scalar::Scalar;
    pub use crate::tensor::Tensor;
    pub use crate::tensor::{Array, NdTensor, StaticMatrix};
    #[cfg(feature = "cpu")]
    pub use crate::tensor::{DynTensor, Matrix};
    pub use nabla_macros::{bcast_all, einsum, generated, mat, named};

    pub use crate::autograd::{Tape, Variable};
    #[cfg(feature = "cpu")]
    pub use crate::backend::Cpu;
    pub use crate::cas::{Expr, ExprKind};
    #[cfg(feature = "cpu")]
    pub use crate::linalg::{Diagonal, Symmetric, TriKind, Triangular};
    pub use crate::ode::{AdaptiveConfig, OdeSolution};
    #[cfg(feature = "cpu")]
    pub use crate::scalar::{c32, c64};
    #[cfg(feature = "cpu")]
    pub use crate::sparse::*;
    #[cfg(feature = "cpu")]
    pub use crate::util::linspace;
    #[cfg(feature = "cpu")]
    pub use nabla_macros::stencil;

    #[cfg(feature = "gpu")]
    pub use crate::backend::Gpu;
}
