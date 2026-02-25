//! nabla — Rust linear algebra DSL.
//!
//! Import [`prelude`] for the most common types and traits.

#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]
#![allow(
    clippy::return_self_not_must_use,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::type_complexity
)]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        clippy::approx_constant,
        clippy::assertions_on_constants
    )
)]

// Re-export nabla-core modules for path-based access.
pub use nabla_core::backend;
pub use nabla_core::error;
pub use nabla_core::layout;
pub use nabla_core::scalar;
pub use nabla_core::tensor;
pub use nabla_core::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};

#[cfg(feature = "gpu")]
pub use nabla_core::gpu;

/// WGSL shader generators (pure string ops, always compiled).
pub use nabla_core::wgsl;

/// CUDA Graph capture/replay for eliminating kernel launch overhead.
#[cfg(feature = "cuda")]
pub use nabla_core::{NablaCudaGraph, cuda_graph_capture, cuda_graph_capture_cached};

/// Dense linear algebra factorizations and solvers.
#[cfg(feature = "cpu")]
pub mod linalg;

/// Sparse matrix support.
#[cfg(feature = "cpu")]
pub mod sparse;

/// Reverse-mode automatic differentiation.
pub mod autograd;

/// Symbolic computer algebra system.
pub mod cas;

/// ODE solvers: Euler, RK4, Dormand-Prince.
pub mod ode;

#[inline]
fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(dur) => {
            let nanos = dur.as_nanos();
            #[allow(clippy::cast_possible_truncation)]
            {
                (nanos as u64) ^ ((nanos >> 64) as u64)
            }
        }
        Err(_) => 0xA11C_E5EE_D5EE_DBAD_u64,
    }
}

/// Allocate a zero-filled tensor of shape `(nrows, ncols)`.
#[must_use]
#[inline]
pub fn zeros<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    tensor::Tensor::zeros(nrows, ncols)
}

/// Allocate a one-filled tensor of shape `(nrows, ncols)`.
#[must_use]
#[inline]
pub fn ones<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    tensor::Tensor::fill(nrows, ncols, T::one())
}

/// Allocate a tensor of shape `(nrows, ncols)` filled with `value`.
#[must_use]
#[inline]
pub fn fill<T: scalar::Scalar>(nrows: usize, ncols: usize, value: T) -> tensor::Tensor<T> {
    tensor::Tensor::fill(nrows, ncols, value)
}

/// Allocate an identity matrix of size `n x n`.
#[must_use]
#[inline]
pub fn eye<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
    tensor::Tensor::identity(n)
}

/// Allocate a tensor whose element `(r, c)` is `f(r, c)`.
#[must_use]
#[inline]
pub fn from_fn<T: scalar::Scalar>(
    nrows: usize,
    ncols: usize,
    f: impl FnMut(usize, usize) -> T,
) -> tensor::Tensor<T> {
    tensor::Tensor::from_fn(nrows, ncols, f)
}

/// Allocate a zero-filled N-D tensor.
#[must_use]
#[inline]
pub fn nd_zeros<T: scalar::Scalar>(shape: &[usize]) -> tensor::NdTensor<T> {
    tensor::NdTensor::zeros(shape)
}

/// Uniform random tensor in `[0, 1)`.
#[must_use]
pub fn rand<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    let mut s = {
        let seed = default_seed();
        if seed == 0 {
            0x1234_5678_9ABC_DEF0_u64
        } else {
            seed
        }
    };
    tensor::Tensor::from_fn(nrows, ncols, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        T::from_f64((s as f64) / (u64::MAX as f64))
    })
}

/// Standard normal random tensor (`mean=0`, `std=1`).
#[must_use]
pub fn randn<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
    let mut s = {
        let seed = default_seed();
        if seed == 0 {
            0x1234_5678_9ABC_DEF0_u64
        } else {
            seed
        }
    };
    let mut xorshift = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64) / (u64::MAX as f64)
    };
    let n = nrows * ncols;
    let mut data = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let u1 = xorshift().max(1e-300);
        let u2 = xorshift();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(T::from_f64(r * theta.cos()));
        if i + 1 < n {
            data.push(T::from_f64(r * theta.sin()));
        }
        i += 2;
    }
    tensor::Tensor::from_fn(nrows, ncols, |r, c| data[r * ncols + c])
}

/// 1-D half-open range tensor: `[start, start+step, ..., < stop]`.
#[must_use]
pub fn arange<T: scalar::Scalar>(start: T, stop: T, step: T) -> tensor::Tensor<T> {
    let step_f = step.to_f64();
    assert!(
        step_f.is_finite() && step_f != 0.0,
        "nabla: arange step must be non-zero finite, got {step_f}"
    );
    let stop_f = stop.to_f64();
    let mut cur = start.to_f64();
    let mut n = 0usize;

    if step_f > 0.0 {
        while cur < stop_f {
            n += 1;
            cur += step_f;
        }
    } else {
        while cur > stop_f {
            n += 1;
            cur += step_f;
        }
    }

    tensor::Tensor::arange(start, step, n)
}

/// Utility macros and functions mirroring Julia math notation.
pub mod util {
    /// Chain comparison: `lo <= x < hi`.
    ///
    /// # Examples
    /// ```
    /// use nabla::between;
    /// assert!(between!(0.0_f64, 0.5, 1.0));
    /// assert!(!between!(0.0_f64, 1.0, 1.0));
    /// ```
    #[macro_export]
    macro_rules! between {
        ($lo:expr, $x:expr, $hi:expr) => {
            $lo <= $x && $x < $hi
        };
    }

    /// Float range with step.
    ///
    /// # Examples
    /// ```
    /// use nabla::frange;
    /// let v = frange!(0.0_f64, 0.25, 1.0);
    /// assert_eq!(v.len(), 5);
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
    #[cfg(feature = "cpu")]
    #[inline]
    #[must_use]
    pub fn c32(re: f32, im: f32) -> crate::scalar::c32 {
        crate::scalar::c32::new(re, im)
    }

    /// Create a 64-bit complex number.
    #[cfg(feature = "cpu")]
    #[inline]
    #[must_use]
    pub fn c64(re: f64, im: f64) -> crate::scalar::c64 {
        crate::scalar::c64::new(re, im)
    }

    /// Linearly spaced vector from `start` to `stop` inclusive.
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

    /// Element-wise broadcast (allocating). Renamed from `bcast!`.
    ///
    /// # Examples
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::map;
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    /// let doubled: Tensor<f64> = map!(|x| x * 2.0, &a);
    /// assert_eq!(doubled.get(0, 0), 2.0);
    /// ```
    #[macro_export]
    macro_rules! map {
        ($f:expr, $a:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("map! is CPU-only. Use fuse! for GPU dispatch.");
            let __a = $a;
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| $f(__a.get(__i, __j)))
        }};
        ($f:expr, $a:expr, $b:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("map! is CPU-only. Use fuse! for GPU dispatch.");
            let __a = $a;
            let __b = $b;
            assert_eq!(__a.shape(), __b.shape(), "map! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j))
            })
        }};
        ($f:expr, $a:expr, $b:expr, $c:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("map! is CPU-only. Use fuse! for GPU dispatch.");
            let __a = $a;
            let __b = $b;
            let __c_t = $c;
            assert_eq!(__a.shape(), __b.shape(), "map! shape mismatch");
            assert_eq!(__a.shape(), __c_t.shape(), "map! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j), __c_t.get(__i, __j))
            })
        }};
    }

    /// Parallel element-wise broadcast via rayon. Renamed from `par_bcast!`.
    ///
    /// # Examples
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::par_map;
    /// let a: Tensor<f64> = Tensor::from_fn(4, 4, |i, j| (i * 4 + j) as f64);
    /// let b: Tensor<f64> = par_map!(|x| x * 2.0, &a);
    /// assert!((b.get(0, 1) - 2.0).abs() < 1e-12);
    /// ```
    #[macro_export]
    macro_rules! par_map {
        ($f:expr, $a:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("par_map! is CPU-only.");
            let __a = $a;
            __a.par_map($f)
        }};
        ($f:expr, $a:expr, $b:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("par_map! is CPU-only.");
            let __a = $a;
            let __b = $b;
            assert_eq!(__a.shape(), __b.shape(), "par_map! shape mismatch");
            let (__r, __c) = __a.shape();
            $crate::tensor::Tensor::par_from_fn(__r, __c, |__i, __j| {
                $f(__a.get(__i, __j), __b.get(__i, __j))
            })
        }};
    }

    /// In-place element-wise broadcast. Renamed from `zip_map!`.
    ///
    /// # Examples
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::map_;
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    /// let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    /// let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    /// map_!(out, |x, y| x * y, &a, &b);
    /// assert_eq!(out.get(0, 0), 2.0);
    /// ```
    #[macro_export]
    macro_rules! map_ {
        ($out:expr, $f:expr, $a:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("map_! is CPU-only. Use fuse! for GPU dispatch.");
            let __a = $a;
            let (__r, __c) = $out.shape();
            assert_eq!((__r, __c), __a.shape(), "map_! shape mismatch");
            for __i in 0..__r {
                for __j in 0..__c {
                    $out.set(__i, __j, $f(__a.get(__i, __j)));
                }
            }
        }};
        ($out:expr, $f:expr, $a:expr, $b:expr) => {{
            #[cfg(feature = "gpu")]
            compile_error!("map_! is CPU-only. Use fuse! for GPU dispatch.");
            let __a = $a;
            let __b = $b;
            let (__r, __c) = $out.shape();
            assert_eq!((__r, __c), __a.shape(), "map_! shape mismatch");
            assert_eq!((__r, __c), __b.shape(), "map_! shape mismatch");
            for __i in 0..__r {
                for __j in 0..__c {
                    $out.set(__i, __j, $f(__a.get(__i, __j), __b.get(__i, __j)));
                }
            }
        }};
    }

    /// Variadic vertical concat: `vcat!(a, b, c)` stacks row-wise.
    ///
    /// # Examples
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::vcat;
    /// let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    /// let b: Tensor<f64> = mat![[3.0, 4.0]];
    /// let c: Tensor<f64> = mat![[5.0, 6.0]];
    /// let r = vcat!(a, b, c);
    /// assert_eq!(r.nrows(), 3);
    /// ```
    #[macro_export]
    macro_rules! vcat {
        ($a:expr, $b:expr) => {
            $crate::tensor::Tensor::vcat(&[&$a, &$b])
        };
        ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
            $crate::tensor::Tensor::vcat(&[&$a, &$b, $(&$rest),+])
        };
    }

    /// Variadic horizontal concat: `hcat!(a, b, c)` stacks column-wise.
    ///
    /// # Examples
    /// ```
    /// use nabla::prelude::*;
    /// use nabla::hcat;
    /// let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    /// let b: Tensor<f64> = mat![[3.0], [4.0]];
    /// let c: Tensor<f64> = mat![[5.0], [6.0]];
    /// let r = hcat!(a, b, c);
    /// assert_eq!(r.ncols(), 3);
    /// ```
    #[macro_export]
    macro_rules! hcat {
        ($a:expr, $b:expr) => {
            $crate::tensor::Tensor::hcat(&[&$a, &$b])
        };
        ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
            $crate::tensor::Tensor::hcat(&[&$a, &$b, $(&$rest),+])
        };
    }

    /// Julia-style pipe operator: `pipe!(val, f, g)` expands to `g(f(val))`.
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
    #[macro_export]
    macro_rules! splat {
        ($f:expr, ($($args:expr),+ $(,)?)) => {
            $f($($args),+)
        };
    }
}

/// Prelude for convenient imports.
pub mod prelude {
    pub use nabla_core::backend::{Backend, DefaultBackend};
    pub use nabla_core::error::{Error, Result};
    pub use nabla_core::layout::{LinearLayout, LinearLayout16, LinearLayout32, LinearLayout64};
    pub use nabla_core::scalar::Scalar;
    pub use nabla_core::tensor::{Array, NdTensor, StaticMatrix};
    #[cfg(feature = "cpu")]
    pub use nabla_core::tensor::{DynTensor, Matrix};
    pub use nabla_core::tensor::{MatmulCompat, Tensor};
    pub use nabla_macros::{
        axis, einsum, fuse, generated, mat, mega_fuse, nabla_grad, named, named_zeros,
    };

    #[cfg(feature = "cpu")]
    pub use crate::autograd::{GradPrep, grad, gradient, gradient_prep};
    pub use crate::autograd::{Tape, Variable};
    pub use crate::cas::{Expr, ExprKind};
    #[cfg(feature = "cpu")]
    pub use crate::linalg::expm;
    #[cfg(feature = "cpu")]
    pub use crate::linalg::{Diagonal, LinalgExt, Side, Symmetric, TriKind, Triangular};
    pub use crate::ode::{
        AdaptiveConfig, Bdf1Config, MetdConfig, OdeSolution, PararealConfig, StormerVerletConfig,
    };
    #[cfg(feature = "cpu")]
    pub use crate::ode::{
        DaeConfig, IfEulerScalarConfig, bdf1, dae_solve, if_euler_scalar, metd_solve,
        parareal_solve, stormer_verlet,
    };
    #[cfg(feature = "cpu")]
    pub use crate::sparse::*;
    #[cfg(feature = "cpu")]
    pub use crate::util::linspace;
    pub use crate::{arange, eye, fill, from_fn, nd_zeros, ones, rand, randn, zeros};
    #[cfg(feature = "cpu")]
    pub use half::{bf16, f16};
    #[cfg(feature = "cpu")]
    pub use nabla_core::backend::Cpu;
    #[cfg(feature = "cpu")]
    pub use nabla_core::scalar::{Dual, MultiDual, c32, c64};
    #[cfg(feature = "cpu")]
    pub use nabla_macros::stencil;

    #[cfg(feature = "gpu")]
    pub use nabla_core::backend::Gpu;

    #[cfg(feature = "cuda")]
    pub use nabla_core::{NablaCudaGraph, cuda_graph_capture, cuda_graph_capture_cached};
}
