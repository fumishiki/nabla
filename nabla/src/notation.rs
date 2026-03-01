//! Utility macros mirroring Julia math notation.
//!
//! All macros in this module use `#[macro_export]` and are therefore available
//! at the crate root: `nabla::between!`, `nabla::map!`, etc.  The containing
//! module exists only for source-code organisation.

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
        if step == 0.0 {
            Vec::new()
        } else {
            let n = ((stop - start) / step).floor() as usize;
            let tol = step.abs() * f64::EPSILON * 2.0;
            let mut v: Vec<f64> = Vec::new();
            for i in 0..=n {
                let val = start + i as f64 * step;
                if (step > 0.0 && val <= stop + tol) || (step < 0.0 && val >= stop - tol) {
                    v.push(val);
                }
            }
            v
        }
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! __nabla_shape2 {
    ($a:expr, $b:expr, $msg:expr) => {{
        let __a = $a;
        let __b = $b;
        assert_eq!(__a.shape(), __b.shape(), $msg);
        (__a, __b)
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! __nabla_cat_refs {
    ($a:expr, $b:expr) => { &[$a, $b] };
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => { &[$a, $b, $(&$rest),+] };
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
        let (__a, __b) = $crate::__nabla_shape2!($a, $b, "map! shape mismatch");
        let (__r, __c) = __a.shape();
        $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
            $f(__a.get(__i, __j), __b.get(__i, __j))
        })
    }};
    ($f:expr, $a:expr, $b:expr, $c:expr) => {{
        #[cfg(feature = "gpu")]
        compile_error!("map! is CPU-only. Use fuse! for GPU dispatch.");
        let (__a, __b) = $crate::__nabla_shape2!($a, $b, "map! shape mismatch");
        let (__a, __c_t) = $crate::__nabla_shape2!(__a, $c, "map! shape mismatch");
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
        let (__a, __b) = $crate::__nabla_shape2!($a, $b, "par_map! shape mismatch");
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
        assert_eq!($out.shape(), __a.shape(), "map_! shape mismatch");
        let (__r, __c) = $out.shape();
        for __i in 0..__r {
            for __j in 0..__c {
                $out.set(__i, __j, $f(__a.get(__i, __j)));
            }
        }
    }};
    ($out:expr, $f:expr, $a:expr, $b:expr) => {{
        #[cfg(feature = "gpu")]
        compile_error!("map_! is CPU-only. Use fuse! for GPU dispatch.");
        let (__a, __b) = $crate::__nabla_shape2!($a, $b, "map_! shape mismatch");
        assert_eq!($out.shape(), __a.shape(), "map_! shape mismatch");
        let (__r, __c) = $out.shape();
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
        $crate::tensor::Tensor::vcat($crate::__nabla_cat_refs!(&$a, &$b))
    };
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
        $crate::tensor::Tensor::vcat($crate::__nabla_cat_refs!(&$a, &$b, $($rest),+))
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
        $crate::tensor::Tensor::hcat($crate::__nabla_cat_refs!(&$a, &$b))
    };
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
        $crate::tensor::Tensor::hcat($crate::__nabla_cat_refs!(&$a, &$b, $($rest),+))
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

/// In-place fused element-wise operation: `fuse_!(out = expr ; tensors...)`.
///
/// Like `fuse!` but writes directly into an existing tensor instead of allocating.
/// Example: `fuse_!(a = sin(b) + c * d);`
#[macro_export]
macro_rules! fuse_ {
    ($out:ident = $($rest:tt)*) => {{
        let __result = $crate::fuse!($($rest)*);
        let (__r, __c) = $out.shape();
        let (__rr, __cc) = __result.shape();
        assert_eq!((__r, __c), (__rr, __cc), "nabla: fuse_! shape mismatch");
        for __i in 0..__r {
            for __j in 0..__c {
                $out.set(__i, __j, __result.get(__i, __j));
            }
        }
    }};
}

/// Unified element-wise transform that works on both CPU and GPU backends.
///
/// Uses `Tensor::from_fn` for element-wise evaluation.
/// Single-tensor form: `tmap!(|x| x * 2.0, &a)`.
/// Two-tensor form: `tmap!(|x, y| x + y, &a, &b)`.
#[macro_export]
macro_rules! tmap {
    ($f:expr, $a:expr) => {{
        let __a = $a;
        let (__r, __c) = __a.shape();
        $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| $f(__a.get(__i, __j)))
    }};
    ($f:expr, $a:expr, $b:expr) => {{
        let (__a, __b) = $crate::__nabla_shape2!($a, $b, "tmap! shape mismatch");
        let (__r, __c) = __a.shape();
        $crate::tensor::Tensor::from_fn(__r, __c, |__i, __j| {
            $f(__a.get(__i, __j), __b.get(__i, __j))
        })
    }};
}

/// Julia-style range: `range!(0.0, 0.25, 1.0)` → `Vec<f64>`.
///
/// Alias for [`frange!`].
///
/// # Examples
/// ```
/// use nabla::range;
/// let v = range!(0.0, 0.25, 1.0);
/// assert_eq!(v.len(), 5);
/// ```
#[macro_export]
macro_rules! range {
    ($start:expr, $step:expr, $stop:expr) => {
        $crate::frange!($start, $step, $stop)
    };
}

/// Float range as a row tensor: `tensor_range!(0.0, 0.25, 1.0)` → `Tensor<f64>` of shape `(1, n)`.
///
/// Like [`frange!`] but wraps the result into a 1-row tensor.
///
/// # Examples
/// ```
/// use nabla::prelude::*;
/// use nabla::tensor_range;
/// let t: Tensor<f64> = tensor_range!(0.0, 0.5, 1.0);
/// assert_eq!(t.shape(), (1, 3));
/// ```
#[macro_export]
macro_rules! tensor_range {
    ($start:expr, $step:expr, $stop:expr) => {{
        let __v = $crate::frange!($start, $step, $stop);
        let __n = __v.len();
        $crate::tensor::Tensor::<f64, $crate::backend::DefaultBackend>::from_fn(
            1,
            __n,
            |_, __c| __v[__c],
        )
    }};
}

/// Assert approximate equality of two tensors within absolute tolerance.
///
/// Panics with a descriptive message if elements differ by more than `atol`.
/// Default `atol` is `1e-10` when omitted.
///
/// # Examples
/// ```
/// use nabla::prelude::*;
/// use nabla::approx;
/// let a: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 1.0);
/// let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 1.0 + 1e-11);
/// approx!(&a, &b);
/// approx!(&a, &b, 1e-10);
/// ```
#[macro_export]
macro_rules! approx {
    ($a:expr, $b:expr, $atol:expr) => {
        assert!(
            $crate::approx_eq($a, $b, $atol),
            "nabla: approx! assertion failed (atol={})",
            $atol
        )
    };
    ($a:expr, $b:expr) => {
        $crate::approx!($a, $b, 1e-10)
    };
}
