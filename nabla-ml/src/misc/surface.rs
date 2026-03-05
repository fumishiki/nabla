pub mod constructors {
    //! Free constructor and math functions for tensors.
    //!
    //! This module provides convenient free functions for creating tensors and
    //! performing basic mathematical operations. All functions are re-exported
    //! from the crate root via `pub use constructors::*;`.

    #[cfg(feature = "cpu")]
    use std::cell::Cell;

    use crate::{scalar, tensor};

    #[cfg(feature = "cpu")]
    thread_local! {
        static GLOBAL_SEED: Cell<Option<u64>> = const { Cell::new(None) };
    }

    /// Set a global RNG seed for reproducible tensor construction.
    ///
    /// Affects all subsequent calls to [`rand`], [`randn`], and any constructor
    /// that uses randomness. The seed auto-increments on each use to produce
    /// different tensors from successive calls.
    ///
    /// # Example
    /// ```ignore
    /// nabla::set_seed(42);
    /// let a = nabla::rand::<f64>(3, 3); // deterministic
    /// ```
    #[cfg(feature = "cpu")]
    pub fn set_seed(seed: u64) {
        GLOBAL_SEED.with(|s| s.set(Some(seed)));
    }

    /// Clear the global RNG seed, reverting to time-based seeding.
    #[cfg(feature = "cpu")]
    pub fn clear_seed() {
        GLOBAL_SEED.with(|s| s.set(None));
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    #[cfg(feature = "cpu")]
    pub(crate) fn default_seed() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|dur| {
                let nanos = dur.as_nanos();
                (nanos as u64) ^ ((nanos >> 64) as u64)
            })
            .unwrap_or(0xA11C_E5EE_D5EE_DBAD_u64)
    }

    #[inline]
    #[cfg(feature = "cpu")]
    pub(crate) fn seed_or_default() -> u64 {
        GLOBAL_SEED.with(|s| {
            if let Some(seed) = s.get() {
                // Auto-increment so successive calls get different tensors
                s.set(Some(seed.wrapping_add(1)));
                seed
            } else {
                let seed = default_seed();
                if seed == 0 {
                    0x1234_5678_9ABC_DEF0_u64
                } else {
                    seed
                }
            }
        })
    }

    #[inline]
    #[cfg(feature = "cpu")]
    fn xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
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

    /// Allocate an identity matrix of size `n x n` (CPU-only).
    #[must_use]
    #[inline]
    #[cfg(feature = "cpu")]
    pub fn eye<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
        tensor::Tensor::identity(n)
    }

    /// Allocate a tensor whose element `(r, c)` is `f(r, c)` (CPU-only).
    #[must_use]
    #[inline]
    #[cfg(feature = "cpu")]
    pub fn from_fn<T: scalar::Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> tensor::Tensor<T> {
        tensor::Tensor::from_fn(nrows, ncols, f)
    }

    /// Allocate a zero-filled N-D tensor (CPU-only).
    #[must_use]
    #[inline]
    #[cfg(feature = "cpu")]
    pub fn nd_zeros<T: scalar::Scalar>(shape: &[usize]) -> tensor::NdTensor<T> {
        tensor::NdTensor::zeros(shape)
    }

    /// Uniform random tensor in `[0, 1)` (CPU-only).
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn rand<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
        let mut s = seed_or_default();
        tensor::Tensor::from_fn(nrows, ncols, |_, _| {
            let val = xorshift64(&mut s);
            T::from_f64((val as f64) / (u64::MAX as f64))
        })
    }

    /// Standard normal random tensor (`mean=0`, `std=1`) (CPU-only).
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn randn<T: scalar::Scalar>(nrows: usize, ncols: usize) -> tensor::Tensor<T> {
        let mut s = seed_or_default();
        let mut xorshift = || {
            let val = xorshift64(&mut s);
            (val as f64) / (u64::MAX as f64)
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

    /// Column vector of zeros with shape `(n, 1)`.
    #[must_use]
    #[inline]
    pub fn zeros_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
        zeros(n, 1)
    }

    /// Column vector of ones with shape `(n, 1)`.
    #[must_use]
    #[inline]
    pub fn ones_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
        ones(n, 1)
    }

    /// Uniform random column vector in `[0, 1)` with shape `(n, 1)` (CPU-only).
    #[must_use]
    #[inline]
    #[cfg(feature = "cpu")]
    pub fn rand_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
        rand(n, 1)
    }

    /// Standard normal random column vector with shape `(n, 1)` (CPU-only).
    #[must_use]
    #[inline]
    #[cfg(feature = "cpu")]
    pub fn randn_vec<T: scalar::Scalar>(n: usize) -> tensor::Tensor<T> {
        randn(n, 1)
    }

    /// 1-D half-open range tensor: `[start, start+step, ..., < stop]` (CPU-only).
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn arange<T: scalar::Scalar>(start: T, stop: T, step: T) -> tensor::Tensor<T> {
        let step_f = step.to_f64();
        assert!(
            step_f.is_finite() && step_f != 0.0,
            "nabla: arange step must be non-zero finite, got {step_f}"
        );
        let stop_f = stop.to_f64();
        let is_forward = step_f > 0.0;
        let mut cur = start.to_f64();
        let mut n = 0usize;

        while (is_forward && cur < stop_f) || (!is_forward && cur > stop_f) {
            n += 1;
            cur += step_f;
        }

        tensor::Tensor::arange(start, step, n)
    }

    /// 1-D tensor of `n` evenly spaced values from `start` to `stop` (inclusive) (CPU-only).
    #[must_use]
    #[inline]
    #[cfg(feature = "cpu")]
    pub fn linspace<T: scalar::Scalar>(start: T, stop: T, n: usize) -> tensor::Tensor<T> {
        tensor::Tensor::linspace(start, stop, n)
    }

    /// 1-D tensor of `n` points logarithmically spaced from `10^start` to `10^stop` (CPU-only).
    ///
    /// Equivalent to NumPy's `np.logspace(start, stop, n)`.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn logspace<T: scalar::Scalar>(start: f64, stop: f64, n: usize) -> tensor::Tensor<T> {
        match n {
            0 => tensor::Tensor::zeros(1, 0),
            1 => tensor::Tensor::from_fn(1, 1, |_, _| T::from_f64(10.0_f64.powf(start))),
            _ => {
                #[allow(clippy::cast_precision_loss)]
                let step = (stop - start) / (n as f64 - 1.0);
                tensor::Tensor::from_fn(1, n, |_, j| {
                    #[allow(clippy::cast_precision_loss)]
                    let exponent = start + j as f64 * step;
                    T::from_f64(10.0_f64.powf(exponent))
                })
            }
        }
    }

    /// 1-D tensor of `n` points geometrically spaced from `start` to `stop` (CPU-only).
    ///
    /// Both `start` and `stop` must be non-zero and have the same sign.
    /// Equivalent to NumPy's `np.geomspace(start, stop, n)`.
    ///
    /// # Panics
    ///
    /// Panics if `start` or `stop` is zero, or if they have different signs.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn geomspace<T: scalar::Scalar>(start: f64, stop: f64, n: usize) -> tensor::Tensor<T> {
        assert!(
            start != 0.0 && stop != 0.0,
            "nabla: geomspace requires non-zero start and stop"
        );
        assert!(
            start.signum() == stop.signum(),
            "nabla: geomspace requires start and stop to have the same sign"
        );
        match n {
            0 => tensor::Tensor::zeros(1, 0),
            1 => tensor::Tensor::from_fn(1, 1, |_, _| T::from_f64(start)),
            _ => {
                let log_start = start.abs().ln();
                let log_stop = stop.abs().ln();
                #[allow(clippy::cast_precision_loss)]
                let step = (log_stop - log_start) / (n as f64 - 1.0);
                let sign = start.signum();
                tensor::Tensor::from_fn(1, n, |_, j| {
                    #[allow(clippy::cast_precision_loss)]
                    let val = sign * (log_start + j as f64 * step).exp();
                    T::from_f64(val)
                })
            }
        }
    }

    /// Inner product (dot product), returning a scalar.
    #[must_use]
    #[inline]
    pub fn dot<T: scalar::Scalar>(a: &tensor::Tensor<T>, b: &tensor::Tensor<T>) -> T {
        a.dot(b)
    }

    /// Kronecker product: `A ⊗ B`.
    #[must_use]
    #[inline]
    pub fn kron<T: scalar::Scalar>(
        a: &tensor::Tensor<T>,
        b: &tensor::Tensor<T>,
    ) -> tensor::Tensor<T> {
        a.kron(b)
    }

    /// Create diagonal matrix from a vector tensor (1×n or n×1).
    #[must_use]
    #[inline]
    pub fn diagm<T: scalar::Scalar>(v: &tensor::Tensor<T>) -> tensor::Tensor<T> {
        tensor::Tensor::from_diag(v)
    }

    /// 3D cross product of two 3-element vectors (CPU-only).
    /// Both must be (3,1) or (1,3) tensors.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn cross<T: scalar::Scalar>(
        a: &tensor::Tensor<T>,
        b: &tensor::Tensor<T>,
    ) -> tensor::Tensor<T> {
        let na = a.nrows() * a.ncols();
        let nb = b.nrows() * b.ncols();
        assert!(
            na == 3 && nb == 3,
            "nabla: cross requires 3-element vectors, got {na} and {nb}"
        );
        let a_is_row = a.ncols() == 3;
        let b_is_row = b.ncols() == 3;
        let comp_a = |row: usize, col: usize| {
            if a_is_row {
                a.get(row, col)
            } else {
                a.get(col, row)
            }
        };
        let comp_b = |row: usize, col: usize| {
            if b_is_row {
                b.get(row, col)
            } else {
                b.get(col, row)
            }
        };

        let a0 = comp_a(0, 0);
        let a1 = comp_a(1, 0);
        let a2 = comp_a(2, 0);
        let b0 = comp_b(0, 0);
        let b1 = comp_b(1, 0);
        let b2 = comp_b(2, 0);
        // Cross product: (a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0)
        tensor::Tensor::from_fn(3, 1, |i, _| match i {
            0 => a1 * b2 - a2 * b1,
            1 => a2 * b0 - a0 * b2,
            _ => a0 * b1 - a1 * b0,
        })
    }

    /// Frobenius/L2 norm of a tensor.
    #[must_use]
    #[inline]
    pub fn norm<T: scalar::Scalar>(a: &tensor::Tensor<T>) -> T {
        a.norm()
    }

    /// Lp norm of a tensor with specified order.
    #[must_use]
    #[inline]
    pub fn norm_ord<T: scalar::Scalar>(a: &tensor::Tensor<T>, p: f64) -> T {
        a.norm_ord(p)
    }

    /// Trace of a matrix (sum of diagonal elements).
    #[must_use]
    #[inline]
    pub fn tr<T: scalar::Scalar>(a: &tensor::Tensor<T>) -> T {
        a.tr()
    }

    /// Check approximate equality of two tensors within absolute tolerance `atol` (CPU-only).
    ///
    /// Returns `false` if shapes differ or any element pair exceeds `atol`.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn approx_eq<T: scalar::Scalar>(
        a: &tensor::Tensor<T>,
        b: &tensor::Tensor<T>,
        atol: f64,
    ) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        let (m, n) = a.shape();
        for r in 0..m {
            for c in 0..n {
                if (a.get(r, c).to_f64() - b.get(r, c).to_f64()).abs() > atol {
                    return false;
                }
            }
        }
        true
    }
}

pub mod notation {
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::map;
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    /// let doubled: Tensor<f64> = map!(|x| x * 2.0, &a);
    /// assert_eq!(doubled.get(0, 0), 2.0);
    /// # }
    /// # }
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::par_map;
    /// let a: Tensor<f64> = Tensor::from_fn(4, 4, |i, j| (i * 4 + j) as f64);
    /// let b: Tensor<f64> = par_map!(|x| x * 2.0, &a);
    /// assert!((b.get(0, 1) - 2.0).abs() < 1e-12);
    /// # }
    /// # }
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::map_;
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    /// let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    /// let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    /// map_!(out, |x, y| x * y, &a, &b);
    /// assert_eq!(out.get(0, 0), 2.0);
    /// # }
    /// # }
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::vcat;
    /// let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    /// let b: Tensor<f64> = mat![[3.0, 4.0]];
    /// let c: Tensor<f64> = mat![[5.0, 6.0]];
    /// let r = vcat!(a, b, c);
    /// assert_eq!(r.nrows(), 3);
    /// # }
    /// # }
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::hcat;
    /// let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    /// let b: Tensor<f64> = mat![[3.0], [4.0]];
    /// let c: Tensor<f64> = mat![[5.0], [6.0]];
    /// let r = hcat!(a, b, c);
    /// assert_eq!(r.ncols(), 3);
    /// # }
    /// # }
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::tensor_range;
    /// let t: Tensor<f64> = tensor_range!(0.0, 0.5, 1.0);
    /// assert_eq!(t.shape(), (1, 3));
    /// # }
    /// # }
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
    /// ```rust
    /// # fn main() {
    /// # #[cfg(feature = "cpu")]
    /// # {
    /// use nabla::prelude::*;
    /// use nabla::approx;
    /// let a: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 1.0);
    /// let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 1.0 + 1e-11);
    /// approx!(&a, &b);
    /// approx!(&a, &b, 1e-10);
    /// # }
    /// # }
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
}
