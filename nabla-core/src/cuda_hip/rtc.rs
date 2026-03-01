use std::sync::Mutex;

use crate::scalar::Scalar;

use super::pool::lock_or_recover;

// ── RtcStorage ──────────────────────────────────────────────────────────────

/// Row-major GPU-backed matrix storage with lazy host cache.
///
/// Generic over `B` (the raw GPU buffer type) — instantiated as
/// `RtcStorage<CuBuffer, T>` for CUDA, `RtcStorage<HipBuffer, T>` for HIP.
pub struct RtcStorage<B, T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) buf: B,
    pub(crate) host_cache: Mutex<Option<Vec<T>>>,
}

impl<B, T: Scalar> RtcStorage<B, T> {
    pub(crate) fn new(nrows: usize, ncols: usize, buf: B) -> Self {
        Self {
            nrows,
            ncols,
            buf,
            host_cache: Mutex::new(None),
        }
    }

    /// Public constructor for external buffer wrapping (e.g. GpuTensor → nabla bridge).
    pub fn from_parts(nrows: usize, ncols: usize, buf: B) -> Self {
        Self {
            nrows,
            ncols,
            buf,
            host_cache: Mutex::new(None),
        }
    }

    /// Returns a reference to the raw GPU buffer.
    pub fn buffer(&self) -> &B {
        &self.buf
    }

    pub(crate) fn new_cached(nrows: usize, ncols: usize, buf: B, cache: Vec<T>) -> Self {
        Self {
            nrows,
            ncols,
            buf,
            host_cache: Mutex::new(Some(cache)),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.nrows * self.ncols
    }

    pub(crate) fn invalidate_cache(&mut self) {
        *lock_or_recover(&self.host_cache) = None;
    }

    pub(crate) fn cached_get(&self, idx: usize) -> T
    where
        Self: EnsureCache,
    {
        with_cached_data(self, |data| data[idx])
    }
}

/// Backend-specific cache fill — implemented per backend since CUDA needs
/// stream synchronization while HIP uses direct memcpy.
pub(crate) trait EnsureCache {
    fn ensure_cache(&self);
}

// ── Reduction ops (host-side, shared) ───────────────────────────────────────

fn with_cached_data<B, T: Scalar, R>(
    a: &RtcStorage<B, T>,
    f: impl FnOnce(&[T]) -> R,
) -> R
where
    RtcStorage<B, T>: EnsureCache,
{
    a.ensure_cache();
    let guard = lock_or_recover(&a.host_cache);
    let data = match guard.as_ref() {
        Some(data) => data,
        None => panic!("cache populated"),
    };
    f(data)
}

// Shared helper: fold-first reduction on cached host data.
fn rtc_fold_first<B, T: Scalar>(a: &RtcStorage<B, T>, f: impl Fn(T, T) -> T) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| {
        let (first, rest) = match data.split_first() {
            Some((first, rest)) => (first, rest),
            None => panic!("reduction on empty matrix"),
        };
        rest.iter().fold(*first, |acc, &x| f(acc, x))
    })
}

// Shared helper: argext on cached host data.
fn rtc_argext<B, T: Scalar>(
    a: &RtcStorage<B, T>,
    is_better: impl Fn(T, T) -> bool,
) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| {
        let mut best = 0usize;
        for i in 1..data.len() {
            if is_better(data[i], data[best]) {
                best = i;
            }
        }
        (best / a.ncols, best % a.ncols)
    })
}

pub(crate) fn rtc_sum_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| data.iter().fold(T::zero(), |acc, &x| acc + x))
}

pub(crate) fn rtc_fold_first_prod<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    with_cached_data(a, |data| data.iter().fold(T::one(), |acc, &x| acc * x))
}

pub(crate) fn rtc_max_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_fold_first(a, |acc, x| acc.reduction_max(x))
}

pub(crate) fn rtc_min_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> T
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_fold_first(a, |acc, x| acc.reduction_min(x))
}

pub(crate) fn rtc_argmax_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_argext(a, |cur, best| cur.reduction_gt(best))
}

pub(crate) fn rtc_argmin_all<B, T: Scalar>(a: &RtcStorage<B, T>) -> (usize, usize)
where
    RtcStorage<B, T>: EnsureCache,
{
    rtc_argext(a, |cur, best| best.reduction_gt(cur))
}

// ── GPU Backend trait method generators ─────────────────────────────────────

/// Generate `fn name<T: Scalar>(a: &$Stor<T>) -> $Stor<T> { launch_unary(a, "name") }`
/// for each unary op name listed.
macro_rules! gpu_unary_ops {
    ($Stor:ident; $($name:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $name<T: Scalar>(a: &$Stor<T>) -> $Stor<T> { launch_unary(a, stringify!($name)) }
        )+
    };
}
pub(crate) use gpu_unary_ops;

/// Generate `fn name<T: Scalar>(a: &$Stor<T>, b: &$Stor<T>) -> $Stor<T> { launch_binary(a, b, "name") }`
/// for each binary op name listed.
macro_rules! gpu_binary_ops {
    ($Stor:ident; $($name:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $name<T: Scalar>(a: &$Stor<T>, b: &$Stor<T>) -> $Stor<T> { launch_binary(a, b, stringify!($name)) }
        )+
    };
}
pub(crate) use gpu_binary_ops;

/// Generate all trivially-delegating `Backend` trait methods shared by CUDA and HIP.
///
/// Methods already handled by `gpu_unary_ops!` / `gpu_binary_ops!` and
/// backend-specific methods (`zeros`, `from_vec`, `sync`, `matmul_into`,
/// `matmul_epilogue`, `bmm`, `addmm`, `baddbmm`, `fuse_*`) are NOT included.
macro_rules! rtc_backend_impl {
    (
        $Stor:ident;
        fill = $fill:ident,
        from_fn = $from_fn:ident,
        from_vec_async = $fva:ident,
        get = $get:ident,
        set = $set:ident,
        transpose = $transpose:ident,
        scale = $scale:ident,
        clone_storage = $clone:ident,
        powf = $powf:ident,
        sum_all = $sum_all:ident,
        max_all = $max_all:ident,
        min_all = $min_all:ident,
        argmax_all = $argmax:ident,
        argmin_all = $argmin:ident,
        softmax = $softmax:ident,
        layer_norm = $ln:ident,
        rms_norm = $rms:ident,
        batch_norm_train = $bn:ident,
        cross_entropy_fused = $ce:ident,
        sdpa = $sdpa:ident,
        axis_reduce = $ar:ident,
        embedding = $emb:ident,
        cumsum_cumprod = $csc:ident,
        prod_all = $pa:ident,
        max_pool2d = $mp2:ident,
        max_pool2d_with_idx = $mpi2:ident,
        avg_pool2d = $ap2:ident,
        adaptive_avg_pool2d = $aap2:ident,
        conv2d = $c2:ident,
        conv1d = $c1:ident,
        conv3d = $c3:ident,
        conv_transpose2d = $ct2:ident $(,)?
    ) => {
        #[inline]
        fn fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> $Stor<T> {
            $fill(nrows, ncols, val)
        }

        #[inline]
        fn identity<T: Scalar>(n: usize) -> $Stor<T> {
            $from_fn(n, n, |r, c| if r == c { T::one() } else { T::zero() })
        }

        #[inline]
        fn from_fn<T: Scalar>(
            nrows: usize,
            ncols: usize,
            f: impl FnMut(usize, usize) -> T,
        ) -> $Stor<T> {
            $from_fn(nrows, ncols, f)
        }

        #[inline]
        fn from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> $Stor<T> {
            $fva(nrows, ncols, data)
        }

        #[inline]
        fn nrows<T: Scalar>(s: &$Stor<T>) -> usize { s.nrows }

        #[inline]
        fn ncols<T: Scalar>(s: &$Stor<T>) -> usize { s.ncols }

        #[inline]
        fn get<T: Scalar>(s: &$Stor<T>, r: usize, c: usize) -> T {
            $get(s, r, c)
        }

        #[inline]
        fn set<T: Scalar>(s: &mut $Stor<T>, r: usize, c: usize, v: T) {
            $set(s, r, c, v)
        }

        #[inline]
        fn neg<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            launch_unary(a, "neg")
        }

        #[inline]
        fn transpose<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $transpose(a)
        }

        #[inline]
        fn scale<T: Scalar>(a: &$Stor<T>, s: T) -> $Stor<T> {
            $scale(a, s)
        }

        #[inline]
        fn clone_storage<T: Scalar>(s: &$Stor<T>) -> $Stor<T> {
            $clone(s)
        }

        #[inline]
        fn leaky_relu<T: Scalar>(a: &$Stor<T>, _negative_slope: T) -> $Stor<T> {
            launch_unary(a, "leaky_relu")
        }

        #[inline]
        fn elu<T: Scalar>(a: &$Stor<T>, _alpha: T) -> $Stor<T> {
            launch_unary(a, "elu")
        }

        #[inline]
        fn powf<T: Scalar>(a: &$Stor<T>, p: T) -> $Stor<T> {
            $powf(a, p)
        }

        #[inline]
        fn sum_all<T: Scalar>(a: &$Stor<T>) -> T {
            $sum_all(a)
        }

        #[inline]
        fn max_all<T: Scalar>(a: &$Stor<T>) -> T {
            $max_all(a)
        }

        #[inline]
        fn min_all<T: Scalar>(a: &$Stor<T>) -> T {
            $min_all(a)
        }

        #[inline]
        fn argmax_all<T: Scalar>(a: &$Stor<T>) -> (usize, usize) {
            $argmax(a)
        }

        #[inline]
        fn argmin_all<T: Scalar>(a: &$Stor<T>) -> (usize, usize) {
            $argmin(a)
        }

        fn softmax<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $softmax(a)
        }

        fn layer_norm<T: Scalar>(
            a: &$Stor<T>,
            gamma: &$Stor<T>,
            beta: &$Stor<T>,
            eps: T,
        ) -> $Stor<T> {
            $ln(a, gamma, beta, eps)
        }

        fn rms_norm<T: Scalar>(a: &$Stor<T>, gamma: &$Stor<T>, eps: T) -> $Stor<T> {
            $rms(a, gamma, eps)
        }

        #[allow(clippy::too_many_arguments)]
        fn batch_norm_train<T: Scalar>(
            a: &$Stor<T>,
            gamma: &$Stor<T>,
            beta: &$Stor<T>,
            running_mean: &mut $Stor<T>,
            running_var: &mut $Stor<T>,
            eps: T,
            momentum: T,
            training: bool,
        ) -> $Stor<T> {
            $bn(a, gamma, beta, running_mean, running_var, eps, momentum, training)
        }

        fn cross_entropy_fused<T: Scalar>(
            input: &$Stor<T>,
            target: &$Stor<T>,
            _n: usize,
            _c: usize,
        ) -> $Stor<T> {
            $ce(input, target)
        }

        #[allow(clippy::too_many_arguments)]
        fn sdpa<T: Scalar>(
            q: &$Stor<T>,
            k: &$Stor<T>,
            v: &$Stor<T>,
            _mask: Option<&$Stor<T>>,
            seq_q: usize,
            seq_k: usize,
            head_dim: usize,
            batch_heads: usize,
        ) -> $Stor<T> {
            $sdpa(q, k, v, seq_q, seq_k, head_dim, batch_heads)
        }

        fn sum_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $ar(a, "sum_axis1")
        }

        fn max_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $ar(a, "max_axis1")
        }

        fn embedding<T: Scalar>(indices: &$Stor<T>, weight: &$Stor<T>) -> $Stor<T> {
            $emb(indices, weight)
        }

        fn cumsum_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $csc(a, "cumsum_axis1")
        }

        fn cumprod_axis1<T: Scalar>(a: &$Stor<T>) -> $Stor<T> {
            $csc(a, "cumprod_axis1")
        }

        #[inline]
        fn prod_all<T: Scalar>(a: &$Stor<T>) -> T {
            $pa(a)
        }

        #[allow(clippy::too_many_arguments)]
        fn max_pool2d<T: Scalar>(
            a: &$Stor<T>,
            h: usize, w: usize,
            kh: usize, kw: usize,
            sh: usize, sw: usize,
            ph: usize, pw: usize,
        ) -> $Stor<T> {
            $mp2(a, h, w, kh, kw, sh, sw, ph, pw)
        }

        #[allow(clippy::too_many_arguments)]
        fn max_pool2d_with_indices<T: Scalar>(
            a: &$Stor<T>,
            h: usize, w: usize,
            kh: usize, kw: usize,
            sh: usize, sw: usize,
            ph: usize, pw: usize,
        ) -> ($Stor<T>, $Stor<T>) {
            $mpi2(a, h, w, kh, kw, sh, sw, ph, pw)
        }

        #[allow(clippy::too_many_arguments)]
        fn avg_pool2d<T: Scalar>(
            a: &$Stor<T>,
            h: usize, w: usize,
            kh: usize, kw: usize,
            sh: usize, sw: usize,
            ph: usize, pw: usize,
        ) -> $Stor<T> {
            $ap2(a, h, w, kh, kw, sh, sw, ph, pw)
        }

        fn adaptive_avg_pool2d<T: Scalar>(
            a: &$Stor<T>,
            in_h: usize, in_w: usize,
            out_h: usize, out_w: usize,
        ) -> $Stor<T> {
            $aap2(a, in_h, in_w, out_h, out_w)
        }

        #[allow(clippy::too_many_arguments)]
        fn conv2d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n: usize, c_in: usize, h: usize, w: usize,
            c_out: usize, kh: usize, kw: usize,
            stride: (usize, usize),
            padding: (usize, usize),
            dilation: (usize, usize),
            groups: usize,
        ) -> $Stor<T> {
            $c2(input, weight, n, c_in, h, w, c_out, kh, kw, stride, padding, dilation, groups)
        }

        #[allow(clippy::too_many_arguments)]
        fn conv1d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n_batch: usize, c_in: usize, length: usize,
            c_out: usize, kernel_size: usize,
            stride: usize, padding: usize,
            dilation: usize, groups: usize,
        ) -> $Stor<T> {
            $c1(
                input, weight, n_batch, c_in, length, c_out, kernel_size,
                stride, padding, dilation, groups,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn conv3d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n_batch: usize, c_in: usize,
            d: usize, h: usize, w: usize,
            c_out: usize, kd: usize, kh: usize, kw: usize,
            stride: (usize, usize, usize),
            padding: (usize, usize, usize),
            dilation: (usize, usize, usize),
            groups: usize,
        ) -> $Stor<T> {
            $c3(
                input, weight, n_batch, c_in, d, h, w, c_out, kd, kh, kw,
                stride, padding, dilation, groups,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn conv_transpose2d<T: Scalar>(
            input: &$Stor<T>,
            weight: &$Stor<T>,
            n_batch: usize, c_in: usize, h: usize, w: usize,
            c_out: usize, kh: usize, kw: usize,
            stride: (usize, usize),
            padding: (usize, usize),
            output_padding: (usize, usize),
        ) -> $Stor<T> {
            $ct2(
                input, weight, n_batch, c_in, h, w, c_out, kh, kw,
                stride, padding, output_padding,
            )
        }
    };
}
pub(crate) use rtc_backend_impl;
