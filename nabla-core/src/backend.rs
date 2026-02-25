// backend.rs — Sealed Backend trait + Cpu implementation backed by CpuStorage (row-major Vec<T>).
//
// Element storage is a plain Vec<T> with
// row-major layout: data[r * ncols + c].
//
// matmul_into uses a tiled i-k-j loop (TILE=64) for cache-friendly access.

use crate::scalar::Scalar;

// Tiled matmul tile size — chosen to fit in L1 cache for f64.
const TILE: usize = 64;

/// Row-major owned storage for a 2-D CPU matrix.
pub struct CpuStorage<T: Scalar> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

// SAFETY: T: Send + Sync (required by Scalar supertrait).
unsafe impl<T: Scalar> Send for CpuStorage<T> {}
unsafe impl<T: Scalar> Sync for CpuStorage<T> {}

impl<T: Scalar> CpuStorage<T> {
    #[inline]
    fn new_zeroed(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.ncols + col
    }

    #[inline]
    fn get_unchecked(&self, row: usize, col: usize) -> T {
        self.data[self.idx(row, col)]
    }

    #[inline]
    fn set_unchecked(&mut self, row: usize, col: usize, val: T) {
        let idx = self.idx(row, col);
        self.data[idx] = val;
    }

    #[inline]
    fn map_elem(&self, f: impl Fn(T) -> T) -> Self {
        Self {
            data: self.data.iter().map(|&x| f(x)).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    #[inline]
    fn zip_map(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&x, &y)| f(x, y))
                .collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn data_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub(crate) fn get_ref(&self, row: usize, col: usize) -> &T {
        &self.data[self.idx(row, col)]
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = self.idx(row, col);
        &mut self.data[idx]
    }
}

// Internal macro: generate a Backend unary method that maps over CpuStorage elements.
macro_rules! cpu_unary_op {
    ($fn_name:ident, |$x:ident| $body:expr) => {
        #[inline]
        fn $fn_name<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
            a.map_elem(|$x| $body)
        }
    };
}

// Internal macro: generate a Backend binary method that zips two CpuStorage.
macro_rules! cpu_binary_op {
    ($fn_name:ident, |$x:ident, $y:ident| $body:expr) => {
        #[inline]
        fn $fn_name<T: Scalar>(a: &CpuStorage<T>, b: &CpuStorage<T>) -> CpuStorage<T> {
            a.zip_map(b, |$x, $y| $body)
        }
    };
}

pub(crate) mod private {
    pub trait Sealed {}
}

/// Compute backend abstraction (sealed — not implementable outside this crate).
pub trait Backend: private::Sealed + Send + Sync + 'static {
    /// Owned storage for a 2-D matrix of element type `T`.
    type Storage<T: Scalar>: Send + Sync;

    /// Allocate a zero-filled matrix.
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> Self::Storage<T>;

    /// Allocate a matrix filled with a constant scalar value.
    fn fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> Self::Storage<T> {
        Self::from_fn(nrows, ncols, |_, _| val)
    }

    /// Allocate an n x n identity matrix.
    #[must_use]
    fn identity<T: Scalar>(n: usize) -> Self::Storage<T> {
        Self::from_fn(n, n, |r, c| if r == c { T::one() } else { T::zero() })
    }

    /// Allocate a matrix and fill it by calling `f(row, col)`.
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> Self::Storage<T>;

    /// Build storage from a pre-allocated row-major `Vec<T>` (zero-copy when possible).
    #[must_use]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> Self::Storage<T> {
        let mut v = data.into_iter();
        Self::from_fn(nrows, ncols, |_, _| {
            v.next().expect("from_vec: data length must equal nrows * ncols")
        })
    }

    /// Row count of `storage`.
    fn nrows<T: Scalar>(storage: &Self::Storage<T>) -> usize;

    /// Column count of `storage`.
    fn ncols<T: Scalar>(storage: &Self::Storage<T>) -> usize;

    /// Read element at `(row, col)`.
    fn get<T: Scalar>(storage: &Self::Storage<T>, row: usize, col: usize) -> T;

    /// Write element at `(row, col)`.
    fn set<T: Scalar>(storage: &mut Self::Storage<T>, row: usize, col: usize, val: T);

    /// Block until all pending operations on this backend's device/stream have completed.
    /// On GPU backends this flushes the command stream; on CPU this is a no-op.
    fn sync<T: Scalar>(_storage: &Self::Storage<T>) {}

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

    /// Clone storage.
    fn clone_storage<T: Scalar>(storage: &Self::Storage<T>) -> Self::Storage<T>;

    // --- Elementwise math operations ---

    /// Element-wise `e^x`.
    fn exp<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise natural logarithm `ln(x)`.
    fn ln<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `ln(1 + x)`.
    fn log1p<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `sin(x)`.
    fn sin<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `cos(x)`.
    fn cos<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `tanh(x)`.
    fn tanh<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `sqrt(x)`.
    fn sqrt<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise absolute value.
    fn abs<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise reciprocal `1/x`.
    fn recip<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise error function.
    fn erf<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `ceil(x)`.
    fn ceil<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `floor(x)`.
    fn floor<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `round(x)`.
    fn round<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise `x^p` for scalar exponent `p`.
    fn powf<T: Scalar>(a: &Self::Storage<T>, p: T) -> Self::Storage<T>;

    /// Element-wise multiplication `a[i,j] * b[i,j]`.
    fn emul<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise division `a[i,j] / b[i,j]`.
    fn ediv<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;

    // --- Reduction operations (whole-matrix -> scalar) ---

    /// Sum all elements of the matrix.
    fn sum_all<T: Scalar>(a: &Self::Storage<T>) -> T;

    /// Element with the maximum value (or maximum magnitude for complex types).
    fn max_all<T: Scalar>(a: &Self::Storage<T>) -> T;

    /// Element with the minimum value (or minimum magnitude for complex types).
    fn min_all<T: Scalar>(a: &Self::Storage<T>) -> T;

    /// `(row, col)` of the element with the maximum value (or magnitude for complex types).
    fn argmax_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);

    /// `(row, col)` of the element with the minimum value (or magnitude for complex types).
    fn argmin_all<T: Scalar>(a: &Self::Storage<T>) -> (usize, usize);

    /// Launch a fused element-wise kernel.
    ///
    /// GPU backends JIT-compile `gpu_expr` (a CUDA/HIP C expression over
    /// `in0[i], in1[i], …`) into a single kernel, caching by `kernel_hash`.
    /// CPU backends ignore the GPU arguments and use `from_fn` with `cpu_fn`.
    fn fuse_launch<T: Scalar>(
        _inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        _gpu_expr: &str,
        _kernel_hash: &str,
        _n_inputs: usize,
    ) -> Self::Storage<T> {
        Self::from_fn(nrows, ncols, cpu_fn)
    }
}

/// CPU backend — row-major `Vec<T>` storage, no external BLAS dependencies.
pub struct Cpu;

impl private::Sealed for Cpu {}

impl Backend for Cpu {
    type Storage<T: Scalar> = CpuStorage<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> CpuStorage<T> {
        CpuStorage::new_zeroed(nrows, ncols)
    }

    #[inline]
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        mut f: impl FnMut(usize, usize) -> T,
    ) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(nrows * ncols);
        for r in 0..nrows {
            for c in 0..ncols {
                data.push(f(r, c));
            }
        }
        CpuStorage { data, nrows, ncols }
    }

    #[inline]
    fn from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> CpuStorage<T> {
        CpuStorage { data, nrows, ncols }
    }

    #[inline]
    fn nrows<T: Scalar>(storage: &CpuStorage<T>) -> usize {
        storage.nrows
    }

    #[inline]
    fn ncols<T: Scalar>(storage: &CpuStorage<T>) -> usize {
        storage.ncols
    }

    #[inline]
    fn get<T: Scalar>(storage: &CpuStorage<T>, row: usize, col: usize) -> T {
        storage.get_unchecked(row, col)
    }

    #[inline]
    fn set<T: Scalar>(storage: &mut CpuStorage<T>, row: usize, col: usize, val: T) {
        storage.set_unchecked(row, col, val);
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_into<T: Scalar>(out: &mut CpuStorage<T>, a: &CpuStorage<T>, b: &CpuStorage<T>) {
        let (m, k, n) = (a.nrows, a.ncols, b.ncols);
        // Zero the output buffer first.
        for x in &mut out.data {
            *x = T::zero();
        }
        // Tiled i-k-j loop: inner loop over j is contiguous in B (row-major), friendly for cache.
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + TILE).min(m);
            let mut kk = 0;
            while kk < k {
                let k_end = (kk + TILE).min(k);
                let mut jj = 0;
                while jj < n {
                    let j_end = (jj + TILE).min(n);
                    for i in ii..i_end {
                        let a_row = &a.data[i * k..(i + 1) * k];
                        let out_row = &mut out.data[i * n..(i + 1) * n];
                        #[allow(clippy::needless_range_loop)]
                        for p in kk..k_end {
                            let a_ip = a_row[p];
                            let b_row = &b.data[p * n..(p + 1) * n];
                            for j in jj..j_end {
                                // out[i,j] += a[i,p] * b[p,j]
                                out_row[j] = out_row[j] + a_ip * b_row[j];
                            }
                        }
                    }
                    jj += TILE;
                }
                kk += TILE;
            }
            ii += TILE;
        }
    }

    cpu_binary_op!(add, |x, y| x + y);

    cpu_binary_op!(sub, |x, y| x - y);

    cpu_unary_op!(neg, |x| -x);

    #[inline]
    fn transpose<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        const BLK: usize = 64;
        let (rows, cols) = (a.nrows, a.ncols);
        let mut out = CpuStorage::new_zeroed(cols, rows);
        let mut i0 = 0;
        while i0 < rows {
            let imax = (i0 + BLK).min(rows);
            let mut j0 = 0;
            while j0 < cols {
                let jmax = (j0 + BLK).min(cols);
                for i in i0..imax {
                    for j in j0..jmax {
                        out.data[j * rows + i] = a.data[i * cols + j];
                    }
                }
                j0 += BLK;
            }
            i0 += BLK;
        }
        out
    }

    #[inline]
    fn scale<T: Scalar>(a: &CpuStorage<T>, s: T) -> CpuStorage<T> {
        a.map_elem(|x| x * s)
    }

    #[inline]
    fn clone_storage<T: Scalar>(storage: &CpuStorage<T>) -> CpuStorage<T> {
        CpuStorage {
            data: storage.data.clone(),
            nrows: storage.nrows,
            ncols: storage.ncols,
        }
    }

    cpu_unary_op!(exp, |x| x.math_exp());

    cpu_unary_op!(ln, |x| x.math_ln());

    cpu_unary_op!(log1p, |x| x.math_log1p());

    cpu_unary_op!(sin, |x| x.math_sin());

    cpu_unary_op!(cos, |x| x.math_cos());

    cpu_unary_op!(tanh, |x| x.math_tanh());

    cpu_unary_op!(sqrt, |x| x.math_sqrt());

    cpu_unary_op!(abs, |x| x.math_abs());

    cpu_unary_op!(recip, |x| x.math_recip());

    cpu_unary_op!(erf, |x| x.math_erf());

    cpu_unary_op!(ceil, |x| x.math_ceil());

    cpu_unary_op!(floor, |x| x.math_floor());

    cpu_unary_op!(round, |x| x.math_round());

    #[inline]
    fn powf<T: Scalar>(a: &CpuStorage<T>, p: T) -> CpuStorage<T> {
        a.map_elem(|x| x.math_powf(p))
    }

    cpu_binary_op!(emul, |x, y| x.math_mul(y));

    cpu_binary_op!(ediv, |x, y| x.math_div(y));

    #[inline]
    fn sum_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        a.data
            .iter()
            .fold(T::zero(), |acc, &x| acc.reduction_add(x))
    }

    #[inline]
    fn max_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        assert!(
            a.nrows > 0 && a.ncols > 0,
            "max_all: matrix must be non-empty"
        );
        let mut it = a.data.iter();
        let init = *it.next().expect("non-empty checked above");
        it.fold(init, |acc, &x| acc.reduction_max(x))
    }

    #[inline]
    fn min_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        assert!(
            a.nrows > 0 && a.ncols > 0,
            "min_all: matrix must be non-empty"
        );
        let mut it = a.data.iter();
        let init = *it.next().expect("non-empty checked above");
        it.fold(init, |acc, &x| acc.reduction_min(x))
    }

    #[inline]
    fn argmax_all<T: Scalar>(a: &CpuStorage<T>) -> (usize, usize) {
        assert!(
            a.nrows > 0 && a.ncols > 0,
            "argmax_all: matrix must be non-empty"
        );
        let (r, c) = (a.nrows, a.ncols);
        let mut best = (0usize, 0usize);
        for i in 0..r {
            for j in 0..c {
                if i == 0 && j == 0 {
                    continue;
                }
                if a.get_unchecked(i, j)
                    .reduction_gt(a.get_unchecked(best.0, best.1))
                {
                    best = (i, j);
                }
            }
        }
        best
    }

    #[inline]
    fn argmin_all<T: Scalar>(a: &CpuStorage<T>) -> (usize, usize) {
        assert!(
            a.nrows > 0 && a.ncols > 0,
            "argmin_all: matrix must be non-empty"
        );
        let (r, c) = (a.nrows, a.ncols);
        let mut best = (0usize, 0usize);
        for i in 0..r {
            for j in 0..c {
                if i == 0 && j == 0 {
                    continue;
                }
                if a.get_unchecked(best.0, best.1)
                    .reduction_gt(a.get_unchecked(i, j))
                {
                    best = (i, j);
                }
            }
        }
        best
    }
}

#[cfg(feature = "gpu")]
/// GPU backend — wgpu + WGSL compute shaders (f32 only).
pub struct Gpu;

#[cfg(feature = "gpu")]
impl private::Sealed for Gpu {}

#[cfg(feature = "cuda")]
/// CUDA backend — cudarc 0.19 + NVRTC JIT (f32/f64).
pub struct Cuda;

#[cfg(feature = "hip")]
/// HIP backend — hip-runtime-sys + hiprtc JIT (f32/f64).
pub struct Hip;

// DefaultBackend: cuda > hip > gpu > cpu
#[cfg(feature = "cuda")]
/// Default backend selected at compile time.
pub type DefaultBackend = Cuda;

#[cfg(all(feature = "hip", not(feature = "cuda")))]
/// Default backend selected at compile time.
pub type DefaultBackend = Hip;

#[cfg(all(feature = "gpu", not(feature = "cuda"), not(feature = "hip")))]
/// Default backend selected at compile time.
pub type DefaultBackend = Gpu;

#[cfg(all(feature = "cpu", not(feature = "gpu"), not(feature = "cuda"), not(feature = "hip")))]
/// Default backend selected at compile time.
pub type DefaultBackend = Cpu;
