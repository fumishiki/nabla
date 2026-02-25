// backend.rs — Sealed Backend trait + Cpu implementation backed by CpuStorage (row-major Vec<T>).
//
// Element storage is a plain Vec<T> with
// row-major layout: data[r * ncols + c].
//
// matmul_into uses a tiled i-k-j loop (TILE=64) for cache-friendly access.

use crate::scalar::Scalar;
use rayon::prelude::*;

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
    fn map_elem(&self, f: impl Fn(T) -> T + Send + Sync) -> Self {
        Self {
            data: self.data.par_iter().map(|&x| f(x)).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    #[inline]
    fn zip_map(&self, other: &Self, f: impl Fn(T, T) -> T + Send + Sync) -> Self {
        Self {
            data: self
                .data
                .par_iter()
                .zip(other.data.par_iter())
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

// Batch version: define multiple unary ops via Scalar::math_* methods.
macro_rules! cpu_unary_ops {
    ($($fn_name:ident => $method:ident),* $(,)?) => {
        $(cpu_unary_op!($fn_name, |x| x.$method());)*
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

    /// Non-blocking H2D upload: data transfer on a separate copy stream overlaps with compute.
    /// Default falls back to synchronous `from_vec`. GPU backends override for overlap.
    #[must_use]
    fn from_vec_async<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> Self::Storage<T> {
        Self::from_vec(nrows, ncols, data)
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

    // --- Activation ops (GPU-accelerable) ---

    /// SiLU activation: x * sigmoid(x)
    fn silu<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, ncols, |r, c| {
            let x = Self::get(a, r, c);
            let s = T::one() / (T::one() + (T::zero() - x).math_exp());
            x * s
        })
    }

    /// Mish activation: x * tanh(softplus(x))
    fn mish<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, ncols, |r, c| {
            let x = Self::get(a, r, c);
            let sp = (T::one() + x.math_exp()).math_ln();
            x * sp.math_tanh()
        })
    }

    /// Leaky ReLU: max(0, x) + negative_slope * min(0, x)
    fn leaky_relu<T: Scalar>(a: &Self::Storage<T>, negative_slope: T) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, ncols, |r, c| {
            let x = Self::get(a, r, c);
            let ax = x.math_abs();
            let pos = (x + ax) * T::from_f64(0.5);
            let neg = (x - ax) * T::from_f64(0.5);
            pos + neg * negative_slope
        })
    }

    /// ELU: x if x > 0, alpha*(exp(x)-1) otherwise
    fn elu<T: Scalar>(a: &Self::Storage<T>, alpha: T) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, ncols, |r, c| {
            let x = Self::get(a, r, c);
            let ax = x.math_abs();
            let eps = T::from_f64(1e-30);
            let denom = if ax.to_f64() > eps.to_f64() { ax } else { eps };
            let sp = (x + ax) / (T::from_f64(2.0) * denom);
            let sp = if sp.to_f64() > 1.0 { T::one() } else { sp };
            sp * x + (T::one() - sp) * alpha * (x.math_exp() - T::one())
        })
    }

    /// HardSwish: x * min(max(x+3, 0), 6) / 6
    fn hardswish<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        let three = T::from_f64(3.0);
        let six = T::from_f64(6.0);
        Self::from_fn(nrows, ncols, |r, c| {
            let x = Self::get(a, r, c);
            let v = x + three;
            let v = if v.to_f64() < 0.0 { T::zero() } else if v.to_f64() > 6.0 { six } else { v };
            x * v / six
        })
    }

    // --- Softmax (GPU-accelerable) ---

    /// Row-wise softmax. Input/output shape: (nrows, ncols).
    fn softmax<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, ncols, |r, c| {
            let mut max = Self::get(a, r, 0);
            for j in 1..ncols {
                let v = Self::get(a, r, j);
                if v.to_f64() > max.to_f64() { max = v; }
            }
            let mut sum = T::zero();
            for j in 0..ncols {
                sum = sum + (Self::get(a, r, j) - max).math_exp();
            }
            (Self::get(a, r, c) - max).math_exp() / sum
        })
    }

    // --- Layer norm / RMS norm (GPU-accelerable) ---

    /// Fused layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.
    /// Input/gamma/beta shape: (nrows, ncols). One normalization per row.
    fn layer_norm<T: Scalar>(
        a: &Self::Storage<T>,
        gamma: &Self::Storage<T>,
        beta: &Self::Storage<T>,
        eps: T,
    ) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        let ncols_f = T::from_f64(ncols as f64);
        Self::from_fn(nrows, ncols, |r, c| {
            let mut sum = T::zero();
            for j in 0..ncols { sum = sum + Self::get(a, r, j); }
            let mean = sum / ncols_f;
            let mut var_sum = T::zero();
            for j in 0..ncols {
                let d = Self::get(a, r, j) - mean;
                var_sum = var_sum + d * d;
            }
            let inv_std = T::one() / (var_sum / ncols_f + eps).math_sqrt();
            (Self::get(a, r, c) - mean) * inv_std * Self::get(gamma, 0, c) + Self::get(beta, 0, c)
        })
    }

    /// Fused RMS normalization: x / sqrt(mean(x^2) + eps) * gamma.
    fn rms_norm<T: Scalar>(
        a: &Self::Storage<T>,
        gamma: &Self::Storage<T>,
        eps: T,
    ) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        let ncols_f = T::from_f64(ncols as f64);
        Self::from_fn(nrows, ncols, |r, c| {
            let mut sq_sum = T::zero();
            for j in 0..ncols {
                let v = Self::get(a, r, j);
                sq_sum = sq_sum + v * v;
            }
            let inv_rms = T::one() / (sq_sum / ncols_f + eps).math_sqrt();
            Self::get(a, r, c) * inv_rms * Self::get(gamma, 0, c)
        })
    }

    // --- Axis reductions (GPU-accelerable) ---

    /// Sum along axis=1 (columns): (nrows, ncols) → (nrows, 1).
    fn sum_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, 1, |r, _c| {
            let mut acc = T::zero();
            for j in 0..ncols { acc = acc + Self::get(a, r, j); }
            acc
        })
    }

    /// Max along axis=1: (nrows, ncols) → (nrows, 1).
    fn max_axis1<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T> {
        let nrows = Self::nrows(a);
        let ncols = Self::ncols(a);
        Self::from_fn(nrows, 1, |r, _c| {
            let mut acc = Self::get(a, r, 0);
            for j in 1..ncols {
                let v = Self::get(a, r, j);
                if v.to_f64() > acc.to_f64() { acc = v; }
            }
            acc
        })
    }

    /// Embedding gather: indices (n_tokens, 1) float-encoded, weight (vocab, embed_dim).
    /// Output: (n_tokens, embed_dim).
    fn embedding<T: Scalar>(
        indices: &Self::Storage<T>,
        weight: &Self::Storage<T>,
    ) -> Self::Storage<T> {
        let n_tokens = Self::nrows(indices) * Self::ncols(indices);
        let embed_dim = Self::ncols(weight);
        Self::from_fn(n_tokens, embed_dim, |r, c| {
            let idx = Self::get(indices, r / Self::ncols(indices), r % Self::ncols(indices)).to_f64() as usize;
            Self::get(weight, idx, c)
        })
    }

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
        _reg_estimate: usize,
    ) -> Self::Storage<T> {
        Self::from_fn(nrows, ncols, cpu_fn)
    }

    /// Launch a mega-fused kernel: multiple element-wise operations in a
    /// single GPU kernel launch, eliminating inter-op launch overhead.
    ///
    /// `ops` is a slice of `(inputs, gpu_expr, n_inputs, cpu_fn)` tuples.
    /// All operations must share the same `(nrows, ncols)` dimensions.
    ///
    /// GPU backends emit a single mega-kernel; the CPU fallback runs each
    /// `cpu_fn` independently via `from_fn`.
    fn mega_fuse_launch<T: Scalar>(
        _ops: &[(Vec<*const u8>, String, usize)],
        nrows: usize,
        ncols: usize,
        cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T>>,
        _kernel_hash: &str,
    ) -> Vec<Self::Storage<T>> {
        cpu_fns.into_iter()
            .map(|mut f| Self::from_fn(nrows, ncols, |r, c| f(r, c)))
            .collect()
    }
}

/// CPU backend — row-major `Vec<T>` storage, no external BLAS dependencies.
pub struct Cpu;

// Shared helpers for CPU reduction ops.
#[inline]
fn cpu_fold_first<T: Scalar>(a: &CpuStorage<T>, f: impl Fn(T, T) -> T) -> T {
    assert!(!a.data.is_empty(), "reduction on empty matrix");
    let mut it = a.data.iter();
    let init = *it.next().unwrap();
    it.fold(init, |acc, &x| f(acc, x))
}

#[inline]
fn cpu_argext<T: Scalar>(a: &CpuStorage<T>, is_better: impl Fn(T, T) -> bool) -> (usize, usize) {
    assert!(!a.data.is_empty(), "argext on empty matrix");
    let mut best = 0usize;
    for i in 1..a.data.len() {
        if is_better(a.data[i], a.data[best]) {
            best = i;
        }
    }
    (best / a.ncols, best % a.ncols)
}

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
        // Zero + parallel tiled i-k-j loop.
        out.data.fill(T::zero());
        let a_data = &a.data;
        let b_data = &b.data;
        out.data
            .par_chunks_mut(TILE * n)
            .enumerate()
            .for_each(|(tile_idx, out_chunk)| {
                let ii = tile_idx * TILE;
                let i_end = (ii + TILE).min(m);
                let rows = i_end - ii;
                let chunk = &mut out_chunk[..rows * n];
                let mut kk = 0;
                while kk < k {
                    let k_end = (kk + TILE).min(k);
                    let mut jj = 0;
                    while jj < n {
                        let j_end = (jj + TILE).min(n);
                        for i in 0..rows {
                            let a_row = &a_data[(ii + i) * k..(ii + i + 1) * k];
                            let out_row = &mut chunk[i * n..(i + 1) * n];
                            #[allow(clippy::needless_range_loop)]
                            for p in kk..k_end {
                                let a_ip = a_row[p];
                                let b_row = &b_data[p * n..(p + 1) * n];
                                for j in jj..j_end {
                                    out_row[j] = out_row[j] + a_ip * b_row[j];
                                }
                            }
                        }
                        jj += TILE;
                    }
                    kk += TILE;
                }
            });
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

    cpu_unary_ops!(
        exp   => math_exp,
        ln    => math_ln,
        log1p => math_log1p,
        sin   => math_sin,
        cos   => math_cos,
        tanh  => math_tanh,
        sqrt  => math_sqrt,
        abs   => math_abs,
        recip => math_recip,
        erf   => math_erf,
        ceil  => math_ceil,
        floor => math_floor,
        round => math_round,
    );

    #[inline]
    fn powf<T: Scalar>(a: &CpuStorage<T>, p: T) -> CpuStorage<T> {
        a.map_elem(|x| x.math_powf(p))
    }

    cpu_binary_op!(emul, |x, y| x.math_mul(y));

    cpu_binary_op!(ediv, |x, y| x.math_div(y));

    #[inline]
    fn sum_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        a.data
            .par_iter()
            .fold(|| T::zero(), |acc, &x| acc.reduction_add(x))
            .reduce(|| T::zero(), |a, b| a.reduction_add(b))
    }

    #[inline]
    fn max_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        cpu_fold_first(a, |acc, x| acc.reduction_max(x))
    }

    #[inline]
    fn min_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        cpu_fold_first(a, |acc, x| acc.reduction_min(x))
    }

    #[inline]
    fn argmax_all<T: Scalar>(a: &CpuStorage<T>) -> (usize, usize) {
        cpu_argext(a, |cur, best| cur.reduction_gt(best))
    }

    #[inline]
    fn argmin_all<T: Scalar>(a: &CpuStorage<T>) -> (usize, usize) {
        cpu_argext(a, |cur, best| best.reduction_gt(cur))
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
