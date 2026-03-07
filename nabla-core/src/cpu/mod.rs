use crate::backend::BackendCore;
use crate::scalar::Scalar;
use rayon::prelude::*;

const TILE: usize = 64;

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
    #[allow(dead_code)]
    pub(crate) fn get_ref(&self, row: usize, col: usize) -> &T {
        &self.data[self.idx(row, col)]
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = self.idx(row, col);
        &mut self.data[idx]
    }
}

macro_rules! cpu_unary_op {
    ($fn_name:ident, |$x:ident| $body:expr) => {
        #[inline]
        fn $fn_name<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
            a.map_elem(|$x| $body)
        }
    };
}

macro_rules! cpu_unary_ops {
    ($($fn_name:ident => $method:ident),* $(,)?) => {
        $(cpu_unary_op!($fn_name, |x| x.$method());)*
    };
}

macro_rules! cpu_binary_op {
    ($fn_name:ident, |$x:ident, $y:ident| $body:expr) => {
        #[inline]
        fn $fn_name<T: Scalar>(a: &CpuStorage<T>, b: &CpuStorage<T>) -> CpuStorage<T> {
            a.zip_map(b, |$x, $y| $body)
        }
    };
}

pub struct Cpu;

#[inline]
fn cpu_fold_first<T: Scalar>(a: &CpuStorage<T>, f: impl Fn(T, T) -> T) -> T {
    assert!(!a.data.is_empty(), "reduction on empty matrix");
    let init = a.data[0];
    a.data.iter().skip(1).fold(init, |acc, &x| f(acc, x))
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

impl crate::backend::private::Sealed for Cpu {}

impl crate::backend::BackendCore for Cpu {
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
    fn axpy_inplace<T: Scalar>(y: &mut CpuStorage<T>, alpha: T, x: &CpuStorage<T>) {
        y.data
            .par_iter_mut()
            .zip(x.data.par_iter())
            .for_each(|(yi, &xi)| {
                *yi = *yi + alpha * xi;
            });
    }

    #[inline]
    fn clone_storage<T: Scalar>(storage: &CpuStorage<T>) -> CpuStorage<T> {
        CpuStorage {
            data: storage.data.clone(),
            nrows: storage.nrows,
            ncols: storage.ncols,
        }
    }
}

impl crate::backend::BackendMath for Cpu {
    cpu_unary_ops!(
        exp   => math_exp,
        ln    => math_ln,
        log1p => math_log1p,
        sin   => math_sin,
        cos   => math_cos,
        tan   => math_tan,
        tanh  => math_tanh,
        sqrt  => math_sqrt,
        abs   => math_abs,
        recip => math_recip,
        erf   => math_erf,
        ceil  => math_ceil,
        floor => math_floor,
        round => math_round,
        asin  => math_asin,
        acos  => math_acos,
        atan  => math_atan,
        sinh  => math_sinh,
        cosh  => math_cosh,
        asinh => math_asinh,
        acosh => math_acosh,
        atanh => math_atanh,
        log2  => math_log2,
        log10 => math_log10,
    );

    #[inline]
    fn powf<T: Scalar>(a: &CpuStorage<T>, p: T) -> CpuStorage<T> {
        a.map_elem(|x| x.math_powf(p))
    }

    cpu_binary_op!(atan2, |x, y| x.math_atan2(y));

    cpu_binary_op!(emul, |x, y| x.math_mul(y));

    cpu_binary_op!(ediv, |x, y| x.math_div(y));
}

impl crate::backend::BackendReduce for Cpu {
    #[inline]
    fn sum_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        a.data
            .par_iter()
            .fold(|| T::zero(), |acc, &x| acc.reduction_add(x))
            .reduce(|| T::zero(), crate::scalar::ReductionOps::reduction_add)
    }

    #[inline]
    fn max_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        cpu_fold_first(a, crate::scalar::ReductionOps::reduction_max)
    }

    #[inline]
    fn min_all<T: Scalar>(a: &CpuStorage<T>) -> T {
        cpu_fold_first(a, crate::scalar::ReductionOps::reduction_min)
    }

    #[inline]
    fn argmax_all<T: Scalar>(a: &CpuStorage<T>) -> (usize, usize) {
        cpu_argext(a, crate::scalar::ReductionOps::reduction_gt)
    }

    #[inline]
    fn argmin_all<T: Scalar>(a: &CpuStorage<T>) -> (usize, usize) {
        cpu_argext(a, |cur, best| best.reduction_gt(cur))
    }

    fn sum_axis1<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        let nrows = a.nrows;
        let ncols = a.ncols;
        let mut data = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let base = r * ncols;
            let mut acc = T::zero();
            for j in 0..ncols {
                acc = acc + a.data[base + j];
            }
            data.push(acc);
        }
        CpuStorage {
            data,
            nrows,
            ncols: 1,
        }
    }

    fn max_axis1<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        let nrows = a.nrows;
        let ncols = a.ncols;
        let mut data = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let base = r * ncols;
            let mut acc = a.data[base];
            for j in 1..ncols {
                let v = a.data[base + j];
                if v.to_f64() > acc.to_f64() {
                    acc = v;
                }
            }
            data.push(acc);
        }
        CpuStorage {
            data,
            nrows,
            ncols: 1,
        }
    }
}

impl crate::backend::BackendShape for Cpu {
    fn reshape_metadata<T: Scalar>(a: &mut CpuStorage<T>, new_rows: usize, new_cols: usize) {
        assert_eq!(
            a.nrows * a.ncols,
            new_rows * new_cols,
            "reshape_metadata: size mismatch"
        );
        a.nrows = new_rows;
        a.ncols = new_cols;
    }

    fn reshape_copy<T: Scalar>(
        a: &CpuStorage<T>,
        out_rows: usize,
        out_cols: usize,
    ) -> CpuStorage<T> {
        let total = out_rows * out_cols;
        assert_eq!(
            total,
            a.nrows * a.ncols,
            "reshape: {}x{} cannot reshape to {}x{}",
            a.nrows,
            a.ncols,
            out_rows,
            out_cols
        );
        let mut data = Vec::with_capacity(total);
        for r in 0..out_rows {
            for c in 0..out_cols {
                let flat = r * out_cols + c;
                let src_r = flat / a.ncols;
                let src_c = flat % a.ncols;
                data.push(a.get_unchecked(src_r, src_c));
            }
        }
        CpuStorage {
            data,
            nrows: out_rows,
            ncols: out_cols,
        }
    }

    fn submatrix<T: Scalar>(
        a: &CpuStorage<T>,
        row_start: usize,
        col_start: usize,
        out_rows: usize,
        out_cols: usize,
    ) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(out_rows * out_cols);
        for r in 0..out_rows {
            for c in 0..out_cols {
                data.push(a.get_unchecked(row_start + r, col_start + c));
            }
        }
        CpuStorage {
            data,
            nrows: out_rows,
            ncols: out_cols,
        }
    }

    fn slice_set<T: Scalar>(
        dst: &mut CpuStorage<T>,
        row_start: usize,
        col_start: usize,
        src: &CpuStorage<T>,
    ) {
        for r in 0..src.nrows {
            for c in 0..src.ncols {
                dst.set_unchecked(row_start + r, col_start + c, src.get_unchecked(r, c));
            }
        }
    }

    fn repeat<T: Scalar>(a: &CpuStorage<T>, row_reps: usize, col_reps: usize) -> CpuStorage<T> {
        let out_rows = a.nrows * row_reps;
        let out_cols = a.ncols * col_reps;
        let mut data = Vec::with_capacity(out_rows * out_cols);
        for r in 0..out_rows {
            for c in 0..out_cols {
                let src_r = r % a.nrows;
                let src_c = c % a.ncols;
                data.push(a.get_unchecked(src_r, src_c));
            }
        }
        CpuStorage {
            data,
            nrows: out_rows,
            ncols: out_cols,
        }
    }

    fn pad<T: Scalar>(
        a: &CpuStorage<T>,
        left: usize,
        right: usize,
        top: usize,
        bottom: usize,
        value: T,
    ) -> CpuStorage<T> {
        let out_rows = a.nrows + top + bottom;
        let out_cols = a.ncols + left + right;
        let mut data = Vec::with_capacity(out_rows * out_cols);
        for r in 0..out_rows {
            for c in 0..out_cols {
                let v = if r >= top && r < top + a.nrows && c >= left && c < left + a.ncols {
                    a.get_unchecked(r - top, c - left)
                } else {
                    value
                };
                data.push(v);
            }
        }
        CpuStorage {
            data,
            nrows: out_rows,
            ncols: out_cols,
        }
    }

    fn triu<T: Scalar>(a: &CpuStorage<T>, diagonal: isize) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(a.nrows * a.ncols);
        for r in 0..a.nrows {
            for c in 0..a.ncols {
                let keep = (c as isize) >= (r as isize) + diagonal;
                data.push(if keep {
                    a.get_unchecked(r, c)
                } else {
                    T::zero()
                });
            }
        }
        CpuStorage {
            data,
            nrows: a.nrows,
            ncols: a.ncols,
        }
    }

    fn tril<T: Scalar>(a: &CpuStorage<T>, diagonal: isize) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(a.nrows * a.ncols);
        for r in 0..a.nrows {
            for c in 0..a.ncols {
                let keep = (c as isize) <= (r as isize) + diagonal;
                data.push(if keep {
                    a.get_unchecked(r, c)
                } else {
                    T::zero()
                });
            }
        }
        CpuStorage {
            data,
            nrows: a.nrows,
            ncols: a.ncols,
        }
    }

    fn roll<T: Scalar>(a: &CpuStorage<T>, shift: isize, axis: usize) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(a.nrows * a.ncols);
        match axis {
            0 => {
                for r in 0..a.nrows {
                    let src_r = ((r as isize - shift).rem_euclid(a.nrows as isize)) as usize;
                    for c in 0..a.ncols {
                        data.push(a.get_unchecked(src_r, c));
                    }
                }
            }
            1 => {
                for r in 0..a.nrows {
                    for c in 0..a.ncols {
                        let src_c = ((c as isize - shift).rem_euclid(a.ncols as isize)) as usize;
                        data.push(a.get_unchecked(r, src_c));
                    }
                }
            }
            _ => panic!("nabla: roll axis must be 0 or 1, got {axis}"),
        }
        CpuStorage {
            data,
            nrows: a.nrows,
            ncols: a.ncols,
        }
    }

    fn flip<T: Scalar>(a: &CpuStorage<T>, axis: usize) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(a.nrows * a.ncols);
        match axis {
            0 => {
                for r in 0..a.nrows {
                    let src_r = a.nrows - 1 - r;
                    for c in 0..a.ncols {
                        data.push(a.get_unchecked(src_r, c));
                    }
                }
            }
            1 => {
                for r in 0..a.nrows {
                    for c in 0..a.ncols {
                        let src_c = a.ncols - 1 - c;
                        data.push(a.get_unchecked(r, src_c));
                    }
                }
            }
            _ => panic!("nabla: flip axis must be 0 or 1, got {axis}"),
        }
        CpuStorage {
            data,
            nrows: a.nrows,
            ncols: a.ncols,
        }
    }

    fn from_diag<T: Scalar>(v: &CpuStorage<T>) -> CpuStorage<T> {
        let n = v.nrows.max(v.ncols);
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            let val = if v.nrows >= v.ncols {
                v.get_unchecked(i, 0)
            } else {
                v.get_unchecked(0, i)
            };
            data[i * n + i] = val;
        }
        CpuStorage {
            data,
            nrows: n,
            ncols: n,
        }
    }

    fn gather_rows<T: Scalar>(a: &CpuStorage<T>, indices: &[usize]) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(indices.len() * a.ncols);
        for &r in indices {
            for c in 0..a.ncols {
                data.push(a.get_unchecked(r, c));
            }
        }
        CpuStorage {
            data,
            nrows: indices.len(),
            ncols: a.ncols,
        }
    }

    fn gather<T: Scalar>(a: &CpuStorage<T>, axis: usize, index: &CpuStorage<T>) -> CpuStorage<T> {
        let (m, n) = (index.nrows, index.ncols);
        let mut data = Vec::with_capacity(m * n);
        match axis {
            0 => {
                for r in 0..m {
                    for c in 0..n {
                        let idx = index.get_unchecked(r, c).to_f64() as usize;
                        data.push(a.get_unchecked(idx, c));
                    }
                }
            }
            1 => {
                for r in 0..m {
                    for c in 0..n {
                        let idx = index.get_unchecked(r, c).to_f64() as usize;
                        data.push(a.get_unchecked(r, idx));
                    }
                }
            }
            _ => panic!("nabla: gather axis must be 0 or 1, got {axis}"),
        }
        CpuStorage {
            data,
            nrows: m,
            ncols: n,
        }
    }

    fn scatter<T: Scalar>(
        a: &CpuStorage<T>,
        axis: usize,
        index: &CpuStorage<T>,
        src: &CpuStorage<T>,
    ) -> CpuStorage<T> {
        let mut out = CpuStorage {
            data: a.data.clone(),
            nrows: a.nrows,
            ncols: a.ncols,
        };
        for r in 0..index.nrows {
            for c in 0..index.ncols {
                let idx = index.get_unchecked(r, c).to_f64() as usize;
                let val = src.get_unchecked(r, c);
                match axis {
                    0 => out.set_unchecked(idx, c, val),
                    1 => out.set_unchecked(r, idx, val),
                    _ => panic!("nabla: scatter axis must be 0 or 1, got {axis}"),
                }
            }
        }
        out
    }

    fn index_select<T: Scalar>(
        a: &CpuStorage<T>,
        axis: usize,
        index: &CpuStorage<T>,
    ) -> CpuStorage<T> {
        let k = index.nrows * index.ncols;
        let get_idx = |i: usize| -> usize {
            if index.nrows == 1 {
                index.get_unchecked(0, i).to_f64() as usize
            } else {
                index.get_unchecked(i, 0).to_f64() as usize
            }
        };
        match axis {
            0 => {
                let mut data = Vec::with_capacity(k * a.ncols);
                for r in 0..k {
                    let src_r = get_idx(r);
                    for c in 0..a.ncols {
                        data.push(a.get_unchecked(src_r, c));
                    }
                }
                CpuStorage {
                    data,
                    nrows: k,
                    ncols: a.ncols,
                }
            }
            1 => {
                let mut data = Vec::with_capacity(a.nrows * k);
                for r in 0..a.nrows {
                    for c in 0..k {
                        let src_c = get_idx(c);
                        data.push(a.get_unchecked(r, src_c));
                    }
                }
                CpuStorage {
                    data,
                    nrows: a.nrows,
                    ncols: k,
                }
            }
            _ => panic!("nabla: index_select axis must be 0 or 1, got {axis}"),
        }
    }

    fn sort_rows<T: Scalar>(a: &CpuStorage<T>, descending: bool) -> (CpuStorage<T>, CpuStorage<T>) {
        let rows = a.nrows;
        let cols = a.ncols;
        let mut vals = vec![T::zero(); rows * cols];
        let mut idxs = vec![T::zero(); rows * cols];
        for r in 0..rows {
            let mut pairs: Vec<(T, usize)> =
                (0..cols).map(|c| (a.get_unchecked(r, c), c)).collect();
            pairs.sort_by(|a, b| {
                let va = a.0.to_f64();
                let vb = b.0.to_f64();
                if descending {
                    vb.total_cmp(&va)
                } else {
                    va.total_cmp(&vb)
                }
            });
            for c in 0..cols {
                vals[r * cols + c] = pairs[c].0;
                idxs[r * cols + c] = T::from_f64(pairs[c].1 as f64);
            }
        }
        (
            CpuStorage {
                data: vals,
                nrows: rows,
                ncols: cols,
            },
            CpuStorage {
                data: idxs,
                nrows: rows,
                ncols: cols,
            },
        )
    }

    fn topk_rows<T: Scalar>(a: &CpuStorage<T>, k: usize) -> (CpuStorage<T>, CpuStorage<T>) {
        let rows = a.nrows;
        let cols = a.ncols;
        let mut vals = vec![T::zero(); rows * k];
        let mut idxs = vec![T::zero(); rows * k];
        for r in 0..rows {
            let mut pairs: Vec<(T, usize)> =
                (0..cols).map(|c| (a.get_unchecked(r, c), c)).collect();
            pairs.sort_by(|a, b| b.0.to_f64().total_cmp(&a.0.to_f64()));
            for c in 0..k {
                vals[r * k + c] = pairs[c].0;
                idxs[r * k + c] = T::from_f64(pairs[c].1 as f64);
            }
        }
        (
            CpuStorage {
                data: vals,
                nrows: rows,
                ncols: k,
            },
            CpuStorage {
                data: idxs,
                nrows: rows,
                ncols: k,
            },
        )
    }

    fn meshgrid<T: Scalar>(x: &CpuStorage<T>, y: &CpuStorage<T>) -> (CpuStorage<T>, CpuStorage<T>) {
        let nx = x.nrows * x.ncols;
        let ny = y.nrows * y.ncols;
        let mut gx = Vec::with_capacity(nx * ny);
        let mut gy = Vec::with_capacity(nx * ny);
        for r in 0..ny {
            for c in 0..nx {
                gx.push(x.data[c]);
                gy.push(y.data[r]);
            }
        }
        (
            CpuStorage {
                data: gx,
                nrows: ny,
                ncols: nx,
            },
            CpuStorage {
                data: gy,
                nrows: ny,
                ncols: nx,
            },
        )
    }

    fn scatter_add_dim0<T: Scalar>(
        dst: &mut CpuStorage<T>,
        indices: &[usize],
        src: &CpuStorage<T>,
    ) {
        for (r, &target_r) in indices.iter().enumerate() {
            for c in 0..src.ncols {
                let idx = dst.idx(target_r, c);
                dst.data[idx] = dst.data[idx] + src.get_unchecked(r, c);
            }
        }
    }

    fn kron<T: Scalar>(
        a: &CpuStorage<T>,
        b: &CpuStorage<T>,
        m: usize,
        n: usize,
        p: usize,
        q: usize,
    ) -> CpuStorage<T> {
        let mut data = Vec::with_capacity(m * p * n * q);
        for i in 0..(m * p) {
            for j in 0..(n * q) {
                let av = a.get_unchecked(i / p, j / q);
                let bv = b.get_unchecked(i % p, j % q);
                data.push(av * bv);
            }
        }
        CpuStorage {
            data,
            nrows: m * p,
            ncols: n * q,
        }
    }
}

impl crate::backend::BackendBlas for Cpu {
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
}

impl crate::backend::BackendNN for Cpu {
    fn sigmoid<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        a.map_elem(|x| T::one() / (T::one() + (T::zero() - x).math_exp()))
    }

    fn silu<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        a.map_elem(|x| {
            let s = T::one() / (T::one() + (T::zero() - x).math_exp());
            x * s
        })
    }

    fn mish<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        a.map_elem(|x| {
            let sp = (T::one() + x.math_exp()).math_ln();
            x * sp.math_tanh()
        })
    }

    fn leaky_relu<T: Scalar>(a: &CpuStorage<T>, negative_slope: T) -> CpuStorage<T> {
        a.map_elem(|x| {
            let ax = x.math_abs();
            let half = T::from_f64(0.5);
            let pos = (x + ax) * half;
            let neg = (x - ax) * half;
            pos + neg * negative_slope
        })
    }

    fn elu<T: Scalar>(a: &CpuStorage<T>, alpha: T) -> CpuStorage<T> {
        a.map_elem(|x| {
            let ax = x.math_abs();
            let eps = T::from_f64(1e-30);
            let two = T::from_f64(2.0);
            let denom = if ax.to_f64() > eps.to_f64() { ax } else { eps };
            let sp = (x + ax) / (two * denom);
            let sp = if sp.to_f64() > 1.0 { T::one() } else { sp };
            sp * x + (T::one() - sp) * alpha * (x.math_exp() - T::one())
        })
    }

    fn hardswish<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        let three = T::from_f64(3.0);
        let six = T::from_f64(6.0);
        a.map_elem(|x| {
            let v = x + three;
            let v = if v.to_f64() < 0.0 {
                T::zero()
            } else if v.to_f64() > 6.0 {
                six
            } else {
                v
            };
            x * v / six
        })
    }

    fn softmax<T: Scalar>(a: &CpuStorage<T>) -> CpuStorage<T> {
        let nrows = a.nrows;
        let ncols = a.ncols;
        let mut data = Vec::with_capacity(nrows * ncols);
        for r in 0..nrows {
            let mut max = a.data[r * ncols];
            for j in 1..ncols {
                let v = a.data[r * ncols + j];
                if v.to_f64() > max.to_f64() {
                    max = v;
                }
            }
            let mut sum = T::zero();
            for j in 0..ncols {
                sum = sum + (a.data[r * ncols + j] - max).math_exp();
            }
            let inv = T::one() / sum;
            for j in 0..ncols {
                data.push((a.data[r * ncols + j] - max).math_exp() * inv);
            }
        }
        CpuStorage { data, nrows, ncols }
    }

    fn layer_norm<T: Scalar>(
        a: &CpuStorage<T>,
        gamma: &CpuStorage<T>,
        beta: &CpuStorage<T>,
        eps: T,
    ) -> CpuStorage<T> {
        let nrows = a.nrows;
        let ncols = a.ncols;
        let ncols_f = T::from_f64(ncols as f64);
        let mut data = Vec::with_capacity(nrows * ncols);
        for r in 0..nrows {
            let base = r * ncols;
            let mut sum = T::zero();
            for j in 0..ncols {
                sum = sum + a.data[base + j];
            }
            let mean = sum / ncols_f;
            let mut var_sum = T::zero();
            for j in 0..ncols {
                let d = a.data[base + j] - mean;
                var_sum = var_sum + d * d;
            }
            let inv_std = T::one() / (var_sum / ncols_f + eps).math_sqrt();
            for j in 0..ncols {
                data.push((a.data[base + j] - mean) * inv_std * gamma.data[j] + beta.data[j]);
            }
        }
        CpuStorage { data, nrows, ncols }
    }

    fn rms_norm<T: Scalar>(a: &CpuStorage<T>, gamma: &CpuStorage<T>, eps: T) -> CpuStorage<T> {
        let nrows = a.nrows;
        let ncols = a.ncols;
        let ncols_f = T::from_f64(ncols as f64);
        let mut data = Vec::with_capacity(nrows * ncols);
        for r in 0..nrows {
            let base = r * ncols;
            let mut sq_sum = T::zero();
            for j in 0..ncols {
                let v = a.data[base + j];
                sq_sum = sq_sum + v * v;
            }
            let inv_rms = T::one() / (sq_sum / ncols_f + eps).math_sqrt();
            for j in 0..ncols {
                data.push(a.data[base + j] * inv_rms * gamma.data[j]);
            }
        }
        CpuStorage { data, nrows, ncols }
    }

    fn embedding<T: Scalar>(indices: &CpuStorage<T>, weight: &CpuStorage<T>) -> CpuStorage<T> {
        let n_tokens = indices.nrows * indices.ncols;
        let embed_dim = weight.ncols;
        let mut data = Vec::with_capacity(n_tokens * embed_dim);
        for i in 0..n_tokens {
            let idx = indices.data[i].to_f64() as usize;
            let base = idx * embed_dim;
            for j in 0..embed_dim {
                data.push(weight.data[base + j]);
            }
        }
        CpuStorage {
            data,
            nrows: n_tokens,
            ncols: embed_dim,
        }
    }
}

impl crate::backend::BackendFusion for Cpu {
    fn fuse_launch<T: Scalar>(
        _inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        _gpu_expr: &str,
        _kernel_hash: &str,
        _n_inputs: usize,
        _reg_estimate: usize,
    ) -> CpuStorage<T> {
        Self::from_fn(nrows, ncols, cpu_fn)
    }

    fn mega_fuse_launch<'a, T: Scalar>(
        _ops: &[(Vec<*const u8>, String, usize, bool)],
        nrows: usize,
        ncols: usize,
        cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T + 'a>>,
        _kernel_hash: &str,
    ) -> Vec<CpuStorage<T>> {
        // CPU backend runs each closure independently.  For `uses_prev` ops the
        // macro inlines the previous expression directly into the closure body,
        // so no special handling is needed here.
        cpu_fns
            .into_iter()
            .map(|mut f| Self::from_fn(nrows, ncols, &mut f))
            .collect()
    }
}
