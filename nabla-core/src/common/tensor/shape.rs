#[cfg(feature = "cpu")]
use rayon::prelude::*;

use crate::backend::Backend;
#[cfg(feature = "cpu")]
use crate::backend::Cpu;
use crate::scalar::Scalar;

use super::Tensor;
#[cfg(feature = "cpu")]
use super::variants::NdTensor;

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Shape manipulation ----

    /// Zero-copy reshape: change metadata only, no allocation or kernel launch.
    #[inline]
    pub fn reshape_inplace(&mut self, m: usize, n: usize) {
        B::reshape_metadata(&mut self.storage, m, n);
    }

    /// Reshape to `(m, n)`, preserving row-major element order.
    #[must_use]
    pub fn reshape(&self, m: usize, n: usize) -> Self {
        let (rows, cols) = self.shape();
        assert_eq!(
            m * n,
            rows * cols,
            "reshape: {rows}x{cols} cannot reshape to {m}x{n}"
        );
        Self::from_storage(B::reshape_copy(&self.storage, m, n))
    }

    /// Flatten to `(1, nrows*ncols)`.
    #[must_use]
    pub fn flatten(&self) -> Self {
        let n = self.nrows() * self.ncols();
        self.reshape(1, n)
    }

    /// Unflatten a dimension: reshape `axis` dim by splitting it into `sizes`.
    #[must_use]
    pub fn unflatten(&self, axis: usize, sizes: (usize, usize)) -> Self {
        let (m, n) = self.shape();
        match axis {
            0 => {
                assert_eq!(
                    sizes.0 * sizes.1,
                    m,
                    "unflatten: {m} != {}*{}",
                    sizes.0,
                    sizes.1
                );
                self.reshape(sizes.0, sizes.1 * n)
            }
            1 => {
                assert_eq!(
                    sizes.0 * sizes.1,
                    n,
                    "unflatten: {n} != {}*{}",
                    sizes.0,
                    sizes.1
                );
                self.reshape(m * sizes.0, sizes.1)
            }
            _ => panic!("unflatten: axis {axis} out of bounds for 2-D tensor"),
        }
    }

    /// Return a contiguous copy of the tensor (forces row-major layout).
    #[must_use]
    pub fn contiguous(&self) -> Self {
        Self::from_storage(B::contiguous(&self.storage))
    }

    /// Detach from the computation graph: returns a clone with no gradient tracking.
    #[must_use]
    pub fn detach(&self) -> Self {
        self.clone()
    }

    /// Reshape alias (PyTorch-style naming). Same as `reshape()`.
    ///
    /// **Warning**: Unlike PyTorch's `view`, this performs a copy. The returned
    /// tensor does not share memory with `self`.
    #[must_use]
    pub fn view(&self, m: usize, n: usize) -> Self {
        self.reshape(m, n)
    }

    /// Remove a size-1 dimension. For 2-D tensors this validates the axis has size 1
    /// and returns a clone (still 2-D -- use `NdTensor` for true rank reduction).
    #[must_use]
    pub fn squeeze(&self, axis: usize) -> Self {
        let d = self.dim(axis);
        assert_eq!(d, 1, "nabla: squeeze({axis}) -- dim is {d}, not 1");
        self.clone()
    }

    /// Insert a size-1 dimension at `axis`, producing an `NdTensor`.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn unsqueeze(&self, axis: usize) -> NdTensor<T> {
        assert!(
            axis <= 2,
            "nabla: unsqueeze axis {axis} out of bounds (max 2 for 2-D tensor)"
        );
        let (r, c) = self.shape();
        let mut new_shape = vec![r, c];
        new_shape.insert(axis, 1);
        NdTensor::from_fn(&new_shape, |idx| {
            let mut orig = idx.to_vec();
            orig.remove(axis);
            self.get(orig[0], orig[1])
        })
    }

    // ---- Concat / stack / chunk / split ----

    fn concat_offsets(tensors: &[&Self], len: impl Fn(&Self) -> usize) -> (Vec<usize>, usize) {
        let offsets: Vec<usize> = tensors
            .iter()
            .scan(0usize, |acc, t| {
                let s = *acc;
                *acc += len(t);
                Some(s)
            })
            .collect();
        let total = offsets.last().copied().unwrap_or(0) + tensors.last().map_or(0, |t| len(t));
        (offsets, total)
    }

    /// Vertical concat: stack `tensors` row-by-row. All must have same ncols.
    #[must_use]
    pub fn vcat(tensors: &[&Self]) -> Self {
        assert!(!tensors.is_empty(), "nabla: vcat on empty slice");
        let ncols = tensors[0].ncols();
        for (i, t) in tensors.iter().enumerate() {
            assert_eq!(
                t.ncols(),
                ncols,
                "nabla: vcat tensor[{i}] has {c} cols, expected {ncols}",
                c = t.ncols()
            );
        }
        let (offsets, total) = Self::concat_offsets(tensors, super::Tensor::nrows);
        let mut out = Self::zeros(total, ncols);
        for (ti, t) in tensors.iter().enumerate() {
            let rs = offsets[ti];
            out.slice_set(rs..(rs + t.nrows()), 0..ncols, t);
        }
        out
    }

    /// Horizontal concat: stack `tensors` column-by-column. All must have same nrows.
    #[must_use]
    pub fn hcat(tensors: &[&Self]) -> Self {
        assert!(!tensors.is_empty(), "nabla: hcat on empty slice");
        let nrows = tensors[0].nrows();
        for (i, t) in tensors.iter().enumerate() {
            assert_eq!(
                t.nrows(),
                nrows,
                "nabla: hcat tensor[{i}] has {r} rows, expected {nrows}",
                r = t.nrows()
            );
        }
        let (offsets, total) = Self::concat_offsets(tensors, super::Tensor::ncols);
        let mut out = Self::zeros(nrows, total);
        for (ti, t) in tensors.iter().enumerate() {
            let cs = offsets[ti];
            out.slice_set(0..nrows, cs..(cs + t.ncols()), t);
        }
        out
    }

    /// Concatenate tensors along `axis` (0=row/vcat, 1=col/hcat).
    #[must_use]
    pub fn cat(tensors: &[&Self], axis: usize) -> Self {
        match axis {
            0 => Self::vcat(tensors),
            1 => Self::hcat(tensors),
            _ => panic!("nabla: cat axis {axis} out of bounds for 2-D tensor"),
        }
    }

    /// Stack tensors along a new axis, producing an `NdTensor`.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn stack(tensors: &[&Self], axis: usize) -> NdTensor<T> {
        assert!(!tensors.is_empty(), "nabla: stack on empty slice");
        assert!(
            axis <= 2,
            "nabla: stack axis {axis} out of bounds (max 2 for 2-D inputs)"
        );
        let (r, c) = tensors[0].shape();
        for (i, t) in tensors.iter().enumerate() {
            let (tr, tc) = t.shape();
            assert!(
                tr == r && tc == c,
                "nabla: stack tensor[{i}] shape ({tr}x{tc}) differs from tensor[0] ({r}x{c})"
            );
        }
        let n = tensors.len();
        let new_shape: [usize; 3] = match axis {
            0 => [n, r, c],
            1 => [r, n, c],
            _ => [r, c, n],
        };
        NdTensor::from_fn(&new_shape, |idx| {
            let (ti, row, col) = match axis {
                0 => (idx[0], idx[1], idx[2]),
                1 => (idx[1], idx[0], idx[2]),
                _ => (idx[2], idx[0], idx[1]),
            };
            tensors[ti].get(row, col)
        })
    }

    /// Split into at most `n` chunks along `axis` (last chunk may be smaller).
    #[must_use]
    pub fn chunk(&self, n: usize, axis: usize) -> Vec<Self> {
        let dim = match axis {
            0 => self.nrows(),
            1 => self.ncols(),
            _ => panic!("nabla: chunk axis {axis} out of bounds for 2-D tensor"),
        };
        let chunk_size = dim.div_ceil(n);
        (0..n)
            .filter_map(|i| {
                let start = i * chunk_size;
                (start < dim).then(|| {
                    let end = ((i + 1) * chunk_size).min(dim);
                    match axis {
                        0 => self.slice_rows(start..end),
                        _ => self.slice_cols(start..end),
                    }
                })
            })
            .collect()
    }

    /// Split into chunks of given sizes along axis.
    #[must_use]
    pub fn split(&self, sizes: &[usize], axis: usize) -> Vec<Self> {
        assert!(axis <= 1, "nabla: split axis must be 0 or 1, got {axis}");
        let mut offset = 0;
        sizes
            .iter()
            .map(|&s| {
                let part = match axis {
                    0 => self.submatrix(offset, offset + s, 0, self.ncols()),
                    _ => self.submatrix(0, self.nrows(), offset, offset + s),
                };
                offset += s;
                part
            })
            .collect()
    }

    // ---- Shape utilities ----

    /// Repeat tensor along each axis.
    #[must_use]
    pub fn repeat(&self, row_reps: usize, col_reps: usize) -> Self {
        Self::from_storage(B::repeat(&self.storage, row_reps, col_reps))
    }

    /// Expand (broadcast view) -- GPU-native broadcast, no D2H during CUDA Graph capture.
    #[must_use]
    pub fn expand(&self, target_rows: usize, target_cols: usize) -> Self {
        let (m, n) = self.shape();
        assert!(
            (m == 1 || m == target_rows) && (n == 1 || n == target_cols),
            "nabla: expand ({m},{n}) -> ({target_rows},{target_cols}) invalid"
        );
        let mut out = Self::zeros(target_rows, target_cols);
        B::expand_into(&mut out.storage, &self.storage, m, n);
        out
    }

    /// Broadcast `self` to match `other`'s shape.
    #[must_use]
    pub fn expand_as(&self, other: &Self) -> Self {
        let (m, n) = other.shape();
        self.expand(m, n)
    }

    /// Pad tensor. `padding = [left, right, top, bottom]`. Fill with `value`.
    #[must_use]
    pub fn pad(&self, padding: [usize; 4], value: T) -> Self {
        let [left, right, top, bottom] = padding;
        Self::from_storage(B::pad(&self.storage, left, right, top, bottom, value))
    }

    /// Upper triangular matrix (zero below diagonal + offset).
    #[must_use]
    pub fn triu(&self, diagonal: isize) -> Self {
        Self::from_storage(B::triu(&self.storage, diagonal))
    }

    /// Lower triangular matrix (zero above diagonal + offset).
    #[must_use]
    pub fn tril(&self, diagonal: isize) -> Self {
        Self::from_storage(B::tril(&self.storage, diagonal))
    }

    /// Roll elements along axis by `shift` positions (circular shift).
    #[must_use]
    pub fn roll(&self, shift: isize, axis: usize) -> Self {
        match axis {
            0 | 1 => Self::from_storage(B::roll(&self.storage, shift, axis)),
            _ => panic!("nabla: roll axis must be 0 or 1, got {axis}"),
        }
    }

    /// Flip (reverse) along axis.
    #[must_use]
    pub fn flip(&self, axis: usize) -> Self {
        match axis {
            0 | 1 => Self::from_storage(B::flip(&self.storage, axis)),
            _ => panic!("nabla: flip axis must be 0 or 1, got {axis}"),
        }
    }

    /// Create an n x n diagonal matrix from a vector (n x 1 or 1 x n).
    #[must_use]
    pub fn from_diag(v: &Self) -> Self {
        Self::from_storage(B::from_diag(&v.storage))
    }

    // ---- Indexing / gather / scatter ----

    /// Select rows by index. Duplicates allowed.
    #[must_use]
    pub fn gather_rows(&self, indices: &[usize]) -> Self {
        Self::from_storage(B::gather_rows(&self.storage, indices))
    }

    /// General gather along dimension.
    #[must_use]
    pub fn gather(&self, axis: usize, index: &Self) -> Self {
        match axis {
            0 | 1 => Self::from_storage(B::gather(&self.storage, axis, &index.storage)),
            _ => panic!("nabla: gather axis must be 0 or 1, got {axis}"),
        }
    }

    /// Scatter: write `src` values into self at positions given by `index` along `axis`.
    #[must_use]
    pub fn scatter(&self, axis: usize, index: &Self, src: &Self) -> Self {
        match axis {
            0 | 1 => Self::from_storage(B::scatter(
                &self.storage,
                axis,
                &index.storage,
                &src.storage,
            )),
            _ => panic!("nabla: scatter axis must be 0 or 1"),
        }
    }

    /// Select elements along axis by index vector.
    #[must_use]
    pub fn index_select(&self, axis: usize, index: &Self) -> Self {
        match axis {
            0 | 1 => Self::from_storage(B::index_select(&self.storage, axis, &index.storage)),
            _ => panic!("nabla: index_select axis must be 0 or 1, got {axis}"),
        }
    }

    /// Top-k values and indices along the given axis. Returns `(values, indices)`.
    ///
    /// - `axis=0`: top-k rows per column. Output shape `(k, ncols)`.
    /// - `axis=1`: top-k columns per row. Output shape `(nrows, k)`.
    #[must_use]
    pub fn topk(&self, k: usize, axis: usize) -> (Self, Self) {
        let (m, n) = self.shape();
        match axis {
            0 => {
                assert!(k <= m, "nabla: topk k={k} > nrows={m}");
                let t = self.t();
                let (vals_t, idxs_t) = B::topk_rows(&t.storage, k);
                (
                    Self::from_storage(vals_t).t(),
                    Self::from_storage(idxs_t).t(),
                )
            }
            1 => {
                assert!(k <= n, "nabla: topk k={k} > ncols={n}");
                let (vals, idxs) = B::topk_rows(&self.storage, k);
                (Self::from_storage(vals), Self::from_storage(idxs))
            }
            _ => panic!("nabla: topk axis must be 0 or 1, got {axis}"),
        }
    }

    /// Sort along axis=1 (rows). Returns `(sorted_values, indices)`.
    #[must_use]
    pub fn sort(&self, axis: usize, descending: bool) -> (Self, Self) {
        assert!(axis == 1, "nabla: sort currently supports axis=1 only");
        let (vals, idxs) = B::sort_rows(&self.storage, descending);
        (Self::from_storage(vals), Self::from_storage(idxs))
    }

    /// Return sorted permutation indices along `axis`.
    ///
    /// - `axis=0`: sort row indices by values in column 0.
    /// - `axis=1`: sort column indices by values in row 0.
    ///
    /// Use [`argsort_by`](Self::argsort_by) for sorting by an arbitrary row/column.
    #[must_use]
    pub fn argsort(&self, axis: usize, descending: bool) -> Vec<usize> {
        match axis {
            0 => self.argsort_by(0, 0, descending),
            1 => self.argsort_by(1, 0, descending),
            _ => panic!("nabla: argsort axis must be 0 or 1, got {axis}"),
        }
    }

    /// Return sorted permutation indices along `axis`, using `key` as the
    /// comparison row (when `axis=1`) or column (when `axis=0`).
    ///
    /// - `axis=0, key=c`: sort row indices by values in column `c`.
    /// - `axis=1, key=r`: sort column indices by values in row `r`.
    #[must_use]
    pub fn argsort_by(&self, axis: usize, key: usize, descending: bool) -> Vec<usize> {
        match axis {
            0 => {
                assert!(
                    key < self.ncols(),
                    "nabla: argsort_by key={key} >= ncols={}",
                    self.ncols()
                );
                let col = self.col(key);
                let (_, idxs) = col.t().sort(1, descending);
                B::to_vec_async(&idxs.storage)
                    .into_iter()
                    .map(|v| v.to_f64() as usize)
                    .collect()
            }
            1 => {
                assert!(
                    key < self.nrows(),
                    "nabla: argsort_by key={key} >= nrows={}",
                    self.nrows()
                );
                let row = self.row(key);
                let (_, idxs) = row.sort(1, descending);
                B::to_vec_async(&idxs.storage)
                    .into_iter()
                    .map(|v| v.to_f64() as usize)
                    .collect()
            }
            _ => panic!("nabla: argsort_by axis must be 0 or 1, got {axis}"),
        }
    }

    /// Create 2-D meshgrid from two 1-D tensors. Returns `(grid_x, grid_y)`.
    #[must_use]
    pub fn meshgrid(x: &Self, y: &Self) -> (Self, Self) {
        assert!(
            (x.nrows() == 1 || x.ncols() == 1) && (y.nrows() == 1 || y.ncols() == 1),
            "nabla: meshgrid expects 1-D inputs"
        );
        let (gx, gy) = B::meshgrid(&x.storage, &y.storage);
        (Self::from_storage(gx), Self::from_storage(gy))
    }

    /// Apply `f` to each row (axis=0) or column (axis=1) and collect.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn map_axis(&self, axis: usize, mut f: impl FnMut(Tensor<T, B>) -> Tensor<T, B>) -> Self {
        match axis {
            0 => {
                let slices: Vec<Self> = (0..self.nrows()).map(|r| f(self.row(r))).collect();
                let refs: Vec<&Self> = slices.iter().collect();
                Self::vcat(&refs)
            }
            1 => {
                let slices: Vec<Self> = (0..self.ncols()).map(|c| f(self.col(c))).collect();
                let refs: Vec<&Self> = slices.iter().collect();
                Self::hcat(&refs)
            }
            _ => panic!("nabla: map_axis axis must be 0 or 1, got {axis}"),
        }
    }

    /// Convert every element to a different scalar type `U` via `f64` as an intermediate.
    #[must_use]
    pub fn cast<U: Scalar>(&self) -> Tensor<U, B> {
        Tensor::from_storage(B::cast(&self.storage))
    }

    /// Convert 2D Tensor to N-D NdTensor with the given shape.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn into_nd(&self, shape: &[usize]) -> NdTensor<T> {
        let total: usize = shape.iter().product();
        let (m, n) = self.shape();
        assert_eq!(
            m * n,
            total,
            "nabla: into_nd shape mismatch: {m}x{n} != {total}"
        );
        let data: Vec<T> = (0..m)
            .flat_map(|r| (0..n).map(move |c| self.get(r, c)))
            .collect();
        NdTensor::from_vec(shape.to_vec(), data)
    }

    /// Fused `self += alpha * x` (BLAS Level 1 AXPY). Avoids intermediate allocation.
    pub fn axpy_(&mut self, alpha: T, x: &Self) {
        let (m, n) = self.shape();
        let (p, q) = x.shape();
        assert!(
            m == p && n == q,
            "nabla: axpy_ ({m}x{n}) vs ({p}x{q}) -- shapes must match"
        );
        let scaled = B::scale(&x.storage, alpha);
        self.storage = B::add(&self.storage, &scaled);
    }

    /// Scatter-add along dimension 0.
    pub fn scatter_add_dim0(&mut self, indices: &[usize], src: &Self) {
        let (sr, sc) = src.shape();
        assert_eq!(
            indices.len(),
            sr,
            "nabla: scatter_add_dim0 indices length {} != src nrows {}",
            indices.len(),
            sr
        );
        let (mr, mc) = self.shape();
        assert_eq!(
            sc, mc,
            "nabla: scatter_add_dim0 ncols mismatch: src {sc} != self {mc}"
        );
        for &target_r in indices {
            assert!(
                target_r < mr,
                "nabla: scatter_add_dim0 index {target_r} out of bounds for {mr} rows"
            );
        }
        B::scatter_add_dim0(&mut self.storage, indices, &src.storage);
    }

    /// Scatter-add along arbitrary axis (0=rows, 1=cols).
    pub fn scatter_add(&mut self, axis: usize, indices: &[usize], src: &Self) {
        let (sr, sc) = src.shape();
        let (mr, mc) = self.shape();
        match axis {
            0 => {
                assert_eq!(
                    indices.len(),
                    sr,
                    "nabla: scatter_add axis=0 indices length {} != src nrows {}",
                    indices.len(),
                    sr
                );
                assert_eq!(
                    sc, mc,
                    "nabla: scatter_add axis=0 ncols mismatch: src {sc} != self {mc}"
                );
                for &idx in indices {
                    assert!(
                        idx < mr,
                        "nabla: scatter_add index {idx} out of bounds for {mr} rows"
                    );
                }
            }
            1 => {
                assert_eq!(
                    indices.len(),
                    sc,
                    "nabla: scatter_add axis=1 indices length {} != src ncols {}",
                    indices.len(),
                    sc
                );
                assert_eq!(
                    sr, mr,
                    "nabla: scatter_add axis=1 nrows mismatch: src {sr} != self {mr}"
                );
                for &idx in indices {
                    assert!(
                        idx < mc,
                        "nabla: scatter_add index {idx} out of bounds for {mc} cols"
                    );
                }
            }
            _ => panic!("nabla: scatter_add axis must be 0 or 1, got {axis}"),
        }
        B::scatter_add(&mut self.storage, axis, indices, &src.storage);
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Parallel element-wise transform -- applies `f` to every element using rayon.
    #[must_use]
    pub fn par_map(&self, f: impl Fn(T) -> T + Send + Sync) -> Self {
        let (nrows, ncols) = self.shape();
        let data: Vec<T> = (0..nrows * ncols)
            .into_par_iter()
            .map(|idx| f(B::get(&self.storage, idx / ncols, idx % ncols)))
            .collect();
        Self::from_storage(B::from_vec(nrows, ncols, data))
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar> Tensor<T, Cpu> {
    /// Borrow the underlying row-major data slice (zero-copy).
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.storage.data_slice()
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar> core::ops::Index<(usize, usize)> for Tensor<T, Cpu> {
    type Output = T;

    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &T {
        let (nrows, ncols) = self.shape();
        assert!(
            r < nrows && c < ncols,
            "nabla: index ({r},{c}) out of bounds for {nrows}x{ncols} tensor"
        );
        self.storage.get_ref(r, c)
    }
}

#[cfg(feature = "cpu")]
impl<T: Scalar> core::ops::IndexMut<(usize, usize)> for Tensor<T, Cpu> {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        let (nrows, ncols) = self.shape();
        assert!(
            r < nrows && c < ncols,
            "nabla: index ({r},{c}) out of bounds for {nrows}x{ncols} tensor"
        );
        self.storage.get_mut(r, c)
    }
}
