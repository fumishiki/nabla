#[cfg(feature = "cpu")]
use rayon::prelude::*;

use crate::backend::Backend;
#[cfg(feature = "cpu")]
use crate::backend::Cpu;
use crate::scalar::Scalar;

use super::Tensor;
use super::variants::NdTensor;

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- Shape manipulation ----

    /// Reshape to `(m, n)`, preserving row-major element order.
    #[must_use]
    pub fn reshape(&self, m: usize, n: usize) -> Self {
        let (rows, cols) = self.shape();
        assert_eq!(
            m * n,
            rows * cols,
            "reshape: {rows}x{cols} cannot reshape to {m}x{n}"
        );
        Self::from_fn(m, n, |r, c| {
            let flat = r * n + c;
            self.get(flat / cols, flat % cols)
        })
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
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| self.get(r, c))
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
        Self::from_fn(total, ncols, |r, c| {
            let ti = offsets.partition_point(|&o| o <= r) - 1;
            tensors[ti].get(r - offsets[ti], c)
        })
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
        Self::from_fn(nrows, total, |r, c| {
            let ti = offsets.partition_point(|&o| o <= c) - 1;
            tensors[ti].get(r, c - offsets[ti])
        })
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
        match axis {
            0 => {
                let r = self.nrows();
                let chunk_size = r.div_ceil(n);
                (0..n).filter_map(|i| {
                    let start = i * chunk_size;
                    if start >= r { return None; }
                    Some(self.slice_rows(start..((i + 1) * chunk_size).min(r)))
                }).collect()
            }
            1 => {
                let c = self.ncols();
                let chunk_size = c.div_ceil(n);
                (0..n).filter_map(|i| {
                    let start = i * chunk_size;
                    if start >= c { return None; }
                    Some(self.slice_cols(start..((i + 1) * chunk_size).min(c)))
                }).collect()
            }
            _ => panic!("nabla: chunk axis {axis} out of bounds for 2-D tensor"),
        }
    }

    /// Split into chunks of given sizes along axis.
    #[must_use]
    pub fn split(&self, sizes: &[usize], axis: usize) -> Vec<Self> {
        match axis {
            0 => {
                let mut offset = 0;
                sizes
                    .iter()
                    .map(|&s| {
                        let part = self.submatrix(offset, offset + s, 0, self.ncols());
                        offset += s;
                        part
                    })
                    .collect()
            }
            1 => {
                let mut offset = 0;
                sizes
                    .iter()
                    .map(|&s| {
                        let part = self.submatrix(0, self.nrows(), offset, offset + s);
                        offset += s;
                        part
                    })
                    .collect()
            }
            _ => panic!("nabla: split axis must be 0 or 1, got {axis}"),
        }
    }

    // ---- Shape utilities ----

    /// Repeat tensor along each axis.
    #[must_use]
    pub fn repeat(&self, row_reps: usize, col_reps: usize) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m * row_reps, n * col_reps, |r, c| self.get(r % m, c % n))
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
        let (m, n) = self.shape();
        Self::from_fn(m + top + bottom, n + left + right, |r, c| {
            if r >= top && r < m + top && c >= left && c < n + left {
                self.get(r - top, c - left)
            } else {
                value
            }
        })
    }

    /// Upper triangular matrix (zero below diagonal + offset).
    #[must_use]
    pub fn triu(&self, diagonal: isize) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if (c as isize) >= (r as isize) + diagonal {
                self.get(r, c)
            } else {
                T::zero()
            }
        })
    }

    /// Lower triangular matrix (zero above diagonal + offset).
    #[must_use]
    pub fn tril(&self, diagonal: isize) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if (c as isize) <= (r as isize) + diagonal {
                self.get(r, c)
            } else {
                T::zero()
            }
        })
    }

    /// Roll elements along axis by `shift` positions (circular shift).
    #[must_use]
    pub fn roll(&self, shift: isize, axis: usize) -> Self {
        let (m, n) = self.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| {
                self.get(((r as isize - shift).rem_euclid(m as isize)) as usize, c)
            }),
            1 => Self::from_fn(m, n, |r, c| {
                self.get(r, ((c as isize - shift).rem_euclid(n as isize)) as usize)
            }),
            _ => panic!("nabla: roll axis must be 0 or 1, got {axis}"),
        }
    }

    /// Flip (reverse) along axis.
    #[must_use]
    pub fn flip(&self, axis: usize) -> Self {
        let (m, n) = self.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| self.get(m - 1 - r, c)),
            1 => Self::from_fn(m, n, |r, c| self.get(r, n - 1 - c)),
            _ => panic!("nabla: flip axis must be 0 or 1, got {axis}"),
        }
    }

    /// Create an n x n diagonal matrix from a vector (n x 1 or 1 x n).
    #[must_use]
    pub fn from_diag(v: &Self) -> Self {
        let n = v.nrows().max(v.ncols());
        let is_col = v.nrows() >= v.ncols();
        Self::from_fn(n, n, |r, c| {
            if r == c {
                if is_col { v.get(r, 0) } else { v.get(0, c) }
            } else {
                T::zero()
            }
        })
    }

    // ---- Indexing / gather / scatter ----

    /// Select rows by index. Duplicates allowed.
    #[must_use]
    pub fn gather_rows(&self, indices: &[usize]) -> Self {
        let nc = self.ncols();
        Self::from_fn(indices.len(), nc, |r, c| self.get(indices[r], c))
    }

    /// General gather along dimension.
    #[must_use]
    pub fn gather(&self, axis: usize, index: &Self) -> Self {
        let (m, n) = index.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| self.get(index.get(r, c).to_f64() as usize, c)),
            1 => Self::from_fn(m, n, |r, c| self.get(r, index.get(r, c).to_f64() as usize)),
            _ => panic!("nabla: gather axis must be 0 or 1, got {axis}"),
        }
    }

    /// Scatter: write `src` values into self at positions given by `index` along `axis`.
    #[must_use]
    pub fn scatter(&self, axis: usize, index: &Self, src: &Self) -> Self {
        let mut out = self.clone();
        let (si, sj) = index.shape();
        for r in 0..si {
            for c in 0..sj {
                let idx = index.get(r, c).to_f64() as usize;
                let val = src.get(r, c);
                match axis {
                    0 => out.set(idx, c, val),
                    1 => out.set(r, idx, val),
                    _ => panic!("nabla: scatter axis must be 0 or 1"),
                }
            }
        }
        out
    }

    /// Select elements along axis by index vector.
    #[must_use]
    pub fn index_select(&self, axis: usize, index: &Self) -> Self {
        let k = index.nrows() * index.ncols();
        let get_idx = |i: usize| -> usize {
            if index.nrows() == 1 {
                index.get(0, i).to_f64() as usize
            } else {
                index.get(i, 0).to_f64() as usize
            }
        };
        match axis {
            0 => Self::from_fn(k, self.ncols(), |r, c| self.get(get_idx(r), c)),
            1 => Self::from_fn(self.nrows(), k, |r, c| self.get(r, get_idx(c))),
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
                let mut all_vals = vec![T::zero(); k * n];
                let mut all_idxs = vec![T::zero(); k * n];
                for c in 0..n {
                    let mut pairs: Vec<(T, usize)> = (0..m).map(|r| (self.get(r, c), r)).collect();
                    pairs.sort_by(|a, b| b.0.to_f64().total_cmp(&a.0.to_f64()));
                    for j in 0..k {
                        all_vals[j * n + c] = pairs[j].0;
                        all_idxs[j * n + c] = T::from_f64(pairs[j].1 as f64);
                    }
                }
                let vals = Self::from_fn(k, n, |r, c| all_vals[r * n + c]);
                let idxs = Self::from_fn(k, n, |r, c| all_idxs[r * n + c]);
                (vals, idxs)
            }
            1 => {
                assert!(k <= n, "nabla: topk k={k} > ncols={n}");
                let mut all_vals = vec![T::zero(); m * k];
                let mut all_idxs = vec![T::zero(); m * k];
                for r in 0..m {
                    let mut pairs: Vec<(T, usize)> = (0..n).map(|c| (self.get(r, c), c)).collect();
                    pairs.sort_by(|a, b| b.0.to_f64().total_cmp(&a.0.to_f64()));
                    for j in 0..k {
                        all_vals[r * k + j] = pairs[j].0;
                        all_idxs[r * k + j] = T::from_f64(pairs[j].1 as f64);
                    }
                }
                let vals = Self::from_fn(m, k, |r, c| all_vals[r * k + c]);
                let idxs = Self::from_fn(m, k, |r, c| all_idxs[r * k + c]);
                (vals, idxs)
            }
            _ => panic!("nabla: topk axis must be 0 or 1, got {axis}"),
        }
    }

    /// Sort along axis=1 (rows). Returns `(sorted_values, indices)`.
    #[must_use]
    pub fn sort(&self, axis: usize, descending: bool) -> (Self, Self) {
        assert!(axis == 1, "nabla: sort currently supports axis=1 only");
        let (m, n) = self.shape();
        let mut all_vals = vec![T::zero(); m * n];
        let mut all_idxs = vec![T::zero(); m * n];
        for r in 0..m {
            let mut pairs: Vec<(T, usize)> = (0..n).map(|c| (self.get(r, c), c)).collect();
            if descending {
                pairs.sort_by(|a, b| b.0.to_f64().total_cmp(&a.0.to_f64()));
            } else {
                pairs.sort_by(|a, b| a.0.to_f64().total_cmp(&b.0.to_f64()));
            }
            for c in 0..n {
                all_vals[r * n + c] = pairs[c].0;
                all_idxs[r * n + c] = T::from_f64(pairs[c].1 as f64);
            }
        }
        let vals = Self::from_fn(m, n, |r, c| all_vals[r * n + c]);
        let idxs = Self::from_fn(m, n, |r, c| all_idxs[r * n + c]);
        (vals, idxs)
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
                let mut indices: Vec<usize> = (0..self.nrows()).collect();
                indices.sort_by(|&a, &b| {
                    let va = self.get(a, key).to_f64();
                    let vb = self.get(b, key).to_f64();
                    if descending {
                        vb.total_cmp(&va)
                    } else {
                        va.total_cmp(&vb)
                    }
                });
                indices
            }
            1 => {
                assert!(
                    key < self.nrows(),
                    "nabla: argsort_by key={key} >= nrows={}",
                    self.nrows()
                );
                let mut indices: Vec<usize> = (0..self.ncols()).collect();
                indices.sort_by(|&a, &b| {
                    let va = self.get(key, a).to_f64();
                    let vb = self.get(key, b).to_f64();
                    if descending {
                        vb.total_cmp(&va)
                    } else {
                        va.total_cmp(&vb)
                    }
                });
                indices
            }
            _ => panic!("nabla: argsort_by axis must be 0 or 1, got {axis}"),
        }
    }

    /// Create 2-D meshgrid from two 1-D tensors. Returns `(grid_x, grid_y)`.
    #[must_use]
    pub fn meshgrid(x: &Self, y: &Self) -> (Self, Self) {
        let nx = x.nrows() * x.ncols();
        let ny = y.nrows() * y.ncols();
        let get_x = |i: usize| {
            if x.nrows() == 1 {
                x.get(0, i)
            } else {
                x.get(i, 0)
            }
        };
        let get_y = |i: usize| {
            if y.nrows() == 1 {
                y.get(0, i)
            } else {
                y.get(i, 0)
            }
        };
        let gx = Self::from_fn(ny, nx, |_, c| get_x(c));
        let gy = Self::from_fn(ny, nx, |r, _| get_y(r));
        (gx, gy)
    }

    /// Apply `f` to each row (axis=0) or column (axis=1) and collect.
    #[must_use]
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
        for (r, &target_r) in indices.iter().enumerate() {
            assert!(
                target_r < mr,
                "nabla: scatter_add_dim0 index {target_r} out of bounds for {mr} rows"
            );
            for c in 0..sc {
                let old = self.get(target_r, c);
                self.set(target_r, c, old + src.get(r, c));
            }
        }
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
