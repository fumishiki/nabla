use core::fmt;
use core::ops::{Add, Mul};

use crate::linalg::{LinalgExt, Side};
use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

#[inline]
fn sparse_error<T: fmt::Display>(op: &'static str, shape: (usize, usize), err: T) -> Error {
    Error::invalid(format!("{op} failed for sparse matrix {shape:?}: {err}"))
}

#[inline]
fn check_triplet_in_bounds(
    op: &'static str,
    nrows: usize,
    ncols: usize,
    row: usize,
    col: usize,
) -> Result<()> {
    if row >= nrows {
        return Err(sparse_error(
            op,
            (nrows, ncols),
            format!("row index {row} out of bounds"),
        ));
    }
    if col >= ncols {
        return Err(sparse_error(
            op,
            (nrows, ncols),
            format!("col index {col} out of bounds"),
        ));
    }
    Ok(())
}

#[inline]
fn check_rhs_rows<T: Scalar>(expected_rows: usize, rhs: &Tensor<T, Cpu>) -> Result<()> {
    if expected_rows == rhs.nrows() {
        Ok(())
    } else {
        Err(Error::mismatch((expected_rows, rhs.ncols()), rhs.shape()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triplet<T: Scalar> {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
    /// Value.
    pub val: T,
}

impl<T: Scalar> Triplet<T> {
    /// Construct a new triplet.
    #[must_use]
    #[inline]
    pub fn new(row: usize, col: usize, val: T) -> Self {
        Self { row, col, val }
    }
}

pub fn sparse<T: Scalar>(
    nrows: usize,
    ncols: usize,
    entries: &[(usize, usize, T)],
) -> Result<SparseMatrix<T>> {
    let triplets: Vec<Triplet<T>> = entries
        .iter()
        .copied()
        .map(|(row, col, val)| Triplet::new(row, col, val))
        .collect();
    SparseMatrix::try_new_from_triplets(nrows, ncols, &triplets)
}

#[derive(Clone)]
pub struct SparseMatrix<T: Scalar> {
    nrows: usize,
    ncols: usize,
    /// Column pointers (length ncols+1).
    col_ptr: Vec<usize>,
    /// Row indices (length nnz).
    row_idx: Vec<usize>,
    /// Values (length nnz).
    values: Vec<T>,
}

impl<T: Scalar> SparseMatrix<T> {
    /// Build from COO triplets.
    ///
    /// # Errors
    /// Returns `Err` when indices are out of bounds.
    pub fn try_new_from_triplets(
        nrows: usize,
        ncols: usize,
        entries: &[Triplet<T>],
    ) -> Result<Self> {
        Self::build_csc(nrows, ncols, entries)
    }

    /// Build from COO triplets (nonnegative indices, same as `try_new_from_triplets`).
    ///
    /// # Errors
    /// Returns `Err` when indices are out of bounds.
    pub fn try_new_from_nonnegative_triplets(
        nrows: usize,
        ncols: usize,
        entries: &[Triplet<T>],
    ) -> Result<Self> {
        Self::try_new_from_triplets(nrows, ncols, entries)
    }

    /// Build an n x n tridiagonal sparse matrix with constant diagonals.
    ///
    /// # Errors
    /// Returns `Err` when `n == 0`.
    pub fn tridiag(n: usize, sub: T, diag: T, sup: T) -> Result<Self> {
        if n == 0 {
            return Err(Error::invalid("tridiag: n must be > 0"));
        }
        let cap = if n == 1 { 1 } else { 3 * n - 2 };
        let mut trips = Vec::with_capacity(cap);
        for i in 0..n {
            if i > 0 {
                trips.push(Triplet::new(i, i - 1, sub));
            }
            trips.push(Triplet::new(i, i, diag));
            if i + 1 < n {
                trips.push(Triplet::new(i, i + 1, sup));
            }
        }
        Self::try_new_from_triplets(n, n, &trips)
    }

    fn build_csc(nrows: usize, ncols: usize, entries: &[Triplet<T>]) -> Result<Self> {
        // Note: only accessible through concrete type impls
        for e in entries {
            check_triplet_in_bounds("build_csc", nrows, ncols, e.row, e.col)?;
        }

        // Count entries per column
        let mut col_counts = vec![0usize; ncols];
        for e in entries {
            col_counts[e.col] += 1;
        }

        // Build col_ptr (prefix sum)
        let mut col_ptr = vec![0usize; ncols + 1];
        for j in 0..ncols {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }

        // Fill row_idx and values (stable sort within each column by row)
        let nnz = entries.len();
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut pos = col_ptr.clone();

        for e in entries {
            let p = pos[e.col];
            row_idx[p] = e.row;
            values[p] = e.val;
            pos[e.col] += 1;
        }

        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Number of rows.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Matrix shape as `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored (non-zero) entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Visit all stored entries as `(row, col, value)`.
    fn for_each_entry(&self, mut f: impl FnMut(usize, usize, T)) {
        for j in 0..self.ncols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                f(self.row_idx[p], j, self.values[p]);
            }
        }
    }

    /// Multiply `self × dense_rhs` into a dense tensor.
    ///
    /// # Errors
    /// Returns `Err` when shapes are incompatible.
    pub fn matmul_dense(&self, rhs: &Tensor<T, Cpu>) -> Result<Tensor<T, Cpu>> {
        if self.ncols != rhs.nrows() {
            return Err(Error::mismatch((self.nrows, rhs.ncols()), rhs.shape()));
        }
        let m = self.nrows;
        let n = rhs.ncols();
        let mut out = Tensor::zeros(m, n);
        self.for_each_entry(|i, j, a_ij| {
            for k in 0..n {
                let old = out.get(i, k);
                out.set(i, k, old + a_ij * rhs.get(j, k));
            }
        });
        Ok(out)
    }

    fn to_dense(&self) -> Tensor<T, Cpu> {
        let mut out = Tensor::zeros(self.nrows, self.ncols);
        self.for_each_entry(|i, j, val| out.set(i, j, val));
        out
    }

    /// Transpose: swap row/col indices and rebuild as CSC.
    ///
    /// The result has shape `(ncols, nrows)`.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.nnz());
        self.for_each_entry(|i, j, val| {
            triplets.push(Triplet::new(j, i, val));
        });
        // Transposed dimensions: rows become cols and vice versa.
        // build_csc cannot fail here because indices are guaranteed in bounds.
        Self::build_csc(self.ncols, self.nrows, &triplets).unwrap_or_else(|_| {
            // Indices originate from a valid matrix, so this is unreachable.
            Self {
                nrows: self.ncols,
                ncols: self.nrows,
                col_ptr: vec![0; self.nrows + 1],
                row_idx: Vec::new(),
                values: Vec::new(),
            }
        })
    }

    /// Short alias for [`transpose`](Self::transpose).
    #[must_use]
    #[inline]
    pub fn t(&self) -> Self {
        self.transpose()
    }
}

pub fn speye<T: Scalar>(n: usize) -> Result<SparseMatrix<T>> {
    let entries: Vec<(usize, usize, T)> = (0..n).map(|i| (i, i, T::one())).collect();
    sparse(n, n, &entries)
}

impl<T: Scalar> SparseMatrix<T> {
    /// Sparse-sparse addition via COO merge.
    ///
    /// Both matrices must have the same shape.
    ///
    /// # Errors
    /// Returns `Err` when shapes do not match.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return Err(Error::mismatch(self.shape(), other.shape()));
        }
        // Collect all triplets from both matrices.
        let capacity = self.nnz() + other.nnz();
        let mut triplets = Vec::with_capacity(capacity);
        self.for_each_entry(|i, j, val| triplets.push(Triplet::new(i, j, val)));
        other.for_each_entry(|i, j, val| triplets.push(Triplet::new(i, j, val)));
        // build_csc will place duplicate (i,j) entries adjacent; they accumulate
        // naturally when iterated by matmul_dense / to_dense. For a clean
        // representation we merge duplicates here.
        Self::build_csc_merged(self.nrows, self.ncols, &triplets)
    }

    /// Build CSC from triplets, summing duplicate (row, col) entries.
    fn build_csc_merged(nrows: usize, ncols: usize, entries: &[Triplet<T>]) -> Result<Self> {
        for e in entries {
            check_triplet_in_bounds("sparse_add", nrows, ncols, e.row, e.col)?;
        }

        // Sort by (col, row) to group duplicates.
        let mut sorted: Vec<Triplet<T>> = entries.to_vec();
        sorted.sort_by(|a, b| a.col.cmp(&b.col).then(a.row.cmp(&b.row)));

        // Merge duplicates.
        let mut merged_row: Vec<usize> = Vec::with_capacity(sorted.len());
        let mut merged_col: Vec<usize> = Vec::with_capacity(sorted.len());
        let mut merged_val: Vec<T> = Vec::with_capacity(sorted.len());

        for e in &sorted {
            if let (Some(&last_r), Some(&last_c)) = (merged_row.last(), merged_col.last())
                && last_r == e.row
                && last_c == e.col
            {
                if let Some(v) = merged_val.last_mut() {
                    *v = *v + e.val;
                }
                continue;
            }
            merged_row.push(e.row);
            merged_col.push(e.col);
            merged_val.push(e.val);
        }

        // Build col_ptr.
        let mut col_ptr = vec![0usize; ncols + 1];
        for &c in &merged_col {
            col_ptr[c + 1] += 1;
        }
        for j in 0..ncols {
            col_ptr[j + 1] += col_ptr[j];
        }

        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx: merged_row,
            values: merged_val,
        })
    }
}

impl<T: Scalar> Add for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        match self.add(rhs) {
            Ok(out) => out,
            Err(err) => panic!("nabla: sparse add failed: {err}"),
        }
    }
}

impl SparseMatrix<f64> {
    /// Solve `A·x = b` via sparse Cholesky (lower-triangle convention).
    ///
    /// # Errors
    /// Returns `Err` when factorization or solve fails.
    pub fn chol_solve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        self.cholesky_solve(Side::Lower, rhs)
    }

    /// Solve `A·x = b` via sparse Cholesky (positive-definite symmetric).
    ///
    /// # Errors
    /// Returns `Err` when factorization or solve fails.
    pub fn cholesky_solve(&self, side: Side, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        check_rhs_rows(self.nrows, rhs)?;
        let dense = self.to_dense();
        let llt = dense.llt(side)?;
        Ok(llt.solve(rhs))
    }

    /// Solve `A·x = b` via sparse LU.
    ///
    /// # Errors
    /// Returns `Err` when factorization or solve fails.
    pub fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        check_rhs_rows(self.nrows, rhs)?;
        let dense = self.to_dense();
        dense.solve(rhs)
    }

    /// Least-squares solve `A·x ≈ b` via sparse QR.
    ///
    /// # Errors
    /// Returns `Err` when shapes are incompatible or solve fails.
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        check_rhs_rows(self.nrows, rhs)?;
        let dense = self.to_dense();
        dense.solve_lstsq(rhs)
    }
}

impl<T: Scalar> Mul<&Tensor<T, Cpu>> for &SparseMatrix<T> {
    type Output = Tensor<T, Cpu>;

    fn mul(self, rhs: &Tensor<T, Cpu>) -> Self::Output {
        match self.matmul_dense(rhs) {
            Ok(out) => out,
            Err(err) => panic!("nabla: sparse matmul failed: {err}"),
        }
    }
}

impl<T: Scalar> Mul<&Tensor<T, Cpu>> for SparseMatrix<T> {
    type Output = Tensor<T, Cpu>;

    fn mul(self, rhs: &Tensor<T, Cpu>) -> Self::Output {
        (&self).mul(rhs)
    }
}

pub type SparseColMat<T> = SparseMatrix<T>;

#[derive(Clone)]
pub struct BcsrMatrix<T: Scalar> {
    block_size: usize,
    nblock_rows: usize,
    nblock_cols: usize,
    /// Block-row pointers (length = nblock_rows + 1).
    row_ptrs: Vec<usize>,
    /// Block-column indices for each non-zero block.
    col_idxs: Vec<usize>,
    /// Dense block values: nnz_blocks * B * B, row-major within each block.
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

fn find_or_insert_block<S: Scalar>(
    map: &mut Vec<((usize, usize), Vec<S>)>,
    br: usize,
    bc: usize,
    b2: usize,
) -> &mut Vec<S> {
    let pos = map.iter().position(|((r, c), _)| *r == br && *c == bc);
    if let Some(i) = pos {
        &mut map[i].1
    } else {
        map.push(((br, bc), vec![S::zero(); b2]));
        let last = map.len() - 1;
        &mut map[last].1
    }
}

impl<T: Scalar> BcsrMatrix<T> {
    /// Convert a CSC `SparseMatrix` into BCSR with given block size.
    #[must_use]
    pub fn from_sparse(s: &SparseMatrix<T>, block_size: usize) -> Self {
        let b = block_size;
        let nblock_rows = s.nrows().div_ceil(b);
        let nblock_cols = s.ncols().div_ceil(b);
        let b2 = b * b;

        let mut block_map: Vec<((usize, usize), Vec<T>)> = Vec::new();

        // Convert via dense (SparseMatrix fields are private)
        let dense = s.matmul_dense(&Tensor::identity(s.ncols()));
        if let Ok(ref d) = dense {
            for i in 0..s.nrows() {
                for j in 0..s.ncols() {
                    let v = d.get(i, j);
                    if v != T::zero() {
                        let br = i / b;
                        let bc = j / b;
                        let block = find_or_insert_block(&mut block_map, br, bc, b2);
                        block[(i % b) * b + j % b] = v;
                    }
                }
            }
        }

        block_map.sort_by_key(|&((br, bc), _)| (br, bc));

        let mut row_ptrs = vec![0usize; nblock_rows + 1];
        let mut col_idxs = Vec::with_capacity(block_map.len());
        let mut values = Vec::with_capacity(block_map.len() * b2);

        for &((br, bc), ref block) in &block_map {
            row_ptrs[br + 1] += 1;
            col_idxs.push(bc);
            values.extend_from_slice(block);
        }

        for i in 0..nblock_rows {
            row_ptrs[i + 1] += row_ptrs[i];
        }

        Self {
            block_size: b,
            nblock_rows,
            nblock_cols,
            row_ptrs,
            col_idxs,
            values,
            nrows: s.nrows(),
            ncols: s.ncols(),
        }
    }

    /// Number of non-zero blocks.
    #[must_use]
    #[inline]
    pub fn nnz_blocks(&self) -> usize {
        self.col_idxs.len()
    }

    /// Block density: nnz_blocks / total_blocks.
    #[must_use]
    #[inline]
    pub fn density(&self) -> f64 {
        let total = self.nblock_rows * self.nblock_cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz_blocks() as f64 / total as f64
    }

    /// Sparse-dense matrix multiply: `self × x`.
    ///
    /// `x` shape: `(ncols, k)`, output shape: `(nrows, k)`.
    #[must_use]
    pub fn spmm(&self, x: &Tensor<T, Cpu>) -> Tensor<T, Cpu> {
        assert_eq!(self.ncols, x.nrows(), "BcsrMatrix::spmm dimension mismatch");
        let k = x.ncols();
        let b = self.block_size;
        let mut out = Tensor::zeros(self.nrows, k);

        for br in 0..self.nblock_rows {
            for p in self.row_ptrs[br]..self.row_ptrs[br + 1] {
                let bc = self.col_idxs[p];
                let block_base = p * b * b;
                let row_base = br * b;
                let col_base = bc * b;
                let row_end = (row_base + b).min(self.nrows);
                let col_end = (col_base + b).min(self.ncols);

                // B×B block at (br, bc) × corresponding rows of x
                for row in row_base..row_end {
                    let row_offset = row - row_base;
                    for col in col_base..col_end {
                        let a_val = self.values[block_base + row_offset * b + (col - col_base)];
                        if a_val == T::zero() {
                            continue;
                        }
                        for c in 0..k {
                            let old = out.get(row, c);
                            out.set(row, c, old + a_val * x.get(col, c));
                        }
                    }
                }
            }
        }
        out
    }
}

pub fn mixed_spmm_f64(
    a: &BcsrMatrix<f32>,
    x: &Tensor<f64, Cpu>,
    tol: f64,
    max_iter: usize,
) -> Tensor<f64, Cpu> {
    let k = x.ncols();
    assert_eq!(a.ncols, x.nrows(), "mixed_spmm_f64: A.ncols != x.nrows");

    // f64 ground truth: reconstruct A element-by-element for residual correction
    let a_f64 = Tensor::<f64, Cpu>::from_fn(a.nrows, a.ncols, |i, j| {
        let br = i / a.block_size;
        let bc = j / a.block_size;
        let lr = i % a.block_size;
        let lc = j % a.block_size;
        for p in a.row_ptrs[br]..a.row_ptrs[br + 1] {
            if a.col_idxs[p] == bc {
                return f64::from(
                    a.values[p * a.block_size * a.block_size + lr * a.block_size + lc],
                );
            }
        }
        0.0
    });

    // Ground truth in f64
    let target = &a_f64 * x;

    // Initial f32 approximation
    let x_f32 = Tensor::<f32, Cpu>::from_fn(x.nrows(), k, |i, j| x.get(i, j) as f32);
    let y0_f32 = a.spmm(&x_f32);
    let mut y = Tensor::<f64, Cpu>::from_fn(a.nrows, k, |i, j| f64::from(y0_f32.get(i, j)));

    // Iterative refinement: correct residual
    for _ in 0..max_iter {
        let r = &target - &y;
        if r.norm() < tol {
            break;
        }
        y = &y + &r;
    }

    y
}
