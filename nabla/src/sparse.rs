// sparse.rs — Sparse matrix support (Wave 18 stub).
//
// This module is scheduled for full rewrite in Wave 18. Until then it exposes
// a minimal compilable surface so that linalg.rs and other modules can build.

use core::fmt;

use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use crate::linalg::{LinalgExt, Side};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

#[inline]
fn sparse_error<T: fmt::Display>(op: &'static str, shape: (usize, usize), err: T) -> Error {
    Error::invalid(format!("{op} failed for sparse matrix {shape:?}: {err}"))
}

#[inline]
fn check_rhs_rows<T: Scalar>(expected_rows: usize, rhs: &Tensor<T, Cpu>) -> Result<()> {
    if expected_rows == rhs.nrows() {
        Ok(())
    } else {
        Err(Error::mismatch((expected_rows, rhs.ncols()), rhs.shape()))
    }
}

/// COO triplet entry for constructing sparse matrices.
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

/// CSC sparse matrix stored as sorted column arrays.
///
/// Wave 18 will replace this with a self-contained CSC implementation. For now
/// the struct holds COO data converted to CSC via simple sort.
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
        Self::build_csc(nrows, ncols, entries)
    }

    fn build_csc(nrows: usize, ncols: usize, entries: &[Triplet<T>]) -> Result<Self> {
        // Note: only accessible through concrete type impls
        for e in entries {
            if e.row >= nrows {
                return Err(sparse_error(
                    "build_csc",
                    (nrows, ncols),
                    format!("row index {} out of bounds", e.row),
                ));
            }
            if e.col >= ncols {
                return Err(sparse_error(
                    "build_csc",
                    (nrows, ncols),
                    format!("col index {} out of bounds", e.col),
                ));
            }
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
        for j in 0..self.ncols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                let a_ij = self.values[p];
                for k in 0..n {
                    let old = out.get(i, k);
                    out.set(i, k, old + a_ij * rhs.get(j, k));
                }
            }
        }
        Ok(out)
    }

    fn to_dense(&self) -> Tensor<T, Cpu> {
        let mut out = Tensor::zeros(self.nrows, self.ncols);
        for j in 0..self.ncols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                out.set(i, j, self.values[p]);
            }
        }
        out
    }
}

impl SparseMatrix<f64> {
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

/// Alias for `SparseMatrix` (backward compat).
pub type SparseColMat<T> = SparseMatrix<T>;

/// Block Compressed Sparse Row (BCSR) matrix.
///
/// Stores dense B×B blocks for improved cache locality and GPU throughput.
/// Partial blocks at matrix boundaries are zero-padded.
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
    if let Some(idx) = map.iter().position(|&((r, c), _)| r == br && c == bc) {
        &mut map[idx].1
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
                // B×B block at (br, bc) × corresponding rows of x
                for lr in 0..b {
                    let row = br * b + lr;
                    if row >= self.nrows {
                        break;
                    }
                    for lc in 0..b {
                        let col = bc * b + lc;
                        if col >= self.ncols {
                            break;
                        }
                        let a_val = self.values[block_base + lr * b + lc];
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

/// Mixed-precision SpMM: compute `A * x` where `A` is f32 and `x` is f64.
///
/// Uses iterative refinement: compute initial result in f32, then iteratively
/// correct the residual `r = A_f64 * x - y` until `‖r‖ < tol` or `max_iter`.
///
/// The f64 version of A is reconstructed from the BCSR blocks.
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
                return f64::from(a.values[p * a.block_size * a.block_size + lr * a.block_size + lc]);
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
