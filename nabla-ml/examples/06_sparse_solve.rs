//! # Sparse Poisson -- explicit but compact
//! Run: cargo run --example 06_sparse_solve --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let n = 5;
    let h = 1.0 / (n as f64 + 1.0);
    let h2 = h * h;
    let diag = 2.0 / h2;
    let off = -1.0 / h2;

    let mut trips = Vec::new();
    for i in 0..n {
        trips.push(Triplet::new(i, i, diag));
        if i > 0 {
            trips.push(Triplet::new(i, i - 1, off));
        }
        if i + 1 < n {
            trips.push(Triplet::new(i, i + 1, off));
        }
    }
    let a = SparseMatrix::try_new_from_triplets(n, n, &trips)?;
    println!("Sparse matrix: {}x{}, nnz={}", a.nrows(), a.ncols(), a.nnz());

    let rhs = Tensor::fill(n, 1, 1.0_f64);

    let u = a.solve(&rhs)?;
    println!("Solution u:");
    for i in 0..n {
        println!("  x={:.4}: u={:.6}", (i + 1) as f64 * h, u.get(i, 0));
    }
}
