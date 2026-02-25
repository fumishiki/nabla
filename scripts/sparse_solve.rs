#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "../nabla", features = ["cpu"] }
//! ```

use nabla::prelude::*;

fn main() -> nabla::error::Result<()> {
    // 5x5 sparse matrix via triplets
    let trips = vec![
        Triplet::new(0, 0, 4.0_f64),
        Triplet::new(0, 1, -1.0),
        Triplet::new(1, 0, -1.0),
        Triplet::new(1, 1, 4.0),
        Triplet::new(1, 2, -1.0),
        Triplet::new(2, 1, -1.0),
        Triplet::new(2, 2, 4.0),
        Triplet::new(2, 3, -1.0),
        Triplet::new(3, 2, -1.0),
        Triplet::new(3, 3, 4.0),
        Triplet::new(3, 4, -1.0),
        Triplet::new(4, 3, -1.0),
        Triplet::new(4, 4, 4.0),
    ];
    let sp = SparseMatrix::try_new_from_triplets(5, 5, &trips)?;
    println!("Sparse 5x5 tridiagonal (nnz={})", sp.nnz());

    // SpMV via matmul_dense
    let b: Tensor<f64> = mat![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let y = sp.matmul_dense(&b)?;
    println!("S * b = [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        y.get(0, 0), y.get(1, 0), y.get(2, 0), y.get(3, 0), y.get(4, 0));

    // Sparse solve: S \ b
    let x = sp.solve(&b)?;
    println!("S \\ b = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        x.get(0, 0), x.get(1, 0), x.get(2, 0), x.get(3, 0), x.get(4, 0));

    // Verify: ||Sx - b||
    let residual = &(sp.matmul_dense(&x)?) - &b;
    let err = residual.emul(&residual).sum_all().sqrt();
    println!("||Sx - b|| = {err:.2e}");

    Ok(())
}
