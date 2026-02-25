//! # Sparse Poisson — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~20 | Julia LOC: ~15 (SparseArrays.jl)
//! Julia advantage: `spdiagm` one-liner for tridiagonal
//! nabla advantage: explicit triplet construction, type-safe solve
//!
//! Run: cargo run --example 06_sparse_solve --features cpu

#[cfg(feature = "cpu")]
use nabla::prelude::*;

#[cfg(feature = "cpu")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 1D Poisson: -u'' = f on [0,1], u(0)=u(1)=0
    // Finite difference: [-1, 2, -1] / h^2
    let n = 5; // interior points
    let h = 1.0 / (n as f64 + 1.0);
    let h2 = h * h;

    // Build tridiagonal sparse matrix
    let mut trips = Vec::new();
    for i in 0..n {
        trips.push(Triplet::new(i, i, 2.0 / h2));
        if i > 0 {
            trips.push(Triplet::new(i, i - 1, -1.0 / h2));
        }
        if i + 1 < n {
            trips.push(Triplet::new(i, i + 1, -1.0 / h2));
        }
    }
    let a = SparseMatrix::try_new_from_triplets(n, n, &trips)?;
    println!(
        "Sparse matrix: {}x{}, nnz={}",
        a.nrows(),
        a.ncols(),
        a.nnz()
    );

    // RHS: f(x) = 1 (constant source)
    let rhs = Tensor::fill(n, 1, 1.0_f64);

    // Solve Au = f
    let u = a.solve(&rhs)?;
    println!("Solution u:");
    for i in 0..n {
        println!("  x={:.4}: u={:.6}", (i + 1) as f64 * h, u.get(i, 0));
    }

    Ok(())
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("example 06_sparse_solve requires --features cpu");
}
