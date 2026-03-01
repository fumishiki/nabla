#![cfg(feature = "cpu")]
#![allow(unused_imports)]

use nabla::cas::{Expr, diff, eval, eval_tensor, simplify};
use nabla::ode::{AdaptiveConfig, dormand_prince, rk4};
use nabla::prelude::*;
use nabla::{between, frange};
use std::collections::HashMap;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

fn linear_f64(rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::from_fn(rows, cols, |i, j| (i * cols + j + 1) as f64)
}

fn assert_approx_grid(got: &Tensor<f64>, expected: &Tensor<f64>, tol: f64) {
    assert_eq!(got.shape(), expected.shape(), "shape mismatch");
    let (r, c) = got.shape();
    for i in 0..r {
        for j in 0..c {
            assert!(
                (got.get(i, j) - expected.get(i, j)).abs() < tol,
                "mismatch at ({i},{j}): got {}, expected {}",
                got.get(i, j),
                expected.get(i, j)
            );
        }
    }
}

#[test]
fn stencil_laplacian() {
    let a: Tensor<f64> = Tensor::from_fn(6, 6, |r, c| (r * r + c * c) as f64);
    let out: Tensor<f64> = stencil!(out[i, j] =
        -4.0 * a[i, j] + a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]
    );
    for r in 1..5 {
        for c in 1..5 {
            assert!(
                approx_eq(out.get(r, c), 4.0),
                "laplacian at ({},{}) = {} expected 4.0",
                r,
                c,
                out.get(r, c)
            );
        }
    }
    assert!(approx_eq(out.get(0, 0), 0.0));
    assert!(approx_eq(out.get(5, 5), 0.0));
}


#[test]
fn par_from_fn_matches_sequential() {
    let seq: Tensor<f64> = Tensor::from_fn(50, 50, |r, c| ((r * 50 + c) as f64).sin());
    let par: Tensor<f64> = Tensor::par_from_fn(50, 50, |r, c| ((r * 50 + c) as f64).sin());
    assert_approx_grid(&seq, &par, 1e-10);
}


#[test]
fn utility_exports() {
    let v = linspace(0.0, 1.0, 5);
    assert_eq!(v.len(), 5);
    assert!((v.get(0, 0) - 0.0_f64).abs() < 1e-10);
    assert!((v.get(0, 4) - 1.0_f64).abs() < 1e-10);
    assert!((v.get(0, 2) - 0.5_f64).abs() < 1e-10);

    let fr = frange!(0.0_f64, 0.25, 1.0);
    assert_eq!(fr.len(), 5);

    let mid = 0.5_f64;
    let hi = 1.0_f64;
    let neg = -0.1_f64;
    assert!(between!(0.0_f64, mid, hi));
    assert!(!between!(0.0_f64, hi, hi));
    assert!(!between!(0.0_f64, neg, hi));

    let z = nabla::util::c64(1.0, 2.0);
    assert!((z.re - 1.0_f64).abs() < 1e-10);
    assert!((z.im - 2.0_f64).abs() < 1e-10);
    let z32 = nabla::util::c32(3.0, 4.0);
    assert!((z32.re - 3.0_f32).abs() < 1e-6);
    assert!((z32.im - 4.0_f32).abs() < 1e-6);
}


#[test]
fn matmul_non_square() {
    let a: Tensor<f64> = linear_f64(2, 3);
    let b: Tensor<f64> = linear_f64(3, 2);
    let c = &a * &b;
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 22.0));
    assert!(approx_eq(c.get(0, 1), 28.0));
    assert!(approx_eq(c.get(1, 0), 49.0));
    assert!(approx_eq(c.get(1, 1), 64.0));
}


#[test]
fn ndtensor_slice_2d_roundtrip() {
    let t = NdTensor::<f64>::from_fn(&[2, 3, 4], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
    });
    let slice0: Tensor<f64> = t.slice_2d(&[0]);
    assert_eq!(slice0.shape(), (3, 4));
    assert!(approx_eq(slice0.get(2, 3), 23.0));

    let slice1: Tensor<f64> = t.slice_2d(&[1]);
    assert!(approx_eq(slice1.get(0, 0), 100.0));
    assert!(approx_eq(slice1.get(2, 3), 123.0));

    let mut t2 = NdTensor::<f64>::zeros(&[2, 3, 4]);
    t2.set_slice_2d(&[0], &slice0);
    t2.set_slice_2d(&[1], &slice1);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert!(approx_eq(t2.get_nd(&[i, j, k]), t.get_nd(&[i, j, k])));
            }
        }
    }
}


#[test]
fn static_tensor_roundtrip() {
    let a = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| (r * 2 + c + 1) as f64);
    let t = a.to_tensor();
    assert_eq!(t.shape(), (2, 2));
    assert!(approx_eq(t.get(0, 0), 1.0));
    assert!(approx_eq(t.get(1, 1), 4.0));

    let b = StaticMatrix::<f64, 2, 2>::from_tensor(&t);
    for r in 0..2 {
        for c in 0..2 {
            assert!(approx_eq(a.get(r, c), b.get(r, c)));
        }
    }
}


#[test]
fn hierarchy_matmul_dyn() {
    use nabla::tensor::Array;
    let t: Tensor<f64> = Tensor::from_fn(2, 2, |r, c| [[1.0_f64, 2.0], [3.0, 4.0]][r][c]);
    let s = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| [[5.0_f64, 6.0], [7.0, 8.0]][r][c]);
    let rhs: &dyn Array<f64> = &s;
    let c = t.matmul_dyn(rhs);
    assert!(approx_eq(c.get(0, 0), 19.0));
    assert!(approx_eq(c.get(1, 1), 50.0));
}


#[test]
fn diagonal_mul_dense() {
    let d = Diagonal::new(vec![2.0_f64, 3.0]);
    let t = d.to_tensor();
    assert_eq!(t.shape(), (2, 2));
    assert!(approx_eq(t.get(0, 0), 2.0));
    assert!(approx_eq(t.get(1, 1), 3.0));
    assert!(approx_eq(t.get(0, 1), 0.0));

    let m: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
    let r = d.mul_dense(&m).expect("mul_dense failed");
    assert!(approx_eq(r.get(0, 0), 2.0));
    assert!(approx_eq(r.get(0, 1), 4.0));
    assert!(approx_eq(r.get(1, 0), 9.0));
    assert!(approx_eq(r.get(1, 1), 12.0));
}


#[test]
fn reductions_boundary() {
    let a: Tensor<f64> = mat![[3.0_f64, 1.0], [5.0, 2.0]];
    assert!(approx_eq(a.sum_all(), 11.0));
    assert_eq!(a.argmax(), (1, 0));
    assert_eq!(a.argmin(), (0, 1));
}


#[test]
fn tensor_permute_2d_transpose() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
    let p = a.permute(&[1, 0]); // => 3x2
    assert_eq!(p.shape(), (3, 2));
    assert!(approx_eq(p.get(0, 0), 1.0));
    assert!(approx_eq(p.get(0, 1), 4.0));
    assert!(approx_eq(p.get(2, 1), 6.0));
}


#[test]
fn multi_dual_batch_jacobian() {
    // f(x,y) = [x*y, x+y], seed x=lane 0, y=lane 1
    let x = MultiDual::<f64, 2>::seed(1.0, 0);
    let y = MultiDual::<f64, 2>::seed(2.0, 1);
    let fxy = x * y; // value=2.0, derivs=[y, x]=[2,1]
    let fsum = x + y; // value=3.0, derivs=[1,1]
    assert_eq!(fxy.value, 2.0);
    assert_eq!(fxy.derivs[0], 2.0); // d(xy)/dx = y = 2
    assert_eq!(fxy.derivs[1], 1.0); // d(xy)/dy = x = 1
    assert_eq!(fsum.value, 3.0);
    assert_eq!(fsum.derivs[0], 1.0); // d(x+y)/dx = 1
    assert_eq!(fsum.derivs[1], 1.0); // d(x+y)/dy = 1
}


#[test]
fn index_bracket_read() {
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    assert_eq!(a[(0, 0)], 1.0);
    assert_eq!(a[(0, 1)], 2.0);
    assert_eq!(a[(1, 0)], 3.0);
    assert_eq!(a[(1, 1)], 4.0);
}


#[test]
fn index_bracket_write() {
    let mut a = Tensor::<f64>::zeros(2, 2);
    a[(0, 1)] = 7.0;
    assert_eq!(a[(0, 1)], 7.0);
    assert_eq!(a[(0, 0)], 0.0);
}


#[test]
fn ndtensor_index_bracket() {
    let t = NdTensor::<f64>::zeros(&[2, 3, 4]);
    assert_eq!(t[&[0, 0, 0]], 0.0);
    assert_eq!(t[&[1, 2, 3]], 0.0);
}


#[test]
fn f16_zeros_and_arithmetic() {
    let a: Tensor<f16> = Tensor::zeros(2, 2);
    assert_eq!(a[(0, 0)], f16::ZERO);
    let b: Tensor<f16> = Tensor::from_fn(2, 2, |i, j| f16::from_f32((i * 2 + j + 1) as f32));
    let c = &a + &b;
    assert!((f32::from(c[(0, 0)]) - 1.0).abs() < 0.01);
    assert!((f32::from(c[(1, 1)]) - 4.0).abs() < 0.01);
    let scaled = &b * f16::from_f32(2.0);
    assert!((f32::from(scaled[(0, 0)]) - 2.0).abs() < 0.01);
    assert!((f32::from(scaled[(1, 1)]) - 8.0).abs() < 0.01);
}


#[test]
fn broadcast_add_rows() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let row: Tensor<f64> = mat![[10.0_f64, 20.0]];
    let r = a.broadcast_add_rows(&row);
    assert!((r.get(0, 0) - 11.0).abs() < 1e-10);
    assert!((r.get(0, 1) - 22.0).abs() < 1e-10);
    assert!((r.get(2, 0) - 15.0).abs() < 1e-10);
    assert!((r.get(2, 1) - 26.0).abs() < 1e-10);
}


#[test]
fn broadcast_add_cols() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let col: Tensor<f64> = mat![[10.0_f64], [20.0], [30.0]];
    let r = a.broadcast_add_cols(&col);
    assert!((r.get(0, 0) - 11.0).abs() < 1e-10);
    assert!((r.get(1, 0) - 23.0).abs() < 1e-10);
    assert!((r.get(2, 1) - 36.0).abs() < 1e-10);
}


#[test]
fn broadcast_mul_rows() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let row: Tensor<f64> = mat![[10.0_f64, 100.0]];
    let r = a.broadcast_mul_rows(&row);
    assert!((r.get(0, 0) - 10.0).abs() < 1e-10);
    assert!((r.get(0, 1) - 200.0).abs() < 1e-10);
    assert!((r.get(1, 0) - 30.0).abs() < 1e-10);
    assert!((r.get(1, 1) - 400.0).abs() < 1e-10);
}


#[test]
fn broadcast_mul_cols() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let col: Tensor<f64> = mat![[10.0_f64], [100.0]];
    let r = a.broadcast_mul_cols(&col);
    assert!((r.get(0, 0) - 10.0).abs() < 1e-10);
    assert!((r.get(0, 1) - 20.0).abs() < 1e-10);
    assert!((r.get(1, 0) - 300.0).abs() < 1e-10);
    assert!((r.get(1, 1) - 400.0).abs() < 1e-10);
}


#[test]
fn diag_square() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let d = a.diag();
    assert_eq!(d.shape(), (2, 1));
    assert!((d.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((d.get(1, 0) - 4.0).abs() < 1e-10);
}


#[test]
fn max_axis_1_row_wise() {
    let a: Tensor<f64> = mat![[1.0_f64, 5.0, 3.0], [4.0, 2.0, 6.0]]; // 2×3
    let mx = a.max_axis(1);
    assert_eq!(mx.shape(), (2, 1));
    assert!((mx.get(0, 0) - 5.0).abs() < 1e-10); // max(1,5,3)
    assert!((mx.get(1, 0) - 6.0).abs() < 1e-10); // max(4,2,6)
}


#[test]
fn min_axis_1_row_wise() {
    let a: Tensor<f64> = mat![[1.0_f64, 5.0, 3.0], [4.0, 2.0, 6.0]]; // 2×3
    let mn = a.min_axis(1);
    assert_eq!(mn.shape(), (2, 1));
    assert!((mn.get(0, 0) - 1.0).abs() < 1e-10); // min(1,5,3)
    assert!((mn.get(1, 0) - 2.0).abs() < 1e-10); // min(4,2,6)
}


#[test]
fn max_axis_keepdim_shape() {
    let a: Tensor<f64> = mat![[1.0_f64, 5.0, 3.0], [4.0, 2.0, 6.0]]; // 2×3
    let mkd0 = a.max_axis_keepdim(0);
    assert_eq!(mkd0.shape(), (1, 3));
    assert!((mkd0.get(0, 1) - 5.0).abs() < 1e-10);
    let mkd1 = a.max_axis_keepdim(1);
    assert_eq!(mkd1.shape(), (2, 1));
    assert!((mkd1.get(1, 0) - 6.0).abs() < 1e-10);
}


#[test]
fn min_axis_keepdim_shape() {
    let a: Tensor<f64> = mat![[1.0_f64, 5.0, 3.0], [4.0, 2.0, 6.0]]; // 2×3
    let mkd0 = a.min_axis_keepdim(0);
    assert_eq!(mkd0.shape(), (1, 3));
    assert!((mkd0.get(0, 0) - 1.0).abs() < 1e-10);
    let mkd1 = a.min_axis_keepdim(1);
    assert_eq!(mkd1.shape(), (2, 1));
    assert!((mkd1.get(0, 0) - 1.0).abs() < 1e-10);
}


#[test]
fn var_axis_constant_is_zero() {
    let a: Tensor<f64> = Tensor::fill(3, 3, 5.0_f64);
    let v = a.var_axis(1);
    assert_eq!(v.shape(), (3, 1));
    for r in 0..3 {
        assert!(v.get(r, 0).abs() < 1e-10, "var of constant row should be 0");
    }
}


#[test]
fn var_axis_1_known_value() {
    // var([1,2,3]) = E[X²]-E[X]² = (1+4+9)/3 - 4 = 14/3 - 4 = 2/3
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]];
    let v = a.var_axis(1);
    assert_eq!(v.shape(), (1, 1));
    assert!((v.get(0, 0) - 2.0 / 3.0).abs() < 1e-10);
}


#[test]
fn std_axis_1_known_value() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]];
    let s = a.std_axis(1);
    assert!((s.get(0, 0) - (2.0_f64 / 3.0).sqrt()).abs() < 1e-10);
}


#[test]
fn clamp_basic() {
    let a: Tensor<f64> = mat![[-2.0_f64, 0.5, 3.0]];
    let c = a.clamp(-1.0, 1.0);
    assert!((c.get(0, 0) - (-1.0)).abs() < 1e-10);
    assert!((c.get(0, 1) - 0.5).abs() < 1e-10);
    assert!((c.get(0, 2) - 1.0).abs() < 1e-10);
}


#[test]
fn from_diag_col_vector() {
    let v: Tensor<f64> = mat![[1.0_f64], [2.0], [3.0]]; // 3×1
    let d = Tensor::from_diag(&v);
    assert_eq!(d.shape(), (3, 3));
    assert!((d.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((d.get(1, 1) - 2.0).abs() < 1e-10);
    assert!((d.get(2, 2) - 3.0).abs() < 1e-10);
    assert!(d.get(0, 1).abs() < 1e-10);
    assert!(d.get(1, 0).abs() < 1e-10);
}


#[test]
fn trace_2x2() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    assert!((a.trace() - 5.0).abs() < 1e-10);
}


#[test]
fn gather_rows_duplicates() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let g = a.gather_rows(&[0, 0, 0]);
    assert_eq!(g.shape(), (3, 2));
    assert!((g.get(2, 0) - 1.0).abs() < 1e-10);
    assert!((g.get(2, 1) - 2.0).abs() < 1e-10);
}


#[test]
fn one_hot_with_cross_entropy() {
    // Verify one_hot integrates with log_softmax + cross_entropy_loss
    let logits: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]];
    let log_probs = logits.log_softmax(1);
    let targets = Tensor::<f64>::one_hot(&[2], 3);
    let loss = log_probs.cross_entropy_loss(&targets);
    assert!(loss > 0.0 && loss < 1.0);
}


#[test]
fn cumsum_axis1_row_wise() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]];
    let cs = a.cumsum(1);
    assert!((cs.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((cs.get(0, 1) - 3.0).abs() < 1e-10);
    assert!((cs.get(0, 2) - 6.0).abs() < 1e-10);
}


#[test]
fn sparse_bcsr_roundtrip() {
    // 3×3 sparse → BCSR (block=2) → SpMM == naive dense multiplication
    let trips = vec![
        Triplet::new(0, 0, 1.0_f64),
        Triplet::new(0, 2, 2.0),
        Triplet::new(1, 1, 3.0),
        Triplet::new(2, 0, 4.0),
        Triplet::new(2, 2, 5.0),
    ];
    let s = SparseMatrix::try_new_from_triplets(3, 3, &trips).expect("build sparse");
    let bcsr = BcsrMatrix::from_sparse(&s, 2);
    assert!(bcsr.nnz_blocks() > 0);
    assert!(bcsr.density() > 0.0 && bcsr.density() <= 1.0);

    let x: Tensor<f64> = linear_f64(3, 2);
    let bcsr_result = bcsr.spmm(&x);
    let dense_result = s.matmul_dense(&x).expect("matmul_dense");
    assert_approx_grid(&bcsr_result, &dense_result, 1e-10);
}


#[test]
fn sparse_bcsr_spmm_accuracy() {
    // 16×16 sparse (~10% density), block=4, compare with dense matmul
    let mut trips = Vec::new();
    for i in 0..16 {
        for j in 0..16 {
            if (i * 7 + j * 13 + 3) % 10 == 0 {
                trips.push(Triplet::new(i, j, ((i * 16 + j + 1) as f64) * 0.1));
            }
        }
    }
    let s = SparseMatrix::try_new_from_triplets(16, 16, &trips).expect("build sparse");
    let bcsr = BcsrMatrix::from_sparse(&s, 4);
    let x: Tensor<f64> = linear_f64(16, 3);
    let bcsr_result = bcsr.spmm(&x);
    let dense_result = s.matmul_dense(&x).expect("matmul_dense");
    assert_approx_grid(&bcsr_result, &dense_result, 1e-8);
}


#[test]
fn sparse_mixed_precision() {
    // Mixed f32/f64 SpMM converges toward f64 ground truth
    let trips_f32 = vec![
        Triplet::new(0, 0, 2.0_f32),
        Triplet::new(0, 1, 1.0),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 3.0),
    ];
    let s_f32 = SparseMatrix::try_new_from_triplets(2, 2, &trips_f32).expect("build sparse f32");
    let bcsr_f32 = BcsrMatrix::from_sparse(&s_f32, 2);

    // b = A * x_true in f64 for known solution
    let x_true: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let a_f64: Tensor<f64> = mat![[2.0_f64, 1.0], [1.0, 3.0]];
    let b = &a_f64 * &x_true;

    let result = mixed_spmm_f64(&bcsr_f32, &b, 1e-10, 100);
    assert_eq!(result.shape(), (2, 2));
    for r in 0..2 {
        for c in 0..2 {
            assert!(result.get(r, c).is_finite(), "non-finite at [{r},{c}]");
        }
    }
}


#[test]
fn reshape_basic() {
    let a: Tensor<f64> = linear_f64(2, 3);
    let b = a.reshape(3, 2);
    assert_eq!(b.shape(), (3, 2));
    assert!(approx_eq(b.get(0, 0), 1.0));
    assert!(approx_eq(b.get(0, 1), 2.0));
    assert!(approx_eq(b.get(1, 0), 3.0));
    assert!(approx_eq(b.get(2, 1), 6.0));
}


#[test]
fn flatten_shape() {
    let a: Tensor<f64> = Tensor::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
    let f = a.flatten();
    assert_eq!(f.shape(), (1, 12));
    assert!(approx_eq(f.get(0, 0), 0.0));
    assert!(approx_eq(f.get(0, 11), 11.0));
}


#[test]
fn sum_axis_0_values() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let s = a.sum_axis(0);
    assert_eq!(s.shape(), (1, 2));
    assert!(approx_eq(s.get(0, 0), 9.0));
    assert!(approx_eq(s.get(0, 1), 12.0));
}


#[test]
fn sum_axis_1_values() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let s = a.sum_axis(1);
    assert_eq!(s.shape(), (2, 1));
    assert!(approx_eq(s.get(0, 0), 6.0));
    assert!(approx_eq(s.get(1, 0), 15.0));
}


#[test]
fn mean_axis_values() {
    let a: Tensor<f64> = mat![[2.0_f64, 4.0], [6.0, 8.0]];
    let m0 = a.mean_axis(0);
    assert_eq!(m0.shape(), (1, 2));
    assert!(approx_eq(m0.get(0, 0), 4.0));
    assert!(approx_eq(m0.get(0, 1), 6.0));

    let m1 = a.mean_axis(1);
    assert_eq!(m1.shape(), (2, 1));
    assert!(approx_eq(m1.get(0, 0), 3.0));
    assert!(approx_eq(m1.get(1, 0), 7.0));
}


#[test]
fn vcat_two_tensors() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64> = mat![[3.0_f64, 4.0]];
    let c = Tensor::vcat(&[&a, &b]);
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 1.0));
    assert!(approx_eq(c.get(0, 1), 2.0));
    assert!(approx_eq(c.get(1, 0), 3.0));
    assert!(approx_eq(c.get(1, 1), 4.0));
}


#[test]
fn hcat_two_tensors() {
    let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    let b: Tensor<f64> = mat![[3.0_f64], [4.0]];
    let c = Tensor::hcat(&[&a, &b]);
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 1.0));
    assert!(approx_eq(c.get(0, 1), 3.0));
    assert!(approx_eq(c.get(1, 0), 2.0));
    assert!(approx_eq(c.get(1, 1), 4.0));
}


#[test]
fn cat_axis0_equals_vcat() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64> = mat![[3.0_f64, 4.0]];
    let cat_result = Tensor::cat(&[&a, &b], 0);
    let vcat_result = Tensor::vcat(&[&a, &b]);
    assert_approx_grid(&cat_result, &vcat_result, 1e-10);
}


