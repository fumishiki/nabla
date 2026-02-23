// tests/basic.rs — Integration tests for nabla public API.
// Includes broadcasting macro tests (bcast!, zip_map!).
//
// Tests exercise the public surface through the prelude only.
// Float comparisons use epsilon-based approx_eq, not exact equality.

use nabla::prelude::*;
use nabla::{between, frange};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

#[test]
fn mat_macro_creation_and_shape() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    assert_eq!(a.shape(), (2, 2));
    assert!(approx_eq(a.get(0, 0), 1.0));
    assert!(approx_eq(a.get(0, 1), 2.0));
    assert!(approx_eq(a.get(1, 0), 3.0));
    assert!(approx_eq(a.get(1, 1), 4.0));
}

#[test]
fn zeros_matrix() {
    let z: Tensor<f64> = Tensor::zeros(3, 4);
    assert_eq!(z.shape(), (3, 4));
    for r in 0..3 {
        for c in 0..4 {
            assert!(approx_eq(z.get(r, c), 0.0));
        }
    }
}

#[test]
fn identity_matrix() {
    let eye: Tensor<f64> = Tensor::identity(3);
    assert_eq!(eye.shape(), (3, 3));
    for r in 0..3 {
        for c in 0..3 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!(approx_eq(eye.get(r, c), expected));
        }
    }
}

#[test]
fn elementwise_add() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j) as f64);
    let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    let c = &a + &b;
    assert_eq!(c.shape(), (2, 2));
    // a = [[0,1],[2,3]], b = [[1,2],[3,4]], c = [[1,3],[5,7]]
    assert!(approx_eq(c.get(0, 0), 1.0));
    assert!(approx_eq(c.get(0, 1), 3.0));
    assert!(approx_eq(c.get(1, 0), 5.0));
    assert!(approx_eq(c.get(1, 1), 7.0));
}

#[test]
fn matmul_2x2() {
    // a = [[1,2],[3,4]], b = [[5,6],[7,8]]
    // a*b = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
    //     = [[19, 22],[43, 50]]
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
    let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
    let c = &a * &b;
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 19.0));
    assert!(approx_eq(c.get(0, 1), 22.0));
    assert!(approx_eq(c.get(1, 0), 43.0));
    assert!(approx_eq(c.get(1, 1), 50.0));
}

#[test]
fn matmul_non_square() {
    // a is 2x3, b is 3x2 → c is 2x2
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
    // a = [[1,2,3],[4,5,6]]
    let b: Tensor<f64> = Tensor::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
    // b = [[1,2],[3,4],[5,6]]
    // c[0][0] = 1*1+2*3+3*5=22, c[0][1]=1*2+2*4+3*6=28
    // c[1][0] = 4*1+5*3+6*5=49, c[1][1]=4*2+5*4+6*6=64
    let c = &a * &b;
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 22.0));
    assert!(approx_eq(c.get(0, 1), 28.0));
    assert!(approx_eq(c.get(1, 0), 49.0));
    assert!(approx_eq(c.get(1, 1), 64.0));
}

#[test]
fn scalar_multiply() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    // a = [[1,2],[3,4]]
    let b = &a * 2.0_f64;
    assert_eq!(b.shape(), (2, 2));
    assert!(approx_eq(b.get(0, 0), 2.0));
    assert!(approx_eq(b.get(0, 1), 4.0));
    assert!(approx_eq(b.get(1, 0), 6.0));
    assert!(approx_eq(b.get(1, 1), 8.0));
}

#[test]
fn transpose_swaps_shape_and_elements() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
    // a = [[0,1,2],[3,4,5]]
    let at = a.t();
    assert_eq!(at.shape(), (3, 2));
    // at[j][i] == a[i][j]
    for i in 0..2 {
        for j in 0..3 {
            assert!(approx_eq(at.get(j, i), a.get(i, j)));
        }
    }
}

#[test]
fn adjoint_real_equals_transpose() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
    let at = a.t();
    let ah = a.adjoint();
    assert_eq!(ah.shape(), at.shape());
    for r in 0..3 {
        for c in 0..2 {
            assert!(approx_eq(ah.get(r, c), at.get(r, c)));
        }
    }
}

#[test]
fn matmul_into_correctness() {
    // Same 2x2 case as matmul_2x2 but via zero-alloc path.
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
    let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    Tensor::matmul_into(&mut out, &a, &b);
    assert!(approx_eq(out.get(0, 0), 19.0));
    assert!(approx_eq(out.get(0, 1), 22.0));
    assert!(approx_eq(out.get(1, 0), 43.0));
    assert!(approx_eq(out.get(1, 1), 50.0));
}

#[test]
#[should_panic]
fn add_shape_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let b: Tensor<f64> = Tensor::zeros(2, 3);
    let _ = &a + &b;
}

#[test]
#[should_panic]
fn matmul_inner_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 3);
    let b: Tensor<f64> = Tensor::zeros(2, 2); // inner dims: 3 vs 2
    let _ = &a * &b;
}

#[test]
fn between_macro() {
    // Use runtime values to avoid compile-time constant folding.
    let mid = 0.5_f64;
    let hi = 1.0_f64;
    let neg = -0.1_f64;
    assert!(between!(0.0_f64, mid, hi));
    assert!(!between!(0.0_f64, hi, hi)); // exclusive upper
    assert!(!between!(0.0_f64, neg, hi));
}

#[test]
fn complex_helpers() {
    let z = nabla::util::c64(1.0, 2.0);
    assert!((z.re - 1.0_f64).abs() < 1e-10);
    assert!((z.im - 2.0_f64).abs() < 1e-10);
    let z32 = nabla::util::c32(3.0, 4.0);
    assert!((z32.re - 3.0_f32).abs() < 1e-6);
    assert!((z32.im - 4.0_f32).abs() < 1e-6);
}

#[test]
fn linspace_basic() {
    let v = linspace(0.0, 1.0, 5);
    assert_eq!(v.len(), 5);
    assert!((v[0] - 0.0).abs() < 1e-10);
    assert!((v[4] - 1.0).abs() < 1e-10);
    assert!((v[2] - 0.5).abs() < 1e-10);
}

#[test]
fn linspace_edge_cases() {
    assert_eq!(linspace(0.0, 1.0, 0).len(), 0);
    let one = linspace(0.0, 1.0, 1);
    assert_eq!(one.len(), 1);
    assert!((one[0] - 0.0).abs() < 1e-10);
}

#[test]
fn frange_basic() {
    let v = frange!(0.0_f64, 0.25, 1.0);
    assert_eq!(v.len(), 5); // 0.0, 0.25, 0.5, 0.75, 1.0
}

#[test]
fn set_and_get() {
    let mut a: Tensor<f64> = Tensor::zeros(2, 2);
    a.set(0, 0, 5.0);
    a.set(1, 1, 3.0);
    assert!(approx_eq(a.get(0, 0), 5.0));
    assert!(approx_eq(a.get(1, 1), 3.0));
    assert!(approx_eq(a.get(0, 1), 0.0));
}

#[test]
fn slice_submatrix() {
    let a: Tensor<f64> = Tensor::from_fn(4, 4, |i, j| (i * 4 + j) as f64);
    let s = a.slice(1..3, 1..3);
    assert_eq!(s.shape(), (2, 2));
    assert!(approx_eq(s.get(0, 0), 5.0)); // a[1][1]
    assert!(approx_eq(s.get(1, 1), 10.0)); // a[2][2]
}

#[test]
fn slice_rows_cols() {
    let a: Tensor<f64> = Tensor::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
    let r = a.slice_rows(..2);
    assert_eq!(r.shape(), (2, 4));
    let c = a.slice_cols(2..);
    assert_eq!(c.shape(), (3, 2));
}

#[test]
fn einsum_matmul() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let c: Tensor<f64> = einsum!(c[i, j] = a[i, k] * b[k, j]);
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 19.0));
    assert!(approx_eq(c.get(0, 1), 22.0));
    assert!(approx_eq(c.get(1, 0), 43.0));
    assert!(approx_eq(c.get(1, 1), 50.0));
}

#[test]
fn einsum_hadamard() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[2.0_f64, 3.0], [4.0, 5.0]];
    let c: Tensor<f64> = einsum!(c[i, j] = a[i, j] * b[i, j]);
    assert!(approx_eq(c.get(0, 0), 2.0));
    assert!(approx_eq(c.get(1, 1), 20.0));
}

#[test]
fn einsum_trace() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let s: f64 = einsum!(s = a[i, i]);
    assert!(approx_eq(s, 5.0)); // 1 + 4
}

#[test]
fn diagonal_to_tensor_and_mul() {
    let d = Diagonal::new(vec![2.0_f64, 3.0]);
    let t = d.to_tensor();
    assert_eq!(t.shape(), (2, 2));
    assert!(approx_eq(t.get(0, 0), 2.0));
    assert!(approx_eq(t.get(1, 1), 3.0));
    assert!(approx_eq(t.get(0, 1), 0.0));

    let m: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
    let r = d.mul_dense(&m).unwrap();
    // D*M: row 0 scaled by 2, row 1 scaled by 3
    assert!(approx_eq(r.get(0, 0), 2.0));
    assert!(approx_eq(r.get(0, 1), 4.0));
    assert!(approx_eq(r.get(1, 0), 9.0));
    assert!(approx_eq(r.get(1, 1), 12.0));
}

#[test]
fn symmetric_eigen() {
    // 2x2 symmetric positive definite matrix
    let a: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |i, j| [[4.0_f64, 2.0], [2.0, 3.0]][i][j]);
    let sym = Symmetric::new(a, faer::Side::Lower).unwrap();
    let evals = sym.eigenvalues().unwrap();
    assert_eq!(evals.len(), 2);
    // Both eigenvalues must be positive (SPD matrix)
    assert!(evals[0] > 0.0);
    assert!(evals[1] > 0.0);
}

#[test]
fn bcast_unary() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    let doubled: Tensor<f64> = nabla::bcast!(|x| x * 2.0, &a);
    assert!(approx_eq(doubled.get(0, 0), 2.0));
    assert!(approx_eq(doubled.get(1, 1), 8.0));
}

#[test]
fn bcast_binary() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j) as f64);
    let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * j) as f64);
    let c: Tensor<f64> = nabla::bcast!(|x, y| x + y, &a, &b);
    assert!(approx_eq(c.get(1, 1), 3.0)); // (1+1) + (1*1) = 3
}

#[test]
fn zip_map_inplace() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    nabla::zip_map!(out, |x, y| x * y, &a, &b);
    assert!(approx_eq(out.get(0, 0), 2.0));
    assert!(approx_eq(out.get(1, 1), 8.0));
}

#[test]
fn tensor_1x1_ops() {
    let a: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 5.0);
    let b: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 3.0);
    let sum = &a + &b;
    assert!(approx_eq(sum.get(0, 0), 8.0));
    let prod = &a * &b;
    assert!(approx_eq(prod.get(0, 0), 15.0));
    let at = a.t();
    assert_eq!(at.shape(), (1, 1));
    assert!(approx_eq(at.get(0, 0), 5.0));
    let neg = -&a;
    assert!(approx_eq(neg.get(0, 0), -5.0));
    let scaled = &a * 4.0_f64;
    assert!(approx_eq(scaled.get(0, 0), 20.0));
}

#[test]
fn clone_independence() {
    let mut a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j) as f64);
    let b = a.clone();
    a.set(0, 0, 99.0);
    assert!(approx_eq(b.get(0, 0), 0.0)); // clone unaffected
    assert!(approx_eq(a.get(0, 0), 99.0));
}

#[test]
fn identity_mul_properties() {
    let a: Tensor<f64> = Tensor::from_fn(3, 3, |i, j| (i * 3 + j + 1) as f64);
    let eye: Tensor<f64> = Tensor::identity(3);
    let ia = &eye * &a;
    let ai = &a * &eye;
    for r in 0..3 {
        for c in 0..3 {
            assert!(approx_eq(ia.get(r, c), a.get(r, c)));
            assert!(approx_eq(ai.get(r, c), a.get(r, c)));
        }
    }
}

#[test]
fn neg_double_negation() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
    let neg_neg_a = -&(-&a);
    for r in 0..2 {
        for c in 0..3 {
            assert!(approx_eq(neg_neg_a.get(r, c), a.get(r, c)));
        }
    }
}

#[test]
fn scalar_mul_zero_and_one() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
    let zeroed = &a * 0.0_f64;
    let ones = &a * 1.0_f64;
    for r in 0..2 {
        for c in 0..3 {
            assert!(approx_eq(zeroed.get(r, c), 0.0));
            assert!(approx_eq(ones.get(r, c), a.get(r, c)));
        }
    }
}

#[test]
fn slice_full_range() {
    let a: Tensor<f64> = Tensor::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
    let (nrows, ncols) = a.shape();
    let s = a.slice(0..nrows, 0..ncols);
    assert_eq!(s.shape(), (nrows, ncols));
    for r in 0..nrows {
        for c in 0..ncols {
            assert!(approx_eq(s.get(r, c), a.get(r, c)));
        }
    }
}

#[test]
fn tall_wide_matmul() {
    // 3x1 * 1x3 → 3x3
    let col: Tensor<f64> = Tensor::from_fn(3, 1, |i, _| (i + 1) as f64); // [1,2,3]^T
    let row: Tensor<f64> = Tensor::from_fn(1, 3, |_, j| (j + 1) as f64); // [1,2,3]
    let outer = &col * &row;
    assert_eq!(outer.shape(), (3, 3));
    for r in 0..3 {
        for c in 0..3 {
            assert!(approx_eq(outer.get(r, c), ((r + 1) * (c + 1)) as f64));
        }
    }
    // 1x3 * 3x1 → 1x1 (dot product = 1+4+9 = 14)
    let dot = &row * &col;
    assert_eq!(dot.shape(), (1, 1));
    assert!(approx_eq(dot.get(0, 0), 14.0));
}

#[test]
fn tensor_0x0_ops() {
    let z: Tensor<f64> = Tensor::zeros(0, 0);
    assert_eq!(z.shape(), (0, 0));
    let zt = z.t();
    assert_eq!(zt.shape(), (0, 0));
    let za = z.adjoint();
    assert_eq!(za.shape(), (0, 0));
    let neg = -&z;
    assert_eq!(neg.shape(), (0, 0));
    let scaled = &z * 2.0_f64;
    assert_eq!(scaled.shape(), (0, 0));
    let sum = &z + &z;
    assert_eq!(sum.shape(), (0, 0));
    // 0x0 * 0x0 → 0x0
    let prod = &z * &z;
    assert_eq!(prod.shape(), (0, 0));
}

#[test]
fn tensor_1x1_adjoint() {
    let a: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 7.0);
    let ah = a.adjoint();
    assert_eq!(ah.shape(), (1, 1));
    assert!(approx_eq(ah.get(0, 0), 7.0));
}

#[test]
fn slice_empty_range() {
    let a: Tensor<f64> = Tensor::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
    let s = a.slice(1..1, 0..3); // 0 rows
    assert_eq!(s.shape(), (0, 3));
    let s2 = a.slice(0..3, 2..2); // 0 cols
    assert_eq!(s2.shape(), (3, 0));
}

// ── Wave 6: reduction tests ───────────────────────────────────────────────────

#[test]
fn sum_all_basic() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
    // [[1,2,3],[4,5,6]] → sum = 21
    assert!(approx_eq(a.sum_all(), 21.0));
}

#[test]
fn sum_all_empty() {
    let z: Tensor<f64> = Tensor::zeros(0, 0);
    assert!(approx_eq(z.sum_all(), 0.0));
}

#[test]
fn max_all_basic() {
    let a: Tensor<f64> = mat![[3.0_f64, 1.0], [5.0, 2.0]];
    assert!(approx_eq(a.max_all(), 5.0));
}

#[test]
fn min_all_basic() {
    let a: Tensor<f64> = mat![[3.0_f64, 1.0], [5.0, 2.0]];
    assert!(approx_eq(a.min_all(), 1.0));
}

#[test]
fn argmax_basic() {
    // max is 5.0 at (1,0)
    let a: Tensor<f64> = mat![[3.0_f64, 1.0], [5.0, 2.0]];
    assert_eq!(a.argmax(), (1, 0));
}

#[test]
fn argmin_basic() {
    // min is 1.0 at (0,1)
    let a: Tensor<f64> = mat![[3.0_f64, 1.0], [5.0, 2.0]];
    assert_eq!(a.argmin(), (0, 1));
}

#[test]
fn argmax_first_element() {
    // max is at (0,0)
    let a: Tensor<f32> = Tensor::from_fn(3, 3, |i, j| -(i as f32) - (j as f32));
    assert_eq!(a.argmax(), (0, 0));
}

#[test]
fn argmin_last_element() {
    // a[i,j] = -(i + j), so the minimum (most negative) is at (2,2)
    let a: Tensor<f32> = Tensor::from_fn(3, 3, |i, j| -((i as f32) + (j as f32)));
    assert_eq!(a.argmin(), (2, 2));
}

#[test]
fn sum_all_f32() {
    let a: Tensor<f32> = Tensor::from_fn(1, 4, |_, j| (j + 1) as f32);
    // [1, 2, 3, 4] → sum = 10
    assert!((a.sum_all() - 10.0_f32).abs() < 1e-5);
}
