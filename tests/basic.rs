// tests/basic.rs — Integration tests for nabla public API.
//
// Tests exercise the public surface through the prelude only.
// Float comparisons use epsilon-based approx_eq, not exact equality.

use nabla::prelude::*;

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
