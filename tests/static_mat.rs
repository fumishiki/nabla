// tests/static_mat.rs — Integration tests for StaticMatrix and hierarchy traits.

use nabla::prelude::*;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

// ── StaticMatrix construction ──────────────────────────────────────────────

#[test]
fn static_zeros() {
    let z = StaticMatrix::<f64, 2, 3>::zeros();
    assert_eq!(z.shape(), (2, 3));
    for r in 0..2 {
        for c in 0..3 {
            assert!(approx_eq(z.get(r, c), 0.0));
        }
    }
}

#[test]
fn static_identity() {
    let eye = StaticMatrix::<f64, 3, 3>::identity();
    assert_eq!(eye.shape(), (3, 3));
    for r in 0..3 {
        for c in 0..3 {
            assert!(approx_eq(eye.get(r, c), if r == c { 1.0 } else { 0.0 }));
        }
    }
}

#[test]
fn static_from_fn() {
    let a = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    // a = [[0,1,2],[3,4,5]]
    assert!(approx_eq(a.get(0, 0), 0.0));
    assert!(approx_eq(a.get(0, 2), 2.0));
    assert!(approx_eq(a.get(1, 0), 3.0));
    assert!(approx_eq(a.get(1, 2), 5.0));
}

// ── Arithmetic ────────────────────────────────────────────────────────────

#[test]
fn static_add() {
    let a = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| (r * 2 + c) as f64);
    let b = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| (r * 2 + c + 1) as f64);
    let c = a + b;
    // a=[[0,1],[2,3]], b=[[1,2],[3,4]], c=[[1,3],[5,7]]
    assert!(approx_eq(c.get(0, 0), 1.0));
    assert!(approx_eq(c.get(1, 1), 7.0));
}

#[test]
fn static_sub() {
    let a = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| (r * 2 + c + 1) as f64);
    let b = StaticMatrix::<f64, 2, 2>::identity();
    let c = a - b;
    // a=[[1,2],[3,4]], b=I, c=[[0,2],[3,3]]
    assert!(approx_eq(c.get(0, 0), 0.0));
    assert!(approx_eq(c.get(0, 1), 2.0));
}

#[test]
fn static_neg() {
    let a = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| (r * 2 + c + 1) as f64);
    let b = -a;
    assert!(approx_eq(b.get(0, 0), -1.0));
    assert!(approx_eq(b.get(1, 1), -4.0));
}

#[test]
fn static_scalar_mul() {
    let a = StaticMatrix::<f64, 2, 2>::identity();
    let b = a * 3.0_f64;
    assert!(approx_eq(b.get(0, 0), 3.0));
    assert!(approx_eq(b.get(0, 1), 0.0));
}

#[test]
fn static_matmul_2x2() {
    // a = [[1,2],[3,4]], b = [[5,6],[7,8]]
    // a*b = [[19,22],[43,50]]
    let a = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| [[1.0_f64, 2.0], [3.0, 4.0]][r][c]);
    let b = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| [[5.0_f64, 6.0], [7.0, 8.0]][r][c]);
    let c = a * b;
    assert!(approx_eq(c.get(0, 0), 19.0));
    assert!(approx_eq(c.get(0, 1), 22.0));
    assert!(approx_eq(c.get(1, 0), 43.0));
    assert!(approx_eq(c.get(1, 1), 50.0));
}

#[test]
fn static_matmul_non_square() {
    // a: 2×3, b: 3×2 → c: 2×2
    let a = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c + 1) as f64);
    // a = [[1,2,3],[4,5,6]]
    let b = StaticMatrix::<f64, 3, 2>::from_fn(|r, c| (r * 2 + c + 1) as f64);
    // b = [[1,2],[3,4],[5,6]]
    let c = a.matmul(&b);
    assert!(approx_eq(c.get(0, 0), 22.0));
    assert!(approx_eq(c.get(0, 1), 28.0));
    assert!(approx_eq(c.get(1, 0), 49.0));
    assert!(approx_eq(c.get(1, 1), 64.0));
}

// ── Transpose / Adjoint ────────────────────────────────────────────────────

#[test]
fn static_transpose() {
    let a = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    // a = [[0,1,2],[3,4,5]]
    let at = a.t();
    assert_eq!(at.shape(), (3, 2));
    for r in 0..2 {
        for c in 0..3 {
            assert!(approx_eq(at.get(c, r), a.get(r, c)));
        }
    }
}

#[test]
fn static_adjoint_real_equals_transpose() {
    let a = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    let at = a.t();
    let ah = a.adjoint();
    assert_eq!(ah.shape(), at.shape());
    for r in 0..3 {
        for c in 0..2 {
            assert!(approx_eq(ah.get(r, c), at.get(r, c)));
        }
    }
}

// ── Tensor conversion ─────────────────────────────────────────────────────

#[test]
fn static_to_tensor_roundtrip() {
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

// ── Set ───────────────────────────────────────────────────────────────────

#[test]
fn static_set_get() {
    let mut a = StaticMatrix::<f64, 2, 2>::zeros();
    a.set(0, 1, 42.0);
    assert!(approx_eq(a.get(0, 1), 42.0));
    assert!(approx_eq(a.get(0, 0), 0.0));
}

// ── Array / Matrix trait dispatch ─────────────────────────────────────────

#[test]
fn hierarchy_array_trait_tensor() {
    use nabla::tensor::Array;
    let t: Tensor<f64> = Tensor::from_fn(2, 3, |r, c| (r * 3 + c) as f64);
    let arr: &dyn Array<f64> = &t;
    assert_eq!(arr.shape(), (2, 3));
    assert!(approx_eq(arr.get(1, 2), 5.0));
}

#[test]
fn hierarchy_array_trait_static() {
    use nabla::tensor::Array;
    let s = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    let arr: &dyn Array<f64> = &s;
    assert_eq!(arr.shape(), (2, 3));
    assert!(approx_eq(arr.get(1, 2), 5.0));
}

#[test]
fn hierarchy_t_dyn_tensor() {
    let t: Tensor<f64> = Tensor::from_fn(2, 3, |r, c| (r * 3 + c) as f64);
    let transposed = t.t_dyn();
    assert_eq!(transposed.shape(), (3, 2));
    assert!(approx_eq(transposed.get(2, 1), 5.0)); // t[1,2] = 5
}

#[test]
fn hierarchy_t_dyn_static() {
    let s = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    let transposed = s.t_dyn();
    assert_eq!(transposed.shape(), (3, 2));
    assert!(approx_eq(transposed.get(2, 1), 5.0));
}

#[test]
fn hierarchy_matmul_dyn_heterogeneous() {
    // Tensor × StaticMatrix via trait objects
    use nabla::tensor::Array;
    let t: Tensor<f64> = Tensor::from_fn(2, 2, |r, c| [[1.0_f64, 2.0], [3.0, 4.0]][r][c]);
    let s = StaticMatrix::<f64, 2, 2>::from_fn(|r, c| [[5.0_f64, 6.0], [7.0, 8.0]][r][c]);
    let rhs: &dyn Array<f64> = &s;
    let c = t.matmul_dyn(rhs);
    assert!(approx_eq(c.get(0, 0), 19.0));
    assert!(approx_eq(c.get(1, 1), 50.0));
}
