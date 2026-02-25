#![cfg(feature = "cpu")]

use nabla::prelude::*;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

#[test]
fn reshape_basic() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
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
#[should_panic]
fn vcat_col_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 3);
    let b: Tensor<f64> = Tensor::zeros(2, 4);
    let _ = Tensor::vcat(&[&a, &b]);
}

#[test]
#[should_panic]
fn reshape_size_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 3);
    let _ = a.reshape(2, 2);
}

#[test]
fn cat_axis0_equals_vcat() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64> = mat![[3.0_f64, 4.0]];
    let cat_result = Tensor::cat(&[&a, &b], 0);
    let vcat_result = Tensor::vcat(&[&a, &b]);
    assert_eq!(cat_result.shape(), vcat_result.shape());
    for r in 0..cat_result.nrows() {
        for c in 0..cat_result.ncols() {
            assert!(approx_eq(cat_result.get(r, c), vcat_result.get(r, c)));
        }
    }
}

#[test]
fn cat_axis1_equals_hcat() {
    let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    let b: Tensor<f64> = mat![[3.0_f64], [4.0]];
    let cat_result = Tensor::cat(&[&a, &b], 1);
    let hcat_result = Tensor::hcat(&[&a, &b]);
    assert_eq!(cat_result.shape(), hcat_result.shape());
    for r in 0..cat_result.nrows() {
        for c in 0..cat_result.ncols() {
            assert!(approx_eq(cat_result.get(r, c), hcat_result.get(r, c)));
        }
    }
}

#[test]
#[should_panic]
fn cat_invalid_axis_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let _ = Tensor::cat(&[&a], 2);
}

#[test]
fn chunk_axis0_splits_evenly() {
    let a: Tensor<f64> = Tensor::from_fn(4, 3, |i, j| (i * 3 + j) as f64);
    let chunks = a.chunk(2, 0);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape(), (2, 3));
    assert_eq!(chunks[1].shape(), (2, 3));
    assert!(approx_eq(chunks[0].get(0, 0), 0.0));
    assert!(approx_eq(chunks[0].get(1, 2), 5.0));
    assert!(approx_eq(chunks[1].get(0, 0), 6.0));
    assert!(approx_eq(chunks[1].get(1, 2), 11.0));
}

#[test]
fn chunk_axis1_splits_evenly() {
    let a: Tensor<f64> = Tensor::from_fn(2, 6, |i, j| (i * 6 + j) as f64);
    let chunks = a.chunk(3, 1);
    assert_eq!(chunks.len(), 3);
    for ch in &chunks {
        assert_eq!(ch.shape(), (2, 2));
    }
    assert!(approx_eq(chunks[0].get(0, 0), 0.0));
    assert!(approx_eq(chunks[1].get(0, 0), 2.0));
    assert!(approx_eq(chunks[2].get(0, 0), 4.0));
}

#[test]
#[should_panic]
fn chunk_not_divisible_panics() {
    let a: Tensor<f64> = Tensor::zeros(5, 3);
    let _ = a.chunk(2, 0);
}

#[test]
fn squeeze_axis0_on_single_row() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0, 4.0]];
    assert_eq!(a.shape(), (1, 4));
    let s = a.squeeze(0);
    assert_eq!(s.shape(), (1, 4));
    for c in 0..4 {
        assert!(approx_eq(s.get(0, c), a.get(0, c)));
    }
}

#[test]
#[should_panic]
fn squeeze_non_unit_dim_panics() {
    let a: Tensor<f64> = Tensor::zeros(3, 4);
    let _ = a.squeeze(0);
}

#[test]
fn view_same_as_reshape() {
    let a: Tensor<f64> = Tensor::from_fn(2, 6, |i, j| (i * 6 + j + 1) as f64);
    let v = a.view(3, 4);
    let r = a.reshape(3, 4);
    assert_eq!(v.shape(), r.shape());
    for i in 0..3 {
        for j in 0..4 {
            assert!(approx_eq(v.get(i, j), r.get(i, j)));
        }
    }
}

#[test]
fn softmax_denominator_pattern() {
    let x: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let denom = x.sum_axis_keepdim(1);
    assert_eq!(denom.shape(), (2, 1));
    assert!(approx_eq(denom.get(0, 0), 6.0));
    assert!(approx_eq(denom.get(1, 0), 15.0));
}

#[test]
fn stack_creates_new_dim() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let s = Tensor::stack(&[&a, &b], 0);
    assert_eq!(s.ndim(), 3);
    assert_eq!(s.dim(0), 2);
    assert_eq!(s.dim(1), 2);
    assert_eq!(s.dim(2), 2);
    assert!(approx_eq(s.get_nd(&[0, 0, 0]), 1.0));
    assert!(approx_eq(s.get_nd(&[0, 1, 1]), 4.0));
    assert!(approx_eq(s.get_nd(&[1, 0, 0]), 5.0));
    assert!(approx_eq(s.get_nd(&[1, 1, 1]), 8.0));
}

#[test]
fn ndtensor_unsqueeze_then_squeeze_roundtrip() {
    let t = NdTensor::<f64>::from_fn(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
    let expanded = t.unsqueeze(1);
    assert_eq!(expanded.ndim(), 3);
    assert_eq!(expanded.dim(0), 3);
    assert_eq!(expanded.dim(1), 1);
    assert_eq!(expanded.dim(2), 4);
    let collapsed = expanded.squeeze(1);
    assert_eq!(collapsed.ndim(), 2);
    assert_eq!(collapsed.dim(0), 3);
    assert_eq!(collapsed.dim(1), 4);
    for i in 0..3 {
        for j in 0..4 {
            assert!(approx_eq(collapsed.get_nd(&[i, j]), t.get_nd(&[i, j])));
        }
    }
}

#[test]
fn gradient_prep_x_squared() {
    // f(x) = sum(x^2), grad = 2x
    let f = |x: &Variable<f64, Cpu>| x.emul(x).sum_all_var();
    let x: Tensor<f64> = mat![[2.0_f64, 3.0]];
    let prep = gradient_prep(&f, &x);
    let g = gradient(&f, &x, &prep).expect("gradient returned None");
    assert!(approx_eq(g.get(0, 0), 4.0));
    assert!(approx_eq(g.get(0, 1), 6.0));
}

#[test]
fn grad_single_use() {
    let x: Tensor<f64> = mat![[3.0_f64, 4.0]];
    let g = grad(|xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var(), &x)
        .expect("grad returned None");
    assert!(approx_eq(g.get(0, 0), 6.0));
    assert!(approx_eq(g.get(0, 1), 8.0));
}

#[test]
fn einsum_canon_swapped_operands_with_renamed_indices() {
    // c[i,j] = a[i,k]*b[k,j]  vs  c[x,y] = b[z,y]*a[x,z]
    // Both should produce the same GEMM result after canonicalization.
    let a = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = mat![[7.0_f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let expected = &a * &b;

    let c1: Tensor<f64> = einsum!(c1[i, j] = a[i, k] * b[k, j]);
    let c2: Tensor<f64> = einsum!(c2[x, y] = b[z, y] * a[x, z]);

    for r in 0..2 {
        for c in 0..2 {
            assert!(approx_eq(c1.get(r, c), expected.get(r, c)));
            assert!(approx_eq(c2.get(r, c), expected.get(r, c)));
        }
    }
}

#[test]
fn einsum_canon_gemv_swapped() {
    // y[i] = a[i,k]*x[k]  vs  y[p] = x[q]*a[p,q]
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let x = mat![[10.0_f64], [20.0]];
    let expected = &a * &x;

    let y1: Tensor<f64> = einsum!(y1[i] = a[i, k] * x[k]);
    let y2: Tensor<f64> = einsum!(y2[p] = x[q] * a[p, q]);

    for r in 0..3 {
        assert!(approx_eq(y1.get(r, 0), expected.get(r, 0)));
        assert!(approx_eq(y2.get(r, 0), expected.get(r, 0)));
    }
}

#[test]
fn einsum_canon_hadamard_renamed() {
    // h[i,j] = a[i,j]*b[i,j]  vs  h[p,q] = b[p,q]*a[p,q]
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b = mat![[5.0_f64, 6.0], [7.0, 8.0]];

    let h1: Tensor<f64> = einsum!(h1[i, j] = a[i, j] * b[i, j]);
    let h2: Tensor<f64> = einsum!(h2[p, q] = b[p, q] * a[p, q]);

    for r in 0..2 {
        for c in 0..2 {
            assert!(approx_eq(h1.get(r, c), h2.get(r, c)));
            assert!(approx_eq(h1.get(r, c), a.get(r, c) * b.get(r, c)));
        }
    }
}
