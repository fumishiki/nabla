// tests/broadcast.rs — Integration tests for bcast! and zip_map! macros.

use nabla::prelude::*;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
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
fn bcast_ternary() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 1.0);
    let b: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    let c: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 3.0);
    let out: Tensor<f64> = nabla::bcast!(|x, y, z| x + y + z, &a, &b, &c);
    assert!(approx_eq(out.get(0, 0), 6.0));
    assert!(approx_eq(out.get(1, 1), 6.0));
}

#[test]
fn zip_map_unary_inplace() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    nabla::zip_map!(out, |x| x * 3.0, &a);
    assert!(approx_eq(out.get(0, 0), 3.0)); // a[0,0]=1 * 3
    assert!(approx_eq(out.get(1, 1), 12.0)); // a[1,1]=4 * 3
}

#[test]
#[should_panic(expected = "bcast! shape mismatch")]
fn bcast_shape_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let b: Tensor<f64> = Tensor::zeros(2, 3);
    let _: Tensor<f64> = nabla::bcast!(|x, y| x + y, &a, &b);
}

#[test]
#[should_panic(expected = "zip_map! shape mismatch")]
fn zip_map_shape_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 3);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    nabla::zip_map!(out, |x| x, &a);
}
