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

fn linear_f64(rows: usize, cols: usize) -> Tensor<f64, Cpu> {
    Tensor::from_fn(rows, cols, |i, j| (i * cols + j + 1) as f64)
}

fn assert_approx_grid(got: &Tensor<f64, Cpu>, expected: &Tensor<f64, Cpu>, tol: f64) {
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
fn mat_macro_roundtrip() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    assert_eq!(a.shape(), (2, 2));
    assert!(approx_eq(a.get(0, 0), 1.0));
    assert!(approx_eq(a.get(0, 1), 2.0));
    assert!(approx_eq(a.get(1, 0), 3.0));
    assert!(approx_eq(a.get(1, 1), 4.0));
}

#[test]
fn einsum_gemm() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let c: Tensor<f64, Cpu> = einsum!(c[i, j] = a[i, k] * b[k, j]);
    assert_eq!(c.shape(), (2, 2));
    assert!(approx_eq(c.get(0, 0), 19.0));
    assert!(approx_eq(c.get(0, 1), 22.0));
    assert!(approx_eq(c.get(1, 0), 43.0));
    assert!(approx_eq(c.get(1, 1), 50.0));
}

#[test]
fn einsum_batch_gemm_3d() {
    let a = NdTensor::<f64>::from_fn(&[2, 3, 4], |idx| {
        (idx[0] * 12 + idx[1] * 4 + idx[2] + 1) as f64
    });
    let m = NdTensor::<f64>::from_fn(&[2, 4, 2], |idx| {
        (idx[0] * 8 + idx[1] * 2 + idx[2] + 1) as f64
    });
    let c: NdTensor<f64> = einsum!(c[b, i, j] = a[b, i, k] * m[b, k, j]);
    assert_eq!(c.ndim(), 3);
    assert_eq!(c.dim(0), 2);
    assert_eq!(c.dim(1), 3);
    assert_eq!(c.dim(2), 2);
    for b in 0..2 {
        for i in 0..3 {
            for j in 0..2 {
                let mut expected = 0.0;
                for k in 0..4 {
                    expected += a.get_nd(&[b, i, k]) * m.get_nd(&[b, k, j]);
                }
                assert!(
                    approx_eq(c.get_nd(&[b, i, j]), expected),
                    "mismatch at [{b},{i},{j}]: got {}, expected {expected}",
                    c.get_nd(&[b, i, j])
                );
            }
        }
    }
}

#[test]
fn einsum_nd_fallback() {
    let t = NdTensor::<f64>::from_fn(&[2, 3, 4], |idx| {
        (idx[0] * 12 + idx[1] * 4 + idx[2] + 1) as f64
    });
    let m: Tensor<f64, Cpu> = Tensor::from_fn(4, 5, |k, l| (k * 5 + l + 1) as f64);
    let r: NdTensor<f64> = einsum!(r[i, j, l] = t[i, j, k] * m[k, l]);
    assert_eq!(r.ndim(), 3);
    assert_eq!(r.dim(0), 2);
    assert_eq!(r.dim(1), 3);
    assert_eq!(r.dim(2), 5);
    for i in 0..2 {
        for j in 0..3 {
            for l in 0..5 {
                let mut expected = 0.0;
                for k in 0..4 {
                    expected += t.get_nd(&[i, j, k]) * m.get(k, l);
                }
                assert!(
                    approx_eq(r.get_nd(&[i, j, l]), expected),
                    "mismatch at [{i},{j},{l}]: got {}, expected {expected}",
                    r.get_nd(&[i, j, l])
                );
            }
        }
    }
}

#[test]
fn fuse_pipeline() {
    let x: Tensor<f64, Cpu> = linear_f64(4, 4);
    let y: Tensor<f64, Cpu> = fuse!(x.sin().powf(2.0); x);
    for r in 0..4 {
        for c in 0..4 {
            let v = ((r * 4 + c + 1) as f64).sin().powf(2.0);
            assert!(approx_eq(y.get(r, c), v));
        }
    }
}

#[test]
fn named_tuple() {
    let p = named!(x: f64 = 1.0, y: f64 = 2.0, z: f64 = 3.0);
    assert!(approx_eq(p.x, 1.0));
    assert!(approx_eq(p.y, 2.0));
    assert!(approx_eq(p.z, 3.0));
}

#[test]
fn generated_specialization() {
    generated! {
        fn norm_sq<const N: usize>(data: &[f64; N]) -> f64 {
            match N {
                1 => data[0] * data[0],
                2 => data[0] * data[0] + data[1] * data[1],
                3 => data[0] * data[0] + data[1] * data[1] + data[2] * data[2],
                _ => {
                    let mut s = 0.0;
                    let mut i = 0;
                    while i < N { s += data[i] * data[i]; i += 1; }
                    s
                }
            }
        }
    }
    assert!(approx_eq(norm_sq(&[3.0_f64]), 9.0));
    assert!(approx_eq(norm_sq(&[3.0_f64, 4.0]), 25.0));
    assert!(approx_eq(norm_sq(&[1.0_f64, 2.0, 3.0]), 14.0));
    assert!(approx_eq(norm_sq(&[1.0_f64, 1.0, 1.0, 1.0]), 4.0));
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn map_and_map_() {
    let a: Tensor<f64, Cpu> = linear_f64(2, 2);
    let doubled: Tensor<f64, Cpu> = nabla::map!(|x| x * 2.0, &a);
    assert!(approx_eq(doubled.get(0, 0), 2.0));
    assert!(approx_eq(doubled.get(1, 1), 8.0));

    let b: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |i, j| (i + j) as f64);
    let c_b: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |i, j| (i * j) as f64);
    let sum: Tensor<f64, Cpu> = nabla::map!(|x, y| x + y, &b, &c_b);
    assert!(approx_eq(sum.get(1, 1), 3.0));

    let scale: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |_, _| 2.0);
    let mut out: Tensor<f64, Cpu> = Tensor::zeros(2, 2);
    nabla::map_!(out, |x, y| x * y, &a, &scale);
    assert!(approx_eq(out.get(0, 0), 2.0));
    assert!(approx_eq(out.get(1, 1), 8.0));
}

#[test]
fn splat_tuple() {
    use nabla::splat;
    fn add3(a: f64, b: f64, c: f64) -> f64 {
        a + b + c
    }
    let result = splat!(add3, (1.0_f64, 2.0, 3.0));
    assert!(approx_eq(result, 6.0));

    fn mul2(a: f64, b: f64) -> f64 {
        a * b
    }
    assert!(approx_eq(splat!(mul2, (3.0_f64, 4.0)), 12.0));
}

#[test]
fn einsum_canon_term_order_independent() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let expected = &a * &b;

    let c1: Tensor<f64, Cpu> = einsum!(c1[i, j] = a[i, k] * b[k, j]);
    let c2: Tensor<f64, Cpu> = einsum!(c2[i, j] = b[k, j] * a[i, k]);
    assert_approx_grid(&c1, &expected, 1e-10);
    assert_approx_grid(&c2, &expected, 1e-10);
}

#[test]
fn einsum_canon_index_rename() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let expected = &a * &b;

    let c: Tensor<f64, Cpu> = einsum!(c[x, y] = a[x, z] * b[z, y]);
    assert_approx_grid(&c, &expected, 1e-10);
}

#[test]
fn einsum_three_tensor_chain() {
    // c[i,j] = a[i,k] * b[k,l] * d[l,j] == a @ b @ d
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[1.0_f64, 0.0], [0.0, 1.0]]; // identity
    let d: Tensor<f64, Cpu> = mat![[2.0_f64, 0.0], [0.0, 2.0]]; // 2*I
    let result: Tensor<f64, Cpu> = einsum!(result[i, j] = a[i, k] * b[k, l] * d[l, j]);
    // a @ I @ 2I = 2a
    let expected = &a * 2.0_f64;
    assert_approx_grid(&result, &expected, 1e-10);
}

#[test]
fn einsum_three_tensor_vs_manual() {
    // Verify 3-tensor einsum matches manual matmul chain
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let d: Tensor<f64, Cpu> = mat![[1.0_f64, 0.0], [0.0, 1.0]]; // identity
    let result: Tensor<f64, Cpu> = einsum!(result[i, j] = a[i, k] * b[k, l] * d[l, j]);
    let manual = &(&a * &b) * &d; // a @ b @ I = a @ b
    assert_approx_grid(&result, &manual, 1e-10);
}

#[test]
fn vcat_macro_three_tensors() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64, Cpu> = mat![[3.0, 4.0]];
    let c: Tensor<f64, Cpu> = mat![[5.0, 6.0]];
    let r = nabla::vcat!(a, b, c);
    assert_eq!(r.nrows(), 3);
    assert_eq!(r.ncols(), 2);
    assert!(approx_eq(r.get(0, 0), 1.0));
    assert!(approx_eq(r.get(1, 0), 3.0));
    assert!(approx_eq(r.get(2, 0), 5.0));
}

#[test]
fn hcat_macro_three_tensors() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64], [2.0]];
    let b: Tensor<f64, Cpu> = mat![[3.0], [4.0]];
    let c: Tensor<f64, Cpu> = mat![[5.0], [6.0]];
    let r = nabla::hcat!(a, b, c);
    assert_eq!(r.ncols(), 3);
    assert_eq!(r.nrows(), 2);
    assert!(approx_eq(r.get(0, 0), 1.0));
    assert!(approx_eq(r.get(0, 1), 3.0));
    assert!(approx_eq(r.get(0, 2), 5.0));
}

#[test]
fn vcat_macro_two_tensors() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64, Cpu> = mat![[3.0, 4.0]];
    let r = nabla::vcat!(a, b);
    assert_eq!(r.shape(), (2, 2));
}

#[test]
fn named_axes_zero_cost() {
    struct Batch;
    struct Features;

    let x: Tensor<f64, Cpu, (Batch, Features)> = Tensor::zeros(32, 128).with_axes();
    assert_eq!(x.shape(), (32, 128));
    // Zero-cost: PhantomData adds no runtime overhead.
    assert_eq!(
        std::mem::size_of::<Tensor<f64, Cpu, (Batch, Features)>>(),
        std::mem::size_of::<Tensor<f64, Cpu>>()
    );
}

#[test]
fn named_axes_named_zeros_macro() {
    axis!(Time, Freq);
    let t: Tensor<f64, Cpu, (Time, Freq)> = named_zeros!(Time, Freq; 16, 256);
    assert_eq!(t.shape(), (16, 256));
    assert!(approx_eq(t.get(0, 0), 0.0));
}

#[test]
fn einsum_nd_fallback_tiled_large() {
    // Verify tiled contraction loops produce correct results with dim > tile size (64)
    let t = NdTensor::<f64>::from_fn(&[3, 2, 128], |idx| {
        ((idx[0] * 256 + idx[1] * 128 + idx[2]) as f64) * 0.01
    });
    let m: Tensor<f64, Cpu> = Tensor::from_fn(128, 4, |k, l| ((k * 4 + l) as f64) * 0.01);
    let r: NdTensor<f64> = einsum!(r[i, j, l] = t[i, j, k] * m[k, l]);
    assert_eq!(r.ndim(), 3);
    assert_eq!(r.dim(0), 3);
    assert_eq!(r.dim(1), 2);
    assert_eq!(r.dim(2), 4);
    for i in 0..3 {
        for j in 0..2 {
            for l in 0..4 {
                let mut expected = 0.0;
                for k in 0..128 {
                    expected += t.get_nd(&[i, j, k]) * m.get(k, l);
                }
                assert!(
                    (r.get_nd(&[i, j, l]) - expected).abs() < 1e-6,
                    "mismatch at [{i},{j},{l}]: got {}, expected {expected}",
                    r.get_nd(&[i, j, l])
                );
            }
        }
    }
}

#[test]
fn einsum_three_tensor_non_square() {
    // 3-tensor contraction with non-square matrices exercises greedy path optimizer
    let a: Tensor<f64, Cpu> = linear_f64(3, 4);
    let b: Tensor<f64, Cpu> = linear_f64(4, 5);
    let d: Tensor<f64, Cpu> = linear_f64(5, 2);
    let result: Tensor<f64, Cpu> = einsum!(result[i, j] = a[i, k] * b[k, l] * d[l, j]);
    let manual = &(&a * &b) * &d;
    assert_approx_grid(&result, &manual, 1e-6);
}

#[test]
fn fuse_multi_tensor() {
    let x: Tensor<f64, Cpu> = Tensor::from_fn(3, 3, |r, c| (r + c + 1) as f64);
    let y: Tensor<f64, Cpu> = Tensor::from_fn(3, 3, |r, c| (r * 3 + c + 1) as f64 * 0.1);
    let z: Tensor<f64, Cpu> = fuse!(x * y + x.sin(); x, y);
    for r in 0..3 {
        for c in 0..3 {
            let xv = (r + c + 1) as f64;
            let yv = (r * 3 + c + 1) as f64 * 0.1;
            let expected = xv * yv + xv.sin();
            assert!(approx_eq(z.get(r, c), expected));
        }
    }
}

#[test]
fn fuse_deep_chain() {
    let x: Tensor<f64, Cpu> = Tensor::from_fn(2, 3, |r, c| (r * 3 + c + 1) as f64 * 0.5);
    let y: Tensor<f64, Cpu> = fuse!(x.exp().ln().abs().sqrt(); x);
    for r in 0..2 {
        for c in 0..3 {
            let v = (r * 3 + c + 1) as f64 * 0.5;
            let expected = v.exp().ln().abs().sqrt();
            assert!(approx_eq(y.get(r, c), expected));
        }
    }
}

#[test]
fn fuse_neg_and_arithmetic() {
    let x: Tensor<f64, Cpu> = linear_f64(2, 2);
    let y: Tensor<f64, Cpu> = fuse!(-x + x * 2.0; x);
    for r in 0..2 {
        for c in 0..2 {
            let v = (r * 2 + c + 1) as f64;
            let expected = -v + v * 2.0;
            assert!(approx_eq(y.get(r, c), expected));
        }
    }
}

#[test]
fn fuse_gemm_sigmoid() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[0.5_f64, 0.1], [0.2, 0.3]];
    let fused: Tensor<f64, Cpu> = fuse!((&a * &b).sigmoid(); a, b);
    let expected = (&a * &b).sigmoid();
    assert_approx_grid(&fused, &expected, 1e-12);
}

#[test]
fn fuse_gemm_relu() {
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, -1.0], [2.0, 0.5]];
    let b: Tensor<f64, Cpu> = mat![[0.5_f64, -0.5], [1.0, 0.3]];
    let fused: Tensor<f64, Cpu> = fuse!((&a * &b).relu(); a, b);
    let expected = (&a * &b).relu();
    assert_approx_grid(&fused, &expected, 1e-12);
}

// Parareal parallel-in-time ODE solver

#[test]
fn einsum_canon_swapped_operands_with_renamed_indices() {
    // c[i,j] = a[i,k]*b[k,j]  vs  c[x,y] = b[z,y]*a[x,z]
    // Both should produce the same GEMM result after canonicalization.
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b: Tensor<f64, Cpu> = mat![[7.0_f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let expected = &a * &b;

    let c1: Tensor<f64, Cpu> = einsum!(c1[i, j] = a[i, k] * b[k, j]);
    let c2: Tensor<f64, Cpu> = einsum!(c2[x, y] = b[z, y] * a[x, z]);
    assert_approx_grid(&c1, &expected, 1e-10);
    assert_approx_grid(&c2, &expected, 1e-10);
}

#[test]
fn einsum_canon_gemv_swapped() {
    // y[i] = a[i,k]*x[k]  vs  y[p] = x[q]*a[p,q]
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let x: Tensor<f64, Cpu> = mat![[10.0_f64], [20.0]];
    let expected = &a * &x;

    let y1: Tensor<f64, Cpu> = einsum!(y1[i] = a[i, k] * x[k]);
    let y2: Tensor<f64, Cpu> = einsum!(y2[p] = x[q] * a[p, q]);

    for r in 0..3 {
        assert!(approx_eq(y1.get(r, 0), expected.get(r, 0)));
        assert!(approx_eq(y2.get(r, 0), expected.get(r, 0)));
    }
}

#[test]
fn einsum_canon_hadamard_renamed() {
    // h[i,j] = a[i,j]*b[i,j]  vs  h[p,q] = b[p,q]*a[p,q]
    let a: Tensor<f64, Cpu> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64, Cpu> = mat![[5.0_f64, 6.0], [7.0, 8.0]];

    let h1: Tensor<f64, Cpu> = einsum!(h1[i, j] = a[i, j] * b[i, j]);
    let h2: Tensor<f64, Cpu> = einsum!(h2[p, q] = b[p, q] * a[p, q]);
    assert_approx_grid(&h1, &h2, 1e-10);
    let expected_h = a.emul(&b);
    assert_approx_grid(&h1, &expected_h, 1e-10);
}

// ── New operations tests (§14.1 parity) ──────────────────────────

#[test]
fn einsum_compile_errors() {
    trybuild::TestCases::new().compile_fail("tests/einsum_errors/*.rs");
}
