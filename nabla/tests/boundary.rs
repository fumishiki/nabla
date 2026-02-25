#![cfg(feature = "cpu")]

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
fn mat_macro_roundtrip() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    assert_eq!(a.shape(), (2, 2));
    assert!(approx_eq(a.get(0, 0), 1.0));
    assert!(approx_eq(a.get(0, 1), 2.0));
    assert!(approx_eq(a.get(1, 0), 3.0));
    assert!(approx_eq(a.get(1, 1), 4.0));
}

#[test]
fn einsum_gemm() {
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
    let m: Tensor<f64> = Tensor::from_fn(4, 5, |k, l| (k * 5 + l + 1) as f64);
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
    let x: Tensor<f64> = linear_f64(4, 4);
    let y: Tensor<f64> = fuse!(x.sin().powf(2.0); x);
    for r in 0..4 {
        for c in 0..4 {
            let v = ((r * 4 + c + 1) as f64).sin().powf(2.0);
            assert!(approx_eq(y.get(r, c), v));
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

#[test]
fn map_and_map_() {
    let a: Tensor<f64> = linear_f64(2, 2);
    let doubled: Tensor<f64> = nabla::map!(|x| x * 2.0, &a);
    assert!(approx_eq(doubled.get(0, 0), 2.0));
    assert!(approx_eq(doubled.get(1, 1), 8.0));

    let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j) as f64);
    let c_b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * j) as f64);
    let sum: Tensor<f64> = nabla::map!(|x, y| x + y, &b, &c_b);
    assert!(approx_eq(sum.get(1, 1), 3.0));

    let scale: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    nabla::map_!(out, |x, y| x * y, &a, &scale);
    assert!(approx_eq(out.get(0, 0), 2.0));
    assert!(approx_eq(out.get(1, 1), 8.0));
}

#[test]
fn par_from_fn_matches_sequential() {
    let seq: Tensor<f64> = Tensor::from_fn(50, 50, |r, c| ((r * 50 + c) as f64).sin());
    let par: Tensor<f64> = Tensor::par_from_fn(50, 50, |r, c| ((r * 50 + c) as f64).sin());
    assert_approx_grid(&seq, &par, 1e-10);
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
fn utility_exports() {
    let v = linspace(0.0, 1.0, 5);
    assert_eq!(v.len(), 5);
    assert!((v[0] - 0.0).abs() < 1e-10);
    assert!((v[4] - 1.0).abs() < 1e-10);
    assert!((v[2] - 0.5).abs() < 1e-10);

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
#[should_panic(expected = "nabla: add")]
fn add_shape_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let b: Tensor<f64> = Tensor::zeros(2, 3);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "nabla: matmul")]
fn matmul_inner_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 3);
    let b: Tensor<f64> = Tensor::zeros(2, 2);
    let _ = &a * &b;
}

#[test]
#[should_panic(expected = "map! shape mismatch")]
fn map_shape_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let b: Tensor<f64> = Tensor::zeros(2, 3);
    let _: Tensor<f64> = nabla::map!(|x, y| x + y, &a, &b);
}

#[test]
#[should_panic(expected = "map_! shape mismatch")]
fn map__shape_mismatch_panics() {
    let a: Tensor<f64> = Tensor::zeros(2, 3);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    nabla::map_!(out, |x| x, &a);
}

#[test]
#[should_panic(expected = "axis must be 0 or 1")]
fn tensor_dim_out_of_range() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let _ = a.dim(2);
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
fn symmetric_eigen() {
    let a: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |i, j| [[4.0_f64, 2.0], [2.0, 3.0]][i][j]);
    let sym = Symmetric::new(a, nabla::linalg::Side::Lower).expect("Symmetric::new failed");
    let evals = sym.eigenvalues().expect("eigenvalues failed");
    assert_eq!(evals.len(), 2);
    assert!(evals[0] > 0.0);
    assert!(evals[1] > 0.0);
}

#[test]
fn reductions_boundary() {
    let a: Tensor<f64> = mat![[3.0_f64, 1.0], [5.0, 2.0]];
    assert!(approx_eq(a.sum_all(), 11.0));
    assert_eq!(a.argmax(), (1, 0));
    assert_eq!(a.argmin(), (0, 1));
}

#[test]
fn cas_diff_simplify_eval_roundtrip() {
    let x = Expr::var("x");
    let expr = Expr::sin(&Expr::pow(&x, &Expr::lit(2.0)));
    let d = simplify(&diff(&expr, "x"));
    let mut vars = HashMap::new();
    vars.insert("x", 1.0_f64);
    let result = eval(&d, &vars).expect("eval failed");
    assert!((result - 2.0 * 1.0_f64.cos()).abs() < 1e-10);
}

#[test]
fn cas_eval_unbound_error() {
    assert!(eval(&Expr::var("x"), &HashMap::new()).is_err());
}

#[test]
fn cas_eval_tensor_roundtrip() {
    let x = Expr::var("x");
    let s = simplify(&(&x * &Expr::lit(2.0)));
    let t: Tensor<f64> = linear_f64(2, 2);
    let mut vars: HashMap<&str, &Tensor<f64>> = HashMap::new();
    vars.insert("x", &t);
    let result = eval_tensor(&s, &vars).expect("eval_tensor failed");
    assert!((result.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((result.get(1, 1) - 8.0).abs() < 1e-10);
}

#[test]
fn ode_rk4_accuracy() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let sol = rk4(|_t, y| Ok(-y), &y0, (0.0, 1.0), 0.01).expect("rk4 failed");
    assert!((sol.final_state().expect("final_state failed").get(0, 0) - (-1.0_f64).exp()).abs() < 1e-6);
}

#[test]
fn ode_dormand_prince_adaptive() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let config = AdaptiveConfig {
        dt_init: 0.1,
        ..Default::default()
    };
    let sol = dormand_prince(|_t, y| Ok(-y), &y0, (0.0, 1.0), &config).expect("dormand_prince failed");
    assert!((sol.final_state().expect("final_state failed").get(0, 0) - (-1.0_f64).exp()).abs() < 1e-4);
    assert!(sol.len() < 100);
}

#[test]
fn ode_error_invalid_inputs() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    assert!(rk4(|_t, y| Ok(-y), &y0, (0.0, 1.0), -0.1).is_err());
    assert!(rk4(|_t, y| Ok(-y), &y0, (1.0, 0.0), 0.01).is_err());
}

#[test]
fn cas_eval_tensor_generic_type() {
    let x = Expr::var("x");
    let expr = Expr::exp(&x);
    let t: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j + 1) as f64);
    let mut vars: HashMap<&str, &Tensor<f64>> = HashMap::new();
    vars.insert("x", &t);
    let result = eval_tensor(&expr, &vars).expect("eval_tensor failed");
    assert!((result.get(0, 0) - 1.0_f64.exp()).abs() < 1e-10);
    assert!((result.get(1, 1) - 3.0_f64.exp()).abs() < 1e-10);
}

#[test]
fn ode_rk4_generic_type() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let sol = rk4(|_t, y| Ok(-y), &y0, (0.0, 0.5), 0.01).expect("rk4 failed");
    assert!((sol.final_state().expect("final_state failed").get(0, 0) - (-0.5_f64).exp()).abs() < 1e-5);
}

#[test]
fn autograd_simple_backward() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let x = tape.variable(linear_f64(2, 2));
    let y = x.emul(&x);
    y.backward().expect("backward failed");
    let grad = x.grad().expect("grad failed");
    assert!((grad.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((grad.get(0, 1) - 4.0).abs() < 1e-10);
    assert!((grad.get(1, 0) - 6.0).abs() < 1e-10);
    assert!((grad.get(1, 1) - 8.0).abs() < 1e-10);
}

#[test]
fn autograd_chain_rule() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let x = tape.variable(Tensor::from_fn(1, 1, |_, _| 1.0_f64));
    let x2 = x.emul(&x);
    let y = x2.sin();
    y.backward().expect("backward failed");
    let grad = x.grad().expect("grad failed");
    let expected = 2.0_f64 * 1.0_f64.cos();
    assert!((grad.get(0, 0) - expected).abs() < 1e-10);
}

#[test]
fn autograd_matmul_backward() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let a = tape.variable(mat![[1.0_f64, 2.0], [3.0, 4.0]]);
    let b = tape.variable(mat![[5.0_f64, 6.0], [7.0, 8.0]]);
    let c = &a * &b;
    c.backward().expect("backward failed");
    let grad_a = a.grad().expect("grad_a failed");
    let grad_b = b.grad().expect("grad_b failed");
    assert!((grad_a.get(0, 0) - 11.0).abs() < 1e-10);
    assert!((grad_a.get(0, 1) - 15.0).abs() < 1e-10);
    assert!((grad_a.get(1, 0) - 11.0).abs() < 1e-10);
    assert!((grad_a.get(1, 1) - 15.0).abs() < 1e-10);
    assert!((grad_b.get(0, 0) - 4.0).abs() < 1e-10);
    assert!((grad_b.get(0, 1) - 4.0).abs() < 1e-10);
    assert!((grad_b.get(1, 0) - 6.0).abs() < 1e-10);
    assert!((grad_b.get(1, 1) - 6.0).abs() < 1e-10);
}

#[test]
fn dual_exp_ln() {
    // f(x) = exp(ln(x)) = x, f'(x) = 1
    use nabla::prelude::Dual;
    use nabla::scalar::MathOps;
    let x = Dual::new(3.0_f64, 1.0);
    let y = x.math_ln().math_exp();
    assert!((y.value - 3.0).abs() < 1e-12);
    assert!((y.deriv - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "nabla: matmul")]
fn matmul_error_message_format() {
    let a = Tensor::<f64>::zeros(3, 4);
    let b = Tensor::<f64>::zeros(5, 2);
    let _ = &a * &b;
}

#[test]
#[should_panic(expected = "nabla: add")]
fn add_error_message_format() {
    let a = Tensor::<f64>::zeros(3, 4);
    let b = Tensor::<f64>::zeros(2, 3);
    let _ = &a + &b;
}

#[test]
fn einsum_canon_term_order_independent() {
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let expected = &a * &b;

    let c1: Tensor<f64> = einsum!(c1[i, j] = a[i, k] * b[k, j]);
    let c2: Tensor<f64> = einsum!(c2[i, j] = b[k, j] * a[i, k]);
    assert_approx_grid(&c1, &expected, 1e-10);
    assert_approx_grid(&c2, &expected, 1e-10);
}

#[test]
fn einsum_canon_index_rename() {
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let expected = &a * &b;

    let c: Tensor<f64> = einsum!(c[x, y] = a[x, z] * b[z, y]);
    assert_approx_grid(&c, &expected, 1e-10);
}

#[test]
fn cas_simplify_constant_folding() {
    use nabla::cas::{Expr, simplify};
    // 2 + 3 = 5
    assert_eq!(format!("{}", simplify(&(&Expr::lit(2.0) + &Expr::lit(3.0)))), "5");
    // 2 * 3 = 6
    assert_eq!(format!("{}", simplify(&(&Expr::lit(2.0) * &Expr::lit(3.0)))), "6");
}

#[test]
fn cas_simplify_eqsat_exp_ln() {
    use nabla::cas::{Expr, simplify};
    let x = Expr::var("x");
    // exp(ln(x)) = x
    assert_eq!(format!("{}", simplify(&Expr::exp(&Expr::ln(&x)))), "x");
    // ln(exp(x)) = x
    assert_eq!(format!("{}", simplify(&Expr::ln(&Expr::exp(&x)))), "x");
}

#[test]
fn bdf1_linear_decay() {
    // dy/dt = -100y, y(0) = 1 => y(t) = exp(-100t)
    use nabla::ode::{bdf1, Bdf1Config};
    let y0 = Tensor::<f64>::fill(1, 1, 1.0);
    let sol = bdf1(
        |_t, y| Ok(y * (-100.0_f64)),
        &y0,
        (0.0, 0.1),
        &Bdf1Config { dt: 0.001, tol: 1e-10, max_iter: 100 },
    ).expect("bdf1 should converge");
    let y_final = sol.final_state().expect("final_state").get(0, 0);
    let expected = (-100.0_f64 * 0.1).exp();
    assert!(
        (y_final - expected).abs() < 0.1,
        "bdf1: got {y_final}, expected ~{expected}"
    );
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
#[should_panic(expected = "nabla: permute")]
fn tensor_permute_invalid_axes() {
    let a = Tensor::<f64>::zeros(2, 2);
    let _ = a.permute(&[0, 0]);
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
fn cas_diff_simplify_matches_diff_simplify() {
    use nabla::cas::{diff, diff_simplify, eval, simplify, Expr};
    // Verify diff_simplify matches diff().simplify() numerically
    let x = Expr::var("x");
    let expr = &x * &Expr::sin(&x);
    let a = diff_simplify(&expr, "x");
    let b = simplify(&diff(&expr, "x"));
    for v in [0.1, 0.5, 1.0, 2.0] {
        let mut vars = HashMap::new();
        vars.insert("x", v);
        let va = eval(&a, &vars).expect("eval a");
        let vb = eval(&b, &vars).expect("eval b");
        assert!((va - vb).abs() < 1e-9, "mismatch at x={v}: {va} vs {vb}");
    }
}

#[test]
fn cas_diff_multivar_product_rule() {
    use nabla::cas::{diff_simplify, eval, Expr};
    // d(x*y)/dx = y
    let expr = &Expr::var("x") * &Expr::var("y");
    let d = diff_simplify(&expr, "x");
    let mut vars = HashMap::new();
    vars.insert("x", 2.0);
    vars.insert("y", 3.0);
    let val = eval(&d, &vars).expect("eval");
    assert!((val - 3.0).abs() < 1e-10);
}

#[test]
fn cas_diff_multivar_other_var() {
    use nabla::cas::{diff_simplify, eval, Expr};
    // d(x^2)/dy = 0
    let expr = Expr::pow(&Expr::var("x"), &Expr::lit(2.0));
    let d = diff_simplify(&expr, "y");
    let mut vars = HashMap::new();
    vars.insert("x", 5.0);
    vars.insert("y", 1.0);
    let val = eval(&d, &vars).expect("eval");
    assert!(val.abs() < 1e-10);
}

#[test]
fn cas_diff_chain_rule_exp() {
    use nabla::cas::{diff_simplify, eval, Expr};
    // d(exp(x^2))/dx = 2*x*exp(x^2)
    let x = Expr::var("x");
    let x2 = Expr::pow(&x, &Expr::lit(2.0));
    let expr = Expr::exp(&x2);
    let d = diff_simplify(&expr, "x");
    let xv = 1.5_f64;
    let expected = 2.0 * xv * (xv * xv).exp();
    let mut vars = HashMap::new();
    vars.insert("x", xv);
    let val = eval(&d, &vars).expect("eval");
    assert!((val - expected).abs() < 1e-6, "got {val}, expected {expected}");
}

#[test]
fn einsum_three_tensor_chain() {
    // c[i,j] = a[i,k] * b[k,l] * d[l,j] == a @ b @ d
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[1.0_f64, 0.0], [0.0, 1.0]]; // identity
    let d: Tensor<f64> = mat![[2.0_f64, 0.0], [0.0, 2.0]]; // 2*I
    let result: Tensor<f64> = einsum!(result[i, j] = a[i, k] * b[k, l] * d[l, j]);
    // a @ I @ 2I = 2a
    let expected = &a * 2.0_f64;
    assert_approx_grid(&result, &expected, 1e-10);
}

#[test]
fn einsum_three_tensor_vs_manual() {
    // Verify 3-tensor einsum matches manual matmul chain
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let d: Tensor<f64> = mat![[1.0_f64, 0.0], [0.0, 1.0]]; // identity
    let result: Tensor<f64> = einsum!(result[i, j] = a[i, k] * b[k, l] * d[l, j]);
    let manual = &(&a * &b) * &d; // a @ b @ I = a @ b
    assert_approx_grid(&result, &manual, 1e-10);
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
fn grad_quadratic() {
    // f(x) = sum(x^2), df/dx = 2x; at x=[[3.0]] => grad = [[6.0]]
    let x = mat![[3.0_f64]];
    let g = grad(
        |xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var(),
        &x,
    )
    .expect("grad returned None");
    assert!((g[(0, 0)] - 6.0).abs() < 1e-10);
}

#[test]
fn gradient_prep_reuse() {
    let f = |xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var();
    let x1 = mat![[2.0_f64]];
    let x2 = mat![[5.0_f64]];
    let prep = gradient_prep(&f, &x1);
    let g1 = gradient(&f, &x1, &prep).expect("gradient returned None");
    let g2 = gradient(&f, &x2, &prep).expect("gradient returned None");
    assert!((g1[(0, 0)] - 4.0).abs() < 1e-10);
    assert!((g2[(0, 0)] - 10.0).abs() < 1e-10);
}

#[test]
#[should_panic(expected = "nabla::gradient")]
fn gradient_prep_shape_mismatch_panics() {
    let f = |xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var();
    let x1 = mat![[1.0_f64]];
    let prep = gradient_prep(&f, &x1);
    let x2 = Tensor::<f64>::zeros(2, 2);
    let _ = gradient(&f, &x2, &prep);
}

#[test]
fn expm_diagonal() {
    // For diagonal A = diag(1.0, 2.0), exp(A) = diag(e, e^2)
    let mut a = Tensor::<f64>::zeros(2, 2);
    a[(0, 0)] = 1.0;
    a[(1, 1)] = 2.0;
    let e = expm(&a).expect("expm failed");
    assert!(
        (e[(0, 0)] - std::f64::consts::E).abs() < 1e-4,
        "expm diag (0,0): got {}, expected {}",
        e[(0, 0)],
        std::f64::consts::E
    );
    assert!(
        (e[(1, 1)] - std::f64::consts::E.powi(2)).abs() < 1e-4,
        "expm diag (1,1): got {}, expected {}",
        e[(1, 1)],
        std::f64::consts::E.powi(2)
    );
    assert!(e[(0, 1)].abs() < 1e-6);
    assert!(e[(1, 0)].abs() < 1e-6);
}

#[test]
fn if_euler_scalar_stiff_stable() {
    // lambda=100, dt=0.1 — forward Euler diverges but IF Euler stays stable
    let y0 = mat![[1.0_f64]];
    let config = IfEulerScalarConfig { dt: 0.1, stiffness: 100.0 };
    let sol = if_euler_scalar(
        |_t, _y| Ok(Tensor::<f64>::zeros(1, 1)),
        &y0,
        (0.0, 0.5),
        &config,
    )
    .expect("if_euler_scalar failed");
    let y_final = sol.final_state().expect("no final state");
    assert!(y_final[(0, 0)] >= 0.0, "IF Euler should remain stable");
    assert!(
        y_final[(0, 0)] < 0.01,
        "should have decayed significantly, got {}",
        y_final[(0, 0)]
    );
}

#[test]
fn dae_simple_constraint() {
    // x' = 1, 0 = z - x  =>  x(t) = t, z(t) = t
    // x0 = 0, z0 = 0, t ∈ [0, 1]
    use nabla::ode::{dae_solve, DaeConfig};
    let x0 = Tensor::<f64>::fill(1, 1, 0.0);
    let z0 = Tensor::<f64>::fill(1, 1, 0.0);
    let sol = dae_solve(
        |_x, _z, _t| Tensor::<f64>::fill(1, 1, 1.0),  // f: x' = 1
        |x, z, _t| {                                     // g: 0 = z - x
            let zv = z.get(0, 0);
            let xv = x.get(0, 0);
            Tensor::<f64>::fill(1, 1, zv - xv)
        },
        x0,
        z0,
        (0.0, 1.0),
        DaeConfig { dt: 0.01, tol: 1e-10, max_iter: 50 },
    )
    .expect("dae_solve failed");
    let x_final = sol.final_state().expect("no final state").get(0, 0);
    assert!(
        (x_final - 1.0).abs() < 0.05,
        "dae x(1) should be ~1.0, got {x_final}"
    );
}

#[test]
fn dae_quadratic_constraint() {
    // x' = z, 0 = z - 2*t  =>  z(t) = 2t, x(t) = t^2
    // x0 = 0, z0 = 0
    use nabla::ode::{dae_solve, DaeConfig};
    let x0 = Tensor::<f64>::fill(1, 1, 0.0);
    let z0 = Tensor::<f64>::fill(1, 1, 0.0);
    let sol = dae_solve(
        |_x, z, _t| z.clone(),              // f: x' = z
        |_x, z, t| {                         // g: 0 = z - 2t
            let zv = z.get(0, 0);
            Tensor::<f64>::fill(1, 1, zv - 2.0 * t)
        },
        x0,
        z0,
        (0.0, 1.0),
        DaeConfig { dt: 0.01, tol: 1e-10, max_iter: 50 },
    )
    .expect("dae_solve failed");
    let x_final = sol.final_state().expect("no final state").get(0, 0);
    // x(1) = 1^2 = 1.0
    assert!(
        (x_final - 1.0).abs() < 0.05,
        "dae x(1) should be ~1.0, got {x_final}"
    );
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
fn vcat_macro_three_tensors() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64> = mat![[3.0, 4.0]];
    let c: Tensor<f64> = mat![[5.0, 6.0]];
    let r = nabla::vcat!(a, b, c);
    assert_eq!(r.nrows(), 3);
    assert_eq!(r.ncols(), 2);
    assert!(approx_eq(r.get(0, 0), 1.0));
    assert!(approx_eq(r.get(1, 0), 3.0));
    assert!(approx_eq(r.get(2, 0), 5.0));
}

#[test]
fn hcat_macro_three_tensors() {
    let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    let b: Tensor<f64> = mat![[3.0], [4.0]];
    let c: Tensor<f64> = mat![[5.0], [6.0]];
    let r = nabla::hcat!(a, b, c);
    assert_eq!(r.ncols(), 3);
    assert_eq!(r.nrows(), 2);
    assert!(approx_eq(r.get(0, 0), 1.0));
    assert!(approx_eq(r.get(0, 1), 3.0));
    assert!(approx_eq(r.get(0, 2), 5.0));
}

#[test]
fn vcat_macro_two_tensors() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0]];
    let b: Tensor<f64> = mat![[3.0, 4.0]];
    let r = nabla::vcat!(a, b);
    assert_eq!(r.shape(), (2, 2));
}

#[test]
fn static_matrix_matmul_shape() {
    let a: StaticMatrix<f64, 2, 3> = StaticMatrix::from_fn(|r, c| (r * 3 + c) as f64);
    let b: StaticMatrix<f64, 3, 2> = StaticMatrix::from_fn(|r, c| (r * 2 + c) as f64);
    let c: StaticMatrix<f64, 2, 2> = &a * &b;
    // row0 of a = [0,1,2], col0 of b = [0,2,4] => 0+2+8 = 10
    assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
}

#[test]
fn static_matrix_add_and_neg() {
    let a: StaticMatrix<f64, 2, 2> = StaticMatrix::from_fn(|r, c| (r + c) as f64);
    let b = &a + &a;
    assert!((b[(0, 1)] - 2.0).abs() < 1e-10);
    let neg = -&a;
    assert!((neg[(0, 1)] - (-1.0)).abs() < 1e-10);
}

#[test]
fn static_matrix_typed_matmul() {
    // 3x4 * 4x2 -> 3x2, compile-time shape checked
    let a = StaticMatrix::<f64, 3, 4>::from_fn(|r, c| (r * 4 + c) as f64);
    let b = StaticMatrix::<f64, 4, 2>::from_fn(|r, c| (r * 2 + c) as f64);
    let c: StaticMatrix<f64, 3, 2> = &a * &b;
    assert_eq!(c.shape(), (3, 2));
    // (0,0): row0=[0,1,2,3] dot col0=[0,2,4,6] = 0+2+8+18 = 28
    assert!((c.get(0, 0) - 28.0).abs() < 1e-10);
}

#[test]
fn static_matrix_typed_transpose() {
    let a = StaticMatrix::<f64, 3, 4>::from_fn(|r, c| (r * 4 + c) as f64);
    let at: StaticMatrix<f64, 4, 3> = a.t();
    assert_eq!(at.shape(), (4, 3));
    assert!((at.get(2, 1) - a.get(1, 2)).abs() < 1e-10);
}

#[test]
fn static_matrix_sub_ref() {
    let a = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c + 1) as f64);
    let b = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    let d = &a - &b;
    // Every element should be 1.0
    for r in 0..2 {
        for c in 0..3 {
            assert!((d.get(r, c) - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn metd_linear_decay() {
    // dy/dt = -y  →  L = [-1] (1x1 matrix), N = 0
    // y(0) = 1, exact: y(t) = exp(-t), so y(1) ≈ 0.3679
    use nabla::ode::{metd_solve, MetdConfig};
    let l: Tensor<f64> = mat![[-1.0_f64]];
    let y0: Tensor<f64> = mat![[1.0_f64]];
    let cfg = MetdConfig { dt: 0.01, order: 8 };
    let sol = metd_solve(
        &l,
        |_t, _y| Tensor::<f64>::zeros(1, 1), // N(t,y) = 0
        y0,
        (0.0, 1.0),
        cfg,
    )
    .expect("metd_solve failed");
    let y_final = sol.final_state().expect("empty solution");
    let val = y_final.get(0, 0);
    let expected = (-1.0_f64).exp(); // e^{-1} ≈ 0.3679
    assert!(
        (val - expected).abs() < 1e-6,
        "metd_linear_decay: got {val}, expected {expected}"
    );
}

#[test]
fn stormer_verlet_harmonic() {
    // Simple harmonic oscillator: V(q) = q^2/2, grad_V(q) = q
    // H = p^2/2 + q^2/2 conserved
    // q(0)=1, p(0)=0, exact: q(t)=cos(t), p(t)=-sin(t), H=0.5
    use nabla::ode::{stormer_verlet, StormerVerletConfig};
    let cfg = StormerVerletConfig { dt: 0.01, mass: 1.0 };
    let (_, qs, ps) = stormer_verlet(
        |q| q.clone(), // grad_V(q) = q
        mat![[1.0_f64]],
        mat![[0.0_f64]],
        (0.0, 2.0 * std::f64::consts::PI),
        cfg,
    )
    .expect("stormer_verlet failed");
    // Check energy conservation: H = (q^2 + p^2)/2 ≈ 0.5 at every step
    for (q, p) in qs.iter().zip(ps.iter()) {
        let qv = q.get(0, 0);
        let pv = p.get(0, 0);
        let h = (qv * qv + pv * pv) * 0.5;
        assert!(
            (h - 0.5).abs() < 0.01,
            "stormer_verlet_harmonic: energy {h} deviates from 0.5"
        );
    }
}

#[test]
fn solve_lstsq_overdetermined_exact() {
    // A*x = b where b is in range(A): x should be recovered exactly.
    let a = mat![[1.0_f64, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let b = mat![[1.0_f64], [2.0], [3.0]];
    let x = a.solve_lstsq(&b).expect("solve_lstsq failed");
    assert!(
        (x.get(0, 0) - 0.0).abs() < 1e-9,
        "intercept: expected 0, got {}",
        x.get(0, 0)
    );
    assert!(
        (x.get(1, 0) - 1.0).abs() < 1e-9,
        "slope: expected 1, got {}",
        x.get(1, 0)
    );
}

#[test]
fn solve_lstsq_overdetermined_approximate() {
    // A*x ≈ b where b is NOT in range(A): least-squares fit.
    let a = mat![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let b = mat![[1.0_f64], [2.0], [4.0]];
    // Normal equations: A^T A x = A^T b
    // A^T A = [[2,1],[1,2]], A^T b = [[5],[6]]
    // x = [[4/3],[7/3]]
    let x = a.solve_lstsq(&b).expect("solve_lstsq failed");
    assert!(
        (x.get(0, 0) - 4.0 / 3.0).abs() < 1e-9,
        "x0: expected {}, got {}",
        4.0 / 3.0,
        x.get(0, 0)
    );
    assert!(
        (x.get(1, 0) - 7.0 / 3.0).abs() < 1e-9,
        "x1: expected {}, got {}",
        7.0 / 3.0,
        x.get(1, 0)
    );
}

#[test]
fn svd_tall_3x2_reconstruction() {
    let a = mat![[3.0_f64, 2.0], [2.0, 3.0], [1.0, 1.0]];
    let svd = a.svd().expect("SVD of 3x2 failed");
    let s = svd.s();
    let u = svd.u();
    let vt = svd.vt();
    assert_eq!(s.len(), 2);
    let (m, n) = a.shape();
    let recon = Tensor::from_fn(m, n, |i, j| {
        (0..s.len()).map(|r| u.get(i, r) * s[r] * vt.get(r, j)).sum::<f64>()
    });
    let err = (&a - &recon).abs().sum_all();
    assert!(err < 1e-12, "reconstruction error: {err}");
}

#[test]
fn svd_rank_deficient() {
    // Rank-1 matrix: outer product
    let a = mat![[1.0_f64, 2.0], [2.0, 4.0]];
    let svd = a.svd().expect("SVD of rank-deficient failed");
    let s = svd.s();
    assert_eq!(s.len(), 2);
    assert!(s[0] > 1e-10, "s[0] should be nonzero: {}", s[0]);
    assert!(s[1] < 1e-10, "s[1] should be ~0: {}", s[1]);
}

#[test]
fn svd_singular_values_descending() {
    let a = mat![[4.0_f64, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0], [0.0, 1.0, 0.0]];
    let svd = a.svd().expect("SVD failed");
    let s = svd.s();
    for w in s.windows(2) {
        assert!(w[0] >= w[1], "not descending: {} < {}", w[0], w[1]);
    }
}

#[test]
fn svd_reconstruct_rank_1_reduces_error() {
    let a = mat![[1.0_f64, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    let svd = a.svd().expect("SVD failed");
    let rank1 = svd.reconstruct_rank(1);
    let rank2 = svd.reconstruct_rank(2);
    assert_eq!(rank1.shape(), a.shape());
    // rank2 error < rank1 error (Eckart-Young)
    let e1 = (&a - &rank1).norm();
    let e2 = (&a - &rank2).norm();
    assert!(e2 < e1, "rank2 error ({e2}) should be < rank1 error ({e1})");
}

#[test]
fn norm_3_4_5_triangle() {
    let a: Tensor<f64> = mat![[3.0_f64, 4.0]];
    assert!((a.norm() - 5.0).abs() < 1e-10, "norm={}", a.norm());
    assert!((a.norm_sq() - 25.0).abs() < 1e-10, "norm_sq={}", a.norm_sq());
}

#[test]
fn norm_matrix_frobenius() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    // Frobenius: sqrt(1+4+9+16) = sqrt(30)
    let expected = 30.0_f64.sqrt();
    assert!((a.norm() - expected).abs() < 1e-10, "norm={}", a.norm());
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
fn log_softmax_row_sums_to_one() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]];
    let ls = a.log_softmax(1);
    let sum_exp: f64 = (0..3).map(|j| ls.get(0, j).exp()).sum();
    assert!((sum_exp - 1.0).abs() < 1e-10);
}

#[test]
fn cross_entropy_loss_one_hot() {
    let logits: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]];
    let log_probs = logits.log_softmax(1);
    let target: Tensor<f64> = mat![[0.0_f64, 0.0, 1.0]]; // class 2
    let loss = log_probs.cross_entropy_loss(&target);
    // loss = -log(softmax(3)) = -log(e^3/(e^1+e^2+e^3))
    let expected = -(3.0_f64.exp() / (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp())).ln();
    assert!((loss - expected).abs() < 1e-10, "loss={loss}, expected={expected}");
}

#[test]
fn cross_entropy_loss_perfect_prediction() {
    // When prediction is very confident and correct, loss approaches 0
    let logits: Tensor<f64> = mat![[-100.0_f64, 100.0]];
    let log_probs = logits.log_softmax(1);
    let target: Tensor<f64> = mat![[0.0_f64, 1.0]];
    let loss = log_probs.cross_entropy_loss(&target);
    assert!(loss < 1e-10, "loss={loss} should be near 0 for perfect prediction");
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
fn relu_mixed_signs() {
    let a: Tensor<f64> = mat![[-1.0_f64, 2.0, -3.0, 4.0]];
    let r = a.relu();
    assert!((r.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((r.get(0, 1) - 2.0).abs() < 1e-10);
    assert!((r.get(0, 2) - 0.0).abs() < 1e-10);
    assert!((r.get(0, 3) - 4.0).abs() < 1e-10);
}

#[test]
fn sigmoid_at_zero_is_half() {
    let a: Tensor<f64> = mat![[0.0_f64]];
    assert!((a.sigmoid().get(0, 0) - 0.5).abs() < 1e-10);
}

#[test]
fn gelu_at_zero_is_zero() {
    let a: Tensor<f64> = mat![[0.0_f64]];
    assert!(a.gelu().get(0, 0).abs() < 1e-6);
}

#[test]
fn softmax_sums_to_one() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0]]; // 1x3
    let s = a.softmax(1);
    let sum = s.get(0, 0) + s.get(0, 1) + s.get(0, 2);
    assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {sum}");
}

#[test]
fn softmax_axis0_sums_to_one() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2
    let s = a.softmax(0);
    // Each column should sum to 1
    let col0_sum = s.get(0, 0) + s.get(1, 0) + s.get(2, 0);
    let col1_sum = s.get(0, 1) + s.get(1, 1) + s.get(2, 1);
    assert!((col0_sum - 1.0).abs() < 1e-10, "col0 sum = {col0_sum}");
    assert!((col1_sum - 1.0).abs() < 1e-10, "col1 sum = {col1_sum}");
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
fn layer_norm_row_zero_mean() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
    let normed = a.layer_norm(1, 1e-8);
    for r in 0..2 {
        let row_mean: f64 = (0..4).map(|c| normed.get(r, c)).sum::<f64>() / 4.0;
        assert!(row_mean.abs() < 1e-6, "row {r} mean = {row_mean}, expected ~0");
    }
}

#[test]
fn layer_norm_row_unit_variance() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
    let normed = a.layer_norm(1, 1e-8);
    for r in 0..2 {
        let var: f64 = (0..4).map(|c| { let x = normed.get(r, c); x * x }).sum::<f64>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "row {r} var = {var}, expected ~1");
    }
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
    let m: Tensor<f64> = Tensor::from_fn(128, 4, |k, l| ((k * 4 + l) as f64) * 0.01);
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
    let a: Tensor<f64> = linear_f64(3, 4);
    let b: Tensor<f64> = linear_f64(4, 5);
    let d: Tensor<f64> = linear_f64(5, 2);
    let result: Tensor<f64> = einsum!(result[i, j] = a[i, k] * b[k, l] * d[l, j]);
    let manual = &(&a * &b) * &d;
    assert_approx_grid(&result, &manual, 1e-6);
}

#[test]
fn fuse_multi_tensor() {
    let x: Tensor<f64> = Tensor::from_fn(3, 3, |r, c| (r + c + 1) as f64);
    let y: Tensor<f64> = Tensor::from_fn(3, 3, |r, c| (r * 3 + c + 1) as f64 * 0.1);
    let z: Tensor<f64> = fuse!(x * y + x.sin(); x, y);
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
    let x: Tensor<f64> = Tensor::from_fn(2, 3, |r, c| (r * 3 + c + 1) as f64 * 0.5);
    let y: Tensor<f64> = fuse!(x.exp().ln().abs().sqrt(); x);
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
    let x: Tensor<f64> = linear_f64(2, 2);
    let y: Tensor<f64> = fuse!(-x + x * 2.0; x);
    for r in 0..2 {
        for c in 0..2 {
            let v = (r * 2 + c + 1) as f64;
            let expected = -v + v * 2.0;
            assert!(approx_eq(y.get(r, c), expected));
        }
    }
}

#[test]
fn static_matrix_outer_product() {
    let u: [f64; 3] = [1.0, 2.0, 3.0];
    let v: [f64; 2] = [4.0, 5.0];
    let m: StaticMatrix<f64, 3, 2> = StaticMatrix::outer(&u, &v);
    assert_eq!(m.shape(), (3, 2));
    // u[i] * v[j]
    assert!(approx_eq(m.get(0, 0), 4.0));
    assert!(approx_eq(m.get(0, 1), 5.0));
    assert!(approx_eq(m.get(1, 0), 8.0));
    assert!(approx_eq(m.get(1, 1), 10.0));
    assert!(approx_eq(m.get(2, 0), 12.0));
    assert!(approx_eq(m.get(2, 1), 15.0));
}

#[test]
fn static_matrix_data_access() {
    let m: StaticMatrix<f64, 2, 2> = StaticMatrix::from_fn(|r, c| (r * 2 + c + 1) as f64);
    let d = m.data();
    assert!(approx_eq(d[0][0], 1.0));
    assert!(approx_eq(d[0][1], 2.0));
    assert!(approx_eq(d[1][0], 3.0));
    assert!(approx_eq(d[1][1], 4.0));
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
fn linear_layout_identity() {
    let id = LinearLayout16::identity();
    for v in 0..16u64 {
        assert_eq!(id.apply(v), v, "identity failed for v={v}");
    }
}

#[test]
fn linear_layout_swizzle_no_conflict() {
    let sw = LinearLayout16::swizzle_for_tile(16, 16, 32);
    // For each row, the 16 column addresses must map to distinct bank slots
    for row in 0..16u64 {
        let mut banks = std::collections::HashSet::new();
        for col in 0..16u64 {
            let addr = (row << 4) | col;
            let swizzled = sw.apply(addr);
            let bank = swizzled & 0x1F; // low 5 bits = bank index (mod 32)
            banks.insert(bank);
        }
        assert_eq!(banks.len(), 16, "bank conflict in row {row}");
    }
}

#[test]
fn linear_layout_compose() {
    let a = LinearLayout16::swizzle_for_tile(16, 16, 32);
    let b = LinearLayout16::identity();
    let ab = a.compose(&b);
    // compose(A, identity) == A
    for v in 0..16u64 {
        assert_eq!(ab.apply(v), a.apply(v), "compose(A,I) != A for v={v}");
    }
    // compose(A, B).apply(v) == A.apply(B.apply(v))
    let ba = b.compose(&a);
    for v in 0..32u64 {
        assert_eq!(ba.apply(v), b.apply(a.apply(v)), "compose(B,A).apply != B(A(v)) for v={v}");
    }
}

#[test]
fn fuse_gemm_sigmoid() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[0.5_f64, 0.1], [0.2, 0.3]];
    let fused: Tensor<f64> = fuse!((&a * &b).sigmoid(); a, b);
    let expected = (&a * &b).sigmoid();
    assert_approx_grid(&fused, &expected, 1e-12);
}

#[test]
fn fuse_gemm_relu() {
    let a: Tensor<f64> = mat![[1.0_f64, -1.0], [2.0, 0.5]];
    let b: Tensor<f64> = mat![[0.5_f64, -0.5], [1.0, 0.3]];
    let fused: Tensor<f64> = fuse!((&a * &b).relu(); a, b);
    let expected = (&a * &b).relu();
    assert_approx_grid(&fused, &expected, 1e-12);
}

// ---------------------------------------------------------------------------
// Parareal parallel-in-time ODE solver
// ---------------------------------------------------------------------------

#[test]
fn parareal_van_der_pol() {
    use nabla::ode::{parareal_solve, PararealConfig};

    // Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0
    // Rewrite as system: x1' = x2, x2' = mu*(1-x1^2)*x2 - x1
    // For scalar parareal we solve just x1' = x2 with x2 coupled analytically via Euler.
    // Simpler: use a scalar test ODE y' = -y (exponential decay) and verify convergence.
    let t0 = 0.0;
    let t1 = 2.0;
    let y0 = 1.0;

    // Coarse propagator: single Euler step per interval
    let coarse = |ta: f64, tb: f64, ya: f64| -> f64 {
        let h = tb - ta;
        ya + h * (-ya) // y' = -y, forward Euler
    };

    // Fine propagator: 100 Euler sub-steps per interval (accurate)
    let fine = |ta: f64, tb: f64, ya: f64| -> f64 {
        let n_sub = 100;
        let h = (tb - ta) / n_sub as f64;
        let mut y = ya;
        for _ in 0..n_sub {
            y += h * (-y);
        }
        y
    };

    let config = PararealConfig { n_intervals: 8, max_iter: 5, tol: 1e-8 };
    let result = parareal_solve(t0, t1, y0, &config, coarse, fine);
    assert!(result.is_ok());
    let vals = result.ok().expect("parareal should converge");

    // Exact solution: y(t) = exp(-t)
    let exact_final = (-t1).exp();
    let computed_final = vals[vals.len() - 1];

    // Fine propagator with 100 sub-steps has ~1e-4 error per interval,
    // parareal corrects to fine-level accuracy
    assert!(
        (computed_final - exact_final).abs() < 1e-3,
        "parareal final value {computed_final} vs exact {exact_final}"
    );

    // Check that all intermediate values are reasonable
    let dt = (t1 - t0) / config.n_intervals as f64;
    for (i, &v) in vals.iter().enumerate() {
        let t = t0 + i as f64 * dt;
        let exact = (-t).exp();
        assert!(
            (v - exact).abs() < 1e-2,
            "parareal checkpoint {i}: computed={v}, exact={exact}"
        );
    }
}

// ---------------------------------------------------------------------------
// #[nabla_grad] proc macro — source-transform forward AD
// ---------------------------------------------------------------------------

#[nabla_grad]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn nabla_grad_sigmoid() {
    let (val, grad) = sigmoid_grad(0.0);
    // sigmoid(0) = 0.5
    assert!((val - 0.5).abs() < 1e-12, "sigmoid(0) = {val}, expected 0.5");
    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25
    assert!((grad - 0.25).abs() < 1e-12, "sigmoid'(0) = {grad}, expected 0.25");
}

#[nabla_grad]
fn poly(x: f64) -> f64 {
    x * x + 2.0 * x
}

#[test]
fn nabla_grad_chain() {
    let (val, grad) = poly_grad(3.0);
    // poly(3) = 9 + 6 = 15
    assert!((val - 15.0).abs() < 1e-12, "poly(3) = {val}, expected 15");
    // poly'(x) = 2x + 2, poly'(3) = 8
    assert!((grad - 8.0).abs() < 1e-12, "poly'(3) = {grad}, expected 8");
}

// ---------------------------------------------------------------------------
// wgpu register-tile software MMA shader generation
// ---------------------------------------------------------------------------

#[test]
fn wgpu_register_tile_shader_gen() {
    let shader = nabla::wgsl::gen_matmul_register_tile(4, 4, 64, 64, 8);
    assert!(shader.contains("workgroup_size"), "missing workgroup_size");
    assert!(shader.contains("smem_a"), "missing smem_a");
    assert!(shader.contains("smem_b"), "missing smem_b");
    assert!(shader.contains("regs"), "missing regs");
    assert!(shader.contains("workgroupBarrier"), "missing workgroupBarrier");
    // Workgroup dims: (64/4, 64/4) = (16, 16)
    assert!(shader.contains("@compute @workgroup_size(16, 16, 1)"), "wrong workgroup dims");
}

#[test]
fn wgpu_select_register_tile_params() {
    assert_eq!(nabla::wgsl::select_register_tile_params(32, 32, 32), (2, 2, 16, 16, 8));
    assert_eq!(nabla::wgsl::select_register_tile_params(100, 100, 100), (4, 4, 32, 32, 8));
    assert_eq!(nabla::wgsl::select_register_tile_params(256, 256, 256), (4, 4, 64, 64, 16));
    assert_eq!(nabla::wgsl::select_register_tile_params(1024, 1024, 512), (4, 4, 64, 64, 16));
}

// --- Migrated from new_ops.rs ---

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
    assert_approx_grid(&cat_result, &vcat_result, 1e-10);
}

#[test]
fn cat_axis1_equals_hcat() {
    let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    let b: Tensor<f64> = mat![[3.0_f64], [4.0]];
    let cat_result = Tensor::cat(&[&a, &b], 1);
    let hcat_result = Tensor::hcat(&[&a, &b]);
    assert_approx_grid(&cat_result, &hcat_result, 1e-10);
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
    let a: Tensor<f64> = linear_f64(2, 6);
    let v = a.view(3, 4);
    let r = a.reshape(3, 4);
    assert_approx_grid(&v, &r, 1e-10);
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
    assert_approx_grid(&c1, &expected, 1e-10);
    assert_approx_grid(&c2, &expected, 1e-10);
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
    assert_approx_grid(&h1, &h2, 1e-10);
    let expected_h = a.emul(&b);
    assert_approx_grid(&h1, &expected_h, 1e-10);
}

// ── New operations tests (§14.1 parity) ──────────────────────────

#[test]
fn silu_known() {
    let x: Tensor<f64> = mat![[0.0, 1.0, -1.0]];
    let y = x.silu();
    assert!((y.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 0.7310585786).abs() < 1e-6);
    assert!((y.get(0, 2) - (-0.2689414214)).abs() < 1e-6);
}

#[test]
fn mish_known() {
    let x: Tensor<f64> = mat![[0.0, 1.0]];
    let y = x.mish();
    assert!((y.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 0.8651).abs() < 1e-3);
}

#[test]
fn leaky_relu_known() {
    let x: Tensor<f64> = mat![[2.0, -3.0, 0.0]];
    let y = x.leaky_relu(0.01);
    assert!((y.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((y.get(0, 1) - (-0.03)).abs() < 1e-10);
}

#[test]
fn elu_known() {
    let x: Tensor<f64> = mat![[1.0, -1.0, 0.0]];
    let y = x.elu(1.0);
    assert!((y.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((y.get(0, 1) - (-0.6321)).abs() < 1e-3);
}

#[test]
fn hardswish_known() {
    let x: Tensor<f64> = mat![[0.0, 3.0, -3.0, 1.0]];
    let y = x.hardswish();
    assert!((y.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 3.0).abs() < 1e-10);
    assert!((y.get(0, 2) - 0.0).abs() < 1e-10);
    assert!((y.get(0, 3) - 0.6667).abs() < 1e-3);
}

#[test]
fn mse_loss_known() {
    let pred: Tensor<f64> = mat![[1.0, 2.0, 3.0]];
    let target: Tensor<f64> = mat![[1.5, 2.5, 3.5]];
    let loss = pred.mse_loss(&target);
    assert!((loss - 0.25).abs() < 1e-10);
}

#[test]
fn l1_loss_known() {
    let pred: Tensor<f64> = mat![[1.0, 2.0, 3.0]];
    let target: Tensor<f64> = mat![[1.5, 2.5, 3.5]];
    assert!((pred.l1_loss(&target) - 0.5).abs() < 1e-10);
}

#[test]
fn smooth_l1_loss_known() {
    let pred: Tensor<f64> = mat![[0.0, 0.0]];
    let target: Tensor<f64> = mat![[0.5, 2.0]];
    assert!((pred.smooth_l1_loss(&target, 1.0) - 0.8125).abs() < 1e-6);
}

#[test]
fn bce_with_logits_known() {
    let logits: Tensor<f64> = mat![[0.0]];
    let target: Tensor<f64> = mat![[1.0]];
    assert!((logits.bce_with_logits(&target) - 0.6931).abs() < 1e-3);
}

#[test]
fn nll_loss_known() {
    let lp: Tensor<f64> = mat![[-0.5, -1.0, -2.0], [-2.0, -0.3, -1.5], [-1.0, -1.0, -0.2]];
    let targets: Tensor<f64> = mat![[0.0], [1.0], [2.0]];
    assert!((lp.nll_loss(&targets) - 0.3333).abs() < 1e-3);
}

#[test]
fn kl_div_known() {
    let log_p: Tensor<f64> = mat![[-1.0, -1.0]];
    let q: Tensor<f64> = mat![[0.5, 0.5]];
    assert!((log_p.kl_div(&q) - 0.3069).abs() < 1e-3);
}

#[test]
fn prod_all_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    assert!((x.prod_all() - 24.0).abs() < 1e-10);
}

#[test]
fn count_nonzero_known() {
    let x: Tensor<f64> = mat![[1.0, 0.0], [0.0, 3.0]];
    assert_eq!(x.count_nonzero(), 2);
}

#[test]
fn argmax_axis_rows() {
    let x: Tensor<f64> = mat![[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]];
    let idx = x.argmax_axis(1);
    assert_eq!(idx.get(0, 0) as usize, 1);
    assert_eq!(idx.get(1, 0) as usize, 2);
}

#[test]
fn argmin_axis_rows() {
    let x: Tensor<f64> = mat![[3.0, 1.0, 2.0]];
    let idx = x.argmin_axis(1);
    assert_eq!(idx.get(0, 0) as usize, 1);
}

#[test]
fn cumprod_axis1() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0]];
    let y = x.cumprod(1);
    assert!((y.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 2.0).abs() < 1e-10);
    assert!((y.get(0, 2) - 6.0).abs() < 1e-10);
}

#[test]
fn norm_axis_l2() {
    let x: Tensor<f64> = mat![[3.0, 4.0]];
    let n = x.norm_axis(2.0, 1);
    assert!((n.get(0, 0) - 5.0).abs() < 1e-10);
}

#[test]
fn arange_known() {
    let x: Tensor<f64> = Tensor::arange(0.0, 1.0, 5);
    assert_eq!(x.shape(), (1, 5));
    assert!((x.get(0, 3) - 3.0).abs() < 1e-10);
}

#[test]
fn linspace_known() {
    let x: Tensor<f64> = Tensor::linspace(0.0, 1.0, 5);
    assert!((x.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((x.get(0, 4) - 1.0).abs() < 1e-10);
    assert!((x.get(0, 2) - 0.5).abs() < 1e-10);
}

#[test]
fn full_like_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let y = x.full_like(7.0);
    assert_eq!(y.shape(), (2, 2));
    assert!((y.get(0, 0) - 7.0).abs() < 1e-10);
}

#[test]
fn split_axis0() {
    let x = linear_f64(4, 2);
    let parts = x.split(&[1, 3], 0);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape(), (1, 2));
    assert_eq!(parts[1].shape(), (3, 2));
}

#[test]
fn repeat_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0]];
    let y = x.repeat(2, 3);
    assert_eq!(y.shape(), (2, 6));
    assert!((y.get(1, 4) - 1.0).abs() < 1e-10);
}

#[test]
fn expand_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0]];
    let y = x.expand(4, 3);
    assert_eq!(y.shape(), (4, 3));
    assert!((y.get(3, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn pad_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let y = x.pad([1, 1, 1, 1], 0.0);
    assert_eq!(y.shape(), (4, 4));
    assert!((y.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((y.get(1, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn gather_axis1() {
    let x: Tensor<f64> = mat![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
    let idx: Tensor<f64> = mat![[2.0, 0.0], [1.0, 1.0]];
    let y = x.gather(1, &idx);
    assert_eq!(y.shape(), (2, 2));
    assert!((y.get(0, 0) - 30.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 10.0).abs() < 1e-10);
}

#[test]
fn scatter_axis1() {
    let x: Tensor<f64> = Tensor::zeros(2, 3);
    let idx: Tensor<f64> = mat![[0.0], [2.0]];
    let src: Tensor<f64> = mat![[10.0], [20.0]];
    let y = x.scatter(1, &idx, &src);
    assert!((y.get(0, 0) - 10.0).abs() < 1e-10);
    assert!((y.get(1, 2) - 20.0).abs() < 1e-10);
}

#[test]
fn index_select_axis0() {
    let x: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let idx: Tensor<f64> = mat![[2.0], [0.0]];
    let y = x.index_select(0, &idx);
    assert_eq!(y.shape(), (2, 2));
    assert!((y.get(0, 0) - 5.0).abs() < 1e-10);
    assert!((y.get(1, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn masked_fill_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let mask: Tensor<f64> = mat![[1.0, 0.0], [0.0, 1.0]];
    let y = x.masked_fill(&mask, -999.0);
    assert!((y.get(0, 0) - (-999.0)).abs() < 1e-10);
    assert!((y.get(0, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn where_cond_known() {
    let a: Tensor<f64> = mat![[1.0, 2.0]];
    let b: Tensor<f64> = mat![[10.0, 20.0]];
    let cond: Tensor<f64> = mat![[1.0, 0.0]];
    let y = a.where_cond(&cond, &b);
    assert!((y.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 20.0).abs() < 1e-10);
}

#[test]
fn triu_known() {
    let x: Tensor<f64> = Tensor::fill(3, 3, 1.0);
    let y = x.triu(0);
    assert!((y.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((y.get(1, 0) - 0.0).abs() < 1e-10);
    assert!((y.get(1, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn tril_known() {
    let x: Tensor<f64> = Tensor::fill(3, 3, 1.0);
    let y = x.tril(0);
    assert!((y.get(0, 1) - 0.0).abs() < 1e-10);
    assert!((y.get(1, 0) - 1.0).abs() < 1e-10);
}

#[test]
fn roll_axis1() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0]];
    let y = x.roll(1, 1);
    assert!((y.get(0, 0) - 3.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn flip_axis1() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0]];
    let y = x.flip(1);
    assert!((y.get(0, 0) - 3.0).abs() < 1e-10);
    assert!((y.get(0, 2) - 1.0).abs() < 1e-10);
}

#[test]
fn topk_known() {
    let x: Tensor<f64> = mat![[1.0, 5.0, 3.0, 2.0, 4.0]];
    let (vals, _) = x.topk(3, 1);
    assert_eq!(vals.shape(), (1, 3));
    assert!((vals.get(0, 0) - 5.0).abs() < 1e-10);
    assert!((vals.get(0, 1) - 4.0).abs() < 1e-10);
    assert!((vals.get(0, 2) - 3.0).abs() < 1e-10);
}

#[test]
fn sort_ascending() {
    let x: Tensor<f64> = mat![[3.0, 1.0, 2.0]];
    let (vals, _) = x.sort(1, false);
    assert!((vals.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((vals.get(0, 2) - 3.0).abs() < 1e-10);
}

#[test]
fn meshgrid_known() {
    let x: Tensor<f64> = Tensor::arange(0.0, 1.0, 3);
    let y: Tensor<f64> = Tensor::arange(0.0, 1.0, 2);
    let (gx, gy) = Tensor::meshgrid(&x, &y);
    assert_eq!(gx.shape(), (2, 3));
    assert!((gx.get(0, 1) - 1.0).abs() < 1e-10);
    assert!((gy.get(1, 0) - 1.0).abs() < 1e-10);
}

#[test]
fn rms_norm_known() {
    let x: Tensor<f64> = mat![[3.0, 4.0]];
    let w: Tensor<f64> = mat![[1.0, 1.0]];
    let y = x.rms_norm(1, &w, 1e-8);
    let rms = (12.5_f64).sqrt();
    assert!((y.get(0, 0) - 3.0 / rms).abs() < 1e-4);
    assert!((y.get(0, 1) - 4.0 / rms).abs() < 1e-4);
}

#[test]
fn batch_norm_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let mean: Tensor<f64> = mat![[2.0, 3.0]];
    let var: Tensor<f64> = mat![[1.0, 1.0]];
    let weight: Tensor<f64> = mat![[1.0, 1.0]];
    let bias: Tensor<f64> = mat![[0.0, 0.0]];
    let y = x.batch_norm(&mean, &var, &weight, &bias, 1e-5);
    assert!((y.get(0, 0) - (-1.0)).abs() < 1e-4);
    assert!((y.get(1, 0) - 1.0).abs() < 1e-4);
}

#[test]
fn group_norm_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0, 4.0]];
    let w: Tensor<f64> = mat![[1.0, 1.0, 1.0, 1.0]];
    let b: Tensor<f64> = mat![[0.0, 0.0, 0.0, 0.0]];
    let y = x.group_norm(2, &w, &b, 1e-5);
    assert!((y.get(0, 0) - (-1.0)).abs() < 1e-3);
    assert!((y.get(0, 1) - 1.0).abs() < 1e-3);
}

#[test]
fn bmm_known() {
    let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let b: Tensor<f64> = mat![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
    let c = a.bmm(&b, 2, 2, 2, 2);
    assert_eq!(c.shape(), (4, 2));
    assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((c.get(2, 0) - 5.0).abs() < 1e-10);
}

#[test]
fn addmm_known() {
    let c: Tensor<f64> = Tensor::fill(2, 2, 1.0);
    let a: Tensor<f64> = mat![[1.0, 0.0], [0.0, 1.0]];
    let b: Tensor<f64> = mat![[2.0, 3.0], [4.0, 5.0]];
    let y = c.addmm(&a, &b, 0.5, 2.0);
    assert!((y.get(0, 0) - 4.5).abs() < 1e-10);
    assert!((y.get(0, 1) - 6.5).abs() < 1e-10);
}

#[test]
fn conv1d_known() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0, 4.0, 5.0]];
    let w: Tensor<f64> = mat![[1.0, 1.0, 1.0]];
    let y = x.conv1d(&w, None, 1, 1, 5, 1, 3, 1, 0, 1, 1);
    assert_eq!(y.shape(), (1, 3));
    assert!((y.get(0, 0) - 6.0).abs() < 1e-10);
    assert!((y.get(0, 1) - 9.0).abs() < 1e-10);
    assert!((y.get(0, 2) - 12.0).abs() < 1e-10);
}

#[test]
fn conv2d_simple() {
    let x: Tensor<f64> = Tensor::from_fn(1, 9, |_, c| (c + 1) as f64); // 3x3
    let w: Tensor<f64> = mat![[1.0, 1.0, 1.0, 1.0]]; // 2x2 kernel
    let y = x.conv2d(&w, None, 1, 1, 3, 3, 1, 2, 2, (1,1), (0,0), (1,1), 1);
    assert_eq!(y.shape(), (1, 4));
    assert!((y.get(0, 0) - 12.0).abs() < 1e-10);
    assert!((y.get(0, 3) - 28.0).abs() < 1e-10);
}

#[test]
fn max_pool2d_known() {
    let x: Tensor<f64> = Tensor::from_fn(1, 16, |_, c| (c + 1) as f64);
    let y = x.max_pool2d(4, 4, 2, 2, (2, 2), (0, 0));
    assert_eq!(y.shape(), (1, 4));
    assert!((y.get(0, 0) - 6.0).abs() < 1e-10);
}

#[test]
fn avg_pool2d_known() {
    let x: Tensor<f64> = Tensor::from_fn(1, 4, |_, c| (c + 1) as f64);
    let y = x.avg_pool2d(2, 2, 2, 2, (2, 2), (0, 0));
    assert_eq!(y.shape(), (1, 1));
    assert!((y.get(0, 0) - 2.5).abs() < 1e-10);
}

#[test]
fn adaptive_avg_pool2d_known() {
    let x: Tensor<f64> = Tensor::from_fn(1, 16, |_, c| (c + 1) as f64);
    let y = x.adaptive_avg_pool2d(4, 4, 2, 2);
    assert_eq!(y.shape(), (1, 4));
    assert!((y.get(0, 0) - 3.5).abs() < 1e-10);
}

#[test]
fn max_pool1d_known() {
    let x: Tensor<f64> = mat![[1.0, 3.0, 2.0, 5.0, 4.0]];
    let y = x.max_pool1d(5, 2, 1, 0);
    assert_eq!(y.shape(), (1, 4));
    assert!((y.get(0, 0) - 3.0).abs() < 1e-10);
    assert!((y.get(0, 3) - 5.0).abs() < 1e-10);
}

#[test]
fn embedding_known() {
    let weight: Tensor<f64> = mat![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
    let indices: Tensor<f64> = mat![[2.0, 0.0, 1.0]];
    let y = Tensor::embedding(&indices, &weight);
    assert_eq!(y.shape(), (3, 2));
    assert!((y.get(0, 0) - 0.5).abs() < 1e-10);
    assert!((y.get(1, 0) - 0.1).abs() < 1e-10);
    assert!((y.get(2, 1) - 0.4).abs() < 1e-10);
}

#[test]
fn sdpa_basic() {
    let q: Tensor<f64> = mat![[1.0, 0.0], [0.0, 1.0]];
    let k: Tensor<f64> = mat![[1.0, 0.0], [0.0, 1.0]];
    let v: Tensor<f64> = mat![[10.0, 20.0], [30.0, 40.0]];
    let out = Tensor::scaled_dot_product_attention(&q, &k, &v, None);
    assert_eq!(out.shape(), (2, 2));
    // First query [1,0] attends more to first key → output[0,0] closer to 10
    assert!(out.get(0, 0) < 20.0);
}

#[test]
fn multi_head_attention_basic() {
    let q: Tensor<f64> = Tensor::fill(4, 8, 1.0);
    let k: Tensor<f64> = Tensor::fill(4, 8, 1.0);
    let v: Tensor<f64> = Tensor::fill(4, 8, 1.0);
    let out = Tensor::multi_head_attention(&q, &k, &v, 2, None);
    assert_eq!(out.shape(), (4, 8));
}

#[test]
fn conv_transpose2d_basic() {
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0, 4.0]]; // 2x2
    let w: Tensor<f64> = mat![[1.0, 0.0, 0.0, 1.0]]; // identity-like 2x2 kernel
    let y = x.conv_transpose2d(&w, None, 1, 1, 2, 2, 1, 2, 2, (1,1), (0,0), (0,0));
    assert_eq!(y.shape(), (1, 9)); // 3x3 output
}

#[test]
fn empty_same_as_zeros() {
    let x: Tensor<f64> = Tensor::empty(3, 4);
    assert_eq!(x.shape(), (3, 4));
    assert!((x.get(0, 0) - 0.0).abs() < 1e-10);
}

#[test]
fn cosine_embedding_loss_same() {
    let x: Tensor<f64> = mat![[1.0, 0.0]];
    let y: Tensor<f64> = mat![[1.0, 0.0]];
    let loss = Tensor::cosine_embedding_loss(&x, &y, 1.0, 0.0);
    assert!((loss - 0.0).abs() < 1e-6);
}

// ── P1 ops tests ─────────────────────────────────────────────────────

#[test]
fn conv3d_basic() {
    // 1 batch, 1 channel, 2x2x2 volume, 1 filter, 2x2x2 kernel
    let x: Tensor<f64> = Tensor::from_fn(1, 8, |_r, c| (c + 1) as f64); // 1..8
    let w: Tensor<f64> = Tensor::fill(1, 8, 1.0); // all 1s kernel
    let out = x.conv3d(&w, None, 1, 1, 2, 2, 2, 1, 2, 2, 2, (1,1,1), (0,0,0), (1,1,1), 1);
    assert_eq!(out.shape(), (1, 1)); // single output voxel
    assert!((out.get(0, 0) - 36.0).abs() < 1e-6); // sum 1..8 = 36
}

#[test]
fn conv3d_with_padding() {
    let x: Tensor<f64> = Tensor::from_fn(1, 8, |_r, c| 1.0);
    let w: Tensor<f64> = Tensor::fill(1, 8, 1.0);
    let out = x.conv3d(&w, None, 1, 1, 2, 2, 2, 1, 2, 2, 2, (1,1,1), (1,1,1), (1,1,1), 1);
    // with padding=1 on each side, output size = (2+2-2)/1+1 = 3 per dim
    assert_eq!(out.shape(), (1, 27));
}

#[test]
fn rand_shape_and_range() {
    let t: Tensor<f64> = Tensor::rand(3, 4, 42);
    assert_eq!(t.shape(), (3, 4));
    for r in 0..3 {
        for c in 0..4 {
            let v = t.get(r, c);
            assert!(v >= 0.0 && v <= 1.0, "rand value {v} out of range");
        }
    }
}

#[test]
fn rand_deterministic() {
    let a: Tensor<f64> = Tensor::rand(2, 3, 123);
    let b: Tensor<f64> = Tensor::rand(2, 3, 123);
    for r in 0..2 {
        for c in 0..3 {
            assert!((a.get(r, c) - b.get(r, c)).abs() < 1e-15);
        }
    }
}

#[test]
fn randn_shape_and_stats() {
    let t: Tensor<f64> = Tensor::randn(1, 10000, 42);
    let mean = t.sum_all() / 10000.0;
    assert!(mean.abs() < 0.1, "randn mean {mean} too far from 0");
}

#[test]
fn dropout_training_off() {
    let x: Tensor<f64> = Tensor::fill(2, 3, 1.0);
    let out = x.dropout(0.5, false, 42);
    for r in 0..2 {
        for c in 0..3 {
            assert!((out.get(r, c) - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn dropout_training_on() {
    let x: Tensor<f64> = Tensor::fill(1, 1000, 1.0);
    let out = x.dropout(0.5, true, 42);
    let nonzero = (0..1000).filter(|&c| out.get(0, c).abs() > 1e-12).count();
    // ~50% should survive, check within reasonable range
    assert!(nonzero > 300 && nonzero < 700, "dropout kept {nonzero}/1000");
    // Surviving values should be scaled by 1/(1-p) = 2.0
    for c in 0..1000 {
        let v = out.get(0, c);
        assert!(v.abs() < 1e-12 || (v - 2.0).abs() < 1e-12, "unexpected value {v}");
    }
}

#[test]
fn dropout_p_zero() {
    let x: Tensor<f64> = Tensor::fill(2, 3, 1.0);
    let out = x.dropout(0.0, true, 42);
    assert!((out.sum_all() - 6.0).abs() < 1e-12);
}

#[test]
fn dropout_p_one() {
    let x: Tensor<f64> = Tensor::fill(2, 3, 1.0);
    let out = x.dropout(1.0, true, 42);
    assert!(out.sum_all().abs() < 1e-12);
}

#[test]
fn interpolate_nearest_upsample() {
    // 1 channel, 2x2 -> 4x4
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0, 4.0]]; // 2x2 flattened
    let out = x.interpolate_nearest(2, 2, 4, 4);
    assert_eq!(out.shape(), (1, 16));
    // top-left 2x2 block should all be 1.0 (from pixel (0,0))
    assert!((out.get(0, 0) - 1.0).abs() < 1e-12);
    assert!((out.get(0, 1) - 1.0).abs() < 1e-12);
    assert!((out.get(0, 4) - 1.0).abs() < 1e-12);
}

#[test]
fn interpolate_nearest_downsample() {
    // 1 channel, 4x4 -> 2x2
    let x: Tensor<f64> = Tensor::from_fn(1, 16, |_r, c| c as f64);
    let out = x.interpolate_nearest(4, 4, 2, 2);
    assert_eq!(out.shape(), (1, 4));
}

#[test]
fn interpolate_bilinear_identity() {
    // Same size => should preserve values (approximately)
    let x: Tensor<f64> = mat![[1.0, 2.0, 3.0, 4.0]]; // 2x2
    let out = x.interpolate_bilinear(2, 2, 2, 2);
    assert_eq!(out.shape(), (1, 4));
    for c in 0..4 {
        assert!((out.get(0, c) - x.get(0, c)).abs() < 1e-6, "c={c}");
    }
}

#[test]
fn interpolate_bilinear_upsample() {
    // 1 channel, 2x2 -> 4x4: should be smooth
    let x: Tensor<f64> = mat![[0.0, 1.0, 0.0, 1.0]]; // 2x2: [[0,1],[0,1]]
    let out = x.interpolate_bilinear(2, 2, 4, 4);
    assert_eq!(out.shape(), (1, 16));
}

#[test]
fn unflatten_axis0() {
    let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let b = a.unflatten(0, (2, 2)); // (4,2) -> (2, 2*2) = (2,4)
    assert_eq!(b.shape(), (2, 4));
    assert_eq!(b.get(0, 0), 1.0);
    assert_eq!(b.get(1, 2), 7.0);
}

#[test]
fn unflatten_axis1() {
    let a: Tensor<f64> = mat![[1.0, 2.0, 3.0, 4.0]];
    let b = a.unflatten(1, (2, 2)); // (1,4) -> (1*2, 2) = (2,2)
    assert_eq!(b.shape(), (2, 2));
    assert_eq!(b.get(0, 0), 1.0);
    assert_eq!(b.get(1, 0), 3.0);
}

#[test]
fn contiguous_identity() {
    let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let b = a.contiguous();
    assert_eq!(b.shape(), (2, 2));
    assert_eq!(b.get(0, 1), 2.0);
    assert_eq!(b.get(1, 0), 3.0);
}

#[test]
fn detach_is_independent_copy() {
    let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let b = a.detach();
    assert_eq!(b.shape(), a.shape());
    assert_eq!(b.get(0, 0), a.get(0, 0));
    assert_eq!(b.get(1, 1), a.get(1, 1));
}
