#![cfg(feature = "cpu")]

use nabla::cas::{Expr, diff, eval, eval_tensor, simplify};
use nabla::ode::{AdaptiveConfig, dormand_prince, rk4};
use nabla::prelude::*;
use nabla::{between, frange};
use std::collections::HashMap;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
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
fn bcast_all_pipeline() {
    let x: Tensor<f64> = Tensor::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f64);
    let y: Tensor<f64> = bcast_all!(x.sin().powf(2.0); x);
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
fn bcast_and_zip_map() {
    let a: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    let doubled: Tensor<f64> = nabla::bcast!(|x| x * 2.0, &a);
    assert!(approx_eq(doubled.get(0, 0), 2.0));
    assert!(approx_eq(doubled.get(1, 1), 8.0));

    let b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j) as f64);
    let c_b: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * j) as f64);
    let sum: Tensor<f64> = nabla::bcast!(|x, y| x + y, &b, &c_b);
    assert!(approx_eq(sum.get(1, 1), 3.0));

    let scale: Tensor<f64> = Tensor::from_fn(2, 2, |_, _| 2.0);
    let mut out: Tensor<f64> = Tensor::zeros(2, 2);
    nabla::zip_map!(out, |x, y| x * y, &a, &scale);
    assert!(approx_eq(out.get(0, 0), 2.0));
    assert!(approx_eq(out.get(1, 1), 8.0));
}

#[test]
fn par_from_fn_matches_sequential() {
    let seq: Tensor<f64> = Tensor::from_fn(50, 50, |r, c| ((r * 50 + c) as f64).sin());
    let par: Tensor<f64> = Tensor::par_from_fn(50, 50, |r, c| ((r * 50 + c) as f64).sin());
    for r in 0..50 {
        for c in 0..50 {
            assert!(approx_eq(seq.get(r, c), par.get(r, c)));
        }
    }
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
    let b: Tensor<f64> = Tensor::zeros(2, 2);
    let _ = &a * &b;
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

#[test]
#[should_panic(expected = "axis must be 0 or 1")]
fn tensor_dim_out_of_range() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let _ = a.dim(2);
}

#[test]
fn matmul_non_square() {
    let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
    let b: Tensor<f64> = Tensor::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
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
    let r = d.mul_dense(&m).unwrap();
    assert!(approx_eq(r.get(0, 0), 2.0));
    assert!(approx_eq(r.get(0, 1), 4.0));
    assert!(approx_eq(r.get(1, 0), 9.0));
    assert!(approx_eq(r.get(1, 1), 12.0));
}

#[test]
fn symmetric_eigen() {
    let a: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |i, j| [[4.0_f64, 2.0], [2.0, 3.0]][i][j]);
    let sym = Symmetric::new(a, nabla::linalg::Side::Lower).unwrap();
    let evals = sym.eigenvalues().unwrap();
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
    let result = eval(&d, &vars).unwrap();
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
    let t: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
    let mut vars: HashMap<&str, &Tensor<f64>> = HashMap::new();
    vars.insert("x", &t);
    let result = eval_tensor(&s, &vars).unwrap();
    assert!((result.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((result.get(1, 1) - 8.0).abs() < 1e-10);
}

#[test]
fn ode_rk4_accuracy() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let sol = rk4(|_t, y| Ok(-y), &y0, (0.0, 1.0), 0.01).unwrap();
    assert!((sol.final_state().unwrap().get(0, 0) - (-1.0_f64).exp()).abs() < 1e-6);
}

#[test]
fn ode_dormand_prince_adaptive() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let config = AdaptiveConfig {
        dt_init: 0.1,
        ..Default::default()
    };
    let sol = dormand_prince(|_t, y| Ok(-y), &y0, (0.0, 1.0), &config).unwrap();
    assert!((sol.final_state().unwrap().get(0, 0) - (-1.0_f64).exp()).abs() < 1e-4);
    assert!(sol.len() < 100);
}

#[test]
fn ode_error_invalid_inputs() {
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    assert!(rk4(|_t, y| Ok(-y), &y0, (0.0, 1.0), -0.1).is_err());
    assert!(rk4(|_t, y| Ok(-y), &y0, (1.0, 0.0), 0.01).is_err());
}

// --- Wave 12: Backend-generic CAS / ODE ---

#[test]
fn cas_eval_tensor_generic_type() {
    // eval_tensor<T, B> is generic — verify it works with Tensor<f64, Cpu>
    let x = Expr::var("x");
    let expr = Expr::exp(&x);
    let t: Tensor<f64> = Tensor::from_fn(2, 2, |i, j| (i + j + 1) as f64);
    let mut vars: HashMap<&str, &Tensor<f64>> = HashMap::new();
    vars.insert("x", &t);
    let result = eval_tensor(&expr, &vars).unwrap();
    assert!((result.get(0, 0) - 1.0_f64.exp()).abs() < 1e-10);
    assert!((result.get(1, 1) - 3.0_f64.exp()).abs() < 1e-10);
}

#[test]
fn ode_rk4_generic_type() {
    // rk4<T, B> after Dev-2 generic refactor — decays as dy/dt = -y => y(t) = e^(-t)
    let y0: Tensor<f64> = Tensor::from_fn(1, 1, |_, _| 1.0_f64);
    let sol = rk4(|_t, y| Ok(-y), &y0, (0.0, 0.5), 0.01).unwrap();
    assert!((sol.final_state().unwrap().get(0, 0) - (-0.5_f64).exp()).abs() < 1e-5);
}

// --- Wave 13: Reverse-mode autodiff ---

#[test]
fn autograd_simple_backward() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let x = tape.variable(Tensor::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64));
    // f(x) = x .* x  =>  df/dx = 2x (element-wise)
    let y = x.mul_elem(&x);
    y.backward().unwrap();
    let grad = x.grad().unwrap();
    assert!((grad.get(0, 0) - 2.0).abs() < 1e-10); // 2 * 1
    assert!((grad.get(0, 1) - 4.0).abs() < 1e-10); // 2 * 2
    assert!((grad.get(1, 0) - 6.0).abs() < 1e-10); // 2 * 3
    assert!((grad.get(1, 1) - 8.0).abs() < 1e-10); // 2 * 4
}

#[test]
fn autograd_chain_rule() {
    use nabla::autograd::Tape;
    // f(x) = sin(x^2)  =>  f'(x) = 2x * cos(x^2)  at x=1: 2 * cos(1)
    let tape = Tape::<f64, Cpu>::new();
    let x = tape.variable(Tensor::from_fn(1, 1, |_, _| 1.0_f64));
    let x2 = x.mul_elem(&x);
    let y = x2.sin();
    y.backward().unwrap();
    let grad = x.grad().unwrap();
    let expected = 2.0_f64 * 1.0_f64.cos();
    assert!((grad.get(0, 0) - expected).abs() < 1e-10);
}

#[test]
fn autograd_matmul_backward() {
    use nabla::autograd::Tape;
    // C = A @ B  =>  dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
    // With dL/dC = ones(2,2):
    //   grad_a = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]... wait
    //   Actually B^T = [[5,7],[6,8]], ones @ B^T = [[11,15],[11,15]]
    //   grad_b = A^T @ ones = [[1,3],[2,4]]^T... = [[1,2],[3,4]] @ [[1,1],[1,1]] = [[3,3],[7,7]]
    let tape = Tape::<f64, Cpu>::new();
    let a = tape.variable(mat![[1.0_f64, 2.0], [3.0, 4.0]]);
    let b = tape.variable(mat![[5.0_f64, 6.0], [7.0, 8.0]]);
    let c = &a * &b; // matmul
    c.backward().unwrap();
    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();
    // grad_a = ones(2,2) @ B^T where B^T = [[5,7],[6,8]]
    //        = [[5+6, 7+8], [5+6, 7+8]] = [[11, 15], [11, 15]]
    assert!((grad_a.get(0, 0) - 11.0).abs() < 1e-10);
    assert!((grad_a.get(0, 1) - 15.0).abs() < 1e-10);
    assert!((grad_a.get(1, 0) - 11.0).abs() < 1e-10);
    assert!((grad_a.get(1, 1) - 15.0).abs() < 1e-10);
    // grad_b = A^T @ ones(2,2) where A^T = [[1,3],[2,4]]
    //        = [[1+1, 1+1], [3+3, 4+4]]... wait:
    //   A^T = [[1,3],[2,4]], A^T @ ones = [[1+3,1+3],[2+4,2+4]] = [[4,4],[6,6]]
    //   Actually: (A^T)[i,k] * ones[k,j] = sum_k A[k,i] = col-sums of A
    //   col0 of A = [1,3] => sum=4; col1 of A = [2,4] => sum=6
    //   So grad_b = [[4,4],[6,6]]
    assert!((grad_b.get(0, 0) - 4.0).abs() < 1e-10);
    assert!((grad_b.get(0, 1) - 4.0).abs() < 1e-10);
    assert!((grad_b.get(1, 0) - 6.0).abs() < 1e-10);
    assert!((grad_b.get(1, 1) - 6.0).abs() < 1e-10);
}
