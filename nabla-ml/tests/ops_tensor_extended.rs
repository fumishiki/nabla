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
fn cat_axis1_equals_hcat() {
    let a: Tensor<f64> = mat![[1.0_f64], [2.0]];
    let b: Tensor<f64> = mat![[3.0_f64], [4.0]];
    let cat_result = Tensor::cat(&[&a, &b], 1);
    let hcat_result = Tensor::hcat(&[&a, &b]);
    assert_approx_grid(&cat_result, &hcat_result, 1e-10);
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
fn view_same_as_reshape() {
    let a: Tensor<f64> = linear_f64(2, 6);
    let v = a.view(3, 4);
    let r = a.reshape(3, 4);
    assert_approx_grid(&v, &r, 1e-10);
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
fn bce_with_logits_known() {
    let logits: Tensor<f64> = mat![[0.0]];
    let target: Tensor<f64> = mat![[1.0]];
    assert!((logits.bce_with_logits(&target) - std::f64::consts::LN_2).abs() < 1e-3);
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
    fn sparse_sugar_api() -> Result<()> {
        let s = sparse(
            2,
            2,
            &[(0, 0, 4.0_f64), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
        )?;
        let b: Tensor<f64> = mat![[1.0], [2.0]];

        let x_short = s.chol_solve(&b)?;
        let x_long = s.cholesky_solve(Side::Lower, &b)?;
        assert_approx_grid(&x_short, &x_long, 1e-10);

        let d: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
        let y_ref = s.matmul_dense(&d)?;
        let y_borrowed = &s * &d;
        let y_owned = s.clone() * &d;
        assert_approx_grid(&y_borrowed, &y_ref, 1e-10);
        assert_approx_grid(&y_owned, &y_ref, 1e-10);
        Ok(())
    }




    #[allow(dead_code)]
    fn smooth_l1_expected(pred: &[f64], target: &[f64], beta: f64) -> f64 {
        let mut sum = 0.0;
        for (p, t) in pred.iter().zip(target.iter()) {
            let d = p - t;
            let ad = d.abs();
            sum += if ad < beta { 0.5 * d * d / beta } else { ad - 0.5 * beta };
        }
        sum / (pred.len() as f64)
    }

    fn bce_with_logits_expected(logits: &[f64], targets: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (x, y) in logits.iter().zip(targets.iter()) {
            let abs_x = x.abs();
            let relu_x = 0.5 * (x + abs_x);
            sum += relu_x - x * y + (1.0 + (-abs_x).exp()).ln();
        }
        sum / (logits.len() as f64)
    }


    #[test]
    fn bce_with_logits_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let logits_data = [0.0, 1.0, -1.0, 2.0];
        let targets_data = [0.0, 1.0, 1.0, 0.0];
        let logits = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| logits_data[r * 2 + c]);
        let targets = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| targets_data[r * 2 + c]);
        let v_logits = match tape.variable(logits) {
            Ok(v) => v,
            Err(e) => panic!("bce_with_logits var error: {e}"),
        };
        let v_targets = match tape.variable(targets) {
            Ok(v) => v,
            Err(e) => panic!("bce_with_logits var error: {e}"),
        };
        let loss = v_logits.bce_with_logits(&v_targets);
        let expected = bce_with_logits_expected(&logits_data, &targets_data);
        assert!(approx_eq(loss.data().get(0, 0), expected));
    }


    #[test]
    fn topk_cpu() {
        let x = Tensor::<f64, Cpu>::from_fn(2, 4, |r, c| {
            let row0 = [1.0, 3.0, 2.0, 4.0];
            let row1 = [0.5, -1.0, 2.5, 2.0];
            if r == 0 { row0[c] } else { row1[c] }
        });
        let (vals, idxs) = x.topk(2, 1);
        assert_eq!(vals.shape(), (2, 2));
        assert_eq!(idxs.shape(), (2, 2));
        assert!(approx_eq(vals.get(0, 0), 4.0));
        assert!(approx_eq(vals.get(0, 1), 3.0));
        assert!(approx_eq(idxs.get(0, 0), 3.0));
        assert!(approx_eq(idxs.get(0, 1), 1.0));
        assert!(approx_eq(vals.get(1, 0), 2.5));
        assert!(approx_eq(vals.get(1, 1), 2.0));
        assert!(approx_eq(idxs.get(1, 0), 2.0));
        assert!(approx_eq(idxs.get(1, 1), 3.0));
    }


    #[test]
    fn sort_cpu() {
        let x = Tensor::<f64, Cpu>::from_fn(1, 4, |_, c| [3.0, 1.0, 2.0, 0.0][c]);
        let (vals, idxs) = x.sort(1, false);
        assert_eq!(vals.shape(), (1, 4));
        assert_eq!(idxs.shape(), (1, 4));
        assert!(approx_eq(vals.get(0, 0), 0.0));
        assert!(approx_eq(vals.get(0, 1), 1.0));
        assert!(approx_eq(vals.get(0, 2), 2.0));
        assert!(approx_eq(vals.get(0, 3), 3.0));
        assert!(approx_eq(idxs.get(0, 0), 3.0));
        assert!(approx_eq(idxs.get(0, 1), 1.0));
        assert!(approx_eq(idxs.get(0, 2), 2.0));
        assert!(approx_eq(idxs.get(0, 3), 0.0));
    }