#![cfg(feature = "cpu")]
#![allow(unused_imports)]

use nabla::cas::{Expr, diff, eval, eval_tensor, simplify};
use nabla::ode::{AdaptiveConfig, dormand_prince, rk4};
use nabla::prelude::*;
use nabla::{between, frange};
use std::collections::HashMap;

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

fn smooth_l1_expected(pred: &[f64], target: &[f64], beta: f64) -> f64 {
    let mut sum = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = p - t;
        let ad = d.abs();
        sum += if ad < beta { 0.5 * d * d / beta } else { ad - 0.5 * beta };
    }
    sum / (pred.len() as f64)
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
    assert!(
        (loss - expected).abs() < 1e-10,
        "loss={loss}, expected={expected}"
    );
}


#[test]
fn cross_entropy_loss_perfect_prediction() {
    // When prediction is very confident and correct, loss approaches 0
    let logits: Tensor<f64> = mat![[-100.0_f64, 100.0]];
    let log_probs = logits.log_softmax(1);
    let target: Tensor<f64> = mat![[0.0_f64, 1.0]];
    let loss = log_probs.cross_entropy_loss(&target);
    assert!(
        loss < 1e-10,
        "loss={loss} should be near 0 for perfect prediction"
    );
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
fn layer_norm_row_zero_mean() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
    let normed = a.layer_norm(1, 1e-8);
    for r in 0..2 {
        let row_mean: f64 = (0..4).map(|c| normed.get(r, c)).sum::<f64>() / 4.0;
        assert!(
            row_mean.abs() < 1e-6,
            "row {r} mean = {row_mean}, expected ~0"
        );
    }
}


#[test]
fn layer_norm_row_unit_variance() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
    let normed = a.layer_norm(1, 1e-8);
    for r in 0..2 {
        let var: f64 = (0..4)
            .map(|c| {
                let x = normed.get(r, c);
                x * x
            })
            .sum::<f64>()
            / 4.0;
        assert!((var - 1.0).abs() < 0.01, "row {r} var = {var}, expected ~1");
    }
}


#[test]
fn softmax_denominator_pattern() {
    let x: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let denom = x.sum_axis_keepdim(1);
    assert_eq!(denom.shape(), (2, 1));
    assert!(approx_eq(denom.get(0, 0), 6.0, 1e-10));
    assert!(approx_eq(denom.get(1, 0), 15.0, 1e-10));
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
fn nll_loss_known() {
    let lp: Tensor<f64> = mat![[-0.5, -1.0, -2.0], [-2.0, -0.3, -1.5], [-1.0, -1.0, -0.2]];
    let targets: Tensor<f64> = mat![[0.0], [1.0], [2.0]];
    assert!((lp.nll_loss(&targets) - 0.3333).abs() < 1e-3);
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
        let y = x.conv2d(&w, None, 1, 1, 3, 3, 1, 2, 2, (1, 1), (0, 0), (1, 1), 1);
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
        let y = x.conv_transpose2d(&w, None, 1, 1, 2, 2, 1, 2, 2, (1, 1), (0, 0), (0, 0));
        assert_eq!(y.shape(), (1, 9)); // 3x3 output
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
        let out = x.conv3d(
            &w,
            None,
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            1,
        );
        assert_eq!(out.shape(), (1, 1)); // single output voxel
        assert!((out.get(0, 0) - 36.0).abs() < 1e-6); // sum 1..8 = 36
    }


    #[test]
    fn conv3d_with_padding() {
        let x: Tensor<f64> = Tensor::from_fn(1, 8, |_r, _c| 1.0);
        let w: Tensor<f64> = Tensor::fill(1, 8, 1.0);
        let out = x.conv3d(
            &w,
            None,
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        );
        // with padding=1 on each side, output size = (2+2-2)/1+1 = 3 per dim
        assert_eq!(out.shape(), (1, 27));
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
    fn mse_loss_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [0.5, 2.5, 2.0, 5.0];
        let a = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| a_data[r * 2 + c]);
        let b = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| b_data[r * 2 + c]);
        let v_a = match tape.variable(a) {
            Ok(v) => v,
            Err(e) => panic!("mse_loss var error: {e}"),
        };
        let v_b = match tape.variable(b) {
            Ok(v) => v,
            Err(e) => panic!("mse_loss var error: {e}"),
        };
        let loss = v_a.mse_loss(&v_b);
        let expected = (0..4)
            .map(|i| {
                let d = a_data[i] - b_data[i];
                d * d
            })
            .sum::<f64>() / 4.0;
        assert!(approx_eq(loss.data().get(0, 0), expected, 1e-10));
    }


    #[test]
    fn l1_loss_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let a_data = [1.0, -2.0, 3.0, -4.0];
        let b_data = [0.5, -1.5, 2.0, -5.0];
        let a = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| a_data[r * 2 + c]);
        let b = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| b_data[r * 2 + c]);
        let v_a = match tape.variable(a) {
            Ok(v) => v,
            Err(e) => panic!("l1_loss var error: {e}"),
        };
        let v_b = match tape.variable(b) {
            Ok(v) => v,
            Err(e) => panic!("l1_loss var error: {e}"),
        };
        let loss = v_a.l1_loss(&v_b);
        let expected = (0..4)
            .map(|i| (a_data[i] - b_data[i]).abs())
            .sum::<f64>() / 4.0;
        assert!(approx_eq(loss.data().get(0, 0), expected, 1e-10));
    }


    #[test]
    fn smooth_l1_loss_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let a_data = [1.0, -2.0, 3.0, -4.0];
        let b_data = [0.5, -1.0, 1.0, -6.0];
        let a = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| a_data[r * 2 + c]);
        let b = Tensor::<f64, Cpu>::from_fn(2, 2, |r, c| b_data[r * 2 + c]);
        let v_a = match tape.variable(a) {
            Ok(v) => v,
            Err(e) => panic!("smooth_l1_loss var error: {e}"),
        };
        let v_b = match tape.variable(b) {
            Ok(v) => v,
            Err(e) => panic!("smooth_l1_loss var error: {e}"),
        };
        let loss = v_a.smooth_l1_loss(&v_b, 1.0);
        let expected = smooth_l1_expected(&a_data, &b_data, 1.0);
        assert!(approx_eq(loss.data().get(0, 0), expected, 1e-10));
    }


    #[test]
    fn nll_loss_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let logp_data = [0.1_f64.ln(), 0.7_f64.ln(), 0.2_f64.ln(),
            0.8_f64.ln(), 0.1_f64.ln(), 0.1_f64.ln()];
        let logp = Tensor::<f64, Cpu>::from_fn(2, 3, |r, c| logp_data[r * 3 + c]);
        let targets = Tensor::<f64, Cpu>::from_fn(2, 1, |r, _| if r == 0 { 1.0 } else { 0.0 });
        let v_logp = match tape.variable(logp) {
            Ok(v) => v,
            Err(e) => panic!("nll_loss var error: {e}"),
        };
        let loss = match v_logp.nll_loss(&targets) {
            Ok(v) => v,
            Err(e) => panic!("nll_loss error: {e}"),
        };
        let expected = -(logp_data[1] + logp_data[3]) / 2.0;
        assert!(approx_eq(loss.data().get(0, 0), expected, 1e-10));
    }


    #[test]
    fn cosine_embedding_loss_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let x1_data = [1.0, 2.0];
        let x2_data = [2.0, 0.0];
        let x1 = Tensor::<f64, Cpu>::from_fn(1, 2, |_, c| x1_data[c]);
        let x2 = Tensor::<f64, Cpu>::from_fn(1, 2, |_, c| x2_data[c]);
        let v_x1 = match tape.variable(x1) {
            Ok(v) => v,
            Err(e) => panic!("cosine_embedding_loss var error: {e}"),
        };
        let v_x2 = match tape.variable(x2) {
            Ok(v) => v,
            Err(e) => panic!("cosine_embedding_loss var error: {e}"),
        };
        let loss = v_x1.cosine_embedding_loss(&v_x2, 1.0, 0.5);
        let dot = x1_data[0] * x2_data[0] + x1_data[1] * x2_data[1];
        let n1 = (x1_data[0] * x1_data[0] + x1_data[1] * x1_data[1]).sqrt();
        let n2 = (x2_data[0] * x2_data[0] + x2_data[1] * x2_data[1]).sqrt();
        let cos = dot / (n1 * n2 + 1e-8);
        let expected = 1.0 - cos;
        assert!(approx_eq(loss.data().get(0, 0), expected, 1e-10));
    }


    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn group_norm_expected(x: &Tensor<f64, Cpu>, groups: usize, eps: f64) -> Tensor<f64, Cpu> {
        let (m, n) = x.shape();
        let g_size = n / groups;
        Tensor::from_fn(m, n, |r, c| {
            let g = c / g_size;
            let g_start = g * g_size;
            let mut mean = 0.0;
            for j in 0..g_size {
                mean += x.get(r, g_start + j);
            }
            mean /= g_size as f64;
            let mut var = 0.0;
            for j in 0..g_size {
                let d = x.get(r, g_start + j) - mean;
                var += d * d;
            }
            var /= g_size as f64;
            (x.get(r, c) - mean) / (var + eps).sqrt()
        })
    }


    #[test]
    fn group_norm_cpu() {
        let tape = Tape::<f64, Cpu>::new();
        let x_data = [1.0, 2.0, 3.0, 4.0,
            2.0, 0.0, -1.0, 1.0];
        let x = Tensor::<f64, Cpu>::from_fn(2, 4, |r, c| x_data[r * 4 + c]);
        let x_ref = x.clone();
        let weight = Tensor::<f64, Cpu>::from_fn(1, 4, |_, _| 1.0);
        let bias = Tensor::<f64, Cpu>::from_fn(1, 4, |_, _| 0.0);
        let v_x = match tape.variable(x) {
            Ok(v) => v,
            Err(e) => panic!("group_norm var error: {e}"),
        };
        let v_w = match tape.variable(weight) {
            Ok(v) => v,
            Err(e) => panic!("group_norm var error: {e}"),
        };
        let v_b = match tape.variable(bias) {
            Ok(v) => v,
            Err(e) => panic!("group_norm var error: {e}"),
        };
        let y = v_x.group_norm(2, &v_w, &v_b, 1e-5);
        let expected = group_norm_expected(&x_ref, 2, 1e-5);
        let (m, n) = y.data().shape();
        for r in 0..m {
            for c in 0..n {
                assert!(approx_eq(y.data().get(r, c), expected.get(r, c), 1e-10));
            }
        }
    }
