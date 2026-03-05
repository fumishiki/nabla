#![cfg(feature = "gpu")]

use nabla::prelude::*;

fn t(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<f32> {
    Tensor::from_vec(data, rows, cols)
}

#[test]
fn wgpu_quick_api_smoke() {
    let a = t(vec![1.0_f32, 2.0, 3.0, 4.0], 2, 2);
    let b = t(vec![4.0_f32, 3.0, 2.0, 1.0], 2, 2);
    let _ = &a + &b;
    let _ = &a - &b;
    let _ = -&a;
    let _ = &a * 2.0;
    let _ = a.exp();
    let _ = a.sum_axis(1);
    let _ = a.cumsum(1);
    let _ = a.reshape(1, 4);
    let _ = a.submatrix(0, 2, 0, 1);
    let _ = a.repeat(2, 1);
    let _ = a.pad([1, 1, 1, 1], 0.0);
    let _ = a.roll(1, 1);
    let _ = a.flip(0);

    let m1 = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m2 = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let mm = &m1 * &m2;
    assert_eq!(mm.shape(), (2, 2));
    let _ = Tensor::matmul_bias(&m1, &m2, &t(vec![1.0_f32, 2.0], 1, 2));

    let x = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
    let gamma = Tensor::ones(1, 4);
    let beta = Tensor::zeros(1, 4);
    let _ = x.softmax(1);
    let _ = x.layer_norm(1, 1e-5);
    let _ = x.rms_norm(1, &gamma, 1e-5);
    let _ = x.group_norm(2, &gamma, &beta, 1e-5);

    let logits = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let targets = t(vec![0.0_f32, 1.0], 2, 1);
    let _ = logits.cross_entropy_fused(&targets);
    let _ = Tensor::embedding(&targets, &t(vec![1.0_f32; 12], 3, 4));

    let img = t(vec![1.0_f32; 32], 2, 16);
    let _ = img.max_pool2d(4, 4, 2, 2, (2, 2), (0, 0));
    let _ = img.avg_pool2d(4, 4, 2, 2, (2, 2), (0, 0));
    let _ = img.adaptive_avg_pool2d(4, 4, 2, 2);
}
