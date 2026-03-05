#![cfg(feature = "gpu")]

use nabla::prelude::*;

fn t(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<f32> {
    Tensor::from_vec(data, rows, cols)
}

#[test]
fn wgpu_feature_matrix_and_shape_smoke() {
    let a = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let c = &a * &b;
    assert_eq!(c.shape(), (2, 2));

    let d = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let e = a.matmul_tn(&d);
    assert_eq!(e.shape(), (3, 2));

    let f = t(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let g = a.matmul_nt(&f);
    assert_eq!(g.shape(), (2, 2));

    let mut target = Tensor::<f32>::zeros(4, 4);
    target.slice_set(1..3, 1..4, &a);
    let _ = a.submatrix(0, 2, 0, 2);
    let _ = a.repeat(2, 2);
    let _ = a.pad([1, 1, 1, 1], 0.0);
    let _ = a.roll(1, 1);
    let _ = a.flip(0);
    let _ = Tensor::<f32>::from_diag(&t(vec![1.0_f32, 2.0, 3.0], 3, 1));
}
