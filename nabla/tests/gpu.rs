//! GPU backend integration tests (wgpu direct).
#![cfg(feature = "gpu")]

use nabla::backend::Cpu;
use nabla::prelude::*;

fn approx_eq_f32(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

fn tensors_close_f32(gpu: &Tensor<f32, Gpu>, cpu: &Tensor<f32, Cpu>) {
    assert_eq!(gpu.shape(), cpu.shape(), "shape mismatch");
    let (r, c) = gpu.shape();
    for i in 0..r {
        for j in 0..c {
            assert!(
                approx_eq_f32(gpu.get(i, j), cpu.get(i, j)),
                "mismatch at ({i},{j}): gpu={} cpu={}",
                gpu.get(i, j),
                cpu.get(i, j)
            );
        }
    }
}

macro_rules! gpu_cpu_unary_test {
    ($name:ident, $op:ident, $rows:expr, $cols:expr, $gen:expr) => {
        #[test]
        fn $name() {
            let a_gpu = Tensor::<f32, Gpu>::from_fn($rows, $cols, $gen);
            let a_cpu = Tensor::<f32, Cpu>::from_fn($rows, $cols, $gen);
            tensors_close_f32(&a_gpu.$op(), &a_cpu.$op());
        }
    };
    ($name:ident, $op:ident ($($arg:expr),*), $rows:expr, $cols:expr, $gen:expr) => {
        #[test]
        fn $name() {
            let a_gpu = Tensor::<f32, Gpu>::from_fn($rows, $cols, $gen);
            let a_cpu = Tensor::<f32, Cpu>::from_fn($rows, $cols, $gen);
            tensors_close_f32(&a_gpu.$op($($arg),*), &a_cpu.$op($($arg),*));
        }
    };
}

gpu_cpu_unary_test!(exp_f32_matches_cpu, exp, 4, 4, |i, j| (i * 4 + j) as f32 * 0.1);
gpu_cpu_unary_test!(ln_f32_matches_cpu, ln, 4, 4, |i, j| (i * 4 + j + 1) as f32 * 0.5);
gpu_cpu_unary_test!(log1p_f32_matches_cpu, log1p, 4, 4, |i, j| (i * 4 + j) as f32 * 0.25);
gpu_cpu_unary_test!(sin_f32_matches_cpu, sin, 4, 4, |i, j| (i * 4 + j) as f32 * 0.2);
gpu_cpu_unary_test!(cos_f32_matches_cpu, cos, 4, 4, |i, j| (i * 4 + j) as f32 * 0.2);
gpu_cpu_unary_test!(tanh_f32_matches_cpu, tanh, 4, 4, |i, j| (i as f32) - (j as f32) * 0.5);
gpu_cpu_unary_test!(sqrt_f32_matches_cpu, sqrt, 4, 4, |i, j| (i * 4 + j + 1) as f32);
gpu_cpu_unary_test!(abs_f32_matches_cpu, abs, 4, 4, |i, j| (i as f32) - (j as f32) * 1.5);
gpu_cpu_unary_test!(recip_f32_matches_cpu, recip, 4, 4, |i, j| (i * 4 + j + 1) as f32);
gpu_cpu_unary_test!(erf_f32_matches_cpu, erf, 4, 4, |i, j| (i as f32) * 0.5 - (j as f32) * 0.3);
gpu_cpu_unary_test!(ceil_f32_matches_cpu, ceil, 4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
gpu_cpu_unary_test!(floor_f32_matches_cpu, floor, 4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
gpu_cpu_unary_test!(round_f32_matches_cpu, round, 4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
gpu_cpu_unary_test!(powf_f32_matches_cpu, powf(2.5f32), 3, 3, |i, j| 1.0f32 + (i * 3 + j) as f32 * 0.1);

#[test]
fn zeros_f32_shape_and_elements() {
    let t = Tensor::<f32, Gpu>::zeros(3, 4);
    assert_eq!(t.shape(), (3, 4));
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(t.get(i, j), 0.0f32);
        }
    }
}

#[test]
fn from_fn_f64_values() {
    let t = Tensor::<f64, Gpu>::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
    assert_eq!(t.shape(), (2, 3));
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(t.get(i, j), (i * 3 + j) as f64);
        }
    }
}

#[test]
fn get_after_from_fn() {
    let t = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i as f32) * 10.0 + j as f32);
    assert_eq!(t.get(2, 3), 23.0f32);
    assert_eq!(t.get(0, 0), 0.0f32);
}

#[test]
fn set_then_get() {
    let mut t = Tensor::<f32, Gpu>::zeros(3, 3);
    t.set(1, 2, 42.0f32);
    assert_eq!(t.get(1, 2), 42.0f32);
    assert_eq!(t.get(0, 0), 0.0f32);
}

#[test]
fn add_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j) as f32);
    tensors_close_f32(&(&a_gpu + &b_gpu), &(&a_cpu + &b_cpu));
}

#[test]
fn sub_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |_i, j| (j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |_i, j| (j + 1) as f32);
    tensors_close_f32(&(&a_gpu - &b_gpu), &(&a_cpu - &b_cpu));
}

#[test]
fn neg_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 4, |i, j| (i as f32) - (j as f32) * 0.5);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 4, |i, j| (i as f32) - (j as f32) * 0.5);
    tensors_close_f32(&(-&a_gpu), &(-&a_cpu));
}

#[test]
fn scale_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    tensors_close_f32(&(&a_gpu * 2.0f32), &(&a_cpu * 2.0f32));
}

#[test]
fn transpose_f32_shape_and_values() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 5, |i, j| (i * 5 + j) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 5, |i, j| (i * 5 + j) as f32);
    let t_gpu = a_gpu.t();
    let t_cpu = a_cpu.t();
    assert_eq!(t_gpu.shape(), (5, 3));
    tensors_close_f32(&t_gpu, &t_cpu);
}

#[test]
fn matmul_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f32);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

#[test]
fn chained_add_matmul_f32() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i + j) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i * j) as f32);
    let d_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 1.0 });

    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| (i + j) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| (i * j) as f32);
    let d_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 1.0 });

    let c_gpu = &(&a_gpu + &b_gpu) * &d_gpu;
    let c_cpu = &(&a_cpu + &b_cpu) * &d_cpu;
    tensors_close_f32(&c_gpu, &c_cpu);
}

#[test]
fn emul_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.5);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.5);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    tensors_close_f32(&a_gpu.emul(&b_gpu), &a_cpu.emul(&b_cpu));
}

#[test]
fn ediv_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    tensors_close_f32(&a_gpu.ediv(&b_gpu), &a_cpu.ediv(&b_cpu));
}

#[test]
fn chained_exp_mul_f32() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i + j) as f32 * 0.5);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i * j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| (i + j) as f32 * 0.5);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| (i * j + 1) as f32);
    tensors_close_f32(&a_gpu.exp().emul(&b_gpu), &a_cpu.exp().emul(&b_cpu));
}

#[test]
fn clone_produces_equal_tensor() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b = a.clone();
    assert_eq!(a.shape(), b.shape());
    let (r, c) = a.shape();
    for i in 0..r {
        for j in 0..c {
            assert_eq!(a.get(i, j), b.get(i, j));
        }
    }
}

#[test]
fn clone_is_independent() {
    let a = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i * 3 + j) as f32);
    let mut b = a.clone();
    b.set(0, 0, 99.0);
    assert_eq!(a.get(0, 0), 0.0f32);
    assert_eq!(b.get(0, 0), 99.0f32);
}

#[test]
fn tiled_matmul_square_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j * 2 + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j * 2 + 1) as f32);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

#[test]
fn tiled_matmul_non_square_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 5, |i, j| (i + j + 1) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(5, 2, |i, j| (i * 2 + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 5, |i, j| (i + j + 1) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(5, 2, |i, j| (i * 2 + j + 1) as f32);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

#[test]
fn tiled_matmul_large_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(20, 20, |i, j| (i + j + 1) as f32 * 0.1);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(20, 20, |i, j| (j + 1) as f32 * 0.1);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(20, 20, |i, j| (i + j + 1) as f32 * 0.1);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(20, 20, |i, j| (j + 1) as f32 * 0.1);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

#[test]
fn gpu_zeros_kernel_f32() {
    let a = Tensor::<f32, Gpu>::zeros(3, 4);
    assert_eq!(a.shape(), (3, 4));
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(a.get(i, j), 0.0_f32);
        }
    }
}

#[test]
fn gpu_fill_f32() {
    let a = Tensor::<f32, Gpu>::fill(3, 4, 5.5_f32);
    assert_eq!(a.shape(), (3, 4));
    for i in 0..3 {
        for j in 0..4 {
            assert!((a.get(i, j) - 5.5_f32).abs() < 1e-6);
        }
    }
}

#[test]
fn gpu_identity_f32() {
    let eye = Tensor::<f32, Gpu>::identity(4);
    assert_eq!(eye.shape(), (4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0_f32 } else { 0.0_f32 };
            assert!((eye.get(i, j) - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn gpu_sum_all() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    let sum = a.sum_all();
    let expected: f32 = (1..=16).map(|x| x as f32).sum();
    assert!((sum - expected).abs() < 1e-4);
}

#[test]
fn gpu_max_all() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    assert!((a.max_all() - 16.0_f32).abs() < 1e-5);
}

#[test]
fn gpu_min_all() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    assert!((a.min_all() - 1.0_f32).abs() < 1e-5);
}

#[test]
fn gpu_argmax_all() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    assert_eq!(a.argmax(), (3, 3));
}

#[test]
fn gpu_argmin_all() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    assert_eq!(a.argmin(), (0, 0));
}
