#[cfg(feature = "cpu")]
mod cpu {
    #![allow(unused_imports)]
use nabla::cas::{Expr, diff, eval, eval_tensor, simplify};
use nabla::ode::{AdaptiveConfig, dormand_prince, rk4};
use nabla::prelude::*;
use nabla::{between, frange};
use std::collections::HashMap;

#[allow(dead_code)]
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

#[allow(dead_code)]
fn linear_f64(rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::from_fn(rows, cols, |i, j| (i * cols + j + 1) as f64)
}

#[allow(dead_code)]
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
    fn empty_same_as_zeros() {
        let x: Tensor<f64> = Tensor::empty(3, 4);
        assert_eq!(x.shape(), (3, 4));
        assert!((x.get(0, 0) - 0.0).abs() < 1e-10);
    }


    #[test]
    fn rand_shape_and_range() {
        let t: Tensor<f64> = Tensor::rand(3, 4, 42);
        assert_eq!(t.shape(), (3, 4));
        for r in 0..3 {
            for c in 0..4 {
                let v = t.get(r, c);
                assert!((0.0..=1.0).contains(&v), "rand value {v} out of range");
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
        assert!(
            nonzero > 300 && nonzero < 700,
            "dropout kept {nonzero}/1000"
        );
        // Surviving values should be scaled by 1/(1-p) = 2.0
        for c in 0..1000 {
            let v = out.get(0, c);
            assert!(
                v.abs() < 1e-12 || (v - 2.0).abs() < 1e-12,
                "unexpected value {v}"
            );
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


    #[test]
    fn free_construction_aliases() {
        let z: Tensor<f64> = zeros(2, 3);
        assert_eq!(z.shape(), (2, 3));
        assert!(z.as_slice().iter().all(|&v| v == 0.0));

        let o: Tensor<f64> = ones(2, 3);
        assert!(o.as_slice().iter().all(|&v| v == 1.0));

        let f: Tensor<f64> = fill(2, 3, 2.5);
        assert!(f.as_slice().iter().all(|&v| v == 2.5));

        let id: Tensor<f64> = eye(3);
        assert_eq!(id.shape(), (3, 3));
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(approx_eq(id.get(r, c), expected));
            }
        }

        let made: Tensor<f64> = from_fn(2, 2, |r, c| (r * 2 + c) as f64);
        assert!(approx_eq(made.get(1, 1), 3.0));

        let nd = nd_zeros::<f64>(&[2, 3, 4]);
        assert_eq!(nd.ndim(), 3);
        assert_eq!(nd.dim(0), 2);
        assert_eq!(nd.dim(1), 3);
        assert_eq!(nd.dim(2), 4);
        assert!(approx_eq(nd.get_nd(&[1, 2, 3]), 0.0));

        let r: Tensor<f64> = arange(0.0_f64, 1.0, 0.25);
        assert_eq!(r.shape(), (1, 4));
        assert!(approx_eq(r.get(0, 0), 0.0));
        assert!(approx_eq(r.get(0, 1), 0.25));
        assert!(approx_eq(r.get(0, 2), 0.5));
        assert!(approx_eq(r.get(0, 3), 0.75));
    }

}

#[cfg(feature = "gpu")]
mod gpu {
    //! GPU backend integration tests (wgpu direct).

    use nabla::backend::Cpu;
    use nabla::prelude::*;

    const TOL_F32: f32 = 1e-5;

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL_F32
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

    gpu_cpu_unary_test!(exp_f32_matches_cpu, exp, 4, 4, |i, j| (i * 4 + j) as f32
        * 0.1);
    gpu_cpu_unary_test!(ln_f32_matches_cpu, ln, 4, 4, |i, j| (i * 4 + j + 1) as f32
        * 0.5);
    gpu_cpu_unary_test!(log1p_f32_matches_cpu, log1p, 4, 4, |i, j| (i * 4 + j)
        as f32
        * 0.25);
    gpu_cpu_unary_test!(sin_f32_matches_cpu, sin, 4, 4, |i, j| (i * 4 + j) as f32
        * 0.2);
    gpu_cpu_unary_test!(cos_f32_matches_cpu, cos, 4, 4, |i, j| (i * 4 + j) as f32
        * 0.2);
    gpu_cpu_unary_test!(tanh_f32_matches_cpu, tanh, 4, 4, |i, j| (i as f32)
        - (j as f32) * 0.5);
    gpu_cpu_unary_test!(sqrt_f32_matches_cpu, sqrt, 4, 4, |i, j| (i * 4 + j + 1)
        as f32);
    gpu_cpu_unary_test!(abs_f32_matches_cpu, abs, 4, 4, |i, j| (i as f32)
        - (j as f32) * 1.5);
    gpu_cpu_unary_test!(recip_f32_matches_cpu, recip, 4, 4, |i, j| (i * 4 + j + 1)
        as f32);
    gpu_cpu_unary_test!(erf_f32_matches_cpu, erf, 4, 4, |i, j| (i as f32) * 0.5
        - (j as f32) * 0.3);
    gpu_cpu_unary_test!(ceil_f32_matches_cpu, ceil, 4, 4, |i, j| (i * 4 + j) as f32
        * 0.7
        - 5.0);
    gpu_cpu_unary_test!(floor_f32_matches_cpu, floor, 4, 4, |i, j| (i * 4 + j)
        as f32
        * 0.7
        - 5.0);
    gpu_cpu_unary_test!(round_f32_matches_cpu, round, 4, 4, |i, j| (i * 4 + j)
        as f32
        * 0.7
        - 5.0);
    gpu_cpu_unary_test!(powf_f32_matches_cpu, powf(2.5f32), 3, 3, |i, j| 1.0f32
        + (i * 3 + j) as f32 * 0.1);

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
}
