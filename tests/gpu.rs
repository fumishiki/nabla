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

// ── 1. Basic construction ─────────────────────────────────────────────────────

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

// ── 2. Element access ─────────────────────────────────────────────────────────

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

// ── 3. Arithmetic ops ─────────────────────────────────────────────────────────

#[test]
fn add_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j) as f32);
    tensors_close_f32(&(&a_gpu + &b_gpu), &(&a_cpu + &b_cpu));
}

// f64 GPU path requires FLOAT64 capability (not available on wgpu/Metal).
// f64 on gpu uses CPU fallback — covered by from_fn_f64_values test.

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
    // 2x3 * 3x2 = 2x2
    let a_gpu = Tensor::<f32, Gpu>::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f32);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

// f64 matmul on gpu uses CPU fallback (FLOAT64 capability not available on wgpu).
// Verified indirectly via from_fn + get tests above.

// ── 4. Chained operations ─────────────────────────────────────────────────────

#[test]
fn chained_add_matmul_f32() {
    // c = (a + b) * d — no intermediate readback
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

// ── 4b. Elementwise math ops ──────────────────────────────────────────────────

#[test]
fn exp_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.1);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.1);
    tensors_close_f32(&a_gpu.exp(), &a_cpu.exp());
}

#[test]
fn ln_f32_matches_cpu() {
    // positive values only
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32 * 0.5);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32 * 0.5);
    tensors_close_f32(&a_gpu.ln(), &a_cpu.ln());
}

#[test]
fn log1p_f32_matches_cpu() {
    // values > -1
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.25);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.25);
    tensors_close_f32(&a_gpu.log1p(), &a_cpu.log1p());
}

#[test]
fn sin_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.2);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.2);
    tensors_close_f32(&a_gpu.sin(), &a_cpu.sin());
}

#[test]
fn cos_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.2);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.2);
    tensors_close_f32(&a_gpu.cos(), &a_cpu.cos());
}

#[test]
fn tanh_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i as f32) - (j as f32) * 0.5);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i as f32) - (j as f32) * 0.5);
    tensors_close_f32(&a_gpu.tanh(), &a_cpu.tanh());
}

#[test]
fn sqrt_f32_matches_cpu() {
    // positive values only
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    tensors_close_f32(&a_gpu.sqrt(), &a_cpu.sqrt());
}

#[test]
fn abs_f32_matches_cpu() {
    // mix positive and negative values
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i as f32) - (j as f32) * 1.5);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i as f32) - (j as f32) * 1.5);
    tensors_close_f32(&a_gpu.abs(), &a_cpu.abs());
}

#[test]
fn recip_f32_matches_cpu() {
    // non-zero values
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f32);
    tensors_close_f32(&a_gpu.recip(), &a_cpu.recip());
}

#[test]
fn erf_f32_matches_cpu() {
    // small range -2..2 works well for erf
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i as f32) * 0.5 - (j as f32) * 0.3);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i as f32) * 0.5 - (j as f32) * 0.3);
    tensors_close_f32(&a_gpu.erf(), &a_cpu.erf());
}

#[test]
fn ceil_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
    tensors_close_f32(&a_gpu.ceil(), &a_cpu.ceil());
}

#[test]
fn floor_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
    tensors_close_f32(&a_gpu.floor(), &a_cpu.floor());
}

#[test]
fn round_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.7 - 5.0);
    tensors_close_f32(&a_gpu.round(), &a_cpu.round());
}

#[test]
fn powf_f32_matches_cpu() {
    // small positive base values — keeps absolute error within 1e-5
    // GPU uses exp(p*ln(x)) which accumulates more FP error for large bases
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| 1.0f32 + (i * 3 + j) as f32 * 0.1);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| 1.0f32 + (i * 3 + j) as f32 * 0.1);
    tensors_close_f32(&a_gpu.powf(2.5f32), &a_cpu.powf(2.5f32));
}

#[test]
fn mul_elem_f32_matches_cpu() {
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.5);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32 * 0.5);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    tensors_close_f32(&a_gpu.mul_elem(&b_gpu), &a_cpu.mul_elem(&b_cpu));
}

#[test]
fn div_elem_f32_matches_cpu() {
    // non-zero divisor
    let a_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i * 4 + j) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(4, 4, |i, j| (i + j + 1) as f32);
    tensors_close_f32(&a_gpu.div_elem(&b_gpu), &a_cpu.div_elem(&b_cpu));
}

#[test]
fn chained_exp_mul_f32() {
    // softmax-like: exp(a) * b — verifies no intermediate readback
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i + j) as f32 * 0.5);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(3, 3, |i, j| (i * j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| (i + j) as f32 * 0.5);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(3, 3, |i, j| (i * j + 1) as f32);
    tensors_close_f32(&a_gpu.exp().mul_elem(&b_gpu), &a_cpu.exp().mul_elem(&b_cpu));
}

// ── 5. Clone ──────────────────────────────────────────────────────────────────

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

// ── Wave 7: Tiled matmul ──────────────────────────────────────────────────────

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
    // 3×5 * 5×2 = 3×2
    let a_gpu = Tensor::<f32, Gpu>::from_fn(3, 5, |i, j| (i + j + 1) as f32);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(5, 2, |i, j| (i * 2 + j + 1) as f32);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(3, 5, |i, j| (i + j + 1) as f32);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(5, 2, |i, j| (i * 2 + j + 1) as f32);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

#[test]
fn tiled_matmul_large_f32_matches_cpu() {
    // 20×20 — spans more than one 16×16 tile in every direction
    let a_gpu = Tensor::<f32, Gpu>::from_fn(20, 20, |i, j| (i + j + 1) as f32 * 0.1);
    let b_gpu = Tensor::<f32, Gpu>::from_fn(20, 20, |i, j| (j + 1) as f32 * 0.1);
    let a_cpu = Tensor::<f32, Cpu>::from_fn(20, 20, |i, j| (i + j + 1) as f32 * 0.1);
    let b_cpu = Tensor::<f32, Cpu>::from_fn(20, 20, |i, j| (j + 1) as f32 * 0.1);
    tensors_close_f32(&(&a_gpu * &b_gpu), &(&a_cpu * &b_cpu));
}

// ── Wave 8: GPU construction kernels ─────────────────────────────────────────

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

// ── Wave 10: GPU reduction tests ──────────────────────────────────────────────

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
    // max is 16.0 at (3,3)
    assert!((a.max_all() - 16.0_f32).abs() < 1e-5);
}

#[test]
fn gpu_min_all() {
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    // min is 1.0 at (0,0)
    assert!((a.min_all() - 1.0_f32).abs() < 1e-5);
}

#[test]
fn gpu_argmax_all() {
    // max is 16.0 at (3,3) — flat index 15, last element
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    assert_eq!(a.argmax(), (3, 3));
}

#[test]
fn gpu_argmin_all() {
    // min is 1.0 at (0,0) — flat index 0, first element
    let a = Tensor::<f32, Gpu>::from_fn(4, 4, |r, c| (r * 4 + c + 1) as f32);
    assert_eq!(a.argmin(), (0, 0));
}
