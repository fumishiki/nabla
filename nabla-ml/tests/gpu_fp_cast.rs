#[cfg(feature = "cuda")]
mod cuda_fp_cast {
    use half::f16;
    use nabla::backend::Cuda;
    use nabla::prelude::*;
    use nabla::scalar::{Fp4E2M1, Fp8E4M3, Fp8E5M2};

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gpu_fp_cast_and_quant() {
        let data: Vec<f32> = vec![-1.5, -0.75, -0.25, 0.0, 0.25, 0.75, 1.5, 2.0];
        let x: Tensor<f32, Cuda> = Tensor::from_vec(data.clone(), 1, 8);

        let f16_t = x.cast::<f16>();
        let f16_back = f16_t.cast::<f32>().to_vec();
        for (i, v) in f16_back.iter().enumerate() {
            assert!(approx_eq(*v, data[i], 1e-2), "f16 cast mismatch at {i}");
        }

        let fp8e4 = x.quantize_fp8_e4m3();
        let fp8e4_back = fp8e4.dequantize_fp8_e4m3().to_vec();
        for (i, v) in fp8e4_back.iter().enumerate() {
            assert!(approx_eq(*v, data[i], 0.5), "fp8e4m3 mismatch at {i}");
        }

        let fp8e5 = x.quantize_fp8_e5m2();
        let fp8e5_back = fp8e5.dequantize_fp8_e5m2().to_vec();
        for (i, v) in fp8e5_back.iter().enumerate() {
            assert!(approx_eq(*v, data[i], 0.75), "fp8e5m2 mismatch at {i}");
        }

        let (fp4, scales) = x.quantize_fp4_blockwise(4);
        let fp4_back = fp4.dequantize_fp4_blockwise(&scales, 4).to_vec();
        for (i, v) in fp4_back.iter().enumerate() {
            assert!(approx_eq(*v, data[i], 1.5), "fp4 blockwise mismatch at {i}");
        }

        let fp4_cast = x.cast::<Fp4E2M1>().cast::<f32>().to_vec();
        for (i, v) in fp4_cast.iter().enumerate() {
            assert!(approx_eq(*v, data[i], 2.0), "fp4 cast mismatch at {i}");
        }
    }
}
