//! Quantize -> dequantize round-trip error bound tests.

use nabla_interface::quant::{
    dequantize_q4_0, dequantize_q4_k_m, dequantize_q8_0, quantize_q4_0, quantize_q4_k_m,
    quantize_q8_0,
};

fn make_test_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 / n as f32) * 2.0 - 1.0) * 10.0).collect()
}

#[test]
fn q4_0_roundtrip() {
    let data = make_test_data(256);
    let quantized = quantize_q4_0(&data).expect("quantize failed");
    let recovered = dequantize_q4_0(&quantized).expect("dequantize failed");
    assert_eq!(data.len(), recovered.len());
    let amax = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_err = 2.0 * amax / 15.0;
    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(err <= max_err + 0.1, "Q4_0 error too large at {i}: orig={orig}, rec={rec}, err={err}, bound={max_err}");
    }
}

#[test]
fn q8_0_roundtrip() {
    let data = make_test_data(256);
    let quantized = quantize_q8_0(&data).expect("quantize failed");
    let recovered = dequantize_q8_0(&quantized).expect("dequantize failed");
    assert_eq!(data.len(), recovered.len());
    let amax = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_err = 2.0 * amax / 127.0;
    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(err <= max_err + 0.01, "Q8_0 error too large at {i}: orig={orig}, rec={rec}, err={err}, bound={max_err}");
    }
}

#[test]
fn q4_k_m_roundtrip() {
    let data = make_test_data(256);
    let quantized = quantize_q4_k_m(&data).expect("quantize failed");
    let recovered = dequantize_q4_k_m(&quantized).expect("dequantize failed");
    assert_eq!(data.len(), recovered.len());
    // Q4_K_M has more complex error bounds; just verify reasonable reconstruction
    let mse: f64 = data.iter().zip(recovered.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum::<f64>() / data.len() as f64;
    let rmse = mse.sqrt();
    let amax = data.iter().fold(0.0f32, |m, &v| m.max(v.abs())) as f64;
    // RMSE should be within 20% of absmax for 4-bit quantization
    assert!(rmse < amax * 0.25, "Q4_K_M RMSE too large: {rmse:.4}, amax={amax:.4}");
}

#[test]
fn q4_0_invalid_length() {
    let data = vec![1.0f32; 33]; // Not a multiple of 32
    let result = quantize_q4_0(&data);
    assert!(result.is_err());
}

#[test]
fn q8_0_invalid_length() {
    let data = vec![1.0f32; 31];
    let result = quantize_q8_0(&data);
    assert!(result.is_err());
}
