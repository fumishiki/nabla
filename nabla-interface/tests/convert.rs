//! Convert module state_dict to GGUF file tests.

use nabla_interface::convert::{export_gguf, map_tensor_name, GgufArchConfig, QuantOverride};
use nabla_interface::quant::GgufQuantType;

#[test]
fn tensor_name_mapping_layers() {
    assert_eq!(map_tensor_name("layers.0.attention.wq.weight"), "blk.0.attn_q.weight");
    assert_eq!(map_tensor_name("layers.5.ffn.w1.weight"), "blk.5.ffn_gate.weight");
    assert_eq!(map_tensor_name("layers.2.ffn.w2.weight"), "blk.2.ffn_down.weight");
    assert_eq!(map_tensor_name("layers.1.attention.wo.weight"), "blk.1.attn_output.weight");
}

#[test]
fn tensor_name_mapping_globals() {
    assert_eq!(map_tensor_name("embedding.weight"), "token_embd.weight");
    assert_eq!(map_tensor_name("norm.weight"), "output_norm.weight");
    assert_eq!(map_tensor_name("output.weight"), "output.weight");
}

#[test]
fn tensor_name_mapping_unknown() {
    assert_eq!(map_tensor_name("custom.layer"), "custom.layer");
}

#[test]
fn export_gguf_two_layer_linear() {
    use nabla_core::tensor::Tensor;

    let w1 = Tensor::<f64, nabla_core::backend::DefaultBackend>::from_fn(4, 8, |r, c| (r * 8 + c) as f64 * 0.01);
    let w2 = Tensor::<f64, nabla_core::backend::DefaultBackend>::from_fn(8, 4, |r, c| (r * 4 + c) as f64 * 0.01);

    let state_dict: Vec<(&str, &Tensor<f64, nabla_core::backend::DefaultBackend>)> = vec![
        ("layers.0.ffn.w1.weight", &w1),
        ("layers.0.ffn.w2.weight", &w2),
    ];

    let config = GgufArchConfig {
        architecture: "llama".into(),
        name: "test-model".into(),
        context_length: 2048,
        embedding_length: 8,
        block_count: 1,
        head_count: 4,
        head_count_kv: 4,
        vocab_size: 32000,
    };

    let tmp = std::env::temp_dir().join("nabla_test_export.gguf");
    export_gguf(&state_dict, &tmp, GgufQuantType::F32, &config, &[]).expect("export failed");

    let metadata = std::fs::metadata(&tmp).expect("file not found");
    assert!(metadata.len() > 100, "GGUF file too small: {} bytes", metadata.len());

    // Verify magic bytes
    let data = std::fs::read(&tmp).expect("read failed");
    assert_eq!(&data[0..4], &[0x47, 0x47, 0x55, 0x46], "bad magic");
    assert_eq!(&data[4..8], &3u32.to_le_bytes(), "bad version");

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn export_with_quant_override() {
    use nabla_core::tensor::Tensor;

    let w = Tensor::<f64, nabla_core::backend::DefaultBackend>::from_fn(32, 32, |r, c| (r * 32 + c) as f64 * 0.001);

    let state_dict: Vec<(&str, &Tensor<f64, nabla_core::backend::DefaultBackend>)> = vec![
        ("layers.0.attention.wq.weight", &w),
    ];

    let config = GgufArchConfig {
        architecture: "llama".into(), name: "test".into(),
        context_length: 2048, embedding_length: 32, block_count: 1,
        head_count: 4, head_count_kv: 4, vocab_size: 32000,
    };

    let overrides = vec![QuantOverride {
        name: "layers.0.attention.wq.weight".into(),
        qtype: GgufQuantType::F16,
    }];

    let tmp = std::env::temp_dir().join("nabla_test_override.gguf");
    export_gguf(&state_dict, &tmp, GgufQuantType::Q8_0, &config, &overrides).expect("export failed");
    assert!(std::fs::metadata(&tmp).expect("no file").len() > 0);
    std::fs::remove_file(&tmp).ok();
}
