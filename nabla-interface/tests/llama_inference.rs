//! llama.cpp inference test (requires `brew install llama.cpp`).

#![cfg(feature = "llama")]

#[test]
#[ignore]
fn generate_text() {
    use nabla_interface::{InferenceConfig, InferenceEngine, SamplingConfig};

    let gguf_path =
        std::env::var("NABLA_TEST_GGUF").unwrap_or_else(|_| "test-model.gguf".to_string());

    if !std::path::Path::new(&gguf_path).exists() {
        eprintln!("Skipping: GGUF file not found at {gguf_path}");
        return;
    }

    let config = InferenceConfig {
        n_ctx: 512,
        n_batch: 128,
        n_threads: 4,
        n_gpu_layers: -1,
    };
    let mut engine = InferenceEngine::new(&gguf_path, config).expect("engine creation failed");

    let output = engine
        .generate("Hello", 32, &SamplingConfig::default())
        .expect("generate failed");
    assert!(!output.is_empty(), "generated text is empty");

    let stats = engine.perf();
    assert!(stats.total_tokens > 0, "no tokens generated");
}
