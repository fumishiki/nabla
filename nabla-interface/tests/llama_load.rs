//! llama.cpp GGUF load test (requires `brew install llama.cpp`).

#![cfg(feature = "llama")]

#[test]
#[ignore]
fn load_gguf_and_tokenize() {
    use nabla_interface::llama::{LlamaBackend, LlamaContext, LlamaModel};

    let _backend = LlamaBackend::init();

    // This test requires a GGUF file at the test path
    let gguf_path = std::env::var("NABLA_TEST_GGUF")
        .unwrap_or_else(|_| "test-model.gguf".to_string());

    if !std::path::Path::new(&gguf_path).exists() {
        eprintln!("Skipping: GGUF file not found at {gguf_path}");
        eprintln!("Set NABLA_TEST_GGUF env var to a valid GGUF path");
        return;
    }

    let model = LlamaModel::load(&gguf_path, -1).expect("model load failed");
    let ctx = LlamaContext::new(&model, 512, 128, 4).expect("context failed");

    let tokens = ctx.tokenize("Hello world", true).expect("tokenize failed");
    assert!(!tokens.is_empty(), "tokenization produced no tokens");

    let text = ctx.detokenize(&tokens).expect("detokenize failed");
    assert!(!text.is_empty(), "detokenized text is empty");
}
