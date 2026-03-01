//! Safe Rust wrappers around llama.cpp C FFI.
//!
//! Gated behind `#[cfg(feature = "llama")]`.

use std::ffi::{c_char, c_void, CString};
use std::marker::PhantomData;
use std::ptr;

use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Opaque FFI types
// ---------------------------------------------------------------------------

#[repr(C)]
pub(crate) struct LlamaModelRaw { _opaque: [u8; 0] }
#[repr(C)]
pub(crate) struct LlamaContextRaw { _opaque: [u8; 0] }
#[repr(C)]
pub(crate) struct LlamaSamplerRaw { _opaque: [u8; 0] }
#[repr(C)]
pub(crate) struct LlamaVocabRaw { _opaque: [u8; 0] }

// ---------------------------------------------------------------------------
// FFI structs (repr(C) mirrors llama.h)
// ---------------------------------------------------------------------------

#[repr(C)]
pub(crate) struct LlamaBatchRaw {
    pub n_tokens: i32,
    pub token: *mut i32,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Clone)]
pub(crate) struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
}

#[repr(C)]
#[derive(Clone)]
pub(crate) struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

#[repr(C)]
#[derive(Clone)]
pub(crate) struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

/// Performance timing data from llama.cpp context.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaPerfContextData {
    /// Absolute start time (ms).
    pub t_start_ms: f64,
    /// Model load time (ms).
    pub t_load_ms: f64,
    /// Prompt evaluation time (ms).
    pub t_p_eval_ms: f64,
    /// Token generation time (ms).
    pub t_eval_ms: f64,
    /// Number of prompt tokens evaluated.
    pub n_p_eval: i32,
    /// Number of tokens generated.
    pub n_eval: i32,
    /// Number of graph reuses.
    pub n_reused: i32,
}

// ---------------------------------------------------------------------------
// FFI extern declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();

    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;

    fn llama_model_load_from_file(path: *const c_char, params: LlamaModelParams) -> *mut LlamaModelRaw;
    fn llama_model_free(model: *mut LlamaModelRaw);
    fn llama_model_get_vocab(model: *const LlamaModelRaw) -> *const LlamaVocabRaw;

    fn llama_init_from_model(model: *mut LlamaModelRaw, params: LlamaContextParams) -> *mut LlamaContextRaw;
    fn llama_free(ctx: *mut LlamaContextRaw);

    fn llama_vocab_n_tokens(vocab: *const LlamaVocabRaw) -> i32;
    fn llama_vocab_is_eog(vocab: *const LlamaVocabRaw, token: i32) -> bool;

    fn llama_tokenize(
        vocab: *const LlamaVocabRaw, text: *const c_char, text_len: i32,
        tokens: *mut i32, n_tokens_max: i32, add_special: bool, parse_special: bool,
    ) -> i32;

    fn llama_token_to_piece(
        vocab: *const LlamaVocabRaw, token: i32,
        buf: *mut c_char, length: i32, lstrip: i32, special: bool,
    ) -> i32;

    fn llama_detokenize(
        vocab: *const LlamaVocabRaw, tokens: *const i32, n_tokens: i32,
        text: *mut c_char, text_len_max: i32, remove_special: bool, unparse_special: bool,
    ) -> i32;

    fn llama_decode(ctx: *mut LlamaContextRaw, batch: LlamaBatchRaw) -> i32;
    fn llama_batch_get_one(tokens: *mut i32, n_tokens: i32) -> LlamaBatchRaw;

    fn llama_get_logits_ith(ctx: *mut LlamaContextRaw, i: i32) -> *mut f32;

    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSamplerRaw;
    fn llama_sampler_chain_add(chain: *mut LlamaSamplerRaw, sampler: *mut LlamaSamplerRaw);
    fn llama_sampler_init_temp(t: f32) -> *mut LlamaSamplerRaw;
    fn llama_sampler_init_top_k(k: i32) -> *mut LlamaSamplerRaw;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSamplerRaw;
    fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSamplerRaw;
    fn llama_sampler_sample(smpl: *mut LlamaSamplerRaw, ctx: *mut LlamaContextRaw, idx: i32) -> i32;

    fn llama_perf_context(ctx: *const LlamaContextRaw) -> LlamaPerfContextData;
    fn llama_perf_context_reset(ctx: *mut LlamaContextRaw);
}

// ---------------------------------------------------------------------------
// LlamaBackend — RAII guard for global init/free
// ---------------------------------------------------------------------------

/// RAII guard for llama.cpp backend initialization.
pub struct LlamaBackend(());

impl LlamaBackend {
    /// Initialize the llama.cpp backend. Call once at program start.
    pub fn init() -> Self {
        // SAFETY: llama_backend_init is idempotent and safe to call from Rust.
        unsafe { llama_backend_init(); }
        Self(())
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        // SAFETY: paired with init(), releases global backend state.
        unsafe { llama_backend_free(); }
    }
}

// ---------------------------------------------------------------------------
// LlamaModel — RAII wrapper for llama_model
// ---------------------------------------------------------------------------

/// Loaded GGUF model. Freed on drop.
pub struct LlamaModel {
    ptr: *mut LlamaModelRaw,
}

// SAFETY: LlamaModel contains a raw pointer to a thread-safe C object.
// llama.cpp models are immutable after load and safe to share across threads.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// Load a GGUF model. `n_gpu_layers = -1` offloads all layers to Metal.
    pub fn load(path: &str, n_gpu_layers: i32) -> Result<Self> {
        let c_path = CString::new(path).map_err(|e| Error::Llama(e.to_string()))?;
        // SAFETY: llama_model_default_params returns a valid C struct.
        let mut params = unsafe { llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        // SAFETY: c_path is a valid null-terminated C string, params is valid.
        let ptr = unsafe { llama_model_load_from_file(c_path.as_ptr(), params) };
        if ptr.is_null() {
            return Err(Error::Llama(format!("failed to load model: {path}")));
        }
        Ok(Self { ptr })
    }

    pub(crate) fn vocab(&self) -> *const LlamaVocabRaw {
        // SAFETY: ptr is valid while LlamaModel is alive.
        unsafe { llama_model_get_vocab(self.ptr) }
    }

    pub(crate) fn as_ptr(&self) -> *mut LlamaModelRaw { self.ptr }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by llama_model_load_from_file and is non-null.
        unsafe { llama_model_free(self.ptr); }
    }
}

// ---------------------------------------------------------------------------
// LlamaContext — RAII wrapper for llama_context
// ---------------------------------------------------------------------------

/// Inference context bound to a model.
pub struct LlamaContext<'m> {
    ptr: *mut LlamaContextRaw,
    model: &'m LlamaModel,
    _marker: PhantomData<&'m LlamaModel>,
}

impl<'m> LlamaContext<'m> {
    /// Create an inference context from a loaded model.
    pub fn new(model: &'m LlamaModel, n_ctx: u32, n_batch: u32, n_threads: u32) -> Result<Self> {
        // SAFETY: llama_context_default_params returns a valid C struct.
        let mut params = unsafe { llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        params.n_threads = n_threads as i32;
        params.n_threads_batch = n_threads as i32;
        // SAFETY: model.ptr is valid, params is valid.
        let ptr = unsafe { llama_init_from_model(model.as_ptr(), params) };
        if ptr.is_null() {
            return Err(Error::Llama("failed to create context".into()));
        }
        Ok(Self { ptr, model, _marker: PhantomData })
    }

    /// Tokenize text into token ids.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        let vocab = self.model.vocab();
        let text_bytes = text.as_bytes();
        let text_len = i32::try_from(text_bytes.len())
            .map_err(|_| Error::Llama("text too long".into()))?;
        // First call to determine required buffer size
        // SAFETY: vocab is valid, text_bytes.as_ptr() is valid for text_len bytes.
        let n = unsafe {
            llama_tokenize(vocab, text_bytes.as_ptr().cast(), text_len, ptr::null_mut(), 0, add_bos, false)
        };
        let capacity = if n < 0 { (-n) as usize } else { n as usize };
        let capacity = capacity.max(1);
        let mut tokens = vec![0i32; capacity];
        // SAFETY: tokens buffer is large enough.
        let n2 = unsafe {
            llama_tokenize(
                vocab, text_bytes.as_ptr().cast(), text_len,
                tokens.as_mut_ptr(), tokens.len() as i32, add_bos, false,
            )
        };
        if n2 < 0 {
            return Err(Error::Llama("tokenization failed".into()));
        }
        tokens.truncate(n2 as usize);
        Ok(tokens)
    }

    /// Detokenize tokens back to text.
    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let vocab = self.model.vocab();
        let n_tokens = i32::try_from(tokens.len())
            .map_err(|_| Error::Llama("too many tokens".into()))?;
        let mut buf = vec![0u8; 4096];
        // SAFETY: vocab, tokens, and buf are valid.
        let n = unsafe {
            llama_detokenize(
                vocab, tokens.as_ptr(), n_tokens,
                buf.as_mut_ptr().cast(), buf.len() as i32, false, false,
            )
        };
        if n < 0 {
            let needed = (-n) as usize;
            buf.resize(needed, 0);
            // SAFETY: buf is now large enough.
            let n2 = unsafe {
                llama_detokenize(
                    vocab, tokens.as_ptr(), n_tokens,
                    buf.as_mut_ptr().cast(), buf.len() as i32, false, false,
                )
            };
            if n2 < 0 {
                return Err(Error::Llama("detokenization failed".into()));
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }
        String::from_utf8(buf).map_err(|e| Error::Llama(e.to_string()))
    }

    /// Decode a batch of tokens. Returns 0 on success.
    pub fn decode(&mut self, batch: &LlamaBatch) -> Result<()> {
        // SAFETY: self.ptr and batch are valid. We use batch_get_one for simple usage.
        let mut tokens = batch.tokens.clone();
        let raw = unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        let ret = unsafe { llama_decode(self.ptr, raw) };
        if ret != 0 {
            return Err(Error::Llama(format!("decode failed with code {ret}")));
        }
        Ok(())
    }

    /// Get logits for the token at position `idx` (-1 = last).
    pub fn get_logits_ith(&self, idx: i32) -> Result<&[f32]> {
        let vocab = self.model.vocab();
        // SAFETY: vocab is valid.
        let n_vocab = unsafe { llama_vocab_n_tokens(vocab) } as usize;
        // SAFETY: logits pointer is valid after decode().
        let ptr = unsafe { llama_get_logits_ith(self.ptr, idx) };
        if ptr.is_null() {
            return Err(Error::Llama("logits pointer is null".into()));
        }
        // SAFETY: llama.cpp guarantees n_vocab contiguous floats.
        Ok(unsafe { std::slice::from_raw_parts(ptr, n_vocab) })
    }

    /// Convert a single token to its text piece.
    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        let vocab = self.model.vocab();
        let mut buf = vec![0u8; 256];
        // SAFETY: vocab and buf are valid.
        let n = unsafe {
            llama_token_to_piece(vocab, token, buf.as_mut_ptr().cast(), buf.len() as i32, 0, false)
        };
        if n < 0 {
            let needed = (-n) as usize;
            buf.resize(needed, 0);
            // SAFETY: buf is now large enough.
            let n2 = unsafe {
                llama_token_to_piece(vocab, token, buf.as_mut_ptr().cast(), buf.len() as i32, 0, false)
            };
            if n2 < 0 {
                return Err(Error::Llama("token_to_piece failed".into()));
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }
        String::from_utf8(buf).map_err(|e| Error::Llama(e.to_string()))
    }

    /// Check if a token is end-of-generation.
    pub fn is_eog(&self, token: i32) -> bool {
        // SAFETY: vocab is valid.
        unsafe { llama_vocab_is_eog(self.model.vocab(), token) }
    }

    /// Get performance data from the context.
    pub fn perf(&self) -> LlamaPerfContextData {
        // SAFETY: self.ptr is valid.
        unsafe { llama_perf_context(self.ptr) }
    }

    /// Reset performance counters.
    pub fn perf_reset(&mut self) {
        // SAFETY: self.ptr is valid.
        unsafe { llama_perf_context_reset(self.ptr); }
    }

    pub(crate) fn as_ptr(&self) -> *mut LlamaContextRaw { self.ptr }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by llama_init_from_model and is non-null.
        unsafe { llama_free(self.ptr); }
    }
}

// ---------------------------------------------------------------------------
// LlamaBatch — token batch builder
// ---------------------------------------------------------------------------

/// Builder for token batches submitted to decode.
pub struct LlamaBatch {
    tokens: Vec<i32>,
}

impl LlamaBatch {
    /// Create a new empty batch.
    #[must_use]
    pub fn new() -> Self { Self { tokens: Vec::new() } }

    /// Add tokens to the batch.
    pub fn add_tokens(&mut self, tokens: &[i32]) -> &mut Self {
        self.tokens.extend_from_slice(tokens);
        self
    }

    /// Number of tokens in the batch.
    #[must_use]
    pub fn len(&self) -> usize { self.tokens.len() }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }

    /// Get the tokens slice.
    #[must_use]
    pub fn tokens(&self) -> &[i32] { &self.tokens }
}

impl Default for LlamaBatch {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// SamplerChain — builder for llama sampler pipeline
// ---------------------------------------------------------------------------

/// Sampler chain with temperature, top-k, top-p, and distribution sampling.
pub struct SamplerChain {
    ptr: *mut LlamaSamplerRaw,
}

impl SamplerChain {
    /// Create a new sampler chain builder.
    pub fn new() -> SamplerChainBuilder {
        SamplerChainBuilder { temperature: 0.8, top_k: 40, top_p: 0.95, seed: 0xFFFF_FFFF }
    }

    /// Sample the next token given a context and batch index.
    pub fn sample(&mut self, ctx: &mut LlamaContext<'_>, idx: i32) -> i32 {
        // SAFETY: both pointers are valid.
        unsafe { llama_sampler_sample(self.ptr, ctx.as_ptr(), idx) }
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by llama_sampler_chain_init.
        // Chain owns its sub-samplers, so freeing the chain frees everything.
        // llama.cpp uses llama_sampler_free which handles chain cleanup.
        unsafe { llama_sampler_free(self.ptr); }
    }
}

unsafe extern "C" {
    fn llama_sampler_free(smpl: *mut LlamaSamplerRaw);
}

/// Builder for constructing a [`SamplerChain`].
pub struct SamplerChainBuilder {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    seed: u32,
}

impl SamplerChainBuilder {
    /// Set sampling temperature.
    #[must_use]
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = t; self }
    /// Set top-k sampling parameter.
    #[must_use]
    pub fn top_k(mut self, k: i32) -> Self { self.top_k = k; self }
    /// Set top-p (nucleus) sampling parameter.
    #[must_use]
    pub fn top_p(mut self, p: f32) -> Self { self.top_p = p; self }
    /// Set RNG seed for sampling.
    #[must_use]
    pub fn seed(mut self, s: u32) -> Self { self.seed = s; self }

    /// Build the sampler chain.
    pub fn build(self) -> Result<SamplerChain> {
        // SAFETY: all llama_sampler_init_* functions return valid pointers.
        unsafe {
            let sparams = llama_sampler_chain_default_params();
            let chain = llama_sampler_chain_init(sparams);
            if chain.is_null() {
                return Err(Error::Llama("failed to init sampler chain".into()));
            }
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(self.top_k));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(self.top_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(self.temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(self.seed));
            Ok(SamplerChain { ptr: chain })
        }
    }
}
