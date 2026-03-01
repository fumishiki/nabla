//! Inference pipeline — tokenize, decode, sample loop.

use crate::Result;
use crate::llama::{LlamaBackend, LlamaBatch, LlamaContext, LlamaModel, SamplerChain};

/// Inference engine configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Context window size.
    pub n_ctx: u32,
    /// Batch size.
    pub n_batch: u32,
    /// Number of threads.
    pub n_threads: u32,
    /// GPU layers (-1 = all on Metal).
    pub n_gpu_layers: i32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_batch: 512,
            n_threads: num_cpus(),
            n_gpu_layers: -1,
        }
    }
}

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k sampling.
    pub top_k: i32,
    /// Top-p (nucleus) sampling.
    pub top_p: f32,
    /// Repeat penalty.
    pub repeat_penalty: f32,
    /// RNG seed (None = random).
    pub seed: Option<u32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: None,
        }
    }
}

/// Performance statistics.
#[derive(Debug, Clone, Copy)]
pub struct PerfStats {
    /// Prompt processing tokens per second.
    pub prompt_tok_per_sec: f64,
    /// Generation tokens per second.
    pub gen_tok_per_sec: f64,
    /// Total tokens processed.
    pub total_tokens: u32,
}

/// High-level inference engine wrapping llama.cpp.
pub struct InferenceEngine {
    _backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    n_batch: u32,
    n_threads: u32,
    last_perf: PerfStats,
}

impl InferenceEngine {
    /// Create a new inference engine from a GGUF file (IF-SERVE-01).
    pub fn new(gguf_path: &str, config: InferenceConfig) -> Result<Self> {
        let backend = LlamaBackend::init();
        let model = LlamaModel::load(gguf_path, config.n_gpu_layers)?;
        Ok(Self {
            _backend: backend,
            model,
            n_ctx: config.n_ctx,
            n_batch: config.n_batch,
            n_threads: config.n_threads,
            last_perf: PerfStats {
                prompt_tok_per_sec: 0.0,
                gen_tok_per_sec: 0.0,
                total_tokens: 0,
            },
        })
    }

    /// Generate text from a prompt (IF-SERVE-02).
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: &SamplingConfig,
    ) -> Result<String> {
        let mut ctx = LlamaContext::new(&self.model, self.n_ctx, self.n_batch, self.n_threads)?;
        let prompt_tokens = ctx.tokenize(prompt, true)?;
        let mut batch = LlamaBatch::new();
        batch.add_tokens(&prompt_tokens);
        ctx.decode(&batch)?;

        let seed = sampling.seed.unwrap_or(0xFFFF_FFFF);
        let mut sampler = SamplerChain::new()
            .temperature(sampling.temperature)
            .top_k(sampling.top_k)
            .top_p(sampling.top_p)
            .seed(seed)
            .build()?;

        let mut output_tokens = Vec::with_capacity(max_tokens as usize);
        for _ in 0..max_tokens {
            let token = sampler.sample(&mut ctx, -1);
            if ctx.is_eog(token) {
                break;
            }
            output_tokens.push(token);
            let mut next_batch = LlamaBatch::new();
            next_batch.add_tokens(&[token]);
            ctx.decode(&next_batch)?;
        }

        self.last_perf = extract_perf(&ctx);
        ctx.detokenize(&output_tokens)
    }

    /// Generate text token-by-token as a streaming iterator (IF-SERVE-03).
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: &SamplingConfig,
    ) -> Result<TokenStream<'_>> {
        let mut ctx = LlamaContext::new(&self.model, self.n_ctx, self.n_batch, self.n_threads)?;
        let prompt_tokens = ctx.tokenize(prompt, true)?;
        let mut batch = LlamaBatch::new();
        batch.add_tokens(&prompt_tokens);
        ctx.decode(&batch)?;

        let seed = sampling.seed.unwrap_or(0xFFFF_FFFF);
        let sampler = SamplerChain::new()
            .temperature(sampling.temperature)
            .top_k(sampling.top_k)
            .top_p(sampling.top_p)
            .seed(seed)
            .build()?;

        Ok(TokenStream {
            ctx,
            sampler,
            remaining: max_tokens,
            done: false,
            perf_out: &mut self.last_perf,
        })
    }

    /// Get performance stats from the last generate/generate_stream call (IF-SERVE-06).
    #[must_use]
    pub fn perf(&self) -> PerfStats {
        self.last_perf
    }
}

fn extract_perf(ctx: &LlamaContext<'_>) -> PerfStats {
    let d = ctx.perf();
    let prompt_tok_per_sec = if d.t_p_eval_ms > 0.0 {
        (d.n_p_eval as f64) / (d.t_p_eval_ms / 1000.0)
    } else {
        0.0
    };
    let gen_tok_per_sec = if d.t_eval_ms > 0.0 {
        (d.n_eval as f64) / (d.t_eval_ms / 1000.0)
    } else {
        0.0
    };
    PerfStats {
        prompt_tok_per_sec,
        gen_tok_per_sec,
        total_tokens: (d.n_p_eval + d.n_eval) as u32,
    }
}

/// Streaming token iterator from [`InferenceEngine::generate_stream`].
pub struct TokenStream<'a> {
    ctx: LlamaContext<'a>,
    sampler: SamplerChain,
    remaining: u32,
    done: bool,
    perf_out: &'a mut PerfStats,
}

impl Iterator for TokenStream<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.done || self.remaining == 0 {
            return None;
        }
        let token = self.sampler.sample(&mut self.ctx, -1);
        if self.ctx.is_eog(token) {
            self.done = true;
            *self.perf_out = extract_perf(&self.ctx);
            return None;
        }
        self.remaining -= 1;
        let piece = self.ctx.token_to_piece(token).ok();
        let mut next_batch = LlamaBatch::new();
        next_batch.add_tokens(&[token]);
        if self.ctx.decode(&next_batch).is_err() {
            self.done = true;
            *self.perf_out = extract_perf(&self.ctx);
            return None;
        }
        if self.remaining == 0 {
            *self.perf_out = extract_perf(&self.ctx);
        }
        piece
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism().map_or(4, |n| n.get() as u32)
}
