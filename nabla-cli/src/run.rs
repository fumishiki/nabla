//! `nabla run` — GGUF streaming inference via nabla-interface + llama.cpp.
//!
//! Only compiled when the `llama` feature is enabled.
//!
//! Usage: nabla run <GGUF_PATH> --prompt <TEXT>
//!                  [--max-tokens 256] [--temp 0.8] [--ctx 2048] [--no-stream]

use std::error::Error;
use std::io::Write as _;

use nabla_interface::{InferenceConfig, InferenceEngine, SamplingConfig};

pub fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: nabla run <GGUF_PATH> --prompt <TEXT> [OPTIONS]\n\n\
            Options:\n  \
            --prompt     <TEXT>   Prompt text (required)\n  \
            --max-tokens <N>      Max generated tokens (default: 256)\n  \
            --temp       <F>      Sampling temperature (default: 0.8)\n  \
            --ctx        <N>      Context window size (default: 2048)\n  \
            --gpu-layers <N>      Layers to offload to GPU (-1 = all, default: -1)\n  \
            --no-stream           Collect and print all tokens at end\n\n\
            Requires nabla-cli compiled with --features llama."
        );
        return Ok(());
    }

    let gguf_path = args
        .first()
        .filter(|a| !a.starts_with('-'))
        .ok_or("missing GGUF_PATH argument")?;

    let prompt = flag_str(args, "--prompt").ok_or("--prompt <TEXT> is required")?;
    let max_tokens  = flag_u32(args, "--max-tokens").unwrap_or(256);
    let temperature = flag_f32(args, "--temp").unwrap_or(0.8);
    let n_ctx       = flag_u32(args, "--ctx").unwrap_or(2048);
    let n_gpu       = flag_i32(args, "--gpu-layers").unwrap_or(-1);
    let stream      = !args.iter().any(|a| a == "--no-stream");

    let inference_cfg = InferenceConfig { n_ctx, n_gpu_layers: n_gpu, ..Default::default() };
    let sampling_cfg  = SamplingConfig { temperature, ..Default::default() };

    eprintln!("Loading {}…", gguf_path);
    let mut engine = InferenceEngine::new(gguf_path, inference_cfg)
        .map_err(|e| format!("load model: {e}"))?;

    let start = std::time::Instant::now();

    if stream {
        let token_iter = engine
            .generate_stream(prompt, max_tokens, &sampling_cfg)
            .map_err(|e| format!("generate_stream: {e}"))?;

        for token in token_iter {
            print!("{token}");
            std::io::stdout().flush().ok();
        }
        println!();
    } else {
        let text = engine
            .generate(prompt, max_tokens, &sampling_cfg)
            .map_err(|e| format!("generate: {e}"))?;
        println!("{text}");
    }

    let elapsed = start.elapsed().as_secs_f64();
    let perf = engine.perf();
    eprintln!(
        "─────────────────────────────────────────────",
    );
    eprintln!(
        "Generated {total} tokens in {elapsed:.1} s  \
        (prompt {pp:.1} tok/s  gen {gp:.1} tok/s)",
        total  = perf.total_tokens,
        pp     = perf.prompt_tok_per_sec,
        gp     = perf.gen_tok_per_sec,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Arg helpers
// ---------------------------------------------------------------------------

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())
}

fn flag_u32(args: &[String], flag: &str) -> Option<u32> {
    flag_str(args, flag)?.parse().ok()
}

fn flag_i32(args: &[String], flag: &str) -> Option<i32> {
    flag_str(args, flag)?.parse().ok()
}

fn flag_f32(args: &[String], flag: &str) -> Option<f32> {
    flag_str(args, flag)?.parse().ok()
}
