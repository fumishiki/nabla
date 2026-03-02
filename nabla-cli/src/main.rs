//! `nabla` CLI — subcommand dispatch.

mod bench;
mod export;
mod info;
mod inspect;
mod serve;
#[cfg(feature = "llama")]
mod run;

use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).map(String::as_str).unwrap_or("--help");
    let rest = if args.len() > 2 { &args[2..] } else { &[] };

    let result: Result<(), Box<dyn std::error::Error>> = match sub {
        "info"    => info::run(rest),
        "bench"   => bench::run(rest),
        "export"  => export::run(rest),
        "inspect" => inspect::run(rest),
        "serve"   => serve::run(rest),
        "run"     => {
            #[cfg(feature = "llama")]
            { run::run(rest) }
            #[cfg(not(feature = "llama"))]
            { Err("`nabla run` requires `--features llama` (link llama.cpp)".into()) }
        }
        "--help" | "-h" | "help" => { print_usage(); Ok(()) }
        cmd => Err(format!("unknown subcommand: `{cmd}`\n\nRun `nabla --help` for usage.").into()),
    };

    if let Err(e) = result {
        eprintln!("nabla: error: {e}");
        process::exit(1);
    }
}

fn print_usage() {
    println!(
        r#"nabla {version}

Usage: nabla <SUBCOMMAND> [OPTIONS]

Subcommands:
  info     Detect hardware backends and display device info
  bench    Run matmul / MLP training-step benchmarks
  inspect  Inspect a nabla checkpoint (tensor shapes, stats)
  export   Export a trained nabla model to GGUF or ONNX
  serve    OpenAI-compatible HTTP inference server (requires --features llama)
  run      Run text generation from a GGUF file (requires --features llama)

Run `nabla <SUBCOMMAND> --help` for subcommand options."#,
        version = env!("CARGO_PKG_VERSION"),
    );
}
