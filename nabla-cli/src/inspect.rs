//! `nabla inspect` — inspect a nabla checkpoint.
//!
//! Usage: nabla inspect <MODEL_PATH> [--filter <pattern>] [--json] [--no-stats]

use std::error::Error;
use std::path::Path;

pub fn run(args: &[String]) -> std::result::Result<(), Box<dyn Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: nabla inspect <MODEL_PATH> [OPTIONS]\n\n\
            Options:\n  \
            --filter  <PAT>   Only show tensors whose name contains PAT\n  \
            --no-stats        Skip min/max/mean/std (faster for large models)\n  \
            --json            Emit JSON array"
        );
        return Ok(());
    }

    let model_path = args
        .first()
        .filter(|a| !a.starts_with('-'))
        .ok_or("missing MODEL_PATH — usage: nabla inspect <path>")?;
    let model_path = Path::new(model_path);

    let filter     = flag_str(args, "--filter");
    let no_stats   = args.iter().any(|a| a == "--no-stats");
    let json       = args.iter().any(|a| a == "--json");

    use nabla::module::load_tensors;
    use nabla_core::backend::DefaultBackend;

    let tensors: Vec<(String, nabla_core::tensor::Tensor<f64, DefaultBackend>)> =
        load_tensors(model_path).map_err(|e| format!("load_tensors: {e}"))?;

    let entries: Vec<TensorEntry> = tensors
        .iter()
        .filter(|(name, _)| filter.map_or(true, |f| name.contains(f)))
        .map(|(name, t)| {
            let (rows, cols) = t.shape();
            let numel = rows * cols;
            let stats = if no_stats {
                None
            } else {
                let data = t.to_vec();
                Some(compute_stats(&data))
            };
            TensorEntry { name: name.clone(), rows, cols, numel, stats }
        })
        .collect();

    if json {
        print_json(&entries);
    } else {
        print_table(&entries, model_path, no_stats);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct Stats {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
}

fn compute_stats(data: &[f64]) -> Stats {
    let n = data.len() as f64;
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    Stats { min, max, mean, std: variance.sqrt() }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

struct TensorEntry {
    name: String,
    rows: usize,
    cols: usize,
    numel: usize,
    stats: Option<Stats>,
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_table(entries: &[TensorEntry], path: &Path, no_stats: bool) {
    let total_params: usize = entries.iter().map(|e| e.numel).sum();
    println!("checkpoint: {}", path.display());
    println!("tensors   : {}", entries.len());
    println!("parameters: {}", fmt_param_count(total_params));
    println!();

    if no_stats {
        println!("{:<40} {:>16}  {:>12}", "name", "shape", "params");
        println!("{}", "─".repeat(72));
        for e in entries {
            println!(
                "{:<40} {:>16}  {:>12}",
                truncate(&e.name, 40),
                format!("[{}×{}]", e.rows, e.cols),
                fmt_param_count(e.numel),
            );
        }
    } else {
        println!(
            "{:<40} {:>14}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}",
            "name", "shape", "params", "min", "max", "mean", "std"
        );
        println!("{}", "─".repeat(113));
        for e in entries {
            let (mn, mx, me, st) = e.stats.as_ref().map_or(
                ("—".into(), "—".into(), "—".into(), "—".into()),
                |s| (fmt_f(s.min), fmt_f(s.max), fmt_f(s.mean), fmt_f(s.std)),
            );
            println!(
                "{:<40} {:>14}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}",
                truncate(&e.name, 40),
                format!("[{}×{}]", e.rows, e.cols),
                fmt_param_count(e.numel),
                mn, mx, me, st,
            );
        }
    }
}

fn print_json(entries: &[TensorEntry]) {
    print!("[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 { print!(","); }
        if let Some(s) = &e.stats {
            print!(
                "{{\"name\":{:?},\"shape\":[{},{}],\"numel\":{},\
                \"min\":{:.6},\"max\":{:.6},\"mean\":{:.6},\"std\":{:.6}}}",
                e.name, e.rows, e.cols, e.numel, s.min, s.max, s.mean, s.std
            );
        } else {
            print!(
                "{{\"name\":{:?},\"shape\":[{},{}],\"numel\":{}}}",
                e.name, e.rows, e.cols, e.numel
            );
        }
    }
    println!("]");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_f(v: f64) -> String {
    format!("{v:.4e}")
}

fn fmt_param_count(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1e6) }
    else if n >= 1_000     { format!("{:.1}K", n as f64 / 1e3) }
    else                   { n.to_string() }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_owned() }
    else { format!("{}…", &s[..max - 1]) }
}

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}
