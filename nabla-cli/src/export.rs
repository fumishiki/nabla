//! `nabla export` — convert a nabla checkpoint to GGUF or ONNX.
//!
//! Usage: `nabla export <MODEL_PATH> --format gguf|onnx`
//!        `[--quant Q4_K_M] [--out PATH] [--arch NAME]`

use std::error::Error;
use std::path::{Path, PathBuf};

pub fn run(args: &[String]) -> std::result::Result<(), Box<dyn Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: nabla export <MODEL_PATH> --format gguf|onnx [OPTIONS]\n\n\
            Options:\n  \
            --format   gguf|onnx            (required)\n  \
            --quant    <TYPE>               GGUF quant type (default: Q4_K_M)\n  \
            --imatrix  <PATH>               Importance matrix for IQ4_NL/IQ4_XS quantization\n  \
            --out      <PATH>               Output path (default: <model>.<format>)\n  \
            --arch     <NAME>               GGUF architecture tag (default: generic)\n  \
            --ctx-len  <N>                  Context length metadata (default: 0)\n  \
            --emb-len  <N>                  Embedding length metadata (default: 0)\n  \
            --blocks   <N>                  Block count metadata (default: 0)\n  \
            --heads    <N>                  Attention head count (default: 0)"
        );
        return Ok(());
    }

    let model_path = args
        .first()
        .filter(|a| !a.starts_with('-'))
        .ok_or("missing MODEL_PATH argument")?;
    let model_path = Path::new(model_path);

    let format = flag_str(args, "--format").ok_or("--format gguf|onnx is required")?;
    let out_path = flag_str(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
            PathBuf::from(format!("{stem}.{format}"))
        });

    match format {
        "gguf" => export_gguf(args, model_path, &out_path),
        "onnx" => export_onnx(model_path, &out_path),
        other => Err(format!("unknown format `{other}` — use gguf or onnx").into()),
    }
}

// ---------------------------------------------------------------------------
// GGUF export
// ---------------------------------------------------------------------------

fn export_gguf(
    args: &[String],
    model_path: &Path,
    out_path: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    use nabla_interface::quant::GgufQuantType;
    use nabla_interface::{GgufArchConfig, Imatrix, QuantOverride, load_imatrix};

    if flag_str(args, "--format") == Some("onnx") {
        return Err("--quant is not supported with --format onnx".into());
    }

    let quant: GgufQuantType = flag_str(args, "--quant")
        .unwrap_or("Q4_K_M")
        .parse()
        .map_err(|e: String| -> Box<dyn Error> { e.into() })?;

    let arch = flag_str(args, "--arch").unwrap_or("generic").to_owned();
    let ctx_len = flag_u32(args, "--ctx-len").unwrap_or(0);
    let emb_len = flag_u32(args, "--emb-len").unwrap_or(0);
    let blocks = flag_u32(args, "--blocks").unwrap_or(0);
    let heads = flag_u32(args, "--heads").unwrap_or(0);

    let config = GgufArchConfig {
        architecture: arch.clone(),
        name: arch,
        context_length: ctx_len,
        embedding_length: emb_len,
        block_count: blocks,
        head_count: heads,
        head_count_kv: heads,
        vocab_size: 0,
    };

    use nabla::module::load_tensors;
    use nabla_core::backend::DefaultBackend;

    eprintln!("Loading checkpoint: {}", model_path.display());
    let tensors: Vec<(String, nabla_core::tensor::Tensor<f64, DefaultBackend>)> =
        load_tensors(model_path).map_err(|e| format!("load_tensors: {e}"))?;

    let refs: Vec<(&str, &nabla_core::tensor::Tensor<f64, DefaultBackend>)> =
        tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();

    // Load imatrix if provided (enables IQ4_NL / IQ4_XS with importance-weighted scales).
    let imatrix: Option<Imatrix> = flag_str(args, "--imatrix")
        .map(|p| load_imatrix(std::path::Path::new(p)))
        .transpose()
        .map_err(|e| format!("load_imatrix: {e}"))?;

    eprintln!(
        "Exporting {} tensors → {} ({}{})",
        refs.len(),
        out_path.display(),
        quant,
        imatrix.as_ref().map_or(String::new(), |im| format!(
            ", imatrix {} entries",
            im.len()
        )),
    );

    if let Some(im) = &imatrix {
        nabla_interface::export_gguf_with_imatrix(
            &refs,
            out_path,
            quant,
            &config,
            &[] as &[QuantOverride],
            im,
        )
        .map_err(|e| format!("export_gguf_with_imatrix: {e}"))?;
    } else {
        nabla_interface::export_gguf(&refs, out_path, quant, &config, &[] as &[QuantOverride])
            .map_err(|e| format!("export_gguf: {e}"))?;
    }

    let size = std::fs::metadata(out_path)?.len();
    eprintln!(
        "Done — {} tensors, {:.1} MiB",
        refs.len(),
        size as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// ONNX export — reconstruct a Sequential from the state_dict tensor names.
// ---------------------------------------------------------------------------

fn export_onnx(model_path: &Path, out_path: &Path) -> std::result::Result<(), Box<dyn Error>> {
    use nabla::module::{Linear, Sequential, load_tensors};
    use nabla_core::backend::DefaultBackend;
    use nabla_train::onnx::export_sequential;

    eprintln!("Loading checkpoint: {}", model_path.display());
    let tensors: Vec<(String, nabla_core::tensor::Tensor<f64, DefaultBackend>)> =
        load_tensors(model_path).map_err(|e| format!("load_tensors: {e}"))?;

    // Infer layer count from weight keys "0.weight", "1.weight", ...
    let max_idx = tensors
        .iter()
        .filter_map(|(n, _)| {
            n.split_once('.').and_then(|(idx, rest)| {
                if rest == "weight" {
                    idx.parse::<usize>().ok()
                } else {
                    None
                }
            })
        })
        .max()
        .ok_or("checkpoint contains no weight tensors (expected `0.weight`, `1.weight`, ...)")?;

    let get_tensor = |name: &str| -> Option<&nabla_core::tensor::Tensor<f64, DefaultBackend>> {
        tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    };

    let first_w = get_tensor("0.weight").ok_or("missing tensor `0.weight`")?;
    let (_, in_features) = first_w.shape();
    let last_key = format!("{max_idx}.weight");
    let last_w = get_tensor(&last_key).ok_or("missing last weight")?;
    let (out_features, _) = last_w.shape();

    // Build Sequential by constructing Linear layers and injecting loaded weights.
    let mut model: Sequential<f64, DefaultBackend> = Sequential::new();
    for i in 0..=max_idx {
        let w = get_tensor(&format!("{i}.weight"))
            .ok_or_else(|| format!("missing tensor `{i}.weight`"))?
            .clone();
        let b = get_tensor(&format!("{i}.bias")).cloned();
        let (out_f, in_f) = w.shape();
        let mut layer = Linear::new(in_f, out_f);
        layer.weight = w;
        layer.bias = b;
        model = model.add(layer);
    }

    eprintln!(
        "Exporting Sequential ({} layers, {in_features}→{out_features}) → {}",
        max_idx + 1,
        out_path.display()
    );

    let onnx = export_sequential(&model, in_features as i64, out_features as i64);
    onnx.save(out_path)?;

    let size = std::fs::metadata(out_path)?.len();
    eprintln!("Done — {:.1} KiB", size as f64 / 1024.0);
    Ok(())
}

// ---------------------------------------------------------------------------
// Arg helpers
// ---------------------------------------------------------------------------

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn flag_u32(args: &[String], flag: &str) -> Option<u32> {
    flag_str(args, flag)?.parse().ok()
}
