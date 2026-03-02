//! Module `state_dict` to GGUF tensor mapping and export.

use std::io::BufWriter;
use std::path::Path;

use nabla_core::backend::DefaultBackend;
use nabla_core::tensor::Tensor;

use crate::gguf::{GgufWriter, MetadataValue, TensorInfo};
use crate::quant::{self, GgufQuantType};
use crate::{Error, Result};

/// Architecture metadata for GGUF export (`GgufArchConfig`).
#[derive(Debug, Clone)]
pub struct GgufArchConfig {
    /// Model architecture identifier (e.g. `"llama"`).
    pub architecture: String,
    /// Human-readable model name.
    pub name: String,
    /// Maximum context length in tokens.
    pub context_length: u32,
    /// Hidden/embedding dimension.
    pub embedding_length: u32,
    /// Number of transformer blocks.
    pub block_count: u32,
    /// Number of attention heads.
    pub head_count: u32,
    /// Number of key-value heads (GQA).
    pub head_count_kv: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
}

/// Per-tensor quantization override (`QuantOverride`).
#[derive(Debug, Clone)]
pub struct QuantOverride {
    /// Tensor name (nabla `state_dict` key) to override.
    pub name: String,
    /// Quantization type to use for this tensor.
    pub qtype: GgufQuantType,
}

/// Map a nabla `state_dict` tensor name to the GGUF tensor name convention.
#[must_use]
pub fn map_tensor_name(name: &str) -> String {
    if name == "embedding.weight" {
        return "token_embd.weight".into();
    }
    if name == "norm.weight" {
        return "output_norm.weight".into();
    }
    if name == "output.weight" {
        return "output.weight".into();
    }
    if let Some(dot_pos) = name
        .strip_prefix("layers.")
        .and_then(|r| r.find('.').map(|p| 7 + p))
    {
        let idx = &name[7..dot_pos];
        let suffix = &name[dot_pos + 1..];
        let mapped = match suffix {
            "attention.wq.weight" => "attn_q.weight",
            "attention.wk.weight" => "attn_k.weight",
            "attention.wv.weight" => "attn_v.weight",
            "attention.wo.weight" => "attn_output.weight",
            "attention_norm.weight" => "attn_norm.weight",
            "ffn.w1.weight" => "ffn_gate.weight",
            "ffn.w2.weight" => "ffn_down.weight",
            "ffn.w3.weight" => "ffn_up.weight",
            "ffn_norm.weight" => "ffn_norm.weight",
            _ => return format!("blk.{idx}.{suffix}"),
        };
        return format!("blk.{idx}.{mapped}");
    }
    name.into()
}

fn quantize_data(data: &[f32], qtype: GgufQuantType) -> Result<Vec<u8>> {
    if !qtype.is_quantizable() {
        return Err(Error::Quant(format!(
            "{qtype:?} requires importance matrix or is not quantizable"
        )));
    }
    quant::quantize(data, qtype)
}

fn quantize_data_with_imatrix(
    gguf_name: &str,
    data: &[f32],
    qtype: GgufQuantType,
    imatrix: &crate::imatrix::Imatrix,
) -> Result<Vec<u8>> {
    if !qtype.is_quantizable() {
        return Err(Error::Quant(format!(
            "{qtype:?} requires importance matrix or is not quantizable"
        )));
    }
    let importance = imatrix.get(gguf_name);
    quant::quantize_with_importance(data, qtype, importance)
}

fn resolve_qtype(name: &str, default: GgufQuantType, overrides: &[QuantOverride]) -> GgufQuantType {
    overrides
        .iter()
        .find(|o| o.name == name)
        .map_or(default, |o| o.qtype)
}

/// Export a `state_dict` to a GGUF file.
///
/// # Errors
/// Returns `Error::Io` on file I/O failure or `Error::Quant` on quantization failure.
pub fn export_gguf(
    state_dict: &[(&str, &Tensor<f64, DefaultBackend>)],
    path: &Path,
    quant_type: GgufQuantType,
    config: &GgufArchConfig,
    overrides: &[QuantOverride],
) -> Result<()> {
    let mut writer: GgufWriter<BufWriter<std::fs::File>> = GgufWriter::new();
    let arch = &config.architecture;
    writer.add_metadata("general.architecture", MetadataValue::String(arch.clone()));
    writer.add_metadata("general.name", MetadataValue::String(config.name.clone()));
    writer.add_metadata(
        "general.file_type",
        MetadataValue::U32(quant_type.type_id()),
    );
    writer.add_metadata(
        &format!("{arch}.context_length"),
        MetadataValue::U32(config.context_length),
    );
    writer.add_metadata(
        &format!("{arch}.embedding_length"),
        MetadataValue::U32(config.embedding_length),
    );
    writer.add_metadata(
        &format!("{arch}.block_count"),
        MetadataValue::U32(config.block_count),
    );
    writer.add_metadata(
        &format!("{arch}.attention.head_count"),
        MetadataValue::U32(config.head_count),
    );
    writer.add_metadata(
        &format!("{arch}.attention.head_count_kv"),
        MetadataValue::U32(config.head_count_kv),
    );
    writer.add_metadata(
        &format!("{arch}.vocab_size"),
        MetadataValue::U32(config.vocab_size),
    );
    for &(name, tensor) in state_dict {
        let gguf_name = map_tensor_name(name);
        let qtype = resolve_qtype(name, quant_type, overrides);
        let (nrows, ncols) = tensor.shape();
        let f32_data: Vec<f32> = tensor.to_vec().iter().map(|&v| v as f32).collect();
        let bs = qtype.block_size();
        let padded = if f32_data.len().is_multiple_of(bs) {
            f32_data
        } else {
            let mut p = f32_data;
            p.resize(p.len().next_multiple_of(bs), 0.0);
            p
        };
        let quantized = quantize_data(&padded, qtype)?;
        let info = TensorInfo {
            name: gguf_name,
            dims: vec![ncols as u64, nrows as u64],
            qtype,
            data_size: quantized.len(),
        };
        writer.add_tensor(info, quantized);
    }
    let file = std::fs::File::create(path)?;
    let mut buf = BufWriter::new(file);
    writer.write_to(&mut buf)?;
    Ok(())
}

/// Export a `state_dict` to a GGUF file with importance-matrix support for IQ types.
///
/// Enables `IQ4_NL` and `IQ4_XS` quantization by using per-column importance scores
/// from the supplied [`Imatrix`] to guide scale selection.
///
/// # Errors
/// Returns `Error::Io` on file I/O failure or `Error::Quant` on quantization failure.
pub fn export_gguf_with_imatrix(
    state_dict: &[(&str, &Tensor<f64, DefaultBackend>)],
    path: &Path,
    quant_type: GgufQuantType,
    config: &GgufArchConfig,
    overrides: &[QuantOverride],
    imatrix: &crate::imatrix::Imatrix,
) -> Result<()> {
    let mut writer: GgufWriter<BufWriter<std::fs::File>> = GgufWriter::new();
    let arch = &config.architecture;
    writer.add_metadata("general.architecture", MetadataValue::String(arch.clone()));
    writer.add_metadata("general.name", MetadataValue::String(config.name.clone()));
    writer.add_metadata("general.file_type", MetadataValue::U32(quant_type.type_id()));
    writer.add_metadata(&format!("{arch}.context_length"),        MetadataValue::U32(config.context_length));
    writer.add_metadata(&format!("{arch}.embedding_length"),      MetadataValue::U32(config.embedding_length));
    writer.add_metadata(&format!("{arch}.block_count"),           MetadataValue::U32(config.block_count));
    writer.add_metadata(&format!("{arch}.attention.head_count"),  MetadataValue::U32(config.head_count));
    writer.add_metadata(&format!("{arch}.attention.head_count_kv"), MetadataValue::U32(config.head_count_kv));
    writer.add_metadata(&format!("{arch}.vocab_size"),            MetadataValue::U32(config.vocab_size));
    for &(name, tensor) in state_dict {
        let gguf_name = map_tensor_name(name);
        let qtype = resolve_qtype(name, quant_type, overrides);
        let (nrows, ncols) = tensor.shape();
        let f32_data: Vec<f32> = tensor.to_vec().iter().map(|&v| v as f32).collect();
        let bs = qtype.block_size();
        let padded = if f32_data.len().is_multiple_of(bs) {
            f32_data
        } else {
            let mut p = f32_data;
            p.resize(p.len().next_multiple_of(bs), 0.0);
            p
        };
        let quantized = quantize_data_with_imatrix(&gguf_name, &padded, qtype, imatrix)?;
        writer.add_tensor(
            TensorInfo { name: gguf_name, dims: vec![ncols as u64, nrows as u64], qtype, data_size: quantized.len() },
            quantized,
        );
    }
    let file = std::fs::File::create(path)?;
    let mut buf = BufWriter::new(file);
    writer.write_to(&mut buf)?;
    Ok(())
}
