#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::cast_possible_truncation)]
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]

//! GGUF export + llama.cpp FFI bridge for nabla.

pub mod convert;
pub mod gguf;
#[cfg(feature = "llama")]
pub mod llama;
pub mod quant;
#[cfg(feature = "llama")]
pub mod serve;

pub use convert::{GgufArchConfig, QuantOverride, export_gguf};
pub use gguf::GgufWriter;
pub use quant::GgufQuantType;

#[cfg(feature = "llama")]
pub use serve::{InferenceConfig, InferenceEngine, PerfStats, SamplingConfig};

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Quantization error.
    #[error("quant: {0}")]
    Quant(String),
    /// Conversion error.
    #[error("convert: {0}")]
    Convert(String),
    /// FFI / llama.cpp error.
    #[error("llama: {0}")]
    Llama(String),
}

/// Crate-level result type.
pub type Result<T> = std::result::Result<T, Error>;
