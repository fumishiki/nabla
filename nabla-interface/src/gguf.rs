//! GGUF v3 binary writer.

use std::io::Write;

use crate::quant::GgufQuantType;
use crate::Result;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

// Metadata value type tags (GGUF spec)
const TYPE_U32: u32 = 4;
const TYPE_F32: u32 = 6;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;

/// Metadata value for GGUF key-value pairs.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// UTF-8 string.
    String(String),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit float.
    F32(f32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// Homogeneous array of metadata values.
    Array(Vec<MetadataValue>),
}

/// Tensor descriptor for GGUF output.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (GGUF string key).
    pub name: String,
    /// Dimension sizes (innermost first).
    pub dims: Vec<u64>,
    /// Quantization type.
    pub qtype: GgufQuantType,
    /// Raw data size in bytes.
    pub data_size: usize,
}

struct TensorEntry {
    info: TensorInfo,
    data: Vec<u8>,
}

struct MetadataEntry {
    key: String,
    value: MetadataValue,
}

/// GGUF v3 binary writer, generic over output sink.
pub struct GgufWriter<W: Write> {
    metadata: Vec<MetadataEntry>,
    tensors: Vec<TensorEntry>,
    _phantom: std::marker::PhantomData<W>,
}

impl<W: Write> GgufWriter<W> {
    /// Create an empty GGUF writer.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: Vec::new(), tensors: Vec::new(), _phantom: std::marker::PhantomData }
    }

    /// Add a metadata key-value pair.
    pub fn add_metadata(&mut self, key: &str, value: MetadataValue) {
        self.metadata.push(MetadataEntry { key: key.to_string(), value });
    }

    /// Add a tensor with its raw data.
    pub fn add_tensor(&mut self, info: TensorInfo, data: Vec<u8>) {
        self.tensors.push(TensorEntry { info, data });
    }

    /// Write the complete GGUF v3 binary to the given writer.
    ///
    /// # Errors
    /// Returns `Error::Io` if the underlying writer fails.
    pub fn write_to(&self, w: &mut W) -> Result<()> {
        // Header
        w.write_all(&GGUF_MAGIC.to_le_bytes())?;
        w.write_all(&GGUF_VERSION.to_le_bytes())?;
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(self.metadata.len() as u64).to_le_bytes())?;

        // Metadata KV pairs
        for entry in &self.metadata {
            write_gguf_string(w, &entry.key)?;
            write_metadata_value(w, &entry.value)?;
        }

        // Compute tensor offsets (relative to data section start, 32B aligned)
        let mut data_offset: u64 = 0;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            offsets.push(data_offset);
            data_offset += align_up(t.data.len(), ALIGNMENT) as u64;
        }

        // Tensor info entries
        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            write_gguf_string(w, &t.info.name)?;
            w.write_all(&(t.info.dims.len() as u32).to_le_bytes())?;
            for &d in &t.info.dims {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&(t.info.qtype as u32).to_le_bytes())?;
            w.write_all(&offset.to_le_bytes())?;
        }

        // Padding to 32-byte alignment before data section
        pad_to_alignment(w, self.header_size())?;

        // Tensor data (each tensor 32B aligned)
        let mut written: usize = 0;
        for t in &self.tensors {
            w.write_all(&t.data)?;
            written += t.data.len();
            let aligned = align_up(written, ALIGNMENT);
            if aligned > written {
                let pad = aligned - written;
                w.write_all(&vec![0u8; pad])?;
                written = aligned;
            }
        }

        Ok(())
    }

    // Compute byte size of header + metadata + tensor_info (before padding)
    fn header_size(&self) -> usize {
        let mut size = 4 + 4 + 8 + 8; // magic + version + tensor_count + metadata_kv_count
        for entry in &self.metadata {
            size += gguf_string_size(&entry.key) + metadata_value_size(&entry.value);
        }
        for t in &self.tensors {
            size += gguf_string_size(&t.info.name);
            size += 4; // n_dims
            size += 8 * t.info.dims.len(); // dims
            size += 4; // type
            size += 8; // offset
        }
        size
    }
}

impl<W: Write> Default for GgufWriter<W> {
    fn default() -> Self { Self::new() }
}

fn write_gguf_string(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_metadata_value(w: &mut impl Write, v: &MetadataValue) -> Result<()> {
    match v {
        MetadataValue::String(s) => {
            w.write_all(&TYPE_STRING.to_le_bytes())?;
            write_gguf_string(w, s)?;
        }
        MetadataValue::U32(val) => {
            w.write_all(&TYPE_U32.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())?;
        }
        MetadataValue::F32(val) => {
            w.write_all(&TYPE_F32.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())?;
        }
        MetadataValue::U64(val) => {
            w.write_all(&TYPE_U64.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())?;
        }
        MetadataValue::Array(items) => {
            w.write_all(&TYPE_ARRAY.to_le_bytes())?;
            let elem_type = items.first().map_or(TYPE_U32, metadata_type_tag);
            w.write_all(&elem_type.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for item in items {
                write_metadata_value_raw(w, item)?;
            }
        }
    }
    Ok(())
}

fn write_metadata_value_raw(w: &mut impl Write, v: &MetadataValue) -> Result<()> {
    match v {
        MetadataValue::String(s) => write_gguf_string(w, s)?,
        MetadataValue::U32(val) => w.write_all(&val.to_le_bytes())?,
        MetadataValue::F32(val) => w.write_all(&val.to_le_bytes())?,
        MetadataValue::U64(val) => w.write_all(&val.to_le_bytes())?,
        MetadataValue::Array(_) => {} // nested arrays not supported in GGUF
    }
    Ok(())
}

fn metadata_type_tag(v: &MetadataValue) -> u32 {
    match v {
        MetadataValue::String(_) => TYPE_STRING,
        MetadataValue::U32(_) => TYPE_U32,
        MetadataValue::F32(_) => TYPE_F32,
        MetadataValue::U64(_) => TYPE_U64,
        MetadataValue::Array(_) => TYPE_ARRAY,
    }
}

fn gguf_string_size(s: &str) -> usize { 8 + s.len() }

fn metadata_value_size(v: &MetadataValue) -> usize {
    4 + match v { // 4 bytes for type tag
        MetadataValue::String(s) => gguf_string_size(s),
        MetadataValue::U32(_) | MetadataValue::F32(_) => 4,
        MetadataValue::U64(_) => 8,
        MetadataValue::Array(items) => {
            4 + 8 + items.iter().map(metadata_value_raw_size).sum::<usize>() // elem_type + len + items
        }
    }
}

fn metadata_value_raw_size(v: &MetadataValue) -> usize {
    match v {
        MetadataValue::String(s) => gguf_string_size(s),
        MetadataValue::U32(_) | MetadataValue::F32(_) => 4,
        MetadataValue::U64(_) => 8,
        MetadataValue::Array(_) => 0,
    }
}

fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

fn pad_to_alignment(w: &mut impl Write, current_pos: usize) -> Result<()> {
    let aligned = align_up(current_pos, ALIGNMENT);
    if aligned > current_pos {
        let pad = aligned - current_pos;
        w.write_all(&vec![0u8; pad])?;
    }
    Ok(())
}
