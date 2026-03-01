//! GGUF v3 binary writer round-trip tests.

use std::io::Cursor;

use nabla_interface::gguf::{GgufWriter, MetadataValue, TensorInfo};
use nabla_interface::quant::GgufQuantType;

#[test]
fn gguf_v3_magic_and_version() {
    let gguf: GgufWriter<Cursor<Vec<u8>>> = GgufWriter::new();
    let mut buf = Cursor::new(Vec::new());
    gguf.write_to(&mut buf).expect("write failed");
    let data = buf.into_inner();
    // Magic: "GGUF" in LE = 0x46554747
    assert_eq!(&data[0..4], &[0x47, 0x47, 0x55, 0x46], "bad magic");
    // Version: 3
    assert_eq!(&data[4..8], &3u32.to_le_bytes(), "bad version");
    // tensor_count: 0
    assert_eq!(&data[8..16], &0u64.to_le_bytes(), "bad tensor_count");
    // metadata_kv_count: 0
    assert_eq!(&data[16..24], &0u64.to_le_bytes(), "bad metadata_count");
}

#[test]
fn gguf_metadata_string() {
    let mut gguf: GgufWriter<Cursor<Vec<u8>>> = GgufWriter::new();
    gguf.add_metadata("general.architecture", MetadataValue::String("llama".into()));
    let mut buf = Cursor::new(Vec::new());
    gguf.write_to(&mut buf).expect("write failed");
    let data = buf.into_inner();
    // metadata_kv_count should be 1
    assert_eq!(u64::from_le_bytes(data[16..24].try_into().expect("slice")), 1);
}

#[test]
fn gguf_metadata_u32() {
    let mut gguf: GgufWriter<Cursor<Vec<u8>>> = GgufWriter::new();
    gguf.add_metadata("test.value", MetadataValue::U32(42));
    let mut buf = Cursor::new(Vec::new());
    gguf.write_to(&mut buf).expect("write failed");
    let data = buf.into_inner();
    assert_eq!(u64::from_le_bytes(data[16..24].try_into().expect("slice")), 1);
}

#[test]
fn gguf_with_tensor_data() {
    let mut gguf: GgufWriter<Cursor<Vec<u8>>> = GgufWriter::new();
    gguf.add_metadata("general.architecture", MetadataValue::String("test".into()));
    let tensor_data = vec![0u8; 128]; // 32 f32 values = Q4_0 wants 32 elements
    let info = TensorInfo {
        name: "weight".into(),
        dims: vec![4, 8],
        qtype: GgufQuantType::F32,
        data_size: 128,
    };
    gguf.add_tensor(info, tensor_data);
    let mut buf = Cursor::new(Vec::new());
    gguf.write_to(&mut buf).expect("write failed");
    let data = buf.into_inner();
    // tensor_count should be 1
    assert_eq!(u64::from_le_bytes(data[8..16].try_into().expect("slice")), 1);
    // File should be non-trivially sized (header + metadata + tensor_info + padding + data)
    assert!(data.len() > 128, "file too small: {}", data.len());
}

#[test]
fn gguf_32byte_alignment() {
    let mut gguf: GgufWriter<Cursor<Vec<u8>>> = GgufWriter::new();
    let tensor_data = vec![1u8; 64];
    let info = TensorInfo {
        name: "t".into(), dims: vec![16], qtype: GgufQuantType::F32, data_size: 64,
    };
    gguf.add_tensor(info, tensor_data);
    let mut buf = Cursor::new(Vec::new());
    gguf.write_to(&mut buf).expect("write failed");
    let data = buf.into_inner();
    // Find the tensor data by looking for the first 0x01 byte after alignment padding
    let pos = data.iter().rposition(|&b| b == 1).expect("no tensor data found");
    // The start of tensor data should be 32-byte aligned
    let start = pos - 63; // 64 bytes of 0x01, so start = pos - 63
    assert_eq!(start % 32, 0, "tensor data not 32-byte aligned: offset {start}");
}
