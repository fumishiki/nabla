//! Tensor serialization: save and load named tensors in the NBLA binary format.

use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};

use crate::{scalar, tensor};

const NBLA_MAGIC: &[u8; 4] = b"NBLA";

/// Save named tensors to a binary file in the NBLA format.
///
/// Format: 4-byte magic "NBLA", u32 count, then per-tensor: u32 name_len, name bytes,
/// u32 nrows, u32 ncols, and `nrows * ncols` little-endian f64 values.
///
/// # Errors
///
/// Returns an `io::Error` if the file cannot be created or written.
pub fn save_tensors(
    tensors: &[(&str, &tensor::Tensor<f64>)],
    path: &Path,
) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(NBLA_MAGIC)?;
    write_u32(&mut file, tensors.len() as u32)?;
    for (name, t) in tensors {
        let name_bytes = name.as_bytes();
        write_u32(&mut file, name_bytes.len() as u32)?;
        file.write_all(name_bytes)?;
        let (m, n) = t.shape();
        write_u32(&mut file, m as u32)?;
        write_u32(&mut file, n as u32)?;
        for r in 0..m {
            for c in 0..n {
                write_f64(&mut file, t.get(r, c))?;
            }
        }
    }
    Ok(())
}

/// Load named tensors from a binary file in the NBLA format.
///
/// # Errors
///
/// Returns an `io::Error` if the file cannot be opened, is not a valid NBLA file,
/// or contains malformed UTF-8 in a tensor name.
pub fn load_tensors(
    path: &Path,
) -> io::Result<Vec<(String, tensor::Tensor<f64>)>> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != NBLA_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a nabla tensor file",
        ));
    }
    let count = read_u32(&mut file)? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u32(&mut file)? as usize;
        let mut name_buf = vec![0u8; name_len];
        file.read_exact(&mut name_buf)?;
        let name =
            String::from_utf8(name_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let nrows = read_u32(&mut file)? as usize;
        let ncols = read_u32(&mut file)? as usize;
        let total = nrows * ncols;
        let mut data = vec![0.0f64; total];
        for v in &mut data {
            *v = read_f64(&mut file)?;
        }
        let t = tensor::Tensor::from_fn(nrows, ncols, |r, c| {
            scalar::Scalar::from_f64(data[r * ncols + c])
        });
        result.push((name, t));
    }
    Ok(result)
}

#[inline]
fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> io::Result<()> {
    writer.write_all(bytes)
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> io::Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}

#[inline]
fn read_bytes<R: Read, const N: usize>(reader: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(reader)?))
}

fn read_f64<R: Read>(reader: &mut R) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_bytes(reader)?))
}
