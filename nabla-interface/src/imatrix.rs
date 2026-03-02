//! Load llama.cpp-format importance matrix (`.imatrix`) files.
//!
//! The binary format (written by `llama-imatrix`):
//! ```text
//! int32 n_entries
//! for each entry:
//!   int32 n_chars         name length (bytes, no null terminator)
//!   u8[n_chars]           tensor name
//!   int32 n_calls         (skipped — forward-pass count, not used for export)
//!   int32 n_values        number of importance values (= n_cols of the tensor)
//!   f32[n_values]         per-column importance scores
//! ```

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::{Error, Result};

/// Per-tensor importance scores loaded from a `.imatrix` file.
///
/// Keys follow the GGUF tensor naming convention (same as used by `llama-imatrix`).
#[derive(Debug, Default)]
pub struct Imatrix(pub HashMap<String, Vec<f32>>);

impl Imatrix {
    /// Look up importance scores for a tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.0.get(name).map(Vec::as_slice)
    }

    /// Number of tensor entries loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the imatrix contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn read_i32<R: Read>(r: &mut R) -> std::result::Result<i32, std::io::Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

/// Load a llama.cpp-format importance matrix file.
///
/// # Errors
/// Returns `Error::Io` on file I/O failure or invalid file structure.
pub fn load_imatrix(path: &Path) -> Result<Imatrix> {
    let file = std::fs::File::open(path).map_err(Error::Io)?;
    let mut reader = BufReader::new(file);

    let n_entries = read_i32(&mut reader).map_err(Error::Io)?;
    if !(0..=500_000).contains(&n_entries) {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("imatrix: implausible entry count {n_entries}"),
        )));
    }

    let mut map = HashMap::with_capacity(n_entries as usize);
    for _ in 0..n_entries {
        let n_chars = read_i32(&mut reader).map_err(Error::Io)? as usize;
        let mut name_bytes = vec![0u8; n_chars];
        reader.read_exact(&mut name_bytes).map_err(Error::Io)?;
        let name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_owned();

        let _n_calls = read_i32(&mut reader).map_err(Error::Io)?;
        let n_values = read_i32(&mut reader).map_err(Error::Io)? as usize;

        let mut values = vec![0f32; n_values];
        for v in &mut values {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).map_err(Error::Io)?;
            *v = f32::from_le_bytes(buf);
        }
        map.insert(name, values);
    }
    Ok(Imatrix(map))
}
