// tensor/display.rs — Debug/Display formatting + fmt_matrix helper.

use core::fmt;

use crate::backend::Backend;
use crate::scalar::Scalar;

use super::Tensor;

fn display_indices(len: usize) -> Vec<Option<usize>> {
    if len > 6 {
        let mut v: Vec<Option<usize>> = (0..3).map(Some).collect();
        v.push(None);
        v.extend((len - 3..len).map(Some));
        v
    } else {
        (0..len).map(Some).collect()
    }
}

/// Write a matrix in `[[a, b], [c, d]]` style.
///
/// `prefix` is written before the outer `[`; when `None` a space is inserted
/// between rows (Display style), otherwise `, ` (Debug style).
///
/// When `rows > 6` or `cols > 6`, only the first 3 and last 3 entries along
/// each over-sized dimension are shown, separated by `...`.
pub(crate) fn fmt_matrix(
    rows: usize,
    cols: usize,
    mut elem: impl FnMut(usize, usize, &mut fmt::Formatter<'_>) -> fmt::Result,
    f: &mut fmt::Formatter<'_>,
    prefix: Option<&str>,
) -> fmt::Result {
    // Build the sequence of row / column indices to display.
    // When a dimension exceeds 6, show indices 0,1,2 and (n-3),(n-2),(n-1).
    let row_indices = display_indices(rows);
    let col_indices = display_indices(cols);

    if let Some(p) = prefix {
        write!(f, "{p}")?;
    }
    write!(f, "[")?;
    let mut first_row = true;
    for row_slot in row_indices {
        if !first_row {
            if prefix.is_some() {
                write!(f, ", ")?;
            } else {
                writeln!(f)?;
                write!(f, " ")?;
            }
        }
        first_row = false;
        match row_slot {
            None => {
                // Row ellipsis: emit a placeholder row
                write!(f, "[...]")?;
            }
            Some(r) => {
                write!(f, "[")?;
                let mut first_col = true;
                for col_slot in col_indices.iter().copied() {
                    if !first_col {
                        write!(f, ", ")?;
                    }
                    first_col = false;
                    match col_slot {
                        None => write!(f, "...")?,
                        Some(c) => elem(r, c, f)?,
                    }
                }
                write!(f, "]")?;
            }
        }
    }
    write!(f, "]")
}

impl<T: Scalar + fmt::Display, B: Backend> fmt::Display for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        fmt_matrix(
            rows,
            cols,
            |r, c, f| write!(f, "{}", self.get(r, c)),
            f,
            None,
        )
    }
}

impl<T: Scalar + fmt::Debug, B: Backend> fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        let prefix = format!("Tensor({rows}x{cols})");
        fmt_matrix(
            rows,
            cols,
            |r, c, f| write!(f, "{:?}", self.get(r, c)),
            f,
            Some(&prefix),
        )
    }
}
