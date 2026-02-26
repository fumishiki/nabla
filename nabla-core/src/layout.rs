//! F₂ binary matrix layout for GPU shared memory bank-conflict-free swizzling.

use core::fmt;

/// N×N binary matrix over GF(2). Represents a linear map on F₂ᴺ.
///
/// Each row is stored as a `u64` bitmask, supporting N ≤ 64.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LinearLayout<const N: usize> {
    rows: [u64; N],
}

impl<const N: usize> LinearLayout<N> {
    /// Identity matrix: `rows[i] = 1 << i`.
    #[must_use]
    pub fn identity() -> Self {
        let mut rows = [0u64; N];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = 1 << i;
        }
        Self { rows }
    }

    /// Matrix product over F₂: C = self * other.
    ///
    /// `C[i][j] = XOR over k of (self[i][k] AND other[k][j])`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = [0u64; N];
        for (i, res) in result.iter_mut().enumerate().take(N) {
            let mut row = 0u64;
            for j in 0..N {
                // Dot product of self row i with other column j over GF(2)
                let mut bit = 0u64;
                for k in 0..N {
                    let a = (self.rows[i] >> k) & 1;
                    let b = (other.rows[k] >> j) & 1;
                    bit ^= a & b;
                }
                row |= bit << j;
            }
            *res = row;
        }
        Self { rows: result }
    }

    /// Matrix-vector product over F₂: output bit `i = popcount(rows[i] & v) mod 2`.
    #[must_use]
    pub fn apply(&self, v: u64) -> u64 {
        let mut out = 0u64;
        for (i, &row) in self.rows.iter().enumerate() {
            let bit = (row & v).count_ones() & 1;
            out |= u64::from(bit) << i;
        }
        out
    }

    /// Construct a swizzle layout for a shared memory tile.
    ///
    /// `banks` must be a power of two (typically 32).
    /// Formula: `swizzle(row, col) = col XOR (row >> (log2(tile_cols) - bank_bits))`.
    #[must_use]
    pub fn swizzle_for_tile(tile_rows: usize, tile_cols: usize, banks: usize) -> Self {
        debug_assert!(banks.is_power_of_two(), "banks must be power of two");
        debug_assert!(
            tile_cols.is_power_of_two(),
            "tile_cols must be power of two"
        );
        let bank_bits = banks.trailing_zeros() as usize;
        let col_bits = tile_cols.trailing_zeros() as usize;
        // Shift amount for extracting high row bits to XOR with column bank bits
        let shift = col_bits.saturating_sub(bank_bits);

        // Address bits: row_bits | col_bits (row in high, col in low)
        let row_bits = tile_rows.trailing_zeros() as usize;
        let total_bits = row_bits + col_bits;
        debug_assert!(total_bits <= N, "tile address exceeds layout dimension");

        // Start from identity, then XOR col bits with shifted row bits
        let mut rows = [0u64; N];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = 1 << i;
        }

        // Column bits [0, col_bits) get XOR'd with row bits shifted by `shift`
        for (c, row) in rows.iter_mut().enumerate().take(bank_bits.min(col_bits)) {
            let row_src = col_bits + shift + c;
            if row_src < total_bits {
                *row |= 1 << row_src;
            }
        }

        Self { rows }
    }

    /// Optimal swizzle for a square tile with 32 banks.
    #[must_use]
    pub fn optimal_tile_swizzle(block_size: usize) -> Self {
        Self::swizzle_for_tile(block_size, block_size, 32)
    }

    /// Emit a WGSL function implementing this layout as XOR + shift ops.
    #[must_use]
    pub fn to_wgsl_swizzle_fn(&self, fn_name: &str) -> String {
        let mut lines = Vec::new();
        lines.push(format!("fn {fn_name}(addr: u32) -> u32 {{"));
        lines.push("    var result: u32 = 0u;".to_owned());
        for i in 0..N {
            if self.rows[i] == 0 {
                continue;
            }
            if self.rows[i] == 1 << i {
                // Identity bit: just extract
                lines.push(format!("    result |= (addr >> {i}u) & 1u;"));
                if i > 0 {
                    // Shift into position
                    let last = lines.len() - 1;
                    lines[last] = format!("    result |= ((addr >> {i}u) & 1u) << {i}u;");
                }
            } else {
                // XOR of multiple input bits
                let set_bits: Vec<usize> =
                    (0..N).filter(|&j| (self.rows[i] >> j) & 1 == 1).collect();
                if set_bits.len() == 1 {
                    let j = set_bits[0];
                    lines.push(format!("    result |= ((addr >> {j}u) & 1u) << {i}u;"));
                } else {
                    let xor_expr: Vec<String> = set_bits
                        .iter()
                        .map(|&j| format!("((addr >> {j}u) & 1u)"))
                        .collect();
                    let expr = xor_expr.join(" ^ ");
                    lines.push(format!("    result |= ({expr}) << {i}u;"));
                }
            }
        }
        lines.push("    return result;".to_owned());
        lines.push("}".to_owned());
        lines.join("\n")
    }
}

impl<const N: usize> fmt::Debug for LinearLayout<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LinearLayout<{N}>[")?;
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{row:#018x}")?;
        }
        write!(f, "]")
    }
}

/// 16×16 binary layout.
pub type LinearLayout16 = LinearLayout<16>;
/// 32×32 binary layout.
pub type LinearLayout32 = LinearLayout<32>;
/// 64×64 binary layout.
pub type LinearLayout64 = LinearLayout<64>;
