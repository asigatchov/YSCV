//! H.264 CAVLC (Context-Adaptive Variable-Length Coding) entropy decoding.
//!
//! CAVLC is the entropy coding method used in H.264 Baseline profile for
//! encoding residual transform coefficients. This module provides a bitstream
//! reader with Exp-Golomb support and a CAVLC block decoder.

// ---------------------------------------------------------------------------
// BitReader — bit-level access to a byte slice (MSB first)
// ---------------------------------------------------------------------------

/// Reads bits from a byte slice in MSB-first order, with Exp-Golomb support.
pub struct BitReader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) byte_pos: usize,
    pub(crate) bit_pos: u8, // 0..8, bits consumed in current byte
}

impl<'a> BitReader<'a> {
    /// Creates a new `BitReader` over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Returns the number of unconsumed bits remaining.
    pub fn bits_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }

    /// Reads `n` bits (1..=32) as a `u32`, MSB first. Returns `None` on
    /// exhaustion.
    pub fn read_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!((1..=32).contains(&n));
        if self.bits_remaining() < n as usize {
            return None;
        }
        let mut value = 0u32;
        for _ in 0..n {
            let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
            value = (value << 1) | bit as u32;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Some(value)
    }

    /// Reads an unsigned Exp-Golomb coded integer (ue(v)).
    pub fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u32;
        loop {
            let bit = self.read_bits(1)?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return None;
            }
        }
        if leading_zeros == 0 {
            return Some(0);
        }
        let suffix = self.read_bits(leading_zeros as u8)?;
        Some((1 << leading_zeros) - 1 + suffix)
    }

    /// Reads a signed Exp-Golomb coded integer (se(v)).
    pub fn read_se(&mut self) -> Option<i32> {
        let code = self.read_ue()?;
        let value = code.div_ceil(2) as i32;
        if code % 2 == 0 {
            Some(-value)
        } else {
            Some(value)
        }
    }
}

// ---------------------------------------------------------------------------
// CAVLC result and VLC tables
// ---------------------------------------------------------------------------

/// Decoded CAVLC residual coefficients for one block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CavlcResult {
    /// Number of non-zero coefficients (0..=16).
    pub total_coeffs: usize,
    /// Number of trailing +/-1 coefficients (0..=3).
    pub trailing_ones: usize,
    /// Non-zero coefficient levels in reverse scan order.
    pub levels: Vec<i32>,
    /// Total number of zero-valued coefficients before the last non-zero.
    pub total_zeros: usize,
    /// Run of zeros before each coefficient (reverse scan order).
    pub runs: Vec<usize>,
}

// coeff_token VLC tables — each entry is (bit_pattern, bit_length, total_coeffs, trailing_ones).
// Tables are indexed by nC category. Only the most common entries are included;
// a production decoder would use the full ITU-T H.264 tables.

struct CoeffTokenEntry {
    pattern: u32,
    length: u8,
    total_coeffs: u8,
    trailing_ones: u8,
}

// Macro to reduce boilerplate in table definitions.
macro_rules! ct {
    ($pat:expr, $len:expr, $tc:expr, $t1:expr) => {
        CoeffTokenEntry {
            pattern: $pat,
            length: $len,
            total_coeffs: $tc,
            trailing_ones: $t1,
        }
    };
}

/// nC = 0..1 (Table 9-5(a) from the spec, truncated to common entries).
const COEFF_TOKEN_NC_0_1: &[CoeffTokenEntry] = &[
    ct!(0b1, 1, 0, 0),           // 1              -> (0,0)
    ct!(0b000101, 6, 1, 0),      // 000101          -> (1,0)
    ct!(0b01, 2, 1, 1),          // 01              -> (1,1)
    ct!(0b00000111, 8, 2, 0),    // 00000111        -> (2,0)
    ct!(0b000100, 6, 2, 1),      // 000100          -> (2,1)
    ct!(0b001, 3, 2, 2),         // 001             -> (2,2)
    ct!(0b000000111, 9, 3, 0),   // 000000111       -> (3,0)
    ct!(0b00000110, 8, 3, 1),    // 00000110        -> (3,1)
    ct!(0b0000101, 7, 3, 2),     // 0000101         -> (3,2)
    ct!(0b00011, 5, 3, 3),       // 00011           -> (3,3)
    ct!(0b0000000111, 10, 4, 0), // 0000000111      -> (4,0)
    ct!(0b000000110, 9, 4, 1),   // 000000110       -> (4,1)
    ct!(0b00000101, 8, 4, 2),    // 00000101        -> (4,2)
    ct!(0b000011, 6, 4, 3),      // 000011          -> (4,3)
];

/// nC = 2..3 (Table 9-5(b), truncated).
const COEFF_TOKEN_NC_2_3: &[CoeffTokenEntry] = &[
    ct!(0b11, 2, 0, 0),       // 11              -> (0,0)
    ct!(0b001011, 6, 1, 0),   // 001011          -> (1,0)
    ct!(0b10, 2, 1, 1),       // 10              -> (1,1)
    ct!(0b000111, 6, 2, 0),   // 000111          -> (2,0)
    ct!(0b00111, 5, 2, 1),    // 00111           -> (2,1)
    ct!(0b011, 3, 2, 2),      // 011             -> (2,2)
    ct!(0b0000111, 7, 3, 0),  // 0000111         -> (3,0)
    ct!(0b001010, 6, 3, 1),   // 001010          -> (3,1)
    ct!(0b001001, 6, 3, 2),   // 001001          -> (3,2)
    ct!(0b00101, 5, 3, 3),    // 00101           -> (3,3)
    ct!(0b00000111, 8, 4, 0), // 00000111        -> (4,0)
    ct!(0b0000110, 7, 4, 1),  // 0000110         -> (4,1)
    ct!(0b000110, 6, 4, 2),   // 000110          -> (4,2)
    ct!(0b00100, 5, 4, 3),    // 00100           -> (4,3)
];

/// nC = 4..7 (Table 9-5(c), truncated).
const COEFF_TOKEN_NC_4_7: &[CoeffTokenEntry] = &[
    ct!(0b1111, 4, 0, 0),    // 1111          -> (0,0)
    ct!(0b001111, 6, 1, 0),  // 001111        -> (1,0)
    ct!(0b1110, 4, 1, 1),    // 1110          -> (1,1)
    ct!(0b001011, 6, 2, 0),  // 001011        -> (2,0)
    ct!(0b01111, 5, 2, 1),   // 01111         -> (2,1)
    ct!(0b1101, 4, 2, 2),    // 1101          -> (2,2)
    ct!(0b001000, 6, 3, 0),  // 001000        -> (3,0)
    ct!(0b01110, 5, 3, 1),   // 01110         -> (3,1)
    ct!(0b01101, 5, 3, 2),   // 01101         -> (3,2)
    ct!(0b1100, 4, 3, 3),    // 1100          -> (3,3)
    ct!(0b0000111, 7, 4, 0), // 0000111       -> (4,0)
    ct!(0b001110, 6, 4, 1),  // 001110        -> (4,1)
    ct!(0b001010, 6, 4, 2),  // 001010        -> (4,2)
    ct!(0b1011, 4, 4, 3),    // 1011          -> (4,3)
];

/// nC >= 8: fixed-length 6-bit code.
fn coeff_token_nc_8plus(reader: &mut BitReader) -> Option<(u8, u8)> {
    let code = reader.read_bits(6)?;
    // For nC >= 8 the coeff_token is a 6-bit FLC:
    //   trailing_ones = code[1:0], total_coeffs = code[5:2] + (trailing_ones > 0 ? 0 : 0)
    // Simplified: the first 4 bits encode total_coeffs-trailing_ones info.
    // ITU spec Table 9-5(d): code = (total_coeffs - 1) * 4 + trailing_ones
    // with total_coeffs 0 mapped to code 3 (special case 0b000011).
    if code == 3 {
        return Some((0, 0));
    }
    let trailing_ones = (code & 0x03) as u8;
    let total_coeffs = ((code >> 2) + 1) as u8;
    // Clamp trailing_ones to min(trailing_ones, total_coeffs, 3)
    let trailing_ones = trailing_ones.min(total_coeffs).min(3);
    Some((total_coeffs, trailing_ones))
}

/// Select the right coeff_token table based on nC.
fn select_coeff_token_table(nc: i32) -> &'static [CoeffTokenEntry] {
    match nc {
        0..=1 => COEFF_TOKEN_NC_0_1,
        2..=3 => COEFF_TOKEN_NC_2_3,
        4..=7 => COEFF_TOKEN_NC_4_7,
        _ => &[], // nc >= 8 uses fixed-length, handled separately
    }
}

/// Reads a coeff_token using VLC lookup.
fn read_coeff_token(reader: &mut BitReader, nc: i32) -> Option<(u8, u8)> {
    if nc >= 8 {
        return coeff_token_nc_8plus(reader);
    }

    let table = select_coeff_token_table(nc);

    // Try matching longest-prefix: peek up to 16 bits and check each entry.
    // We need to save/restore position for mismatches.
    let save_byte = reader.byte_pos;
    let save_bit = reader.bit_pos;
    let avail = reader.bits_remaining();

    for entry in table {
        let len = entry.length;
        if (len as usize) > avail {
            continue;
        }
        // Restore position for each attempt.
        reader.byte_pos = save_byte;
        reader.bit_pos = save_bit;
        let bits = reader.read_bits(len)?;
        if bits == entry.pattern {
            return Some((entry.total_coeffs, entry.trailing_ones));
        }
    }

    // No match found — restore position and return None.
    reader.byte_pos = save_byte;
    reader.bit_pos = save_bit;
    None
}

// ---------------------------------------------------------------------------
// total_zeros VLC tables (Table 9-7 from the spec, simplified)
// ---------------------------------------------------------------------------

struct VlcEntry {
    pattern: u32,
    length: u8,
    value: u8,
}

macro_rules! vlc {
    ($pat:expr, $len:expr, $val:expr) => {
        VlcEntry {
            pattern: $pat,
            length: $len,
            value: $val,
        }
    };
}

/// total_zeros tables indexed by total_coeffs (1-based). Only tc=1..4 provided.
const TOTAL_ZEROS_TC1: &[VlcEntry] = &[
    vlc!(0b1, 1, 0),
    vlc!(0b011, 3, 1),
    vlc!(0b010, 3, 2),
    vlc!(0b0011, 4, 3),
    vlc!(0b0010, 4, 4),
    vlc!(0b00011, 5, 5),
    vlc!(0b00010, 5, 6),
    vlc!(0b00001, 5, 7),
    vlc!(0b000001, 6, 8),
    vlc!(0b0000001, 7, 9),
    vlc!(0b00000001, 8, 10),
    vlc!(0b000000001, 9, 11),
    vlc!(0b0000000001, 10, 12),
    vlc!(0b00000000011, 11, 13),
    vlc!(0b00000000010, 11, 14),
    vlc!(0b00000000001, 11, 15),
];

const TOTAL_ZEROS_TC2: &[VlcEntry] = &[
    vlc!(0b111, 3, 0),
    vlc!(0b110, 3, 1),
    vlc!(0b101, 3, 2),
    vlc!(0b100, 3, 3),
    vlc!(0b011, 3, 4),
    vlc!(0b0101, 4, 5),
    vlc!(0b0100, 4, 6),
    vlc!(0b0011, 4, 7),
    vlc!(0b0010, 4, 8),
    vlc!(0b00011, 5, 9),
    vlc!(0b00010, 5, 10),
    vlc!(0b000011, 6, 11),
    vlc!(0b000010, 6, 12),
    vlc!(0b000001, 6, 13),
    vlc!(0b000000, 6, 14),
];

const TOTAL_ZEROS_TC3: &[VlcEntry] = &[
    vlc!(0b0101, 4, 0),
    vlc!(0b111, 3, 1),
    vlc!(0b110, 3, 2),
    vlc!(0b101, 3, 3),
    vlc!(0b0100, 4, 4),
    vlc!(0b0011, 4, 5),
    vlc!(0b100, 3, 6),
    vlc!(0b011, 3, 7),
    vlc!(0b0010, 4, 8),
    vlc!(0b00011, 5, 9),
    vlc!(0b00010, 5, 10),
    vlc!(0b000001, 6, 11),
    vlc!(0b00001, 5, 12),
    vlc!(0b000000, 6, 13),
];

const TOTAL_ZEROS_TC4: &[VlcEntry] = &[
    vlc!(0b00011, 5, 0),
    vlc!(0b111, 3, 1),
    vlc!(0b0101, 4, 2),
    vlc!(0b0100, 4, 3),
    vlc!(0b110, 3, 4),
    vlc!(0b101, 3, 5),
    vlc!(0b100, 3, 6),
    vlc!(0b0011, 4, 7),
    vlc!(0b011, 3, 8),
    vlc!(0b00010, 5, 9),
    vlc!(0b00001, 5, 10),
    vlc!(0b00000, 5, 11),
    vlc!(0b0010, 4, 12),
];

fn total_zeros_table(total_coeffs: usize) -> &'static [VlcEntry] {
    match total_coeffs {
        1 => TOTAL_ZEROS_TC1,
        2 => TOTAL_ZEROS_TC2,
        3 => TOTAL_ZEROS_TC3,
        4 => TOTAL_ZEROS_TC4,
        _ => &[],
    }
}

/// Reads a value from a VLC table by trying each entry.
fn read_vlc(reader: &mut BitReader, table: &[VlcEntry]) -> Option<u8> {
    let save_byte = reader.byte_pos;
    let save_bit = reader.bit_pos;
    let avail = reader.bits_remaining();

    for entry in table {
        let len = entry.length;
        if (len as usize) > avail {
            continue;
        }
        reader.byte_pos = save_byte;
        reader.bit_pos = save_bit;
        let bits = reader.read_bits(len)?;
        if bits == entry.pattern {
            return Some(entry.value);
        }
    }

    reader.byte_pos = save_byte;
    reader.bit_pos = save_bit;
    None
}

// ---------------------------------------------------------------------------
// run_before VLC (Table 9-10)
// ---------------------------------------------------------------------------

/// Reads run_before given zeros_left.
fn read_run_before(reader: &mut BitReader, zeros_left: usize) -> Option<usize> {
    match zeros_left {
        0 => Some(0),
        1 => {
            let bit = reader.read_bits(1)?;
            Some(if bit == 1 { 0 } else { 1 })
        }
        2 => {
            let bit = reader.read_bits(1)?;
            if bit == 1 {
                return Some(0);
            }
            let bit2 = reader.read_bits(1)?;
            Some(if bit2 == 1 { 1 } else { 2 })
        }
        3 => {
            let bits = reader.read_bits(2)?;
            match bits {
                0b11 => Some(0),
                0b10 => Some(1),
                0b01 => Some(2),
                0b00 => Some(3),
                _ => None,
            }
        }
        4 => {
            let bits = reader.read_bits(2)?;
            if bits != 0 {
                return Some(match bits {
                    0b11 => 0,
                    0b10 => 1,
                    0b01 => 2,
                    _ => unreachable!(),
                });
            }
            let bit = reader.read_bits(1)?;
            Some(if bit == 1 { 3 } else { 4 })
        }
        5 => {
            let bits = reader.read_bits(2)?;
            if bits != 0 {
                return Some(match bits {
                    0b11 => 0,
                    0b10 => 1,
                    0b01 => 2,
                    _ => unreachable!(),
                });
            }
            let bits2 = reader.read_bits(1)?;
            if bits2 == 1 {
                return Some(3);
            }
            let bits3 = reader.read_bits(1)?;
            Some(if bits3 == 1 { 4 } else { 5 })
        }
        6 => {
            let bits = reader.read_bits(2)?;
            if bits != 0 {
                return Some(match bits {
                    0b11 => 0,
                    0b10 => 1,
                    0b01 => 2,
                    _ => unreachable!(),
                });
            }
            let bits2 = reader.read_bits(1)?;
            if bits2 == 1 {
                return Some(3);
            }
            let bits3 = reader.read_bits(1)?;
            if bits3 == 1 {
                return Some(4);
            }
            let bits4 = reader.read_bits(1)?;
            Some(if bits4 == 1 { 5 } else { 6 })
        }
        _ => {
            // zeros_left >= 7: prefix code 0..0 1 xxxx (leading zeros + 1)
            let mut run = 0usize;
            loop {
                let bit = reader.read_bits(1)?;
                if bit == 1 {
                    break;
                }
                run += 1;
                if run > 15 {
                    return None;
                }
            }
            Some(run)
        }
    }
}

// ---------------------------------------------------------------------------
// Main CAVLC block decoder
// ---------------------------------------------------------------------------

/// Decodes one CAVLC-coded 4x4 residual block from the bitstream.
///
/// `nc` is the predicted number of non-zero coefficients derived from
/// neighbouring blocks (used to select the coeff_token VLC table).
pub fn decode_cavlc_block(reader: &mut BitReader, nc: i32) -> Option<CavlcResult> {
    // Step (a): read coeff_token -> (total_coeffs, trailing_ones)
    let (total_coeffs_u8, trailing_ones_u8) = read_coeff_token(reader, nc)?;
    let total_coeffs = total_coeffs_u8 as usize;
    let trailing_ones = trailing_ones_u8 as usize;

    if total_coeffs == 0 {
        return Some(CavlcResult {
            total_coeffs: 0,
            trailing_ones: 0,
            levels: Vec::new(),
            total_zeros: 0,
            runs: Vec::new(),
        });
    }

    let mut levels = Vec::with_capacity(total_coeffs);

    // Step (b): read sign of trailing ones (1 bit each, in reverse order)
    for _ in 0..trailing_ones {
        let sign_bit = reader.read_bits(1)?;
        levels.push(if sign_bit == 0 { 1 } else { -1 });
    }

    // Step (c): read remaining levels (from trailing_ones..total_coeffs)
    let mut suffix_length: u8 = if total_coeffs > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };

    for i in trailing_ones..total_coeffs {
        // Read level_prefix: count leading zeros, then a 1-bit.
        let mut level_prefix = 0u32;
        loop {
            let bit = reader.read_bits(1)?;
            if bit == 1 {
                break;
            }
            level_prefix += 1;
            if level_prefix > 20 {
                return None;
            }
        }

        let mut level_code = level_prefix as i32;

        // Read level_suffix.
        let suffix_len = if level_prefix == 14 && suffix_length == 0 {
            4 // special case
        } else if level_prefix >= 15 {
            (level_prefix as u8).saturating_sub(3)
        } else {
            suffix_length
        };

        if suffix_len > 0 {
            let level_suffix = reader.read_bits(suffix_len)? as i32;
            level_code = (level_code << suffix_len) + level_suffix;
        }

        // First non-trailing coefficient gets an offset when trailing_ones < 3.
        if i == trailing_ones && trailing_ones < 3 {
            level_code += 2;
        }

        // Convert level_code to signed level.
        let level = if level_code % 2 == 0 {
            (level_code + 2) >> 1
        } else {
            (-level_code - 1) >> 1
        };

        levels.push(level);

        // Update suffix_length based on decoded level magnitude.
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if level.unsigned_abs() > (3 << (suffix_length - 1)) {
            suffix_length += 1;
        }
    }

    // Step (d): read total_zeros.
    let max_zeros = 16 - total_coeffs;
    let total_zeros = if total_coeffs < 16 && max_zeros > 0 {
        let table = total_zeros_table(total_coeffs);
        if table.is_empty() {
            // Fallback: read as Exp-Golomb for unsupported table indices.
            reader.read_ue()? as usize
        } else {
            read_vlc(reader, table)? as usize
        }
    } else {
        0
    };

    // Step (e): read run_before for each coefficient.
    let mut runs = vec![0usize; total_coeffs];
    let mut zeros_left = total_zeros;
    for i in 0..total_coeffs - 1 {
        if zeros_left == 0 {
            break;
        }
        let run = read_run_before(reader, zeros_left)?;
        runs[i] = run;
        zeros_left = zeros_left.saturating_sub(run);
    }
    // Last coefficient gets all remaining zeros.
    if total_coeffs > 0 {
        runs[total_coeffs - 1] = zeros_left;
    }

    Some(CavlcResult {
        total_coeffs,
        trailing_ones,
        levels,
        total_zeros,
        runs,
    })
}

// ---------------------------------------------------------------------------
// Coefficient expansion
// ---------------------------------------------------------------------------

/// Expands a `CavlcResult` into a full coefficient array of `block_size`
/// elements, inserting zero runs at the correct positions.
///
/// Coefficients in the `CavlcResult` are stored in reverse scan order
/// (highest frequency first). This function places them into forward scan
/// order (DC first) and inserts the decoded zero runs.
pub fn expand_cavlc_to_coefficients(result: &CavlcResult, block_size: usize) -> Vec<i32> {
    let mut coeffs = vec![0i32; block_size];

    if result.total_coeffs == 0 {
        return coeffs;
    }

    // Walk the levels in reverse (they are stored highest-frequency-first)
    // and place them with their run-before gaps.
    let n = result.total_coeffs;
    let mut pos = block_size;

    for i in 0..n {
        // Number of zeros before this coefficient (in reverse scan).
        let run = result.runs[i];

        // Reserve space for the run of zeros.
        if pos < run {
            break;
        }
        pos -= run;

        // Place the coefficient.
        if pos == 0 {
            break;
        }
        pos -= 1;
        coeffs[pos] = result.levels[n - 1 - i];
    }

    coeffs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basic() {
        // 0b10110100 = 0xB4
        let data = [0xB4];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(1), Some(1)); // 1
        assert_eq!(r.read_bits(1), Some(0)); // 0
        assert_eq!(r.read_bits(1), Some(1)); // 1
        assert_eq!(r.read_bits(1), Some(1)); // 1
        assert_eq!(r.read_bits(1), Some(0)); // 0
        assert_eq!(r.read_bits(1), Some(1)); // 1
        assert_eq!(r.read_bits(1), Some(0)); // 0
        assert_eq!(r.read_bits(1), Some(0)); // 0
        assert_eq!(r.read_bits(1), None); // exhausted
    }

    #[test]
    fn test_bit_reader_exp_golomb_unsigned() {
        // ue(0) = 1              -> 1 bit
        // ue(1) = 010            -> 3 bits
        // ue(2) = 011            -> 3 bits
        // ue(3) = 00100          -> 5 bits
        // ue(4) = 00101          -> 5 bits
        // Total = 1+3+3+5+5 = 17 bits, padded to 3 bytes = 24 bits
        // Binary: 1 010 011 00100 00101 0000000
        //       = 1010_0110_0100_0010_1000_0000
        //       = 0xA6 0x42 0x80
        let data = [0xA6, 0x42, 0x80];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_ue(), Some(0));
        assert_eq!(r.read_ue(), Some(1));
        assert_eq!(r.read_ue(), Some(2));
        assert_eq!(r.read_ue(), Some(3));
        assert_eq!(r.read_ue(), Some(4));
    }

    #[test]
    fn test_bit_reader_exp_golomb_signed() {
        // se(0)  = ue(0) = 1           -> 1 bit
        // se(1)  = ue(1) = 010         -> 3 bits
        // se(-1) = ue(2) = 011         -> 3 bits
        // se(2)  = ue(3) = 00100       -> 5 bits
        // se(-2) = ue(4) = 00101       -> 5 bits
        // Same bitstream as unsigned test.
        let data = [0xA6, 0x42, 0x80];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_se(), Some(0));
        assert_eq!(r.read_se(), Some(1));
        assert_eq!(r.read_se(), Some(-1));
        assert_eq!(r.read_se(), Some(2));
        assert_eq!(r.read_se(), Some(-2));
    }

    #[test]
    fn test_bit_reader_multi_byte() {
        // Read a 12-bit value that spans two bytes.
        // 0xFF 0xF0 = 1111_1111 1111_0000
        let data = [0xFF, 0xF0];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(12), Some(0xFFF)); // 12 ones
        assert_eq!(r.read_bits(4), Some(0x0)); // 4 zeros
    }

    #[test]
    fn test_cavlc_all_zeros() {
        // For nc=0..1, coeff_token (0,0) is encoded as a single 1-bit.
        let data = [0x80]; // 1000_0000
        let mut r = BitReader::new(&data);
        let result = decode_cavlc_block(&mut r, 0).unwrap();
        assert_eq!(result.total_coeffs, 0);
        assert_eq!(result.trailing_ones, 0);
        assert!(result.levels.is_empty());
        assert_eq!(result.total_zeros, 0);
        assert!(result.runs.is_empty());
    }

    #[test]
    fn test_expand_coefficients() {
        // Manually construct a CavlcResult and verify expansion.
        let result = CavlcResult {
            total_coeffs: 3,
            trailing_ones: 0,
            levels: vec![5, -3, 2], // reverse scan: highest freq first
            total_zeros: 2,
            runs: vec![1, 0, 1], // run before each coeff (reverse scan order)
        };

        let coeffs = expand_cavlc_to_coefficients(&result, 8);
        // Expansion logic (placing from end, reverse scan order):
        //   coeff[0] (level=levels[2]=2): runs[0]=1 zero, then coeff -> pos skips 1 zero, places 2
        //   coeff[1] (level=levels[1]=-3): runs[1]=0, places -3
        //   coeff[2] (level=levels[0]=5): runs[2]=1 zero, then 5
        //
        // Working backwards from position 8:
        //   i=0: run=1, pos=8-1=7, pos=7-1=6 -> coeffs[6] = levels[2] = 2
        //   i=1: run=0, pos=6-0=6, pos=6-1=5 -> coeffs[5] = levels[1] = -3
        //   i=2: run=1, pos=5-1=4, pos=4-1=3 -> coeffs[3] = levels[0] = 5
        assert_eq!(coeffs, [0, 0, 0, 5, 0, -3, 2, 0]);
    }

    #[test]
    fn test_bits_remaining() {
        let data = [0xAB, 0xCD];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits_remaining(), 16);
        let _ = r.read_bits(5);
        assert_eq!(r.bits_remaining(), 11);
        let _ = r.read_bits(11);
        assert_eq!(r.bits_remaining(), 0);
    }

    #[test]
    fn test_expand_empty_block() {
        let result = CavlcResult {
            total_coeffs: 0,
            trailing_ones: 0,
            levels: Vec::new(),
            total_zeros: 0,
            runs: Vec::new(),
        };
        let coeffs = expand_cavlc_to_coefficients(&result, 16);
        assert_eq!(coeffs, vec![0; 16]);
    }

    #[test]
    fn test_cavlc_nc2_all_zeros() {
        // For nc=2..3, coeff_token (0,0) is encoded as 0b11 (2 bits).
        let data = [0xC0]; // 1100_0000
        let mut r = BitReader::new(&data);
        let result = decode_cavlc_block(&mut r, 2).unwrap();
        assert_eq!(result.total_coeffs, 0);
        assert_eq!(result.trailing_ones, 0);
    }
}
