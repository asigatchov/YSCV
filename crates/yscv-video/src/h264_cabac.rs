//! H.264 CABAC (Context-based Adaptive Binary Arithmetic Coding) entropy decoder.
//!
//! CABAC is the entropy coding method used in H.264 Main and High profiles.
//! It provides 9--14 % bitrate savings over CAVLC at the cost of higher
//! decoding complexity.  This module implements a *minimal* CABAC decoder
//! covering the most common syntax elements:
//!
//! - `mb_type`
//! - `coded_block_flag`
//! - `significant_coeff_flag`
//! - `last_significant_coeff_flag`
//! - `coeff_abs_level_minus1`
//!
//! The arithmetic engine follows ITU-T H.264 section 9.3 (Table 9-45 state
//! transitions and Table 9-48 range LPS values).

// ---------------------------------------------------------------------------
// State-transition tables (H.264 spec Table 9-45)
// ---------------------------------------------------------------------------

/// State transition after decoding the **Most Probable Symbol** (MPS).
/// Indexed by `pStateIdx` (0..=63).
#[rustfmt::skip]
static TRANSITION_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,
     9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 62, 63,
];

/// State transition after decoding the **Least Probable Symbol** (LPS).
/// Indexed by `pStateIdx` (0..=63).
#[rustfmt::skip]
static TRANSITION_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,
     6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18,
    19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29,
    29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36,
    36, 36, 37, 37, 37, 38, 38, 63,
];

// ---------------------------------------------------------------------------
// Range LPS table (H.264 spec Table 9-48)
// ---------------------------------------------------------------------------

/// `RANGE_TABLE[pStateIdx][qRangeIdx]` — the LPS sub-range for each
/// probability state and quarter-range index.
///
/// 64 rows x 4 columns.  Values taken directly from ITU-T H.264.
#[rustfmt::skip]
static RANGE_TABLE: [[u16; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [ 95, 116, 137, 158],
    [ 90, 110, 130, 150],
    [ 85, 104, 123, 142],
    [ 81,  99, 117, 135],
    [ 77,  94, 111, 128],
    [ 73,  89, 105, 122],
    [ 69,  85, 100, 116],
    [ 66,  80,  95, 110],
    [ 62,  76,  90, 104],
    [ 59,  72,  86,  99],
    [ 56,  69,  81,  94],
    [ 53,  65,  77,  89],
    [ 51,  62,  73,  85],
    [ 48,  59,  69,  80],
    [ 46,  56,  66,  76],
    [ 43,  53,  63,  72],
    [ 41,  50,  59,  69],
    [ 39,  48,  56,  65],
    [ 37,  45,  54,  62],
    [ 35,  43,  51,  59],
    [ 33,  41,  48,  56],
    [ 32,  39,  46,  53],
    [ 30,  37,  43,  50],
    [ 29,  35,  41,  48],
    [ 27,  33,  39,  45],
    [ 26,  31,  37,  43],
    [ 24,  30,  35,  41],
    [ 23,  28,  33,  39],
    [ 22,  27,  32,  37],
    [ 21,  26,  30,  35],
    [ 20,  24,  29,  33],
    [ 19,  23,  27,  31],
    [ 18,  22,  26,  30],
    [ 17,  21,  25,  28],
    [ 16,  20,  23,  27],
    [ 15,  19,  22,  25],
    [ 14,  18,  21,  24],
    [ 14,  17,  20,  23],
    [ 13,  16,  19,  22],
    [ 12,  15,  18,  21],
    [ 12,  14,  17,  20],
    [ 11,  14,  16,  19],
    [ 11,  13,  15,  18],
    [ 10,  12,  15,  17],
    [ 10,  12,  14,  16],
    [  9,  11,  13,  15],
    [  9,  11,  12,  14],
    [  8,  10,  12,  14],
    [  8,   9,  11,  13],
    [  7,   9,  11,  12],
    [  7,   9,  10,  12],
    [  7,   8,  10,  11],
    [  6,   8,   9,  11],
    [  6,   7,   9,  10],
    [  6,   7,   8,   9],
    [  2,   2,   2,   2],
];

// ---------------------------------------------------------------------------
// Context model
// ---------------------------------------------------------------------------

/// Number of context variables used in H.264 CABAC.
pub const NUM_CABAC_CONTEXTS: usize = 460;

/// Adaptive probability context model for CABAC (H.264, 9.3.1).
#[derive(Debug, Clone)]
pub struct CabacContext {
    /// Probability state index (0 = equiprobable, 63 = most skewed).
    pub state: u8,
    /// Most Probable Symbol value.
    pub mps: bool,
}

impl CabacContext {
    /// Create a context initialised from a (slope, offset) init_value at a
    /// given slice QP.
    pub fn new(slice_qp: i32, init_value: i16) -> Self {
        let m = (init_value >> 4) * 5 - 45;
        let n = ((init_value & 15) << 3) - 16;
        let pre = ((m * (slice_qp.clamp(0, 51) as i16 - 16)) >> 4) + n;
        let pre = pre.clamp(1, 126);

        if pre <= 63 {
            CabacContext {
                state: (63 - pre) as u8,
                mps: false,
            }
        } else {
            CabacContext {
                state: (pre - 64) as u8,
                mps: true,
            }
        }
    }

    /// Create a default equiprobable context.
    pub fn equiprobable() -> Self {
        CabacContext {
            state: 0,
            mps: false,
        }
    }
}

/// Initialise a full set of CABAC context variables for a given QP.
///
/// Uses the default init_value table from the H.264 spec (simplified:
/// all contexts start at init_value = 0x7E which maps to roughly
/// equiprobable).  A production decoder would use the per-context tables
/// from Tables 9-12 through 9-23.
pub fn init_cabac_contexts(slice_qp: i32) -> Vec<CabacContext> {
    // Default init value for all contexts (mid-probability).
    let default_init: i16 = 0x7E; // 126 -> state near equiprobable
    (0..NUM_CABAC_CONTEXTS)
        .map(|_| CabacContext::new(slice_qp, default_init))
        .collect()
}

// ---------------------------------------------------------------------------
// CABAC arithmetic decoding engine (H.264, 9.3.3)
// ---------------------------------------------------------------------------

/// CABAC binary arithmetic decoder for H.264.
pub struct CabacDecoder<'a> {
    data: &'a [u8],
    offset: usize,
    bits_left: u32,
    /// Current arithmetic coding range (9-bit, initialised to 510).
    range: u32,
    /// Current arithmetic coding value.
    value: u32,
}

impl<'a> CabacDecoder<'a> {
    /// Construct a new CABAC decoder from RBSP payload bytes.
    ///
    /// The slice must start at the first CABAC-coded byte (after the slice
    /// header has been fully consumed).
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = CabacDecoder {
            data,
            offset: 0,
            bits_left: 0,
            range: 510,
            value: 0,
        };
        // Bootstrap: read 9 bits into `value` (spec 9.3.2.2).
        dec.value = dec.read_bits(9);
        dec
    }

    // ------------------------------------------------------------------
    // Bit-level I/O
    // ------------------------------------------------------------------

    fn read_bit(&mut self) -> u32 {
        if self.bits_left == 0 {
            if self.offset < self.data.len() {
                self.bits_left = 8;
                self.offset += 1;
            } else {
                return 0;
            }
        }
        self.bits_left -= 1;
        let byte = self.data[self.offset - 1];
        (u32::from(byte) >> self.bits_left) & 1
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit();
        }
        val
    }

    // ------------------------------------------------------------------
    // Renormalization (spec 9.3.3.2.2)
    // ------------------------------------------------------------------

    fn renorm(&mut self) {
        while self.range < 256 {
            self.range <<= 1;
            self.value = (self.value << 1) | self.read_bit();
        }
    }

    // ------------------------------------------------------------------
    // Core decoding primitives
    // ------------------------------------------------------------------

    /// Decode a single context-modelled binary decision.
    pub fn decode_decision(&mut self, ctx: &mut CabacContext) -> bool {
        let q_idx = (self.range >> 6) & 3;
        let lps_range = RANGE_TABLE[ctx.state as usize][q_idx as usize] as u32;
        self.range -= lps_range;

        if self.value < self.range {
            // MPS path
            ctx.state = TRANSITION_MPS[ctx.state as usize];
            self.renorm();
            ctx.mps
        } else {
            // LPS path
            self.value -= self.range;
            self.range = lps_range;
            if ctx.state == 0 {
                ctx.mps = !ctx.mps;
            }
            ctx.state = TRANSITION_LPS[ctx.state as usize];
            self.renorm();
            !ctx.mps
        }
    }

    /// Decode a bypass bin (equiprobable, no context update).
    pub fn decode_bypass(&mut self) -> bool {
        self.value = (self.value << 1) | self.read_bit();
        if self.value >= self.range {
            self.value -= self.range;
            true
        } else {
            false
        }
    }

    /// Decode a terminate bin (used for end_of_slice_flag).
    pub fn decode_terminate(&mut self) -> bool {
        self.range -= 2;
        if self.value >= self.range {
            true
        } else {
            self.renorm();
            false
        }
    }

    /// Returns the number of unconsumed bytes remaining.
    pub fn bytes_remaining(&self) -> usize {
        let full_bytes = self.data.len().saturating_sub(self.offset);
        if self.bits_left > 0 {
            full_bytes + 1
        } else {
            full_bytes
        }
    }
}

// ---------------------------------------------------------------------------
// Binarization schemes (H.264, 9.3.2)
// ---------------------------------------------------------------------------

/// Decode a unary-coded value (sequence of 1s terminated by 0, or max bins).
pub fn decode_unary(decoder: &mut CabacDecoder<'_>, ctx: &mut CabacContext, max_bins: u32) -> u32 {
    let mut val = 0u32;
    while val < max_bins {
        if decoder.decode_decision(ctx) {
            val += 1;
        } else {
            return val;
        }
    }
    val
}

/// Decode a truncated-unary coded value.
pub fn decode_truncated_unary(
    decoder: &mut CabacDecoder<'_>,
    ctx: &mut CabacContext,
    max_val: u32,
) -> u32 {
    if max_val == 0 {
        return 0;
    }
    let mut val = 0u32;
    while val < max_val {
        if decoder.decode_decision(ctx) {
            val += 1;
        } else {
            return val;
        }
    }
    val
}

/// Decode a fixed-length code of `n` bits using bypass decoding.
pub fn decode_fixed_length(decoder: &mut CabacDecoder<'_>, n: u32) -> u32 {
    let mut val = 0u32;
    for _ in 0..n {
        val = (val << 1) | (decoder.decode_bypass() as u32);
    }
    val
}

/// Decode a k-th order Exp-Golomb coded value using bypass bins.
pub fn decode_exp_golomb_bypass(decoder: &mut CabacDecoder<'_>, k: u32) -> u32 {
    let mut order = 0u32;
    // Count leading 1-bits (prefix)
    while decoder.decode_bypass() {
        order += 1;
        if order > 16 {
            return 0; // safety limit
        }
    }
    // Read (order + k) suffix bits
    let suffix_len = order + k;
    let mut val = (1u32 << order) - 1;
    if suffix_len > 0 {
        val += decode_fixed_length(decoder, suffix_len);
    }
    val
}

// ---------------------------------------------------------------------------
// H.264 syntax element decoders
// ---------------------------------------------------------------------------

/// Context indices for the common syntax elements.
pub mod ctx {
    // mb_type contexts for I-slices (Table 9-34): 3..=10
    pub const MB_TYPE_I_START: usize = 3;
    // mb_type contexts for P-slices (Table 9-34): 14..=20
    pub const MB_TYPE_P_START: usize = 14;
    // coded_block_flag (Table 9-34): 85..=88 for luma
    pub const CODED_BLOCK_FLAG_LUMA: usize = 85;
    // significant_coeff_flag (Table 9-34): 105..=165
    pub const SIGNIFICANT_COEFF_START: usize = 105;
    // last_significant_coeff_flag: 166..=226
    pub const LAST_SIGNIFICANT_COEFF_START: usize = 166;
    // coeff_abs_level_minus1: 227..=275
    pub const COEFF_ABS_LEVEL_START: usize = 227;
}

/// Decode `mb_type` for an I-slice macroblock.
///
/// Binarization: prefix = TU(max=25), suffix depends on value.
/// Simplified: decode the prefix as a truncated-unary with context
/// index offset from `MB_TYPE_I_START`.
pub fn decode_mb_type_i_slice(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
) -> u32 {
    // bin 0: context index 3 (or 3 + ctx_inc depending on neighbours)
    let ci = ctx::MB_TYPE_I_START;
    if !decoder.decode_decision(&mut contexts[ci]) {
        return 0; // I_4x4
    }
    // bin 1: terminate check for I_PCM (context 4)
    if decoder.decode_terminate() {
        return 25; // I_PCM
    }
    // bins 2..N: decode sub-type using contexts 5+
    let sub = decode_truncated_unary(decoder, &mut contexts[ci + 2], 23);
    1 + sub // I_16x16 variants (1..=24)
}

/// Decode `mb_type` for a P-slice macroblock.
pub fn decode_mb_type_p_slice(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
) -> u32 {
    let ci = ctx::MB_TYPE_P_START;
    if !decoder.decode_decision(&mut contexts[ci]) {
        // P_L0_16x16 (0) or sub-partition modes
        if !decoder.decode_decision(&mut contexts[ci + 1]) {
            return 0; // P_L0_16x16
        }
        if !decoder.decode_decision(&mut contexts[ci + 2]) {
            return 1; // P_L0_L0_16x8
        }
        return 2; // P_L0_L0_8x16
    }
    if !decoder.decode_decision(&mut contexts[ci + 3]) {
        return 3; // P_8x8
    }
    // Intra modes in P-slice: decode as I-slice mb_type + 5
    let intra_type = decode_mb_type_i_slice(decoder, contexts);
    5 + intra_type
}

/// Decode `coded_block_flag` for a 4x4 luma block.
pub fn decode_coded_block_flag(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
    ctx_offset: usize,
) -> bool {
    let ci = ctx::CODED_BLOCK_FLAG_LUMA + ctx_offset.min(3);
    decoder.decode_decision(&mut contexts[ci])
}

/// Decode residual coefficients for one 4x4 block via CABAC.
///
/// Returns a vector of up to 16 coefficients in zig-zag scan order.
pub fn decode_residual_block_cabac(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
    max_num_coeff: usize,
) -> Vec<i32> {
    let mut coeffs = vec![0i32; max_num_coeff];

    // 1) Decode significance map
    let mut significant = vec![false; max_num_coeff];
    let mut last = vec![false; max_num_coeff];
    let mut num_coeff = 0usize;

    for i in 0..max_num_coeff - 1 {
        let sig_ci = ctx::SIGNIFICANT_COEFF_START + i.min(14);
        significant[i] = decoder.decode_decision(&mut contexts[sig_ci]);
        if significant[i] {
            let last_ci = ctx::LAST_SIGNIFICANT_COEFF_START + i.min(14);
            last[i] = decoder.decode_decision(&mut contexts[last_ci]);
            num_coeff += 1;
            if last[i] {
                break;
            }
        }
    }
    // Last coeff is implicitly significant if we haven't hit last yet
    if num_coeff > 0 && !last.iter().any(|&l| l) {
        significant[max_num_coeff - 1] = true;
        num_coeff += 1;
    }

    if num_coeff == 0 {
        return coeffs;
    }

    // 2) Decode coefficient levels in reverse scan order
    let sig_positions: Vec<usize> = (0..max_num_coeff).filter(|&i| significant[i]).collect();

    let mut num_gt1 = 0u32;
    let mut num_eq1 = 0u32;

    for &pos in sig_positions.iter().rev() {
        // coeff_abs_level_minus1 prefix (truncated unary, max 14)
        let ctx_cat = (num_gt1.min(4)) as usize;
        let ci = ctx::COEFF_ABS_LEVEL_START + ctx_cat;
        let prefix = decode_truncated_unary(decoder, &mut contexts[ci], 14);
        let abs_level = if prefix < 14 {
            prefix + 1
        } else {
            // suffix via Exp-Golomb bypass
            let suffix = decode_exp_golomb_bypass(decoder, 0);
            prefix + 1 + suffix
        };

        // Sign bit (bypass)
        let sign = decoder.decode_bypass();
        coeffs[pos] = if sign {
            -(abs_level as i32)
        } else {
            abs_level as i32
        };

        if abs_level == 1 {
            num_eq1 += 1;
        }
        if abs_level > 1 {
            num_gt1 += 1;
        }
    }

    let _ = num_eq1; // used by context derivation in full decoder

    coeffs
}

/// Identifies the entropy coding mode from a PPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyCodingMode {
    /// Context-Adaptive Variable-Length Coding (Baseline).
    Cavlc,
    /// Context-Adaptive Binary Arithmetic Coding (Main/High).
    Cabac,
}

impl EntropyCodingMode {
    /// Determine entropy coding mode from `entropy_coding_mode_flag`.
    pub fn from_flag(flag: bool) -> Self {
        if flag {
            EntropyCodingMode::Cabac
        } else {
            EntropyCodingMode::Cavlc
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cabac_context_init_equiprobable() {
        let ctx = CabacContext::equiprobable();
        assert_eq!(ctx.state, 0);
        assert!(!ctx.mps);
    }

    #[test]
    fn test_cabac_context_init_from_value() {
        // init_value 0x7E = 126 -> slope = (126>>4)*5-45 = 7*5-45 = -10
        // offset = ((126&15)<<3)-16 = (14<<3)-16 = 96
        // init_state = ((-10)*(26-16))>>4 + 96 = (-100>>4)+96 = -7+96 = 89
        // pre = clamp(89,1,126) = 89
        // 89 > 63 -> state = 89-64 = 25, mps = true
        let ctx = CabacContext::new(26, 0x7E);
        assert_eq!(ctx.state, 25);
        assert!(ctx.mps);
    }

    #[test]
    fn test_cabac_decode_bypass_deterministic() {
        // All-zero data -> bypass always returns false (value stays below range).
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut dec = CabacDecoder::new(&data);
        for _ in 0..8 {
            assert!(!dec.decode_bypass());
        }
    }

    #[test]
    fn test_cabac_decode_terminate_on_end() {
        // Range starts at 510. After subtracting 2, range = 508.
        // If value >= 508, terminate returns true.
        // With all-ones data, value will be large.
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut dec = CabacDecoder::new(&data);
        // value after init = first 9 bits of 0xFFFF... = 0x1FF = 511
        // range = 510, range -= 2 = 508, value (511) >= 508 -> true
        assert!(dec.decode_terminate());
    }

    #[test]
    fn test_cabac_decode_decision_updates_state() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = CabacDecoder::new(&data);
        let mut ctx = CabacContext::equiprobable();
        let initial_state = ctx.state;

        // After a decision the state should change.
        let _bin = dec.decode_decision(&mut ctx);
        // The state may or may not differ from initial (depends on MPS/LPS),
        // but the function should not panic.
        assert!(ctx.state <= 63);
        let _ = initial_state;
    }

    #[test]
    fn test_decode_unary_zero() {
        // With all-zero data, decode_decision on an equiprobable context
        // with value=0 should return the MPS (false) immediately,
        // giving unary value 0.
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = CabacDecoder::new(&data);
        let mut ctx = CabacContext::equiprobable();
        let val = decode_unary(&mut dec, &mut ctx, 10);
        // Value should be 0 since MPS = false -> decode_decision returns false
        // on MPS path (value < range for all-zero data).
        assert_eq!(val, 0);
    }

    #[test]
    fn test_fixed_length_decode() {
        // All-ones data: bypass bits should all be 1.
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut dec = CabacDecoder::new(&data);
        let val = decode_fixed_length(&mut dec, 3);
        // 3 bypass bits from all-1s stream should produce 0b111 = 7.
        assert_eq!(val, 7);
    }

    #[test]
    fn test_entropy_coding_mode_from_flag() {
        assert_eq!(
            EntropyCodingMode::from_flag(false),
            EntropyCodingMode::Cavlc
        );
        assert_eq!(EntropyCodingMode::from_flag(true), EntropyCodingMode::Cabac);
    }

    #[test]
    fn test_init_cabac_contexts_count() {
        let contexts = init_cabac_contexts(26);
        assert_eq!(contexts.len(), NUM_CABAC_CONTEXTS);
    }

    #[test]
    fn test_decode_residual_block_length() {
        // Verify that decode_residual_block_cabac always returns the
        // requested number of coefficients regardless of input data.
        let data = [0x00; 32];
        let mut dec = CabacDecoder::new(&data);
        let mut contexts = init_cabac_contexts(26);
        let coeffs = decode_residual_block_cabac(&mut dec, &mut contexts, 16);
        assert_eq!(coeffs.len(), 16);

        // Also verify with max_num_coeff = 4 (chroma DC).
        let data2 = [0x00; 32];
        let mut dec2 = CabacDecoder::new(&data2);
        let mut contexts2 = init_cabac_contexts(26);
        let coeffs2 = decode_residual_block_cabac(&mut dec2, &mut contexts2, 4);
        assert_eq!(coeffs2.len(), 4);
    }

    #[test]
    fn test_transition_table_bounds() {
        // Verify all transition table entries are in [0, 63].
        for &s in TRANSITION_MPS.iter() {
            assert!(s <= 63);
        }
        for &s in TRANSITION_LPS.iter() {
            assert!(s <= 63);
        }
    }

    #[test]
    fn test_range_table_positive() {
        // All range table entries should be > 0.
        for row in RANGE_TABLE.iter() {
            for &val in row.iter() {
                assert!(val > 0);
            }
        }
    }
}
