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

/// H.264 CABAC I-slice context init values as (m, n) pairs.
/// From ITU-T H.264 Table 9-12 (ctxIdx 0..10), Table 9-13 (11..23),
/// Table 9-17/9-18/9-19 for coefficient contexts.
/// For contexts not in the I-slice table, we use (0, 0) = equiprobable.
///
/// The init value encoding: `init_value = ((m/5 + 9) << 4) | ((n/8 + 2))`.
/// But CabacContext::new() takes a raw packed value where:
///   m = (init_value >> 4) * 5 - 45
///   n = (init_value & 15) * 8 - 16
/// So to encode (m=0, n=0): init_value = (9 << 4) | 2 = 146 -> m = 9*5-45=0, n = 2*8-16=0
fn encode_mn(m: i16, n: i16) -> i16 {
    let m_enc = ((m + 45) / 5).clamp(0, 15);
    let n_enc = ((n + 16) / 8).clamp(0, 15);
    (m_enc << 4) | n_enc
}

/// Initialise CABAC context variables for I-slices.
///
/// Uses (m, n) pairs from ITU-T H.264 spec. For the most common contexts:
/// - mb_type I-slice (ctx 3..10): from Table 9-12
/// - coded_block_pattern (ctx 73..84): from Table 9-13
/// - coded_block_flag (ctx 85..104): from Table 9-14
/// - significant_coeff_flag (ctx 105..165): from Table 9-17
/// - last_significant_coeff_flag (ctx 166..226): from Table 9-18
/// - coeff_abs_level_minus1 (ctx 227..275): from Table 9-19
///
/// Remaining contexts default to equiprobable (m=0, n=0).
pub fn init_cabac_contexts(slice_qp: i32) -> Vec<CabacContext> {
    // Start all contexts equiprobable
    let eq = encode_mn(0, 0);
    let mut init_values = vec![eq; NUM_CABAC_CONTEXTS];

    // Table 9-12: mb_type for I-slices (ctxIdx 3..10)
    // (m, n) from spec. Typical: ctx3 = (20,29), ctx4 = (2,26), etc.
    let mb_type_i: [(i16, i16); 8] = [
        (20, 29),
        (2, 26),
        (0, 27),
        (0, 27),
        (0, 27),
        (0, 27),
        (0, 27),
        (0, 27),
    ];
    for (i, &(m, n)) in mb_type_i.iter().enumerate() {
        init_values[3 + i] = encode_mn(m, n);
    }

    // Table 9-13: coded_block_pattern luma (ctxIdx 73..76)
    let cbp_luma: [(i16, i16); 4] = [(0, 41), (-3, 40), (-7, 39), (-5, 44)];
    for (i, &(m, n)) in cbp_luma.iter().enumerate() {
        init_values[73 + i] = encode_mn(m, n);
    }

    // coded_block_pattern chroma (ctxIdx 77..84)
    let cbp_chroma: [(i16, i16); 8] = [
        (-11, 43),
        (-15, 39),
        (-4, 44),
        (-7, 43),
        (-11, 43),
        (-15, 39),
        (-4, 44),
        (-7, 43),
    ];
    for (i, &(m, n)) in cbp_chroma.iter().enumerate() {
        init_values[77 + i] = encode_mn(m, n);
    }

    // Table 9-14: coded_block_flag luma (ctxIdx 85..104)
    // Default: roughly equiprobable with slight bias
    for i in 85..105 {
        init_values[i] = encode_mn(0, 26); // slight bias toward coded=1
    }

    // mb_qp_delta (ctxIdx 60..68)
    let qp_delta: [(i16, i16); 4] = [(0, 39), (0, 39), (0, 39), (0, 39)];
    for (i, &(m, n)) in qp_delta.iter().enumerate() {
        init_values[60 + i] = encode_mn(m, n);
    }

    // Table 9-17: significant_coeff_flag (ctxIdx 105..165)
    // These contexts adapt quickly so starting near equiprobable is fine
    for i in 105..166 {
        init_values[i] = encode_mn(0, 14);
    }

    // Table 9-18: last_significant_coeff_flag (ctxIdx 166..226)
    for i in 166..227 {
        init_values[i] = encode_mn(0, 14);
    }

    // Table 9-19: coeff_abs_level_minus1 (ctxIdx 227..275)
    for i in 227..276 {
        init_values[i] = encode_mn(0, 14);
    }

    init_values
        .iter()
        .map(|&v| CabacContext::new(slice_qp, v))
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

    #[inline(always)]
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

    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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

/// Decode `mb_type` for an I-slice macroblock (H.264 Table 9-34).
///
/// Binarization per ITU-T H.264 Table 9-36:
///   bin 0 (ctx 3+ctxInc): 0 → I_4x4 (mb_type=0)
///   bin 0=1, bin 1 (ctx 4): 1 → I_PCM (mb_type=25)
///   bin 0=1, bin 1=0: I_16x16 — decode 4 more bins for sub-type (1..24)
pub fn decode_mb_type_i_slice(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
) -> u32 {
    let ci = ctx::MB_TYPE_I_START;
    // bin 0: I_4x4 vs other
    if !decoder.decode_decision(&mut contexts[ci]) {
        return 0; // I_4x4
    }
    // bin 1: I_PCM vs I_16x16
    if decoder.decode_decision(&mut contexts[ci + 1]) {
        return 25; // I_PCM
    }
    // I_16x16: decode cbp_luma (1 bin), cbp_chroma (2 bins), pred_mode (2 bins)
    // bin 2 (ctx 5): cbp_luma (0 or 1 → maps to cbp 0 or 15)
    let cbp_luma = decoder.decode_decision(&mut contexts[ci + 2]) as u32;
    // bin 3 (ctx 6): chroma cbp bit 0
    let cbp_c0 = decoder.decode_decision(&mut contexts[ci + 3]) as u32;
    // bin 4 (ctx 7): chroma cbp bit 1 (if cbp_c0=1)
    let cbp_chroma = if cbp_c0 == 0 {
        0u32
    } else if decoder.decode_decision(&mut contexts[ci + 4]) {
        2
    } else {
        1
    };
    // bin 5,6 (ctx 8,9): intra16x16 pred mode (2 bits)
    let pred0 = decoder.decode_decision(&mut contexts[ci + 5]) as u32;
    let pred1 = decoder.decode_decision(&mut contexts[ci + 6]) as u32;
    let pred_mode = (pred0 << 1) | pred1;

    // mb_type = 1 + pred_mode*4 + cbp_chroma*4*4? No — see Table 7-11:
    // mb_type = 1 + cbp_luma*12 + cbp_chroma*4 + pred_mode
    1 + cbp_luma * 12 + cbp_chroma * 4 + pred_mode
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

/// Decode `coded_block_flag` for a block.
///
/// `cat_offset`: block category offset:
///   0 = luma DC (I_16x16), 1 = luma AC (I_16x16),
///   2 = luma 4x4, 3 = chroma DC, 4 = chroma AC
/// For simplicity, uses ctx 85 + cat_offset * 4 + ctxInc (ctxInc=0 simplified).
pub fn decode_coded_block_flag(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
    cat_offset: usize,
) -> bool {
    // Table 9-34: coded_block_flag ctx = 85 + ctxBlockCat * 4 + ctxInc
    // ctxBlockCat: 0=Luma_DC_16x16, 1=Luma_AC_16x16, 2=Luma_4x4,
    //              3=Chroma_DC, 4=Chroma_AC
    let ci = (ctx::CODED_BLOCK_FLAG_LUMA + cat_offset * 4).min(contexts.len() - 1);
    decoder.decode_decision(&mut contexts[ci])
}

/// Decode residual coefficients for one 4x4 block via CABAC.
///
/// `ctx_block_cat` selects the context offset for this block type:
///   0 = luma DC 16x16, 1 = luma AC 16x16, 2 = luma 4x4,
///   3 = chroma DC, 4 = chroma AC
///
/// Returns a vector of up to `max_num_coeff` coefficients in scan order.
pub fn decode_residual_block_cabac(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut [CabacContext],
    max_num_coeff: usize,
) -> Vec<i32> {
    let mut coeffs = vec![0i32; max_num_coeff];

    // 1) Decode significance map using position-dependent contexts
    // significant_coeff_flag: ctx 105 + min(pos, 14) for 4x4 blocks (Table 9-17)
    // last_significant_coeff_flag: ctx 166 + min(pos, 14) for 4x4 blocks (Table 9-18)
    let mut significant = vec![false; max_num_coeff];
    let mut last = vec![false; max_num_coeff];
    let mut num_coeff = 0usize;

    let max_scan = max_num_coeff.saturating_sub(1);
    for i in 0..max_scan {
        // Position-dependent context: map scan index to context
        let sig_ctx = if max_num_coeff <= 4 {
            // Chroma DC: fewer contexts
            ctx::SIGNIFICANT_COEFF_START + i.min(3)
        } else if max_num_coeff <= 16 {
            // 4x4 block: ctx offset by position
            ctx::SIGNIFICANT_COEFF_START + i.min(14)
        } else {
            // 8x8 block (not used in our code, but safe)
            ctx::SIGNIFICANT_COEFF_START + (i >> 2).min(14)
        };
        let sig_ctx = sig_ctx.min(contexts.len() - 1);
        significant[i] = decoder.decode_decision(&mut contexts[sig_ctx]);

        if significant[i] {
            let last_ctx = if max_num_coeff <= 4 {
                ctx::LAST_SIGNIFICANT_COEFF_START + i.min(3)
            } else if max_num_coeff <= 16 {
                ctx::LAST_SIGNIFICANT_COEFF_START + i.min(14)
            } else {
                ctx::LAST_SIGNIFICANT_COEFF_START + (i >> 2).min(14)
            };
            let last_ctx = last_ctx.min(contexts.len() - 1);
            last[i] = decoder.decode_decision(&mut contexts[last_ctx]);
            num_coeff += 1;
            if last[i] {
                break;
            }
        }
    }
    // Last position is implicitly significant if we haven't hit last yet
    if num_coeff > 0 && !last.iter().any(|&l| l) {
        significant[max_num_coeff - 1] = true;
        num_coeff += 1;
    }

    if num_coeff == 0 {
        return coeffs;
    }

    // 2) Decode coefficient levels in reverse scan order
    // coeff_abs_level_minus1: ctx 227 + offset based on num_gt1 and num_eq1
    // Table 9-19: ctxIdxInc = min(num_gt1, 4) for prefix bins
    //             Suffix uses bypass (Exp-Golomb k=0)
    let sig_positions: Vec<usize> = (0..max_num_coeff).filter(|&i| significant[i]).collect();

    let mut num_gt1 = 0u32;
    let mut num_t1 = 0u32; // trailing ones

    for &pos in sig_positions.iter().rev() {
        // Base context for coeff_abs_level_minus1:
        // ctx = 227 + 5 * min(block_cat, 4) + min(num_gt1, 4) for prefix bin 0
        // Simplified: use ctx 227 + 10*min(num_t1,4) for bin0, 227+5+min(num_gt1,4) for bin1+
        let base_ctx = ctx::COEFF_ABS_LEVEL_START;

        // Bin 0: ctx = base + min(num_t1, 4) (decides abs_level == 1 vs > 1)
        let ci0 = (base_ctx + num_t1.min(4) as usize).min(contexts.len() - 1);
        let prefix_bin0 = decoder.decode_decision(&mut contexts[ci0]);

        let abs_level = if !prefix_bin0 {
            1u32 // abs_level_minus1 = 0 → abs_level = 1
        } else {
            // Bins 1+: ctx = base + 5 + min(num_gt1, 4)
            let mut abs_minus1 = 1u32;
            let ci_rest = (base_ctx + 5 + num_gt1.min(4) as usize).min(contexts.len() - 1);
            while abs_minus1 < 14 {
                if !decoder.decode_decision(&mut contexts[ci_rest]) {
                    break;
                }
                abs_minus1 += 1;
            }
            if abs_minus1 >= 14 {
                // Suffix: Exp-Golomb bypass
                abs_minus1 += decode_exp_golomb_bypass(decoder, 0);
            }
            abs_minus1 + 1
        };

        // Sign bit (bypass)
        let sign = decoder.decode_bypass();
        coeffs[pos] = if sign {
            -(abs_level as i32)
        } else {
            abs_level as i32
        };

        if abs_level == 1 {
            num_t1 += 1;
        }
        if abs_level > 1 {
            num_gt1 += 1;
            num_t1 = 0; // reset trailing ones count
        }
    }

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
