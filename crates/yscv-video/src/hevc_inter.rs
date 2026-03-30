//! HEVC inter prediction: reference picture management, motion compensation,
//! merge mode, AMVP, and inter CU parsing (ITU-T H.265 sections 8.5, 8.5.3).
//!
//! This module enables P-slice and B-slice decoding by providing:
//! - [`HevcDpb`] — Decoded Picture Buffer for reference picture management.
//! - [`HevcMv`] / [`HevcMvField`] — Quarter-pel motion vector types.
//! - Luma motion compensation with 8-tap interpolation (Table 8-4).
//! - Bi-prediction averaging.
//! - Merge candidate list construction (spatial candidates).
//! - AMVP candidate list construction.
//! - MVD parsing from CABAC.
//! - Inter CU prediction data parsing.

use super::hevc_cabac::CabacDecoder;
use super::hevc_decoder::{HevcSliceType, HevcSps};
use super::hevc_syntax::HevcSliceCabacState;

// ---------------------------------------------------------------------------
// Reference picture
// ---------------------------------------------------------------------------

/// A reconstructed reference picture stored in the DPB.
#[derive(Debug, Clone)]
pub struct HevcReferencePicture {
    /// Picture Order Count.
    pub poc: i32,
    /// Reconstructed luma samples (row-major, `width * height` elements).
    pub luma: Vec<u8>,
    /// Picture width in luma samples.
    pub width: usize,
    /// Picture height in luma samples.
    pub height: usize,
    /// Whether this is a long-term reference picture.
    pub is_long_term: bool,
}

// ---------------------------------------------------------------------------
// Decoded Picture Buffer (DPB)
// ---------------------------------------------------------------------------

/// Decoded Picture Buffer — manages reference pictures for inter prediction.
///
/// The maximum size is derived from `sps_max_dec_pic_buffering_minus1 + 1`.
#[derive(Debug)]
pub struct HevcDpb {
    pictures: Vec<HevcReferencePicture>,
    max_size: usize,
}

impl HevcDpb {
    /// Create a new DPB that holds at most `max_size` reference pictures.
    pub fn new(max_size: usize) -> Self {
        Self {
            pictures: Vec::new(),
            max_size: max_size.max(1),
        }
    }

    /// Insert a reference picture into the DPB.
    ///
    /// If the buffer is at capacity the oldest picture is bumped first.
    pub fn add(&mut self, pic: HevcReferencePicture) {
        if self.pictures.len() >= self.max_size {
            self.bump();
        }
        self.pictures.push(pic);
    }

    /// Look up a reference picture by its POC.
    pub fn get_by_poc(&self, poc: i32) -> Option<&HevcReferencePicture> {
        self.pictures.iter().find(|p| p.poc == poc)
    }

    /// Mark a picture as unused by setting its long-term flag to false and
    /// removing it from the buffer.
    pub fn mark_unused(&mut self, poc: i32) {
        self.pictures.retain(|p| p.poc != poc);
    }

    /// Remove the oldest picture (lowest POC) to make room.
    pub fn bump(&mut self) {
        if self.pictures.is_empty() {
            return;
        }
        // Find the picture with the smallest POC.
        let min_idx = self
            .pictures
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.poc)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.pictures.remove(min_idx);
    }

    /// Remove all pictures from the DPB (IDR flush).
    pub fn clear(&mut self) {
        self.pictures.clear();
    }

    /// Number of pictures currently in the buffer.
    pub fn len(&self) -> usize {
        self.pictures.len()
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.pictures.is_empty()
    }

    /// Maximum capacity.
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

// ---------------------------------------------------------------------------
// Motion vector types
// ---------------------------------------------------------------------------

/// Quarter-pel motion vector (14-bit range + sign per component).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HevcMv {
    /// Horizontal displacement in quarter-pel units.
    pub x: i16,
    /// Vertical displacement in quarter-pel units.
    pub y: i16,
}

impl HevcMv {
    /// Create a motion vector from integer-pel coordinates (internally scaled
    /// to quarter-pel).
    pub fn from_fullpel(x: i16, y: i16) -> Self {
        Self { x: x * 4, y: y * 4 }
    }

    /// Add two motion vectors (e.g. predictor + difference).
    pub fn add(self, other: HevcMv) -> HevcMv {
        HevcMv {
            x: self.x.saturating_add(other.x),
            y: self.y.saturating_add(other.y),
        }
    }

    /// Negate both components.
    pub fn negate(self) -> HevcMv {
        HevcMv {
            x: self.x.saturating_neg(),
            y: self.y.saturating_neg(),
        }
    }
}

/// Per-PU motion information for L0 and L1 reference lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HevcMvField {
    /// Motion vectors for L0 and L1.
    pub mv: [HevcMv; 2],
    /// Reference picture indices for L0 and L1 (-1 = unused).
    pub ref_idx: [i8; 2],
    /// Whether L0 / L1 prediction is active.
    pub pred_flag: [bool; 2],
}

impl HevcMvField {
    /// An empty (no-prediction) field.
    pub fn unavailable() -> Self {
        Self {
            mv: [HevcMv::default(); 2],
            ref_idx: [-1, -1],
            pred_flag: [false, false],
        }
    }

    /// Returns `true` when at least one list is active.
    pub fn is_available(&self) -> bool {
        self.pred_flag[0] || self.pred_flag[1]
    }
}

// ---------------------------------------------------------------------------
// Luma interpolation filter (HEVC spec Table 8-4)
// ---------------------------------------------------------------------------

/// 8-tap luma interpolation filter coefficients.
///
/// Indexed by fractional position (0 = integer, 1 = quarter, 2 = half,
/// 3 = three-quarter).
const HEVC_LUMA_FILTER: [[i16; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],        // Integer pel
    [-1, 4, -10, 58, 17, -5, 1, 0],   // Quarter pel
    [-1, 4, -11, 40, 40, -11, 4, -1], // Half pel
    [0, 1, -5, 17, 58, -10, 4, -1],   // Three-quarter pel
];

// ---------------------------------------------------------------------------
// Motion compensation
// ---------------------------------------------------------------------------

/// Fetch a reference sample with edge clamping.
#[inline]
fn ref_sample(pic: &HevcReferencePicture, x: i32, y: i32) -> i16 {
    let cx = x.clamp(0, pic.width as i32 - 1) as usize;
    let cy = y.clamp(0, pic.height as i32 - 1) as usize;
    pic.luma[cy * pic.width + cx] as i16
}

/// Luma motion compensation with quarter-pel interpolation (8-tap filter).
///
/// `x`, `y` are the block position in the current picture (full-pel).
/// `mv` is in quarter-pel units.
/// `output` receives `block_w * block_h` samples as `i16` (for bi-pred
/// averaging before final clipping).
#[allow(clippy::too_many_arguments)]
pub fn hevc_mc_luma(
    ref_pic: &HevcReferencePicture,
    x: i32,
    y: i32,
    mv: HevcMv,
    block_w: usize,
    block_h: usize,
    output: &mut [i16],
) {
    debug_assert!(output.len() >= block_w * block_h);

    let frac_x = ((mv.x as i32) & 3) as usize;
    let frac_y = ((mv.y as i32) & 3) as usize;
    let int_x = x + (mv.x as i32 >> 2);
    let int_y = y + (mv.y as i32 >> 2);

    let filter_h = &HEVC_LUMA_FILTER[frac_x];
    let filter_v = &HEVC_LUMA_FILTER[frac_y];

    if frac_x == 0 && frac_y == 0 {
        // Integer-pel: direct copy.
        for row in 0..block_h {
            for col in 0..block_w {
                output[row * block_w + col] =
                    ref_sample(ref_pic, int_x + col as i32, int_y + row as i32);
            }
        }
        return;
    }

    if frac_y == 0 {
        // Horizontal-only filtering.
        for row in 0..block_h {
            for col in 0..block_w {
                let sy = int_y + row as i32;
                let mut sum = 0i32;
                for k in 0..8 {
                    let sx = int_x + col as i32 + k as i32 - 3;
                    sum += ref_sample(ref_pic, sx, sy) as i32 * filter_h[k] as i32;
                }
                output[row * block_w + col] = ((sum + 32) >> 6) as i16;
            }
        }
        return;
    }

    if frac_x == 0 {
        // Vertical-only filtering.
        for row in 0..block_h {
            for col in 0..block_w {
                let sx = int_x + col as i32;
                let mut sum = 0i32;
                for k in 0..8 {
                    let sy = int_y + row as i32 + k as i32 - 3;
                    sum += ref_sample(ref_pic, sx, sy) as i32 * filter_v[k] as i32;
                }
                output[row * block_w + col] = ((sum + 32) >> 6) as i16;
            }
        }
        return;
    }

    // Both horizontal and vertical fractional position: two-pass filtering.
    // First filter horizontally into a temporary buffer (with extra rows for
    // the vertical filter tap extent), then filter vertically.
    let ext_h = block_h + 7; // 3 rows above + 4 rows below
    // Stack buffer: max 71*64 = 4544 i32 = 18KB (safe for stack, not in recursion)
    let tmp_size = ext_h * block_w;
    let mut tmp_buf = [0i32; 71 * 64];
    let tmp = &mut tmp_buf[..tmp_size];

    // Horizontal pass: produce (block_h + 7) rows of block_w intermediate
    // samples shifted by 6 bits.
    for row in 0..ext_h {
        let sy = int_y + row as i32 - 3;
        for col in 0..block_w {
            let mut sum = 0i32;
            for k in 0..8 {
                let sx = int_x + col as i32 + k as i32 - 3;
                sum += ref_sample(ref_pic, sx, sy) as i32 * filter_h[k] as i32;
            }
            tmp[row * block_w + col] = sum; // keep <<6 headroom
        }
    }

    // Vertical pass over the intermediate buffer.
    for row in 0..block_h {
        for col in 0..block_w {
            let mut sum = 0i64;
            for k in 0..8 {
                sum += tmp[(row + k) * block_w + col] as i64 * filter_v[k] as i64;
            }
            // Two shifts: +32 from h-pass kept, +2048 for v-pass rounding.
            output[row * block_w + col] = ((sum + 2048) >> 12) as i16;
        }
    }
}

/// Bi-prediction average: combine L0 and L1 predictions and clip to [0, 255].
///
/// `pred_l0` and `pred_l1` are intermediate i16 values produced by
/// [`hevc_mc_luma`].  The final result in `output` is clipped to u8.
pub fn hevc_bipred_average(pred_l0: &[i16], pred_l1: &[i16], output: &mut [u8], size: usize) {
    debug_assert!(pred_l0.len() >= size);
    debug_assert!(pred_l1.len() >= size);
    debug_assert!(output.len() >= size);

    for i in 0..size {
        let avg = (pred_l0[i] as i32 + pred_l1[i] as i32 + 1) >> 1;
        output[i] = avg.clamp(0, 255) as u8;
    }
}

/// Uni-prediction clip: convert an intermediate i16 buffer to u8.
pub fn hevc_unipred_clip(pred: &[i16], output: &mut [u8], size: usize) {
    debug_assert!(pred.len() >= size);
    debug_assert!(output.len() >= size);
    for i in 0..size {
        output[i] = (pred[i] as i32).clamp(0, 255) as u8;
    }
}

// ---------------------------------------------------------------------------
// Merge mode (ITU-T H.265, 8.5.3.2)
// ---------------------------------------------------------------------------

/// Minimum PU size in luma samples (4x4 in HEVC).
const MIN_PU_SIZE: usize = 4;

/// Build the merge candidate list from spatial neighbours (simplified).
///
/// Full HEVC spec constructs up to 5 spatial candidates (A0, A1, B0, B1, B2),
/// one temporal candidate, and combined bi-pred candidates.  This
/// implementation covers the spatial candidates.
///
/// `mv_field` is the per-min-PU motion field of the current picture,
/// dimensioned `pic_width_in_min_pu * pic_height_in_min_pu`.
pub fn build_merge_candidates(
    mv_field: &[HevcMvField],
    pic_width_in_min_pu: usize,
    x: usize,
    y: usize,
    block_w: usize,
    block_h: usize,
) -> Vec<HevcMvField> {
    let mut candidates: Vec<HevcMvField> = Vec::with_capacity(5);
    let max_candidates = 5usize;

    let pu_x = x / MIN_PU_SIZE;
    let pu_y = y / MIN_PU_SIZE;
    let pu_w = block_w / MIN_PU_SIZE;
    let pu_h = block_h / MIN_PU_SIZE;

    // Helper to fetch a candidate if available.
    let get = |px: usize, py: usize| -> Option<HevcMvField> {
        if px < pic_width_in_min_pu {
            let idx = py * pic_width_in_min_pu + px;
            if idx < mv_field.len() {
                let f = mv_field[idx];
                if f.is_available() {
                    return Some(f);
                }
            }
        }
        None
    };

    // A1: left-bottom  — (x - 1, y + block_h - 1)
    if pu_x > 0
        && let Some(c) = get(pu_x - 1, pu_y + pu_h - 1)
    {
        candidates.push(c);
    }

    // B1: top-right   — (x + block_w - 1, y - 1)
    if candidates.len() < max_candidates
        && pu_y > 0
        && let Some(c) = get(pu_x + pu_w - 1, pu_y - 1)
        && (candidates.is_empty() || candidates[candidates.len() - 1] != c)
    {
        candidates.push(c);
    }

    // B0: top-right+1 — (x + block_w, y - 1)
    if candidates.len() < max_candidates
        && pu_y > 0
        && let Some(c) = get(pu_x + pu_w, pu_y - 1)
        && candidates.last() != Some(&c)
    {
        candidates.push(c);
    }

    // A0: left-bottom+1 — (x - 1, y + block_h)
    if candidates.len() < max_candidates
        && pu_x > 0
        && let Some(c) = get(pu_x - 1, pu_y + pu_h)
        && candidates.last() != Some(&c)
    {
        candidates.push(c);
    }

    // B2: top-left    — (x - 1, y - 1)
    if candidates.len() < max_candidates
        && pu_x > 0
        && pu_y > 0
        && let Some(c) = get(pu_x - 1, pu_y - 1)
        && candidates.last() != Some(&c)
    {
        candidates.push(c);
    }

    // Pad with zero-MV fields up to max_candidates if fewer were found.
    while candidates.len() < max_candidates {
        candidates.push(HevcMvField {
            mv: [HevcMv::default(); 2],
            ref_idx: [0, -1],
            pred_flag: [true, false],
        });
    }

    candidates
}

/// Parse `merge_idx` from the CABAC bitstream (truncated unary, bypass coded).
///
/// Returns a value in `0 ..= max_merge_cand - 1`.
pub fn parse_merge_idx(cabac: &mut CabacDecoder<'_>, max_merge_cand: u32) -> u32 {
    if max_merge_cand <= 1 {
        return 0;
    }
    // First bin is bypass-coded, remaining bins are bypass-coded truncated
    // unary.
    let mut idx = 0u32;
    if cabac.decode_bypass() {
        idx += 1;
        while idx < max_merge_cand - 1 {
            if cabac.decode_bypass() {
                idx += 1;
            } else {
                break;
            }
        }
    }
    idx
}

// ---------------------------------------------------------------------------
// AMVP (ITU-T H.265, 8.5.3.1)
// ---------------------------------------------------------------------------

/// Build the AMVP candidate list (2 candidates).
///
/// Spatial candidates are derived from left (A0/A1) and above (B0/B1/B2)
/// neighbours.  When fewer than 2 are found, zero-MVs fill the rest.
pub fn build_amvp_candidates(
    mv_field: &[HevcMvField],
    pic_width_in_min_pu: usize,
    x: usize,
    y: usize,
    ref_idx: i8,
    list: usize,
) -> [HevcMv; 2] {
    let mut cands = [HevcMv::default(); 2];
    let mut count = 0usize;

    let pu_x = x / MIN_PU_SIZE;
    let pu_y = y / MIN_PU_SIZE;

    let get = |px: usize, py: usize| -> Option<HevcMv> {
        if px < pic_width_in_min_pu {
            let idx = py * pic_width_in_min_pu + px;
            if idx < mv_field.len() {
                let f = &mv_field[idx];
                if f.pred_flag[list] && f.ref_idx[list] == ref_idx {
                    return Some(f.mv[list]);
                }
                // Try the other list with same ref_idx (scaling omitted).
                let other = 1 - list;
                if f.pred_flag[other] && f.ref_idx[other] == ref_idx {
                    return Some(f.mv[other]);
                }
            }
        }
        None
    };

    // Left group: A0 then A1.
    if pu_x > 0 {
        if let Some(mv) = get(pu_x - 1, pu_y) {
            cands[count] = mv;
            count += 1;
        } else if pu_y > 0
            && let Some(mv) = get(pu_x - 1, pu_y - 1)
        {
            cands[count] = mv;
            count += 1;
        }
    }

    // Above group: B0 then B1 then B2.
    if count < 2
        && pu_y > 0
        && let Some(mv) = get(pu_x, pu_y - 1)
        && (count == 0 || cands[0] != mv)
    {
        cands[count] = mv;
        count += 1;
    }
    if count < 2
        && pu_x > 0
        && pu_y > 0
        && let Some(mv) = get(pu_x - 1, pu_y - 1)
        && (count == 0 || cands[0] != mv)
    {
        cands[count] = mv;
        // count += 1; (last candidate, value not read again)
    }

    // Zero-MV fill.
    // (cands already defaults to zero.)

    cands
}

// ---------------------------------------------------------------------------
// MVD parsing (ITU-T H.265, 7.3.8.9)
// ---------------------------------------------------------------------------

/// Parse a motion vector difference from the CABAC bitstream.
///
/// Each component (x, y) is coded as:
///   `abs_mvd_greater0_flag` (bypass) → if set, `abs_mvd_greater1_flag`
///   (bypass) → if set, `abs_mvd_minus2` (Exp-Golomb order-1, bypass)
///   → `mvd_sign_flag` (bypass).
pub fn parse_mvd(cabac: &mut CabacDecoder<'_>) -> HevcMv {
    let abs_x_gt0 = cabac.decode_bypass();
    let abs_y_gt0 = cabac.decode_bypass();

    let abs_x_gt1 = if abs_x_gt0 {
        cabac.decode_bypass()
    } else {
        false
    };
    let abs_y_gt1 = if abs_y_gt0 {
        cabac.decode_bypass()
    } else {
        false
    };

    let mut abs_x: i16 = 0;
    if abs_x_gt0 {
        abs_x = 1;
        if abs_x_gt1 {
            abs_x += 1 + cabac.decode_eg(1) as i16;
        }
    }

    let mut abs_y: i16 = 0;
    if abs_y_gt0 {
        abs_y = 1;
        if abs_y_gt1 {
            abs_y += 1 + cabac.decode_eg(1) as i16;
        }
    }

    let sign_x = if abs_x_gt0 {
        cabac.decode_bypass()
    } else {
        false
    };
    let sign_y = if abs_y_gt0 {
        cabac.decode_bypass()
    } else {
        false
    };

    HevcMv {
        x: if sign_x { -abs_x } else { abs_x },
        y: if sign_y { -abs_y } else { abs_y },
    }
}

// ---------------------------------------------------------------------------
// Inter CU prediction parsing
// ---------------------------------------------------------------------------

/// Parse inter prediction data for one CU from the CABAC bitstream.
///
/// Returns the resulting [`HevcMvField`] describing L0/L1 motion.
///
/// This is a simplified implementation handling merge mode and single-list
/// explicit MV coding for P-slices, plus basic B-slice bi-prediction.
#[allow(clippy::too_many_arguments)]
pub fn parse_inter_prediction(
    state: &mut HevcSliceCabacState<'_>,
    sps: &HevcSps,
    slice_type: HevcSliceType,
    mv_field: &[HevcMvField],
    pic_width_in_min_pu: usize,
    x: usize,
    y: usize,
    cu_size: usize,
) -> HevcMvField {
    // merge_flag — decoded as bypass (simplified; spec uses context 0).
    let merge_flag = state.cabac.decode_bypass();

    if merge_flag {
        // Merge mode: pick candidate.
        let max_merge = 5u32;
        let merge_idx = parse_merge_idx(&mut state.cabac, max_merge);
        let candidates =
            build_merge_candidates(mv_field, pic_width_in_min_pu, x, y, cu_size, cu_size);
        let idx = (merge_idx as usize).min(candidates.len().saturating_sub(1));
        return candidates[idx];
    }

    // Explicit MV coding.
    match slice_type {
        HevcSliceType::P => {
            // ref_idx_l0 — bypass coded unary (simplified).
            let ref_idx_l0 = if sps.num_short_term_ref_pic_sets > 1 {
                let mut idx = 0i8;
                while (idx as u8) < sps.num_short_term_ref_pic_sets.saturating_sub(1) {
                    if state.cabac.decode_bypass() {
                        idx += 1;
                    } else {
                        break;
                    }
                }
                idx
            } else {
                0i8
            };

            let mvd = parse_mvd(&mut state.cabac);

            // AMVP predictor.
            let amvp = build_amvp_candidates(mv_field, pic_width_in_min_pu, x, y, ref_idx_l0, 0);
            let mvp_flag = state.cabac.decode_bypass();
            let predictor = if mvp_flag { amvp[1] } else { amvp[0] };

            HevcMvField {
                mv: [predictor.add(mvd), HevcMv::default()],
                ref_idx: [ref_idx_l0, -1],
                pred_flag: [true, false],
            }
        }
        HevcSliceType::B => {
            // Simplified B-slice: decode both L0 and L1.
            let ref_idx_l0 = 0i8;
            let ref_idx_l1 = 0i8;

            let mvd_l0 = parse_mvd(&mut state.cabac);
            let mvd_l1 = parse_mvd(&mut state.cabac);

            let amvp_l0 = build_amvp_candidates(mv_field, pic_width_in_min_pu, x, y, ref_idx_l0, 0);
            let amvp_l1 = build_amvp_candidates(mv_field, pic_width_in_min_pu, x, y, ref_idx_l1, 1);

            let mvp0_flag = state.cabac.decode_bypass();
            let mvp1_flag = state.cabac.decode_bypass();

            let pred0 = if mvp0_flag { amvp_l0[1] } else { amvp_l0[0] };
            let pred1 = if mvp1_flag { amvp_l1[1] } else { amvp_l1[0] };

            HevcMvField {
                mv: [pred0.add(mvd_l0), pred1.add(mvd_l1)],
                ref_idx: [ref_idx_l0, ref_idx_l1],
                pred_flag: [true, true],
            }
        }
        HevcSliceType::I => {
            // Should never reach inter prediction in an I-slice.
            HevcMvField::unavailable()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DPB tests ----------------------------------------------------------

    #[test]
    fn dpb_new_is_empty() {
        let dpb = HevcDpb::new(5);
        assert!(dpb.is_empty());
        assert_eq!(dpb.len(), 0);
        assert_eq!(dpb.max_size(), 5);
    }

    #[test]
    fn dpb_add_and_get_by_poc() {
        let mut dpb = HevcDpb::new(4);
        dpb.add(HevcReferencePicture {
            poc: 0,
            luma: vec![128; 16],
            width: 4,
            height: 4,
            is_long_term: false,
        });
        dpb.add(HevcReferencePicture {
            poc: 2,
            luma: vec![200; 16],
            width: 4,
            height: 4,
            is_long_term: false,
        });
        assert_eq!(dpb.len(), 2);
        assert!(dpb.get_by_poc(0).is_some());
        assert_eq!(dpb.get_by_poc(0).unwrap().luma[0], 128);
        assert!(dpb.get_by_poc(2).is_some());
        assert!(dpb.get_by_poc(99).is_none());
    }

    #[test]
    fn dpb_bump_removes_oldest() {
        let mut dpb = HevcDpb::new(3);
        for poc in 0..3 {
            dpb.add(HevcReferencePicture {
                poc,
                luma: vec![poc as u8; 16],
                width: 4,
                height: 4,
                is_long_term: false,
            });
        }
        assert_eq!(dpb.len(), 3);
        // Adding a 4th triggers bump of the lowest POC (0).
        dpb.add(HevcReferencePicture {
            poc: 5,
            luma: vec![5; 16],
            width: 4,
            height: 4,
            is_long_term: false,
        });
        assert_eq!(dpb.len(), 3);
        assert!(dpb.get_by_poc(0).is_none());
        assert!(dpb.get_by_poc(1).is_some());
        assert!(dpb.get_by_poc(5).is_some());
    }

    #[test]
    fn dpb_mark_unused() {
        let mut dpb = HevcDpb::new(4);
        dpb.add(HevcReferencePicture {
            poc: 10,
            luma: vec![0; 16],
            width: 4,
            height: 4,
            is_long_term: false,
        });
        dpb.add(HevcReferencePicture {
            poc: 20,
            luma: vec![0; 16],
            width: 4,
            height: 4,
            is_long_term: false,
        });
        dpb.mark_unused(10);
        assert_eq!(dpb.len(), 1);
        assert!(dpb.get_by_poc(10).is_none());
        assert!(dpb.get_by_poc(20).is_some());
    }

    #[test]
    fn dpb_clear() {
        let mut dpb = HevcDpb::new(4);
        for poc in 0..4 {
            dpb.add(HevcReferencePicture {
                poc,
                luma: vec![0; 16],
                width: 4,
                height: 4,
                is_long_term: false,
            });
        }
        assert_eq!(dpb.len(), 4);
        dpb.clear();
        assert!(dpb.is_empty());
        assert_eq!(dpb.len(), 0);
    }

    // -- MV arithmetic tests ------------------------------------------------

    #[test]
    fn mv_from_fullpel() {
        let mv = HevcMv::from_fullpel(3, -5);
        assert_eq!(mv.x, 12);
        assert_eq!(mv.y, -20);
    }

    #[test]
    fn mv_add() {
        let a = HevcMv { x: 10, y: -4 };
        let b = HevcMv { x: -3, y: 7 };
        let c = a.add(b);
        assert_eq!(c.x, 7);
        assert_eq!(c.y, 3);
    }

    #[test]
    fn mv_negate() {
        let mv = HevcMv { x: 5, y: -8 };
        let neg = mv.negate();
        assert_eq!(neg.x, -5);
        assert_eq!(neg.y, 8);
    }

    #[test]
    fn mv_default_is_zero() {
        let mv = HevcMv::default();
        assert_eq!(mv.x, 0);
        assert_eq!(mv.y, 0);
    }

    // -- Luma interpolation tests -------------------------------------------

    fn make_ref_pic(w: usize, h: usize, val: u8) -> HevcReferencePicture {
        HevcReferencePicture {
            poc: 0,
            luma: vec![val; w * h],
            width: w,
            height: h,
            is_long_term: false,
        }
    }

    #[test]
    fn mc_luma_integer_pel() {
        let pic = make_ref_pic(16, 16, 100);
        let mut out = vec![0i16; 4 * 4];
        hevc_mc_luma(&pic, 2, 2, HevcMv { x: 0, y: 0 }, 4, 4, &mut out);
        for v in &out {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn mc_luma_half_pel_uniform() {
        // With a uniform reference, half-pel filtering should yield the same
        // value (filter coefficients sum to 64).
        let pic = make_ref_pic(32, 32, 80);
        let mut out = vec![0i16; 8 * 8];
        // frac_x = 2 (half-pel)
        hevc_mc_luma(&pic, 4, 4, HevcMv { x: 2, y: 0 }, 8, 8, &mut out);
        for v in &out {
            assert_eq!(*v, 80);
        }
    }

    #[test]
    fn mc_luma_quarter_pel_uniform() {
        let pic = make_ref_pic(32, 32, 60);
        let mut out = vec![0i16; 4 * 4];
        hevc_mc_luma(&pic, 4, 4, HevcMv { x: 1, y: 0 }, 4, 4, &mut out);
        for v in &out {
            assert_eq!(*v, 60);
        }
    }

    #[test]
    fn mc_luma_three_quarter_pel_uniform() {
        let pic = make_ref_pic(32, 32, 120);
        let mut out = vec![0i16; 4 * 4];
        hevc_mc_luma(&pic, 4, 4, HevcMv { x: 3, y: 0 }, 4, 4, &mut out);
        for v in &out {
            assert_eq!(*v, 120);
        }
    }

    #[test]
    fn mc_luma_vertical_half_pel_uniform() {
        let pic = make_ref_pic(32, 32, 90);
        let mut out = vec![0i16; 4 * 4];
        hevc_mc_luma(&pic, 4, 4, HevcMv { x: 0, y: 2 }, 4, 4, &mut out);
        for v in &out {
            assert_eq!(*v, 90);
        }
    }

    #[test]
    fn mc_luma_diagonal_half_pel_uniform() {
        let pic = make_ref_pic(32, 32, 70);
        let mut out = vec![0i16; 4 * 4];
        hevc_mc_luma(&pic, 4, 4, HevcMv { x: 2, y: 2 }, 4, 4, &mut out);
        for v in &out {
            assert_eq!(*v, 70);
        }
    }

    #[test]
    fn mc_luma_gradient_horizontal() {
        // Horizontal gradient: column c has value 10*c.
        let w = 32usize;
        let h = 16usize;
        let mut luma = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                luma[row * w + col] = (col * 8).min(255) as u8;
            }
        }
        let pic = HevcReferencePicture {
            poc: 0,
            luma,
            width: w,
            height: h,
            is_long_term: false,
        };
        let mut out = vec![0i16; 4 * 4];
        // Integer-pel fetch.
        hevc_mc_luma(&pic, 4, 2, HevcMv { x: 0, y: 0 }, 4, 4, &mut out);
        // First column should be col=4 => 32
        assert_eq!(out[0], 32);
        assert_eq!(out[1], 40);
    }

    // -- Bi-prediction averaging tests --------------------------------------

    #[test]
    fn bipred_average_uniform() {
        let l0 = vec![100i16; 16];
        let l1 = vec![200i16; 16];
        let mut out = vec![0u8; 16];
        hevc_bipred_average(&l0, &l1, &mut out, 16);
        // (100 + 200 + 1) >> 1 = 150
        for v in &out {
            assert_eq!(*v, 150);
        }
    }

    #[test]
    fn bipred_average_clamping() {
        let l0 = vec![255i16; 4];
        let l1 = vec![255i16; 4];
        let mut out = vec![0u8; 4];
        hevc_bipred_average(&l0, &l1, &mut out, 4);
        for v in &out {
            assert_eq!(*v, 255);
        }

        let l0n = vec![-10i16; 4];
        let l1n = vec![-20i16; 4];
        let mut outn = vec![255u8; 4];
        hevc_bipred_average(&l0n, &l1n, &mut outn, 4);
        for v in &outn {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn unipred_clip_basic() {
        let pred = vec![128i16, -5, 300, 0];
        let mut out = vec![0u8; 4];
        hevc_unipred_clip(&pred, &mut out, 4);
        assert_eq!(out[0], 128);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 255);
        assert_eq!(out[3], 0);
    }

    // -- Merge candidate list tests -----------------------------------------

    #[test]
    fn merge_candidates_no_neighbours() {
        // Empty motion field — all candidates should be zero-MV fill.
        let field = vec![HevcMvField::unavailable(); 16 * 16];
        let cands = build_merge_candidates(&field, 16, 0, 0, 8, 8);
        assert_eq!(cands.len(), 5);
        // All should have pred_flag[0] true (zero-MV default fill).
        for c in &cands {
            assert!(c.pred_flag[0]);
        }
    }

    #[test]
    fn merge_candidates_with_left_neighbour() {
        let pw = 16usize; // pic_width_in_min_pu
        let mut field = vec![HevcMvField::unavailable(); pw * pw];
        // Set left neighbour at min-PU (0, 1) — that's A1 for a block at (4, 0).
        let left = HevcMvField {
            mv: [HevcMv { x: 8, y: 4 }, HevcMv::default()],
            ref_idx: [0, -1],
            pred_flag: [true, false],
        };
        // Block at pixel (4, 0), size 4x4 -> pu_x=1, pu_y=0, pu_w=1, pu_h=1.
        // A1 is at (pu_x-1, pu_y+pu_h-1) = (0, 0).
        field[0] = left;
        let cands = build_merge_candidates(&field, pw, 4, 0, 4, 4);
        assert_eq!(cands.len(), 5);
        assert_eq!(cands[0].mv[0].x, 8);
        assert_eq!(cands[0].mv[0].y, 4);
    }

    // -- AMVP candidate tests -----------------------------------------------

    #[test]
    fn amvp_no_neighbours() {
        let field = vec![HevcMvField::unavailable(); 16 * 16];
        let cands = build_amvp_candidates(&field, 16, 4, 4, 0, 0);
        // No neighbours -> zero MV.
        assert_eq!(cands[0], HevcMv::default());
        assert_eq!(cands[1], HevcMv::default());
    }

    #[test]
    fn amvp_with_left_neighbour() {
        let pw = 16usize;
        let mut field = vec![HevcMvField::unavailable(); pw * pw];
        let left = HevcMvField {
            mv: [HevcMv { x: 12, y: -8 }, HevcMv::default()],
            ref_idx: [0, -1],
            pred_flag: [true, false],
        };
        // Block at pixel (4, 4) -> pu (1, 1). Left A0 at (0, 1).
        field[pw] = left;
        let cands = build_amvp_candidates(&field, pw, 4, 4, 0, 0);
        assert_eq!(cands[0].x, 12);
        assert_eq!(cands[0].y, -8);
    }

    // -- MVD parsing tests --------------------------------------------------

    #[test]
    fn parse_mvd_zero() {
        // All-zero stream: abs_x_gt0 = false, abs_y_gt0 = false -> (0, 0).
        let data = [0x00u8; 16];
        let mut cabac = CabacDecoder::new(&data);
        let mv = parse_mvd(&mut cabac);
        assert_eq!(mv.x, 0);
        assert_eq!(mv.y, 0);
    }

    #[test]
    fn parse_mvd_deterministic() {
        // Non-trivial stream should produce deterministic MVD.
        let data = [0xFFu8; 32];
        let mut cabac = CabacDecoder::new(&data);
        let mv1 = parse_mvd(&mut cabac);

        let mut cabac2 = CabacDecoder::new(&data);
        let mv2 = parse_mvd(&mut cabac2);
        assert_eq!(mv1, mv2);
    }

    // -- parse_merge_idx tests ----------------------------------------------

    #[test]
    fn parse_merge_idx_single_candidate() {
        let data = [0xFFu8; 8];
        let mut cabac = CabacDecoder::new(&data);
        let idx = parse_merge_idx(&mut cabac, 1);
        assert_eq!(idx, 0);
    }

    #[test]
    fn parse_merge_idx_zero_stream() {
        let data = [0x00u8; 16];
        let mut cabac = CabacDecoder::new(&data);
        let idx = parse_merge_idx(&mut cabac, 5);
        assert_eq!(idx, 0);
    }

    // -- HevcMvField tests --------------------------------------------------

    #[test]
    fn mvfield_unavailable() {
        let f = HevcMvField::unavailable();
        assert!(!f.is_available());
        assert_eq!(f.ref_idx[0], -1);
        assert_eq!(f.ref_idx[1], -1);
    }

    #[test]
    fn mvfield_available() {
        let f = HevcMvField {
            mv: [HevcMv { x: 1, y: 2 }, HevcMv::default()],
            ref_idx: [0, -1],
            pred_flag: [true, false],
        };
        assert!(f.is_available());
    }

    // -- Full inter CU decode on synthetic data -----------------------------

    #[test]
    fn parse_inter_prediction_p_slice_synthetic() {
        let data = [0x55u8; 128];
        let mut state = HevcSliceCabacState::new(&data, 26);
        let sps = test_sps();
        let field = vec![HevcMvField::unavailable(); 16 * 16];
        let mvf = parse_inter_prediction(&mut state, &sps, HevcSliceType::P, &field, 16, 0, 0, 8);
        // Should produce some prediction field.
        assert!(mvf.pred_flag[0] || mvf.pred_flag[1]);
    }

    #[test]
    fn parse_inter_prediction_b_slice_synthetic() {
        let data = [0xAAu8; 128];
        let mut state = HevcSliceCabacState::new(&data, 26);
        let sps = test_sps();
        let field = vec![HevcMvField::unavailable(); 16 * 16];
        let mvf = parse_inter_prediction(&mut state, &sps, HevcSliceType::B, &field, 16, 0, 0, 8);
        assert!(mvf.pred_flag[0] || mvf.pred_flag[1]);
    }

    #[test]
    fn parse_inter_prediction_merge_mode() {
        // Build a stream where merge_flag = true (first bypass bin = 1 in
        // the CABAC stream). We use 0xFF which makes bypass bins decode as 1.
        let data = [0xFFu8; 128];
        let mut state = HevcSliceCabacState::new(&data, 26);
        let sps = test_sps();
        let field = vec![HevcMvField::unavailable(); 16 * 16];
        let mvf = parse_inter_prediction(&mut state, &sps, HevcSliceType::P, &field, 16, 4, 4, 8);
        // Merge mode with no real neighbours falls back to zero-MV fill.
        assert!(mvf.pred_flag[0]);
    }

    // -- Luma filter coefficient sanity -------------------------------------

    #[test]
    fn luma_filter_coefficients_sum() {
        // Each set of filter coefficients should sum to 64.
        for (i, row) in HEVC_LUMA_FILTER.iter().enumerate() {
            let sum: i16 = row.iter().sum();
            assert_eq!(sum, 64, "filter row {i} sums to {sum}, expected 64");
        }
    }

    // -- Helper for tests ---------------------------------------------------

    fn test_sps() -> HevcSps {
        HevcSps {
            sps_id: 0,
            vps_id: 0,
            max_sub_layers: 1,
            chroma_format_idc: 1,
            pic_width: 64,
            pic_height: 64,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_pic_order_cnt: 4,
            log2_min_cb_size: 3,
            log2_diff_max_min_cb_size: 3,
            log2_min_transform_size: 2,
            log2_diff_max_min_transform_size: 3,
            max_transform_hierarchy_depth_inter: 1,
            max_transform_hierarchy_depth_intra: 1,
            sample_adaptive_offset_enabled: false,
            pcm_enabled: false,
            num_short_term_ref_pic_sets: 0,
            long_term_ref_pics_present: false,
            sps_temporal_mvp_enabled: false,
            strong_intra_smoothing_enabled: false,
        }
    }
}
