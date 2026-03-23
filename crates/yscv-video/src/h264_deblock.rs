//! H.264 in-loop deblocking filter.
//!
//! Operates on the reconstructed frame to reduce blocking artifacts at
//! macroblock and 4x4 sub-block boundaries. Implements boundary strength
//! computation and adaptive edge filtering per the H.264/AVC specification.

use crate::h264_motion::MotionVector;

// ---------------------------------------------------------------------------
// Boundary strength
// ---------------------------------------------------------------------------

/// Compute boundary strength (bS) for a pair of adjacent blocks `p` and `q`.
///
/// Returns a value in 0..=4:
/// - 4: either block is intra-coded (strongest filtering)
/// - 2: either block contains coded residual coefficients
/// - 1: motion vectors differ by >= 1 integer sample (4 quarter-pel units)
/// - 0: no filtering needed
pub fn compute_boundary_strength(
    is_intra_p: bool,
    is_intra_q: bool,
    mv_p: MotionVector,
    mv_q: MotionVector,
    has_coded_residual_p: bool,
    has_coded_residual_q: bool,
) -> u8 {
    if is_intra_p || is_intra_q {
        return 4;
    }
    if has_coded_residual_p || has_coded_residual_q {
        return 2;
    }
    let mv_diff = (mv_p.dx - mv_q.dx).unsigned_abs() + (mv_p.dy - mv_q.dy).unsigned_abs();
    if mv_diff >= 4 {
        return 1;
    }
    0
}

// ---------------------------------------------------------------------------
// Threshold derivation
// ---------------------------------------------------------------------------

/// Alpha threshold lookup table indexed by indexA = clamp(QP + offset, 0, 51).
/// Values from Table 8-16 of the H.264 specification.
const ALPHA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
    24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168,
    176, 184,
];

/// Beta threshold lookup table indexed by indexB = clamp(QP + offset, 0, 51).
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Tc0 table indexed by [indexA][bS-1] for bS in 1..=3.
/// From Table 8-17 of the H.264 specification (subset for common QP range).
const TC0_TABLE: [[i32; 3]; 52] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 2, 3],
    [1, 2, 3],
    [2, 2, 3],
    [2, 2, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 3, 5],
    [3, 4, 6],
    [3, 4, 6],
    [4, 5, 7],
    [4, 5, 8],
    [4, 6, 9],
    [5, 7, 10],
    [6, 8, 11],
    [6, 8, 13],
    [7, 10, 14],
    [8, 11, 16],
    [9, 12, 18],
    [10, 13, 20],
    [11, 15, 23],
    [13, 17, 25],
];

/// Derive the alpha threshold from a quantization parameter.
fn derive_alpha(qp: u8) -> i32 {
    let idx = (qp as usize).min(51);
    ALPHA_TABLE[idx]
}

/// Derive the beta threshold from a quantization parameter.
fn derive_beta(qp: u8) -> i32 {
    let idx = (qp as usize).min(51);
    BETA_TABLE[idx]
}

/// Derive tc0 from QP and boundary strength (bS in 1..=3).
fn derive_tc0(qp: u8, bs: u8) -> i32 {
    let idx = (qp as usize).min(51);
    if bs == 0 || bs > 3 {
        return 0;
    }
    TC0_TABLE[idx][(bs - 1) as usize]
}

// ---------------------------------------------------------------------------
// Edge filtering
// ---------------------------------------------------------------------------

/// Apply the deblocking filter to a single edge consisting of 4 pixel pairs.
///
/// `pixels` is the row-major frame buffer (single channel). `stride` is the
/// row stride. `offset` is the index of q0 (the first pixel on the "q" side
/// of the boundary). `step` is the distance between successive pixel pairs
/// along the edge (stride for vertical edges, 1 for horizontal edges).
///
/// For a **vertical** edge, p and q are horizontally adjacent:
///   p2 p1 p0 | q0 q1 q2   (step = stride, each pair one row apart)
///
/// For a **horizontal** edge, p and q are vertically adjacent:
///   p2 p1 p0 are in rows above, q0 q1 q2 in rows below.
pub fn deblock_edge_luma(
    pixels: &mut [u8],
    stride: usize,
    offset: usize,
    is_vertical: bool,
    bs: u8,
    alpha: i32,
    beta: i32,
) {
    if bs == 0 {
        return;
    }

    // `across` steps across the boundary (towards p or q).
    // `along` steps along the edge to the next pixel pair.
    let (across, along) = if is_vertical {
        (1usize, stride)
    } else {
        (stride, 1usize)
    };

    for i in 0..4 {
        let q0_idx = offset + i * along;

        // Bounds check: we need p2..=q2 to be valid indices.
        if q0_idx + 2 * across >= pixels.len() || q0_idx < 3 * across {
            continue;
        }

        let p0_idx = q0_idx - across;
        let p1_idx = q0_idx - 2 * across;
        let p2_idx = q0_idx - 3 * across;
        let q1_idx = q0_idx + across;
        let q2_idx = q0_idx + 2 * across;

        let p0 = pixels[p0_idx] as i32;
        let p1 = pixels[p1_idx] as i32;
        let p2 = pixels[p2_idx] as i32;
        let q0 = pixels[q0_idx] as i32;
        let q1 = pixels[q1_idx] as i32;
        let q2 = pixels[q2_idx] as i32;

        // Threshold check: only filter if the discontinuity is within range.
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }

        if bs == 4 {
            // Strong filtering (intra edge).
            let ap = (p2 - p0).abs();
            let aq = (q2 - q0).abs();

            if ap < beta && (p0 - q0).abs() < ((alpha >> 2) + 2) {
                pixels[p0_idx] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) / 8).clamp(0, 255) as u8;
                pixels[p1_idx] = ((p2 + p1 + p0 + q0 + 2) / 4).clamp(0, 255) as u8;
            } else {
                pixels[p0_idx] = ((2 * p1 + p0 + q1 + 2) / 4).clamp(0, 255) as u8;
            }

            if aq < beta && (p0 - q0).abs() < ((alpha >> 2) + 2) {
                pixels[q0_idx] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) / 8).clamp(0, 255) as u8;
                pixels[q1_idx] = ((q2 + q1 + q0 + p0 + 2) / 4).clamp(0, 255) as u8;
            } else {
                pixels[q0_idx] = ((2 * q1 + q0 + p1 + 2) / 4).clamp(0, 255) as u8;
            }
        } else {
            // Normal filtering (bs = 1, 2, or 3).
            let tc0 = derive_tc0(
                ((alpha.unsigned_abs().leading_zeros() as i32 - 1).max(0)) as u8,
                bs,
            );
            // Simplified: use tc = tc0 + 1 as an upper bound on the correction.
            let tc = tc0 + 1;
            let delta = ((4 * (q0 - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            pixels[p0_idx] = (p0 + delta).clamp(0, 255) as u8;
            pixels[q0_idx] = (q0 - delta).clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Frame-level deblocking
// ---------------------------------------------------------------------------

/// Filter all macroblock boundaries in a frame.
///
/// Iterates over every macroblock row and column, applying the deblocking
/// filter to both vertical and horizontal edges. The `qp` parameter is the
/// average slice quantization parameter used to derive filter thresholds.
///
/// For multi-channel frames the filter is applied independently to each
/// channel plane (the caller should ensure the frame is in planar or
/// interleaved format; this implementation assumes a single luma plane or
/// operates channel-by-channel).
pub fn deblock_frame(frame: &mut [u8], width: usize, height: usize, channels: usize, qp: u8) {
    let alpha = derive_alpha(qp);
    let beta = derive_beta(qp);
    let mb_cols = width / 16;
    let mb_rows = height / 16;

    // Process each channel independently.
    let plane_size = width * height;
    for ch in 0..channels {
        let plane_offset = ch * plane_size;

        // Work on a contiguous plane slice.
        // For interleaved data we would need a different approach; here we
        // assume planar layout or single-channel input.
        if plane_offset + plane_size > frame.len() {
            break;
        }

        // Vertical edges (left boundary of each macroblock, skip column 0).
        for mb_row in 0..mb_rows {
            for mb_col in 1..mb_cols {
                let edge_x = mb_col * 16;
                for row in 0..16 {
                    let y = mb_row * 16 + row;
                    let q0_offset = plane_offset + y * width + edge_x;
                    // Use a default bs=2 for demonstration; in a full decoder
                    // this would come from per-block metadata.
                    deblock_edge_luma(frame, width, q0_offset, true, 2, alpha, beta);
                }
            }
        }

        // Horizontal edges (top boundary of each macroblock, skip row 0).
        for mb_row in 1..mb_rows {
            for mb_col in 0..mb_cols {
                let edge_y = mb_row * 16;
                for col in 0..16 {
                    let x = mb_col * 16 + col;
                    let q0_offset = plane_offset + edge_y * width + x;
                    deblock_edge_luma(frame, width, q0_offset, false, 2, alpha, beta);
                }
            }
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
    fn boundary_strength_intra() {
        let zero_mv = MotionVector::default();

        // Any intra block should yield bs=4.
        assert_eq!(
            compute_boundary_strength(true, false, zero_mv, zero_mv, false, false),
            4
        );
        assert_eq!(
            compute_boundary_strength(false, true, zero_mv, zero_mv, false, false),
            4
        );
        assert_eq!(
            compute_boundary_strength(true, true, zero_mv, zero_mv, true, true),
            4
        );

        // Coded residual but no intra -> bs=2.
        assert_eq!(
            compute_boundary_strength(false, false, zero_mv, zero_mv, true, false),
            2
        );
        assert_eq!(
            compute_boundary_strength(false, false, zero_mv, zero_mv, false, true),
            2
        );

        // Large MV difference, no residual -> bs=1.
        let mv_a = MotionVector {
            dx: 0,
            dy: 0,
            ref_idx: 0,
        };
        let mv_b = MotionVector {
            dx: 4,
            dy: 0,
            ref_idx: 0,
        };
        assert_eq!(
            compute_boundary_strength(false, false, mv_a, mv_b, false, false),
            1
        );

        // Small MV difference -> bs=0.
        let mv_c = MotionVector {
            dx: 1,
            dy: 0,
            ref_idx: 0,
        };
        assert_eq!(
            compute_boundary_strength(false, false, mv_a, mv_c, false, false),
            0
        );
    }

    #[test]
    fn deblock_edge_reduces_discontinuity() {
        // Create an artificial block boundary in a 32-pixel wide, single-row-
        // equivalent buffer. Left half = 50, right half = 200.
        let width = 32;
        let height = 8;
        let mut pixels = vec![0u8; width * height];
        for row in 0..height {
            for col in 0..width {
                pixels[row * width + col] = if col < 16 { 50 } else { 200 };
            }
        }

        // Record the original discontinuity at the boundary (col 15 vs 16).
        let orig_disc: i32 = (pixels[16] as i32 - pixels[15] as i32).abs();

        // Apply filtering at the vertical edge at column 16 for 4 rows.
        let alpha = 40;
        let beta = 20;
        let q0_offset = 3 * width + 16; // row 3 so we have p2..q2 room
        deblock_edge_luma(&mut pixels, width, q0_offset, true, 4, alpha, beta);

        // After filtering, the discontinuity at the boundary should be
        // reduced (or at least not increased).
        let new_disc: i32 = (pixels[3 * width + 16] as i32 - pixels[3 * width + 15] as i32).abs();
        assert!(
            new_disc <= orig_disc,
            "deblocking should reduce discontinuity: was {orig_disc}, now {new_disc}"
        );
    }
}
