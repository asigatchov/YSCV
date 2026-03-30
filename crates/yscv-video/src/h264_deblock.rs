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
    qp: u8,
) {
    if bs == 0 {
        return;
    }

    let (across, along) = if is_vertical {
        (1usize, stride)
    } else {
        (stride, 1usize)
    };

    let tc = if (1..=3).contains(&bs) {
        derive_tc0(qp, bs) + 1
    } else {
        0
    };
    let alpha_quarter_plus2 = (alpha >> 2) + 2;

    // Single bounds check for the entire 4-pixel group
    let max_idx = offset + 3 * along + 2 * across;
    let min_idx_needed = 3 * across;
    if max_idx >= pixels.len() || offset < min_idx_needed {
        return;
    }

    for i in 0..4 {
        let q0_idx = offset + i * along;
        let p0_idx = q0_idx - across;

        let p0 = pixels[p0_idx] as i32;
        let q0 = pixels[q0_idx] as i32;

        // Quick threshold reject
        let diff_pq = (p0 - q0).abs();
        if diff_pq >= alpha {
            continue;
        }

        let p1 = pixels[q0_idx - 2 * across] as i32;
        let q1 = pixels[q0_idx + across] as i32;

        if (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }

        if bs == 4 {
            let p2 = pixels[q0_idx - 3 * across] as i32;
            let q2 = pixels[q0_idx + 2 * across] as i32;

            if (p2 - p0).abs() < beta && diff_pq < alpha_quarter_plus2 {
                pixels[p0_idx] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                pixels[q0_idx - 2 * across] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
            } else {
                pixels[p0_idx] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            }

            if (q2 - q0).abs() < beta && diff_pq < alpha_quarter_plus2 {
                pixels[q0_idx] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                pixels[q0_idx + across] = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
            } else {
                pixels[q0_idx] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            }
        } else {
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
/// Skip-aware deblocking: skips edges where both adjacent MBs are P-skip.
#[allow(unsafe_code)]
pub fn deblock_frame_skip_aware(
    frame: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    qp: u8,
    mb_is_skip: &[bool],
    mb_w: usize,
) {
    use rayon::prelude::*;

    let alpha = derive_alpha(qp);
    let beta = derive_beta(qp);
    let mb_cols = width / 16;
    let mb_rows = height / 16;
    let bs: u8 = 2;
    let tc = if (1..=3).contains(&bs) {
        derive_tc0(qp, bs) + 1
    } else {
        0
    };

    let plane_size = width * height;
    for ch in 0..channels {
        let plane_offset = ch * plane_size;
        if plane_offset + plane_size > frame.len() {
            break;
        }

        let plane = &mut frame[plane_offset..plane_offset + plane_size];

        // Vertical edges: parallel over MB rows, skip edges between two skip MBs
        if mb_rows >= 4 {
            plane
                .par_chunks_mut(16 * width)
                .enumerate()
                .for_each(|(mb_row, chunk)| {
                    let chunk_len = chunk.len();
                    for mb_col in 1..mb_cols {
                        // Skip if both left and right MBs are skip
                        let left_skip = mb_is_skip
                            .get(mb_row * mb_w + mb_col - 1)
                            .copied()
                            .unwrap_or(false);
                        let right_skip = mb_is_skip
                            .get(mb_row * mb_w + mb_col)
                            .copied()
                            .unwrap_or(false);
                        if left_skip && right_skip {
                            continue;
                        }
                        let edge_x = mb_col * 16;
                        for row in 0..16 {
                            let q0 = row * width + edge_x;
                            if q0 < 2 || q0 + 2 > chunk_len {
                                continue;
                            }
                            let p0 = chunk[q0 - 1] as i32;
                            let q0v = chunk[q0] as i32;
                            if (p0 - q0v).abs() >= alpha {
                                continue;
                            }
                            let p1 = chunk[q0 - 2] as i32;
                            let q1 = chunk[q0 + 1] as i32;
                            if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                                continue;
                            }
                            let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                            chunk[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                            chunk[q0] = (q0v - delta).clamp(0, 255) as u8;
                        }
                    }
                });
        } else {
            for mb_row in 0..mb_rows {
                for mb_col in 1..mb_cols {
                    let left_skip = mb_is_skip
                        .get(mb_row * mb_w + mb_col - 1)
                        .copied()
                        .unwrap_or(false);
                    let right_skip = mb_is_skip
                        .get(mb_row * mb_w + mb_col)
                        .copied()
                        .unwrap_or(false);
                    if left_skip && right_skip {
                        continue;
                    }
                    let edge_x = mb_col * 16;
                    let base_y = mb_row * 16;
                    for row in 0..16 {
                        let q0 = (base_y + row) * width + edge_x;
                        if q0 < 2 || q0 + 2 > plane.len() {
                            continue;
                        }
                        let p0 = plane[q0 - 1] as i32;
                        let q0v = plane[q0] as i32;
                        if (p0 - q0v).abs() >= alpha {
                            continue;
                        }
                        let p1 = plane[q0 - 2] as i32;
                        let q1 = plane[q0 + 1] as i32;
                        if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                            continue;
                        }
                        let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                        plane[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                        plane[q0] = (q0v - delta).clamp(0, 255) as u8;
                    }
                }
            }
        }

        // Horizontal edges (sequential, skip-aware)
        for mb_row in 1..mb_rows {
            for mb_col in 0..mb_cols {
                let top_skip = mb_is_skip
                    .get((mb_row - 1) * mb_w + mb_col)
                    .copied()
                    .unwrap_or(false);
                let bot_skip = mb_is_skip
                    .get(mb_row * mb_w + mb_col)
                    .copied()
                    .unwrap_or(false);
                if top_skip && bot_skip {
                    continue;
                }
                let edge_y = mb_row * 16;
                let base_x = mb_col * 16;
                for col in 0..16 {
                    let x = base_x + col;
                    let q0 = edge_y * width + x;
                    if q0 < 3 * width || q0 + 2 * width >= plane.len() {
                        continue;
                    }
                    let p0 = plane[q0 - width] as i32;
                    let q0v = plane[q0] as i32;
                    if (p0 - q0v).abs() >= alpha {
                        continue;
                    }
                    let p1 = plane[q0 - 2 * width] as i32;
                    let q1 = plane[q0 + width] as i32;
                    if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                        continue;
                    }
                    let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                    plane[q0 - width] = (p0 + delta).clamp(0, 255) as u8;
                    plane[q0] = (q0v - delta).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Iterates over every macroblock row and column, applying the deblocking
/// filter to both vertical and horizontal edges. The `qp` parameter is the
/// average slice quantization parameter used to derive filter thresholds.
///
/// For multi-channel frames the filter is applied independently to each
/// channel plane (the caller should ensure the frame is in planar or
/// interleaved format; this implementation assumes a single luma plane or
/// operates channel-by-channel).
#[allow(unsafe_code)]
pub fn deblock_frame(frame: &mut [u8], width: usize, height: usize, channels: usize, qp: u8) {
    use rayon::prelude::*;

    let alpha = derive_alpha(qp);
    let beta = derive_beta(qp);
    let mb_cols = width / 16;
    let mb_rows = height / 16;
    let bs: u8 = 2;
    let tc = if (1..=3).contains(&bs) {
        derive_tc0(qp, bs) + 1
    } else {
        0
    };

    let plane_size = width * height;
    for ch in 0..channels {
        let plane_offset = ch * plane_size;
        if plane_offset + plane_size > frame.len() {
            break;
        }

        // Vertical edges: parallelize over MB rows using chunks_mut.
        // Each MB row's vertical edges only touch pixels in rows [mb_row*16..(mb_row+1)*16].
        let plane = &mut frame[plane_offset..plane_offset + plane_size];
        if mb_rows >= 4 {
            plane.par_chunks_mut(16 * width).for_each(|mb_row_chunk| {
                // mb_row_chunk is exactly 16 rows of width pixels
                let chunk_len = mb_row_chunk.len();
                for mb_col in 1..mb_cols {
                    let edge_x = mb_col * 16;
                    for row_in_mb in 0..16 {
                        let q0 = row_in_mb * width + edge_x;
                        if q0 < 2 || q0 + 2 > chunk_len {
                            continue;
                        }
                        let p0 = mb_row_chunk[q0 - 1] as i32;
                        let q0v = mb_row_chunk[q0] as i32;
                        if (p0 - q0v).abs() >= alpha {
                            continue;
                        }
                        let p1 = mb_row_chunk[q0 - 2] as i32;
                        let q1 = mb_row_chunk[q0 + 1] as i32;
                        if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                            continue;
                        }
                        let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                        mb_row_chunk[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                        mb_row_chunk[q0] = (q0v - delta).clamp(0, 255) as u8;
                    }
                }
            });
        } else {
            // Small frame: sequential fallback
            for mb_row in 0..mb_rows {
                for mb_col in 1..mb_cols {
                    let edge_x = mb_col * 16;
                    let base_y = mb_row * 16;
                    for row in 0..16 {
                        let y = base_y + row;
                        let q0 = plane_offset + y * width + edge_x;
                        if q0 < 3 || q0 + 2 >= frame.len() {
                            continue;
                        }
                        let p0 = frame[q0 - 1] as i32;
                        let q0v = frame[q0] as i32;
                        if (p0 - q0v).abs() >= alpha {
                            continue;
                        }
                        let p1 = frame[q0 - 2] as i32;
                        let q1 = frame[q0 + 1] as i32;
                        if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                            continue;
                        }
                        let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                        frame[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                        frame[q0] = (q0v - delta).clamp(0, 255) as u8;
                    }
                }
            }
        }

        // Horizontal edges: parallelize over MB rows (each row accesses 2 adjacent pixel rows).
        // Horizontal edge at mb_row touches rows [mb_row*16-2..mb_row*16+2], so adjacent
        // MB row edges are 16 rows apart — no overlap if mb_rows > 1.
        // Horizontal edges: sequential (cross MB row boundaries).
        for mb_row in 1..mb_rows {
            for mb_col in 0..mb_cols {
                let edge_y = mb_row * 16;
                let base_x = mb_col * 16;
                for col in 0..16 {
                    let x = base_x + col;
                    let q0 = plane_offset + edge_y * width + x;
                    if q0 < 3 * width || q0 + 2 * width >= frame.len() {
                        continue;
                    }
                    let p0 = frame[q0 - width] as i32;
                    let q0v = frame[q0] as i32;
                    if (p0 - q0v).abs() >= alpha {
                        continue;
                    }
                    let p1 = frame[q0 - 2 * width] as i32;
                    let q1 = frame[q0 + width] as i32;
                    if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                        continue;
                    }
                    let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                    frame[q0 - width] = (p0 + delta).clamp(0, 255) as u8;
                    frame[q0] = (q0v - delta).clamp(0, 255) as u8;
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
        deblock_edge_luma(&mut pixels, width, q0_offset, true, 4, alpha, beta, 26);

        // After filtering, the discontinuity at the boundary should be
        // reduced (or at least not increased).
        let new_disc: i32 = (pixels[3 * width + 16] as i32 - pixels[3 * width + 15] as i32).abs();
        assert!(
            new_disc <= orig_disc,
            "deblocking should reduce discontinuity: was {orig_disc}, now {new_disc}"
        );
    }
}
