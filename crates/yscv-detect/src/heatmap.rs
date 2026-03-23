use yscv_tensor::Tensor;

use crate::nms::validate_nms_args;
use crate::{BoundingBox, CLASS_ID_PERSON, DetectError, Detection, non_max_suppression};

/// Reusable scratch storage for connected-component heatmap detection.
///
/// This allows callers with stable heatmap dimensions (for example camera loops)
/// to avoid reallocating traversal buffers on each detection call.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct HeatmapDetectScratch {
    active: Vec<bool>,
    visited: Vec<bool>,
    stack: Vec<usize>,
    detections: Vec<Detection>,
}

/// Connected-component detector over heatmaps `[H, W, 1]`.
pub fn detect_from_heatmap(
    heatmap: &Tensor,
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>, DetectError> {
    let mut scratch = HeatmapDetectScratch::default();
    detect_from_heatmap_with_scratch(
        heatmap,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch,
    )
}

/// Connected-component detector over heatmaps `[H, W, 1]` with reusable scratch storage.
pub fn detect_from_heatmap_with_scratch(
    heatmap: &Tensor,
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    scratch: &mut HeatmapDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    let (h, w, c) = map_shape(heatmap)?;
    if c != 1 {
        return Err(DetectError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    detect_from_heatmap_data_with_scratch(
        (h, w),
        heatmap.data(),
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        scratch,
    )
}

pub(crate) fn detect_from_heatmap_data_with_scratch(
    shape: (usize, usize),
    data: &[f32],
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    scratch: &mut HeatmapDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    let (h, w) = shape;
    if !score_threshold.is_finite() || !(0.0..=1.0).contains(&score_threshold) {
        return Err(DetectError::InvalidThreshold {
            threshold: score_threshold,
        });
    }
    if min_area == 0 {
        return Err(DetectError::InvalidMinArea { min_area });
    }
    validate_nms_args(iou_threshold, max_detections)?;
    let pixel_count = h.saturating_mul(w);
    debug_assert_eq!(data.len(), pixel_count);

    if scratch.active.len() != pixel_count {
        scratch.active.resize(pixel_count, false);
    }
    if scratch.visited.len() != pixel_count {
        scratch.visited.resize(pixel_count, false);
    }

    for ((active, visited), value) in scratch
        .active
        .iter_mut()
        .zip(scratch.visited.iter_mut())
        .zip(data.iter().copied())
    {
        *active = is_active_score(value, score_threshold);
        *visited = false;
    }

    scratch.stack.clear();
    scratch.detections.clear();
    for start in 0..pixel_count {
        if scratch.visited[start] || !scratch.active[start] {
            continue;
        }

        scratch.visited[start] = true;
        scratch.stack.clear();
        scratch.stack.push(start);

        let start_y = start / w;
        let start_x = start - start_y * w;
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;
        let mut area = 0usize;
        let mut score_sum = 0.0f32;
        let mut score_max = 0.0f32;

        while let Some(current) = scratch.stack.pop() {
            let cy = current / w;
            let cx = current - cy * w;
            let current_score = data[current];

            area += 1;
            score_sum += current_score;
            score_max = score_max.max(current_score);
            min_x = min_x.min(cx);
            max_x = max_x.max(cx);
            min_y = min_y.min(cy);
            max_y = max_y.max(cy);

            if cx > 0 {
                visit_neighbor(
                    current - 1,
                    &scratch.active,
                    &mut scratch.visited,
                    &mut scratch.stack,
                );
            }
            if cx + 1 < w {
                visit_neighbor(
                    current + 1,
                    &scratch.active,
                    &mut scratch.visited,
                    &mut scratch.stack,
                );
            }
            if cy > 0 {
                visit_neighbor(
                    current - w,
                    &scratch.active,
                    &mut scratch.visited,
                    &mut scratch.stack,
                );
            }
            if cy + 1 < h {
                visit_neighbor(
                    current + w,
                    &scratch.active,
                    &mut scratch.visited,
                    &mut scratch.stack,
                );
            }
        }

        if area >= min_area {
            let avg_score = score_sum / area as f32;
            scratch.detections.push(Detection {
                bbox: BoundingBox {
                    x1: min_x as f32,
                    y1: min_y as f32,
                    x2: (max_x + 1) as f32,
                    y2: (max_y + 1) as f32,
                },
                score: (avg_score + score_max) * 0.5,
                class_id: CLASS_ID_PERSON,
            });
        }
    }

    Ok(non_max_suppression(
        &scratch.detections,
        iou_threshold,
        max_detections,
    ))
}

pub(crate) fn map_shape(input: &Tensor) -> Result<(usize, usize, usize), DetectError> {
    if input.rank() != 3 {
        return Err(DetectError::InvalidMapShape {
            expected_rank: 3,
            got: input.shape().to_vec(),
        });
    }
    Ok((input.shape()[0], input.shape()[1], input.shape()[2]))
}

fn is_active_score(value: f32, threshold: f32) -> bool {
    value.is_finite() && value >= threshold
}

fn visit_neighbor(index: usize, active: &[bool], visited: &mut [bool], stack: &mut Vec<usize>) {
    if visited[index] {
        return;
    }
    visited[index] = true;
    if active[index] {
        stack.push(index);
    }
}
