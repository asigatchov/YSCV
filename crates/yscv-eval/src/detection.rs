use yscv_detect::{BoundingBox, Detection, iou};

use crate::EvalError;
use crate::util::{harmonic_mean, safe_ratio, validate_iou_threshold, validate_score_threshold};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabeledBox {
    pub bbox: BoundingBox,
    pub class_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionFrame<'a> {
    pub ground_truth: &'a [LabeledBox],
    pub predictions: &'a [Detection],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionEvalConfig {
    pub iou_threshold: f32,
    pub score_threshold: f32,
}

impl Default for DetectionEvalConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            score_threshold: 0.0,
        }
    }
}

impl DetectionEvalConfig {
    pub fn validate(&self) -> Result<(), EvalError> {
        validate_iou_threshold(self.iou_threshold)?;
        validate_score_threshold(self.score_threshold)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionMetrics {
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub average_precision: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionDatasetFrame {
    pub ground_truth: Vec<LabeledBox>,
    pub predictions: Vec<Detection>,
}

impl DetectionDatasetFrame {
    pub fn as_view(&self) -> DetectionFrame<'_> {
        DetectionFrame {
            ground_truth: &self.ground_truth,
            predictions: &self.predictions,
        }
    }
}

pub fn detection_frames_as_view(frames: &[DetectionDatasetFrame]) -> Vec<DetectionFrame<'_>> {
    frames.iter().map(DetectionDatasetFrame::as_view).collect()
}

pub fn evaluate_detections_from_dataset(
    frames: &[DetectionDatasetFrame],
    config: DetectionEvalConfig,
) -> Result<DetectionMetrics, EvalError> {
    let borrowed = detection_frames_as_view(frames);
    evaluate_detections(&borrowed, config)
}

pub fn evaluate_detections(
    frames: &[DetectionFrame<'_>],
    config: DetectionEvalConfig,
) -> Result<DetectionMetrics, EvalError> {
    config.validate()?;

    let mut true_positives = 0u64;
    let mut false_positives = 0u64;
    let mut false_negatives = 0u64;

    for frame in frames {
        let mut predictions: Vec<Detection> = frame
            .predictions
            .iter()
            .copied()
            .filter(|prediction| prediction.score >= config.score_threshold)
            .collect();
        predictions.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut gt_taken = vec![false; frame.ground_truth.len()];
        for prediction in predictions {
            if let Some(best_gt_idx) = best_gt_match(
                prediction,
                frame.ground_truth,
                &gt_taken,
                config.iou_threshold,
            ) {
                gt_taken[best_gt_idx] = true;
                true_positives += 1;
            } else {
                false_positives += 1;
            }
        }

        false_negatives += gt_taken.iter().filter(|matched| !**matched).count() as u64;
    }

    let precision = safe_ratio(true_positives, true_positives + false_positives);
    let recall = safe_ratio(true_positives, true_positives + false_negatives);
    let f1 = harmonic_mean(precision, recall);
    let average_precision = average_precision(frames, config);

    Ok(DetectionMetrics {
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
        average_precision,
    })
}

fn best_gt_match(
    prediction: Detection,
    ground_truth: &[LabeledBox],
    gt_taken: &[bool],
    iou_threshold: f32,
) -> Option<usize> {
    let mut best_iou = iou_threshold;
    let mut best_idx = None;

    for (idx, gt) in ground_truth.iter().enumerate() {
        if gt_taken[idx] || gt.class_id != prediction.class_id {
            continue;
        }
        let overlap = iou(gt.bbox, prediction.bbox);
        if overlap >= best_iou {
            best_iou = overlap;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn average_precision(frames: &[DetectionFrame<'_>], config: DetectionEvalConfig) -> f32 {
    let total_ground_truth = frames
        .iter()
        .map(|frame| frame.ground_truth.len() as u64)
        .sum::<u64>();
    if total_ground_truth == 0 {
        return 0.0;
    }

    let mut ranked_predictions = Vec::new();
    for (frame_idx, frame) in frames.iter().enumerate() {
        for prediction in frame.predictions {
            if prediction.score >= config.score_threshold {
                ranked_predictions.push((frame_idx, *prediction));
            }
        }
    }
    ranked_predictions.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));

    if ranked_predictions.is_empty() {
        return 0.0;
    }

    let mut gt_taken: Vec<Vec<bool>> = frames
        .iter()
        .map(|frame| vec![false; frame.ground_truth.len()])
        .collect();
    let mut precisions = Vec::with_capacity(ranked_predictions.len());
    let mut recalls = Vec::with_capacity(ranked_predictions.len());

    let mut true_positives = 0u64;
    let mut false_positives = 0u64;

    for (frame_idx, prediction) in ranked_predictions {
        if let Some(best_gt_idx) = best_gt_match(
            prediction,
            frames[frame_idx].ground_truth,
            &gt_taken[frame_idx],
            config.iou_threshold,
        ) {
            gt_taken[frame_idx][best_gt_idx] = true;
            true_positives += 1;
        } else {
            false_positives += 1;
        }

        precisions.push(safe_ratio(true_positives, true_positives + false_positives));
        recalls.push(safe_ratio(true_positives, total_ground_truth));
    }

    let mut monotonic_precisions = Vec::with_capacity(precisions.len() + 2);
    let mut padded_recalls = Vec::with_capacity(recalls.len() + 2);

    padded_recalls.push(0.0);
    padded_recalls.extend(recalls.iter().copied());
    padded_recalls.push(1.0);

    monotonic_precisions.push(0.0);
    monotonic_precisions.extend(precisions.iter().copied());
    monotonic_precisions.push(0.0);

    for idx in (0..monotonic_precisions.len() - 1).rev() {
        monotonic_precisions[idx] = monotonic_precisions[idx].max(monotonic_precisions[idx + 1]);
    }

    let mut ap = 0.0f32;
    for idx in 0..padded_recalls.len() - 1 {
        let recall_delta = padded_recalls[idx + 1] - padded_recalls[idx];
        if recall_delta > 0.0 {
            ap += recall_delta * monotonic_precisions[idx + 1];
        }
    }
    ap.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// COCO-style multi-threshold evaluation
// ---------------------------------------------------------------------------

const COCO_IOU_THRESHOLDS: [f32; 10] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

const SMALL_AREA_MAX: f32 = 32.0 * 32.0;
const MEDIUM_AREA_MAX: f32 = 96.0 * 96.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CocoMetrics {
    /// AP averaged over IoU thresholds 0.50..=0.95 (step 0.05).
    pub ap: f32,
    /// AP at IoU = 0.50.
    pub ap50: f32,
    /// AP at IoU = 0.75.
    pub ap75: f32,
    /// AP for small objects (area < 32²).
    pub ap_small: f32,
    /// AP for medium objects (32² <= area < 96²).
    pub ap_medium: f32,
    /// AP for large objects (area >= 96²).
    pub ap_large: f32,
    /// Average Recall: mean of max recall across IoU thresholds.
    pub ar: f32,
}

fn box_area(b: &BoundingBox) -> f32 {
    (b.x2 - b.x1) * (b.y2 - b.y1)
}

/// Filter frames so that only ground-truth boxes satisfying `pred` are kept.
/// Predictions are kept unchanged (they are matched against the filtered GT).
fn filter_gt_by<F>(frames: &[DetectionFrame<'_>], pred: F) -> Vec<DetectionDatasetFrame>
where
    F: Fn(&LabeledBox) -> bool,
{
    frames
        .iter()
        .map(|frame| {
            let ground_truth: Vec<LabeledBox> = frame
                .ground_truth
                .iter()
                .filter(|lb| pred(lb))
                .copied()
                .collect();
            DetectionDatasetFrame {
                ground_truth,
                predictions: frame.predictions.to_vec(),
            }
        })
        .collect()
}

/// Evaluate detections using COCO-style multi-threshold metrics.
pub fn evaluate_detections_coco(
    frames: &[DetectionFrame<'_>],
    score_threshold: f32,
) -> Result<CocoMetrics, EvalError> {
    validate_score_threshold(score_threshold)?;

    // Compute per-threshold AP and recall.
    let mut aps = [0.0f32; 10];
    let mut recalls = [0.0f32; 10];

    for (i, &iou_thresh) in COCO_IOU_THRESHOLDS.iter().enumerate() {
        let config = DetectionEvalConfig {
            iou_threshold: iou_thresh,
            score_threshold,
        };
        let m = evaluate_detections(frames, config)?;
        aps[i] = m.average_precision;
        recalls[i] = m.recall;
    }

    let ap = aps.iter().sum::<f32>() / aps.len() as f32;
    let ap50 = aps[0]; // IoU 0.50
    let ap75 = aps[5]; // IoU 0.75
    let ar = recalls.iter().sum::<f32>() / recalls.len() as f32;

    // Size-based AP (computed at all 10 thresholds, then averaged).
    let ap_small = size_ap(frames, score_threshold, |a| a < SMALL_AREA_MAX)?;
    let ap_medium = size_ap(frames, score_threshold, |a| {
        (SMALL_AREA_MAX..MEDIUM_AREA_MAX).contains(&a)
    })?;
    let ap_large = size_ap(frames, score_threshold, |a| a >= MEDIUM_AREA_MAX)?;

    Ok(CocoMetrics {
        ap,
        ap50,
        ap75,
        ap_small,
        ap_medium,
        ap_large,
        ar,
    })
}

fn size_ap<F>(
    frames: &[DetectionFrame<'_>],
    score_threshold: f32,
    area_filter: F,
) -> Result<f32, EvalError>
where
    F: Fn(f32) -> bool,
{
    let owned = filter_gt_by(frames, |lb| area_filter(box_area(&lb.bbox)));
    let views = detection_frames_as_view(&owned);

    let mut sum = 0.0f32;
    for &iou_thresh in &COCO_IOU_THRESHOLDS {
        let config = DetectionEvalConfig {
            iou_threshold: iou_thresh,
            score_threshold,
        };
        let m = evaluate_detections(&views, config)?;
        sum += m.average_precision;
    }
    Ok(sum / COCO_IOU_THRESHOLDS.len() as f32)
}
