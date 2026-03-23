//! Trait-based model detector interface.
//!
//! Provides an abstract `ModelDetector` trait so that model-backed detectors
//! (YOLO, SSD, FCOS, etc.) can be plugged in alongside the existing heatmap
//! pipeline.

use yscv_tensor::Tensor;

use crate::{DetectError, Detection};

/// Configuration for a model-based detector.
#[derive(Debug, Clone)]
pub struct ModelDetectorConfig {
    /// Minimum score to keep a detection.
    pub score_threshold: f32,
    /// IoU threshold for NMS post-processing.
    pub nms_iou_threshold: f32,
    /// Maximum number of detections to return.
    pub max_detections: usize,
    /// Expected input height for the model.
    pub input_height: usize,
    /// Expected input width for the model.
    pub input_width: usize,
}

impl Default for ModelDetectorConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            nms_iou_threshold: 0.45,
            max_detections: 100,
            input_height: 640,
            input_width: 640,
        }
    }
}

/// Abstract interface for model-backed object detectors.
///
/// Implementors provide `detect_tensor` which takes a preprocessed input
/// tensor and returns detections. The framework handles NMS and thresholding
/// via the config.
pub trait ModelDetector {
    /// Run detection on a preprocessed input tensor.
    ///
    /// The tensor format depends on the model. Typical NHWC shape:
    /// `[1, H, W, C]` where `H` and `W` match the config dimensions.
    ///
    /// Returns raw detections (before NMS). The caller may apply
    /// `non_max_suppression` from this crate.
    fn detect_tensor(&self, input: &Tensor) -> Result<Vec<Detection>, DetectError>;

    /// Returns the class labels this detector can produce.
    fn class_labels(&self) -> &[&str];

    /// Returns the expected input shape `[H, W, C]`.
    fn input_shape(&self) -> [usize; 3];
}

/// Post-processes raw model output into final detections.
///
/// Applies score thresholding and NMS.
pub fn postprocess_detections(raw: &[Detection], config: &ModelDetectorConfig) -> Vec<Detection> {
    // Score filter
    let mut filtered: Vec<Detection> = raw
        .iter()
        .copied()
        .filter(|d| d.score >= config.score_threshold)
        .collect();

    // Sort by score descending
    filtered.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // NMS per class
    let mut result = Vec::new();
    let mut suppressed = vec![false; filtered.len()];

    for i in 0..filtered.len() {
        if suppressed[i] {
            continue;
        }
        result.push(filtered[i]);
        if result.len() >= config.max_detections {
            break;
        }
        for j in i + 1..filtered.len() {
            if suppressed[j] || filtered[j].class_id != filtered[i].class_id {
                continue;
            }
            if crate::iou(filtered[i].bbox, filtered[j].bbox) > config.nms_iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    result
}

/// Preprocess an RGB8 image for model input.
///
/// Resizes to `(target_h, target_w)`, normalizes to `[0, 1]`, and returns
/// an NHWC tensor `[1, target_h, target_w, 3]`.
pub fn preprocess_rgb8_for_model(
    rgb8: &[u8],
    width: usize,
    height: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor, DetectError> {
    if rgb8.len() < width * height * 3 {
        return Err(DetectError::InvalidRgb8BufferSize {
            expected: width * height * 3,
            got: rgb8.len(),
        });
    }

    // Simple bilinear resize + normalize
    let mut data = Vec::with_capacity(target_h * target_w * 3);
    let scale_y = height as f32 / target_h as f32;
    let scale_x = width as f32 / target_w as f32;

    for row in 0..target_h {
        let src_y = (row as f32 * scale_y).min((height - 1) as f32);
        let y0 = src_y as usize;
        let y1 = (y0 + 1).min(height - 1);
        let fy = src_y - y0 as f32;

        for col in 0..target_w {
            let src_x = (col as f32 * scale_x).min((width - 1) as f32);
            let x0 = src_x as usize;
            let x1 = (x0 + 1).min(width - 1);
            let fx = src_x - x0 as f32;

            for ch in 0..3 {
                let v00 = rgb8[(y0 * width + x0) * 3 + ch] as f32;
                let v01 = rgb8[(y0 * width + x1) * 3 + ch] as f32;
                let v10 = rgb8[(y1 * width + x0) * 3 + ch] as f32;
                let v11 = rgb8[(y1 * width + x1) * 3 + ch] as f32;
                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                data.push(v / 255.0);
            }
        }
    }

    Tensor::from_vec(vec![1, target_h, target_w, 3], data).map_err(DetectError::Tensor)
}
