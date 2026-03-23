use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Bounding box with confidence for NMS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// Greedy non-maximum suppression on bounding boxes.
///
/// Returns indices of kept boxes, sorted by descending confidence.
pub fn nms(boxes: &[BBox], iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b]
            .score
            .partial_cmp(&boxes[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &indices {
            if suppressed[j] || j == i {
                continue;
            }
            if iou(&boxes[i], &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

fn iou(a: &BBox, b: &BBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Template matching method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateMatchMethod {
    /// Sum of squared differences (lower = better match).
    Ssd,
    /// Normalized cross-correlation (higher = better match).
    Ncc,
}

/// Template matching result: top-left location and score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemplateMatchResult {
    pub x: usize,
    pub y: usize,
    pub score: f32,
}

/// Slides `template` over `image` computing a similarity map, returns the best match.
///
/// Both must be single-channel `[H, W, 1]`.
pub fn template_match(
    image: &Tensor,
    template: &Tensor,
    method: TemplateMatchMethod,
) -> Result<TemplateMatchResult, ImgProcError> {
    let (ih, iw, ic) = hwc_shape(image)?;
    let (th, tw, tc) = hwc_shape(template)?;
    if ic != 1 || tc != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: if ic != 1 { ic } else { tc },
        });
    }
    if th > ih || tw > iw {
        return Err(ImgProcError::InvalidSize {
            height: th,
            width: tw,
        });
    }
    let img = image.data();
    let tmpl = template.data();
    let rh = ih - th + 1;
    let rw = iw - tw + 1;

    let mut best = TemplateMatchResult {
        x: 0,
        y: 0,
        score: match method {
            TemplateMatchMethod::Ssd => f32::MAX,
            TemplateMatchMethod::Ncc => f32::NEG_INFINITY,
        },
    };

    // Precompute template stats for NCC
    let tmpl_mean: f32 = tmpl.iter().sum::<f32>() / tmpl.len() as f32;
    let tmpl_std: f32 = {
        let var: f32 = tmpl
            .iter()
            .map(|&v| (v - tmpl_mean) * (v - tmpl_mean))
            .sum::<f32>()
            / tmpl.len() as f32;
        var.sqrt()
    };

    for y in 0..rh {
        for x in 0..rw {
            let score = match method {
                TemplateMatchMethod::Ssd => {
                    let mut sum = 0.0f32;
                    for ty in 0..th {
                        for tx in 0..tw {
                            let diff = img[(y + ty) * iw + x + tx] - tmpl[ty * tw + tx];
                            sum += diff * diff;
                        }
                    }
                    sum
                }
                TemplateMatchMethod::Ncc => {
                    let patch_size = (th * tw) as f32;
                    let mut patch_mean = 0.0f32;
                    for ty in 0..th {
                        for tx in 0..tw {
                            patch_mean += img[(y + ty) * iw + x + tx];
                        }
                    }
                    patch_mean /= patch_size;
                    let mut num = 0.0f32;
                    let mut den_patch = 0.0f32;
                    for ty in 0..th {
                        for tx in 0..tw {
                            let pi = img[(y + ty) * iw + x + tx] - patch_mean;
                            let ti = tmpl[ty * tw + tx] - tmpl_mean;
                            num += pi * ti;
                            den_patch += pi * pi;
                        }
                    }
                    let den = (den_patch.sqrt()) * (tmpl_std * patch_size.sqrt());
                    if den.abs() < 1e-10 { 0.0 } else { num / den }
                }
            };

            let is_better = match method {
                TemplateMatchMethod::Ssd => score < best.score,
                TemplateMatchMethod::Ncc => score > best.score,
            };
            if is_better {
                best = TemplateMatchResult { x, y, score };
            }
        }
    }

    Ok(best)
}
