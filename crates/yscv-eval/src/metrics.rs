//! Advanced evaluation metrics: top-k accuracy, ROC/AUC, IoU, SSIM, PSNR.

use crate::EvalError;

/// Top-k accuracy: fraction of samples where the correct label is in the top-k predictions.
///
/// `scores`: `[N, C]` matrix (N samples, C classes) — raw scores or probabilities.
/// `targets`: `[N]` — true class indices.
pub fn top_k_accuracy(
    scores: &[f32],
    num_classes: usize,
    targets: &[usize],
    k: usize,
) -> Result<f32, EvalError> {
    if scores.is_empty() || num_classes == 0 {
        return Ok(0.0);
    }
    let n = scores.len() / num_classes;
    if n != targets.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: targets.len(),
            predictions: n,
        });
    }

    let mut correct = 0;
    for i in 0..n {
        let row = &scores[i * num_classes..(i + 1) * num_classes];
        let mut indices: Vec<usize> = (0..num_classes).collect();
        indices.sort_unstable_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if indices[..k.min(num_classes)].contains(&targets[i]) {
            correct += 1;
        }
    }
    Ok(correct as f32 / n as f32)
}

/// Compute ROC curve from binary classification scores and labels.
///
/// Returns `(fpr, tpr, thresholds)` — sorted by decreasing threshold.
pub fn roc_curve(
    scores: &[f32],
    labels: &[bool],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), EvalError> {
    if scores.len() != labels.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: labels.len(),
            predictions: scores.len(),
        });
    }

    let n = scores.len();
    let total_pos = labels.iter().filter(|&&l| l).count() as f32;
    let total_neg = n as f32 - total_pos;

    if total_pos == 0.0 || total_neg == 0.0 {
        return Ok((
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![f32::INFINITY, f32::NEG_INFINITY],
        ));
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fpr_list = vec![0.0f32];
    let mut tpr_list = vec![0.0f32];
    let mut thresholds = vec![f32::INFINITY];

    let mut tp = 0.0f32;
    let mut fp = 0.0f32;

    for &i in &indices {
        if labels[i] {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        fpr_list.push(fp / total_neg);
        tpr_list.push(tp / total_pos);
        thresholds.push(scores[i]);
    }

    Ok((fpr_list, tpr_list, thresholds))
}

/// Area under the curve using the trapezoidal rule.
///
/// `x` and `y` must have the same length and be sorted by x.
pub fn auc(x: &[f32], y: &[f32]) -> Result<f32, EvalError> {
    if x.len() != y.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: x.len(),
            predictions: y.len(),
        });
    }
    if x.len() < 2 {
        return Ok(0.0);
    }

    let mut area = 0.0f32;
    for i in 1..x.len() {
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }
    Ok(area.abs())
}

/// Mean Intersection over Union for semantic segmentation.
///
/// `predictions` and `targets` are flat label maps (same length), `num_classes` classes.
pub fn mean_iou(
    predictions: &[usize],
    targets: &[usize],
    num_classes: usize,
) -> Result<f32, EvalError> {
    if predictions.len() != targets.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: targets.len(),
            predictions: predictions.len(),
        });
    }

    let mut intersection = vec![0usize; num_classes];
    let mut union = vec![0usize; num_classes];

    for (&p, &t) in predictions.iter().zip(targets.iter()) {
        if t < num_classes {
            if p == t {
                intersection[t] += 1;
            }
            union[t] += 1;
        }
        if p < num_classes && p != t {
            union[p] += 1;
        }
    }

    let mut sum_iou = 0.0f32;
    let mut valid_classes = 0;
    for c in 0..num_classes {
        if union[c] > 0 {
            sum_iou += intersection[c] as f32 / union[c] as f32;
            valid_classes += 1;
        }
    }

    if valid_classes == 0 {
        return Ok(0.0);
    }
    Ok(sum_iou / valid_classes as f32)
}

/// Per-class Dice coefficient: 2 * |pred ∩ target| / (|pred| + |target|).
///
/// Returns a `Vec` of Dice scores, one per class.  Classes that appear in
/// neither predictions nor targets receive a score of 0.0.
pub fn dice_score(predictions: &[usize], targets: &[usize], num_classes: usize) -> Vec<f32> {
    let mut tp = vec![0usize; num_classes];
    let mut fp = vec![0usize; num_classes];
    let mut fn_ = vec![0usize; num_classes];

    for (&p, &t) in predictions.iter().zip(targets.iter()) {
        if p == t {
            if p < num_classes {
                tp[p] += 1;
            }
        } else {
            if p < num_classes {
                fp[p] += 1;
            }
            if t < num_classes {
                fn_[t] += 1;
            }
        }
    }

    (0..num_classes)
        .map(|c| {
            let denom = 2 * tp[c] + fp[c] + fn_[c];
            if denom == 0 {
                0.0
            } else {
                (2 * tp[c]) as f32 / denom as f32
            }
        })
        .collect()
}

/// Per-class Intersection over Union: |pred ∩ target| / |pred ∪ target|.
///
/// Returns a `Vec` of IoU scores, one per class.  Classes that appear in
/// neither predictions nor targets receive a score of 0.0.
pub fn per_class_iou(predictions: &[usize], targets: &[usize], num_classes: usize) -> Vec<f32> {
    let mut tp = vec![0usize; num_classes];
    let mut fp = vec![0usize; num_classes];
    let mut fn_ = vec![0usize; num_classes];

    for (&p, &t) in predictions.iter().zip(targets.iter()) {
        if p == t {
            if p < num_classes {
                tp[p] += 1;
            }
        } else {
            if p < num_classes {
                fp[p] += 1;
            }
            if t < num_classes {
                fn_[t] += 1;
            }
        }
    }

    (0..num_classes)
        .map(|c| {
            let denom = tp[c] + fp[c] + fn_[c];
            if denom == 0 {
                0.0
            } else {
                tp[c] as f32 / denom as f32
            }
        })
        .collect()
}

/// Structural Similarity Index (SSIM) between two grayscale images.
///
/// Both inputs are flat f32 slices of the same length (H*W).
pub fn ssim(img1: &[f32], img2: &[f32]) -> Result<f32, EvalError> {
    if img1.len() != img2.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: img1.len(),
            predictions: img2.len(),
        });
    }
    let n = img1.len() as f32;
    if n == 0.0 {
        return Ok(1.0);
    }

    let c1 = (0.01f32 * 1.0).powi(2); // L=1.0 for [0,1] range
    let c2 = (0.03f32 * 1.0).powi(2);

    let mu1: f32 = img1.iter().sum::<f32>() / n;
    let mu2: f32 = img2.iter().sum::<f32>() / n;

    let sigma1_sq: f32 = img1.iter().map(|&v| (v - mu1).powi(2)).sum::<f32>() / n;
    let sigma2_sq: f32 = img2.iter().map(|&v| (v - mu2).powi(2)).sum::<f32>() / n;
    let sigma12: f32 = img1
        .iter()
        .zip(img2.iter())
        .map(|(&a, &b)| (a - mu1) * (b - mu2))
        .sum::<f32>()
        / n;

    let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
    let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);

    Ok(numerator / denominator)
}

/// Peak Signal-to-Noise Ratio between two images.
///
/// Both inputs are flat f32 slices (same length). `max_val` is the maximum pixel value.
pub fn psnr(img1: &[f32], img2: &[f32], max_val: f32) -> Result<f32, EvalError> {
    if img1.len() != img2.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: img1.len(),
            predictions: img2.len(),
        });
    }
    let mse: f32 = img1
        .iter()
        .zip(img2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / img1.len() as f32;

    if mse == 0.0 {
        return Ok(f32::INFINITY);
    }
    Ok(10.0 * (max_val.powi(2) / mse).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_accuracy() {
        // 2 samples, 3 classes
        // sample 0: scores=[0.1, 0.8, 0.5], target=2 → top-1=class1 → wrong, top-2=[1,2] → correct
        // sample 1: scores=[0.7, 0.2, 0.1], target=0 → top-1=class0 → correct
        let scores = vec![0.1, 0.8, 0.5, 0.7, 0.2, 0.1];
        let targets = vec![2, 0];
        let acc = top_k_accuracy(&scores, 3, &targets, 1).unwrap();
        assert!((acc - 0.5).abs() < 1e-6); // only second sample correct at k=1
        let acc_k2 = top_k_accuracy(&scores, 3, &targets, 2).unwrap();
        assert!((acc_k2 - 1.0).abs() < 1e-6); // both correct at k=2
    }

    #[test]
    fn test_roc_curve_and_auc() {
        let scores = vec![0.9, 0.8, 0.4, 0.3, 0.1];
        let labels = vec![true, true, false, false, false];
        let (fpr, tpr, _) = roc_curve(&scores, &labels).unwrap();
        let area = auc(&fpr, &tpr).unwrap();
        assert!(area > 0.9, "AUC should be high: {area}");
    }

    #[test]
    fn test_auc_perfect() {
        let fpr = vec![0.0, 0.0, 1.0];
        let tpr = vec![0.0, 1.0, 1.0];
        let area = auc(&fpr, &tpr).unwrap();
        assert!((area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_iou() {
        let preds = vec![0, 0, 1, 1, 2, 2];
        let targets = vec![0, 0, 1, 1, 2, 2];
        let miou = mean_iou(&preds, &targets, 3).unwrap();
        assert!((miou - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_iou_partial() {
        let preds = vec![0, 1, 1, 0];
        let targets = vec![0, 0, 1, 1];
        let miou = mean_iou(&preds, &targets, 2).unwrap();
        // class 0: intersection=1, union=3 → 1/3
        // class 1: intersection=1, union=3 → 1/3
        // mean = 1/3
        assert!((miou - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_ssim_identical() {
        let img = vec![0.5f32; 100];
        let val = ssim(&img, &img).unwrap();
        assert!((val - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_psnr_identical() {
        let img = vec![0.5f32; 100];
        let val = psnr(&img, &img, 1.0).unwrap();
        assert!(val.is_infinite() && val > 0.0);
    }

    #[test]
    fn dice_score_perfect() {
        let preds = vec![0, 0, 1, 1, 2, 2];
        let targets = vec![0, 0, 1, 1, 2, 2];
        let scores = dice_score(&preds, &targets, 3);
        for &s in &scores {
            assert!((s - 1.0).abs() < 1e-6, "expected 1.0, got {s}");
        }
    }

    #[test]
    fn dice_score_partial() {
        // preds:   [0, 1, 1, 0]
        // targets: [0, 0, 1, 1]
        // class 0: tp=1, fp=1, fn=1 → dice = 2/4 = 0.5
        // class 1: tp=1, fp=1, fn=1 → dice = 2/4 = 0.5
        let preds = vec![0, 1, 1, 0];
        let targets = vec![0, 0, 1, 1];
        let scores = dice_score(&preds, &targets, 2);
        assert!((scores[0] - 0.5).abs() < 1e-6, "class 0: {}", scores[0]);
        assert!((scores[1] - 0.5).abs() < 1e-6, "class 1: {}", scores[1]);
    }

    #[test]
    fn per_class_iou_known_values() {
        // preds:   [0, 1, 1, 0]
        // targets: [0, 0, 1, 1]
        // class 0: tp=1, fp=1, fn=1 → iou = 1/3
        // class 1: tp=1, fp=1, fn=1 → iou = 1/3
        let preds = vec![0, 1, 1, 0];
        let targets = vec![0, 0, 1, 1];
        let ious = per_class_iou(&preds, &targets, 2);
        assert!((ious[0] - 1.0 / 3.0).abs() < 1e-6, "class 0: {}", ious[0]);
        assert!((ious[1] - 1.0 / 3.0).abs() < 1e-6, "class 1: {}", ious[1]);
    }

    #[test]
    fn per_class_iou_no_overlap() {
        // preds are all class 0, targets are all class 1
        let preds = vec![0, 0, 0, 0];
        let targets = vec![1, 1, 1, 1];
        let ious = per_class_iou(&preds, &targets, 2);
        assert!((ious[0]).abs() < 1e-6, "class 0 should be 0: {}", ious[0]);
        assert!((ious[1]).abs() < 1e-6, "class 1 should be 0: {}", ious[1]);
    }

    #[test]
    fn test_psnr_different() {
        let img1 = vec![0.0f32; 100];
        let img2 = vec![1.0f32; 100];
        let val = psnr(&img1, &img2, 1.0).unwrap();
        assert!((val - 0.0).abs() < 1e-6); // MSE=1, PSNR=0
    }
}
