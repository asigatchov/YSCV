use std::fmt::Write as FmtWrite;

use crate::EvalError;

/// Averaging strategy for multi-class F1 score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F1Average {
    /// Compute F1 per class and average (unweighted).
    Macro,
    /// Compute global TP/FP/FN then derive a single F1.
    Micro,
    /// Compute F1 per class and average weighted by support (class count in targets).
    Weighted,
}

/// Compute classification accuracy as the fraction of correct predictions.
///
/// Returns a value in `[0, 1]`. Returns an error if the slices have different lengths.
pub fn accuracy(predictions: &[usize], targets: &[usize]) -> Result<f32, EvalError> {
    if predictions.len() != targets.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: targets.len(),
            predictions: predictions.len(),
        });
    }
    if predictions.is_empty() {
        return Ok(0.0);
    }
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    Ok(correct as f32 / predictions.len() as f32)
}

/// Compute a confusion matrix for `num_classes` classes.
///
/// `result[actual][predicted]` contains the count of samples with true class
/// `actual` that were predicted as class `predicted`.
pub fn confusion_matrix(
    predictions: &[usize],
    targets: &[usize],
    num_classes: usize,
) -> Result<Vec<Vec<usize>>, EvalError> {
    if predictions.len() != targets.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: targets.len(),
            predictions: predictions.len(),
        });
    }
    let mut cm = vec![vec![0usize; num_classes]; num_classes];
    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        if target < num_classes && pred < num_classes {
            cm[target][pred] += 1;
        }
    }
    Ok(cm)
}

/// Compute per-class precision and recall from a confusion matrix.
///
/// Returns a `Vec` of `(precision, recall)` tuples, one per class.
/// If a class has no predictions, precision is `0.0`; if no ground truth, recall is `0.0`.
pub fn per_class_precision_recall(cm: &[Vec<usize>]) -> Vec<(f32, f32)> {
    let n = cm.len();
    let mut result = Vec::with_capacity(n);
    for c in 0..n {
        let tp = cm[c][c] as f32;
        let col_sum: f32 = cm.iter().map(|row| row[c] as f32).sum();
        let row_sum: f32 = cm[c].iter().sum::<usize>() as f32;

        let precision = if col_sum > 0.0 { tp / col_sum } else { 0.0 };
        let recall = if row_sum > 0.0 { tp / row_sum } else { 0.0 };
        result.push((precision, recall));
    }
    result
}

/// Generate a human-readable classification report (similar to scikit-learn's
/// `classification_report`).
///
/// Example output:
/// ```text
///               precision  recall  f1-score  support
///          cat      0.800   0.889     0.842        9
///          dog      0.857   0.750     0.800        8
///     accuracy                        0.824       17
/// ```
pub fn classification_report(
    predictions: &[usize],
    targets: &[usize],
    labels: &[&str],
) -> Result<String, EvalError> {
    let num_classes = labels.len();
    let cm = confusion_matrix(predictions, targets, num_classes)?;
    let pr = per_class_precision_recall(&cm);
    let acc = accuracy(predictions, targets)?;

    let max_label = labels.iter().map(|l| l.len()).max().unwrap_or(5).max(10);
    let mut report = String::new();

    writeln!(
        report,
        "{:>width$}  precision  recall  f1-score  support",
        "",
        width = max_label
    )
    .expect("write to String");

    let total_support = targets.len();

    for (i, label) in labels.iter().enumerate() {
        let (prec, rec) = pr[i];
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        let support: usize = cm[i].iter().sum();
        writeln!(
            report,
            "{:>width$}    {:.3}   {:.3}     {:.3}     {:>4}",
            label,
            prec,
            rec,
            f1,
            support,
            width = max_label
        )
        .expect("write to String");
    }

    writeln!(
        report,
        "{:>width$}                      {:.3}     {:>4}",
        "accuracy",
        acc,
        total_support,
        width = max_label
    )
    .expect("write to String");

    Ok(report)
}

/// Compute F1 score with the specified averaging strategy.
///
/// `num_classes` must be at least as large as the maximum label value + 1.
pub fn f1_score(
    predictions: &[usize],
    targets: &[usize],
    num_classes: usize,
    average: F1Average,
) -> Result<f32, EvalError> {
    if predictions.len() != targets.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: targets.len(),
            predictions: predictions.len(),
        });
    }

    let cm = confusion_matrix(predictions, targets, num_classes)?;

    match average {
        F1Average::Macro => {
            let pr = per_class_precision_recall(&cm);
            let mut sum_f1 = 0.0f32;
            for &(prec, rec) in &pr {
                let f1 = if prec + rec > 0.0 {
                    2.0 * prec * rec / (prec + rec)
                } else {
                    0.0
                };
                sum_f1 += f1;
            }
            Ok(sum_f1 / num_classes as f32)
        }
        F1Average::Micro => {
            let mut tp_total = 0usize;
            let mut fp_total = 0usize;
            let mut fn_total = 0usize;
            for c in 0..num_classes {
                let tp = cm[c][c];
                let fp: usize = cm.iter().map(|row| row[c]).sum::<usize>() - tp;
                let fn_c: usize = cm[c].iter().sum::<usize>() - tp;
                tp_total += tp;
                fp_total += fp;
                fn_total += fn_c;
            }
            let precision = if tp_total + fp_total > 0 {
                tp_total as f32 / (tp_total + fp_total) as f32
            } else {
                0.0
            };
            let recall = if tp_total + fn_total > 0 {
                tp_total as f32 / (tp_total + fn_total) as f32
            } else {
                0.0
            };
            if precision + recall > 0.0 {
                Ok(2.0 * precision * recall / (precision + recall))
            } else {
                Ok(0.0)
            }
        }
        F1Average::Weighted => {
            let pr = per_class_precision_recall(&cm);
            let mut weighted_f1 = 0.0f32;
            let total: usize = targets.len();
            for c in 0..num_classes {
                let support: usize = cm[c].iter().sum();
                let (prec, rec) = pr[c];
                let f1 = if prec + rec > 0.0 {
                    2.0 * prec * rec / (prec + rec)
                } else {
                    0.0
                };
                weighted_f1 += f1 * support as f32;
            }
            if total > 0 {
                Ok(weighted_f1 / total as f32)
            } else {
                Ok(0.0)
            }
        }
    }
}

/// Compute precision-recall curve from binary classification scores and labels.
///
/// Returns `(precisions, recalls, thresholds)` sorted by decreasing threshold.
pub fn precision_recall_curve(
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

    // Sort indices by score descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut precisions = Vec::with_capacity(n);
    let mut recalls = Vec::with_capacity(n);
    let mut thresholds = Vec::with_capacity(n);

    let mut tp = 0.0f32;

    for (rank, &i) in indices.iter().enumerate() {
        if labels[i] {
            tp += 1.0;
        }
        let predicted_pos = (rank + 1) as f32;
        precisions.push(tp / predicted_pos);
        recalls.push(if total_pos > 0.0 { tp / total_pos } else { 0.0 });
        thresholds.push(scores[i]);
    }

    Ok((precisions, recalls, thresholds))
}

/// Compute average precision (area under the precision-recall curve) using the trapezoidal rule.
///
/// Prepends the point (recall=0, precision=1) to ensure the full area is captured.
pub fn average_precision(scores: &[f32], labels: &[bool]) -> Result<f32, EvalError> {
    let (precisions, recalls, _) = precision_recall_curve(scores, labels)?;

    if recalls.is_empty() {
        return Ok(0.0);
    }

    // Prepend (recall=0, precision=1.0) as the starting point of the PR curve.
    let mut full_recalls = Vec::with_capacity(recalls.len() + 1);
    let mut full_precisions = Vec::with_capacity(precisions.len() + 1);
    full_recalls.push(0.0f32);
    full_precisions.push(1.0f32);
    full_recalls.extend_from_slice(&recalls);
    full_precisions.extend_from_slice(&precisions);

    // Trapezoidal rule over recall (which is monotonically non-decreasing)
    let mut ap = 0.0f32;
    for i in 1..full_recalls.len() {
        let dr = full_recalls[i] - full_recalls[i - 1];
        ap += dr * (full_precisions[i] + full_precisions[i - 1]) / 2.0;
    }
    Ok(ap)
}

/// Cohen's kappa coefficient measuring inter-annotator agreement.
///
/// κ = (p_o - p_e) / (1 - p_e) where p_o is observed agreement and p_e is expected agreement.
pub fn cohens_kappa(
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

    let n = predictions.len();
    if n == 0 {
        return Ok(0.0);
    }

    let cm = confusion_matrix(predictions, targets, num_classes)?;
    let n_f = n as f32;

    // Observed agreement
    let p_o: f32 = (0..num_classes).map(|c| cm[c][c] as f32).sum::<f32>() / n_f;

    // Expected agreement
    let mut p_e = 0.0f32;
    for c in 0..num_classes {
        let row_sum: f32 = cm[c].iter().sum::<usize>() as f32; // true class c count
        let col_sum: f32 = cm.iter().map(|row| row[c]).sum::<usize>() as f32; // predicted class c count
        p_e += (row_sum / n_f) * (col_sum / n_f);
    }

    if (1.0 - p_e).abs() < 1e-10 {
        return Ok(1.0); // perfect agreement when p_e ≈ 1
    }

    Ok((p_o - p_e) / (1.0 - p_e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let preds = vec![0, 1, 2, 0, 1];
        let targets = vec![0, 1, 2, 0, 1];
        let acc = accuracy(&preds, &targets).unwrap();
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_half() {
        let preds = vec![0, 0, 1, 1];
        let targets = vec![0, 1, 0, 1];
        let acc = accuracy(&preds, &targets).unwrap();
        assert!((acc - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_length_mismatch() {
        assert!(accuracy(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn test_confusion_matrix_basic() {
        let preds = vec![0, 0, 1, 1, 2, 2];
        let targets = vec![0, 1, 1, 2, 2, 0];
        let cm = confusion_matrix(&preds, &targets, 3).unwrap();

        // Diagonal: correct predictions.
        assert_eq!(cm[0][0], 1); // target=0, pred=0
        assert_eq!(cm[1][1], 1); // target=1, pred=1
        assert_eq!(cm[2][2], 1); // target=2, pred=2

        // Off-diagonal: misclassifications.
        assert_eq!(cm[1][0], 1); // target=1, pred=0
        assert_eq!(cm[2][1], 1); // target=2, pred=1
        assert_eq!(cm[0][2], 1); // target=0, pred=2
    }

    #[test]
    fn test_per_class_precision_recall() {
        // 2 classes: [0, 0, 1, 1] vs [0, 1, 0, 1]
        let cm = confusion_matrix(&[0, 0, 1, 1], &[0, 1, 0, 1], 2).unwrap();
        let pr = per_class_precision_recall(&cm);
        // Class 0: TP=1, FP=1 (pred=0 when target=1), FN=1 => precision=0.5, recall=0.5
        assert!((pr[0].0 - 0.5).abs() < 1e-5);
        assert!((pr[0].1 - 0.5).abs() < 1e-5);
        // Class 1: same situation
        assert!((pr[1].0 - 0.5).abs() < 1e-5);
        assert!((pr[1].1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_classification_report_format() {
        let preds = vec![0, 0, 1, 1, 1];
        let targets = vec![0, 1, 1, 1, 0];
        let report = classification_report(&preds, &targets, &["cat", "dog"]).unwrap();

        assert!(report.contains("precision"));
        assert!(report.contains("recall"));
        assert!(report.contains("cat"));
        assert!(report.contains("dog"));
        assert!(report.contains("accuracy"));
    }

    #[test]
    fn test_accuracy_empty() {
        let acc = accuracy(&[], &[]).unwrap();
        assert_eq!(acc, 0.0);
    }
}
