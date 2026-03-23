use crate::{
    F1Average, accuracy, average_precision, cohens_kappa, f1_score, precision_recall_curve,
};

use super::approx_eq;

#[test]
fn f1_perfect_score_is_one() {
    let preds = vec![0, 1, 2, 0, 1, 2];
    let targets = vec![0, 1, 2, 0, 1, 2];
    let f1_macro = f1_score(&preds, &targets, 3, F1Average::Macro).unwrap();
    let f1_micro = f1_score(&preds, &targets, 3, F1Average::Micro).unwrap();
    let f1_weighted = f1_score(&preds, &targets, 3, F1Average::Weighted).unwrap();
    approx_eq(f1_macro, 1.0);
    approx_eq(f1_micro, 1.0);
    approx_eq(f1_weighted, 1.0);
}

#[test]
fn f1_micro_vs_macro() {
    // Imbalanced: class 0 has 4 samples, class 1 has 1 sample.
    // Predictions are all class 0 — class 1 recall is 0.
    let preds = vec![0, 0, 0, 0, 0];
    let targets = vec![0, 0, 0, 0, 1];
    let f1_macro = f1_score(&preds, &targets, 2, F1Average::Macro).unwrap();
    let f1_micro = f1_score(&preds, &targets, 2, F1Average::Micro).unwrap();
    // Micro F1 == accuracy for multiclass when every sample gets exactly one label.
    let acc = accuracy(&preds, &targets).unwrap();
    approx_eq(f1_micro, acc);
    // Macro will be lower because class 1 has F1=0 and drags the average down.
    assert!(
        (f1_macro - f1_micro).abs() > 1e-3,
        "macro ({f1_macro}) and micro ({f1_micro}) should differ"
    );
}

#[test]
fn f1_weighted_accounts_for_support() {
    // Class 0: 4 samples, class 1: 1 sample. Predict all class 0.
    let preds = vec![0, 0, 0, 0, 0];
    let targets = vec![0, 0, 0, 0, 1];
    let f1_macro = f1_score(&preds, &targets, 2, F1Average::Macro).unwrap();
    let f1_weighted = f1_score(&preds, &targets, 2, F1Average::Weighted).unwrap();
    // Weighted should be higher than macro because class 0 (high support, high F1)
    // dominates, while macro gives equal weight to class 1 (F1=0).
    assert!(
        f1_weighted > f1_macro,
        "weighted ({f1_weighted}) should be > macro ({f1_macro})"
    );
}

#[test]
fn pr_curve_perfect_classifier() {
    // Perfect: positive samples get high scores, negative get low.
    let scores = vec![0.9, 0.8, 0.7, 0.2, 0.1];
    let labels = vec![true, true, true, false, false];
    let (precisions, _recalls, _thresholds) = precision_recall_curve(&scores, &labels).unwrap();
    // Since all positives are ranked above negatives, precision should be 1.0
    // for the first 3 entries (all positives seen before any negative).
    for &p in &precisions[..3] {
        approx_eq(p, 1.0);
    }
}

#[test]
fn average_precision_perfect() {
    let scores = vec![0.9, 0.8, 0.7, 0.2, 0.1];
    let labels = vec![true, true, true, false, false];
    let ap = average_precision(&scores, &labels).unwrap();
    approx_eq(ap, 1.0);
}

#[test]
fn cohens_kappa_perfect_agreement() {
    let preds = vec![0, 1, 2, 0, 1, 2];
    let targets = vec![0, 1, 2, 0, 1, 2];
    let kappa = cohens_kappa(&preds, &targets, 3).unwrap();
    approx_eq(kappa, 1.0);
}

#[test]
fn cohens_kappa_random_is_near_zero() {
    // Predictions that are independent of targets but have the same marginal
    // distribution. With 3 classes and uniform marginals, p_e ~ 1/3.
    // We construct predictions where observed agreement is also ~ 1/3 (chance),
    // so kappa should be near zero.
    //
    // targets: [0,0,0, 1,1,1, 2,2,2, 0,0,0, ...] (30 of each class)
    // preds:   [0,1,2, 0,1,2, 0,1,2, 0,1,2, ...] (30 of each class)
    // Agreement happens when i%3 == (i/3)%3, i.e., roughly 1/3 of the time.
    let n = 90;
    let targets: Vec<usize> = (0..n).map(|i| (i / 3) % 3).collect();
    let preds: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let kappa = cohens_kappa(&preds, &targets, 3).unwrap();
    assert!(
        kappa.abs() < 0.15,
        "kappa for random-like predictions should be near zero, got {kappa}"
    );
}
