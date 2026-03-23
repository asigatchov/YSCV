use super::*;

#[test]
fn detection_perfect_case_returns_full_scores() {
    let gt = [gt_box(0.0, 0.0, 2.0, 2.0), gt_box(3.0, 3.0, 5.0, 5.0)];
    let pred = [det(0.0, 0.0, 2.0, 2.0, 0.95), det(3.0, 3.0, 5.0, 5.0, 0.88)];
    let frames = [DetectionFrame {
        ground_truth: &gt,
        predictions: &pred,
    }];

    let metrics = evaluate_detections(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 2);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
    approx_eq(metrics.precision, 1.0);
    approx_eq(metrics.recall, 1.0);
    approx_eq(metrics.f1, 1.0);
    approx_eq(metrics.average_precision, 1.0);
}

#[test]
fn detection_mixed_case_counts_fp_and_fn() {
    let gt = [gt_box(0.0, 0.0, 2.0, 2.0), gt_box(3.0, 3.0, 5.0, 5.0)];
    let pred = [
        det(0.0, 0.0, 2.0, 2.0, 0.9),
        det(10.0, 10.0, 12.0, 12.0, 0.8),
    ];
    let frames = [DetectionFrame {
        ground_truth: &gt,
        predictions: &pred,
    }];

    let metrics = evaluate_detections(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 1);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 1);
    approx_eq(metrics.precision, 0.5);
    approx_eq(metrics.recall, 0.5);
    approx_eq(metrics.f1, 0.5);
}

#[test]
fn detection_score_threshold_filters_predictions() {
    let gt = [gt_box(0.0, 0.0, 2.0, 2.0)];
    let pred = [det(0.0, 0.0, 2.0, 2.0, 0.4)];
    let frames = [DetectionFrame {
        ground_truth: &gt,
        predictions: &pred,
    }];

    let metrics = evaluate_detections(
        &frames,
        DetectionEvalConfig {
            iou_threshold: 0.5,
            score_threshold: 0.5,
        },
    )
    .unwrap();

    assert_eq!(metrics.true_positives, 0);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 1);
    approx_eq(metrics.average_precision, 0.0);
}

#[test]
fn detection_rejects_invalid_config() {
    let empty = [DetectionFrame {
        ground_truth: &[],
        predictions: &[],
    }];
    let err = evaluate_detections(
        &empty,
        DetectionEvalConfig {
            iou_threshold: 1.2,
            score_threshold: 0.5,
        },
    )
    .unwrap_err();
    assert_eq!(err, EvalError::InvalidIouThreshold { value: 1.2 });

    let err = evaluate_detections(
        &empty,
        DetectionEvalConfig {
            iou_threshold: 0.5,
            score_threshold: f32::NAN,
        },
    )
    .unwrap_err();
    match err {
        EvalError::InvalidScoreThreshold { value } => assert!(value.is_nan()),
        other => panic!("unexpected error: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// COCO-style multi-threshold tests
// ---------------------------------------------------------------------------

#[test]
fn coco_perfect_detections() {
    // Perfect predictions with IoU=1.0 should yield AP close to 1.0 at every threshold.
    let gt = [
        gt_box(0.0, 0.0, 100.0, 100.0),
        gt_box(200.0, 200.0, 300.0, 300.0),
    ];
    let pred = [
        det(0.0, 0.0, 100.0, 100.0, 0.99),
        det(200.0, 200.0, 300.0, 300.0, 0.95),
    ];
    let frames = [DetectionFrame {
        ground_truth: &gt,
        predictions: &pred,
    }];

    let coco = evaluate_detections_coco(&frames, 0.0).unwrap();
    assert!(coco.ap > 0.99, "ap should be ~1.0 but got {}", coco.ap);
    assert!(
        coco.ap50 > 0.99,
        "ap50 should be ~1.0 but got {}",
        coco.ap50
    );
    assert!(
        coco.ap75 > 0.99,
        "ap75 should be ~1.0 but got {}",
        coco.ap75
    );
    assert!(coco.ar > 0.99, "ar should be ~1.0 but got {}", coco.ar);
}

#[test]
fn coco_no_detections() {
    let gt = [gt_box(0.0, 0.0, 50.0, 50.0)];
    let pred: [Detection; 0] = [];
    let frames = [DetectionFrame {
        ground_truth: &gt,
        predictions: &pred,
    }];

    let coco = evaluate_detections_coco(&frames, 0.0).unwrap();
    approx_eq(coco.ap, 0.0);
    approx_eq(coco.ap50, 0.0);
    approx_eq(coco.ap75, 0.0);
    approx_eq(coco.ar, 0.0);
}

#[test]
fn coco_ap50_higher_than_ap75() {
    // Predictions that partially overlap: good enough for IoU=0.5 but worse at 0.75.
    // GT box is 0..100, prediction is shifted so IoU is around 0.58 (passes 0.5, fails 0.75).
    let gt = [gt_box(0.0, 0.0, 100.0, 100.0)];
    // Overlap region: 30..100 x 0..100 = 70*100 = 7000
    // Union: 100*100 + 100*100 - 7000 = 13000
    // IoU = 7000/13000 ≈ 0.538
    let pred = [det(30.0, 0.0, 130.0, 100.0, 0.9)];
    let frames = [DetectionFrame {
        ground_truth: &gt,
        predictions: &pred,
    }];

    let coco = evaluate_detections_coco(&frames, 0.0).unwrap();
    assert!(
        coco.ap50 >= coco.ap75,
        "ap50 ({}) should be >= ap75 ({})",
        coco.ap50,
        coco.ap75,
    );
    // ap50 should be 1.0 (IoU 0.538 > 0.5) and ap75 should be 0.0 (IoU 0.538 < 0.75)
    assert!(
        coco.ap50 > 0.99,
        "ap50 should be ~1.0 but got {}",
        coco.ap50
    );
    approx_eq(coco.ap75, 0.0);
}
