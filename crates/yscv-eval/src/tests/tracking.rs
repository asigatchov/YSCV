use super::*;

#[test]
fn tracking_perfect_assignment_has_no_errors() {
    let gt1 = [gt_track(1, 0.0, 0.0, 2.0, 2.0)];
    let gt2 = [gt_track(1, 0.2, 0.1, 2.2, 2.1)];
    let pred1 = [tracked(10, 0.0, 0.0, 2.0, 2.0, 0.9)];
    let pred2 = [tracked(10, 0.2, 0.1, 2.2, 2.1, 0.8)];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];

    let metrics = evaluate_tracking(&frames, TrackingEvalConfig::default()).unwrap();
    assert_eq!(metrics.total_ground_truth, 2);
    assert_eq!(metrics.matches, 2);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
    assert_eq!(metrics.id_switches, 0);
    approx_eq(metrics.precision, 1.0);
    approx_eq(metrics.recall, 1.0);
    approx_eq(metrics.f1, 1.0);
    approx_eq(metrics.mota, 1.0);
    approx_eq(metrics.motp, 1.0);
}

#[test]
fn tracking_counts_id_switches() {
    let gt1 = [gt_track(1, 0.0, 0.0, 2.0, 2.0)];
    let gt2 = [gt_track(1, 0.1, 0.1, 2.1, 2.1)];
    let pred1 = [tracked(10, 0.0, 0.0, 2.0, 2.0, 0.9)];
    let pred2 = [tracked(20, 0.1, 0.1, 2.1, 2.1, 0.9)];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];

    let metrics = evaluate_tracking(&frames, TrackingEvalConfig::default()).unwrap();
    assert_eq!(metrics.id_switches, 1);
    approx_eq(metrics.mota, 0.5);
}

#[test]
fn tracking_counts_fp_and_fn() {
    let gt1 = [gt_track(1, 0.0, 0.0, 2.0, 2.0)];
    let gt2: [GroundTruthTrack; 0] = [];
    let pred1: [TrackedDetection; 0] = [];
    let pred2 = [tracked(7, 0.0, 0.0, 2.0, 2.0, 0.9)];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];

    let metrics = evaluate_tracking(&frames, TrackingEvalConfig::default()).unwrap();
    assert_eq!(metrics.matches, 0);
    assert_eq!(metrics.false_negatives, 1);
    assert_eq!(metrics.false_positives, 1);
    approx_eq(metrics.mota, -1.0);
    approx_eq(metrics.motp, 0.0);
}

#[test]
fn tracking_rejects_invalid_iou_threshold() {
    let empty = [TrackingFrame {
        ground_truth: &[],
        predictions: &[],
    }];
    let err = evaluate_tracking(
        &empty,
        TrackingEvalConfig {
            iou_threshold: -0.1,
        },
    )
    .unwrap_err();
    assert_eq!(err, EvalError::InvalidIouThreshold { value: -0.1 });
}

#[test]
fn idf1_perfect_tracking() {
    let gt1 = [
        gt_track(1, 0.0, 0.0, 2.0, 2.0),
        gt_track(2, 3.0, 3.0, 5.0, 5.0),
    ];
    let gt2 = [
        gt_track(1, 0.1, 0.1, 2.1, 2.1),
        gt_track(2, 3.1, 3.1, 5.1, 5.1),
    ];
    let pred1 = [
        tracked(10, 0.0, 0.0, 2.0, 2.0, 0.9),
        tracked(20, 3.0, 3.0, 5.0, 5.0, 0.9),
    ];
    let pred2 = [
        tracked(10, 0.1, 0.1, 2.1, 2.1, 0.9),
        tracked(20, 3.1, 3.1, 5.1, 5.1, 0.9),
    ];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];
    let score = idf1(&frames, TrackingEvalConfig::default()).unwrap();
    approx_eq(score, 1.0);
}

#[test]
fn idf1_swapped_ids() {
    // GT ids 1,2 consistently mapped to pred ids 20,10 (swapped but consistent)
    let gt1 = [
        gt_track(1, 0.0, 0.0, 2.0, 2.0),
        gt_track(2, 3.0, 3.0, 5.0, 5.0),
    ];
    let gt2 = [
        gt_track(1, 0.1, 0.1, 2.1, 2.1),
        gt_track(2, 3.1, 3.1, 5.1, 5.1),
    ];
    let pred1 = [
        tracked(20, 0.0, 0.0, 2.0, 2.0, 0.9),
        tracked(10, 3.0, 3.0, 5.0, 5.0, 0.9),
    ];
    let pred2 = [
        tracked(20, 0.1, 0.1, 2.1, 2.1, 0.9),
        tracked(10, 3.1, 3.1, 5.1, 5.1, 0.9),
    ];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];
    let score = idf1(&frames, TrackingEvalConfig::default()).unwrap();
    approx_eq(score, 1.0);
}

#[test]
fn idf1_all_wrong() {
    // Predictions don't overlap with GT at all
    let gt1 = [gt_track(1, 0.0, 0.0, 1.0, 1.0)];
    let gt2 = [gt_track(1, 0.0, 0.0, 1.0, 1.0)];
    let pred1 = [tracked(10, 5.0, 5.0, 6.0, 6.0, 0.9)];
    let pred2 = [tracked(10, 5.0, 5.0, 6.0, 6.0, 0.9)];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];
    let score = idf1(&frames, TrackingEvalConfig::default()).unwrap();
    approx_eq(score, 0.0);
}

#[test]
fn hota_perfect_tracking() {
    let gt1 = [gt_track(1, 0.0, 0.0, 2.0, 2.0)];
    let gt2 = [gt_track(1, 0.1, 0.1, 2.1, 2.1)];
    let pred1 = [tracked(10, 0.0, 0.0, 2.0, 2.0, 0.9)];
    let pred2 = [tracked(10, 0.1, 0.1, 2.1, 2.1, 0.9)];
    let frames = [
        TrackingFrame {
            ground_truth: &gt1,
            predictions: &pred1,
        },
        TrackingFrame {
            ground_truth: &gt2,
            predictions: &pred2,
        },
    ];
    let score = hota(&frames, TrackingEvalConfig::default()).unwrap();
    approx_eq(score, 1.0);
}

#[test]
fn hota_no_detections() {
    let gt1 = [gt_track(1, 0.0, 0.0, 2.0, 2.0)];
    let pred1: [TrackedDetection; 0] = [];
    let frames = [TrackingFrame {
        ground_truth: &gt1,
        predictions: &pred1,
    }];
    let score = hota(&frames, TrackingEvalConfig::default()).unwrap();
    approx_eq(score, 0.0);
}
