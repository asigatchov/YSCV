use crate::{BoundingBox, Detection, batched_nms, iou, non_max_suppression, soft_nms};

#[test]
fn iou_returns_expected_value() {
    let a = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 2.0,
        y2: 2.0,
    };
    let b = BoundingBox {
        x1: 1.0,
        y1: 1.0,
        x2: 3.0,
        y2: 3.0,
    };
    assert!((iou(a, b) - (1.0 / 7.0)).abs() < 1e-6);
}

#[test]
fn nms_keeps_highest_scoring_overlap() {
    let dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 2.0,
                y2: 2.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.2,
                y1: 0.2,
                x2: 2.2,
                y2: 2.2,
            },
            score: 0.8,
            class_id: 0,
        },
    ];
    let out = non_max_suppression(&dets, 0.3, 10);
    assert_eq!(out.len(), 1);
    assert!((out[0].score - 0.9).abs() < 1e-6);
}

#[test]
fn nms_skips_non_finite_candidates() {
    let dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 2.0,
                y2: 2.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.2,
                y1: 0.2,
                x2: 2.2,
                y2: 2.2,
            },
            score: f32::NAN,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: f32::INFINITY,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            score: 0.8,
            class_id: 0,
        },
    ];
    let out = non_max_suppression(&dets, 0.3, 10);
    assert_eq!(out.len(), 1);
    assert!((out[0].score - 0.9).abs() < 1e-6);
}

// ── soft_nms tests ──────────────────────────────────────────────────────

#[test]
fn soft_nms_decays_overlapping() {
    // Two overlapping boxes; the lower-score one should get decayed but not removed.
    let mut dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 2.0,
                y2: 2.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.5,
                y1: 0.5,
                x2: 2.5,
                y2: 2.5,
            },
            score: 0.8,
            class_id: 0,
        },
    ];
    soft_nms(&mut dets, 0.5, 0.01);
    assert_eq!(dets.len(), 2);
    // The first (highest) should keep its score.
    assert!((dets[0].score - 0.9).abs() < 1e-6);
    // The second should be decayed below its original 0.8.
    assert!(dets[1].score < 0.8);
    assert!(dets[1].score > 0.01); // but still above threshold
}

#[test]
fn soft_nms_removes_below_threshold() {
    // Two heavily overlapping boxes; with aggressive decay the lower one is removed.
    let mut dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 2.0,
                y2: 2.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.1,
                y1: 0.1,
                x2: 2.1,
                y2: 2.1,
            },
            score: 0.5,
            class_id: 0,
        },
    ];
    // Use very small sigma so decay is aggressive, and a moderate threshold.
    soft_nms(&mut dets, 0.1, 0.4);
    assert_eq!(dets.len(), 1);
    assert!((dets[0].score - 0.9).abs() < 1e-6);
}

#[test]
fn soft_nms_non_overlapping_unchanged() {
    let mut dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 5.0,
                y1: 5.0,
                x2: 6.0,
                y2: 6.0,
            },
            score: 0.8,
            class_id: 0,
        },
    ];
    soft_nms(&mut dets, 0.5, 0.01);
    assert_eq!(dets.len(), 2);
    assert!((dets[0].score - 0.9).abs() < 1e-6);
    assert!((dets[1].score - 0.8).abs() < 1e-6);
}

// ── batched_nms tests ───────────────────────────────────────────────────

#[test]
fn batched_nms_per_class() {
    // Two overlapping boxes with different class_ids should both survive.
    let dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 2.0,
                y2: 2.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.2,
                y1: 0.2,
                x2: 2.2,
                y2: 2.2,
            },
            score: 0.8,
            class_id: 1,
        },
    ];
    let out = batched_nms(&dets, 0.3);
    assert_eq!(out.len(), 2);
}

#[test]
fn batched_nms_same_class_suppresses() {
    // Two overlapping boxes with the same class_id; one should be suppressed.
    let dets = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 2.0,
                y2: 2.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.2,
                y1: 0.2,
                x2: 2.2,
                y2: 2.2,
            },
            score: 0.8,
            class_id: 0,
        },
    ];
    let out = batched_nms(&dets, 0.3);
    assert_eq!(out.len(), 1);
    assert!((out[0].score - 0.9).abs() < 1e-6);
}
