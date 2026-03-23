use crate::{
    BoundingBox, Detection, ModelDetectorConfig, postprocess_detections, preprocess_rgb8_for_model,
};

#[test]
fn preprocess_rgb8_basic_shape() {
    let rgb = vec![128u8; 4 * 4 * 3]; // 4x4 RGB
    let t = preprocess_rgb8_for_model(&rgb, 4, 4, 2, 2).unwrap();
    assert_eq!(t.shape(), &[1, 2, 2, 3]);
    // All values should be 128/255 ~ 0.502
    for &v in t.data() {
        assert!((v - 128.0 / 255.0).abs() < 0.01);
    }
}

#[test]
fn preprocess_rgb8_rejects_short_buffer() {
    let rgb = vec![0u8; 5]; // too short for 2x2x3=12
    assert!(preprocess_rgb8_for_model(&rgb, 2, 2, 2, 2).is_err());
}

#[test]
fn postprocess_detections_filters_by_score() {
    let config = ModelDetectorConfig {
        score_threshold: 0.5,
        nms_iou_threshold: 0.5,
        max_detections: 10,
        input_height: 640,
        input_width: 640,
    };
    let raw = vec![
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
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            score: 0.3,
            class_id: 0,
        },
    ];
    let result = postprocess_detections(&raw, &config);
    assert_eq!(result.len(), 1);
    assert!((result[0].score - 0.9).abs() < 1e-5);
}

#[test]
fn postprocess_detections_nms_suppresses_overlapping() {
    let config = ModelDetectorConfig {
        score_threshold: 0.1,
        nms_iou_threshold: 0.3,
        max_detections: 10,
        input_height: 640,
        input_width: 640,
    };
    let raw = vec![
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 10.0,
                y2: 10.0,
            },
            score: 0.9,
            class_id: 0,
        },
        Detection {
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 10.0,
                y2: 10.0,
            },
            score: 0.8,
            class_id: 0,
        },
    ];
    let result = postprocess_detections(&raw, &config);
    // Second detection is fully overlapping, should be suppressed
    assert_eq!(result.len(), 1);
}

#[test]
fn postprocess_detections_respects_max() {
    let config = ModelDetectorConfig {
        score_threshold: 0.0,
        nms_iou_threshold: 1.0, // no NMS
        max_detections: 2,
        input_height: 640,
        input_width: 640,
    };
    let raw: Vec<Detection> = (0..5)
        .map(|i| Detection {
            bbox: BoundingBox {
                x1: i as f32 * 20.0,
                y1: 0.0,
                x2: i as f32 * 20.0 + 10.0,
                y2: 10.0,
            },
            score: 0.5,
            class_id: 0,
        })
        .collect();
    let result = postprocess_detections(&raw, &config);
    assert_eq!(result.len(), 2);
}
