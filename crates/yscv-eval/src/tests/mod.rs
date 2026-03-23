mod camera_diagnostics;
mod classification;
mod counting;
mod dataset_detection;
mod dataset_tracking;
mod detection;
mod pipeline;
mod regression;
mod timing;
mod tracking;

use std::path::PathBuf;
use std::time::Duration;

use crate::{
    CameraDiagnosticsThresholds, CountingMetrics, DetectionEvalConfig, DetectionFrame, EvalError,
    GroundTruthTrack, LabeledBox, PipelineBenchmarkReport, PipelineBenchmarkThresholds,
    PipelineDurations, StageThresholds, TimingStats, TrackingEvalConfig, TrackingFrame,
    evaluate_counts, evaluate_detections, evaluate_detections_coco,
    evaluate_detections_from_dataset, evaluate_tracking, evaluate_tracking_from_dataset, hota,
    idf1, load_detection_dataset_coco_files, load_detection_dataset_jsonl_file,
    load_detection_dataset_kitti_label_dirs, load_detection_dataset_openimages_csv_files,
    load_detection_dataset_voc_xml_dirs, load_detection_dataset_widerface_files,
    load_detection_dataset_yolo_label_dirs, load_tracking_dataset_mot_txt_files,
    parse_camera_diagnostics_report_json, parse_detection_dataset_coco,
    parse_detection_dataset_jsonl, parse_detection_dataset_openimages_csv,
    parse_detection_dataset_widerface, parse_pipeline_benchmark_thresholds,
    parse_tracking_dataset_jsonl, parse_tracking_dataset_mot, summarize_durations,
    summarize_pipeline_durations, validate_camera_diagnostics_report,
    validate_pipeline_benchmark_thresholds,
};
use yscv_detect::{BoundingBox, Detection};
use yscv_track::TrackedDetection;

fn approx_eq(left: f32, right: f32) {
    assert!((left - right).abs() < 1e-5, "left={left}, right={right}");
}

fn approx_eq64(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
}

fn bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BoundingBox {
    BoundingBox { x1, y1, x2, y2 }
}

fn workspace_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn gt_box(x1: f32, y1: f32, x2: f32, y2: f32) -> LabeledBox {
    LabeledBox {
        bbox: bbox(x1, y1, x2, y2),
        class_id: 0,
    }
}

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Detection {
    Detection {
        bbox: bbox(x1, y1, x2, y2),
        score,
        class_id: 0,
    }
}

fn gt_track(id: u64, x1: f32, y1: f32, x2: f32, y2: f32) -> GroundTruthTrack {
    GroundTruthTrack {
        object_id: id,
        bbox: bbox(x1, y1, x2, y2),
        class_id: 0,
    }
}

fn tracked(track_id: u64, x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> TrackedDetection {
    TrackedDetection {
        track_id,
        detection: det(x1, y1, x2, y2, score),
    }
}
