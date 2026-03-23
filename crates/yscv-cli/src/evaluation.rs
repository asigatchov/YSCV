use yscv_eval::{
    CameraDiagnosticsThresholds, DetectionEvalConfig, TrackingEvalConfig,
    evaluate_detections_from_dataset, evaluate_tracking_from_dataset,
    load_camera_diagnostics_report_json_file, load_detection_dataset_coco_files,
    load_detection_dataset_jsonl_file, load_detection_dataset_kitti_label_dirs,
    load_detection_dataset_openimages_csv_files, load_detection_dataset_voc_xml_dirs,
    load_detection_dataset_widerface_files, load_detection_dataset_yolo_label_dirs,
    load_tracking_dataset_jsonl_file, load_tracking_dataset_mot_txt_files,
    validate_camera_diagnostics_report,
};

use crate::config::{CliConfig, CliError};
use crate::error::AppError;

pub fn run_dataset_evaluation(cli: &CliConfig) -> Result<(), AppError> {
    println!("yscv-cli eval: starting dataset evaluation");
    if let Some(path) = cli.eval_detection_dataset_path.as_deref() {
        let frames = load_detection_dataset_jsonl_file(path)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval dataset={} frames={}",
            path.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let (Some(gt_path), Some(pred_path)) = (
        cli.eval_detection_coco_gt_path.as_deref(),
        cli.eval_detection_coco_pred_path.as_deref(),
    ) {
        let frames = load_detection_dataset_coco_files(gt_path, pred_path)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval_coco ground_truth={} predictions={} frames={}",
            gt_path.display(),
            pred_path.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let (Some(gt_path), Some(pred_path)) = (
        cli.eval_detection_openimages_gt_path.as_deref(),
        cli.eval_detection_openimages_pred_path.as_deref(),
    ) {
        let frames = load_detection_dataset_openimages_csv_files(gt_path, pred_path)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval_openimages ground_truth={} predictions={} frames={}",
            gt_path.display(),
            pred_path.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let (Some(manifest_path), Some(gt_dir), Some(pred_dir)) = (
        cli.eval_detection_yolo_manifest_path.as_deref(),
        cli.eval_detection_yolo_gt_dir_path.as_deref(),
        cli.eval_detection_yolo_pred_dir_path.as_deref(),
    ) {
        let frames = load_detection_dataset_yolo_label_dirs(manifest_path, gt_dir, pred_dir)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval_yolo manifest={} gt_dir={} pred_dir={} frames={}",
            manifest_path.display(),
            gt_dir.display(),
            pred_dir.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let (Some(manifest_path), Some(gt_dir), Some(pred_dir)) = (
        cli.eval_detection_voc_manifest_path.as_deref(),
        cli.eval_detection_voc_gt_dir_path.as_deref(),
        cli.eval_detection_voc_pred_dir_path.as_deref(),
    ) {
        let frames = load_detection_dataset_voc_xml_dirs(manifest_path, gt_dir, pred_dir)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval_voc manifest={} gt_dir={} pred_dir={} frames={}",
            manifest_path.display(),
            gt_dir.display(),
            pred_dir.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let (Some(manifest_path), Some(gt_dir), Some(pred_dir)) = (
        cli.eval_detection_kitti_manifest_path.as_deref(),
        cli.eval_detection_kitti_gt_dir_path.as_deref(),
        cli.eval_detection_kitti_pred_dir_path.as_deref(),
    ) {
        let frames = load_detection_dataset_kitti_label_dirs(manifest_path, gt_dir, pred_dir)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval_kitti manifest={} gt_dir={} pred_dir={} frames={}",
            manifest_path.display(),
            gt_dir.display(),
            pred_dir.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let (Some(gt_path), Some(pred_path)) = (
        cli.eval_detection_widerface_gt_path.as_deref(),
        cli.eval_detection_widerface_pred_path.as_deref(),
    ) {
        let frames = load_detection_dataset_widerface_files(gt_path, pred_path)?;
        let metrics = evaluate_detections_from_dataset(
            &frames,
            DetectionEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
                score_threshold: cli.eval_score_threshold,
            },
        )?;
        println!(
            "detection_eval_widerface ground_truth={} predictions={} frames={}",
            gt_path.display(),
            pred_path.display(),
            frames.len()
        );
        println!(
            "  tp={} fp={} fn={} precision={:.4} recall={:.4} f1={:.4} ap={:.4}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
        );
    }

    if let Some(path) = cli.eval_tracking_dataset_path.as_deref() {
        let frames = load_tracking_dataset_jsonl_file(path)?;
        let metrics = evaluate_tracking_from_dataset(
            &frames,
            TrackingEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
            },
        )?;
        println!(
            "tracking_eval dataset={} frames={}",
            path.display(),
            frames.len()
        );
        println!(
            "  gt={} matches={} fp={} fn={} idsw={} precision={:.4} recall={:.4} f1={:.4} mota={:.4} motp={:.4}",
            metrics.total_ground_truth,
            metrics.matches,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.id_switches,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.mota,
            metrics.motp,
        );
    }

    if let (Some(gt_path), Some(pred_path)) = (
        cli.eval_tracking_mot_gt_path.as_deref(),
        cli.eval_tracking_mot_pred_path.as_deref(),
    ) {
        let frames = load_tracking_dataset_mot_txt_files(gt_path, pred_path)?;
        let metrics = evaluate_tracking_from_dataset(
            &frames,
            TrackingEvalConfig {
                iou_threshold: cli.eval_iou_threshold,
            },
        )?;
        println!(
            "tracking_eval_mot ground_truth={} predictions={} frames={}",
            gt_path.display(),
            pred_path.display(),
            frames.len()
        );
        println!(
            "  gt={} matches={} fp={} fn={} idsw={} precision={:.4} recall={:.4} f1={:.4} mota={:.4} motp={:.4}",
            metrics.total_ground_truth,
            metrics.matches,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.id_switches,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.mota,
            metrics.motp,
        );
    }
    println!("yscv-cli eval: completed");
    Ok(())
}

pub fn run_diagnostics_report_validation(cli: &CliConfig) -> Result<(), AppError> {
    let Some(path) = cli.validate_diagnostics_report_path.as_deref() else {
        return Err(CliError::Message(
            "missing diagnostics report path for validation mode".to_string(),
        )
        .into());
    };
    println!(
        "yscv-cli eval: validating diagnostics report {}",
        path.display()
    );
    let report = load_camera_diagnostics_report_json_file(path)?;
    let thresholds = CameraDiagnosticsThresholds {
        min_collected_frames: cli.validate_diagnostics_min_frames,
        max_abs_wall_drift_pct: cli.validate_diagnostics_max_drift_pct,
        max_abs_sensor_drift_pct: cli.validate_diagnostics_max_drift_pct,
        max_dropped_frames: cli.validate_diagnostics_max_dropped_frames,
    };
    let violations = validate_camera_diagnostics_report(&report, thresholds);
    if violations.is_empty() {
        println!(
            "diagnostics_report_validation passed: status={} collected_frames={} max_abs_drift_pct={} max_dropped_frames={}",
            report.status,
            report
                .capture
                .as_ref()
                .map(|capture| capture.collected_frames)
                .unwrap_or(0),
            cli.validate_diagnostics_max_drift_pct,
            cli.validate_diagnostics_max_dropped_frames,
        );
        return Ok(());
    }

    println!("diagnostics_report_validation failed:");
    for violation in &violations {
        println!("  - {}: {}", violation.field, violation.message);
    }
    Err(CliError::Message("diagnostics report validation failed".to_string()).into())
}
