use super::*;

#[test]
fn tracking_dataset_jsonl_parses_and_evaluates() {
    let text = r#"
            {"ground_truth":[{"object_id":1,"bbox":{"x1":0.0,"y1":0.0,"x2":2.0,"y2":2.0},"class_id":0}],"predictions":[{"track_id":17,"detection":{"bbox":{"x1":0.0,"y1":0.0,"x2":2.0,"y2":2.0},"score":0.8,"class_id":0}}]}
        "#;
    let frames = parse_tracking_dataset_jsonl(text).unwrap();
    assert_eq!(frames.len(), 1);
    let metrics = evaluate_tracking_from_dataset(&frames, TrackingEvalConfig::default()).unwrap();
    assert_eq!(metrics.matches, 1);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn tracking_dataset_jsonl_rejects_bad_line() {
    let err = parse_tracking_dataset_jsonl("{bad json}").unwrap_err();
    match err {
        EvalError::InvalidDatasetEntry { line, message } => {
            assert_eq!(line, 1);
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn tracking_dataset_mot_parses_and_evaluates() {
    let ground_truth = r#"
        # frame, id, left, top, width, height, conf, class, visibility
        1,1,0.0,0.0,2.0,2.0,1,0,1
        2,1,0.2,0.1,2.0,2.0,1,0,1
    "#;
    let predictions = r#"
        # frame, id, left, top, width, height, score
        1,11,0.0,0.0,2.0,2.0,0.9
        2,11,0.2,0.1,2.0,2.0,0.88
    "#;
    let frames = parse_tracking_dataset_mot(ground_truth, predictions).unwrap();
    assert_eq!(frames.len(), 2);
    let metrics = evaluate_tracking_from_dataset(&frames, TrackingEvalConfig::default()).unwrap();
    assert_eq!(metrics.total_ground_truth, 2);
    assert_eq!(metrics.matches, 2);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
    assert_eq!(metrics.id_switches, 0);
}

#[test]
fn tracking_dataset_mot_txt_files_load_and_evaluate() {
    let gt_path = workspace_path("benchmarks/eval-tracking-mot-gt-sample.txt");
    let pred_path = workspace_path("benchmarks/eval-tracking-mot-pred-sample.txt");
    let frames = load_tracking_dataset_mot_txt_files(&gt_path, &pred_path).unwrap();
    assert_eq!(frames.len(), 2);
    let metrics = evaluate_tracking_from_dataset(&frames, TrackingEvalConfig::default()).unwrap();
    assert_eq!(metrics.total_ground_truth, 2);
    assert_eq!(metrics.matches, 2);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
    assert_eq!(metrics.id_switches, 0);
}

#[test]
fn tracking_dataset_mot_ignores_zero_conf_ground_truth() {
    let ground_truth = r#"
        1,1,0.0,0.0,2.0,2.0,1,1,1
        1,2,5.0,5.0,1.0,1.0,0,1,1
    "#;
    let predictions = r#"
        1,11,0.0,0.0,2.0,2.0,0.9
    "#;
    let frames = parse_tracking_dataset_mot(ground_truth, predictions).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].ground_truth.len(), 1);
    assert_eq!(frames[0].predictions.len(), 1);
}

#[test]
fn tracking_dataset_mot_rejects_invalid_bbox_size() {
    let ground_truth = "1,1,0.0,0.0,0.0,2.0,1,1,1";
    let predictions = "1,11,0.0,0.0,2.0,2.0,0.9";
    let err = parse_tracking_dataset_mot(ground_truth, predictions).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "mot");
            assert!(message.contains("bbox_width and bbox_height must be > 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn load_tracking_dataset_mot_reports_io_errors() {
    let missing_gt = workspace_path("benchmarks/__missing_eval_tracking_mot_gt__.txt");
    let pred = workspace_path("benchmarks/eval-tracking-mot-pred-sample.txt");
    let err = load_tracking_dataset_mot_txt_files(&missing_gt, &pred).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_gt.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
