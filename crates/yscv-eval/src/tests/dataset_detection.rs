use super::*;

#[test]
fn detection_dataset_jsonl_parses_and_evaluates() {
    let text = r#"
            # frame 0
            {"ground_truth":[{"bbox":{"x1":0.0,"y1":0.0,"x2":2.0,"y2":2.0},"class_id":0}],"predictions":[{"bbox":{"x1":0.0,"y1":0.0,"x2":2.0,"y2":2.0},"score":0.9,"class_id":0}]}
        "#;
    let frames = parse_detection_dataset_jsonl(text).unwrap();
    assert_eq!(frames.len(), 1);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 1);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_jsonl_rejects_bad_line() {
    let err = parse_detection_dataset_jsonl("{bad json}").unwrap_err();
    match err {
        EvalError::InvalidDatasetEntry { line, message } => {
            assert_eq!(line, 1);
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn load_detection_dataset_jsonl_reports_io_errors() {
    let missing = PathBuf::from("benchmarks/__missing_detection_dataset__.jsonl");
    let err = load_detection_dataset_jsonl_file(&missing).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_coco_parses_and_evaluates() {
    let ground_truth = r#"
        {
          "images": [{"id": 101}, {"id": 202}],
          "annotations": [
            {"image_id": 101, "category_id": 0, "bbox": [0.0, 0.0, 2.0, 2.0]},
            {"image_id": 202, "category_id": 0, "bbox": [4.0, 4.0, 2.0, 2.0]}
          ]
        }
    "#;
    let predictions = r#"
        [
          {"image_id": 101, "category_id": 0, "bbox": [0.0, 0.0, 2.0, 2.0], "score": 0.95},
          {"image_id": 202, "category_id": 0, "bbox": [4.0, 4.0, 2.0, 2.0], "score": 0.80}
        ]
    "#;

    let frames = parse_detection_dataset_coco(ground_truth, predictions).unwrap();
    assert_eq!(frames.len(), 2);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 2);
    assert_eq!(metrics.false_positives, 0);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_coco_rejects_unknown_image_reference() {
    let ground_truth = r#"
        {
          "images": [{"id": 1}],
          "annotations": [{"image_id": 1, "category_id": 0, "bbox": [0.0, 0.0, 1.0, 1.0]}]
        }
    "#;
    let predictions = r#"
        [
          {"image_id": 99, "category_id": 0, "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.9}
        ]
    "#;
    let err = parse_detection_dataset_coco(ground_truth, predictions).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "coco");
            assert!(message.contains("unknown image_id 99"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_coco_rejects_negative_bbox_size() {
    let ground_truth = r#"
        {
          "images": [{"id": 1}],
          "annotations": [{"image_id": 1, "category_id": 0, "bbox": [0.0, 0.0, -1.0, 1.0]}]
        }
    "#;
    let predictions = "[]";
    let err = parse_detection_dataset_coco(ground_truth, predictions).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "coco");
            assert!(message.contains("negative bbox size"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn load_detection_dataset_coco_reports_io_errors() {
    let missing_gt = PathBuf::from("benchmarks/__missing_detection_coco_gt__.json");
    let pred = PathBuf::from("benchmarks/eval-detection-coco-pred-sample.json");
    let err = load_detection_dataset_coco_files(&missing_gt, &pred).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_gt.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_openimages_csv_parses_and_evaluates() {
    let ground_truth = "ImageID,LabelName,XMin,XMax,YMin,YMax\n\
frame0,/m/person,0.0,0.2,0.0,0.2\n\
frame1,/m/person,0.4,0.6,0.4,0.6\n";
    let predictions = "ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n\
frame0,/m/person,0.95,0.0,0.2,0.0,0.2\n\
frame1,/m/person,0.80,0.4,0.6,0.4,0.6\n\
frame1,/m/car,0.70,0.7,0.9,0.7,0.9\n";
    let frames = parse_detection_dataset_openimages_csv(ground_truth, predictions).unwrap();
    assert_eq!(frames.len(), 2);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 2);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_openimages_csv_files_load_and_evaluate() {
    let gt_path = workspace_path("benchmarks/eval-detection-openimages-gt-sample.csv");
    let pred_path = workspace_path("benchmarks/eval-detection-openimages-pred-sample.csv");
    let frames = load_detection_dataset_openimages_csv_files(&gt_path, &pred_path).unwrap();
    assert_eq!(frames.len(), 3);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 3);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_openimages_csv_rejects_invalid_bbox() {
    let ground_truth = "ImageID,LabelName,XMin,XMax,YMin,YMax\nframe0,/m/person,0.2,0.2,0.0,0.2\n";
    let predictions =
        "ImageID,LabelName,Score,XMin,XMax,YMin,YMax\nframe0,/m/person,0.9,0.2,0.4,0.0,0.2\n";
    let err = parse_detection_dataset_openimages_csv(ground_truth, predictions).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "openimages");
            assert!(message.contains("expected XMax>XMin and YMax>YMin"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn load_detection_dataset_openimages_reports_io_errors() {
    let missing_gt = workspace_path("benchmarks/__missing_eval_detection_openimages_gt__.csv");
    let pred_path = workspace_path("benchmarks/eval-detection-openimages-pred-sample.csv");
    let err = load_detection_dataset_openimages_csv_files(&missing_gt, &pred_path).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_gt.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_yolo_label_dirs_load_and_evaluate() {
    let manifest = workspace_path("benchmarks/eval-detection-yolo-manifest-sample.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-yolo-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-yolo-pred");
    let frames = load_detection_dataset_yolo_label_dirs(&manifest, &gt_dir, &pred_dir).unwrap();
    assert_eq!(frames.len(), 3);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 3);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_yolo_label_dirs_rejects_invalid_manifest() {
    let manifest = workspace_path("benchmarks/eval-detection-yolo-manifest-invalid.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-yolo-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-yolo-pred");
    let err = load_detection_dataset_yolo_label_dirs(&manifest, &gt_dir, &pred_dir).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "yolo");
            assert!(message.contains("must be > 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_yolo_label_dirs_reports_manifest_io_error() {
    let missing_manifest =
        workspace_path("benchmarks/__missing_eval_detection_yolo_manifest__.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-yolo-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-yolo-pred");
    let err =
        load_detection_dataset_yolo_label_dirs(&missing_manifest, &gt_dir, &pred_dir).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_manifest.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_voc_xml_dirs_load_and_evaluate() {
    let manifest = workspace_path("benchmarks/eval-detection-voc-manifest-sample.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-voc-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-voc-pred");
    let frames = load_detection_dataset_voc_xml_dirs(&manifest, &gt_dir, &pred_dir).unwrap();
    assert_eq!(frames.len(), 3);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 3);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_voc_xml_dirs_rejects_invalid_bbox() {
    let manifest = workspace_path("benchmarks/eval-detection-voc-manifest-invalid.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-voc-gt-invalid");
    let pred_dir = workspace_path("benchmarks/eval-detection-voc-pred");
    let err = load_detection_dataset_voc_xml_dirs(&manifest, &gt_dir, &pred_dir).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "voc");
            assert!(message.contains("expected xmax>xmin and ymax>ymin"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_voc_xml_dirs_reports_manifest_io_error() {
    let missing_manifest = workspace_path("benchmarks/__missing_eval_detection_voc_manifest__.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-voc-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-voc-pred");
    let err =
        load_detection_dataset_voc_xml_dirs(&missing_manifest, &gt_dir, &pred_dir).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_manifest.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_kitti_label_dirs_load_and_evaluate() {
    let manifest = workspace_path("benchmarks/eval-detection-kitti-manifest-sample.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-kitti-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-kitti-pred");
    let frames = load_detection_dataset_kitti_label_dirs(&manifest, &gt_dir, &pred_dir).unwrap();
    assert_eq!(frames.len(), 3);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 3);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_kitti_label_dirs_rejects_invalid_bbox() {
    let manifest = workspace_path("benchmarks/eval-detection-kitti-manifest-invalid.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-kitti-gt-invalid");
    let pred_dir = workspace_path("benchmarks/eval-detection-kitti-pred");
    let err = load_detection_dataset_kitti_label_dirs(&manifest, &gt_dir, &pred_dir).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "kitti");
            assert!(message.contains("expected bbox_right>bbox_left and bbox_bottom>bbox_top"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_kitti_label_dirs_reports_manifest_io_error() {
    let missing_manifest =
        workspace_path("benchmarks/__missing_eval_detection_kitti_manifest__.txt");
    let gt_dir = workspace_path("benchmarks/eval-detection-kitti-gt");
    let pred_dir = workspace_path("benchmarks/eval-detection-kitti-pred");
    let err =
        load_detection_dataset_kitti_label_dirs(&missing_manifest, &gt_dir, &pred_dir).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_manifest.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_widerface_parses_and_evaluates() {
    let ground_truth = r#"
img_0.jpg
1
10 20 20 20 0 0 0 0 0 0
img_1.jpg
1
50 30 15 10 0 0 0 0 0 0
"#;
    let predictions = r#"
img_0.jpg
2
10 20 20 20 0.95
0 0 5 5 0.30
img_1.jpg
1
50 30 15 10 0.90
"#;
    let frames = parse_detection_dataset_widerface(ground_truth, predictions).unwrap();
    assert_eq!(frames.len(), 2);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 2);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_widerface_file_loader_loads_and_evaluates() {
    let gt_path = workspace_path("benchmarks/eval-detection-widerface-gt-sample.txt");
    let pred_path = workspace_path("benchmarks/eval-detection-widerface-pred-sample.txt");
    let frames = load_detection_dataset_widerface_files(&gt_path, &pred_path).unwrap();
    assert_eq!(frames.len(), 2);
    let metrics =
        evaluate_detections_from_dataset(&frames, DetectionEvalConfig::default()).unwrap();
    assert_eq!(metrics.true_positives, 2);
    assert_eq!(metrics.false_positives, 1);
    assert_eq!(metrics.false_negatives, 0);
}

#[test]
fn detection_dataset_widerface_rejects_unknown_prediction_image() {
    let ground_truth = "img_0.jpg\n0\n";
    let predictions = "img_1.jpg\n1\n1 1 1 1 0.9\n";
    let err = parse_detection_dataset_widerface(ground_truth, predictions).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "widerface");
            assert!(message.contains("unknown image id"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn detection_dataset_widerface_rejects_invalid_bbox_size() {
    let ground_truth = "img_0.jpg\n1\n10 20 0 10 0 0 0 0 0 0\n";
    let predictions = "img_0.jpg\n0\n";
    let err = parse_detection_dataset_widerface(ground_truth, predictions).unwrap_err();
    match err {
        EvalError::InvalidDatasetFormat { format, message } => {
            assert_eq!(format, "widerface");
            assert!(message.contains("`w` and `h` must be > 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn load_detection_dataset_widerface_reports_io_errors() {
    let missing_gt = workspace_path("benchmarks/__missing_eval_detection_widerface_gt__.txt");
    let pred_path = workspace_path("benchmarks/eval-detection-widerface-pred-sample.txt");
    let err = load_detection_dataset_widerface_files(&missing_gt, &pred_path).unwrap_err();
    match err {
        EvalError::DatasetIo { path, message } => {
            assert_eq!(path, missing_gt.display().to_string());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
