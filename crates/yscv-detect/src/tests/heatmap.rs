use crate::{
    DetectError, HeatmapDetectScratch, detect_from_heatmap, detect_from_heatmap_with_scratch,
};
use yscv_tensor::Tensor;

#[test]
fn detect_from_heatmap_finds_connected_component_boxes() {
    let heatmap = Tensor::from_vec(
        vec![4, 4, 1],
        vec![
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.8, 0.9, 0.0, //
            0.0, 0.7, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let dets = detect_from_heatmap(&heatmap, 0.5, 2, 0.5, 10).unwrap();
    assert_eq!(dets.len(), 1);
    let d = dets[0];
    assert_eq!(d.bbox.x1, 1.0);
    assert_eq!(d.bbox.y1, 1.0);
    assert_eq!(d.bbox.x2, 3.0);
    assert_eq!(d.bbox.y2, 3.0);
}

#[test]
fn detect_from_heatmap_ignores_non_finite_cells() {
    let heatmap = Tensor::from_vec(vec![2, 2, 1], vec![f32::NAN, 0.9, f32::INFINITY, 0.0]).unwrap();

    let dets = detect_from_heatmap(&heatmap, 0.5, 1, 0.5, 10).unwrap();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].bbox.x1, 1.0);
    assert_eq!(dets[0].bbox.y1, 0.0);
    assert_eq!(dets[0].bbox.x2, 2.0);
    assert_eq!(dets[0].bbox.y2, 1.0);
}

#[test]
fn detect_from_heatmap_finds_multiple_components() {
    let heatmap = Tensor::from_vec(
        vec![4, 5, 1],
        vec![
            0.0, 0.0, 0.0, 0.9, 0.9, //
            0.0, 0.8, 0.8, 0.0, 0.0, //
            0.0, 0.8, 0.8, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
        ],
    )
    .unwrap();

    let mut dets = detect_from_heatmap(&heatmap, 0.5, 2, 0.5, 10).unwrap();
    dets.sort_by(|left, right| left.bbox.x1.total_cmp(&right.bbox.x1));
    assert_eq!(dets.len(), 2);
    assert_eq!(dets[0].bbox.x1, 1.0);
    assert_eq!(dets[0].bbox.y1, 1.0);
    assert_eq!(dets[0].bbox.x2, 3.0);
    assert_eq!(dets[0].bbox.y2, 3.0);
    assert_eq!(dets[1].bbox.x1, 3.0);
    assert_eq!(dets[1].bbox.y1, 0.0);
    assert_eq!(dets[1].bbox.x2, 5.0);
    assert_eq!(dets[1].bbox.y2, 1.0);
}

#[test]
fn detect_from_heatmap_with_scratch_finds_connected_component_boxes() {
    let heatmap = Tensor::from_vec(
        vec![4, 4, 1],
        vec![
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.8, 0.9, 0.0, //
            0.0, 0.7, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let mut scratch = HeatmapDetectScratch::default();
    let dets = detect_from_heatmap_with_scratch(&heatmap, 0.5, 2, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(dets.len(), 1);
    let d = dets[0];
    assert_eq!(d.bbox.x1, 1.0);
    assert_eq!(d.bbox.y1, 1.0);
    assert_eq!(d.bbox.x2, 3.0);
    assert_eq!(d.bbox.y2, 3.0);
}

#[test]
fn detect_from_heatmap_with_scratch_reuses_buffer_for_resized_maps() {
    let mut scratch = HeatmapDetectScratch::default();

    let heatmap_small =
        Tensor::from_vec(vec![2, 2, 1], vec![0.8, 0.0, 0.8, 0.0]).expect("valid heatmap");
    let dets_small =
        detect_from_heatmap_with_scratch(&heatmap_small, 0.5, 1, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(dets_small.len(), 1);

    let heatmap_large = Tensor::from_vec(
        vec![4, 5, 1],
        vec![
            0.0, 0.0, 0.0, 0.9, 0.9, //
            0.0, 0.8, 0.8, 0.0, 0.0, //
            0.0, 0.8, 0.8, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
        ],
    )
    .expect("valid heatmap");
    let dets_large =
        detect_from_heatmap_with_scratch(&heatmap_large, 0.5, 2, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(dets_large.len(), 2);
}

#[test]
fn detect_from_heatmap_rejects_invalid_threshold() {
    let heatmap = Tensor::zeros(vec![2, 2, 1]).unwrap();
    let err = detect_from_heatmap(&heatmap, 1.2, 1, 0.5, 10).unwrap_err();
    assert_eq!(err, DetectError::InvalidThreshold { threshold: 1.2 });
}

#[test]
fn detect_from_heatmap_rejects_invalid_iou_threshold() {
    let heatmap = Tensor::zeros(vec![2, 2, 1]).unwrap();
    let err = detect_from_heatmap(&heatmap, 0.5, 1, 1.2, 10).unwrap_err();
    assert_eq!(err, DetectError::InvalidIouThreshold { iou_threshold: 1.2 });
}

#[test]
fn detect_from_heatmap_rejects_zero_max_detections() {
    let heatmap = Tensor::zeros(vec![2, 2, 1]).unwrap();
    let err = detect_from_heatmap(&heatmap, 0.5, 1, 0.5, 0).unwrap_err();
    assert_eq!(err, DetectError::InvalidMaxDetections { max_detections: 0 });
}
