use crate::generate_anchors;

#[test]
fn anchor_count_correct() {
    let h = 4;
    let w = 5;
    let sizes = [32.0, 64.0, 128.0];
    let ratios = [0.5, 1.0, 2.0];
    let anchors = generate_anchors(h, w, &sizes, &ratios, 16.0);
    assert_eq!(anchors.len(), h * w * sizes.len() * ratios.len());
}

#[test]
fn anchor_centers_on_grid() {
    let stride = 16.0;
    let anchors = generate_anchors(2, 3, &[32.0], &[1.0], stride);
    // First anchor should be centred at (0.5 * stride, 0.5 * stride).
    let first = &anchors[0];
    let cx = (first.x1 + first.x2) / 2.0;
    let cy = (first.y1 + first.y2) / 2.0;
    assert!((cx - 0.5 * stride).abs() < 1e-5);
    assert!((cy - 0.5 * stride).abs() < 1e-5);
}

#[test]
fn anchor_dimensions_match_size_and_ratio() {
    let anchors = generate_anchors(1, 1, &[64.0], &[4.0], 16.0);
    assert_eq!(anchors.len(), 1);
    let a = &anchors[0];
    let w = a.x2 - a.x1;
    let h = a.y2 - a.y1;
    // w = size * sqrt(ratio) = 64 * 2 = 128
    // h = size / sqrt(ratio) = 64 / 2 = 32
    assert!((w - 128.0).abs() < 1e-4);
    assert!((h - 32.0).abs() < 1e-4);
}

#[test]
fn anchor_empty_feature_map() {
    let anchors = generate_anchors(0, 5, &[32.0], &[1.0], 16.0);
    assert!(anchors.is_empty());
}
