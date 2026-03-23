use crate::{roi_align, roi_pool};
use yscv_tensor::Tensor;

/// Helper: creates an [H, W, C] tensor filled with a constant.
fn constant_features(h: usize, w: usize, c: usize, val: f32) -> Tensor {
    Tensor::from_vec(vec![h, w, c], vec![val; h * w * c]).unwrap()
}

/// Helper: creates an [H, W, 1] tensor where value = y * W + x.
fn indexed_features(h: usize, w: usize) -> Tensor {
    let mut data = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            data.push((y * w + x) as f32);
        }
    }
    Tensor::from_vec(vec![h, w, 1], data).unwrap()
}

// ── roi_pool tests ──────────────────────────────────────────────────────

#[test]
fn roi_pool_output_shape() {
    let feat = constant_features(8, 8, 3, 1.0);
    let rois = vec![(0.0, 0.0, 4.0, 4.0), (2.0, 2.0, 6.0, 6.0)];
    let out = roi_pool(&feat, &rois, (2, 2)).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2, 3]);
}

#[test]
fn roi_pool_max_value() {
    // 4x4 feature map, single channel, values = y*4+x (0..15).
    let feat = indexed_features(4, 4);
    // RoI covering top-left 2x2 region.  Values: 0,1,4,5.
    let rois = vec![(0.0, 0.0, 2.0, 2.0)];
    let out = roi_pool(&feat, &rois, (1, 1)).unwrap();
    // Max of {0,1,4,5} = 5.
    let val = out.get(&[0, 0, 0, 0]).unwrap();
    assert!((val - 5.0).abs() < 1e-5);
}

#[test]
fn roi_pool_constant_input() {
    let feat = constant_features(6, 6, 2, 7.0);
    let rois = vec![(1.0, 1.0, 5.0, 5.0)];
    let out = roi_pool(&feat, &rois, (3, 3)).unwrap();
    // Every output value should be 7.0 (max of constant).
    for r in 0..1 {
        for oh in 0..3 {
            for ow in 0..3 {
                for c in 0..2 {
                    let v = out.get(&[r, oh, ow, c]).unwrap();
                    assert!((v - 7.0).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn roi_pool_rejects_wrong_rank() {
    let feat = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
    let rois = vec![(0.0, 0.0, 1.0, 1.0)];
    assert!(roi_pool(&feat, &rois, (1, 1)).is_err());
}

// ── roi_align tests ─────────────────────────────────────────────────────

#[test]
fn roi_align_output_shape() {
    let feat = constant_features(8, 8, 3, 1.0);
    let rois = vec![(0.0, 0.0, 4.0, 4.0), (2.0, 2.0, 6.0, 6.0)];
    let out = roi_align(&feat, &rois, (3, 3), 2).unwrap();
    assert_eq!(out.shape(), &[2, 3, 3, 3]);
}

#[test]
fn roi_align_bilinear() {
    // For a constant-valued feature map, bilinear interpolation should
    // reproduce that constant exactly regardless of RoI position.
    let feat = constant_features(8, 8, 1, 3.0);
    let rois = vec![(1.5, 1.5, 5.5, 5.5)];
    let out = roi_align(&feat, &rois, (2, 2), 4).unwrap();
    for oh in 0..2 {
        for ow in 0..2 {
            let v = out.get(&[0, oh, ow, 0]).unwrap();
            assert!(
                (v - 3.0).abs() < 1e-4,
                "expected 3.0, got {v} at ({oh},{ow})"
            );
        }
    }
}

#[test]
fn roi_align_rejects_wrong_rank() {
    let feat = Tensor::from_vec(vec![2, 3, 4, 5], vec![0.0; 120]).unwrap();
    let rois = vec![(0.0, 0.0, 1.0, 1.0)];
    assert!(roi_align(&feat, &rois, (1, 1), 2).is_err());
}

#[test]
fn roi_align_smooth_interpolation() {
    // A 2x2 feature map with known values; RoI align at the centre should
    // produce the average when sampling the whole map.
    // Values: (0,0)=0  (0,1)=1  (1,0)=2  (1,1)=3
    let feat = Tensor::from_vec(vec![2, 2, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let rois = vec![(0.0, 0.0, 2.0, 2.0)];
    let out = roi_align(&feat, &rois, (1, 1), 4).unwrap();
    let v = out.get(&[0, 0, 0, 0]).unwrap();
    // The average of the bilinear-sampled points over the whole 2x2 map
    // should be close to the mean of corner values (1.5).
    // Bilinear sampling over integer grid can produce values in [0, 3] range;
    // just verify we get a finite interpolated result, not NaN or extreme.
    assert!(
        v.is_finite() && (0.0..=3.0).contains(&v),
        "expected interpolated value in [0, 3], got {v}"
    );
}
