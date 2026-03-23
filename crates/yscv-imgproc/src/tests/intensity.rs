use yscv_tensor::Tensor;

use super::super::{adjust_gamma, adjust_log, rescale_intensity};

// ── adjust_gamma ───────────────────────────────────────────────────

#[test]
fn gamma_one_is_identity() {
    let data = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.3, 0.6];
    let img = Tensor::from_vec(vec![3, 3, 1], data.clone()).unwrap();
    let result = adjust_gamma(&img, 1.0).unwrap();
    assert_eq!(result.shape(), &[3, 3, 1]);
    for (a, b) in result.data().iter().zip(data.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "gamma=1 should be identity, got {} vs {}",
            a,
            b,
        );
    }
}

#[test]
fn gamma_two_squares_values() {
    let img = Tensor::from_vec(vec![1, 3, 1], vec![0.0, 0.5, 1.0]).unwrap();
    let result = adjust_gamma(&img, 2.0).unwrap();
    let d = result.data();
    assert!((d[0] - 0.0).abs() < 1e-6);
    assert!((d[1] - 0.25).abs() < 1e-6);
    assert!((d[2] - 1.0).abs() < 1e-6);
}

// ── rescale_intensity ──────────────────────────────────────────────

#[test]
fn rescale_doubles_values() {
    let img = Tensor::from_vec(vec![1, 4, 1], vec![0.0, 0.1, 0.25, 0.5]).unwrap();
    let result = rescale_intensity(&img, 0.0, 0.5, 0.0, 1.0).unwrap();
    let d = result.data();
    assert!((d[0] - 0.0).abs() < 1e-6, "expected 0.0, got {}", d[0]);
    assert!((d[1] - 0.2).abs() < 1e-6, "expected 0.2, got {}", d[1]);
    assert!((d[2] - 0.5).abs() < 1e-6, "expected 0.5, got {}", d[2]);
    assert!((d[3] - 1.0).abs() < 1e-6, "expected 1.0, got {}", d[3]);
}

#[test]
fn rescale_clamps_outside_range() {
    let img = Tensor::from_vec(vec![1, 3, 1], vec![0.0, 0.5, 1.0]).unwrap();
    let result = rescale_intensity(&img, 0.25, 0.75, 0.0, 1.0).unwrap();
    let d = result.data();
    // 0.0 maps to (0.0 - 0.25) / 0.5 = -0.5, clamped to 0.0
    assert!((d[0] - 0.0).abs() < 1e-6);
    // 0.5 maps to (0.5 - 0.25) / 0.5 = 0.5
    assert!((d[1] - 0.5).abs() < 1e-6);
    // 1.0 maps to (1.0 - 0.25) / 0.5 = 1.5, clamped to 1.0
    assert!((d[2] - 1.0).abs() < 1e-6);
}

// ── adjust_log ─────────────────────────────────────────────────────

#[test]
fn log_boosts_low_values_more() {
    let img = Tensor::from_vec(vec![1, 3, 1], vec![0.1, 0.5, 1.0]).unwrap();
    let result = adjust_log(&img).unwrap();
    let d = result.data();

    // After log transform, low values should be boosted relative to high values
    // compared to linear scaling.
    // Ratio of low/high should be larger after log transform
    let original_ratio = 0.1 / 1.0;
    let log_ratio = d[0] / d[2];
    assert!(
        log_ratio > original_ratio,
        "log should boost low values more: log_ratio={}, original_ratio={}",
        log_ratio,
        original_ratio,
    );
    // Max should be 1.0 (normalized)
    assert!(
        (d[2] - 1.0).abs() < 1e-6,
        "max should be 1.0 after normalization"
    );
}

#[test]
fn log_zero_input_gives_zero() {
    let img = Tensor::from_vec(vec![1, 2, 1], vec![0.0, 1.0]).unwrap();
    let result = adjust_log(&img).unwrap();
    let d = result.data();
    assert!((d[0] - 0.0).abs() < 1e-6, "log(1+0) = 0");
    assert!((d[1] - 1.0).abs() < 1e-6, "max should normalise to 1.0");
}
