use crate::{LrFinderConfig, lr_range_test};

#[test]
fn test_lr_finder_generates_correct_steps() {
    let config = LrFinderConfig {
        num_steps: 50,
        ..Default::default()
    };
    let result = lr_range_test(&config, |_lr| 1.0);
    assert_eq!(result.lr_values.len(), 50);
    assert_eq!(result.loss_values.len(), 50);
}

#[test]
fn test_lr_finder_log_scale() {
    let config = LrFinderConfig {
        start_lr: 1e-4,
        end_lr: 1.0,
        num_steps: 5,
        log_scale: true,
        smoothing: 1.0, // no smoothing (use raw)
    };
    let result = lr_range_test(&config, |_lr| 1.0);
    let lrs = &result.lr_values;
    assert!((lrs[0] - 1e-4).abs() < 1e-6);
    assert!((lrs[4] - 1.0).abs() < 1e-4);
    // Log-spaced: ratios between consecutive values should be roughly equal
    let r1 = lrs[1] / lrs[0];
    let r2 = lrs[2] / lrs[1];
    assert!((r1 - r2).abs() / r1 < 0.01);
}

#[test]
fn test_lr_finder_linear_scale() {
    let config = LrFinderConfig {
        start_lr: 0.0,
        end_lr: 1.0,
        num_steps: 5,
        log_scale: false,
        smoothing: 1.0,
    };
    let result = lr_range_test(&config, |_lr| 1.0);
    let lrs = &result.lr_values;
    assert!((lrs[0] - 0.0).abs() < 1e-6);
    assert!((lrs[1] - 0.25).abs() < 1e-6);
    assert!((lrs[2] - 0.5).abs() < 1e-6);
    assert!((lrs[3] - 0.75).abs() < 1e-6);
    assert!((lrs[4] - 1.0).abs() < 1e-6);
}

#[test]
fn test_lr_finder_suggested_lr() {
    // Simulate a convex loss curve: loss = (lr - 0.5)^2 + 0.1
    // Loss decreases most steeply just before lr=0.5
    let config = LrFinderConfig {
        start_lr: 0.0,
        end_lr: 1.0,
        num_steps: 101,
        log_scale: false,
        smoothing: 1.0, // use raw values so derivative is clean
    };
    let result = lr_range_test(&config, |lr| (lr - 0.5) * (lr - 0.5) + 0.1);
    // Suggested LR should be in the region where loss is decreasing steeply,
    // i.e., lr < 0.5 and close to the start (steepest descent at lr=0.0).
    // For a parabola the derivative is 2*(lr-0.5), most negative at lr=0.
    // With linear steps the largest drop is at step 1 (lr=0.01).
    assert!(result.suggested_lr < 0.5);
}

#[test]
fn test_lr_finder_default_config() {
    let config = LrFinderConfig::default();
    assert!((config.start_lr - 1e-7).abs() < 1e-10);
    assert!((config.end_lr - 10.0).abs() < 1e-6);
    assert_eq!(config.num_steps, 100);
    assert!(config.log_scale);
    assert!((config.smoothing - 0.05).abs() < 1e-6);
}
