use super::*;

#[test]
fn counting_computes_expected_errors() {
    let metrics = evaluate_counts(&[1, 2, 0], &[1, 0, 2]).unwrap();
    assert_eq!(
        metrics,
        CountingMetrics {
            num_frames: 3,
            mae: 4.0 / 3.0,
            rmse: (8.0f32 / 3.0).sqrt(),
            max_abs_error: 2,
        }
    );
}

#[test]
fn counting_rejects_length_mismatch() {
    let err = evaluate_counts(&[1, 2], &[1]).unwrap_err();
    assert_eq!(
        err,
        EvalError::CountLengthMismatch {
            ground_truth: 2,
            predictions: 1
        }
    );
}

#[test]
fn counting_handles_empty_series() {
    let metrics = evaluate_counts(&[], &[]).unwrap();
    assert_eq!(
        metrics,
        CountingMetrics {
            num_frames: 0,
            mae: 0.0,
            rmse: 0.0,
            max_abs_error: 0,
        }
    );
}
