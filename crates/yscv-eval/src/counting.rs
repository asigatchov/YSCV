use crate::EvalError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CountingMetrics {
    pub num_frames: usize,
    pub mae: f32,
    pub rmse: f32,
    pub max_abs_error: usize,
}

pub fn evaluate_counts(
    ground_truth: &[usize],
    predictions: &[usize],
) -> Result<CountingMetrics, EvalError> {
    if ground_truth.len() != predictions.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: ground_truth.len(),
            predictions: predictions.len(),
        });
    }

    if ground_truth.is_empty() {
        return Ok(CountingMetrics {
            num_frames: 0,
            mae: 0.0,
            rmse: 0.0,
            max_abs_error: 0,
        });
    }

    let mut abs_error_sum = 0.0f32;
    let mut sq_error_sum = 0.0f32;
    let mut max_abs_error = 0usize;
    for (&gt, &prediction) in ground_truth.iter().zip(predictions.iter()) {
        let error = prediction as i64 - gt as i64;
        let abs_error = error.unsigned_abs() as usize;
        abs_error_sum += abs_error as f32;
        sq_error_sum += (error as f32).powi(2);
        max_abs_error = max_abs_error.max(abs_error);
    }

    let denom = ground_truth.len() as f32;
    Ok(CountingMetrics {
        num_frames: ground_truth.len(),
        mae: abs_error_sum / denom,
        rmse: (sq_error_sum / denom).sqrt(),
        max_abs_error,
    })
}
