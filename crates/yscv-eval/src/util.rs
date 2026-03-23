use crate::EvalError;

pub(crate) fn safe_ratio(numerator: u64, denominator: u64) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

pub(crate) fn harmonic_mean(a: f32, b: f32) -> f32 {
    let denom = a + b;
    if denom == 0.0 {
        0.0
    } else {
        2.0 * a * b / denom
    }
}

pub(crate) fn validate_iou_threshold(value: f32) -> Result<(), EvalError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EvalError::InvalidIouThreshold { value });
    }
    Ok(())
}

pub(crate) fn validate_score_threshold(value: f32) -> Result<(), EvalError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EvalError::InvalidScoreThreshold { value });
    }
    Ok(())
}
