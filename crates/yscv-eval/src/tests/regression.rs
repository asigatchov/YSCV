use crate::{EvalError, mae, mape, r2_score, rmse};

use super::approx_eq;

#[test]
fn r2_perfect_predictions() {
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    approx_eq(r2_score(&vals, &vals).unwrap(), 1.0);
}

#[test]
fn r2_mean_predictions() {
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = 3.0;
    let preds = vec![mean; 5];
    approx_eq(r2_score(&preds, &targets).unwrap(), 0.0);
}

#[test]
fn mae_perfect() {
    let vals = vec![1.0, 2.0, 3.0];
    approx_eq(mae(&vals, &vals).unwrap(), 0.0);
}

#[test]
fn mae_simple() {
    let preds = vec![1.0, 2.0, 3.0];
    let targets = vec![2.0, 3.0, 4.0];
    approx_eq(mae(&preds, &targets).unwrap(), 1.0);
}

#[test]
fn rmse_perfect() {
    let vals = vec![1.0, 2.0, 3.0];
    approx_eq(rmse(&vals, &vals).unwrap(), 0.0);
}

#[test]
fn rmse_simple() {
    // preds=[1,2,3], targets=[2,4,6] → diffs=[1,2,3] → mse=(1+4+9)/3=14/3 → rmse=sqrt(14/3)
    let preds = vec![1.0, 2.0, 3.0];
    let targets = vec![2.0, 4.0, 6.0];
    let expected = (14.0_f32 / 3.0).sqrt();
    approx_eq(rmse(&preds, &targets).unwrap(), expected);
}

#[test]
fn mape_perfect() {
    let vals = vec![1.0, 2.0, 3.0];
    approx_eq(mape(&vals, &vals).unwrap(), 0.0);
}

#[test]
fn regression_length_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    let expected = EvalError::CountLengthMismatch {
        ground_truth: 1,
        predictions: 2,
    };
    assert_eq!(r2_score(&a, &b).unwrap_err(), expected);
    assert_eq!(mae(&a, &b).unwrap_err(), expected);
    assert_eq!(rmse(&a, &b).unwrap_err(), expected);
    assert_eq!(mape(&a, &b).unwrap_err(), expected);
}
