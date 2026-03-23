use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{ModelError, hinge_loss, huber_loss, kl_div_loss, mae_loss, mse_loss, smooth_l1_loss};

#[test]
fn mse_loss_computes_scalar_mean() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![2.0, 4.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());

    let loss = mse_loss(&mut graph, pred, target).unwrap();
    let value = graph.value(loss).unwrap();
    assert!(value.shape().is_empty());
    assert_eq!(value.data(), &[5.0]);
}

#[test]
fn mae_loss_computes_scalar_mean() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![2.0, 4.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());

    let loss = mae_loss(&mut graph, pred, target).unwrap();
    let value = graph.value(loss).unwrap();
    assert!(value.shape().is_empty());
    assert_eq!(value.data(), &[2.0]);
}

#[test]
fn huber_loss_computes_scalar_mean() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![0.0, 3.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap());

    let loss = huber_loss(&mut graph, pred, target, 1.0).unwrap();
    let value = graph.value(loss).unwrap();
    assert!(value.shape().is_empty());
    assert_eq!(value.data(), &[0.75]);
}

#[test]
fn huber_loss_rejects_invalid_delta() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());

    let err = huber_loss(&mut graph, pred, target, 0.0).unwrap_err();
    assert_eq!(err, ModelError::InvalidHuberDelta { delta: 0.0 });
}

#[test]
fn hinge_loss_computes_scalar_mean() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![2.0, -0.5]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![1.0, -1.0]).unwrap());

    let loss = hinge_loss(&mut graph, pred, target, 1.0).unwrap();
    let value = graph.value(loss).unwrap();
    assert!(value.shape().is_empty());
    assert_eq!(value.data(), &[0.25]);
}

#[test]
fn hinge_loss_rejects_invalid_margin() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![1.0, -1.0]).unwrap());

    let err = hinge_loss(&mut graph, pred, target, 0.0).unwrap_err();
    assert_eq!(err, ModelError::InvalidHingeMargin { margin: 0.0 });
}

#[test]
fn smooth_l1_loss_basic() {
    // beta = 1.0: residuals 0.5 (quadratic region) and 2.0 (linear region)
    // x=0.5: 0.5 * 0.25 / 1.0 = 0.125
    // x=2.0: 2.0 - 0.5 * 1.0 = 1.5
    // mean = (0.125 + 1.5) / 2 = 0.8125
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![2], vec![1.5, 4.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());

    let loss = smooth_l1_loss(&mut graph, pred, target, 1.0).unwrap();
    let value = graph.value(loss).unwrap();
    assert!(value.shape().is_empty());
    assert!((value.data()[0] - 0.8125).abs() < 1e-6);
}

#[test]
fn smooth_l1_loss_zero_residual() {
    let mut graph = Graph::new();
    let pred = graph.variable(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap());

    let loss = smooth_l1_loss(&mut graph, pred, target, 1.0).unwrap();
    let value = graph.value(loss).unwrap();
    assert_eq!(value.data(), &[0.0]);
}

#[test]
fn kl_div_loss_basic() {
    // target = [0.25, 0.75], log_pred = [ln(0.5), ln(0.5)]
    // KL = 0.25 * (ln(0.25) - ln(0.5)) + 0.75 * (ln(0.75) - ln(0.5))
    //    = 0.25 * ln(0.5) + 0.75 * ln(1.5)
    //    = 0.25 * (-0.6931...) + 0.75 * (0.4055...)
    //    = -0.1733 + 0.3041 = 0.1309
    // mean = 0.1309 / 2
    let mut graph = Graph::new();
    let log_pred =
        graph.variable(Tensor::from_vec(vec![2], vec![0.5_f32.ln(), 0.5_f32.ln()]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![0.25, 0.75]).unwrap());

    let loss = kl_div_loss(&mut graph, log_pred, target).unwrap();
    let value = graph.value(loss).unwrap();
    let expected =
        (0.25 * (0.25_f32.ln() - 0.5_f32.ln()) + 0.75 * (0.75_f32.ln() - 0.5_f32.ln())) / 2.0;
    assert!((value.data()[0] - expected).abs() < 1e-6);
}

#[test]
fn kl_div_loss_same_distribution() {
    // When target == pred, KL divergence is zero.
    // target = [0.3, 0.7], log_pred = [ln(0.3), ln(0.7)]
    let mut graph = Graph::new();
    let log_pred =
        graph.variable(Tensor::from_vec(vec![2], vec![0.3_f32.ln(), 0.7_f32.ln()]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![2], vec![0.3, 0.7]).unwrap());

    let loss = kl_div_loss(&mut graph, log_pred, target).unwrap();
    let value = graph.value(loss).unwrap();
    assert!(value.data()[0].abs() < 1e-6);
}
