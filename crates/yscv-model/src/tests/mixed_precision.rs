use yscv_tensor::Tensor;

#[test]
fn dynamic_loss_scaler_reduces_on_overflow() {
    use crate::DynamicLossScaler;
    let mut scaler = DynamicLossScaler::new(1024.0);
    assert_eq!(scaler.scale(), 1024.0);
    let applied = scaler.update(true);
    assert!(!applied);
    assert!(scaler.scale() < 1024.0);
}

#[test]
fn dynamic_loss_scaler_grows_without_overflow() {
    use crate::DynamicLossScaler;
    let mut scaler = DynamicLossScaler::new(1.0);
    for _ in 0..2001 {
        scaler.update(false);
    }
    assert!(scaler.scale() > 1.0);
}

#[test]
fn mixed_precision_config_default() {
    use crate::MixedPrecisionConfig;
    use yscv_tensor::DType;
    let config = MixedPrecisionConfig::default();
    assert_eq!(config.forward_dtype, DType::F16);
    assert_eq!(config.master_dtype, DType::F32);
    assert!(config.loss_scale > 0.0);
}

#[test]
fn cast_params_for_forward_dtype_conversion() {
    use crate::cast_params_for_forward;
    use yscv_tensor::DType;
    let mut graph = yscv_autograd::Graph::new();
    let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let node = graph.variable(t);
    let casted = cast_params_for_forward(&graph, &[node], DType::F16).unwrap();
    assert_eq!(casted.len(), 1);
    assert_eq!(casted[0].dtype(), DType::F16);
}
