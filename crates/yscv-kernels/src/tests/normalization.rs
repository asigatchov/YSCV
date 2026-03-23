use yscv_tensor::Tensor;

use crate::{
    BatchNorm2dParams, GroupNormNhwcParams, KernelError, LayerNormLastDimParams,
    ParallelElementwiseConfig, RmsNormLastDimParams, batch_norm2d_nhwc,
    batch_norm2d_nhwc_with_config, group_norm_nhwc, layer_norm_last_dim,
    layer_norm_last_dim_with_config, rms_norm_last_dim,
};

use super::{assert_slice_close, build_tensor};

#[test]
fn layer_norm_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gamma = Tensor::from_vec(vec![2], vec![2.0, 0.5]).unwrap();
    let beta = Tensor::from_vec(vec![2], vec![0.1, -0.2]).unwrap();
    let out = layer_norm_last_dim(
        &input,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 1e-5,
        },
    )
    .unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_slice_close(out.data(), &[-1.89996, 0.29999, -1.89996, 0.29999], 1e-4);
}

#[test]
fn layer_norm_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let gamma = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let beta = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let err = layer_norm_last_dim(
        &input,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 1e-5,
        },
    )
    .unwrap_err();
    assert_eq!(err, KernelError::InvalidLayerNormRank { got_rank: 0 });
}

#[test]
fn layer_norm_last_dim_rejects_parameter_shape_mismatch() {
    let input = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
    let gamma = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
    let err = layer_norm_last_dim(
        &input,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 1e-5,
        },
    )
    .unwrap_err();
    assert_eq!(
        err,
        KernelError::LayerNormParameterShapeMismatch {
            parameter: "gamma",
            shape: vec![2],
            expected_features: 3,
        }
    );
}

#[test]
fn layer_norm_last_dim_rejects_invalid_epsilon() {
    let input = Tensor::from_vec(vec![2, 2], vec![0.0; 4]).unwrap();
    let gamma = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let err = layer_norm_last_dim(
        &input,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 0.0,
        },
    )
    .unwrap_err();
    assert_eq!(err, KernelError::InvalidLayerNormEpsilon);
}

#[test]
fn layer_norm_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[32, 64], 0.84);
    let gamma = build_tensor(&[64], 0.22);
    let beta = build_tensor(&[64], 0.63);
    let baseline = layer_norm_last_dim(
        &input,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 1e-5,
        },
    )
    .unwrap();
    let disabled = layer_norm_last_dim_with_config(
        &input,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 1e-5,
        },
        ParallelElementwiseConfig::disabled(),
    )
    .unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn batch_norm2d_nhwc_computes_expected_result() {
    let input = Tensor::from_vec(vec![1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gamma = Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap();
    let beta = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    let mean = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let variance = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();

    let out = batch_norm2d_nhwc(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1.0,
        },
    )
    .unwrap();
    assert_eq!(out.shape(), &[1, 2, 1, 2]);
    assert_eq!(out.data(), &[0.5, -1.0, 4.5, 5.0]);
}

#[test]
fn batch_norm2d_nhwc_rejects_invalid_rank() {
    let input = Tensor::from_vec(vec![2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gamma = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let mean = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let variance = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();

    let err = batch_norm2d_nhwc(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1e-5,
        },
    )
    .unwrap_err();
    assert_eq!(err, KernelError::InvalidBatchNormRank { got_rank: 3 });
}

#[test]
fn batch_norm2d_nhwc_rejects_parameter_shape_mismatch() {
    let input = Tensor::from_vec(vec![1, 1, 1, 2], vec![1.0, 2.0]).unwrap();
    let gamma = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let beta = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let mean = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let variance = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();

    let err = batch_norm2d_nhwc(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1e-5,
        },
    )
    .unwrap_err();
    assert_eq!(
        err,
        KernelError::BatchNormParameterShapeMismatch {
            parameter: "gamma",
            shape: vec![1],
            expected_channels: 2,
        }
    );
}

#[test]
fn batch_norm2d_nhwc_rejects_invalid_epsilon() {
    let input = Tensor::from_vec(vec![1, 1, 1, 1], vec![1.0]).unwrap();
    let gamma = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let beta = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let mean = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let variance = Tensor::from_vec(vec![1], vec![1.0]).unwrap();

    let err = batch_norm2d_nhwc(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 0.0,
        },
    )
    .unwrap_err();
    assert_eq!(err, KernelError::InvalidBatchNormEpsilon);
}

#[test]
fn batch_norm2d_nhwc_rejects_invalid_variance_envelope() {
    let input = Tensor::from_vec(vec![1, 1, 1, 2], vec![1.0, 2.0]).unwrap();
    let gamma = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let mean = Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap();
    let variance = Tensor::from_vec(vec![2], vec![-0.2, 0.5]).unwrap();

    let err = batch_norm2d_nhwc(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 0.1,
        },
    )
    .unwrap_err();
    assert_eq!(err, KernelError::InvalidBatchNormVariance { channel: 0 });
}

#[test]
fn batch_norm2d_with_config_disabled_matches_default() {
    let input = build_tensor(&[2, 16, 16, 8], 0.37);
    let gamma = build_tensor(&[8], 0.21);
    let beta = build_tensor(&[8], 0.73);
    let mean = build_tensor(&[8], 0.18);
    let variance = build_tensor(&[8], 1.19);

    let baseline = batch_norm2d_nhwc(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1e-5,
        },
    )
    .unwrap();
    let disabled = batch_norm2d_nhwc_with_config(
        &input,
        BatchNorm2dParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1e-5,
        },
        ParallelElementwiseConfig::disabled(),
    )
    .unwrap();
    assert_eq!(baseline, disabled);
}

// ---------------------------------------------------------------------------
// GroupNorm tests
// ---------------------------------------------------------------------------

#[test]
fn group_norm_uniform_input() {
    // Uniform input => after normalization each value is 0, so output = gamma*0 + beta = beta
    let input = Tensor::from_vec(vec![1, 2, 2, 4], vec![5.0; 16]).unwrap();
    let gamma = Tensor::from_vec(vec![4], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let beta = Tensor::from_vec(vec![4], vec![0.5, 1.0, 1.5, 2.0]).unwrap();

    let out = group_norm_nhwc(
        &input,
        GroupNormNhwcParams {
            gamma: &gamma,
            beta: &beta,
            num_groups: 2,
            epsilon: 1e-5,
        },
    )
    .unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 4]);
    // Each spatial position should have beta values
    let expected: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0]
        .into_iter()
        .cycle()
        .take(16)
        .collect();
    assert_slice_close(out.data(), &expected, 1e-4);
}

#[test]
fn group_norm_two_groups() {
    // 1 sample, 1x2 spatial, 4 channels, 2 groups (channels 0-1 and 2-3)
    let input = Tensor::from_vec(
        vec![1, 1, 2, 4],
        vec![1.0, 3.0, 10.0, 20.0, 2.0, 4.0, 15.0, 25.0],
    )
    .unwrap();
    let gamma = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();

    let out = group_norm_nhwc(
        &input,
        GroupNormNhwcParams {
            gamma: &gamma,
            beta: &beta,
            num_groups: 2,
            epsilon: 1e-5,
        },
    )
    .unwrap();

    assert_eq!(out.shape(), &[1, 1, 2, 4]);
    let data = out.data();

    // Group 0 (channels 0,1) values: [1, 3, 2, 4] => mean=2.5, var=1.25
    // Group 1 (channels 2,3) values: [10, 20, 15, 25] => mean=17.5, var=31.25
    // Check that within each group, the mean is ~0
    let group0: Vec<f32> = vec![data[0], data[1], data[4], data[5]];
    let group0_mean: f32 = group0.iter().sum::<f32>() / 4.0;
    assert!(
        group0_mean.abs() < 1e-4,
        "group0 mean should be ~0, got {group0_mean}"
    );

    let group1: Vec<f32> = vec![data[2], data[3], data[6], data[7]];
    let group1_mean: f32 = group1.iter().sum::<f32>() / 4.0;
    assert!(
        group1_mean.abs() < 1e-4,
        "group1 mean should be ~0, got {group1_mean}"
    );
}

// ---------------------------------------------------------------------------
// RMSNorm tests
// ---------------------------------------------------------------------------

#[test]
fn rms_norm_identity_gamma() {
    // gamma=1 => output = x / rms(x)
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gamma = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let out = rms_norm_last_dim(
        &input,
        RmsNormLastDimParams {
            gamma: &gamma,
            epsilon: 1e-5,
        },
    )
    .unwrap();

    assert_eq!(out.shape(), &[1, 4]);
    // rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps)
    let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
    let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
    assert_slice_close(out.data(), &expected, 1e-4);
}

#[test]
fn rms_norm_scales_by_gamma() {
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let gamma = Tensor::from_vec(vec![4], vec![2.0, 2.0, 2.0, 2.0]).unwrap();

    let out = rms_norm_last_dim(
        &input,
        RmsNormLastDimParams {
            gamma: &gamma,
            epsilon: 1e-5,
        },
    )
    .unwrap();

    // Same as identity but doubled
    let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
    let expected: Vec<f32> = vec![2.0 / rms, 4.0 / rms, 6.0 / rms, 8.0 / rms];
    assert_slice_close(out.data(), &expected, 1e-4);
}
