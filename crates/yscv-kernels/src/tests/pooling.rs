use yscv_tensor::Tensor;

use crate::{
    KernelError, ParallelElementwiseConfig, avg_pool2d_nhwc, avg_pool2d_nhwc_with_config,
    max_pool2d_nhwc, max_pool2d_nhwc_with_config,
};

use super::build_tensor;

#[test]
fn max_pool2d_nhwc_computes_expected_result() {
    let input = Tensor::from_vec(
        vec![1, 4, 4, 1],
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();
    let out = max_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    assert_eq!(out.data(), &[6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn avg_pool2d_nhwc_computes_expected_result() {
    let input = Tensor::from_vec(
        vec![1, 4, 4, 1],
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();
    let out = avg_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    assert_eq!(out.data(), &[3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn pool2d_nhwc_rejects_invalid_rank() {
    let input = Tensor::from_vec(vec![4, 4, 1], vec![0.0; 16]).unwrap();
    let err = max_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap_err();
    assert_eq!(err, KernelError::InvalidPoolRank { got_rank: 3 });
}

#[test]
fn pool2d_nhwc_rejects_invalid_parameters() {
    let input = Tensor::from_vec(vec![1, 4, 4, 1], vec![0.0; 16]).unwrap();
    let err = avg_pool2d_nhwc(&input, 0, 2, 2, 2).unwrap_err();
    assert_eq!(
        err,
        KernelError::InvalidPoolParameters {
            kernel_h: 0,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        }
    );
}

#[test]
fn pool2d_nhwc_rejects_kernel_larger_than_input() {
    let input = Tensor::from_vec(vec![1, 3, 3, 1], vec![0.0; 9]).unwrap();
    let err = max_pool2d_nhwc(&input, 4, 3, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::PoolKernelLargerThanInput {
            input_h: 3,
            input_w: 3,
            kernel_h: 4,
            kernel_w: 3,
        }
    );
}

#[test]
fn pool2d_with_config_disabled_matches_default() {
    let input = build_tensor(&[2, 16, 16, 3], 0.39);
    let max_default = max_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    let max_disabled =
        max_pool2d_nhwc_with_config(&input, 2, 2, 2, 2, ParallelElementwiseConfig::disabled())
            .unwrap();
    assert_eq!(max_default, max_disabled);

    let avg_default = avg_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    let avg_disabled =
        avg_pool2d_nhwc_with_config(&input, 2, 2, 2, 2, ParallelElementwiseConfig::disabled())
            .unwrap();
    assert_eq!(avg_default, avg_disabled);
}
