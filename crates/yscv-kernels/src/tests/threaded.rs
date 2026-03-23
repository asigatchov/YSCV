#![cfg(not(miri))]

use std::num::NonZeroUsize;

use yscv_tensor::Tensor;

use crate::{
    Backend, BatchNorm2dParams, LayerNormLastDimParams, ParallelElementwiseConfig,
    ParallelMatmulConfig, SeparableConv2dParams, ThreadedCpuBackend, ThreadedCpuBackendConfig, add,
    avg_pool2d_nhwc, batch_norm2d_nhwc, conv2d_nhwc, depthwise_conv2d_nhwc, layer_norm_last_dim,
    log_softmax_last_dim, logsumexp_last_dim, matmul_2d_sequential, matmul_2d_with_threads,
    max_pool2d_nhwc, mul, relu, separable_conv2d_nhwc, sigmoid, softmax_last_dim, sub,
};

use super::build_tensor;

#[test]
fn threaded_cpu_backend_matches_sequential() {
    let lhs = build_tensor(&[128, 96], 0.29);
    let rhs = build_tensor(&[96, 72], 0.73);
    let backend = ThreadedCpuBackend::new(NonZeroUsize::new(2).unwrap()).unwrap();
    let threaded = backend.matmul_2d(&lhs, &rhs).unwrap();
    let sequential = matmul_2d_sequential(&lhs, &rhs).unwrap();
    assert_eq!(threaded, sequential);
}

#[test]
fn matmul_2d_with_threads_matches_sequential() {
    let lhs = build_tensor(&[128, 64], 0.31);
    let rhs = build_tensor(&[64, 128], 0.83);
    let threaded = matmul_2d_with_threads(
        &lhs,
        &rhs,
        NonZeroUsize::new(2).unwrap(),
        ParallelMatmulConfig::default(),
    )
    .unwrap();
    let sequential = matmul_2d_sequential(&lhs, &rhs).unwrap();
    assert_eq!(threaded, sequential);
}

#[test]
fn threaded_backend_elementwise_matches_cpu() {
    let lhs = build_tensor(&[256, 192], 0.17);
    let rhs = build_tensor(&[256, 192], 0.67);
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let add_threaded = backend.add(&lhs, &rhs).unwrap();
    let add_cpu = add(&lhs, &rhs).unwrap();
    assert_eq!(add_threaded, add_cpu);

    let sub_threaded = backend.sub(&lhs, &rhs).unwrap();
    let sub_cpu = sub(&lhs, &rhs).unwrap();
    assert_eq!(sub_threaded, sub_cpu);

    let mul_threaded = backend.mul(&lhs, &rhs).unwrap();
    let mul_cpu = mul(&lhs, &rhs).unwrap();
    assert_eq!(mul_threaded, mul_cpu);

    let relu_threaded = backend.relu(&lhs);
    let relu_cpu = relu(&lhs);
    assert_eq!(relu_threaded, relu_cpu);

    let sigmoid_threaded = backend.sigmoid(&lhs);
    let sigmoid_cpu = sigmoid(&lhs);
    assert_eq!(sigmoid_threaded, sigmoid_cpu);

    let softmax_threaded = backend.softmax_last_dim(&lhs).unwrap();
    let softmax_cpu = softmax_last_dim(&lhs).unwrap();
    assert_eq!(softmax_threaded, softmax_cpu);

    let log_softmax_threaded = backend.log_softmax_last_dim(&lhs).unwrap();
    let log_softmax_cpu = log_softmax_last_dim(&lhs).unwrap();
    assert_eq!(log_softmax_threaded, log_softmax_cpu);

    let logsumexp_threaded = backend.logsumexp_last_dim(&lhs).unwrap();
    let logsumexp_cpu = logsumexp_last_dim(&lhs).unwrap();
    assert_eq!(logsumexp_threaded, logsumexp_cpu);

    let gamma = build_tensor(&[192], 0.41);
    let beta = build_tensor(&[192], 0.93);
    let layer_norm_threaded = backend
        .layer_norm_last_dim(
            &lhs,
            LayerNormLastDimParams {
                gamma: &gamma,
                beta: &beta,
                epsilon: 1e-5,
            },
        )
        .unwrap();
    let layer_norm_cpu = layer_norm_last_dim(
        &lhs,
        LayerNormLastDimParams {
            gamma: &gamma,
            beta: &beta,
            epsilon: 1e-5,
        },
    )
    .unwrap();
    assert_eq!(layer_norm_threaded, layer_norm_cpu);
}

#[test]
fn threaded_backend_pool2d_matches_cpu() {
    let input = build_tensor(&[2, 64, 64, 3], 0.27);
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let max_threaded = backend.max_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    let max_cpu = max_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    assert_eq!(max_threaded, max_cpu);

    let avg_threaded = backend.avg_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    let avg_cpu = avg_pool2d_nhwc(&input, 2, 2, 2, 2).unwrap();
    assert_eq!(avg_threaded, avg_cpu);
}

#[test]
fn threaded_backend_conv2d_matches_cpu() {
    let input = build_tensor(&[2, 32, 32, 8], 0.53);
    let kernel = build_tensor(&[3, 3, 8, 12], 0.91);
    let bias = build_tensor(&[12], 0.07);
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let threaded = backend
        .conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1)
        .unwrap();
    let cpu = conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();
    assert_eq!(threaded, cpu);
}

#[test]
fn threaded_backend_depthwise_conv2d_matches_cpu() {
    let input = build_tensor(&[2, 32, 32, 8], 0.24);
    let kernel = build_tensor(&[3, 3, 8, 2], 0.88);
    let bias = build_tensor(&[16], 0.31);
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let threaded = backend
        .depthwise_conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1)
        .unwrap();
    let cpu = depthwise_conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();
    assert_eq!(threaded, cpu);
}

#[test]
fn threaded_backend_separable_conv2d_matches_cpu() {
    let input = build_tensor(&[2, 32, 32, 8], 0.28);
    let depthwise_kernel = build_tensor(&[3, 3, 8, 2], 0.73);
    let depthwise_bias = build_tensor(&[16], 0.14);
    let pointwise_kernel = build_tensor(&[1, 1, 16, 12], 0.49);
    let pointwise_bias = build_tensor(&[12], 0.81);
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let threaded = backend
        .separable_conv2d_nhwc(
            &input,
            SeparableConv2dParams {
                depthwise_kernel: &depthwise_kernel,
                depthwise_bias: Some(&depthwise_bias),
                pointwise_kernel: &pointwise_kernel,
                pointwise_bias: Some(&pointwise_bias),
            },
            1,
            1,
        )
        .unwrap();
    let cpu = separable_conv2d_nhwc(
        &input,
        SeparableConv2dParams {
            depthwise_kernel: &depthwise_kernel,
            depthwise_bias: Some(&depthwise_bias),
            pointwise_kernel: &pointwise_kernel,
            pointwise_bias: Some(&pointwise_bias),
        },
        1,
        1,
    )
    .unwrap();
    assert_eq!(threaded, cpu);
}

#[test]
fn threaded_backend_batch_norm2d_matches_cpu() {
    let input = build_tensor(&[2, 64, 64, 8], 0.19);
    let gamma = build_tensor(&[8], 0.44);
    let beta = build_tensor(&[8], 0.91);
    let mean = build_tensor(&[8], 0.38);
    let variance = build_tensor(&[8], 1.24);
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let threaded = backend
        .batch_norm2d_nhwc(
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
    let cpu = batch_norm2d_nhwc(
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
    assert_eq!(threaded, cpu);
}

#[test]
fn threaded_backend_add_broadcast_falls_back_to_tensor_path() {
    let lhs = build_tensor(&[32, 3], 0.23);
    let rhs = Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap();
    let config = ThreadedCpuBackendConfig {
        matmul: ParallelMatmulConfig::disabled(),
        elementwise: ParallelElementwiseConfig {
            min_parallel_elements: 1,
        },
    };
    let backend =
        ThreadedCpuBackend::with_full_config(NonZeroUsize::new(2).unwrap(), config).unwrap();

    let threaded = backend.add(&lhs, &rhs).unwrap();
    let cpu = add(&lhs, &rhs).unwrap();
    assert_eq!(threaded, cpu);
}
