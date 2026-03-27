use yscv_tensor::Tensor;

use super::assert_slice_close;
use crate::{BatchNorm2dParams, LayerNormLastDimParams, batch_norm2d_nhwc, layer_norm_last_dim};

/// Compare GPU batch-norm result with CPU reference.
/// If GPU is not available, the test validates CPU-only logic.
#[test]
fn gpu_batch_norm_matches_cpu() {
    // NHWC input: [1, 2, 2, 3]
    let input = Tensor::from_vec(
        vec![1, 2, 2, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let gamma = Tensor::from_vec(vec![3], vec![1.0, 1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
    let mean = Tensor::from_vec(vec![3], vec![5.5, 6.5, 7.5]).unwrap();
    let var = Tensor::from_vec(vec![3], vec![11.25, 11.25, 11.25]).unwrap();
    let epsilon = 1e-5;

    let params = BatchNorm2dParams {
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &var,
        epsilon,
    };
    let cpu_result = batch_norm2d_nhwc(&input, params).unwrap();

    // Try GPU if available
    #[cfg(feature = "gpu")]
    {
        match crate::gpu_batch_norm(&input, &gamma, &beta, &mean, &var, epsilon) {
            Ok(gpu_result) => {
                assert_eq!(gpu_result.shape(), cpu_result.shape());
                assert_slice_close(gpu_result.data(), cpu_result.data(), 1e-4);
            }
            Err(_) => {
                // GPU not available, skip GPU comparison
            }
        }
    }

    // Verify CPU result is reasonable: normalized values should be centered
    assert_eq!(cpu_result.shape(), &[1, 2, 2, 3]);
    let d = cpu_result.data();
    // First element: (1.0 - 5.5) / sqrt(11.25 + 1e-5) * 1.0 + 0.0
    let expected_0 = (1.0 - 5.5) / (11.25_f32 + epsilon).sqrt();
    assert!((d[0] - expected_0).abs() < 1e-4);
}

/// Compare GPU layer-norm result with CPU reference.
/// If GPU is not available, the test validates CPU-only logic.
#[test]
fn gpu_layer_norm_matches_cpu() {
    let input = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let gamma = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let beta = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
    let epsilon = 1e-5;

    let params = LayerNormLastDimParams {
        gamma: &gamma,
        beta: &beta,
        epsilon,
    };
    let cpu_result = layer_norm_last_dim(&input, params).unwrap();

    #[cfg(feature = "gpu")]
    {
        match crate::gpu_layer_norm(&input, &gamma, &beta, epsilon) {
            Ok(gpu_result) => {
                assert_eq!(gpu_result.shape(), cpu_result.shape());
                assert_slice_close(gpu_result.data(), cpu_result.data(), 1e-4);
            }
            Err(_) => {
                // GPU not available, skip
            }
        }
    }

    // Verify CPU result is reasonable
    assert_eq!(cpu_result.shape(), &[2, 4]);
    let d = cpu_result.data();
    // Row 0: mean=2.5, var=1.25
    // (1.0 - 2.5) / sqrt(1.25 + eps) = -1.5 / ~1.118 = ~-1.3416
    let row0_mean = 2.5_f32;
    let row0_var = 1.25_f32;
    let expected_0 = (1.0 - row0_mean) / (row0_var + epsilon).sqrt();
    assert!((d[0] - expected_0).abs() < 1e-4);
}

/// Compare GPU transpose result with CPU reference.
/// If GPU is not available, the test validates CPU-only logic.
#[test]
fn gpu_transpose_matches_cpu() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let cpu_result = input.transpose_2d().unwrap();
    assert_eq!(cpu_result.shape(), &[3, 2]);
    assert_eq!(cpu_result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    #[cfg(feature = "gpu")]
    {
        match crate::gpu_transpose(&input) {
            Ok(gpu_result) => {
                assert_eq!(gpu_result.shape(), cpu_result.shape());
                assert_slice_close(gpu_result.data(), cpu_result.data(), 1e-6);
            }
            Err(_) => {
                // GPU not available, skip
            }
        }
    }
}
