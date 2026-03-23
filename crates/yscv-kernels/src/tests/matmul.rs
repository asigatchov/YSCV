use yscv_tensor::Tensor;

use crate::{
    KernelError, ParallelMatmulConfig, matmul_2d, matmul_2d_sequential, matmul_2d_with_config,
};

use super::build_tensor;

#[test]
fn matmul_2d_computes_expected_result() {
    let lhs = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Tensor::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let out = matmul_2d(&lhs, &rhs).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_2d_rejects_non_rank_2_inputs() {
    let lhs = Tensor::scalar(1.0);
    let rhs = Tensor::from_vec(vec![1, 1], vec![2.0]).unwrap();

    let err = matmul_2d(&lhs, &rhs).unwrap_err();
    assert_eq!(
        err,
        KernelError::InvalidMatMulRank {
            left_rank: 0,
            right_rank: 2
        }
    );
}

#[test]
fn matmul_2d_rejects_shape_mismatch() {
    let lhs = Tensor::zeros(vec![2, 3]).unwrap();
    let rhs = Tensor::zeros(vec![4, 2]).unwrap();

    let err = matmul_2d(&lhs, &rhs).unwrap_err();
    assert_eq!(
        err,
        KernelError::MatMulShapeMismatch {
            left: vec![2, 3],
            right: vec![4, 2]
        }
    );
}

#[test]
fn matmul_2d_parallel_matches_sequential() {
    let lhs = build_tensor(&[96, 128], 0.13);
    let rhs = build_tensor(&[128, 64], 0.61);
    let sequential = matmul_2d_sequential(&lhs, &rhs).unwrap();
    let adaptive = matmul_2d(&lhs, &rhs).unwrap();
    assert_eq!(adaptive, sequential);
}

#[test]
fn matmul_2d_disabled_parallel_matches_sequential() {
    let lhs = build_tensor(&[64, 96], 0.19);
    let rhs = build_tensor(&[96, 80], 0.47);
    let sequential = matmul_2d_sequential(&lhs, &rhs).unwrap();
    let disabled = matmul_2d_with_config(&lhs, &rhs, ParallelMatmulConfig::disabled()).unwrap();
    assert_eq!(disabled, sequential);
}
