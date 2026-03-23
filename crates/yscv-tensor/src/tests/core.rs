use crate::{Tensor, TensorError};

#[test]
fn creates_scalar_tensor() {
    let scalar = Tensor::scalar(3.25);
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.data(), &[3.25]);
}

#[test]
fn from_vec_accepts_valid_shape_and_computes_strides() {
    let tensor = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.strides(), &[12, 4, 1]);
    assert_eq!(tensor.len(), 24);
}

#[test]
fn from_vec_rejects_size_mismatch() {
    let err = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0]).unwrap_err();
    assert_eq!(
        err,
        TensorError::SizeMismatch {
            shape: vec![2, 2],
            data_len: 3
        }
    );
}

#[test]
fn get_and_set_use_multidimensional_indexing() {
    let mut tensor = Tensor::zeros(vec![2, 2]).unwrap();
    tensor.set(&[1, 0], 4.5).unwrap();
    assert_eq!(tensor.get(&[1, 0]).unwrap(), 4.5);
    assert_eq!(tensor.get(&[0, 0]).unwrap(), 0.0);
}

#[test]
fn get_rejects_invalid_index_rank() {
    let tensor = Tensor::zeros(vec![2, 2]).unwrap();
    let err = tensor.get(&[1]).unwrap_err();
    assert_eq!(
        err,
        TensorError::InvalidIndexRank {
            expected: 2,
            got: 1
        }
    );
}

#[test]
fn set_rejects_out_of_bounds_index() {
    let mut tensor = Tensor::zeros(vec![2, 2]).unwrap();
    let err = tensor.set(&[2, 0], 1.0).unwrap_err();
    assert_eq!(
        err,
        TensorError::IndexOutOfBounds {
            axis: 0,
            index: 2,
            dim: 2
        }
    );
}

#[test]
fn reshape_changes_metadata_not_data_order() {
    let tensor = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reshaped = tensor.reshape(vec![3, 2]).unwrap();

    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.strides(), &[2, 1]);
    assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_rejects_size_mismatch() {
    let tensor = Tensor::zeros(vec![2, 3]).unwrap();
    let err = tensor.reshape(vec![5]).unwrap_err();
    assert_eq!(
        err,
        TensorError::ReshapeSizeMismatch {
            from: vec![2, 3],
            to: vec![5]
        }
    );
}

#[test]
fn rejects_shape_that_overflows_total_size() {
    let err = Tensor::zeros(vec![usize::MAX, 2]).unwrap_err();
    assert_eq!(
        err,
        TensorError::SizeOverflow {
            shape: vec![usize::MAX, 2]
        }
    );
}
