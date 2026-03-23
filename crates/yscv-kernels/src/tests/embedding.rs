use yscv_tensor::Tensor;

use crate::{dropout, embedding_lookup};

#[test]
fn embedding_lookup_shape() {
    // weight: [vocab=10, dim=4], indices: [3]
    let weight = Tensor::from_vec(vec![10, 4], (0..40).map(|i| i as f32).collect()).unwrap();
    let indices = Tensor::from_vec(vec![3], vec![0.0, 5.0, 9.0]).unwrap();
    let result = embedding_lookup(&weight, &indices).unwrap();
    assert_eq!(result.shape(), &[3, 4]);
}

#[test]
fn embedding_lookup_correct_values() {
    // weight: [4, 3] — 4 vocab entries of dim 3
    let weight = Tensor::from_vec(
        vec![4, 3],
        vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ],
    )
    .unwrap();
    let indices = Tensor::from_vec(vec![4], vec![3.0, 0.0, 2.0, 1.0]).unwrap();
    let result = embedding_lookup(&weight, &indices).unwrap();
    assert_eq!(result.shape(), &[4, 3]);
    let data = result.data();
    // row 3, row 0, row 2, row 1
    assert_eq!(
        data,
        &[
            10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0
        ]
    );
}

#[test]
fn dropout_inference_identity() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = dropout(&input, 0.5, 42, false).unwrap();
    assert_eq!(result.data(), input.data());
}

#[test]
fn dropout_zeros_ratio() {
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 1.0).collect();
    let input = Tensor::from_vec(vec![n], data).unwrap();
    let p = 0.3;
    let result = dropout(&input, p, 12345, true).unwrap();
    let out = result.data();

    let num_zeros = out.iter().filter(|&&v| v == 0.0).count();
    let ratio = num_zeros as f64 / n as f64;
    // Check within tolerance of 0.05
    assert!(
        (ratio - p as f64).abs() < 0.05,
        "Expected ~{p} fraction of zeros, got {ratio} ({num_zeros}/{n})"
    );

    // Verify surviving elements are scaled by 1/(1-p)
    let scale = 1.0 / (1.0 - p);
    let orig = input.data();
    for i in 0..n {
        if out[i] != 0.0 {
            let expected = orig[i] * scale;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "Element {i}: expected {expected}, got {}",
                out[i]
            );
        }
    }
}
