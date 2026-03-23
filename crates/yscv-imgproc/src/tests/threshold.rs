use yscv_tensor::Tensor;

use super::super::{
    adaptive_threshold_gaussian, adaptive_threshold_mean, threshold_binary, threshold_binary_inv,
    threshold_otsu, threshold_truncate,
};

#[test]
fn threshold_binary_produces_binary_output() {
    let img = Tensor::from_vec(vec![1, 4, 1], vec![0.1, 0.4, 0.6, 0.9]).unwrap();
    let out = threshold_binary(&img, 0.5, 1.0).unwrap();
    assert_eq!(out.data(), &[0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn threshold_binary_inv_produces_inverted_output() {
    let img = Tensor::from_vec(vec![1, 4, 1], vec![0.1, 0.4, 0.6, 0.9]).unwrap();
    let out = threshold_binary_inv(&img, 0.5, 1.0).unwrap();
    assert_eq!(out.data(), &[1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn threshold_truncate_caps_values() {
    let img = Tensor::from_vec(vec![1, 3, 1], vec![0.1, 0.6, 0.9]).unwrap();
    let out = threshold_truncate(&img, 0.5).unwrap();
    assert_eq!(out.data(), &[0.1, 0.5, 0.5]);
}

#[test]
fn threshold_otsu_finds_bimodal_threshold() {
    let mut data = vec![0.2f32; 100];
    for v in &mut data[50..] {
        *v = 0.8;
    }
    let img = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let (thresh, _thresholded) = threshold_otsu(&img, 1.0).unwrap();
    assert!(thresh > 0.1 && thresh < 0.9, "otsu thresh={thresh}");
}

#[test]
fn adaptive_threshold_mean_produces_binary() {
    let data: Vec<f32> = (0..25).map(|i| i as f32 / 25.0).collect();
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let out = adaptive_threshold_mean(&img, 1.0, 3, 0.0).unwrap();
    assert_eq!(out.shape(), &[5, 5, 1]);
    for &v in out.data() {
        assert!(v == 0.0 || v == 1.0);
    }
}

#[test]
fn adaptive_threshold_mean_rejects_even_block() {
    let img = Tensor::filled(vec![5, 5, 1], 0.5).unwrap();
    assert!(adaptive_threshold_mean(&img, 1.0, 4, 0.0).is_err());
}

#[test]
fn adaptive_threshold_gaussian_produces_binary() {
    let data: Vec<f32> = (0..25).map(|i| i as f32 / 25.0).collect();
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let out = adaptive_threshold_gaussian(&img, 1.0, 3, 0.0).unwrap();
    assert_eq!(out.shape(), &[5, 5, 1]);
    for &v in out.data() {
        assert!(v == 0.0 || v == 1.0);
    }
}

#[test]
fn adaptive_threshold_mean_shape() {
    let img = Tensor::from_vec(vec![8, 8, 1], vec![0.5; 64]).unwrap();
    let result = adaptive_threshold_mean(&img, 1.0, 3, 0.0).unwrap();
    assert_eq!(result.shape(), &[8, 8, 1]);
    for &v in result.data() {
        assert!(v == 0.0 || v == 1.0);
    }
}
