use yscv_tensor::Tensor;

use super::super::{resize_bilinear, resize_nearest};

#[test]
fn resize_nearest_scales_2x2_to_4x4() {
    let input = Tensor::from_vec(
        vec![2, 2, 1],
        vec![
            1.0, 2.0, //
            3.0, 4.0,
        ],
    )
    .unwrap();
    let out = resize_nearest(&input, 4, 4).unwrap();
    assert_eq!(out.shape(), &[4, 4, 1]);
    assert_eq!(
        out.data(),
        &[
            1.0, 1.0, 2.0, 2.0, //
            1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, //
            3.0, 3.0, 4.0, 4.0,
        ]
    );
}

#[test]
fn resize_bilinear_scales_2x2_to_4x4() {
    let img = Tensor::from_vec(vec![2, 2, 1], vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    let resized = resize_bilinear(&img, 4, 4).unwrap();
    assert_eq!(resized.shape(), &[4, 4, 1]);
    assert!((resized.data()[0] - 0.0).abs() < 1e-5);
    assert!((resized.data()[3] - 1.0).abs() < 1e-5);
}

#[test]
fn resize_bilinear_rejects_zero_dimensions() {
    let img = Tensor::filled(vec![2, 2, 1], 0.5).unwrap();
    assert!(resize_bilinear(&img, 0, 4).is_err());
}
