use yscv_tensor::Tensor;

use super::super::{histogram_256, histogram_equalize};

#[test]
fn histogram_256_counts_correctly() {
    let img = Tensor::from_vec(vec![1, 4, 1], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let hist = histogram_256(&img).unwrap();
    assert_eq!(hist[0], 2);
    assert_eq!(hist[255], 2);
}

#[test]
fn histogram_equalize_preserves_shape() {
    let img = Tensor::filled(vec![4, 4, 1], 0.5).unwrap();
    let eq = histogram_equalize(&img).unwrap();
    assert_eq!(eq.shape(), &[4, 4, 1]);
}

#[test]
fn clahe_preserves_shape() {
    let img = Tensor::from_vec(vec![8, 8, 1], vec![0.5; 64]).unwrap();
    let result = super::super::clahe(&img, 2, 2, 2.0).unwrap();
    assert_eq!(result.shape(), &[8, 8, 1]);
    for &v in result.data() {
        assert!((0.0..=1.0).contains(&v));
    }
}
