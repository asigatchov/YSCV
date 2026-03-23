use yscv_tensor::Tensor;

use super::super::error::ModelError;
use super::super::transform::{
    CenterCrop, Compose, GaussianBlur, Normalize, RandomHorizontalFlip, Resize, ScaleValues,
    Transform,
};

#[test]
fn compose_chain() {
    // 1x3 tensor with values [0.5, 1.0, 1.5]
    let input = Tensor::from_vec(vec![1, 3], vec![0.5, 1.0, 1.5]).unwrap();

    let pipeline = Compose::new()
        .add(Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]))
        .add(ScaleValues::new(2.0));

    let output = pipeline.apply(&input).unwrap();
    let data = output.data();

    // After normalize: (0.5-0.5)/0.5=0, (1.0-0.5)/0.5=1, (1.5-0.5)/0.5=2
    // After scale*2:   0, 2, 4
    super::assert_slice_approx_eq(data, &[0.0, 2.0, 4.0], 1e-6);
}

#[test]
fn custom_transform() {
    struct AddOne;

    impl Transform for AddOne {
        fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
            let data = input.data();
            let out: Vec<f32> = data.iter().map(|v| v + 1.0).collect();
            Ok(Tensor::from_vec(input.shape().to_vec(), out)?)
        }
    }

    let input = Tensor::from_vec(vec![2], vec![3.0, 7.0]).unwrap();

    let pipeline = Compose::new().add(AddOne).add(ScaleValues::new(0.5));

    let output = pipeline.apply(&input).unwrap();
    // (3+1)*0.5=2, (7+1)*0.5=4
    super::assert_slice_approx_eq(output.data(), &[2.0, 4.0], 1e-6);
}

#[test]
fn empty_compose() {
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let pipeline = Compose::new();
    let output = pipeline.apply(&input).unwrap();
    assert_eq!(output.shape(), input.shape());
    super::assert_slice_approx_eq(output.data(), input.data(), 0.0);
}

#[test]
fn resize_changes_shape() {
    // [10, 10, 3] filled with 1.0
    let input = Tensor::from_vec(vec![10, 10, 3], vec![1.0; 10 * 10 * 3]).unwrap();
    let resize = Resize::new(5, 5);
    let output = resize.apply(&input).unwrap();
    assert_eq!(output.shape(), &[5, 5, 3]);
}

#[test]
fn center_crop_extracts_center() {
    // [10, 10, 1] with sequential values
    let data: Vec<f32> = (0..100).map(|v| v as f32).collect();
    let input = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let crop = CenterCrop::new(4);
    let output = crop.apply(&input).unwrap();
    assert_eq!(output.shape(), &[4, 4, 1]);
    // Center crop starts at row 3, col 3
    // First element should be row=3, col=3 → index 33
    assert_eq!(output.data()[0], 33.0);
}

#[test]
fn random_flip_preserves_shape() {
    let input = Tensor::from_vec(vec![5, 7, 3], vec![1.0; 5 * 7 * 3]).unwrap();
    // p=1.0 so it always flips
    let flip = RandomHorizontalFlip::new(1.0, 42);
    let output = flip.apply(&input).unwrap();
    assert_eq!(output.shape(), &[5, 7, 3]);
}

#[test]
fn gaussian_blur_preserves_shape() {
    let input = Tensor::from_vec(vec![8, 8, 3], vec![1.0; 8 * 8 * 3]).unwrap();
    let blur = GaussianBlur::new(3, 1.0);
    let output = blur.apply(&input).unwrap();
    assert_eq!(output.shape(), &[8, 8, 3]);
}
