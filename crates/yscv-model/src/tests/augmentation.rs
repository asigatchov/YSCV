use std::sync::Arc;

use yscv_tensor::Tensor;

use crate::{ImageAugmentationOp, ImageAugmentationPipeline, ModelError, SupervisedDataset};

use super::assert_slice_approx_eq;

#[test]
fn augmentation_pipeline_rejects_invalid_probability() {
    let err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::HorizontalFlip {
        probability: 1.5,
    }])
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationProbability {
            operation: "horizontal_flip",
            value: 1.5
        }
    );

    let rotate_err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::RandomRotate90 {
        probability: -0.1,
    }])
    .unwrap_err();
    assert_eq!(
        rotate_err,
        ModelError::InvalidAugmentationProbability {
            operation: "random_rotate90",
            value: -0.1
        }
    );
}

#[test]
fn augmentation_pipeline_rejects_invalid_contrast_jitter() {
    let err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::ContrastJitter {
        max_scale_delta: -0.1,
    }])
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationArgument {
            operation: "contrast_jitter",
            message: "max_scale_delta must be finite and >= 0, got -0.1".to_string()
        }
    );
}

#[test]
fn augmentation_pipeline_rejects_invalid_gamma_jitter() {
    let err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::GammaJitter {
        max_gamma_delta: -0.1,
    }])
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationArgument {
            operation: "gamma_jitter",
            message: "max_gamma_delta must be finite and >= 0, got -0.1".to_string()
        }
    );
}

#[test]
fn augmentation_pipeline_rejects_invalid_gaussian_noise_arguments() {
    let err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::GaussianNoise {
        probability: 1.0,
        std_dev: -0.1,
    }])
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationArgument {
            operation: "gaussian_noise",
            message: "std_dev must be finite and >= 0, got -0.1".to_string()
        }
    );
}

#[test]
fn augmentation_pipeline_rejects_invalid_cutout_arguments() {
    let err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::Cutout {
        probability: 1.0,
        max_height_fraction: 0.0,
        max_width_fraction: 0.5,
        fill_value: 0.0,
    }])
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationArgument {
            operation: "cutout",
            message: "max_height_fraction must be finite in (0, 1], got 0".to_string()
        }
    );
}

#[test]
fn augmentation_pipeline_rejects_invalid_random_resized_crop_arguments() {
    let err = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::RandomResizedCrop {
        probability: 1.0,
        min_scale: 0.8,
        max_scale: 0.2,
    }])
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationArgument {
            operation: "random_resized_crop",
            message: "min_scale must be <= max_scale, got min_scale=0.8, max_scale=0.2".to_string()
        }
    );
}

#[test]
fn augmentation_pipeline_requires_rank4_nhwc() {
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::HorizontalFlip {
        probability: 1.0,
    }])
    .unwrap();
    let input = Tensor::from_vec(vec![2, 2, 3], vec![0.0; 12]).unwrap();
    let err = pipeline.apply_nhwc(&input, 7).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationInputShape { got: vec![2, 2, 3] }
    );
}

#[test]
fn augmentation_pipeline_applies_flips_and_preserves_targets() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![1, 2, 2, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![1, 1], vec![9.0]).unwrap(),
    )
    .unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![
        ImageAugmentationOp::HorizontalFlip { probability: 1.0 },
        ImageAugmentationOp::VerticalFlip { probability: 1.0 },
    ])
    .unwrap();

    let augmented = dataset.augment_nhwc(&pipeline, 42).unwrap();
    assert_eq!(augmented.targets().data(), dataset.targets().data());
    assert_eq!(augmented.inputs().shape(), &[1, 2, 2, 1]);
    assert_eq!(augmented.inputs().data(), &[3.0, 2.0, 1.0, 0.0]);
}

#[test]
fn augmentation_pipeline_random_rotate90_is_seed_deterministic_for_square_inputs() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::RandomRotate90 {
        probability: 1.0,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 17).unwrap();
    let b = pipeline.apply_nhwc(&input, 17).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.shape(), input.shape());

    let candidates = [
        vec![0.0, 1.0, 2.0, 3.0], // 0 deg
        vec![2.0, 0.0, 3.0, 1.0], // 90 deg
        vec![3.0, 2.0, 1.0, 0.0], // 180 deg
        vec![1.0, 3.0, 0.0, 2.0], // 270 deg
    ];
    assert!(candidates.iter().any(|candidate| candidate == a.data()));
}

#[test]
fn augmentation_pipeline_random_rotate90_preserves_non_square_shape() {
    let input = Tensor::from_vec(vec![1, 2, 3, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::RandomRotate90 {
        probability: 1.0,
    }])
    .unwrap();

    let out = pipeline.apply_nhwc(&input, 9).unwrap();
    assert_eq!(out.shape(), input.shape());

    let candidates = [
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], // 0 deg
        vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0], // 180 deg
    ];
    assert!(candidates.iter().any(|candidate| candidate == out.data()));
}

#[test]
fn augmentation_pipeline_brightness_jitter_is_seed_deterministic() {
    let input = Tensor::from_vec(vec![2, 2, 2, 1], vec![0.2; 8]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::BrightnessJitter {
        max_delta: 0.2,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 99).unwrap();
    let b = pipeline.apply_nhwc(&input, 99).unwrap();
    let c = pipeline.apply_nhwc(&input, 100).unwrap();

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn augmentation_pipeline_contrast_jitter_is_seed_deterministic() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.2, 0.4, 0.6, 0.8]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::ContrastJitter {
        max_scale_delta: 0.4,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 17).unwrap();
    let b = pipeline.apply_nhwc(&input, 17).unwrap();
    let c = pipeline.apply_nhwc(&input, 18).unwrap();

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn augmentation_pipeline_gamma_jitter_is_seed_deterministic() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.1, 0.3, 0.7, 0.9]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::GammaJitter {
        max_gamma_delta: 0.4,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 31).unwrap();
    let b = pipeline.apply_nhwc(&input, 31).unwrap();
    let c = pipeline.apply_nhwc(&input, 32).unwrap();

    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a.data().iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn augmentation_pipeline_gaussian_noise_is_seed_deterministic() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.5; 4]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::GaussianNoise {
        probability: 1.0,
        std_dev: 0.1,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 211).unwrap();
    let b = pipeline.apply_nhwc(&input, 211).unwrap();
    let c = pipeline.apply_nhwc(&input, 212).unwrap();

    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a.data().iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn augmentation_pipeline_gaussian_noise_clamps_to_unit_range() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::GaussianNoise {
        probability: 1.0,
        std_dev: 10.0,
    }])
    .unwrap();

    let output = pipeline.apply_nhwc(&input, 7).unwrap();
    assert!(
        output
            .data()
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
    );
}

#[test]
fn augmentation_pipeline_cutout_is_seed_deterministic() {
    let input = Tensor::from_vec(
        vec![1, 4, 4, 1],
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0,
        ],
    )
    .unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::Cutout {
        probability: 1.0,
        max_height_fraction: 0.5,
        max_width_fraction: 0.5,
        fill_value: -1.0,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 123).unwrap();
    let b = pipeline.apply_nhwc(&input, 123).unwrap();

    assert_eq!(a, b);
    assert_eq!(a.shape(), input.shape());
    assert!(a.data().iter().any(|value| *value == -1.0));
}

#[test]
fn augmentation_pipeline_box_blur_changes_values_when_applied() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 9.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let pipeline =
        ImageAugmentationPipeline::new(vec![ImageAugmentationOp::BoxBlur3x3 { probability: 1.0 }])
            .unwrap();

    let output = pipeline.apply_nhwc(&input, 0).unwrap();
    assert_eq!(output.shape(), input.shape());
    assert_ne!(output, input);
}

#[test]
fn augmentation_pipeline_random_resized_crop_preserves_shape_and_is_seed_deterministic() {
    let input = Tensor::from_vec(
        vec![1, 4, 4, 1],
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0,
        ],
    )
    .unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::RandomResizedCrop {
        probability: 1.0,
        min_scale: 0.5,
        max_scale: 0.5,
    }])
    .unwrap();

    let a = pipeline.apply_nhwc(&input, 77).unwrap();
    let b = pipeline.apply_nhwc(&input, 77).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.shape(), input.shape());
    assert_ne!(a, input);
}

#[test]
fn augmentation_pipeline_channel_normalize_matches_expected_values() {
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![0.5, 0.8, 0.3, 0.6]).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::ChannelNormalize {
        mean: vec![0.4, 0.5],
        std: vec![0.2, 0.1],
    }])
    .unwrap();
    let output = pipeline.apply_nhwc(&input, 0).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_slice_approx_eq(output.data(), &[0.5, 3.0, -0.5, 1.0], 1e-5);
}

#[test]
fn custom_augmentation_op() {
    // Custom closure that doubles every value.
    let op = ImageAugmentationOp::Custom(Arc::new(|input: &Tensor| {
        let doubled: Vec<f32> = input.data().iter().map(|v| v * 2.0).collect();
        Tensor::from_vec(input.shape().to_vec(), doubled).map_err(Into::into)
    }));
    let pipeline = ImageAugmentationPipeline::new(vec![op]).unwrap();
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    let output = pipeline.apply_nhwc(&input, 0).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    assert_slice_approx_eq(output.data(), &[0.2, 0.4, 0.6, 0.8], 1e-6);
}

#[test]
fn random_crop_reduces_size() {
    // 1 sample of 4x4x1, crop to 2x2.
    let input = Tensor::from_vec(vec![1, 4, 4, 1], (0..16).map(|i| i as f32).collect()).unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::RandomCrop {
        height: 2,
        width: 2,
    }])
    .unwrap();
    let output = pipeline.apply_nhwc(&input, 42).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    // Output should contain 4 values that are a subset of the original 16.
    assert_eq!(output.data().len(), 4);
    for v in output.data() {
        assert!(*v >= 0.0 && *v < 16.0);
    }
}

#[test]
fn gaussian_blur_preserves_shape() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let pipeline =
        ImageAugmentationPipeline::new(vec![ImageAugmentationOp::GaussianBlur { kernel_size: 3 }])
            .unwrap();
    let output = pipeline.apply_nhwc(&input, 0).unwrap();
    assert_eq!(output.shape(), input.shape());
    // The center impulse should be spread out, so the center value decreases.
    assert!(output.data()[4] < 1.0);
    // Total energy is roughly conserved (sum should be close to 1.0).
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 0.1);
}

#[test]
fn from_transform_bridge() {
    // Use ScaleValues transform via the bridge.
    let scale = crate::ScaleValues::new(3.0);
    let op = ImageAugmentationOp::from_transform(scale);
    let pipeline = ImageAugmentationPipeline::new(vec![op]).unwrap();
    let input = Tensor::from_vec(vec![1, 2, 1, 1], vec![1.0, 2.0]).unwrap();
    let output = pipeline.apply_nhwc(&input, 0).unwrap();
    assert_eq!(output.shape(), &[1, 2, 1, 1]);
    assert_slice_approx_eq(output.data(), &[3.0, 6.0], 1e-6);
}
