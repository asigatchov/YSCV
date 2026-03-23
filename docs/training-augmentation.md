# yscv Training Augmentation Pipeline

This page documents training-time augmentation and batch regularization currently provided by `yscv-model`.

## Scope
- Surface: `ImageAugmentationPipeline`, `ImageAugmentationOp`, `SupervisedDataset::augment_nhwc`, `MixUpConfig`, `CutMixConfig`.
- Batch/data surface: `BatchIterOptions`, `SamplingPolicy`, `SupervisedDataset::batches_with_options`, `SupervisedDataset::split_by_counts`, `SupervisedDataset::split_by_ratio`, `SupervisedDataset::split_by_class_ratio`.
- Intended tensor layout: rank-4 `NHWC` (`[batch, height, width, channels]`).
- Goal: deterministic, reproducible augmentation for supervised training workflows.

## Supported Operations
- `HorizontalFlip { probability }`
  - Per-sample horizontal mirror with probability in `[0, 1]`.
- `VerticalFlip { probability }`
  - Per-sample vertical mirror with probability in `[0, 1]`.
- `RandomRotate90 { probability }`
  - Rotates sample by random multiples of 90 degrees with probability in `[0, 1]`.
  - Square samples use `{0, 90, 180, 270}` degrees; non-square samples use `{0, 180}` to preserve shape.
- `BrightnessJitter { max_delta }`
  - Adds random uniform delta in `[-max_delta, +max_delta]` and clamps output to `[0, 1]`.
- `ContrastJitter { max_scale_delta }`
  - Scales contrast around per-sample mean by factor in `[1-max_scale_delta, 1+max_scale_delta]`.
- `GammaJitter { max_gamma_delta }`
  - Applies gamma correction with gamma sampled in `[1-max_gamma_delta, 1+max_gamma_delta]`.
- `GaussianNoise { probability, std_dev }`
  - Adds Gaussian noise sampled with standard deviation `std_dev` when Bernoulli trial succeeds.
  - Output values are clamped to `[0, 1]`.
- `BoxBlur3x3 { probability }`
  - Applies deterministic 3x3 box blur to a sample when Bernoulli trial succeeds.
- `RandomResizedCrop { probability, min_scale, max_scale }`
  - Crops a random window from the sample and resizes it back to original `H x W`.
  - Scale range is deterministic by seed and constrained by `min_scale..=max_scale`.
- `Cutout { probability, max_height_fraction, max_width_fraction, fill_value }`
  - Applies random rectangular erasing with deterministic seed control.
  - Rectangle size is sampled per sample, bounded by configured max height/width fractions.
- `ChannelNormalize { mean, std }`
  - Per-channel normalization in HWC layout: `(x - mean[c]) / std[c]`.

## Validation Rules
- Pipeline construction validates operation arguments:
  - flip probabilities must be finite and in `[0, 1]`,
  - random-rotate90 probability must be finite and in `[0, 1]`,
  - `max_delta` must be finite and `>= 0`,
  - `max_scale_delta` must be finite and `>= 0`,
  - `max_gamma_delta` must be finite and `>= 0`,
  - gaussian-noise probability must be finite and in `[0, 1]`,
  - gaussian-noise `std_dev` must be finite and `>= 0`,
  - `box_blur_3x3` probability must be finite and in `[0, 1]`,
  - random-resized-crop probability must be finite and in `[0, 1]`,
  - random-resized-crop `min_scale`/`max_scale` must be finite in `(0, 1]` with `min_scale <= max_scale`,
  - cutout `max_height_fraction` and `max_width_fraction` must be finite in `(0, 1]`,
  - cutout `fill_value` must be finite,
  - `mean/std` must be non-empty and have identical lengths,
  - each `mean` value must be finite,
  - each `std` value must be finite and `> 0`.
- Applying pipeline validates input tensor shape:
  - input must be rank-4 `NHWC`.
- Augmentations must preserve per-sample shape.

## Reproducibility Contract
- `apply_nhwc` and `augment_nhwc` accept a `seed: u64`.
- Same seed + same input + same op list => deterministic output.
- Shuffled mini-batches are deterministic by `BatchIterOptions.shuffle_seed`.
- Batch-level augmentation is deterministic by `BatchIterOptions.augmentation_seed`.
- `SamplingPolicy`-driven order is deterministic by policy-provided seed.

## Mini-Batch Data Pipeline Controls
- `BatchIterOptions`:
  - `shuffle`: deterministic sample-order shuffle toggle.
  - `shuffle_seed`: seed for deterministic shuffle order.
  - `sampling`: optional explicit policy override (`Sequential`, `Shuffled`, `BalancedByClass`, `Weighted`).
  - `drop_last`: drop trailing incomplete batch.
  - `augmentation`: optional `ImageAugmentationPipeline` applied to batch inputs.
  - `augmentation_seed`: deterministic seed base for per-batch augmentation RNG.
  - `mixup`: optional `MixUpConfig` for per-batch sample/target interpolation.
  - `mixup_seed`: deterministic seed base for per-batch mixup pairing/lambda.
  - `cutmix`: optional `CutMixConfig` for per-batch patch replacement interpolation.
  - `cutmix_seed`: deterministic seed base for per-batch CutMix pairing/patch sampling.

## Batch MixUp
- `MixUpConfig`:
  - `probability`: apply mixup for a batch with probability in `[0, 1]`.
  - `lambda_min`: lower bound for interpolation weight in `[0, 0.5]`.
- Runtime behavior:
  - when enabled, both inputs and targets are mixed using the same sample pairing and lambda,
  - batch shape and tensor rank are preserved,
  - same seed + same batch order => deterministic mixed outputs.

## Batch CutMix
- `CutMixConfig`:
  - `probability`: apply cutmix for a batch with probability in `[0, 1]`.
  - `min_patch_fraction`: lower bound for square patch side fraction in `[0, 1]`.
  - `max_patch_fraction`: upper bound for square patch side fraction in `[0, 1]` and `>= min_patch_fraction`.
- Runtime behavior:
  - cutmix expects rank-4 `NHWC` batch inputs,
  - for each sample, a partner sample is selected and a random rectangular patch is copied from partner input,
  - target rows are mixed with lambda derived from replaced patch area ratio,
  - same seed + same batch order => deterministic cutmix outputs.

## Sampling Policies
- `SamplingPolicy::Sequential`
  - emits dataset indices in natural order (`0..len`).
- `SamplingPolicy::Shuffled { seed }`
  - deterministic seeded shuffle of full epoch order.
- `SamplingPolicy::BalancedByClass { seed, with_replacement }`
  - derives per-sample weights from inverse class frequency using:
    - scalar class labels from `targets` shape `[N, 1]`, or
    - one-hot class labels from `targets` shape `[N, C]`,
  - scalar class labels must be finite non-negative integers,
  - one-hot rows must contain exactly one active class and values close to `0` or `1`,
  - `with_replacement=true` draws `dataset_len` balanced samples with replacement,
  - `with_replacement=false` produces a deterministic class-balanced weighted order without replacement.
- `SamplingPolicy::Weighted { weights, seed, with_replacement }`
  - validates `weights.len() == dataset_len`,
  - each weight must be finite and `>= 0`,
  - at least one weight must be `> 0`,
  - `with_replacement=true` samples `dataset_len` draws from weighted distribution,
  - `with_replacement=false` produces deterministic weighted order without replacement.

## Dataset Split API
- `split_by_counts(train_count, validation_count, shuffle, seed)`:
  - returns `DatasetSplit { train, validation, test }`,
  - validates `train_count + validation_count <= dataset_len`.
- `split_by_ratio(train_ratio, validation_ratio, shuffle, seed)`:
  - expects finite ratios in `[0, 1]`,
  - requires `train_ratio + validation_ratio <= 1`,
  - test split receives remaining samples.
- `split_by_class_ratio(train_ratio, validation_ratio, shuffle, seed)`:
  - applies split ratios independently per class label and merges class-wise partitions into final train/validation/test subsets,
  - supports scalar class labels (`targets` shape `[N, 1]`) and one-hot class labels (`targets` shape `[N, C]`),
  - keeps deterministic behavior under identical seed/configuration.

## API Example
```rust
use yscv_model::{
    BatchIterOptions, CutMixConfig, ImageAugmentationOp, ImageAugmentationPipeline, MixUpConfig,
    SamplingPolicy, SupervisedDataset,
};
use yscv_tensor::Tensor;

let dataset = SupervisedDataset::new(
    Tensor::from_vec(vec![2, 2, 2, 3], vec![0.2; 24])?,
    Tensor::from_vec(vec![2, 1], vec![1.0, 0.0])?,
)?;

let pipeline = ImageAugmentationPipeline::new(vec![
    ImageAugmentationOp::HorizontalFlip { probability: 0.5 },
    ImageAugmentationOp::RandomRotate90 { probability: 0.3 },
    ImageAugmentationOp::BrightnessJitter { max_delta: 0.1 },
    ImageAugmentationOp::ContrastJitter { max_scale_delta: 0.2 },
    ImageAugmentationOp::GammaJitter { max_gamma_delta: 0.25 },
    ImageAugmentationOp::GaussianNoise {
        probability: 0.2,
        std_dev: 0.03,
    },
    ImageAugmentationOp::BoxBlur3x3 { probability: 0.2 },
    ImageAugmentationOp::RandomResizedCrop {
        probability: 0.4,
        min_scale: 0.6,
        max_scale: 1.0,
    },
    ImageAugmentationOp::Cutout {
        probability: 0.3,
        max_height_fraction: 0.2,
        max_width_fraction: 0.2,
        fill_value: 0.0,
    },
    ImageAugmentationOp::ChannelNormalize {
        mean: vec![0.485, 0.456, 0.406],
        std: vec![0.229, 0.224, 0.225],
    },
])?;

let augmented = dataset.augment_nhwc(&pipeline, 12345)?;
assert_eq!(augmented.targets().shape(), dataset.targets().shape());

let split = augmented.split_by_ratio(0.7, 0.15, true, 7)?;
let mixup = MixUpConfig::new()
    .with_probability(0.5)?
    .with_lambda_min(0.1)?;
let cutmix = CutMixConfig::new()
    .with_probability(0.3)?
    .with_min_patch_fraction(0.2)?
    .with_max_patch_fraction(0.5)?;
let batches = split.train.batches_with_options(
    8,
    BatchIterOptions {
        shuffle: true,
        shuffle_seed: 77,
        sampling: Some(SamplingPolicy::Weighted {
            weights: vec![1.0; split.train.len()],
            seed: 5,
            with_replacement: true,
        }),
        drop_last: true,
        augmentation: Some(pipeline.clone()),
        augmentation_seed: 777,
        mixup: Some(mixup),
        mixup_seed: 1777,
        cutmix: Some(cutmix),
        cutmix_seed: 2777,
    },
)?;

for batch in batches {
    assert_eq!(batch.inputs.rank(), 4);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Compose Transform Pipeline

In addition to training-time augmentation, `yscv-model` provides a `Compose`-based transform pipeline (analogous to `torchvision.transforms`) for inference and preprocessing:

- `Normalize { mean, std }` — per-channel normalization.
- `ScaleValues { scale }` — multiply all values by a scalar (e.g., `1.0/255.0`).
- `PermuteDims { order }` — reorder tensor dimensions (e.g., HWC → CHW).
- `Resize { height, width }` — nearest-neighbor resize.
- `CenterCrop { height, width }` — center crop to target size.
- `RandomHorizontalFlip { probability }` — random horizontal flip.
- `GaussianBlur { kernel_size, sigma }` — Gaussian blur.

Usage:
```rust
use yscv_model::{Transform, Compose};

let transform = Compose::new(vec![
    Transform::Resize { height: 224, width: 224 },
    Transform::CenterCrop { height: 224, width: 224 },
    Transform::ScaleValues { scale: 1.0 / 255.0 },
    Transform::Normalize {
        mean: vec![0.485, 0.456, 0.406],
        std: vec![0.229, 0.224, 0.225],
    },
]);

let output = transform.apply(&input_tensor)?;
```

## DataLoader

`yscv-model` also provides `DataLoader` for production data pipelines:
- Configurable batch size and drop-last behavior.
- Samplers: `RandomSampler`, `SequentialSampler`, `WeightedSampler`.
- Optional prefetch for overlapping data loading with training.

## Current Limits
- Augmentation path currently targets rank-4 `NHWC` tensors only.
- No multi-worker dataset sharding/streaming execution yet.
