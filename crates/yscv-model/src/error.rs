use thiserror::Error;
use yscv_autograd::AutogradError;
use yscv_imgproc::ImgProcError;
use yscv_kernels::KernelError;
use yscv_optim::OptimError;
use yscv_tensor::TensorError;

/// Errors returned by model-layer assembly, checkpoints, and training helpers.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ModelError {
    #[error("invalid shape for {parameter}: expected {expected:?}, got {got:?}")]
    InvalidParameterShape {
        parameter: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error(
        "invalid linear input shape: expected rank-2 with last dim {expected_features}, got {got:?}"
    )]
    InvalidInputShape {
        expected_features: usize,
        got: Vec<usize>,
    },
    #[error("invalid leaky-relu negative slope: {negative_slope}; expected finite value >= 0")]
    InvalidLeakyReluSlope { negative_slope: f32 },
    #[error("invalid dropout rate: {rate}; expected finite value in [0, 1)")]
    InvalidDropoutRate { rate: f32 },
    #[error("prediction/target shape mismatch: prediction={prediction:?}, target={target:?}")]
    PredictionTargetShapeMismatch {
        prediction: Vec<usize>,
        target: Vec<usize>,
    },
    #[error("cannot compute mean loss for empty tensor")]
    EmptyLossTensor,
    #[error("invalid huber delta: {delta}; expected finite value > 0")]
    InvalidHuberDelta { delta: f32 },
    #[error("invalid hinge margin: {margin}; expected finite value > 0")]
    InvalidHingeMargin { margin: f32 },
    #[error(
        "dataset tensors must have rank >= 1, got inputs_rank={inputs_rank}, targets_rank={targets_rank}"
    )]
    InvalidDatasetRank {
        inputs_rank: usize,
        targets_rank: usize,
    },
    #[error("dataset sample mismatch: inputs={inputs:?}, targets={targets:?}")]
    DatasetShapeMismatch {
        inputs: Vec<usize>,
        targets: Vec<usize>,
    },
    #[error("dataset is empty")]
    EmptyDataset,
    #[error("invalid batch size: {batch_size}; expected batch_size > 0")]
    InvalidBatchSize { batch_size: usize },
    #[error("invalid epoch count: {epochs}; expected epochs > 0")]
    InvalidEpochCount { epochs: usize },
    #[error(
        "invalid split ratios: train_ratio={train_ratio}, validation_ratio={validation_ratio}; expected finite values in [0, 1] with train+validation <= 1"
    )]
    InvalidSplitRatios {
        train_ratio: f32,
        validation_ratio: f32,
    },
    #[error(
        "invalid split counts: train_count={train_count}, validation_count={validation_count}, dataset_len={dataset_len}"
    )]
    InvalidSplitCounts {
        train_count: usize,
        validation_count: usize,
        dataset_len: usize,
    },
    #[error("invalid sampling weights length: expected {expected} weights, got {got}")]
    InvalidSamplingWeightsLength { expected: usize, got: usize },
    #[error("invalid sampling weight at index {index}: {value}; expected finite value >= 0")]
    InvalidSamplingWeight { index: usize, value: f32 },
    #[error("invalid sampling distribution: at least one weight must be > 0")]
    InvalidSamplingDistribution,
    #[error(
        "invalid class-balanced sampling target shape: expected scalar class labels ([N,1]) or one-hot labels ([N,C]), got {got:?}"
    )]
    InvalidClassSamplingTargetShape { got: Vec<usize> },
    #[error("invalid class-balanced sampling target at sample {index}: {value}; {reason}")]
    InvalidClassSamplingTargetValue {
        index: usize,
        value: f32,
        reason: &'static str,
    },
    #[error(
        "invalid augmentation probability for {operation}: {value}; expected finite value in [0, 1]"
    )]
    InvalidAugmentationProbability { operation: &'static str, value: f32 },
    #[error("invalid augmentation argument for {operation}: {message}")]
    InvalidAugmentationArgument {
        operation: &'static str,
        message: String,
    },
    #[error("invalid augmentation input shape: expected rank-4 NHWC, got {got:?}")]
    InvalidAugmentationInputShape { got: Vec<usize> },
    #[error("invalid mixup argument for {field}: {value}; {message}")]
    InvalidMixupArgument {
        field: &'static str,
        value: f32,
        message: String,
    },
    #[error("invalid cutmix argument for {field}: {value}; {message}")]
    InvalidCutMixArgument {
        field: &'static str,
        value: f32,
        message: String,
    },
    #[error("invalid cutmix input shape: expected rank-4 NHWC, got {got:?}")]
    InvalidCutMixInputShape { got: Vec<usize> },
    #[error("invalid dataset-adapter shape for {field}: {shape:?}; {message}")]
    InvalidDatasetAdapterShape {
        field: &'static str,
        shape: Vec<usize>,
        message: String,
    },
    #[error("invalid image-folder extension configuration for {extension}: {message}")]
    InvalidImageFolderExtension { extension: String, message: String },
    #[error("invalid CSV delimiter: {delimiter:?}; expected a non-control character")]
    InvalidCsvDelimiter { delimiter: char },
    #[error("invalid CSV dataset column count at line {line}: expected {expected}, got {got}")]
    InvalidDatasetRecordColumns {
        line: usize,
        expected: usize,
        got: usize,
    },
    #[error("invalid dataset record path at line {line}: {message}")]
    InvalidDatasetRecordPath { line: usize, message: String },
    #[error("failed to parse CSV dataset value at line {line}, column {column}: {message}")]
    DatasetCsvParse {
        line: usize,
        column: usize,
        message: String,
    },
    #[error(
        "invalid JSONL dataset record length at line {line} for {field}: expected {expected}, got {got}"
    )]
    InvalidDatasetRecordLength {
        line: usize,
        field: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid JSONL dataset record value at line {line} for {field}[{index}]: {reason}")]
    InvalidDatasetRecordValue {
        line: usize,
        field: &'static str,
        index: usize,
        reason: &'static str,
    },
    #[error("failed to parse JSONL dataset record at line {line}: {message}")]
    DatasetJsonlParse { line: usize, message: String },
    #[error("failed to read dataset file {path}: {message}")]
    DatasetLoadIo { path: String, message: String },
    #[error("failed to decode dataset image {path}: {message}")]
    DatasetImageDecode { path: String, message: String },
    #[error("invalid conv2d stride: stride_h={stride_h}, stride_w={stride_w}; both must be > 0")]
    InvalidConv2dStride { stride_h: usize, stride_w: usize },
    #[error("invalid batch-norm epsilon: {epsilon}; expected finite value > 0")]
    InvalidBatchNormEpsilon { epsilon: f32 },
    #[error("invalid pool kernel: kernel_h={kernel_h}, kernel_w={kernel_w}; both must be > 0")]
    InvalidPoolKernel { kernel_h: usize, kernel_w: usize },
    #[error("invalid pool stride: stride_h={stride_h}, stride_w={stride_w}; both must be > 0")]
    InvalidPoolStride { stride_h: usize, stride_w: usize },
    #[error("invalid flatten input shape: expected rank >= 2, got {got:?}")]
    InvalidFlattenShape { got: Vec<usize> },
    #[error("layer is inference-only and cannot be used in autograd graph forward pass")]
    InferenceOnlyLayer,
    #[error("layer {layer} parameters not registered in graph; call register_params first")]
    ParamsNotRegistered { layer: &'static str },
    #[error("layer is graph-only and cannot be used in direct tensor inference forward pass")]
    GraphOnlyLayer,
    #[error("checkpoint serialization error: {message}")]
    CheckpointSerialization { message: String },
    #[error("invalid accumulation steps: {steps}; expected steps > 0")]
    InvalidAccumulationSteps { steps: usize },
    #[error("ONNX export error: {0}")]
    OnnxExport(String),
    #[error("invalid layer index {index}: model has {count} layers")]
    InvalidLayerIndex { index: usize, count: usize },
    #[error("missing weight tensor: {name}")]
    WeightNotFound { name: String },
    #[error("safetensors parse error: {message}")]
    SafeTensorsParse { message: String },
    #[error("safetensors I/O error for {path}: {message}")]
    SafeTensorsIo { path: String, message: String },
    #[error("download failed for {url}: {reason}")]
    DownloadFailed { url: String, reason: String },
    #[error("transport error: {0}")]
    TransportError(String),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    ImgProc(#[from] ImgProcError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Optim(#[from] OptimError),
}
