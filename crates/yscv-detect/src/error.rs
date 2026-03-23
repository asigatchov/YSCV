use thiserror::Error;
use yscv_tensor::TensorError;
use yscv_video::VideoError;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DetectError {
    #[error("invalid map shape: expected rank {expected_rank}, got {got:?}")]
    InvalidMapShape {
        expected_rank: usize,
        got: Vec<usize>,
    },
    #[error("invalid channel count: expected {expected}, got {got}")]
    InvalidChannelCount { expected: usize, got: usize },
    #[error("invalid threshold: {threshold}; expected finite threshold in [0, 1]")]
    InvalidThreshold { threshold: f32 },
    #[error("invalid iou_threshold: {iou_threshold}; expected finite threshold in [0, 1]")]
    InvalidIouThreshold { iou_threshold: f32 },
    #[error("invalid min_area: {min_area}; expected min_area > 0")]
    InvalidMinArea { min_area: usize },
    #[error("invalid max_detections: {max_detections}; expected max_detections > 0")]
    InvalidMaxDetections { max_detections: usize },
    #[error("rgb8 frame dimensions overflow for width={width}, height={height}")]
    Rgb8DimensionsOverflow { width: usize, height: usize },
    #[error("invalid rgb8 frame buffer size: expected {expected} bytes, got {got}")]
    InvalidRgb8BufferSize { expected: usize, got: usize },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Video(#[from] VideoError),
    #[cfg(feature = "onnx")]
    #[error("onnx error: {0}")]
    Onnx(#[from] yscv_onnx::OnnxError),
}
