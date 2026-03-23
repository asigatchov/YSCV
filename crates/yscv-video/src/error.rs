use thiserror::Error;
use yscv_tensor::TensorError;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VideoError {
    #[error("invalid frame tensor shape: expected [H, W, C], got {got:?}")]
    InvalidFrameShape { got: Vec<usize> },
    #[error("unsupported channel count: {channels}; expected 1 or 3")]
    UnsupportedChannelCount { channels: usize },
    #[error(
        "invalid camera resolution: width={width}, height={height}; expected width > 0 and height > 0"
    )]
    InvalidCameraResolution { width: u32, height: u32 },
    #[error("invalid camera fps: {fps}; expected fps > 0")]
    InvalidCameraFps { fps: u32 },
    #[error("invalid camera device query `{query}`; expected non-empty value")]
    InvalidCameraDeviceQuery { query: String },
    #[error("no camera device matched query `{query}`; run device listing to inspect names")]
    CameraDeviceNotFound { query: String },
    #[error(
        "camera query `{query}` matched multiple devices: {}; refine query",
        matches.join(", ")
    )]
    CameraDeviceAmbiguous { query: String, matches: Vec<String> },
    #[error("invalid raw frame buffer size: expected {expected} bytes, got {got}")]
    RawFrameSizeMismatch { expected: usize, got: usize },
    #[error("invalid normalized output buffer size: expected {expected} f32 values, got {got}")]
    NormalizedBufferSizeMismatch { expected: usize, got: usize },
    #[error("native camera backend is disabled; enable `yscv-video` feature `native-camera`")]
    CameraBackendDisabled,
    #[error("frame source error: {0}")]
    Source(String),
    #[error("codec error: {0}")]
    Codec(String),
    #[error("container parse error: {0}")]
    ContainerParse(String),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}
