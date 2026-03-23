use thiserror::Error;

/// Errors returned by ONNX model loading and conversion.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OnnxError {
    #[error("failed to decode ONNX protobuf: {message}")]
    DecodeFailed { message: String },
    #[error("ONNX model has no graph")]
    MissingGraph,
    #[error("unsupported ONNX data type: {data_type}")]
    UnsupportedDataType { data_type: i32 },
    #[error(
        "tensor shape mismatch in initializer '{name}': expected {expected} elements, got {got}"
    )]
    InitializerShapeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("unsupported ONNX op type: {op_type}")]
    UnsupportedOpType { op_type: String },
    #[error("missing input '{input}' for node '{node}'")]
    MissingInput { node: String, input: String },
    #[error("I/O error: {message}")]
    Io { message: String },
    #[error("shape mismatch: {detail}")]
    ShapeMismatch { detail: String },
    #[error("missing required attribute '{attr}' on node '{node}'")]
    MissingAttribute { node: String, attr: String },
}
