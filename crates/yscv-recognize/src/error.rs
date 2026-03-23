use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum RecognizeError {
    #[error("invalid embedding shape: expected rank-1, got {got:?}")]
    InvalidEmbeddingShape { got: Vec<usize> },
    #[error("embedding must contain at least one element")]
    EmptyEmbedding,
    #[error("embedding contains non-finite value at index {index}")]
    NonFiniteEmbeddingValue { index: usize },
    #[error("invalid recognition threshold: {value}; expected finite threshold in [-1, 1]")]
    InvalidThreshold { value: f32 },
    #[error("identity already enrolled: {id}")]
    DuplicateIdentity { id: String },
    #[error("embedding dim mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },
    #[error("recognizer serialization error: {message}")]
    Serialization { message: String },
    #[error("recognizer IO error: {message}")]
    Io { message: String },
}
