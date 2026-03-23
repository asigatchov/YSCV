use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum EvalError {
    #[error("invalid IoU threshold: {value}; expected finite threshold in [0, 1]")]
    InvalidIouThreshold { value: f32 },
    #[error("invalid score threshold: {value}; expected finite threshold in [0, 1]")]
    InvalidScoreThreshold { value: f32 },
    #[error(
        "count sequence length mismatch: ground_truth={ground_truth}, predictions={predictions}"
    )]
    CountLengthMismatch {
        ground_truth: usize,
        predictions: usize,
    },
    #[error("duration series must contain at least one item")]
    EmptyDurationSeries,
    #[error("duration series length mismatch for `{series}`: expected {expected}, got {got}")]
    DurationSeriesLengthMismatch {
        expected: usize,
        got: usize,
        series: &'static str,
    },
    #[error("invalid threshold entry at line {line}: {message}")]
    InvalidThresholdEntry { line: usize, message: String },
    #[error("invalid dataset entry at line {line}: {message}")]
    InvalidDatasetEntry { line: usize, message: String },
    #[error("invalid dataset format `{format}`: {message}")]
    InvalidDatasetFormat {
        format: &'static str,
        message: String,
    },
    #[error("dataset I/O error for `{path}`: {message}")]
    DatasetIo { path: String, message: String },
    #[error("invalid diagnostics report: {message}")]
    InvalidDiagnosticsReport { message: String },
    #[error("diagnostics report I/O error for `{path}`: {message}")]
    DiagnosticsReportIo { path: String, message: String },
}
