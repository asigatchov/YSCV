use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    Config(#[from] crate::config::CliError),
    #[error(transparent)]
    Video(#[from] yscv_video::VideoError),
    #[error(transparent)]
    Detect(#[from] yscv_detect::DetectError),
    #[error(transparent)]
    Eval(#[from] yscv_eval::EvalError),
    #[error(transparent)]
    Recognize(#[from] yscv_recognize::RecognizeError),
    #[error(transparent)]
    Tensor(#[from] yscv_tensor::TensorError),
    #[error(transparent)]
    Track(#[from] yscv_track::TrackError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
