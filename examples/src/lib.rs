//! Unified yscv facade crate and examples.
//!
//! This crate re-exports all public yscv crate APIs and provides a unified
//! error type [`RustcvError`] that wraps all domain-specific errors.
//!
//! Run examples with: `cargo run --example <name>`

use thiserror::Error;

/// Unified error type wrapping all yscv crate errors.
///
/// Allows `?` to propagate any yscv error through a single type.
#[derive(Debug, Error)]
pub enum RustcvError {
    #[error(transparent)]
    Tensor(#[from] yscv_tensor::TensorError),
    #[error(transparent)]
    Autograd(#[from] yscv_autograd::AutogradError),
    #[error(transparent)]
    Model(#[from] yscv_model::ModelError),
    #[error(transparent)]
    ImgProc(#[from] yscv_imgproc::ImgProcError),
    #[error(transparent)]
    Detect(#[from] yscv_detect::DetectError),
    #[error(transparent)]
    Eval(#[from] yscv_eval::EvalError),
    #[error(transparent)]
    Video(#[from] yscv_video::VideoError),
    #[error(transparent)]
    Track(#[from] yscv_track::TrackError),
    #[error(transparent)]
    Optim(#[from] yscv_optim::OptimError),
    #[error(transparent)]
    Kernel(#[from] yscv_kernels::KernelError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience type alias for functions returning `RustcvError`.
pub type Result<T> = std::result::Result<T, RustcvError>;
