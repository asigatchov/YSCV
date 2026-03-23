//! Optimizers and training helpers for yscv models.
#![deny(unsafe_code)]

pub const CRATE_ID: &str = "yscv-optim";

#[path = "adagrad.rs"]
mod adagrad;
#[path = "adam.rs"]
mod adam;
#[path = "adamw.rs"]
mod adamw;
#[path = "clip.rs"]
mod clip;
#[path = "error.rs"]
mod error;
#[path = "lamb.rs"]
mod lamb;
#[path = "lars.rs"]
mod lars;
#[path = "lookahead.rs"]
mod lookahead;
#[path = "lr.rs"]
mod lr;
#[path = "radam.rs"]
mod radam;
#[path = "rmsprop.rs"]
mod rmsprop;
#[path = "scheduler.rs"]
mod scheduler;
#[path = "sgd.rs"]
mod sgd;
#[path = "validate.rs"]
mod validate;

pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use clip::{clip_grad_norm_, clip_grad_value_};
pub use error::OptimError;
pub use lamb::Lamb;
pub use lars::Lars;
pub use lookahead::{Lookahead, StepOptimizer};
pub use lr::LearningRate;
pub use radam::RAdam;
pub use rmsprop::RmsProp;
pub use scheduler::{
    CosineAnnealingLr, CosineAnnealingWarmRestarts, CyclicLr, ExponentialLr, LambdaLr,
    LinearWarmupLr, LrScheduler, MultiStepLr, OneCycleLr, PolynomialDecayLr, ReduceLrOnPlateau,
    StepLr,
};
pub use sgd::Sgd;

#[path = "tests.rs"]
#[cfg(test)]
mod tests;
