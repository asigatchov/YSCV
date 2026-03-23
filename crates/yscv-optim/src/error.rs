use thiserror::Error;
use yscv_autograd::AutogradError;
use yscv_tensor::TensorError;

/// Errors returned by optimizer configuration and update steps.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptimError {
    #[error("invalid learning rate: {lr}; expected finite lr >= 0")]
    InvalidLearningRate { lr: f32 },
    #[error("invalid momentum: {momentum}; expected finite momentum in [0, 1)")]
    InvalidMomentum { momentum: f32 },
    #[error("invalid beta1: {beta1}; expected finite beta1 in [0, 1)")]
    InvalidBeta1 { beta1: f32 },
    #[error("invalid beta2: {beta2}; expected finite beta2 in [0, 1)")]
    InvalidBeta2 { beta2: f32 },
    #[error("invalid epsilon: {epsilon}; expected finite epsilon > 0")]
    InvalidEpsilon { epsilon: f32 },
    #[error("invalid rmsprop alpha: {alpha}; expected finite alpha in [0, 1)")]
    InvalidRmsPropAlpha { alpha: f32 },
    #[error("invalid step scheduler gamma: {gamma}; expected finite gamma in (0, 1]")]
    InvalidStepGamma { gamma: f32 },
    #[error("invalid step scheduler step_size: {step_size}; expected step_size > 0")]
    InvalidStepSize { step_size: usize },
    #[error("invalid cosine scheduler t_max: {t_max}; expected t_max > 0")]
    InvalidCosineTMax { t_max: usize },
    #[error("invalid warmup scheduler warmup_steps: {warmup_steps}; expected warmup_steps > 0")]
    InvalidWarmupSteps { warmup_steps: usize },
    #[error("invalid one-cycle total_steps: {total_steps}; expected total_steps > 0")]
    InvalidOneCycleTotalSteps { total_steps: usize },
    #[error("invalid one-cycle pct_start: {pct_start}; expected finite pct_start in (0, 1]")]
    InvalidOneCyclePctStart { pct_start: f32 },
    #[error(
        "invalid one-cycle final_div_factor: {final_div_factor}; expected finite final_div_factor > 1"
    )]
    InvalidOneCycleFinalDivFactor { final_div_factor: f32 },
    #[error(
        "scheduler start lr exceeds base lr: start_lr={start_lr}, base_lr={base_lr}; expected start_lr <= base_lr"
    )]
    SchedulerStartLrExceedsBase { start_lr: f32, base_lr: f32 },
    #[error(
        "scheduler max lr below initial lr: max_lr={max_lr}, initial_lr={initial_lr}; expected max_lr >= initial_lr"
    )]
    SchedulerMaxLrBelowInitial { max_lr: f32, initial_lr: f32 },
    #[error(
        "scheduler min_lr exceeds base lr: min_lr={min_lr}, base_lr={base_lr}; expected min_lr <= base_lr"
    )]
    SchedulerMinLrExceedsBase { min_lr: f32, base_lr: f32 },
    #[error("invalid dampening: {dampening}; expected finite dampening in [0, 1]")]
    InvalidDampening { dampening: f32 },
    #[error("invalid weight_decay: {weight_decay}; expected finite weight_decay >= 0")]
    InvalidWeightDecay { weight_decay: f32 },
    #[error("nesterov momentum requires momentum > 0")]
    NesterovRequiresMomentum,
    #[error("optimizer shape mismatch: weights={weights:?}, grad={grad:?}")]
    ShapeMismatch {
        weights: Vec<usize>,
        grad: Vec<usize>,
    },
    #[error("missing gradient for node {node}; call backward() first")]
    MissingGradient { node: usize },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
}
