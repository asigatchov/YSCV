use super::OptimError;

/// Shared learning-rate control surface for optimizers.
pub trait LearningRate {
    /// Returns current optimizer learning rate.
    fn learning_rate(&self) -> f32;

    /// Sets optimizer learning rate after validation.
    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError>;
}
