use yscv_tensor::Tensor;

use super::error::ModelError;
use super::sequential::SequentialModel;

/// Builder-style inference pipeline that wraps a [`SequentialModel`] with
/// optional pre- and post-processing closures.
pub struct InferencePipeline {
    preprocess: Option<Box<dyn Fn(&Tensor) -> Result<Tensor, ModelError>>>,
    model: SequentialModel,
    postprocess: Option<Box<dyn Fn(&Tensor) -> Result<Tensor, ModelError>>>,
}

impl InferencePipeline {
    /// Create a new pipeline wrapping the given model with no pre/post processing.
    pub fn new(model: SequentialModel) -> Self {
        Self {
            preprocess: None,
            model,
            postprocess: None,
        }
    }

    /// Attach a preprocessing closure that transforms the input tensor before
    /// it is fed to the model.
    pub fn with_preprocess<F>(mut self, f: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Tensor, ModelError> + 'static,
    {
        self.preprocess = Some(Box::new(f));
        self
    }

    /// Attach a postprocessing closure that transforms the model output tensor
    /// before it is returned to the caller.
    pub fn with_postprocess<F>(mut self, f: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Tensor, ModelError> + 'static,
    {
        self.postprocess = Some(Box::new(f));
        self
    }

    /// Run the full pipeline on a single input tensor:
    /// preprocess (if set) -> model forward inference -> postprocess (if set).
    pub fn run(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let preprocessed = match &self.preprocess {
            Some(f) => f(input)?,
            None => input.clone(),
        };
        let output = self.model.forward_inference(&preprocessed)?;
        match &self.postprocess {
            Some(f) => f(&output),
            None => Ok(output),
        }
    }

    /// Run the pipeline on a batch of input tensors, collecting results.
    pub fn run_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, ModelError> {
        inputs.iter().map(|input| self.run(input)).collect()
    }
}
