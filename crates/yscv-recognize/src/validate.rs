use yscv_tensor::Tensor;

use super::RecognizeError;

pub(crate) fn validate_threshold(threshold: f32) -> Result<(), RecognizeError> {
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err(RecognizeError::InvalidThreshold { value: threshold });
    }
    Ok(())
}

pub(crate) fn validate_embedding(embedding: &Tensor) -> Result<(), RecognizeError> {
    if embedding.rank() != 1 {
        return Err(RecognizeError::InvalidEmbeddingShape {
            got: embedding.shape().to_vec(),
        });
    }
    validate_embedding_slice(embedding.data())
}

pub(crate) fn validate_embedding_slice(embedding: &[f32]) -> Result<(), RecognizeError> {
    if embedding.is_empty() {
        return Err(RecognizeError::EmptyEmbedding);
    }
    for (index, value) in embedding.iter().enumerate() {
        if !value.is_finite() {
            return Err(RecognizeError::NonFiniteEmbeddingValue { index });
        }
    }
    Ok(())
}
