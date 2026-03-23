use yscv_tensor::Tensor;

use super::RecognizeError;
use super::validate::{validate_embedding, validate_embedding_slice};

#[allow(unsafe_code)]
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32, RecognizeError> {
    validate_embedding(a)?;
    validate_embedding(b)?;
    cosine_similarity_prevalidated(a.data(), b.data())
}

#[allow(unsafe_code)]
pub fn cosine_similarity_slice(a: &[f32], b: &[f32]) -> Result<f32, RecognizeError> {
    validate_embedding_slice(a)?;
    validate_embedding_slice(b)?;
    cosine_similarity_prevalidated(a, b)
}

#[allow(unsafe_code)]
pub(crate) fn cosine_similarity_prevalidated(a: &[f32], b: &[f32]) -> Result<f32, RecognizeError> {
    if a.len() != b.len() {
        return Err(RecognizeError::EmbeddingDimMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(RecognizeError::EmptyEmbedding);
    }

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // SAFETY:
    // - `a_ptr` and `b_ptr` originate from slices of equal length `len`.
    // - Loop bounds ensure all pointer offsets are in-bounds.
    // - We perform read-only access, so there is no aliasing mutation.
    unsafe {
        let mut i = 0usize;
        while i + 4 <= len {
            let a0 = *a_ptr.add(i);
            let b0 = *b_ptr.add(i);
            dot += a0 * b0;
            norm_a += a0 * a0;
            norm_b += b0 * b0;

            let a1 = *a_ptr.add(i + 1);
            let b1 = *b_ptr.add(i + 1);
            dot += a1 * b1;
            norm_a += a1 * a1;
            norm_b += b1 * b1;

            let a2 = *a_ptr.add(i + 2);
            let b2 = *b_ptr.add(i + 2);
            dot += a2 * b2;
            norm_a += a2 * a2;
            norm_b += b2 * b2;

            let a3 = *a_ptr.add(i + 3);
            let b3 = *b_ptr.add(i + 3);
            dot += a3 * b3;
            norm_a += a3 * a3;
            norm_b += b3 * b3;

            i += 4;
        }
        while i < len {
            let av = *a_ptr.add(i);
            let bv = *b_ptr.add(i);
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
            i += 1;
        }
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}
