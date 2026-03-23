use yscv_kernels::{matmul_2d, softmax_last_dim};
use yscv_tensor::Tensor;

use crate::ModelError;

/// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
///
/// Q, K, V all have shape `[seq_len, d_model]` (single head) or
/// `[num_heads * seq_len, d_k]` (multi-head, pre-split).
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor, ModelError> {
    let q_shape = query.shape();
    let k_shape = key.shape();
    if q_shape.len() != 2 || k_shape.len() != 2 {
        return Err(ModelError::InvalidParameterShape {
            parameter: "attention QKV",
            expected: vec![0, 0],
            got: q_shape.to_vec(),
        });
    }
    let d_k = q_shape[1] as f32;

    let kt = key.transpose_2d()?;
    let scores = matmul_2d(query, &kt)?;
    let scale = 1.0 / d_k.sqrt();
    let scaled = scores.scale(scale);
    let attn_weights = softmax_last_dim(&scaled)?;
    let output = matmul_2d(&attn_weights, value)?;
    Ok(output)
}

/// Multi-head attention configuration.
pub struct MultiHeadAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
}

/// Multi-head attention weights.
pub struct MultiHeadAttention {
    pub w_q: Tensor, // [d_model, d_model]
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor, // [d_model, d_model]
    pub num_heads: usize,
    pub d_k: usize,
}

impl MultiHeadAttention {
    /// Creates zero-initialized multi-head attention weights.
    pub fn new(config: &MultiHeadAttentionConfig) -> Result<Self, ModelError> {
        let d = config.d_model;
        let h = config.num_heads;
        if !d.is_multiple_of(h) {
            return Err(ModelError::InvalidParameterShape {
                parameter: "d_model must be divisible by num_heads",
                expected: vec![d, h],
                got: vec![d % h],
            });
        }
        let d_k = d / h;
        let z = vec![0.0f32; d * d];
        Ok(Self {
            w_q: Tensor::from_vec(vec![d, d], z.clone())?,
            w_k: Tensor::from_vec(vec![d, d], z.clone())?,
            w_v: Tensor::from_vec(vec![d, d], z.clone())?,
            w_o: Tensor::from_vec(vec![d, d], z)?,
            num_heads: h,
            d_k,
        })
    }

    /// Forward pass: input `[seq_len, d_model]` -> output `[seq_len, d_model]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        let _seq_len = shape[0];
        let _d_model = shape[1];

        let q = matmul_2d(input, &self.w_q)?;
        let k = matmul_2d(input, &self.w_k)?;
        let v = matmul_2d(input, &self.w_v)?;

        let mut head_outputs = Vec::new();
        for h in 0..self.num_heads {
            let start = h * self.d_k;
            let qh = q.narrow(1, start, self.d_k)?;
            let kh = k.narrow(1, start, self.d_k)?;
            let vh = v.narrow(1, start, self.d_k)?;
            let attn = scaled_dot_product_attention(&qh, &kh, &vh)?;
            head_outputs.push(attn);
        }

        // Concatenate heads along last dim -> [seq_len, d_model]
        let concat = Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)?;
        let output = matmul_2d(&concat, &self.w_o)?;
        Ok(output)
    }
}

/// Feed-forward network: Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model).
pub struct FeedForward {
    pub w1: Tensor, // [d_model, d_ff]
    pub b1: Tensor, // [d_ff]
    pub w2: Tensor, // [d_ff, d_model]
    pub b2: Tensor, // [d_model]
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Result<Self, ModelError> {
        Ok(Self {
            w1: Tensor::from_vec(vec![d_model, d_ff], vec![0.0; d_model * d_ff])?,
            b1: Tensor::from_vec(vec![d_ff], vec![0.0; d_ff])?,
            w2: Tensor::from_vec(vec![d_ff, d_model], vec![0.0; d_ff * d_model])?,
            b2: Tensor::from_vec(vec![d_model], vec![0.0; d_model])?,
        })
    }

    /// Forward: `[seq_len, d_model]` -> `[seq_len, d_model]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let h = matmul_2d(input, &self.w1)?;
        let h = h.add(&self.b1.unsqueeze(0)?)?;
        let data: Vec<f32> = h.data().iter().map(|&v| v.max(0.0)).collect();
        let h = Tensor::from_vec(h.shape().to_vec(), data)?;
        let out = matmul_2d(&h, &self.w2)?;
        let out = out.add(&self.b2.unsqueeze(0)?)?;
        Ok(out)
    }
}

/// Generates a causal (lower-triangular) attention mask.
/// Returns [seq_len, seq_len] tensor where:
/// - 0.0 on and below diagonal (allowed positions)
/// - f32::NEG_INFINITY above diagonal (masked positions)
pub fn generate_causal_mask(seq_len: usize) -> Result<Tensor, ModelError> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(vec![seq_len, seq_len], data)?)
}

/// Generates a padding mask for batched sequences with different lengths.
/// lengths: actual length of each sequence in the batch
/// max_len: maximum sequence length (pad length)
/// Returns [batch, max_len] tensor where:
/// - 0.0 for valid positions (index < length)
/// - f32::NEG_INFINITY for padding positions (index >= length)
pub fn generate_padding_mask(lengths: &[usize], max_len: usize) -> Result<Tensor, ModelError> {
    let batch = lengths.len();
    let mut data = vec![0.0f32; batch * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for j in len..max_len {
            data[b * max_len + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(vec![batch, max_len], data)?)
}

/// Transformer encoder block: MHA -> Add&Norm -> FFN -> Add&Norm.
pub struct TransformerEncoderBlock {
    pub mha: MultiHeadAttention,
    pub ffn: FeedForward,
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub d_model: usize,
}

impl TransformerEncoderBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Result<Self, ModelError> {
        let config = MultiHeadAttentionConfig { d_model, num_heads };
        Ok(Self {
            mha: MultiHeadAttention::new(&config)?,
            ffn: FeedForward::new(d_model, d_ff)?,
            ln1_gamma: Tensor::from_vec(vec![d_model], vec![1.0; d_model])?,
            ln1_beta: Tensor::from_vec(vec![d_model], vec![0.0; d_model])?,
            ln2_gamma: Tensor::from_vec(vec![d_model], vec![1.0; d_model])?,
            ln2_beta: Tensor::from_vec(vec![d_model], vec![0.0; d_model])?,
            d_model,
        })
    }

    /// Forward: `[seq_len, d_model]` -> `[seq_len, d_model]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let attn_out = self.mha.forward(input)?;
        let residual1 = input.add(&attn_out)?;
        let norm1 = layer_norm_2d(&residual1, &self.ln1_gamma, &self.ln1_beta, self.d_model)?;

        let ffn_out = self.ffn.forward(&norm1)?;
        let residual2 = norm1.add(&ffn_out)?;
        let norm2 = layer_norm_2d(&residual2, &self.ln2_gamma, &self.ln2_beta, self.d_model)?;
        Ok(norm2)
    }
}

fn layer_norm_2d(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    _d: usize,
) -> Result<Tensor, ModelError> {
    let params = yscv_kernels::LayerNormLastDimParams {
        gamma,
        beta,
        epsilon: 1e-5,
    };
    yscv_kernels::layer_norm_last_dim(input, params).map_err(Into::into)
}
