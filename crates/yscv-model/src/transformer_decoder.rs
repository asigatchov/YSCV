use yscv_kernels::matmul_2d;
use yscv_tensor::Tensor;

use super::ModelError;
use super::attention::{
    FeedForward, MultiHeadAttention, MultiHeadAttentionConfig, scaled_dot_product_attention,
};

/// Cross-attention: query from decoder, key/value from encoder output.
///
/// Stores separate projection matrices for Q (applied to decoder state)
/// and K/V (applied to encoder memory).
pub struct CrossAttention {
    pub w_q: Tensor, // [d_model, d_model]
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    pub num_heads: usize,
    pub d_k: usize,
    pub d_model: usize,
}

impl CrossAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Result<Self, ModelError> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(ModelError::InvalidParameterShape {
                parameter: "d_model must be divisible by num_heads",
                expected: vec![d_model, num_heads],
                got: vec![d_model % num_heads],
            });
        }
        let d_k = d_model / num_heads;
        let z = vec![0.0f32; d_model * d_model];
        Ok(Self {
            w_q: Tensor::from_vec(vec![d_model, d_model], z.clone())?,
            w_k: Tensor::from_vec(vec![d_model, d_model], z.clone())?,
            w_v: Tensor::from_vec(vec![d_model, d_model], z.clone())?,
            w_o: Tensor::from_vec(vec![d_model, d_model], z)?,
            num_heads,
            d_k,
            d_model,
        })
    }

    /// Forward pass.
    ///
    /// `query`: `[seq_q, d_model]` — decoder state.
    /// `kv`:    `[seq_kv, d_model]` — encoder memory.
    ///
    /// Returns `[seq_q, d_model]`.
    pub fn forward(&self, query: &Tensor, kv: &Tensor) -> Result<Tensor, ModelError> {
        let q = matmul_2d(query, &self.w_q)?;
        let k = matmul_2d(kv, &self.w_k)?;
        let v = matmul_2d(kv, &self.w_v)?;

        let mut head_outputs = Vec::new();
        for h in 0..self.num_heads {
            let start = h * self.d_k;
            let qh = q.narrow(1, start, self.d_k)?;
            let kh = k.narrow(1, start, self.d_k)?;
            let vh = v.narrow(1, start, self.d_k)?;
            let attn = scaled_dot_product_attention(&qh, &kh, &vh)?;
            head_outputs.push(attn);
        }

        let concat = Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)?;
        let output = matmul_2d(&concat, &self.w_o)?;
        Ok(output)
    }
}

fn layer_norm_2d(input: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor, ModelError> {
    let params = yscv_kernels::LayerNormLastDimParams {
        gamma,
        beta,
        epsilon: 1e-5,
    };
    yscv_kernels::layer_norm_last_dim(input, params).map_err(Into::into)
}

/// Single transformer decoder block: masked self-attention → cross-attention → FFN,
/// each sub-layer wrapped with residual connection and layer normalization.
pub struct TransformerDecoderBlock {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: CrossAttention,
    pub ffn: FeedForward,
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub ln3_gamma: Tensor,
    pub ln3_beta: Tensor,
    pub d_model: usize,
}

impl TransformerDecoderBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Result<Self, ModelError> {
        let mha_config = MultiHeadAttentionConfig { d_model, num_heads };
        Ok(Self {
            self_attn: MultiHeadAttention::new(&mha_config)?,
            cross_attn: CrossAttention::new(d_model, num_heads)?,
            ffn: FeedForward::new(d_model, d_ff)?,
            ln1_gamma: Tensor::from_vec(vec![d_model], vec![1.0; d_model])?,
            ln1_beta: Tensor::from_vec(vec![d_model], vec![0.0; d_model])?,
            ln2_gamma: Tensor::from_vec(vec![d_model], vec![1.0; d_model])?,
            ln2_beta: Tensor::from_vec(vec![d_model], vec![0.0; d_model])?,
            ln3_gamma: Tensor::from_vec(vec![d_model], vec![1.0; d_model])?,
            ln3_beta: Tensor::from_vec(vec![d_model], vec![0.0; d_model])?,
            d_model,
        })
    }

    /// Forward pass.
    ///
    /// `target`: `[seq_t, d_model]` — decoder input / previous decoder layer output.
    /// `memory`: `[seq_s, d_model]` — encoder output.
    ///
    /// Returns `[seq_t, d_model]`.
    pub fn forward(&self, target: &Tensor, memory: &Tensor) -> Result<Tensor, ModelError> {
        // Sub-layer 1: masked self-attention
        let sa_out = self.self_attn.forward(target)?;
        let residual1 = target.add(&sa_out)?;
        let x = layer_norm_2d(&residual1, &self.ln1_gamma, &self.ln1_beta)?;

        // Sub-layer 2: cross-attention (Q from decoder, K/V from encoder)
        let ca_out = self.cross_attn.forward(&x, memory)?;
        let residual2 = x.add(&ca_out)?;
        let x = layer_norm_2d(&residual2, &self.ln2_gamma, &self.ln2_beta)?;

        // Sub-layer 3: feed-forward network
        let ffn_out = self.ffn.forward(&x)?;
        let residual3 = x.add(&ffn_out)?;
        let x = layer_norm_2d(&residual3, &self.ln3_gamma, &self.ln3_beta)?;

        Ok(x)
    }
}

/// Stack of `TransformerDecoderBlock` layers.
pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderBlock>,
}

impl TransformerDecoder {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        num_layers: usize,
    ) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerDecoderBlock::new(d_model, num_heads, d_ff)?);
        }
        Ok(Self { layers })
    }

    /// Forward pass through all decoder layers.
    ///
    /// `target`: `[seq_t, d_model]` — decoder input.
    /// `memory`: `[seq_s, d_model]` — encoder output.
    ///
    /// Returns `[seq_t, d_model]`.
    pub fn forward(&self, target: &Tensor, memory: &Tensor) -> Result<Tensor, ModelError> {
        let mut x = target.clone();
        for layer in &self.layers {
            x = layer.forward(&x, memory)?;
        }
        Ok(x)
    }
}
