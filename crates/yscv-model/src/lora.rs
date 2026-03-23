use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::ModelError;

/// LoRA configuration.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices (default 4).
    pub rank: usize,
    /// Alpha scaling factor (default 1.0).
    pub alpha: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
        }
    }
}

/// A LoRA adapter for a linear layer.
///
/// Wraps a frozen weight matrix with trainable low-rank A and B matrices.
/// The effective weight is `W + A @ B * scaling` where `scaling = alpha / rank`.
///
/// In this crate, weight is stored as `[in_features, out_features]` (matching
/// `LinearLayer`), so:
///   - `lora_a` has shape `[in_features, rank]`
///   - `lora_b` has shape `[rank, out_features]`
///   - Forward: `y = x @ W + x @ A @ B * scaling + bias`
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Original frozen weight `[in_features, out_features]`.
    pub frozen_weight: NodeId,
    /// Low-rank matrix A `[in_features, rank]` — initialized with small random values.
    pub lora_a: NodeId,
    /// Low-rank matrix B `[rank, out_features]` — initialized to zeros.
    pub lora_b: NodeId,
    /// Optional bias (frozen).
    pub bias: Option<NodeId>,
    /// Scaling factor = alpha / rank.
    pub scaling: f32,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
}

impl LoraLinear {
    /// Creates a LoRA adapter from dimensions.
    ///
    /// Initializes `frozen_weight` and `bias` as graph constants (`requires_grad=false`),
    /// `lora_a` with small Gaussian-like values (`requires_grad=true`),
    /// `lora_b` with zeros (`requires_grad=true`).
    pub fn new(
        graph: &mut Graph,
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
    ) -> Result<Self, ModelError> {
        let rank = config.rank;
        let scaling = config.alpha / rank as f32;

        // Frozen weight initialized to zeros (would normally be loaded from a pretrained model).
        let frozen_weight_tensor = Tensor::zeros(vec![in_features, out_features])?;
        let frozen_weight = graph.constant(frozen_weight_tensor);

        // lora_a: small values using a simple deterministic initialization
        // (Kaiming-like: stddev = 1/sqrt(in_features))
        let stddev = 1.0 / (in_features as f32).sqrt();
        let a_len = in_features * rank;
        let a_data: Vec<f32> = (0..a_len)
            .map(|i| {
                // Simple deterministic pseudo-random using a hash-like function
                let x = ((i as f32 + 1.0) * 0.618_034).fract() * 2.0 - 1.0;
                x * stddev
            })
            .collect();
        let lora_a_tensor = Tensor::from_vec(vec![in_features, rank], a_data)?;
        let lora_a = graph.variable(lora_a_tensor);

        // lora_b: initialized to zeros so initial LoRA contribution is zero
        let lora_b_tensor = Tensor::zeros(vec![rank, out_features])?;
        let lora_b = graph.variable(lora_b_tensor);

        Ok(Self {
            frozen_weight,
            lora_a,
            lora_b,
            bias: None,
            scaling,
            in_features,
            out_features,
            rank,
        })
    }

    /// Creates a LoRA adapter from an existing `LinearLayer`'s weights.
    ///
    /// The original weight and bias are frozen (stored as constants).
    /// New trainable `lora_a` and `lora_b` matrices are created.
    pub fn from_linear(
        graph: &mut Graph,
        weight_node: NodeId,
        bias_node: NodeId,
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
    ) -> Result<Self, ModelError> {
        let rank = config.rank;
        let scaling = config.alpha / rank as f32;

        // Freeze the original weight: copy tensor into a constant node.
        let weight_tensor = graph.value(weight_node)?.clone();
        let frozen_weight = graph.constant(weight_tensor);

        // Freeze the original bias: copy tensor into a constant node.
        let bias_tensor = graph.value(bias_node)?.clone();
        let frozen_bias = graph.constant(bias_tensor);

        // lora_a: small values (Kaiming-like)
        let stddev = 1.0 / (in_features as f32).sqrt();
        let a_len = in_features * rank;
        let a_data: Vec<f32> = (0..a_len)
            .map(|i| {
                let x = ((i as f32 + 1.0) * 0.618_034).fract() * 2.0 - 1.0;
                x * stddev
            })
            .collect();
        let lora_a_tensor = Tensor::from_vec(vec![in_features, rank], a_data)?;
        let lora_a = graph.variable(lora_a_tensor);

        // lora_b: zeros so initial LoRA contribution is zero
        let lora_b_tensor = Tensor::zeros(vec![rank, out_features])?;
        let lora_b = graph.variable(lora_b_tensor);

        Ok(Self {
            frozen_weight,
            lora_a,
            lora_b,
            bias: Some(frozen_bias),
            scaling,
            in_features,
            out_features,
            rank,
        })
    }

    /// Creates a LoRA adapter with a frozen bias term.
    pub fn with_bias(mut self, graph: &mut Graph, bias_tensor: Tensor) -> Result<Self, ModelError> {
        if bias_tensor.shape() != [self.out_features] {
            return Err(ModelError::InvalidParameterShape {
                parameter: "bias",
                expected: vec![self.out_features],
                got: bias_tensor.shape().to_vec(),
            });
        }
        self.bias = Some(graph.constant(bias_tensor));
        Ok(self)
    }

    /// Forward pass: `y = x @ W + x @ A @ B * scaling + bias`.
    ///
    /// Gradients flow through A and B but not through the frozen weight.
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let input_shape = graph.value(input)?.shape().to_vec();
        if input_shape.len() != 2 || input_shape[1] != self.in_features {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.in_features,
                got: input_shape,
            });
        }

        // Frozen path: x @ W  [batch, in] @ [in, out] -> [batch, out]
        let frozen_out = graph.matmul_2d(input, self.frozen_weight)?;

        // LoRA path: x @ A @ B * scaling
        // x @ A: [batch, in] @ [in, rank] -> [batch, rank]
        let lora_mid = graph.matmul_2d(input, self.lora_a)?;
        // (x @ A) @ B: [batch, rank] @ [rank, out] -> [batch, out]
        let lora_out = graph.matmul_2d(lora_mid, self.lora_b)?;

        // Scale by alpha / rank
        let scale_node = graph.constant(Tensor::scalar(self.scaling));
        let lora_scaled = graph.mul(lora_out, scale_node)?;

        // Combine frozen + lora
        let mut output = graph.add(frozen_out, lora_scaled)?;

        // Add bias if present
        if let Some(bias) = self.bias {
            output = graph.add(output, bias)?;
        }

        Ok(output)
    }

    /// Returns the trainable parameter `NodeId`s (only `lora_a` and `lora_b`).
    pub fn trainable_params(&self) -> Vec<NodeId> {
        vec![self.lora_a, self.lora_b]
    }

    /// Merges LoRA weights into the frozen weight, returning the effective weight tensor.
    ///
    /// `W_eff = W + A @ B * scaling`
    ///
    /// Useful for inference after training to avoid the LoRA overhead.
    pub fn merge(&self, graph: &Graph) -> Result<Tensor, ModelError> {
        let w = graph.value(self.frozen_weight)?;
        let a = graph.value(self.lora_a)?;
        let b = graph.value(self.lora_b)?;

        // A @ B: [in, rank] @ [rank, out] -> [in, out]
        let ab = yscv_kernels::matmul_2d(a, b)?;

        // Scale and add to frozen weight
        let w_data = w.data();
        let ab_data = ab.data();
        let merged_data: Vec<f32> = w_data
            .iter()
            .zip(ab_data.iter())
            .map(|(&wi, &abi)| wi + abi * self.scaling)
            .collect();

        Ok(Tensor::from_vec(w.shape().to_vec(), merged_data)?)
    }
}
