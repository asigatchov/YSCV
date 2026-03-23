use serde::{Deserialize, Serialize};
use yscv_tensor::Tensor;

use crate::ModelError;

/// Serializable tensor snapshot used in model checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSnapshot {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorSnapshot {
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            data: tensor.data().to_vec(),
        }
    }

    pub fn into_tensor(self) -> Result<Tensor, ModelError> {
        Tensor::from_vec(self.shape, self.data).map_err(Into::into)
    }
}

/// Serializable layer checkpoint payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "layer", content = "payload")]
pub enum LayerCheckpoint {
    Linear {
        in_features: usize,
        out_features: usize,
        weight: TensorSnapshot,
        bias: TensorSnapshot,
    },
    ReLU,
    LeakyReLU {
        negative_slope: f32,
    },
    Sigmoid,
    Tanh,
    Dropout {
        rate: f32,
    },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: TensorSnapshot,
        bias: Option<TensorSnapshot>,
    },
    BatchNorm2d {
        num_features: usize,
        epsilon: f32,
        gamma: TensorSnapshot,
        beta: TensorSnapshot,
        running_mean: TensorSnapshot,
        running_var: TensorSnapshot,
    },
    MaxPool2d {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    AvgPool2d {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    Flatten,
    GlobalAvgPool2d,
    Softmax,
    Embedding {
        num_embeddings: usize,
        embedding_dim: usize,
        weight: TensorSnapshot,
    },
    LayerNorm {
        normalized_shape: usize,
        eps: f32,
        gamma: TensorSnapshot,
        beta: TensorSnapshot,
    },
    GroupNorm {
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        gamma: TensorSnapshot,
        beta: TensorSnapshot,
    },
    DepthwiseConv2d {
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: TensorSnapshot,
        bias: Option<TensorSnapshot>,
    },
    SeparableConv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        depthwise_weight: TensorSnapshot,
        pointwise_weight: TensorSnapshot,
        bias: Option<TensorSnapshot>,
    },
}

/// Serializable sequential model checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialCheckpoint {
    pub layers: Vec<LayerCheckpoint>,
}

pub fn checkpoint_to_json(checkpoint: &SequentialCheckpoint) -> Result<String, ModelError> {
    serde_json::to_string_pretty(checkpoint).map_err(|err| ModelError::CheckpointSerialization {
        message: err.to_string(),
    })
}

pub fn checkpoint_from_json(json: &str) -> Result<SequentialCheckpoint, ModelError> {
    serde_json::from_str(json).map_err(|err| ModelError::CheckpointSerialization {
        message: err.to_string(),
    })
}
