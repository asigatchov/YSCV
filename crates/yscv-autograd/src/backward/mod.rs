use yscv_kernels::{matmul_2d, mul as kernel_mul};
use yscv_tensor::Tensor;

use super::checkpoint::CheckpointConfig;
use super::error::AutogradError;
use super::graph::Graph;
use super::node::{NodeId, Op};

mod activation;
mod attention;
mod elementwise;
mod linalg;
mod norm;
mod pool;
mod recurrent;
mod reduce;
mod shape;

// Re-export for sub-modules.
pub(super) use super::error;
pub(super) use super::graph;
pub(super) use super::node;

impl Graph {
    /// Backpropagates gradients from a scalar target.
    pub fn backward(&mut self, target: NodeId) -> Result<(), AutogradError> {
        if !self.node(target)?.value.shape().is_empty() {
            return Err(AutogradError::NonScalarTarget {
                shape: self.node(target)?.value.shape().to_vec(),
            });
        }

        self.zero_grads();
        self.node_mut(target)?.grad = Some(Tensor::scalar(1.0));

        for index in (0..=target.0).rev() {
            let op = self.nodes[index].op.clone();
            let upstream = match self.nodes[index].grad.take() {
                Some(grad) => grad,
                None => continue,
            };

            match op {
                Op::Leaf => {
                    // Restore gradient for leaf nodes — users query these after backward.
                    self.nodes[index].grad = Some(upstream);
                }
                Op::Add(left, right) => {
                    elementwise::add_backward(self, upstream, left, right)?;
                }
                Op::Sub(left, right) => {
                    elementwise::sub_backward(self, upstream, left, right)?;
                }
                Op::Mul(left, right) => {
                    elementwise::mul_backward(self, upstream, left, right)?;
                }
                Op::Div(left, right) => {
                    elementwise::div_backward(self, upstream, left, right)?;
                }
                Op::Neg(input) => {
                    elementwise::neg_backward(self, upstream, input)?;
                }
                Op::Pow(base, exponent) => {
                    elementwise::pow_backward(self, upstream, index, base, exponent)?;
                }
                Op::Abs(input) => {
                    elementwise::abs_backward(self, upstream, input)?;
                }
                Op::Clamp {
                    input,
                    min_bits,
                    max_bits,
                } => {
                    elementwise::clamp_backward(self, upstream, input, min_bits, max_bits)?;
                }

                // Activations
                Op::Relu(input) => {
                    activation::relu_backward(self, upstream, index, input)?;
                }
                Op::LeakyRelu {
                    input,
                    negative_slope,
                } => {
                    activation::leaky_relu_backward(self, upstream, input, negative_slope)?;
                }
                Op::Sigmoid(input) => {
                    activation::sigmoid_backward(self, upstream, index, input)?;
                }
                Op::Tanh(input) => {
                    activation::tanh_backward(self, upstream, index, input)?;
                }
                Op::Exp(input) => {
                    activation::exp_backward(self, upstream, index, input)?;
                }
                Op::Log(input) => {
                    activation::log_backward(self, upstream, input)?;
                }
                Op::Sqrt(input) => {
                    activation::sqrt_backward(self, upstream, index, input)?;
                }
                Op::Gelu(input) => {
                    activation::gelu_backward(self, upstream, input)?;
                }
                Op::Silu(input) => {
                    activation::silu_backward(self, upstream, input)?;
                }
                Op::Mish(input) => {
                    activation::mish_backward(self, upstream, input)?;
                }
                Op::Softmax(input) => {
                    activation::softmax_backward(self, upstream, index, input)?;
                }
                Op::LogSoftmax(input) => {
                    activation::log_softmax_backward(self, upstream, index, input)?;
                }

                // Linear algebra
                Op::MatMul2D(left, right) => {
                    linalg::matmul2d_backward(self, upstream, left, right)?;
                }
                Op::Transpose2D(input) => {
                    linalg::transpose2d_backward(self, upstream, input)?;
                }
                Op::Conv2dNhwc {
                    input,
                    weight,
                    bias,
                    stride_h,
                    stride_w,
                } => {
                    linalg::conv2d_nhwc_backward(
                        self,
                        &upstream,
                        input,
                        weight,
                        bias,
                        stride_h as usize,
                        stride_w as usize,
                    )?;
                }
                Op::DepthwiseConv2dNhwc {
                    input,
                    weight,
                    bias,
                    stride_h,
                    stride_w,
                } => {
                    linalg::depthwise_conv2d_nhwc_backward(
                        self,
                        &upstream,
                        input,
                        weight,
                        bias,
                        stride_h as usize,
                        stride_w as usize,
                    )?;
                }
                Op::ConvTranspose2dNhwc {
                    input,
                    weight,
                    bias,
                    stride_h,
                    stride_w,
                } => {
                    linalg::conv_transpose2d_nhwc_backward(
                        self,
                        &upstream,
                        input,
                        weight,
                        bias,
                        stride_h as usize,
                        stride_w as usize,
                    )?;
                }
                Op::Conv1dNlc {
                    input,
                    weight,
                    bias,
                    stride,
                } => {
                    linalg::conv1d_nlc_backward(
                        self,
                        &upstream,
                        input,
                        weight,
                        bias,
                        stride as usize,
                    )?;
                }
                Op::Conv3dNdhwc {
                    input,
                    weight,
                    bias,
                    stride_d,
                    stride_h,
                    stride_w,
                } => {
                    linalg::conv3d_ndhwc_backward(
                        self,
                        &upstream,
                        input,
                        weight,
                        bias,
                        stride_d as usize,
                        stride_h as usize,
                        stride_w as usize,
                    )?;
                }
                Op::DeformableConv2dNhwc {
                    input,
                    weight,
                    offsets,
                    bias,
                    stride,
                    padding,
                } => {
                    linalg::deformable_conv2d_nhwc_backward(
                        self,
                        &upstream,
                        input,
                        weight,
                        offsets,
                        bias,
                        stride as usize,
                        padding as usize,
                    )?;
                }

                // Shape operations
                Op::ReshapeView { input } => {
                    shape::reshape_backward(self, upstream, input)?;
                }
                Op::Flatten(input) => {
                    shape::flatten_backward(self, upstream, input)?;
                }
                Op::UnsqueezeView { input, axis } => {
                    shape::unsqueeze_backward(self, upstream, input, axis)?;
                }
                Op::SqueezeView { input, axis } => {
                    shape::squeeze_backward(self, upstream, input, axis)?;
                }
                Op::Cat { ref inputs, axis } => {
                    shape::cat_backward(self, upstream, inputs, axis)?;
                }
                Op::Select { input, axis, index } => {
                    shape::select_backward(self, upstream, input, axis, index)?;
                }
                Op::Narrow {
                    input,
                    axis,
                    start,
                    len,
                } => {
                    shape::narrow_backward(self, upstream, input, axis, start, len)?;
                }
                Op::Gather { input, axis, index } => {
                    shape::gather_backward(self, upstream, input, axis, index)?;
                }
                Op::ScatterAdd {
                    input,
                    axis,
                    index,
                    src,
                } => {
                    shape::scatter_add_backward(self, upstream, input, axis, index, src)?;
                }
                #[allow(clippy::needless_range_loop)]
                Op::Pad {
                    input,
                    ref pad_before,
                    pad_after: _,
                } => {
                    shape::pad_backward(self, upstream, input, pad_before)?;
                }
                #[allow(clippy::needless_range_loop)]
                Op::Repeat { input, repeats: _ } => {
                    shape::repeat_backward(self, upstream, input)?;
                }
                Op::Scatter {
                    input,
                    indices,
                    src,
                } => {
                    shape::scatter_backward(self, upstream, input, indices, src)?;
                }
                Op::EmbeddingLookup { weight, indices } => {
                    shape::embedding_lookup_backward(self, upstream, weight, indices)?;
                }
                Op::PixelShuffle {
                    input,
                    upscale_factor,
                } => {
                    shape::pixel_shuffle_backward(self, &upstream, input, upscale_factor as usize)?;
                }
                Op::UpsampleNearest {
                    input,
                    scale_factor,
                } => {
                    shape::upsample_nearest_backward(
                        self,
                        &upstream,
                        input,
                        scale_factor as usize,
                    )?;
                }

                // Reductions
                Op::Sum(input) => {
                    reduce::sum_backward(self, upstream, index, input)?;
                }
                Op::Mean(input) => {
                    reduce::mean_backward(self, upstream, index, input)?;
                }
                Op::SumAxis { input, axis } => {
                    reduce::sum_axis_backward(self, upstream, input, axis)?;
                }
                Op::MeanAxis { input, axis } => {
                    reduce::mean_axis_backward(self, upstream, input, axis)?;
                }

                // Recurrent
                Op::Rnn {
                    input,
                    w_ih,
                    w_hh,
                    bias,
                } => {
                    recurrent::rnn_backward(self, &upstream, index, input, w_ih, w_hh, bias)?;
                }
                Op::Lstm {
                    input,
                    w_ih,
                    w_hh,
                    bias,
                } => {
                    recurrent::lstm_backward(self, &upstream, index, input, w_ih, w_hh, bias)?;
                }
                Op::Gru {
                    input,
                    w_ih,
                    w_hh,
                    bias_ih,
                    bias_hh,
                } => {
                    recurrent::gru_backward(
                        self, &upstream, index, input, w_ih, w_hh, bias_ih, bias_hh,
                    )?;
                }

                // Normalization
                Op::BatchNorm2dNhwc {
                    input,
                    gamma,
                    beta,
                    running_mean: _,
                    running_var,
                    epsilon,
                } => {
                    let eps = f32::from_bits(epsilon);
                    norm::batch_norm2d_nhwc_backward(
                        self,
                        index,
                        &upstream,
                        input,
                        gamma,
                        beta,
                        running_var,
                        eps,
                    )?;
                }
                Op::LayerNorm {
                    input,
                    gamma,
                    beta,
                    eps_bits,
                } => {
                    let eps = f32::from_bits(eps_bits);
                    norm::layer_norm_backward(self, index, &upstream, input, gamma, beta, eps)?;
                }
                Op::GroupNorm {
                    input,
                    gamma,
                    beta,
                    num_groups,
                    eps_bits,
                } => {
                    let eps = f32::from_bits(eps_bits);
                    norm::group_norm_backward(
                        self,
                        index,
                        &upstream,
                        input,
                        gamma,
                        beta,
                        num_groups as usize,
                        eps,
                    )?;
                }
                Op::InstanceNormNhwc {
                    input,
                    gamma,
                    beta,
                    eps_bits,
                } => {
                    let eps = f32::from_bits(eps_bits);
                    norm::instance_norm_nhwc_backward(
                        self, index, &upstream, input, gamma, beta, eps,
                    )?;
                }

                // Pooling
                Op::MaxPool2dNhwc {
                    input,
                    kernel_h: _,
                    kernel_w: _,
                    stride_h: _,
                    stride_w: _,
                } => {
                    pool::max_pool2d_nhwc_backward(self, upstream, index, input)?;
                }
                Op::AvgPool2dNhwc {
                    input,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                } => {
                    pool::avg_pool2d_nhwc_backward(
                        self,
                        &upstream,
                        input,
                        kernel_h as usize,
                        kernel_w as usize,
                        stride_h as usize,
                        stride_w as usize,
                    )?;
                }
                Op::AdaptiveAvgPool2dNhwc {
                    input,
                    out_h,
                    out_w,
                } => {
                    pool::adaptive_avg_pool2d_nhwc_backward(
                        self,
                        &upstream,
                        input,
                        out_h as usize,
                        out_w as usize,
                    )?;
                }
                Op::AdaptiveMaxPool2dNhwc {
                    input,
                    out_h: _,
                    out_w: _,
                } => {
                    pool::adaptive_max_pool2d_nhwc_backward(self, upstream, index, input)?;
                }

                // Attention & PReLU
                Op::PRelu { input, alpha } => {
                    attention::prelu_backward(self, &upstream, input, alpha)?;
                }
                Op::ScaledDotProductAttention { query, key, value } => {
                    attention::scaled_dot_product_attention_backward(
                        self, &upstream, index, query, key, value,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Backward pass with activation checkpointing.
    pub fn backward_with_checkpoints(
        &mut self,
        target: NodeId,
        config: &CheckpointConfig,
    ) -> Result<(), AutogradError> {
        self.backward(target)?;

        for index in 0..self.nodes.len() {
            if config.should_checkpoint(index) && !matches!(self.nodes[index].op, Op::Leaf) {
                self.nodes[index].value = Tensor::scalar(0.0);
                self.nodes[index].aux = None;
            }
        }

        Ok(())
    }

    pub(crate) fn accumulate_grad(
        &mut self,
        node_id: NodeId,
        contribution: Tensor,
    ) -> Result<(), AutogradError> {
        if !self.node(node_id)?.requires_grad {
            return Ok(());
        }

        let node = self.node_mut(node_id)?;
        match &mut node.grad {
            Some(existing) => {
                if existing.shape() != contribution.shape() {
                    return Err(AutogradError::InvalidGradientShape {
                        node: node_id.0,
                        expected: existing.shape().to_vec(),
                        got: contribution.shape().to_vec(),
                    });
                }
                existing.add_inplace(&contribution);
            }
            None => node.grad = Some(contribution),
        }
        Ok(())
    }

    // ── Backend dispatch helpers for backward pass ──────────────────

    pub(crate) fn dispatch_mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, AutogradError> {
        if let Some(ref backend) = self.backend {
            Ok(backend.mul(lhs, rhs)?)
        } else {
            Ok(kernel_mul(lhs, rhs)?)
        }
    }

    pub(crate) fn dispatch_matmul_2d(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<Tensor, AutogradError> {
        if let Some(ref backend) = self.backend {
            Ok(backend.matmul_2d(lhs, rhs)?)
        } else {
            Ok(matmul_2d(lhs, rhs)?)
        }
    }

    pub(crate) fn dispatch_transpose_2d(&self, input: &Tensor) -> Result<Tensor, AutogradError> {
        if let Some(ref backend) = self.backend {
            Ok(backend.transpose_2d(input)?)
        } else {
            transpose_2d(input)
        }
    }

    pub(crate) fn dispatch_neg(&self, input: &Tensor) -> Tensor {
        if let Some(ref backend) = self.backend {
            backend.neg(input)
        } else {
            input.neg()
        }
    }
}

fn transpose_2d(input: &Tensor) -> Result<Tensor, AutogradError> {
    if input.rank() != 2 {
        return Err(AutogradError::InvalidRankForOperation {
            op: "transpose_2d",
            expected: 2,
            got: input.rank(),
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let mut data = vec![0.0f32; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            data[col * rows + row] = input.data()[row * cols + col];
        }
    }
    Tensor::from_vec(vec![cols, rows], data).map_err(Into::into)
}

fn reduce_broadcast_gradient(
    upstream: &Tensor,
    target_shape: &[usize],
) -> Result<Tensor, AutogradError> {
    if upstream.shape() == target_shape {
        return Ok(upstream.clone());
    }
    if target_shape.len() > upstream.rank() {
        return Err(AutogradError::BroadcastGradientIncompatible {
            upstream: upstream.shape().to_vec(),
            target: target_shape.to_vec(),
        });
    }

    let leading_axes = upstream.rank() - target_shape.len();
    let mut reduced: Option<Tensor> = None;
    for axis in (0..leading_axes).rev() {
        reduced = Some(match reduced {
            Some(r) => r.sum_axis(axis)?,
            None => upstream.sum_axis(axis)?,
        });
    }

    let mut axes_to_reduce = Vec::new();
    let check_shape = reduced.as_ref().unwrap_or(upstream);
    for (axis, target_dim) in target_shape.iter().enumerate() {
        let current_dim = check_shape.shape()[axis];
        if current_dim == *target_dim {
            continue;
        }
        if *target_dim == 1 && current_dim > 1 {
            axes_to_reduce.push(axis);
            continue;
        }
        return Err(AutogradError::BroadcastGradientIncompatible {
            upstream: upstream.shape().to_vec(),
            target: target_shape.to_vec(),
        });
    }

    if !axes_to_reduce.is_empty() && reduced.is_none() {
        reduced = Some(upstream.clone());
    }
    let mut reduced = reduced.unwrap_or_else(|| upstream.clone());

    for axis in axes_to_reduce.into_iter().rev() {
        reduced = reduced.sum_axis(axis)?;
    }

    if reduced.shape() != target_shape {
        reduced = reduced.reshape(target_shape.to_vec())?;
    }
    Ok(reduced)
}
