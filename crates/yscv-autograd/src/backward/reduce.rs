use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::NodeId;

/// Sum backward (scalar reduction).
pub(crate) fn sum_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    if !upstream.shape().is_empty() {
        return Err(AutogradError::InvalidGradientShape {
            node: index,
            expected: Vec::new(),
            got: upstream.shape().to_vec(),
        });
    }
    let scalar_grad = upstream.data()[0];
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let input_grad = Tensor::filled(input_shape, scalar_grad)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Mean backward (scalar reduction).
pub(crate) fn mean_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    if !upstream.shape().is_empty() {
        return Err(AutogradError::InvalidGradientShape {
            node: index,
            expected: Vec::new(),
            got: upstream.shape().to_vec(),
        });
    }
    let input_len = graph.node(input)?.value.len() as f32;
    let scalar_grad = upstream.data()[0] / input_len;
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let input_grad = Tensor::filled(input_shape, scalar_grad)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// SumAxis backward.
pub(crate) fn sum_axis_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
) -> Result<(), AutogradError> {
    let ax = axis as usize;
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let input_grad = upstream.unsqueeze(ax)?;
    let mut repeats = vec![1usize; input_shape.len()];
    repeats[ax] = input_shape[ax];
    let input_grad = input_grad.repeat(&repeats)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// MeanAxis backward.
pub(crate) fn mean_axis_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
) -> Result<(), AutogradError> {
    let ax = axis as usize;
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let dim_size = input_shape[ax] as f32;
    let input_grad = upstream.unsqueeze(ax)?;
    let mut repeats = vec![1usize; input_shape.len()];
    repeats[ax] = input_shape[ax];
    let input_grad = input_grad.repeat(&repeats)?.scale(1.0 / dim_size);
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}
