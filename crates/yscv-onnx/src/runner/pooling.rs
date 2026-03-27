use super::conv::{nchw_to_nhwc, pad_nhwc, pad_nhwc_val};
use super::*;

pub(super) fn exec_max_pool(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let kernel_shape = get_attr_ints(node, "kernel_shape").unwrap_or_else(|| vec![2, 2]);
    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };
    let padded_owned;
    let input_padded: &Tensor = if pads.iter().any(|&p| p > 0) {
        padded_owned = pad_nhwc_val(
            input_nhwc,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
            f32::NEG_INFINITY,
        )?;
        &padded_owned
    } else {
        input_nhwc
    };
    let out_nhwc = max_pool2d_nhwc(
        input_padded,
        kernel_shape[0] as usize,
        kernel_shape[1] as usize,
        strides[0] as usize,
        strides[1] as usize,
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

pub(super) fn exec_avg_pool(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let kernel_shape = get_attr_ints(node, "kernel_shape").unwrap_or_else(|| vec![2, 2]);
    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };
    let padded_owned;
    let input_padded: &Tensor = if pads.iter().any(|&p| p > 0) {
        padded_owned = pad_nhwc(
            input_nhwc,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        )?;
        &padded_owned
    } else {
        input_nhwc
    };
    let out_nhwc = avg_pool2d_nhwc(
        input_padded,
        kernel_shape[0] as usize,
        kernel_shape[1] as usize,
        strides[0] as usize,
        strides[1] as usize,
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

pub(super) fn exec_global_avg_pool(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape = input.shape();
    let (h, w) = if input_is_nhwc {
        (shape[1], shape[2])
    } else {
        (shape[2], shape[3])
    };
    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };
    let out_nhwc =
        avg_pool2d_nhwc(input_nhwc, h, w, 1, 1).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}
