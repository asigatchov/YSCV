use super::conv::{nchw_to_nhwc, nhwc_to_nchw};
use super::*;

pub(super) fn exec_batch_norm(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let gamma = get_tensor(env, &node.name, &node.inputs[1])?;
    let beta = get_tensor(env, &node.name, &node.inputs[2])?;
    let mean = get_tensor(env, &node.name, &node.inputs[3])?;
    let var = get_tensor(env, &node.name, &node.inputs[4])?;
    let epsilon = get_attr_float(node, "epsilon").unwrap_or(1e-5);

    let input_nhwc = nchw_to_nhwc(input)?;
    let params = BatchNorm2dParams {
        gamma,
        beta,
        mean,
        variance: var,
        epsilon,
    };
    let out_nhwc = batch_norm2d_nhwc(&input_nhwc, params).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    let out_nchw = nhwc_to_nchw(&out_nhwc)?;
    env.insert(node.outputs[0].clone(), out_nchw);
    Ok(())
}

pub(super) fn exec_softmax(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = softmax_last_dim(input).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_dropout(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    env.insert(node.outputs[0].clone(), input.clone());
    Ok(())
}

pub(super) fn exec_instance_norm(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let scale = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias_t = get_tensor(env, &node.name, &node.inputs[2])?;
    let eps = get_attr_float(node, "epsilon").unwrap_or(1e-5);

    let shape = input.shape();
    if shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: "InstanceNorm expects 4D NCHW".into(),
        });
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let hw = (h * w) as f32;
    let in_data = input.data();
    let scale_data = scale.data();
    let bias_data = bias_t.data();
    let mut out_data = vec![0.0f32; n * c * h * w];

    for b in 0..n {
        for ch in 0..c {
            let base = (b * c + ch) * h * w;
            let mean: f32 = (0..h * w).map(|i| in_data[base + i]).sum::<f32>() / hw;
            let var: f32 = (0..h * w)
                .map(|i| {
                    let d = in_data[base + i] - mean;
                    d * d
                })
                .sum::<f32>()
                / hw;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..h * w {
                out_data[base + i] =
                    (in_data[base + i] - mean) * inv_std * scale_data[ch] + bias_data[ch];
            }
        }
    }

    let result =
        Tensor::from_vec(shape.to_vec(), out_data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_lp_norm(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let p = get_attr_int(node, "p").unwrap_or(2) as i32;
    let axis = get_attr_int(node, "axis").unwrap_or(-1);

    let shape = input.shape();
    let rank = shape.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let data = input.data();
    let mut out_data = data.to_vec();
    let outer: usize = shape[..axis].iter().product();
    let dim = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            let norm: f32 = if p == 1 {
                (0..dim)
                    .map(|d| data[(o * dim + d) * inner + i].abs())
                    .sum::<f32>()
                    .max(1e-12)
            } else {
                (0..dim)
                    .map(|d| data[(o * dim + d) * inner + i].powi(2))
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12)
            };
            for d in 0..dim {
                out_data[(o * dim + d) * inner + i] /= norm;
            }
        }
    }

    let result =
        Tensor::from_vec(shape.to_vec(), out_data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_layer_norm(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let scale = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };
    let eps = get_attr_float(node, "epsilon").unwrap_or(1e-5);
    let axis = get_attr_int(node, "axis").unwrap_or(-1);

    let shape = input.shape();
    let rank = shape.len() as i64;
    let axis_norm = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let outer: usize = shape[..axis_norm].iter().product();
    let inner: usize = shape[axis_norm..].iter().product();
    let data = input.data();
    let scale_data = scale.data();
    let bias_data = bias.as_ref().map(|b| b.data());
    let inner_f = inner as f32;

    let mut out_data = vec![0.0f32; data.len()];
    for o in 0..outer {
        let base = o * inner;
        let mean: f32 = data[base..base + inner].iter().sum::<f32>() / inner_f;
        let var: f32 = data[base..base + inner]
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / inner_f;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..inner {
            let normalized = (data[base + i] - mean) * inv_std;
            let si = i % scale_data.len();
            out_data[base + i] = normalized * scale_data[si] + bias_data.map_or(0.0, |b| b[si]);
        }
    }

    let result =
        Tensor::from_vec(shape.to_vec(), out_data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_hardmax(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(-1);
    let shape = input.shape();
    let rank = shape.len() as i64;
    let axis_idx = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let outer: usize = shape[..axis_idx].iter().product();
    let dim = shape[axis_idx];
    let inner: usize = shape[axis_idx + 1..].iter().product();
    let data = input.data();
    let mut out_data = vec![0.0f32; data.len()];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let v = data[(o * dim + d) * inner + i];
                if v > max_val {
                    max_val = v;
                    max_idx = d;
                }
            }
            out_data[(o * dim + max_idx) * inner + i] = 1.0;
        }
    }

    let result =
        Tensor::from_vec(shape.to_vec(), out_data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

#[allow(clippy::needless_range_loop)]
pub(super) fn exec_lrn(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, "alpha").unwrap_or(0.0001);
    let beta = get_attr_float(node, "beta").unwrap_or(0.75);
    let bias = get_attr_float(node, "bias").unwrap_or(1.0);
    let size = get_attr_int(node, "size").unwrap_or(1) as usize;
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch {
            detail: format!("LRN expects 4D NCHW, got {:?}", shape),
        });
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let data = input.data();
    let mut out = vec![0.0f32; data.len()];
    let half = (size / 2) as i64;

    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let mut sq_sum = 0.0f32;
                    for k in -half..=half {
                        let ck = ci as i64 + k;
                        if ck >= 0 && (ck as usize) < c {
                            let kidx = ((ni * c + ck as usize) * h + hi) * w + wi;
                            sq_sum += data[kidx] * data[kidx];
                        }
                    }
                    let norm = (bias + alpha / size as f32 * sq_sum).powf(beta);
                    out[idx] = data[idx] / norm;
                }
            }
        }
    }

    let result = Tensor::from_vec(shape.to_vec(), out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}
