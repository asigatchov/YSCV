use super::*;

/// ONNX Conv: NCHW input, OIHW weight -> convert to NHWC for yscv kernels, then back.
pub(super) fn exec_conv(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let group = get_attr_int(node, "group").unwrap_or(1) as usize;

    let sh = strides[0] as usize;
    let sw = strides[1] as usize;

    // NCHW -> NHWC
    let input_nhwc = nchw_to_nhwc(input)?;

    // Weight: ONNX [O, I/group, KH, KW] -> yscv [KH, KW, I/group, O]
    let w_shape = weight.shape();
    let (o_ch, i_per_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    // Apply padding if non-zero
    let input_padded = if pads.iter().any(|&p| p > 0) {
        pad_nhwc(
            &input_nhwc,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        )?
    } else {
        input_nhwc
    };

    if group == 1 {
        let w_nhwc = oihw_to_khwc_cout(weight)?;
        let out_nhwc = conv2d_nhwc(&input_padded, &w_nhwc, bias, sh, sw).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        let out_nchw = nhwc_to_nchw(&out_nhwc)?;
        env.insert(node.outputs[0].clone(), out_nchw);
    } else if group as usize == o_ch && group as usize == input.shape()[1] {
        // Depthwise: each group has 1 input channel and 1 output channel
        // Convert to yscv depthwise format [KH, KW, C, depth_mult=1]
        let c = group;
        let depth_mult = o_ch / c;
        let mut dw_data = vec![0.0f32; kh * kw * c * depth_mult];
        let w_data = weight.data();
        for oc in 0..o_ch {
            let g = oc / depth_mult;
            let dm = oc % depth_mult;
            for ki in 0..kh {
                for kj in 0..kw {
                    let src = ((oc * i_per_g) * kh + ki) * kw + kj;
                    let dst = ((ki * kw + kj) * c + g) * depth_mult + dm;
                    dw_data[dst] = w_data[src];
                }
            }
        }
        let dw_kernel = Tensor::from_vec(vec![kh, kw, c, depth_mult], dw_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        let out_nhwc = yscv_kernels::depthwise_conv2d_nhwc(&input_padded, &dw_kernel, bias, sh, sw)
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        let out_nchw = nhwc_to_nchw(&out_nhwc)?;
        env.insert(node.outputs[0].clone(), out_nchw);
    } else {
        // Grouped convolution: split input channels, run per-group conv, concat
        let in_shape = input_padded.shape();
        let (n, ih, iw, total_ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let ic_per_group = total_ic / group;
        let oc_per_group = o_ch / group;
        let oh = (ih - kh) / sh + 1;
        let ow = (iw - kw) / sw + 1;
        let mut out_data = vec![0.0f32; n * oh * ow * o_ch];

        let in_data = input_padded.data();
        let w_data = weight.data();

        for batch in 0..n {
            for g in 0..group {
                let ic_start = g * ic_per_group;
                let oc_start = g * oc_per_group;
                for or in 0..oh {
                    for oc_col in 0..ow {
                        for oc in 0..oc_per_group {
                            let abs_oc = oc_start + oc;
                            let mut val = if let Some(b) = bias {
                                b.data()[abs_oc]
                            } else {
                                0.0
                            };
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ir = or * sh + ki;
                                    let ic_pos = oc_col * sw + kj;
                                    for ci in 0..ic_per_group {
                                        let in_idx = ((batch * ih + ir) * iw + ic_pos) * total_ic
                                            + ic_start
                                            + ci;
                                        let w_idx =
                                            ((abs_oc * ic_per_group + ci) * kh + ki) * kw + kj;
                                        val += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                            let out_idx = ((batch * oh + or) * ow + oc_col) * o_ch + abs_oc;
                            out_data[out_idx] = val;
                        }
                    }
                }
            }
        }
        let out_nhwc = Tensor::from_vec(vec![n, oh, ow, o_ch], out_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        let out_nchw = nhwc_to_nchw(&out_nhwc)?;
        env.insert(node.outputs[0].clone(), out_nchw);
    }

    Ok(())
}

pub(super) fn exec_conv_transpose(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;

    let input_nhwc = nchw_to_nhwc(input)?;
    let w_shape = weight.shape();
    // ONNX ConvTranspose weight: [C_in, C_out, KH, KW]
    let (ic, oc, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    let wd = weight.data();
    let mut w_nhwc = vec![0.0f32; kh * kw * ic * oc];
    for ci in 0..ic {
        for co in 0..oc {
            for ky in 0..kh {
                for kx in 0..kw {
                    w_nhwc[((ky * kw + kx) * ic + ci) * oc + co] =
                        wd[((ci * oc + co) * kh + ky) * kw + kx];
                }
            }
        }
    }
    let w_t =
        Tensor::from_vec(vec![kh, kw, ic, oc], w_nhwc).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let n = input_nhwc.shape()[0];
    let ih = input_nhwc.shape()[1];
    let iw = input_nhwc.shape()[2];
    let oh = (ih - 1) * sh + kh;
    let ow = (iw - 1) * sw + kw;

    let bias_data = match &bias {
        Some(b) => b.data().to_vec(),
        None => vec![0.0f32; oc],
    };

    let in_d = input_nhwc.data();
    let w_d = w_t.data();
    let mut out = vec![0.0f32; n * oh * ow * oc];

    for b in 0..n {
        for iy in 0..ih {
            for ix in 0..iw {
                for ci in 0..ic {
                    let in_val = in_d[((b * ih + iy) * iw + ix) * ic + ci];
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let oy = iy * sh + ky;
                            let ox = ix * sw + kx;
                            for co in 0..oc {
                                let w_val = w_d[((ky * kw + kx) * ic + ci) * oc + co];
                                out[((b * oh + oy) * ow + ox) * oc + co] += in_val * w_val;
                            }
                        }
                    }
                }
            }
        }
        for oy in 0..oh {
            for ox in 0..ow {
                for co in 0..oc {
                    out[((b * oh + oy) * ow + ox) * oc + co] += bias_data[co];
                }
            }
        }
    }

    let out_nhwc =
        Tensor::from_vec(vec![n, oh, ow, oc], out).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let out_nchw = nhwc_to_nchw(&out_nhwc)?;
    env.insert(node.outputs[0].clone(), out_nchw);
    Ok(())
}

pub(super) fn exec_qlinear_conv(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let x = get_tensor(env, &node.name, &node.inputs[0])?;
    let x_scale = get_tensor(env, &node.name, &node.inputs[1])?.data()[0];
    let x_zp = get_tensor(env, &node.name, &node.inputs[2])?.data()[0];
    let w = get_tensor(env, &node.name, &node.inputs[3])?;
    let w_scale = get_tensor(env, &node.name, &node.inputs[4])?.data()[0];
    let w_zp = get_tensor(env, &node.name, &node.inputs[5])?.data()[0];
    let y_scale = get_tensor(env, &node.name, &node.inputs[6])?.data()[0];
    let y_zp = get_tensor(env, &node.name, &node.inputs[7])?.data()[0];
    let bias = if node.inputs.len() > 8 && !node.inputs[8].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[8])?.clone())
    } else {
        None
    };

    let deq_x: Vec<f32> = x.data().iter().map(|&v| (v - x_zp) * x_scale).collect();
    let deq_w: Vec<f32> = w.data().iter().map(|&v| (v - w_zp) * w_scale).collect();

    let deq_x_t =
        Tensor::from_vec(x.shape().to_vec(), deq_x).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let deq_w_t =
        Tensor::from_vec(w.shape().to_vec(), deq_w).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let float_node = OnnxNode {
        name: node.name.clone(),
        op_type: "Conv".to_string(),
        inputs: vec!["__qx".into(), "__qw".into(), "__qb".into()],
        outputs: vec!["__qconv_out".into()],
        attributes: node.attributes.clone(),
    };
    env.insert("__qx".into(), deq_x_t);
    env.insert("__qw".into(), deq_w_t);
    if let Some(b) = bias {
        env.insert("__qb".into(), b);
    }
    exec_conv(&float_node, env)?;
    let float_out = env
        .remove("__qconv_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__qconv_out".into(),
        })?;
    env.remove("__qx");
    env.remove("__qw");
    env.remove("__qb");

    let quant: Vec<f32> = float_out
        .data()
        .iter()
        .map(|&v| (v / y_scale + y_zp).round().clamp(-128.0, 127.0))
        .collect();
    let out = Tensor::from_vec(float_out.shape().to_vec(), quant).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_conv_integer(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let x = get_tensor(env, &node.name, &node.inputs[0])?;
    let w = get_tensor(env, &node.name, &node.inputs[1])?;
    let x_zp = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0]
    } else {
        0.0
    };
    let w_zp = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        get_tensor(env, &node.name, &node.inputs[3])?.data()[0]
    } else {
        0.0
    };

    let deq_x: Vec<f32> = x.data().iter().map(|&v| v - x_zp).collect();
    let deq_w: Vec<f32> = w.data().iter().map(|&v| v - w_zp).collect();

    let t_x = Tensor::from_vec(x.shape().to_vec(), deq_x).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    let t_w = Tensor::from_vec(w.shape().to_vec(), deq_w).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let conv_node = OnnxNode {
        name: node.name.clone(),
        op_type: "Conv".to_string(),
        inputs: vec!["__ci_x".into(), "__ci_w".into(), "".into()],
        outputs: vec!["__ci_out".into()],
        attributes: node.attributes.clone(),
    };
    env.insert("__ci_x".into(), t_x);
    env.insert("__ci_w".into(), t_w);
    exec_conv(&conv_node, env)?;
    let out = env
        .remove("__ci_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__ci_out".into(),
        })?;
    env.remove("__ci_x");
    env.remove("__ci_w");

    let rounded: Vec<f32> = out.data().iter().map(|&v| v.round()).collect();
    let result =
        Tensor::from_vec(out.shape().to_vec(), rounded).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

// ── layout conversion helpers ──────────────────────────────────────

pub(super) fn nchw_to_nhwc(input: &Tensor) -> Result<Tensor, OnnxError> {
    input
        .permute(&[0, 2, 3, 1])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })
}

pub(super) fn nhwc_to_nchw(input: &Tensor) -> Result<Tensor, OnnxError> {
    input
        .permute(&[0, 3, 1, 2])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })
}

/// Convert ONNX Conv weight [O, I, KH, KW] to yscv [KH, KW, I, O]
pub(super) fn oihw_to_khwc_cout(weight: &Tensor) -> Result<Tensor, OnnxError> {
    weight
        .permute(&[2, 3, 1, 0])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })
}

/// Zero-pad an NHWC tensor on H/W dimensions.
pub(super) fn pad_nhwc(
    input: &Tensor,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
) -> Result<Tensor, OnnxError> {
    pad_nhwc_val(input, top, left, bottom, right, 0.0)
}

pub(super) fn pad_nhwc_val(
    input: &Tensor,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
    val: f32,
) -> Result<Tensor, OnnxError> {
    let shape = input.shape();
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let oh = h + top + bottom;
    let ow = w + left + right;
    let mut out = vec![val; n * oh * ow * c];
    let in_data = input.data();
    for batch in 0..n {
        for row in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    let src = ((batch * h + row) * w + col) * c + ch;
                    let dst = ((batch * oh + row + top) * ow + col + left) * c + ch;
                    out[dst] = in_data[src];
                }
            }
        }
    }
    Tensor::from_vec(vec![n, oh, ow, c], out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}
