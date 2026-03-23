use super::*;

pub(super) fn exec_gemm(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let c = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let alpha = get_attr_float(node, "alpha").unwrap_or(1.0);
    let beta_val = get_attr_float(node, "beta").unwrap_or(1.0);
    let trans_a = get_attr_int(node, "transA").unwrap_or(0) != 0;
    let trans_b = get_attr_int(node, "transB").unwrap_or(0) != 0;

    let a_final = if trans_a {
        a.transpose_2d().map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?
    } else {
        a.clone()
    };
    let b_final = if trans_b {
        b.transpose_2d().map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?
    } else {
        b.clone()
    };

    let mut out = matmul_2d(&a_final, &b_final).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    if (alpha - 1.0).abs() > f32::EPSILON {
        out = out.scale(alpha);
    }
    if let Some(c_tensor) = c {
        if (beta_val - 1.0).abs() > f32::EPSILON {
            let scaled_c = c_tensor.scale(beta_val);
            out = kernel_add(&out, &scaled_c).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        } else {
            out = kernel_add(&out, c_tensor).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        }
    }
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_matmul(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = matmul_2d(a, b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_einsum(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let equation = get_attr_string(node, "equation").unwrap_or_default();

    if equation == "ij,jk->ik" && node.inputs.len() == 2 {
        let a = get_tensor(env, &node.name, &node.inputs[0])?;
        let b = get_tensor(env, &node.name, &node.inputs[1])?;
        let result = yscv_kernels::matmul_2d(a, b).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), result);
        return Ok(());
    }

    if equation == "ij->ji" && node.inputs.len() == 1 {
        let a = get_tensor(env, &node.name, &node.inputs[0])?;
        let result = a.transpose_2d().map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), result);
        return Ok(());
    }

    Err(OnnxError::UnsupportedOpType {
        op_type: format!("Einsum({})", equation),
    })
}

pub(super) fn exec_qlinear_matmul(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let a_scale = get_tensor(env, &node.name, &node.inputs[1])?.data()[0];
    let a_zp = get_tensor(env, &node.name, &node.inputs[2])?.data()[0];
    let b = get_tensor(env, &node.name, &node.inputs[3])?;
    let b_scale = get_tensor(env, &node.name, &node.inputs[4])?.data()[0];
    let b_zp = get_tensor(env, &node.name, &node.inputs[5])?.data()[0];
    let y_scale = get_tensor(env, &node.name, &node.inputs[6])?.data()[0];
    let y_zp = get_tensor(env, &node.name, &node.inputs[7])?.data()[0];

    let deq_a: Vec<f32> = a.data().iter().map(|&v| (v - a_zp) * a_scale).collect();
    let deq_b: Vec<f32> = b.data().iter().map(|&v| (v - b_zp) * b_scale).collect();

    let deq_a_t =
        Tensor::from_vec(a.shape().to_vec(), deq_a).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let deq_b_t =
        Tensor::from_vec(b.shape().to_vec(), deq_b).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let float_node = OnnxNode {
        name: node.name.clone(),
        op_type: "MatMul".to_string(),
        inputs: vec!["__qa".into(), "__qb_mat".into()],
        outputs: vec!["__qmm_out".into()],
        attributes: HashMap::new(),
    };
    env.insert("__qa".into(), deq_a_t);
    env.insert("__qb_mat".into(), deq_b_t);
    exec_matmul(&float_node, env)?;
    let float_out = env
        .remove("__qmm_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__qmm_out".into(),
        })?;
    env.remove("__qa");
    env.remove("__qb_mat");

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

pub(super) fn exec_matmul_integer(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let a_zp = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0]
    } else {
        0.0
    };
    let b_zp = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        get_tensor(env, &node.name, &node.inputs[3])?.data()[0]
    } else {
        0.0
    };

    let deq_a: Vec<f32> = a.data().iter().map(|&v| v - a_zp).collect();
    let deq_b: Vec<f32> = b.data().iter().map(|&v| v - b_zp).collect();

    let t_a = Tensor::from_vec(a.shape().to_vec(), deq_a).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    let t_b = Tensor::from_vec(b.shape().to_vec(), deq_b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let mm_node = OnnxNode {
        name: node.name.clone(),
        op_type: "MatMul".to_string(),
        inputs: vec!["__mmi_a".into(), "__mmi_b".into()],
        outputs: vec!["__mmi_out".into()],
        attributes: HashMap::new(),
    };
    env.insert("__mmi_a".into(), t_a);
    env.insert("__mmi_b".into(), t_b);
    exec_matmul(&mm_node, env)?;
    let out = env
        .remove("__mmi_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__mmi_out".into(),
        })?;
    env.remove("__mmi_a");
    env.remove("__mmi_b");

    let rounded: Vec<f32> = out.data().iter().map(|&v| v.round()).collect();
    let result =
        Tensor::from_vec(out.shape().to_vec(), rounded).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}
