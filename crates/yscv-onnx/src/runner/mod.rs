pub(crate) use std::collections::{HashMap, HashSet};

pub(crate) use yscv_kernels::{
    BatchNorm2dParams, add as kernel_add, avg_pool2d_nhwc, batch_norm2d_nhwc, conv2d_nhwc,
    matmul_2d, max_pool2d_nhwc, mul as kernel_mul, relu, sigmoid, softmax_last_dim,
    sub as kernel_sub,
};
pub(crate) use yscv_tensor::{DType, Tensor};

pub(crate) use crate::error::OnnxError;
pub(crate) use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};

mod compare;
mod conv;
mod elementwise;
mod gather_scatter;
mod linear;
mod misc;
mod normalization;
mod pooling;
mod reduce;
mod reshape;

use compare::*;
use conv::*;
use elementwise::*;
use gather_scatter::*;
use linear::*;
use misc::*;
use normalization::*;
use pooling::*;
use reduce::*;
use reshape::*;

/// A tensor environment backed by a `Vec<Option<Tensor>>` for O(1) lookups
/// by integer index. Tensor names are mapped to dense integer IDs during
/// construction, eliminating string hashing in the hot execution loop.
///
/// This type exposes a `HashMap`-like API so that existing node-execution
/// functions can remain unchanged.
pub(crate) struct TensorEnv {
    name_to_id: HashMap<String, usize>,
    slots: Vec<Option<Tensor>>,
    /// Counter for dynamically allocated temporary names that were not in
    /// the pre-built mapping (e.g., "__qa", "__qb_mat").
    next_dynamic: usize,
}

impl TensorEnv {
    /// Build from the model, pre-allocating a slot for every known tensor name.
    fn from_model(model: &OnnxModel) -> Self {
        let mut names: HashSet<&str> = HashSet::new();
        for name in &model.inputs {
            names.insert(name.as_str());
        }
        for name in &model.outputs {
            names.insert(name.as_str());
        }
        for name in model.initializers.keys() {
            names.insert(name.as_str());
        }
        for node in &model.nodes {
            for name in &node.inputs {
                names.insert(name.as_str());
            }
            for name in &node.outputs {
                names.insert(name.as_str());
            }
        }
        let name_to_id: HashMap<String, usize> = names
            .into_iter()
            .enumerate()
            .map(|(id, name)| (name.to_string(), id))
            .collect();
        let num_slots = name_to_id.len();
        TensorEnv {
            next_dynamic: num_slots,
            name_to_id,
            slots: vec![None; num_slots],
        }
    }

    /// Look up a tensor by name.
    #[inline]
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> {
        let id = self.name_to_id.get(name)?;
        self.slots[*id].as_ref()
    }

    /// Insert a tensor by name. If the name is unknown, a new slot is
    /// allocated dynamically (this handles temporary names created by
    /// quantization ops, etc.).
    #[inline]
    pub(crate) fn insert(&mut self, name: String, tensor: Tensor) {
        if let Some(&id) = self.name_to_id.get(&name) {
            self.slots[id] = Some(tensor);
        } else {
            let id = self.next_dynamic;
            self.next_dynamic += 1;
            self.name_to_id.insert(name, id);
            self.slots.push(Some(tensor));
        }
    }

    /// Get a mutable reference to a tensor by name.
    #[inline]
    pub(crate) fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        let id = *self.name_to_id.get(name)?;
        self.slots[id].as_mut()
    }

    /// Remove a tensor by name (sets the slot to `None`).
    #[inline]
    pub(crate) fn remove(&mut self, name: &str) -> Option<Tensor> {
        let id = *self.name_to_id.get(name)?;
        self.slots[id].take()
    }

    /// Create an alias: make `alias_name` refer to the same tensor as `target_name`.
    /// This copies the tensor from `target_name` into the slot for `alias_name`.
    #[inline]
    pub(crate) fn alias(&mut self, alias_name: &str, target_name: &str) {
        let tensor = self
            .name_to_id
            .get(target_name)
            .and_then(|&id| self.slots[id].clone());
        if let Some(tensor) = tensor {
            if let Some(&alias_id) = self.name_to_id.get(alias_name) {
                self.slots[alias_id] = Some(tensor);
            } else {
                let id = self.next_dynamic;
                self.next_dynamic += 1;
                self.name_to_id.insert(alias_name.to_string(), id);
                self.slots.push(Some(tensor));
            }
        }
    }
}

/// Runs inference on a loaded ONNX model with the given named inputs.
///
/// Returns a map of output-name -> tensor for the graph's declared outputs.
pub fn run_onnx_model(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let mut env = TensorEnv::from_model(model);

    for (name, tensor) in &model.initializers {
        env.insert(name.clone(), tensor.clone());
    }
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    // --- Operator fusion: scan for fusible patterns ---
    // Build a set of node indices that should be skipped because they were
    // fused into the preceding node.  We also create synthetic "fused" nodes
    // that carry a combined op_type (e.g. "Conv_Relu").
    let nodes = &model.nodes;
    let mut skip: HashSet<usize> = HashSet::new();

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }

        // --- Conv → BatchNorm → Relu 3-node fusion ---
        if node.op_type == "Conv"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "BatchNormalization"
            && !next.inputs.is_empty()
            && next.inputs[0] == node.outputs[0]
            && let Some(next2) = nodes.get(i + 2)
            && next2.op_type == "Relu"
            && next2.inputs.len() == 1
            && next2.inputs[0] == next.outputs[0]
        {
            execute_node(node, &mut env)?;
            execute_node(next, &mut env)?;
            if let Some(tensor) = env.get_mut(&next.outputs[0]) {
                for v in tensor.data_mut() {
                    *v = v.max(0.0);
                }
            }
            env.alias(&next2.outputs[0], &next.outputs[0]);
            skip.insert(i + 1);
            skip.insert(i + 2);
            continue;
        }

        // --- Conv + Relu fusion ---
        if node.op_type == "Conv"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            // Execute conv, then apply relu in-place, then alias output
            execute_node(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                for v in tensor.data_mut() {
                    *v = v.max(0.0);
                }
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- BatchNormalization + Relu fusion ---
        if node.op_type == "BatchNormalization"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                for v in tensor.data_mut() {
                    *v = v.max(0.0);
                }
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- Gemm + Relu fusion ---
        if node.op_type == "Gemm"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                for v in tensor.data_mut() {
                    *v = v.max(0.0);
                }
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- Add + Relu fusion ---
        if node.op_type == "Add"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                for v in tensor.data_mut() {
                    *v = v.max(0.0);
                }
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- MatMul + Add fusion (Gemm-like) ---
        if node.op_type == "MatMul"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Add"
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            // Execute matmul then add back-to-back (skip separate Add node)
            execute_node(node, &mut env)?;
            execute_node(next, &mut env)?;
            skip.insert(i + 1);
            continue;
        }

        execute_node(node, &mut env)?;
    }

    let mut result = HashMap::new();
    for name in &model.outputs {
        if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        } else {
            eprintln!("warning: ONNX output '{}' not found in environment", name);
        }
    }
    Ok(result)
}

#[inline]
pub(crate) fn get_tensor<'a>(
    env: &'a TensorEnv,
    node_name: &str,
    input_name: &str,
) -> Result<&'a Tensor, OnnxError> {
    env.get(input_name).ok_or_else(|| OnnxError::MissingInput {
        node: node_name.to_string(),
        input: input_name.to_string(),
    })
}

#[inline]
pub(crate) fn get_attr_ints(node: &OnnxNode, name: &str) -> Option<Vec<i64>> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_int(node: &OnnxNode, name: &str) -> Option<i64> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Int(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_float(node: &OnnxNode, name: &str) -> Option<f32> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Float(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_string(node: &OnnxNode, name: &str) -> Option<String> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::String(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Converts inputs in the environment to f32 before executing a node, then converts
/// outputs back to the original dtype. Ops that handle dtypes themselves (Cast, Shape,
/// Identity, quantization ops) are exempt.
fn execute_node(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    // Ops that should NOT have automatic dtype conversion
    let dtype_exempt = matches!(
        node.op_type.as_str(),
        "Cast"
            | "Shape"
            | "Identity"
            | "Constant"
            | "ConstantOfShape"
            | "QuantizeLinear"
            | "DequantizeLinear"
            | "DynamicQuantizeLinear"
            | "QLinearConv"
            | "QLinearMatMul"
            | "MatMulInteger"
            | "ConvInteger"
    );

    // Detect original dtype from first input (if any) and convert inputs to f32
    let orig_dtype = if !dtype_exempt && !node.inputs.is_empty() {
        let first_dt = node
            .inputs
            .iter()
            .filter_map(|name| env.get(name))
            .map(|t| t.dtype())
            .find(|&dt| dt != DType::F32);

        if let Some(dt) = first_dt {
            // Convert all non-f32 inputs to f32 in-place
            for input_name in &node.inputs {
                if let Some(tensor) = env.get(input_name)
                    && tensor.dtype() != DType::F32
                {
                    let converted = tensor.to_dtype(DType::F32);
                    env.insert(input_name.clone(), converted);
                }
            }
            Some(dt)
        } else {
            None
        }
    } else {
        None
    };

    // Execute the actual op
    execute_node_inner(node, env)?;

    // Convert outputs back to original dtype if needed
    if let Some(dt) = orig_dtype {
        for output_name in &node.outputs {
            if let Some(tensor) = env.get(output_name)
                && tensor.dtype() == DType::F32
            {
                let converted = tensor.to_dtype(dt);
                env.insert(output_name.clone(), converted);
            }
        }
    }

    Ok(())
}

fn execute_node_inner(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        "Conv" => exec_conv(node, env),
        "Relu" => exec_relu(node, env),
        "MaxPool" => exec_max_pool(node, env),
        "AveragePool" => exec_avg_pool(node, env),
        "GlobalAveragePool" => exec_global_avg_pool(node, env),
        "BatchNormalization" => exec_batch_norm(node, env),
        "Flatten" => exec_flatten(node, env),
        "Gemm" => exec_gemm(node, env),
        "MatMul" => exec_matmul(node, env),
        "Add" => exec_add(node, env),
        "Sub" => exec_sub(node, env),
        "Mul" => exec_mul(node, env),
        "Softmax" => exec_softmax(node, env),
        "Sigmoid" => exec_sigmoid(node, env),
        "Reshape" => exec_reshape(node, env),
        "Transpose" => exec_transpose(node, env),
        "Concat" => exec_concat(node, env),
        "Unsqueeze" => exec_unsqueeze(node, env),
        "Squeeze" => exec_squeeze(node, env),
        "Clip" => exec_clip(node, env),
        "Shape" => exec_shape(node, env),
        "Gather" => exec_gather(node, env),
        "Constant" => exec_constant(node, env),
        "Dropout" => exec_dropout(node, env),
        "Pad" => exec_pad(node, env),
        "Pow" => exec_pow(node, env),
        "Sqrt" => exec_sqrt(node, env),
        "Exp" => exec_exp(node, env),
        "Log" => exec_log(node, env),
        "Neg" => exec_neg(node, env),
        "Abs" => exec_abs(node, env),
        "Reciprocal" => exec_reciprocal(node, env),
        "Tanh" => exec_tanh(node, env),
        "Floor" => exec_floor(node, env),
        "Ceil" => exec_ceil(node, env),
        "Equal" => exec_cmp(node, env, 0),
        "Greater" => exec_cmp(node, env, 1),
        "Less" => exec_cmp(node, env, 2),
        "Where" => exec_where(node, env),
        "ReduceMean" => exec_reduce_mean(node, env),
        "ReduceSum" => exec_reduce_sum(node, env),
        "Split" => exec_split(node, env),
        "Slice" => exec_slice(node, env),
        "Expand" => exec_expand(node, env),
        "Tile" => exec_tile(node, env),
        "Cast" => exec_cast(node, env),
        "Div" => exec_div(node, env),
        "Min" => exec_min_max(node, env, false),
        "Max" => exec_min_max(node, env, true),
        "ReduceMax" => exec_reduce_max(node, env),
        "ConvTranspose" => exec_conv_transpose(node, env),
        "Resize" => exec_resize(node, env),
        "LeakyRelu" => exec_leaky_relu(node, env),
        "Elu" => exec_elu(node, env),
        "ReduceMin" => exec_reduce_min(node, env),
        "ReduceProd" => exec_reduce_prod(node, env),
        "Identity" => exec_identity(node, env),
        "QuantizeLinear" => exec_quantize_linear(node, env),
        "DequantizeLinear" => exec_dequantize_linear(node, env),
        "Gelu" => exec_gelu(node, env),
        "Erf" => exec_erf(node, env),
        "HardSigmoid" => exec_hard_sigmoid(node, env),
        "InstanceNormalization" => exec_instance_norm(node, env),
        "LpNormalization" => exec_lp_norm(node, env),
        "Upsample" => exec_resize(node, env),
        "Selu" => exec_selu(node, env),
        "Celu" => exec_celu(node, env),
        "ThresholdedRelu" => exec_thresholded_relu(node, env),
        "Hardmax" => exec_hardmax(node, env),
        "OneHot" => exec_onehot(node, env),
        "Range" => exec_range(node, env),
        "NonZero" => exec_nonzero(node, env),
        "LayerNormalization" => exec_layer_norm(node, env),
        "GatherElements" => exec_gather_elements(node, env),
        "ScatterElements" => exec_scatter_elements(node, env),
        "Einsum" => exec_einsum(node, env),
        "ReduceL2" => exec_reduce_l2(node, env),
        "ReduceL1" => exec_reduce_l1(node, env),
        "CumSum" => exec_cumsum(node, env),
        "ArgMax" => exec_argmax(node, env),
        "ArgMin" => exec_argmin(node, env),
        "TopK" => exec_topk(node, env),
        "ScatterND" => exec_scatter_nd(node, env),
        "GatherND" => exec_gather_nd(node, env),
        "DepthToSpace" => exec_depth_to_space(node, env),
        "SpaceToDepth" => exec_space_to_depth(node, env),
        "GridSample" => exec_grid_sample(node, env),
        "RoiAlign" => exec_roi_align(node, env),
        "Compress" => exec_compress(node, env),
        "QLinearConv" => exec_qlinear_conv(node, env),
        "QLinearMatMul" => exec_qlinear_matmul(node, env),
        "MatMulInteger" => exec_matmul_integer(node, env),
        "ConvInteger" => exec_conv_integer(node, env),
        "DynamicQuantizeLinear" => exec_dynamic_quantize_linear(node, env),
        "Not" => exec_not(node, env),
        "And" => exec_logical_bin(node, env, 0),
        "Or" => exec_logical_bin(node, env, 1),
        "Xor" => exec_logical_bin(node, env, 2),
        "Sin" => exec_tensor_op(node, env, |t| t.sin()),
        "Cos" => exec_tensor_op(node, env, |t| t.cos()),
        "Tan" => exec_unary(node, env, |v| v.tan()),
        "Asin" => exec_unary(node, env, |v| v.asin()),
        "Acos" => exec_unary(node, env, |v| v.acos()),
        "Atan" => exec_unary(node, env, |v| v.atan()),
        "Sinh" => exec_unary(node, env, |v| v.sinh()),
        "Cosh" => exec_unary(node, env, |v| v.cosh()),
        "Asinh" => exec_unary(node, env, |v| v.asinh()),
        "Acosh" => exec_unary(node, env, |v| v.acosh()),
        "Atanh" => exec_unary(node, env, |v| v.atanh()),
        "Round" => exec_unary(node, env, |v| v.round()),
        "Sign" => exec_unary(node, env, |v| v.signum()),
        "IsNaN" => exec_unary(node, env, |v| if v.is_nan() { 1.0 } else { 0.0 }),
        "IsInf" => exec_unary(node, env, |v| if v.is_infinite() { 1.0 } else { 0.0 }),
        "Mod" => exec_mod(node, env),
        "GreaterOrEqual" => exec_cmp(node, env, 3),
        "LessOrEqual" => exec_cmp(node, env, 4),
        "BitShift" => exec_bitshift(node, env),
        "Mean" => exec_variadic_mean(node, env),
        "Sum" => exec_variadic_sum(node, env),
        "ConstantOfShape" => exec_constant_of_shape(node, env),
        "LRN" => exec_lrn(node, env),
        "Softplus" => exec_unary(node, env, |v| (1.0 + v.exp()).ln()),
        "Softsign" => exec_unary(node, env, |v| v / (1.0 + v.abs())),
        "HardSwish" => exec_unary(node, env, |v| v * ((v + 3.0).clamp(0.0, 6.0) / 6.0)),
        "Mish" => exec_unary(node, env, |v| v * (1.0 + v.exp()).ln().tanh()),
        "NonMaxSuppression" => exec_nms(node, env),
        "Conv_Relu" => {
            exec_conv(node, env)?;
            exec_relu_inplace(node, env)
        }
        "BatchNormalization_Relu" => {
            exec_batch_norm(node, env)?;
            exec_relu_inplace(node, env)
        }
        other => Err(OnnxError::UnsupportedOpType {
            op_type: other.to_string(),
        }),
    }
}
