use std::collections::{HashMap, HashSet};

use prost::Message;
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::proto::onnx;

/// A named tensor extracted from an ONNX model initializer.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub tensor: Tensor,
}

/// An ONNX operator node with its type, inputs, outputs, and attributes.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// Supported ONNX attribute value types.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Tensor(Tensor),
}

/// Parsed ONNX model containing graph topology and weight tensors.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub graph_name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub initializers: HashMap<String, Tensor>,
    pub nodes: Vec<OnnxNode>,
    /// Conv weight names that were pre-permuted OIHW → KHWC at load time.
    pub(crate) khwc_weights: HashSet<String>,
}

impl OnnxModel {
    /// Returns the weight tensor for a given initializer name, if present.
    pub fn get_initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.get(name)
    }

    /// Returns the number of operator nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Loads an ONNX model from raw protobuf bytes.
pub fn load_onnx_model(data: &[u8]) -> Result<OnnxModel, OnnxError> {
    let model_proto = onnx::ModelProto::decode(data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let graph = model_proto.graph.ok_or(OnnxError::MissingGraph)?;

    let opset_version = model_proto
        .opset_import
        .first()
        .and_then(|o| o.version)
        .unwrap_or(0);

    let inputs: Vec<String> = graph
        .input
        .iter()
        .map(|v| v.name.clone().unwrap_or_default())
        .collect();

    let outputs: Vec<String> = graph
        .output
        .iter()
        .map(|v| v.name.clone().unwrap_or_default())
        .collect();

    let mut initializers = HashMap::new();
    for init in &graph.initializer {
        let name = init.name.clone().unwrap_or_default();
        let tensor = convert_tensor_proto(init)?;
        initializers.insert(name, tensor);
    }

    let mut nodes = Vec::new();
    for node_proto in &graph.node {
        let mut attributes = HashMap::new();
        for attr in &node_proto.attribute {
            let attr_name = attr.name.clone().unwrap_or_default();
            let value = convert_attribute(attr);
            if let Some(v) = value {
                attributes.insert(attr_name, v);
            }
        }
        nodes.push(OnnxNode {
            op_type: node_proto.op_type.clone().unwrap_or_default(),
            name: node_proto.name.clone().unwrap_or_default(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            attributes,
        });
    }

    // Pre-permute group=1 Conv weights OIHW → KHWC at load time
    // so we don't pay the ~11ms permutation cost on every inference call.
    let mut khwc_weights = HashSet::new();
    for node in &nodes {
        if node.op_type != "Conv" || node.inputs.len() < 2 {
            continue;
        }
        let weight_name = &node.inputs[1];
        if khwc_weights.contains(weight_name) {
            continue;
        }
        // Only pre-permute group=1 conv weights
        let group = node
            .attributes
            .get("group")
            .and_then(|a| match a {
                OnnxAttribute::Int(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(1);
        if group != 1 {
            continue;
        }
        if let Some(w) = initializers.get(weight_name)
            && w.rank() == 4
            && let Ok(permuted) = w.permute(&[2, 3, 1, 0])
        {
            initializers.insert(weight_name.clone(), permuted);
            khwc_weights.insert(weight_name.clone());
        }
    }

    Ok(OnnxModel {
        ir_version: model_proto.ir_version.unwrap_or(0),
        opset_version,
        producer_name: model_proto.producer_name.unwrap_or_default(),
        graph_name: graph.name.unwrap_or_default(),
        inputs,
        outputs,
        initializers,
        nodes,
        khwc_weights,
    })
}

/// Loads an ONNX model from a file path.
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
pub fn load_onnx_model_from_file(
    path: impl AsRef<std::path::Path>,
) -> Result<OnnxModel, OnnxError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| OnnxError::Io {
        message: format!("{}: {e}", path.display()),
    })?;
    load_onnx_model(&data)
}

fn convert_tensor_proto(tp: &onnx::TensorProto) -> Result<Tensor, OnnxError> {
    let shape: Vec<usize> = tp.dims.iter().map(|&d| d as usize).collect();
    let expected_len: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let data_type = tp.data_type.unwrap_or(0);

    let data = match data_type {
        // FLOAT = 1
        1 => {
            if !tp.float_data.is_empty() {
                tp.float_data.clone()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // DOUBLE = 11
        11 => {
            if !tp.double_data.is_empty() {
                tp.double_data.iter().map(|&d| d as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_f64_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT64 = 7
        7 => {
            if !tp.int64_data.is_empty() {
                tp.int64_data.iter().map(|&v| v as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_i64_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT32 = 6
        6 => {
            if !tp.int32_data.is_empty() {
                tp.int32_data.iter().map(|&v| v as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_i32_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        other => {
            return Err(OnnxError::UnsupportedDataType { data_type: other });
        }
    };

    if data.len() != expected_len {
        return Err(OnnxError::InitializerShapeMismatch {
            name: tp.name.clone().unwrap_or_default(),
            expected: expected_len,
            got: data.len(),
        });
    }

    // Preserve 0-D scalar shapes: ONNX TensorProto with dims=[] is a 0-D
    // scalar, not a 1-D tensor.  Many graph patterns (Gather with scalar
    // indices → Unsqueeze → Concat for reshape targets) depend on correct
    // rank propagation.
    Tensor::from_vec(shape, data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

fn raw_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn raw_bytes_to_f64_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(8)
        .map(|c| {
            let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            v as f32
        })
        .collect()
}

fn raw_bytes_to_i64_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(8)
        .map(|c| {
            let v = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            v as f32
        })
        .collect()
}

fn raw_bytes_to_i32_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| {
            let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            v as f32
        })
        .collect()
}

fn convert_attribute(attr: &onnx::AttributeProto) -> Option<OnnxAttribute> {
    let attr_type = attr.r#type.unwrap_or(0);
    match attr_type {
        1 => Some(OnnxAttribute::Float(attr.f.unwrap_or(0.0))),
        2 => Some(OnnxAttribute::Int(attr.i.unwrap_or(0))),
        3 => {
            let s = attr
                .s
                .as_deref()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .unwrap_or_default();
            Some(OnnxAttribute::String(s))
        }
        // TENSOR — used by Constant nodes to embed full tensor values
        4 => {
            let tp = attr.t.as_ref()?;
            convert_tensor_proto(tp).ok().map(OnnxAttribute::Tensor)
        }
        6 => Some(OnnxAttribute::Floats(attr.floats.clone())),
        7 => Some(OnnxAttribute::Ints(attr.ints.clone())),
        _ => None,
    }
}
