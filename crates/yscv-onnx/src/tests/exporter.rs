use super::*;
use crate::exporter::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, export_onnx_model,
};

#[test]
fn export_roundtrip_relu_graph() {
    let graph = OnnxExportGraph {
        nodes: vec![OnnxExportNode {
            op_type: "Relu".into(),
            name: "relu0".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attributes: vec![],
        }],
        initializers: vec![],
        inputs: vec![OnnxExportValueInfo {
            name: "x".into(),
            shape: vec![1, 4],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "y".into(),
            shape: vec![1, 4],
        }],
    };
    let bytes = export_onnx_model(&graph, "yscv-test", "relu_model").unwrap();
    let model = load_onnx_model(&bytes).unwrap();
    assert_eq!(model.node_count(), 1);
    assert_eq!(model.nodes[0].op_type, "Relu");

    let input = Tensor::from_vec(vec![1, 4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    assert_eq!(result["y"].data(), &[0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn export_roundtrip_gemm_with_weights() {
    let weight = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.5, -0.5]).unwrap();

    let graph = OnnxExportGraph {
        nodes: vec![OnnxExportNode {
            op_type: "Gemm".into(),
            name: "fc".into(),
            inputs: vec!["x".into(), "w".into(), "b".into()],
            outputs: vec!["y".into()],
            attributes: vec![OnnxExportAttr::Int("transB".into(), 1)],
        }],
        initializers: vec![("w".into(), weight), ("b".into(), bias)],
        inputs: vec![OnnxExportValueInfo {
            name: "x".into(),
            shape: vec![1, 3],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "y".into(),
            shape: vec![1, 2],
        }],
    };
    let bytes = export_onnx_model(&graph, "yscv", "gemm_model").unwrap();
    let model = load_onnx_model(&bytes).unwrap();

    let input = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    let out = &result["y"];
    assert_eq!(out.shape(), &[1, 2]);
    assert!((out.data()[0] - 1.5).abs() < 1e-5); // 1*1 + 0.5
    assert!((out.data()[1] - 1.5).abs() < 1e-5); // 2*1 + (-0.5)
}

#[test]
fn export_to_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.onnx");

    let graph = OnnxExportGraph {
        nodes: vec![OnnxExportNode {
            op_type: "Relu".into(),
            name: "r".into(),
            inputs: vec!["in".into()],
            outputs: vec!["out".into()],
            attributes: vec![],
        }],
        initializers: vec![],
        inputs: vec![OnnxExportValueInfo {
            name: "in".into(),
            shape: vec![2],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "out".into(),
            shape: vec![2],
        }],
    };

    crate::exporter::export_onnx_model_to_file(&graph, "test", "test", &path).unwrap();
    let model = crate::loader::load_onnx_model_from_file(&path).unwrap();
    assert_eq!(model.node_count(), 1);
}
