//! ONNX model import, inference execution, and export for yscv.

pub const CRATE_ID: &str = "yscv-onnx";

mod dtype;
#[path = "error.rs"]
mod error;
#[path = "exporter.rs"]
mod exporter;
#[path = "loader.rs"]
mod loader;
#[path = "optimizer.rs"]
mod optimizer;
mod proto;
mod runner;

pub use dtype::{OnnxDtype, OnnxTensorData};
pub use error::OnnxError;
pub use exporter::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, export_onnx_model,
    export_onnx_model_to_file,
};
pub use loader::{
    OnnxAttribute, OnnxModel, OnnxNode, OnnxTensor, load_onnx_model, load_onnx_model_from_file,
};
pub use optimizer::{
    GraphStats, fold_constants, fold_conv_bn, fuse_bn_relu, fuse_conv_relu, graph_stats,
    optimize_onnx_graph,
};
pub use runner::run_onnx_model;

#[cfg(test)]
mod tests;
