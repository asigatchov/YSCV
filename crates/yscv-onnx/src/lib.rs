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
#[cfg(feature = "gpu")]
pub use runner::gpu::profile_onnx_model_gpu;
#[cfg(feature = "gpu")]
pub use runner::gpu::run_onnx_model_gpu;
#[cfg(feature = "gpu")]
pub use runner::gpu::run_onnx_model_gpu_with;
#[cfg(feature = "gpu")]
pub use runner::gpu::{
    CompiledGpuPlan, compile_gpu_plan, run_compiled_gpu, run_compiled_gpu_fused,
    run_compiled_gpu_fused_timed,
};
#[cfg(feature = "gpu")]
pub use runner::gpu::{GpuExecAction, GpuExecPlan, plan_gpu_execution};
#[cfg(feature = "gpu")]
pub use runner::gpu::{GpuWeightCache, run_onnx_model_gpu_cached};
#[cfg(feature = "gpu")]
pub use runner::gpu::{compile_gpu_plan_f16, run_compiled_gpu_f16_fused};
pub use runner::profile_onnx_model_cpu;
pub use runner::run_onnx_model;

#[cfg(feature = "metal-backend")]
pub use runner::metal_runner::{MetalPlan, compile_metal_plan, run_metal_plan};

#[cfg(test)]
mod tests;
