//! Execution kernels and backend abstraction for yscv.
//!
//! ## GPU Inference (Cross-Platform via wgpu)
//!
//! The `gpu` feature enables compute shader acceleration via wgpu —
//! Vulkan (Linux/Windows/Android), Metal (macOS/iOS), DX12 (Windows).
//! No CUDA dependency. GPU-accelerated operations:
//! - Matrix multiplication (tiled 16×16 workgroups)
//! - Elementwise: add, sub, mul
//! - Activations: relu, sigmoid
//! - Normalization: batch_norm, layer_norm, group_norm, rms_norm, softmax
//! - Convolution: conv2d, depthwise_conv2d, separable_conv2d, transpose_conv2d
//! - Pooling: max_pool2d, avg_pool2d
//!
//! GPU training (backward passes) is on the roadmap.
//! CPU backend is fully optimized with NEON/AVX/SSE SIMD on all platforms.
#![deny(unsafe_code)]

mod core;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unsafe_code)]
pub mod metal_backend;

pub use core::*;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
pub use metal_backend::metal_conv::{MetalConv, MetalInference};
