//! High-level GPU session that wraps [`GpuBackend`] and provides a Tensor-in / Tensor-out API.
//!
//! `GpuSession` handles the full lifecycle of uploading tensor data to the GPU,
//! dispatching a compute kernel, reading back the result, and returning a new
//! [`Tensor`] tagged with [`Device::Cpu`].
//!
//! # Example
//! ```ignore
//! use yscv_kernels::GpuSession;
//! use yscv_tensor::Tensor;
//!
//! let session = GpuSession::new().expect("no GPU");
//! let a = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
//! let b = Tensor::from_vec(vec![3, 2], vec![1.0; 6]).unwrap();
//! let c = session.matmul(&a, &b).unwrap();
//! ```

use yscv_tensor::Tensor;

use super::gpu_backend::GpuBackend;
use crate::{
    Backend, BatchNorm2dParams, GroupNormNhwcParams, KernelError, LayerNormLastDimParams,
    RmsNormLastDimParams, SeparableConv2dParams,
};

/// A high-level GPU compute session.
///
/// Wraps [`GpuBackend`] and exposes every [`Backend`] operation as a
/// `&Tensor -> Tensor` method, hiding buffer upload / readback.
pub struct GpuSession {
    backend: GpuBackend,
}

impl GpuSession {
    /// Create a session on the best available GPU adapter.
    pub fn new() -> Result<Self, KernelError> {
        Ok(Self {
            backend: GpuBackend::new()?,
        })
    }

    /// GPU adapter name reported by the driver.
    pub fn adapter_name(&self) -> &str {
        self.backend.adapter_name()
    }

    // ── Elementwise ────────────────────────────────────────────────────

    /// Element-wise addition.
    pub fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.add(lhs, rhs)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.sub(lhs, rhs)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.mul(lhs, rhs)
    }

    /// ReLU activation.
    pub fn relu(&self, input: &Tensor) -> Tensor {
        self.backend.relu(input)
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self, input: &Tensor) -> Tensor {
        self.backend.sigmoid(input)
    }

    // ── Softmax / normalization ────────────────────────────────────────

    /// Softmax along last dimension.
    pub fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.softmax_last_dim(input)
    }

    /// Log-softmax along last dimension.
    pub fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.log_softmax_last_dim(input)
    }

    /// Log-sum-exp reduction along last dimension.
    pub fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.logsumexp_last_dim(input)
    }

    /// Layer normalization over last dimension.
    pub fn layer_norm_last_dim(
        &self,
        input: &Tensor,
        params: LayerNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.backend.layer_norm_last_dim(input, params)
    }

    /// Batch normalization (NHWC).
    pub fn batch_norm2d_nhwc(
        &self,
        input: &Tensor,
        params: BatchNorm2dParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.backend.batch_norm2d_nhwc(input, params)
    }

    /// Group normalization (NHWC).
    pub fn group_norm_nhwc(
        &self,
        input: &Tensor,
        params: GroupNormNhwcParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.backend.group_norm_nhwc(input, params)
    }

    /// RMS normalization over last dimension.
    pub fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.backend.rms_norm_last_dim(input, params)
    }

    // ── Pooling ────────────────────────────────────────────────────────

    /// Max pool 2D (NHWC).
    pub fn max_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.backend
            .max_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w)
    }

    /// Average pool 2D (NHWC).
    pub fn avg_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.backend
            .avg_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w)
    }

    // ── Convolution ────────────────────────────────────────────────────

    /// Conv2D (NHWC), kernel `[KH, KW, C_in, C_out]`.
    pub fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.backend
            .conv2d_nhwc(input, kernel, bias, stride_h, stride_w)
    }

    /// Depthwise conv2D (NHWC), kernel `[KH, KW, C, depth_multiplier]`.
    pub fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.backend
            .depthwise_conv2d_nhwc(input, kernel, bias, stride_h, stride_w)
    }

    /// Separable conv2D (NHWC).
    pub fn separable_conv2d_nhwc(
        &self,
        input: &Tensor,
        params: SeparableConv2dParams<'_>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.backend
            .separable_conv2d_nhwc(input, params, stride_h, stride_w)
    }

    // ── Linear algebra ─────────────────────────────────────────────────

    /// Rank-2 matrix multiplication: `(M x K) * (K x N) -> (M x N)`.
    pub fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.backend.matmul_2d(lhs, rhs)
    }
}
