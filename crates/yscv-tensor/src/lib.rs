//! Tensor types and numeric primitives for the yscv framework.
//!
//! # Tensor type
//!
//! [`Tensor`] is the core n-dimensional array. It stores contiguous, 64-byte
//! aligned `f32` data and carries its own shape and strides. FP16 and BF16
//! storage is supported via [`DType`] and the conversion helpers
//! [`Tensor::to_dtype`], [`Tensor::from_f16`], and [`Tensor::from_bf16`].
//!
//! # Supported dtypes
//!
//! | Variant | Backing store | Notes |
//! |---------|---------------|-------|
//! | `F32`   | `AlignedVec<f32>` | Default, SIMD-accelerated ops |
//! | `F16`   | `AlignedVec<u16>` | IEEE 754 half-precision bit patterns |
//! | `BF16`  | `AlignedVec<u16>` | Brain floating-point bit patterns |
//!
//! Arithmetic operations require `F32`. Convert with [`Tensor::to_dtype`]
//! before performing math on F16/BF16 tensors.
//!
//! # Broadcasting
//!
//! Binary operations (`add`, `sub`, `mul`, `div`, `pow`, etc.) follow
//! NumPy-style broadcasting rules:
//!
//! 1. Shapes are right-aligned. Missing leading dimensions are treated as 1.
//! 2. Dimensions of size 1 are stretched to match the other operand.
//! 3. If dimensions differ and neither is 1, the operation returns
//!    [`TensorError::BroadcastIncompatible`].
//!
//! Example: `[3, 1, 5] + [4, 5]` broadcasts to `[3, 4, 5]`.
#![allow(unsafe_code)]

mod core;

pub use core::*;
