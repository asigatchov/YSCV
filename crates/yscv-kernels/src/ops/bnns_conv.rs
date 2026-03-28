//! BNNS (Apple Accelerate) accelerated convolution.
//!
//! Operates on **NCHW** data directly — no transpose needed when the caller
//! keeps tensors in NCHW layout (ONNX native).  This avoids the per-layer
//! NHWC ↔ NCHW conversion overhead that makes per-layer dispatch impractical.

#![allow(non_upper_case_globals)]

use std::ptr;

use yscv_tensor::Tensor;

// ── BNNS constants ──────────────────────────────────────────────────────────

const BNNSDataTypeFloat32: u32 = 0x10020;
const BNNSDataLayoutVector: u32 = 0x10000;
const BNNSDataLayoutImageCHW: u32 = 0x30000;
const BNNSDataLayoutConvolutionWeightsOIHW: u32 = 0x40000;
const BNNSActivationFunctionIdentity: u32 = 0;
const BNNSActivationFunctionRectifiedLinear: u32 = 1;
const BNNSActivationFunctionSiLU: u32 = 31;

// ── BNNS FFI types ──────────────────────────────────────────────────────────

type BNNSFilter = *mut std::ffi::c_void;

#[repr(C)]
struct BNNSNDArrayDescriptor {
    flags: u32,
    layout: u32,
    size: [usize; 8],
    stride: [usize; 8],
    data: *mut std::ffi::c_void,
    data_type: u32,
    table_data: *const std::ffi::c_void,
    table_data_type: u32,
    data_scale: f32,
    data_bias: f32,
}

#[repr(C)]
struct BNNSActivation {
    function: u32,
    alpha: f32,
    beta: f32,
    iscale: i32,
    ioffset: i32,
    ishift: i32,
    iscale_per_channel: *const i32,
    ioffset_per_channel: *const i32,
    ishift_per_channel: *const i32,
}

#[repr(C)]
struct BNNSLayerParametersConvolution {
    i_desc: BNNSNDArrayDescriptor,
    w_desc: BNNSNDArrayDescriptor,
    o_desc: BNNSNDArrayDescriptor,
    bias: BNNSNDArrayDescriptor,
    activation: BNNSActivation,
    x_stride: usize,
    y_stride: usize,
    x_dilation_stride: usize,
    y_dilation_stride: usize,
    x_padding: usize,
    y_padding: usize,
    groups: usize,
    pad: [usize; 4],
}

#[repr(C)]
struct BNNSFilterParameters {
    flags: u32,
    n_threads: usize,
    alloc_memory: *const std::ffi::c_void,
    free_memory: *const std::ffi::c_void,
}

#[allow(unsafe_code)]
unsafe extern "C" {
    fn BNNSFilterCreateLayerConvolution(
        layer_params: *const BNNSLayerParametersConvolution,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    fn BNNSFilterApplyBatch(
        filter: BNNSFilter,
        batch_size: usize,
        input: *const std::ffi::c_void,
        in_stride: usize,
        output: *mut std::ffi::c_void,
        out_stride: usize,
    ) -> i32;

    fn BNNSFilterDestroy(filter: BNNSFilter);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn zeros_ndarray() -> BNNSNDArrayDescriptor {
    BNNSNDArrayDescriptor {
        flags: 0,
        layout: 0,
        size: [0; 8],
        stride: [0; 8],
        data: ptr::null_mut(),
        data_type: 0,
        table_data: ptr::null(),
        table_data_type: 0,
        data_scale: 1.0,
        data_bias: 0.0,
    }
}

fn make_activation(fused: BnnsActivation) -> BNNSActivation {
    let func = match fused {
        BnnsActivation::None => BNNSActivationFunctionIdentity,
        BnnsActivation::Relu => BNNSActivationFunctionRectifiedLinear,
        BnnsActivation::Silu => BNNSActivationFunctionSiLU,
    };
    BNNSActivation {
        function: func,
        alpha: 0.0,
        beta: 0.0,
        iscale: 0,
        ioffset: 0,
        ishift: 0,
        iscale_per_channel: ptr::null(),
        ioffset_per_channel: ptr::null(),
        ishift_per_channel: ptr::null(),
    }
}

// ── Public types ────────────────────────────────────────────────────────────

/// Fused activation to apply inside the BNNS conv call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnnsActivation {
    None,
    Relu,
    Silu,
}

/// Parameters for a BNNS convolution (NCHW layout).
#[derive(Clone, Debug)]
pub struct BnnsConvParams {
    pub batch: usize,
    pub in_c: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_c: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub groups: usize,
    pub activation: BnnsActivation,
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Run conv2d via Apple BNNS on **NCHW** tensors.
///
/// - `input`:  NCHW `[N, in_C, in_H, in_W]`
/// - `weight`: OIHW `[out_C, in_C/groups, kH, kW]`
/// - `bias`:   1-D `[out_C]` or `None`
///
/// Returns NCHW `[N, out_C, out_H, out_W]`, or `None` on failure.
#[allow(unsafe_code)]
pub fn conv2d_nchw_bnns(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    p: &BnnsConvParams,
) -> Option<Tensor> {
    let in_data = input.data();
    let w_data = weight.data();
    let b_data = bias.map(Tensor::data);

    // ── Input descriptor ──
    let mut i_desc = zeros_ndarray();
    i_desc.layout = BNNSDataLayoutImageCHW;
    i_desc.data_type = BNNSDataTypeFloat32;
    i_desc.size[0] = p.in_w;
    i_desc.size[1] = p.in_h;
    i_desc.size[2] = p.in_c;

    // ── Weight descriptor: OIHW ──
    let ic_per_group = p.in_c / p.groups;
    let mut w_desc = zeros_ndarray();
    w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    w_desc.data_type = BNNSDataTypeFloat32;
    w_desc.data = w_data.as_ptr() as *mut _;
    w_desc.size[0] = p.kw;
    w_desc.size[1] = p.kh;
    w_desc.size[2] = ic_per_group;
    w_desc.size[3] = p.out_c; // total output channels

    // ── Output descriptor ──
    let mut o_desc = zeros_ndarray();
    o_desc.layout = BNNSDataLayoutImageCHW;
    o_desc.data_type = BNNSDataTypeFloat32;
    o_desc.size[0] = p.out_w;
    o_desc.size[1] = p.out_h;
    o_desc.size[2] = p.out_c;

    // ── Bias ──
    let mut bias_desc = zeros_ndarray();
    if let Some(b) = b_data {
        bias_desc.layout = BNNSDataLayoutVector;
        bias_desc.data_type = BNNSDataTypeFloat32;
        bias_desc.data = b.as_ptr() as *mut _;
        bias_desc.size[0] = p.out_c;
    }

    // Use asymmetric padding via `pad` field
    let params = BNNSLayerParametersConvolution {
        i_desc,
        w_desc,
        o_desc,
        bias: bias_desc,
        activation: make_activation(p.activation),
        x_stride: p.stride_w,
        y_stride: p.stride_h,
        x_dilation_stride: 0,
        y_dilation_stride: 0,
        x_padding: 0,
        y_padding: 0,
        groups: p.groups,
        pad: [p.pad_left, p.pad_right, p.pad_top, p.pad_bottom],
    };

    unsafe {
        let filter = BNNSFilterCreateLayerConvolution(&params, ptr::null());
        if filter.is_null() {
            return None;
        }

        let out_elems = p.batch * p.out_c * p.out_h * p.out_w;
        let mut output = vec![0.0f32; out_elems];
        let in_stride = p.in_c * p.in_h * p.in_w;
        let out_stride = p.out_c * p.out_h * p.out_w;

        let ret = BNNSFilterApplyBatch(
            filter,
            p.batch,
            in_data.as_ptr() as *const _,
            in_stride,
            output.as_mut_ptr() as *mut _,
            out_stride,
        );

        BNNSFilterDestroy(filter);

        if ret != 0 {
            return None;
        }

        Tensor::from_vec(vec![p.batch, p.out_c, p.out_h, p.out_w], output).ok()
    }
}
