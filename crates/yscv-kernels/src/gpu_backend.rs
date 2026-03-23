//! wgpu-based GPU backend — dispatches to Vulkan (Linux/Win), Metal (macOS), DX12 (Win).

use wgpu::util::DeviceExt;
use yscv_tensor::Tensor;

use crate::{
    Backend, BatchNorm2dParams, GroupNormNhwcParams, KernelError, LayerNormLastDimParams,
    RmsNormLastDimParams, SeparableConv2dParams,
};

const MIN_GPU_ELEMENTS: usize = 4096;

// ── WGSL Shaders ───────────────────────────────────────────────────

const MATMUL_WGSL: &str = r#"
struct Params { m: u32, n: u32, k: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
const T: u32 = 16u;
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    var sum = 0.0;
    let tiles = (p.k + T - 1u) / T;
    for (var t = 0u; t < tiles; t++) {
        let ac = t * T + lid.x;
        if (row < p.m && ac < p.k) { ta[lid.y * T + lid.x] = a[row * p.k + ac]; }
        else { ta[lid.y * T + lid.x] = 0.0; }
        let br = t * T + lid.y;
        if (br < p.k && col < p.n) { tb[lid.y * T + lid.x] = b[br * p.n + col]; }
        else { tb[lid.y * T + lid.x] = 0.0; }
        workgroupBarrier();
        for (var i = 0u; i < T; i++) { sum += ta[lid.y * T + i] * tb[i * T + lid.x]; }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) { out[row * p.n + col] = sum; }
}
"#;

const ELEMENTWISE_WGSL: &str = r#"
struct Params { len: u32, op: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    switch p.op {
        case 0u: { out[i] = a[i] + b[i]; }
        case 1u: { out[i] = a[i] - b[i]; }
        case 2u: { out[i] = a[i] * b[i]; }
        default: {}
    }
}
"#;

const UNARY_WGSL: &str = r#"
struct Params { len: u32, op: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    switch p.op {
        case 0u: { out[i] = max(a[i], 0.0); }
        case 1u: { out[i] = 1.0 / (1.0 + exp(-a[i])); }
        case 2u: { out[i] = exp(a[i]); }
        case 3u: { out[i] = tanh(a[i]); }
        default: {}
    }
}
"#;

const SOFTMAX_WGSL: &str = r#"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var mx = inp[base];
    for (var j = 1u; j < p.cols; j++) { mx = max(mx, inp[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < p.cols; j++) {
        let e = exp(inp[base + j] - mx);
        out[base + j] = e;
        s += e;
    }
    for (var j = 0u; j < p.cols; j++) { out[base + j] /= s; }
}
"#;

const LOG_SOFTMAX_WGSL: &str = r#"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var mx = inp[base];
    for (var j = 1u; j < p.cols; j++) { mx = max(mx, inp[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < p.cols; j++) { s += exp(inp[base + j] - mx); }
    let lse = mx + log(s);
    for (var j = 0u; j < p.cols; j++) { out[base + j] = inp[base + j] - lse; }
}
"#;

const LOGSUMEXP_WGSL: &str = r#"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var mx = inp[base];
    for (var j = 1u; j < p.cols; j++) { mx = max(mx, inp[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < p.cols; j++) { s += exp(inp[base + j] - mx); }
    out[row] = mx + log(s);
}
"#;

const CONV2D_WGSL: &str = r#"
struct Params {
    n: u32, ih: u32, iw: u32, ic: u32,
    oc: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow_idx = gid.x; let oh_idx = gid.y;
    let batch_oc = gid.z;
    let batch = batch_oc / p.oc;
    let co = batch_oc % p.oc;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var sum = bias[co];
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let iy = oh_idx * p.sh + ky;
            let ix = ow_idx * p.sw + kx;
            for (var ci = 0u; ci < p.ic; ci++) {
                let in_val = inp[((batch * p.ih + iy) * p.iw + ix) * p.ic + ci];
                let k_val = kern[((ky * p.kw + kx) * p.ic + ci) * p.oc + co];
                sum += in_val * k_val;
            }
        }
    }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * p.oc + co] = sum;
}
"#;

const POOL2D_WGSL: &str = r#"
struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    oh: u32, ow: u32, mode: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow_idx = gid.x; let oh_idx = gid.y;
    let batch_c = gid.z;
    let batch = batch_c / p.c;
    let ch = batch_c % p.c;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var val: f32;
    if (p.mode == 0u) { val = -1e38; } else { val = 0.0; }
    var cnt = 0.0;
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let iy = oh_idx * p.sh + ky;
            let ix = ow_idx * p.sw + kx;
            let v = inp[((batch * p.ih + iy) * p.iw + ix) * p.c + ch];
            if (p.mode == 0u) { val = max(val, v); }
            else { val += v; cnt += 1.0; }
        }
    }
    if (p.mode == 1u && cnt > 0.0) { val /= cnt; }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * p.c + ch] = val;
}
"#;

const BATCH_NORM_WGSL: &str = r#"
struct Params { total: u32, c: u32, eps: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> variance: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.total) { return; }
    let ch = i % p.c;
    let inv_std = 1.0 / sqrt(variance[ch] + p.eps);
    out[i] = gamma[ch] * (inp[i] - mean[ch]) * inv_std + beta[ch];
}
"#;

const LAYER_NORM_WGSL: &str = r#"
struct Params { rows: u32, cols: u32, eps: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var mu = 0.0;
    for (var j = 0u; j < p.cols; j++) { mu += inp[base + j]; }
    mu /= f32(p.cols);
    var v = 0.0;
    for (var j = 0u; j < p.cols; j++) { let d = inp[base + j] - mu; v += d * d; }
    v /= f32(p.cols);
    let inv = 1.0 / sqrt(v + p.eps);
    for (var j = 0u; j < p.cols; j++) {
        out[base + j] = gamma[j] * (inp[base + j] - mu) * inv + beta[j];
    }
}
"#;

const DEPTHWISE_CONV2D_WGSL: &str = r#"
struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    dm: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow_idx = gid.x; let oh_idx = gid.y;
    let batch_cdm = gid.z;
    let oc = p.c * p.dm;
    let batch = batch_cdm / oc;
    let cdm = batch_cdm % oc;
    let ch = cdm / p.dm;
    let dm_idx = cdm % p.dm;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var sum = bias[cdm];
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let iy = oh_idx * p.sh + ky;
            let ix = ow_idx * p.sw + kx;
            let in_val = inp[((batch * p.ih + iy) * p.iw + ix) * p.c + ch];
            let k_val = kern[((ky * p.kw + kx) * p.c + ch) * p.dm + dm_idx];
            sum += in_val * k_val;
        }
    }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * oc + cdm] = sum;
}
"#;

const TRANSPOSE_CONV2D_WGSL: &str = r#"
struct Params {
    n: u32, ih: u32, iw: u32, ic: u32,
    oc: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow_idx = gid.x; let oh_idx = gid.y;
    let batch_co = gid.z;
    let batch = batch_co / p.oc;
    let co = batch_co % p.oc;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var sum = bias[co];
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let in_y_f = i32(oh_idx) - i32(ky);
            let in_x_f = i32(ow_idx) - i32(kx);
            if (in_y_f < 0 || in_x_f < 0) { continue; }
            let in_y = u32(in_y_f);
            let in_x = u32(in_x_f);
            if (in_y % p.sh != 0u || in_x % p.sw != 0u) { continue; }
            let iy = in_y / p.sh;
            let ix = in_x / p.sw;
            if (iy >= p.ih || ix >= p.iw) { continue; }
            for (var ci = 0u; ci < p.ic; ci++) {
                let in_val = inp[((batch * p.ih + iy) * p.iw + ix) * p.ic + ci];
                let k_val = kern[((ky * p.kw + kx) * p.ic + ci) * p.oc + co];
                sum += in_val * k_val;
            }
        }
    }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * p.oc + co] = sum;
}
"#;

const TRANSPOSE_2D_WGSL: &str = r#"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= p.rows || col >= p.cols) { return; }
    out[col * p.rows + row] = inp[row * p.cols + col];
}
"#;

const GATHER_WGSL: &str = r#"
struct Params { len: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    out[i] = data[indices[i]];
}
"#;

const ATTENTION_WGSL: &str = r#"
struct Params { seq_q: u32, seq_k: u32, d_k: u32, d_v: u32, scale: f32, _pad1: u32, _pad2: u32, _pad3: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.y;
    let vi = gid.x;
    if (qi >= p.seq_q || vi >= p.d_v) { return; }
    var sum = 0.0;
    for (var ki = 0u; ki < p.seq_k; ki++) {
        var dot = 0.0;
        for (var d = 0u; d < p.d_k; d++) {
            dot += q[qi * p.d_k + d] * k[ki * p.d_k + d];
        }
        let score = dot * p.scale;
        // Simplified: no softmax in this kernel (applied separately)
        sum += score * v[ki * p.d_v + vi];
    }
    out[qi * p.d_v + vi] = sum;
}
"#;

const GROUP_NORM_WGSL: &str = r#"
struct Params { n: u32, c: u32, spatial: u32, groups: u32, eps: f32, _pad1: u32, _pad2: u32, _pad3: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.spatial;
    if (idx >= total) { return; }
    let s = idx % p.spatial;
    let c = (idx / p.spatial) % p.c;
    let n_idx = idx / (p.c * p.spatial);
    let g = c / (p.c / p.groups);
    let g_size = (p.c / p.groups) * p.spatial;
    let g_start = n_idx * p.c * p.spatial + g * g_size;
    var mean = 0.0;
    for (var i = 0u; i < g_size; i++) { mean += input[g_start + i]; }
    mean /= f32(g_size);
    var variance = 0.0;
    for (var i = 0u; i < g_size; i++) {
        let d = input[g_start + i] - mean;
        variance += d * d;
    }
    variance /= f32(g_size);
    let inv_std = 1.0 / sqrt(variance + p.eps);
    out[idx] = (input[idx] - mean) * inv_std * gamma[c] + beta[c];
}
"#;

const RMS_NORM_WGSL: &str = r#"
struct Params { rows: u32, cols: u32, eps: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var sq_sum = 0.0;
    for (var c = 0u; c < p.cols; c++) { sq_sum += input[base + c] * input[base + c]; }
    let rms = sqrt(sq_sum / f32(p.cols) + p.eps);
    let inv_rms = 1.0 / rms;
    for (var c = 0u; c < p.cols; c++) { out[base + c] = input[base + c] * inv_rms * gamma[c]; }
}
"#;

const BACKWARD_BINARY_WGSL: &str = r#"
struct Params { len: u32, op: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> forward_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    switch p.op {
        // relu_backward: upstream * (forward_input > 0 ? 1 : 0)
        case 0u: { grad_out[i] = upstream[i] * select(0.0, 1.0, forward_val[i] > 0.0); }
        // sigmoid_backward: upstream * s * (1 - s)  where s = forward_output
        case 1u: {
            let s = forward_val[i];
            grad_out[i] = upstream[i] * s * (1.0 - s);
        }
        // tanh_backward: upstream * (1 - t*t)  where t = forward_output
        case 2u: {
            let t = forward_val[i];
            grad_out[i] = upstream[i] * (1.0 - t * t);
        }
        // exp_backward: upstream * e  where e = forward_output
        case 3u: { grad_out[i] = upstream[i] * forward_val[i]; }
        default: {}
    }
}
"#;

const CONV2D_INPUT_GRAD_WGSL: &str = r#"
struct Params {
    n: u32, ih: u32, iw: u32, ic: u32,
    oc: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iw_idx = gid.x; let ih_idx = gid.y;
    let batch_ic = gid.z;
    let batch = batch_ic / p.ic;
    let ci = batch_ic % p.ic;
    if (ih_idx >= p.ih || iw_idx >= p.iw || batch >= p.n) { return; }
    var sum = 0.0;
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let oy_check = ih_idx - ky;
            let ox_check = iw_idx - kx;
            // Check that the subtraction didn't underflow (unsigned) and alignment.
            if (ih_idx < ky || iw_idx < kx) { continue; }
            if (oy_check % p.sh != 0u || ox_check % p.sw != 0u) { continue; }
            let oy = oy_check / p.sh;
            let ox = ox_check / p.sw;
            if (oy >= p.oh || ox >= p.ow) { continue; }
            for (var co = 0u; co < p.oc; co++) {
                let g = upstream[((batch * p.oh + oy) * p.ow + ox) * p.oc + co];
                let k_val = kern[((ky * p.kw + kx) * p.ic + ci) * p.oc + co];
                sum += g * k_val;
            }
        }
    }
    grad_input[((batch * p.ih + ih_idx) * p.iw + iw_idx) * p.ic + ci] = sum;
}
"#;

const REDUCE_SUM_BACKWARD_WGSL: &str = r#"
struct Params { len: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    grad_out[i] = upstream[0];
}
"#;

// ── Pipeline cache ─────────────────────────────────────────────────

struct Pipelines {
    matmul: wgpu::ComputePipeline,
    elementwise: wgpu::ComputePipeline,
    unary: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    log_softmax: wgpu::ComputePipeline,
    logsumexp: wgpu::ComputePipeline,
    conv2d: wgpu::ComputePipeline,
    pool2d: wgpu::ComputePipeline,
    batch_norm: wgpu::ComputePipeline,
    layer_norm: wgpu::ComputePipeline,
    depthwise_conv2d: wgpu::ComputePipeline,
    transpose_conv2d: wgpu::ComputePipeline,
    transpose_2d: wgpu::ComputePipeline,
    backward_binary: wgpu::ComputePipeline,
    reduce_sum_backward: wgpu::ComputePipeline,
    conv2d_input_grad: wgpu::ComputePipeline,
}

// ── GpuBackend ─────────────────────────────────────────────────────

/// Simple size-bucketed buffer pool for GPU buffer reuse across dispatches.
/// Uses `RefCell` for interior mutability so `&self` methods can pool/reclaim.
struct BufferPool {
    /// Available output buffers keyed by capacity in bytes.
    output: std::cell::RefCell<Vec<(u64, wgpu::Buffer)>>,
    /// Available storage buffers keyed by capacity in bytes.
    storage: std::cell::RefCell<Vec<(u64, wgpu::Buffer)>>,
    /// Maximum pool depth per category.
    max_depth: usize,
    /// Total allocations saved (diagnostic counter).
    hits: std::cell::Cell<u64>,
}

impl BufferPool {
    fn new(max_depth: usize) -> Self {
        Self {
            output: std::cell::RefCell::new(Vec::with_capacity(max_depth)),
            storage: std::cell::RefCell::new(Vec::with_capacity(max_depth)),
            max_depth,
            hits: std::cell::Cell::new(0),
        }
    }

    /// Try to reclaim an output buffer with at least `size_bytes` capacity.
    fn take_output(&self, size_bytes: u64) -> Option<wgpu::Buffer> {
        let mut pool = self.output.borrow_mut();
        if let Some(pos) = pool.iter().position(|(cap, _)| *cap >= size_bytes) {
            self.hits.set(self.hits.get() + 1);
            Some(pool.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Return an output buffer to the pool for future reuse.
    fn return_output(&self, size_bytes: u64, buf: wgpu::Buffer) {
        let mut pool = self.output.borrow_mut();
        if pool.len() < self.max_depth {
            pool.push((size_bytes, buf));
        }
        // else: drop the buffer (exceeds pool capacity)
    }

    /// Try to reclaim a storage buffer with at least `size_bytes` capacity.
    fn take_storage(&self, size_bytes: u64) -> Option<wgpu::Buffer> {
        let mut pool = self.storage.borrow_mut();
        if let Some(pos) = pool.iter().position(|(cap, _)| *cap >= size_bytes) {
            self.hits.set(self.hits.get() + 1);
            Some(pool.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Return a storage buffer to the pool.
    fn return_storage(&self, size_bytes: u64, buf: wgpu::Buffer) {
        let mut pool = self.storage.borrow_mut();
        if pool.len() < self.max_depth {
            pool.push((size_bytes, buf));
        }
    }

    /// Total cache hits (diagnostic).
    fn cache_hits(&self) -> u64 {
        self.hits.get()
    }
}

/// Cross-platform GPU compute backend via wgpu (Vulkan/Metal/DX12).
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    adapter_name: String,
    pool: BufferPool,
}

impl GpuBackend {
    /// Auto-selects the best available GPU adapter.
    pub fn new() -> Result<Self, KernelError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok_or_else(|| KernelError::Gpu {
            message: "no GPU adapter found".into(),
        })?;

        let adapter_name = adapter.get_info().name;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("yscv-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| KernelError::Gpu {
            message: format!("device request failed: {e}"),
        })?;

        let mk = |src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let pipe = |module: &wgpu::ShaderModule| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipelines = Pipelines {
            matmul: pipe(&mk(MATMUL_WGSL)),
            elementwise: pipe(&mk(ELEMENTWISE_WGSL)),
            unary: pipe(&mk(UNARY_WGSL)),
            softmax: pipe(&mk(SOFTMAX_WGSL)),
            log_softmax: pipe(&mk(LOG_SOFTMAX_WGSL)),
            logsumexp: pipe(&mk(LOGSUMEXP_WGSL)),
            conv2d: pipe(&mk(CONV2D_WGSL)),
            pool2d: pipe(&mk(POOL2D_WGSL)),
            batch_norm: pipe(&mk(BATCH_NORM_WGSL)),
            layer_norm: pipe(&mk(LAYER_NORM_WGSL)),
            depthwise_conv2d: pipe(&mk(DEPTHWISE_CONV2D_WGSL)),
            transpose_conv2d: pipe(&mk(TRANSPOSE_CONV2D_WGSL)),
            transpose_2d: pipe(&mk(TRANSPOSE_2D_WGSL)),
            backward_binary: pipe(&mk(BACKWARD_BINARY_WGSL)),
            reduce_sum_backward: pipe(&mk(REDUCE_SUM_BACKWARD_WGSL)),
            conv2d_input_grad: pipe(&mk(CONV2D_INPUT_GRAD_WGSL)),
        };

        Ok(Self {
            device,
            queue,
            pipelines,
            adapter_name,
            pool: BufferPool::new(32),
        })
    }

    /// Returns the GPU adapter name (e.g. "NVIDIA GeForce RTX 4090").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    // ── Buffer helpers ─────────────────────────────────────────────

    fn storage_buf(&self, data: &[f32]) -> wgpu::Buffer {
        let size_bytes = (data.len() * 4) as u64;
        if let Some(buf) = self.pool.take_storage(size_bytes) {
            self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            return buf;
        }
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn output_buf(&self, len: usize) -> wgpu::Buffer {
        let size_bytes = (len * 4) as u64;
        if let Some(buf) = self.pool.take_output(size_bytes) {
            return buf;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Returns the number of GPU buffer pool cache hits (diagnostic).
    pub fn pool_cache_hits(&self) -> u64 {
        self.pool.cache_hits()
    }

    /// Returns a buffer to the output pool for future reuse.
    pub fn return_output_buf(&self, len: usize, buf: wgpu::Buffer) {
        self.pool.return_output((len * 4) as u64, buf);
    }

    /// Returns a buffer to the storage pool for future reuse.
    pub fn return_storage_buf(&self, len: usize, buf: wgpu::Buffer) {
        self.pool.return_storage((len * 4) as u64, buf);
    }

    fn uniform_buf(&self, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn read_buf(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("channel closed").expect("GPU map failed");

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }

    // ── Dispatch helpers ───────────────────────────────────────────

    fn dispatch_elementwise(&self, a: &[f32], b: &[f32], len: usize, op: u32) -> Vec<f32> {
        let buf_a = self.storage_buf(a);
        let buf_b = self.storage_buf(b);
        let buf_out = self.output_buf(len);
        let params: [u32; 2] = [len as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.elementwise.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.elementwise);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(len as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.read_buf(&buf_out, len)
    }

    fn dispatch_backward_binary(
        &self,
        upstream: &[f32],
        forward_val: &[f32],
        len: usize,
        op: u32,
    ) -> Vec<f32> {
        let buf_up = self.storage_buf(upstream);
        let buf_fv = self.storage_buf(forward_val);
        let buf_out = self.output_buf(len);
        let params: [u32; 2] = [len as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.backward_binary.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_fv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.backward_binary);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(len as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.read_buf(&buf_out, len)
    }

    fn dispatch_reduce_sum_backward(&self, upstream: &[f32], output_len: usize) -> Vec<f32> {
        let buf_up = self.storage_buf(upstream);
        let buf_out = self.output_buf(output_len);
        let params: [u32; 2] = [output_len as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.reduce_sum_backward.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.reduce_sum_backward);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(output_len as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.read_buf(&buf_out, output_len)
    }

    fn dispatch_unary(&self, a: &[f32], len: usize, op: u32) -> Vec<f32> {
        let buf_a = self.storage_buf(a);
        let buf_out = self.output_buf(len);
        let params: [u32; 2] = [len as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.unary.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.unary);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(len as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.read_buf(&buf_out, len)
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn same_shape_data(lhs: &Tensor, rhs: &Tensor) -> Option<usize> {
    if lhs.shape() == rhs.shape() {
        Some(lhs.data().len())
    } else {
        None
    }
}

// ── Backend trait implementation ───────────────────────────────────

impl Backend for GpuBackend {
    fn matmul_2d(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if lhs.shape().len() != 2 || rhs.shape().len() != 2 {
            return Err(KernelError::InvalidMatMulRank {
                left_rank: lhs.shape().len(),
                right_rank: rhs.shape().len(),
            });
        }
        let (m, k) = (lhs.shape()[0], lhs.shape()[1]);
        let (k2, n) = (rhs.shape()[0], rhs.shape()[1]);
        if k != k2 {
            return Err(KernelError::MatMulShapeMismatch {
                left: lhs.shape().to_vec(),
                right: rhs.shape().to_vec(),
            });
        }

        if m * n < MIN_GPU_ELEMENTS {
            return crate::matmul_2d(lhs, rhs);
        }

        let buf_a = self.storage_buf(lhs.data());
        let buf_b = self.storage_buf(rhs.data());
        let buf_out = self.output_buf(m * n);
        let params: [u32; 4] = [m as u32, n as u32, k as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.matmul.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.matmul);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(n as u32, 16), div_ceil(m as u32, 16), 1);
        }
        self.queue.submit(Some(enc.finish()));

        let data = self.read_buf(&buf_out, m * n);
        Tensor::from_vec(vec![m, n], data).map_err(Into::into)
    }

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if let Some(len) = same_shape_data(lhs, rhs)
            && len >= MIN_GPU_ELEMENTS
        {
            let out = self.dispatch_elementwise(lhs.data(), rhs.data(), len, 0);
            return Tensor::from_vec(lhs.shape().to_vec(), out).map_err(Into::into);
        }
        crate::add(lhs, rhs)
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if let Some(len) = same_shape_data(lhs, rhs)
            && len >= MIN_GPU_ELEMENTS
        {
            let out = self.dispatch_elementwise(lhs.data(), rhs.data(), len, 1);
            return Tensor::from_vec(lhs.shape().to_vec(), out).map_err(Into::into);
        }
        crate::sub(lhs, rhs)
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if let Some(len) = same_shape_data(lhs, rhs)
            && len >= MIN_GPU_ELEMENTS
        {
            let out = self.dispatch_elementwise(lhs.data(), rhs.data(), len, 2);
            return Tensor::from_vec(lhs.shape().to_vec(), out).map_err(Into::into);
        }
        crate::mul(lhs, rhs)
    }

    fn relu(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::relu(input);
        }
        let out = self.dispatch_unary(input.data(), len, 0);
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn sigmoid(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::sigmoid(input);
        }
        let out = self.dispatch_unary(input.data(), len, 1);
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn exp(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::exp(input);
        }
        let out = self.dispatch_unary(input.data(), len, 2);
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn tanh_act(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::tanh_act(input);
        }
        let out = self.dispatch_unary(input.data(), len, 3);
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidSoftmaxRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::softmax_last_dim(input);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(rows * cols);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.softmax.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.softmax);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(rows as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, rows * cols);
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLogSoftmaxRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::log_softmax_last_dim(input);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(rows * cols);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.log_softmax.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.log_softmax);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(rows as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, rows * cols);
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLogSumExpRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::logsumexp_last_dim(input);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(rows);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.logsumexp.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.logsumexp);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(rows as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, rows);
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Tensor::from_vec(out_shape, data).map_err(Into::into)
    }

    fn layer_norm_last_dim(
        &self,
        input: &Tensor,
        params: LayerNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLayerNormRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::layer_norm_last_dim(input, params);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_g = self.storage_buf(params.gamma.data());
        let buf_b = self.storage_buf(params.beta.data());
        let buf_out = self.output_buf(rows * cols);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            rows: u32,
            cols: u32,
            eps: f32,
            _pad: u32,
        }
        let p = P {
            rows: rows as u32,
            cols: cols as u32,
            eps: params.epsilon,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.layer_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.layer_norm);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(rows as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, rows * cols);
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let is = input.shape();
        let ks = kernel.shape();
        if is.len() != 4 || ks.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: is.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (is[0], is[1], is[2], is[3]);
        let (kh, kw, kc, oc) = (ks[0], ks[1], ks[2], ks[3]);
        if ic != kc {
            return Err(KernelError::ConvChannelMismatch {
                input_channels: ic,
                kernel_in_channels: kc,
            });
        }
        let oh = (ih - kh) / stride_h + 1;
        let ow = (iw - kw) / stride_w + 1;
        let total = n * oh * ow * oc;

        if total < MIN_GPU_ELEMENTS {
            return crate::conv2d_nhwc(input, kernel, bias, stride_h, stride_w);
        }

        let bias_data = if let Some(b) = bias {
            b.data().to_vec()
        } else {
            vec![0.0f32; oc]
        };

        let buf_in = self.storage_buf(input.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_bias = self.storage_buf(&bias_data);
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.conv2d);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * oc) as u32,
            );
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, total);
        Tensor::from_vec(vec![n, oh, ow, oc], data).map_err(Into::into)
    }

    fn max_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.dispatch_pool(input, kernel_h, kernel_w, stride_h, stride_w, 0)
    }

    fn avg_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.dispatch_pool(input, kernel_h, kernel_w, stride_h, stride_w, 1)
    }

    fn batch_norm2d_nhwc(
        &self,
        input: &Tensor,
        params: BatchNorm2dParams<'_>,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(KernelError::InvalidBatchNormRank {
                got_rank: shape.len(),
            });
        }
        let c = shape[3];
        let total = input.data().len();

        if total < MIN_GPU_ELEMENTS {
            return crate::batch_norm2d_nhwc(input, params);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_g = self.storage_buf(params.gamma.data());
        let buf_b = self.storage_buf(params.beta.data());
        let buf_m = self.storage_buf(params.mean.data());
        let buf_v = self.storage_buf(params.variance.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            total: u32,
            c: u32,
            eps: f32,
            _pad: u32,
        }
        let p = P {
            total: total as u32,
            c: c as u32,
            eps: params.epsilon,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.batch_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_m.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.batch_norm);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(div_ceil(total as u32, 256), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, total);
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let is = input.shape();
        let ks = kernel.shape();
        if is.len() != 4 || ks.len() != 4 {
            return crate::depthwise_conv2d_nhwc(input, kernel, bias, stride_h, stride_w);
        }
        let (n, ih, iw, c) = (is[0], is[1], is[2], is[3]);
        let (kh, kw, _kc, dm) = (ks[0], ks[1], ks[2], ks[3]);
        let oh = (ih - kh) / stride_h + 1;
        let ow = (iw - kw) / stride_w + 1;
        let oc = c * dm;
        let total = n * oh * ow * oc;
        if total < MIN_GPU_ELEMENTS {
            return crate::depthwise_conv2d_nhwc(input, kernel, bias, stride_h, stride_w);
        }
        let bias_data = if let Some(b) = bias {
            b.data().to_vec()
        } else {
            vec![0.0f32; oc]
        };
        let buf_in = self.storage_buf(input.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_bias = self.storage_buf(&bias_data);
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            dm: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            c: c as u32,
            dm: dm as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
        let bgl = self.pipelines.depthwise_conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.depthwise_conv2d);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * oc) as u32,
            );
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, total);
        Tensor::from_vec(vec![n, oh, ow, oc], data).map_err(Into::into)
    }

    fn separable_conv2d_nhwc(
        &self,
        input: &Tensor,
        params: SeparableConv2dParams<'_>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        crate::separable_conv2d_nhwc(input, params, stride_h, stride_w)
    }

    fn group_norm_nhwc(
        &self,
        input: &Tensor,
        params: GroupNormNhwcParams<'_>,
    ) -> Result<Tensor, KernelError> {
        crate::group_norm_nhwc(input, params)
    }

    fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        crate::rms_norm_last_dim(input, params)
    }
}

impl crate::BackwardOps for GpuBackend {
    fn relu_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            // Fall back to default CPU implementation.
            let u = upstream.data();
            let f = forward_input.data();
            let out: Vec<f32> = u
                .iter()
                .zip(f.iter())
                .map(|(&u, &x)| if x > 0.0 { u } else { 0.0 })
                .collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_input.data(), len, 0);
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn sigmoid_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            let u = upstream.data();
            let s = forward_output.data();
            let out: Vec<f32> = u
                .iter()
                .zip(s.iter())
                .map(|(&u, &s)| u * s * (1.0 - s))
                .collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_output.data(), len, 1);
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn tanh_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            let u = upstream.data();
            let t = forward_output.data();
            let out: Vec<f32> = u
                .iter()
                .zip(t.iter())
                .map(|(&u, &t)| u * (1.0 - t * t))
                .collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_output.data(), len, 2);
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn exp_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            let u = upstream.data();
            let e = forward_output.data();
            let out: Vec<f32> = u.iter().zip(e.iter()).map(|(&u, &e)| u * e).collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_output.data(), len, 3);
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn reduce_sum_backward(
        &self,
        upstream: &Tensor,
        original_shape: &[usize],
    ) -> Result<Tensor, KernelError> {
        let len: usize = original_shape.iter().product();
        if len < MIN_GPU_ELEMENTS {
            let grad_val = upstream.data()[0];
            let out = vec![grad_val; len];
            return Tensor::from_vec(original_shape.to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_reduce_sum_backward(upstream.data(), len);
        Tensor::from_vec(original_shape.to_vec(), out).map_err(Into::into)
    }

    fn matmul_backward(
        &self,
        upstream: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let rt = self.transpose_2d(rhs)?;
        let lt = self.transpose_2d(lhs)?;
        let grad_lhs = self.matmul_2d(upstream, &rt)?;
        let grad_rhs = self.matmul_2d(&lt, upstream)?;
        Ok((grad_lhs, grad_rhs))
    }

    fn add_backward(
        &self,
        upstream: &Tensor,
        _lhs: &Tensor,
        _rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        Ok((upstream.clone(), upstream.clone()))
    }

    fn sub_backward(
        &self,
        upstream: &Tensor,
        _lhs: &Tensor,
        _rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        Ok((upstream.clone(), self.neg(upstream)))
    }

    fn mul_backward(
        &self,
        upstream: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let grad_lhs = self.mul(upstream, rhs)?;
        let grad_rhs = self.mul(upstream, lhs)?;
        Ok((grad_lhs, grad_rhs))
    }

    fn conv2d_input_backward(
        &self,
        upstream: &Tensor,
        kernel: &Tensor,
        input_shape: &[usize],
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let us = upstream.shape();
        let ks = kernel.shape();
        if us.len() != 4 || ks.len() != 4 || input_shape.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: input_shape.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (_n, oh, ow, oc) = (us[0], us[1], us[2], us[3]);
        let (kh, kw) = (ks[0], ks[1]);
        let total = n * ih * iw * ic;

        if total < MIN_GPU_ELEMENTS {
            // CPU fallback via default trait implementation.
            let u_data = upstream.data();
            let k_data = kernel.data();
            let mut grad_input = vec![0.0f32; total];
            for b in 0..n {
                for oy in 0..oh {
                    for ox in 0..ow {
                        for co in 0..oc {
                            let g = u_data[((b * oh + oy) * ow + ox) * oc + co];
                            if g == 0.0 {
                                continue;
                            }
                            for ky_i in 0..kh {
                                for kx_i in 0..kw {
                                    let iy = oy * stride_h + ky_i;
                                    let ix = ox * stride_w + kx_i;
                                    if iy < ih && ix < iw {
                                        for ci in 0..ic {
                                            let k_val =
                                                k_data[((ky_i * kw + kx_i) * ic + ci) * oc + co];
                                            grad_input[((b * ih + iy) * iw + ix) * ic + ci] +=
                                                g * k_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_vec(input_shape.to_vec(), grad_input).map_err(Into::into);
        }

        let buf_up = self.storage_buf(upstream.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.conv2d_input_grad.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.conv2d_input_grad);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                div_ceil(iw as u32, 8),
                div_ceil(ih as u32, 8),
                (n * ic) as u32,
            );
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, total);
        Tensor::from_vec(input_shape.to_vec(), data).map_err(Into::into)
    }
}

impl GpuBackend {
    fn dispatch_pool(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        mode: u32,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(KernelError::InvalidPoolRank {
                got_rank: shape.len(),
            });
        }
        let (n, ih, iw, c) = (shape[0], shape[1], shape[2], shape[3]);
        let oh = (ih - kernel_h) / stride_h + 1;
        let ow = (iw - kernel_w) / stride_w + 1;
        let total = n * oh * ow * c;

        if total < MIN_GPU_ELEMENTS {
            if mode == 0 {
                return crate::max_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w);
            } else {
                return crate::avg_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w);
            }
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            mode: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            c: c as u32,
            kh: kernel_h as u32,
            kw: kernel_w as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            mode,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.pool2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.pool2d);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * c) as u32,
            );
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, total);
        Tensor::from_vec(vec![n, oh, ow, c], data).map_err(Into::into)
    }

    /// Transposed convolution (deconvolution) on GPU.
    ///
    /// Input: `[N,H,W,C_in]`, kernel: `[KH,KW,C_in,C_out]`, bias: `[C_out]`.
    /// Output: `[N, (H-1)*stride_h + KH, (W-1)*stride_w + KW, C_out]`.
    pub fn transpose_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let is = input.shape();
        let ks = kernel.shape();
        if is.len() != 4 || ks.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: is.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (is[0], is[1], is[2], is[3]);
        let (kh, kw, _kc, oc) = (ks[0], ks[1], ks[2], ks[3]);
        let oh = (ih - 1) * stride_h + kh;
        let ow = (iw - 1) * stride_w + kw;
        let total = n * oh * ow * oc;

        let bias_data = if let Some(b) = bias {
            b.data().to_vec()
        } else {
            vec![0.0f32; oc]
        };
        let buf_in = self.storage_buf(input.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_bias = self.storage_buf(&bias_data);
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
        let bgl = self.pipelines.transpose_conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.transpose_conv2d);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * oc) as u32,
            );
        }
        self.queue.submit(Some(enc.finish()));
        let data = self.read_buf(&buf_out, total);
        Tensor::from_vec(vec![n, oh, ow, oc], data).map_err(Into::into)
    }
}

// ── Standalone GPU dispatch functions ─────────────────────────────

/// GPU batch normalization: normalize across spatial dims (NHWC layout).
///
/// Applies: `output[i] = gamma[ch] * (input[i] - mean[ch]) / sqrt(var[ch] + epsilon) + beta[ch]`
/// where `ch = i % C`.
///
/// Falls back to CPU if the GPU backend cannot be created.
pub fn gpu_batch_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, KernelError> {
    let backend = GpuBackend::new()?;
    let params = BatchNorm2dParams {
        gamma,
        beta,
        mean,
        variance: var,
        epsilon,
    };
    backend.batch_norm2d_nhwc(input, params)
}

/// GPU layer normalization: normalize across the last dimension.
///
/// For each row, computes mean and variance over the last dim, then applies:
/// `output[row, j] = gamma[j] * (input[row, j] - mu) / sqrt(var + epsilon) + beta[j]`
///
/// Falls back to CPU if the GPU backend cannot be created.
pub fn gpu_layer_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    epsilon: f32,
) -> Result<Tensor, KernelError> {
    let backend = GpuBackend::new()?;
    let params = LayerNormLastDimParams {
        gamma,
        beta,
        epsilon,
    };
    backend.layer_norm_last_dim(input, params)
}

/// GPU 2D matrix transpose: transposes a rank-2 tensor `[M, N]` to `[N, M]`.
///
/// Falls back to CPU if the GPU backend cannot be created.
pub fn gpu_transpose(input: &Tensor) -> Result<Tensor, KernelError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(KernelError::InvalidMatMulRank {
            left_rank: shape.len(),
            right_rank: 0,
        });
    }
    let backend = GpuBackend::new()?;
    let rows = shape[0];
    let cols = shape[1];

    let buf_in = backend.storage_buf(input.data());
    let buf_out = backend.output_buf(rows * cols);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        rows: u32,
        cols: u32,
    }
    let p = P {
        rows: rows as u32,
        cols: cols as u32,
    };
    let buf_p = backend.uniform_buf(bytemuck::bytes_of(&p));

    let bgl = backend.pipelines.transpose_2d.get_bind_group_layout(0);
    let bg = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

    let mut enc = backend.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.pipelines.transpose_2d);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(div_ceil(cols as u32, 16), div_ceil(rows as u32, 16), 1);
    }
    backend.queue.submit(Some(enc.finish()));
    let data = backend.read_buf(&buf_out, rows * cols);
    Tensor::from_vec(vec![cols, rows], data).map_err(Into::into)
}
