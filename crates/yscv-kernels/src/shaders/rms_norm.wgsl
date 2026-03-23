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
