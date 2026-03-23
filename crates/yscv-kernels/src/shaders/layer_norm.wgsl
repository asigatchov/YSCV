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
