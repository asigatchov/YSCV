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
