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
