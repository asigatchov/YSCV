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
