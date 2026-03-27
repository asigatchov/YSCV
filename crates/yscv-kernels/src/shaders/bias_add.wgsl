// Add bias per-channel: out[i] += bias[i % oc]
struct Params { total: u32, oc: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.total) { return; }
    data[i] += bias[i % p.oc];
}
