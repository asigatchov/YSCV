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
