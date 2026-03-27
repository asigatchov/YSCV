// Vec4 elementwise: 4 elements per thread for bandwidth efficiency.
// p.len is in vec4 units (original_len / 4).
struct Params { len: u32, op: u32 }
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    switch p.op {
        case 0u: { out[i] = a[i] + b[i]; }
        case 1u: { out[i] = a[i] - b[i]; }
        case 2u: { out[i] = a[i] * b[i]; }
        case 3u: { out[i] = a[i] / b[i]; }
        default: {}
    }
}
