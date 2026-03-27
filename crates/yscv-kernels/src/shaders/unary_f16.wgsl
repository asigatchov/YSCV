// Vec4 unary on f16 storage: 4 f16 elements per thread.
// p.len is in vec4 units (original_len / 4).
enable f16;
struct Params { len: u32, op: u32 }
@group(0) @binding(0) var<storage, read> a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let v = a[i];
    let z = vec4<f16>(0.0h);
    let one = vec4<f16>(1.0h);
    switch p.op {
        case 0u: { out[i] = max(v, z); }
        case 1u: { out[i] = one / (one + exp(-v)); }
        case 2u: { out[i] = exp(v); }
        case 3u: { out[i] = tanh(v); }
        case 4u: { out[i] = v / (one + exp(-v)); }
        default: {}
    }
}
