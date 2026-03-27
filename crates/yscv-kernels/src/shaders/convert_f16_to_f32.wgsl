// Convert f16 buffer to f32 buffer. One thread per element.
enable f16;
struct Params { len: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    out[i] = f32(inp[i]);
}
