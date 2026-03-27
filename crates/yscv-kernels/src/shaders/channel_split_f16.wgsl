// Channel-axis (last-axis) split on f16 storage — one dispatch per output tensor.
enable f16;
struct Params {
    spatial: u32,
    c_in: u32,
    c_out: u32,
    ch_offset: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.spatial * p.c_out;
    if (idx >= total) { return; }
    let pos = idx / p.c_out;
    let ch = idx % p.c_out;
    out[idx] = inp[pos * p.c_in + p.ch_offset + ch];
}
