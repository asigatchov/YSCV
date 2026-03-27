struct Params {
    spatial: u32,    // product of all dims except last (N * H * W for NHWC)
    c_in: u32,      // total input channels
    c_out: u32,     // channels to extract
    ch_offset: u32, // start channel in input
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.spatial * p.c_out) { return; }
    let pos = idx / p.c_out;
    let ch = idx % p.c_out;
    out[idx] = inp[pos * p.c_in + p.ch_offset + ch];
}
