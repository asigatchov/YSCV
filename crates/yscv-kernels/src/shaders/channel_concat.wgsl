// Channel-axis (last-axis) concat — one dispatch per input tensor.
// Each thread copies one element from input to output at the correct channel offset.
struct Params {
    spatial: u32,    // product of all dims except last (N * H * W for NHWC)
    c_in: u32,      // channels in this input
    c_out: u32,     // total output channels
    ch_offset: u32, // start channel offset in output
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.spatial * p.c_in;
    if (idx >= total) { return; }

    // Each thread copies one (pos, channel) pair.
    // Compute the spatial position and channel from linear index.
    let pos = idx / p.c_in;
    let ch = idx % p.c_in;
    out[pos * p.c_out + p.ch_offset + ch] = inp[idx];
}
