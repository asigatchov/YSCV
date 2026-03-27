// General concat along any axis.
// outer = product of dims before concat axis
// inner = product of dims after concat axis
// One dispatch per input slice.
struct Params {
    outer: u32,
    inner: u32,
    c_in: u32,     // this input's size along concat axis
    c_out: u32,    // total output size along concat axis
    offset: u32,   // this input's offset in concat axis
    _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.outer * p.c_in * p.inner;
    if (idx >= total) { return; }
    let inner_idx = idx % p.inner;
    let tmp = idx / p.inner;
    let a_idx = tmp % p.c_in;
    let outer_idx = tmp / p.c_in;
    let out_idx = (outer_idx * p.c_out + p.offset + a_idx) * p.inner + inner_idx;
    out[out_idx] = inp[idx];
}
