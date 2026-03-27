struct Params {
    n: u32, h: u32, w: u32, c: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.h * p.w;
    if (idx >= total) { return; }
    // Output is NCHW: idx = ((n * C + c) * H + h) * W + w
    let w_idx = idx % p.w;
    let tmp = idx / p.w;
    let h_idx = tmp % p.h;
    let tmp2 = tmp / p.h;
    let c_idx = tmp2 % p.c;
    let n_idx = tmp2 / p.c;
    // Input is NHWC: src = ((n * H + h) * W + w) * C + c
    let src = ((n_idx * p.h + h_idx) * p.w + w_idx) * p.c + c_idx;
    out[idx] = inp[src];
}
