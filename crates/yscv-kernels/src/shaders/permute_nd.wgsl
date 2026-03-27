// General N-dimensional permute (up to 6D).
// Output shape and permuted input strides are packed left-aligned;
// unused trailing dims have shape=1, stride=0.
struct Params {
    total: u32, _p1: u32, _p2: u32, _p3: u32,
    // output shape (dims 0-5, unused = 1)
    s0: u32, s1: u32, s2: u32, s3: u32, s4: u32, s5: u32, _s6: u32, _s7: u32,
    // permuted input strides: t[i] = input_stride[perm[i]]
    t0: u32, t1: u32, t2: u32, t3: u32, t4: u32, t5: u32, _t6: u32, _t7: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    var rem = idx;
    let c5 = rem % p.s5; rem = rem / p.s5;
    let c4 = rem % p.s4; rem = rem / p.s4;
    let c3 = rem % p.s3; rem = rem / p.s3;
    let c2 = rem % p.s2; rem = rem / p.s2;
    let c1 = rem % p.s1; rem = rem / p.s1;
    let c0 = rem;
    let in_idx = c0 * p.t0 + c1 * p.t1 + c2 * p.t2 + c3 * p.t3 + c4 * p.t4 + c5 * p.t5;
    out[idx] = inp[in_idx];
}
