// General N-dimensional slice with step=1 (up to 6D).
// Extracts a sub-tensor given starts and output shape.
// Unused trailing dims: shape=1, stride=0, start=0.
struct Params {
    total: u32, _p1: u32, _p2: u32, _p3: u32,
    // output shape
    s0: u32, s1: u32, s2: u32, s3: u32, s4: u32, s5: u32, _a: u32, _b: u32,
    // input strides (row-major)
    t0: u32, t1: u32, t2: u32, t3: u32, t4: u32, t5: u32, _c: u32, _d: u32,
    // start offsets per dim
    o0: u32, o1: u32, o2: u32, o3: u32, o4: u32, o5: u32, _e: u32, _f: u32,
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
    let in_idx = (c0 + p.o0) * p.t0 + (c1 + p.o1) * p.t1 + (c2 + p.o2) * p.t2
               + (c3 + p.o3) * p.t3 + (c4 + p.o4) * p.t4 + (c5 + p.o5) * p.t5;
    out[idx] = inp[in_idx];
}
