// Broadcasting binary op for up to 6D tensors.
// Strides are 0 for broadcast dimensions (size=1).
struct Params {
    total: u32, op: u32, _p1: u32, _p2: u32,
    // output shape (unused dims = 1)
    s0: u32, s1: u32, s2: u32, s3: u32, s4: u32, s5: u32, _a: u32, _b: u32,
    // A strides (0 for broadcast dims)
    a0: u32, a1: u32, a2: u32, a3: u32, a4: u32, a5: u32, _c: u32, _d: u32,
    // B strides (0 for broadcast dims)
    b0: u32, b1: u32, b2: u32, b3: u32, b4: u32, b5: u32, _e: u32, _f: u32,
}
@group(0) @binding(0) var<storage, read> inp_a: array<f32>;
@group(0) @binding(1) var<storage, read> inp_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
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
    let ai = c0*p.a0 + c1*p.a1 + c2*p.a2 + c3*p.a3 + c4*p.a4 + c5*p.a5;
    let bi = c0*p.b0 + c1*p.b1 + c2*p.b2 + c3*p.b3 + c4*p.b4 + c5*p.b5;
    let va = inp_a[ai];
    let vb = inp_b[bi];
    switch p.op {
        case 0u: { out[idx] = va + vb; }
        case 1u: { out[idx] = va - vb; }
        case 2u: { out[idx] = va * vb; }
        case 3u: { out[idx] = va / vb; }
        default: {}
    }
}
