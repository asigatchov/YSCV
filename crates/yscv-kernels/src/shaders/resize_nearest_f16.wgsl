// Nearest-neighbor resize on f16 NHWC tensor.
enable f16;
struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    oh: u32, ow: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.n * p.oh * p.ow * p.c) { return; }
    let ch = idx % p.c;
    let tmp = idx / p.c;
    let ox = tmp % p.ow;
    let tmp2 = tmp / p.ow;
    let oy = tmp2 % p.oh;
    let batch = tmp2 / p.oh;
    let sy = f32(p.ih) / f32(p.oh);
    let sx = f32(p.iw) / f32(p.ow);
    let iy = min(u32(floor((f32(oy) + 0.5) * sy)), p.ih - 1u);
    let ix = min(u32(floor((f32(ox) + 0.5) * sx)), p.iw - 1u);
    out[idx] = inp[((batch * p.ih + iy) * p.iw + ix) * p.c + ch];
}
