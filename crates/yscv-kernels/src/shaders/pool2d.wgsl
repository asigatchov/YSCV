struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    oh: u32, ow: u32, mode: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow_idx = gid.x; let oh_idx = gid.y;
    let batch_c = gid.z;
    let batch = batch_c / p.c;
    let ch = batch_c % p.c;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var val: f32;
    if (p.mode == 0u) { val = -1e38; } else { val = 0.0; }
    var cnt = 0.0;
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let iy = oh_idx * p.sh + ky;
            let ix = ow_idx * p.sw + kx;
            let v = inp[((batch * p.ih + iy) * p.iw + ix) * p.c + ch];
            if (p.mode == 0u) { val = max(val, v); }
            else { val += v; cnt += 1.0; }
        }
    }
    if (p.mode == 1u && cnt > 0.0) { val /= cnt; }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * p.c + ch] = val;
}
