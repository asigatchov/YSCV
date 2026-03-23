struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    dm: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow_idx = gid.x; let oh_idx = gid.y;
    let batch_cdm = gid.z;
    let oc = p.c * p.dm;
    let batch = batch_cdm / oc;
    let cdm = batch_cdm % oc;
    let ch = cdm / p.dm;
    let dm_idx = cdm % p.dm;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var sum = bias[cdm];
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let iy = oh_idx * p.sh + ky;
            let ix = ow_idx * p.sw + kx;
            let in_val = inp[((batch * p.ih + iy) * p.iw + ix) * p.c + ch];
            let k_val = kern[((ky * p.kw + kx) * p.c + ch) * p.dm + dm_idx];
            sum += in_val * k_val;
        }
    }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * oc + cdm] = sum;
}
