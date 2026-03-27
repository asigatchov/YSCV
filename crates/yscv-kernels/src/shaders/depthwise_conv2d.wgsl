struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    dm: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32,
    pad_h: u32, pad_w: u32, act: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
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
        let iy_s = i32(oh_idx * p.sh + ky) - i32(p.pad_h);
        if (iy_s < 0 || u32(iy_s) >= p.ih) { continue; }
        let iy = u32(iy_s);
        for (var kx = 0u; kx < p.kw; kx++) {
            let ix_s = i32(ow_idx * p.sw + kx) - i32(p.pad_w);
            if (ix_s < 0 || u32(ix_s) >= p.iw) { continue; }
            let ix = u32(ix_s);
            sum = fma(
                inp[((batch * p.ih + iy) * p.iw + ix) * p.c + ch],
                kern[((ky * p.kw + kx) * p.c + ch) * p.dm + dm_idx],
                sum
            );
        }
    }
    if (p.act == 1u) { sum = max(sum, 0.0); }
    else if (p.act == 2u) { sum = sum / (1.0 + exp(-sum)); }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * oc + cdm] = sum;
}
