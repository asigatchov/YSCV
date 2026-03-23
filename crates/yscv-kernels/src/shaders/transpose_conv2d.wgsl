struct Params {
    n: u32, ih: u32, iw: u32, ic: u32,
    oc: u32, kh: u32, kw: u32,
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
    let batch_co = gid.z;
    let batch = batch_co / p.oc;
    let co = batch_co % p.oc;
    if (oh_idx >= p.oh || ow_idx >= p.ow || batch >= p.n) { return; }
    var sum = bias[co];
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let in_y_f = i32(oh_idx) - i32(ky);
            let in_x_f = i32(ow_idx) - i32(kx);
            if (in_y_f < 0 || in_x_f < 0) { continue; }
            let in_y = u32(in_y_f);
            let in_x = u32(in_x_f);
            if (in_y % p.sh != 0u || in_x % p.sw != 0u) { continue; }
            let iy = in_y / p.sh;
            let ix = in_x / p.sw;
            if (iy >= p.ih || ix >= p.iw) { continue; }
            for (var ci = 0u; ci < p.ic; ci++) {
                let in_val = inp[((batch * p.ih + iy) * p.iw + ix) * p.ic + ci];
                let k_val = kern[((ky * p.kw + kx) * p.ic + ci) * p.oc + co];
                sum += in_val * k_val;
            }
        }
    }
    out[((batch * p.oh + oh_idx) * p.ow + ow_idx) * p.oc + co] = sum;
}
