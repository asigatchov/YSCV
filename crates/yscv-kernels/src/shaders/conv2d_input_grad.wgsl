struct Params {
    n: u32, ih: u32, iw: u32, ic: u32,
    oc: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, oh: u32, ow: u32, _pad: u32,
}
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iw_idx = gid.x; let ih_idx = gid.y;
    let batch_ic = gid.z;
    let batch = batch_ic / p.ic;
    let ci = batch_ic % p.ic;
    if (ih_idx >= p.ih || iw_idx >= p.iw || batch >= p.n) { return; }
    var sum = 0.0;
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let oy_check = ih_idx - ky;
            let ox_check = iw_idx - kx;
            // Check that the subtraction didn't underflow (unsigned) and alignment.
            if (ih_idx < ky || iw_idx < kx) { continue; }
            if (oy_check % p.sh != 0u || ox_check % p.sw != 0u) { continue; }
            let oy = oy_check / p.sh;
            let ox = ox_check / p.sw;
            if (oy >= p.oh || ox >= p.ow) { continue; }
            for (var co = 0u; co < p.oc; co++) {
                let g = upstream[((batch * p.oh + oy) * p.ow + ox) * p.oc + co];
                let k_val = kern[((ky * p.kw + kx) * p.ic + ci) * p.oc + co];
                sum += g * k_val;
            }
        }
    }
    grad_input[((batch * p.ih + ih_idx) * p.iw + iw_idx) * p.ic + ci] = sum;
}
