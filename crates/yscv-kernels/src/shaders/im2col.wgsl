// im2col: unfold conv input neighborhoods into a matrix.
// Input:  NHWC [N, IH, IW, IC]
// Output: [N*OH*OW, KH*KW*IC]  (each row = one receptive field)
struct Params {
    n: u32, ih: u32, iw: u32, ic: u32,
    oh: u32, ow: u32, kh: u32, kw: u32,
    sh: u32, sw: u32, pad_h: u32, pad_w: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> col: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.oh * p.ow;
    if (idx >= total) { return; }
    // idx = batch*OH*OW + oh*OW + ow
    let ow_idx = idx % p.ow;
    let oh_idx = (idx / p.ow) % p.oh;
    let batch = idx / (p.oh * p.ow);
    let col_width = p.kh * p.kw * p.ic;
    let col_row = idx * col_width;
    let base_h = i32(oh_idx * p.sh) - i32(p.pad_h);
    let base_w = i32(ow_idx * p.sw) - i32(p.pad_w);
    for (var ky = 0u; ky < p.kh; ky++) {
        let iy = base_h + i32(ky);
        for (var kx = 0u; kx < p.kw; kx++) {
            let ix = base_w + i32(kx);
            let offset = (ky * p.kw + kx) * p.ic;
            if (iy >= 0 && u32(iy) < p.ih && ix >= 0 && u32(ix) < p.iw) {
                let in_base = ((batch * p.ih + u32(iy)) * p.iw + u32(ix)) * p.ic;
                for (var c = 0u; c < p.ic; c++) {
                    col[col_row + offset + c] = inp[in_base + c];
                }
            } else {
                for (var c = 0u; c < p.ic; c++) {
                    col[col_row + offset + c] = 0.0;
                }
            }
        }
    }
}
