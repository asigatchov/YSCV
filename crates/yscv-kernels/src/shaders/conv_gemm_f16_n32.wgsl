// Small-N variant: BN=32 for conv layers with OC ≤ 48.
// Eliminates tile waste for N=32 (0% vs 50%) and N=16 (50% vs 75%).
// Workgroup 8×16=128 threads. Same bind group layout as conv_gemm_f16.
enable f16;

struct Params {
    m: u32, n_out: u32, k: u32, act: u32,
    ih: u32, iw: u32, ic: u32, oh: u32,
    ow: u32, kh: u32, kw: u32, sh: u32,
    sw: u32, pad_h: u32, pad_w: u32, batch: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

const BM: u32 = 64u;
const BN: u32 = 32u;
const BK: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const WG: u32 = 128u;   // 8 × 16
const SA_STRIDE: u32 = 17u;   // BK + 1
const SB_STRIDE: u32 = 33u;   // BN + 1
var<workgroup> sa: array<f16, 1088>;  // BM * SA_STRIDE = 64*17
var<workgroup> sb: array<f16, 528>;   // BK * SB_STRIDE = 16*33

@compute @workgroup_size(8, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tx = lid.x;  // 0..7
    let ty = lid.y;  // 0..15
    let tid = ty * 8u + tx;

    let row0 = wid.y * BM;
    let col0 = wid.x * BN;

    var acc0 = vec4<f16>(0.0h);
    var acc1 = vec4<f16>(0.0h);
    var acc2 = vec4<f16>(0.0h);
    var acc3 = vec4<f16>(0.0h);

    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t++) {
        let k0 = t * BK;

        // Load A tile (64×16 = 1024 elements, 128 threads, 8 per thread)
        if (p.kh == 1u && p.kw == 1u && p.sh == 1u && p.sw == 1u
            && p.pad_h == 0u && p.pad_w == 0u) {
            for (var i = 0u; i < 8u; i++) {
                let idx = tid + i * WG;
                let r = idx / BK;
                let c = idx % BK;
                let gr = row0 + r;
                let gk = k0 + c;
                if (gr < p.m && gk < p.k) {
                    sa[r * SA_STRIDE + c] = f16(inp[gr * p.ic + gk]);
                } else {
                    sa[r * SA_STRIDE + c] = 0.0h;
                }
            }
        } else {
            for (var i = 0u; i < 8u; i++) {
                let idx = tid + i * WG;
                let r = idx / BK;
                let c = idx % BK;
                let gr = row0 + r;
                let gk = k0 + c;
                if (gr < p.m && gk < p.k) {
                    let ow_idx = gr % p.ow;
                    let oh_idx = (gr / p.ow) % p.oh;
                    let b = gr / (p.oh * p.ow);
                    let ci = gk % p.ic;
                    let kx_idx = (gk / p.ic) % p.kw;
                    let ky_idx = gk / (p.ic * p.kw);
                    let iy = i32(oh_idx * p.sh + ky_idx) - i32(p.pad_h);
                    let ix = i32(ow_idx * p.sw + kx_idx) - i32(p.pad_w);
                    if (iy >= 0 && u32(iy) < p.ih && ix >= 0 && u32(ix) < p.iw) {
                        sa[r * SA_STRIDE + c] = f16(inp[((b * p.ih + u32(iy)) * p.iw + u32(ix)) * p.ic + ci]);
                    } else {
                        sa[r * SA_STRIDE + c] = 0.0h;
                    }
                } else {
                    sa[r * SA_STRIDE + c] = 0.0h;
                }
            }
        }

        // Load B tile (16×32 = 512 elements, 128 threads, 4 per thread)
        for (var i = 0u; i < 4u; i++) {
            let idx = tid + i * WG;
            let r = idx / BN;
            let c = idx % BN;
            let gr = k0 + r;
            let gc_val = col0 + c;
            if (gr < p.k && gc_val < p.n_out) {
                sb[r * SB_STRIDE + c] = f16(weight[gr * p.n_out + gc_val]);
            } else {
                sb[r * SB_STRIDE + c] = 0.0h;
            }
        }

        workgroupBarrier();

        // GEMM inner loop — f16 FMA, 2x unrolled.
        let b_base = tx * TN;
        let a_base0 = (ty * TM + 0u) * SA_STRIDE;
        let a_base1 = (ty * TM + 1u) * SA_STRIDE;
        let a_base2 = (ty * TM + 2u) * SA_STRIDE;
        let a_base3 = (ty * TM + 3u) * SA_STRIDE;
        for (var kk = 0u; kk < BK; kk += 2u) {
            var bv = vec4<f16>(
                sb[kk * SB_STRIDE + b_base],
                sb[kk * SB_STRIDE + b_base + 1u],
                sb[kk * SB_STRIDE + b_base + 2u],
                sb[kk * SB_STRIDE + b_base + 3u],
            );
            acc0 = fma(vec4<f16>(sa[a_base0 + kk]), bv, acc0);
            acc1 = fma(vec4<f16>(sa[a_base1 + kk]), bv, acc1);
            acc2 = fma(vec4<f16>(sa[a_base2 + kk]), bv, acc2);
            acc3 = fma(vec4<f16>(sa[a_base3 + kk]), bv, acc3);
            let kk1 = kk + 1u;
            bv = vec4<f16>(
                sb[kk1 * SB_STRIDE + b_base],
                sb[kk1 * SB_STRIDE + b_base + 1u],
                sb[kk1 * SB_STRIDE + b_base + 2u],
                sb[kk1 * SB_STRIDE + b_base + 3u],
            );
            acc0 = fma(vec4<f16>(sa[a_base0 + kk1]), bv, acc0);
            acc1 = fma(vec4<f16>(sa[a_base1 + kk1]), bv, acc1);
            acc2 = fma(vec4<f16>(sa[a_base2 + kk1]), bv, acc2);
            acc3 = fma(vec4<f16>(sa[a_base3 + kk1]), bv, acc3);
        }

        workgroupBarrier();
    }

    // Write results: f16 → f32 + bias + activation.
    let c = col0 + tx * TN;
    if (c + 3u < p.n_out) {
        let bv = vec4<f32>(bias[c], bias[c + 1u], bias[c + 2u], bias[c + 3u]);
        var r0 = vec4<f32>(acc0) + bv;
        var r1 = vec4<f32>(acc1) + bv;
        var r2 = vec4<f32>(acc2) + bv;
        var r3 = vec4<f32>(acc3) + bv;
        var results = array<vec4<f32>, 4>(r0, r1, r2, r3);
        for (var ri = 0u; ri < TM; ri++) {
            let r = row0 + ty * TM + ri;
            if (r >= p.m) { continue; }
            var v = results[ri];
            if (p.act == 1u) {
                v = max(v, vec4<f32>(0.0));
            } else if (p.act == 2u) {
                v = v / (vec4<f32>(1.0) + exp(-v));
            }
            let base = r * p.n_out + c;
            out[base]      = v.x;
            out[base + 1u] = v.y;
            out[base + 2u] = v.z;
            out[base + 3u] = v.w;
        }
    } else {
        for (var ri = 0u; ri < TM; ri++) {
            let r = row0 + ty * TM + ri;
            if (r >= p.m) { continue; }
            var acc_r: vec4<f16>;
            switch ri {
                case 0u: { acc_r = acc0; }
                case 1u: { acc_r = acc1; }
                case 2u: { acc_r = acc2; }
                default: { acc_r = acc3; }
            }
            let acc_f32 = vec4<f32>(acc_r);
            for (var ci = 0u; ci < TN; ci++) {
                let col = c + ci;
                if (col < p.n_out) {
                    var val = acc_f32[ci] + bias[col];
                    if (p.act == 1u) { val = max(val, 0.0); }
                    else if (p.act == 2u) { val = val / (1.0 + exp(-val)); }
                    out[r * p.n_out + col] = val;
                }
            }
        }
    }
}
