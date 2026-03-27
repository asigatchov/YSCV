// Small-N variant: BN=32, BM=64, BK=16, workgroup 8×16=128.
// Bitwise ops + fully unrolled inner loop (same optimizations as conv_gemm_fast).
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
const BK_SHIFT: u32 = 4u;
const BK_MASK: u32 = 15u;
const BN_SHIFT: u32 = 5u;  // 32 = 2^5
const BN_MASK: u32 = 31u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const WG: u32 = 128u;
const SA_STRIDE: u32 = 17u;
const SB_STRIDE: u32 = 33u;
var<workgroup> sa: array<f16, 1088>;  // 64 * 17
var<workgroup> sb: array<f16, 528>;   // 16 * 33

fn load_a(gr: u32, gk: u32) -> f16 {
    let ow_idx = gr % p.ow;
    let oh_idx = (gr / p.ow) % p.oh;
    let b = gr / (p.oh * p.ow);
    let ci = gk % p.ic;
    let kx_idx = (gk / p.ic) % p.kw;
    let ky_idx = gk / (p.ic * p.kw);
    let iy = i32(oh_idx * p.sh + ky_idx) - i32(p.pad_h);
    let ix = i32(ow_idx * p.sw + kx_idx) - i32(p.pad_w);
    if (iy >= 0 && u32(iy) < p.ih && ix >= 0 && u32(ix) < p.iw) {
        return f16(inp[((b * p.ih + u32(iy)) * p.iw + u32(ix)) * p.ic + ci]);
    }
    return 0.0h;
}

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

    let is_1x1 = p.kh == 1u && p.kw == 1u && p.sh == 1u && p.sw == 1u
                 && p.pad_h == 0u && p.pad_w == 0u;

    let tiles = (p.k + BK - 1u) >> BK_SHIFT;
    for (var t = 0u; t < tiles; t++) {
        let k0 = t << BK_SHIFT;

        // Load A tile: 64×16 = 1024 elements / 128 threads = 8 per thread.
        // Use bitwise ops for division/modulo.
        if (is_1x1) {
            for (var i = 0u; i < 8u; i++) {
                let idx = tid + i * WG;
                let r = idx >> BK_SHIFT;
                let c = idx & BK_MASK;
                let gr = row0 + r;
                let gk = k0 + c;
                sa[r * SA_STRIDE + c] = select(0.0h,
                    f16(inp[gr * p.ic + gk]),
                    gr < p.m && gk < p.k);
            }
        } else {
            for (var i = 0u; i < 8u; i++) {
                let idx = tid + i * WG;
                let r = idx >> BK_SHIFT;
                let c = idx & BK_MASK;
                let gr = row0 + r;
                let gk = k0 + c;
                sa[r * SA_STRIDE + c] = select(0.0h,
                    load_a(gr, gk),
                    gr < p.m && gk < p.k);
            }
        }

        // Load B tile: 16×32 = 512 / 128 = 4 per thread.
        for (var i = 0u; i < 4u; i++) {
            let idx = tid + i * WG;
            let r = idx >> BN_SHIFT;
            let c = idx & BN_MASK;
            let gr = k0 + r;
            let gc = col0 + c;
            sb[r * SB_STRIDE + c] = select(0.0h,
                f16(weight[gr * p.n_out + gc]),
                gr < p.k && gc < p.n_out);
        }

        workgroupBarrier();

        // Fully unrolled GEMM inner loop: BK=16 steps.
        let b_base = tx * TN;
        let a0b = (ty * TM) * SA_STRIDE;
        let a1b = (ty * TM + 1u) * SA_STRIDE;
        let a2b = (ty * TM + 2u) * SA_STRIDE;
        let a3b = (ty * TM + 3u) * SA_STRIDE;

        // kk = 0
        var bv = vec4<f16>(sb[0u * SB_STRIDE + b_base], sb[0u * SB_STRIDE + b_base + 1u],
                           sb[0u * SB_STRIDE + b_base + 2u], sb[0u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 0u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 0u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 0u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 0u]), bv, acc3);
        // kk = 1
        bv = vec4<f16>(sb[1u * SB_STRIDE + b_base], sb[1u * SB_STRIDE + b_base + 1u],
                       sb[1u * SB_STRIDE + b_base + 2u], sb[1u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 1u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 1u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 1u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 1u]), bv, acc3);
        // kk = 2
        bv = vec4<f16>(sb[2u * SB_STRIDE + b_base], sb[2u * SB_STRIDE + b_base + 1u],
                       sb[2u * SB_STRIDE + b_base + 2u], sb[2u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 2u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 2u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 2u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 2u]), bv, acc3);
        // kk = 3
        bv = vec4<f16>(sb[3u * SB_STRIDE + b_base], sb[3u * SB_STRIDE + b_base + 1u],
                       sb[3u * SB_STRIDE + b_base + 2u], sb[3u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 3u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 3u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 3u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 3u]), bv, acc3);
        // kk = 4
        bv = vec4<f16>(sb[4u * SB_STRIDE + b_base], sb[4u * SB_STRIDE + b_base + 1u],
                       sb[4u * SB_STRIDE + b_base + 2u], sb[4u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 4u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 4u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 4u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 4u]), bv, acc3);
        // kk = 5
        bv = vec4<f16>(sb[5u * SB_STRIDE + b_base], sb[5u * SB_STRIDE + b_base + 1u],
                       sb[5u * SB_STRIDE + b_base + 2u], sb[5u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 5u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 5u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 5u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 5u]), bv, acc3);
        // kk = 6
        bv = vec4<f16>(sb[6u * SB_STRIDE + b_base], sb[6u * SB_STRIDE + b_base + 1u],
                       sb[6u * SB_STRIDE + b_base + 2u], sb[6u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 6u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 6u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 6u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 6u]), bv, acc3);
        // kk = 7
        bv = vec4<f16>(sb[7u * SB_STRIDE + b_base], sb[7u * SB_STRIDE + b_base + 1u],
                       sb[7u * SB_STRIDE + b_base + 2u], sb[7u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 7u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 7u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 7u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 7u]), bv, acc3);
        // kk = 8
        bv = vec4<f16>(sb[8u * SB_STRIDE + b_base], sb[8u * SB_STRIDE + b_base + 1u],
                       sb[8u * SB_STRIDE + b_base + 2u], sb[8u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 8u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 8u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 8u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 8u]), bv, acc3);
        // kk = 9
        bv = vec4<f16>(sb[9u * SB_STRIDE + b_base], sb[9u * SB_STRIDE + b_base + 1u],
                       sb[9u * SB_STRIDE + b_base + 2u], sb[9u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 9u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 9u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 9u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 9u]), bv, acc3);
        // kk = 10
        bv = vec4<f16>(sb[10u * SB_STRIDE + b_base], sb[10u * SB_STRIDE + b_base + 1u],
                       sb[10u * SB_STRIDE + b_base + 2u], sb[10u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 10u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 10u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 10u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 10u]), bv, acc3);
        // kk = 11
        bv = vec4<f16>(sb[11u * SB_STRIDE + b_base], sb[11u * SB_STRIDE + b_base + 1u],
                       sb[11u * SB_STRIDE + b_base + 2u], sb[11u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 11u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 11u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 11u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 11u]), bv, acc3);
        // kk = 12
        bv = vec4<f16>(sb[12u * SB_STRIDE + b_base], sb[12u * SB_STRIDE + b_base + 1u],
                       sb[12u * SB_STRIDE + b_base + 2u], sb[12u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 12u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 12u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 12u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 12u]), bv, acc3);
        // kk = 13
        bv = vec4<f16>(sb[13u * SB_STRIDE + b_base], sb[13u * SB_STRIDE + b_base + 1u],
                       sb[13u * SB_STRIDE + b_base + 2u], sb[13u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 13u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 13u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 13u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 13u]), bv, acc3);
        // kk = 14
        bv = vec4<f16>(sb[14u * SB_STRIDE + b_base], sb[14u * SB_STRIDE + b_base + 1u],
                       sb[14u * SB_STRIDE + b_base + 2u], sb[14u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 14u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 14u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 14u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 14u]), bv, acc3);
        // kk = 15
        bv = vec4<f16>(sb[15u * SB_STRIDE + b_base], sb[15u * SB_STRIDE + b_base + 1u],
                       sb[15u * SB_STRIDE + b_base + 2u], sb[15u * SB_STRIDE + b_base + 3u]);
        acc0 = fma(vec4<f16>(sa[a0b + 15u]), bv, acc0);
        acc1 = fma(vec4<f16>(sa[a1b + 15u]), bv, acc1);
        acc2 = fma(vec4<f16>(sa[a2b + 15u]), bv, acc2);
        acc3 = fma(vec4<f16>(sa[a3b + 15u]), bv, acc3);

        workgroupBarrier();
    }

    // Write results: f16 → f32 + bias + activation.
    let c = col0 + tx * TN;
    if (c + 3u < p.n_out) {
        let bv2 = vec4<f32>(bias[c], bias[c + 1u], bias[c + 2u], bias[c + 3u]);
        var r0 = vec4<f32>(acc0) + bv2;
        var r1 = vec4<f32>(acc1) + bv2;
        var r2 = vec4<f32>(acc2) + bv2;
        var r3 = vec4<f32>(acc3) + bv2;
        for (var ri = 0u; ri < TM; ri++) {
            let r = row0 + ty * TM + ri;
            if (r >= p.m) { continue; }
            var v: vec4<f32>;
            switch ri {
                case 0u: { v = r0; }
                case 1u: { v = r1; }
                case 2u: { v = r2; }
                default: { v = r3; }
            }
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
