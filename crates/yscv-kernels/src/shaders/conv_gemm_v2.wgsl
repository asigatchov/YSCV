// V2 conv_gemm: BM=128, BN=64, BK=32, TM=8, TN=4.
// Larger M-tile + wider K-tile = 2x better compute intensity.
// f16 shared memory + accumulators for 2x FMA throughput on Apple Silicon.
// Input/Weight/Bias/Output remain f32 in global memory.
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

const BM: u32 = 128u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const TM: u32 = 8u;
const TN: u32 = 4u;
// Thread layout: 16 × 16 = 256 threads.
// M: 128 / 8 = 16 threads. N: 64 / 4 = 16 threads.
const SA_STRIDE: u32 = 33u;  // BK + 1 (bank conflict avoidance)
const SB_STRIDE: u32 = 65u;  // BN + 1
// Shared memory: 128 * 33 = 4224 f16 + 32 * 65 = 2080 f16 = 6304 f16 = 12608 bytes
var<workgroup> sa: array<f16, 4224>;  // BM * SA_STRIDE
var<workgroup> sb: array<f16, 2080>;  // BK * SB_STRIDE

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tx = lid.x;  // 0..15 → column thread
    let ty = lid.y;  // 0..15 → row thread
    let tid = ty * 16u + tx;

    let row0 = wid.y * BM;
    let col0 = wid.x * BN;

    // 8 row accumulators × 4 columns = 32 outputs per thread.
    var acc0 = vec4<f16>(0.0h);
    var acc1 = vec4<f16>(0.0h);
    var acc2 = vec4<f16>(0.0h);
    var acc3 = vec4<f16>(0.0h);
    var acc4 = vec4<f16>(0.0h);
    var acc5 = vec4<f16>(0.0h);
    var acc6 = vec4<f16>(0.0h);
    var acc7 = vec4<f16>(0.0h);

    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t++) {
        let k0 = t * BK;

        // Load A tile: BM × BK = 128 × 32 = 4096 elements.
        // 256 threads → 16 elements per thread.
        if (p.kh == 1u && p.kw == 1u && p.sh == 1u && p.sw == 1u
            && p.pad_h == 0u && p.pad_w == 0u) {
            // 1×1 conv fast path — simple address
            for (var i = 0u; i < 16u; i++) {
                let idx = tid + i * 256u;
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
            // General conv — im2col address computation
            for (var i = 0u; i < 16u; i++) {
                let idx = tid + i * 256u;
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

        // Load B tile: BK × BN = 32 × 64 = 2048 elements.
        // 256 threads → 8 elements per thread.
        for (var i = 0u; i < 8u; i++) {
            let idx = tid + i * 256u;
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

        // GEMM inner loop — 8 row accumulators × 4 col = 32 FMAs per kk step.
        // 4× unrolled for reduced loop overhead.
        let b_base = tx * TN;
        let a_base0 = (ty * TM + 0u) * SA_STRIDE;
        let a_base1 = (ty * TM + 1u) * SA_STRIDE;
        let a_base2 = (ty * TM + 2u) * SA_STRIDE;
        let a_base3 = (ty * TM + 3u) * SA_STRIDE;
        let a_base4 = (ty * TM + 4u) * SA_STRIDE;
        let a_base5 = (ty * TM + 5u) * SA_STRIDE;
        let a_base6 = (ty * TM + 6u) * SA_STRIDE;
        let a_base7 = (ty * TM + 7u) * SA_STRIDE;

        for (var kk = 0u; kk < BK; kk += 4u) {
            // kk+0
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
            acc4 = fma(vec4<f16>(sa[a_base4 + kk]), bv, acc4);
            acc5 = fma(vec4<f16>(sa[a_base5 + kk]), bv, acc5);
            acc6 = fma(vec4<f16>(sa[a_base6 + kk]), bv, acc6);
            acc7 = fma(vec4<f16>(sa[a_base7 + kk]), bv, acc7);

            // kk+1
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
            acc4 = fma(vec4<f16>(sa[a_base4 + kk1]), bv, acc4);
            acc5 = fma(vec4<f16>(sa[a_base5 + kk1]), bv, acc5);
            acc6 = fma(vec4<f16>(sa[a_base6 + kk1]), bv, acc6);
            acc7 = fma(vec4<f16>(sa[a_base7 + kk1]), bv, acc7);

            // kk+2
            let kk2 = kk + 2u;
            bv = vec4<f16>(
                sb[kk2 * SB_STRIDE + b_base],
                sb[kk2 * SB_STRIDE + b_base + 1u],
                sb[kk2 * SB_STRIDE + b_base + 2u],
                sb[kk2 * SB_STRIDE + b_base + 3u],
            );
            acc0 = fma(vec4<f16>(sa[a_base0 + kk2]), bv, acc0);
            acc1 = fma(vec4<f16>(sa[a_base1 + kk2]), bv, acc1);
            acc2 = fma(vec4<f16>(sa[a_base2 + kk2]), bv, acc2);
            acc3 = fma(vec4<f16>(sa[a_base3 + kk2]), bv, acc3);
            acc4 = fma(vec4<f16>(sa[a_base4 + kk2]), bv, acc4);
            acc5 = fma(vec4<f16>(sa[a_base5 + kk2]), bv, acc5);
            acc6 = fma(vec4<f16>(sa[a_base6 + kk2]), bv, acc6);
            acc7 = fma(vec4<f16>(sa[a_base7 + kk2]), bv, acc7);

            // kk+3
            let kk3 = kk + 3u;
            bv = vec4<f16>(
                sb[kk3 * SB_STRIDE + b_base],
                sb[kk3 * SB_STRIDE + b_base + 1u],
                sb[kk3 * SB_STRIDE + b_base + 2u],
                sb[kk3 * SB_STRIDE + b_base + 3u],
            );
            acc0 = fma(vec4<f16>(sa[a_base0 + kk3]), bv, acc0);
            acc1 = fma(vec4<f16>(sa[a_base1 + kk3]), bv, acc1);
            acc2 = fma(vec4<f16>(sa[a_base2 + kk3]), bv, acc2);
            acc3 = fma(vec4<f16>(sa[a_base3 + kk3]), bv, acc3);
            acc4 = fma(vec4<f16>(sa[a_base4 + kk3]), bv, acc4);
            acc5 = fma(vec4<f16>(sa[a_base5 + kk3]), bv, acc5);
            acc6 = fma(vec4<f16>(sa[a_base6 + kk3]), bv, acc6);
            acc7 = fma(vec4<f16>(sa[a_base7 + kk3]), bv, acc7);
        }

        workgroupBarrier();
    }

    // Write results: f16 → f32 + bias + activation.
    let c = col0 + tx * TN;
    let accs = array<vec4<f16>, 8>(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

    if (c + 3u < p.n_out) {
        let bv = vec4<f32>(bias[c], bias[c + 1u], bias[c + 2u], bias[c + 3u]);
        for (var ri = 0u; ri < TM; ri++) {
            let r = row0 + ty * TM + ri;
            if (r >= p.m) { continue; }
            var v = vec4<f32>(accs[ri]) + bv;
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
        // Boundary: scalar fallback
        for (var ri = 0u; ri < TM; ri++) {
            let r = row0 + ty * TM + ri;
            if (r >= p.m) { continue; }
            let acc_f32 = vec4<f32>(accs[ri]);
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
