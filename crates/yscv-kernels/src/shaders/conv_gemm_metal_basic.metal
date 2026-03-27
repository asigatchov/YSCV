// Metal conv kernels for ONNX inference.
// conv_gemm_basic: Tiled GEMM, BM=64, BN=64, BK=16, 16×16=256 threads.
// conv_direct: Per-pixel kernel for small K×N, zero shared memory, max occupancy.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint m; uint n_out; uint k; uint act;
    uint ih; uint iw; uint ic; uint oh;
    uint ow; uint kh; uint kw; uint sh;
    uint sw; uint pad_h; uint pad_w; uint batch;
    uint out_stride; uint out_offset; // for strided output (concat fusion)
    uint in_stride;  uint in_offset;  // for strided input (split fusion)
    uint has_residual; uint _pad;     // residual add before activation (buffer 5)
};

// ── Direct per-pixel conv ────────────────────────────────────────
// Thread grid: (M, ceil(n_out/4)). Each thread computes 4 output channels
// for one spatial position. No shared memory. Best when K and N are small.

kernel void conv_direct(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint pixel = gid.x;
    const uint oc_base = gid.y * 4;
    if (pixel >= p.m || oc_base >= p.n_out) return;

    const uint nc = min(p.n_out - oc_base, 4u);
    const uint ow_idx = pixel % p.ow;
    const uint oh_idx = (pixel / p.ow) % p.oh;
    const uint b_idx = pixel / (p.oh * p.ow);

    half4 acc = half4(0.0h);
    for (uint i = 0; i < nc; i++) acc[i] = half(bias[oc_base + i]);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    if (is_1x1) {
        const uint in_base = pixel * p.ic;
        for (uint ci = 0; ci < p.ic; ci++) {
            half in_val = half(input[in_base + ci]);
            const uint w_base = ci * p.n_out + oc_base;
            half4 wv;
            if (nc >= 4) {
                wv = half4(half(weight[w_base]), half(weight[w_base + 1]),
                           half(weight[w_base + 2]), half(weight[w_base + 3]));
            } else {
                wv = half4(0.0h);
                for (uint i = 0; i < nc; i++) wv[i] = half(weight[w_base + i]);
            }
            acc = fma(half4(in_val), wv, acc);
        }
    } else {
        for (uint ky = 0; ky < p.kh; ky++) {
            const int iy = int(oh_idx * p.sh + ky) - int(p.pad_h);
            if (iy < 0 || uint(iy) >= p.ih) continue;
            for (uint kx = 0; kx < p.kw; kx++) {
                const int ix = int(ow_idx * p.sw + kx) - int(p.pad_w);
                if (ix < 0 || uint(ix) >= p.iw) continue;
                const uint in_base = ((b_idx * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic;
                const uint k_off = (ky * p.kw + kx) * p.ic;
                for (uint ci = 0; ci < p.ic; ci++) {
                    half in_val = half(input[in_base + ci]);
                    const uint w_base = (k_off + ci) * p.n_out + oc_base;
                    half4 wv;
                    if (nc >= 4) {
                        wv = half4(half(weight[w_base]), half(weight[w_base + 1]),
                                   half(weight[w_base + 2]), half(weight[w_base + 3]));
                    } else {
                        wv = half4(0.0h);
                        for (uint i = 0; i < nc; i++) wv[i] = half(weight[w_base + i]);
                    }
                    acc = fma(half4(in_val), wv, acc);
                }
            }
        }
    }

    float4 v = float4(acc);
    if (p.act == 1u) v = max(v, float4(0.0f));
    else if (p.act == 2u) v = v / (float4(1.0f) + exp(-v));

    const uint out_base = pixel * p.n_out + oc_base;
    for (uint i = 0; i < nc; i++) output[out_base + i] = v[i];
}

// ── F16 weight variants ──────────────────────────────────────────
// Same kernels but weight buffer is pre-converted half*. Saves
// f32→f16 cast on every load = half weight bandwidth.

kernel void conv_direct_f16w(
    device const float* input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint pixel = gid.x;
    const uint oc_base = gid.y * 4;
    if (pixel >= p.m || oc_base >= p.n_out) return;

    const uint nc = min(p.n_out - oc_base, 4u);
    const uint ow_idx = pixel % p.ow;
    const uint oh_idx = (pixel / p.ow) % p.oh;
    const uint b_idx = pixel / (p.oh * p.ow);

    half4 acc = half4(0.0h);
    for (uint i = 0; i < nc; i++) acc[i] = half(bias[oc_base + i]);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    if (is_1x1) {
        const uint in_base = pixel * p.ic;
        for (uint ci = 0; ci < p.ic; ci++) {
            half in_val = half(input[in_base + ci]);
            const uint w_base = ci * p.n_out + oc_base;
            half4 wv;
            if (nc >= 4) {
                wv = half4(weight[w_base], weight[w_base + 1],
                           weight[w_base + 2], weight[w_base + 3]);
            } else {
                wv = half4(0.0h);
                for (uint i = 0; i < nc; i++) wv[i] = weight[w_base + i];
            }
            acc = fma(half4(in_val), wv, acc);
        }
    } else {
        for (uint ky = 0; ky < p.kh; ky++) {
            const int iy = int(oh_idx * p.sh + ky) - int(p.pad_h);
            if (iy < 0 || uint(iy) >= p.ih) continue;
            for (uint kx = 0; kx < p.kw; kx++) {
                const int ix = int(ow_idx * p.sw + kx) - int(p.pad_w);
                if (ix < 0 || uint(ix) >= p.iw) continue;
                const uint in_base = ((b_idx * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic;
                const uint k_off = (ky * p.kw + kx) * p.ic;
                for (uint ci = 0; ci < p.ic; ci++) {
                    half in_val = half(input[in_base + ci]);
                    const uint w_base = (k_off + ci) * p.n_out + oc_base;
                    half4 wv;
                    if (nc >= 4) {
                        wv = half4(weight[w_base], weight[w_base + 1],
                                   weight[w_base + 2], weight[w_base + 3]);
                    } else {
                        wv = half4(0.0h);
                        for (uint i = 0; i < nc; i++) wv[i] = weight[w_base + i];
                    }
                    acc = fma(half4(in_val), wv, acc);
                }
            }
        }
    }

    float4 v = float4(acc);
    if (p.act == 1u) v = max(v, float4(0.0f));
    else if (p.act == 2u) v = v / (float4(1.0f) + exp(-v));

    const uint out_base = pixel * p.n_out + oc_base;
    for (uint i = 0; i < nc; i++) output[out_base + i] = v[i];
}

// ── Tiled GEMM conv ─────────────────────────────────────────────

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;
constant uint TM = 4;
constant uint TN = 4;
constant uint SA_STRIDE = 17;   // BK + 1
constant uint SB_STRIDE = 65;   // BN + 1

// Fast integer divmod via float. Avoids GPU's slow integer division (~20 cyc).
// Uses float division (~4 cyc) + correction. Valid for a < 2^24, b > 0.
inline uint2 fast_divmod(uint a, uint b) {
    uint q = uint(float(a) / float(b));
    uint r = a - q * b;
    // Correct for float rounding (at most off by 1)
    if (r >= b) { q++; r -= b; }
    return uint2(q, r);  // (quotient, remainder)
}

kernel void conv_gemm_basic(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 gid   [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[BM * SA_STRIDE];   // 64 × 17 = 1088
    threadgroup half sb[BK * SB_STRIDE];   // 16 × 65 = 1040

    const uint tx = lid.x;  // 0..15
    const uint ty = lid.y;  // 0..15
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * BM;
    const uint col0 = gid.x * BN;

    // 4×4 = 16 accumulators per thread
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    const uint tiles = (p.k + BK - 1) / BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * BK;

        // Load A: BM × BK = 1024 elements, 256 threads → 4 elements/thread
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;   // / 16
            const uint c = idx & 15;   // % 16
            const uint gr = row0 + r;
            const uint gk = k0 + c;

            half val = 0.0h;
            if (gr < p.m && gk < p.k) {
                if (is_1x1) {
                    val = half(input[gr * p.in_stride + p.in_offset + gk]);
                } else {
                    // Im2col address via fast_divmod (float div ~4 cyc vs int div ~20 cyc)
                    uint2 dm1 = fast_divmod(gr, p.ow);  // (gr/ow, gr%ow)
                    uint2 dm2 = fast_divmod(dm1.x, p.oh);  // (b, oh_idx)
                    uint2 dm3 = fast_divmod(gk, p.ic);  // (ky*kw+kx, ci)
                    uint2 dm4 = fast_divmod(dm3.x, p.kw);  // (ky, kx)

                    const int iy = int(dm2.y * p.sh + dm4.x) - int(p.pad_h);
                    const int ix = int(dm1.y * p.sw + dm4.y) - int(p.pad_w);
                    if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                        val = half(input[((dm2.x * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + dm3.y]);
                    }
                }
            }
            sa[r * SA_STRIDE + c] = val;
        }

        // Load B: BK × BN = 1024 elements, 256 threads → 4 elements/thread
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;   // / 64
            const uint c = idx & 63;   // % 64
            const uint gr = k0 + r;
            const uint gc = col0 + c;

            half val = 0.0h;
            if (gr < p.k && gc < p.n_out) {
                val = half(weight[gr * p.n_out + gc]);
            }
            sb[r * SB_STRIDE + c] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner compute — fully unrolled BK=16 steps
        const uint b_base = tx * TN;
        const uint a_base0 = (ty * TM + 0) * SA_STRIDE;
        const uint a_base1 = (ty * TM + 1) * SA_STRIDE;
        const uint a_base2 = (ty * TM + 2) * SA_STRIDE;
        const uint a_base3 = (ty * TM + 3) * SA_STRIDE;

        for (uint kk = 0; kk < BK; kk++) {
            float4 bv = float4(
                sb[kk * SB_STRIDE + b_base],
                sb[kk * SB_STRIDE + b_base + 1],
                sb[kk * SB_STRIDE + b_base + 2],
                sb[kk * SB_STRIDE + b_base + 3]
            );
            acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
            acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
            acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
            acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: f32 accumulators + bias + activation
    const uint c = col0 + tx * TN;
    if (c + 3 < p.n_out) {
        float4 bv = float4(bias[c], bias[c + 1], bias[c + 2], bias[c + 3]);
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            float4 v = accs[ri] + bv;
            if (p.act == 1u) {
                v = max(v, float4(0.0));
            } else if (p.act == 2u) {
                v = v / (float4(1.0) + exp(-v));
            }
            const uint base = r * p.n_out + c;
            output[base]     = v.x;
            output[base + 1] = v.y;
            output[base + 2] = v.z;
            output[base + 3] = v.w;
        }
    } else {
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < TN; ci++) {
                const uint col = c + ci;
                if (col < p.n_out) {
                    float val = accs[ri][ci] + bias[col];
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    output[r * p.n_out + col] = val;
                }
            }
        }
    }
}

// ── Tiled GEMM conv with f16 pre-packed weights ──────────────────
// Identical to conv_gemm_basic but weight buffer is half* — no f32→f16
// cast on load, half the weight bandwidth.

kernel void conv_gemm_basic_f16w(
    device const float* input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 gid   [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[BM * SA_STRIDE];
    threadgroup half sb[BK * SB_STRIDE];

    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * BM;
    const uint col0 = gid.x * BN;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    const uint tiles = (p.k + BK - 1) / BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * BK;

        // Load A (input, still f32)
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;
            const uint c = idx & 15;
            const uint gr = row0 + r;
            const uint gk = k0 + c;

            half val = 0.0h;
            if (gr < p.m && gk < p.k) {
                if (is_1x1) {
                    val = half(input[gr * p.in_stride + p.in_offset + gk]);
                } else {
                    uint2 dm1 = fast_divmod(gr, p.ow);
                    uint2 dm2 = fast_divmod(dm1.x, p.oh);
                    uint2 dm3 = fast_divmod(gk, p.ic);
                    uint2 dm4 = fast_divmod(dm3.x, p.kw);

                    const int iy = int(dm2.y * p.sh + dm4.x) - int(p.pad_h);
                    const int ix = int(dm1.y * p.sw + dm4.y) - int(p.pad_w);
                    if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                        val = half(input[((dm2.x * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + dm3.y]);
                    }
                }
            }
            sa[r * SA_STRIDE + c] = val;
        }

        // Load B (weights, already f16 — direct load, no cast)
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;

            half val = (gr < p.k && gc < p.n_out) ? weight[gr * p.n_out + gc] : 0.0h;
            sb[r * SB_STRIDE + c] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b_base = tx * TN;
        const uint a_base0 = (ty * TM + 0) * SA_STRIDE;
        const uint a_base1 = (ty * TM + 1) * SA_STRIDE;
        const uint a_base2 = (ty * TM + 2) * SA_STRIDE;
        const uint a_base3 = (ty * TM + 3) * SA_STRIDE;

        for (uint kk = 0; kk < BK; kk++) {
            float4 bv = float4(
                sb[kk * SB_STRIDE + b_base],
                sb[kk * SB_STRIDE + b_base + 1],
                sb[kk * SB_STRIDE + b_base + 2],
                sb[kk * SB_STRIDE + b_base + 3]
            );
            acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
            acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
            acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
            acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint c = col0 + tx * TN;
    if (c + 3 < p.n_out) {
        float4 bv = float4(bias[c], bias[c + 1], bias[c + 2], bias[c + 3]);
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            float4 v = accs[ri] + bv;
            if (p.act == 1u) {
                v = max(v, float4(0.0));
            } else if (p.act == 2u) {
                v = v / (float4(1.0) + exp(-v));
            }
            const uint base = r * p.n_out + c;
            output[base]     = v.x;
            output[base + 1] = v.y;
            output[base + 2] = v.z;
            output[base + 3] = v.w;
        }
    } else {
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < TN; ci++) {
                const uint col = c + ci;
                if (col < p.n_out) {
                    float val = accs[ri][ci] + bias[col];
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    output[r * p.n_out + col] = val;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// F16 full I/O — f16 input, f16 weight, f16 output.
// Bias stays f32. All intermediate buffers are half precision.
// ═══════════════════════════════════════════════════════════════════════

kernel void conv_direct_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint pixel = gid.x;
    const uint oc_base = gid.y * 4;
    if (pixel >= p.m || oc_base >= p.n_out) return;

    const uint nc = min(p.n_out - oc_base, 4u);
    const uint ow_idx = pixel % p.ow;
    const uint oh_idx = (pixel / p.ow) % p.oh;
    const uint b_idx = pixel / (p.oh * p.ow);

    half4 acc = half4(0.0h);
    for (uint i = 0; i < nc; i++) acc[i] = half(bias[oc_base + i]);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    if (is_1x1) {
        const uint in_base = pixel * p.ic;
        for (uint ci = 0; ci < p.ic; ci++) {
            half in_val = input[in_base + ci];
            const uint w_base = ci * p.n_out + oc_base;
            half4 wv;
            if (nc >= 4) {
                wv = half4(weight[w_base], weight[w_base + 1],
                           weight[w_base + 2], weight[w_base + 3]);
            } else {
                wv = half4(0.0h);
                for (uint i = 0; i < nc; i++) wv[i] = weight[w_base + i];
            }
            acc = fma(half4(in_val), wv, acc);
        }
    } else {
        for (uint ky = 0; ky < p.kh; ky++) {
            const int iy = int(oh_idx * p.sh + ky) - int(p.pad_h);
            if (iy < 0 || uint(iy) >= p.ih) continue;
            for (uint kx = 0; kx < p.kw; kx++) {
                const int ix = int(ow_idx * p.sw + kx) - int(p.pad_w);
                if (ix < 0 || uint(ix) >= p.iw) continue;
                const uint in_base = ((b_idx * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic;
                const uint k_off = (ky * p.kw + kx) * p.ic;
                for (uint ci = 0; ci < p.ic; ci++) {
                    half in_val = input[in_base + ci];
                    const uint w_base = (k_off + ci) * p.n_out + oc_base;
                    half4 wv;
                    if (nc >= 4) {
                        wv = half4(weight[w_base], weight[w_base + 1],
                                   weight[w_base + 2], weight[w_base + 3]);
                    } else {
                        wv = half4(0.0h);
                        for (uint i = 0; i < nc; i++) wv[i] = weight[w_base + i];
                    }
                    acc = fma(half4(in_val), wv, acc);
                }
            }
        }
    }

    float4 v = float4(acc);
    if (p.act == 1u) v = max(v, float4(0.0f));
    else if (p.act == 2u) v = v / (float4(1.0f) + exp(-v));
    v = clamp(v, float4(-65504.0f), float4(65504.0f));

    const uint out_base = pixel * p.n_out + oc_base;
    for (uint i = 0; i < nc; i++) output[out_base + i] = half(v[i]);
}

kernel void conv_gemm_basic_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 gid   [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[BM * SA_STRIDE];   // 64×17
    threadgroup half sb[BK * SB_STRIDE];   // 16×65

    // Precomputed spatial decode for tile rows
    threadgroup uint row_ow[BM];
    threadgroup uint row_oh[BM];
    threadgroup uint row_b[BM];

    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * BM;
    const uint col0 = gid.x * BN;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    // Precompute spatial decode once for all tile rows
    if (!is_1x1) {
        for (uint i = tid; i < BM; i += 256) {
            const uint gr = row0 + i;
            if (gr < p.m) {
                uint2 dm1 = fast_divmod(gr, p.ow);
                uint2 dm2 = fast_divmod(dm1.x, p.oh);
                row_ow[i] = dm1.y;
                row_oh[i] = dm2.y;
                row_b[i] = dm2.x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint b_base = tx * TN;
    const uint a_base0 = (ty * TM + 0) * SA_STRIDE;
    const uint a_base1 = (ty * TM + 1) * SA_STRIDE;
    const uint a_base2 = (ty * TM + 2) * SA_STRIDE;
    const uint a_base3 = (ty * TM + 3) * SA_STRIDE;

    if (is_1x1) {
        const uint tiles = (p.k + BK - 1) / BK;
        for (uint t = 0; t < tiles; t++) {
            const uint k0 = t * BK;
            // Load A: 64×16 = 1024 elements, 256 threads → 4 each
            for (uint i = 0; i < 4; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 4;
                const uint c = idx & 15;
                const uint gr = row0 + r;
                const uint gk = k0 + c;
                sa[r * SA_STRIDE + c] = (gr < p.m && gk < p.k)
                    ? input[gr * p.in_stride + p.in_offset + gk] : half(0.0h);
            }
            // Load B: 16×64 = 1024 elements
            for (uint i = 0; i < 4; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 6;
                const uint c = idx & 63;
                const uint gr = k0 + r;
                const uint gc = col0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                    ? weight[gr * p.n_out + gc] : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk++) {
                float4 bv = float4(
                    sb[kk * SB_STRIDE + b_base],
                    sb[kk * SB_STRIDE + b_base + 1],
                    sb[kk * SB_STRIDE + b_base + 2],
                    sb[kk * SB_STRIDE + b_base + 3]
                );
                acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
                acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
                acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
                acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // 3×3+ conv: iterate (ky, kx, ic_tile) — zero divmod in inner loop
        const uint ic_tiles = (p.ic + BK - 1) / BK;
        for (uint ky = 0; ky < p.kh; ky++) {
            for (uint kx = 0; kx < p.kw; kx++) {
                const uint kykx_ic_base = (ky * p.kw + kx) * p.ic;
                for (uint ic_t = 0; ic_t < ic_tiles; ic_t++) {
                    const uint ic_off = ic_t * BK;
                    // Load A
                    for (uint i = 0; i < 4; i++) {
                        const uint idx = tid + i * 256;
                        const uint r = idx >> 4;
                        const uint c = idx & 15;
                        const uint gr = row0 + r;
                        const uint ci = ic_off + c;
                        half val = 0.0h;
                        if (gr < p.m && ci < p.ic) {
                            const int iy = int(row_oh[r] * p.sh + ky) - int(p.pad_h);
                            const int ix = int(row_ow[r] * p.sw + kx) - int(p.pad_w);
                            if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                                val = input[((row_b[r] * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + ci];
                            }
                        }
                        sa[r * SA_STRIDE + c] = val;
                    }
                    // Load B
                    for (uint i = 0; i < 4; i++) {
                        const uint idx = tid + i * 256;
                        const uint r = idx >> 6;
                        const uint c = idx & 63;
                        const uint k_row = kykx_ic_base + ic_off + r;
                        const uint gc = col0 + c;
                        sb[r * SB_STRIDE + c] = (k_row < p.k && gc < p.n_out)
                            ? weight[k_row * p.n_out + gc] : half(0.0h);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint kk = 0; kk < BK; kk++) {
                        float4 bv = float4(
                            sb[kk * SB_STRIDE + b_base],
                            sb[kk * SB_STRIDE + b_base + 1],
                            sb[kk * SB_STRIDE + b_base + 2],
                            sb[kk * SB_STRIDE + b_base + 3]
                        );
                        acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
                        acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
                        acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
                        acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    // Store
    const uint c = col0 + tx * TN;
    if (c + 3 < p.n_out) {
        float4 bv = float4(bias[c], bias[c + 1], bias[c + 2], bias[c + 3]);
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            float4 v = accs[ri] + bv;
            if (p.act == 1u) v = max(v, float4(0.0));
            else if (p.act == 2u) v = v / (float4(1.0) + exp(-v));
            v = clamp(v, float4(-65504.0f), float4(65504.0f));
            *((device half4*)(&output[r * p.n_out + c])) = half4(v);
        }
    } else {
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < TN; ci++) {
                const uint col = c + ci;
                if (col < p.n_out) {
                    float val = accs[ri][ci] + bias[col];
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    val = clamp(val, -65504.0f, 65504.0f);
                    output[r * p.n_out + col] = half(val);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Large-M variant: BM=128, BN=64, BK=16, TM=8, TN=4, 16×16=256 threads.
// 2× M-tile height → doubles B-tile reuse, halves threadgroup launches.
// Best for layers with large M (≥256). Each thread accumulates 32 outputs.
// ═══════════════════════════════════════════════════════════════════════

constant uint LG_BM = 128;
constant uint LG_TM = 8;
constant uint LG_SA_STRIDE = 17;  // BK + 1

kernel void conv_gemm_large_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 gid   [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[LG_BM * LG_SA_STRIDE];  // 128×17 = 4352 bytes
    threadgroup half sb[BK * SB_STRIDE];         // 16×65 = 2080 bytes

    threadgroup uint row_ow[LG_BM];
    threadgroup uint row_oh[LG_BM];
    threadgroup uint row_b[LG_BM];

    const uint tx = lid.x;   // 0..15
    const uint ty = lid.y;   // 0..15
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * LG_BM;
    const uint col0 = gid.x * BN;

    // TM=8 × TN=4 = 32 accumulators per thread (f32 to prevent overflow)
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    float4 acc4 = float4(0.0f);
    float4 acc5 = float4(0.0f);
    float4 acc6 = float4(0.0f);
    float4 acc7 = float4(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    if (!is_1x1) {
        for (uint i = tid; i < LG_BM; i += 256) {
            const uint gr = row0 + i;
            if (gr < p.m) {
                uint2 dm1 = fast_divmod(gr, p.ow);
                uint2 dm2 = fast_divmod(dm1.x, p.oh);
                row_ow[i] = dm1.y;
                row_oh[i] = dm2.y;
                row_b[i] = dm2.x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint b_base = tx * TN;
    const uint a_base0 = (ty * LG_TM + 0) * LG_SA_STRIDE;
    const uint a_base1 = (ty * LG_TM + 1) * LG_SA_STRIDE;
    const uint a_base2 = (ty * LG_TM + 2) * LG_SA_STRIDE;
    const uint a_base3 = (ty * LG_TM + 3) * LG_SA_STRIDE;
    const uint a_base4 = (ty * LG_TM + 4) * LG_SA_STRIDE;
    const uint a_base5 = (ty * LG_TM + 5) * LG_SA_STRIDE;
    const uint a_base6 = (ty * LG_TM + 6) * LG_SA_STRIDE;
    const uint a_base7 = (ty * LG_TM + 7) * LG_SA_STRIDE;

    if (is_1x1) {
        const uint tiles = (p.k + BK - 1) / BK;
        for (uint t = 0; t < tiles; t++) {
            const uint k0 = t * BK;
            // Load A: 128×16 = 2048 elements, 256 threads → 8 each
            for (uint i = 0; i < 8; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 4;
                const uint c = idx & 15;
                const uint gr = row0 + r;
                const uint gk = k0 + c;
                sa[r * LG_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                    ? input[gr * p.in_stride + p.in_offset + gk] : half(0.0h);
            }
            // Load B: 16×64 = 1024 elements, 256 threads → 4 each
            for (uint i = 0; i < 4; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 6;
                const uint c = idx & 63;
                const uint gr = k0 + r;
                const uint gc = col0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                    ? weight[gr * p.n_out + gc] : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk++) {
                float4 bv = float4(
                    sb[kk * SB_STRIDE + b_base],
                    sb[kk * SB_STRIDE + b_base + 1],
                    sb[kk * SB_STRIDE + b_base + 2],
                    sb[kk * SB_STRIDE + b_base + 3]
                );
                acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
                acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
                acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
                acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
                acc4 = fma(float4(sa[a_base4 + kk]), bv, acc4);
                acc5 = fma(float4(sa[a_base5 + kk]), bv, acc5);
                acc6 = fma(float4(sa[a_base6 + kk]), bv, acc6);
                acc7 = fma(float4(sa[a_base7 + kk]), bv, acc7);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const uint ic_tiles = (p.ic + BK - 1) / BK;
        for (uint ky = 0; ky < p.kh; ky++) {
            for (uint kx = 0; kx < p.kw; kx++) {
                const uint kykx_ic_base = (ky * p.kw + kx) * p.ic;
                for (uint ic_t = 0; ic_t < ic_tiles; ic_t++) {
                    const uint ic_off = ic_t * BK;
                    // Load A: 128×16 = 2048 elements, 256 threads → 8 each
                    for (uint i = 0; i < 8; i++) {
                        const uint idx = tid + i * 256;
                        const uint r = idx >> 4;
                        const uint c = idx & 15;
                        const uint gr = row0 + r;
                        const uint ci = ic_off + c;
                        half val = 0.0h;
                        if (gr < p.m && ci < p.ic) {
                            const int iy = int(row_oh[r] * p.sh + ky) - int(p.pad_h);
                            const int ix = int(row_ow[r] * p.sw + kx) - int(p.pad_w);
                            if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                                val = input[((row_b[r] * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + ci];
                            }
                        }
                        sa[r * LG_SA_STRIDE + c] = val;
                    }
                    // Load B: 16×64 = 1024, 256 threads → 4 each
                    for (uint i = 0; i < 4; i++) {
                        const uint idx = tid + i * 256;
                        const uint r = idx >> 6;
                        const uint c = idx & 63;
                        const uint k_row = kykx_ic_base + ic_off + r;
                        const uint gc = col0 + c;
                        sb[r * SB_STRIDE + c] = (k_row < p.k && gc < p.n_out)
                            ? weight[k_row * p.n_out + gc] : half(0.0h);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint kk = 0; kk < BK; kk++) {
                        float4 bv = float4(
                            sb[kk * SB_STRIDE + b_base],
                            sb[kk * SB_STRIDE + b_base + 1],
                            sb[kk * SB_STRIDE + b_base + 2],
                            sb[kk * SB_STRIDE + b_base + 3]
                        );
                        acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
                        acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
                        acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
                        acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
                        acc4 = fma(float4(sa[a_base4 + kk]), bv, acc4);
                        acc5 = fma(float4(sa[a_base5 + kk]), bv, acc5);
                        acc6 = fma(float4(sa[a_base6 + kk]), bv, acc6);
                        acc7 = fma(float4(sa[a_base7 + kk]), bv, acc7);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    // Store: 8 rows per thread
    const uint c = col0 + tx * TN;
    float4 accs[8] = { acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 };
    if (c + 3 < p.n_out) {
        float4 bv = float4(bias[c], bias[c + 1], bias[c + 2], bias[c + 3]);
        for (uint ri = 0; ri < LG_TM; ri++) {
            const uint r = row0 + ty * LG_TM + ri;
            if (r >= p.m) continue;
            float4 v = accs[ri] + bv;
            if (p.act == 1u) v = max(v, float4(0.0));
            else if (p.act == 2u) v = v / (float4(1.0) + exp(-v));
            v = clamp(v, float4(-65504.0f), float4(65504.0f));
            *((device half4*)(&output[r * p.n_out + c])) = half4(v);
        }
    } else {
        for (uint ri = 0; ri < LG_TM; ri++) {
            const uint r = row0 + ty * LG_TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < TN; ci++) {
                const uint col = c + ci;
                if (col < p.n_out) {
                    float val = accs[ri][ci] + bias[col];
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    val = clamp(val, -65504.0f, 65504.0f);
                    output[r * p.n_out + col] = half(val);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Small-N variant: BM=64, BN=32, BK=16, TM=4, TN=2, 16×16=256 threads.
// For layers with N≤32, eliminates 50-75% tile waste vs BN=64.
// ═══════════════════════════════════════════════════════════════════════

constant uint SM_BN  = 32;
constant uint SM_TN  = 2;
constant uint SM_SBS = 33;   // BN + 1

kernel void conv_gemm_small_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    device const half*  residual [[buffer(5)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 gid   [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[BM * SA_STRIDE];    // 64×17
    threadgroup half sb[BK * SM_SBS];       // 16×33 (half the size of BN=64 variant)

    threadgroup uint row_ow[BM];
    threadgroup uint row_oh[BM];
    threadgroup uint row_b[BM];

    const uint tx = lid.x;   // 0..15
    const uint ty = lid.y;   // 0..15
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * BM;
    const uint col0 = gid.x * SM_BN;

    // TM=4 × TN=2 = 8 accumulators per thread (f32 to prevent overflow)
    float2 acc0 = float2(0.0f);
    float2 acc1 = float2(0.0f);
    float2 acc2 = float2(0.0f);
    float2 acc3 = float2(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    if (!is_1x1) {
        for (uint i = tid; i < BM; i += 256) {
            const uint gr = row0 + i;
            if (gr < p.m) {
                uint2 dm1 = fast_divmod(gr, p.ow);
                uint2 dm2 = fast_divmod(dm1.x, p.oh);
                row_ow[i] = dm1.y;
                row_oh[i] = dm2.y;
                row_b[i] = dm2.x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint b_base = tx * SM_TN;
    const uint a_base0 = (ty * TM + 0) * SA_STRIDE;
    const uint a_base1 = (ty * TM + 1) * SA_STRIDE;
    const uint a_base2 = (ty * TM + 2) * SA_STRIDE;
    const uint a_base3 = (ty * TM + 3) * SA_STRIDE;

    if (is_1x1) {
        const uint tiles = (p.k + BK - 1) / BK;
        for (uint t = 0; t < tiles; t++) {
            const uint k0 = t * BK;
            // Load A: 64×16 = 1024 elements, 256 threads → 4 each
            for (uint i = 0; i < 4; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 4;
                const uint c = idx & 15;
                const uint gr = row0 + r;
                const uint gk = k0 + c;
                sa[r * SA_STRIDE + c] = (gr < p.m && gk < p.k)
                    ? input[gr * p.in_stride + p.in_offset + gk] : half(0.0h);
            }
            // Load B: 16×32 = 512 elements, 256 threads → 2 each
            for (uint i = 0; i < 2; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 5;    // /32
                const uint c = idx & 31;    // %32
                const uint gr = k0 + r;
                const uint gc = col0 + c;
                sb[r * SM_SBS + c] = (gr < p.k && gc < p.n_out)
                    ? weight[gr * p.n_out + gc] : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk++) {
                float2 bv = float2(
                    sb[kk * SM_SBS + b_base],
                    sb[kk * SM_SBS + b_base + 1]
                );
                acc0 = fma(float2(sa[a_base0 + kk]), bv, acc0);
                acc1 = fma(float2(sa[a_base1 + kk]), bv, acc1);
                acc2 = fma(float2(sa[a_base2 + kk]), bv, acc2);
                acc3 = fma(float2(sa[a_base3 + kk]), bv, acc3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const uint ic_tiles = (p.ic + BK - 1) / BK;
        for (uint ky = 0; ky < p.kh; ky++) {
            for (uint kx = 0; kx < p.kw; kx++) {
                const uint kykx_ic_base = (ky * p.kw + kx) * p.ic;
                for (uint ic_t = 0; ic_t < ic_tiles; ic_t++) {
                    const uint ic_off = ic_t * BK;
                    // Load A: same as BN=64 variant
                    for (uint i = 0; i < 4; i++) {
                        const uint idx = tid + i * 256;
                        const uint r = idx >> 4;
                        const uint c = idx & 15;
                        const uint gr = row0 + r;
                        const uint ci = ic_off + c;
                        half val = 0.0h;
                        if (gr < p.m && ci < p.ic) {
                            const int iy = int(row_oh[r] * p.sh + ky) - int(p.pad_h);
                            const int ix = int(row_ow[r] * p.sw + kx) - int(p.pad_w);
                            if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                                val = input[((row_b[r] * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + ci];
                            }
                        }
                        sa[r * SA_STRIDE + c] = val;
                    }
                    // Load B: 16×32 = 512 elements
                    for (uint i = 0; i < 2; i++) {
                        const uint idx = tid + i * 256;
                        const uint r = idx >> 5;
                        const uint c = idx & 31;
                        const uint k_row = kykx_ic_base + ic_off + r;
                        const uint gc = col0 + c;
                        sb[r * SM_SBS + c] = (k_row < p.k && gc < p.n_out)
                            ? weight[k_row * p.n_out + gc] : half(0.0h);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint kk = 0; kk < BK; kk++) {
                        float2 bv = float2(
                            sb[kk * SM_SBS + b_base],
                            sb[kk * SM_SBS + b_base + 1]
                        );
                        acc0 = fma(float2(sa[a_base0 + kk]), bv, acc0);
                        acc1 = fma(float2(sa[a_base1 + kk]), bv, acc1);
                        acc2 = fma(float2(sa[a_base2 + kk]), bv, acc2);
                        acc3 = fma(float2(sa[a_base3 + kk]), bv, acc3);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    // Store
    const uint c = col0 + tx * SM_TN;
    if (c + 1 < p.n_out) {
        float2 bv = float2(bias[c], bias[c + 1]);
        float2 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            float2 v = accs[ri] + bv;
            if (p.has_residual) {
                const uint base = r * p.n_out + c;
                v += float2(residual[base], residual[base + 1]);
            }
            if (p.act == 1u) v = max(v, float2(0.0));
            else if (p.act == 2u) v = v / (float2(1.0) + exp(-v));
            v = clamp(v, float2(-65504.0f), float2(65504.0f));
            const uint base = r * p.n_out + c;
            output[base]     = half(v.x);
            output[base + 1] = half(v.y);
        }
    } else {
        float2 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < SM_TN; ci++) {
                const uint col = c + ci;
                if (col < p.n_out) {
                    float val = accs[ri][ci] + bias[col];
                    if (p.has_residual) val += float(residual[r * p.n_out + col]);
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    val = clamp(val, -65504.0f, 65504.0f);
                    output[r * p.n_out + col] = half(val);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Optimized tiled GEMM: BK=32 (fewer barriers), vectorized half4
// stores/reads. Same BM=64, BN=64 tile, 16×16=256 threads.
// ═══════════════════════════════════════════════════════════════════════

kernel void conv_gemm_v2_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 gid   [[threadgroup_position_in_grid]]
) {
    // BK=32: double K tile → half the iterations, half the barriers
    constexpr uint BK2 = 32;
    constexpr uint SA2S = 33;   // BK2+1, bank conflict padding
    constexpr uint SB2S = 68;   // BN+4, aligned to half4 (68%4==0)

    threadgroup half sa[BM * SA2S];   // 64×33
    threadgroup half sb[BK2 * SB2S];  // 32×68

    threadgroup uint row_ow[BM];
    threadgroup uint row_oh[BM];
    threadgroup uint row_b[BM];

    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * BM;
    const uint col0 = gid.x * BN;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    // Precompute spatial decode once for all tile rows
    if (!is_1x1) {
        for (uint i = tid; i < BM; i += 256) {
            const uint gr = row0 + i;
            if (gr < p.m) {
                uint2 dm1 = fast_divmod(gr, p.ow);
                uint2 dm2 = fast_divmod(dm1.x, p.oh);
                row_ow[i] = dm1.y;
                row_oh[i] = dm2.y;
                row_b[i] = dm2.x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint tiles = (p.k + BK2 - 1) / BK2;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * BK2;

        // Load A: 64×32 = 2048 elements, 256 threads → 8 per thread
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 5;   // /32
            const uint c = idx & 31;
            const uint gr = row0 + r;
            const uint gk = k0 + c;

            half val = 0.0h;
            if (gr < p.m && gk < p.k) {
                if (is_1x1) {
                    val = input[gr * p.in_stride + p.in_offset + gk];
                } else {
                    uint2 dm3 = fast_divmod(gk, p.ic);
                    uint2 dm4 = fast_divmod(dm3.x, p.kw);
                    const int iy = int(row_oh[r] * p.sh + dm4.x) - int(p.pad_h);
                    const int ix = int(row_ow[r] * p.sw + dm4.y) - int(p.pad_w);
                    if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                        val = input[((row_b[r] * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + dm3.y];
                    }
                }
            }
            sa[r * SA2S + c] = val;
        }

        // Load B: 32×64 = 2048 elements, 256 threads → 8 per thread
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;   // /64
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * SB2S + c] = (gr < p.k && gc < p.n_out) ? weight[gr * p.n_out + gc] : 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: 32 FMA iterations (double the work per barrier pair)
        const uint b_base = tx * TN;
        const uint a_base0 = (ty * TM + 0) * SA2S;
        const uint a_base1 = (ty * TM + 1) * SA2S;
        const uint a_base2 = (ty * TM + 2) * SA2S;
        const uint a_base3 = (ty * TM + 3) * SA2S;

        for (uint kk = 0; kk < BK2; kk++) {
            // half4 vectorized read (SB2S=68 ensures 4-element alignment)
            half4 bv_h = *((threadgroup const half4*)(&sb[kk * SB2S + b_base]));
            float4 bv = float4(bv_h);
            acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
            acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
            acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
            acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store with vectorized half4 writes
    const uint c = col0 + tx * TN;
    if (c + 3 < p.n_out) {
        float4 bv = float4(bias[c], bias[c + 1], bias[c + 2], bias[c + 3]);
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            float4 v = accs[ri] + bv;
            if (p.act == 1u) v = max(v, float4(0.0));
            else if (p.act == 2u) v = v / (float4(1.0) + exp(-v));
            v = clamp(v, float4(-65504.0f), float4(65504.0f));
            // Single 8-byte write instead of 4× 2-byte writes
            *((device half4*)(&output[r * p.n_out + c])) = half4(v);
        }
    } else {
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < TM; ri++) {
            const uint r = row0 + ty * TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < TN; ci++) {
                const uint col = c + ci;
                if (col < p.n_out) {
                    float val = accs[ri][ci] + bias[col];
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    val = clamp(val, -65504.0f, 65504.0f);
                    output[r * p.n_out + col] = half(val);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Depthwise conv — each output channel depends only on its own input channel.
// Grid: (M=N*OH*OW, ceil(C/4)). Each thread handles 4 channels for 1 pixel.
// Weight layout: [KH*KW, C] (KHWC with I=1), weight[(ky*kw+kx)*n_out + ch].
// Uses f32 accumulators, f16 I/O. p.ic == p.n_out for depthwise.
// ═══════════════════════════════════════════════════════════════════════

kernel void depthwise_conv_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint pixel = gid.x;
    const uint ch_base = gid.y * 4;
    if (pixel >= p.m || ch_base >= p.n_out) return;

    const uint nc = min(p.n_out - ch_base, 4u);
    const uint ow_idx = pixel % p.ow;
    const uint oh_idx = (pixel / p.ow) % p.oh;
    const uint b_idx = pixel / (p.oh * p.ow);

    float4 acc = float4(0.0f);
    for (uint i = 0; i < nc; i++) acc[i] = bias[ch_base + i];

    for (uint ky = 0; ky < p.kh; ky++) {
        const int iy = int(oh_idx * p.sh + ky) - int(p.pad_h);
        if (iy < 0 || uint(iy) >= p.ih) continue;
        for (uint kx = 0; kx < p.kw; kx++) {
            const int ix = int(ow_idx * p.sw + kx) - int(p.pad_w);
            if (ix < 0 || uint(ix) >= p.iw) continue;

            const uint in_base = ((b_idx * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + ch_base;
            const uint w_base = (ky * p.kw + kx) * p.n_out + ch_base;

            if (nc >= 4) {
                acc += float4(input[in_base], input[in_base+1], input[in_base+2], input[in_base+3])
                     * float4(weight[w_base], weight[w_base+1], weight[w_base+2], weight[w_base+3]);
            } else {
                for (uint i = 0; i < nc; i++) {
                    acc[i] += float(input[in_base + i]) * float(weight[w_base + i]);
                }
            }
        }
    }

    if (p.act == 1u) acc = max(acc, float4(0.0f));
    else if (p.act == 2u) acc = acc / (float4(1.0f) + exp(-acc));
    acc = clamp(acc, float4(-65504.0f), float4(65504.0f));

    const uint out_base = pixel * p.n_out + ch_base;
    for (uint i = 0; i < nc; i++) output[out_base + i] = half(acc[i]);
}

// ═══════════════════════════════════════════════════════════════════════
// SIMD conv: uses simdgroup_multiply_accumulate for half×half→float.
// Handles both 1×1 (direct) and 3×3+ (im2col) convolutions.
// BM=64, BN=64, BK=16, 8 simdgroups (256 threads), each covers 32×16.
// Fuses bias add and optional activation at the store step.
// ═══════════════════════════════════════════════════════════════════════

constant uint SIMD_BM  = 64;
constant uint SIMD_BN  = 64;
constant uint SIMD_BK  = 16;
constant uint SIMD_SA_STRIDE = 17;  // BK + 1 to avoid bank conflicts
constant uint SIMD_SB_STRIDE = 65;  // BN + 1

kernel void conv1x1_simd_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    device const half*  residual [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[SIMD_BM * SIMD_SA_STRIDE];  // 64×17
    threadgroup half sb[SIMD_BK * SIMD_SB_STRIDE];  // 16×65

    const uint tid = simd_idx * 32 + simd_lane;
    const uint row0 = tgid.y * SIMD_BM;
    const uint col0 = tgid.x * SIMD_BN;

    const uint sg_m = simd_idx / 4;
    const uint sg_n = simd_idx % 4;

    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint tiles = (p.k + SIMD_BK - 1) / SIMD_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * SIMD_BK;

        // Load A tile: 64×16 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;
            const uint c = idx & 15;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * SIMD_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? input[gr * p.in_stride + p.in_offset + gk] : half(0);
        }

        // Load B tile: 16×64 = 1024 elems
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * SIMD_SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                ? weight[gr * p.n_out + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint km = 0; km < 2; km++) {
            simdgroup_matrix<half, 8, 8> b_mat[2];
            for (uint bn = 0; bn < 2; bn++)
                simdgroup_load(b_mat[bn],
                    &sb[km * 8 * SIMD_SB_STRIDE + sg_n * 16 + bn * 8],
                    SIMD_SB_STRIDE);
            for (uint bm = 0; bm < 4; bm++) {
                simdgroup_matrix<half, 8, 8> a_mat;
                simdgroup_load(a_mat,
                    &sa[(sg_m * 32 + bm * 8) * SIMD_SA_STRIDE + km * 8],
                    SIMD_SA_STRIDE);
                for (uint bn = 0; bn < 2; bn++)
                    simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store with fused bias + activation
    threadgroup float sg_scratch[8 * 64];

    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 2; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 16 + bn * 8;

            simdgroup_store(acc[bm][bn], &sg_scratch[simd_idx * 64], 8);

            if (m_off < p.m && n_off < p.n_out) {
                for (uint e = 0; e < 2; e++) {
                    const uint flat = simd_lane * 2 + e;
                    const uint lr = flat >> 3;
                    const uint lc = flat & 7;
                    const uint gr = m_off + lr;
                    const uint gc = n_off + lc;
                    if (gr < p.m && gc < p.n_out) {
                        float val = sg_scratch[simd_idx * 64 + lr * 8 + lc] + bias[gc];
                        if (p.has_residual) val += float(residual[gr * p.out_stride + p.out_offset + gc]);
                        if (p.act == 1u) val = max(val, 0.0f);
                        else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                        val = clamp(val, -65504.0f, 65504.0f);
                        output[gr * p.out_stride + p.out_offset + gc] = half(val);
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SIMD conv1x1 BK=32: BM=64, BN=64, BK=32.
// Same 2×4 simdgroup layout as conv1x1_simd_f16io but double K-tile.
// Halves barrier count for large-K convolutions. Trades higher shared memory
// (10.4KB vs 6.3KB per threadgroup) for fewer barrier stalls.
// Best for K >= 128 where many K-tiles dominate runtime.
// ═══════════════════════════════════════════════════════════════════════

constant uint S32_BM  = 64;
constant uint S32_BN  = 64;
constant uint S32_BK  = 32;
constant uint S32_SA_STRIDE = 33;  // BK + 1
constant uint S32_SB_STRIDE = 65;  // BN + 1

kernel void conv1x1_simd_bk32_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[S32_BM * S32_SA_STRIDE];   // 64×33
    threadgroup half sb[S32_BK * S32_SB_STRIDE];   // 32×65

    const uint tid = simd_idx * 32 + simd_lane;
    const uint row0 = tgid.y * S32_BM;
    const uint col0 = tgid.x * S32_BN;

    const uint sg_m = simd_idx / 4;
    const uint sg_n = simd_idx % 4;

    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint tiles = (p.k + S32_BK - 1) / S32_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * S32_BK;

        // Load A tile: 64×32 = 2048 elems, 256 threads → 8 each
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 5;   // / 32
            const uint c = idx & 31;   // % 32
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * S32_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? input[gr * p.in_stride + p.in_offset + gk] : half(0);
        }

        // Load B tile: 32×64 = 2048 elems, 256 threads → 8 each
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;   // / 64
            const uint c = idx & 63;   // % 64
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * S32_SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                ? weight[gr * p.n_out + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4 K-blocks of 8 (BK=32)
        for (uint km = 0; km < 4; km++) {
            simdgroup_matrix<half, 8, 8> b_mat[2];
            for (uint bn = 0; bn < 2; bn++)
                simdgroup_load(b_mat[bn],
                    &sb[km * 8 * S32_SB_STRIDE + sg_n * 16 + bn * 8],
                    S32_SB_STRIDE);
            for (uint bm = 0; bm < 4; bm++) {
                simdgroup_matrix<half, 8, 8> a_mat;
                simdgroup_load(a_mat,
                    &sa[(sg_m * 32 + bm * 8) * S32_SA_STRIDE + km * 8],
                    S32_SA_STRIDE);
                for (uint bn = 0; bn < 2; bn++)
                    simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store with fused bias + activation
    threadgroup float sg_scratch[8 * 64];

    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 2; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 16 + bn * 8;

            simdgroup_store(acc[bm][bn], &sg_scratch[simd_idx * 64], 8);

            if (m_off < p.m && n_off < p.n_out) {
                for (uint e = 0; e < 2; e++) {
                    const uint flat = simd_lane * 2 + e;
                    const uint lr = flat >> 3;
                    const uint lc = flat & 7;
                    const uint gr = m_off + lr;
                    const uint gc = n_off + lc;
                    if (gr < p.m && gc < p.n_out) {
                        float val = sg_scratch[simd_idx * 64 + lr * 8 + lc] + bias[gc];
                        if (p.act == 1u) val = max(val, 0.0f);
                        else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                        val = clamp(val, -65504.0f, 65504.0f);
                        output[gr * p.out_stride + p.out_offset + gc] = half(val);
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SIMD conv1x1 4-SG: BM=64, BN=64, BK=16, 4 simdgroups (128 threads).
// 2×2 simdgroup layout, each sg covers 32×32 with acc[4][4] = 16 accumulators.
// 67% MAC utilization (vs 57% for 2×4 layout).
// Only 5.3KB shared memory → 6 TGs per core (vs 5 for 8-sg kernel).
// ═══════════════════════════════════════════════════════════════════════

kernel void conv1x1_simd_4sg_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint  tid        [[thread_index_in_threadgroup]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[SIMD_BM * SIMD_SA_STRIDE];  // 64×17
    threadgroup half sb[SIMD_BK * SIMD_SB_STRIDE];  // 16×65
    threadgroup float sg_scratch[4 * 64];  // 4 simdgroups × 64 floats

    const uint row0 = tgid.y * SIMD_BM;
    const uint col0 = tgid.x * SIMD_BN;

    // 2×2 simdgroup layout: each covers 32×32
    const uint sg_m = simd_idx / 2;
    const uint sg_n = simd_idx % 2;

    simdgroup_matrix<float, 8, 8> acc[16];  // flat [bm*4+bn]
    for (uint i = 0; i < 16; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint tiles = (p.k + SIMD_BK - 1) / SIMD_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * SIMD_BK;

        // Load A tile: 64×16 = 1024 elems, 128 threads → 8 each
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 128;
            const uint r = idx >> 4;
            const uint c = idx & 15;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * SIMD_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? input[gr * p.in_stride + p.in_offset + gk] : half(0);
        }

        // Load B tile: 16×64 = 1024 elems, 128 threads → 8 each
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 128;
            const uint r = idx >> 6;
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * SIMD_SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                ? weight[gr * p.n_out + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint km = 0; km < 2; km++) {
            simdgroup_matrix<half, 8, 8> b_mat[4];
            for (uint bn = 0; bn < 4; bn++)
                simdgroup_load(b_mat[bn],
                    &sb[km * 8 * SIMD_SB_STRIDE + sg_n * 32 + bn * 8],
                    SIMD_SB_STRIDE);
            for (uint bm = 0; bm < 4; bm++) {
                simdgroup_matrix<half, 8, 8> a_mat;
                simdgroup_load(a_mat,
                    &sa[(sg_m * 32 + bm * 8) * SIMD_SA_STRIDE + km * 8],
                    SIMD_SA_STRIDE);
                for (uint bn = 0; bn < 4; bn++)
                    simdgroup_multiply_accumulate(acc[bm * 4 + bn], a_mat, b_mat[bn], acc[bm * 4 + bn]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store with fused bias + activation
    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 4; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 32 + bn * 8;

            simdgroup_store(acc[bm * 4 + bn], &sg_scratch[simd_idx * 64], 8);

            if (m_off < p.m && n_off < p.n_out) {
                for (uint e = 0; e < 2; e++) {
                    const uint flat = simd_lane * 2 + e;
                    const uint lr = flat >> 3;
                    const uint lc = flat & 7;
                    const uint gr = m_off + lr;
                    const uint gc = n_off + lc;
                    if (gr < p.m && gc < p.n_out) {
                        float val = sg_scratch[simd_idx * 64 + lr * 8 + lc] + bias[gc];
                        if (p.act == 1u) val = max(val, 0.0f);
                        else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                        val = clamp(val, -65504.0f, 65504.0f);
                        output[gr * p.out_stride + p.out_offset + gc] = half(val);
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SIMD conv1x1 LARGE: BM=128, BN=64, BK=16.
// 4×2 simdgroup layout (4 rows, 2 cols), each sg covers 32×32.
// Each sg: 4 bm × 4 bn = 16 accumulators → higher arithmetic intensity.
// AI = 128*64/(128+64) = 42.67 FLOP/byte (vs 32 for BM=64).
// ═══════════════════════════════════════════════════════════════════════

constant uint SL_BM = 128;
constant uint SL_BN = 64;
constant uint SL_BK = 16;
constant uint SL_SA_STRIDE = 17;  // BK + 1
constant uint SL_SB_STRIDE = 65;  // BN + 1

kernel void conv1x1_simd_large_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[SL_BM * SL_SA_STRIDE];   // 128×17
    threadgroup half sb[SL_BK * SL_SB_STRIDE];   // 16×65

    const uint tid = simd_idx * 32 + simd_lane;
    const uint row0 = tgid.y * SL_BM;
    const uint col0 = tgid.x * SL_BN;

    // 4×2 simdgroup layout
    const uint sg_m = simd_idx / 2;  // 0..3
    const uint sg_n = simd_idx % 2;  // 0..1

    // 4 bm × 4 bn = 16 accumulators per simdgroup
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint tiles = (p.k + SL_BK - 1) / SL_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * SL_BK;

        // Load A tile: 128×16 = 2048 elems, 256 threads → 8 each
        for (uint i = 0; i < 8; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;
            const uint c = idx & 15;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * SL_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? input[gr * p.in_stride + p.in_offset + gk] : half(0);
        }

        // Load B tile: 16×64 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * SL_SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                ? weight[gr * p.n_out + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint km = 0; km < 2; km++) {
            simdgroup_matrix<half, 8, 8> b_mat[4];
            for (uint bn = 0; bn < 4; bn++)
                simdgroup_load(b_mat[bn],
                    &sb[km * 8 * SL_SB_STRIDE + sg_n * 32 + bn * 8],
                    SL_SB_STRIDE);
            for (uint bm = 0; bm < 4; bm++) {
                simdgroup_matrix<half, 8, 8> a_mat;
                simdgroup_load(a_mat,
                    &sa[(sg_m * 32 + bm * 8) * SL_SA_STRIDE + km * 8],
                    SL_SA_STRIDE);
                for (uint bn = 0; bn < 4; bn++)
                    simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store with fused bias + activation
    threadgroup float sg_scratch[8 * 64];

    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 4; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 32 + bn * 8;

            simdgroup_store(acc[bm][bn], &sg_scratch[simd_idx * 64], 8);

            if (m_off < p.m && n_off < p.n_out) {
                for (uint e = 0; e < 2; e++) {
                    const uint flat = simd_lane * 2 + e;
                    const uint lr = flat >> 3;
                    const uint lc = flat & 7;
                    const uint gr = m_off + lr;
                    const uint gc = n_off + lc;
                    if (gr < p.m && gc < p.n_out) {
                        float val = sg_scratch[simd_idx * 64 + lr * 8 + lc] + bias[gc];
                        if (p.act == 1u) val = max(val, 0.0f);
                        else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                        val = clamp(val, -65504.0f, 65504.0f);
                        output[gr * p.out_stride + p.out_offset + gc] = half(val);
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SIMD 3×3+ conv with im2col: same SIMD matmul core, but loads A via
// im2col (spatial decode + padding check). Handles stride/padding.
// ═══════════════════════════════════════════════════════════════════════

kernel void conv3x3_simd_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    device const half*  residual [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[SIMD_BM * SIMD_SA_STRIDE];
    threadgroup half sb[SIMD_BK * SIMD_SB_STRIDE];

    // Precomputed spatial decode for tile rows
    threadgroup uint row_ow[SIMD_BM];
    threadgroup uint row_oh[SIMD_BM];
    threadgroup uint row_b[SIMD_BM];

    const uint tid = simd_idx * 32 + simd_lane;
    const uint row0 = tgid.y * SIMD_BM;
    const uint col0 = tgid.x * SIMD_BN;

    const uint sg_m = simd_idx / 4;
    const uint sg_n = simd_idx % 4;

    // Precompute spatial decode
    for (uint i = tid; i < SIMD_BM; i += 256) {
        const uint gr = row0 + i;
        if (gr < p.m) {
            uint2 dm1 = fast_divmod(gr, p.ow);
            uint2 dm2 = fast_divmod(dm1.x, p.oh);
            row_ow[i] = dm1.y;
            row_oh[i] = dm2.y;
            row_b[i] = dm2.x;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint ic_tiles = (p.ic + SIMD_BK - 1) / SIMD_BK;
    for (uint ky = 0; ky < p.kh; ky++) {
        for (uint kx = 0; kx < p.kw; kx++) {
            const uint kykx_ic_base = (ky * p.kw + kx) * p.ic;
            for (uint ic_t = 0; ic_t < ic_tiles; ic_t++) {
                const uint ic_off = ic_t * SIMD_BK;

                // Load A: im2col — read input at (oh*sh+ky, ow*sw+kx, ic)
                for (uint i = 0; i < 4; i++) {
                    const uint idx = tid + i * 256;
                    const uint r = idx >> 4;
                    const uint c = idx & 15;
                    const uint gr = row0 + r;
                    const uint ci = ic_off + c;
                    half val = 0.0h;
                    if (gr < p.m && ci < p.ic) {
                        const int iy = int(row_oh[r] * p.sh + ky) - int(p.pad_h);
                        const int ix = int(row_ow[r] * p.sw + kx) - int(p.pad_w);
                        if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                            val = input[((row_b[r] * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + ci];
                        }
                    }
                    sa[r * SIMD_SA_STRIDE + c] = val;
                }

                // Load B: weight[(ky*kw+kx)*ic + ic_off + r, col0 + c]
                for (uint i = 0; i < 4; i++) {
                    const uint idx = tid + i * 256;
                    const uint r = idx >> 6;
                    const uint c = idx & 63;
                    const uint k_row = kykx_ic_base + ic_off + r;
                    const uint gc = col0 + c;
                    sb[r * SIMD_SB_STRIDE + c] = (k_row < p.k && gc < p.n_out)
                        ? weight[k_row * p.n_out + gc] : half(0.0h);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // SIMD matmul
                for (uint km = 0; km < 2; km++) {
                    simdgroup_matrix<half, 8, 8> b_mat[2];
                    for (uint bn = 0; bn < 2; bn++)
                        simdgroup_load(b_mat[bn],
                            &sb[km * 8 * SIMD_SB_STRIDE + sg_n * 16 + bn * 8],
                            SIMD_SB_STRIDE);
                    for (uint bm = 0; bm < 4; bm++) {
                        simdgroup_matrix<half, 8, 8> a_mat;
                        simdgroup_load(a_mat,
                            &sa[(sg_m * 32 + bm * 8) * SIMD_SA_STRIDE + km * 8],
                            SIMD_SA_STRIDE);
                        for (uint bn = 0; bn < 2; bn++)
                            simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Store with fused bias + residual + activation
    threadgroup float sg_scratch[8 * 64];

    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 2; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 16 + bn * 8;

            simdgroup_store(acc[bm][bn], &sg_scratch[simd_idx * 64], 8);

            if (m_off < p.m && n_off < p.n_out) {
                for (uint e = 0; e < 2; e++) {
                    const uint flat = simd_lane * 2 + e;
                    const uint lr = flat >> 3;
                    const uint lc = flat & 7;
                    const uint gr = m_off + lr;
                    const uint gc = n_off + lc;
                    if (gr < p.m && gc < p.n_out) {
                        float val = sg_scratch[simd_idx * 64 + lr * 8 + lc] + bias[gc];
                        if (p.has_residual) val += float(residual[gr * p.out_stride + p.out_offset + gc]);
                        if (p.act == 1u) val = max(val, 0.0f);
                        else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                        val = clamp(val, -65504.0f, 65504.0f);
                        output[gr * p.out_stride + p.out_offset + gc] = half(val);
                    }
                }
            }
        }
    }
}
