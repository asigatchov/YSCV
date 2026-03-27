// Metal conv_gemm using simdgroup_matrix_multiply_accumulate (Apple Silicon).
// BM=64, BN=64, BK=16. 8 simdgroups × 32 threads = 256 threads per threadgroup.
// Each simdgroup handles a 32×16 output sub-tile using 4×2 grid of 8×8 matrices.
// F16 input/weight, f32 accumulators (native hw throughput), f16 output. Zero divmod for 3×3+ convs.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Params {
    uint m;       // output rows = batch * oh * ow
    uint n_out;   // output channels
    uint k;       // kh * kw * ic
    uint act;     // 0=none, 1=relu, 2=silu
    uint ih; uint iw; uint ic; uint oh;
    uint ow; uint kh; uint kw; uint sh;
    uint sw; uint pad_h; uint pad_w; uint batch;
};

inline uint2 fast_divmod(uint a, uint b) {
    uint q = uint(float(a) / float(b));
    uint r = a - q * b;
    if (r >= b) { q++; r -= b; }
    return uint2(q, r);
}

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;
constant uint SA_STRIDE = BK + 1;  // 17 — bank conflict avoidance
constant uint SB_STRIDE = BN + 1;  // 65

kernel void conv_gemm_simd_f16io(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device half*        output  [[buffer(3)]],
    constant Params&    p       [[buffer(4)]],
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[BM * SA_STRIDE];   // 64×17
    threadgroup half sb[BK * SB_STRIDE];   // 16×65

    // Precomputed spatial decode for tile rows
    threadgroup uint row_ow[BM];
    threadgroup uint row_oh[BM];
    threadgroup uint row_b[BM];

    // Per-simdgroup scratch for store-back (no global barrier needed)
    threadgroup float sg_scratch[8 * 64];  // 8 simdgroups × 64 floats = 2048 bytes

    const uint row0 = tgid.y * BM;
    const uint col0 = tgid.x * BN;
    const uint tid  = simd_idx * 32 + simd_lane;  // 0..255

    // 8 simdgroups: 2×4 layout covering 64×64
    const uint sg_m = simd_idx / 4;  // 0..1, each 32 M-rows
    const uint sg_n = simd_idx % 4;  // 0..3, each 16 N-cols

    // Float accumulators: half×half→float — native hw throughput, prevents overflow
    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const bool is_1x1 = (p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1
                          && p.pad_h == 0 && p.pad_w == 0);

    // Precompute spatial decode once for tile rows
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

    if (is_1x1) {
        const uint tiles = (p.k + BK - 1) / BK;
        for (uint t = 0; t < tiles; t++) {
            const uint k0 = t * BK;
            // Load A: 64×16 = 1024 elements, 256 threads → 4 each
            for (uint i = 0; i < 4; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 4;   // / 16
                const uint c = idx & 15;   // % 16
                const uint gr = row0 + r;
                const uint gk = k0 + c;
                sa[r * SA_STRIDE + c] = (gr < p.m && gk < p.k)
                    ? input[gr * p.ic + gk] : half(0.0h);
            }
            // Load B: 16×64 = 1024 elements, 256 threads → 4 each
            for (uint i = 0; i < 4; i++) {
                const uint idx = tid + i * 256;
                const uint r = idx >> 6;   // / 64
                const uint c = idx & 63;   // % 64
                const uint gr = k0 + r;
                const uint gc = col0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.k && gc < p.n_out)
                    ? weight[gr * p.n_out + gc] : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Simdgroup multiply: half×half→float, BK=16 = 2 K-blocks of 8
            for (uint km = 0; km < 2; km++) {
                simdgroup_matrix<half, 8, 8> b_mat[2];
                for (uint bn = 0; bn < 2; bn++)
                    simdgroup_load(b_mat[bn], &sb[km * 8 * SB_STRIDE + sg_n * 16 + bn * 8], SB_STRIDE);
                for (uint bm = 0; bm < 4; bm++) {
                    simdgroup_matrix<half, 8, 8> a_mat;
                    simdgroup_load(a_mat, &sa[(sg_m * 32 + bm * 8) * SA_STRIDE + km * 8], SA_STRIDE);
                    for (uint bn = 0; bn < 2; bn++)
                        simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // 3×3+ convs: iterate (ky, kx, ic_tile) — zero divmod in inner loop
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
                                val = input[((row_b[r] * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic + ci];
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

                    for (uint km = 0; km < 2; km++) {
                        simdgroup_matrix<half, 8, 8> b_mat[2];
                        for (uint bn = 0; bn < 2; bn++)
                            simdgroup_load(b_mat[bn], &sb[km * 8 * SB_STRIDE + sg_n * 16 + bn * 8], SB_STRIDE);
                        for (uint bm = 0; bm < 4; bm++) {
                            simdgroup_matrix<half, 8, 8> a_mat;
                            simdgroup_load(a_mat, &sa[(sg_m * 32 + bm * 8) * SA_STRIDE + km * 8], SA_STRIDE);
                            for (uint bn = 0; bn < 2; bn++)
                                simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    // Store: float acc → per-simdgroup scratch → bias+act → half global memory.
    // No threadgroup barrier needed: simdgroup threads execute in lock-step.
    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 2; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 16 + bn * 8;

            simdgroup_store(acc[bm][bn], &sg_scratch[simd_idx * 64], 8);

            for (uint e = 0; e < 2; e++) {
                const uint flat = simd_lane * 2 + e;
                const uint lr = flat / 8;
                const uint lc = flat % 8;
                const uint gr = m_off + lr;
                const uint gc = n_off + lc;
                if (gr < p.m && gc < p.n_out) {
                    float val = sg_scratch[simd_idx * 64 + lr * 8 + lc] + bias[gc];
                    if (p.act == 1u) val = max(val, 0.0f);
                    else if (p.act == 2u) val = val / (1.0f + fast::exp(-val));
                    output[gr * p.n_out + gc] = half(val);
                }
            }
        }
    }
}
