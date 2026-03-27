// Winograd F(2×2, 3×3) convolution for Metal — f16 I/O, f32 accumulation.
// Three kernels: input transform, batched GEMM (16 alpha slices), output transform.
// Layout: transformed data in (16, n_tiles, channels) format.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Shared parameter structs ──

struct WinogradParams {
    uint batch;
    uint ih; uint iw; uint ic;
    uint oh; uint ow; uint oc;
    uint pad_h; uint pad_w;
    uint tile_h; uint tile_w;
    uint n_tiles;  // batch * tile_h * tile_w
    uint act;      // 0=none, 1=relu, 2=silu
    uint out_stride;  // output channel stride (default = oc, > oc for concat fusion)
    uint out_offset;  // output channel offset (default = 0)
    uint in_stride;   // input channel stride (default = ic, > ic for split fusion)
    uint in_offset;   // input channel offset (default = 0)
};

struct BatchedGemmParams {
    uint m;     // n_tiles
    uint n;     // oc
    uint k;     // ic
};

// ── Kernel 1: Input Transform ──
// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
// V = B^T * d * B for each 4×4 input tile
// Input: NHWC (half), Output: (16, n_tiles, ic) (half)
// Each thread handles one (tile_idx, channel) pair.

kernel void winograd_input_transform_f16(
    device const half*    input  [[buffer(0)]],
    device half*          output [[buffer(1)]],
    constant WinogradParams& p  [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint c = tid.x;
    const uint tile_idx = tid.y;
    if (c >= p.ic || tile_idx >= p.n_tiles) return;

    // Decode tile_idx → (batch, tile_y, tile_x)
    const uint tw = p.tile_w;
    const uint tiles_per_batch = p.tile_h * tw;
    const uint b = tile_idx / tiles_per_batch;
    const uint rem = tile_idx % tiles_per_batch;
    const uint tile_y = rem / tw;
    const uint tile_x = rem % tw;

    // Top-left of 4×4 input tile
    const int base_y = int(tile_y * 2) - int(p.pad_h);
    const int base_x = int(tile_x * 2) - int(p.pad_w);

    // Load 4×4 tile with zero-padding
    half d[4][4];
    for (uint i = 0; i < 4; i++) {
        const int iy = base_y + int(i);
        const bool iy_ok = (iy >= 0 && uint(iy) < p.ih);
        for (uint j = 0; j < 4; j++) {
            const int ix = base_x + int(j);
            if (iy_ok && ix >= 0 && uint(ix) < p.iw) {
                d[i][j] = input[((b * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + c];
            } else {
                d[i][j] = half(0);
            }
        }
    }

    // B^T * d (column transform)
    half t[4][4];
    for (uint j = 0; j < 4; j++) {
        t[0][j] = d[0][j] - d[2][j];
        t[1][j] = d[1][j] + d[2][j];
        t[2][j] = -d[1][j] + d[2][j];
        t[3][j] = d[1][j] - d[3][j];
    }

    // (B^T * d) * B (row transform) → store to (16, n_tiles, ic)
    const uint stride = p.n_tiles * p.ic;
    const uint base_out = tile_idx * p.ic + c;
    for (uint i = 0; i < 4; i++) {
        half r0 = t[i][0] - t[i][2];
        half r1 = t[i][1] + t[i][2];
        half r2 = -t[i][1] + t[i][2];
        half r3 = t[i][1] - t[i][3];
        output[(i * 4 + 0) * stride + base_out] = r0;
        output[(i * 4 + 1) * stride + base_out] = r1;
        output[(i * 4 + 2) * stride + base_out] = r2;
        output[(i * 4 + 3) * stride + base_out] = r3;
    }
}

// ── Kernel 2: Batched GEMM ──
// For alpha in 0..15: C[alpha] = A[alpha] × B[alpha]
// A: (16, n_tiles, ic), B: (16, ic, oc), C: (16, n_tiles, oc) — all half
// Grid: (ceil(oc/BN), ceil(n_tiles/BM), 16)
// Threads per group: 16×16 = 256

constant uint WG_BM = 64;
constant uint WG_BN = 64;
constant uint WG_BK = 16;
constant uint WG_TM = 4;
constant uint WG_TN = 4;
constant uint WG_SA_STRIDE = WG_BK + 1;  // 17 — bank-conflict avoidance
constant uint WG_SB_STRIDE = WG_BN + 1;  // 65

kernel void winograd_batched_gemm_f16io(
    device const half*       A  [[buffer(0)]],
    device const half*       B  [[buffer(1)]],
    device half*             C  [[buffer(2)]],
    constant BatchedGemmParams& p [[buffer(3)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint  tid_local  [[thread_index_in_threadgroup]]
) {
    threadgroup half sa[WG_BM * WG_SA_STRIDE];   // 64×17
    threadgroup half sb[WG_BK * WG_SB_STRIDE];   // 16×65

    const uint batch_idx = tgid.z;
    const uint row0 = tgid.y * WG_BM;
    const uint col0 = tgid.x * WG_BN;

    const uint tr = tid_local >> 4;   // 0..15 (thread row in 16×16 block)
    const uint tc = tid_local & 15;   // 0..15 (thread col)

    // Batch offsets into A, B, C
    const uint a_off = batch_idx * p.m * p.k;
    const uint b_off = batch_idx * p.k * p.n;
    const uint c_off = batch_idx * p.m * p.n;

    // Accumulators: TM×TN = 4×4 = 16 values per thread (f32 to prevent overflow)
    float4 acc[WG_TM];
    for (uint i = 0; i < WG_TM; i++) acc[i] = float4(0);

    const uint tiles = (p.k + WG_BK - 1) / WG_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * WG_BK;

        // Load A tile: BM×BK = 64×16 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid_local + i * 256;
            const uint r = idx >> 4;   // / 16
            const uint c = idx & 15;   // % 16
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * WG_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? A[a_off + gr * p.k + gk] : half(0);
        }

        // Load B tile: BK×BN = 16×64 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid_local + i * 256;
            const uint r = idx >> 6;   // / 64
            const uint c = idx & 63;   // % 64
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * WG_SB_STRIDE + c] = (gr < p.k && gc < p.n)
                ? B[b_off + gr * p.n + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner FMA loop: BK=16 steps — vectorized B-tile load
        for (uint kk = 0; kk < WG_BK; kk++) {
            half4 b_val_h = *((threadgroup const half4*)&sb[kk * WG_SB_STRIDE + tc * WG_TN]);
            float4 b_val = float4(b_val_h);
            float a0 = float(sa[(tr * WG_TM    ) * WG_SA_STRIDE + kk]);
            float a1 = float(sa[(tr * WG_TM + 1) * WG_SA_STRIDE + kk]);
            float a2 = float(sa[(tr * WG_TM + 2) * WG_SA_STRIDE + kk]);
            float a3 = float(sa[(tr * WG_TM + 3) * WG_SA_STRIDE + kk]);
            acc[0] = fma(float4(a0), b_val, acc[0]);
            acc[1] = fma(float4(a1), b_val, acc[1]);
            acc[2] = fma(float4(a2), b_val, acc[2]);
            acc[3] = fma(float4(a3), b_val, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results — clamp to f16 range to prevent inf in intermediate buffers
    for (uint tm = 0; tm < WG_TM; tm++) {
        const uint r = row0 + tr * WG_TM + tm;
        const uint c = col0 + tc * WG_TN;
        if (r < p.m) {
            half4 out_val = half4(clamp(acc[tm], float4(-65504.0f), float4(65504.0f)));
            if (c + 3 < p.n) {
                *((device half4*)(&C[c_off + r * p.n + c])) = out_val;
            } else {
                for (uint j = 0; j < WG_TN && c + j < p.n; j++)
                    C[c_off + r * p.n + c + j] = out_val[j];
            }
        }
    }
}

// ── Kernel 2b: Batched GEMM (simdgroup variant) ──
// Same as above but uses simdgroup_matrix_multiply_accumulate.
// No staging buffer needed (no bias/act), so shared memory = basic kernel.
// 8 simdgroups × 32 threads = 256 threads per threadgroup.
// Each simdgroup: 4×2 grid of 8×8 = 32×16 output sub-tile.

kernel void winograd_batched_gemm_simd_f16io(
    device const half*       A  [[buffer(0)]],
    device const half*       B  [[buffer(1)]],
    device half*             C  [[buffer(2)]],
    constant BatchedGemmParams& p [[buffer(3)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_idx   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[WG_BM * WG_SA_STRIDE];   // 64×17
    threadgroup half sb[WG_BK * WG_SB_STRIDE];   // 16×65

    const uint batch_idx = tgid.z;
    const uint row0 = tgid.y * WG_BM;
    const uint col0 = tgid.x * WG_BN;
    const uint tid  = simd_idx * 32 + simd_lane;

    // 8 simdgroups: 2×4 layout covering 64×64
    const uint sg_m = simd_idx / 4;  // 0..1
    const uint sg_n = simd_idx % 4;  // 0..3

    // Float accumulators: half×half→float prevents overflow, same hardware throughput
    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint a_off = batch_idx * p.m * p.k;
    const uint b_off = batch_idx * p.k * p.n;
    const uint c_off = batch_idx * p.m * p.n;

    const uint tiles = (p.k + WG_BK - 1) / WG_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * WG_BK;

        // Load A tile: BM×BK = 64×16 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;
            const uint c = idx & 15;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * WG_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? A[a_off + gr * p.k + gk] : half(0);
        }

        // Load B tile: BK×BN = 16×64 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * WG_SB_STRIDE + c] = (gr < p.k && gc < p.n)
                ? B[b_off + gr * p.n + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Simdgroup multiply: half×half→float accumulation, BK=16 = 2 K-blocks of 8
        for (uint km = 0; km < 2; km++) {
            simdgroup_matrix<half, 8, 8> b_mat[2];
            for (uint bn = 0; bn < 2; bn++)
                simdgroup_load(b_mat[bn], &sb[km * 8 * WG_SB_STRIDE + sg_n * 16 + bn * 8], WG_SB_STRIDE);
            for (uint bm = 0; bm < 4; bm++) {
                simdgroup_matrix<half, 8, 8> a_mat;
                simdgroup_load(a_mat, &sa[(sg_m * 32 + bm * 8) * WG_SA_STRIDE + km * 8], WG_SA_STRIDE);
                for (uint bn = 0; bn < 2; bn++)
                    simdgroup_multiply_accumulate(acc[bm][bn], a_mat, b_mat[bn], acc[bm][bn]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: float acc → threadgroup scratch → clamp → half device memory.
    // Each simdgroup has its own scratch region; simdgroup threads execute in
    // lock-step, so no threadgroup barriers are needed between store and load.
    threadgroup float sg_scratch[8 * 64];  // 8 simdgroups × 64 floats = 2048 bytes

    for (uint bm = 0; bm < 4; bm++) {
        for (uint bn = 0; bn < 2; bn++) {
            const uint m_off = row0 + sg_m * 32 + bm * 8;
            const uint n_off = col0 + sg_n * 16 + bn * 8;

            simdgroup_store(acc[bm][bn], &sg_scratch[simd_idx * 64], 8);
            // No barrier needed: simdgroup threads are lock-step, so all stores
            // complete before any thread proceeds to the reads below.

            if (m_off < p.m && n_off < p.n) {
                for (uint e = 0; e < 2; e++) {
                    const uint flat = simd_lane * 2 + e;
                    const uint lr = flat >> 3;
                    const uint lc = flat & 7;
                    const uint gr = m_off + lr;
                    const uint gc = n_off + lc;
                    if (gr < p.m && gc < p.n) {
                        float val = clamp(sg_scratch[simd_idx * 64 + lr * 8 + lc],
                                          -65504.0f, 65504.0f);
                        C[c_off + gr * p.n + gc] = half(val);
                    }
                }
            }
            // No barrier needed: next simdgroup_store is a cooperative op that
            // implicitly synchronizes all threads before overwriting scratch.
        }
    }
}

// ── Kernel 3: Output Transform + Bias + Activation ──
// A^T = [[1,1,1,0],[0,1,-1,-1]]
// Y = A^T * M * A → 2×2 output tile
// Input: (16, n_tiles, oc) half from GEMM. Output: NHWC half.

kernel void winograd_output_transform_f16(
    device const half*    gemm_out [[buffer(0)]],
    device const float*   bias     [[buffer(1)]],
    device half*          output   [[buffer(2)]],
    constant WinogradParams& p    [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint oc = tid.x;
    const uint tile_idx = tid.y;
    if (oc >= p.oc || tile_idx >= p.n_tiles) return;

    // Decode tile_idx
    const uint tw = p.tile_w;
    const uint tiles_per_batch = p.tile_h * tw;
    const uint b = tile_idx / tiles_per_batch;
    const uint rem = tile_idx % tiles_per_batch;
    const uint tile_y = rem / tw;
    const uint tile_x = rem % tw;

    // Gather 4×4 values from GEMM output — read as float to avoid half overflow
    const uint stride = p.n_tiles * p.oc;
    const uint base = tile_idx * p.oc + oc;
    float m[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            m[i][j] = float(gemm_out[(i * 4 + j) * stride + base]);

    // A^T * M (column transform) → 2×4 in float
    float s[2][4];
    for (uint j = 0; j < 4; j++) {
        s[0][j] = m[0][j] + m[1][j] + m[2][j];
        s[1][j] = m[1][j] - m[2][j] - m[3][j];
    }

    // (A^T * M) * A → 2×2 + bias + activation → write NHWC
    const float bias_val = bias[oc];
    const uint act = p.act;
    for (uint dy = 0; dy < 2; dy++) {
        const uint oy = tile_y * 2 + dy;
        if (oy >= p.oh) continue;
        for (uint dx = 0; dx < 2; dx++) {
            const uint ox = tile_x * 2 + dx;
            if (ox >= p.ow) continue;
            float val;
            if (dx == 0)
                val = s[dy][0] + s[dy][1] + s[dy][2] + bias_val;
            else
                val = s[dy][1] - s[dy][2] - s[dy][3] + bias_val;
            if (act == 1u) val = max(val, 0.0f);
            else if (act == 2u) val = val / (1.0f + fast::exp(-val));
            output[((b * p.oh + oy) * p.ow + ox) * p.oc + oc] = half(val);
        }
    }
}

// ============================================================================
// Winograd F(4×4, 3×3) convolution kernels
// Input tile: 6×6, output tile: 4×4, 36 alpha slices.
// Reuses WinogradParams (Rust side sets tile_h=ceil(oh/4), tile_w=ceil(ow/4))
// and BatchedGemmParams.
// ============================================================================

// ── Kernel 1 (F4×4): Input Transform ──
// B^T (6×6):
//  [4,  0, -5,  0,  1,  0]
//  [0, -4, -4,  1,  1,  0]
//  [0,  4, -4, -1,  1,  0]
//  [0, -2, -1,  2,  1,  0]
//  [0,  2, -1, -2,  1,  0]
//  [0,  4,  0, -5,  0,  1]
// V = B^T * d * B   (B = transpose of B^T)
// Input: NHWC (half), Output: (36, n_tiles, ic) (half)
// Each thread handles one (tile_idx, channel) pair.

kernel void winograd4x4_input_transform_f16(
    device const half*    input  [[buffer(0)]],
    device half*          output [[buffer(1)]],
    constant WinogradParams& p  [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint c = tid.x;
    const uint tile_idx = tid.y;
    if (c >= p.ic || tile_idx >= p.n_tiles) return;

    // Decode tile_idx → (batch, tile_y, tile_x)
    const uint tw = p.tile_w;
    const uint tiles_per_batch = p.tile_h * tw;
    const uint b = tile_idx / tiles_per_batch;
    const uint rem = tile_idx % tiles_per_batch;
    const uint tile_y = rem / tw;
    const uint tile_x = rem % tw;

    // Top-left of 6×6 input tile (stride = 4 for F(4,3))
    const int base_y = int(tile_y * 4) - int(p.pad_h);
    const int base_x = int(tile_x * 4) - int(p.pad_w);

    // Load 6×6 tile with zero-padding
    half d[6][6];
    for (uint i = 0; i < 6; i++) {
        const int iy = base_y + int(i);
        const bool iy_ok = (iy >= 0 && uint(iy) < p.ih);
        for (uint j = 0; j < 6; j++) {
            const int ix = base_x + int(j);
            if (iy_ok && ix >= 0 && uint(ix) < p.iw) {
                d[i][j] = input[((b * p.ih + uint(iy)) * p.iw + uint(ix)) * p.in_stride + p.in_offset + c];
            } else {
                d[i][j] = half(0);
            }
        }
    }

    // B^T * d (column transform) — use float to avoid half overflow with coefficients up to 5
    float t[6][6];
    for (uint j = 0; j < 6; j++) {
        float d0 = float(d[0][j]), d1 = float(d[1][j]), d2 = float(d[2][j]);
        float d3 = float(d[3][j]), d4 = float(d[4][j]), d5 = float(d[5][j]);
        t[0][j] = 4.0f * d0 - 5.0f * d2 + d4;
        t[1][j] = -4.0f * d1 - 4.0f * d2 + d3 + d4;
        t[2][j] = 4.0f * d1 - 4.0f * d2 - d3 + d4;
        t[3][j] = -2.0f * d1 - d2 + 2.0f * d3 + d4;
        t[4][j] = 2.0f * d1 - d2 - 2.0f * d3 + d4;
        t[5][j] = 4.0f * d1 - 5.0f * d3 + d5;
    }

    // (B^T * d) * B (row transform) → clamp and store as half
    const uint stride = p.n_tiles * p.ic;
    const uint base_out = tile_idx * p.ic + c;
    for (uint i = 0; i < 6; i++) {
        float t0 = t[i][0], t1 = t[i][1], t2 = t[i][2];
        float t3 = t[i][3], t4 = t[i][4], t5 = t[i][5];
        output[(i * 6 + 0) * stride + base_out] = half(clamp(4.0f * t0 - 5.0f * t2 + t4, -65504.0f, 65504.0f));
        output[(i * 6 + 1) * stride + base_out] = half(clamp(-4.0f * t1 - 4.0f * t2 + t3 + t4, -65504.0f, 65504.0f));
        output[(i * 6 + 2) * stride + base_out] = half(clamp(4.0f * t1 - 4.0f * t2 - t3 + t4, -65504.0f, 65504.0f));
        output[(i * 6 + 3) * stride + base_out] = half(clamp(-2.0f * t1 - t2 + 2.0f * t3 + t4, -65504.0f, 65504.0f));
        output[(i * 6 + 4) * stride + base_out] = half(clamp(2.0f * t1 - t2 - 2.0f * t3 + t4, -65504.0f, 65504.0f));
        output[(i * 6 + 5) * stride + base_out] = half(clamp(4.0f * t1 - 5.0f * t3 + t5, -65504.0f, 65504.0f));
    }
}

// ── Kernel 2 (F4×4): Batched GEMM ──
// For alpha in 0..35: C[alpha] = A[alpha] × B[alpha]
// A: (36, n_tiles, ic), B: (36, ic, oc), C: (36, n_tiles, oc) — all half
// Grid: (ceil(oc/BN), ceil(n_tiles/BM), 36)
// Threads per group: 16×16 = 256
// Same tile sizes as F(2,3) GEMM: BM=64, BN=64, BK=16, TM=4, TN=4

kernel void winograd4x4_batched_gemm_f16io(
    device const half*       A  [[buffer(0)]],
    device const half*       B  [[buffer(1)]],
    device half*             C  [[buffer(2)]],
    constant BatchedGemmParams& p [[buffer(3)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint  tid_local  [[thread_index_in_threadgroup]]
) {
    threadgroup half sa[WG_BM * WG_SA_STRIDE];   // 64×17
    threadgroup half sb[WG_BK * WG_SB_STRIDE];   // 16×65

    const uint batch_idx = tgid.z;   // 0..35
    const uint row0 = tgid.y * WG_BM;
    const uint col0 = tgid.x * WG_BN;

    const uint tr = tid_local >> 4;   // 0..15 (thread row in 16×16 block)
    const uint tc = tid_local & 15;   // 0..15 (thread col)

    // Batch offsets into A, B, C
    const uint a_off = batch_idx * p.m * p.k;
    const uint b_off = batch_idx * p.k * p.n;
    const uint c_off = batch_idx * p.m * p.n;

    // Accumulators: TM×TN = 4×4 = 16 values per thread (f32 to prevent overflow)
    float4 acc[WG_TM];
    for (uint i = 0; i < WG_TM; i++) acc[i] = float4(0);

    const uint tiles = (p.k + WG_BK - 1) / WG_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * WG_BK;

        // Load A tile: BM×BK = 64×16 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid_local + i * 256;
            const uint r = idx >> 4;   // / 16
            const uint c = idx & 15;   // % 16
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * WG_SA_STRIDE + c] = (gr < p.m && gk < p.k)
                ? A[a_off + gr * p.k + gk] : half(0);
        }

        // Load B tile: BK×BN = 16×64 = 1024 elems, 256 threads → 4 each
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid_local + i * 256;
            const uint r = idx >> 6;   // / 64
            const uint c = idx & 63;   // % 64
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * WG_SB_STRIDE + c] = (gr < p.k && gc < p.n)
                ? B[b_off + gr * p.n + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner FMA loop: BK=16 steps — vectorized B-tile load
        for (uint kk = 0; kk < WG_BK; kk++) {
            half4 b_val_h = *((threadgroup const half4*)&sb[kk * WG_SB_STRIDE + tc * WG_TN]);
            float4 b_val = float4(b_val_h);
            float a0 = float(sa[(tr * WG_TM    ) * WG_SA_STRIDE + kk]);
            float a1 = float(sa[(tr * WG_TM + 1) * WG_SA_STRIDE + kk]);
            float a2 = float(sa[(tr * WG_TM + 2) * WG_SA_STRIDE + kk]);
            float a3 = float(sa[(tr * WG_TM + 3) * WG_SA_STRIDE + kk]);
            acc[0] = fma(float4(a0), b_val, acc[0]);
            acc[1] = fma(float4(a1), b_val, acc[1]);
            acc[2] = fma(float4(a2), b_val, acc[2]);
            acc[3] = fma(float4(a3), b_val, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results — clamp to f16 range to prevent inf in intermediate buffers
    for (uint tm = 0; tm < WG_TM; tm++) {
        const uint r = row0 + tr * WG_TM + tm;
        const uint c = col0 + tc * WG_TN;
        if (r < p.m) {
            half4 out_val = half4(clamp(acc[tm], float4(-65504.0f), float4(65504.0f)));
            if (c + 3 < p.n) {
                *((device half4*)(&C[c_off + r * p.n + c])) = out_val;
            } else {
                for (uint j = 0; j < WG_TN && c + j < p.n; j++)
                    C[c_off + r * p.n + c + j] = out_val[j];
            }
        }
    }
}

// ── Kernel 3 (F4×4): Output Transform + Bias + Activation ──
// A^T (4×6):
//  [1,  1,  1,  1,  1,  0]
//  [0,  1, -1,  2, -2,  0]
//  [0,  1,  1,  4,  4,  0]
//  [0,  1, -1,  8, -8,  1]
// Y = A^T * M * A → 4×4 output tile
// Input: (36, n_tiles, oc) half from GEMM. Output: NHWC half.

kernel void winograd4x4_output_transform_f16(
    device const half*    gemm_out [[buffer(0)]],
    device const float*   bias     [[buffer(1)]],
    device half*          output   [[buffer(2)]],
    constant WinogradParams& p    [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint oc = tid.x;
    const uint tile_idx = tid.y;
    if (oc >= p.oc || tile_idx >= p.n_tiles) return;

    // Decode tile_idx
    const uint tw = p.tile_w;
    const uint tiles_per_batch = p.tile_h * tw;
    const uint b = tile_idx / tiles_per_batch;
    const uint rem = tile_idx % tiles_per_batch;
    const uint tile_y = rem / tw;
    const uint tile_x = rem % tw;

    // Gather 6×6 values from GEMM output (36 alpha slices)
    const uint stride = p.n_tiles * p.oc;
    const uint base = tile_idx * p.oc + oc;
    half m[6][6];
    for (uint i = 0; i < 6; i++)
        for (uint j = 0; j < 6; j++)
            m[i][j] = gemm_out[(i * 6 + j) * stride + base];

    // A^T * M (column transform) — float to avoid half overflow with coefficients up to 8
    float s[4][6];
    for (uint j = 0; j < 6; j++) {
        float m0 = float(m[0][j]), m1 = float(m[1][j]), m2 = float(m[2][j]);
        float m3 = float(m[3][j]), m4 = float(m[4][j]), m5 = float(m[5][j]);
        s[0][j] = m0 + m1 + m2 + m3 + m4;
        s[1][j] = m1 - m2 + 2.0f * m3 - 2.0f * m4;
        s[2][j] = m1 + m2 + 4.0f * m3 + 4.0f * m4;
        s[3][j] = m1 - m2 + 8.0f * m3 - 8.0f * m4 + m5;
    }

    // (A^T * M) * A (row transform) + bias + activation → write NHWC
    const float bias_val = bias[oc];
    const uint act = p.act;
    for (uint dy = 0; dy < 4; dy++) {
        const uint oy = tile_y * 4 + dy;
        if (oy >= p.oh) continue;
        float s0 = s[dy][0], s1 = s[dy][1], s2 = s[dy][2];
        float s3 = s[dy][3], s4 = s[dy][4], s5 = s[dy][5];
        float y[4];
        y[0] = s0 + s1 + s2 + s3 + s4;
        y[1] = s1 - s2 + 2.0f * s3 - 2.0f * s4;
        y[2] = s1 + s2 + 4.0f * s3 + 4.0f * s4;
        y[3] = s1 - s2 + 8.0f * s3 - 8.0f * s4 + s5;
        for (uint dx = 0; dx < 4; dx++) {
            const uint ox = tile_x * 4 + dx;
            if (ox >= p.ow) continue;
            float val = y[dx] + bias_val;
            if (act == 1u) val = max(val, 0.0f);
            else if (act == 2u) val = val / (1.0f + fast::exp(-val));
            // Clamp to half range to prevent inf→NaN propagation through subsequent layers
            val = clamp(val, -65504.0f, 65504.0f);
            output[((b * p.oh + oy) * p.ow + ox) * p.out_stride + p.out_offset + oc] = half(val);
        }
    }
}

// ── Winograd F(4,3) output transform with fused residual add ──
// Same as above but adds residual[..] to output. Used for Conv→Add residual
// connections: eliminates separate Binary(Add) dispatch + its memory traffic.
kernel void winograd4x4_output_transform_residual_f16(
    device const half*    gemm_out  [[buffer(0)]],
    device const float*   bias      [[buffer(1)]],
    device half*          output    [[buffer(2)]],
    constant WinogradParams& p     [[buffer(3)]],
    device const half*    residual  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint oc = tid.x;
    const uint tile_idx = tid.y;
    if (oc >= p.oc || tile_idx >= p.n_tiles) return;

    const uint tw = p.tile_w;
    const uint tiles_per_batch = p.tile_h * tw;
    const uint b = tile_idx / tiles_per_batch;
    const uint rem = tile_idx % tiles_per_batch;
    const uint tile_y = rem / tw;
    const uint tile_x = rem % tw;

    const uint stride = p.n_tiles * p.oc;
    const uint base = tile_idx * p.oc + oc;
    half m[6][6];
    for (uint i = 0; i < 6; i++)
        for (uint j = 0; j < 6; j++)
            m[i][j] = gemm_out[(i * 6 + j) * stride + base];

    float s[4][6];
    for (uint j = 0; j < 6; j++) {
        float m0 = float(m[0][j]), m1 = float(m[1][j]), m2 = float(m[2][j]);
        float m3 = float(m[3][j]), m4 = float(m[4][j]), m5 = float(m[5][j]);
        s[0][j] = m0 + m1 + m2 + m3 + m4;
        s[1][j] = m1 - m2 + 2.0f * m3 - 2.0f * m4;
        s[2][j] = m1 + m2 + 4.0f * m3 + 4.0f * m4;
        s[3][j] = m1 - m2 + 8.0f * m3 - 8.0f * m4 + m5;
    }

    const float bias_val = bias[oc];
    const uint act = p.act;
    for (uint dy = 0; dy < 4; dy++) {
        const uint oy = tile_y * 4 + dy;
        if (oy >= p.oh) continue;
        float s0 = s[dy][0], s1 = s[dy][1], s2 = s[dy][2];
        float s3 = s[dy][3], s4 = s[dy][4], s5 = s[dy][5];
        float y[4];
        y[0] = s0 + s1 + s2 + s3 + s4;
        y[1] = s1 - s2 + 2.0f * s3 - 2.0f * s4;
        y[2] = s1 + s2 + 4.0f * s3 + 4.0f * s4;
        y[3] = s1 - s2 + 8.0f * s3 - 8.0f * s4 + s5;
        for (uint dx = 0; dx < 4; dx++) {
            const uint ox = tile_x * 4 + dx;
            if (ox >= p.ow) continue;
            const uint spatial_idx = (b * p.oh + oy) * p.ow + ox;
            float val = y[dx] + bias_val;
            if (act == 1u) val = max(val, 0.0f);
            else if (act == 2u) val = val / (1.0f + fast::exp(-val));
            // Add residual AFTER activation (matches Conv+Act+Add pattern in YOLO)
            // Residual uses oc stride (its own layout), output uses out_stride
            val += float(residual[spatial_idx * p.oc + oc]);
            val = clamp(val, -65504.0f, 65504.0f);
            output[spatial_idx * p.out_stride + p.out_offset + oc] = half(val);
        }
    }
}
