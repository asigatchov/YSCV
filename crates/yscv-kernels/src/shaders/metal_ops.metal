// Unified Metal compute shaders for ONNX inference.
// All ops needed for YOLO: elementwise, unary, concat, split, resize, maxpool, etc.

#include <metal_stdlib>
using namespace metal;

// ── Elementwise binary ops ──────────────────────────────────────
// op: 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min
struct BinaryParams { uint n; uint op; };

kernel void binary_elementwise(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant BinaryParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float va = a[tid], vb = b[tid];
    float r;
    switch (p.op) {
        case 0: r = va + vb; break;
        case 1: r = va - vb; break;
        case 2: r = va * vb; break;
        case 3: r = va / vb; break;
        case 4: r = max(va, vb); break;
        case 5: r = min(va, vb); break;
        default: r = va + vb;
    }
    out[tid] = r;
}

// Broadcast binary: b is broadcast over a's last dim
struct BroadcastBinaryParams { uint n; uint broadcast_dim; uint op; };

kernel void broadcast_binary(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant BroadcastBinaryParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float va = a[tid], vb = b[tid % p.broadcast_dim];
    float r;
    switch (p.op) {
        case 0: r = va + vb; break;
        case 1: r = va - vb; break;
        case 2: r = va * vb; break;
        case 3: r = va / vb; break;
        default: r = va + vb;
    }
    out[tid] = r;
}

// ── Unary ops ───────────────────────────────────────────────────
// op: 0=relu, 1=sigmoid, 2=silu, 3=neg, 4=exp, 5=sqrt, 6=tanh, 7=abs, 8=floor, 9=ceil
struct UnaryParams { uint n; uint op; };

kernel void unary_op(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant UnaryParams& p   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float v = input[tid];
    switch (p.op) {
        case 0: v = max(v, 0.0f); break;                    // relu
        case 1: v = 1.0f / (1.0f + fast::exp(-v)); break; // sigmoid
        case 2: v = v / (1.0f + fast::exp(-v)); break; // silu
        case 3: v = -v; break;                               // neg
        case 4: v = exp(v); break;                           // exp
        case 5: v = sqrt(v); break;                          // sqrt
        case 6: v = tanh(v); break;                          // tanh
        case 7: v = abs(v); break;                           // abs
        case 8: v = floor(v); break;                         // floor
        case 9: v = ceil(v); break;                          // ceil
        default: break;
    }
    out[tid] = v;
}

// SiLU fused: out = input * sigmoid(input)  (common in YOLO)
kernel void silu(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float v = input[tid];
    out[tid] = v / (1.0f + fast::exp(-v));
}

// ── Concat along last dim (channel concat for NHWC) ────────────
struct ConcatParams {
    uint total_elements;  // total output elements
    uint out_c;           // total output channels
    uint n_inputs;        // number of inputs to concat
    uint channels[16];    // per-input channel counts (max 16 inputs)
};

kernel void concat_channels(
    device const float* in0  [[buffer(0)]],
    device const float* in1  [[buffer(1)]],
    device const float* in2  [[buffer(2)]],
    device const float* in3  [[buffer(3)]],
    device float* out        [[buffer(4)]],
    constant ConcatParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total_elements) return;
    uint spatial = tid / p.out_c;
    uint c = tid % p.out_c;

    // Find which input this channel belongs to
    uint offset = 0;
    if (c < p.channels[0]) {
        out[tid] = in0[spatial * p.channels[0] + c];
    } else {
        offset += p.channels[0];
        if (p.n_inputs > 1 && c < offset + p.channels[1]) {
            out[tid] = in1[spatial * p.channels[1] + (c - offset)];
        } else {
            offset += p.channels[1];
            if (p.n_inputs > 2 && c < offset + p.channels[2]) {
                out[tid] = in2[spatial * p.channels[2] + (c - offset)];
            } else {
                offset += p.channels[2];
                if (p.n_inputs > 3 && c < offset + p.channels[3]) {
                    out[tid] = in3[spatial * p.channels[3] + (c - offset)];
                } else {
                    out[tid] = 0.0f;
                }
            }
        }
    }
}

// ── Split along last dim ────────────────────────────────────────
struct SplitParams {
    uint spatial;       // N*H*W
    uint in_c;          // input channels
    uint out_c;         // this output's channels
    uint offset_c;      // channel offset in input
};

kernel void split_channels(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant SplitParams& p   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.spatial * p.out_c;
    if (tid >= n) return;
    uint s = tid / p.out_c;
    uint c = tid % p.out_c;
    out[tid] = input[s * p.in_c + p.offset_c + c];
}

// ── Resize nearest (2D, NHWC) ───────────────────────────────────
struct ResizeParams {
    uint batch; uint ih; uint iw; uint ic;
    uint oh; uint ow;
    float scale_h; float scale_w;
};

kernel void resize_nearest(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant ResizeParams& p  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.batch * p.oh * p.ow * p.ic;
    if (tid >= n) return;
    uint c  = tid % p.ic;
    uint ow_idx = (tid / p.ic) % p.ow;
    uint oh_idx = (tid / (p.ic * p.ow)) % p.oh;
    uint b  = tid / (p.ic * p.ow * p.oh);

    uint iy = uint(float(oh_idx) / p.scale_h);
    uint ix = uint(float(ow_idx) / p.scale_w);
    iy = min(iy, p.ih - 1);
    ix = min(ix, p.iw - 1);

    out[tid] = input[((b * p.ih + iy) * p.iw + ix) * p.ic + c];
}

// ── MaxPool 2D (NHWC) ──────────────────────────────────────────
struct PoolParams {
    uint batch; uint ih; uint iw; uint ic;
    uint oh; uint ow;
    uint kh; uint kw; uint sh; uint sw;
    uint pad_h; uint pad_w;
};

kernel void maxpool2d(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant PoolParams& p    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.batch * p.oh * p.ow * p.ic;
    if (tid >= n) return;
    uint c  = tid % p.ic;
    uint ow_idx = (tid / p.ic) % p.ow;
    uint oh_idx = (tid / (p.ic * p.ow)) % p.oh;
    uint b  = tid / (p.ic * p.ow * p.oh);

    float max_val = -INFINITY;
    for (uint ky = 0; ky < p.kh; ky++) {
        for (uint kx = 0; kx < p.kw; kx++) {
            int iy = int(oh_idx * p.sh + ky) - int(p.pad_h);
            int ix = int(ow_idx * p.sw + kx) - int(p.pad_w);
            if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                float v = input[((b * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic + c];
                max_val = max(max_val, v);
            }
        }
    }
    out[tid] = max_val;
}

// ── Softmax (last dim) — threadgroup-parallel for large dims ────
// One threadgroup per row, 256 threads cooperate via shared memory reduction.
struct SoftmaxParams { uint outer; uint dim; };

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant SoftmaxParams& p [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= p.outer) return;
    const uint base = tgid * p.dim;

    threadgroup float shared[256];

    // Pass 1: find max (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = lid; i < p.dim; i += tg_size) {
        local_max = max(local_max, input[base + i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: compute exp and sum (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += tg_size) {
        float e = exp(input[base + i] - max_val);
        out[base + i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: normalize
    for (uint i = lid; i < p.dim; i += tg_size) {
        out[base + i] *= inv_sum;
    }
}

// ── Transpose 2D ────────────────────────────────────────────────
struct Transpose2DParams { uint rows; uint cols; };

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant Transpose2DParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.rows * p.cols;
    if (tid >= n) return;
    uint r = tid / p.cols;
    uint c = tid % p.cols;
    out[c * p.rows + r] = input[tid];
}

// ── Slice (simple 1D flat copy with offset) ─────────────────────
struct SliceParams { uint n; uint src_offset; };

kernel void slice_copy(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant SliceParams& p   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    out[tid] = input[p.src_offset + tid];
}

// ── Permute [0,2,1,3] — swap dim1 and dim2 of 4D tensor ─────
// Input [D0, D1, D2, D3] → Output [D0, D2, D1, D3]
struct Permute0213Params { uint d0; uint d1; uint d2; uint d3; };

kernel void permute_0213(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant Permute0213Params& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.d0 * p.d1 * p.d2 * p.d3;
    if (tid >= total) return;
    // Decompose tid in output layout [D0, D2, D1, D3]
    uint o3    = tid % p.d3;
    uint o_d1  = (tid / p.d3) % p.d1;
    uint o_d2  = (tid / (p.d3 * p.d1)) % p.d2;
    uint o_d0  = tid / (p.d3 * p.d1 * p.d2);
    // Source: input[o_d0, o_d1, o_d2, o3] in layout [D0, D1, D2, D3]
    uint src = ((o_d0 * p.d1 + o_d1) * p.d2 + o_d2) * p.d3 + o3;
    out[tid] = input[src];
}

// ── Permute NHWC ↔ NCHW (4D) ────────────────────────────────
struct PermuteParams { uint n; uint h; uint w; uint c; };

kernel void permute_nhwc_to_nchw(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant PermuteParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.n * p.h * p.w * p.c;
    if (tid >= total) return;
    // NHWC input: index = ((n*H+h)*W+w)*C+c
    uint cc = tid % p.c;
    uint ww = (tid / p.c) % p.w;
    uint hh = (tid / (p.c * p.w)) % p.h;
    uint nn = tid / (p.c * p.w * p.h);
    // NCHW output: index = ((n*C+c)*H+h)*W+w
    out[((nn * p.c + cc) * p.h + hh) * p.w + ww] = input[tid];
}

kernel void permute_nchw_to_nhwc(
    device const float* input [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant PermuteParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.n * p.h * p.w * p.c;
    if (tid >= total) return;
    // NCHW input: index = ((n*C+c)*H+h)*W+w
    uint ww = tid % p.w;
    uint hh = (tid / p.w) % p.h;
    uint cc = (tid / (p.w * p.h)) % p.c;
    uint nn = tid / (p.w * p.h * p.c);
    // NHWC output: index = ((n*H+h)*W+w)*C+c
    out[((nn * p.h + hh) * p.w + ww) * p.c + cc] = input[tid];
}

// ── Matmul (tiled, f16 accumulators) ────────────────────────────
// BM=64, BN=64, BK=16, TM=4, TN=4 — 16×16=256 threads, 16 accumulators/thread.
// Matching conv_gemm_basic tile sizes for proven occupancy (~100%).

struct MatmulParams { uint m; uint n; uint k; };

constant uint MM_BM = 64;
constant uint MM_BN = 64;
constant uint MM_BK = 16;
constant uint MM_TM = 4;
constant uint MM_TN = 4;
constant uint MM_SA_STRIDE = 17;   // BK + 1 (bank-conflict padding)
constant uint MM_SB_STRIDE = 65;   // BN + 1

kernel void matmul(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float* out       [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 gid  [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[MM_BM * MM_SA_STRIDE];   // 64 × 17 = 1088
    threadgroup half sb[MM_BK * MM_SB_STRIDE];   // 16 × 65 = 1040

    const uint tx = lid.x;  // 0..15
    const uint ty = lid.y;  // 0..15
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * MM_BM;
    const uint col0 = gid.x * MM_BN;

    // 4×4 = 16 accumulators per thread
    half4 acc0 = half4(0.0h);
    half4 acc1 = half4(0.0h);
    half4 acc2 = half4(0.0h);
    half4 acc3 = half4(0.0h);

    const uint tiles = (p.k + MM_BK - 1) / MM_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * MM_BK;

        // Load A: BM × BK = 1024 elements, 256 threads → 4 elements/thread
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;   // / 16
            const uint c = idx & 15;   // % 16
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            half val = (gr < p.m && gk < p.k) ? half(a[gr * p.k + gk]) : 0.0h;
            sa[r * MM_SA_STRIDE + c] = val;
        }

        // Load B: BK × BN = 1024 elements, 256 threads → 4 elements/thread
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;   // / 64
            const uint c = idx & 63;   // % 64
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            half val = (gr < p.k && gc < p.n) ? half(b[gr * p.n + gc]) : 0.0h;
            sb[r * MM_SB_STRIDE + c] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner compute — fully unrolled BK=16 steps
        const uint b_base = tx * MM_TN;
        const uint a_base0 = (ty * MM_TM + 0) * MM_SA_STRIDE;
        const uint a_base1 = (ty * MM_TM + 1) * MM_SA_STRIDE;
        const uint a_base2 = (ty * MM_TM + 2) * MM_SA_STRIDE;
        const uint a_base3 = (ty * MM_TM + 3) * MM_SA_STRIDE;

        for (uint kk = 0; kk < MM_BK; kk++) {
            half4 bv = half4(
                sb[kk * MM_SB_STRIDE + b_base],
                sb[kk * MM_SB_STRIDE + b_base + 1],
                sb[kk * MM_SB_STRIDE + b_base + 2],
                sb[kk * MM_SB_STRIDE + b_base + 3]
            );
            acc0 = fma(half4(sa[a_base0 + kk]), bv, acc0);
            acc1 = fma(half4(sa[a_base1 + kk]), bv, acc1);
            acc2 = fma(half4(sa[a_base2 + kk]), bv, acc2);
            acc3 = fma(half4(sa[a_base3 + kk]), bv, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: f16 → f32
    const uint c = col0 + tx * MM_TN;
    if (c + 3 < p.n) {
        half4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < MM_TM; ri++) {
            const uint r = row0 + ty * MM_TM + ri;
            if (r >= p.m) continue;
            float4 v = float4(accs[ri]);
            const uint base = r * p.n + c;
            out[base]     = v.x;
            out[base + 1] = v.y;
            out[base + 2] = v.z;
            out[base + 3] = v.w;
        }
    } else {
        half4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < MM_TM; ri++) {
            const uint r = row0 + ty * MM_TM + ri;
            if (r >= p.m) continue;
            float4 acc_f32 = float4(accs[ri]);
            for (uint ci = 0; ci < MM_TN; ci++) {
                const uint col = c + ci;
                if (col < p.n) {
                    out[r * p.n + col] = acc_f32[ci];
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// F16 I/O kernel variants — all intermediate buffers use half precision
// to halve memory bandwidth. Internal arithmetic stays f32 where needed.
// ═══════════════════════════════════════════════════════════════════════

// ── Cast kernels (graph boundary) ────────────────────────────────────

kernel void cast_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* out          [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    out[tid] = half(input[tid]);
}

kernel void cast_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    out[tid] = float(input[tid]);
}

// ── Fused cast f32→f16 + NCHW→NHWC permutation (input boundary) ──────
struct CastPermuteParams { uint n; uint c; uint h; uint w; };

kernel void cast_f32_to_f16_nchw_to_nhwc(
    device const float* input [[buffer(0)]],
    device half* out          [[buffer(1)]],
    constant CastPermuteParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.n * p.c * p.h * p.w;
    if (tid >= total) return;
    // Decompose tid as NHWC linear index
    uint ci = tid % p.c;
    uint wi = (tid / p.c) % p.w;
    uint hi = (tid / (p.c * p.w)) % p.h;
    uint ni = tid / (p.c * p.w * p.h);
    // Read from NCHW layout
    uint src_idx = ((ni * p.c + ci) * p.h + hi) * p.w + wi;
    out[tid] = half(input[src_idx]);
}

// ── Binary elementwise f16 ───────────────────────────────────────────

kernel void binary_elementwise_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out     [[buffer(2)]],
    constant BinaryParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float va = float(a[tid]), vb = float(b[tid]);
    float r;
    switch (p.op) {
        case 0: r = va + vb; break;
        case 1: r = va - vb; break;
        case 2: r = va * vb; break;
        case 3: r = va / vb; break;
        case 4: r = max(va, vb); break;
        case 5: r = min(va, vb); break;
        default: r = va + vb;
    }
    out[tid] = half(clamp(r, -65504.0f, 65504.0f));
}

// ── Broadcast binary f16 ────────────────────────────────────────────

kernel void broadcast_binary_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out     [[buffer(2)]],
    constant BroadcastBinaryParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float va = float(a[tid]), vb = float(b[tid % p.broadcast_dim]);
    float r;
    switch (p.op) {
        case 0: r = va + vb; break;
        case 1: r = va - vb; break;
        case 2: r = va * vb; break;
        case 3: r = va / vb; break;
        default: r = va + vb;
    }
    out[tid] = half(clamp(r, -65504.0f, 65504.0f));
}

// ── Unary op f16 ────────────────────────────────────────────────────

kernel void unary_op_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant UnaryParams& p  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float v = float(input[tid]);
    switch (p.op) {
        case 0: v = max(v, 0.0f); break;
        case 1: v = 1.0f / (1.0f + fast::exp(-v)); break;
        case 2: v = v / (1.0f + fast::exp(-v)); break;
        case 3: v = -v; break;
        case 4: v = exp(v); break;
        case 5: v = sqrt(v); break;
        case 6: v = tanh(v); break;
        case 7: v = abs(v); break;
        case 8: v = floor(v); break;
        case 9: v = ceil(v); break;
        default: break;
    }
    out[tid] = half(v);
}

// ── SiLU f16 ────────────────────────────────────────────────────────

kernel void silu_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float v = float(input[tid]);
    out[tid] = half(v / (1.0f + fast::exp(-v)));
}

// ── Concat channels f16 ────────────────────────────────────────────

kernel void concat_channels_f16(
    device const half* in0  [[buffer(0)]],
    device const half* in1  [[buffer(1)]],
    device const half* in2  [[buffer(2)]],
    device const half* in3  [[buffer(3)]],
    device half* out        [[buffer(4)]],
    constant ConcatParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]  // x=channel, y=spatial
) {
    const uint c = gid.x;
    const uint spatial = gid.y;
    const uint n_spatial = p.total_elements / p.out_c;
    if (c >= p.out_c || spatial >= n_spatial) return;

    const uint out_idx = spatial * p.out_c + c;

    if (c < p.channels[0]) {
        out[out_idx] = in0[spatial * p.channels[0] + c];
    } else {
        uint offset = p.channels[0];
        if (p.n_inputs > 1 && c < offset + p.channels[1]) {
            out[out_idx] = in1[spatial * p.channels[1] + (c - offset)];
        } else {
            offset += p.channels[1];
            if (p.n_inputs > 2 && c < offset + p.channels[2]) {
                out[out_idx] = in2[spatial * p.channels[2] + (c - offset)];
            } else {
                offset += p.channels[2];
                if (p.n_inputs > 3 && c < offset + p.channels[3]) {
                    out[out_idx] = in3[spatial * p.channels[3] + (c - offset)];
                } else {
                    out[out_idx] = half(0.0h);
                }
            }
        }
    }
}

// ── Split channels f16 ─────────────────────────────────────────────

kernel void split_channels_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant SplitParams& p  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.spatial * p.out_c;
    if (tid >= n) return;
    uint s = tid / p.out_c;
    uint c = tid % p.out_c;
    out[tid] = input[s * p.in_c + p.offset_c + c];
}

// ── Fused N-way split: read input once, write to up to 3 outputs ───

struct FusedSplitParams {
    uint spatial;       // N*H*W (outer dimensions)
    uint in_c;          // total input channels
    uint n_outputs;     // 2 or 3
    uint split_c0;      // channels for output 0
    uint split_c1;      // channels for output 1
    uint split_c2;      // channels for output 2 (0 if n_outputs < 3)
};

kernel void split_fused_f16(
    device const half* input [[buffer(0)]],
    device half* out0        [[buffer(1)]],
    device half* out1        [[buffer(2)]],
    device half* out2        [[buffer(3)]],
    constant FusedSplitParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.spatial * p.in_c;
    if (tid >= n) return;
    uint s = tid / p.in_c;
    uint c = tid % p.in_c;
    half val = input[tid];
    if (c < p.split_c0) {
        out0[s * p.split_c0 + c] = val;
    } else if (c < p.split_c0 + p.split_c1) {
        out1[s * p.split_c1 + (c - p.split_c0)] = val;
    } else if (p.n_outputs > 2) {
        out2[s * p.split_c2 + (c - p.split_c0 - p.split_c1)] = val;
    }
}

// ── Fused N-way split: vectorized half4, 2D grid (c/4, spatial) ────

kernel void split_fused_f16v4(
    device const half* input [[buffer(0)]],
    device half* out0        [[buffer(1)]],
    device half* out1        [[buffer(2)]],
    device half* out2        [[buffer(3)]],
    constant FusedSplitParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c4 = gid.x;      // channel group index (ic/4)
    const uint s  = gid.y;      // spatial position
    if (c4 * 4 >= p.in_c || s >= p.spatial) return;

    const uint c_base = c4 * 4;
    const uint in_off = s * p.in_c + c_base;

    // Determine which output(s) this half4 belongs to
    // For YOLO split sizes are always multiples of 4, so a half4 never straddles
    // two outputs. Fast path: check which output range c_base falls in.
    if (c_base + 4 <= p.in_c) {
        half4 v = *(device const half4*)(input + in_off);
        if (c_base + 4 <= p.split_c0) {
            // Entirely in output 0
            *(device half4*)(out0 + s * p.split_c0 + c_base) = v;
        } else if (c_base >= p.split_c0 && c_base + 4 <= p.split_c0 + p.split_c1) {
            // Entirely in output 1
            *(device half4*)(out1 + s * p.split_c1 + (c_base - p.split_c0)) = v;
        } else if (c_base >= p.split_c0 + p.split_c1 && p.n_outputs > 2) {
            // Entirely in output 2
            uint off2 = c_base - p.split_c0 - p.split_c1;
            *(device half4*)(out2 + s * p.split_c2 + off2) = v;
        } else {
            // Boundary case: straddles two outputs — scalar fallback
            for (uint i = 0; i < 4; i++) {
                uint c = c_base + i;
                half val = v[i];
                if (c < p.split_c0) {
                    out0[s * p.split_c0 + c] = val;
                } else if (c < p.split_c0 + p.split_c1) {
                    out1[s * p.split_c1 + (c - p.split_c0)] = val;
                } else if (p.n_outputs > 2) {
                    out2[s * p.split_c2 + (c - p.split_c0 - p.split_c1)] = val;
                }
            }
        }
    } else {
        // Tail: fewer than 4 channels remaining
        for (uint i = 0; i < p.in_c - c_base; i++) {
            uint c = c_base + i;
            half val = input[in_off + i];
            if (c < p.split_c0) {
                out0[s * p.split_c0 + c] = val;
            } else if (c < p.split_c0 + p.split_c1) {
                out1[s * p.split_c1 + (c - p.split_c0)] = val;
            } else if (p.n_outputs > 2) {
                out2[s * p.split_c2 + (c - p.split_c0 - p.split_c1)] = val;
            }
        }
    }
}

// ── Resize nearest f16 ─────────────────────────────────────────────

kernel void resize_nearest_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant ResizeParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.batch * p.oh * p.ow * p.ic;
    if (tid >= n) return;
    uint c  = tid % p.ic;
    uint ow_idx = (tid / p.ic) % p.ow;
    uint oh_idx = (tid / (p.ic * p.ow)) % p.oh;
    uint b  = tid / (p.ic * p.ow * p.oh);

    uint iy = uint(float(oh_idx) / p.scale_h);
    uint ix = uint(float(ow_idx) / p.scale_w);
    iy = min(iy, p.ih - 1);
    ix = min(ix, p.iw - 1);

    out[tid] = input[((b * p.ih + iy) * p.iw + ix) * p.ic + c];
}

// ── MaxPool 2D f16 ─────────────────────────────────────────────────

kernel void maxpool2d_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant PoolParams& p   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.batch * p.oh * p.ow * p.ic;
    if (tid >= n) return;
    uint c  = tid % p.ic;
    uint ow_idx = (tid / p.ic) % p.ow;
    uint oh_idx = (tid / (p.ic * p.ow)) % p.oh;
    uint b  = tid / (p.ic * p.ow * p.oh);

    half max_val = half(-INFINITY);
    for (uint ky = 0; ky < p.kh; ky++) {
        for (uint kx = 0; kx < p.kw; kx++) {
            int iy = int(oh_idx * p.sh + ky) - int(p.pad_h);
            int ix = int(ow_idx * p.sw + kx) - int(p.pad_w);
            if (iy >= 0 && uint(iy) < p.ih && ix >= 0 && uint(ix) < p.iw) {
                half v = input[((b * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic + c];
                max_val = max(max_val, v);
            }
        }
    }
    out[tid] = max_val;
}

// ── Softmax f16 (f32 accumulation internally) ──────────────────────

kernel void softmax_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant SoftmaxParams& p [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= p.outer) return;
    const uint base = tgid * p.dim;

    threadgroup float shared[256];

    float local_max = -INFINITY;
    for (uint i = lid; i < p.dim; i += tg_size) {
        local_max = max(local_max, float(input[base + i]));
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += tg_size) {
        float e = exp(float(input[base + i]) - max_val);
        out[base + i] = half(e);
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < p.dim; i += tg_size) {
        out[base + i] = half(float(out[base + i]) * inv_sum);
    }
}

// ── Scalar softmax f16 (dim ≤ ~128) ────────────────────────────────
// One thread per row — no threadgroup reduction, no barriers.
// 10-100× faster than threadgroup softmax for small dim.

kernel void softmax_scalar_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant SoftmaxParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.outer) return;
    const uint base = tid * p.dim;
    const uint dim = p.dim;

    // Pass 1: find max (f32 accumulation)
    float max_val = float(input[base]);
    for (uint i = 1; i < dim; i++) {
        max_val = max(max_val, float(input[base + i]));
    }

    // Pass 2: exp and sum, write exp to output
    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float e = exp(float(input[base + i]) - max_val);
        out[base + i] = half(e);
        sum += e;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < dim; i++) {
        out[base + i] = half(float(out[base + i]) * inv_sum);
    }
}

// ── Transpose 2D f16 ───────────────────────────────────────────────

kernel void transpose_2d_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant Transpose2DParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = p.rows * p.cols;
    if (tid >= n) return;
    uint r = tid / p.cols;
    uint c = tid % p.cols;
    out[c * p.rows + r] = input[tid];
}

// ── Slice copy f16 ─────────────────────────────────────────────────

kernel void slice_copy_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant SliceParams& p  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    out[tid] = input[p.src_offset + tid];
}

// ── Permute [0,2,1,3] f16 ──────────────────────────────────────────

kernel void permute_0213_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant Permute0213Params& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.d0 * p.d1 * p.d2 * p.d3;
    if (tid >= total) return;
    uint o3    = tid % p.d3;
    uint o_d1  = (tid / p.d3) % p.d1;
    uint o_d2  = (tid / (p.d3 * p.d1)) % p.d2;
    uint o_d0  = tid / (p.d3 * p.d1 * p.d2);
    uint src = ((o_d0 * p.d1 + o_d1) * p.d2 + o_d2) * p.d3 + o3;
    out[tid] = input[src];
}

// ── Permute NHWC ↔ NCHW f16 ────────────────────────────────────────

kernel void permute_nhwc_to_nchw_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant PermuteParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.n * p.h * p.w * p.c;
    if (tid >= total) return;
    uint cc = tid % p.c;
    uint ww = (tid / p.c) % p.w;
    uint hh = (tid / (p.c * p.w)) % p.h;
    uint nn = tid / (p.c * p.w * p.h);
    out[((nn * p.c + cc) * p.h + hh) * p.w + ww] = input[tid];
}

kernel void permute_nchw_to_nhwc_f16(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant PermuteParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.n * p.h * p.w * p.c;
    if (tid >= total) return;
    uint ww = tid % p.w;
    uint hh = (tid / p.w) % p.h;
    uint cc = (tid / (p.w * p.h)) % p.c;
    uint nn = tid / (p.w * p.h * p.c);
    out[((nn * p.h + hh) * p.w + ww) * p.c + cc] = input[tid];
}

// ── Vectorized Permute NHWC→NCHW f16: 3D grid (w, h*batch, c/4) ───
// Eliminates divmod by using grid position directly.
// Each thread reads half4 from NHWC (4 consecutive channels) and writes
// 4 scalar values to 4 channel planes in NCHW.

kernel void permute_nhwc_to_nchw_f16v4(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant PermuteParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint ww = gid.x;
    const uint hb = gid.y;
    const uint c4 = gid.z;
    if (ww >= p.w || hb >= p.h * p.n || c4 * 4 >= p.c) return;
    const uint hh = hb % p.h;
    const uint nn = hb / p.h;
    const uint hw = p.h * p.w;
    const uint c_base = c4 * 4;
    const uint in_off = ((nn * p.h + hh) * p.w + ww) * p.c + c_base;
    const uint out_base = (nn * p.c * hw) + hh * p.w + ww;
    if (c_base + 4 <= p.c) {
        half4 v = *(device const half4*)(input + in_off);
        out[out_base + (c_base + 0) * hw] = v[0];
        out[out_base + (c_base + 1) * hw] = v[1];
        out[out_base + (c_base + 2) * hw] = v[2];
        out[out_base + (c_base + 3) * hw] = v[3];
    } else {
        for (uint i = 0; i < p.c - c_base; i++) {
            out[out_base + (c_base + i) * hw] = input[in_off + i];
        }
    }
}

// ── Vectorized Permute NCHW→NHWC f16: 3D grid (w, h*batch, c/4) ───

kernel void permute_nchw_to_nhwc_f16v4(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant PermuteParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint ww = gid.x;
    const uint hb = gid.y;
    const uint c4 = gid.z;
    if (ww >= p.w || hb >= p.h * p.n || c4 * 4 >= p.c) return;
    const uint hh = hb % p.h;
    const uint nn = hb / p.h;
    const uint hw = p.h * p.w;
    const uint c_base = c4 * 4;
    const uint in_base = (nn * p.c * hw) + hh * p.w + ww;
    const uint out_off = ((nn * p.h + hh) * p.w + ww) * p.c + c_base;
    if (c_base + 4 <= p.c) {
        half4 v;
        v[0] = input[in_base + (c_base + 0) * hw];
        v[1] = input[in_base + (c_base + 1) * hw];
        v[2] = input[in_base + (c_base + 2) * hw];
        v[3] = input[in_base + (c_base + 3) * hw];
        *(device half4*)(out + out_off) = v;
    } else {
        for (uint i = 0; i < p.c - c_base; i++) {
            out[out_off + i] = input[in_base + (c_base + i) * hw];
        }
    }
}

// ── MatMul f16 I/O (f32 accumulators for precision, f16 input/output) ──

kernel void matmul_f16io(
    device const half* a    [[buffer(0)]],
    device const half* b    [[buffer(1)]],
    device half* out        [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 gid  [[threadgroup_position_in_grid]]
) {
    threadgroup half sa[MM_BM * MM_SA_STRIDE];
    threadgroup half sb[MM_BK * MM_SB_STRIDE];

    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint tid = ty * 16 + tx;

    const uint row0 = gid.y * MM_BM;
    const uint col0 = gid.x * MM_BN;

    // f32 accumulators to avoid f16 overflow in attention-style matmuls
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    const uint tiles = (p.k + MM_BK - 1) / MM_BK;
    for (uint t = 0; t < tiles; t++) {
        const uint k0 = t * MM_BK;

        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 4;
            const uint c = idx & 15;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            sa[r * MM_SA_STRIDE + c] = (gr < p.m && gk < p.k) ? a[gr * p.k + gk] : 0.0h;
        }

        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + i * 256;
            const uint r = idx >> 6;
            const uint c = idx & 63;
            const uint gr = k0 + r;
            const uint gc = col0 + c;
            sb[r * MM_SB_STRIDE + c] = (gr < p.k && gc < p.n) ? b[gr * p.n + gc] : 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b_base = tx * MM_TN;
        const uint a_base0 = (ty * MM_TM + 0) * MM_SA_STRIDE;
        const uint a_base1 = (ty * MM_TM + 1) * MM_SA_STRIDE;
        const uint a_base2 = (ty * MM_TM + 2) * MM_SA_STRIDE;
        const uint a_base3 = (ty * MM_TM + 3) * MM_SA_STRIDE;

        for (uint kk = 0; kk < MM_BK; kk++) {
            float4 bv = float4(
                sb[kk * MM_SB_STRIDE + b_base],
                sb[kk * MM_SB_STRIDE + b_base + 1],
                sb[kk * MM_SB_STRIDE + b_base + 2],
                sb[kk * MM_SB_STRIDE + b_base + 3]
            );
            acc0 = fma(float4(sa[a_base0 + kk]), bv, acc0);
            acc1 = fma(float4(sa[a_base1 + kk]), bv, acc1);
            acc2 = fma(float4(sa[a_base2 + kk]), bv, acc2);
            acc3 = fma(float4(sa[a_base3 + kk]), bv, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint c_col = col0 + tx * MM_TN;
    if (c_col + 3 < p.n) {
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < MM_TM; ri++) {
            const uint r = row0 + ty * MM_TM + ri;
            if (r >= p.m) continue;
            float4 v = clamp(accs[ri], float4(-65504.0f), float4(65504.0f));
            const uint base = r * p.n + c_col;
            out[base]     = half(v.x);
            out[base + 1] = half(v.y);
            out[base + 2] = half(v.z);
            out[base + 3] = half(v.w);
        }
    } else {
        float4 accs[4] = { acc0, acc1, acc2, acc3 };
        for (uint ri = 0; ri < MM_TM; ri++) {
            const uint r = row0 + ty * MM_TM + ri;
            if (r >= p.m) continue;
            for (uint ci = 0; ci < MM_TN; ci++) {
                const uint col = c_col + ci;
                if (col < p.n) {
                    float val = clamp(accs[ri][ci], -65504.0f, 65504.0f);
                    out[r * p.n + col] = half(val);
                }
            }
        }
    }
}

// ── Bias-add + activation (f16) ──────────────────────────────────
// Adds per-channel bias to [M, N] output in NHWC layout (N = num channels).
// Fuses optional activation: act: 0=none, 1=relu, 2=silu.
struct BiasActParams { uint total; uint n_channels; uint act; };

kernel void bias_act_f16(
    device half* data          [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    constant BiasActParams& p  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    const uint ch = tid % p.n_channels;
    float val = float(data[tid]) + bias[ch];
    if (p.act == 1) { val = max(val, 0.0f); }        // ReLU
    else if (p.act == 2) { val = val * 0.5f * (1.0f + fast::tanh(val * 0.5f)); } // SiLU
    data[tid] = half(val);
}

// ── Im2col (f16) ─────────────────────────────────────────────────
// Converts NHWC input to [M, K] column matrix for GEMM-based convolution.
// M = batch * oh * ow, K = kh * kw * ic.
// Output layout: row m, col k → im2col[m * K + k]
struct Im2colParams {
    uint batch; uint ih; uint iw; uint ic;
    uint oh; uint ow;
    uint kh; uint kw;
    uint sh; uint sw;
    uint pad_h; uint pad_w;
    uint m; uint k;
};

kernel void im2col_f16(
    device const half* input    [[buffer(0)]],
    device half* col            [[buffer(1)]],
    constant Im2colParams& p    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one element of the output [M, K] matrix
    if (tid >= p.m * p.k) return;
    const uint row = tid / p.k;  // spatial position index
    const uint gk = tid % p.k;   // within kh*kw*ic

    // Decode row → (b, oh_idx, ow_idx)
    const uint ow_idx = row % p.ow;
    const uint tmp = row / p.ow;
    const uint oh_idx = tmp % p.oh;
    const uint b = tmp / p.oh;

    // Decode gk → (c, kh_idx, kw_idx)
    const uint c = gk % p.ic;
    const uint tmp2 = gk / p.ic;
    const uint kw_idx = tmp2 % p.kw;
    const uint kh_idx = tmp2 / p.kw;

    const int iy = int(oh_idx * p.sh + kh_idx) - int(p.pad_h);
    const int ix = int(ow_idx * p.sw + kw_idx) - int(p.pad_w);

    half val = 0.0h;
    if (iy >= 0 && iy < int(p.ih) && ix >= 0 && ix < int(p.iw)) {
        val = input[((b * p.ih + uint(iy)) * p.iw + uint(ix)) * p.ic + c];
    }
    col[tid] = val;
}

// ═══════════════════════════════════════════════════════════════════════
// Vectorized resize nearest (f16): 3D grid eliminates all integer division.
// grid=(ic/4, ow, oh*batch), threads per group=(min(ic/4,32), 1, 1).
// Each thread copies one half4 (4 channels) via vectorized load/store.
// ═══════════════════════════════════════════════════════════════════════

kernel void resize_nearest_f16v4(
    device const half* input [[buffer(0)]],
    device half* out         [[buffer(1)]],
    constant ResizeParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]  // x=ic/4, y=ow, z=oh*batch
) {
    const uint c4 = gid.x;
    const uint ow_idx = gid.y;
    const uint oh_b = gid.z;
    if (c4 * 4 >= p.ic || ow_idx >= p.ow || oh_b >= p.oh * p.batch) return;

    const uint oh_idx = oh_b % p.oh;
    const uint b = oh_b / p.oh;

    const uint iy = min(uint(float(oh_idx) / p.scale_h), p.ih - 1);
    const uint ix = min(uint(float(ow_idx) / p.scale_w), p.iw - 1);

    const uint in_off = ((b * p.ih + iy) * p.iw + ix) * p.ic + c4 * 4;
    const uint out_off = ((oh_b * p.ow + ow_idx)) * p.ic + c4 * 4;

    if (c4 * 4 + 4 <= p.ic) {
        *(device half4*)(out + out_off) = *(device const half4*)(input + in_off);
    } else {
        for (uint i = 0; i < p.ic - c4 * 4; i++)
            out[out_off + i] = input[in_off + i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Vectorized concat channels (f16): processes half4 chunks.
// 2D grid: x=out_c/4, y=spatial. Eliminates per-element branching
// by computing input index via prefix-sum on channel boundaries.
// ═══════════════════════════════════════════════════════════════════════

kernel void concat_channels_f16v4(
    device const half* in0  [[buffer(0)]],
    device const half* in1  [[buffer(1)]],
    device const half* in2  [[buffer(2)]],
    device const half* in3  [[buffer(3)]],
    device half* out        [[buffer(4)]],
    constant ConcatParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]  // x=c4, y=spatial
) {
    const uint c4 = gid.x;
    const uint spatial = gid.y;
    const uint c_base = c4 * 4;
    const uint n_spatial = p.total_elements / p.out_c;
    if (c_base >= p.out_c || spatial >= n_spatial) return;

    const uint out_base = spatial * p.out_c + c_base;

    // Determine which input and local offset
    uint cum = 0;
    device const half* src = in0;
    uint src_c = p.channels[0];
    uint local_c = c_base;

    if (c_base >= p.channels[0]) {
        cum = p.channels[0];
        src = in1; src_c = p.channels[1]; local_c = c_base - cum;
        if (p.n_inputs > 2 && c_base >= cum + p.channels[1]) {
            cum += p.channels[1];
            src = in2; src_c = p.channels[2]; local_c = c_base - cum;
            if (p.n_inputs > 3 && c_base >= cum + p.channels[2]) {
                cum += p.channels[2];
                src = in3; src_c = p.channels[3]; local_c = c_base - cum;
            }
        }
    }

    const uint in_base = spatial * src_c + local_c;

    // Check if this 4-element chunk stays within one input
    if (local_c + 4 <= src_c && c_base + 4 <= p.out_c) {
        *(device half4*)(out + out_base) = *(device const half4*)(src + in_base);
    } else {
        // Scalar fallback for boundary
        uint rem = min(4u, p.out_c - c_base);
        for (uint i = 0; i < rem; i++) {
            uint gc = c_base + i;
            // Find input for this channel
            uint c2 = 0;
            if (gc < p.channels[0]) {
                out[out_base + i] = in0[spatial * p.channels[0] + gc];
            } else {
                c2 = p.channels[0];
                if (gc < c2 + p.channels[1]) {
                    out[out_base + i] = in1[spatial * p.channels[1] + (gc - c2)];
                } else {
                    c2 += p.channels[1];
                    if (p.n_inputs > 2 && gc < c2 + p.channels[2]) {
                        out[out_base + i] = in2[spatial * p.channels[2] + (gc - c2)];
                    } else {
                        c2 += p.channels[2];
                        out[out_base + i] = in3[spatial * p.channels[3] + (gc - c2)];
                    }
                }
            }
        }
    }
}

// ── Fused NHWC→flat-NCHW concat ──────────────────────────────────
// Reads up to 3 NHWC [1,Hi,Wi,C] input buffers and writes to flat
// NCHW [1,C,sum(Hi*Wi)] output in a single pass.
// Eliminates separate NHWC→NCHW permutation + FlatConcat copies.
struct NhwcToFlatParams {
    uint c;             // channel count (same for all inputs)
    uint n_inputs;      // 1-3
    uint h0; uint w0;   // input 0 spatial dims
    uint h1; uint w1;   // input 1 spatial dims
    uint h2; uint w2;   // input 2 spatial dims
    uint total_spatial;  // h0*w0 + h1*w1 + h2*w2
};

kernel void nhwc_to_flat_concat_f16(
    device const half* in0 [[buffer(0)]],
    device const half* in1 [[buffer(1)]],
    device const half* in2 [[buffer(2)]],
    device half* out       [[buffer(3)]],
    constant NhwcToFlatParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]  // x=spatial, y=channel
) {
    const uint s = gid.x;
    const uint c = gid.y;
    if (s >= p.total_spatial || c >= p.c) return;

    const uint s0 = p.h0 * p.w0;
    const uint s01 = s0 + p.h1 * p.w1;
    half val;
    if (s < s0) {
        val = in0[s * p.c + c];
    } else if (s < s01) {
        val = in1[(s - s0) * p.c + c];
    } else {
        val = in2[(s - s01) * p.c + c];
    }
    out[c * p.total_spatial + s] = val;
}

// ── Channel scatter (reverse of split_channels) ────────────────
// Copies dense [spatial, src_c] → strided [spatial, dst_c] at channel offset.
// Used for partial concat fusion: copies non-conv inputs into concat buffer.
kernel void channel_scatter_f16(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint4& p      [[buffer(2)]],  // (spatial, src_c, dst_c, dst_offset)
    uint2 gid [[thread_position_in_grid]]  // x=channel, y=spatial
) {
    const uint c = gid.x;
    const uint s = gid.y;
    if (c >= p.y || s >= p.x) return;
    dst[s * p.z + p.w + c] = src[s * p.y + c];
}
