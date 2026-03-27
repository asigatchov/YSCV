// Batched matmul: A[batch, M, K] × B[batch, K, N] = C[batch, M, N]
// Vec4 accumulators for the TN=4 inner dimension.
struct Params { m: u32, n: u32, k: u32, batch: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const SA_STRIDE: u32 = 17u;
const SB_STRIDE: u32 = 65u;
var<workgroup> sa: array<f32, 1088>;
var<workgroup> sb: array<f32, 1040>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let batch = wid.z;
    if (batch >= p.batch) { return; }

    let a_off = batch * p.m * p.k;
    let b_off = batch * p.k * p.n;
    let c_off = batch * p.m * p.n;

    let row0 = wid.y * BM;
    let col0 = wid.x * BN;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t++) {
        let k0 = t * BK;
        for (var i = 0u; i < 4u; i++) {
            let idx = tid + i * 256u;
            let r = idx / BK;
            let c = idx % BK;
            let gr = row0 + r;
            let gc_val = k0 + c;
            if (gr < p.m && gc_val < p.k) {
                sa[r * SA_STRIDE + c] = a[a_off + gr * p.k + gc_val];
            } else {
                sa[r * SA_STRIDE + c] = 0.0;
            }
        }
        for (var i = 0u; i < 4u; i++) {
            let idx = tid + i * 256u;
            let r = idx / BN;
            let c = idx % BN;
            let gr = k0 + r;
            let gc_val = col0 + c;
            if (gr < p.k && gc_val < p.n) {
                sb[r * SB_STRIDE + c] = b[b_off + gr * p.n + gc_val];
            } else {
                sb[r * SB_STRIDE + c] = 0.0;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let a0 = sa[(ty * TM + 0u) * SA_STRIDE + kk];
            let a1 = sa[(ty * TM + 1u) * SA_STRIDE + kk];
            let a2 = sa[(ty * TM + 2u) * SA_STRIDE + kk];
            let a3 = sa[(ty * TM + 3u) * SA_STRIDE + kk];
            let bv = vec4<f32>(
                sb[kk * SB_STRIDE + tx * TN + 0u],
                sb[kk * SB_STRIDE + tx * TN + 1u],
                sb[kk * SB_STRIDE + tx * TN + 2u],
                sb[kk * SB_STRIDE + tx * TN + 3u],
            );
            acc0 += a0 * bv;
            acc1 += a1 * bv;
            acc2 += a2 * bv;
            acc3 += a3 * bv;
        }
        workgroupBarrier();
    }

    let c = col0 + tx * TN;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var ri = 0u; ri < TM; ri++) {
        let r = row0 + ty * TM + ri;
        if (r >= p.m) { continue; }
        let v = accs[ri];
        if (c + 3u < p.n) {
            let base = c_off + r * p.n + c;
            out[base]      = v.x;
            out[base + 1u] = v.y;
            out[base + 2u] = v.z;
            out[base + 3u] = v.w;
        } else {
            for (var ci = 0u; ci < TN; ci++) {
                if (c + ci < p.n) {
                    out[c_off + r * p.n + c + ci] = v[ci];
                }
            }
        }
    }
}
