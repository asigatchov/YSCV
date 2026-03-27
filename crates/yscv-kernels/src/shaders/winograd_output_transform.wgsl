// Winograd F(2,3) output transform + bias + activation.
// Input: [16, num_tiles, OC] from batched GEMM.
// Output: NHWC [N, OH, OW, OC].
// A^T = [[1,1,1,0],[0,1,-1,-1]]
// Y = A^T * M * A → 2x2 output tile
struct Params {
    num_tiles: u32, oc: u32, oh: u32, ow: u32,
    tiles_h: u32, tiles_w: u32, act: u32, batch: u32,
}
@group(0) @binding(0) var<storage, read> gemm_out: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.num_tiles * p.oc;
    if (idx >= total) { return; }

    let oc = idx % p.oc;
    let tile_idx = idx / p.oc;

    // Decode tile_idx → (batch, tile_h, tile_w)
    let tiles_per_batch = p.tiles_h * p.tiles_w;
    let b = tile_idx / tiles_per_batch;
    let rem = tile_idx % tiles_per_batch;
    let tile_h = rem / p.tiles_w;
    let tile_w = rem % p.tiles_w;

    // Gather 4x4 from GEMM output: gemm_out[alpha * num_tiles * OC + tile_idx * OC + oc]
    let stride = p.num_tiles * p.oc;
    var m: array<f32, 16>;
    for (var a = 0u; a < 16u; a++) {
        m[a] = gemm_out[a * stride + tile_idx * p.oc + oc];
    }

    // temp = A^T * M (2x4)
    // A^T rows: [1,1,1,0], [0,1,-1,-1]
    var t: array<f32, 8>; // 2x4
    for (var j = 0u; j < 4u; j++) {
        t[0u * 4u + j] = m[0u * 4u + j] + m[1u * 4u + j] + m[2u * 4u + j];
        t[1u * 4u + j] = m[1u * 4u + j] - m[2u * 4u + j] - m[3u * 4u + j];
    }

    // Y = temp * A (2x2)
    var y: array<f32, 4>;
    y[0] = t[0] + t[1] + t[2];           // Y[0][0]
    y[1] = t[1] - t[2] - t[3];           // Y[0][1]
    y[2] = t[4] + t[5] + t[6];           // Y[1][0]
    y[3] = t[5] - t[6] - t[7];           // Y[1][1]

    // Add bias + activation + write to NHWC output
    let b_val = bias[oc];
    for (var dy = 0u; dy < 2u; dy++) {
        let oh_idx = tile_h * 2u + dy;
        if (oh_idx >= p.oh) { continue; }
        for (var dx = 0u; dx < 2u; dx++) {
            let ow_idx = tile_w * 2u + dx;
            if (ow_idx >= p.ow) { continue; }
            var v = y[dy * 2u + dx] + b_val;
            if (p.act == 1u) { v = max(v, 0.0); }
            else if (p.act == 2u) { v = v / (1.0 + exp(-v)); }
            out[((b * p.oh + oh_idx) * p.ow + ow_idx) * p.oc + oc] = v;
        }
    }
}
