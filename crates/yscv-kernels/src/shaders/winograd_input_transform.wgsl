// Winograd F(2,3) input transform: NHWC input → [16, num_tiles, IC]
// Each thread handles one (tile_idx, ic) pair.
// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
// V = B^T * d * B where d is 4x4 input tile
struct Params {
    num_tiles: u32, ic: u32, ih: u32, iw: u32,
    tiles_h: u32, tiles_w: u32, pad_h: u32, pad_w: u32,
    batch: u32, _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

fn load_input(b: u32, h: i32, w: i32, c: u32) -> f32 {
    if (h >= 0 && u32(h) < p.ih && w >= 0 && u32(w) < p.iw) {
        return inp[((b * p.ih + u32(h)) * p.iw + u32(w)) * p.ic + c];
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.num_tiles * p.ic;
    if (idx >= total) { return; }

    let ic = idx % p.ic;
    let tile_idx = idx / p.ic;

    // Decode tile_idx → (batch, tile_h, tile_w)
    let tiles_per_batch = p.tiles_h * p.tiles_w;
    let b = tile_idx / tiles_per_batch;
    let rem = tile_idx % tiles_per_batch;
    let tile_h = rem / p.tiles_w;
    let tile_w = rem % p.tiles_w;

    // Top-left corner of the 4x4 input tile
    let h0 = i32(tile_h * 2u) - i32(p.pad_h);
    let w0 = i32(tile_w * 2u) - i32(p.pad_w);

    // Load 4x4 input tile
    var d: array<f32, 16>; // d[row * 4 + col]
    for (var r = 0u; r < 4u; r++) {
        for (var c = 0u; c < 4u; c++) {
            d[r * 4u + c] = load_input(b, h0 + i32(r), w0 + i32(c), ic);
        }
    }

    // temp = B^T * d (4x4)
    // B^T rows: [1,0,-1,0], [0,1,1,0], [0,-1,1,0], [0,1,0,-1]
    var t: array<f32, 16>;
    for (var j = 0u; j < 4u; j++) {
        t[0u * 4u + j] = d[0u * 4u + j] - d[2u * 4u + j];
        t[1u * 4u + j] = d[1u * 4u + j] + d[2u * 4u + j];
        t[2u * 4u + j] = -d[1u * 4u + j] + d[2u * 4u + j];
        t[3u * 4u + j] = d[1u * 4u + j] - d[3u * 4u + j];
    }

    // V = temp * B (4x4)
    // B cols (= B^T rows transposed): same structure
    var v: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        v[i * 4u + 0u] = t[i * 4u + 0u] - t[i * 4u + 2u];
        v[i * 4u + 1u] = t[i * 4u + 1u] + t[i * 4u + 2u];
        v[i * 4u + 2u] = -t[i * 4u + 1u] + t[i * 4u + 2u];
        v[i * 4u + 3u] = t[i * 4u + 1u] - t[i * 4u + 3u];
    }

    // Scatter to output: out[alpha * num_tiles * IC + tile_idx * IC + ic]
    let stride = p.num_tiles * p.ic;
    for (var a = 0u; a < 4u; a++) {
        for (var bb = 0u; bb < 4u; bb++) {
            out[(a * 4u + bb) * stride + tile_idx * p.ic + ic] = v[a * 4u + bb];
        }
    }
}
