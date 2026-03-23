struct Params { m: u32, n: u32, k: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
const T: u32 = 16u;
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    var sum = 0.0;
    let tiles = (p.k + T - 1u) / T;
    for (var t = 0u; t < tiles; t++) {
        let ac = t * T + lid.x;
        if (row < p.m && ac < p.k) { ta[lid.y * T + lid.x] = a[row * p.k + ac]; }
        else { ta[lid.y * T + lid.x] = 0.0; }
        let br = t * T + lid.y;
        if (br < p.k && col < p.n) { tb[lid.y * T + lid.x] = b[br * p.n + col]; }
        else { tb[lid.y * T + lid.x] = 0.0; }
        workgroupBarrier();
        for (var i = 0u; i < T; i++) { sum += ta[lid.y * T + i] * tb[i * T + lid.x]; }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) { out[row * p.n + col] = sum; }
}
