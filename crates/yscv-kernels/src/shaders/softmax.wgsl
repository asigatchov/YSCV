struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var mx = inp[base];
    for (var j = 1u; j < p.cols; j++) { mx = max(mx, inp[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < p.cols; j++) {
        let e = exp(inp[base + j] - mx);
        out[base + j] = e;
        s += e;
    }
    for (var j = 0u; j < p.cols; j++) { out[base + j] /= s; }
}
