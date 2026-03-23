struct Params { seq_q: u32, seq_k: u32, d_k: u32, d_v: u32, scale: f32, _pad1: u32, _pad2: u32, _pad3: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.y;
    let vi = gid.x;
    if (qi >= p.seq_q || vi >= p.d_v) { return; }
    var sum = 0.0;
    for (var ki = 0u; ki < p.seq_k; ki++) {
        var dot = 0.0;
        for (var d = 0u; d < p.d_k; d++) {
            dot += q[qi * p.d_k + d] * k[ki * p.d_k + d];
        }
        let score = dot * p.scale;
        // Simplified: no softmax in this kernel (applied separately)
        sum += score * v[ki * p.d_v + vi];
    }
    out[qi * p.d_v + vi] = sum;
}
