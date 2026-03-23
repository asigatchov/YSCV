struct Params { len: u32, op: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> forward_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    switch p.op {
        // relu_backward: upstream * (forward_input > 0 ? 1 : 0)
        case 0u: { grad_out[i] = upstream[i] * select(0.0, 1.0, forward_val[i] > 0.0); }
        // sigmoid_backward: upstream * s * (1 - s)  where s = forward_output
        case 1u: {
            let s = forward_val[i];
            grad_out[i] = upstream[i] * s * (1.0 - s);
        }
        // tanh_backward: upstream * (1 - t*t)  where t = forward_output
        case 2u: {
            let t = forward_val[i];
            grad_out[i] = upstream[i] * (1.0 - t * t);
        }
        // exp_backward: upstream * e  where e = forward_output
        case 3u: { grad_out[i] = upstream[i] * forward_val[i]; }
        default: {}
    }
}
