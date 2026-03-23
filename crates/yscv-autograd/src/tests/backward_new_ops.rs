use yscv_tensor::Tensor;

use crate::Graph;

// ---- ConvTranspose2d backward ----

#[test]
fn backward_conv_transpose2d_nhwc_computes_weight_and_input_grads() {
    let mut graph = Graph::new();
    // input: [1, 2, 2, 1], weight: [2, 2, 1, 1], no bias, stride=1
    // output: [1, 3, 3, 1]
    let input =
        graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let weight =
        graph.variable(Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 0.5, 0.5, 0.25]).unwrap());

    let out = graph
        .conv_transpose2d_nhwc(input, weight, None, 1, 1)
        .unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 3, 3, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let w_grad = graph.grad(weight).unwrap().unwrap();
    assert_eq!(w_grad.shape(), &[2, 2, 1, 1]);

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 1]);

    // Each input element contributes to 2x2 output region via weight.
    // grad_input[ih,iw] = sum over ki,kj of weight[ki,kj,0,0] * upstream(=1)
    // = 1.0 + 0.5 + 0.5 + 0.25 = 2.25 for all positions
    for &g in i_grad.data() {
        assert!(
            (g - 2.25).abs() < 1e-4,
            "input grad mismatch: got {g}, expected 2.25"
        );
    }

    // grad_weight[ki,kj] = sum over batch,ih,iw of input[batch,ih,iw,0] * upstream[batch,ih*s+ki,iw*s+kj,0]
    // = sum of input = 1+2+3+4 = 10 for each (ki,kj) since upstream is all 1s
    for &g in w_grad.data() {
        assert!(
            (g - 10.0).abs() < 1e-4,
            "weight grad mismatch: got {g}, expected 10.0"
        );
    }
}

#[test]
fn backward_conv_transpose2d_nhwc_with_bias() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 2, 2, 2], 1.0).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 1, 2], 0.5).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![1], vec![0.1]).unwrap());

    let out = graph
        .conv_transpose2d_nhwc(input, weight, Some(bias), 1, 1)
        .unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 3, 3, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let b_grad = graph.grad(bias).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[1]);
    // bias grad = number of output elements = 3*3 = 9
    assert!((b_grad.data()[0] - 9.0).abs() < 1e-4);
}

#[test]
fn backward_conv_transpose2d_nhwc_stride2() {
    let mut graph = Graph::new();
    // input: [1, 2, 2, 1], stride=2, kernel=2x2 -> output [1, 4, 4, 1]
    // out_h = (2-1)*2 + 2 = 4, out_w = (2-1)*2 + 2 = 4
    let input =
        graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 1, 1], 1.0).unwrap());

    let out = graph
        .conv_transpose2d_nhwc(input, weight, None, 2, 2)
        .unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 4, 4, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 1]);
    // Each input element contributes to exactly 2x2 non-overlapping region (stride=2, kernel=2)
    // grad_input[ih,iw] = sum of weight = 4.0
    for &g in i_grad.data() {
        assert!((g - 4.0).abs() < 1e-4, "stride2 input grad: got {g}");
    }
}

// ---- AdaptiveAvgPool2d backward ----

#[test]
fn backward_adaptive_avg_pool2d_nhwc_distributes_uniformly() {
    let mut graph = Graph::new();
    // input [1,4,4,1] -> adaptive_avg_pool to [1,2,2,1]
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 4, 4, 1], data).unwrap());
    let out = graph.adaptive_avg_pool2d_nhwc(input, 2, 2).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 4, 4, 1]);
    // Each output covers a 2x2 region, so each input gets 1/4 of the upstream gradient (=1.0)
    for &g in i_grad.data() {
        assert!(
            (g - 0.25).abs() < 1e-6,
            "adaptive avg pool grad: got {g}, expected 0.25"
        );
    }
}

#[test]
fn backward_adaptive_avg_pool2d_to_1x1() {
    let mut graph = Graph::new();
    // input [1,3,3,2] -> adaptive pool to [1,1,1,2] = global average pool
    let input = graph.variable(Tensor::filled(vec![1, 3, 3, 2], 1.0).unwrap());
    let out = graph.adaptive_avg_pool2d_nhwc(input, 1, 1).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 1, 1, 2]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    // Each of 9 spatial positions gets 1/9
    for &g in i_grad.data() {
        assert!(
            (g - 1.0 / 9.0).abs() < 1e-6,
            "global avg pool grad: got {g}"
        );
    }
}

// ---- AdaptiveMaxPool2d backward ----

#[test]
fn backward_adaptive_max_pool2d_nhwc_scatters_to_argmax() {
    let mut graph = Graph::new();
    // input [1,4,4,1] -> adaptive max pool to [1,2,2,1]
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 4, 4, 1], data).unwrap());
    let out = graph.adaptive_max_pool2d_nhwc(input, 2, 2).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 4, 4, 1]);
    // Only max elements in each 2x2 window should get gradient
    let nonzero_count = i_grad.data().iter().filter(|&&g| g > 0.5).count();
    assert_eq!(
        nonzero_count, 4,
        "expected 4 max elements to receive gradient"
    );
}

// ---- InstanceNorm backward ----

#[test]
fn backward_instance_norm_nhwc_computes_gamma_beta_input_grads() {
    let mut graph = Graph::new();
    // input [1,2,2,2] with known values
    let input = graph.variable(
        Tensor::from_vec(
            vec![1, 2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap(),
    );
    let gamma = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());
    let beta = graph.variable(Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap());

    let out = graph.instance_norm_nhwc(input, gamma, beta, 1e-5).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 2]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let g_grad = graph.grad(gamma).unwrap().unwrap();
    assert_eq!(g_grad.shape(), &[2]);

    let b_grad = graph.grad(beta).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[2]);
    // beta grad = sum of upstream = 4 per channel
    assert!((b_grad.data()[0] - 4.0).abs() < 1e-4);
    assert!((b_grad.data()[1] - 4.0).abs() < 1e-4);

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 2]);
    // Instance norm backward with upstream=1: gradient should be approximately 0
    // since sum of normalized values is 0 and gradient of affine through norm is 0.
    for &g in i_grad.data() {
        assert!(
            g.abs() < 1e-3,
            "instance norm input grad should be near 0, got {g}"
        );
    }
}

#[test]
fn backward_instance_norm_nhwc_multi_batch() {
    let mut graph = Graph::new();
    // input [2,2,2,1]
    let input = graph.variable(
        Tensor::from_vec(
            vec![2, 2, 2, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap(),
    );
    let gamma = graph.variable(Tensor::from_vec(vec![1], vec![2.0]).unwrap());
    let beta = graph.variable(Tensor::from_vec(vec![1], vec![0.0]).unwrap());

    let out = graph.instance_norm_nhwc(input, gamma, beta, 1e-5).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[2, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let g_grad = graph.grad(gamma).unwrap().unwrap();
    assert_eq!(g_grad.shape(), &[1]);

    let b_grad = graph.grad(beta).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[1]);
    // 2 batches * 4 spatial positions
    assert!((b_grad.data()[0] - 8.0).abs() < 1e-4);
}

// ---- PReLU backward ----

#[test]
fn backward_prelu_scalar_alpha() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 4], vec![1.0, -2.0, 3.0, -4.0]).unwrap());
    let alpha = graph.variable(Tensor::from_vec(vec![1], vec![0.1]).unwrap());

    let out = graph.prelu(input, alpha).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    // For positive x: grad = 1, for negative x: grad = alpha = 0.1
    assert!((i_grad.data()[0] - 1.0).abs() < 1e-6);
    assert!((i_grad.data()[1] - 0.1).abs() < 1e-6);
    assert!((i_grad.data()[2] - 1.0).abs() < 1e-6);
    assert!((i_grad.data()[3] - 0.1).abs() < 1e-6);

    let a_grad = graph.grad(alpha).unwrap().unwrap();
    // dalpha = sum(dout * min(x, 0)) = 1 * (-2.0) + 1 * (-4.0) = -6.0
    assert!((a_grad.data()[0] - (-6.0)).abs() < 1e-6);
}

#[test]
fn backward_prelu_per_channel_alpha() {
    let mut graph = Graph::new();
    // 4 channels, 2 negative, 2 positive
    let input = graph.variable(Tensor::from_vec(vec![1, 4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap());
    let alpha = graph.variable(Tensor::from_vec(vec![4], vec![0.1, 0.2, 0.3, 0.4]).unwrap());

    let out = graph.prelu(input, alpha).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert!((i_grad.data()[0] - 0.1).abs() < 1e-6); // negative, alpha[0] = 0.1
    assert!((i_grad.data()[1] - 1.0).abs() < 1e-6); // positive
    assert!((i_grad.data()[2] - 0.3).abs() < 1e-6); // negative, alpha[2] = 0.3
    assert!((i_grad.data()[3] - 1.0).abs() < 1e-6); // positive

    let a_grad = graph.grad(alpha).unwrap().unwrap();
    // dalpha[0] = -1.0 (from input=-1.0), dalpha[1] = 0 (positive), dalpha[2] = -3.0, dalpha[3] = 0
    assert!((a_grad.data()[0] - (-1.0)).abs() < 1e-6);
    assert!((a_grad.data()[1] - 0.0).abs() < 1e-6);
    assert!((a_grad.data()[2] - (-3.0)).abs() < 1e-6);
    assert!((a_grad.data()[3] - 0.0).abs() < 1e-6);
}

// ---- SeparableConv2d backward (composition test) ----

#[test]
fn backward_separable_conv2d_produces_grads_through_composition() {
    // SeparableConv2d = DepthwiseConv2d + pointwise Conv2d.
    // Both already have backward passes, so this tests the composition.
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 3, 3, 2], 1.0).unwrap());

    // Depthwise: [2,2,2,1], stride=1
    let dw_weight = graph.variable(Tensor::filled(vec![2, 2, 2, 1], 0.5).unwrap());
    // Pointwise: [1,1,2,1], stride=1
    let pw_weight = graph.variable(Tensor::filled(vec![1, 1, 2, 1], 1.0).unwrap());

    // Depthwise conv
    let dw_out = graph
        .depthwise_conv2d_nhwc(input, dw_weight, None, 1, 1)
        .unwrap();
    assert_eq!(graph.value(dw_out).unwrap().shape(), &[1, 2, 2, 2]);

    // Pointwise conv
    let out = graph.conv2d_nhwc(dw_out, pw_weight, None, 1, 1).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    // All three should have gradients
    assert!(graph.grad(input).unwrap().is_some());
    assert!(graph.grad(dw_weight).unwrap().is_some());
    assert!(graph.grad(pw_weight).unwrap().is_some());
}

// ---- Numerical gradient checks ----

#[test]
fn backward_conv_transpose2d_numerical_gradient_check() {
    // Numerical gradient check for conv_transpose2d weight
    let eps = 1e-3;
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let weight_data = vec![1.0, 0.5, 0.5, 0.25];

    // Forward + backward
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], input_data.clone()).unwrap());
    let weight = graph.variable(Tensor::from_vec(vec![2, 2, 1, 1], weight_data.clone()).unwrap());
    let out = graph
        .conv_transpose2d_nhwc(input, weight, None, 1, 1)
        .unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_grad = graph.grad(weight).unwrap().unwrap().data().to_vec();

    // Numerical gradient for each weight element
    for w_idx in 0..4 {
        let mut wp = weight_data.clone();
        wp[w_idx] += eps;
        let mut graph_p = Graph::new();
        let inp = graph_p.variable(Tensor::from_vec(vec![1, 2, 2, 1], input_data.clone()).unwrap());
        let wt = graph_p.variable(Tensor::from_vec(vec![2, 2, 1, 1], wp).unwrap());
        let o = graph_p.conv_transpose2d_nhwc(inp, wt, None, 1, 1).unwrap();
        let loss_p = graph_p.value(o).unwrap().sum();

        let mut wm = weight_data.clone();
        wm[w_idx] -= eps;
        let mut graph_m = Graph::new();
        let inp = graph_m.variable(Tensor::from_vec(vec![1, 2, 2, 1], input_data.clone()).unwrap());
        let wt = graph_m.variable(Tensor::from_vec(vec![2, 2, 1, 1], wm).unwrap());
        let o = graph_m.conv_transpose2d_nhwc(inp, wt, None, 1, 1).unwrap();
        let loss_m = graph_m.value(o).unwrap().sum();

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_grad[w_idx] - numerical).abs() < 1e-2,
            "conv_transpose2d weight grad mismatch at {w_idx}: analytic={}, numerical={}",
            analytic_grad[w_idx],
            numerical
        );
    }
}

#[test]
fn backward_prelu_numerical_gradient_check() {
    let eps = 1e-3;
    let input_data = vec![1.0, -2.0, 3.0, -4.0];
    let alpha_val = 0.1f32;

    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 4], input_data.clone()).unwrap());
    let alpha = graph.variable(Tensor::from_vec(vec![1], vec![alpha_val]).unwrap());
    let out = graph.prelu(input, alpha).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_alpha_grad = graph.grad(alpha).unwrap().unwrap().data()[0];

    // Numerical for alpha
    let mut graph_p = Graph::new();
    let inp = graph_p.variable(Tensor::from_vec(vec![1, 4], input_data.clone()).unwrap());
    let ap = graph_p.variable(Tensor::from_vec(vec![1], vec![alpha_val + eps]).unwrap());
    let o = graph_p.prelu(inp, ap).unwrap();
    let loss_p = graph_p.value(o).unwrap().sum();

    let mut graph_m = Graph::new();
    let inp = graph_m.variable(Tensor::from_vec(vec![1, 4], input_data.clone()).unwrap());
    let am = graph_m.variable(Tensor::from_vec(vec![1], vec![alpha_val - eps]).unwrap());
    let o = graph_m.prelu(inp, am).unwrap();
    let loss_m = graph_m.value(o).unwrap().sum();

    let numerical = (loss_p - loss_m) / (2.0 * eps);
    assert!(
        (analytic_alpha_grad - numerical).abs() < 1e-2,
        "prelu alpha grad: analytic={analytic_alpha_grad}, numerical={numerical}"
    );
}
