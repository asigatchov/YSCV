use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_conv2d_nhwc_computes_weight_and_input_grads() {
    let mut graph = Graph::new();
    // input: [1, 3, 3, 1], weight: [2, 2, 1, 1], no bias, stride=1
    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 3, 3, 1], input_data).unwrap());
    let weight =
        graph.variable(Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 0.0, 0.0, -1.0]).unwrap());

    let out = graph.conv2d_nhwc(input, weight, None, 1, 1).unwrap();
    // out shape: [1, 2, 2, 1]
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let w_grad = graph.grad(weight).unwrap().unwrap();
    assert_eq!(w_grad.shape(), &[2, 2, 1, 1]);

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 3, 3, 1]);

    // Numerical gradient check for weight:
    // grad_w[kh,kw,0,0] = sum_{oh,ow} input[oh+kh, ow+kw, 0]
    // For kh=0,kw=0: input at (0,0),(0,1),(1,0),(1,1) = 1+2+4+5 = 12
    // For kh=0,kw=1: (0,1),(0,2),(1,1),(1,2) = 2+3+5+6 = 16
    // For kh=1,kw=0: (1,0),(1,1),(2,0),(2,1) = 4+5+7+8 = 24
    // For kh=1,kw=1: (1,1),(1,2),(2,1),(2,2) = 5+6+8+9 = 28
    let expected_wg = [12.0, 16.0, 24.0, 28.0];
    for (i, &e) in expected_wg.iter().enumerate() {
        assert!(
            (w_grad.data()[i] - e).abs() < 1e-4,
            "weight grad mismatch at {i}: got {}, expected {e}",
            w_grad.data()[i]
        );
    }
}

#[test]
fn backward_conv2d_nhwc_with_bias_produces_bias_grad() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 3, 3, 2], 1.0).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 2, 1], 0.5).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![1], vec![0.1]).unwrap());

    let out = graph.conv2d_nhwc(input, weight, Some(bias), 1, 1).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    // bias grad = number of output spatial elements = 2*2 = 4
    let b_grad = graph.grad(bias).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[1]);
    assert!((b_grad.data()[0] - 4.0).abs() < 1e-4);
}

#[test]
fn backward_conv2d_nhwc_stride2_shape_and_grad() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 4, 4, 1], 1.0).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 1, 1], 1.0).unwrap());

    let out = graph.conv2d_nhwc(input, weight, None, 2, 2).unwrap();
    // OH = (4-2)/2+1 = 2, OW = 2
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 4, 4, 1]);
    // Each input pixel contributes to at most one output with stride=2 and kernel=2
    for &g in i_grad.data() {
        assert!((0.0..=1.0 + 1e-6).contains(&g));
    }
}

#[test]
fn backward_conv2d_nhwc_numerical_gradient_check() {
    let eps = 1e-3;
    let input_data: Vec<f32> = vec![
        0.5, 1.2, 0.3, 0.8, 1.0, 0.7, 0.9, 1.5, 0.2, 0.6, 1.1, 0.4, 0.8, 0.3, 0.7, 1.0,
    ];
    let weight_data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];

    // Compute analytical gradients
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 4, 4, 1], input_data.clone()).unwrap());
    let weight = graph.variable(Tensor::from_vec(vec![2, 2, 1, 1], weight_data.clone()).unwrap());
    let out = graph.conv2d_nhwc(input, weight, None, 1, 1).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_wg = graph.grad(weight).unwrap().unwrap().data().to_vec();
    let analytic_ig = graph.grad(input).unwrap().unwrap().data().to_vec();

    // Numerical gradient check for weight
    for wi in 0..4 {
        let mut wp = weight_data.clone();
        wp[wi] += eps;
        let mut wm = weight_data.clone();
        wm[wi] -= eps;

        let mut gp = Graph::new();
        let ip = gp.variable(Tensor::from_vec(vec![1, 4, 4, 1], input_data.clone()).unwrap());
        let wp_node = gp.variable(Tensor::from_vec(vec![2, 2, 1, 1], wp).unwrap());
        let op = gp.conv2d_nhwc(ip, wp_node, None, 1, 1).unwrap();
        let lp = gp.sum(op).unwrap();
        let loss_p = gp.value(lp).unwrap().data()[0];

        let mut gm = Graph::new();
        let im = gm.variable(Tensor::from_vec(vec![1, 4, 4, 1], input_data.clone()).unwrap());
        let wm_node = gm.variable(Tensor::from_vec(vec![2, 2, 1, 1], wm).unwrap());
        let om = gm.conv2d_nhwc(im, wm_node, None, 1, 1).unwrap();
        let lm = gm.sum(om).unwrap();
        let loss_m = gm.value(lm).unwrap().data()[0];

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_wg[wi] - numerical).abs() < 1e-2,
            "weight numerical grad mismatch at {wi}: analytic={}, numerical={numerical}",
            analytic_wg[wi]
        );
    }

    // Numerical gradient check for input (spot check first 4 elements)
    for ii in 0..4 {
        let mut ip_data = input_data.clone();
        ip_data[ii] += eps;
        let mut im_data = input_data.clone();
        im_data[ii] -= eps;

        let mut gp = Graph::new();
        let ip = gp.variable(Tensor::from_vec(vec![1, 4, 4, 1], ip_data).unwrap());
        let wp = gp.variable(Tensor::from_vec(vec![2, 2, 1, 1], weight_data.clone()).unwrap());
        let op = gp.conv2d_nhwc(ip, wp, None, 1, 1).unwrap();
        let lp = gp.sum(op).unwrap();
        let loss_p = gp.value(lp).unwrap().data()[0];

        let mut gm = Graph::new();
        let im = gm.variable(Tensor::from_vec(vec![1, 4, 4, 1], im_data).unwrap());
        let wm = gm.variable(Tensor::from_vec(vec![2, 2, 1, 1], weight_data.clone()).unwrap());
        let om = gm.conv2d_nhwc(im, wm, None, 1, 1).unwrap();
        let lm = gm.sum(om).unwrap();
        let loss_m = gm.value(lm).unwrap().data()[0];

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_ig[ii] - numerical).abs() < 1e-2,
            "input numerical grad mismatch at {ii}: analytic={}, numerical={numerical}",
            analytic_ig[ii]
        );
    }
}

#[test]
fn end_to_end_mini_cnn_forward_backward() {
    let mut graph = Graph::new();

    // Build a tiny CNN: conv2d -> relu -> avgpool -> flatten -> matmul -> sum
    // Input: [1, 4, 4, 1]
    let input_data: Vec<f32> = (0..16).map(|v| v as f32 / 16.0).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 4, 4, 1], input_data).unwrap());

    // Conv: [2,2,1,2] -> 2 output channels
    let w_conv = graph.variable(
        Tensor::from_vec(
            vec![2, 2, 1, 2],
            vec![0.1, -0.1, 0.2, 0.2, -0.1, 0.3, 0.15, -0.15],
        )
        .unwrap(),
    );
    let conv_out = graph.conv2d_nhwc(input, w_conv, None, 1, 1).unwrap();
    // [1, 3, 3, 2]
    assert_eq!(graph.value(conv_out).unwrap().shape(), &[1, 3, 3, 2]);

    let relu_out = graph.relu(conv_out).unwrap();

    // AvgPool 3x3 stride 1 -> [1,1,1,2]
    let pool_out = graph.avg_pool2d_nhwc(relu_out, 3, 3, 1, 1).unwrap();
    assert_eq!(graph.value(pool_out).unwrap().shape(), &[1, 1, 1, 2]);

    let flat = graph.flatten(pool_out).unwrap();
    assert_eq!(graph.value(flat).unwrap().shape(), &[1, 2]);

    // Linear: [2, 1]
    let w_fc = graph.variable(Tensor::from_vec(vec![2, 1], vec![1.0, -1.0]).unwrap());
    let logit = graph.matmul_2d(flat, w_fc).unwrap();
    assert_eq!(graph.value(logit).unwrap().shape(), &[1, 1]);

    let loss = graph.sum(logit).unwrap();
    graph.backward(loss).unwrap();

    // All trainable nodes should have gradients
    assert!(graph.grad(input).unwrap().is_some());
    assert!(graph.grad(w_conv).unwrap().is_some());
    assert!(graph.grad(w_fc).unwrap().is_some());

    // Sanity: gradients should be finite
    for &g in graph.grad(w_conv).unwrap().unwrap().data() {
        assert!(g.is_finite(), "conv weight grad not finite: {g}");
    }
    for &g in graph.grad(w_fc).unwrap().unwrap().data() {
        assert!(g.is_finite(), "fc weight grad not finite: {g}");
    }
}

#[test]
fn depthwise_conv2d_nhwc_forward_and_backward() {
    let mut g = Graph::new();
    // Input: [1, 3, 3, 2] (batch=1, 3x3, 2 channels)
    let input_data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let x = g.variable(Tensor::from_vec(vec![1, 3, 3, 2], input_data).unwrap());
    // Weight: [2, 2, 2, 1] (2x2 kernel, 2 channels, depth_multiplier=1)
    let w = g.variable(Tensor::filled(vec![2, 2, 2, 1], 1.0).unwrap());
    let out = g.depthwise_conv2d_nhwc(x, w, None, 1, 1).unwrap();
    let out_shape = g.value(out).unwrap().shape().to_vec();
    // Output: [1, 2, 2, 2]
    assert_eq!(out_shape, &[1, 2, 2, 2]);
    // Verify backward runs without error
    let loss = g.sum(out).unwrap();
    g.backward(loss).unwrap();
    let grad_x = g.grad(x).unwrap().unwrap();
    let grad_w = g.grad(w).unwrap().unwrap();
    assert_eq!(grad_x.shape(), &[1, 3, 3, 2]);
    assert_eq!(grad_w.shape(), &[2, 2, 2, 1]);
}
