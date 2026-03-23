use yscv_tensor::Tensor;

use crate::Graph;

// ---------- Conv1d backward tests ----------

#[test]
fn backward_conv1d_nlc_computes_weight_and_input_grads() {
    let mut graph = Graph::new();
    // input: [1, 5, 2], weight: [3, 2, 1], no bias, stride=1
    let input_data: Vec<f32> = (1..=10).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 5, 2], input_data).unwrap());
    let weight = graph
        .variable(Tensor::from_vec(vec![3, 2, 1], vec![1.0, -1.0, 0.5, 0.5, -0.5, 1.0]).unwrap());

    let out = graph.conv1d_nlc(input, weight, None, 1).unwrap();
    // out_len = (5-3)/1+1 = 3, shape: [1, 3, 1]
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 3, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let w_grad = graph.grad(weight).unwrap().unwrap();
    assert_eq!(w_grad.shape(), &[3, 2, 1]);

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 5, 2]);

    // All gradients should be finite
    for &g in w_grad.data() {
        assert!(g.is_finite(), "weight grad not finite: {g}");
    }
    for &g in i_grad.data() {
        assert!(g.is_finite(), "input grad not finite: {g}");
    }
}

#[test]
fn backward_conv1d_nlc_with_bias() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 4, 1], 1.0).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 1, 1], 0.5).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![1], vec![0.1]).unwrap());

    let out = graph.conv1d_nlc(input, weight, Some(bias), 1).unwrap();
    // out_len = (4-2)/1+1 = 3
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 3, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    // bias grad = number of output elements = 3
    let b_grad = graph.grad(bias).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[1]);
    assert!((b_grad.data()[0] - 3.0).abs() < 1e-4);
}

#[test]
fn backward_conv1d_nlc_numerical_gradient_check() {
    let eps = 1e-3;
    let input_data: Vec<f32> = vec![0.5, 1.2, 0.3, 0.8, 1.0, 0.7];
    let weight_data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];

    // Compute analytical gradients
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 3, 2], input_data.clone()).unwrap());
    let weight = graph.variable(Tensor::from_vec(vec![2, 2, 1], weight_data.clone()).unwrap());
    let out = graph.conv1d_nlc(input, weight, None, 1).unwrap();
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
        let ip = gp.variable(Tensor::from_vec(vec![1, 3, 2], input_data.clone()).unwrap());
        let wp_node = gp.variable(Tensor::from_vec(vec![2, 2, 1], wp).unwrap());
        let op = gp.conv1d_nlc(ip, wp_node, None, 1).unwrap();
        let lp = gp.sum(op).unwrap();
        let loss_p = gp.value(lp).unwrap().data()[0];

        let mut gm = Graph::new();
        let im = gm.variable(Tensor::from_vec(vec![1, 3, 2], input_data.clone()).unwrap());
        let wm_node = gm.variable(Tensor::from_vec(vec![2, 2, 1], wm).unwrap());
        let om = gm.conv1d_nlc(im, wm_node, None, 1).unwrap();
        let lm = gm.sum(om).unwrap();
        let loss_m = gm.value(lm).unwrap().data()[0];

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_wg[wi] - numerical).abs() < 1e-2,
            "weight numerical grad mismatch at {wi}: analytic={}, numerical={numerical}",
            analytic_wg[wi]
        );
    }

    // Numerical gradient check for input (spot check first 4)
    for ii in 0..4 {
        let mut ip_data = input_data.clone();
        ip_data[ii] += eps;
        let mut im_data = input_data.clone();
        im_data[ii] -= eps;

        let mut gp = Graph::new();
        let ip = gp.variable(Tensor::from_vec(vec![1, 3, 2], ip_data).unwrap());
        let wp = gp.variable(Tensor::from_vec(vec![2, 2, 1], weight_data.clone()).unwrap());
        let op = gp.conv1d_nlc(ip, wp, None, 1).unwrap();
        let lp = gp.sum(op).unwrap();
        let loss_p = gp.value(lp).unwrap().data()[0];

        let mut gm = Graph::new();
        let im = gm.variable(Tensor::from_vec(vec![1, 3, 2], im_data).unwrap());
        let wm = gm.variable(Tensor::from_vec(vec![2, 2, 1], weight_data.clone()).unwrap());
        let om = gm.conv1d_nlc(im, wm, None, 1).unwrap();
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

// ---------- Conv3d backward tests ----------

#[test]
fn backward_conv3d_ndhwc_computes_weight_and_input_grads() {
    let mut graph = Graph::new();
    // input: [1, 3, 3, 3, 1], weight: [2, 2, 2, 1, 1], no bias, stride=1
    let input_data: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 3, 3, 3, 1], input_data).unwrap());
    let weight = graph.variable(
        Tensor::from_vec(
            vec![2, 2, 2, 1, 1],
            vec![1.0, -1.0, 0.5, 0.5, -0.5, 1.0, 0.0, -0.5],
        )
        .unwrap(),
    );

    let out = graph.conv3d_ndhwc(input, weight, None, 1, 1, 1).unwrap();
    // out: [1, 2, 2, 2, 1]
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let w_grad = graph.grad(weight).unwrap().unwrap();
    assert_eq!(w_grad.shape(), &[2, 2, 2, 1, 1]);

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 3, 3, 3, 1]);

    for &g in w_grad.data() {
        assert!(g.is_finite(), "weight grad not finite: {g}");
    }
    for &g in i_grad.data() {
        assert!(g.is_finite(), "input grad not finite: {g}");
    }
}

#[test]
fn backward_conv3d_ndhwc_with_bias() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 3, 3, 3, 1], 1.0).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 2, 1, 1], 0.5).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![1], vec![0.1]).unwrap());

    let out = graph
        .conv3d_ndhwc(input, weight, Some(bias), 1, 1, 1)
        .unwrap();
    // out: [1, 2, 2, 2, 1]
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    // bias grad = number of output spatial elements = 2*2*2 = 8
    let b_grad = graph.grad(bias).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[1]);
    assert!((b_grad.data()[0] - 8.0).abs() < 1e-4);
}

#[test]
fn backward_conv3d_ndhwc_numerical_gradient_check() {
    let eps = 1e-3;
    // Small 3D input: [1, 2, 2, 2, 1]
    let input_data: Vec<f32> = vec![0.5, 1.2, 0.3, 0.8, 1.0, 0.7, 0.4, 0.9];
    // Kernel: [2, 2, 2, 1, 1]
    let weight_data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3];

    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 2, 2, 2, 1], input_data.clone()).unwrap());
    let weight =
        graph.variable(Tensor::from_vec(vec![2, 2, 2, 1, 1], weight_data.clone()).unwrap());
    let out = graph.conv3d_ndhwc(input, weight, None, 1, 1, 1).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_wg = graph.grad(weight).unwrap().unwrap().data().to_vec();
    let analytic_ig = graph.grad(input).unwrap().unwrap().data().to_vec();

    // Numerical gradient check for weight
    for wi in 0..8 {
        let mut wp = weight_data.clone();
        wp[wi] += eps;
        let mut wm = weight_data.clone();
        wm[wi] -= eps;

        let mut gp = Graph::new();
        let ip = gp.variable(Tensor::from_vec(vec![1, 2, 2, 2, 1], input_data.clone()).unwrap());
        let wp_node = gp.variable(Tensor::from_vec(vec![2, 2, 2, 1, 1], wp).unwrap());
        let op = gp.conv3d_ndhwc(ip, wp_node, None, 1, 1, 1).unwrap();
        let lp = gp.sum(op).unwrap();
        let loss_p = gp.value(lp).unwrap().data()[0];

        let mut gm = Graph::new();
        let im = gm.variable(Tensor::from_vec(vec![1, 2, 2, 2, 1], input_data.clone()).unwrap());
        let wm_node = gm.variable(Tensor::from_vec(vec![2, 2, 2, 1, 1], wm).unwrap());
        let om = gm.conv3d_ndhwc(im, wm_node, None, 1, 1, 1).unwrap();
        let lm = gm.sum(om).unwrap();
        let loss_m = gm.value(lm).unwrap().data()[0];

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_wg[wi] - numerical).abs() < 1e-2,
            "weight numerical grad mismatch at {wi}: analytic={}, numerical={numerical}",
            analytic_wg[wi]
        );
    }

    // Numerical gradient check for input
    for ii in 0..8 {
        let mut ip_data = input_data.clone();
        ip_data[ii] += eps;
        let mut im_data = input_data.clone();
        im_data[ii] -= eps;

        let mut gp = Graph::new();
        let ip = gp.variable(Tensor::from_vec(vec![1, 2, 2, 2, 1], ip_data).unwrap());
        let wp = gp.variable(Tensor::from_vec(vec![2, 2, 2, 1, 1], weight_data.clone()).unwrap());
        let op = gp.conv3d_ndhwc(ip, wp, None, 1, 1, 1).unwrap();
        let lp = gp.sum(op).unwrap();
        let loss_p = gp.value(lp).unwrap().data()[0];

        let mut gm = Graph::new();
        let im = gm.variable(Tensor::from_vec(vec![1, 2, 2, 2, 1], im_data).unwrap());
        let wm = gm.variable(Tensor::from_vec(vec![2, 2, 2, 1, 1], weight_data.clone()).unwrap());
        let om = gm.conv3d_ndhwc(im, wm, None, 1, 1, 1).unwrap();
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

// ---------- Scaled dot-product attention backward tests ----------

#[test]
fn backward_scaled_dot_product_attention_shapes() {
    let mut graph = Graph::new();
    // Q: [3, 4], K: [5, 4], V: [5, 2]
    let q_data: Vec<f32> = (0..12).map(|v| v as f32 * 0.1).collect();
    let k_data: Vec<f32> = (0..20).map(|v| v as f32 * 0.1).collect();
    let v_data: Vec<f32> = (0..10).map(|v| v as f32 * 0.1).collect();

    let q = graph.variable(Tensor::from_vec(vec![3, 4], q_data).unwrap());
    let k = graph.variable(Tensor::from_vec(vec![5, 4], k_data).unwrap());
    let v = graph.variable(Tensor::from_vec(vec![5, 2], v_data).unwrap());

    let out = graph.scaled_dot_product_attention(q, k, v).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[3, 2]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let q_grad = graph.grad(q).unwrap().unwrap();
    let k_grad = graph.grad(k).unwrap().unwrap();
    let v_grad = graph.grad(v).unwrap().unwrap();

    assert_eq!(q_grad.shape(), &[3, 4]);
    assert_eq!(k_grad.shape(), &[5, 4]);
    assert_eq!(v_grad.shape(), &[5, 2]);

    for &g in q_grad.data() {
        assert!(g.is_finite(), "q grad not finite: {g}");
    }
    for &g in k_grad.data() {
        assert!(g.is_finite(), "k grad not finite: {g}");
    }
    for &g in v_grad.data() {
        assert!(g.is_finite(), "v grad not finite: {g}");
    }
}

#[test]
fn backward_attention_numerical_gradient_check() {
    let eps = 1e-3;
    // Small attention: Q [2,2], K [2,2], V [2,2]
    let q_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let k_data: Vec<f32> = vec![0.5, 0.6, 0.7, 0.8];
    let v_data: Vec<f32> = vec![0.9, 1.0, 1.1, 1.2];

    // Analytical
    let mut graph = Graph::new();
    let q = graph.variable(Tensor::from_vec(vec![2, 2], q_data.clone()).unwrap());
    let k = graph.variable(Tensor::from_vec(vec![2, 2], k_data.clone()).unwrap());
    let v = graph.variable(Tensor::from_vec(vec![2, 2], v_data.clone()).unwrap());
    let out = graph.scaled_dot_product_attention(q, k, v).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_qg = graph.grad(q).unwrap().unwrap().data().to_vec();
    let analytic_kg = graph.grad(k).unwrap().unwrap().data().to_vec();
    let analytic_vg = graph.grad(v).unwrap().unwrap().data().to_vec();

    // Numerical gradient check for Q
    for qi in 0..4 {
        let mut qp = q_data.clone();
        qp[qi] += eps;
        let mut qm = q_data.clone();
        qm[qi] -= eps;

        let lp = {
            let mut g = Graph::new();
            let q = g.variable(Tensor::from_vec(vec![2, 2], qp).unwrap());
            let k = g.variable(Tensor::from_vec(vec![2, 2], k_data.clone()).unwrap());
            let v = g.variable(Tensor::from_vec(vec![2, 2], v_data.clone()).unwrap());
            let out = g.scaled_dot_product_attention(q, k, v).unwrap();
            let l = g.sum(out).unwrap();
            g.value(l).unwrap().data()[0]
        };
        let lm = {
            let mut g = Graph::new();
            let q = g.variable(Tensor::from_vec(vec![2, 2], qm).unwrap());
            let k = g.variable(Tensor::from_vec(vec![2, 2], k_data.clone()).unwrap());
            let v = g.variable(Tensor::from_vec(vec![2, 2], v_data.clone()).unwrap());
            let out = g.scaled_dot_product_attention(q, k, v).unwrap();
            let l = g.sum(out).unwrap();
            g.value(l).unwrap().data()[0]
        };
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_qg[qi] - numerical).abs() < 1e-2,
            "Q numerical grad mismatch at {qi}: analytic={}, numerical={numerical}",
            analytic_qg[qi]
        );
    }

    // Numerical gradient check for K
    for ki in 0..4 {
        let mut kp = k_data.clone();
        kp[ki] += eps;
        let mut km = k_data.clone();
        km[ki] -= eps;

        let lp = {
            let mut g = Graph::new();
            let q = g.variable(Tensor::from_vec(vec![2, 2], q_data.clone()).unwrap());
            let k = g.variable(Tensor::from_vec(vec![2, 2], kp).unwrap());
            let v = g.variable(Tensor::from_vec(vec![2, 2], v_data.clone()).unwrap());
            let out = g.scaled_dot_product_attention(q, k, v).unwrap();
            let l = g.sum(out).unwrap();
            g.value(l).unwrap().data()[0]
        };
        let lm = {
            let mut g = Graph::new();
            let q = g.variable(Tensor::from_vec(vec![2, 2], q_data.clone()).unwrap());
            let k = g.variable(Tensor::from_vec(vec![2, 2], km).unwrap());
            let v = g.variable(Tensor::from_vec(vec![2, 2], v_data.clone()).unwrap());
            let out = g.scaled_dot_product_attention(q, k, v).unwrap();
            let l = g.sum(out).unwrap();
            g.value(l).unwrap().data()[0]
        };
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_kg[ki] - numerical).abs() < 1e-2,
            "K numerical grad mismatch at {ki}: analytic={}, numerical={numerical}",
            analytic_kg[ki]
        );
    }

    // Numerical gradient check for V
    for vi in 0..4 {
        let mut vp = v_data.clone();
        vp[vi] += eps;
        let mut vm = v_data.clone();
        vm[vi] -= eps;

        let lp = {
            let mut g = Graph::new();
            let q = g.variable(Tensor::from_vec(vec![2, 2], q_data.clone()).unwrap());
            let k = g.variable(Tensor::from_vec(vec![2, 2], k_data.clone()).unwrap());
            let v = g.variable(Tensor::from_vec(vec![2, 2], vp).unwrap());
            let out = g.scaled_dot_product_attention(q, k, v).unwrap();
            let l = g.sum(out).unwrap();
            g.value(l).unwrap().data()[0]
        };
        let lm = {
            let mut g = Graph::new();
            let q = g.variable(Tensor::from_vec(vec![2, 2], q_data.clone()).unwrap());
            let k = g.variable(Tensor::from_vec(vec![2, 2], k_data.clone()).unwrap());
            let v = g.variable(Tensor::from_vec(vec![2, 2], vm).unwrap());
            let out = g.scaled_dot_product_attention(q, k, v).unwrap();
            let l = g.sum(out).unwrap();
            g.value(l).unwrap().data()[0]
        };
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_vg[vi] - numerical).abs() < 1e-2,
            "V numerical grad mismatch at {vi}: analytic={}, numerical={numerical}",
            analytic_vg[vi]
        );
    }
}
