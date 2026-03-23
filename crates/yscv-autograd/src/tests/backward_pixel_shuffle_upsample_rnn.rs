use yscv_tensor::Tensor;

use crate::Graph;

// ---- PixelShuffle backward ----

#[test]
fn backward_pixel_shuffle_basic() {
    let mut graph = Graph::new();
    // input [1, 1, 1, 4] with r=2 -> output [1, 2, 2, 1]
    let input =
        graph.variable(Tensor::from_vec(vec![1, 1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let out = graph.pixel_shuffle(input, 2).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 1, 1, 4]);
    // Each input element maps to exactly one output element, so grad = 1
    for &g in i_grad.data() {
        assert!(
            (g - 1.0).abs() < 1e-6,
            "pixel shuffle grad: got {g}, expected 1.0"
        );
    }
}

#[test]
fn backward_pixel_shuffle_2x2_input() {
    let mut graph = Graph::new();
    // input [1, 2, 2, 4], r=2 -> output [1, 4, 4, 1]
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 2, 2, 4], data).unwrap());
    let out = graph.pixel_shuffle(input, 2).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 4, 4, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 4]);
    // Pixel shuffle is a permutation, so each grad element is 1
    for &g in i_grad.data() {
        assert!((g - 1.0).abs() < 1e-6);
    }
}

#[test]
fn backward_pixel_shuffle_numerical() {
    let eps = 1e-3;
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![1, 1, 1, 4], input_data.clone()).unwrap());
    let out = graph.pixel_shuffle(input, 2).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_grad = graph.grad(input).unwrap().unwrap().data().to_vec();

    for idx in 0..4 {
        let mut dp = input_data.clone();
        dp[idx] += eps;
        let mut gp = Graph::new();
        let inp = gp.variable(Tensor::from_vec(vec![1, 1, 1, 4], dp).unwrap());
        let o = gp.pixel_shuffle(inp, 2).unwrap();
        let lp = gp.value(o).unwrap().sum();

        let mut dm = input_data.clone();
        dm[idx] -= eps;
        let mut gm = Graph::new();
        let inp = gm.variable(Tensor::from_vec(vec![1, 1, 1, 4], dm).unwrap());
        let o = gm.pixel_shuffle(inp, 2).unwrap();
        let lm = gm.value(o).unwrap().sum();

        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_grad[idx] - numerical).abs() < 1e-2,
            "pixel_shuffle numerical gradient mismatch at {idx}: analytic={}, numerical={}",
            analytic_grad[idx],
            numerical
        );
    }
}

// ---- UpsampleNearest backward ----

#[test]
fn backward_upsample_nearest_basic() {
    let mut graph = Graph::new();
    // input [1, 2, 2, 1], r=2 -> output [1, 4, 4, 1]
    let input =
        graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let out = graph.upsample_nearest(input, 2).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 4, 4, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 1]);
    // Each input element is repeated 2x2=4 times, so grad = 4
    for &g in i_grad.data() {
        assert!(
            (g - 4.0).abs() < 1e-6,
            "upsample nearest grad: got {g}, expected 4.0"
        );
    }
}

#[test]
fn backward_upsample_nearest_scale3() {
    let mut graph = Graph::new();
    // input [1, 1, 1, 2], r=3 -> output [1, 3, 3, 2]
    let input = graph.variable(Tensor::from_vec(vec![1, 1, 1, 2], vec![1.0, 2.0]).unwrap());
    let out = graph.upsample_nearest(input, 3).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 3, 3, 2]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    // Each element repeated 3x3=9 times, so grad = 9
    for &g in i_grad.data() {
        assert!(
            (g - 9.0).abs() < 1e-6,
            "upsample nearest r=3 grad: got {g}, expected 9.0"
        );
    }
}

// ---- RNN backward ----

#[test]
fn backward_rnn_produces_grads() {
    let mut graph = Graph::new();
    let input_size = 2;
    let hidden_size = 3;
    let seq_len = 4;

    let input = graph.variable(Tensor::filled(vec![seq_len, input_size], 0.5).unwrap());
    let w_ih = graph.variable(Tensor::filled(vec![input_size, hidden_size], 0.1).unwrap());
    let w_hh = graph.variable(Tensor::filled(vec![hidden_size, hidden_size], 0.1).unwrap());
    let bias = graph.variable(Tensor::zeros(vec![hidden_size]).unwrap());

    let out = graph.rnn_forward(input, w_ih, w_hh, bias).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[seq_len, hidden_size]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    assert!(graph.grad(input).unwrap().is_some());
    assert!(graph.grad(w_ih).unwrap().is_some());
    assert!(graph.grad(w_hh).unwrap().is_some());
    assert!(graph.grad(bias).unwrap().is_some());

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[seq_len, input_size]);
    // Check grads are not all zero
    assert!(i_grad.data().iter().any(|&g| g.abs() > 1e-8));
}

#[test]
fn backward_rnn_numerical_gradient_check() {
    let eps = 1e-3;
    let input_data = vec![0.5, -0.3, 0.2, 0.7];
    let wih_data = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1];
    let whh_data = vec![0.1, 0.05, -0.1, -0.05, 0.1, 0.02, 0.03, -0.1, 0.05];
    let bias_data = vec![0.0, 0.0, 0.0];

    // Analytic
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
    let w_ih = graph.variable(Tensor::from_vec(vec![2, 3], wih_data.clone()).unwrap());
    let w_hh = graph.variable(Tensor::from_vec(vec![3, 3], whh_data.clone()).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![3], bias_data.clone()).unwrap());
    let out = graph.rnn_forward(input, w_ih, w_hh, bias).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_wih = graph.grad(w_ih).unwrap().unwrap().data().to_vec();

    // Numerical for w_ih
    for idx in 0..wih_data.len() {
        let mut wp = wih_data.clone();
        wp[idx] += eps;
        let mut gp = Graph::new();
        let inp = gp.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
        let wih_p = gp.variable(Tensor::from_vec(vec![2, 3], wp).unwrap());
        let whh_p = gp.variable(Tensor::from_vec(vec![3, 3], whh_data.clone()).unwrap());
        let b_p = gp.variable(Tensor::from_vec(vec![3], bias_data.clone()).unwrap());
        let o = gp.rnn_forward(inp, wih_p, whh_p, b_p).unwrap();
        let lp = gp.value(o).unwrap().sum();

        let mut wm = wih_data.clone();
        wm[idx] -= eps;
        let mut gm = Graph::new();
        let inp = gm.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
        let wih_m = gm.variable(Tensor::from_vec(vec![2, 3], wm).unwrap());
        let whh_m = gm.variable(Tensor::from_vec(vec![3, 3], whh_data.clone()).unwrap());
        let b_m = gm.variable(Tensor::from_vec(vec![3], bias_data.clone()).unwrap());
        let o = gm.rnn_forward(inp, wih_m, whh_m, b_m).unwrap();
        let lm = gm.value(o).unwrap().sum();

        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_wih[idx] - numerical).abs() < 1e-2,
            "rnn w_ih grad mismatch at {idx}: analytic={}, numerical={}",
            analytic_wih[idx],
            numerical
        );
    }
}

// ---- LSTM backward ----

#[test]
fn backward_lstm_produces_grads() {
    let mut graph = Graph::new();
    let input_size = 2;
    let hidden_size = 3;
    let seq_len = 4;

    let input = graph.variable(Tensor::filled(vec![seq_len, input_size], 0.5).unwrap());
    let w_ih = graph.variable(Tensor::filled(vec![input_size, 4 * hidden_size], 0.05).unwrap());
    let w_hh = graph.variable(Tensor::filled(vec![hidden_size, 4 * hidden_size], 0.05).unwrap());
    let bias = graph.variable(Tensor::zeros(vec![4 * hidden_size]).unwrap());

    let out = graph.lstm_forward(input, w_ih, w_hh, bias).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[seq_len, hidden_size]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    assert!(graph.grad(input).unwrap().is_some());
    assert!(graph.grad(w_ih).unwrap().is_some());
    assert!(graph.grad(w_hh).unwrap().is_some());
    assert!(graph.grad(bias).unwrap().is_some());

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[seq_len, input_size]);
    assert!(i_grad.data().iter().any(|&g| g.abs() > 1e-8));
}

#[test]
fn backward_lstm_numerical_gradient_check() {
    let eps = 1e-3;
    let input_data = vec![0.5, -0.3, 0.2, 0.7];
    let input_size = 2;
    let hidden_size = 2;
    let h4 = 4 * hidden_size;

    let mut wih_data = vec![0.0f32; input_size * h4];
    for (i, v) in wih_data.iter_mut().enumerate() {
        *v = ((i as f32 * 0.1) - 0.4).clamp(-0.3, 0.3);
    }
    let mut whh_data = vec![0.0f32; hidden_size * h4];
    for (i, v) in whh_data.iter_mut().enumerate() {
        *v = ((i as f32 * 0.05) - 0.2).clamp(-0.2, 0.2);
    }
    let bias_data = vec![0.0f32; h4];

    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
    let w_ih = graph.variable(Tensor::from_vec(vec![input_size, h4], wih_data.clone()).unwrap());
    let w_hh = graph.variable(Tensor::from_vec(vec![hidden_size, h4], whh_data.clone()).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![h4], bias_data.clone()).unwrap());
    let out = graph.lstm_forward(input, w_ih, w_hh, bias).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_wih = graph.grad(w_ih).unwrap().unwrap().data().to_vec();

    for idx in 0..wih_data.len() {
        let mut wp = wih_data.clone();
        wp[idx] += eps;
        let mut gp = Graph::new();
        let inp = gp.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
        let wih_p = gp.variable(Tensor::from_vec(vec![input_size, h4], wp).unwrap());
        let whh_p = gp.variable(Tensor::from_vec(vec![hidden_size, h4], whh_data.clone()).unwrap());
        let b_p = gp.variable(Tensor::from_vec(vec![h4], bias_data.clone()).unwrap());
        let o = gp.lstm_forward(inp, wih_p, whh_p, b_p).unwrap();
        let lp = gp.value(o).unwrap().sum();

        let mut wm = wih_data.clone();
        wm[idx] -= eps;
        let mut gm = Graph::new();
        let inp = gm.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
        let wih_m = gm.variable(Tensor::from_vec(vec![input_size, h4], wm).unwrap());
        let whh_m = gm.variable(Tensor::from_vec(vec![hidden_size, h4], whh_data.clone()).unwrap());
        let b_m = gm.variable(Tensor::from_vec(vec![h4], bias_data.clone()).unwrap());
        let o = gm.lstm_forward(inp, wih_m, whh_m, b_m).unwrap();
        let lm = gm.value(o).unwrap().sum();

        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_wih[idx] - numerical).abs() < 5e-2,
            "lstm w_ih grad mismatch at {idx}: analytic={}, numerical={}",
            analytic_wih[idx],
            numerical
        );
    }
}

// ---- GRU backward ----

#[test]
fn backward_gru_produces_grads() {
    let mut graph = Graph::new();
    let input_size = 2;
    let hidden_size = 3;
    let seq_len = 4;
    let h3 = 3 * hidden_size;

    let input = graph.variable(Tensor::filled(vec![seq_len, input_size], 0.5).unwrap());
    let w_ih = graph.variable(Tensor::filled(vec![input_size, h3], 0.05).unwrap());
    let w_hh = graph.variable(Tensor::filled(vec![hidden_size, h3], 0.05).unwrap());
    let bias_ih = graph.variable(Tensor::zeros(vec![h3]).unwrap());
    let bias_hh = graph.variable(Tensor::zeros(vec![h3]).unwrap());

    let out = graph
        .gru_forward(input, w_ih, w_hh, bias_ih, bias_hh)
        .unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[seq_len, hidden_size]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    assert!(graph.grad(input).unwrap().is_some());
    assert!(graph.grad(w_ih).unwrap().is_some());
    assert!(graph.grad(w_hh).unwrap().is_some());
    assert!(graph.grad(bias_ih).unwrap().is_some());
    assert!(graph.grad(bias_hh).unwrap().is_some());

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[seq_len, input_size]);
    assert!(i_grad.data().iter().any(|&g| g.abs() > 1e-8));
}

#[test]
fn backward_gru_numerical_gradient_check() {
    let eps = 1e-3;
    let input_data = vec![0.5, -0.3, 0.2, 0.7];
    let input_size = 2;
    let hidden_size = 2;
    let h3 = 3 * hidden_size;

    let mut wih_data = vec![0.0f32; input_size * h3];
    for (i, v) in wih_data.iter_mut().enumerate() {
        *v = ((i as f32 * 0.1) - 0.3).clamp(-0.3, 0.3);
    }
    let mut whh_data = vec![0.0f32; hidden_size * h3];
    for (i, v) in whh_data.iter_mut().enumerate() {
        *v = ((i as f32 * 0.05) - 0.15).clamp(-0.2, 0.2);
    }
    let bih_data = vec![0.0f32; h3];
    let bhh_data = vec![0.0f32; h3];

    let mut graph = Graph::new();
    let input = graph.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
    let w_ih = graph.variable(Tensor::from_vec(vec![input_size, h3], wih_data.clone()).unwrap());
    let w_hh = graph.variable(Tensor::from_vec(vec![hidden_size, h3], whh_data.clone()).unwrap());
    let bih = graph.variable(Tensor::from_vec(vec![h3], bih_data.clone()).unwrap());
    let bhh = graph.variable(Tensor::from_vec(vec![h3], bhh_data.clone()).unwrap());
    let out = graph.gru_forward(input, w_ih, w_hh, bih, bhh).unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let analytic_wih = graph.grad(w_ih).unwrap().unwrap().data().to_vec();

    for idx in 0..wih_data.len() {
        let mut wp = wih_data.clone();
        wp[idx] += eps;
        let mut gp = Graph::new();
        let inp = gp.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
        let wih_p = gp.variable(Tensor::from_vec(vec![input_size, h3], wp).unwrap());
        let whh_p = gp.variable(Tensor::from_vec(vec![hidden_size, h3], whh_data.clone()).unwrap());
        let bih_p = gp.variable(Tensor::from_vec(vec![h3], bih_data.clone()).unwrap());
        let bhh_p = gp.variable(Tensor::from_vec(vec![h3], bhh_data.clone()).unwrap());
        let o = gp.gru_forward(inp, wih_p, whh_p, bih_p, bhh_p).unwrap();
        let lp = gp.value(o).unwrap().sum();

        let mut wm = wih_data.clone();
        wm[idx] -= eps;
        let mut gm = Graph::new();
        let inp = gm.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
        let wih_m = gm.variable(Tensor::from_vec(vec![input_size, h3], wm).unwrap());
        let whh_m = gm.variable(Tensor::from_vec(vec![hidden_size, h3], whh_data.clone()).unwrap());
        let bih_m = gm.variable(Tensor::from_vec(vec![h3], bih_data.clone()).unwrap());
        let bhh_m = gm.variable(Tensor::from_vec(vec![h3], bhh_data.clone()).unwrap());
        let o = gm.gru_forward(inp, wih_m, whh_m, bih_m, bhh_m).unwrap();
        let lm = gm.value(o).unwrap().sum();

        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic_wih[idx] - numerical).abs() < 5e-2,
            "gru w_ih grad mismatch at {idx}: analytic={}, numerical={}",
            analytic_wih[idx],
            numerical
        );
    }
}

// ---- DeformableConv2d backward ----

#[test]
fn backward_deformable_conv2d_produces_grads() {
    let mut graph = Graph::new();
    // input [1, 3, 3, 1], weight [2, 2, 1, 1], zero offsets, stride=1, padding=0
    // output = [1, 2, 2, 1]
    let input = graph
        .variable(Tensor::from_vec(vec![1, 3, 3, 1], (1..=9).map(|v| v as f32).collect()).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 1, 1], 0.25).unwrap());
    // offsets: [1, 2, 2, 2*2*2] = [1, 2, 2, 8], all zeros
    let offsets = graph.variable(Tensor::zeros(vec![1, 2, 2, 8]).unwrap());

    let out = graph
        .deformable_conv2d_nhwc(input, weight, offsets, None, 1, 0)
        .unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 3, 3, 1]);
    assert!(i_grad.data().iter().any(|&g| g.abs() > 1e-8));

    let w_grad = graph.grad(weight).unwrap().unwrap();
    assert_eq!(w_grad.shape(), &[2, 2, 1, 1]);
    assert!(w_grad.data().iter().any(|&g| g.abs() > 1e-8));
}

#[test]
fn backward_deformable_conv2d_with_bias() {
    let mut graph = Graph::new();
    let input = graph.variable(Tensor::filled(vec![1, 3, 3, 1], 1.0).unwrap());
    let weight = graph.variable(Tensor::filled(vec![2, 2, 1, 1], 0.5).unwrap());
    let offsets = graph.variable(Tensor::zeros(vec![1, 2, 2, 8]).unwrap());
    let bias = graph.variable(Tensor::from_vec(vec![1], vec![0.1]).unwrap());

    let out = graph
        .deformable_conv2d_nhwc(input, weight, offsets, Some(bias), 1, 0)
        .unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let b_grad = graph.grad(bias).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[1]);
    // bias grad = number of output elements = 2*2 = 4
    assert!((b_grad.data()[0] - 4.0).abs() < 1e-4);
}

#[test]
fn backward_deformable_conv2d_zero_offsets_matches_standard_conv() {
    // With zero offsets, deformable conv should produce same result as standard conv (no padding)
    let mut graph = Graph::new();
    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let weight_data = vec![0.25f32; 4];

    let input = graph.variable(Tensor::from_vec(vec![1, 3, 3, 1], input_data.clone()).unwrap());
    let weight = graph.variable(Tensor::from_vec(vec![2, 2, 1, 1], weight_data.clone()).unwrap());
    let offsets = graph.variable(Tensor::zeros(vec![1, 2, 2, 8]).unwrap());
    let out = graph
        .deformable_conv2d_nhwc(input, weight, offsets, None, 1, 0)
        .unwrap();
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let deform_w_grad = graph.grad(weight).unwrap().unwrap().data().to_vec();

    // Standard conv2d
    let mut graph2 = Graph::new();
    let input2 = graph2.variable(Tensor::from_vec(vec![1, 3, 3, 1], input_data).unwrap());
    let weight2 = graph2.variable(Tensor::from_vec(vec![2, 2, 1, 1], weight_data).unwrap());
    let out2 = graph2.conv2d_nhwc(input2, weight2, None, 1, 1).unwrap();
    let loss2 = graph2.sum(out2).unwrap();
    graph2.backward(loss2).unwrap();
    let standard_w_grad = graph2.grad(weight2).unwrap().unwrap().data().to_vec();

    for (i, (&d, &s)) in deform_w_grad.iter().zip(standard_w_grad.iter()).enumerate() {
        assert!(
            (d - s).abs() < 1e-3,
            "deformable vs standard weight grad mismatch at {i}: deform={d}, standard={s}"
        );
    }
}
