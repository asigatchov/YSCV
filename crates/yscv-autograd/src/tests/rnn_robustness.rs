use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn rnn_long_sequence_256() {
    let mut graph = Graph::new();
    let seq_len = 256;
    let input_size = 32;
    let hidden_size = 64;

    // Use small weights to avoid gradient explosion in long sequences
    let input_data = Tensor::randn(vec![seq_len, input_size], 42).unwrap();
    let wih_data = {
        let mut t = Tensor::randn(vec![input_size, hidden_size], 43).unwrap();
        let scale = 1.0 / (input_size as f32).sqrt();
        for v in t.data_mut() {
            *v *= scale;
        }
        t
    };
    let whh_data = {
        let mut t = Tensor::randn(vec![hidden_size, hidden_size], 44).unwrap();
        let scale = 1.0 / (hidden_size as f32).sqrt();
        for v in t.data_mut() {
            *v *= scale;
        }
        t
    };

    let input = graph.variable(input_data);
    let w_ih = graph.variable(wih_data);
    let w_hh = graph.variable(whh_data);
    let bias = graph.variable(Tensor::zeros(vec![hidden_size]).unwrap());

    let output = graph
        .rnn_forward(input, w_ih, w_hh, bias)
        .expect("rnn forward");
    let loss = graph.sum(output).expect("sum");
    graph.backward(loss).expect("backward");

    for &node in &[w_ih, w_hh, bias] {
        let grad = graph.grad(node).expect("grad lookup").expect("grad exists");
        for &v in grad.data() {
            assert!(v.is_finite(), "gradient is NaN/Inf at seq_len={}", seq_len);
        }
    }
}

#[test]
fn lstm_long_sequence_512() {
    let mut graph = Graph::new();
    let seq_len = 512;
    let input_size = 32;
    let hidden_size = 128;

    let input_data = Tensor::randn(vec![seq_len, input_size], 100).unwrap();
    let wih_data = {
        let mut t = Tensor::randn(vec![input_size, 4 * hidden_size], 101).unwrap();
        let scale = 1.0 / (input_size as f32).sqrt();
        for v in t.data_mut() {
            *v *= scale;
        }
        t
    };
    let whh_data = {
        let mut t = Tensor::randn(vec![hidden_size, 4 * hidden_size], 102).unwrap();
        let scale = 1.0 / (hidden_size as f32).sqrt();
        for v in t.data_mut() {
            *v *= scale;
        }
        t
    };

    let input = graph.variable(input_data);
    let w_ih = graph.variable(wih_data);
    let w_hh = graph.variable(whh_data);
    let bias = graph.variable(Tensor::zeros(vec![4 * hidden_size]).unwrap());

    let output = graph
        .lstm_forward(input, w_ih, w_hh, bias)
        .expect("lstm forward");
    let loss = graph.sum(output).expect("sum");
    graph.backward(loss).expect("backward");

    for &node in &[w_ih, w_hh, bias] {
        let grad = graph.grad(node).expect("grad lookup").expect("grad exists");
        for &v in grad.data() {
            assert!(
                v.is_finite(),
                "LSTM gradient is NaN/Inf at seq_len={}",
                seq_len
            );
        }
    }
}

#[test]
fn gradient_clipping_prevents_explosion() {
    let mut graph = Graph::new();

    // Create variables and produce large gradients via scaling
    let x = graph.variable(Tensor::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap());
    let w = graph.variable(Tensor::from_vec(vec![4], vec![5.0, 5.0, 5.0, 5.0]).unwrap());
    let product = graph.mul(x, w).unwrap();
    let loss = graph.sum(product).unwrap();
    graph.backward(loss).unwrap();

    let params = vec![x, w];
    let norm_before = graph.clip_grad_norm(&params, 1.0);
    assert!(
        norm_before > 1.0,
        "expected large norm before clipping, got {}",
        norm_before
    );

    // Verify all grads now have total norm <= 1.0 (within epsilon)
    let mut total_sq = 0.0f32;
    for &p in &params {
        if let Some(grad) = graph.grad(p).unwrap() {
            for &v in grad.data() {
                total_sq += v * v;
            }
        }
    }
    let total_norm_after = total_sq.sqrt();
    assert!(
        (total_norm_after - 1.0).abs() < 1e-5,
        "expected norm ~1.0 after clipping, got {}",
        total_norm_after,
    );
}

#[test]
fn gradient_value_clipping() {
    let mut graph = Graph::new();

    let x = graph.variable(Tensor::from_vec(vec![4], vec![10.0, -20.0, 30.0, -40.0]).unwrap());
    let loss = graph.sum(x).unwrap();
    graph.backward(loss).unwrap();

    // Manually set large gradients
    let big_grad = Tensor::from_vec(vec![4], vec![100.0, -200.0, 0.5, -0.3]).unwrap();
    graph.set_grad(x, big_grad).unwrap();

    let params = vec![x];
    graph.clip_grad_value(&params, 1.0);

    let grad = graph.grad(x).unwrap().unwrap();
    let data = grad.data();
    assert_eq!(data.len(), 4);
    assert!(
        (data[0] - 1.0).abs() < 1e-6,
        "expected 1.0, got {}",
        data[0]
    );
    assert!(
        (data[1] - (-1.0)).abs() < 1e-6,
        "expected -1.0, got {}",
        data[1]
    );
    assert!(
        (data[2] - 0.5).abs() < 1e-6,
        "expected 0.5, got {}",
        data[2]
    );
    assert!(
        (data[3] - (-0.3)).abs() < 1e-6,
        "expected -0.3, got {}",
        data[3]
    );
}
