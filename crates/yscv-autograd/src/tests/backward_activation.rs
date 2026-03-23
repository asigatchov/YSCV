use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_relu_masks_negative_inputs() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap());
    let y = graph.relu(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.data(), &[0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn backward_exp_multiplies_by_exp_output() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap());
    let y = graph.exp(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    // d(exp(x))/dx = exp(x)
    let x_grad = graph.grad(x).unwrap().unwrap();
    assert!((x_grad.data()[0] - 1.0).abs() < 1e-5); // exp(0)=1
    assert!((x_grad.data()[1] - std::f32::consts::E).abs() < 1e-4);
}

#[test]
fn backward_log_divides_by_input() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let y = graph.log(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    // d(ln(x))/dx = 1/x
    let x_grad = graph.grad(x).unwrap().unwrap();
    assert!((x_grad.data()[0] - 1.0).abs() < 1e-5);
    assert!((x_grad.data()[1] - 0.5).abs() < 1e-5);
}

#[test]
fn backward_sqrt_computes_half_reciprocal_sqrt() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![4.0, 9.0]).unwrap());
    let y = graph.sqrt(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    // d(sqrt(x))/dx = 0.5 / sqrt(x)
    let x_grad = graph.grad(x).unwrap().unwrap();
    assert!((x_grad.data()[0] - 0.25).abs() < 1e-5); // 0.5/2
    assert!((x_grad.data()[1] - 1.0 / 6.0).abs() < 1e-5); // 0.5/3
}

#[test]
fn backward_sigmoid_computes_s_times_1_minus_s() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![0.0, 2.0]).unwrap());
    let y = graph.sigmoid(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let y_val = graph.value(y).unwrap();
    let x_grad = graph.grad(x).unwrap().unwrap();
    for i in 0..2 {
        let s = y_val.data()[i];
        let expected = s * (1.0 - s);
        assert!(
            (x_grad.data()[i] - expected).abs() < 1e-5,
            "sigmoid grad mismatch at {i}"
        );
    }
}

#[test]
fn backward_tanh_computes_1_minus_tanh_sq() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap());
    let y = graph.tanh(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let y_val = graph.value(y).unwrap();
    let x_grad = graph.grad(x).unwrap().unwrap();
    for i in 0..2 {
        let t = y_val.data()[i];
        let expected = 1.0 - t * t;
        assert!(
            (x_grad.data()[i] - expected).abs() < 1e-5,
            "tanh grad mismatch at {i}"
        );
    }
}

#[test]
fn leaky_relu_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3], vec![-2.0, 0.0, 3.0]).unwrap());
    let y = g.leaky_relu(x, 0.1).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();
    assert!((g.value(y).unwrap().data()[0] - (-0.2)).abs() < 1e-6);
    assert!((g.value(y).unwrap().data()[2] - 3.0).abs() < 1e-6);
    let grad = g.grad(x).unwrap().unwrap();
    assert!((grad.data()[0] - 0.1).abs() < 1e-6);
    assert!((grad.data()[2] - 1.0).abs() < 1e-6);
}

#[test]
fn gelu_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3], vec![-1.0, 0.0, 1.0]).unwrap());
    let y = g.gelu(x).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();

    // GELU(0) = 0
    let y_val = g.value(y).unwrap();
    assert!((y_val.data()[1] - 0.0).abs() < 1e-6);

    // Check gradient via finite differences
    let grad = g.grad(x).unwrap().unwrap();
    let eps = 1e-4_f32;
    for i in 0..3 {
        let xi = [-1.0_f32, 0.0, 1.0][i];
        let gelu_plus = {
            let xp = xi + eps;
            xp * (1.0 / (1.0 + (-1.702 * xp).exp()))
        };
        let gelu_minus = {
            let xm = xi - eps;
            xm * (1.0 / (1.0 + (-1.702 * xm).exp()))
        };
        let numerical = (gelu_plus - gelu_minus) / (2.0 * eps);
        assert!(
            (grad.data()[i] - numerical).abs() < 1e-3,
            "gelu grad mismatch at {i}: got {} expected {}",
            grad.data()[i],
            numerical
        );
    }
}

#[test]
fn silu_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3], vec![-1.0, 0.0, 1.0]).unwrap());
    let y = g.silu(x).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();

    // SiLU(0) = 0
    let y_val = g.value(y).unwrap();
    assert!((y_val.data()[1] - 0.0).abs() < 1e-6);

    // Check gradient: SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    let grad = g.grad(x).unwrap().unwrap();
    for i in 0..3 {
        let xi = [-1.0_f32, 0.0, 1.0][i];
        let s = 1.0 / (1.0 + (-xi).exp());
        let expected = s + xi * s * (1.0 - s);
        assert!(
            (grad.data()[i] - expected).abs() < 1e-5,
            "silu grad mismatch at {i}: got {} expected {}",
            grad.data()[i],
            expected
        );
    }
}

#[test]
fn mish_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3], vec![-1.0, 0.0, 1.0]).unwrap());
    let y = g.mish(x).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();

    // Mish(0) = 0 * tanh(ln(2)) = 0
    let y_val = g.value(y).unwrap();
    assert!((y_val.data()[1] - 0.0).abs() < 1e-6);

    // Check gradient via finite differences
    let grad = g.grad(x).unwrap().unwrap();
    let eps = 1e-4_f32;
    for i in 0..3 {
        let xi = [-1.0_f32, 0.0, 1.0][i];
        let mish_val = |x: f32| -> f32 {
            let sp = (1.0 + x.exp()).ln();
            x * sp.tanh()
        };
        let numerical = (mish_val(xi + eps) - mish_val(xi - eps)) / (2.0 * eps);
        assert!(
            (grad.data()[i] - numerical).abs() < 1e-3,
            "mish grad mismatch at {i}: got {} expected {}",
            grad.data()[i],
            numerical
        );
    }
}

#[test]
fn softmax_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap());
    let sm = g.softmax(x).unwrap();
    let loss = g.sum(sm).unwrap();
    g.backward(loss).unwrap();
    // softmax sums to 1 per row
    let sm_val = g.value(sm).unwrap();
    let sum: f32 = sm_val.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // gradients should exist
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.shape(), &[1, 3]);
}
