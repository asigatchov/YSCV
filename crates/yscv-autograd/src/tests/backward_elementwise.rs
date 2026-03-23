use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_add_with_broadcast_distributes_gradients() {
    let mut graph = Graph::new();
    let x =
        graph.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let b = graph.variable(Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap());

    let y = graph.add(x, b).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.shape(), &[2, 3]);
    assert_eq!(x_grad.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let b_grad = graph.grad(b).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[3]);
    assert_eq!(b_grad.data(), &[2.0, 2.0, 2.0]);
}

#[test]
fn backward_sub_propagates_signed_gradients() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![5.0, 7.0]).unwrap());
    let y = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let z = graph.sub(x, y).unwrap();
    let loss = graph.sum(z).unwrap();
    graph.backward(loss).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.data(), &[1.0, 1.0]);

    let y_grad = graph.grad(y).unwrap().unwrap();
    assert_eq!(y_grad.data(), &[-1.0, -1.0]);
}

#[test]
fn backward_mul_matches_elementwise_derivative() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap());
    let y = graph.variable(Tensor::from_vec(vec![2], vec![4.0, 5.0]).unwrap());
    let z = graph.mul(x, y).unwrap();
    let loss = graph.sum(z).unwrap();
    graph.backward(loss).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.data(), &[4.0, 5.0]);

    let y_grad = graph.grad(y).unwrap().unwrap();
    assert_eq!(y_grad.data(), &[2.0, 3.0]);
}

#[test]
fn backward_div_computes_quotient_rule_gradients() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![6.0, 10.0]).unwrap());
    let b = graph.variable(Tensor::from_vec(vec![2], vec![2.0, 5.0]).unwrap());
    let y = graph.div(a, b).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    // da = 1/b = [0.5, 0.2]
    let a_grad = graph.grad(a).unwrap().unwrap();
    assert!((a_grad.data()[0] - 0.5).abs() < 1e-5);
    assert!((a_grad.data()[1] - 0.2).abs() < 1e-5);

    // db = -a/b^2 = [-6/4, -10/25] = [-1.5, -0.4]
    let b_grad = graph.grad(b).unwrap().unwrap();
    assert!((b_grad.data()[0] - (-1.5)).abs() < 1e-5);
    assert!((b_grad.data()[1] - (-0.4)).abs() < 1e-5);
}

#[test]
fn backward_neg_negates_gradient() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let y = graph.neg(x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.data(), &[-1.0, -1.0, -1.0]);
}

#[test]
fn abs_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3], vec![-2.0, 0.5, -1.0]).unwrap());
    let y = g.abs(x).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();
    assert_eq!(g.value(y).unwrap().data(), &[2.0, 0.5, 1.0]);
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.data(), &[-1.0, 1.0, -1.0]);
}

#[test]
fn pow_forward_and_backward() {
    let mut g = Graph::new();
    let base = g.variable(Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap());
    let exp = g.constant(Tensor::from_vec(vec![2], vec![3.0, 2.0]).unwrap());
    let y = g.pow(base, exp).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();
    // d(2^3)/d2 = 3*2^2 = 12, d(3^2)/d3 = 2*3 = 6
    let grad = g.grad(base).unwrap().unwrap();
    assert!((grad.data()[0] - 12.0).abs() < 1e-3);
    assert!((grad.data()[1] - 6.0).abs() < 1e-3);
}

#[test]
fn clamp_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![4], vec![-1.0, 0.5, 1.5, 3.0]).unwrap());
    let y = g.clamp(x, 0.0, 2.0).unwrap();
    let loss = g.sum(y).unwrap();
    g.backward(loss).unwrap();
    assert_eq!(g.value(y).unwrap().data(), &[0.0, 0.5, 1.5, 2.0]);
    // grad passes through only where 0 <= x <= 2
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.data(), &[0.0, 1.0, 1.0, 0.0]);
}
