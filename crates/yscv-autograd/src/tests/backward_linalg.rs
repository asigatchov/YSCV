use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_matmul_2d_computes_expected_grads() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let b = graph.variable(Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap());

    let y = graph.matmul_2d(a, b).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let a_grad = graph.grad(a).unwrap().unwrap();
    assert_eq!(a_grad.shape(), &[2, 2]);
    assert_eq!(a_grad.data(), &[11.0, 15.0, 11.0, 15.0]);

    let b_grad = graph.grad(b).unwrap().unwrap();
    assert_eq!(b_grad.shape(), &[2, 2]);
    assert_eq!(b_grad.data(), &[4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn transpose_2d_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let t = g.transpose_2d(x).unwrap();
    assert_eq!(g.value(t).unwrap().shape(), &[3, 2]);
    let loss = g.sum(t).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
    assert!(grad.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
}
