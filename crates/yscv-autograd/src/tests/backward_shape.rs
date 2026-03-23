use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_flatten_restores_input_shape() {
    let mut graph = Graph::new();
    // [2, 2, 2, 3] -> flatten -> [2, 12]
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![2, 2, 2, 3], data).unwrap());
    let flat = graph.flatten(input).unwrap();
    assert_eq!(graph.value(flat).unwrap().shape(), &[2, 12]);

    let loss = graph.sum(flat).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[2, 2, 2, 3]);
    for &g in i_grad.data() {
        assert!((g - 1.0).abs() < 1e-6);
    }
}

#[test]
fn reshape_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let r = g.reshape(x, vec![6]).unwrap();
    assert_eq!(g.value(r).unwrap().shape(), &[6]);
    let loss = g.sum(r).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
}

#[test]
fn unsqueeze_squeeze_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let u = g.unsqueeze(x, 0).unwrap();
    assert_eq!(g.value(u).unwrap().shape(), &[1, 3]);
    let s = g.squeeze(u, 0).unwrap();
    assert_eq!(g.value(s).unwrap().shape(), &[3]);
    let loss = g.sum(s).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.shape(), &[3]);
    assert!(grad.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

#[test]
fn cat_forward_and_backward() {
    let mut g = Graph::new();
    let a = g.variable(Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let b = g.variable(Tensor::from_vec(vec![1, 2], vec![5.0, 6.0]).unwrap());
    let c = g.cat(&[a, b], 0).unwrap();
    assert_eq!(g.value(c).unwrap().shape(), &[3, 2]);
    assert_eq!(g.value(c).unwrap().data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let loss = g.sum(c).unwrap();
    g.backward(loss).unwrap();
    let grad_a = g.grad(a).unwrap().unwrap();
    assert!(grad_a.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    let grad_b = g.grad(b).unwrap().unwrap();
    assert!(grad_b.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

#[test]
fn select_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let s = g.select(x, 0, 1).unwrap();
    assert_eq!(g.value(s).unwrap().shape(), &[2]);
    assert_eq!(g.value(s).unwrap().data(), &[3.0, 4.0]);
    let loss = g.sum(s).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.data(), &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn narrow_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(
        Tensor::from_vec(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
    );
    let n = g.narrow(x, 0, 1, 2).unwrap();
    assert_eq!(g.value(n).unwrap().shape(), &[2, 2]);
    assert_eq!(g.value(n).unwrap().data(), &[3.0, 4.0, 5.0, 6.0]);
    let loss = g.sum(n).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert_eq!(grad.data(), &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn pad_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let padded = g.pad(x, &[1, 1, 0, 0], 0.0).unwrap();
    assert_eq!(g.value(padded).unwrap().shape(), &[4, 3]);
    assert_eq!(
        &g.value(padded).unwrap().data()[3..9],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let loss = g.sum(padded).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert!(grad.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

#[test]
fn repeat_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap());
    let rep = g.repeat(x, &[2, 1]).unwrap();
    assert_eq!(g.value(rep).unwrap().shape(), &[2, 3]);
    assert_eq!(
        g.value(rep).unwrap().data(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
    let loss = g.sum(rep).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    assert!(grad.data().iter().all(|&v| (v - 2.0).abs() < 1e-5));
}
