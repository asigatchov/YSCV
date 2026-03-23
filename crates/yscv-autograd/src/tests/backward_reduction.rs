use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_mean_distributes_uniform_gradient() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let y = graph.mean(x).unwrap();
    graph.backward(y).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.shape(), &[4]);
    for &g in x_grad.data() {
        assert!((g - 0.25).abs() < 1e-6);
    }
}

#[test]
fn sum_axis_forward_and_backward() {
    let mut g = Graph::new();
    // 2x3 matrix
    let x = g.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let s = g.sum_axis(x, 1).unwrap();
    // sum along axis 1 -> [2]
    assert_eq!(g.value(s).unwrap().shape(), &[2]);
    assert_eq!(g.value(s).unwrap().data(), &[6.0, 15.0]);
    let loss = g.sum(s).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    // d(sum)/d(x_ij) = 1 for all elements
    assert!(grad.data().iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

#[test]
fn mean_axis_forward_and_backward() {
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let m = g.mean_axis(x, 1).unwrap();
    assert_eq!(g.value(m).unwrap().shape(), &[2]);
    let data = g.value(m).unwrap().data().to_vec();
    assert!((data[0] - 2.0).abs() < 1e-5); // mean([1,2,3]) = 2
    assert!((data[1] - 5.0).abs() < 1e-5); // mean([4,5,6]) = 5
    let loss = g.sum(m).unwrap();
    g.backward(loss).unwrap();
    let grad = g.grad(x).unwrap().unwrap();
    // d(mean)/d(x_ij) = 1/3
    assert!(grad.data().iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-5));
}
