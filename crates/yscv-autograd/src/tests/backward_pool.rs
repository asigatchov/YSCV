use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn backward_max_pool2d_nhwc_scatters_to_argmax() {
    let mut graph = Graph::new();
    // input [1,2,2,1] with distinct values
    let input =
        graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 3.0, 2.0, 4.0]).unwrap());
    // pool entire 2x2 -> [1,1,1,1]
    let out = graph.max_pool2d_nhwc(input, 2, 2, 1, 1).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 1, 1, 1]);
    assert!((graph.value(out).unwrap().data()[0] - 4.0).abs() < 1e-6);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 1]);
    // gradient goes only to the max element (index 3 = value 4.0)
    assert_eq!(i_grad.data(), &[0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn backward_max_pool2d_nhwc_stride2() {
    let mut graph = Graph::new();
    // [1,4,4,1] -> pool 2x2 stride 2 -> [1,2,2,1]
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let input = graph.variable(Tensor::from_vec(vec![1, 4, 4, 1], data).unwrap());
    let out = graph.max_pool2d_nhwc(input, 2, 2, 2, 2).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    // Only 4 max positions should get gradient = 1.0, rest = 0.0
    let nonzero_count = i_grad.data().iter().filter(|&&g| g > 0.5).count();
    assert_eq!(nonzero_count, 4);
}

#[test]
fn backward_avg_pool2d_nhwc_distributes_uniformly() {
    let mut graph = Graph::new();
    let input =
        graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let out = graph.avg_pool2d_nhwc(input, 2, 2, 1, 1).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 1, 1, 1]);
    assert!((graph.value(out).unwrap().data()[0] - 2.5).abs() < 1e-6);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let i_grad = graph.grad(input).unwrap().unwrap();
    // Each of 4 inputs gets 1/4 of the upstream gradient (1.0)
    for &g in i_grad.data() {
        assert!((g - 0.25).abs() < 1e-6);
    }
}
