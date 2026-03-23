use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{
    FeedForwardLayer, GruLayer, LstmLayer, ModelLayer, PixelShuffleLayer, ResidualBlock, RnnLayer,
    SoftmaxLayer, TransformerEncoderLayer, UpsampleLayer,
};

#[test]
fn softmax_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let layer = SoftmaxLayer::new();
    let input =
        graph.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 1.0, 0.0, -1.0]).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    let val = graph.value(out).unwrap();
    assert_eq!(val.shape(), &[2, 3]);

    // Each row should sum to ~1.0
    let d = val.data();
    assert!((d[0] + d[1] + d[2] - 1.0).abs() < 1e-5);
    assert!((d[3] + d[4] + d[5] - 1.0).abs() < 1e-5);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.grad(input).unwrap().is_some());
}

#[test]
fn pixel_shuffle_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let layer = PixelShuffleLayer::new(2);
    let input = graph.variable(
        Tensor::from_vec(vec![1, 2, 2, 4], (1..=16).map(|v| v as f32).collect()).unwrap(),
    );
    let out = layer.forward(&mut graph, input).unwrap();
    let val = graph.value(out).unwrap();
    assert_eq!(val.shape(), &[1, 4, 4, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 4]);
}

#[test]
fn upsample_nearest_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let layer = UpsampleLayer::new(2, false);
    let input =
        graph.variable(Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    let val = graph.value(out).unwrap();
    assert_eq!(val.shape(), &[1, 4, 4, 1]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let i_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(i_grad.shape(), &[1, 2, 2, 1]);
    for &g in i_grad.data() {
        assert!((g - 4.0).abs() < 1e-6);
    }
}

#[test]
fn upsample_bilinear_returns_inference_only_error() {
    let mut graph = Graph::new();
    let layer = UpsampleLayer::new(2, true);
    let input = graph.variable(Tensor::filled(vec![1, 2, 2, 1], 1.0).unwrap());
    let result = layer.forward(&mut graph, input);
    assert!(result.is_err());
}

#[test]
fn feedforward_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let mut layer = FeedForwardLayer::new(4, 8, 42);
    layer.register_params(&mut graph);

    let input = graph.variable(Tensor::filled(vec![2, 4], 0.5).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    let val = graph.value(out).unwrap();
    assert_eq!(val.shape(), &[2, 4]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.grad(input).unwrap().is_some());
}

#[test]
fn residual_block_graph_forward_produces_skip_connection() {
    let mut graph = Graph::new();
    // ResidualBlock with just a ReLU layer: output = relu(input) + input
    let block = ResidualBlock::new(vec![ModelLayer::ReLU(crate::ReLULayer)]);
    let input = graph.variable(Tensor::from_vec(vec![1, 4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap());
    let out = block.forward(&mut graph, input).unwrap();
    let val = graph.value(out).unwrap();
    // relu([-1,2,-3,4]) + [-1,2,-3,4] = [0,2,0,4] + [-1,2,-3,4] = [-1,4,-3,8]
    assert_eq!(val.shape(), &[1, 4]);
    let d = val.data();
    assert!((d[0] - (-1.0)).abs() < 1e-6);
    assert!((d[1] - 4.0).abs() < 1e-6);
    assert!((d[2] - (-3.0)).abs() < 1e-6);
    assert!((d[3] - 8.0).abs() < 1e-6);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    let i_grad = graph.grad(input).unwrap().unwrap();
    // For x < 0: relu'(x)=0, so grad = 0 + 1 (skip) = 1
    // For x > 0: relu'(x)=1, so grad = 1 + 1 (skip) = 2
    assert!((i_grad.data()[0] - 1.0).abs() < 1e-6);
    assert!((i_grad.data()[1] - 2.0).abs() < 1e-6);
    assert!((i_grad.data()[2] - 1.0).abs() < 1e-6);
    assert!((i_grad.data()[3] - 2.0).abs() < 1e-6);
}

#[test]
fn transformer_encoder_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let d_model = 4;
    let num_heads = 2;
    let d_ff = 8;
    let mut layer = TransformerEncoderLayer::new(d_model, num_heads, d_ff, 42);
    layer.register_params(&mut graph);

    let input = graph.variable(Tensor::filled(vec![3, d_model], 0.5).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    let val = graph.value(out).unwrap();
    assert_eq!(val.shape(), &[3, d_model]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.grad(input).unwrap().is_some());
}

#[test]
fn rnn_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let mut layer = RnnLayer::new(2, 3, 42);
    layer.register_params(&mut graph);

    let input = graph.variable(Tensor::filled(vec![4, 2], 0.5).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[4, 3]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.grad(input).unwrap().is_some());
}

#[test]
fn lstm_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let mut layer = LstmLayer::new(2, 3, 42);
    layer.register_params(&mut graph);

    let input = graph.variable(Tensor::filled(vec![4, 2], 0.5).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[4, 3]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.grad(input).unwrap().is_some());
}

#[test]
fn gru_layer_graph_forward_and_backward() {
    let mut graph = Graph::new();
    let mut layer = GruLayer::new(2, 3, 42);
    layer.register_params(&mut graph);

    let input = graph.variable(Tensor::filled(vec![4, 2], 0.5).unwrap());
    let out = layer.forward(&mut graph, input).unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[4, 3]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.grad(input).unwrap().is_some());
}

#[test]
fn model_layer_forward_no_longer_inference_only_for_softmax() {
    let mut graph = Graph::new();
    let layer = ModelLayer::Softmax(SoftmaxLayer::new());
    assert!(layer.supports_graph_forward());
    let input = graph.variable(Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap());
    let result = layer.forward(&mut graph, input);
    assert!(result.is_ok());
}

#[test]
fn model_layer_forward_no_longer_inference_only_for_pixel_shuffle() {
    let mut graph = Graph::new();
    let layer = ModelLayer::PixelShuffle(PixelShuffleLayer::new(2));
    assert!(layer.supports_graph_forward());
    let input = graph.variable(Tensor::filled(vec![1, 1, 1, 4], 1.0).unwrap());
    let result = layer.forward(&mut graph, input);
    assert!(result.is_ok());
}
