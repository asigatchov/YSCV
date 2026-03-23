use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{
    AvgPool2dLayer, BatchNorm2dLayer, Conv2dLayer, EmbeddingLayer, FlattenLayer, GroupNormLayer,
    LayerNormLayer, LeakyReLULayer, LinearLayer, MaxPool2dLayer, ModelError, SoftmaxLayer,
};

#[test]
fn linear_forward_produces_expected_values() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        3,
        2,
        Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        Tensor::from_vec(vec![2], vec![0.5, -0.5]).unwrap(),
    )
    .unwrap();
    let input =
        graph.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 1.0, 2.0, 1.0, 0.0]).unwrap());

    let out = layer.forward(&mut graph, input).unwrap();
    let value = graph.value(out).unwrap();

    assert_eq!(value.shape(), &[2, 2]);
    assert_eq!(value.data(), &[6.5, 7.5, 5.5, 7.5]);
}

#[test]
fn leaky_relu_forward_produces_expected_values() {
    let mut graph = Graph::new();
    let layer = LeakyReLULayer::new(0.1).unwrap();
    let input = graph.variable(Tensor::from_vec(vec![1, 3], vec![-2.0, 0.0, 3.0]).unwrap());

    let out = layer.forward(&mut graph, input).unwrap();
    let value = graph.value(out).unwrap();

    assert_eq!(value.shape(), &[1, 3]);
    assert_eq!(value.data(), &[-0.2, 0.0, 3.0]);
}

#[test]
fn leaky_relu_rejects_invalid_slope() {
    let err = LeakyReLULayer::new(-0.1).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidLeakyReluSlope {
            negative_slope: -0.1
        }
    );
}

// ── Conv2dLayer tests ───────────────────────────────────────────────

#[test]
fn conv2d_layer_forward_inference_produces_correct_shape() {
    let layer = Conv2dLayer::zero_init(3, 8, 3, 3, 1, 1, true).unwrap();
    let input = Tensor::zeros(vec![1, 6, 6, 3]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4, 4, 8]);
}

#[test]
fn conv2d_layer_rejects_bad_weight_shape() {
    let weight = Tensor::zeros(vec![3, 3, 3, 8]).unwrap();
    let bad_bias = Tensor::zeros(vec![4]).unwrap();
    let result = Conv2dLayer::new(3, 8, 3, 3, 1, 1, weight, Some(bad_bias));
    assert!(result.is_err());
}

#[test]
fn conv2d_layer_rejects_zero_stride() {
    let result = Conv2dLayer::zero_init(3, 8, 3, 3, 0, 1, false);
    assert!(result.is_err());
}

#[test]
fn conv2d_layer_no_bias_forward_works() {
    let layer = Conv2dLayer::zero_init(1, 2, 2, 2, 1, 1, false).unwrap();
    assert!(layer.bias().is_none());
    let input = Tensor::filled(vec![1, 3, 3, 1], 1.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);
}

// ── BatchNorm2dLayer tests ──────────────────────────────────────────

#[test]
fn batch_norm2d_identity_init_passes_through() {
    let layer = BatchNorm2dLayer::identity_init(3, 1e-5).unwrap();
    let input = Tensor::filled(vec![1, 4, 4, 3], 2.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4, 4, 3]);
    for &v in out.data() {
        assert!((v - 2.0).abs() < 1e-3, "expected ~2.0, got {v}");
    }
}

#[test]
fn batch_norm2d_rejects_invalid_epsilon() {
    assert!(BatchNorm2dLayer::identity_init(3, 0.0).is_err());
    assert!(BatchNorm2dLayer::identity_init(3, -1.0).is_err());
}

// ── MaxPool2dLayer tests ────────────────────────────────────────────

#[test]
fn max_pool2d_layer_forward_inference_produces_correct_shape() {
    let layer = MaxPool2dLayer::new(2, 2, 2, 2).unwrap();
    let input = Tensor::filled(vec![1, 4, 4, 1], 1.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
}

#[test]
fn max_pool2d_rejects_zero_kernel() {
    assert!(MaxPool2dLayer::new(0, 2, 2, 2).is_err());
}

#[test]
fn max_pool2d_rejects_zero_stride() {
    assert!(MaxPool2dLayer::new(2, 2, 0, 2).is_err());
}

// ── AvgPool2dLayer tests ────────────────────────────────────────────

#[test]
fn avg_pool2d_layer_forward_inference_correct_value() {
    let layer = AvgPool2dLayer::new(2, 2, 2, 2).unwrap();
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert!((out.data()[0] - 2.5).abs() < 1e-5);
}

// ── FlattenLayer tests ──────────────────────────────────────────────

#[test]
fn flatten_layer_reshapes_nhwc_to_rank2() {
    let layer = FlattenLayer::new();
    let input = Tensor::filled(vec![2, 3, 3, 4], 1.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[2, 36]);
}

#[test]
fn flatten_layer_rejects_rank1() {
    let layer = FlattenLayer::new();
    let input = Tensor::zeros(vec![5]).unwrap();
    assert!(layer.forward_inference(&input).is_err());
}

// ── SoftmaxLayer tests ─────────────────────────────────────────────

#[test]
fn softmax_layer_output_sums_to_one() {
    let layer = SoftmaxLayer::new();
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    let sum: f32 = out.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ── EmbeddingLayer tests ────────────────────────────────────────────

#[test]
fn embedding_layer_forward_inference() {
    let mut graph = Graph::new();
    let weight = Tensor::from_vec(vec![5, 3], (0..15).map(|i| i as f32).collect()).unwrap();
    let emb = EmbeddingLayer::new(&mut graph, 5, 3, weight).unwrap();
    let indices = Tensor::from_vec(vec![2], vec![0.0, 3.0]).unwrap();
    let out = emb.forward_inference(&graph, &indices).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(&out.data()[0..3], &[0.0, 1.0, 2.0]);
    assert_eq!(&out.data()[3..6], &[9.0, 10.0, 11.0]);
}

// ── LayerNormLayer tests ────────────────────────────────────────────

#[test]
fn layer_norm_forward_inference() {
    let mut graph = Graph::new();
    let ln = LayerNormLayer::new(&mut graph, 4, 1e-5).unwrap();
    let input = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let out = ln.forward_inference(&graph, &input).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    let d = out.data();
    let mean: f32 = d[0..4].iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 0.01);
}

// ── GroupNormLayer tests ────────────────────────────────────────────

#[test]
fn group_norm_forward_inference_nhwc() {
    let mut graph = Graph::new();
    let gn = GroupNormLayer::new(&mut graph, 2, 4, 1e-5).unwrap();
    let input = Tensor::from_vec(vec![1, 2, 2, 4], (0..16).map(|i| i as f32).collect()).unwrap();
    let out = gn.forward_inference(&graph, &input).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 4]);
    assert!(out.data().iter().all(|v: &f32| v.is_finite()));
}

// ── DepthwiseConv2dLayer / SeparableConv2dLayer tests ─────────────────

#[test]
fn depthwise_conv2d_layer_zero_init_inference() {
    let layer = crate::DepthwiseConv2dLayer::zero_init(3, 2, 2, 1, 1, false).unwrap();
    assert_eq!(layer.channels(), 3);
    assert_eq!(layer.weight().shape(), &[2, 2, 3, 1]);
    // Zero weights → output is all zeros
    let input = Tensor::filled(vec![1, 4, 4, 3], 1.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 3, 3, 3]);
    assert!(out.data().iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn depthwise_conv2d_layer_with_bias() {
    let weight = Tensor::filled(vec![1, 1, 2, 1], 1.0).unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.5, -0.5]).unwrap();
    let layer = crate::DepthwiseConv2dLayer::new(2, 1, 1, 1, 1, weight, Some(bias)).unwrap();
    let input = Tensor::filled(vec![1, 2, 2, 2], 2.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    // Each channel: 2.0 * 1.0 + bias
    let data = out.data();
    for i in (0..data.len()).step_by(2) {
        assert!((data[i] - 2.5).abs() < 1e-5); // 2+0.5
        assert!((data[i + 1] - 1.5).abs() < 1e-5); // 2-0.5
    }
}

#[test]
fn separable_conv2d_layer_zero_init_inference() {
    let layer = crate::SeparableConv2dLayer::zero_init(3, 8, 3, 3, 1, 1, false).unwrap();
    assert_eq!(layer.in_channels(), 3);
    assert_eq!(layer.out_channels(), 8);
    let input = Tensor::filled(vec![1, 5, 5, 3], 1.0).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 3, 3, 8]);
    // Zero weights → all zeros
    assert!(out.data().iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn depthwise_conv2d_layer_graph_forward() {
    let mut g = yscv_autograd::Graph::new();
    let mut layer = crate::DepthwiseConv2dLayer::zero_init(2, 2, 2, 1, 1, false).unwrap();
    layer.register_params(&mut g);
    let input = g.variable(Tensor::filled(vec![1, 3, 3, 2], 1.0).unwrap());
    let out = layer.forward(&mut g, input).unwrap();
    assert_eq!(g.value(out).unwrap().shape(), &[1, 2, 2, 2]);
}

#[test]
fn separable_conv2d_layer_graph_forward() {
    let mut g = yscv_autograd::Graph::new();
    let mut layer = crate::SeparableConv2dLayer::zero_init(2, 4, 2, 2, 1, 1, true).unwrap();
    layer.register_params(&mut g);
    let input = g.variable(Tensor::filled(vec![1, 3, 3, 2], 1.0).unwrap());
    let out = layer.forward(&mut g, input).unwrap();
    assert_eq!(g.value(out).unwrap().shape(), &[1, 2, 2, 4]);
}

#[test]
fn depthwise_conv2d_invalid_weight_shape() {
    let weight = Tensor::zeros(vec![2, 2, 3, 2]).unwrap(); // last dim should be 1
    assert!(crate::DepthwiseConv2dLayer::new(3, 2, 2, 1, 1, weight, None).is_err());
}

// ── ResidualBlock tests ─────────────────────────────────────────────

#[test]
fn residual_block_preserves_shape() {
    use crate::{ModelLayer, ResidualBlock};
    // Conv2d with stride=1 and zero-init preserves spatial dims when input is
    // same-padded manually (we use a 1x1 kernel to guarantee shape preservation).
    let conv = crate::Conv2dLayer::zero_init(3, 3, 1, 1, 1, 1, true).unwrap();
    let block = ResidualBlock::new(vec![ModelLayer::Conv2d(conv)]);

    let input = Tensor::filled(vec![1, 4, 4, 3], 1.0).unwrap();
    let out = block.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), input.shape());
}

#[test]
fn residual_block_adds_skip() {
    use crate::{ModelLayer, ResidualBlock};
    // With a zero-weight 1x1 conv, inner layers produce all zeros,
    // so the output should equal the input (0 + input = input).
    let conv = crate::Conv2dLayer::zero_init(3, 3, 1, 1, 1, 1, true).unwrap();
    let block = ResidualBlock::new(vec![ModelLayer::Conv2d(conv)]);

    let input = Tensor::filled(vec![1, 4, 4, 3], 2.5).unwrap();
    let out = block.forward_inference(&input).unwrap();
    for &v in out.data() {
        assert!(
            (v - 2.5).abs() < 1e-5,
            "expected ~2.5 (skip connection), got {v}"
        );
    }
}

#[test]
fn residual_block_in_sequential() {
    use crate::{ModelLayer, SequentialModel};
    let mut model = SequentialModel::new(&Graph::new());

    // Add a residual block with a 1x1 zero-init conv.
    let conv = crate::Conv2dLayer::zero_init(3, 3, 1, 1, 1, 1, true).unwrap();
    model.add_residual_block(vec![ModelLayer::Conv2d(conv)]);

    let input = Tensor::filled(vec![1, 4, 4, 3], 3.0).unwrap();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4, 4, 3]);
    for &v in out.data() {
        assert!(
            (v - 3.0).abs() < 1e-5,
            "expected ~3.0 from residual skip, got {v}"
        );
    }
}

// ── RnnLayer tests ──────────────────────────────────────────────────

#[test]
fn rnn_layer_forward_shape() {
    let layer = crate::RnnLayer::new(8, 16, 42);
    let input = Tensor::from_vec(vec![4, 8], vec![0.1; 4 * 8]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[4, 16]);
}

// ── LstmLayer tests ─────────────────────────────────────────────────

#[test]
fn lstm_layer_forward_shape() {
    let layer = crate::LstmLayer::new(8, 16, 42);
    let input = Tensor::from_vec(vec![4, 8], vec![0.1; 4 * 8]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[4, 16]);
}

// ── GruLayer tests ──────────────────────────────────────────────────

#[test]
fn gru_layer_forward_shape() {
    let layer = crate::GruLayer::new(8, 16, 42);
    let input = Tensor::from_vec(vec![4, 8], vec![0.1; 4 * 8]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[4, 16]);
}

// ── MultiHeadAttentionLayer tests ───────────────────────────────────

#[test]
fn mha_layer_forward_shape() {
    let layer = crate::MultiHeadAttentionLayer::new(16, 4, 42);
    let input = Tensor::from_vec(vec![5, 16], vec![0.0; 5 * 16]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[5, 16]);
}

// ── TransformerEncoderLayer tests ───────────────────────────────────

#[test]
fn transformer_encoder_layer_forward_shape() {
    let layer = crate::TransformerEncoderLayer::new(16, 4, 32, 42);
    let input = Tensor::from_vec(vec![5, 16], vec![0.0; 5 * 16]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[5, 16]);
}

// ── FeedForwardLayer tests ──────────────────────────────────────────

#[test]
fn feed_forward_layer_forward_shape() {
    let layer = crate::FeedForwardLayer::new(16, 32, 42);
    let input = Tensor::from_vec(vec![5, 16], vec![0.0; 5 * 16]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[5, 16]);
}

// --- DeformableConv2d ---

#[test]
fn deformable_conv2d_layer_forward() {
    use crate::DeformableConv2dLayer;

    // 4x4 input, 1 channel, 2x2 kernel, 2 output channels, stride=1, padding=0
    let input = Tensor::from_vec(
        vec![1, 4, 4, 1],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();

    let weight = Tensor::from_vec(
        vec![2, 2, 1, 2],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0],
    )
    .unwrap();
    // offset_weight: [2, 2, 1, 8] (kH*kW*2 = 8)
    // Zero offset weights produce zero offsets from conv
    let offset_weight = Tensor::zeros(vec![2, 2, 1, 8]).unwrap();

    let layer = DeformableConv2dLayer::new(1, 2, 2, 2, 1, 0, weight, offset_weight, None).unwrap();

    let out = layer.forward_inference(&input).unwrap();
    // out_h = (4-2)/1+1 = 3, out_w = 3
    assert_eq!(out.shape(), &[1, 3, 3, 2]);
}

#[test]
fn deformable_conv2d_layer_zero_init() {
    use crate::DeformableConv2dLayer;

    let layer = DeformableConv2dLayer::zero_init(3, 8, 3, 3, 1, 1, true).unwrap();
    assert_eq!(layer.in_channels(), 3);
    assert_eq!(layer.out_channels(), 8);
    assert_eq!(layer.kernel_h(), 3);
    assert_eq!(layer.kernel_w(), 3);
    assert_eq!(layer.stride(), 1);
    assert_eq!(layer.padding(), 1);
    assert!(layer.bias().is_some());

    let input = Tensor::zeros(vec![1, 5, 5, 3]).unwrap();
    let out = layer.forward_inference(&input).unwrap();
    // With padding=1, stride=1: out_h = (5+2-3)/1+1 = 5
    assert_eq!(out.shape(), &[1, 5, 5, 8]);
}

#[test]
fn deformable_conv2d_sequential_forward() {
    use crate::SequentialModel;
    use yscv_autograd::Graph;

    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    let weight = Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let offset_weight = Tensor::zeros(vec![2, 2, 1, 8]).unwrap();

    model
        .add_deformable_conv2d(1, 1, 2, 2, 1, 0, weight, offset_weight, None)
        .unwrap();

    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
}
