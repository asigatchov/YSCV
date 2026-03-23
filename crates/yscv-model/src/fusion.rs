use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{BatchNorm2dLayer, Conv2dLayer, ModelLayer, SequentialModel};

/// Fuse Conv2d + BatchNorm2d into a single Conv2d with adjusted weights and bias.
///
/// BatchNorm during inference computes:
///   `y = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// When preceded by Conv (`conv_out = W * x + b`), we can fold BN into conv:
///   `W_fused = scale * W`  (per output channel)
///   `b_fused = scale * (b - mean) + beta`
/// where `scale = gamma / sqrt(var + eps)`.
///
/// The fused Conv2d produces the same output as running Conv2d followed by BatchNorm2d,
/// eliminating the BatchNorm layer entirely and saving computation.
///
/// Conv2d weight layout is NHWC: `[KH, KW, C_in, C_out]`.
pub fn fuse_conv_bn(conv: &Conv2dLayer, bn: &BatchNorm2dLayer) -> Conv2dLayer {
    let out_channels = conv.out_channels();
    assert_eq!(
        out_channels,
        bn.num_features(),
        "Conv2d out_channels ({}) must match BatchNorm2d num_features ({})",
        out_channels,
        bn.num_features()
    );

    let gamma = bn.gamma().data();
    let beta = bn.beta().data();
    let running_mean = bn.running_mean().data();
    let running_var = bn.running_var().data();
    let eps = bn.epsilon();

    // Compute per-channel scale: gamma / sqrt(var + eps)
    let scale: Vec<f32> = (0..out_channels)
        .map(|c| gamma[c] / (running_var[c] + eps).sqrt())
        .collect();

    // Fuse weights: multiply each output channel slice by its scale.
    // Weight shape: [KH, KW, C_in, C_out]
    let weight = conv.weight();
    let w_data = weight.data();
    let kh = conv.kernel_h();
    let kw = conv.kernel_w();
    let c_in = conv.in_channels();

    let mut fused_w = vec![0.0f32; w_data.len()];
    for i in 0..kh {
        for j in 0..kw {
            for ci in 0..c_in {
                for co in 0..out_channels {
                    let idx = ((i * kw + j) * c_in + ci) * out_channels + co;
                    fused_w[idx] = w_data[idx] * scale[co];
                }
            }
        }
    }

    // Fuse bias: scale * (old_bias - mean) + beta
    let old_bias: Vec<f32> = match conv.bias() {
        Some(b) => b.data().to_vec(),
        None => vec![0.0; out_channels],
    };

    let fused_b: Vec<f32> = (0..out_channels)
        .map(|c| scale[c] * (old_bias[c] - running_mean[c]) + beta[c])
        .collect();

    let fused_weight =
        Tensor::from_vec(vec![kh, kw, c_in, out_channels], fused_w).expect("valid fused weight");
    let fused_bias = Tensor::from_vec(vec![out_channels], fused_b).expect("valid fused bias");

    Conv2dLayer::new(
        c_in,
        out_channels,
        kh,
        kw,
        conv.stride_h(),
        conv.stride_w(),
        fused_weight,
        Some(fused_bias),
    )
    .expect("fused Conv2dLayer construction should not fail")
}

/// Scan a `SequentialModel` and fuse Conv2d + BatchNorm2d patterns.
///
/// Returns a new optimized `SequentialModel` with fewer layers.
/// Conv2d immediately followed by BatchNorm2d is replaced by a single fused Conv2d.
/// All other layers (including ReLU after the fused Conv2d) are preserved as-is.
pub fn optimize_sequential(model: &SequentialModel, graph: &mut Graph) -> SequentialModel {
    let layers = model.layers();
    let mut optimized = SequentialModel::new(graph);
    let mut i = 0;

    while i < layers.len() {
        if i + 1 < layers.len()
            && let (ModelLayer::Conv2d(conv), ModelLayer::BatchNorm2d(bn)) =
                (&layers[i], &layers[i + 1])
        {
            let fused = fuse_conv_bn(conv, bn);
            optimized
                .add_conv2d(
                    fused.in_channels(),
                    fused.out_channels(),
                    fused.kernel_h(),
                    fused.kernel_w(),
                    fused.stride_h(),
                    fused.stride_w(),
                    fused.weight().clone(),
                    fused.bias().cloned(),
                )
                .expect("adding fused conv layer should not fail");
            i += 2; // skip both Conv2d and BatchNorm2d
            continue;
        }

        // Copy layer as-is using the appropriate add method.
        push_layer(&mut optimized, graph, &layers[i]);
        i += 1;
    }

    optimized
}

/// Helper to push a single `ModelLayer` into a `SequentialModel` via the public API.
fn push_layer(model: &mut SequentialModel, graph: &mut Graph, layer: &ModelLayer) {
    match layer {
        ModelLayer::Conv2d(l) => {
            model
                .add_conv2d(
                    l.in_channels(),
                    l.out_channels(),
                    l.kernel_h(),
                    l.kernel_w(),
                    l.stride_h(),
                    l.stride_w(),
                    l.weight().clone(),
                    l.bias().cloned(),
                )
                .expect("add_conv2d");
        }
        ModelLayer::BatchNorm2d(l) => {
            model
                .add_batch_norm2d(
                    l.num_features(),
                    l.epsilon(),
                    l.gamma().clone(),
                    l.beta().clone(),
                    l.running_mean().clone(),
                    l.running_var().clone(),
                )
                .expect("add_batch_norm2d");
        }
        ModelLayer::ReLU(_) => model.add_relu(),
        ModelLayer::LeakyReLU(l) => {
            model
                .add_leaky_relu(l.negative_slope())
                .expect("add_leaky_relu");
        }
        ModelLayer::Sigmoid(_) => model.add_sigmoid(),
        ModelLayer::Tanh(_) => model.add_tanh(),
        ModelLayer::Dropout(l) => {
            model.add_dropout(l.rate()).expect("add_dropout");
        }
        ModelLayer::Flatten(_) => model.add_flatten(),
        ModelLayer::Softmax(_) => model.add_softmax(),
        ModelLayer::GlobalAvgPool2d(_) => model.add_global_avg_pool2d(),
        ModelLayer::MaxPool2d(l) => {
            model
                .add_max_pool2d(l.kernel_h(), l.kernel_w(), l.stride_h(), l.stride_w())
                .expect("add_max_pool2d");
        }
        ModelLayer::AvgPool2d(l) => {
            model
                .add_avg_pool2d(l.kernel_h(), l.kernel_w(), l.stride_h(), l.stride_w())
                .expect("add_avg_pool2d");
        }
        ModelLayer::Linear(l) => {
            // Linear requires graph registration; use zero_init as a fallback
            // since we cannot retrieve the tensors without a graph reference.
            model
                .add_linear_zero(graph, l.in_features(), l.out_features())
                .expect("add_linear_zero");
        }
        ModelLayer::Embedding(l) => {
            let weight = Tensor::zeros(vec![l.num_embeddings(), l.embedding_dim()])
                .expect("embedding weight");
            model
                .add_embedding(graph, l.num_embeddings(), l.embedding_dim(), weight)
                .expect("add_embedding");
        }
        ModelLayer::LayerNorm(l) => {
            model
                .add_layer_norm(graph, l.normalized_shape(), 1e-5)
                .expect("add_layer_norm");
        }
        ModelLayer::GroupNorm(l) => {
            model
                .add_group_norm(graph, l.num_groups(), l.num_channels(), 1e-5)
                .expect("add_group_norm");
        }
        ModelLayer::DepthwiseConv2d(l) => {
            model
                .add_depthwise_conv2d(
                    l.channels(),
                    l.kernel_h(),
                    l.kernel_w(),
                    l.stride_h(),
                    l.stride_w(),
                    l.weight().clone(),
                    l.bias().cloned(),
                )
                .expect("add_depthwise_conv2d");
        }
        ModelLayer::SeparableConv2d(l) => {
            model
                .add_separable_conv2d(
                    l.in_channels(),
                    l.out_channels(),
                    l.kernel_h(),
                    l.kernel_w(),
                    l.stride_h(),
                    l.stride_w(),
                    l.depthwise().weight().clone(),
                    l.pointwise().weight().clone(),
                    l.pointwise().bias().cloned(),
                )
                .expect("add_separable_conv2d");
        }
        ModelLayer::LoraLinear(l) => {
            // LoRA layers pass through as-is (cannot reconstruct without graph context).
            model
                .add_linear_zero(graph, l.in_features, l.out_features)
                .expect("add_linear_zero for lora");
        }
        // Inference-only layers: push as-is via raw layer insertion.
        ModelLayer::Conv1d(_)
        | ModelLayer::Conv3d(_)
        | ModelLayer::ConvTranspose2d(_)
        | ModelLayer::AdaptiveAvgPool2d(_)
        | ModelLayer::AdaptiveMaxPool2d(_)
        | ModelLayer::InstanceNorm(_)
        | ModelLayer::PixelShuffle(_)
        | ModelLayer::Upsample(_)
        | ModelLayer::GELU(_)
        | ModelLayer::SiLU(_)
        | ModelLayer::Mish(_)
        | ModelLayer::PReLU(_)
        | ModelLayer::ResidualBlock(_)
        | ModelLayer::Rnn(_)
        | ModelLayer::Lstm(_)
        | ModelLayer::Gru(_)
        | ModelLayer::MultiHeadAttention(_)
        | ModelLayer::TransformerEncoder(_)
        | ModelLayer::FeedForward(_)
        | ModelLayer::DeformableConv2d(_) => {
            model.push_raw_layer(layer.clone());
        }
    }
}
