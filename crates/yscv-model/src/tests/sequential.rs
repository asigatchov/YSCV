use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::SequentialModel;

#[test]
fn sequential_cnn_forward_inference_end_to_end() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_conv2d_zero(1, 4, 3, 3, 1, 1, true).unwrap();
    model.add_batch_norm2d_identity(4, 1e-5).unwrap();
    model.add_max_pool2d(2, 2, 2, 2).unwrap();
    model.add_flatten();

    let input = Tensor::filled(vec![1, 8, 8, 1], 0.5).unwrap();
    let out = model.forward_inference(&input).unwrap();
    // Conv: [1,8,8,1] -> [1,6,6,4] (3x3 no-pad, stride 1)
    // BN: same shape
    // MaxPool 2x2/2: [1,6,6,4] -> [1,3,3,4]
    // Flatten: [1, 36]
    assert_eq!(out.shape().len(), 2, "expected rank-2 output");
    assert_eq!(out.shape(), &[1, 36]);
}

#[test]
fn end_to_end_cnn_graph_training_reduces_loss() {
    use yscv_optim::Sgd;

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Conv2d: [1,4,4,1] -> [1,3,3,2] (2x2 kernel, 1 in_ch, 2 out_ch, stride 1)
    model
        .add_conv2d(
            1,
            2,
            2,
            2,
            1,
            1,
            Tensor::filled(vec![2, 2, 1, 2], 0.25).unwrap(),
            Some(Tensor::zeros(vec![2]).unwrap()),
        )
        .unwrap();
    model.add_relu();
    // AvgPool: [1,3,3,2] -> [1,1,1,2] with 3x3 kernel stride 1
    model.add_avg_pool2d(3, 3, 1, 1).unwrap();
    model.add_flatten();
    // Linear: [1,2] -> [1,1]
    model
        .add_linear(
            &mut graph,
            2,
            1,
            Tensor::filled(vec![2, 1], 0.5).unwrap(),
            Tensor::zeros(vec![1]).unwrap(),
        )
        .unwrap();

    // Register CNN params in graph for autograd
    model.register_cnn_params(&mut graph);
    let param_nodes = model.trainable_nodes();
    assert!(param_nodes.len() >= 4); // conv_w, conv_b, fc_w, fc_b

    let mut optimizer = Sgd::new(0.01).unwrap();
    let persistent = model.persistent_node_count();

    // Dummy target: make the output approach 1.0
    let target = Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap();
    let input_data: Vec<f32> = (0..16).map(|v| v as f32 / 16.0).collect();

    let mut losses = Vec::new();
    for _ in 0..5 {
        graph.truncate(persistent).unwrap();

        let input_node =
            graph.constant(Tensor::from_vec(vec![1, 4, 4, 1], input_data.clone()).unwrap());
        let output = model.forward(&mut graph, input_node).unwrap();
        let target_node = graph.constant(target.clone());
        let diff = graph.sub(output, target_node).unwrap();
        let sq = graph.mul(diff, diff).unwrap();
        let loss = graph.mean(sq).unwrap();
        losses.push(graph.value(loss).unwrap().data()[0]);

        graph.backward(loss).unwrap();
        for &node in &param_nodes {
            optimizer.step_graph_node(&mut graph, node).unwrap();
        }
        model.sync_cnn_from_graph(&graph).unwrap();
    }

    // Loss should decrease over iterations
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease: {losses:?}"
    );
}

#[test]
fn num_parameters_counts_linear_and_conv_params() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    // Linear: 4*2 + 2 = 10 params
    model.add_linear_zero(&mut graph, 4, 2).unwrap();
    model.add_relu();
    // Conv2d with bias: 1*2*3*3 + 2 = 20 params
    model.add_conv2d_zero(1, 2, 3, 3, 1, 1, true).unwrap();
    assert_eq!(model.num_parameters(), 10 + 20);
}

#[test]
fn num_parameters_empty_model_is_zero() {
    let graph = Graph::new();
    let model = SequentialModel::new(&graph);
    assert_eq!(model.num_parameters(), 0);
}

#[test]
fn named_parameters_returns_correct_names_and_count() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    // Linear layer -> linear0_weight, linear0_bias
    model.add_linear_zero(&mut graph, 4, 2).unwrap();
    model.add_relu();
    // Another linear -> linear1_weight, linear1_bias
    model.add_linear_zero(&mut graph, 2, 3).unwrap();

    let params = model.named_parameters(&graph).unwrap();
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "linear0_weight",
            "linear0_bias",
            "linear1_weight",
            "linear1_bias"
        ]
    );
    // linear0 weight shape [4, 2]
    assert_eq!(params[0].1.shape(), &[4, 2]);
    // linear0 bias shape [2]
    assert_eq!(params[1].1.shape(), &[2]);
    // linear1 weight shape [2, 3]
    assert_eq!(params[2].1.shape(), &[2, 3]);
    // linear1 bias shape [3]
    assert_eq!(params[3].1.shape(), &[3]);
}

#[test]
fn named_parameters_includes_conv_and_batchnorm() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_conv2d_zero(1, 4, 3, 3, 1, 1, true).unwrap();
    model.add_batch_norm2d_identity(4, 1e-5).unwrap();

    let params = model.named_parameters(&graph).unwrap();
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "conv2d0_weight",
            "conv2d0_bias",
            "batchnorm2d0_gamma",
            "batchnorm2d0_beta",
        ]
    );
}

#[test]
fn freeze_layer_excludes_from_trainable() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    // Linear 0: 4*2 + 2 = 10 params
    model.add_linear_zero(&mut graph, 4, 2).unwrap();
    model.add_relu();
    // Linear 1: 2*3 + 3 = 9 params
    model.add_linear_zero(&mut graph, 2, 3).unwrap();

    assert_eq!(model.num_parameters(), 19);
    assert_eq!(model.trainable_parameters(), 19);

    // Freeze the first linear layer (index 0)
    model.freeze_layer(0).unwrap();
    assert_eq!(model.trainable_parameters(), 9);
    // Total parameters unchanged
    assert_eq!(model.num_parameters(), 19);
}

#[test]
fn unfreeze_layer_restores_trainable() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 4, 2).unwrap();
    model.add_relu();
    model.add_linear_zero(&mut graph, 2, 3).unwrap();

    let total = model.trainable_parameters();
    model.freeze_layer(0).unwrap();
    assert!(model.trainable_parameters() < total);

    model.unfreeze_layer(0).unwrap();
    assert_eq!(model.trainable_parameters(), total);
}

#[test]
fn freeze_invalid_index_returns_error() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    let err = model.freeze_layer(0);
    assert!(err.is_err());

    let err = model.unfreeze_layer(5);
    assert!(err.is_err());
}

#[test]
fn frozen_mask_reflects_state() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 4, 2).unwrap();
    model.add_relu();
    model.add_linear_zero(&mut graph, 2, 3).unwrap();

    assert_eq!(model.frozen_mask(), &[false, false, false]);

    model.freeze_layer(1).unwrap();
    assert_eq!(model.frozen_mask(), &[false, true, false]);

    model.freeze_layer(0).unwrap();
    assert_eq!(model.frozen_mask(), &[true, true, false]);

    model.unfreeze_layer(1).unwrap();
    assert_eq!(model.frozen_mask(), &[true, false, false]);
}

#[test]
fn gelu_layer_forward_inference() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_gelu();

    let input = Tensor::from_vec(vec![2, 3], vec![-1.0, 0.0, 1.0, 2.0, -0.5, 0.5]).unwrap();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert!(out.data().iter().all(|v| v.is_finite()));
}

#[test]
fn silu_layer_forward_inference() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_silu();

    let input = Tensor::from_vec(vec![2, 3], vec![-1.0, 0.0, 1.0, 2.0, -0.5, 0.5]).unwrap();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert!(out.data().iter().all(|v| v.is_finite()));
}

#[test]
fn mish_layer_forward_inference() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_mish();

    let input = Tensor::from_vec(vec![2, 3], vec![-1.0, 0.0, 1.0, 2.0, -0.5, 0.5]).unwrap();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert!(out.data().iter().all(|v| v.is_finite()));
}

#[test]
fn prelu_layer_single_alpha() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_prelu(vec![0.25]);

    let input = Tensor::from_vec(vec![1, 4], vec![-2.0, -1.0, 0.0, 1.0]).unwrap();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4]);
    let data = out.data();
    // Negative values scaled by alpha=0.25
    assert!((data[0] - (-0.5)).abs() < 1e-6);
    assert!((data[1] - (-0.25)).abs() < 1e-6);
    // Non-negative values unchanged
    assert!((data[2] - 0.0).abs() < 1e-6);
    assert!((data[3] - 1.0).abs() < 1e-6);
}

#[test]
fn prelu_layer_per_channel() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_prelu(vec![0.1, 0.2]);

    // Shape: [1, 2, 2] — batch=1, channels=2, spatial=2
    let input = Tensor::from_vec(vec![1, 2, 2], vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2]);
    let data = out.data();
    // Channel 0 (alpha=0.1): -1.0*0.1=-0.1, -2.0*0.1=-0.2
    assert!((data[0] - (-0.1)).abs() < 1e-6);
    assert!((data[1] - (-0.2)).abs() < 1e-6);
    // Channel 1 (alpha=0.2): -3.0*0.2=-0.6, -4.0*0.2=-0.8
    assert!((data[2] - (-0.6)).abs() < 1e-6);
    assert!((data[3] - (-0.8)).abs() < 1e-6);
}

#[test]
fn eval_mode_disables_dropout() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_dropout(0.5).unwrap();

    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    // In eval mode, dropout should pass input unchanged
    model.eval();
    let out = model.forward_inference(&input).unwrap();
    assert_eq!(out.data(), input.data());
}

#[test]
fn train_mode_enables_dropout() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_dropout(0.5).unwrap();

    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    // In train mode, dropout applies inverted scaling: x / (1 - rate)
    model.train_mode();
    let out = model.forward_inference(&input).unwrap();
    let scale = 1.0 / (1.0 - 0.5);
    for (got, &orig) in out.data().iter().zip(input.data().iter()) {
        assert!(
            (got - orig * scale).abs() < 1e-6,
            "expected {} but got {}",
            orig * scale,
            got
        );
    }
}

#[test]
fn is_training_getter() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Default is training
    assert!(model.is_training());

    model.eval();
    assert!(!model.is_training());

    model.train_mode();
    assert!(model.is_training());
}

fn _onnx_export_linear_model_produces_bytes() {
    use yscv_autograd::Graph;
    let mut graph = Graph::new();
    let mut model = crate::SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 4, 2).unwrap();
    model.add_relu();
    let bytes =
        crate::export_sequential_to_onnx(&model, &graph, &[1, 4], "yscv-test", "test_model")
            .unwrap();
    assert!(bytes.len() > 20);
}
