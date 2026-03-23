use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{LoraConfig, LoraLinear};

#[test]
fn test_lora_creation() {
    let mut graph = Graph::new();
    let config = LoraConfig {
        rank: 4,
        alpha: 1.0,
    };
    let lora = LoraLinear::new(&mut graph, 8, 6, &config).unwrap();

    // A: [in_features, rank] = [8, 4]
    let a_shape = graph.value(lora.lora_a).unwrap().shape().to_vec();
    assert_eq!(a_shape, vec![8, 4]);

    // B: [rank, out_features] = [4, 6]
    let b_shape = graph.value(lora.lora_b).unwrap().shape().to_vec();
    assert_eq!(b_shape, vec![4, 6]);

    // Frozen weight: [8, 6]
    let w_shape = graph.value(lora.frozen_weight).unwrap().shape().to_vec();
    assert_eq!(w_shape, vec![8, 6]);

    assert_eq!(lora.rank, 4);
    assert_eq!(lora.in_features, 8);
    assert_eq!(lora.out_features, 6);
}

#[test]
fn test_lora_forward_shape() {
    let mut graph = Graph::new();
    let config = LoraConfig {
        rank: 2,
        alpha: 1.0,
    };
    let lora = LoraLinear::new(&mut graph, 5, 3, &config).unwrap();

    let batch_size = 4;
    let input = graph.variable(Tensor::zeros(vec![batch_size, 5]).unwrap());
    let out = lora.forward(&mut graph, input).unwrap();

    let out_shape = graph.value(out).unwrap().shape().to_vec();
    assert_eq!(out_shape, vec![4, 3]);
}

#[test]
fn test_lora_trainable_params() {
    let mut graph = Graph::new();
    let config = LoraConfig::default();
    let lora = LoraLinear::new(&mut graph, 10, 5, &config).unwrap();

    let params = lora.trainable_params();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], lora.lora_a);
    assert_eq!(params[1], lora.lora_b);

    // Verify A and B require grad, frozen weight does not
    assert!(graph.requires_grad(lora.lora_a).unwrap());
    assert!(graph.requires_grad(lora.lora_b).unwrap());
    assert!(!graph.requires_grad(lora.frozen_weight).unwrap());
}

#[test]
fn test_lora_zero_init_matches_frozen() {
    // With B initialized to zeros, the LoRA contribution should be zero,
    // so output should equal the frozen-only output (x @ W).
    let mut graph = Graph::new();
    let config = LoraConfig {
        rank: 3,
        alpha: 2.0,
    };
    let lora = LoraLinear::new(&mut graph, 4, 3, &config).unwrap();

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = graph.variable(Tensor::from_vec(vec![2, 4], input_data.clone()).unwrap());

    let out = lora.forward(&mut graph, input).unwrap();
    let out_val = graph.value(out).unwrap();

    // Frozen weight is zeros, so x @ W = zeros. LoRA B is zeros, so LoRA = zeros.
    // Output should be all zeros.
    let expected = [0.0f32; 6];
    assert_eq!(out_val.data(), &expected[..]);
}

#[test]
fn test_lora_merge() {
    let mut graph = Graph::new();
    let config = LoraConfig {
        rank: 2,
        alpha: 1.0,
    };
    let lora = LoraLinear::new(&mut graph, 4, 3, &config).unwrap();

    let merged = lora.merge(&graph).unwrap();
    assert_eq!(merged.shape(), &[4, 3]);

    // Since B is zeros, merged weight should equal frozen weight (also zeros)
    let w = graph.value(lora.frozen_weight).unwrap();
    assert_eq!(merged.data(), w.data());
}

#[test]
fn test_lora_merge_with_nonzero_b() {
    let mut graph = Graph::new();
    let config = LoraConfig {
        rank: 2,
        alpha: 2.0,
    };
    let lora = LoraLinear::new(&mut graph, 3, 2, &config).unwrap();

    // Overwrite B with known nonzero values
    let b_data = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
    *graph.value_mut(lora.lora_b).unwrap() = Tensor::from_vec(vec![2, 2], b_data).unwrap();

    let merged = lora.merge(&graph).unwrap();
    assert_eq!(merged.shape(), &[3, 2]);

    // With nonzero B, merged should differ from frozen weight
    let w = graph.value(lora.frozen_weight).unwrap();
    let merged_data = merged.data();
    let w_data = w.data();
    let has_diff = merged_data
        .iter()
        .zip(w_data.iter())
        .any(|(m, w)| (m - w).abs() > 1e-9);
    assert!(
        has_diff,
        "Merged weight should differ from frozen weight when B is nonzero"
    );
}

#[test]
fn test_lora_gradient_flow() {
    let mut graph = Graph::new();
    let config = LoraConfig {
        rank: 2,
        alpha: 1.0,
    };
    let lora = LoraLinear::new(&mut graph, 3, 2, &config).unwrap();

    // Set B to nonzero so gradients flow
    *graph.value_mut(lora.lora_b).unwrap() =
        Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap());
    let out = lora.forward(&mut graph, input).unwrap();

    // Reduce to scalar for backward
    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    // lora_a should have non-zero gradient
    let grad_a = graph.grad(lora.lora_a).unwrap();
    assert!(grad_a.is_some(), "lora_a should have gradient");
    let grad_a_data = grad_a.unwrap().data();
    let a_has_nonzero = grad_a_data.iter().any(|&v| v.abs() > 1e-12);
    assert!(a_has_nonzero, "lora_a gradient should be non-zero");

    // lora_b should have non-zero gradient
    let grad_b = graph.grad(lora.lora_b).unwrap();
    assert!(grad_b.is_some(), "lora_b should have gradient");
    let grad_b_data = grad_b.unwrap().data();
    let b_has_nonzero = grad_b_data.iter().any(|&v| v.abs() > 1e-12);
    assert!(b_has_nonzero, "lora_b gradient should be non-zero");

    // frozen_weight should have NO gradient (it's a constant)
    let grad_w = graph.grad(lora.frozen_weight).unwrap();
    if let Some(gw) = grad_w {
        let all_zero = gw.data().iter().all(|&v| v.abs() < 1e-12);
        assert!(all_zero, "frozen_weight gradient should be zero");
    }
    // None is also acceptable — constant nodes may not accumulate gradients at all
}

#[test]
fn test_apply_lora_converts_linear_layers() {
    use crate::{ModelLayer, SequentialModel};

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Add two Linear layers with a ReLU in between.
    model.add_linear_zero(&mut graph, 8, 4).unwrap();
    model.add_relu();
    model.add_linear_zero(&mut graph, 4, 2).unwrap();

    assert_eq!(model.layers().len(), 3);
    assert!(matches!(model.layers()[0], ModelLayer::Linear(_)));
    assert!(matches!(model.layers()[2], ModelLayer::Linear(_)));

    let config = LoraConfig {
        rank: 2,
        alpha: 1.0,
    };
    let converted = model.apply_lora(&mut graph, &config).unwrap();

    assert_eq!(converted, 2);
    assert!(matches!(model.layers()[0], ModelLayer::LoraLinear(_)));
    assert!(matches!(model.layers()[1], ModelLayer::ReLU(_)));
    assert!(matches!(model.layers()[2], ModelLayer::LoraLinear(_)));
}

#[test]
fn test_merge_lora_restores_linear() {
    use crate::{ModelLayer, SequentialModel};

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    model.add_linear_zero(&mut graph, 6, 3).unwrap();
    model.add_relu();
    model.add_linear_zero(&mut graph, 3, 2).unwrap();

    let config = LoraConfig {
        rank: 2,
        alpha: 1.0,
    };
    let applied = model.apply_lora(&mut graph, &config).unwrap();
    assert_eq!(applied, 2);

    let merged = model.merge_lora(&mut graph).unwrap();
    assert_eq!(merged, 2);

    assert!(matches!(model.layers()[0], ModelLayer::Linear(_)));
    assert!(matches!(model.layers()[1], ModelLayer::ReLU(_)));
    assert!(matches!(model.layers()[2], ModelLayer::Linear(_)));
}

#[test]
fn test_lora_trainable_param_count() {
    use crate::SequentialModel;

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    let in_f = 16;
    let out_f = 8;
    model.add_linear_zero(&mut graph, in_f, out_f).unwrap();
    model.add_relu();
    model.add_linear_zero(&mut graph, out_f, 4).unwrap();

    // Before LoRA: trainable params = weight + bias per Linear layer
    let before = model.trainable_nodes().len();
    // 2 Linear layers * 2 params (weight + bias) = 4
    assert_eq!(before, 4);

    let rank = 2;
    let config = LoraConfig { rank, alpha: 1.0 };
    model.apply_lora(&mut graph, &config).unwrap();

    // After LoRA: only lora_a and lora_b per layer are trainable = 2 * 2 = 4 nodes
    // but the total number of *elements* is much smaller.
    let after = model.trainable_nodes().len();
    assert_eq!(after, 4); // 2 layers * 2 params (A + B)

    // Verify the actual element count is smaller.
    // Original: layer1 weight=16*8=128, bias=8; layer2 weight=8*4=32, bias=4 => total=172
    // LoRA: layer1 A=16*2=32, B=2*8=16; layer2 A=8*2=16, B=2*4=8 => total=72
    let lora_params: usize = model
        .trainable_nodes()
        .iter()
        .map(|&n| graph.value(n).unwrap().data().len())
        .sum();
    let original_params = in_f * out_f + out_f + out_f * 4 + 4; // 172
    assert!(
        lora_params < original_params,
        "LoRA params ({lora_params}) should be fewer than original ({original_params})"
    );
    assert_eq!(lora_params, 72); // 32 + 16 + 16 + 8
}
