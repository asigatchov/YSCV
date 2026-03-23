use crate::Graph;
use crate::checkpoint::{CheckpointConfig, CheckpointSegment};
use yscv_tensor::Tensor;

#[test]
fn test_backward_with_checkpoints_produces_gradients() {
    // Build a simple computation: loss = sum((a * b + c)^2)
    // Compare gradients from backward() vs backward_with_checkpoints().
    let mut g1 = Graph::new();
    let a1 = g1.variable(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let b1 = g1.variable(Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap());
    let c1 = g1.variable(Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap());
    let mul1 = g1.mul(a1, b1).unwrap();
    let add1 = g1.add(mul1, c1).unwrap();
    let sq1 = g1.mul(add1, add1).unwrap();
    let loss1 = g1.sum(sq1).unwrap();
    g1.backward(loss1).unwrap();

    let grad_a1 = g1.grad(a1).unwrap().unwrap().data().to_vec();
    let grad_b1 = g1.grad(b1).unwrap().unwrap().data().to_vec();
    let grad_c1 = g1.grad(c1).unwrap().unwrap().data().to_vec();

    // Same computation with checkpointing.
    let mut g2 = Graph::new();
    let a2 = g2.variable(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let b2 = g2.variable(Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap());
    let c2 = g2.variable(Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap());
    let mul2 = g2.mul(a2, b2).unwrap();
    let add2 = g2.add(mul2, c2).unwrap();
    let sq2 = g2.mul(add2, add2).unwrap();
    let loss2 = g2.sum(sq2).unwrap();

    // Checkpoint intermediate nodes (mul, add, sq).
    let config = CheckpointConfig {
        segments: vec![CheckpointSegment {
            start_node: mul2.0,
            end_node: sq2.0,
        }],
    };
    g2.backward_with_checkpoints(loss2, &config).unwrap();

    let grad_a2 = g2.grad(a2).unwrap().unwrap().data().to_vec();
    let grad_b2 = g2.grad(b2).unwrap().unwrap().data().to_vec();
    let grad_c2 = g2.grad(c2).unwrap().unwrap().data().to_vec();

    // Gradients should be identical.
    for i in 0..3 {
        assert!(
            (grad_a1[i] - grad_a2[i]).abs() < 1e-6,
            "grad_a mismatch at {i}: {} vs {}",
            grad_a1[i],
            grad_a2[i]
        );
        assert!(
            (grad_b1[i] - grad_b2[i]).abs() < 1e-6,
            "grad_b mismatch at {i}: {} vs {}",
            grad_b1[i],
            grad_b2[i]
        );
        assert!(
            (grad_c1[i] - grad_c2[i]).abs() < 1e-6,
            "grad_c mismatch at {i}: {} vs {}",
            grad_c1[i],
            grad_c2[i]
        );
    }

    // Verify that checkpointed intermediate nodes had their values cleared.
    // The mul, add, sq nodes should now hold scalar(0.0).
    for node_id in [mul2, add2, sq2] {
        let val = g2.value(node_id).unwrap();
        assert_eq!(
            val.len(),
            1,
            "checkpointed node {} should be cleared",
            node_id.0
        );
    }
}

#[test]
fn test_backward_with_empty_checkpoint_config() {
    // With no checkpoint segments, backward_with_checkpoints should behave
    // identically to backward (no activations dropped).
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
    let y = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let prod = graph.mul(x, y).unwrap();
    let loss = graph.sum(prod).unwrap();

    let config = CheckpointConfig::default();
    graph.backward_with_checkpoints(loss, &config).unwrap();

    let grad_x = graph.grad(x).unwrap().unwrap().data().to_vec();
    assert!((grad_x[0] - 1.0).abs() < 1e-6);
    assert!((grad_x[1] - 2.0).abs() < 1e-6);

    // Intermediate node value should NOT be cleared (no segments).
    assert_eq!(graph.value(prod).unwrap().len(), 2);
}
