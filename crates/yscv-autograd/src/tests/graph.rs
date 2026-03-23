use yscv_kernels::CpuBackend;
use yscv_tensor::Tensor;

use crate::{AutogradError, Graph};

#[test]
fn backward_accumulates_gradient_from_multiple_paths() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let y = graph.add(x, x).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let x_grad = graph.grad(x).unwrap().unwrap();
    assert_eq!(x_grad.shape(), &[2]);
    assert_eq!(x_grad.data(), &[2.0, 2.0]);
}

#[test]
fn backward_does_not_produce_grad_for_constant_leaf() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());

    let y = graph.add(x, c).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    assert!(graph.grad(c).unwrap().is_none());
    assert!(graph.grad(x).unwrap().is_some());
}

#[test]
fn backward_requires_scalar_target() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let err = graph.backward(x).unwrap_err();
    assert_eq!(err, AutogradError::NonScalarTarget { shape: vec![2] });
}

#[test]
fn truncate_keeps_only_requested_prefix() {
    let mut graph = Graph::new();
    let _a = graph.variable(Tensor::from_vec(vec![1], vec![1.0]).unwrap());
    let _b = graph.variable(Tensor::from_vec(vec![1], vec![2.0]).unwrap());
    let _c = graph.variable(Tensor::from_vec(vec![1], vec![3.0]).unwrap());
    assert_eq!(graph.node_count(), 3);

    graph.truncate(1).unwrap();
    assert_eq!(graph.node_count(), 1);
}

#[test]
fn truncate_rejects_out_of_range_size() {
    let mut graph = Graph::new();
    let _a = graph.variable(Tensor::from_vec(vec![1], vec![1.0]).unwrap());
    let err = graph.truncate(2).unwrap_err();
    assert_eq!(
        err,
        AutogradError::InvalidTruncate {
            requested: 2,
            available: 1
        }
    );
}

#[test]
fn test_graph_default_no_backend() {
    // Verify existing behavior is unchanged when no backend is set.
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let b = graph.variable(Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap());

    let sum = graph.add(a, b).unwrap();
    let sum_val = graph.value(sum).unwrap();
    assert_eq!(sum_val.data(), &[6.0, 8.0, 10.0, 12.0]);

    let mm = graph.matmul_2d(a, b).unwrap();
    let mm_val = graph.value(mm).unwrap();
    assert_eq!(mm_val.shape(), &[2, 2]);
    assert_eq!(mm_val.data(), &[19.0, 22.0, 43.0, 50.0]);

    let r = graph.relu(a).unwrap();
    let r_val = graph.value(r).unwrap();
    assert_eq!(r_val.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_graph_with_cpu_backend() {
    // Create a graph with CpuBackend and verify results match default (no backend).
    let mut graph_default = Graph::new();
    let mut graph_backend = Graph::new();
    graph_backend.set_backend(Box::new(CpuBackend));

    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];

    let a1 = graph_default.variable(Tensor::from_vec(vec![2, 2], data_a.clone()).unwrap());
    let b1 = graph_default.variable(Tensor::from_vec(vec![2, 2], data_b.clone()).unwrap());

    let a2 = graph_backend.variable(Tensor::from_vec(vec![2, 2], data_a.clone()).unwrap());
    let b2 = graph_backend.variable(Tensor::from_vec(vec![2, 2], data_b.clone()).unwrap());

    // Test add
    let sum1 = graph_default.add(a1, b1).unwrap();
    let sum2 = graph_backend.add(a2, b2).unwrap();
    assert_eq!(
        graph_default.value(sum1).unwrap().data(),
        graph_backend.value(sum2).unwrap().data(),
    );

    // Test matmul
    let mm1 = graph_default.matmul_2d(a1, b1).unwrap();
    let mm2 = graph_backend.matmul_2d(a2, b2).unwrap();
    assert_eq!(
        graph_default.value(mm1).unwrap().data(),
        graph_backend.value(mm2).unwrap().data(),
    );

    // Test relu
    let r1 = graph_default.relu(a1).unwrap();
    let r2 = graph_backend.relu(a2).unwrap();
    assert_eq!(
        graph_default.value(r1).unwrap().data(),
        graph_backend.value(r2).unwrap().data(),
    );

    // Test sub
    let sub1 = graph_default.sub(a1, b1).unwrap();
    let sub2 = graph_backend.sub(a2, b2).unwrap();
    assert_eq!(
        graph_default.value(sub1).unwrap().data(),
        graph_backend.value(sub2).unwrap().data(),
    );

    // Test mul
    let mul1 = graph_default.mul(a1, b1).unwrap();
    let mul2 = graph_backend.mul(a2, b2).unwrap();
    assert_eq!(
        graph_default.value(mul1).unwrap().data(),
        graph_backend.value(mul2).unwrap().data(),
    );

    // Test sigmoid
    let sig1 = graph_default.sigmoid(a1).unwrap();
    let sig2 = graph_backend.sigmoid(a2).unwrap();
    assert_eq!(
        graph_default.value(sig1).unwrap().data(),
        graph_backend.value(sig2).unwrap().data(),
    );

    // Test softmax
    let sm1 = graph_default.softmax(a1).unwrap();
    let sm2 = graph_backend.softmax(a2).unwrap();
    let sm1_data = graph_default.value(sm1).unwrap().data();
    let sm2_data = graph_backend.value(sm2).unwrap().data();
    for (v1, v2) in sm1_data.iter().zip(sm2_data.iter()) {
        assert!((v1 - v2).abs() < 1e-6, "softmax mismatch: {v1} vs {v2}");
    }

    // Test clear_backend reverts to default behavior
    graph_backend.clear_backend();
    let a3 = graph_backend.variable(Tensor::from_vec(vec![2, 2], data_a).unwrap());
    let b3 = graph_backend.variable(Tensor::from_vec(vec![2, 2], data_b).unwrap());
    let sum3 = graph_backend.add(a3, b3).unwrap();
    assert_eq!(
        graph_default.value(sum1).unwrap().data(),
        graph_backend.value(sum3).unwrap().data(),
    );
}
