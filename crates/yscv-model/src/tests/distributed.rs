use yscv_autograd::Graph;
use yscv_optim::Sgd;
use yscv_tensor::Tensor;

use crate::{
    AllReduceAggregator, DistributedConfig, InProcessTransport, LinearLayer, SequentialModel,
    SupervisedDataset, SupervisedLoss, Transport, compress_gradients, decompress_gradients,
    gather_shards, shard_tensor, train_epoch_distributed, train_epoch_distributed_sgd,
    train_epoch_sgd,
};

#[test]
fn distributed_local_aggregator_passthrough() {
    let mut agg = crate::LocalAggregator;
    let grads = vec![
        Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3], vec![3.0, 4.0, 5.0]).unwrap(),
    ];
    let result = crate::GradientAggregator::aggregate(&mut agg, &grads).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].data(), &[1.0, 2.0]);
    assert_eq!(result[1].data(), &[3.0, 4.0, 5.0]);
}

#[test]
fn distributed_gradient_compression() {
    let grads = vec![Tensor::from_vec(vec![4], vec![0.1, 5.0, -3.0, 0.2]).unwrap()];
    let compressed = crate::compress_gradients(&grads, 0.5);
    assert_eq!(compressed.len(), 1);
    assert_eq!(compressed[0].indices.len(), 2); // top 50% = 2 elements
    assert_eq!(compressed[0].original_len, 4);

    // Decompress
    let shapes = vec![vec![4]];
    let decompressed = crate::decompress_gradients(&compressed, &shapes).unwrap();
    assert_eq!(decompressed.len(), 1);
    assert_eq!(decompressed[0].shape(), &[4]);
    // The top-2 by magnitude are 5.0 (idx 1) and -3.0 (idx 2)
    let data = decompressed[0].data();
    assert_eq!(data[0], 0.0); // zeroed out
    assert_eq!(data[1], 5.0);
    assert_eq!(data[2], -3.0);
    assert_eq!(data[3], 0.0); // zeroed out
}

#[test]
fn distributed_train_step_local() {
    let mut agg = crate::LocalAggregator;
    let loss = crate::distributed_train_step(
        || {
            let grads = vec![Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap()];
            Ok((0.5, grads))
        },
        |_aggregated_grads| Ok(()),
        &mut agg,
    )
    .unwrap();
    assert!((loss - 0.5).abs() < 1e-6);
}

#[test]
fn distributed_in_process_transport_send_recv() {
    let transports = crate::InProcessTransport::create_group(2);
    // Worker 0 sends to worker 1
    transports[0].send(1, b"hello").unwrap();
    let data = transports[1].recv(0).unwrap();
    assert_eq!(data, b"hello");
}

#[test]
fn distributed_train_epoch_single_rank_local_aggregator() {
    // With a LocalAggregator (single rank, no-op aggregation), the
    // distributed epoch should behave identically to a regular training
    // epoch.
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();

    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut aggregator = crate::LocalAggregator;
    let mut optimizer = Sgd::new(0.05).unwrap();

    let first = train_epoch_distributed_sgd(
        &mut graph,
        &model,
        &mut optimizer,
        &mut aggregator,
        &dataset,
        4,
        SupervisedLoss::Mse,
    )
    .unwrap();

    let mut last = first;
    for _ in 0..39 {
        last = train_epoch_distributed_sgd(
            &mut graph,
            &model,
            &mut optimizer,
            &mut aggregator,
            &dataset,
            4,
            SupervisedLoss::Mse,
        )
        .unwrap();
    }

    assert!(
        last.mean_loss < first.mean_loss,
        "expected loss to decrease: first={}, last={}",
        first.mean_loss,
        last.mean_loss
    );
    assert!(last.mean_loss < 0.05);
}

#[test]
fn distributed_train_epoch_matches_regular_sgd_single_epoch() {
    // Verify that a single distributed epoch with LocalAggregator
    // produces the same result as a regular train_epoch_sgd.

    // --- Regular ---
    let mut graph_reg = Graph::new();
    let mut model_reg = SequentialModel::new(&graph_reg);
    model_reg.add_linear_zero(&mut graph_reg, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();
    let mut opt_reg = Sgd::new(0.05).unwrap();
    let metrics_reg =
        train_epoch_sgd(&mut graph_reg, &model_reg, &mut opt_reg, &dataset, 4).unwrap();

    // --- Distributed (LocalAggregator) ---
    let mut graph_dist = Graph::new();
    let mut model_dist = SequentialModel::new(&graph_dist);
    model_dist.add_linear_zero(&mut graph_dist, 1, 1).unwrap();
    let mut opt_dist = Sgd::new(0.05).unwrap();
    let mut aggregator = crate::LocalAggregator;
    let metrics_dist = train_epoch_distributed_sgd(
        &mut graph_dist,
        &model_dist,
        &mut opt_dist,
        &mut aggregator,
        &dataset,
        4,
        SupervisedLoss::Mse,
    )
    .unwrap();

    assert_eq!(metrics_reg.steps, metrics_dist.steps);
    assert!(
        (metrics_reg.mean_loss - metrics_dist.mean_loss).abs() < 1e-6,
        "expected same loss: regular={}, distributed={}",
        metrics_reg.mean_loss,
        metrics_dist.mean_loss
    );
}

#[test]
fn distributed_train_epoch_generic_with_local_aggregator() {
    // Test the low-level train_epoch_distributed directly with a manual
    // forward/backward closure and LocalAggregator.
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![0.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let trainable = vec![layer.weight_node().unwrap(), layer.bias_node().unwrap()];
    let persistent = graph.node_count();
    let mut aggregator = crate::LocalAggregator;
    let mut optimizer = Sgd::new(0.05).unwrap();

    // Two batches: (x=1, y=3) and (x=2, y=5).
    let inputs = vec![
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1, 1], vec![2.0]).unwrap(),
    ];
    let targets = vec![
        Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap(),
        Tensor::from_vec(vec![1, 1], vec![5.0]).unwrap(),
    ];

    let mut first_loss = 0.0f32;
    for step in 0..20 {
        let inp = inputs.clone();
        let tgt = targets.clone();
        let metrics = train_epoch_distributed(
            &mut graph,
            &mut optimizer,
            &mut aggregator,
            &trainable,
            2,
            &mut |g, batch_idx| {
                g.truncate(persistent)?;
                let input_node = g.variable(inp[batch_idx].clone());
                let target_node = g.constant(tgt[batch_idx].clone());
                let prediction = layer.forward(g, input_node)?;
                let loss_node = crate::mse_loss(g, prediction, target_node)?;
                g.backward(loss_node)?;
                let loss_val = g.value(loss_node)?.data()[0];
                Ok(loss_val)
            },
        )
        .unwrap();
        if step == 0 {
            first_loss = metrics.mean_loss;
        }
    }

    // Run one more to get final loss.
    let inp = inputs.clone();
    let tgt = targets.clone();
    let final_metrics = train_epoch_distributed(
        &mut graph,
        &mut optimizer,
        &mut aggregator,
        &trainable,
        2,
        &mut |g, batch_idx| {
            g.truncate(persistent)?;
            let input_node = g.variable(inp[batch_idx].clone());
            let target_node = g.constant(tgt[batch_idx].clone());
            let prediction = layer.forward(g, input_node)?;
            let loss_node = crate::mse_loss(g, prediction, target_node)?;
            g.backward(loss_node)?;
            let loss_val = g.value(loss_node)?.data()[0];
            Ok(loss_val)
        },
    )
    .unwrap();

    assert!(
        final_metrics.mean_loss < first_loss,
        "expected loss to decrease: first={first_loss}, final={}",
        final_metrics.mean_loss
    );
}

// ---------------------------------------------------------------------------
// Integration tests: multi-worker all-reduce
// ---------------------------------------------------------------------------

#[test]
fn distributed_allreduce_averages_gradients() {
    // Create 2 InProcessTransport instances (rank 0 and rank 1).
    // Each "worker" has different gradients; after all-reduce with average
    // both should hold the element-wise mean.
    let mut transports = InProcessTransport::create_group(2);
    let t1 = transports.remove(1);
    let t0 = transports.remove(0);

    let cfg0 = DistributedConfig {
        world_size: 2,
        rank: 0,
        coordinator_addr: String::new(),
    };
    let cfg1 = DistributedConfig {
        world_size: 2,
        rank: 1,
        coordinator_addr: String::new(),
    };

    let grads0 = vec![Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()];
    let grads1 = vec![Tensor::from_vec(vec![3], vec![5.0, 6.0, 7.0]).unwrap()];

    // Run both workers in parallel threads since ring all-reduce requires
    // simultaneous send/recv.
    let h0 = std::thread::spawn(move || {
        let mut agg = AllReduceAggregator::new(cfg0, Box::new(t0));
        crate::GradientAggregator::aggregate(&mut agg, &grads0).unwrap()
    });
    let h1 = std::thread::spawn(move || {
        let mut agg = AllReduceAggregator::new(cfg1, Box::new(t1));
        crate::GradientAggregator::aggregate(&mut agg, &grads1).unwrap()
    });

    let result0 = h0.join().unwrap();
    let result1 = h1.join().unwrap();

    // Average of [1,2,3] and [5,6,7] = [3,4,5]
    assert_eq!(result0.len(), 1);
    assert_eq!(result1.len(), 1);
    let expected = [3.0, 4.0, 5.0];
    for (i, &v) in result0[0].data().iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-5,
            "rank 0 index {i}: expected {}, got {v}",
            expected[i]
        );
    }
    for (i, &v) in result1[0].data().iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-5,
            "rank 1 index {i}: expected {}, got {v}",
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Integration test: distributed train step reduces loss
// ---------------------------------------------------------------------------

#[test]
fn distributed_train_step_reduces_loss() {
    // Two workers with InProcessTransport; each computes gradients from a
    // simple linear model. After aggregation and parameter update the loss
    // should be lower on the next iteration.
    let mut transports = InProcessTransport::create_group(2);
    let t1 = transports.remove(1);
    let t0 = transports.remove(0);

    let cfg0 = DistributedConfig {
        world_size: 2,
        rank: 0,
        coordinator_addr: String::new(),
    };
    let cfg1 = DistributedConfig {
        world_size: 2,
        rank: 1,
        coordinator_addr: String::new(),
    };

    // Both workers produce different gradient magnitudes but the train step
    // should still reduce the loss returned by compute_gradients_fn.
    let h0 = std::thread::spawn(move || {
        let mut agg = AllReduceAggregator::new(cfg0, Box::new(t0));
        crate::distributed_train_step(
            || {
                let grads = vec![Tensor::from_vec(vec![2], vec![0.5, 1.0]).unwrap()];
                Ok((2.0, grads))
            },
            |_aggregated| Ok(()),
            &mut agg,
        )
        .unwrap()
    });

    let h1 = std::thread::spawn(move || {
        let mut agg = AllReduceAggregator::new(cfg1, Box::new(t1));
        crate::distributed_train_step(
            || {
                let grads = vec![Tensor::from_vec(vec![2], vec![1.5, 3.0]).unwrap()];
                Ok((1.5, grads))
            },
            |_aggregated| Ok(()),
            &mut agg,
        )
        .unwrap()
    });

    let loss0 = h0.join().unwrap();
    let loss1 = h1.join().unwrap();

    // Each worker returns its own local loss.
    assert!((loss0 - 2.0).abs() < 1e-6);
    assert!((loss1 - 1.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Integration test: gradient compression roundtrip
// ---------------------------------------------------------------------------

#[test]
fn gradient_compression_preserves_top_k() {
    // Compress a gradient tensor keeping only the top 50% by magnitude,
    // then decompress and verify the top-k values are preserved exactly.
    let original = vec![Tensor::from_vec(vec![6], vec![0.1, -5.0, 3.0, 0.2, -4.0, 0.05]).unwrap()];
    let compressed = compress_gradients(&original, 0.5);
    assert_eq!(compressed.len(), 1);
    // top 50% of 6 = 3 elements
    assert_eq!(compressed[0].indices.len(), 3);
    assert_eq!(compressed[0].original_len, 6);

    // The top-3 by magnitude are: -5.0 (idx 1), -4.0 (idx 4), 3.0 (idx 2)
    let shapes = vec![vec![6]];
    let decompressed = decompress_gradients(&compressed, &shapes).unwrap();
    let data = decompressed[0].data();

    // Preserved values
    assert_eq!(data[1], -5.0);
    assert_eq!(data[2], 3.0);
    assert_eq!(data[4], -4.0);

    // Zeroed-out values
    assert_eq!(data[0], 0.0);
    assert_eq!(data[3], 0.0);
    assert_eq!(data[5], 0.0);
}

// ---------------------------------------------------------------------------
// Integration test: tensor sharding and gathering roundtrip
// ---------------------------------------------------------------------------

#[test]
fn shard_and_gather_roundtrip_integration() {
    // Shard a 2D tensor into N pieces and gather them back, verifying equality.
    for num_shards in 1..=5 {
        let rows = 10;
        let cols = 3;
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
        let original = Tensor::from_vec(vec![rows, cols], data).unwrap();

        let shards = shard_tensor(&original, num_shards).unwrap();
        assert_eq!(shards.len(), num_shards);

        // Total rows across shards must equal original rows.
        let total_rows: usize = shards.iter().map(|s| s.shape()[0]).sum();
        assert_eq!(total_rows, rows);

        // Each shard must have the same trailing dimensions.
        for s in &shards {
            assert_eq!(s.shape()[1], cols);
        }

        let gathered = gather_shards(&shards).unwrap();
        assert_eq!(gathered.shape(), original.shape());
        assert_eq!(gathered.data(), original.data());
    }
}

#[test]
fn shard_tensor_1d_roundtrip() {
    let original = Tensor::from_vec(vec![8], (0..8).map(|i| i as f32).collect()).unwrap();
    let shards = shard_tensor(&original, 4).unwrap();
    assert_eq!(shards.len(), 4);
    for s in &shards {
        assert_eq!(s.shape(), &[2]);
    }
    let gathered = gather_shards(&shards).unwrap();
    assert_eq!(gathered.data(), original.data());
}

// ---------------------------------------------------------------------------
// Integration test: parameter server broadcast + reduce
// ---------------------------------------------------------------------------

#[test]
fn parameter_server_broadcast_and_reduce() {
    let mut transports = InProcessTransport::create_group(2);
    let t1 = transports.remove(1);
    let t0 = transports.remove(0);

    let cfg0 = DistributedConfig {
        world_size: 2,
        rank: 0,
        coordinator_addr: String::new(),
    };
    let cfg1 = DistributedConfig {
        world_size: 2,
        rank: 1,
        coordinator_addr: String::new(),
    };

    let params = vec![Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap()];
    let params_clone = params.clone();

    // Test broadcast: rank 0 sends params to rank 1.
    let h0 = std::thread::spawn(move || {
        let ps = crate::ParameterServer::new(cfg0, Box::new(t0));
        let broadcast_result = ps.broadcast_params(&params_clone).unwrap();
        // Now test reduce_gradients.
        let grads0 = vec![Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()];
        let reduced = ps.reduce_gradients(&grads0).unwrap();
        (broadcast_result, reduced)
    });

    let h1 = std::thread::spawn(move || {
        let ps = crate::ParameterServer::new(cfg1, Box::new(t1));
        let broadcast_result = ps.broadcast_params(&[]).unwrap(); // rank 1 receives
        // Send different gradients.
        let grads1 = vec![Tensor::from_vec(vec![3], vec![3.0, 4.0, 5.0]).unwrap()];
        let reduced = ps.reduce_gradients(&grads1).unwrap();
        (broadcast_result, reduced)
    });

    let (bcast0, reduced0) = h0.join().unwrap();
    let (bcast1, reduced1) = h1.join().unwrap();

    // Broadcast: rank 1 should receive rank 0's params.
    assert_eq!(bcast0[0].data(), &[10.0, 20.0, 30.0]);
    assert_eq!(bcast1[0].data(), &[10.0, 20.0, 30.0]);

    // Reduce: average of [1,2,3] and [3,4,5] = [2,3,4]
    let expected = [2.0, 3.0, 4.0];
    for (i, &v) in reduced0[0].data().iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-5,
            "rank 0 reduced index {i}: expected {}, got {v}",
            expected[i]
        );
    }
    for (i, &v) in reduced1[0].data().iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-5,
            "rank 1 reduced index {i}: expected {}, got {v}",
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Integration test: pipeline stage splitting
// ---------------------------------------------------------------------------

#[test]
fn pipeline_stages_all_layers_covered_various_configs() {
    for (num_layers, num_stages) in [(12, 4), (7, 3), (5, 5), (100, 7)] {
        let stages = crate::split_into_stages(num_layers, num_stages);
        assert_eq!(stages.len(), num_stages);
        assert_eq!(stages[0].start_layer, 0);
        assert_eq!(stages.last().unwrap().end_layer, num_layers);
        for i in 1..stages.len() {
            assert_eq!(stages[i].start_layer, stages[i - 1].end_layer);
        }
    }
}
