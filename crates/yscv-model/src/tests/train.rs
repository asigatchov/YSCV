use yscv_autograd::Graph;
use yscv_optim::{Adam, AdamW, CosineAnnealingLr, OneCycleLr, RmsProp, Sgd, StepLr};
use yscv_tensor::Tensor;

use crate::{
    BatchIterOptions, EpochTrainOptions, LinearLayer, ModelError, SequentialModel,
    SupervisedDataset, SupervisedLoss, accumulate_gradients, collect_gradients, scale_gradients,
    train_epoch_adam, train_epoch_adam_with_options, train_epoch_adamw,
    train_epoch_adamw_with_options, train_epoch_rmsprop, train_epoch_rmsprop_with_options,
    train_epoch_sgd, train_epoch_sgd_with_options, train_epoch_sgd_with_options_and_loss,
    train_epochs_adamw_with_scheduler, train_epochs_rmsprop_with_scheduler,
    train_epochs_sgd_with_scheduler, train_step_adam, train_step_adam_with_accumulation,
    train_step_adamw, train_step_rmsprop, train_step_sgd, train_step_sgd_with_accumulation,
    train_step_sgd_with_loss,
};

use super::assert_slice_approx_eq;

#[test]
fn train_step_sgd_updates_linear_parameters() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = Sgd::new(0.1).unwrap();
    let loss = train_step_sgd(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
    )
    .unwrap();
    assert_eq!(loss, 4.0);

    let weight = graph.value(layer.weight_node().unwrap()).unwrap();
    let bias = graph.value(layer.bias_node().unwrap()).unwrap();
    assert_eq!(weight.data(), &[1.4]);
    assert_eq!(bias.data(), &[0.4]);
}

#[test]
fn train_step_sgd_with_mae_loss_updates_linear_parameters() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = Sgd::new(0.1).unwrap();
    let loss = train_step_sgd_with_loss(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
        SupervisedLoss::Mae,
    )
    .unwrap();
    assert_eq!(loss, 2.0);

    let weight = graph.value(layer.weight_node().unwrap()).unwrap();
    let bias = graph.value(layer.bias_node().unwrap()).unwrap();
    assert_eq!(weight.data(), &[1.1]);
    assert_eq!(bias.data(), &[0.1]);
}

#[test]
fn train_step_sgd_with_hinge_loss_updates_linear_parameters() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![0.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = Sgd::new(0.1).unwrap();
    let loss = train_step_sgd_with_loss(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
        SupervisedLoss::Hinge { margin: 1.0 },
    )
    .unwrap();
    assert_eq!(loss, 1.0);

    let weight = graph.value(layer.weight_node().unwrap()).unwrap();
    let bias = graph.value(layer.bias_node().unwrap()).unwrap();
    assert_eq!(weight.data(), &[0.1]);
    assert_eq!(bias.data(), &[0.1]);
}

#[test]
fn train_step_sgd_with_hinge_loss_rejects_invalid_margin() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![0.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = Sgd::new(0.1).unwrap();
    let err = train_step_sgd_with_loss(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
        SupervisedLoss::Hinge { margin: 0.0 },
    )
    .unwrap_err();
    assert_eq!(err, ModelError::InvalidHingeMargin { margin: 0.0 });
}

#[test]
fn train_step_adam_updates_linear_parameters() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = Adam::new(0.1).unwrap();
    let loss = train_step_adam(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
    )
    .unwrap();
    assert_eq!(loss, 4.0);

    let weight = graph.value(layer.weight_node().unwrap()).unwrap();
    let bias = graph.value(layer.bias_node().unwrap()).unwrap();
    assert_eq!(weight.data(), &[1.1]);
    assert_eq!(bias.data(), &[0.1]);
}

#[test]
fn train_step_adamw_updates_linear_parameters() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = AdamW::new(0.1).unwrap().with_weight_decay(0.1).unwrap();
    let loss = train_step_adamw(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
    )
    .unwrap();
    assert_eq!(loss, 4.0);

    let weight = graph.value(layer.weight_node().unwrap()).unwrap();
    let bias = graph.value(layer.bias_node().unwrap()).unwrap();
    assert_eq!(weight.data(), &[1.09]);
    assert_eq!(bias.data(), &[0.1]);
}

#[test]
fn train_step_rmsprop_updates_linear_parameters() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let mut optimizer = RmsProp::new(0.1).unwrap();
    let loss = train_step_rmsprop(
        &mut graph,
        &mut optimizer,
        prediction,
        target,
        &[layer.weight_node().unwrap(), layer.bias_node().unwrap()],
    )
    .unwrap();
    assert_eq!(loss, 4.0);

    let weight = graph.value(layer.weight_node().unwrap()).unwrap();
    let bias = graph.value(layer.bias_node().unwrap()).unwrap();
    assert_slice_approx_eq(weight.data(), &[2.0], 1e-5);
    assert_slice_approx_eq(bias.data(), &[1.0], 1e-5);
}

#[test]
fn train_epoch_sgd_reduces_loss_on_simple_linear_problem() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();

    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = Sgd::new(0.05).unwrap();
    let first = train_epoch_sgd(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    let mut last = first;
    for _ in 0..39 {
        last = train_epoch_sgd(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
    assert!(last.mean_loss < 0.05);
}

#[test]
fn train_epoch_sgd_with_options_and_huber_loss_reduces_loss() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();

    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = Sgd::new(0.05).unwrap();
    let first = train_epoch_sgd_with_options_and_loss(
        &mut graph,
        &model,
        &mut optimizer,
        &dataset,
        EpochTrainOptions {
            batch_size: 4,
            batch_iter_options: BatchIterOptions::default(),
        },
        SupervisedLoss::Huber { delta: 1.0 },
    )
    .unwrap();
    let mut last = first;
    for _ in 0..39 {
        last = train_epoch_sgd_with_options_and_loss(
            &mut graph,
            &model,
            &mut optimizer,
            &dataset,
            EpochTrainOptions {
                batch_size: 4,
                batch_iter_options: BatchIterOptions::default(),
            },
            SupervisedLoss::Huber { delta: 1.0 },
        )
        .unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
}

#[test]
fn train_epoch_adam_reduces_loss_on_simple_linear_problem() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();

    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = Adam::new(0.05).unwrap();
    let first = train_epoch_adam(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    let mut last = first;
    for _ in 0..59 {
        last = train_epoch_adam(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
}

#[test]
fn train_epoch_adamw_reduces_loss_on_simple_linear_problem() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();

    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = AdamW::new(0.05).unwrap().with_weight_decay(0.01).unwrap();
    let first = train_epoch_adamw(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    let mut last = first;
    for _ in 0..59 {
        last = train_epoch_adamw(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
}

#[test]
fn train_epoch_rmsprop_reduces_loss_on_simple_linear_problem() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();

    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = RmsProp::new(0.01).unwrap().with_alpha(0.9).unwrap();
    let first = train_epoch_rmsprop_with_options(
        &mut graph,
        &model,
        &mut optimizer,
        &dataset,
        EpochTrainOptions {
            batch_size: 4,
            batch_iter_options: BatchIterOptions::default(),
        },
    )
    .unwrap();
    let mut last = first;
    for _ in 0..79 {
        last = train_epoch_rmsprop(&mut graph, &model, &mut optimizer, &dataset, 4).unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
}

#[test]
fn train_epoch_with_options_shuffled_batches_reduces_loss() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![8, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        Tensor::from_vec(vec![8, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = Sgd::new(0.02).unwrap();
    let first = train_epoch_sgd_with_options(
        &mut graph,
        &model,
        &mut optimizer,
        &dataset,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions {
                shuffle: true,
                shuffle_seed: 7,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        },
    )
    .unwrap();

    let mut last = first;
    for epoch in 1..40 {
        last = train_epoch_sgd_with_options(
            &mut graph,
            &model,
            &mut optimizer,
            &dataset,
            EpochTrainOptions {
                batch_size: 2,
                batch_iter_options: BatchIterOptions {
                    shuffle: true,
                    shuffle_seed: epoch as u64 + 7,
                    drop_last: false,
                    augmentation: None,
                    augmentation_seed: 0,
                    mixup: None,
                    mixup_seed: 0,
                    cutmix: None,
                    cutmix_seed: 0,
                    sampling: None,
                },
            },
        )
        .unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
    assert!(last.mean_loss < 0.1);
}

#[test]
fn train_epoch_adam_with_options_shuffled_batches_reduces_loss() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![8, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        Tensor::from_vec(vec![8, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = Adam::new(0.02).unwrap();
    let first = train_epoch_adam_with_options(
        &mut graph,
        &model,
        &mut optimizer,
        &dataset,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions {
                shuffle: true,
                shuffle_seed: 7,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        },
    )
    .unwrap();

    let mut last = first;
    for epoch in 1..40 {
        last = train_epoch_adam_with_options(
            &mut graph,
            &model,
            &mut optimizer,
            &dataset,
            EpochTrainOptions {
                batch_size: 2,
                batch_iter_options: BatchIterOptions {
                    shuffle: true,
                    shuffle_seed: epoch as u64 + 7,
                    drop_last: false,
                    augmentation: None,
                    augmentation_seed: 0,
                    mixup: None,
                    mixup_seed: 0,
                    cutmix: None,
                    cutmix_seed: 0,
                    sampling: None,
                },
            },
        )
        .unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
}

#[test]
fn train_epoch_adamw_with_options_shuffled_batches_reduces_loss() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![8, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        Tensor::from_vec(vec![8, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = AdamW::new(0.02).unwrap().with_weight_decay(0.01).unwrap();
    let first = train_epoch_adamw_with_options(
        &mut graph,
        &model,
        &mut optimizer,
        &dataset,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions {
                shuffle: true,
                shuffle_seed: 7,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        },
    )
    .unwrap();

    let mut last = first;
    for epoch in 1..40 {
        last = train_epoch_adamw_with_options(
            &mut graph,
            &model,
            &mut optimizer,
            &dataset,
            EpochTrainOptions {
                batch_size: 2,
                batch_iter_options: BatchIterOptions {
                    shuffle: true,
                    shuffle_seed: epoch as u64 + 7,
                    drop_last: false,
                    augmentation: None,
                    augmentation_seed: 0,
                    mixup: None,
                    mixup_seed: 0,
                    cutmix: None,
                    cutmix_seed: 0,
                    sampling: None,
                },
            },
        )
        .unwrap();
    }

    assert!(last.mean_loss < first.mean_loss);
}

#[test]
fn train_epochs_adamw_with_step_scheduler_tracks_lr_and_reduces_loss() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![8, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        Tensor::from_vec(vec![8, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = AdamW::new(0.02).unwrap().with_weight_decay(0.01).unwrap();
    let mut scheduler = StepLr::new(2, 0.5).unwrap();
    let per_epoch = train_epochs_adamw_with_scheduler(
        &mut graph,
        &model,
        &mut optimizer,
        &mut scheduler,
        &dataset,
        4,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions {
                shuffle: true,
                shuffle_seed: 7,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        },
    )
    .unwrap();

    assert_eq!(per_epoch.len(), 4);
    assert!(per_epoch[3].mean_loss < per_epoch[0].mean_loss);
    assert!((per_epoch[0].learning_rate - 0.02).abs() < 1e-8);
    assert!((per_epoch[1].learning_rate - 0.01).abs() < 1e-8);
    assert!((per_epoch[2].learning_rate - 0.01).abs() < 1e-8);
    assert!((per_epoch[3].learning_rate - 0.005).abs() < 1e-8);
}

#[test]
fn train_epochs_rmsprop_with_step_scheduler_tracks_lr_and_reduces_loss() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![8, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        Tensor::from_vec(vec![8, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = RmsProp::new(0.01)
        .unwrap()
        .with_alpha(0.9)
        .unwrap()
        .with_momentum(0.8)
        .unwrap();
    let mut scheduler = StepLr::new(2, 0.5).unwrap();
    let per_epoch = train_epochs_rmsprop_with_scheduler(
        &mut graph,
        &model,
        &mut optimizer,
        &mut scheduler,
        &dataset,
        4,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions {
                shuffle: true,
                shuffle_seed: 7,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        },
    )
    .unwrap();

    assert_eq!(per_epoch.len(), 4);
    assert!(per_epoch[3].mean_loss < per_epoch[0].mean_loss);
    assert!((per_epoch[0].learning_rate - 0.01).abs() < 1e-8);
    assert!((per_epoch[1].learning_rate - 0.005).abs() < 1e-8);
    assert!((per_epoch[2].learning_rate - 0.005).abs() < 1e-8);
    assert!((per_epoch[3].learning_rate - 0.0025).abs() < 1e-8);
}

#[test]
fn train_epochs_rmsprop_with_one_cycle_scheduler_tracks_lr_profile() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![8, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        Tensor::from_vec(vec![8, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = RmsProp::new(0.01).unwrap().with_alpha(0.9).unwrap();
    let mut scheduler = OneCycleLr::new(4, 0.1)
        .unwrap()
        .with_pct_start(0.5)
        .unwrap()
        .with_final_div_factor(10.0)
        .unwrap();
    let per_epoch = train_epochs_rmsprop_with_scheduler(
        &mut graph,
        &model,
        &mut optimizer,
        &mut scheduler,
        &dataset,
        4,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions {
                shuffle: true,
                shuffle_seed: 5,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        },
    )
    .unwrap();

    assert_eq!(per_epoch.len(), 4);
    assert!(per_epoch[0].learning_rate > 0.01);
    assert!((per_epoch[1].learning_rate - 0.1).abs() < 1e-8);
    assert!(per_epoch[2].learning_rate < per_epoch[1].learning_rate);
    assert!((per_epoch[3].learning_rate - 0.001).abs() < 1e-8);
}

#[test]
fn train_epochs_sgd_with_scheduler_rejects_zero_epoch_count() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear_zero(&mut graph, 1, 1).unwrap();
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap(),
    )
    .unwrap();

    let mut optimizer = Sgd::new(0.01).unwrap();
    let mut scheduler = CosineAnnealingLr::new(4, 0.0).unwrap();
    let err = train_epochs_sgd_with_scheduler(
        &mut graph,
        &model,
        &mut optimizer,
        &mut scheduler,
        &dataset,
        0,
        EpochTrainOptions {
            batch_size: 2,
            batch_iter_options: BatchIterOptions::default(),
        },
    )
    .unwrap_err();
    assert_eq!(err, ModelError::InvalidEpochCount { epochs: 0 });
}

// ── Gradient accumulation tests ──────────────────────────────────

#[test]
fn scale_gradients_scales_existing_grads() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let loss_node = crate::mse_loss(&mut graph, prediction, target).unwrap();
    graph.backward(loss_node).unwrap();

    let nodes = [layer.weight_node().unwrap(), layer.bias_node().unwrap()];
    let grad_before_w = graph
        .grad(layer.weight_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];
    let grad_before_b = graph
        .grad(layer.bias_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];

    scale_gradients(&mut graph, &nodes, 0.5).unwrap();

    let grad_after_w = graph
        .grad(layer.weight_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];
    let grad_after_b = graph
        .grad(layer.bias_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];
    assert_slice_approx_eq(&[grad_after_w], &[grad_before_w * 0.5], 1e-6);
    assert_slice_approx_eq(&[grad_after_b], &[grad_before_b * 0.5], 1e-6);
}

#[test]
fn collect_and_accumulate_gradients_round_trips() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let loss_node = crate::mse_loss(&mut graph, prediction, target).unwrap();
    graph.backward(loss_node).unwrap();

    let nodes = [layer.weight_node().unwrap(), layer.bias_node().unwrap()];
    let grads = collect_gradients(&graph, &nodes).unwrap();

    // Both grads should be Some.
    assert!(grads[0].is_some());
    assert!(grads[1].is_some());

    let w_grad = grads[0].as_ref().unwrap().data()[0];
    let b_grad = grads[1].as_ref().unwrap().data()[0];

    // Zero out and accumulate back -- should restore original.
    graph.zero_grads();
    accumulate_gradients(&mut graph, &nodes, &grads).unwrap();

    let w_grad_after = graph
        .grad(layer.weight_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];
    let b_grad_after = graph
        .grad(layer.bias_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];
    assert_slice_approx_eq(&[w_grad_after], &[w_grad], 1e-6);
    assert_slice_approx_eq(&[b_grad_after], &[b_grad], 1e-6);
}

#[test]
fn accumulate_gradients_adds_to_existing() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let input = graph.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer.forward(&mut graph, input).unwrap();

    let loss_node = crate::mse_loss(&mut graph, prediction, target).unwrap();
    graph.backward(loss_node).unwrap();

    let nodes = [layer.weight_node().unwrap(), layer.bias_node().unwrap()];
    let grads = collect_gradients(&graph, &nodes).unwrap();
    let w_grad = grads[0].as_ref().unwrap().data()[0];

    // Accumulate again -- should double.
    accumulate_gradients(&mut graph, &nodes, &grads).unwrap();
    let w_grad_doubled = graph
        .grad(layer.weight_node().unwrap())
        .unwrap()
        .unwrap()
        .data()[0];
    assert_slice_approx_eq(&[w_grad_doubled], &[w_grad * 2.0], 1e-6);
}

#[test]
fn train_step_sgd_with_accumulation_reduces_loss() {
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

    // Two micro-batches: (x=1, y=3) and (x=2, y=5) -- linear y=2x+1.
    let micro_batches = vec![
        (
            Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
            Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap(),
        ),
        (
            Tensor::from_vec(vec![1, 1], vec![2.0]).unwrap(),
            Tensor::from_vec(vec![1, 1], vec![5.0]).unwrap(),
        ),
    ];

    let mut optimizer = Sgd::new(0.05).unwrap();
    let mut first_loss = 0.0f32;

    for step in 0..20 {
        let mb = micro_batches.clone();
        let mut mb_iter = mb.into_iter();
        let p = persistent;
        let loss = train_step_sgd_with_accumulation(
            &mut graph,
            &mut optimizer,
            &trainable,
            2,
            SupervisedLoss::Mse,
            |g| {
                g.truncate(p).unwrap();
                let (inp, tgt) = mb_iter.next().unwrap();
                let input_node = g.variable(inp);
                let target_node = g.constant(tgt);
                let pred = layer.forward(g, input_node).unwrap();
                Ok((pred, target_node))
            },
        )
        .unwrap();
        if step == 0 {
            first_loss = loss;
        }
    }

    // After 20 accumulated-gradient steps the loss must have decreased.
    let mb = micro_batches.clone();
    let mut mb_iter = mb.into_iter();
    let p = persistent;
    let final_loss = train_step_sgd_with_accumulation(
        &mut graph,
        &mut optimizer,
        &trainable,
        2,
        SupervisedLoss::Mse,
        |g| {
            g.truncate(p).unwrap();
            let (inp, tgt) = mb_iter.next().unwrap();
            let input_node = g.variable(inp);
            let target_node = g.constant(tgt);
            let pred = layer.forward(g, input_node).unwrap();
            Ok((pred, target_node))
        },
    )
    .unwrap();

    assert!(
        final_loss < first_loss,
        "expected loss to decrease: first={first_loss}, final={final_loss}"
    );
}

#[test]
fn train_step_adam_with_accumulation_reduces_loss() {
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

    let micro_batches = vec![
        (
            Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
            Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap(),
        ),
        (
            Tensor::from_vec(vec![1, 1], vec![2.0]).unwrap(),
            Tensor::from_vec(vec![1, 1], vec![5.0]).unwrap(),
        ),
    ];

    let mut optimizer = Adam::new(0.05).unwrap();
    let mut first_loss = 0.0f32;

    for step in 0..20 {
        let mb = micro_batches.clone();
        let mut mb_iter = mb.into_iter();
        let p = persistent;
        let loss = train_step_adam_with_accumulation(
            &mut graph,
            &mut optimizer,
            &trainable,
            2,
            SupervisedLoss::Mse,
            |g| {
                g.truncate(p).unwrap();
                let (inp, tgt) = mb_iter.next().unwrap();
                let input_node = g.variable(inp);
                let target_node = g.constant(tgt);
                let pred = layer.forward(g, input_node).unwrap();
                Ok((pred, target_node))
            },
        )
        .unwrap();
        if step == 0 {
            first_loss = loss;
        }
    }

    // Measure final loss.
    let mb = micro_batches.clone();
    let mut mb_iter = mb.into_iter();
    let p = persistent;
    let final_loss = train_step_adam_with_accumulation(
        &mut graph,
        &mut optimizer,
        &trainable,
        2,
        SupervisedLoss::Mse,
        |g| {
            g.truncate(p).unwrap();
            let (inp, tgt) = mb_iter.next().unwrap();
            let input_node = g.variable(inp);
            let target_node = g.constant(tgt);
            let pred = layer.forward(g, input_node).unwrap();
            Ok((pred, target_node))
        },
    )
    .unwrap();

    assert!(
        final_loss < first_loss,
        "expected loss to decrease: first={first_loss}, final={final_loss}"
    );
}

#[test]
fn train_step_sgd_with_accumulation_rejects_zero_steps() {
    let mut graph = Graph::new();
    let layer = LinearLayer::new(
        &mut graph,
        1,
        1,
        Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    )
    .unwrap();

    let trainable = vec![layer.weight_node().unwrap(), layer.bias_node().unwrap()];
    let mut optimizer = Sgd::new(0.1).unwrap();

    let err = train_step_sgd_with_accumulation(
        &mut graph,
        &mut optimizer,
        &trainable,
        0,
        SupervisedLoss::Mse,
        |_g| unreachable!("should not be called"),
    )
    .unwrap_err();

    assert_eq!(err, ModelError::InvalidAccumulationSteps { steps: 0 });
}

#[test]
fn accumulation_with_one_step_matches_regular_train_step() {
    // With accumulation_steps=1, the result should match a regular train_step_sgd.
    let make_layer = |graph: &mut Graph| {
        LinearLayer::new(
            graph,
            1,
            1,
            Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(),
            Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
        )
        .unwrap()
    };

    // Regular train step.
    let mut graph_regular = Graph::new();
    let layer_regular = make_layer(&mut graph_regular);
    let input = graph_regular.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
    let target = graph_regular.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
    let prediction = layer_regular.forward(&mut graph_regular, input).unwrap();
    let mut opt_regular = Sgd::new(0.1).unwrap();
    let loss_regular = train_step_sgd(
        &mut graph_regular,
        &mut opt_regular,
        prediction,
        target,
        &[
            layer_regular.weight_node().unwrap(),
            layer_regular.bias_node().unwrap(),
        ],
    )
    .unwrap();

    // Accumulated train step with 1 micro-batch.
    let mut graph_accum = Graph::new();
    let layer_accum = make_layer(&mut graph_accum);
    let persistent = graph_accum.node_count();
    let trainable = vec![
        layer_accum.weight_node().unwrap(),
        layer_accum.bias_node().unwrap(),
    ];
    let mut opt_accum = Sgd::new(0.1).unwrap();
    let loss_accum = train_step_sgd_with_accumulation(
        &mut graph_accum,
        &mut opt_accum,
        &trainable,
        1,
        SupervisedLoss::Mse,
        |g| {
            g.truncate(persistent).unwrap();
            let inp = g.variable(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap());
            let tgt = g.constant(Tensor::from_vec(vec![1, 1], vec![3.0]).unwrap());
            let pred = layer_accum.forward(g, inp).unwrap();
            Ok((pred, tgt))
        },
    )
    .unwrap();

    assert_slice_approx_eq(&[loss_accum], &[loss_regular], 1e-6);

    // Weights should match.
    let w_regular = graph_regular
        .value(layer_regular.weight_node().unwrap())
        .unwrap()
        .data()[0];
    let w_accum = graph_accum
        .value(layer_accum.weight_node().unwrap())
        .unwrap()
        .data()[0];
    assert_slice_approx_eq(&[w_accum], &[w_regular], 1e-6);

    let b_regular = graph_regular
        .value(layer_regular.bias_node().unwrap())
        .unwrap()
        .data()[0];
    let b_accum = graph_accum
        .value(layer_accum.bias_node().unwrap())
        .unwrap()
        .data()[0];
    assert_slice_approx_eq(&[b_accum], &[b_regular], 1e-6);
}
