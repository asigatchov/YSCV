use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{
    EarlyStopping, LossKind, MonitorMode, OptimizerKind, SequentialModel, Trainer, TrainerConfig,
};

#[test]
fn test_trainer_default_config() {
    let config = TrainerConfig::default();
    assert_eq!(config.epochs, 10);
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.loss, LossKind::Mse);
    assert_eq!(
        config.optimizer,
        OptimizerKind::Sgd {
            lr: 0.01,
            momentum: 0.0,
        }
    );
    assert_eq!(config.validation_split, None);
}

#[test]
fn test_trainer_basic_sgd_mse() {
    // Train y = 2x with a single linear layer.
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            1,
            1,
            Tensor::from_vec(vec![1, 1], vec![0.5]).unwrap(),
            Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
        )
        .unwrap();

    // 4 training samples: x = [1, 2, 3, 4], y = [2, 4, 6, 8]
    let inputs = Tensor::from_vec(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let targets = Tensor::from_vec(vec![4, 1], vec![2.0, 4.0, 6.0, 8.0]).unwrap();

    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd {
            lr: 0.001,
            momentum: 0.0,
        },
        loss: LossKind::Mse,
        epochs: 20,
        batch_size: 4,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .unwrap();

    assert_eq!(result.epochs_trained, 20);
    assert_eq!(result.history.len(), 20);

    // Loss should decrease over training.
    let first_loss = result.history[0]["loss"];
    let last_loss = result.final_loss;
    assert!(
        last_loss < first_loss,
        "loss should decrease: first={first_loss} last={last_loss}"
    );
}

#[test]
fn test_trainer_with_early_stopping() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            1,
            1,
            Tensor::from_vec(vec![1, 1], vec![0.5]).unwrap(),
            Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
        )
        .unwrap();

    let inputs = Tensor::from_vec(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let targets = Tensor::from_vec(vec![4, 1], vec![2.0, 4.0, 6.0, 8.0]).unwrap();

    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd {
            lr: 0.0001,
            momentum: 0.0,
        },
        loss: LossKind::Mse,
        epochs: 1000,
        batch_size: 4,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);
    // Very tight early stopping: patience=2, min_delta=1.0 means it will stop
    // almost immediately since improvements per epoch with tiny lr will be < 1.0.
    let es = EarlyStopping::new(2, 1.0, MonitorMode::Min);
    trainer.add_callback(Box::new(es));

    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .unwrap();

    // Should stop well before 1000 epochs.
    assert!(
        result.epochs_trained < 1000,
        "early stopping should have triggered, got {} epochs",
        result.epochs_trained
    );
}

#[test]
fn test_trainer_adam_mse() {
    // Test Adam optimizer with MSE loss on y = 3x + 1
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            1,
            1,
            Tensor::from_vec(vec![1, 1], vec![0.1]).unwrap(),
            Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
        )
        .unwrap();

    let inputs = Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let targets = Tensor::from_vec(vec![4, 1], vec![1.0, 4.0, 7.0, 10.0]).unwrap();

    let config = TrainerConfig {
        optimizer: OptimizerKind::Adam { lr: 0.05 },
        loss: LossKind::Mse,
        epochs: 30,
        batch_size: 4,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .unwrap();

    assert_eq!(result.epochs_trained, 30);
    assert_eq!(result.history.len(), 30);

    // Loss should decrease.
    let first_loss = result.history[0]["loss"];
    let last_loss = result.final_loss;
    assert!(
        last_loss < first_loss,
        "cross-entropy loss should decrease: first={first_loss} last={last_loss}"
    );
}

#[test]
fn fit_with_validation_split() {
    // Train y = 2x with a 70/30 validation split.
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            1,
            1,
            Tensor::from_vec(vec![1, 1], vec![0.5]).unwrap(),
            Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
        )
        .unwrap();

    // 10 samples so the 30% split gives 3 val samples.
    let inputs = Tensor::from_vec(
        vec![10, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let targets = Tensor::from_vec(
        vec![10, 1],
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )
    .unwrap();

    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd {
            lr: 0.001,
            momentum: 0.0,
        },
        loss: LossKind::Mse,
        epochs: 10,
        batch_size: 7,
        validation_split: Some(0.3),
    };

    let mut trainer = Trainer::new(config);
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .unwrap();

    assert_eq!(result.epochs_trained, 10);

    // Every epoch should have both "loss" and "val_loss" recorded.
    for (i, entry) in result.history.iter().enumerate() {
        assert!(entry.contains_key("loss"), "epoch {i} missing 'loss'");
        assert!(
            entry.contains_key("val_loss"),
            "epoch {i} missing 'val_loss'"
        );
    }

    // The training log should also contain val_loss history.
    let val_history = result.log.get_metric_history("val_loss");
    assert_eq!(val_history.len(), 10);

    // val_loss values should be finite.
    for v in &val_history {
        assert!(v.is_finite(), "val_loss should be finite, got {v}");
    }
}

#[test]
fn fit_without_validation() {
    // Backward-compatible: no validation_split means no val_loss in metrics.
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            1,
            1,
            Tensor::from_vec(vec![1, 1], vec![0.5]).unwrap(),
            Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
        )
        .unwrap();

    let inputs = Tensor::from_vec(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let targets = Tensor::from_vec(vec![4, 1], vec![2.0, 4.0, 6.0, 8.0]).unwrap();

    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd {
            lr: 0.001,
            momentum: 0.0,
        },
        loss: LossKind::Mse,
        epochs: 5,
        batch_size: 4,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .unwrap();

    assert_eq!(result.epochs_trained, 5);

    // No val_loss should appear in any epoch.
    for (i, entry) in result.history.iter().enumerate() {
        assert!(entry.contains_key("loss"), "epoch {i} missing 'loss'");
        assert!(
            !entry.contains_key("val_loss"),
            "epoch {i} should not have 'val_loss' without validation_split"
        );
    }

    // Training log should have no val_loss entries.
    let val_history = result.log.get_metric_history("val_loss");
    assert!(val_history.is_empty());
}
