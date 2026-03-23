//! Example: Train a simple linear regression model.
//!
//! Learns y = 2x + 1 using SGD with MSE loss and demonstrates:
//! - TrainerConfig with optimizer/loss selection
//! - EarlyStopping callback
//! - TrainingLog with CSV export
//!
//! Usage: cargo run --example train_linear

use yscv_autograd::Graph;
use yscv_model::{
    EarlyStopping, LossKind, MonitorMode, OptimizerKind, SequentialModel, Trainer, TrainerConfig,
};
use yscv_tensor::Tensor;

fn main() {
    // Training data: y = 2x + 1
    let inputs =
        Tensor::from_vec(vec![8, 1], vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).expect("inputs");
    let targets = Tensor::from_vec(vec![8, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("targets");

    // Build a single-layer linear model.
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
        .expect("add_linear");

    // Configure training.
    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd {
            lr: 0.01,
            momentum: 0.0,
        },
        loss: LossKind::Mse,
        epochs: 200,
        batch_size: 8,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);

    // Add early stopping: stop if loss doesn't improve by > 0.001 for 10 epochs.
    let es = EarlyStopping::new(10, 0.001, MonitorMode::Min);
    trainer.add_callback(Box::new(es));

    // Train!
    println!("Training y = 2x + 1 with SGD...\n");
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .expect("training failed");

    println!("Epochs trained: {}", result.epochs_trained);
    println!("Final loss:     {:.6}", result.final_loss);

    // Show loss history from the integrated TrainingLog.
    let loss_history = result.log.get_metric_history("loss");
    println!("\nLoss progression (first 10 epochs):");
    for (i, loss) in loss_history.iter().take(10).enumerate() {
        println!("  Epoch {:>3}: {:.6}", i + 1, loss);
    }

    // Export training log as CSV.
    let csv = result.log.to_csv();
    println!("\nCSV export (first 5 lines):");
    for line in csv.lines().take(5) {
        println!("  {line}");
    }
}
