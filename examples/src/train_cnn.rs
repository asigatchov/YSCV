//! Example: Train a CNN classifier (Conv2d -> ReLU -> Flatten -> Linear).
//!
//! Demonstrates:
//! 1. Building a simple CNN with SequentialModel
//! 2. Creating synthetic [N, H, W, C] training data
//! 3. Training with Trainer and SGD optimizer
//! 4. Printing training results
//!
//! Usage: cargo run --example train_cnn

use yscv_autograd::Graph;
use yscv_model::{LossKind, OptimizerKind, SequentialModel, Trainer, TrainerConfig};
use yscv_tensor::Tensor;

fn main() {
    println!("Building CNN: Conv2d(1,4,3x3) -> ReLU -> Flatten -> Linear(4,2)\n");

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Conv2d layer: 1 input channel, 4 output channels, 3x3 kernel, stride 1x1
    model
        .add_conv2d_zero(1, 4, 3, 3, 1, 1, true)
        .expect("add_conv2d");
    model.add_relu();
    model.add_flatten();
    // After conv on a 4x4 input with 3x3 kernel stride 1 and no padding:
    // output spatial = (4 - 3)/1 + 1 = 2, so flattened = 4 channels * 2 * 2 = 16
    model
        .add_linear_zero(&mut graph, 16, 2)
        .expect("add_linear");

    // Create synthetic training data: 8 samples of 4x4 single-channel "images".
    // Class 0: low values, Class 1: high values.
    let n = 8;
    let h = 4;
    let w = 4;
    let c = 1;
    let mut input_data = Vec::with_capacity(n * h * w * c);
    let mut target_data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let base = if i < n / 2 { 0.1 } else { 0.9 };
        for _ in 0..(h * w * c) {
            input_data.push(base);
        }
        // One-hot target: class 0 for low, class 1 for high
        if i < n / 2 {
            target_data.push(1.0);
            target_data.push(0.0);
        } else {
            target_data.push(0.0);
            target_data.push(1.0);
        }
    }

    let inputs = Tensor::from_vec(vec![n, h, w, c], input_data).expect("inputs");
    let targets = Tensor::from_vec(vec![n, 2], target_data).expect("targets");

    // Configure training
    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd {
            lr: 0.01,
            momentum: 0.0,
        },
        loss: LossKind::Mse,
        epochs: 50,
        batch_size: n,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);

    println!("Training on {} synthetic samples for 50 epochs...\n", n);
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .expect("training failed");

    println!("Epochs trained: {}", result.epochs_trained);
    println!("Final loss:     {:.6}", result.final_loss);

    // Show loss progression
    let loss_history = result.log.get_metric_history("loss");
    println!("\nLoss progression (first 10 epochs):");
    for (i, loss) in loss_history.iter().take(10).enumerate() {
        println!("  Epoch {:>3}: {:.6}", i + 1, loss);
    }

    println!("\nDone!");
}
