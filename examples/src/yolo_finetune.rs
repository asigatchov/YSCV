//! Example: Fine-tune a detection model on custom data.
//!
//! Demonstrates:
//! 1. Building a simple detection head (Conv2d → ReLU → Conv2d)
//! 2. Creating synthetic training data (fake bounding boxes)
//! 3. Training with Adam optimizer and MSE loss
//! 4. Showing loss convergence
//!
//! In a real scenario, you would:
//! - Load a pretrained backbone (ResNet/MobileNet from ModelHub)
//! - Freeze backbone layers
//! - Train only the detection head on your labeled data
//! - Export to ONNX for deployment
//!
//! Usage: cargo run --example yolo_finetune

use yscv_autograd::Graph;
use yscv_model::{LossKind, OptimizerKind, SequentialModel, Trainer, TrainerConfig};
use yscv_tensor::Tensor;

fn main() {
    println!("=== Detection Head Fine-tuning Example ===\n");

    // Build a simple detection head:
    // Input: [N, 8, 8, 32] feature map (from backbone)
    // Output: [N, 8, 8, 5] — 5 values per cell: (cx, cy, w, h, objectness)
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Conv 1x1: 32 channels → 64 channels (pointwise, preserves spatial dims)
    model
        .add_conv2d_zero(32, 64, 1, 1, 1, 1, true)
        .expect("conv1");
    model.add_relu();

    // Conv 1x1: 64 channels → 5 channels (bbox + objectness)
    model
        .add_conv2d_zero(64, 5, 1, 1, 1, 1, true)
        .expect("conv2");

    println!("Model: Conv2d(32→64, 1x1) → ReLU → Conv2d(64→5, 1x1)");
    println!("Input:  [batch, 8, 8, 32] feature maps");
    println!("Output: [batch, 8, 8, 5]  (cx, cy, w, h, obj)\n");

    // Create synthetic training data
    let batch = 4;
    let h = 8;
    let w = 8;
    let in_ch = 32;
    let out_ch = 5;

    // Random feature maps as input (simulating backbone output)
    let mut input_data = Vec::with_capacity(batch * h * w * in_ch);
    for i in 0..(batch * h * w * in_ch) {
        input_data.push(((i % 997) as f32 * 0.001).sin().abs());
    }
    let inputs = Tensor::from_vec(vec![batch, h, w, in_ch], input_data).expect("inputs");

    // Target: fake ground truth boxes
    // Each cell predicts (cx, cy, w, h, objectness)
    // Most cells have objectness = 0 (no object), a few have objectness = 1
    let mut target_data = vec![0.0f32; batch * h * w * out_ch];
    for b in 0..batch {
        // Place one object per sample at different positions
        let obj_y = (b * 2 + 1) % h;
        let obj_x = (b * 3 + 2) % w;
        let idx = (b * h * w + obj_y * w + obj_x) * out_ch;
        target_data[idx] = 0.5; // cx (relative to cell)
        target_data[idx + 1] = 0.5; // cy
        target_data[idx + 2] = 0.3; // width (relative to image)
        target_data[idx + 3] = 0.4; // height
        target_data[idx + 4] = 1.0; // objectness (this cell has an object)
    }
    let targets = Tensor::from_vec(vec![batch, h, w, out_ch], target_data).expect("targets");

    // Configure training
    let config = TrainerConfig {
        optimizer: OptimizerKind::Adam { lr: 0.001 },
        loss: LossKind::Mse,
        epochs: 100,
        batch_size: batch,
        validation_split: None,
    };

    println!(
        "Training for {} epochs with Adam (lr=0.001)...\n",
        config.epochs
    );
    let mut trainer = Trainer::new(config);
    let result = trainer
        .fit(&mut model, &mut graph, &inputs, &targets)
        .expect("training failed");

    // Print results
    println!("Epochs trained: {}", result.epochs_trained);
    println!("Final loss:     {:.6}", result.final_loss);

    let history = result.log.get_metric_history("loss");
    println!("\nLoss progression:");
    for (i, loss) in history.iter().take(10).enumerate() {
        println!("  Epoch {:>3}: {:.6}", i + 1, loss);
    }
    if history.len() > 10 {
        println!("  ...");
        println!(
            "  Epoch {:>3}: {:.6}",
            history.len(),
            history.last().unwrap_or(&f32::NAN)
        );
    }

    println!("\n=== Next steps in a real project ===");
    println!("1. Replace synthetic data with real labeled images");
    println!("2. Use ModelHub to load a pretrained backbone:");
    println!("     let hub = ModelHub::new();");
    println!("     let weights = hub.load_weights(\"resnet50\")?;");
    println!("3. Freeze backbone, train only detection head");
    println!("4. Export to ONNX for deployment:");
    println!("     model.export_onnx(\"detector.onnx\")?;");
    println!("5. Deploy with yscv-onnx runtime");

    println!("\nDone!");
}
