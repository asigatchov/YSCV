//! Example: Image classification with a pretrained model.
//!
//! Demonstrates the full inference pipeline:
//! 1. Load image from disk
//! 2. Apply ImageNet preprocessing
//! 3. Run forward inference through a pretrained model
//! 4. Print top-5 predictions
//!
//! Usage: cargo run --example classify_image -- <image_path>

use yscv_autograd::Graph;
use yscv_imgproc::{imagenet_preprocess, imread};
use yscv_model::{ModelArchitecture, ModelHub, build_resnet, remap_state_dict};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: classify_image <image_path>");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example classify_image -- photo.jpg");
        std::process::exit(1);
    }
    let image_path = std::path::Path::new(&args[1]);

    // Step 1: Load image.
    println!("Loading image: {}", image_path.display());
    let img = imread(image_path).expect("Failed to load image");
    println!("  Image shape: {:?}", img.shape());

    // Step 2: Preprocess (resize → crop → normalize → HWC→CHW).
    let input = imagenet_preprocess(&img).expect("Preprocessing failed");
    println!("  Preprocessed shape: {:?}", input.shape());

    // Step 3: Build model and load weights.
    let mut graph = Graph::new();
    let config = ModelArchitecture::ResNet18.config();
    let model = build_resnet(&mut graph, &config).expect("Failed to build ResNet18");
    println!("  Model: ResNet18 (1000 classes)");

    // Try to load pretrained weights from hub.
    let hub = ModelHub::new();
    match hub.load_weights("resnet18") {
        Ok(timm_weights) => {
            let weights = remap_state_dict(&timm_weights, ModelArchitecture::ResNet18);
            println!(
                "  Loaded {} weight tensors ({} mapped)",
                timm_weights.len(),
                weights.len()
            );
        }
        Err(e) => {
            eprintln!("  Warning: could not load pretrained weights: {e}");
            eprintln!("  Running with random initialization.");
        }
    }

    // Step 4: Inference.
    match model.forward_inference(&input) {
        Ok(output) => {
            let scores: &[f32] = output.data();
            // Find top-5 class indices by score.
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nTop-5 predictions:");
            for (rank, (class_id, score)) in indexed.iter().take(5).enumerate() {
                println!("  #{}: class {} (score: {:.4})", rank + 1, class_id, score);
            }
        }
        Err(e) => eprintln!("Inference failed: {e}"),
    }
}
