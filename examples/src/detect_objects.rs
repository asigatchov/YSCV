//! Example: Object detection with bounding box visualization.
//!
//! Demonstrates:
//! 1. Loading an image
//! 2. Running detection (simulated)
//! 3. Drawing bounding boxes and labels
//! 4. Saving annotated result
//! 5. Computing detection metrics
//!
//! Usage: cargo run --example detect_objects -- <image_path> <output_path>

use std::path::Path;

use yscv_imgproc::{DrawDetection, draw_detections, imread, imwrite};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (input_path, output_path) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        eprintln!("Usage: detect_objects <input_image> <output_image>");
        eprintln!();
        eprintln!("This example loads an image, draws synthetic detections,");
        eprintln!("and saves the annotated result.");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example detect_objects -- photo.jpg result.png");
        std::process::exit(1);
    };

    // Step 1: Load image.
    println!("Loading: {input_path}");
    let mut img = imread(Path::new(input_path)).expect("Failed to load image");
    let shape = img.shape().to_vec();
    println!("  Size: {}x{}", shape[1], shape[0]);

    // Step 2: Simulated detections (in a real app, use ONNX inference).
    let h = shape[0];
    let w = shape[1];
    let detections = vec![
        DrawDetection {
            x: w / 10,
            y: h / 10,
            width: w / 3,
            height: h / 2,
            score: 0.95,
            class_id: 0,
        },
        DrawDetection {
            x: w / 2,
            y: h / 4,
            width: w / 4,
            height: h / 3,
            score: 0.82,
            class_id: 1,
        },
    ];

    let labels = ["person", "car", "dog", "cat", "bicycle"];

    // Step 3: Draw detections on the image.
    println!("Drawing {} detections...", detections.len());
    draw_detections(&mut img, &detections, &labels).expect("Drawing failed");

    // Step 4: Save result.
    println!("Saving: {output_path}");
    imwrite(Path::new(output_path), &img).expect("Failed to save image");

    println!("Done!");
}
