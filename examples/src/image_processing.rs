//! Example: Image processing pipeline.
//!
//! Demonstrates common imgproc operations:
//! 1. Load image
//! 2. Convert to grayscale
//! 3. Apply Gaussian blur
//! 4. Detect edges with Canny
//! 5. Apply threshold
//! 6. Save results
//!
//! Usage: cargo run --example image_processing -- <image_path> <output_dir>

use std::path::Path;

use yscv_imgproc::{canny, gaussian_blur_3x3, imread, imwrite, rgb_to_grayscale, threshold_binary};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: image_processing <input_image> <output_dir>");
        eprintln!();
        eprintln!("Applies a chain of image processing operations and saves");
        eprintln!("intermediate results to the output directory.");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example image_processing -- photo.jpg ./output/");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_dir = Path::new(&args[2]);
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Step 1: Load as RGB.
    println!("Loading: {}", input_path.display());
    let rgb = imread(input_path).expect("Failed to load image");
    println!("  Shape: {:?}", rgb.shape());

    // Step 2: Convert to grayscale.
    let gray = rgb_to_grayscale(&rgb).expect("Grayscale conversion failed");
    println!("  Grayscale: {:?}", gray.shape());
    imwrite(output_dir.join("01_grayscale.png"), &gray).expect("Save failed");

    // Step 3: Gaussian blur.
    let blurred = gaussian_blur_3x3(&gray).expect("Blur failed");
    println!("  Blurred: {:?}", blurred.shape());
    imwrite(output_dir.join("02_blurred.png"), &blurred).expect("Save failed");

    // Step 4: Canny edge detection.
    let edges = canny(&blurred, 0.1, 0.3).expect("Canny failed");
    println!("  Edges: {:?}", edges.shape());
    imwrite(output_dir.join("03_edges.png"), &edges).expect("Save failed");

    // Step 5: Binary threshold.
    let binary = threshold_binary(&gray, 0.5, 1.0).expect("Threshold failed");
    println!("  Binary: {:?}", binary.shape());
    imwrite(output_dir.join("04_binary.png"), &binary).expect("Save failed");

    println!("\nResults saved to: {}", output_dir.display());
}
