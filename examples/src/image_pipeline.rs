//! Example: Image preprocessing pipeline with transforms.
//!
//! Demonstrates:
//! 1. Creating a Compose pipeline with Resize, Normalize, and ScaleValues
//! 2. Applying the pipeline to a dummy image tensor
//! 3. Wrapping a model in an InferencePipeline with pre/postprocessing
//! 4. Running end-to-end inference
//!
//! Usage: cargo run --example image_pipeline

use yscv_autograd::Graph;
use yscv_model::{
    Compose, InferencePipeline, Normalize, Resize, ScaleValues, SequentialModel, Transform,
};
use yscv_tensor::Tensor;

fn main() {
    // Step 1: Build a preprocessing pipeline.
    // Resize to 4x4, scale pixel values by 1/255, then channel-normalize.
    let preprocess = Compose::new()
        .add(Resize::new(4, 4))
        .add(ScaleValues::new(1.0 / 255.0))
        .add(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        ));

    // Step 2: Create a dummy 8x8 RGB image (values in [0, 255]).
    let h = 8;
    let w = 8;
    let c = 3;
    let data: Vec<f32> = (0..(h * w * c)).map(|i| (i % 256) as f32).collect();
    let image = Tensor::from_vec(vec![h, w, c], data).expect("image tensor");
    println!("Input image shape:  {:?}", image.shape());

    // Step 3: Apply the preprocessing pipeline.
    let preprocessed = preprocess.apply(&image).expect("preprocessing failed");
    println!("Preprocessed shape: {:?}", preprocessed.shape());
    println!(
        "First 6 values:     [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        preprocessed.data()[0],
        preprocessed.data()[1],
        preprocessed.data()[2],
        preprocessed.data()[3],
        preprocessed.data()[4],
        preprocessed.data()[5],
    );

    // Step 4: Build a simple model and wrap in InferencePipeline.
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    // A minimal model: just flatten + identity-like pass-through.
    model.add_flatten();
    model.add_relu();

    let pipeline = InferencePipeline::new(model).with_preprocess(move |input| {
        // Apply the same Compose transform chain
        let resized = Resize::new(4, 4).apply(input)?;
        let scaled = ScaleValues::new(1.0 / 255.0).apply(&resized)?;
        Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]).apply(&scaled)
    });

    // Step 5: Run the full pipeline.
    let output = pipeline.run(&image).expect("pipeline inference failed");
    println!("\nInferencePipeline output shape: {:?}", output.shape());
    println!("Output length: {}", output.data().len());

    println!("\nDone!");
}
