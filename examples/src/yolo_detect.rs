//! Example: YOLOv8/v11 object detection with ONNX model.
//!
//! Demonstrates the full detection pipeline:
//! 1. Load an ONNX model (YOLOv8 or YOLOv11)
//! 2. Load and preprocess an image (letterbox)
//! 3. Run inference
//! 4. Decode detections with NMS
//! 5. Print results
//!
//! Usage:
//!   cargo run --example yolo_detect -- <model.onnx> <image.jpg>
//!
//! Example with YOLOv8:
//!   cargo run --example yolo_detect -- yolov8n.onnx photo.jpg
//!
//! Example with YOLOv11:
//!   cargo run --example yolo_detect -- yolo11n.onnx photo.jpg
//!
//! To get models:
//!   pip install ultralytics
//!   yolo export model=yolov8n.pt format=onnx   # YOLOv8
//!   yolo export model=yolo11n.pt format=onnx   # YOLOv11

use yscv_detect::{
    Detection, coco_labels, decode_yolov8_output, decode_yolov11_output, letterbox_preprocess,
    yolov8_coco_config,
};
use yscv_imgproc::imread;
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: yolo_detect <model.onnx> <image.jpg>");
        eprintln!();
        eprintln!("Supports both YOLOv8 and YOLOv11 ONNX models.");
        eprintln!("The format is auto-detected from the output tensor shape.");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example yolo_detect -- yolov8n.onnx photo.jpg");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let image_path = &args[2];

    // Step 1: Load ONNX model
    println!("Loading model: {model_path}");
    let model = load_onnx_model_from_file(model_path).expect("Failed to load ONNX model");
    println!("  Nodes: {}", model.nodes.len());

    // Step 2: Load and preprocess image
    println!("Loading image: {image_path}");
    let img = imread(image_path).expect("Failed to load image");
    let shape = img.shape().to_vec();
    let (orig_h, orig_w) = (shape[0], shape[1]);
    println!("  Size: {orig_w}x{orig_h}");

    let config = yolov8_coco_config();
    let (letterboxed, _scale, _pad_x, _pad_y) = letterbox_preprocess(&img, config.input_size);
    let sz = config.input_size;
    println!("  Preprocessed to {sz}x{sz}");

    // Convert HWC [H, W, 3] → NCHW [1, 3, H, W] for ONNX inference
    let hwc_data = letterboxed.data();
    let mut nchw = vec![0.0f32; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let src = (y * sz + x) * 3;
            for c in 0..3 {
                nchw[c * sz * sz + y * sz + x] = hwc_data[src + c];
            }
        }
    }
    let input_tensor =
        yscv_tensor::Tensor::from_vec(vec![1, 3, sz, sz], nchw).expect("Failed to create tensor");

    // Step 3: Run inference
    println!("Running inference...");
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("images".to_string(), input_tensor);
    let outputs = run_onnx_model(&model, inputs).expect("Inference failed");

    // Find the output tensor (usually "output0")
    let output = outputs.values().next().expect("Model produced no outputs");
    let out_shape = output.shape();
    println!("  Output shape: {:?}", out_shape);

    // Step 4: Auto-detect YOLOv8 vs YOLOv11 format and decode
    let detections: Vec<Detection> = if out_shape.len() == 3 && out_shape[1] < out_shape[2] {
        // [1, 84, 8400] → YOLOv8 format (features × predictions)
        println!("  Detected YOLOv8 output format");
        decode_yolov8_output(output, &config, orig_w, orig_h)
    } else {
        // [1, 8400, 84] → YOLOv11 format (predictions × features)
        println!("  Detected YOLOv11 output format");
        decode_yolov11_output(output, &config, orig_w, orig_h)
    };

    // Step 5: Print results
    let labels = coco_labels();
    println!("\nDetected {} objects:", detections.len());
    for (i, det) in detections.iter().enumerate() {
        let label = labels
            .get(det.class_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        println!(
            "  [{}] {} ({:.1}%) at ({:.0}, {:.0}, {:.0}, {:.0})",
            i + 1,
            label,
            det.score * 100.0,
            det.bbox.x1,
            det.bbox.y1,
            det.bbox.x2,
            det.bbox.y2,
        );
    }

    if detections.is_empty() {
        println!("  (no objects detected above confidence threshold)");
    }

    println!("\nDone!");
}
