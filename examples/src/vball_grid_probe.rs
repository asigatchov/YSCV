use std::collections::HashMap;
use std::path::Path;

use yscv_imgproc::{imread, resize_bilinear, rgb_to_grayscale};
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};
use yscv_tensor::Tensor;

const INPUT_WIDTH: usize = 768;
const INPUT_HEIGHT: usize = 432;
const GRID_COLS: usize = 48;
const GRID_ROWS: usize = 27;
const SEQ_LEN: usize = 9;

fn preprocess_frame(path: &Path) -> Result<Tensor, Box<dyn std::error::Error>> {
    let rgb = imread(path)?;
    let gray = rgb_to_grayscale(&rgb)?;
    Ok(resize_bilinear(&gray, INPUT_HEIGHT, INPUT_WIDTH)?)
}

fn build_clip(frame: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let plane = INPUT_HEIGHT * INPUT_WIDTH;
    let mut data = vec![0.0f32; SEQ_LEN * plane];
    for channel in 0..SEQ_LEN {
        let start = channel * plane;
        data[start..start + plane].copy_from_slice(frame.data());
    }
    Ok(Tensor::from_vec(
        vec![1, SEQ_LEN, INPUT_HEIGHT, INPUT_WIDTH],
        data,
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: vball_grid_probe <model.onnx> <frame.jpg>");
        std::process::exit(1);
    }

    let model = load_onnx_model_from_file(&args[1])?;
    let input_name = model.inputs.first().ok_or("missing input")?.clone();
    let output_name = model.outputs.first().ok_or("missing output")?.clone();

    for name in [
        "features.0.dsconv.depthwise.weight",
        "features.0.dsconv.pointwise.weight",
        "features.0.dsconv.pointwise.bias",
        "head.0.weight",
        "head.0.bias",
    ] {
        if let Some(t) = model.initializers.get(name) {
            let data = t.data();
            let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            println!("init {name} shape={:?} min={min} max={max} mean={mean}", t.shape());
        } else {
            println!("missing initializer {name}");
        }
    }

    let frame = preprocess_frame(Path::new(&args[2]))?;
    let clip = build_clip(&frame)?;
    let mut inputs = HashMap::new();
    inputs.insert(input_name, clip);
    let outputs = run_onnx_model(&model, inputs)?;
    let output = outputs.get(&output_name).ok_or("missing model output")?;

    let data = output.data();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    println!("output shape {:?}", output.shape());
    println!("output stats min={min} max={max} mean={mean}");

    let plane = GRID_ROWS * GRID_COLS;
    for frame_idx in [0usize, 8usize] {
        let conf_base = (frame_idx * 3) * plane;
        let x_base = conf_base + plane;
        let y_base = x_base + plane;
        let conf = &data[conf_base..conf_base + plane];
        let (best_idx, &score) = conf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or("empty confidence")?;
        let row = best_idx / GRID_COLS;
        let col = best_idx % GRID_COLS;
        println!(
            "frame {frame_idx}: best=({row},{col}) score={} xoff={} yoff={}",
            score,
            data[x_base + best_idx],
            data[y_base + best_idx]
        );
    }

    Ok(())
}
