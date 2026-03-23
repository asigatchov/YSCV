mod attention;
mod augmentation;
mod backward_new_layers;
mod batch_infer;
mod blocks;
mod callbacks;
mod checkpoint;
mod data_loader;
mod dataset;
mod decoder;
mod distributed;
mod ema;
mod fusion;
mod hub;
mod layers;
mod lora;
mod loss;
mod lr_finder;
mod mixed_precision;
mod pipeline;
mod quantize;
mod recurrent;
mod safetensors;
mod sequential;
mod train;
mod trainer;
mod training_log;
mod transform;
mod zoo;

use image::{ImageBuffer, Rgb};
use std::path::PathBuf;

fn assert_slice_approx_eq(left: &[f32], right: &[f32], eps: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (a - b).abs() <= eps,
            "index={idx} left={a} right={b} eps={eps}"
        );
    }
}

fn unique_temp_path(prefix: &str) -> PathBuf {
    unique_temp_path_with_extension(prefix, "jsonl")
}

fn unique_temp_path_with_extension(prefix: &str, extension: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "{prefix}-{}-{now_ns}.{extension}",
        std::process::id()
    ));
    path
}

fn write_test_rgb_png(path: &std::path::Path) {
    let image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
        2,
        2,
        vec![
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            255, 255, 255, // white
        ],
    )
    .unwrap();
    image.save(path).unwrap();
}

fn write_solid_rgb_png(path: &std::path::Path, rgb: [u8; 3]) {
    let image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
        2,
        2,
        vec![
            rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2], rgb[0], rgb[1],
            rgb[2],
        ],
    )
    .unwrap();
    image.save(path).unwrap();
}
