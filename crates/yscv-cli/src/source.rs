use yscv_detect::BoundingBox;
use yscv_recognize::Recognizer;
use yscv_tensor::Tensor;
use yscv_video::{
    CameraConfig, CameraFrameSource, Frame, FrameSource, InMemoryFrameSource, resolve_camera_device,
};

use crate::config::{CliConfig, DetectTarget};
use crate::error::AppError;
use crate::util::embedding_from_bbox;

pub fn open_camera_source(cli: &CliConfig) -> Result<CameraFrameSource, AppError> {
    let device_index = if let Some(query) = cli.device_name_query.as_deref() {
        let selected = resolve_camera_device(query)?;
        println!(
            "camera device resolved: query=`{}` -> {}: {}",
            query, selected.index, selected.label
        );
        selected.index
    } else {
        cli.device_index
    };

    let source = CameraFrameSource::open(CameraConfig {
        device_index,
        width: cli.width,
        height: cli.height,
        fps: cli.fps,
    })?;
    Ok(source)
}

pub fn build_source(
    cli: &CliConfig,
    recognizer: &mut Recognizer,
) -> Result<Box<dyn FrameSource>, AppError> {
    if cli.camera {
        let source = open_camera_source(cli)?;
        return Ok(Box::new(source));
    }

    let frame_height = 6usize;
    let frame_width = 6usize;
    let frames = match cli.detect_target {
        DetectTarget::People => vec![
            frame_with_boxes(0, 0, frame_height, frame_width, &[(1, 1, 3, 3, 0.9)])?,
            frame_with_boxes(1, 33_333, frame_height, frame_width, &[(1, 2, 3, 4, 0.85)])?,
            frame_with_boxes(
                2,
                66_666,
                frame_height,
                frame_width,
                &[(1, 3, 3, 5, 0.9), (4, 0, 6, 2, 0.88)],
            )?,
        ],
        DetectTarget::Faces => vec![
            frame_with_rgb_boxes(
                0,
                0,
                frame_height,
                frame_width,
                &[(1, 1, 3, 3, [0.78, 0.60, 0.46])],
                [0.10, 0.10, 0.12],
            )?,
            frame_with_rgb_boxes(
                1,
                33_333,
                frame_height,
                frame_width,
                &[(1, 2, 3, 4, [0.79, 0.61, 0.47])],
                [0.10, 0.10, 0.12],
            )?,
            frame_with_rgb_boxes(
                2,
                66_666,
                frame_height,
                frame_width,
                &[
                    (1, 3, 3, 5, [0.78, 0.60, 0.46]),
                    (4, 0, 6, 2, [0.72, 0.55, 0.41]),
                ],
                [0.10, 0.10, 0.12],
            )?,
        ],
    };

    if recognizer.identities().is_empty() {
        let known_embedding = embedding_from_bbox(
            BoundingBox {
                x1: 1.0,
                y1: 1.0,
                x2: 3.0,
                y2: 3.0,
            },
            frame_width as f32,
            frame_height as f32,
        )?;
        recognizer.enroll("alice", known_embedding)?;
    }

    Ok(Box::new(InMemoryFrameSource::new(frames)))
}

fn frame_with_boxes(
    index: u64,
    ts_us: u64,
    height: usize,
    width: usize,
    boxes: &[(usize, usize, usize, usize, f32)],
) -> Result<Frame, AppError> {
    let mut data = vec![0.0f32; height * width];
    for (x1, y1, x2, y2, value) in boxes {
        for y in *y1..*y2 {
            for x in *x1..*x2 {
                data[y * width + x] = *value;
            }
        }
    }
    Ok(Frame::new(
        index,
        ts_us,
        Tensor::from_vec(vec![height, width, 1], data)?,
    )?)
}

fn frame_with_rgb_boxes(
    index: u64,
    ts_us: u64,
    height: usize,
    width: usize,
    boxes: &[(usize, usize, usize, usize, [f32; 3])],
    background_rgb: [f32; 3],
) -> Result<Frame, AppError> {
    let mut data = vec![0.0f32; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            data[base] = background_rgb[0];
            data[base + 1] = background_rgb[1];
            data[base + 2] = background_rgb[2];
        }
    }

    for (x1, y1, x2, y2, rgb) in boxes {
        for y in *y1..*y2 {
            for x in *x1..*x2 {
                let base = (y * width + x) * 3;
                data[base] = rgb[0];
                data[base + 1] = rgb[1];
                data[base + 2] = rgb[2];
            }
        }
    }

    Ok(Frame::new(
        index,
        ts_us,
        Tensor::from_vec(vec![height, width, 3], data)?,
    )?)
}
