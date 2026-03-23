use std::fs;
use std::path::Path;
use std::time::Duration;

use yscv_detect::BoundingBox;
use yscv_tensor::Tensor;

use crate::error::AppError;

pub fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub fn face_min_area(frame_width: usize, frame_height: usize) -> usize {
    let frame_area = frame_width.saturating_mul(frame_height);
    ((frame_area as f32 * 0.003).round() as usize).max(4)
}

pub fn embedding_from_bbox(
    bbox: BoundingBox,
    frame_width: f32,
    frame_height: f32,
) -> Result<Tensor, AppError> {
    let cx = ((bbox.x1 + bbox.x2) * 0.5) / frame_width;
    let cy = ((bbox.y1 + bbox.y2) * 0.5) / frame_height;
    let area = bbox.area() / (frame_width * frame_height);
    Ok(Tensor::from_vec(vec![3], vec![cx, cy, area])?)
}
