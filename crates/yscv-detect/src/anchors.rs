use crate::BoundingBox;

/// Generates anchor boxes for a feature map.
///
/// For each grid cell `(y, x)` and each `(size, ratio)` pair, produces an
/// anchor in `(x1, y1, x2, y2)` format centred on the cell.
pub fn generate_anchors(
    feature_map_height: usize,
    feature_map_width: usize,
    sizes: &[f32],
    aspect_ratios: &[f32],
    stride: f32,
) -> Vec<BoundingBox> {
    let num_anchors_per_cell = sizes.len() * aspect_ratios.len();
    let total = feature_map_height * feature_map_width * num_anchors_per_cell;
    let mut anchors = Vec::with_capacity(total);

    for y in 0..feature_map_height {
        let cy = (y as f32 + 0.5) * stride;
        for x in 0..feature_map_width {
            let cx = (x as f32 + 0.5) * stride;
            for &size in sizes {
                for &ratio in aspect_ratios {
                    let sqrt_ratio = ratio.sqrt();
                    let w = size * sqrt_ratio;
                    let h = size / sqrt_ratio;
                    anchors.push(BoundingBox {
                        x1: cx - w * 0.5,
                        y1: cy - h * 0.5,
                        x2: cx + w * 0.5,
                        y2: cy + h * 0.5,
                    });
                }
            }
        }
    }

    anchors
}
