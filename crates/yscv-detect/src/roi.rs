use crate::DetectError;
use yscv_tensor::Tensor;

/// RoI Pooling: for each RoI, divides the region into an `output_size` grid
/// and max-pools each cell.
///
/// * `features` -- `[H, W, C]` single-image feature map.
/// * `rois` -- slice of `(x1, y1, x2, y2)` in feature-map coordinates.
/// * `output_size` -- `(out_h, out_w)`.
///
/// Returns a tensor of shape `[num_rois, out_h, out_w, C]`.
pub fn roi_pool(
    features: &Tensor,
    rois: &[(f32, f32, f32, f32)],
    output_size: (usize, usize),
) -> Result<Tensor, DetectError> {
    let shape = features.shape();
    if shape.len() != 3 {
        return Err(DetectError::InvalidMapShape {
            expected_rank: 3,
            got: shape.to_vec(),
        });
    }
    let feat_h = shape[0];
    let feat_w = shape[1];
    let channels = shape[2];
    let (out_h, out_w) = output_size;
    let num_rois = rois.len();

    let total = num_rois * out_h * out_w * channels;
    let mut data = vec![f32::NEG_INFINITY; total];

    for (roi_idx, &(rx1, ry1, rx2, ry2)) in rois.iter().enumerate() {
        let roi_h = (ry2 - ry1).max(0.0);
        let roi_w = (rx2 - rx1).max(0.0);
        let bin_h = roi_h / out_h as f32;
        let bin_w = roi_w / out_w as f32;

        for oh in 0..out_h {
            for ow in 0..out_w {
                let y_start = (ry1 + oh as f32 * bin_h).floor() as isize;
                let y_end = (ry1 + (oh + 1) as f32 * bin_h).ceil() as isize;
                let x_start = (rx1 + ow as f32 * bin_w).floor() as isize;
                let x_end = (rx1 + (ow + 1) as f32 * bin_w).ceil() as isize;

                let y_start = y_start.max(0) as usize;
                let y_end = (y_end as usize).min(feat_h);
                let x_start = x_start.max(0) as usize;
                let x_end = (x_end as usize).min(feat_w);

                if y_start >= y_end || x_start >= x_end {
                    // Empty bin -- leave as NEG_INFINITY (or could use 0).
                    let base = ((roi_idx * out_h + oh) * out_w + ow) * channels;
                    for c in 0..channels {
                        data[base + c] = 0.0;
                    }
                    continue;
                }

                for fy in y_start..y_end {
                    for fx in x_start..x_end {
                        for c in 0..channels {
                            let val = features.get(&[fy, fx, c])?;
                            let out_idx = ((roi_idx * out_h + oh) * out_w + ow) * channels + c;
                            if val > data[out_idx] {
                                data[out_idx] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        vec![num_rois, out_h, out_w, channels],
        data,
    )?)
}

/// RoI Align: bilinear-interpolation version of RoI pooling (no quantisation).
///
/// * `features` -- `[H, W, C]` single-image feature map.
/// * `rois` -- slice of `(x1, y1, x2, y2)` in feature-map coordinates.
/// * `output_size` -- `(out_h, out_w)`.
/// * `sampling_ratio` -- number of sampling points per bin dimension
///   (0 means adaptive: `ceil(bin_size)`).
///
/// Returns a tensor of shape `[num_rois, out_h, out_w, C]`.
pub fn roi_align(
    features: &Tensor,
    rois: &[(f32, f32, f32, f32)],
    output_size: (usize, usize),
    sampling_ratio: usize,
) -> Result<Tensor, DetectError> {
    let shape = features.shape();
    if shape.len() != 3 {
        return Err(DetectError::InvalidMapShape {
            expected_rank: 3,
            got: shape.to_vec(),
        });
    }
    let feat_h = shape[0];
    let feat_w = shape[1];
    let channels = shape[2];
    let (out_h, out_w) = output_size;
    let num_rois = rois.len();

    let total = num_rois * out_h * out_w * channels;
    let mut data = vec![0.0f32; total];

    for (roi_idx, &(rx1, ry1, rx2, ry2)) in rois.iter().enumerate() {
        let roi_h = (ry2 - ry1).max(1e-6);
        let roi_w = (rx2 - rx1).max(1e-6);
        let bin_h = roi_h / out_h as f32;
        let bin_w = roi_w / out_w as f32;

        let sample_h = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            bin_h.ceil() as usize
        };
        let sample_w = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            bin_w.ceil() as usize
        };

        let count = (sample_h * sample_w) as f32;

        for oh in 0..out_h {
            for ow in 0..out_w {
                let base = ((roi_idx * out_h + oh) * out_w + ow) * channels;

                for sy in 0..sample_h {
                    let y = ry1 + bin_h * (oh as f32 + (sy as f32 + 0.5) / sample_h as f32);
                    for sx in 0..sample_w {
                        let x = rx1 + bin_w * (ow as f32 + (sx as f32 + 0.5) / sample_w as f32);

                        // Bilinear interpolation.
                        if y < -1.0 || y > feat_h as f32 || x < -1.0 || x > feat_w as f32 {
                            continue; // outside -- contributes 0
                        }

                        let y = y.max(0.0).min((feat_h - 1) as f32);
                        let x = x.max(0.0).min((feat_w - 1) as f32);

                        let y_low = y.floor() as usize;
                        let x_low = x.floor() as usize;
                        let y_high = (y_low + 1).min(feat_h - 1);
                        let x_high = (x_low + 1).min(feat_w - 1);

                        let ly = y - y_low as f32;
                        let lx = x - x_low as f32;
                        let hy = 1.0 - ly;
                        let hx = 1.0 - lx;

                        let w1 = hy * hx;
                        let w2 = hy * lx;
                        let w3 = ly * hx;
                        let w4 = ly * lx;

                        for c in 0..channels {
                            let v1 = features.get(&[y_low, x_low, c])?;
                            let v2 = features.get(&[y_low, x_high, c])?;
                            let v3 = features.get(&[y_high, x_low, c])?;
                            let v4 = features.get(&[y_high, x_high, c])?;
                            data[base + c] += (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4) / count;
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        vec![num_rois, out_h, out_w, channels],
        data,
    )?)
}
