use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Applies gamma correction to each pixel: `pixel^gamma`.
///
/// Input is `[H, W, C]` with values typically in `[0, 1]`.
/// Negative pixel values are clamped to 0 before the power operation.
pub fn adjust_gamma(img: &Tensor, gamma: f32) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    let data = img.data();
    let mut out = vec![0.0f32; h * w * c];

    for i in 0..h * w * c {
        out[i] = data[i].max(0.0).powf(gamma);
    }

    Tensor::from_vec(vec![h, w, c], out).map_err(Into::into)
}

/// Linearly rescales pixel intensities from `[in_min, in_max]` to `[out_min, out_max]`.
///
/// Values below `in_min` are clamped to `out_min`, values above `in_max` to `out_max`.
/// Input is `[H, W, C]`.
pub fn rescale_intensity(
    img: &Tensor,
    in_min: f32,
    in_max: f32,
    out_min: f32,
    out_max: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    let data = img.data();
    let mut out = vec![0.0f32; h * w * c];

    let in_range = in_max - in_min;
    let out_range = out_max - out_min;

    if in_range.abs() < 1e-10 {
        // Degenerate input range: map everything to out_min
        for i in 0..h * w * c {
            out[i] = out_min;
        }
    } else {
        let scale = out_range / in_range;
        for i in 0..h * w * c {
            let v = (data[i] - in_min) * scale + out_min;
            out[i] = v.clamp(out_min.min(out_max), out_min.max(out_max));
        }
    }

    Tensor::from_vec(vec![h, w, c], out).map_err(Into::into)
}

/// Applies logarithmic transform: `log(1 + pixel)`, normalized to `[0, 1]`.
///
/// Input is `[H, W, C]` with non-negative values.
/// The result is scaled so that the maximum output value is 1.0.
pub fn adjust_log(img: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    let data = img.data();
    let mut out = vec![0.0f32; h * w * c];

    let mut max_val = 0.0f32;
    for i in 0..h * w * c {
        let v = (1.0 + data[i].max(0.0)).ln();
        out[i] = v;
        if v > max_val {
            max_val = v;
        }
    }

    // Normalize to [0, 1]
    if max_val > 1e-10 {
        let inv = 1.0 / max_val;
        for v in &mut out {
            *v *= inv;
        }
    }

    Tensor::from_vec(vec![h, w, c], out).map_err(Into::into)
}
