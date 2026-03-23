use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Telea-style fast marching inpainting.
///
/// Fills in masked pixels (mask > 0.5) using a weighted average of known
/// neighbours within `radius`. The image must be `[H, W, C]` and the mask
/// must be `[H, W, 1]`. Iterates until all masked pixels are filled or no
/// further progress is made.
pub fn inpaint_telea(img: &Tensor, mask: &Tensor, radius: usize) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(img)?;
    let (mh, mw, mc) = hwc_shape(mask)?;
    if mc != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: mc,
        });
    }
    if mh != h || mw != w {
        return Err(ImgProcError::InvalidOutputDimensions {
            out_h: mh,
            out_w: mw,
        });
    }

    let mask_data = mask.data();
    let mut out: Vec<f32> = img.data().to_vec();
    let mut known: Vec<bool> = mask_data.iter().map(|&v| v <= 0.5).collect();
    let r = radius as isize;

    // Iterative filling: each pass fills masked pixels that have at least one
    // known neighbour within the radius, using inverse-distance weighting.
    // Repeat until nothing changes.
    loop {
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                if known[y * w + x] {
                    continue;
                }
                // Weighted average over known neighbours
                let mut weight_sum = 0.0f64;
                let mut chan_sums = vec![0.0f64; channels];
                for dy in -r..=r {
                    for dx in -r..=r {
                        let ny = y as isize + dy;
                        let nx = x as isize + dx;
                        if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
                            continue;
                        }
                        let nidx = ny as usize * w + nx as usize;
                        if !known[nidx] {
                            continue;
                        }
                        let dist_sq = (dx * dx + dy * dy) as f64;
                        if dist_sq == 0.0 {
                            continue;
                        }
                        let wt = 1.0 / dist_sq.sqrt();
                        weight_sum += wt;
                        for c in 0..channels {
                            chan_sums[c] += wt * out[nidx * channels + c] as f64;
                        }
                    }
                }
                if weight_sum > 0.0 {
                    for c in 0..channels {
                        out[(y * w + x) * channels + c] = (chan_sums[c] / weight_sum) as f32;
                    }
                    known[y * w + x] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}
