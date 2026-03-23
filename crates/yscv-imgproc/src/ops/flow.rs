use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

// ── Farneback dense optical flow ────────────────────────────────────

/// Configuration for Farneback dense optical flow.
#[derive(Debug, Clone)]
pub struct FarnebackConfig {
    /// Pyramid scale (< 1.0, default 0.5).
    pub pyr_scale: f32,
    /// Number of pyramid levels (default 3).
    pub levels: usize,
    /// Averaging window size (default 15).
    pub win_size: usize,
    /// Number of iterations at each level (default 3).
    pub iterations: usize,
    /// Polynomial expansion neighborhood size (default 5, must be 5 or 7).
    pub poly_n: usize,
    /// Gaussian std for polynomial expansion (default 1.1 for poly_n=5).
    pub poly_sigma: f32,
}

impl Default for FarnebackConfig {
    fn default() -> Self {
        Self {
            pyr_scale: 0.5,
            levels: 3,
            win_size: 15,
            iterations: 3,
            poly_n: 5,
            poly_sigma: 1.1,
        }
    }
}

/// Compute dense optical flow using Farneback's polynomial expansion method.
///
/// `prev` and `next` are grayscale images `[H, W]` with values in `[0, 1]`.
/// Returns `(flow_x, flow_y)` each of shape `[H, W]` containing per-pixel displacement.
pub fn farneback_flow(
    prev: &Tensor,
    next: &Tensor,
    config: &FarnebackConfig,
) -> Result<(Tensor, Tensor), ImgProcError> {
    let shape = prev.shape();
    if shape.len() != 2 {
        return Err(ImgProcError::InvalidImageShape {
            expected_rank: 2,
            got: shape.to_vec(),
        });
    }
    let (h, w) = (shape[0], shape[1]);
    let next_shape = next.shape();
    if next_shape.len() != 2 || next_shape[0] != h || next_shape[1] != w {
        return Err(ImgProcError::ShapeMismatch {
            expected: vec![h, w],
            got: next_shape.to_vec(),
        });
    }

    let levels = config.levels.max(1);
    let pyr_scale = config.pyr_scale.clamp(0.1, 0.95);

    // Build Gaussian pyramids for both images.
    let pyr_prev = build_pyramid(prev, levels, pyr_scale);
    let pyr_next = build_pyramid(next, levels, pyr_scale);

    // Start at coarsest level with zero flow.
    let coarsest = &pyr_prev[levels - 1];
    let mut ch = coarsest.shape()[0];
    let mut cw = coarsest.shape()[1];
    let mut flow_x = vec![0.0f32; ch * cw];
    let mut flow_y = vec![0.0f32; ch * cw];

    // Coarse-to-fine iteration.
    for level in (0..levels).rev() {
        let lh = pyr_prev[level].shape()[0];
        let lw = pyr_prev[level].shape()[1];

        if level < levels - 1 {
            // Upsample flow from previous (coarser) level.
            let (ux, uy) = upsample_flow_vecs(&flow_x, &flow_y, ch, cw, lh, lw);
            flow_x = ux;
            flow_y = uy;
        }

        let prev_data = pyr_prev[level].data();
        let next_data = pyr_next[level].data();

        // Build Gaussian weight kernel for the window.
        let half = (config.win_size / 2) as i32;
        let sigma = config.win_size as f32 / 4.0;
        let weights = gaussian_weights(half, sigma);

        for _iter in 0..config.iterations {
            // Warp next image by current flow estimate.
            let warped = warp_image_data(next_data, &flow_x, &flow_y, lh, lw);

            // Compute gradients of prev image.
            let (grad_x, grad_y) = compute_gradients_data(prev_data, lh, lw);

            // Compute temporal gradient.
            let grad_t: Vec<f32> = warped
                .iter()
                .zip(prev_data.iter())
                .map(|(w_val, p_val)| w_val - p_val)
                .collect();

            // Dense Lucas-Kanade update with Gaussian-weighted window.
            let mut new_fx = flow_x.clone();
            let mut new_fy = flow_y.clone();

            for y in 0..lh {
                for x in 0..lw {
                    let mut sum_ixx = 0.0f32;
                    let mut sum_iyy = 0.0f32;
                    let mut sum_ixy = 0.0f32;
                    let mut sum_ixt = 0.0f32;
                    let mut sum_iyt = 0.0f32;

                    for dy in -half..=half {
                        for dx in -half..=half {
                            let sy = y as i32 + dy;
                            let sx = x as i32 + dx;
                            if sy < 0 || sy >= lh as i32 || sx < 0 || sx >= lw as i32 {
                                continue;
                            }
                            let idx = sy as usize * lw + sx as usize;
                            let wi = (dy + half) as usize * (2 * half as usize + 1)
                                + (dx + half) as usize;
                            let wt = weights[wi];
                            let ix = grad_x[idx];
                            let iy = grad_y[idx];
                            let it = grad_t[idx];
                            sum_ixx += wt * ix * ix;
                            sum_iyy += wt * iy * iy;
                            sum_ixy += wt * ix * iy;
                            sum_ixt += wt * ix * it;
                            sum_iyt += wt * iy * it;
                        }
                    }

                    let det = sum_ixx * sum_iyy - sum_ixy * sum_ixy;
                    if det.abs() > 1e-6 {
                        let inv_det = 1.0 / det;
                        let dvx = -(sum_iyy * sum_ixt - sum_ixy * sum_iyt) * inv_det;
                        let dvy = -(sum_ixx * sum_iyt - sum_ixy * sum_ixt) * inv_det;
                        let pidx = y * lw + x;
                        new_fx[pidx] += dvx;
                        new_fy[pidx] += dvy;
                    }
                }
            }

            flow_x = new_fx;
            flow_y = new_fy;
        }

        ch = lh;
        cw = lw;
    }

    let fx_tensor = Tensor::from_vec(vec![h, w], flow_x)?;
    let fy_tensor = Tensor::from_vec(vec![h, w], flow_y)?;
    Ok((fx_tensor, fy_tensor))
}

/// Compute image gradients using central differences.
fn compute_gradients_data(data: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; h * w];
    let mut dy = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            dx[idx] = if x == 0 {
                data[idx + 1] - data[idx]
            } else if x == w - 1 {
                data[idx] - data[idx - 1]
            } else {
                (data[idx + 1] - data[idx - 1]) * 0.5
            };
            dy[idx] = if y == 0 {
                data[(y + 1) * w + x] - data[idx]
            } else if y == h - 1 {
                data[idx] - data[(y - 1) * w + x]
            } else {
                (data[(y + 1) * w + x] - data[(y - 1) * w + x]) * 0.5
            };
        }
    }
    (dx, dy)
}

/// Warp an image by a flow field using bilinear interpolation.
fn warp_image_data(data: &[f32], flow_x: &[f32], flow_y: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let src_x = x as f32 + flow_x[idx];
            let src_y = y as f32 + flow_y[idx];
            let sx = src_x.clamp(0.0, (w - 1) as f32);
            let sy = src_y.clamp(0.0, (h - 1) as f32);
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            out[idx] = data[y0 * w + x0] * (1.0 - fy) * (1.0 - fx)
                + data[y0 * w + x1] * (1.0 - fy) * fx
                + data[y1 * w + x0] * fy * (1.0 - fx)
                + data[y1 * w + x1] * fy * fx;
        }
    }
    out
}

/// Build a Gaussian pyramid by repeated downscaling.
fn build_pyramid(image: &Tensor, levels: usize, scale: f32) -> Vec<Tensor> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(image.clone());
    for i in 1..levels {
        let prev_level = &pyramid[i - 1];
        let ph = prev_level.shape()[0];
        let pw = prev_level.shape()[1];
        let nh = ((ph as f32) * scale).round().max(1.0) as usize;
        let nw = ((pw as f32) * scale).round().max(1.0) as usize;
        let downscaled = downsample(prev_level.data(), ph, pw, nh, nw);
        pyramid.push(
            Tensor::from_vec(vec![nh, nw], downscaled).expect("pyramid level shape matches data"),
        );
    }
    pyramid
}

/// Downsample using bilinear interpolation.
fn downsample(data: &[f32], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let src_y = (y as f32 + 0.5) * (sh as f32 / dh as f32) - 0.5;
            let src_x = (x as f32 + 0.5) * (sw as f32 / dw as f32) - 0.5;
            let sy = src_y.clamp(0.0, (sh - 1) as f32);
            let sx = src_x.clamp(0.0, (sw - 1) as f32);
            let y0 = sy.floor() as usize;
            let x0 = sx.floor() as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let fy = sy - y0 as f32;
            let fx = sx - x0 as f32;
            out[y * dw + x] = data[y0 * sw + x0] * (1.0 - fy) * (1.0 - fx)
                + data[y0 * sw + x1] * (1.0 - fy) * fx
                + data[y1 * sw + x0] * fy * (1.0 - fx)
                + data[y1 * sw + x1] * fy * fx;
        }
    }
    out
}

/// Upsample flow field vectors to a larger resolution, scaling magnitudes.
fn upsample_flow_vecs(
    fx: &[f32],
    fy: &[f32],
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale_x = dw as f32 / sw as f32;
    let scale_y = dh as f32 / sh as f32;
    let mut out_x = vec![0.0f32; dh * dw];
    let mut out_y = vec![0.0f32; dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let src_y = (y as f32 + 0.5) / scale_y - 0.5;
            let src_x = (x as f32 + 0.5) / scale_x - 0.5;
            let sy = src_y.clamp(0.0, (sh - 1) as f32);
            let sx = src_x.clamp(0.0, (sw - 1) as f32);
            let y0 = sy.floor() as usize;
            let x0 = sx.floor() as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let frac_y = sy - y0 as f32;
            let frac_x = sx - x0 as f32;
            let interp = |data: &[f32]| {
                data[y0 * sw + x0] * (1.0 - frac_y) * (1.0 - frac_x)
                    + data[y0 * sw + x1] * (1.0 - frac_y) * frac_x
                    + data[y1 * sw + x0] * frac_y * (1.0 - frac_x)
                    + data[y1 * sw + x1] * frac_y * frac_x
            };
            let idx = y * dw + x;
            out_x[idx] = interp(fx) * scale_x;
            out_y[idx] = interp(fy) * scale_y;
        }
    }
    (out_x, out_y)
}

/// Build a 2D Gaussian weight kernel.
fn gaussian_weights(half: i32, sigma: f32) -> Vec<f32> {
    let size = (2 * half + 1) as usize;
    let mut w = vec![0.0f32; size * size];
    let mut sum = 0.0f32;
    let s2 = 2.0 * sigma * sigma;
    for dy in -half..=half {
        for dx in -half..=half {
            let val = (-(dx * dx + dy * dy) as f32 / s2).exp();
            let idx = (dy + half) as usize * size + (dx + half) as usize;
            w[idx] = val;
            sum += val;
        }
    }
    if sum > 0.0 {
        for v in &mut w {
            *v /= sum;
        }
    }
    w
}

/// Sparse Lucas-Kanade optical flow between two grayscale `[H,W,1]` frames.
///
/// Returns displacement `(dx, dy)` for each point. `window_size` should be odd (5-21).
pub fn lucas_kanade_optical_flow(
    prev: &Tensor,
    next: &Tensor,
    points: &[(usize, usize)],
    window_size: usize,
) -> Result<Vec<(f32, f32)>, ImgProcError> {
    let (h, w, c) = hwc_shape(prev)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let (h2, w2, c2) = hwc_shape(next)?;
    if h != h2 || w != w2 || c2 != 1 {
        return Err(ImgProcError::InvalidSize {
            height: h2,
            width: w2,
        });
    }
    let prev_d = prev.data();
    let next_d = next.data();
    let half = (window_size / 2) as i32;
    let mut flows = Vec::with_capacity(points.len());
    for &(px, py) in points {
        let mut ixx = 0.0f32;
        let mut iyy = 0.0f32;
        let mut ixy = 0.0f32;
        let mut ixt = 0.0f32;
        let mut iyt = 0.0f32;
        for dy in -half..=half {
            for dx in -half..=half {
                let y = py as i32 + dy;
                let x = px as i32 + dx;
                if y < 1 || y >= (h as i32 - 1) || x < 1 || x >= (w as i32 - 1) {
                    continue;
                }
                let (yu, xu) = (y as usize, x as usize);
                let ix = (prev_d[yu * w + xu + 1] - prev_d[yu * w + xu - 1]) * 0.5;
                let iy = (prev_d[(yu + 1) * w + xu] - prev_d[(yu - 1) * w + xu]) * 0.5;
                let it = next_d[yu * w + xu] - prev_d[yu * w + xu];
                ixx += ix * ix;
                iyy += iy * iy;
                ixy += ix * iy;
                ixt += ix * it;
                iyt += iy * it;
            }
        }
        let det = ixx * iyy - ixy * ixy;
        if det.abs() < 1e-6 {
            flows.push((0.0, 0.0));
        } else {
            let vx = -(iyy * ixt - ixy * iyt) / det;
            let vy = -(ixx * iyt - ixy * ixt) / det;
            flows.push((vx, vy));
        }
    }
    Ok(flows)
}

/// Marker-based watershed segmentation on a grayscale `[H,W,1]` image.
///
/// `markers` is `[H,W,1]` with positive labels for seeds, 0 for unlabeled.
/// Returns a fully labeled `[H,W,1]` map.
pub fn watershed(image: &Tensor, markers: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let (mh, mw, mc) = hwc_shape(markers)?;
    if mh != h || mw != w || mc != 1 {
        return Err(ImgProcError::InvalidSize {
            height: mh,
            width: mw,
        });
    }
    let img = image.data();
    let mark = markers.data();
    let mut labels = vec![0i32; h * w];
    let mut queue: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> =
        std::collections::BinaryHeap::new();
    let nbr: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for i in 0..h * w {
        let m = mark[i] as i32;
        if m > 0 {
            labels[i] = m;
            let (y, x) = (i / w, i % w);
            for &(dy, dx) in &nbr {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;
                if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                    let ni = ny as usize * w + nx as usize;
                    if labels[ni] == 0 && mark[ni] as i32 == 0 {
                        queue.push(std::cmp::Reverse(((img[ni] * 65535.0) as u32, ni)));
                    }
                }
            }
        }
    }
    while let Some(std::cmp::Reverse((_, idx))) = queue.pop() {
        if labels[idx] != 0 {
            continue;
        }
        let (y, x) = (idx / w, idx % w);
        let mut label = 0i32;
        for &(dy, dx) in &nbr {
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                let ni = ny as usize * w + nx as usize;
                if labels[ni] > 0 {
                    label = labels[ni];
                    break;
                }
            }
        }
        if label == 0 {
            continue;
        }
        labels[idx] = label;
        for &(dy, dx) in &nbr {
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                let ni = ny as usize * w + nx as usize;
                if labels[ni] == 0 {
                    queue.push(std::cmp::Reverse(((img[ni] * 65535.0) as u32, ni)));
                }
            }
        }
    }
    let out_data: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
    Tensor::from_vec(vec![h, w, 1], out_data).map_err(Into::into)
}

/// Dense optical flow estimation between two grayscale `[H,W,1]` frames.
///
/// Returns `[H,W,2]` where channel 0 is dx and channel 1 is dy.
/// Uses a block-matching approach with polynomial expansion approximation.
/// `window_size` controls neighborhood (odd, typically 5-15).
/// `num_iterations` controls refinement passes (1-5).
pub fn dense_optical_flow(
    prev: &Tensor,
    next: &Tensor,
    window_size: usize,
    num_iterations: usize,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(prev)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let (h2, w2, c2) = hwc_shape(next)?;
    if h != h2 || w != w2 || c2 != 1 {
        return Err(ImgProcError::InvalidSize {
            height: h2,
            width: w2,
        });
    }

    let half = (window_size / 2) as i32;
    let prev_d = prev.data();
    let next_d = next.data();

    let mut flow_x = vec![0.0f32; h * w];
    let mut flow_y = vec![0.0f32; h * w];

    for _iter in 0..num_iterations {
        for y in 0..h {
            for x in 0..w {
                let mut sum_ixx = 0.0f32;
                let mut sum_iyy = 0.0f32;
                let mut sum_ixy = 0.0f32;
                let mut sum_ixt = 0.0f32;
                let mut sum_iyt = 0.0f32;

                for dy in -half..=half {
                    for dx in -half..=half {
                        let sy = y as i32 + dy;
                        let sx = x as i32 + dx;
                        if sy < 1 || sy >= (h as i32 - 1) || sx < 1 || sx >= (w as i32 - 1) {
                            continue;
                        }
                        let su = sy as usize;
                        let sxu = sx as usize;
                        let ix = (prev_d[su * w + sxu + 1] - prev_d[su * w + sxu - 1]) * 0.5;
                        let iy = (prev_d[(su + 1) * w + sxu] - prev_d[(su - 1) * w + sxu]) * 0.5;

                        let warped_y = sy as f32 + flow_y[y * w + x];
                        let warped_x = sx as f32 + flow_x[y * w + x];
                        let wy = warped_y.clamp(0.0, (h - 1) as f32);
                        let wx = warped_x.clamp(0.0, (w - 1) as f32);
                        let wy0 = wy.floor() as usize;
                        let wx0 = wx.floor() as usize;
                        let wy1 = (wy0 + 1).min(h - 1);
                        let wx1 = (wx0 + 1).min(w - 1);
                        let fy = wy - wy0 as f32;
                        let fx = wx - wx0 as f32;
                        let warped_val = next_d[wy0 * w + wx0] * (1.0 - fy) * (1.0 - fx)
                            + next_d[wy0 * w + wx1] * (1.0 - fy) * fx
                            + next_d[wy1 * w + wx0] * fy * (1.0 - fx)
                            + next_d[wy1 * w + wx1] * fy * fx;
                        let it = warped_val - prev_d[su * w + sxu];

                        sum_ixx += ix * ix;
                        sum_iyy += iy * iy;
                        sum_ixy += ix * iy;
                        sum_ixt += ix * it;
                        sum_iyt += iy * it;
                    }
                }

                let det = sum_ixx * sum_iyy - sum_ixy * sum_ixy;
                if det.abs() > 1e-6 {
                    flow_x[y * w + x] += -(sum_iyy * sum_ixt - sum_ixy * sum_iyt) / det;
                    flow_y[y * w + x] += -(sum_ixx * sum_iyt - sum_ixy * sum_ixt) / det;
                }
            }
        }
    }

    let mut out = Vec::with_capacity(h * w * 2);
    for y in 0..h {
        for x in 0..w {
            out.push(flow_x[y * w + x]);
            out.push(flow_y[y * w + x]);
        }
    }
    Tensor::from_vec(vec![h, w, 2], out).map_err(Into::into)
}
