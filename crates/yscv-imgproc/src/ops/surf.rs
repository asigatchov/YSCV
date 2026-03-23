use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// SURF keypoint with position, scale, orientation, and response.
#[derive(Debug, Clone)]
pub struct SurfKeypoint {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub orientation: f32,
    pub response: f32,
    pub laplacian_sign: i8, // +1 or -1
}

/// SURF descriptor (64-element vector).
pub type SurfDescriptor = Vec<f32>;

/// Build integral image for fast box filter computation.
pub fn build_integral_image(image: &[f32], width: usize, height: usize) -> Vec<f64> {
    let mut integral = vec![0.0f64; width * height];
    for y in 0..height {
        let mut row_sum = 0.0f64;
        for x in 0..width {
            row_sum += image[y * width + x] as f64;
            integral[y * width + x] = row_sum
                + if y > 0 {
                    integral[(y - 1) * width + x]
                } else {
                    0.0
                };
        }
    }
    integral
}

/// Query box sum from integral image.
/// Computes the sum of pixels in the rectangle [x1, y1] to [x2, y2] (inclusive).
fn box_sum(integral: &[f64], width: usize, x1: usize, y1: usize, x2: usize, y2: usize) -> f64 {
    let a = if x1 > 0 && y1 > 0 {
        integral[(y1 - 1) * width + x1 - 1]
    } else {
        0.0
    };
    let b = if y1 > 0 {
        integral[(y1 - 1) * width + x2]
    } else {
        0.0
    };
    let c = if x1 > 0 {
        integral[y2 * width + x1 - 1]
    } else {
        0.0
    };
    let d = integral[y2 * width + x2];
    d - b - c + a
}

/// Safe box sum that clamps coordinates to image bounds.
fn box_sum_safe(
    integral: &[f64],
    width: usize,
    height: usize,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
) -> f64 {
    let x1 = x1.max(0) as usize;
    let y1 = y1.max(0) as usize;
    let x2 = (x2.min(width as i32 - 1)).max(0) as usize;
    let y2 = (y2.min(height as i32 - 1)).max(0) as usize;
    if x2 < x1 || y2 < y1 {
        return 0.0;
    }
    box_sum(integral, width, x1, y1, x2, y2)
}

/// Compute the approximate Hessian determinant at a given point and filter size.
/// Uses box-filter approximation of second-order Gaussian derivatives.
fn hessian_det(
    integral: &[f64],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    filter_size: usize,
) -> (f64, f64) {
    let fs = filter_size as i32;
    let l = fs / 3; // lobe size

    // Dxx: three horizontal lobes
    let dxx = box_sum_safe(integral, width, height, x - l, y - l / 2, x + l, y + l / 2)
        - 3.0
            * box_sum_safe(
                integral,
                width,
                height,
                x - l / 2,
                y - l / 2,
                x + l / 2,
                y + l / 2,
            );

    // Dyy: three vertical lobes
    let dyy = box_sum_safe(integral, width, height, x - l / 2, y - l, x + l / 2, y + l)
        - 3.0
            * box_sum_safe(
                integral,
                width,
                height,
                x - l / 2,
                y - l / 2,
                x + l / 2,
                y + l / 2,
            );

    // Dxy: four quadrant lobes
    let dxy = box_sum_safe(integral, width, height, x + 1, y - l, x + l, y - 1)
        + box_sum_safe(integral, width, height, x - l, y + 1, x - 1, y + l)
        - box_sum_safe(integral, width, height, x - l, y - l, x - 1, y - 1)
        - box_sum_safe(integral, width, height, x + 1, y + 1, x + l, y + l);

    // Normalize by filter area
    let area = (fs * fs) as f64;
    let dxx = dxx / area;
    let dyy = dyy / area;
    let dxy = dxy / area;

    // Hessian determinant approximation (weight 0.9 for Dxy per SURF paper)
    let det = dxx * dyy - 0.81 * dxy * dxy;
    let trace = dxx + dyy;
    (det, trace)
}

/// Detect SURF keypoints using box-filter approximation of Hessian.
pub fn detect_surf_keypoints(
    image: &Tensor,
    hessian_threshold: f32,
    num_octaves: usize,
    num_scales: usize,
) -> Result<Vec<SurfKeypoint>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = image.data();

    // 1. Build integral image
    let integral = build_integral_image(data, w, h);

    // 2. Compute Hessian response at multiple scales
    // SURF uses filter sizes: 9, 15, 21, 27 for octave 1; 15, 27, 39, 51 for octave 2; etc.
    let mut scale_responses: Vec<(Vec<f64>, usize)> = Vec::new(); // (response_map, filter_size)

    for octave in 0..num_octaves {
        let step = 1usize << octave; // sampling step doubles per octave
        for scale in 0..num_scales {
            let filter_size = 3 * ((2usize.pow(octave as u32)) * (scale + 1) + 1);
            if filter_size / 2 >= h.min(w) {
                continue;
            }
            let mut response = vec![0.0f64; h * w];
            let margin = (filter_size / 2 + 1) as i32;

            for y in (margin as usize..h.saturating_sub(margin as usize)).step_by(step) {
                for x in (margin as usize..w.saturating_sub(margin as usize)).step_by(step) {
                    let (det, _trace) =
                        hessian_det(&integral, w, h, x as i32, y as i32, filter_size);
                    response[y * w + x] = det;
                }
            }
            scale_responses.push((response, filter_size));
        }
    }

    // 3. Non-maximum suppression in 3x3x3 neighborhood (x, y, scale)
    let mut keypoints = Vec::new();
    let thresh = hessian_threshold as f64;

    for si in 1..scale_responses.len().saturating_sub(1) {
        let filter_size = scale_responses[si].1;
        let margin = filter_size / 2 + 1;
        let _step = 1usize.max(filter_size / 9);

        for y in margin..h.saturating_sub(margin) {
            for x in margin..w.saturating_sub(margin) {
                let val = scale_responses[si].0[y * w + x];
                if val < thresh {
                    continue;
                }

                let mut is_max = true;
                'nms: for ds in -1i32..=1 {
                    let si2 = (si as i32 + ds) as usize;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if ds == 0 && dy == 0 && dx == 0 {
                                continue;
                            }
                            let ny = (y as i32 + dy) as usize;
                            let nx = (x as i32 + dx) as usize;
                            if ny < h && nx < w && scale_responses[si2].0[ny * w + nx] >= val {
                                is_max = false;
                                break 'nms;
                            }
                        }
                    }
                }

                if is_max {
                    // Compute orientation using Haar wavelets in circular neighborhood
                    let scale = filter_size as f32 * 1.2 / 9.0;
                    let orientation =
                        compute_orientation(&integral, w, h, x as f32, y as f32, scale);

                    let (_, trace) = hessian_det(&integral, w, h, x as i32, y as i32, filter_size);

                    keypoints.push(SurfKeypoint {
                        x: x as f32,
                        y: y as f32,
                        scale,
                        orientation,
                        response: val as f32,
                        laplacian_sign: if trace > 0.0 { 1 } else { -1 },
                    });
                }
            }
        }
    }

    // Sort by response (descending)
    keypoints.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(keypoints)
}

/// Compute dominant orientation using Haar wavelet responses in a circular region.
fn compute_orientation(
    integral: &[f64],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    scale: f32,
) -> f32 {
    let radius = (6.0 * scale).round() as i32;
    let haar_size = (4.0 * scale).round().max(1.0) as i32;
    let half_haar = haar_size / 2;

    let mut dx_responses = Vec::new();
    let mut dy_responses = Vec::new();
    let mut angles = Vec::new();

    // Sample Haar wavelet responses in circular neighborhood
    for i in -radius..=radius {
        for j in -radius..=radius {
            if i * i + j * j > radius * radius {
                continue;
            }
            let px = x as i32 + j;
            let py = y as i32 + i;

            // Haar wavelet response in x direction
            let dx = box_sum_safe(
                integral,
                width,
                height,
                px,
                py - half_haar,
                px + half_haar,
                py + half_haar,
            ) - box_sum_safe(
                integral,
                width,
                height,
                px - half_haar,
                py - half_haar,
                px,
                py + half_haar,
            );

            // Haar wavelet response in y direction
            let dy = box_sum_safe(
                integral,
                width,
                height,
                px - half_haar,
                py,
                px + half_haar,
                py + half_haar,
            ) - box_sum_safe(
                integral,
                width,
                height,
                px - half_haar,
                py - half_haar,
                px + half_haar,
                py,
            );

            // Gaussian weight
            let sigma = 2.5 * scale;
            let weight = (-(i * i + j * j) as f32 / (2.0 * sigma * sigma)).exp();

            dx_responses.push(dx as f32 * weight);
            dy_responses.push(dy as f32 * weight);
            angles.push((dy as f32).atan2(dx as f32));
        }
    }

    if dx_responses.is_empty() {
        return 0.0;
    }

    // Sliding window of pi/3 to find dominant orientation
    let window = std::f32::consts::PI / 3.0;
    let mut best_angle = 0.0f32;
    let mut best_magnitude = 0.0f32;

    let steps = 36;
    for step in 0..steps {
        let angle = -std::f32::consts::PI + step as f32 * 2.0 * std::f32::consts::PI / steps as f32;
        let mut sum_dx = 0.0f32;
        let mut sum_dy = 0.0f32;

        for i in 0..angles.len() {
            let mut diff = angles[i] - angle;
            // Normalize to [-pi, pi]
            while diff > std::f32::consts::PI {
                diff -= 2.0 * std::f32::consts::PI;
            }
            while diff < -std::f32::consts::PI {
                diff += 2.0 * std::f32::consts::PI;
            }
            if diff.abs() < window / 2.0 {
                sum_dx += dx_responses[i];
                sum_dy += dy_responses[i];
            }
        }

        let mag = sum_dx * sum_dx + sum_dy * sum_dy;
        if mag > best_magnitude {
            best_magnitude = mag;
            best_angle = sum_dy.atan2(sum_dx);
        }
    }

    best_angle
}

/// Compute SURF descriptors for detected keypoints.
pub fn compute_surf_descriptors(
    image: &Tensor,
    keypoints: &[SurfKeypoint],
) -> Result<Vec<SurfDescriptor>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = image.data();
    let integral = build_integral_image(data, w, h);

    let mut descriptors = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let scale = kp.scale;
        let cos_ori = kp.orientation.cos();
        let sin_ori = kp.orientation.sin();
        let haar_size = (2.0 * scale).round().max(1.0) as i32;
        let half_haar = haar_size / 2;

        let mut desc = vec![0.0f32; 64];

        // 4x4 sub-regions, each covering a 5s x 5s area
        let _sub_region_size = 5.0 * scale;

        for i in 0..4 {
            for j in 0..4 {
                let mut sum_dx = 0.0f32;
                let mut sum_abs_dx = 0.0f32;
                let mut sum_dy = 0.0f32;
                let mut sum_abs_dy = 0.0f32;

                // 5x5 sample points per sub-region
                for k in 0..5 {
                    for l in 0..5 {
                        // Position relative to keypoint (in rotated coordinates)
                        let sample_x = ((i as f32 - 2.0) * 5.0 + l as f32 + 0.5) * scale;
                        let sample_y = ((j as f32 - 2.0) * 5.0 + k as f32 + 0.5) * scale;

                        // Rotate to image coordinates
                        let rx = (cos_ori * sample_x - sin_ori * sample_y + kp.x).round() as i32;
                        let ry = (sin_ori * sample_x + cos_ori * sample_y + kp.y).round() as i32;

                        // Haar wavelet responses
                        let dx = box_sum_safe(
                            &integral,
                            w,
                            h,
                            rx,
                            ry - half_haar,
                            rx + half_haar,
                            ry + half_haar,
                        ) - box_sum_safe(
                            &integral,
                            w,
                            h,
                            rx - half_haar,
                            ry - half_haar,
                            rx,
                            ry + half_haar,
                        );

                        let dy = box_sum_safe(
                            &integral,
                            w,
                            h,
                            rx - half_haar,
                            ry,
                            rx + half_haar,
                            ry + half_haar,
                        ) - box_sum_safe(
                            &integral,
                            w,
                            h,
                            rx - half_haar,
                            ry - half_haar,
                            rx + half_haar,
                            ry,
                        );

                        // Gaussian weight centered on sub-region
                        let cx = ((i as f32 - 1.5) * 5.0 + 2.5) * scale;
                        let cy = ((j as f32 - 1.5) * 5.0 + 2.5) * scale;
                        let dist_sq = (sample_x - cx).powi(2) + (sample_y - cy).powi(2);
                        let sigma = 3.3 * scale;
                        let gauss = (-dist_sq / (2.0 * sigma * sigma)).exp();

                        // Rotate wavelet responses to be relative to keypoint orientation
                        let rdx = cos_ori * dx as f32 + sin_ori * dy as f32;
                        let rdy = -sin_ori * dx as f32 + cos_ori * dy as f32;

                        sum_dx += rdx * gauss;
                        sum_abs_dx += rdx.abs() * gauss;
                        sum_dy += rdy * gauss;
                        sum_abs_dy += rdy.abs() * gauss;
                    }
                }

                let idx = (i * 4 + j) * 4;
                desc[idx] = sum_dx;
                desc[idx + 1] = sum_abs_dx;
                desc[idx + 2] = sum_dy;
                desc[idx + 3] = sum_abs_dy;
            }
        }

        // Normalize to unit vector
        let norm = desc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-7);
        for v in &mut desc {
            *v /= norm;
        }

        descriptors.push(desc);
    }

    Ok(descriptors)
}

/// Match SURF descriptors between two sets using nearest-neighbor ratio test.
///
/// Returns `(idx1, idx2, distance)` for accepted matches where the ratio of
/// best to second-best distance is below `ratio_threshold`.
pub fn match_surf_descriptors(
    desc1: &[SurfDescriptor],
    desc2: &[SurfDescriptor],
    ratio_threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let mut matches = Vec::new();

    for (i, d1) in desc1.iter().enumerate() {
        let mut best_dist = f32::MAX;
        let mut second_dist = f32::MAX;
        let mut best_idx = 0;

        for (j, d2) in desc2.iter().enumerate() {
            let dist: f32 = d1
                .iter()
                .zip(d2.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = j;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        // Accept exact matches (dist ≈ 0) unconditionally; otherwise apply ratio test.
        if best_dist < 1e-9 || (second_dist > 0.0 && best_dist / second_dist < ratio_threshold) {
            matches.push((i, best_idx, best_dist));
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surf_integral_image() {
        // 3x3 image, all ones
        let data = vec![1.0f32; 9];
        let integral = build_integral_image(&data, 3, 3);
        // Expected integral image:
        // 1  2  3
        // 2  4  6
        // 3  6  9
        assert_eq!(integral[0], 1.0); // (0,0)
        assert_eq!(integral[1], 2.0); // (1,0)
        assert_eq!(integral[2], 3.0); // (2,0)
        assert_eq!(integral[3], 2.0); // (0,1)
        assert_eq!(integral[4], 4.0); // (1,1)
        assert_eq!(integral[5], 6.0); // (2,1)
        assert_eq!(integral[6], 3.0); // (0,2)
        assert_eq!(integral[7], 6.0); // (1,2)
        assert_eq!(integral[8], 9.0); // (2,2)

        // Test with varying values
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let integral2 = build_integral_image(&data2, 3, 3);
        // Row 0: 1, 3, 6
        // Row 1: 5, 12, 21
        // Row 2: 12, 27, 45
        assert_eq!(integral2[0], 1.0);
        assert_eq!(integral2[1], 3.0);
        assert_eq!(integral2[2], 6.0);
        assert_eq!(integral2[3], 5.0);
        assert_eq!(integral2[4], 12.0);
        assert_eq!(integral2[5], 21.0);
        assert_eq!(integral2[6], 12.0);
        assert_eq!(integral2[7], 27.0);
        assert_eq!(integral2[8], 45.0);

        // Verify box_sum on a region
        let sum = box_sum(&integral2, 3, 1, 1, 2, 2);
        // Should be 5+6+8+9 = 28
        assert_eq!(sum, 28.0);
    }

    #[test]
    fn surf_detect_on_gradient() {
        // Create a 64x64 image with a strong corner (bright square on dark background)
        let (h, w) = (64, 64);
        let mut data = vec![0.0f32; h * w];
        for y in 20..44 {
            for x in 20..44 {
                data[y * w + x] = 1.0;
            }
        }
        let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();
        let keypoints = detect_surf_keypoints(&img, 0.0001, 2, 4).unwrap();
        assert!(
            !keypoints.is_empty(),
            "image with strong edges should produce SURF keypoints"
        );
        // Keypoints should be near the edges of the bright square
        for kp in &keypoints {
            assert!(kp.response > 0.0, "keypoint response should be positive");
        }
    }

    #[test]
    fn surf_descriptor_dimension() {
        // Create a simple image and manually-placed keypoint
        let (h, w) = (64, 64);
        let data: Vec<f32> = (0..h * w).map(|i| (i % w) as f32 / w as f32).collect();
        let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();

        let keypoints = vec![SurfKeypoint {
            x: 32.0,
            y: 32.0,
            scale: 1.2,
            orientation: 0.0,
            response: 1.0,
            laplacian_sign: 1,
        }];
        let descriptors = compute_surf_descriptors(&img, &keypoints).unwrap();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(
            descriptors[0].len(),
            64,
            "SURF descriptor should be 64-element"
        );
        // Should be normalized to unit vector
        let norm: f32 = descriptors[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "descriptor should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn surf_match_identical() {
        // Create image, detect keypoints, compute descriptors, match against itself
        let (h, w) = (64, 64);
        let mut data = vec![0.1f32; h * w];
        // Add multiple distinct features
        for y in 10..20 {
            for x in 10..20 {
                data[y * w + x] = 0.9;
            }
        }
        for y in 40..50 {
            for x in 40..50 {
                data[y * w + x] = 0.9;
            }
        }
        let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();

        // Create keypoints at known positions
        let keypoints = vec![
            SurfKeypoint {
                x: 15.0,
                y: 15.0,
                scale: 1.2,
                orientation: 0.0,
                response: 1.0,
                laplacian_sign: 1,
            },
            SurfKeypoint {
                x: 45.0,
                y: 45.0,
                scale: 1.2,
                orientation: 0.0,
                response: 1.0,
                laplacian_sign: 1,
            },
        ];

        let descriptors = compute_surf_descriptors(&img, &keypoints).unwrap();
        assert_eq!(descriptors.len(), 2);

        // Match against itself — every descriptor should find a near-zero distance match.
        // Note: if two descriptors are identical (symmetric patches), the matcher may
        // pair them in any order, so we only check distances, not index identity.
        let matches = match_surf_descriptors(&descriptors, &descriptors, 0.99);
        assert!(
            !matches.is_empty(),
            "matching descriptors against themselves should produce matches"
        );
        for &(_i, _j, dist) in &matches {
            assert!(
                dist < 1e-5,
                "self-match distance should be ~0, got {}",
                dist
            );
        }
    }
}
