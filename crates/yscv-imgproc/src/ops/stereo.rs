use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Configuration for stereo block matching.
#[derive(Debug, Clone)]
pub struct StereoConfig {
    /// Number of disparity levels to search (must be > 0).
    pub num_disparities: usize,
    /// Side length of the matching block (must be odd and > 0).
    pub block_size: usize,
    /// Minimum disparity offset (can be 0).
    pub min_disparity: usize,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            num_disparities: 64,
            block_size: 9,
            min_disparity: 0,
        }
    }
}

/// Compute a disparity map from a rectified stereo pair using block matching (SAD).
///
/// Both `left` and `right` must be single-channel images with shape `[H, W, 1]`.
/// Returns an `[H, W, 1]` disparity map of type `f32`.  Pixels where no valid
/// disparity could be found are set to `0.0`.
pub fn stereo_block_matching(
    left: &Tensor,
    right: &Tensor,
    config: &StereoConfig,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(left)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let (rh, rw, rc) = hwc_shape(right)?;
    if rh != h || rw != w || rc != 1 {
        return Err(ImgProcError::ShapeMismatch {
            expected: vec![h, w, 1],
            got: vec![rh, rw, rc],
        });
    }
    if config.block_size == 0 || config.block_size.is_multiple_of(2) {
        return Err(ImgProcError::InvalidBlockSize {
            block_size: config.block_size,
        });
    }
    if config.num_disparities == 0 {
        return Err(ImgProcError::InvalidSize {
            height: 0,
            width: config.num_disparities,
        });
    }

    let half = (config.block_size / 2) as isize;
    let left_data = left.data();
    let right_data = right.data();
    let mut out = vec![0.0f32; h * w];

    for y in 0..h {
        for x in 0..w {
            let mut best_sad = f32::MAX;
            let mut best_d: usize = 0;

            for d in config.min_disparity..config.min_disparity + config.num_disparities {
                // If the block in the right image would go out of bounds, skip.
                if (x as isize - d as isize) < half {
                    continue;
                }

                let mut sad = 0.0f32;
                for ky in -half..=half {
                    let sy = y as isize + ky;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in -half..=half {
                        let lx = x as isize + kx;
                        let rx = lx - d as isize;
                        if lx < 0 || lx >= w as isize || rx < 0 || rx >= w as isize {
                            continue;
                        }
                        let li = sy as usize * w + lx as usize;
                        let ri = sy as usize * w + rx as usize;
                        sad += (left_data[li] - right_data[ri]).abs();
                    }
                }

                if sad < best_sad {
                    best_sad = sad;
                    best_d = d;
                }
            }

            out[y * w + x] = best_d as f32;
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gray(h: usize, w: usize, data: Vec<f32>) -> Tensor {
        Tensor::from_vec(vec![h, w, 1], data).unwrap()
    }

    #[test]
    fn test_zero_disparity_identical_images() {
        let data = vec![1.0; 8 * 8];
        let img = make_gray(8, 8, data);
        let config = StereoConfig {
            num_disparities: 4,
            block_size: 3,
            min_disparity: 0,
        };
        let disp = stereo_block_matching(&img, &img, &config).unwrap();
        assert_eq!(disp.shape(), &[8, 8, 1]);
        // Identical images → disparity should be 0 everywhere.
        for &v in disp.data() {
            assert!((v - 0.0).abs() < f32::EPSILON, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_known_shift() {
        // Create a 1-row image with a clear feature shifted by 2 pixels.
        let h = 1;
        let w = 16;
        let left_data: Vec<f32> = (0..w).map(|i| if i == 8 { 100.0 } else { 0.0 }).collect();
        let right_data: Vec<f32> = (0..w).map(|i| if i == 6 { 100.0 } else { 0.0 }).collect();
        let left = make_gray(h, w, left_data);
        let right = make_gray(h, w, right_data);
        let config = StereoConfig {
            num_disparities: 8,
            block_size: 1,
            min_disparity: 0,
        };
        let disp = stereo_block_matching(&left, &right, &config).unwrap();
        // At x=8 the best match in right is at x=6 → disparity = 2.
        assert!((disp.data()[8] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = make_gray(4, 4, vec![0.0; 16]);
        let b = make_gray(4, 5, vec![0.0; 20]);
        let config = StereoConfig::default();
        assert!(stereo_block_matching(&a, &b, &config).is_err());
    }

    #[test]
    fn test_invalid_block_size() {
        let img = make_gray(4, 4, vec![0.0; 16]);
        let config = StereoConfig {
            num_disparities: 4,
            block_size: 4, // even – invalid
            min_disparity: 0,
        };
        assert!(stereo_block_matching(&img, &img, &config).is_err());
    }
}
