use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;
use super::brief::{BriefDescriptor, compute_rotated_brief, hamming_distance};
use super::fast::{Keypoint, fast9_detect_raw, intensity_centroid_angle};

/// ORB feature: oriented FAST keypoint paired with a rotated BRIEF descriptor.
#[derive(Debug, Clone)]
pub struct OrbFeature {
    pub keypoint: Keypoint,
    pub descriptor: BriefDescriptor,
}

/// Configuration for the ORB feature detector.
#[derive(Debug, Clone)]
pub struct OrbConfig {
    /// Maximum number of features to retain (default 500).
    pub num_features: usize,
    /// Scale factor between pyramid levels (default 1.2).
    pub scale_factor: f32,
    /// Number of pyramid levels (default 8).
    pub num_levels: usize,
    /// FAST threshold (default 20.0/255.0).
    pub fast_threshold: f32,
}

impl Default for OrbConfig {
    fn default() -> Self {
        Self {
            num_features: 500,
            scale_factor: 1.2,
            num_levels: 8,
            fast_threshold: 20.0 / 255.0,
        }
    }
}

/// Detect ORB features: oriented FAST keypoints + rotated BRIEF descriptors.
///
/// Input must be a single-channel `[H, W, 1]` image.
///
/// 1. Build an image pyramid.
/// 2. At each level, run FAST-9 detection.
/// 3. Compute orientation via the intensity centroid method.
/// 4. Compute rotated BRIEF descriptors.
/// 5. Keep top `num_features` by response across all levels.
/// 6. Scale coordinates back to the original image.
pub fn detect_orb(image: &Tensor, config: &OrbConfig) -> Result<Vec<OrbFeature>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }

    let mut all_features: Vec<OrbFeature> = Vec::new();

    // Build pyramid levels — hierarchical: level N from level N-1.
    // Nearest-neighbor from previous level (fast, each level reads smaller image).
    let mut pyramid: Vec<(Vec<f32>, usize, usize, f32)> = Vec::new(); // (data, h, w, scale)
    {
        let base_data = image.data().to_vec();
        pyramid.push((base_data, h, w, 1.0));
        let sf = config.scale_factor;
        for level in 1..config.num_levels {
            let (ref prev_data, prev_h, prev_w, prev_scale) = pyramid[level - 1];
            let nh = (prev_h as f32 / sf).round() as usize;
            let nw = (prev_w as f32 / sf).round() as usize;
            if nh < 10 || nw < 10 {
                break;
            }
            let mut dst = vec![0.0f32; nh * nw];
            let y_ratio = prev_h as f32 / nh as f32;
            let x_ratio = prev_w as f32 / nw as f32;
            for dy in 0..nh {
                let sy = ((dy as f32 + 0.5) * y_ratio) as usize;
                let sy = sy.min(prev_h - 1);
                let src_row = sy * prev_w;
                let dst_row = dy * nw;
                for dx in 0..nw {
                    let sx = ((dx as f32 + 0.5) * x_ratio) as usize;
                    dst[dst_row + dx] = prev_data[src_row + sx.min(prev_w - 1)];
                }
            }
            pyramid.push((dst, nh, nw, prev_scale * sf));
        }
    }

    // Process all pyramid levels in parallel (FAST + orientation + BRIEF per level).
    {
        use rayon::prelude::*;
        let fast_threshold = config.fast_threshold;
        let level_features: Vec<Vec<OrbFeature>> = pyramid
            .par_iter()
            .enumerate()
            .map(|(level, (pdata, ph, pw, scale))| {
                let ph = *ph;
                let pw = *pw;
                let scale = *scale;
                let mut keypoints = fast9_detect_raw(pdata, ph, pw, fast_threshold, true);

                let centroid_radius = 15i32.min((ph.min(pw) / 2) as i32 - 1);
                if centroid_radius > 0 {
                    for kp in &mut keypoints {
                        let kx = kp.x.round() as usize;
                        let ky = kp.y.round() as usize;
                        kp.angle = intensity_centroid_angle(pdata, pw, ph, kx, ky, centroid_radius);
                        kp.octave = level;
                    }
                }

                let descs = compute_rotated_brief(pdata, pw, ph, &keypoints);
                let mut features = Vec::new();
                for (kp, desc_opt) in keypoints.into_iter().zip(descs) {
                    if let Some(descriptor) = desc_opt {
                        let mut scaled_kp = kp;
                        scaled_kp.x *= scale;
                        scaled_kp.y *= scale;
                        features.push(OrbFeature {
                            keypoint: scaled_kp,
                            descriptor,
                        });
                    }
                }
                features
            })
            .collect();

        for features in level_features {
            all_features.extend(features);
        }
    }

    // Sort by response (descending) and keep top N
    all_features.sort_by(|a, b| {
        b.keypoint
            .response
            .partial_cmp(&a.keypoint.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_features.truncate(config.num_features);

    Ok(all_features)
}

/// Match ORB features between two sets using brute-force Hamming distance.
///
/// Returns pairs `(idx_a, idx_b)` for the best match of each feature in `a`
/// against `b`, provided the distance is at most `max_distance`.
pub fn match_features(
    features_a: &[OrbFeature],
    features_b: &[OrbFeature],
    max_distance: u32,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    for (i, fa) in features_a.iter().enumerate() {
        let mut best_j = 0usize;
        let mut best_dist = u32::MAX;
        for (j, fb) in features_b.iter().enumerate() {
            let d = hamming_distance(&fa.descriptor, &fb.descriptor);
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        if best_dist <= max_distance {
            matches.push((i, best_j));
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orb_detect_returns_features() {
        // 128x128 with multiple L-shaped corners well inside the BRIEF margin (21px)
        let size = 128;
        let mut data = vec![0.0f32; size * size];
        // Place several L-shaped bright features centered around the image
        // Each L: horizontal bar + vertical bar meeting at a corner
        let centers = [(50, 50), (50, 80), (80, 50), (80, 80)];
        for &(cy, cx) in &centers {
            // Horizontal bright bar
            for x in cx..cx + 12 {
                if x < size {
                    data[cy * size + x] = 1.0;
                }
            }
            // Vertical bright bar
            for y in cy..cy + 12 {
                if y < size {
                    data[y * size + cx] = 1.0;
                }
            }
        }
        let img = Tensor::from_vec(vec![size, size, 1], data).unwrap();
        let config = OrbConfig {
            num_features: 100,
            num_levels: 1,
            fast_threshold: 0.05,
            ..OrbConfig::default()
        };
        let features = detect_orb(&img, &config).unwrap();
        assert!(
            !features.is_empty(),
            "image with L-shaped corners should produce ORB features"
        );
        // Each feature should have a valid descriptor
        for f in &features {
            assert_eq!(f.descriptor.bits.len(), 32);
        }
    }

    #[test]
    fn test_orb_config_defaults() {
        let cfg = OrbConfig::default();
        assert_eq!(cfg.num_features, 500);
        assert!((cfg.scale_factor - 1.2).abs() < 1e-6);
        assert_eq!(cfg.num_levels, 8);
        assert!((cfg.fast_threshold - 20.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_match_features_self() {
        // Create textured image and detect features, then match against self
        let size = 64;
        let data: Vec<f32> = (0..size * size)
            .map(|i| {
                let x = i % size;
                let y = i / size;
                if (x / 4 + y / 4) % 2 == 0 { 0.9 } else { 0.1 }
            })
            .collect();
        let img = Tensor::from_vec(vec![size, size, 1], data).unwrap();
        let config = OrbConfig {
            num_features: 50,
            num_levels: 1,
            ..OrbConfig::default()
        };
        let features = detect_orb(&img, &config).unwrap();
        if features.is_empty() {
            // Can't test matching without features; skip gracefully
            return;
        }
        let matches = match_features(&features, &features, 0);
        // Every feature should match itself at distance 0
        assert_eq!(matches.len(), features.len());
        for &(a, b) in &matches {
            assert_eq!(a, b, "self-matching should produce identity pairs");
        }
    }
}
