use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;
use super::fast::Keypoint;

/// BRIEF descriptor: 256-bit binary descriptor stored as 32 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BriefDescriptor {
    pub bits: [u8; 32],
}

/// Fixed set of 256 point-pair sampling offsets within a 31x31 patch.
/// Each entry is (ax, ay, bx, by) relative to the keypoint centre.
/// Generated from a deterministic xorshift PRNG.
fn brief_pattern() -> Vec<(i32, i32, i32, i32)> {
    let mut pattern = Vec::with_capacity(256);
    let mut seed: u32 = 0xDEAD_BEEF;
    for _ in 0..256 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let ax = ((seed % 31) as i32) - 15;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let ay = ((seed % 31) as i32) - 15;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let bx = ((seed % 31) as i32) - 15;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let by = ((seed % 31) as i32) - 15;
        pattern.push((ax, ay, bx, by));
    }
    pattern
}

/// Compute the rotated BRIEF pattern for a given angle (radians).
/// Returns the pattern with each point rotated around the origin.
fn rotated_brief_pattern(angle: f32) -> Vec<(i32, i32, i32, i32)> {
    let base = brief_pattern();
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    base.iter()
        .map(|&(ax, ay, bx, by)| {
            let rax = (ax as f32 * cos_a - ay as f32 * sin_a).round() as i32;
            let ray = (ax as f32 * sin_a + ay as f32 * cos_a).round() as i32;
            let rbx = (bx as f32 * cos_a - by as f32 * sin_a).round() as i32;
            let rby = (bx as f32 * sin_a + by as f32 * cos_a).round() as i32;
            (rax, ray, rbx, rby)
        })
        .collect()
}

/// Compute BRIEF descriptors for keypoints on a grayscale `[H, W, 1]` image.
///
/// Uses a fixed set of 256 point-pair comparisons in a 31x31 patch.
/// Keypoints too close to the image border are skipped (not included in output).
pub fn compute_brief(
    image: &Tensor,
    keypoints: &[Keypoint],
) -> Result<Vec<BriefDescriptor>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = image.data();
    let pattern = brief_pattern();
    let patch_radius = 15usize; // 31x31 patch

    let mut descriptors = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let kx = kp.x.round() as i32;
        let ky = kp.y.round() as i32;

        // Skip if too close to border
        if kx < patch_radius as i32
            || ky < patch_radius as i32
            || kx + patch_radius as i32 >= w as i32
            || ky + patch_radius as i32 >= h as i32
        {
            continue;
        }

        let mut bits = [0u8; 32];
        for (i, &(ax, ay, bx, by)) in pattern.iter().enumerate() {
            let pa = data[(ky + ay) as usize * w + (kx + ax) as usize];
            let pb = data[(ky + by) as usize * w + (kx + bx) as usize];
            if pa < pb {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        descriptors.push(BriefDescriptor { bits });
    }

    Ok(descriptors)
}

/// Compute rotated BRIEF descriptors for oriented keypoints.
///
/// The sampling pattern is rotated by each keypoint's angle.
pub(crate) fn compute_rotated_brief(
    data: &[f32],
    w: usize,
    h: usize,
    keypoints: &[Keypoint],
) -> Vec<Option<BriefDescriptor>> {
    keypoints
        .iter()
        .map(|kp| {
            let kx = kp.x.round() as i32;
            let ky = kp.y.round() as i32;

            // Need extra margin for rotated offsets (worst case ~21 pixels)
            let margin = 21i32;
            if kx < margin || ky < margin || kx + margin >= w as i32 || ky + margin >= h as i32 {
                return None;
            }

            let pattern = rotated_brief_pattern(kp.angle);
            let mut bits = [0u8; 32];
            let mut valid = true;
            for (i, &(ax, ay, bx, by)) in pattern.iter().enumerate() {
                let pax = kx + ax;
                let pay = ky + ay;
                let pbx = kx + bx;
                let pby = ky + by;
                if pax < 0
                    || pay < 0
                    || pbx < 0
                    || pby < 0
                    || pax >= w as i32
                    || pay >= h as i32
                    || pbx >= w as i32
                    || pby >= h as i32
                {
                    valid = false;
                    break;
                }
                let pa = data[pay as usize * w + pax as usize];
                let pb = data[pby as usize * w + pbx as usize];
                if pa < pb {
                    bits[i / 8] |= 1 << (i % 8);
                }
            }
            if valid {
                Some(BriefDescriptor { bits })
            } else {
                None
            }
        })
        .collect()
}

/// Hamming distance between two BRIEF descriptors.
pub fn hamming_distance(a: &BriefDescriptor, b: &BriefDescriptor) -> u32 {
    a.bits
        .iter()
        .zip(b.bits.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brief_descriptor_length() {
        // Create a textured image so we get a descriptor
        let data: Vec<f32> = (0..40 * 40)
            .map(|i| ((i as f32) * 0.1).sin().abs())
            .collect();
        let img = Tensor::from_vec(vec![40, 40, 1], data).unwrap();
        let kps = vec![Keypoint {
            x: 20.0,
            y: 20.0,
            response: 1.0,
            angle: 0.0,
            octave: 0,
        }];
        let descs = compute_brief(&img, &kps).unwrap();
        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].bits.len(), 32);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let d = BriefDescriptor { bits: [0xAB; 32] };
        assert_eq!(hamming_distance(&d, &d), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let a = BriefDescriptor { bits: [0x00; 32] };
        let b = BriefDescriptor { bits: [0xFF; 32] };
        assert_eq!(hamming_distance(&a, &b), 256);
    }
}
