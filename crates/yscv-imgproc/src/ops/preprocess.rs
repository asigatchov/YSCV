use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;
use super::resize::resize_bilinear;

/// Convert an image from `[H, W, C]` (HWC) layout to `[C, H, W]` (CHW) layout.
pub fn hwc_to_chw(image: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    let src = image.data();
    let mut out = vec![0.0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            let hwc_base = (y * w + x) * c;
            for ch in 0..c {
                out[ch * h * w + y * w + x] = src[hwc_base + ch];
            }
        }
    }
    Tensor::from_vec(vec![c, h, w], out).map_err(Into::into)
}

/// Convert an image from `[C, H, W]` (CHW) layout to `[H, W, C]` (HWC) layout.
pub fn chw_to_hwc(image: &Tensor) -> Result<Tensor, ImgProcError> {
    let shape = image.shape();
    if shape.len() != 3 {
        return Err(ImgProcError::InvalidImageShape {
            expected_rank: 3,
            got: shape.to_vec(),
        });
    }
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let src = image.data();
    let mut out = vec![0.0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let hwc_base = (y * w + x) * c;
            for ch in 0..c {
                out[hwc_base + ch] = src[ch * h * w + y * w + x];
            }
        }
    }
    Tensor::from_vec(vec![h, w, c], out).map_err(Into::into)
}

/// Per-channel normalization: `(pixel - mean[c]) / std[c]` for an HWC image.
pub fn normalize_image(image: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor, ImgProcError> {
    // Delegate to the existing normalize function.
    super::normalize::normalize(image, mean, std)
}

/// Center-crop an `[H, W, C]` image to `size × size`.
///
/// If the image is smaller than `size` in either dimension the original is returned.
pub fn center_crop(image: &Tensor, size: usize) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if h <= size && w <= size {
        return Ok(image.clone());
    }
    let crop_h = size.min(h);
    let crop_w = size.min(w);
    let y_off = (h - crop_h) / 2;
    let x_off = (w - crop_w) / 2;

    let src = image.data();
    let mut out = vec![0.0f32; crop_h * crop_w * c];
    for y in 0..crop_h {
        let src_row = ((y_off + y) * w + x_off) * c;
        let dst_row = y * crop_w * c;
        out[dst_row..dst_row + crop_w * c].copy_from_slice(&src[src_row..src_row + crop_w * c]);
    }
    Tensor::from_vec(vec![crop_h, crop_w, c], out).map_err(Into::into)
}

/// Standard ImageNet preprocessing pipeline.
///
/// 1. Resize shortest side to 256 (bilinear)
/// 2. Center crop to 224×224
/// 3. Normalize with ImageNet mean/std
/// 4. Convert HWC → CHW
///
/// Input: `[H, W, 3]` float32 in `[0, 1]`.
/// Output: `[3, 224, 224]` float32 normalized.
pub fn imagenet_preprocess(image: &Tensor) -> Result<Tensor, ImgProcError> {
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    let (h, w, _c) = hwc_shape(image)?;

    // Step 1: Resize shortest side to 256.
    let (new_h, new_w) = if h < w {
        (256, (256 * w + h / 2) / h)
    } else {
        ((256 * h + w / 2) / w, 256)
    };
    let resized = resize_bilinear(image, new_h, new_w)?;

    // Step 2: Center crop to 224×224.
    let cropped = center_crop(&resized, 224)?;

    // Step 3: Normalize.
    let normalized = normalize_image(&cropped, &IMAGENET_MEAN, &IMAGENET_STD)?;

    // Step 4: HWC → CHW.
    hwc_to_chw(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hwc_to_chw_roundtrip() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let hwc = Tensor::from_vec(vec![2, 4, 3], data.clone()).unwrap();

        let chw = hwc_to_chw(&hwc).unwrap();
        assert_eq!(chw.shape(), &[3, 2, 4]);

        let back = chw_to_hwc(&chw).unwrap();
        assert_eq!(back.shape(), &[2, 4, 3]);
        assert_eq!(back.data(), &data[..]);
    }

    #[test]
    fn test_center_crop_exact() {
        let img = Tensor::from_vec(vec![10, 10, 3], vec![0.5f32; 300]).unwrap();
        let cropped = center_crop(&img, 6).unwrap();
        assert_eq!(cropped.shape(), &[6, 6, 3]);
    }

    #[test]
    fn test_center_crop_smaller_than_size() {
        let img = Tensor::from_vec(vec![4, 4, 3], vec![0.5f32; 48]).unwrap();
        let cropped = center_crop(&img, 10).unwrap();
        assert_eq!(cropped.shape(), &[4, 4, 3]);
    }

    #[test]
    fn test_normalize_image() {
        // 1×1×3 image with value 0.5 for all channels.
        let img = Tensor::from_vec(vec![1, 1, 3], vec![0.485, 0.456, 0.406]).unwrap();
        let norm = normalize_image(&img, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]).unwrap();
        // After normalization all channels should be ~0.
        for &v in norm.data() {
            assert!(v.abs() < 1e-5, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_imagenet_preprocess_shape() {
        // 300×400×3 image.
        let img = Tensor::from_vec(vec![300, 400, 3], vec![0.5f32; 300 * 400 * 3]).unwrap();
        let result = imagenet_preprocess(&img).unwrap();
        assert_eq!(result.shape(), &[3, 224, 224]);
    }
}
