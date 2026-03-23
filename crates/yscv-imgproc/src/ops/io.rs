use std::path::Path;

use image::{DynamicImage, ImageFormat, ImageReader};
use yscv_tensor::Tensor;

use super::super::ImgProcError;

/// Read an image from disk and return it as a `[H, W, C]` `f32` tensor with values in `[0, 1]`.
///
/// Supported formats: PNG, JPEG, BMP, WebP.
/// RGB images produce shape `[H, W, 3]`, grayscale images are converted to RGB.
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
pub fn imread(path: impl AsRef<Path>) -> Result<Tensor, ImgProcError> {
    let path = path.as_ref();
    let reader = ImageReader::open(path).map_err(|e| ImgProcError::Io {
        message: format!("{}: {e}", path.display()),
    })?;
    let img = reader
        .decode()
        .map_err(|e| ImgProcError::ImageDecode {
            message: format!("{}: {e}", path.display()),
        })?
        .into_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();
    let data: Vec<f32> = raw.iter().map(|&b| b as f32 / 255.0).collect();
    Tensor::from_vec(vec![h, w, 3], data).map_err(Into::into)
}

/// Read an image as grayscale and return it as a `[H, W, 1]` `f32` tensor with values in `[0, 1]`.
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
pub fn imread_gray(path: impl AsRef<Path>) -> Result<Tensor, ImgProcError> {
    let path = path.as_ref();
    let reader = ImageReader::open(path).map_err(|e| ImgProcError::Io {
        message: format!("{}: {e}", path.display()),
    })?;
    let img = reader
        .decode()
        .map_err(|e| ImgProcError::ImageDecode {
            message: format!("{}: {e}", path.display()),
        })?
        .into_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();
    let data: Vec<f32> = raw.iter().map(|&b| b as f32 / 255.0).collect();
    Tensor::from_vec(vec![h, w, 1], data).map_err(Into::into)
}

/// Write a `[H, W, C]` `f32` tensor (values in `[0, 1]`) to disk.
///
/// Format is inferred from the file extension. Channels must be 1 (grayscale) or 3 (RGB).
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
pub fn imwrite(path: impl AsRef<Path>, image: &Tensor) -> Result<(), ImgProcError> {
    let path = path.as_ref();
    let shape = image.shape();
    if shape.len() != 3 {
        return Err(ImgProcError::InvalidImageShape {
            expected_rank: 3,
            got: shape.to_vec(),
        });
    }
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 1 && c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let format = ImageFormat::from_path(path).map_err(|_| ImgProcError::UnsupportedFormat {
        path: path.display().to_string(),
    })?;

    let bytes: Vec<u8> = image
        .data()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    let dyn_img = if c == 1 {
        DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, bytes).ok_or_else(|| {
                ImgProcError::ImageEncode {
                    message: "failed to construct grayscale image buffer".into(),
                }
            })?,
        )
    } else {
        DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(w as u32, h as u32, bytes).ok_or_else(|| {
                ImgProcError::ImageEncode {
                    message: "failed to construct RGB image buffer".into(),
                }
            })?,
        )
    };

    dyn_img
        .save_with_format(path, format)
        .map_err(|e| ImgProcError::ImageEncode {
            message: format!("{}: {e}", path.display()),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_imread_imwrite_roundtrip_rgb() {
        let dir = std::env::temp_dir().join("yscv_io_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_rgb.png");

        // Create a small 2x3 RGB image tensor.
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // row 0: R, G, B
            0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, // row 1
        ];
        let img = Tensor::from_vec(vec![2, 3, 3], data).unwrap();
        imwrite(&path, &img).unwrap();

        let loaded = imread(&path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3, 3]);

        // Values should round-trip within 1/255 tolerance.
        for (a, b) in img.data().iter().zip(loaded.data().iter()) {
            assert!((a - b).abs() < 2.0 / 255.0, "a={a} b={b}");
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_imread_gray() {
        let dir = std::env::temp_dir().join("yscv_io_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_gray.png");

        let data: Vec<f32> = vec![0.0, 0.5, 1.0, 0.25];
        let img = Tensor::from_vec(vec![2, 2, 1], data).unwrap();
        imwrite(&path, &img).unwrap();

        let loaded = imread_gray(&path).unwrap();
        assert_eq!(loaded.shape(), &[2, 2, 1]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_imwrite_invalid_rank() {
        let path = Path::new("/tmp/invalid.png");
        let img = Tensor::from_vec(vec![4], vec![0.0; 4]).unwrap();
        assert!(imwrite(path, &img).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_imread_nonexistent_file() {
        let result = imread(Path::new("/tmp/nonexistent_yscv_test_image.png"));
        assert!(result.is_err());
    }
}
