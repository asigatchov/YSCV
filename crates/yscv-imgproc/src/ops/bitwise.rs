//! Bitwise and pixel-level operations (OpenCV-style in_range, bitwise, blending).

use yscv_tensor::Tensor;

use super::super::ImgProcError;

/// Create a binary mask where each pixel is 1.0 if all channels are within
/// `[lower, upper]` (inclusive), 0.0 otherwise.
///
/// Input: `[H, W, C]`, output: `[H, W, 1]`.
pub fn in_range(image: &Tensor, lower: &[f32], upper: &[f32]) -> Result<Tensor, ImgProcError> {
    let shape = image.shape();
    if shape.len() != 3 {
        return Err(ImgProcError::InvalidImageShape {
            expected_rank: 3,
            got: shape.to_vec(),
        });
    }
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if lower.len() != c || upper.len() != c {
        return Err(ImgProcError::InvalidChannelCount {
            expected: c,
            got: lower.len(),
        });
    }

    let data = image.data();
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            let mut inside = true;
            for ch in 0..c {
                let v = data[base + ch];
                if v < lower[ch] || v > upper[ch] {
                    inside = false;
                    break;
                }
            }
            out[y * w + x] = if inside { 1.0 } else { 0.0 };
        }
    }
    Ok(Tensor::from_vec(vec![h, w, 1], out)?)
}

/// Elementwise AND of two single-channel binary masks.
///
/// Values > 0.5 are treated as 1, otherwise 0.
pub fn bitwise_and(a: &Tensor, b: &Tensor) -> Result<Tensor, ImgProcError> {
    if a.shape() != b.shape() {
        return Err(ImgProcError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let out: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| if *av > 0.5 && *bv > 0.5 { 1.0 } else { 0.0 })
        .collect();
    Ok(Tensor::from_vec(a.shape().to_vec(), out)?)
}

/// Elementwise OR of two single-channel binary masks.
pub fn bitwise_or(a: &Tensor, b: &Tensor) -> Result<Tensor, ImgProcError> {
    if a.shape() != b.shape() {
        return Err(ImgProcError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let out: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| if *av > 0.5 || *bv > 0.5 { 1.0 } else { 0.0 })
        .collect();
    Ok(Tensor::from_vec(a.shape().to_vec(), out)?)
}

/// Elementwise NOT of a single-channel binary mask.
pub fn bitwise_not(a: &Tensor) -> Result<Tensor, ImgProcError> {
    let out: Vec<f32> = a
        .data()
        .iter()
        .map(|v| if *v > 0.5 { 0.0 } else { 1.0 })
        .collect();
    Ok(Tensor::from_vec(a.shape().to_vec(), out)?)
}

/// Alpha-blend two images: `dst = alpha * a + beta * b + gamma`.
///
/// Both images must have the same shape.
pub fn add_weighted(
    a: &Tensor,
    alpha: f32,
    b: &Tensor,
    beta: f32,
    gamma: f32,
) -> Result<Tensor, ImgProcError> {
    if a.shape() != b.shape() {
        return Err(ImgProcError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let out: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| alpha * av + beta * bv + gamma)
        .collect();
    Ok(Tensor::from_vec(a.shape().to_vec(), out)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_range() {
        let img = Tensor::from_vec(
            vec![2, 2, 3],
            vec![0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5],
        )
        .unwrap();
        let mask = in_range(&img, &[0.4, 0.4, 0.4], &[0.6, 0.6, 0.6]).unwrap();
        assert_eq!(mask.shape(), &[2, 2, 1]);
        assert_eq!(mask.data(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_bitwise_and_or_not() {
        let a = Tensor::from_vec(vec![4], vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let b = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(bitwise_and(&a, &b).unwrap().data(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(bitwise_or(&a, &b).unwrap().data(), &[1.0, 1.0, 1.0, 0.0]);
        assert_eq!(bitwise_not(&a).unwrap().data(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_add_weighted() {
        let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
        let out = add_weighted(&a, 0.5, &b, 0.5, 0.0).unwrap();
        assert_eq!(out.data(), &[2.5, 3.5, 4.5]);
    }
}
