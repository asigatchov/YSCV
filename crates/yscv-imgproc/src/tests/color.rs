use yscv_tensor::Tensor;

use super::super::{
    hsv_to_rgb, lab_to_rgb, rgb_to_bgr, rgb_to_grayscale, rgb_to_hsv, rgb_to_lab, rgb_to_yuv,
    yuv_to_rgb,
};

#[test]
fn rgb_to_grayscale_converts_single_pixel() {
    let rgb = Tensor::from_vec(vec![1, 1, 3], vec![10.0, 20.0, 30.0]).unwrap();
    let gray = rgb_to_grayscale(&rgb).unwrap();
    assert_eq!(gray.shape(), &[1, 1, 1]);
    assert!((gray.data()[0] - 18.15).abs() < 1e-4);
}

#[test]
fn rgb_to_hsv_pure_red() {
    let img = Tensor::from_vec(vec![1, 1, 3], vec![1.0, 0.0, 0.0]).unwrap();
    let hsv = rgb_to_hsv(&img).unwrap();
    assert!((hsv.data()[0] - 0.0).abs() < 1e-5); // H=0
    assert!((hsv.data()[1] - 1.0).abs() < 1e-5); // S=1
    assert!((hsv.data()[2] - 1.0).abs() < 1e-5); // V=1
}

#[test]
fn hsv_to_rgb_roundtrip() {
    let img = Tensor::from_vec(vec![1, 1, 3], vec![0.8, 0.3, 0.5]).unwrap();
    let hsv = rgb_to_hsv(&img).unwrap();
    let rgb = hsv_to_rgb(&hsv).unwrap();
    for i in 0..3 {
        assert!(
            (rgb.data()[i] - img.data()[i]).abs() < 1e-4,
            "roundtrip mismatch at channel {i}"
        );
    }
}

#[test]
fn rgb_to_bgr_swaps_channels() {
    let img = Tensor::from_vec(vec![1, 1, 3], vec![0.1, 0.2, 0.3]).unwrap();
    let bgr = rgb_to_bgr(&img).unwrap();
    assert_eq!(bgr.data(), &[0.3, 0.2, 0.1]);
}

// --- LAB tests ---

#[test]
fn rgb_to_lab_known_white() {
    // Pure white (1,1,1) should map to L=100, a≈0, b≈0
    let img = Tensor::from_vec(vec![1, 1, 3], vec![1.0, 1.0, 1.0]).unwrap();
    let lab = rgb_to_lab(&img).unwrap();
    let d = lab.data();
    assert!(
        (d[0] - 100.0).abs() < 0.01,
        "L should be ~100, got {}",
        d[0]
    );
    assert!(d[1].abs() < 0.01, "a should be ~0, got {}", d[1]);
    assert!(d[2].abs() < 0.01, "b should be ~0, got {}", d[2]);
}

#[test]
fn rgb_to_lab_known_black() {
    let img = Tensor::from_vec(vec![1, 1, 3], vec![0.0, 0.0, 0.0]).unwrap();
    let lab = rgb_to_lab(&img).unwrap();
    let d = lab.data();
    assert!(d[0].abs() < 0.01, "L should be ~0, got {}", d[0]);
    assert!(d[1].abs() < 0.01, "a should be ~0, got {}", d[1]);
    assert!(d[2].abs() < 0.01, "b should be ~0, got {}", d[2]);
}

#[test]
fn rgb_lab_roundtrip() {
    let img = Tensor::from_vec(vec![1, 2, 3], vec![0.8, 0.3, 0.5, 0.1, 0.6, 0.9]).unwrap();
    let lab = rgb_to_lab(&img).unwrap();
    let rgb = lab_to_rgb(&lab).unwrap();
    for i in 0..6 {
        assert!(
            (rgb.data()[i] - img.data()[i]).abs() < 1e-4,
            "LAB roundtrip mismatch at index {}: got {} expected {}",
            i,
            rgb.data()[i],
            img.data()[i],
        );
    }
}

#[test]
fn rgb_to_lab_known_red() {
    // Pure red (1,0,0) → L≈53.23, a≈80.11, b≈67.22
    let img = Tensor::from_vec(vec![1, 1, 3], vec![1.0, 0.0, 0.0]).unwrap();
    let lab = rgb_to_lab(&img).unwrap();
    let d = lab.data();
    assert!(
        (d[0] - 53.23).abs() < 0.5,
        "L should be ~53.23, got {}",
        d[0]
    );
    assert!(
        (d[1] - 80.11).abs() < 0.5,
        "a should be ~80.11, got {}",
        d[1]
    );
    assert!(
        (d[2] - 67.22).abs() < 0.5,
        "b should be ~67.22, got {}",
        d[2]
    );
}

// --- YUV tests ---

#[test]
fn rgb_to_yuv_known_white() {
    let img = Tensor::from_vec(vec![1, 1, 3], vec![1.0, 1.0, 1.0]).unwrap();
    let yuv = rgb_to_yuv(&img).unwrap();
    let d = yuv.data();
    assert!((d[0] - 1.0).abs() < 1e-4, "Y should be ~1.0, got {}", d[0]);
    assert!(d[1].abs() < 1e-3, "U should be ~0, got {}", d[1]);
    assert!(d[2].abs() < 1e-3, "V should be ~0, got {}", d[2]);
}

#[test]
fn rgb_yuv_roundtrip() {
    let img = Tensor::from_vec(vec![1, 2, 3], vec![0.8, 0.3, 0.5, 0.1, 0.6, 0.9]).unwrap();
    let yuv = rgb_to_yuv(&img).unwrap();
    let rgb = yuv_to_rgb(&yuv).unwrap();
    for i in 0..6 {
        assert!(
            (rgb.data()[i] - img.data()[i]).abs() < 1e-4,
            "YUV roundtrip mismatch at index {}: got {} expected {}",
            i,
            rgb.data()[i],
            img.data()[i],
        );
    }
}

#[test]
fn rgb_to_yuv_known_red() {
    let img = Tensor::from_vec(vec![1, 1, 3], vec![1.0, 0.0, 0.0]).unwrap();
    let yuv = rgb_to_yuv(&img).unwrap();
    let d = yuv.data();
    assert!(
        (d[0] - 0.299).abs() < 1e-4,
        "Y should be 0.299, got {}",
        d[0]
    );
    assert!(
        (d[1] - (-0.14713)).abs() < 1e-4,
        "U should be -0.14713, got {}",
        d[1]
    );
    assert!(
        (d[2] - 0.615).abs() < 1e-4,
        "V should be 0.615, got {}",
        d[2]
    );
}
