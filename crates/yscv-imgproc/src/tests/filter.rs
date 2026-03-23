use yscv_tensor::Tensor;

use super::super::{
    TemplateMatchMethod, bilateral_filter, box_blur_3x3, canny, filter2d, gaussian_blur_3x3,
    gaussian_blur_5x5, inpaint_telea, laplacian_3x3, median_blur_3x3, median_filter,
    scharr_3x3_gradients, scharr_3x3_magnitude, sobel_3x3_gradients, sobel_3x3_magnitude,
    template_match,
};

#[test]
fn box_blur_3x3_smooths_center_pixel() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 9.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let out = box_blur_3x3(&input).unwrap();
    assert_eq!(out.shape(), &[3, 3, 1]);
    // Center averages all 3x3 values -> 1.0
    assert!((out.data()[4] - 1.0).abs() < 1e-6);
}

#[test]
fn gaussian_blur_3x3_uniform_image_center_preserved() {
    let img = Tensor::filled(vec![5, 5, 1], 0.5).unwrap();
    let blurred = gaussian_blur_3x3(&img).unwrap();
    assert_eq!(blurred.shape(), &[5, 5, 1]);
    let center = blurred.data()[2 * 5 + 2];
    assert!((center - 0.5_f32).abs() < 1e-5);
}

#[test]
fn gaussian_blur_5x5_preserves_shape() {
    let img = Tensor::filled(vec![5, 5, 3], 0.3).unwrap();
    let blurred = gaussian_blur_5x5(&img).unwrap();
    assert_eq!(blurred.shape(), &[5, 5, 3]);
}

#[test]
fn laplacian_3x3_flat_center_is_zero() {
    let img = Tensor::filled(vec![5, 5, 1], 0.5).unwrap();
    let lap = laplacian_3x3(&img).unwrap();
    let center = lap.data()[2 * 5 + 2];
    assert!((center as f64).abs() < 1e-5);
}

#[test]
fn median_blur_3x3_preserves_shape() {
    let img = Tensor::filled(vec![3, 3, 1], 0.5).unwrap();
    let blurred = median_blur_3x3(&img).unwrap();
    assert_eq!(blurred.shape(), &[3, 3, 1]);
}

#[test]
fn sobel_3x3_gradients_detect_axis_edges() {
    let vertical_edge = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let (gx_vertical, gy_vertical) = sobel_3x3_gradients(&vertical_edge).unwrap();
    assert_eq!(gx_vertical.shape(), &[3, 3, 1]);
    assert_eq!(gy_vertical.shape(), &[3, 3, 1]);
    assert!((gx_vertical.data()[4] - 4.0).abs() < 1e-6);
    assert!(gy_vertical.data()[4].abs() < 1e-6);

    let horizontal_edge = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let (gx_horizontal, gy_horizontal) = sobel_3x3_gradients(&horizontal_edge).unwrap();
    assert!(gx_horizontal.data()[4].abs() < 1e-6);
    assert!((gy_horizontal.data()[4] - 4.0).abs() < 1e-6);
}

#[test]
fn sobel_3x3_magnitude_matches_gradient_norm() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 1.0, 2.0, //
            1.0, 2.0, 3.0, //
            2.0, 3.0, 4.0,
        ],
    )
    .unwrap();
    let (gx, gy) = sobel_3x3_gradients(&input).unwrap();
    let magnitude = sobel_3x3_magnitude(&input).unwrap();

    assert_eq!(magnitude.shape(), &[3, 3, 1]);
    for idx in 0..magnitude.len() {
        let expected = (gx.data()[idx] * gx.data()[idx] + gy.data()[idx] * gy.data()[idx]).sqrt();
        assert!((magnitude.data()[idx] - expected).abs() < 1e-6);
    }
}

#[test]
fn canny_produces_binary_edge_map() {
    let mut data = vec![0.0f32; 10 * 10];
    // vertical edge in the middle
    for y in 0..10 {
        for x in 5..10 {
            data[y * 10 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let edges = canny(&img, 0.1, 0.3).unwrap();
    assert_eq!(edges.shape(), &[10, 10, 1]);
    for &v in edges.data() {
        assert!(v == 0.0 || v == 1.0);
    }
    let edge_count: usize = edges.data().iter().filter(|&&v| v == 1.0).count();
    assert!(edge_count > 0, "should detect at least one edge pixel");
}

#[test]
fn canny_rejects_multi_channel() {
    let img = Tensor::filled(vec![5, 5, 3], 0.5).unwrap();
    assert!(canny(&img, 0.1, 0.3).is_err());
}

#[test]
fn bilateral_filter_preserves_flat_image() {
    let img = Tensor::from_vec(vec![5, 5, 1], vec![0.5; 25]).unwrap();
    let out = bilateral_filter(&img, 2, 0.1, 2.0).unwrap();
    for &v in out.data() {
        assert!((v - 0.5).abs() < 1e-4);
    }
}

#[test]
fn bilateral_filter_reduces_noise() {
    let mut data = vec![0.5f32; 7 * 7];
    data[3 * 7 + 3] = 1.0; // single noisy pixel
    let img = Tensor::from_vec(vec![7, 7, 1], data).unwrap();
    let out = bilateral_filter(&img, 2, 0.3, 2.0).unwrap();
    // The noisy pixel should be somewhat smoothed toward 0.5
    assert!(out.data()[3 * 7 + 3] < 0.9, "bilateral should reduce noise");
}

#[test]
fn filter2d_identity_kernel() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = Tensor::from_vec(vec![3, 3, 1], data).unwrap();
    let identity = Tensor::from_vec(
        vec![3, 3, 1],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let out = filter2d(&img, &identity).unwrap();
    // Interior pixel (1,1) should match: 5.0
    assert!((out.data()[4] - 5.0).abs() < 1e-5);
}

#[test]
fn filter2d_box_blur_kernel() {
    let img = Tensor::from_vec(
        vec![3, 3, 1],
        vec![0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let box_k = Tensor::from_vec(vec![3, 3, 1], vec![1.0 / 9.0; 9]).unwrap();
    let out = filter2d(&img, &box_k).unwrap();
    assert!((out.data()[4] - 1.0).abs() < 1e-5);
}

#[test]
fn median_filter_3x3_removes_salt_pepper() {
    // 7x7 uniform image at 0.5, with salt-and-pepper noise
    let mut data = vec![0.5f32; 7 * 7];
    // salt
    data[7 + 3] = 1.0;
    data[3 * 7 + 5] = 1.0;
    data[5 * 7 + 1] = 1.0;
    // pepper
    data[2 * 7 + 2] = 0.0;
    data[4 * 7 + 4] = 0.0;
    let img = Tensor::from_vec(vec![7, 7, 1], data).unwrap();
    let out = median_filter(&img, 3).unwrap();
    assert_eq!(out.shape(), &[7, 7, 1]);
    // Interior noisy pixels should be restored to 0.5 (median of mostly-0.5 neighborhood)
    let noisy_indices = [7 + 3, 3 * 7 + 5, 2 * 7 + 2, 4 * 7 + 4];
    for &idx in &noisy_indices {
        assert!(
            (out.data()[idx] - 0.5).abs() < 1e-6,
            "noisy pixel at index {} not cleaned: got {}",
            idx,
            out.data()[idx]
        );
    }
}

#[test]
fn median_filter_5x5_on_uniform() {
    let img = Tensor::filled(vec![9, 9, 1], 0.7).unwrap();
    let out = median_filter(&img, 5).unwrap();
    assert_eq!(out.shape(), &[9, 9, 1]);
    for (i, &v) in out.data().iter().enumerate() {
        assert!(
            (v - 0.7).abs() < 1e-6,
            "pixel {} changed from uniform: got {}",
            i,
            v
        );
    }
}

#[test]
fn median_filter_rejects_even_kernel() {
    let img = Tensor::filled(vec![5, 5, 1], 0.5).unwrap();
    assert!(median_filter(&img, 4).is_err());
    assert!(median_filter(&img, 0).is_err());
}

#[test]
fn scharr_magnitude_on_gradient() {
    // Horizontal gradient: left half 0, right half 1 -> Scharr should detect strong edges
    let mut data = vec![0.0f32; 5 * 5];
    for y in 0..5 {
        for x in 3..5 {
            data[y * 5 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let (gx, _gy) = scharr_3x3_gradients(&img).unwrap();
    assert_eq!(gx.shape(), &[5, 5, 1]);
    // The center pixel (2,2) should have a strong horizontal gradient
    let center_gx = gx.data()[2 * 5 + 2];
    assert!(
        center_gx.abs() > 1.0,
        "Scharr gx at transition should be strong, got {}",
        center_gx
    );

    let mag = scharr_3x3_magnitude(&img).unwrap();
    assert_eq!(mag.shape(), &[5, 5, 1]);
    let center_mag = mag.data()[2 * 5 + 2];
    assert!(
        center_mag > 1.0,
        "Scharr magnitude at transition should be strong, got {}",
        center_mag
    );
}

#[test]
fn inpaint_fills_hole() {
    // 5x5 image filled with 0.8, but the center pixel is 0.0 and masked
    let mut img_data = vec![0.8f32; 5 * 5];
    img_data[2 * 5 + 2] = 0.0;
    let img = Tensor::from_vec(vec![5, 5, 1], img_data).unwrap();

    let mut mask_data = vec![0.0f32; 5 * 5];
    mask_data[2 * 5 + 2] = 1.0; // mask the center pixel
    let mask = Tensor::from_vec(vec![5, 5, 1], mask_data).unwrap();

    let result = inpaint_telea(&img, &mask, 2).unwrap();
    assert_eq!(result.shape(), &[5, 5, 1]);

    // The center pixel should be filled with approximately 0.8
    let center = result.data()[2 * 5 + 2];
    assert!(
        (center - 0.8).abs() < 0.1,
        "inpainted center should be ~0.8, got {}",
        center
    );
}

#[test]
fn template_match_finds_pattern() {
    // 10x10 image with a 3x3 bright patch at position (4,5)
    let mut img_data = vec![0.0f32; 10 * 10];
    for y in 5..8 {
        for x in 4..7 {
            img_data[y * 10 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![10, 10, 1], img_data).unwrap();

    // Template: 3x3 bright patch
    let tmpl = Tensor::filled(vec![3, 3, 1], 1.0).unwrap();

    let result = template_match(&img, &tmpl, TemplateMatchMethod::Ssd).unwrap();
    assert_eq!(result.x, 4, "expected match at x=4, got {}", result.x);
    assert_eq!(result.y, 5, "expected match at y=5, got {}", result.y);
}
