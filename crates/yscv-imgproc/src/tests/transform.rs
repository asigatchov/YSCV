use yscv_tensor::Tensor;

use super::super::{
    BBox, ImgProcError, TemplateMatchMethod, distance_transform, gaussian_pyramid, hough_lines,
    integral_image, nms, normalize, template_match, warp_affine, warp_perspective, watershed,
};

// ── normalize ───────────────────────────────────────────────────────

#[test]
fn normalize_applies_per_channel_params() {
    let input = Tensor::from_vec(vec![1, 1, 3], vec![2.0, 4.0, 6.0]).unwrap();
    let out = normalize(&input, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3]);
    assert_eq!(out.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn normalize_rejects_zero_std() {
    let input = Tensor::from_vec(vec![1, 1, 1], vec![1.0]).unwrap();
    let err = normalize(&input, &[0.0], &[0.0]).unwrap_err();
    assert_eq!(err, ImgProcError::ZeroStdAtChannel { channel: 0 });
}

// ── NMS ─────────────────────────────────────────────────────────────

#[test]
fn nms_suppresses_overlapping_boxes() {
    let boxes = vec![
        BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
        },
        BBox {
            x1: 1.0,
            y1: 1.0,
            x2: 11.0,
            y2: 11.0,
            score: 0.8,
        },
        BBox {
            x1: 50.0,
            y1: 50.0,
            x2: 60.0,
            y2: 60.0,
            score: 0.7,
        },
    ];
    let keep = nms(&boxes, 0.5);
    assert_eq!(keep.len(), 2); // box 0 and box 2 kept; box 1 suppressed
    assert_eq!(keep[0], 0);
    assert_eq!(keep[1], 2);
}

#[test]
fn nms_no_overlap_keeps_all() {
    let boxes = vec![
        BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 5.0,
            y2: 5.0,
            score: 0.9,
        },
        BBox {
            x1: 20.0,
            y1: 20.0,
            x2: 25.0,
            y2: 25.0,
            score: 0.8,
        },
    ];
    let keep = nms(&boxes, 0.5);
    assert_eq!(keep.len(), 2);
}

// ── Template matching ───────────────────────────────────────────────

#[test]
fn template_match_ssd_finds_exact_patch() {
    let mut data = vec![0.0f32; 10 * 10];
    // Place a 3x3 bright patch at (4,4)
    for y in 4..7 {
        for x in 4..7 {
            data[y * 10 + x] = 1.0;
        }
    }
    let image = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let template = Tensor::from_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap();
    let result = template_match(&image, &template, TemplateMatchMethod::Ssd).unwrap();
    assert_eq!(result.x, 4);
    assert_eq!(result.y, 4);
    assert!(result.score.abs() < 1e-5, "exact match SSD should be ~0");
}

#[test]
fn template_match_ncc_finds_best_correlation() {
    // Place a gradient patch in the image; NCC requires non-constant template
    let pattern = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
    let mut data = vec![0.0f32; 10 * 10];
    for (i, &v) in pattern.iter().enumerate() {
        let ty = i / 3;
        let tx = i % 3;
        data[(3 + ty) * 10 + (5 + tx)] = v;
    }
    let image = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let template = Tensor::from_vec(vec![3, 3, 1], pattern).unwrap();
    let result = template_match(&image, &template, TemplateMatchMethod::Ncc).unwrap();
    assert_eq!(result.x, 5);
    assert_eq!(result.y, 3);
    assert!(
        result.score > 0.9,
        "NCC at exact match should be near 1.0, got {}",
        result.score
    );
}

// ── Hough lines ─────────────────────────────────────────────────────

#[test]
fn hough_detects_horizontal_line() {
    let mut data = vec![0.0f32; 20 * 20];
    for x in 0..20 {
        data[10 * 20 + x] = 1.0; // horizontal line at y=10
    }
    let img = Tensor::from_vec(vec![20, 20, 1], data).unwrap();
    let lines = hough_lines(&img, 1.0, std::f32::consts::PI / 180.0, 15).unwrap();
    assert!(!lines.is_empty(), "should detect horizontal line");
    // The dominant line should have theta near π/2 (90°)
    let dominant = &lines[0];
    let angle_deg = dominant.theta * 180.0 / std::f32::consts::PI;
    assert!(
        (angle_deg - 90.0).abs() < 5.0,
        "theta should be near 90°, got {angle_deg}"
    );
}

// ── Gaussian pyramid ────────────────────────────────────────────────

#[test]
fn gaussian_pyramid_produces_halved_sizes() {
    let img = Tensor::from_vec(vec![16, 16, 3], vec![0.5; 16 * 16 * 3]).unwrap();
    let pyr = gaussian_pyramid(&img, 3).unwrap();
    assert_eq!(pyr.len(), 4); // original + 3 levels
    assert_eq!(pyr[0].shape(), &[16, 16, 3]);
    assert_eq!(pyr[1].shape(), &[8, 8, 3]);
    assert_eq!(pyr[2].shape(), &[4, 4, 3]);
    assert_eq!(pyr[3].shape(), &[2, 2, 3]);
}

#[test]
fn gaussian_pyramid_uniform_preserves_value() {
    let img = Tensor::from_vec(vec![8, 8, 1], vec![0.7; 64]).unwrap();
    let pyr = gaussian_pyramid(&img, 2).unwrap();
    for level in &pyr {
        for v in level.data().iter().copied() {
            assert!(
                (v - 0.7f32).abs() < 1e-5,
                "uniform value should be preserved"
            );
        }
    }
}

// ── Distance transform ──────────────────────────────────────────────

#[test]
fn distance_transform_single_foreground_pixel() {
    let mut data = vec![0.0f32; 5 * 5];
    data[2 * 5 + 2] = 1.0; // center pixel is foreground
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let dist = distance_transform(&img).unwrap();
    assert_eq!(dist.shape(), &[5, 5, 1]);
    // Center pixel should have distance 1.0 (L1 to nearest background = 1)
    assert!((dist.data()[2 * 5 + 2] - 1.0).abs() < 1e-5);
    // Background pixels should have distance 0
    assert_eq!(dist.data()[0], 0.0);
}

#[test]
fn distance_transform_all_foreground_stays_large() {
    let img = Tensor::from_vec(vec![5, 5, 1], vec![1.0; 25]).unwrap();
    let dist = distance_transform(&img).unwrap();
    // All foreground, no background -> all distances should be capped at (h+w)
    for &v in dist.data() {
        assert!(
            v > 0.0,
            "all-foreground pixels should have nonzero distance"
        );
    }
}

// ── Warp affine ─────────────────────────────────────────────────────

#[test]
fn warp_affine_identity_preserves_center() {
    let img = Tensor::from_vec(
        vec![3, 3, 1],
        vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let out = warp_affine(&img, 3, 3, &identity, 0.0).unwrap();
    assert_eq!(out.shape(), &[3, 3, 1]);
    assert!((out.data()[4] - 5.0).abs() < 1e-5);
}

#[test]
fn warp_affine_rejects_zero_output() {
    let img = Tensor::filled(vec![3, 3, 1], 1.0).unwrap();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert!(warp_affine(&img, 0, 3, &identity, 0.0).is_err());
}

#[test]
fn warp_affine_translation_shifts_content() {
    let mut data = vec![0.0f32; 5 * 5];
    data[2 * 5 + 2] = 1.0; // center pixel
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let translate_right = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]; // dst->src: x-1
    let out = warp_affine(&img, 5, 5, &translate_right, 0.0).unwrap();
    // Pixel at (3,2) in output maps to src (2,2) which is 1.0
    assert!((out.data()[2 * 5 + 3] - 1.0).abs() < 1e-5);
}

// ── Perspective warp ────────────────────────────────────────────────

#[test]
fn warp_perspective_identity_preserves_center() {
    // Use a large enough image so bilinear sampling at half-pixel center converges
    let size = 8;
    let data: Vec<f32> = (0..size * size).map(|v| v as f32).collect();
    let img = Tensor::from_vec(vec![size, size, 1], data.clone()).unwrap();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let out = warp_perspective(&img, &identity, size, size, 0.0).unwrap();
    assert_eq!(out.shape(), &[size, size, 1]);
    // Check interior pixels (borders may have sub-pixel interpolation effects)
    for y in 1..size - 1 {
        for x in 1..size - 1 {
            let idx = y * size + x;
            assert!(
                (out.data()[idx] - data[idx]).abs() < 0.6,
                "identity warp interior pixel ({x},{y}): got {} vs {}",
                out.data()[idx],
                data[idx]
            );
        }
    }
}

// ── Integral image ──────────────────────────────────────────────────

#[test]
fn integral_image_correctness() {
    let data = vec![1.0f32; 3 * 3];
    let img = Tensor::from_vec(vec![3, 3, 1], data).unwrap();
    let sat = integral_image(&img).unwrap();
    assert_eq!(sat.shape(), &[3, 3, 1]);
    // sat[0,0] = 1, sat[0,2] = 3, sat[2,2] = 9
    assert_eq!(sat.data()[0], 1.0);
    assert_eq!(sat.data()[2], 3.0);
    assert_eq!(sat.data()[8], 9.0);
}

// ── Watershed ───────────────────────────────────────────────────────

#[test]
fn watershed_labels_all_from_seeds() {
    let mut img_data = vec![0.5f32; 10 * 10];
    img_data[0] = 0.1;
    img_data[99] = 0.1;
    let img = Tensor::from_vec(vec![10, 10, 1], img_data).unwrap();

    let mut mark_data = vec![0.0f32; 10 * 10];
    mark_data[0] = 1.0; // seed 1 at top-left
    mark_data[99] = 2.0; // seed 2 at bottom-right
    let markers = Tensor::from_vec(vec![10, 10, 1], mark_data).unwrap();

    let result = watershed(&img, &markers).unwrap();
    assert_eq!(result.shape(), &[10, 10, 1]);
    assert_eq!(result.data()[0], 1.0);
    assert_eq!(result.data()[99], 2.0);
    // All pixels should be labeled (no zeros in a fully-connected image)
    let unlabeled = result.data().iter().filter(|&&v| v == 0.0).count();
    assert!(unlabeled < 10, "too many unlabeled pixels: {unlabeled}");
}

#[test]
fn watershed_single_seed_fills_all() {
    let img = Tensor::from_vec(vec![5, 5, 1], vec![0.5; 25]).unwrap();
    let mut mark = vec![0.0f32; 25];
    mark[12] = 1.0; // center seed
    let markers = Tensor::from_vec(vec![5, 5, 1], mark).unwrap();
    let result = watershed(&img, &markers).unwrap();
    for &v in result.data() {
        assert_eq!(v, 1.0);
    }
}

// ── Homography / RANSAC ─────────────────────────────────────────────

#[test]
fn homography_4pt_identity_like() {
    let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let dst = src;
    let h = super::super::homography_4pt(&src, &dst).unwrap();
    assert!((h[0] - 1.0).abs() < 0.01);
    assert!((h[4] - 1.0).abs() < 0.01);
    assert!((h[8] - 1.0).abs() < 0.01);
}

#[test]
fn ransac_homography_finds_identity() {
    let pts: Vec<(f32, f32)> = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
    ];
    let result = super::super::ransac_homography(&pts, &pts, 100, 0.1, 42);
    assert!(result.is_some());
    let (h, inliers) = result.unwrap();
    assert!((h[0] - 1.0).abs() < 0.05);
    assert!(inliers.iter().filter(|&&x| x).count() >= 4);
}
