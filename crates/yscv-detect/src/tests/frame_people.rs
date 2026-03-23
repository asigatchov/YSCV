use crate::{
    DetectError, FramePeopleDetectScratch, Rgb8PeopleDetectScratch, detect_people_from_frame,
    detect_people_from_frame_with_scratch, detect_people_from_rgb8,
    detect_people_from_rgb8_with_scratch,
};
use yscv_tensor::Tensor;
use yscv_video::Frame;

#[test]
fn detect_people_from_rgb_frame_uses_grayscale_adapter() {
    let frame = Frame::new(
        0,
        0,
        Tensor::from_vec(
            vec![2, 2, 3],
            vec![
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1, //
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let dets = detect_people_from_frame(&frame, 0.5, 1, 0.5, 10).unwrap();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].bbox.x1, 0.0);
    assert_eq!(dets[0].bbox.x2, 1.0);
}

#[test]
fn detect_people_from_frame_with_scratch_uses_grayscale_adapter() {
    let frame = Frame::new(
        0,
        0,
        Tensor::from_vec(
            vec![2, 2, 3],
            vec![
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1, //
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let mut scratch = FramePeopleDetectScratch::default();
    let dets =
        detect_people_from_frame_with_scratch(&frame, 0.5, 1, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].bbox.x1, 0.0);
    assert_eq!(dets[0].bbox.x2, 1.0);
}

#[test]
fn detect_people_from_frame_with_scratch_accepts_grayscale_heatmap_input() {
    let frame = Frame::new(
        0,
        0,
        Tensor::from_vec(
            vec![2, 2, 1],
            vec![
                0.8, 0.1, //
                0.8, 0.1,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let mut scratch = FramePeopleDetectScratch::default();
    let dets =
        detect_people_from_frame_with_scratch(&frame, 0.5, 1, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].bbox.x1, 0.0);
    assert_eq!(dets[0].bbox.x2, 1.0);
}

#[test]
fn detect_people_from_frame_with_scratch_reuses_buffer_for_resized_frames() {
    let mut scratch = FramePeopleDetectScratch::default();

    let small = Frame::new(
        0,
        0,
        Tensor::from_vec(
            vec![2, 2, 3],
            vec![
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1, //
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let small_dets =
        detect_people_from_frame_with_scratch(&small, 0.5, 1, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(small_dets.len(), 1);

    let large = Frame::new(
        0,
        0,
        Tensor::from_vec(
            vec![2, 3, 3],
            vec![
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, //
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let large_dets =
        detect_people_from_frame_with_scratch(&large, 0.5, 1, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(large_dets.len(), 1);
    assert_eq!(large_dets[0].bbox.x1, 0.0);
    assert_eq!(large_dets[0].bbox.x2, 2.0);
}

#[test]
fn detect_people_from_rgb8_uses_grayscale_adapter() {
    let rgb8 = vec![
        230, 230, 230, 20, 20, 20, //
        230, 230, 230, 20, 20, 20,
    ];
    let dets = detect_people_from_rgb8(2, 2, &rgb8, 0.5, 1, 0.5, 10).unwrap();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].bbox.x1, 0.0);
    assert_eq!(dets[0].bbox.x2, 1.0);
}

#[test]
fn detect_people_from_rgb8_with_scratch_uses_grayscale_adapter() {
    let rgb8 = vec![
        230, 230, 230, 20, 20, 20, //
        230, 230, 230, 20, 20, 20,
    ];
    let mut scratch = Rgb8PeopleDetectScratch::default();
    let dets =
        detect_people_from_rgb8_with_scratch((2, 2), &rgb8, 0.5, 1, 0.5, 10, &mut scratch).unwrap();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].bbox.x1, 0.0);
    assert_eq!(dets[0].bbox.x2, 1.0);
}

#[test]
fn detect_people_from_rgb8_with_scratch_reuses_buffer_for_resized_frames() {
    let mut scratch = Rgb8PeopleDetectScratch::default();

    let rgb8_small = vec![
        230, 230, 230, 20, 20, 20, //
        230, 230, 230, 20, 20, 20,
    ];
    let dets_small =
        detect_people_from_rgb8_with_scratch((2, 2), &rgb8_small, 0.5, 1, 0.5, 10, &mut scratch)
            .unwrap();
    assert_eq!(dets_small.len(), 1);

    let rgb8_large = vec![
        230, 230, 230, 230, 230, 230, 20, 20, 20, //
        230, 230, 230, 230, 230, 230, 20, 20, 20,
    ];
    let dets_large =
        detect_people_from_rgb8_with_scratch((3, 2), &rgb8_large, 0.5, 1, 0.5, 10, &mut scratch)
            .unwrap();
    assert_eq!(dets_large.len(), 1);
    assert_eq!(dets_large[0].bbox.x1, 0.0);
    assert_eq!(dets_large[0].bbox.x2, 2.0);
}

#[test]
fn detect_people_from_rgb8_rejects_invalid_buffer_size() {
    let err = detect_people_from_rgb8(2, 2, &[1, 2, 3], 0.5, 2, 0.4, 4).unwrap_err();
    assert_eq!(
        err,
        DetectError::InvalidRgb8BufferSize {
            expected: 12,
            got: 3
        }
    );
}
