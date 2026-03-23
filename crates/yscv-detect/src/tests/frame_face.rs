use crate::{
    CLASS_ID_FACE, DetectError, FrameFaceDetectScratch, Rgb8FaceDetectScratch,
    detect_faces_from_frame, detect_faces_from_frame_with_scratch, detect_faces_from_rgb8,
    detect_faces_from_rgb8_with_scratch,
};
use yscv_tensor::Tensor;
use yscv_video::Frame;

#[test]
fn detect_faces_from_rgb_frame_finds_skin_region() {
    let background = [0.10, 0.10, 0.12];
    let skin = [0.78, 0.60, 0.46];
    let mut data = vec![0.0f32; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let base = (y * 6 + x) * 3;
            let color = if (1..5).contains(&y) && (2..5).contains(&x) {
                skin
            } else {
                background
            };
            data[base] = color[0];
            data[base + 1] = color[1];
            data[base + 2] = color[2];
        }
    }
    let frame = Frame::new(0, 0, Tensor::from_vec(vec![6, 6, 3], data).unwrap()).unwrap();

    let detections = detect_faces_from_frame(&frame, 0.35, 6, 0.4, 8).unwrap();
    assert_eq!(detections.len(), 1);
    let detected = detections[0];
    assert_eq!(detected.class_id, CLASS_ID_FACE);
    assert_eq!(detected.bbox.x1, 2.0);
    assert_eq!(detected.bbox.y1, 1.0);
    assert_eq!(detected.bbox.x2, 5.0);
    assert_eq!(detected.bbox.y2, 5.0);
}

#[test]
fn detect_faces_from_frame_with_scratch_finds_skin_region() {
    let background = [0.10, 0.10, 0.12];
    let skin = [0.78, 0.60, 0.46];
    let mut data = vec![0.0f32; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let base = (y * 6 + x) * 3;
            let color = if (1..5).contains(&y) && (2..5).contains(&x) {
                skin
            } else {
                background
            };
            data[base] = color[0];
            data[base + 1] = color[1];
            data[base + 2] = color[2];
        }
    }
    let frame = Frame::new(0, 0, Tensor::from_vec(vec![6, 6, 3], data).unwrap()).unwrap();

    let mut scratch = FrameFaceDetectScratch::default();
    let detections =
        detect_faces_from_frame_with_scratch(&frame, 0.35, 6, 0.4, 8, &mut scratch).unwrap();
    assert_eq!(detections.len(), 1);
    let detected = detections[0];
    assert_eq!(detected.class_id, CLASS_ID_FACE);
    assert_eq!(detected.bbox.x1, 2.0);
    assert_eq!(detected.bbox.y1, 1.0);
    assert_eq!(detected.bbox.x2, 5.0);
    assert_eq!(detected.bbox.y2, 5.0);
}

#[test]
fn detect_faces_from_frame_with_scratch_reuses_buffer_for_resized_frames() {
    let mut scratch = FrameFaceDetectScratch::default();

    let background = [0.10, 0.10, 0.12];
    let skin = [0.78, 0.60, 0.46];

    let mut small = vec![0.0f32; 4 * 4 * 3];
    for y in 0..4 {
        for x in 0..4 {
            let base = (y * 4 + x) * 3;
            let color = if (1..3).contains(&y) && (1..3).contains(&x) {
                skin
            } else {
                background
            };
            small[base] = color[0];
            small[base + 1] = color[1];
            small[base + 2] = color[2];
        }
    }
    let small_frame = Frame::new(0, 0, Tensor::from_vec(vec![4, 4, 3], small).unwrap()).unwrap();
    let first =
        detect_faces_from_frame_with_scratch(&small_frame, 0.35, 2, 0.4, 8, &mut scratch).unwrap();
    assert_eq!(first.len(), 1);

    let mut large = vec![0.0f32; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let base = (y * 6 + x) * 3;
            let color = if (1..5).contains(&y) && (2..5).contains(&x) {
                skin
            } else {
                background
            };
            large[base] = color[0];
            large[base + 1] = color[1];
            large[base + 2] = color[2];
        }
    }
    let large_frame = Frame::new(0, 0, Tensor::from_vec(vec![6, 6, 3], large).unwrap()).unwrap();
    let second =
        detect_faces_from_frame_with_scratch(&large_frame, 0.35, 6, 0.4, 8, &mut scratch).unwrap();
    assert_eq!(second.len(), 1);
}

#[test]
fn detect_faces_rejects_grayscale_input() {
    let frame = Frame::new(0, 0, Tensor::from_vec(vec![2, 2, 1], vec![0.5; 4]).unwrap()).unwrap();
    let err = detect_faces_from_frame(&frame, 0.5, 2, 0.4, 4).unwrap_err();
    assert_eq!(
        err,
        DetectError::InvalidChannelCount {
            expected: 3,
            got: 1
        }
    );
}

#[test]
fn detect_faces_rejects_invalid_nms_config() {
    let frame = Frame::new(
        0,
        0,
        Tensor::from_vec(vec![2, 2, 3], vec![0.5; 12]).unwrap(),
    )
    .unwrap();
    let err = detect_faces_from_frame(&frame, 0.5, 2, 0.4, 0).unwrap_err();
    assert_eq!(err, DetectError::InvalidMaxDetections { max_detections: 0 });
}

#[test]
fn detect_faces_from_rgb8_finds_skin_region() {
    let background = [26u8, 26u8, 31u8];
    let skin = [199u8, 153u8, 117u8];
    let mut rgb8 = vec![0u8; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let base = (y * 6 + x) * 3;
            let color = if (1..5).contains(&y) && (2..5).contains(&x) {
                skin
            } else {
                background
            };
            rgb8[base] = color[0];
            rgb8[base + 1] = color[1];
            rgb8[base + 2] = color[2];
        }
    }

    let detections = detect_faces_from_rgb8(6, 6, &rgb8, 0.35, 6, 0.4, 8).unwrap();
    assert_eq!(detections.len(), 1);
    let detected = detections[0];
    assert_eq!(detected.class_id, CLASS_ID_FACE);
    assert_eq!(detected.bbox.x1, 2.0);
    assert_eq!(detected.bbox.y1, 1.0);
    assert_eq!(detected.bbox.x2, 5.0);
    assert_eq!(detected.bbox.y2, 5.0);
}

#[test]
fn detect_faces_from_rgb8_with_scratch_finds_skin_region() {
    let background = [26u8, 26u8, 31u8];
    let skin = [199u8, 153u8, 117u8];
    let mut rgb8 = vec![0u8; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let base = (y * 6 + x) * 3;
            let color = if (1..5).contains(&y) && (2..5).contains(&x) {
                skin
            } else {
                background
            };
            rgb8[base] = color[0];
            rgb8[base + 1] = color[1];
            rgb8[base + 2] = color[2];
        }
    }

    let mut scratch = Rgb8FaceDetectScratch::default();
    let detections =
        detect_faces_from_rgb8_with_scratch((6, 6), &rgb8, 0.35, 6, 0.4, 8, &mut scratch).unwrap();
    assert_eq!(detections.len(), 1);
    let detected = detections[0];
    assert_eq!(detected.class_id, CLASS_ID_FACE);
    assert_eq!(detected.bbox.x1, 2.0);
    assert_eq!(detected.bbox.y1, 1.0);
    assert_eq!(detected.bbox.x2, 5.0);
    assert_eq!(detected.bbox.y2, 5.0);
}

#[test]
fn detect_faces_from_rgb8_with_scratch_reuses_buffer_for_resized_frames() {
    let mut scratch = Rgb8FaceDetectScratch::default();

    let background = [26u8, 26u8, 31u8];
    let skin = [199u8, 153u8, 117u8];
    let mut rgb8_small = vec![0u8; 4 * 4 * 3];
    for y in 0..4 {
        for x in 0..4 {
            let base = (y * 4 + x) * 3;
            let color = if (1..3).contains(&y) && (1..3).contains(&x) {
                skin
            } else {
                background
            };
            rgb8_small[base] = color[0];
            rgb8_small[base + 1] = color[1];
            rgb8_small[base + 2] = color[2];
        }
    }

    let first =
        detect_faces_from_rgb8_with_scratch((4, 4), &rgb8_small, 0.35, 2, 0.4, 8, &mut scratch)
            .unwrap();
    assert_eq!(first.len(), 1);

    let mut rgb8_large = vec![0u8; 6 * 6 * 3];
    for y in 0..6 {
        for x in 0..6 {
            let base = (y * 6 + x) * 3;
            let color = if (1..5).contains(&y) && (2..5).contains(&x) {
                skin
            } else {
                background
            };
            rgb8_large[base] = color[0];
            rgb8_large[base + 1] = color[1];
            rgb8_large[base + 2] = color[2];
        }
    }
    let second =
        detect_faces_from_rgb8_with_scratch((6, 6), &rgb8_large, 0.35, 6, 0.4, 8, &mut scratch)
            .unwrap();
    assert_eq!(second.len(), 1);
}

#[test]
fn detect_faces_from_rgb8_rejects_invalid_buffer_size() {
    let err = detect_faces_from_rgb8(2, 2, &[1, 2, 3], 0.5, 2, 0.4, 4).unwrap_err();
    assert_eq!(
        err,
        DetectError::InvalidRgb8BufferSize {
            expected: 12,
            got: 3
        }
    );
}

#[test]
fn detect_faces_from_rgb8_with_scratch_rejects_invalid_buffer_size() {
    let mut scratch = Rgb8FaceDetectScratch::default();
    let err = detect_faces_from_rgb8_with_scratch((2, 2), &[1, 2, 3], 0.5, 2, 0.4, 4, &mut scratch)
        .unwrap_err();
    assert_eq!(
        err,
        DetectError::InvalidRgb8BufferSize {
            expected: 12,
            got: 3
        }
    );
}
