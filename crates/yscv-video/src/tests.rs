use super::VideoError;
use super::camera::{
    CameraConfig, CameraDeviceInfo, CameraFrameSource, filter_camera_devices, select_camera_device,
};
use super::convert::rgb8_bytes_to_frame;
use super::frame::{Frame, PixelFormat, Rgb8Frame};
use super::normalize_rgb8_to_f32_inplace;
use super::source::{FrameSource, InMemoryFrameSource};
use super::stream::FrameStream;
use bytes::Bytes;
use yscv_tensor::Tensor;

#[test]
fn frame_new_accepts_rgb_and_gray() {
    let rgb = Frame::new(
        0,
        0,
        Tensor::from_vec(vec![2, 2, 3], vec![0.0; 12]).unwrap(),
    )
    .unwrap();
    assert_eq!(rgb.pixel_format(), PixelFormat::RgbF32);

    let gray = Frame::new(
        1,
        1_000,
        Tensor::from_vec(vec![2, 2, 1], vec![0.0; 4]).unwrap(),
    )
    .unwrap();
    assert_eq!(gray.pixel_format(), PixelFormat::GrayF32);
}

#[test]
fn frame_new_rejects_invalid_shape() {
    let err = Frame::new(
        0,
        0,
        Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap_err();
    assert_eq!(err, VideoError::InvalidFrameShape { got: vec![4] });
}

#[test]
fn in_memory_source_returns_frames_in_order() {
    let frames = vec![
        Frame::new(0, 0, Tensor::from_vec(vec![1, 1, 1], vec![1.0]).unwrap()).unwrap(),
        Frame::new(
            1,
            1_000,
            Tensor::from_vec(vec![1, 1, 1], vec![2.0]).unwrap(),
        )
        .unwrap(),
    ];
    let mut source = InMemoryFrameSource::new(frames);

    let a = source.next_frame().unwrap().unwrap();
    let b = source.next_frame().unwrap().unwrap();
    let c = source.next_frame().unwrap();

    assert_eq!(a.index(), 0);
    assert_eq!(b.index(), 1);
    assert!(c.is_none());
}

#[test]
fn frame_stream_respects_max_frames() {
    let frames = vec![
        Frame::new(0, 0, Tensor::from_vec(vec![1, 1, 1], vec![1.0]).unwrap()).unwrap(),
        Frame::new(
            1,
            1_000,
            Tensor::from_vec(vec![1, 1, 1], vec![2.0]).unwrap(),
        )
        .unwrap(),
    ];
    let source = InMemoryFrameSource::new(frames);
    let mut stream = FrameStream::new(source).with_max_frames(1);

    assert!(stream.try_next().unwrap().is_some());
    assert!(stream.try_next().unwrap().is_none());
}

#[derive(Debug)]
struct FailingSource;

impl FrameSource for FailingSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        Err(VideoError::Source("boom".to_string()))
    }
}

#[test]
fn frame_stream_propagates_source_errors() {
    let mut stream = FrameStream::new(FailingSource);
    let err = stream.try_next().unwrap_err();
    assert_eq!(err, VideoError::Source("boom".to_string()));
}

#[test]
fn camera_config_validation_rejects_invalid_values() {
    let err = CameraConfig {
        device_index: 0,
        width: 0,
        height: 480,
        fps: 30,
    }
    .validate()
    .unwrap_err();
    assert_eq!(
        err,
        VideoError::InvalidCameraResolution {
            width: 0,
            height: 480
        }
    );

    let err = CameraConfig {
        device_index: 0,
        width: 640,
        height: 480,
        fps: 0,
    }
    .validate()
    .unwrap_err();
    assert_eq!(err, VideoError::InvalidCameraFps { fps: 0 });
}

#[test]
fn rgb8_bytes_to_frame_normalizes_values() {
    let frame = rgb8_bytes_to_frame(7, 99, 1, 1, &[255, 128, 0]).unwrap();
    assert_eq!(frame.index(), 7);
    assert_eq!(frame.timestamp_us(), 99);
    assert_eq!(frame.pixel_format(), PixelFormat::RgbF32);
    let pixels = frame.image().data();
    assert!((pixels[0] - 1.0).abs() < 1e-6);
    assert!((pixels[1] - (128.0 / 255.0)).abs() < 1e-6);
    assert!((pixels[2] - 0.0).abs() < 1e-6);
}

#[test]
fn rgb8_bytes_to_frame_rejects_wrong_buffer_size() {
    let err = rgb8_bytes_to_frame(0, 0, 2, 2, &[1, 2, 3]).unwrap_err();
    assert_eq!(
        err,
        VideoError::RawFrameSizeMismatch {
            expected: 12,
            got: 3
        }
    );
}

#[test]
fn rgb8_frame_new_validates_raw_buffer_size() {
    let frame = Rgb8Frame::new(3, 44, 2, 1, vec![10, 20, 30, 40, 50, 60]).unwrap();
    assert_eq!(frame.index(), 3);
    assert_eq!(frame.timestamp_us(), 44);
    assert_eq!(frame.width(), 2);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.data(), &[10, 20, 30, 40, 50, 60]);

    let err = Rgb8Frame::new(0, 0, 2, 2, vec![1, 2, 3]).unwrap_err();
    assert_eq!(
        err,
        VideoError::RawFrameSizeMismatch {
            expected: 12,
            got: 3
        }
    );
}

#[test]
fn rgb8_frame_from_bytes_roundtrip() {
    let bytes = Bytes::from_static(&[1, 2, 3, 4, 5, 6]);
    let frame = Rgb8Frame::from_bytes(8, 99, 2, 1, bytes.clone()).unwrap();
    assert_eq!(frame.index(), 8);
    assert_eq!(frame.timestamp_us(), 99);
    assert_eq!(frame.width(), 2);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.data(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(frame.into_bytes(), bytes);
}

#[test]
fn rgb8_bytes_to_frame_handles_vectorized_and_tail_segments() {
    let bytes = (0u8..18u8).collect::<Vec<_>>();
    let frame = rgb8_bytes_to_frame(9, 123, 3, 2, &bytes).unwrap();
    let data = frame.image().data();
    assert_eq!(data.len(), bytes.len());
    for (index, raw) in bytes.iter().copied().enumerate() {
        let expected = raw as f32 / 255.0;
        assert!((data[index] - expected).abs() < 1e-6);
    }
}

#[test]
fn normalize_rgb8_to_f32_inplace_rejects_buffer_size_mismatch() {
    let mut out = vec![0.0f32; 2];
    let err = normalize_rgb8_to_f32_inplace(&[1, 2, 3], &mut out).unwrap_err();
    assert_eq!(
        err,
        VideoError::NormalizedBufferSizeMismatch {
            expected: 3,
            got: 2,
        }
    );
}

#[test]
fn normalize_rgb8_to_f32_inplace_overwrites_reused_buffer() {
    let mut out = vec![0.0f32; 6];

    normalize_rgb8_to_f32_inplace(&[0, 64, 128, 192, 255, 32], &mut out).unwrap();
    let first = out.clone();
    assert!((first[0] - 0.0).abs() < 1e-6);
    assert!((first[1] - (64.0 / 255.0)).abs() < 1e-6);
    assert!((first[4] - 1.0).abs() < 1e-6);

    normalize_rgb8_to_f32_inplace(&[255, 0, 255, 0, 255, 0], &mut out).unwrap();
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);
    assert!((out[2] - 1.0).abs() < 1e-6);
    assert!((out[3] - 0.0).abs() < 1e-6);
    assert!((out[4] - 1.0).abs() < 1e-6);
    assert!((out[5] - 0.0).abs() < 1e-6);
}

#[test]
fn select_camera_device_prefers_exact_match() {
    let devices = vec![
        CameraDeviceInfo {
            index: 0,
            label: "Laptop Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 1,
            label: "USB Camera".to_string(),
        },
    ];
    let device = select_camera_device(&devices, "usb camera").unwrap();
    assert_eq!(device.index, 1);
}

#[test]
fn select_camera_device_supports_substring_match() {
    let devices = vec![
        CameraDeviceInfo {
            index: 2,
            label: "Front Sensor".to_string(),
        },
        CameraDeviceInfo {
            index: 3,
            label: "Studio Camera".to_string(),
        },
    ];
    let device = select_camera_device(&devices, "studio").unwrap();
    assert_eq!(device.index, 3);
}

#[test]
fn select_camera_device_rejects_ambiguous_query() {
    let devices = vec![
        CameraDeviceInfo {
            index: 2,
            label: "Front Sensor".to_string(),
        },
        CameraDeviceInfo {
            index: 3,
            label: "Studio Sensor".to_string(),
        },
    ];
    let err = select_camera_device(&devices, "sensor").unwrap_err();
    assert_eq!(
        err,
        VideoError::CameraDeviceAmbiguous {
            query: "sensor".to_string(),
            matches: vec![
                "2: Front Sensor".to_string(),
                "3: Studio Sensor".to_string(),
            ],
        }
    );
}

#[test]
fn select_camera_device_rejects_unknown_query() {
    let devices = vec![CameraDeviceInfo {
        index: 0,
        label: "Laptop Camera".to_string(),
    }];
    let err = select_camera_device(&devices, "external").unwrap_err();
    assert_eq!(
        err,
        VideoError::CameraDeviceNotFound {
            query: "external".to_string()
        }
    );
}

#[test]
fn select_camera_device_rejects_empty_query() {
    let devices = vec![CameraDeviceInfo {
        index: 0,
        label: "Laptop Camera".to_string(),
    }];
    let err = select_camera_device(&devices, "   ").unwrap_err();
    assert_eq!(
        err,
        VideoError::InvalidCameraDeviceQuery {
            query: "   ".to_string()
        }
    );
}

#[test]
fn select_camera_device_supports_numeric_query_by_index() {
    let devices = vec![
        CameraDeviceInfo {
            index: 3,
            label: "Studio Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
    ];
    let device = select_camera_device(&devices, "7").unwrap();
    assert_eq!(device.index, 7);
    assert_eq!(device.label, "USB Camera");
}

#[test]
fn filter_camera_devices_supports_substring_query() {
    let devices = vec![
        CameraDeviceInfo {
            index: 0,
            label: "Laptop Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 2,
            label: "Studio Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 3,
            label: "USB Capture".to_string(),
        },
    ];
    let matches = filter_camera_devices(&devices, "studio").unwrap();
    assert_eq!(
        matches,
        vec![CameraDeviceInfo {
            index: 2,
            label: "Studio Camera".to_string(),
        }]
    );
}

#[test]
fn filter_camera_devices_supports_numeric_query() {
    let devices = vec![
        CameraDeviceInfo {
            index: 5,
            label: "Front Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
    ];
    let matches = filter_camera_devices(&devices, "7").unwrap();
    assert_eq!(
        matches,
        vec![CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        }]
    );
}

#[test]
fn filter_camera_devices_rejects_empty_query() {
    let devices = vec![CameraDeviceInfo {
        index: 0,
        label: "Laptop Camera".to_string(),
    }];
    let err = filter_camera_devices(&devices, "   ").unwrap_err();
    assert_eq!(
        err,
        VideoError::InvalidCameraDeviceQuery {
            query: "   ".to_string(),
        }
    );
}

#[test]
fn filter_camera_devices_returns_sorted_unique_matches() {
    let devices = vec![
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 2,
            label: "Front Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
    ];
    let matches = filter_camera_devices(&devices, "camera").unwrap();
    assert_eq!(
        matches,
        vec![
            CameraDeviceInfo {
                index: 2,
                label: "Front Camera".to_string(),
            },
            CameraDeviceInfo {
                index: 7,
                label: "USB Camera".to_string(),
            },
        ]
    );
}

#[cfg(not(feature = "native-camera"))]
#[test]
fn camera_source_returns_disabled_error_without_feature() {
    let err = CameraFrameSource::open(CameraConfig::default()).unwrap_err();
    assert_eq!(err, VideoError::CameraBackendDisabled);

    let mut source = CameraFrameSource;
    let err = source.next_frame().unwrap_err();
    assert_eq!(err, VideoError::CameraBackendDisabled);
}

// ── Raw video I/O tests ────────────────────────────────────────────

use super::video_io::{RawVideoReader, RawVideoWriter};

#[test]
fn raw_video_roundtrip() {
    let dir = std::env::temp_dir().join("yscv_test_video_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.rcv");

    let mut writer = RawVideoWriter::new(2, 2, 30.0);
    let frame_data = vec![255u8; 12];
    writer.push_frame(&frame_data).unwrap();
    writer.push_frame(&frame_data).unwrap();
    writer.save(&path).unwrap();
    assert_eq!(writer.frame_count(), 2);

    let mut reader = RawVideoReader::open(&path).unwrap();
    assert_eq!(reader.frame_count(), 2);
    assert_eq!(reader.meta.width, 2);
    assert_eq!(reader.meta.height, 2);

    let f0 = reader.next_frame().unwrap();
    assert_eq!(f0.width(), 2);
    assert_eq!(f0.height(), 2);

    let f1 = reader.next_frame().unwrap();
    assert_eq!(f1.width(), 2);

    assert!(reader.next_frame().is_none());

    reader.seek_start();
    assert!(reader.next_frame().is_some());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn raw_video_writer_rejects_wrong_size() {
    let mut writer = RawVideoWriter::new(2, 2, 30.0);
    let result = writer.push_frame(&[0u8; 11]);
    assert!(result.is_err());
}

#[test]
fn annex_b_parse_extracts_nal_units() {
    use super::codec::{NalUnitType, parse_annex_b};
    let mut stream = Vec::new();
    // SPS NAL
    stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    stream.push(0x67); // forbidden=0, ref_idc=3, type=7 (SPS)
    stream.extend_from_slice(&[0x42, 0x00, 0x1e]); // payload
    // PPS NAL
    stream.extend_from_slice(&[0x00, 0x00, 0x01]);
    stream.push(0x68); // forbidden=0, ref_idc=3, type=8 (PPS)
    stream.extend_from_slice(&[0xce, 0x38, 0x80]);
    // IDR NAL
    stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    stream.push(0x65); // forbidden=0, ref_idc=3, type=5 (IDR)
    stream.extend_from_slice(&[0x88, 0x84, 0x21]);

    let nals = parse_annex_b(&stream);
    assert_eq!(nals.len(), 3);
    assert_eq!(nals[0].nal_type, NalUnitType::Sps);
    assert_eq!(nals[1].nal_type, NalUnitType::Pps);
    assert_eq!(nals[2].nal_type, NalUnitType::Idr);
    assert!(nals[2].nal_type.is_vcl());
    assert!(!nals[0].nal_type.is_vcl());
}

#[test]
fn extract_sps_pps_from_nals() {
    use super::codec::{NalUnitType, extract_parameter_sets, parse_annex_b};
    let mut stream = Vec::new();
    stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x42]);
    stream.extend_from_slice(&[0x00, 0x00, 0x01, 0x68, 0xce]);
    let nals = parse_annex_b(&stream);
    let (sps, pps) = extract_parameter_sets(&nals);
    assert!(sps.is_some());
    assert_eq!(sps.unwrap().nal_type, NalUnitType::Sps);
    assert!(pps.is_some());
    assert_eq!(pps.unwrap().nal_type, NalUnitType::Pps);
}

#[test]
fn mp4_box_parse_basic() {
    use super::codec::parse_mp4_boxes;
    let mut data = Vec::new();
    // ftyp box: size=12, type=ftyp, payload=[0x00; 4]
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0C]); // size=12
    data.extend_from_slice(b"ftyp");
    data.extend_from_slice(&[0x00; 4]); // payload

    // moov box: size=8 (empty)
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x08]);
    data.extend_from_slice(b"moov");

    let boxes = parse_mp4_boxes(&data).unwrap();
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].type_str(), "ftyp");
    assert_eq!(boxes[0].size, 12);
    assert_eq!(boxes[1].type_str(), "moov");
    assert_eq!(boxes[1].size, 8);
}

#[test]
fn extract_avcc_nals_parses_sps_pps() {
    // Build a synthetic moov chunk containing an avcC box
    let mut moov = Vec::new();

    // Some padding (like other boxes before avcC)
    moov.extend_from_slice(&[0x00; 20]);

    // "avcC" tag
    moov.extend_from_slice(b"avcC");

    // avcC config: version(1) + profile(66=baseline) + compat(0xC0) + level(30) + lengthSizeMinusOne(0xFF = 3+0xFC)
    moov.push(1); // configurationVersion
    moov.push(66); // AVCProfileIndication (Baseline)
    moov.push(0xC0); // profile_compatibility
    moov.push(30); // AVCLevelIndication (3.0)
    moov.push(0xFF); // lengthSizeMinusOne = 3 (lower 2 bits) + reserved bits

    // numSPS = 1 (lower 5 bits of 0xE1 = 1)
    moov.push(0xE1);

    // SPS: length=4, data=[0x67, 0x42, 0xC0, 0x1E] (typical SPS NAL header 0x67 = nal_type=7 SPS)
    let sps_data = [0x67, 0x42, 0xC0, 0x1E];
    moov.extend_from_slice(&(sps_data.len() as u16).to_be_bytes());
    moov.extend_from_slice(&sps_data);

    // numPPS = 1
    moov.push(1);

    // PPS: length=3, data=[0x68, 0xCE, 0x38] (typical PPS NAL header 0x68 = nal_type=8 PPS)
    let pps_data = [0x68, 0xCE, 0x38];
    moov.extend_from_slice(&(pps_data.len() as u16).to_be_bytes());
    moov.extend_from_slice(&pps_data);

    let nals = super::video_io::extract_avcc_nals(&moov);
    assert_eq!(nals.len(), 2, "should extract 1 SPS + 1 PPS");

    assert_eq!(nals[0].nal_type, super::codec::NalUnitType::Sps);
    assert_eq!(nals[0].data, sps_data);

    assert_eq!(nals[1].nal_type, super::codec::NalUnitType::Pps);
    assert_eq!(nals[1].data, pps_data);
}

#[test]
fn extract_avcc_nals_empty_on_no_avcc() {
    let moov = vec![0x00; 100]; // no avcC tag
    let nals = super::video_io::extract_avcc_nals(&moov);
    assert!(nals.is_empty());
}

#[test]
fn mp4_h264_decode_real_file() {
    // Integration test: decode a real H.264 MP4 if the test file exists
    let path = std::path::Path::new("/tmp/test_h264.mp4");
    if !path.exists() {
        // Skip test if file not present (created by ffmpeg in dev environment)
        return;
    }
    let mut reader =
        super::video_io::Mp4VideoReader::open(path).expect("should open H.264 MP4 without error");
    let nal_count = reader.nal_count();
    assert!(nal_count > 0, "should find NAL units, got 0");

    // Try to decode frames, collecting results
    let mut frames = Vec::new();
    let mut errors = Vec::new();
    for _ in 0..nal_count {
        match reader.next_frame() {
            Ok(Some(f)) => frames.push(f),
            Ok(None) => break,
            Err(e) => errors.push(format!("{e}")),
        }
    }

    // Debug: check what NAL types we got
    reader.seek_start();
    // We can't inspect NAL types directly but we know: 3 NALs = SPS + PPS + 1 video NAL
    // If no frames decoded, the video NAL is probably non-IDR Slice type

    assert!(
        !frames.is_empty(),
        "should decode at least one frame from {nal_count} NALs. Errors: {errors:?}"
    );
    let frame = &frames[0];

    assert!(frame.width > 0 && frame.height > 0, "valid dimensions");
    assert_eq!(frame.rgb8_data.len(), frame.width * frame.height * 3);

    // Verify it's not all-gray (which would mean decode failure)
    let min = frame.rgb8_data.iter().copied().min().unwrap_or(0);
    let max = frame.rgb8_data.iter().copied().max().unwrap_or(0);
    assert!(
        max > min,
        "frame should not be uniform gray — actual min={min} max={max}"
    );
}

#[test]
fn mp4_h264_high_profile_decode() {
    // Test H.264 High profile (CABAC) MP4
    let path = std::path::Path::new("/tmp/test_h264_high.mp4");
    if !path.exists() {
        return;
    }
    let mut reader =
        super::video_io::Mp4VideoReader::open(path).expect("should open H.264 High profile MP4");
    assert!(reader.nal_count() > 0);

    match reader.next_frame() {
        Ok(Some(frame)) => {
            assert!(frame.width > 0 && frame.height > 0);
            let min = frame.rgb8_data.iter().copied().min().unwrap_or(0);
            let max = frame.rgb8_data.iter().copied().max().unwrap_or(0);
            assert!(
                max > min,
                "CABAC frame should not be uniform — min={min} max={max}"
            );
        }
        Ok(None) => panic!("no frame decoded from High profile MP4"),
        Err(e) => panic!("decode error: {e}"),
    }
}

#[test]
fn video_codec_enum_basics() {
    use super::codec::VideoCodec;
    assert_eq!(VideoCodec::H264, VideoCodec::H264);
    assert_ne!(VideoCodec::H264, VideoCodec::H265);
}
