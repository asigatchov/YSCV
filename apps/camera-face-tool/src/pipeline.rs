use std::time::{Duration, Instant};

use serde_json::{Value, json};
use yscv_detect::{
    BoundingBox, CLASS_ID_FACE, Rgb8FaceDetectScratch, detect_faces_from_rgb8_with_scratch,
};
use yscv_recognize::Recognizer;
use yscv_track::{TrackedDetection, Tracker, TrackerConfig};
use yscv_video::{CameraConfig, CameraFrameSource, resolve_camera_device};

use crate::config::{FaceAppConfig, FaceAppError};
use crate::error::AppError;
use crate::event_log::JsonlEventWriter;
use crate::util::duration_to_ms;

pub(crate) fn run_pipeline(config: &FaceAppConfig) -> Result<(), AppError> {
    if !cfg!(feature = "native-camera") {
        return Err(FaceAppError::Message(
            "camera support is disabled; rebuild with `--features native-camera`".to_string(),
        )
        .into());
    }

    let mut recognizer = if let Some(path) = config.identities_path.as_deref() {
        Recognizer::load_json_file(path)?
    } else {
        Recognizer::new(config.recognition_threshold)?
    };
    recognizer.set_threshold(config.recognition_threshold)?;
    let mut event_log = if let Some(path) = config.event_log_path.as_deref() {
        Some(JsonlEventWriter::create(path)?)
    } else {
        None
    };

    let device_index = if let Some(query) = config.device_name_query.as_deref() {
        let selected = resolve_camera_device(query)?;
        println!(
            "camera device resolved: query=`{}` -> {}: {}",
            query, selected.index, selected.label
        );
        selected.index
    } else {
        config.device_index
    };
    let mut source = CameraFrameSource::open(CameraConfig {
        device_index,
        width: config.width,
        height: config.height,
        fps: config.fps,
    })?;
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: config.track_iou_threshold,
        max_missed_frames: config.track_max_missed_frames,
        max_tracks: config.track_max_tracks,
    })?;

    println!(
        "camera-face-app: capture started (device={}, {}x{}@{}fps)",
        device_index, config.width, config.height, config.fps
    );

    let mut frames_seen = 0usize;
    let mut tracked: Vec<TrackedDetection> = Vec::new();
    let mut tracked_event_records: Vec<Value> = Vec::new();
    let mut face_scratch = Rgb8FaceDetectScratch::default();
    while let Some(frame) = source.next_rgb8_frame()? {
        if let Some(max_frames) = config.max_frames
            && frames_seen >= max_frames
        {
            break;
        }
        frames_seen += 1;

        let frame_start = Instant::now();
        let frame_height = frame.height();
        let frame_width = frame.width();
        let min_area = config
            .detect_min_area_override
            .unwrap_or_else(|| face_min_area(frame_width, frame_height));

        let detect_start = Instant::now();
        let detections = detect_faces_from_rgb8_with_scratch(
            (frame_width, frame_height),
            frame.data(),
            config.detect_score_threshold,
            min_area,
            config.detect_iou_threshold,
            config.detect_max_detections,
            &mut face_scratch,
        )?;
        let detect_duration = detect_start.elapsed();

        let track_start = Instant::now();
        tracker.update_into(&detections, &mut tracked);
        let track_duration = track_start.elapsed();
        let tracked_faces = tracker.count_by_class(CLASS_ID_FACE);

        println!(
            "frame={} ts_us={} faces={} tracked_faces={}",
            frame.index(),
            frame.timestamp_us(),
            detections.len(),
            tracked_faces
        );

        let mut recognize_duration = Duration::ZERO;
        let collect_event_records = event_log.is_some();
        if collect_event_records {
            tracked_event_records.clear();
            if tracked_event_records.capacity() < tracked.len() {
                tracked_event_records.reserve(tracked.len() - tracked_event_records.capacity());
            }
        }
        for (idx, item) in tracked.iter().enumerate() {
            let recognize_start = Instant::now();
            let embedding =
                embedding_from_bbox(item.detection.bbox, frame_width as f32, frame_height as f32);
            let recognition = recognizer.recognize_slice(&embedding)?;
            recognize_duration += recognize_start.elapsed();
            let identity_label = recognition.identity.as_deref().unwrap_or("unknown");
            println!(
                "  face#{idx} track_id={} score={:.3} identity={} sim={:.3} bbox=({:.1},{:.1},{:.1},{:.1})",
                item.track_id,
                item.detection.score,
                identity_label,
                recognition.score,
                item.detection.bbox.x1,
                item.detection.bbox.y1,
                item.detection.bbox.x2,
                item.detection.bbox.y2,
            );
            if collect_event_records {
                tracked_event_records.push(json!({
                    "det_index": idx,
                    "track_id": item.track_id,
                    "class_id": item.detection.class_id,
                    "score": item.detection.score,
                    "identity": recognition.identity.clone(),
                    "similarity": recognition.score,
                    "bbox": {
                        "x1": item.detection.bbox.x1,
                        "y1": item.detection.bbox.y1,
                        "x2": item.detection.bbox.x2,
                        "y2": item.detection.bbox.y2,
                    },
                }));
            }
        }

        let end_to_end_duration = frame_start.elapsed();
        if let Some(writer) = event_log.as_mut() {
            writer.write_record(&json!({
                "frame_index": frame.index(),
                "timestamp_us": frame.timestamp_us(),
                "mode": "camera",
                "detect_target": "face",
                "detection_count": detections.len(),
                "tracked_target_count": tracked_faces,
                "timings_ms": {
                    "detect": duration_to_ms(detect_duration),
                    "track": duration_to_ms(track_duration),
                    "recognize": duration_to_ms(recognize_duration),
                    "end_to_end": duration_to_ms(end_to_end_duration),
                },
                "tracked": &tracked_event_records,
            }))?;
        }
    }

    if let Some(mut writer) = event_log {
        writer.flush()?;
        println!("event log saved to {}", writer.path().display());
    }
    println!("camera-face-app: completed");
    Ok(())
}

fn face_min_area(frame_width: usize, frame_height: usize) -> usize {
    let frame_area = frame_width.saturating_mul(frame_height);
    ((frame_area as f32 * 0.003).round() as usize).max(4)
}

fn embedding_from_bbox(bbox: BoundingBox, frame_width: f32, frame_height: f32) -> [f32; 3] {
    let cx = ((bbox.x1 + bbox.x2) * 0.5) / frame_width;
    let cy = ((bbox.y1 + bbox.y2) * 0.5) / frame_height;
    let area = bbox.area() / (frame_width * frame_height);
    [cx, cy, area]
}
