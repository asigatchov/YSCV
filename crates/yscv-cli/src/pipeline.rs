use std::fs;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use yscv_detect::{
    BoundingBox, Detection, FrameFaceDetectScratch, FramePeopleDetectScratch,
    Rgb8FaceDetectScratch, detect_faces_from_frame_with_scratch,
    detect_faces_from_rgb8_with_scratch, detect_people_from_frame_with_scratch,
    detect_people_from_rgb8_with_scratch,
};
use yscv_eval::{
    PipelineDurations, parse_pipeline_benchmark_thresholds, summarize_pipeline_durations,
    validate_pipeline_benchmark_thresholds,
};
use yscv_recognize::Recognizer;
use yscv_track::{TrackedDetection, Tracker, TrackerConfig};
use yscv_video::FrameStream;

use crate::benchmark::{BenchmarkCollector, format_benchmark_report, format_benchmark_violations};
use crate::config::{CliConfig, CliError, DetectTarget, resolve_detect_config};
use crate::error::AppError;
use crate::event_log::JsonlEventWriter;
use crate::source::{build_source, open_camera_source};
use crate::util::{duration_to_ms, ensure_parent_dir};

#[derive(Debug, Clone, Copy)]
struct FrameMeta {
    index: u64,
    timestamp_us: u64,
    width: usize,
    height: usize,
}

struct FrameProcessContext<'a> {
    cli: &'a CliConfig,
    mode: &'a str,
    tracker: &'a mut Tracker,
    recognizer: &'a mut Recognizer,
    event_log: &'a mut Option<JsonlEventWriter>,
    benchmark: &'a mut Option<BenchmarkCollector>,
    tracked_detections: Vec<TrackedDetection>,
    tracked_event_records: Vec<Value>,
}

struct FrameProcessInput<'a> {
    frame: FrameMeta,
    detections: &'a [Detection],
    detect_duration: Duration,
    frame_start: Instant,
}

pub fn run_pipeline(cli: &CliConfig) -> Result<(), AppError> {
    let mut recognizer = if let Some(path) = cli.identities_path.as_deref() {
        Recognizer::load_json_file(path)?
    } else {
        Recognizer::new(cli.recognition_threshold)?
    };
    recognizer.set_threshold(cli.recognition_threshold)?;

    let mut event_log = if let Some(path) = cli.event_log_path.as_deref() {
        Some(JsonlEventWriter::create(path)?)
    } else {
        None
    };
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: cli.track_iou_threshold,
        max_missed_frames: cli.track_max_missed_frames,
        max_tracks: cli.track_max_tracks,
    })?;
    let mut benchmark = cli.benchmark.then(BenchmarkCollector::default);
    if cli.camera {
        println!(
            "yscv-cli demo: starting camera frame stream (raw rgb8 {} path)",
            cli.detect_target.as_str()
        );
        run_camera_rgb8_pipeline(
            cli,
            &mut recognizer,
            &mut tracker,
            &mut event_log,
            &mut benchmark,
        )?;
    } else {
        let mode = "deterministic";
        println!("yscv-cli demo: starting {mode} frame stream");
        run_standard_pipeline(
            cli,
            mode,
            &mut recognizer,
            &mut tracker,
            &mut event_log,
            &mut benchmark,
        )?;
    }

    finalize_benchmark(cli, benchmark)?;
    flush_event_log(event_log)?;
    println!("yscv-cli demo: stream completed");
    Ok(())
}

fn run_standard_pipeline(
    cli: &CliConfig,
    mode: &str,
    recognizer: &mut Recognizer,
    tracker: &mut Tracker,
    event_log: &mut Option<JsonlEventWriter>,
    benchmark: &mut Option<BenchmarkCollector>,
) -> Result<(), AppError> {
    let source = build_source(cli, recognizer)?;
    let mut stream = FrameStream::new(source);
    if let Some(max_frames) = cli.max_frames {
        stream = stream.with_max_frames(max_frames);
    }
    let mut process_context = FrameProcessContext {
        cli,
        mode,
        tracker,
        recognizer,
        event_log,
        benchmark,
        tracked_detections: Vec::new(),
        tracked_event_records: Vec::new(),
    };
    let mut people_scratch = FramePeopleDetectScratch::default();
    let mut face_scratch = FrameFaceDetectScratch::default();

    while let Some(frame) = stream.try_next()? {
        let frame_start = Instant::now();
        let frame_height = frame.image().shape()[0];
        let frame_width = frame.image().shape()[1];
        let detect_config = resolve_detect_config(cli, frame_width, frame_height);
        let detect_start = Instant::now();
        let detections = match cli.detect_target {
            DetectTarget::People => detect_people_from_frame_with_scratch(
                &frame,
                detect_config.score_threshold,
                detect_config.min_area,
                detect_config.iou_threshold,
                detect_config.max_detections,
                &mut people_scratch,
            )?,
            DetectTarget::Faces => detect_faces_from_frame_with_scratch(
                &frame,
                detect_config.score_threshold,
                detect_config.min_area,
                detect_config.iou_threshold,
                detect_config.max_detections,
                &mut face_scratch,
            )?,
        };
        let detect_duration = detect_start.elapsed();

        process_frame(
            &mut process_context,
            FrameProcessInput {
                frame: FrameMeta {
                    index: frame.index(),
                    timestamp_us: frame.timestamp_us(),
                    width: frame_width,
                    height: frame_height,
                },
                detections: &detections,
                detect_duration,
                frame_start,
            },
        )?;
    }

    Ok(())
}

fn run_camera_rgb8_pipeline(
    cli: &CliConfig,
    recognizer: &mut Recognizer,
    tracker: &mut Tracker,
    event_log: &mut Option<JsonlEventWriter>,
    benchmark: &mut Option<BenchmarkCollector>,
) -> Result<(), AppError> {
    let mut source = open_camera_source(cli)?;
    let max_frames = cli.max_frames.unwrap_or(usize::MAX);
    let mut frames_seen = 0usize;
    let mut people_scratch = yscv_detect::Rgb8PeopleDetectScratch::default();
    let mut face_scratch = Rgb8FaceDetectScratch::default();
    let mut process_context = FrameProcessContext {
        cli,
        mode: "camera",
        tracker,
        recognizer,
        event_log,
        benchmark,
        tracked_detections: Vec::new(),
        tracked_event_records: Vec::new(),
    };

    while frames_seen < max_frames {
        let Some(frame) = source.next_rgb8_frame()? else {
            break;
        };
        frames_seen += 1;

        let frame_start = Instant::now();
        let frame_width = frame.width();
        let frame_height = frame.height();
        let detect_config = resolve_detect_config(cli, frame_width, frame_height);
        let detect_start = Instant::now();
        let detections = match cli.detect_target {
            DetectTarget::People => detect_people_from_rgb8_with_scratch(
                (frame_width, frame_height),
                frame.data(),
                detect_config.score_threshold,
                detect_config.min_area,
                detect_config.iou_threshold,
                detect_config.max_detections,
                &mut people_scratch,
            )?,
            DetectTarget::Faces => detect_faces_from_rgb8_with_scratch(
                (frame_width, frame_height),
                frame.data(),
                detect_config.score_threshold,
                detect_config.min_area,
                detect_config.iou_threshold,
                detect_config.max_detections,
                &mut face_scratch,
            )?,
        };
        let detect_duration = detect_start.elapsed();

        process_frame(
            &mut process_context,
            FrameProcessInput {
                frame: FrameMeta {
                    index: frame.index(),
                    timestamp_us: frame.timestamp_us(),
                    width: frame_width,
                    height: frame_height,
                },
                detections: &detections,
                detect_duration,
                frame_start,
            },
        )?;
    }

    Ok(())
}

fn process_frame(
    context: &mut FrameProcessContext<'_>,
    input: FrameProcessInput<'_>,
) -> Result<(), AppError> {
    let FrameProcessInput {
        frame,
        detections,
        detect_duration,
        frame_start,
    } = input;
    let track_start = Instant::now();
    context
        .tracker
        .update_into(detections, &mut context.tracked_detections);
    let track_duration = track_start.elapsed();
    let tracked = context.tracked_detections.as_slice();

    let tracked_targets = context
        .tracker
        .count_by_class(context.cli.detect_target.class_id());
    println!(
        "frame={} ts_us={} detections={} tracked_{}={}",
        frame.index,
        frame.timestamp_us,
        detections.len(),
        context.cli.detect_target.count_label(),
        tracked_targets,
    );

    let mut recognize_duration = Duration::ZERO;
    let collect_event_records = context.event_log.is_some();
    if collect_event_records {
        context.tracked_event_records.clear();
        if context.tracked_event_records.capacity() < tracked.len() {
            context
                .tracked_event_records
                .reserve(tracked.len() - context.tracked_event_records.capacity());
        }
    }

    for (idx, item) in tracked.iter().enumerate() {
        let recognize_start = Instant::now();
        let embedding =
            bbox_embedding_components(item.detection.bbox, frame.width as f32, frame.height as f32);
        let recognition = context.recognizer.recognize_slice(&embedding)?;
        recognize_duration += recognize_start.elapsed();
        let identity_label = recognition.identity.as_deref().unwrap_or("unknown");
        println!(
            "  det#{idx} track_id={} score={:.3} identity={} sim={:.3} bbox=({:.1},{:.1},{:.1},{:.1})",
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
            context.tracked_event_records.push(json!({
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
    if let Some(writer) = context.event_log.as_mut() {
        writer.write_record(&json!({
            "frame_index": frame.index,
            "timestamp_us": frame.timestamp_us,
            "mode": context.mode,
            "detect_target": context.cli.detect_target.as_str(),
            "detection_count": detections.len(),
            "tracked_target_count": tracked_targets,
            "timings_ms": {
                "detect": duration_to_ms(detect_duration),
                "track": duration_to_ms(track_duration),
                "recognize": duration_to_ms(recognize_duration),
                "end_to_end": duration_to_ms(end_to_end_duration),
            },
            "tracked": &context.tracked_event_records,
        }))?;
    }
    if let Some(collector) = context.benchmark.as_mut() {
        collector.detect.push(detect_duration);
        collector.track.push(track_duration);
        collector.recognize.push(recognize_duration);
        collector.end_to_end.push(end_to_end_duration);
    }
    Ok(())
}

fn bbox_embedding_components(bbox: BoundingBox, frame_width: f32, frame_height: f32) -> [f32; 3] {
    let cx = ((bbox.x1 + bbox.x2) * 0.5) / frame_width;
    let cy = ((bbox.y1 + bbox.y2) * 0.5) / frame_height;
    let area = bbox.area() / (frame_width * frame_height);
    [cx, cy, area]
}

fn finalize_benchmark(
    cli: &CliConfig,
    benchmark: Option<BenchmarkCollector>,
) -> Result<(), AppError> {
    if let Some(collector) = benchmark {
        let report = summarize_pipeline_durations(PipelineDurations {
            detect: &collector.detect,
            track: &collector.track,
            recognize: &collector.recognize,
            end_to_end: &collector.end_to_end,
        })?;
        let text_report = format_benchmark_report(&report);
        println!("{text_report}");

        if let Some(path) = cli.benchmark_baseline_path.as_deref() {
            let baseline_text = fs::read_to_string(path)?;
            let thresholds = parse_pipeline_benchmark_thresholds(&baseline_text)?;
            let violations = validate_pipeline_benchmark_thresholds(&report, &thresholds);
            if violations.is_empty() {
                println!("benchmark baseline check passed ({})", path.display());
            } else {
                println!(
                    "benchmark baseline check failed ({}):\n{}",
                    path.display(),
                    format_benchmark_violations(&violations)
                );
                return Err(CliError::Message("benchmark regression detected".to_string()).into());
            }
        }
        if let Some(path) = cli.benchmark_report_path.as_deref() {
            ensure_parent_dir(path)?;
            fs::write(path, &text_report)?;
            println!("benchmark report saved to {}", path.display());
        }
    }
    Ok(())
}

fn flush_event_log(event_log: Option<JsonlEventWriter>) -> Result<(), AppError> {
    if let Some(mut writer) = event_log {
        writer.flush()?;
        println!("event log saved to {}", writer.path().display());
    }
    Ok(())
}
