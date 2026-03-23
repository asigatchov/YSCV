use std::env;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaceAppConfig {
    pub(crate) list_cameras: bool,
    pub(crate) diagnose_camera: bool,
    pub(crate) diagnose_frames: usize,
    pub(crate) diagnose_report_path: Option<PathBuf>,
    pub(crate) device_index: u32,
    pub(crate) device_name_query: Option<String>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) fps: u32,
    pub(crate) max_frames: Option<usize>,
    pub(crate) identities_path: Option<PathBuf>,
    pub(crate) event_log_path: Option<PathBuf>,
    pub(crate) recognition_threshold: f32,
    pub(crate) detect_score_threshold: f32,
    pub(crate) detect_iou_threshold: f32,
    pub(crate) detect_max_detections: usize,
    pub(crate) detect_min_area_override: Option<usize>,
    pub(crate) track_iou_threshold: f32,
    pub(crate) track_max_missed_frames: u32,
    pub(crate) track_max_tracks: usize,
}

impl Default for FaceAppConfig {
    fn default() -> Self {
        Self {
            list_cameras: false,
            diagnose_camera: false,
            diagnose_frames: 30,
            diagnose_report_path: None,
            device_index: 0,
            device_name_query: None,
            width: 1280,
            height: 720,
            fps: 30,
            max_frames: None,
            identities_path: None,
            event_log_path: None,
            recognition_threshold: 0.9,
            detect_score_threshold: 0.35,
            detect_iou_threshold: 0.4,
            detect_max_detections: 32,
            detect_min_area_override: None,
            track_iou_threshold: 0.25,
            track_max_missed_frames: 3,
            track_max_tracks: 128,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum FaceAppError {
    #[error("help requested")]
    HelpRequested,
    #[error("{0}")]
    Message(String),
}

impl FaceAppConfig {
    pub(crate) fn from_env() -> Result<Self, FaceAppError> {
        Self::parse_from(env::args().skip(1))
    }

    pub(crate) fn parse_from<I>(args: I) -> Result<Self, FaceAppError>
    where
        I: IntoIterator<Item = String>,
    {
        let args = args.into_iter().collect::<Vec<_>>();
        let mut config = Self::default();
        let mut index = 0usize;
        let mut has_device_index = false;
        let mut has_device_name = false;

        while index < args.len() {
            match args[index].as_str() {
                "--list-cameras" => {
                    config.list_cameras = true;
                }
                "--diagnose-camera" => {
                    config.diagnose_camera = true;
                }
                "--diagnose-frames" => {
                    let raw = next_value(&args, &mut index, "--diagnose-frames")?;
                    config.diagnose_frames = parse_usize("--diagnose-frames", raw)?;
                }
                "--diagnose-report" => {
                    let raw = next_value(&args, &mut index, "--diagnose-report")?;
                    config.diagnose_report_path = Some(parse_path(raw));
                }
                "--device" => {
                    if has_device_name {
                        return Err(FaceAppError::Message(
                            "`--device` cannot be used together with `--device-name`".to_string(),
                        ));
                    }
                    let raw = next_value(&args, &mut index, "--device")?;
                    config.device_index = parse_u32("--device", raw)?;
                    has_device_index = true;
                }
                "--device-name" => {
                    if has_device_index {
                        return Err(FaceAppError::Message(
                            "`--device-name` cannot be used together with `--device`".to_string(),
                        ));
                    }
                    let raw = next_value(&args, &mut index, "--device-name")?;
                    config.device_name_query = Some(parse_non_empty("--device-name", raw)?);
                    has_device_name = true;
                }
                "--width" => {
                    let raw = next_value(&args, &mut index, "--width")?;
                    config.width = parse_u32("--width", raw)?;
                }
                "--height" => {
                    let raw = next_value(&args, &mut index, "--height")?;
                    config.height = parse_u32("--height", raw)?;
                }
                "--fps" => {
                    let raw = next_value(&args, &mut index, "--fps")?;
                    config.fps = parse_u32("--fps", raw)?;
                }
                "--max-frames" => {
                    let raw = next_value(&args, &mut index, "--max-frames")?;
                    config.max_frames = Some(parse_usize("--max-frames", raw)?);
                }
                "--identities" => {
                    let raw = next_value(&args, &mut index, "--identities")?;
                    config.identities_path = Some(parse_path(raw));
                }
                "--event-log" => {
                    let raw = next_value(&args, &mut index, "--event-log")?;
                    config.event_log_path = Some(parse_path(raw));
                }
                "--recognition-threshold" => {
                    let raw = next_value(&args, &mut index, "--recognition-threshold")?;
                    config.recognition_threshold = parse_f32("--recognition-threshold", raw)?;
                }
                "--detect-score" => {
                    let raw = next_value(&args, &mut index, "--detect-score")?;
                    config.detect_score_threshold = parse_f32("--detect-score", raw)?;
                }
                "--detect-min-area" => {
                    let raw = next_value(&args, &mut index, "--detect-min-area")?;
                    config.detect_min_area_override = Some(parse_usize("--detect-min-area", raw)?);
                }
                "--detect-iou" => {
                    let raw = next_value(&args, &mut index, "--detect-iou")?;
                    config.detect_iou_threshold = parse_f32("--detect-iou", raw)?;
                }
                "--detect-max" => {
                    let raw = next_value(&args, &mut index, "--detect-max")?;
                    config.detect_max_detections = parse_usize("--detect-max", raw)?;
                }
                "--track-iou" => {
                    let raw = next_value(&args, &mut index, "--track-iou")?;
                    config.track_iou_threshold = parse_f32("--track-iou", raw)?;
                }
                "--track-max-missed" => {
                    let raw = next_value(&args, &mut index, "--track-max-missed")?;
                    config.track_max_missed_frames = parse_u32("--track-max-missed", raw)?;
                }
                "--track-max" => {
                    let raw = next_value(&args, &mut index, "--track-max")?;
                    config.track_max_tracks = parse_usize("--track-max", raw)?;
                }
                "--help" | "-h" => return Err(FaceAppError::HelpRequested),
                unknown => {
                    return Err(FaceAppError::Message(format!(
                        "unknown argument: {unknown}; run with --help for usage"
                    )));
                }
            }
            index += 1;
        }

        validate_positive("--width", config.width)?;
        validate_positive("--height", config.height)?;
        validate_positive("--fps", config.fps)?;
        validate_positive_usize("--diagnose-frames", config.diagnose_frames)?;
        validate_positive_usize("--detect-max", config.detect_max_detections)?;
        validate_positive_usize("--track-max", config.track_max_tracks)?;
        validate_in_unit_interval("--detect-score", config.detect_score_threshold)?;
        validate_in_unit_interval("--detect-iou", config.detect_iou_threshold)?;
        validate_in_unit_interval("--track-iou", config.track_iou_threshold)?;
        validate_in_closed_range(
            "--recognition-threshold",
            config.recognition_threshold,
            -1.0,
            1.0,
        )?;
        if let Some(max_frames) = config.max_frames
            && max_frames == 0
        {
            return Err(FaceAppError::Message(
                "`--max-frames` must be greater than 0".to_string(),
            ));
        }
        if let Some(min_area) = config.detect_min_area_override {
            validate_positive_usize("--detect-min-area", min_area)?;
        }
        if config.event_log_path.is_some() && config.list_cameras {
            return Err(FaceAppError::Message(
                "`--event-log` cannot be used together with `--list-cameras`".to_string(),
            ));
        }
        if config.diagnose_camera && config.list_cameras {
            return Err(FaceAppError::Message(
                "`--diagnose-camera` cannot be used together with `--list-cameras`".to_string(),
            ));
        }
        if config.diagnose_camera && config.event_log_path.is_some() {
            return Err(FaceAppError::Message(
                "`--diagnose-camera` cannot be used together with `--event-log`".to_string(),
            ));
        }
        if config.diagnose_report_path.is_some() && !config.diagnose_camera {
            return Err(FaceAppError::Message(
                "`--diagnose-report` requires `--diagnose-camera`".to_string(),
            ));
        }
        Ok(config)
    }
}

pub(crate) fn print_usage() {
    println!("camera-face-app usage:");
    println!("  cargo run -p camera-face-tool --features native-camera -- [options]");
    println!();
    println!("options:");
    println!("  --list-cameras           list available camera devices and exit");
    println!("  --diagnose-camera        run camera environment/capture diagnostics and exit");
    println!(
        "  --diagnose-frames <n>    number of frames to sample in diagnostics mode (default: 30)"
    );
    println!("  --diagnose-report <path> write diagnostics summary JSON report");
    println!("  --device <index>         camera device index (default: 0)");
    println!(
        "  --device-name <query>    camera device query by label substring (also filters --list-cameras)"
    );
    println!("  --width <pixels>         camera frame width (default: 1280)");
    println!("  --height <pixels>        camera frame height (default: 720)");
    println!("  --fps <value>            camera target FPS (default: 30)");
    println!("  --max-frames <count>     stop after N frames");
    println!("  --identities <path>      load recognizer identities JSON");
    println!("  --event-log <path>       write per-frame JSONL events for downstream tooling");
    println!("  --recognition-threshold <value> threshold in [-1, 1] (default: 0.9)");
    println!("  --detect-score <value>   face detector threshold in [0, 1] (default: 0.35)");
    println!("  --detect-min-area <n>    override minimum face blob area in pixels");
    println!("  --detect-iou <value>     face NMS IoU threshold in [0, 1] (default: 0.4)");
    println!("  --detect-max <n>         max detections per frame (default: 32)");
    println!("  --track-iou <value>      tracker IoU match threshold in [0, 1] (default: 0.25)");
    println!("  --track-max-missed <n>   tracker missed-frame budget (default: 3)");
    println!("  --track-max <n>          max simultaneous tracks (default: 128)");
    println!("  -h, --help               print this help");
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, FaceAppError> {
    *index += 1;
    if let Some(value) = args.get(*index) {
        Ok(value.clone())
    } else {
        Err(FaceAppError::Message(format!(
            "missing value for {flag}; run with --help for usage"
        )))
    }
}

fn parse_u32(flag: &str, raw: String) -> Result<u32, FaceAppError> {
    raw.parse::<u32>().map_err(|_| {
        FaceAppError::Message(format!(
            "failed to parse {flag} value `{raw}` as unsigned integer"
        ))
    })
}

fn parse_usize(flag: &str, raw: String) -> Result<usize, FaceAppError> {
    raw.parse::<usize>().map_err(|_| {
        FaceAppError::Message(format!(
            "failed to parse {flag} value `{raw}` as unsigned integer"
        ))
    })
}

fn parse_f32(flag: &str, raw: String) -> Result<f32, FaceAppError> {
    raw.parse::<f32>().map_err(|_| {
        FaceAppError::Message(format!("failed to parse {flag} value `{raw}` as number"))
    })
}

fn parse_non_empty(flag: &str, raw: String) -> Result<String, FaceAppError> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(FaceAppError::Message(format!("{flag} must not be empty")));
    }
    Ok(normalized.to_string())
}

fn parse_path(raw: String) -> PathBuf {
    PathBuf::from(raw)
}

fn validate_positive(flag: &str, value: u32) -> Result<(), FaceAppError> {
    if value == 0 {
        return Err(FaceAppError::Message(format!(
            "{flag} must be greater than 0"
        )));
    }
    Ok(())
}

fn validate_positive_usize(flag: &str, value: usize) -> Result<(), FaceAppError> {
    if value == 0 {
        return Err(FaceAppError::Message(format!(
            "{flag} must be greater than 0"
        )));
    }
    Ok(())
}

fn validate_in_unit_interval(flag: &str, value: f32) -> Result<(), FaceAppError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(FaceAppError::Message(format!(
            "{flag} must be finite and in [0, 1]"
        )));
    }
    Ok(())
}

fn validate_in_closed_range(
    flag: &str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), FaceAppError> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(FaceAppError::Message(format!(
            "{flag} must be finite and in [{min}, {max}]"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FaceAppConfig, FaceAppError};

    #[test]
    fn parse_defaults_without_args() {
        let config = FaceAppConfig::parse_from(Vec::<String>::new()).unwrap();
        assert_eq!(config, FaceAppConfig::default());
    }

    #[test]
    fn parse_list_cameras_flag() {
        let config = FaceAppConfig::parse_from(vec!["--list-cameras".to_string()]).unwrap();
        assert!(config.list_cameras);
        assert!(!config.diagnose_camera);
        assert_eq!(config.max_frames, None);
    }

    #[test]
    fn parse_diagnose_camera_flags() {
        let config = FaceAppConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--diagnose-frames".to_string(),
            "12".to_string(),
        ])
        .unwrap();
        assert!(config.diagnose_camera);
        assert_eq!(config.diagnose_frames, 12);
    }

    #[test]
    fn parse_diagnose_report_path() {
        let config = FaceAppConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--diagnose-report".to_string(),
            "diag.json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            config.diagnose_report_path,
            Some(std::path::PathBuf::from("diag.json"))
        );
    }

    #[test]
    fn parse_device_name_query() {
        let config =
            FaceAppConfig::parse_from(vec!["--device-name".to_string(), "Studio".to_string()])
                .unwrap();
        assert_eq!(config.device_name_query.as_deref(), Some("Studio"));
    }

    #[test]
    fn parse_rejects_conflicting_device_flags() {
        let err = FaceAppConfig::parse_from(vec![
            "--device".to_string(),
            "0".to_string(),
            "--device-name".to_string(),
            "cam".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            FaceAppError::Message(
                "`--device-name` cannot be used together with `--device`".to_string()
            )
        );
    }

    #[test]
    fn parse_list_cameras_with_device_name() {
        let config = FaceAppConfig::parse_from(vec![
            "--list-cameras".to_string(),
            "--device-name".to_string(),
            "cam".to_string(),
        ])
        .unwrap();
        assert!(config.list_cameras);
        assert_eq!(config.device_name_query.as_deref(), Some("cam"));
    }

    #[test]
    fn parse_event_log_path() {
        let config =
            FaceAppConfig::parse_from(vec!["--event-log".to_string(), "events.jsonl".to_string()])
                .unwrap();
        assert_eq!(
            config.event_log_path,
            Some(std::path::PathBuf::from("events.jsonl"))
        );
    }

    #[test]
    fn parse_rejects_event_log_with_list_cameras() {
        let err = FaceAppConfig::parse_from(vec![
            "--list-cameras".to_string(),
            "--event-log".to_string(),
            "events.jsonl".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            FaceAppError::Message(
                "`--event-log` cannot be used together with `--list-cameras`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_zero_diagnose_frames() {
        let err = FaceAppConfig::parse_from(vec!["--diagnose-frames".to_string(), "0".to_string()])
            .unwrap_err();
        assert_eq!(
            err,
            FaceAppError::Message("--diagnose-frames must be greater than 0".to_string())
        );
    }

    #[test]
    fn parse_rejects_diagnostics_with_list_cameras() {
        let err = FaceAppConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--list-cameras".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            FaceAppError::Message(
                "`--diagnose-camera` cannot be used together with `--list-cameras`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnostics_with_event_log() {
        let err = FaceAppConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--event-log".to_string(),
            "events.jsonl".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            FaceAppError::Message(
                "`--diagnose-camera` cannot be used together with `--event-log`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnose_report_without_diagnostics_mode() {
        let err = FaceAppConfig::parse_from(vec![
            "--diagnose-report".to_string(),
            "diag.json".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            FaceAppError::Message("`--diagnose-report` requires `--diagnose-camera`".to_string())
        );
    }
}
