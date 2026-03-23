use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::EvalError;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CameraDiagnosticsReport {
    pub tool: String,
    pub mode: String,
    pub status: String,
    pub requested: CameraDiagnosticsRequested,
    #[serde(default)]
    pub discovered_devices: Vec<CameraDiagnosticsDevice>,
    pub selected_device: Option<CameraDiagnosticsDevice>,
    pub capture: Option<CameraDiagnosticsCapture>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CameraDiagnosticsRequested {
    pub device_index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub diagnose_frames: usize,
    pub device_name_query: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CameraDiagnosticsDevice {
    pub index: u32,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CameraDiagnosticsCapture {
    pub requested_frames: usize,
    pub collected_frames: usize,
    pub wall_ms: f64,
    pub first_frame: Option<CameraDiagnosticsFirstFrame>,
    pub timing: Option<CameraDiagnosticsTiming>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CameraDiagnosticsFirstFrame {
    pub index: u64,
    pub timestamp_us: u64,
    pub shape: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CameraDiagnosticsTiming {
    pub target_fps: f64,
    pub wall_fps: f64,
    pub sensor_fps: f64,
    pub wall_drift_pct: f64,
    pub sensor_drift_pct: f64,
    pub mean_gap_us: f64,
    pub min_gap_us: u64,
    pub max_gap_us: u64,
    pub dropped_frames: u64,
    pub drift_warning: bool,
    pub dropped_frames_warning: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraDiagnosticsThresholds {
    pub min_collected_frames: usize,
    pub max_abs_wall_drift_pct: f64,
    pub max_abs_sensor_drift_pct: f64,
    pub max_dropped_frames: u64,
}

impl Default for CameraDiagnosticsThresholds {
    fn default() -> Self {
        Self {
            min_collected_frames: 2,
            max_abs_wall_drift_pct: 25.0,
            max_abs_sensor_drift_pct: 25.0,
            max_dropped_frames: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CameraDiagnosticsViolation {
    pub field: &'static str,
    pub message: String,
}

pub fn parse_camera_diagnostics_report_json(
    text: &str,
) -> Result<CameraDiagnosticsReport, EvalError> {
    serde_json::from_str(text).map_err(|err| EvalError::InvalidDiagnosticsReport {
        message: err.to_string(),
    })
}

pub fn load_camera_diagnostics_report_json_file(
    path: &Path,
) -> Result<CameraDiagnosticsReport, EvalError> {
    let body = fs::read_to_string(path).map_err(|err| EvalError::DiagnosticsReportIo {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    parse_camera_diagnostics_report_json(&body)
}

pub fn validate_camera_diagnostics_report(
    report: &CameraDiagnosticsReport,
    thresholds: CameraDiagnosticsThresholds,
) -> Vec<CameraDiagnosticsViolation> {
    let mut violations = Vec::new();
    if report.mode != "diagnostics" {
        violations.push(CameraDiagnosticsViolation {
            field: "mode",
            message: format!("expected `diagnostics`, got `{}`", report.mode),
        });
    }
    if report.status != "ok" {
        violations.push(CameraDiagnosticsViolation {
            field: "status",
            message: format!("expected `ok`, got `{}`", report.status),
        });
    }

    let Some(capture) = report.capture.as_ref() else {
        violations.push(CameraDiagnosticsViolation {
            field: "capture",
            message: "capture section is missing".to_string(),
        });
        return violations;
    };

    if capture.collected_frames < thresholds.min_collected_frames {
        violations.push(CameraDiagnosticsViolation {
            field: "capture.collected_frames",
            message: format!(
                "expected >= {}, got {}",
                thresholds.min_collected_frames, capture.collected_frames
            ),
        });
    }
    if !capture.wall_ms.is_finite() || capture.wall_ms < 0.0 {
        violations.push(CameraDiagnosticsViolation {
            field: "capture.wall_ms",
            message: format!(
                "expected finite non-negative value, got {}",
                capture.wall_ms
            ),
        });
    }

    let Some(timing) = capture.timing.as_ref() else {
        violations.push(CameraDiagnosticsViolation {
            field: "capture.timing",
            message: "timing section is missing".to_string(),
        });
        return violations;
    };

    for (field, value) in [
        ("capture.timing.target_fps", timing.target_fps),
        ("capture.timing.wall_fps", timing.wall_fps),
        ("capture.timing.sensor_fps", timing.sensor_fps),
        ("capture.timing.mean_gap_us", timing.mean_gap_us),
    ] {
        if !value.is_finite() {
            violations.push(CameraDiagnosticsViolation {
                field,
                message: format!("expected finite value, got {value}"),
            });
        }
    }

    if timing.wall_drift_pct.abs() > thresholds.max_abs_wall_drift_pct {
        violations.push(CameraDiagnosticsViolation {
            field: "capture.timing.wall_drift_pct",
            message: format!(
                "abs drift {} exceeds threshold {}",
                timing.wall_drift_pct.abs(),
                thresholds.max_abs_wall_drift_pct
            ),
        });
    }
    if timing.sensor_drift_pct.abs() > thresholds.max_abs_sensor_drift_pct {
        violations.push(CameraDiagnosticsViolation {
            field: "capture.timing.sensor_drift_pct",
            message: format!(
                "abs drift {} exceeds threshold {}",
                timing.sensor_drift_pct.abs(),
                thresholds.max_abs_sensor_drift_pct
            ),
        });
    }
    if timing.dropped_frames > thresholds.max_dropped_frames {
        violations.push(CameraDiagnosticsViolation {
            field: "capture.timing.dropped_frames",
            message: format!(
                "dropped frames {} exceeds threshold {}",
                timing.dropped_frames, thresholds.max_dropped_frames
            ),
        });
    }
    violations
}
