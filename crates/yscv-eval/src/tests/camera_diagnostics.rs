use super::*;

#[test]
fn parse_camera_diagnostics_report_parses_ok_report() {
    let json = r#"
        {
          "tool": "yscv-cli",
          "mode": "diagnostics",
          "status": "ok",
          "requested": {
            "device_index": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "diagnose_frames": 60,
            "device_name_query": null
          },
          "discovered_devices": [
            {"index": 0, "label": "Camera"}
          ],
          "selected_device": {"index": 0, "label": "Camera"},
          "capture": {
            "requested_frames": 60,
            "collected_frames": 60,
            "wall_ms": 2000.0,
            "first_frame": {"index": 0, "timestamp_us": 1000, "shape": [480, 640, 3]},
            "timing": {
              "target_fps": 30.0,
              "wall_fps": 30.0,
              "sensor_fps": 29.8,
              "wall_drift_pct": 0.0,
              "sensor_drift_pct": -0.7,
              "mean_gap_us": 33333.3,
              "min_gap_us": 32000,
              "max_gap_us": 35000,
              "dropped_frames": 0,
              "drift_warning": false,
              "dropped_frames_warning": false
            }
          }
        }
    "#;
    let report = parse_camera_diagnostics_report_json(json).unwrap();
    assert_eq!(report.mode, "diagnostics");
    assert_eq!(report.status, "ok");
    assert_eq!(report.capture.as_ref().unwrap().collected_frames, 60);
}

#[test]
fn validate_camera_diagnostics_report_accepts_within_thresholds() {
    let json = r#"
        {
          "tool": "yscv-cli",
          "mode": "diagnostics",
          "status": "ok",
          "requested": {
            "device_index": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "diagnose_frames": 60,
            "device_name_query": null
          },
          "discovered_devices": [],
          "selected_device": {"index": 0, "label": "Camera"},
          "capture": {
            "requested_frames": 60,
            "collected_frames": 60,
            "wall_ms": 2000.0,
            "first_frame": {"index": 0, "timestamp_us": 1000, "shape": [480, 640, 3]},
            "timing": {
              "target_fps": 30.0,
              "wall_fps": 29.5,
              "sensor_fps": 29.0,
              "wall_drift_pct": -1.7,
              "sensor_drift_pct": -3.3,
              "mean_gap_us": 34482.7,
              "min_gap_us": 33000,
              "max_gap_us": 36000,
              "dropped_frames": 0,
              "drift_warning": false,
              "dropped_frames_warning": false
            }
          }
        }
    "#;
    let report = parse_camera_diagnostics_report_json(json).unwrap();
    let violations = validate_camera_diagnostics_report(
        &report,
        CameraDiagnosticsThresholds {
            min_collected_frames: 30,
            max_abs_wall_drift_pct: 25.0,
            max_abs_sensor_drift_pct: 25.0,
            max_dropped_frames: 0,
        },
    );
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn validate_camera_diagnostics_report_reports_failures() {
    let json = r#"
        {
          "tool": "yscv-cli",
          "mode": "diagnostics",
          "status": "no_frames",
          "requested": {
            "device_index": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "diagnose_frames": 60,
            "device_name_query": null
          },
          "discovered_devices": [],
          "selected_device": {"index": 0, "label": "Camera"},
          "capture": {
            "requested_frames": 60,
            "collected_frames": 1,
            "wall_ms": 100.0,
            "first_frame": {"index": 0, "timestamp_us": 1000, "shape": [480, 640, 3]},
            "timing": {
              "target_fps": 30.0,
              "wall_fps": 10.0,
              "sensor_fps": 10.0,
              "wall_drift_pct": -66.0,
              "sensor_drift_pct": -66.0,
              "mean_gap_us": 100000.0,
              "min_gap_us": 100000,
              "max_gap_us": 100000,
              "dropped_frames": 3,
              "drift_warning": true,
              "dropped_frames_warning": true
            }
          }
        }
    "#;
    let report = parse_camera_diagnostics_report_json(json).unwrap();
    let violations =
        validate_camera_diagnostics_report(&report, CameraDiagnosticsThresholds::default());
    assert!(violations.iter().any(|v| v.field == "status"));
    assert!(
        violations
            .iter()
            .any(|v| v.field == "capture.collected_frames")
    );
    assert!(
        violations
            .iter()
            .any(|v| v.field == "capture.timing.wall_drift_pct")
    );
    assert!(
        violations
            .iter()
            .any(|v| v.field == "capture.timing.sensor_drift_pct")
    );
    assert!(
        violations
            .iter()
            .any(|v| v.field == "capture.timing.dropped_frames")
    );
}
