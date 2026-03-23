use std::fs;
use std::time::Duration;

use serde_json::{Value, json};
use yscv_video::{
    CameraConfig, CameraFrameSource, VideoError, filter_camera_devices, list_camera_devices,
    resolve_camera_device,
};

use crate::config::{FaceAppConfig, FaceAppError};
use crate::error::AppError;
use crate::util::{duration_to_ms, ensure_parent_dir};

#[derive(Debug, Clone, Copy)]
struct TimingMetrics {
    target_fps: f64,
    wall_fps: f64,
    sensor_fps: f64,
    wall_drift_pct: f64,
    sensor_drift_pct: f64,
    mean_gap_us: f64,
    min_gap_us: u64,
    max_gap_us: u64,
    dropped_frames: u64,
}

#[derive(Debug, Clone, Copy)]
struct FrameSample {
    index: u64,
    timestamp_us: u64,
    width: usize,
    height: usize,
}

pub(crate) fn run_camera_diagnostics(config: &FaceAppConfig) -> Result<(), AppError> {
    println!("camera-face-app diagnostics: starting");
    println!(
        "requested capture config: device={} {}x{}@{}fps",
        config.device_index, config.width, config.height, config.fps
    );

    let requested = json!({
        "device_index": config.device_index,
        "width": config.width,
        "height": config.height,
        "fps": config.fps,
        "diagnose_frames": config.diagnose_frames,
        "device_name_query": config.device_name_query,
    });

    let devices = match list_camera_devices() {
        Ok(devices) => devices,
        Err(VideoError::CameraBackendDisabled) => {
            println!("camera support is disabled; rebuild with `--features native-camera`");
            let report = json!({
                "tool": "camera-face-app",
                "mode": "diagnostics",
                "status": "backend_disabled",
                "requested": requested,
                "discovered_devices": [],
            });
            maybe_write_report(config, &report)?;
            println!("camera-face-app diagnostics: completed");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let discovered_devices = devices
        .iter()
        .map(|device| {
            json!({
                "index": device.index,
                "label": device.label,
            })
        })
        .collect::<Vec<_>>();

    if devices.is_empty() {
        println!("diagnostics: no camera devices discovered");
        let report = json!({
            "tool": "camera-face-app",
            "mode": "diagnostics",
            "status": "no_devices",
            "requested": requested,
            "discovered_devices": discovered_devices,
        });
        maybe_write_report(config, &report)?;
        println!("camera-face-app diagnostics: completed");
        return Ok(());
    }

    println!("diagnostics: discovered {} device(s)", devices.len());
    for device in &devices {
        println!("  {}: {}", device.index, device.label);
    }

    let selected = if let Some(query) = config.device_name_query.as_deref() {
        let selected = resolve_camera_device(query)?;
        println!(
            "diagnostics: selected via query `{}` -> {}: {}",
            query, selected.index, selected.label
        );
        selected
    } else if let Some(found) = devices
        .iter()
        .find(|device| device.index == config.device_index)
    {
        found.clone()
    } else {
        return Err(FaceAppError::Message(format!(
            "diagnostics: requested device index {} was not found in discovery list",
            config.device_index
        ))
        .into());
    };

    let mut source = CameraFrameSource::open(CameraConfig {
        device_index: selected.index,
        width: config.width,
        height: config.height,
        fps: config.fps,
    })?;
    println!(
        "diagnostics: capturing up to {} frame(s) for timing analysis",
        config.diagnose_frames
    );

    let mut frames = Vec::with_capacity(config.diagnose_frames);
    let capture_started = std::time::Instant::now();
    for _ in 0..config.diagnose_frames {
        match source.next_rgb8_frame()? {
            Some(frame) => frames.push(FrameSample {
                index: frame.index(),
                timestamp_us: frame.timestamp_us(),
                width: frame.width(),
                height: frame.height(),
            }),
            None => break,
        }
    }
    let capture_elapsed = capture_started.elapsed();

    let first_frame = frames.first().map(|first| {
        json!({
            "index": first.index,
            "timestamp_us": first.timestamp_us,
            "shape": [first.height, first.width, 3],
        })
    });

    let timing = compute_timing_metrics(&frames, capture_elapsed, config.fps);
    if frames.is_empty() {
        println!("diagnostics: capture opened but produced no frames");
    } else if let Some(first) = frames.first() {
        println!(
            "diagnostics: capture ok first_frame_index={} ts_us={} shape=[{}, {}, {}] collected_frames={} wall_ms={:.3}",
            first.index,
            first.timestamp_us,
            first.height,
            first.width,
            3,
            frames.len(),
            duration_to_ms(capture_elapsed),
        );

        if let Some(metrics) = timing {
            println!(
                "diagnostics: timing target_fps={:.2} wall_fps={:.2} sensor_fps={:.2} wall_drift={:+.1}% sensor_drift={:+.1}%",
                metrics.target_fps,
                metrics.wall_fps,
                metrics.sensor_fps,
                metrics.wall_drift_pct,
                metrics.sensor_drift_pct
            );
            println!(
                "diagnostics: frame_interval_us mean={:.1} min={} max={} dropped_frames={}",
                metrics.mean_gap_us, metrics.min_gap_us, metrics.max_gap_us, metrics.dropped_frames
            );

            if metrics.dropped_frames > 0 {
                println!(
                    "diagnostics: warning dropped frame indices observed (count={})",
                    metrics.dropped_frames
                );
            }
            if metrics.wall_drift_pct.abs() > 25.0 || metrics.sensor_drift_pct.abs() > 25.0 {
                println!(
                    "diagnostics: warning fps drift exceeds 25%; check camera backend/device load"
                );
            }
        } else {
            println!("diagnostics: collected fewer than 2 frames; timing analysis skipped");
        }
    }

    let timing_json = timing.map(|metrics| {
        json!({
            "target_fps": metrics.target_fps,
            "wall_fps": metrics.wall_fps,
            "sensor_fps": metrics.sensor_fps,
            "wall_drift_pct": metrics.wall_drift_pct,
            "sensor_drift_pct": metrics.sensor_drift_pct,
            "mean_gap_us": metrics.mean_gap_us,
            "min_gap_us": metrics.min_gap_us,
            "max_gap_us": metrics.max_gap_us,
            "dropped_frames": metrics.dropped_frames,
            "drift_warning": metrics.wall_drift_pct.abs() > 25.0 || metrics.sensor_drift_pct.abs() > 25.0,
            "dropped_frames_warning": metrics.dropped_frames > 0,
        })
    });

    let report = json!({
        "tool": "camera-face-app",
        "mode": "diagnostics",
        "status": if frames.is_empty() { "no_frames" } else { "ok" },
        "requested": requested,
        "discovered_devices": discovered_devices,
        "selected_device": {
            "index": selected.index,
            "label": selected.label,
        },
        "capture": {
            "requested_frames": config.diagnose_frames,
            "collected_frames": frames.len(),
            "wall_ms": duration_to_ms(capture_elapsed),
            "first_frame": first_frame,
            "timing": timing_json,
        },
    });
    maybe_write_report(config, &report)?;

    println!("camera-face-app diagnostics: completed");
    Ok(())
}

pub(crate) fn print_camera_devices(query: Option<&str>) -> Result<(), AppError> {
    let devices = match list_camera_devices() {
        Ok(devices) => devices,
        Err(VideoError::CameraBackendDisabled) => {
            println!("camera support is disabled; rebuild with `--features native-camera`");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let devices = if let Some(query) = query {
        filter_camera_devices(&devices, query)?
    } else {
        devices
    };

    if devices.is_empty() {
        if let Some(query) = query {
            println!("no camera devices matched query `{query}`");
        } else {
            println!("no camera devices were found");
        }
        return Ok(());
    }

    if let Some(query) = query {
        println!("available camera devices matching `{query}`:");
    } else {
        println!("available camera devices:");
    }
    for device in devices {
        println!("  {}: {}", device.index, device.label);
    }
    Ok(())
}

fn compute_timing_metrics(
    frames: &[FrameSample],
    capture_elapsed: Duration,
    requested_fps: u32,
) -> Option<TimingMetrics> {
    if frames.len() < 2 {
        return None;
    }

    let mut gap_sum_us = 0u128;
    let mut min_gap_us = u64::MAX;
    let mut max_gap_us = 0u64;
    let mut dropped_frames = 0u64;

    for pair in frames.windows(2) {
        let prev = &pair[0];
        let next = &pair[1];

        let ts_gap_us = next.timestamp_us.saturating_sub(prev.timestamp_us);
        gap_sum_us += u128::from(ts_gap_us);
        min_gap_us = min_gap_us.min(ts_gap_us);
        max_gap_us = max_gap_us.max(ts_gap_us);

        let index_gap = next.index.saturating_sub(prev.index);
        if index_gap > 1 {
            dropped_frames += index_gap - 1;
        }
    }

    let interval_count = (frames.len() - 1) as f64;
    let mean_gap_us = gap_sum_us as f64 / interval_count;
    let sensor_fps = if mean_gap_us > 0.0 {
        1_000_000.0 / mean_gap_us
    } else {
        0.0
    };
    let wall_fps = if capture_elapsed.as_secs_f64() > 0.0 {
        frames.len() as f64 / capture_elapsed.as_secs_f64()
    } else {
        0.0
    };
    let target_fps = f64::from(requested_fps);
    let wall_drift_pct = ((wall_fps - target_fps) / target_fps) * 100.0;
    let sensor_drift_pct = ((sensor_fps - target_fps) / target_fps) * 100.0;

    Some(TimingMetrics {
        target_fps,
        wall_fps,
        sensor_fps,
        wall_drift_pct,
        sensor_drift_pct,
        mean_gap_us,
        min_gap_us,
        max_gap_us,
        dropped_frames,
    })
}

fn maybe_write_report(config: &FaceAppConfig, report: &Value) -> Result<(), AppError> {
    if let Some(path) = config.diagnose_report_path.as_deref() {
        ensure_parent_dir(path)?;
        let body = serde_json::to_vec_pretty(report)?;
        fs::write(path, body)?;
        println!("diagnostics: report saved to {}", path.display());
    }
    Ok(())
}
