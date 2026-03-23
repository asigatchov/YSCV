# Native Camera Validation Checklist

This checklist is for real-device validation of `native-camera` runtime behavior on macOS, Linux, and Windows.

## 1. Build Validation

Run on each target OS:

```bash
cargo check -p yscv-cli --features native-camera --bin yscv-cli
cargo check -p camera-face-tool --features native-camera
```

Pass criteria:
- both commands complete without warnings promoted to errors in CI,
- no feature-gating or linking failures for camera backend dependencies.

## 2. Device Discovery Validation

```bash
cargo run -p yscv-cli --bin yscv-cli --features native-camera -- --list-cameras
cargo run -p camera-face-tool --features native-camera -- --list-cameras
```

Pass criteria:
- at least one expected device appears,
- device labels are stable across repeated runs,
- `--device-name <query>` filters deterministically.

## 3. Capture Diagnostics Validation

```bash
cargo run -p yscv-cli --bin yscv-cli --features native-camera -- \
  --diagnose-camera --diagnose-frames 60 --device 0 --width 640 --height 480 --fps 30 \
  --diagnose-report artifacts/diagnostics/yscv-cli-diagnostics.json

cargo run -p camera-face-tool --features native-camera -- \
  --diagnose-camera --diagnose-frames 60 --device 0 --width 640 --height 480 --fps 30 \
  --diagnose-report artifacts/diagnostics/camera-face-tool-diagnostics.json
```

Capture output fields (from diagnostics report JSON):
- `status`,
- `capture.timing.wall_fps`,
- `capture.timing.sensor_fps`,
- `capture.timing.wall_drift_pct`,
- `capture.timing.sensor_drift_pct`,
- `capture.timing.dropped_frames`.

Pass criteria:
- at least 2 frames captured,
- `dropped_frames = 0` in low-load local run,
- absolute drift for wall/sensor FPS is within 25% of requested FPS.
- diagnostics report JSON files are created and non-empty.

Optional strict gate command (report validator):

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --validate-diagnostics-report artifacts/diagnostics/yscv-cli-diagnostics.json \
  --validate-diagnostics-min-frames 2 \
  --validate-diagnostics-max-drift-pct 25 \
  --validate-diagnostics-max-dropped 0
```

Pass criteria:
- command exits with code `0`,
- output contains `diagnostics_report_validation passed`.

## 4. End-to-End Face Pipeline Validation

```bash
cargo run -p camera-face-tool --features native-camera -- \
  --device 0 --width 640 --height 480 --fps 30 --max-frames 120
```

Pass criteria:
- frames are processed continuously,
- detection + tracking outputs are present,
- no runtime panics or backend read errors.

## 5. Optional Event-Log Evidence

```bash
cargo run -p yscv-cli --bin yscv-cli --features native-camera -- \
  --camera --max-frames 120 --event-log artifacts/camera-events.jsonl
```

Pass criteria:
- JSONL file is created and non-empty,
- each record contains frame index, timings, and tracked objects.

## 6. Regression Protocol

When camera drivers, backend versions, or OS versions change:
1. Re-run sections 1-5.
2. Save command output snippets, diagnostics report JSON, and event-log sample in CI/job artifacts.
3. Record drift and dropped-frame changes in release notes if behavior changed.
