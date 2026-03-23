use yscv_detect::{BoundingBox, Detection};
use yscv_track::TrackedDetection;

use crate::{EvalError, GroundTruthTrack, TrackingDatasetFrame};

use super::helpers::ensure_frame_slot;

pub(crate) fn parse_and_build_mot(
    ground_truth_text: &str,
    predictions_text: &str,
) -> Result<Vec<TrackingDatasetFrame>, EvalError> {
    let mut ground_truth_frames = parse_mot_ground_truth(ground_truth_text)?;
    let mut prediction_frames = parse_mot_predictions(predictions_text)?;
    let frame_count = ground_truth_frames.len().max(prediction_frames.len());
    ground_truth_frames.resize_with(frame_count, Vec::new);
    prediction_frames.resize_with(frame_count, Vec::new);

    Ok(ground_truth_frames
        .into_iter()
        .zip(prediction_frames)
        .map(|(ground_truth, predictions)| TrackingDatasetFrame {
            ground_truth,
            predictions,
        })
        .collect())
}

fn parse_mot_ground_truth(text: &str) -> Result<Vec<Vec<GroundTruthTrack>>, EvalError> {
    let mut frames: Vec<Vec<GroundTruthTrack>> = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = parse_mot_tokens(line);
        if tokens.len() < 6 {
            return Err(EvalError::InvalidDatasetFormat {
                format: "mot",
                message: format!(
                    "invalid ground-truth line {line_no}: expected at least 6 comma-separated fields"
                ),
            });
        }

        let frame_idx = parse_mot_frame_index(tokens[0], "ground-truth", line_no)?;
        let object_id = parse_mot_positive_u64(tokens[1], "track id", "ground-truth", line_no)?;
        let bbox = parse_mot_bbox(
            tokens[2],
            tokens[3],
            tokens[4],
            tokens[5],
            "ground-truth",
            line_no,
        )?;

        if let Some(confidence_raw) = tokens.get(6).copied() {
            let confidence = parse_mot_f32(confidence_raw, "confidence", "ground-truth", line_no)?;
            if confidence <= 0.0 {
                continue;
            }
        }

        let class_id =
            parse_mot_optional_class_id(tokens.get(7).copied(), "ground-truth", line_no)?;

        ensure_frame_slot(&mut frames, frame_idx);
        frames[frame_idx].push(GroundTruthTrack {
            object_id,
            bbox,
            class_id,
        });
    }
    Ok(frames)
}

fn parse_mot_predictions(text: &str) -> Result<Vec<Vec<TrackedDetection>>, EvalError> {
    let mut frames: Vec<Vec<TrackedDetection>> = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = parse_mot_tokens(line);
        if tokens.len() < 6 {
            return Err(EvalError::InvalidDatasetFormat {
                format: "mot",
                message: format!(
                    "invalid prediction line {line_no}: expected at least 6 comma-separated fields"
                ),
            });
        }

        let frame_idx = parse_mot_frame_index(tokens[0], "prediction", line_no)?;
        let track_id = parse_mot_positive_u64(tokens[1], "track id", "prediction", line_no)?;
        let bbox = parse_mot_bbox(
            tokens[2],
            tokens[3],
            tokens[4],
            tokens[5],
            "prediction",
            line_no,
        )?;
        let score = match tokens.get(6).copied() {
            Some(raw) => parse_mot_f32(raw, "score", "prediction", line_no)?,
            None => 1.0,
        };
        let class_id = parse_mot_optional_class_id(tokens.get(7).copied(), "prediction", line_no)?;

        ensure_frame_slot(&mut frames, frame_idx);
        frames[frame_idx].push(TrackedDetection {
            track_id,
            detection: Detection {
                bbox,
                score,
                class_id,
            },
        });
    }
    Ok(frames)
}

fn parse_mot_tokens(line: &str) -> Vec<&str> {
    line.split(',').map(str::trim).collect()
}

fn parse_mot_frame_index(token: &str, source: &str, line_no: usize) -> Result<usize, EvalError> {
    let frame_number = parse_mot_positive_u64(token, "frame index", source, line_no)?;
    usize::try_from(frame_number - 1).map_err(|_| EvalError::InvalidDatasetFormat {
        format: "mot",
        message: format!(
            "invalid {source} line {line_no}: frame index `{token}` exceeds usize::MAX on this platform"
        ),
    })
}

fn parse_mot_positive_u64(
    token: &str,
    field: &'static str,
    source: &str,
    line_no: usize,
) -> Result<u64, EvalError> {
    let value = token
        .parse::<i64>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!("invalid {source} line {line_no} `{field}` value `{token}`: {err}"),
        })?;
    if value <= 0 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!("invalid {source} line {line_no}: `{field}` must be > 0"),
        });
    }
    Ok(value as u64)
}

fn parse_mot_f32(
    token: &str,
    field: &'static str,
    source: &str,
    line_no: usize,
) -> Result<f32, EvalError> {
    let value = token
        .parse::<f32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!("invalid {source} line {line_no} `{field}` value `{token}`: {err}"),
        })?;
    if !value.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!("invalid {source} line {line_no}: `{field}` must be finite"),
        });
    }
    Ok(value)
}

fn parse_mot_bbox(
    left_raw: &str,
    top_raw: &str,
    width_raw: &str,
    height_raw: &str,
    source: &str,
    line_no: usize,
) -> Result<BoundingBox, EvalError> {
    let left = parse_mot_f32(left_raw, "bbox_left", source, line_no)?;
    let top = parse_mot_f32(top_raw, "bbox_top", source, line_no)?;
    let width = parse_mot_f32(width_raw, "bbox_width", source, line_no)?;
    let height = parse_mot_f32(height_raw, "bbox_height", source, line_no)?;
    if width <= 0.0 || height <= 0.0 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!(
                "invalid {source} line {line_no}: bbox_width and bbox_height must be > 0"
            ),
        });
    }

    let x2 = left + width;
    let y2 = top + height;
    if !x2.is_finite() || !y2.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!("invalid {source} line {line_no}: bbox corner overflow"),
        });
    }

    Ok(BoundingBox {
        x1: left,
        y1: top,
        x2,
        y2,
    })
}

fn parse_mot_optional_class_id(
    token: Option<&str>,
    source: &str,
    line_no: usize,
) -> Result<usize, EvalError> {
    let Some(raw) = token else {
        return Ok(0);
    };
    if raw.is_empty() {
        return Ok(0);
    }

    let parsed = raw
        .parse::<i64>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "mot",
            message: format!("invalid {source} line {line_no} `class_id` value `{raw}`: {err}"),
        })?;
    if parsed <= 0 {
        return Ok(0);
    }
    usize::try_from(parsed).map_err(|_| EvalError::InvalidDatasetFormat {
        format: "mot",
        message: format!(
            "invalid {source} line {line_no}: `class_id` value `{raw}` exceeds usize::MAX on this platform"
        ),
    })
}
