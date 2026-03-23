use std::path::Path;

use roxmltree::Node;
use yscv_detect::{BoundingBox, Detection};

use crate::{DetectionDatasetFrame, EvalError, LabeledBox};

use super::helpers::{parse_image_id_manifest, read_optional_text_file};

pub(crate) fn load_voc_xml_dirs(
    manifest_text: &str,
    ground_truth_xml_dir: &Path,
    prediction_xml_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let image_ids = parse_image_id_manifest(manifest_text, "voc")?;
    let mut frames = Vec::with_capacity(image_ids.len());

    for image_id in image_ids {
        let gt_path = ground_truth_xml_dir.join(format!("{image_id}.xml"));
        let pred_path = prediction_xml_dir.join(format!("{image_id}.xml"));

        let ground_truth = match read_optional_text_file(&gt_path)? {
            Some(xml) => parse_voc_ground_truth_annotation(&xml, &gt_path.display().to_string())?,
            None => Vec::new(),
        };
        let predictions = match read_optional_text_file(&pred_path)? {
            Some(xml) => parse_voc_prediction_annotation(&xml, &pred_path.display().to_string())?,
            None => Vec::new(),
        };

        frames.push(DetectionDatasetFrame {
            ground_truth,
            predictions,
        });
    }

    Ok(frames)
}

fn parse_voc_ground_truth_annotation(
    xml: &str,
    source: &str,
) -> Result<Vec<LabeledBox>, EvalError> {
    let doc = roxmltree::Document::parse(xml).map_err(|err| EvalError::InvalidDatasetFormat {
        format: "voc",
        message: format!("invalid XML in `{source}`: {err}"),
    })?;
    let mut labels = Vec::new();
    for (object_idx, object) in doc
        .descendants()
        .filter(|node| node.has_tag_name("object"))
        .enumerate()
    {
        let class_id = parse_voc_class_id(object, source, object_idx)?;
        let bbox = parse_voc_bbox(object, source, object_idx)?;
        labels.push(LabeledBox { bbox, class_id });
    }
    Ok(labels)
}

fn parse_voc_prediction_annotation(xml: &str, source: &str) -> Result<Vec<Detection>, EvalError> {
    let doc = roxmltree::Document::parse(xml).map_err(|err| EvalError::InvalidDatasetFormat {
        format: "voc",
        message: format!("invalid XML in `{source}`: {err}"),
    })?;
    let mut detections = Vec::new();
    for (object_idx, object) in doc
        .descendants()
        .filter(|node| node.has_tag_name("object"))
        .enumerate()
    {
        let class_id = parse_voc_class_id(object, source, object_idx)?;
        let bbox = parse_voc_bbox(object, source, object_idx)?;
        let score = parse_voc_score(object, source, object_idx)?;
        detections.push(Detection {
            bbox,
            score,
            class_id,
        });
    }
    Ok(detections)
}

fn parse_voc_class_id(
    object: Node<'_, '_>,
    source: &str,
    object_idx: usize,
) -> Result<usize, EvalError> {
    if let Some(raw) = find_child_text(object, "class_id") {
        return parse_voc_usize(raw, "class_id", source, object_idx);
    }
    if let Some(raw) = find_child_text(object, "name") {
        return parse_voc_usize(raw, "name", source, object_idx);
    }
    Err(EvalError::InvalidDatasetFormat {
        format: "voc",
        message: format!("missing `class_id`/`name` in `{source}` object[{object_idx}]"),
    })
}

fn parse_voc_bbox(
    object: Node<'_, '_>,
    source: &str,
    object_idx: usize,
) -> Result<BoundingBox, EvalError> {
    let Some(bndbox) = object
        .children()
        .find(|child| child.is_element() && child.has_tag_name("bndbox"))
    else {
        return Err(EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!("missing `bndbox` in `{source}` object[{object_idx}]"),
        });
    };
    let x1 = parse_voc_f32_required(bndbox, "xmin", source, object_idx)?;
    let y1 = parse_voc_f32_required(bndbox, "ymin", source, object_idx)?;
    let x2 = parse_voc_f32_required(bndbox, "xmax", source, object_idx)?;
    let y2 = parse_voc_f32_required(bndbox, "ymax", source, object_idx)?;

    if x2 <= x1 || y2 <= y1 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!(
                "invalid bbox in `{source}` object[{object_idx}]: expected xmax>xmin and ymax>ymin"
            ),
        });
    }

    Ok(BoundingBox { x1, y1, x2, y2 })
}

fn parse_voc_score(
    object: Node<'_, '_>,
    source: &str,
    object_idx: usize,
) -> Result<f32, EvalError> {
    let Some(raw) = find_child_text(object, "score") else {
        return Ok(1.0);
    };
    let score = raw
        .parse::<f32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!(
                "invalid `score` in `{source}` object[{object_idx}] value `{raw}`: {err}"
            ),
        })?;
    if !score.is_finite() || !(0.0..=1.0).contains(&score) {
        return Err(EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!(
                "invalid `score` in `{source}` object[{object_idx}]: expected finite value in [0, 1]"
            ),
        });
    }
    Ok(score)
}

fn parse_voc_f32_required(
    node: Node<'_, '_>,
    field: &'static str,
    source: &str,
    object_idx: usize,
) -> Result<f32, EvalError> {
    let Some(raw) = find_child_text(node, field) else {
        return Err(EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!("missing `{field}` in `{source}` object[{object_idx}]"),
        });
    };
    let value = raw
        .parse::<f32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!(
                "invalid `{field}` in `{source}` object[{object_idx}] value `{raw}`: {err}"
            ),
        })?;
    if !value.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!(
                "invalid `{field}` in `{source}` object[{object_idx}]: expected finite number"
            ),
        });
    }
    Ok(value)
}

fn parse_voc_usize(
    raw: &str,
    field: &'static str,
    source: &str,
    object_idx: usize,
) -> Result<usize, EvalError> {
    raw.parse::<usize>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "voc",
            message: format!(
                "invalid `{field}` in `{source}` object[{object_idx}] value `{raw}`: {err}"
            ),
        })
}

fn find_child_text<'a>(node: Node<'a, 'a>, child_name: &str) -> Option<&'a str> {
    node.children()
        .find(|child| child.is_element() && child.tag_name().name() == child_name)
        .and_then(|child| child.text())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}
