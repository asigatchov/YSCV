use serde::Deserialize;
use std::path::Path;

use crate::ModelError;

use super::helpers::{
    adapter_sample_len, build_supervised_dataset_from_flat_values, load_dataset_text_file,
    validate_adapter_sample_shape, validate_finite_values,
};

/// Configuration for parsing/loading supervised JSONL datasets.
#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedJsonlConfig {
    input_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

impl SupervisedJsonlConfig {
    pub fn new(input_shape: Vec<usize>, target_shape: Vec<usize>) -> Result<Self, ModelError> {
        validate_adapter_sample_shape("input_shape", &input_shape)?;
        validate_adapter_sample_shape("target_shape", &target_shape)?;
        Ok(Self {
            input_shape,
            target_shape,
        })
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }
}

#[derive(Debug, Deserialize)]
struct JsonlSupervisedRecord {
    #[serde(alias = "inputs")]
    #[serde(alias = "features")]
    input: JsonlNumericField,
    #[serde(alias = "targets")]
    #[serde(alias = "label")]
    #[serde(alias = "labels")]
    target: JsonlNumericField,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum JsonlNumericField {
    Scalar(f32),
    Vector(Vec<f32>),
}

fn jsonl_field_to_row(field: JsonlNumericField) -> Vec<f32> {
    match field {
        JsonlNumericField::Vector(values) => values,
        JsonlNumericField::Scalar(value) => vec![value],
    }
}

/// Parses supervised training samples from JSONL text into a `SupervisedDataset`.
///
/// Each non-empty line must be a JSON object with:
/// - `input` (or alias `inputs`/`features`): flat sample values matching `config.input_shape`
/// - `target` (or alias `targets`/`label`/`labels`): flat sample values matching `config.target_shape`
pub fn parse_supervised_dataset_jsonl(
    content: &str,
    config: &SupervisedJsonlConfig,
) -> Result<super::types::SupervisedDataset, ModelError> {
    let input_row_len = adapter_sample_len("input_shape", config.input_shape())?;
    let target_row_len = adapter_sample_len("target_shape", config.target_shape())?;

    let mut input_values = Vec::new();
    let mut target_values = Vec::new();
    let mut sample_count = 0usize;

    for (line_idx, raw_line) in content.lines().enumerate() {
        let line_number = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let record: JsonlSupervisedRecord =
            serde_json::from_str(line).map_err(|error| ModelError::DatasetJsonlParse {
                line: line_number,
                message: error.to_string(),
            })?;

        let input_row = jsonl_field_to_row(record.input);
        if input_row.len() != input_row_len {
            return Err(ModelError::InvalidDatasetRecordLength {
                line: line_number,
                field: "input",
                expected: input_row_len,
                got: input_row.len(),
            });
        }
        let target_row = jsonl_field_to_row(record.target);
        if target_row.len() != target_row_len {
            return Err(ModelError::InvalidDatasetRecordLength {
                line: line_number,
                field: "target",
                expected: target_row_len,
                got: target_row.len(),
            });
        }

        validate_finite_values(line_number, "input", &input_row)?;
        validate_finite_values(line_number, "target", &target_row)?;

        input_values.extend_from_slice(&input_row);
        target_values.extend_from_slice(&target_row);
        sample_count =
            sample_count
                .checked_add(1)
                .ok_or_else(|| ModelError::InvalidDatasetAdapterShape {
                    field: "sample_count",
                    shape: vec![sample_count],
                    message: "sample count overflow".to_string(),
                })?;
    }

    if sample_count == 0 {
        return Err(ModelError::EmptyDataset);
    }

    build_supervised_dataset_from_flat_values(
        config.input_shape(),
        config.target_shape(),
        sample_count,
        input_values,
        target_values,
    )
}

/// Loads supervised training samples from a JSONL file.
pub fn load_supervised_dataset_jsonl_file<P: AsRef<Path>>(
    path: P,
    config: &SupervisedJsonlConfig,
) -> Result<super::types::SupervisedDataset, ModelError> {
    let content = load_dataset_text_file(path)?;
    parse_supervised_dataset_jsonl(&content, config)
}
