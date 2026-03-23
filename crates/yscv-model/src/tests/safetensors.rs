use crate::{SafeTensorDType, SafeTensorFile, load_state_dict};

/// Build a synthetic safetensors file in memory.
///
/// Each entry is `(name, dtype_str, shape, raw_data)`.
fn build_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
    // Concatenate raw data and compute offsets
    let mut data_section = Vec::new();
    let mut entries: Vec<(&str, &str, &[usize], usize, usize)> = Vec::new();
    for &(name, dtype, shape, raw) in tensors {
        let start = data_section.len();
        data_section.extend_from_slice(raw);
        let end = data_section.len();
        entries.push((name, dtype, shape, start, end));
    }

    // Build JSON header manually
    let mut json_parts: Vec<String> = Vec::new();
    for (name, dtype, shape, start, end) in &entries {
        let shape_str = format!(
            "[{}]",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        json_parts.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{shape_str},"data_offsets":[{start},{end}]}}"#
        ));
    }
    let header_json = format!("{{{}}}", json_parts.join(","));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut result = Vec::new();
    result.extend_from_slice(&header_len.to_le_bytes());
    result.extend_from_slice(header_bytes);
    result.extend_from_slice(&data_section);
    result
}

#[test]
fn test_parse_safetensors_header() {
    let weight_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let bias_data: Vec<u8> = [0.5f32, -0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let bytes = build_safetensors(&[
        ("layer.weight", "F32", &[2, 3], &weight_data),
        ("layer.bias", "F32", &[2], &bias_data),
    ]);

    let file = SafeTensorFile::from_bytes(&bytes).unwrap();

    let mut names = file.tensor_names();
    names.sort();
    assert_eq!(names, vec!["layer.bias", "layer.weight"]);

    let w_info = file.tensor_info("layer.weight").unwrap();
    assert_eq!(w_info.dtype, SafeTensorDType::F32);
    assert_eq!(w_info.shape, vec![2, 3]);

    let b_info = file.tensor_info("layer.bias").unwrap();
    assert_eq!(b_info.dtype, SafeTensorDType::F32);
    assert_eq!(b_info.shape, vec![2]);
}

#[test]
fn test_load_f32_tensor() {
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let raw: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_safetensors(&[("w", "F32", &[2, 3], &raw)]);
    let file = SafeTensorFile::from_bytes(&bytes).unwrap();
    let tensor = file.load_tensor("w").unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.data(), &expected[..]);
}

#[test]
fn test_load_f16_tensor() {
    // Use known F16 bit patterns:
    // 1.0 in FP16 = 0x3C00
    // 0.5 in FP16 = 0x3800
    // 2.0 in FP16 = 0x4000
    let f16_values: Vec<u16> = vec![0x3C00, 0x3800, 0x4000];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_safetensors(&[("h", "F16", &[3], &raw)]);
    let file = SafeTensorFile::from_bytes(&bytes).unwrap();
    let tensor = file.load_tensor("h").unwrap();

    assert_eq!(tensor.shape(), &[3]);
    // Should be converted to F32
    let data = tensor.data();
    super::assert_slice_approx_eq(data, &[1.0, 0.5, 2.0], 1e-3);
}

#[test]
fn test_empty_file_error() {
    let result = SafeTensorFile::from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_truncated_header_error() {
    // Header says 1000 bytes but file is too short
    let header_len: u64 = 1000;
    let bytes = header_len.to_le_bytes().to_vec();
    let result = SafeTensorFile::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_unknown_tensor_name() {
    let raw: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let bytes = build_safetensors(&[("a", "F32", &[1], &raw)]);
    let file = SafeTensorFile::from_bytes(&bytes).unwrap();

    let result = file.load_tensor("nonexistent");
    assert!(result.is_err());

    assert!(file.tensor_info("nonexistent").is_none());
}

#[test]
fn test_load_state_dict() {
    let w_data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_data: Vec<u8> = [0.1f32].iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_safetensors(&[
        ("weight", "F32", &[2], &w_data),
        ("bias", "F32", &[1], &b_data),
    ]);

    // Write to a temp file and load via load_state_dict
    let tmp = super::unique_temp_path_with_extension("safetensors_test", "safetensors");
    std::fs::write(&tmp, &bytes).unwrap();

    let dict = load_state_dict(&tmp).unwrap();
    assert_eq!(dict.len(), 2);
    assert!(dict.contains_key("weight"));
    assert!(dict.contains_key("bias"));

    let w = &dict["weight"];
    assert_eq!(w.shape(), &[2]);
    assert_eq!(w.data(), &[1.0, 2.0]);

    let b = &dict["bias"];
    assert_eq!(b.shape(), &[1]);
    super::assert_slice_approx_eq(b.data(), &[0.1], 1e-6);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_metadata_key_skipped() {
    // Build a file with __metadata__ key in the header
    let raw: Vec<u8> = 1.0f32.to_le_bytes().to_vec();

    // Manually build with metadata
    let header_json =
        r#"{"__metadata__":{"format":"pt"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header_bytes);
    bytes.extend_from_slice(&raw);

    let file = SafeTensorFile::from_bytes(&bytes).unwrap();
    let names = file.tensor_names();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "x");
}

#[test]
fn test_load_bf16_tensor() {
    // BF16 1.0 = 0x3F80 (upper 16 bits of f32 1.0)
    // BF16 2.0 = 0x4000
    let bf16_values: Vec<u16> = vec![0x3F80, 0x4000];
    let raw: Vec<u8> = bf16_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_safetensors(&[("b", "BF16", &[2], &raw)]);
    let file = SafeTensorFile::from_bytes(&bytes).unwrap();
    let tensor = file.load_tensor("b").unwrap();

    assert_eq!(tensor.shape(), &[2]);
    let data = tensor.data();
    super::assert_slice_approx_eq(data, &[1.0, 2.0], 1e-3);
}
