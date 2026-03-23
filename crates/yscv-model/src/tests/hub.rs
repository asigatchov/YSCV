use crate::{ModelHub, default_cache_dir};

#[test]
fn test_default_cache_dir() {
    // default_cache_dir should return a valid path
    let dir = default_cache_dir();
    let dir_str = dir.to_string_lossy();
    // Either env var is set or it falls back to ~/.yscv/models
    assert!(!dir_str.is_empty(), "cache dir should not be empty");
}

#[test]
fn test_model_hub_creation() {
    let hub = ModelHub::new();
    assert!(!hub.registry().is_empty());
}

#[test]
fn test_hub_entry_registry_has_known_models() {
    let hub = ModelHub::new();
    let reg = hub.registry();
    for name in &[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "vgg16",
        "vgg19",
        "mobilenet_v2",
        "efficientnet_b0",
        "alexnet",
    ] {
        assert!(reg.contains_key(*name), "registry should contain '{name}'");
    }
}

#[test]
fn test_download_missing_model_returns_error() {
    let hub = ModelHub::new();
    let result = hub.download_if_missing("nonexistent_model");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("not in the hub registry"),
        "error should mention registry, got: {err_msg}"
    );
}

#[test]
fn test_hub_entries_have_valid_fields() {
    let hub = ModelHub::new();
    for (name, entry) in hub.registry() {
        assert!(
            !entry.url.is_empty(),
            "entry '{name}' should have a non-empty URL"
        );
        assert!(
            entry.expected_size > 0,
            "entry '{name}' should have a positive expected_size"
        );
        assert!(
            entry.filename.ends_with(".safetensors"),
            "entry '{name}' filename should end with .safetensors, got: {}",
            entry.filename
        );
    }
}
