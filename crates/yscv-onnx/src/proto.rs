/// Re-export generated ONNX protobuf types.
#[allow(
    clippy::doc_overindented_list_items,
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq
)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
