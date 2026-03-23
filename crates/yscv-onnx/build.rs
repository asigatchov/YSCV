fn main() {
    // Try to compile protos; if protoc is not found, use pre-generated code
    if let Err(e) = prost_build::Config::new().compile_protos(&["proto/onnx-ml.proto"], &["proto/"])
    {
        eprintln!("cargo:warning=protoc not found, using pre-generated proto code: {e}");
        // The pre-generated code should already be checked in
    }
}
