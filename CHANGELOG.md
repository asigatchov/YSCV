# Changelog

All notable changes to the yscv workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-03-18

### Added
- **yscv-imgproc**: Hand-written NEON and SSE/SSSE3 SIMD for all 12 u8 image operations (grayscale, dilate, erode, gaussian, box blur, sobel, median, canny sobel, canny NMS, resize 1ch, resize RGB H-pass, resize RGB V-pass).
- **yscv-imgproc**: GCD `dispatch_apply` threading on macOS with rayon fallback on all platforms.
- **yscv-imgproc**: Direct 3x3 gaussian blur (vextq/alignr, zero intermediate buffers).
- **yscv-imgproc**: Stride-2 fast path for ~2x downscale resize.
- **yscv-track**: 27 new tests for DeepSORT and ByteTrack (57 total).
- **CI**: ARM64 Linux runner (`ubuntu-24.04-arm`).
- **CI**: GPU feature compilation check (`cargo check -p yscv-kernels --features gpu`).
- **build**: Release profile with `lto = "thin"`, `codegen-units = 1`.
- **build**: Target-specific CPU flags in `.cargo/config.toml` (apple-m1, neoverse-n1, x86-64-v3).
- **bench**: OpenCV comparison benchmarks for u8 and f32 operations.
- **bench**: CPU frequency warm-up for Apple Silicon benchmarks.
- **docs**: Architecture guide (`docs/architecture.md`).
- **docs**: OpenCV vs yscv comparison with full methodology in `docs/performance-benchmarks.md`.

### Changed
- **yscv-imgproc**: Grayscale u8 processes entire image as flat array (removed per-row GCD overhead).
- **yscv-imgproc**: Gaussian blur uses direct 3x3 approach instead of separable tiles.
- **yscv-imgproc**: Morphology uses branchless vextq/alignr inner loop.

### Fixed
- **yscv-imgproc**: Canny hysteresis buffer overflow on negative offset underflow.
- **yscv-imgproc**: `to_tensor()` uses `expect()` instead of `unwrap()` with diagnostic message.
- **docs**: All rustdoc unresolved link warnings fixed (29 warnings eliminated).
- **workspace**: All clippy warnings fixed (`cargo clippy -- -D warnings` clean).

### Removed
- `goals.md` — replaced by `docs/ecosystem-capability-matrix.md` as canonical progress tracker.

### Added
- **yscv-optim**: LAMB optimizer with trust ratio scaling for large-batch training.
- **yscv-optim**: LARS optimizer with layer-wise adaptive rate scaling.
- **yscv-optim**: Lookahead meta-optimizer wrapping any `StepOptimizer` with slow-weight interpolation.
- **yscv-tensor**: `scatter_add` operation for index-based additive scatter.
- **yscv-autograd**: Differentiable `gather` and `scatter_add` ops with full backward support.
- **yscv-recognize**: VP-Tree (vantage-point tree) for approximate nearest-neighbor search (`build_index()`, `search_indexed()`).
- **yscv-video**: H.264 P-slice motion compensation (`MotionVector`, `motion_compensate_16x16`, `ReferenceFrameBuffer`).
- **yscv-video**: H.264 B-slice bidirectional prediction (`BiMotionVector`, `BPredMode`, `motion_compensate_bipred`).
- **yscv-video**: H.264 deblocking filter (`boundary_strength`, `deblock_edge_luma`, `deblock_frame`).
- **yscv-video**: HEVC/H.265 decoder infrastructure (VPS/SPS/PPS parsing, `CodingTreeUnit`, `HevcSliceType`).
- **yscv-kernels**: Deformable Conv2d kernel (`deformable_conv2d_nhwc`) with bilinear sampling.
- **yscv-model**: `DeformableConv2dLayer` with `ModelLayer::DeformableConv2d` variant.
- **yscv-track**: Re-identification module (`ReIdExtractor` trait, `ColorHistogramReId`, `ReIdGallery`).
- **yscv-kernels**: GPU compute shaders for batch_norm, layer_norm, and transpose via wgpu.
- **yscv-imgproc**: SURF keypoint detection and descriptor matching (`detect_surf_keypoints`, `compute_surf_descriptors`, `match_surf_descriptors`).
- **yscv-onnx**: `OnnxDtype` enum (Float32/Float16/Int8/UInt8/Int32/Int64/Bool) with `OnnxTensorData` quantize/dequantize support.
- **yscv-model**: TCP transport for distributed training (`TcpTransport` with coordinator/worker roles, `send`/`recv`, `allreduce_sum`).
- **scripts**: `publish.sh` for dependency-ordered crate publishing.
- **scripts**: `bump-version.sh` for workspace-wide version bumps.
- **examples**: `train_cnn` — CNN training recipe with Conv2d + BatchNorm + pooling.
- **examples**: `image_pipeline` — composable image preprocessing pipeline.
- **yscv-model**: Pretrained model zoo with architecture builders (ResNet, VGG, MobileNetV2, EfficientNet, AlexNet) and `ModelHub` remote weight download with caching.
- **yscv-model**: Distributed training primitives — `GradientAggregator` trait, `AllReduceAggregator`, `ParameterServer`, `InProcessTransport`, gradient compression (`TopKCompressor`).
- **yscv-model**: High-level `Trainer` API with `TrainerConfig`, validation split, `EarlyStopping`, `BestModelCheckpoint` callbacks.
- **yscv-model**: Eval/train mode toggle for layers (dropout, batch norm behavior).
- **yscv-model**: Compose-based `Transform` pipeline (Resize, CenterCrop, Normalize, GaussianBlur, RandomHorizontalFlip, ScaleValues, PermuteDims).
- **yscv-kernels**: GPU multi-device scheduling — `MultiGpuBackend`, device enumeration, round-robin/data-parallel/manual scheduling strategies.
- **yscv-video**: H.264 baseline decoder infrastructure — SPS/PPS parsing, bitstream reader, Exp-Golomb decoding, YUV420-to-RGB8 conversion, H.265 NAL type classification.
- **yscv-tensor**: Native FP16/BF16 dtype support with `DType` enum, typed constructors, and `to_dtype()` conversion.
- **yscv-model**: Mixed-precision training (`MixedPrecisionConfig`, `DynamicLossScaler`, `mixed_precision_train_step`).
- **yscv-model**: Embedding, LayerNorm, GroupNorm, InstanceNorm layers with checkpoint roundtrip.
- **yscv-model**: LoRA fine-tuning, EMA, LR finder.
- **yscv-model**: SafeTensors format support.
- **yscv-onnx**: Quantized ONNX runtime ops (QLinearConv, QLinearMatMul, MatMulInteger, ConvInteger, DynamicQuantizeLinear).
- **yscv-onnx**: Expanded opset from 90 to 123 operations.
- **yscv-video**: H.264/H.265 codec infrastructure (NAL parser, MP4 box parser, VideoDecoder/VideoEncoder traits, CAVLC).
- **docs**: API stability policy and release governance (`docs/api-stability.md`).
- **docs**: Full documentation suite (ecosystem capability matrix, performance benchmarks, dataset adapters, training augmentation, training optimizers).

### Changed
- **yscv-tensor**: `DType` enum now supports F32, F16, and BF16 storage variants.
- **yscv-imgproc**: SURF descriptor matching accepts exact matches (dist < 1e-9) unconditionally, bypassing ratio test.
