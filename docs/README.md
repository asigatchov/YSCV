# yscv Documentation

This directory contains the technical documentation for the yscv framework. Each document serves a specific audience — whether you are a contributor writing new ops, an engineer evaluating performance, or an AI agent reasoning about project structure.

## Getting oriented

If you are new to the project, start with the root [README.md](../README.md) for a high-level overview of what yscv is and how to build it. Then come back here for the details.

## Documents

### [architecture.md](architecture.md)

How the framework is put together. Explains the crate dependency layers, the SIMD dispatch model (AVX/SSE/NEON with scalar fallback), the threading strategy (GCD on macOS, rayon everywhere), memory patterns, and a map of key source files. Start here if you want to understand how things connect before making changes.

### [performance-benchmarks.md](performance-benchmarks.md)

How we measure performance and how yscv compares to OpenCV. Covers the full benchmark methodology (hardware, measurement protocol, warm-up, statistical aggregation), CI regression gates with trend tracking, and the definitive OpenCV vs yscv comparison table (8 wins, 2 parity, 0 losses on 10 u8 imgproc operations). Includes exact commands to reproduce every number.

### [ecosystem-capability-matrix.md](ecosystem-capability-matrix.md)

The canonical map of what yscv can do today and what gaps remain relative to a full Python CV/DL stack (OpenCV + NumPy + PyTorch). Organized by capability area (tensor ops, autograd, optimizers, model layers, image processing, video, detection, tracking, ONNX, GPU, etc.) with status markers. This is the primary planning artifact for deciding what to build next.

### [api-stability.md](api-stability.md)

Versioning policy, stability tiers for each crate, the release checklist, and the publish order for the 15 crates in the workspace. Currently pre-1.0 (semver 0.x.y), so breaking changes are expected but tracked in the changelog.

### [training-optimizers.md](training-optimizers.md)

Reference for the training subsystem: 8 optimizers (SGD through LARS), the Lookahead meta-optimizer, 11 learning-rate schedulers, 14+ loss functions, gradient clipping utilities, and the high-level Trainer API that ties them together.

### [training-augmentation.md](training-augmentation.md)

The data augmentation pipeline for training: 12+ transform operations (flips, crops, jitter, noise, cutout), batch-level regularization (MixUp, CutMix), sampling policies, dataset splitting, and the reproducibility contract (deterministic via seed).

### [dataset-adapters.md](dataset-adapters.md)

Supported dataset formats for both training (JSONL, CSV, ImageManifest, ImageFolder) and evaluation (COCO, OpenImages, YOLO, VOC, KITTI, WIDER FACE, MOTChallenge, and more). Documents the field mapping rules and CLI integration.

### [native-camera-validation.md](native-camera-validation.md)

Step-by-step checklist for validating native camera capture on macOS, Linux, and Windows. Covers build verification, device discovery, capture diagnostics, and the end-to-end face detection pipeline.
