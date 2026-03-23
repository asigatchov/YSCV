# Architecture Guide

This document explains how the yscv framework is put together — the crate dependency structure, the SIMD dispatch model, the threading strategy, and the key design patterns used throughout the codebase. It is written for contributors who need to understand the system before making changes, and for AI agents that need to reason about where code lives and how it connects.

## Crate layers

The crates form a layered architecture. Lower layers know nothing about higher layers.

**Layer 0 — Foundation:**
`yscv-tensor` provides the `Tensor` type with 115+ operations, f32/f16/bf16 dtype support, operator overloading (`+`, `-`, `*`, `/`), `Display` impl, and SIMD-accelerated reductions. Everything else depends on this.

**Layer 1 — Compute:**
`yscv-kernels` provides the `Backend` trait with 50+ methods (conv2d, matmul, pool, normalization, activation, backward ops). It has a `CpuBackend` with rayon parallelism and SIMD dispatch, a `ThreadedCpuBackend`, and an optional `GpuBackend` using wgpu compute shaders (20 WGSL shaders including backward kernels). The SIMD code lives in `crates/yscv-kernels/src/ops/simd.rs` (3000+ lines) with AVX, SSE, and NEON implementations for every kernel.

**Layer 2 — Autograd and Optimization:**
`yscv-autograd` builds on kernels to provide a dynamic computation graph with tape-based reverse-mode autodiff. `yscv-optim` provides optimizers and schedulers.

**Layer 3 — Model and Training:**
`yscv-model` combines autograd, kernels, and optim into a high-level training API with 39 layer types (25 trainable), the `Trainer` helper, model zoo (13 architectures), TensorBoard logging, StreamingDataLoader, LoRA, EMA, mixed precision, distributed training (AllReduce + pipeline parallel + tensor sharding), and gradient clipping.

**Layer 4 — Domain:**
`yscv-imgproc` (image processing), `yscv-video` (codecs and camera), `yscv-detect` (YOLOv8), `yscv-track` (DeepSORT/ByteTrack), `yscv-recognize` (VP-Tree matching), `yscv-eval` (metrics), and `yscv-onnx` (128+ op runtime) each handle a specific domain. They depend on the foundation but not on each other (except detect → video for frame types, track → detect for detection types).

**Layer 5 — Applications:**
`yscv-cli` and `camera-face-tool` are end-to-end binaries that wire everything together.

## SIMD dispatch model

There are two distinct SIMD systems in the project.

### f32 operations (yscv-kernels, yscv-tensor)

These use a three-tier dispatch: AVX → SSE → NEON → scalar. The pattern is always the same:

```rust
pub fn relu_slice_dispatch(data: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") { return unsafe { relu_avx(data) }; }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse") { return unsafe { relu_sse(data) }; }
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") { return unsafe { relu_neon(data) }; }
    relu_scalar(data);
}
```

Each implementation is marked with `#[target_feature(enable = "...")]` so the compiler generates appropriate instructions. The feature detection call is cached after the first invocation (atomic load, ~1ns).

### u8 image operations (yscv-imgproc)

These use a three-tier dispatch: NEON or AVX2 or SSSE3/SSE2 → scalar. The operations are more specialized than f32 kernels and use architecture-specific tricks:

- **aarch64**: `vextq_u8` for byte shifts, `vld3q_u8` for RGB deinterleave, `vqtbl1q_u8` for gather
- **x86_64 AVX2**: `_mm256_max_epu8`/`_mm256_min_epu8` for morphology, `_mm256_maddubs_epi16` for grayscale
- **x86_64 SSE**: `_mm_alignr_epi8` (SSSE3) for byte shifts, `_mm_shuffle_epi8` for gather, manual deinterleave for RGB

All u8 sub-operations (grayscale, dilate, erode, gaussian, box blur, sobel, median, canny, resize nearest, resize bilinear) have NEON, AVX2/SSE, and scalar implementations.

## Threading model

### macOS: GCD

On macOS, the project uses Apple's Grand Central Dispatch directly via FFI (`dispatch_apply_f`). This is the same primitive OpenCV uses. Dispatch latency is ~0.3 us compared to rayon's ~3-5 us, which matters for sub-millisecond operations on small images.

The GCD wrapper lives in `crates/yscv-imgproc/src/ops/u8ops.rs` as a small `mod gcd` with `parallel_for<F: Fn(usize) + Sync>`.

### Linux/Windows: scoped threads

On non-macOS platforms, `std::thread::scope` with `available_parallelism()` provides ~1 us dispatch overhead — faster than rayon's ~3-5 us for small fixed-iteration parallel_for patterns. The implementation lives in the same `mod gcd` module, selected via `#[cfg(not(target_os = "macos"))]`.

### All platforms: rayon

Rayon is the cross-platform parallel backend. It provides work-stealing thread pool scheduling. Used for operations where rayon's chunking is natural (kernel elementwise ops, large batch processing).

### Parallelism threshold

`RAYON_THRESHOLD = 4096` pixels. Below this, operations run sequentially to avoid thread dispatch overhead. This matters because many image operations take only 20-100 us on small images, and thread wake-up alone can cost 3-5 us.

## Memory patterns

- **mimalloc** global allocator in benchmark harness for faster large allocations.
- `AlignedVec` in yscv-tensor (32-byte aligned for AVX) with `uninitialized(len)` to skip zeroing output buffers.
- **Ring buffers** for streaming row processing (canny magnitude/direction, morph separable passes).
- **Thread-local scratch** via rayon's per-task closures (not explicit thread-local storage).
- **Zero-copy boundaries**: `Bytes`-backed frames in video, slice-based tensor views, caller-owned output buffers in detection/tracking.

## Key files

If you need to change something, these are the most important files:

| What | Where |
|---|---|
| u8 image ops (SIMD) | `crates/yscv-imgproc/src/ops/u8ops.rs` (~5500 lines) |
| f32 SIMD kernels | `crates/yscv-kernels/src/ops/simd.rs` (~3000 lines) |
| Tensor SIMD | `crates/yscv-tensor/src/simd.rs` (~3000 lines) |
| Tensor core | `crates/yscv-tensor/src/tensor.rs` |
| Autograd graph | `crates/yscv-autograd/src/graph.rs` |
| CPU backend | `crates/yscv-kernels/src/backend.rs` |
| ONNX runtime | `crates/yscv-onnx/src/runtime.rs` |
| Benchmark harness | `apps/bench/src/main.rs` |
| OpenCV comparison | `bench_opencv.py` |
| CI pipeline | `.github/workflows/ci.yml` |
| Release config | `Cargo.toml` (workspace root), `.cargo/config.toml` |

## Cross-platform optimization matrix

| Feature | macOS (aarch64) | Linux (aarch64) | Linux/Windows (x86_64) |
|---------|:-:|:-:|:-:|
| Threading | GCD dispatch_apply | std::thread::scope | std::thread::scope |
| SIMD f32 ops | NEON 4× unroll | NEON 4× unroll | AVX 4× unroll + SSE |
| SIMD u8 ops | NEON | NEON | AVX2 + SSE2/SSSE3 |
| Sigmoid/tanh | NEON 3-term poly | NEON 3-term poly | AVX/SSE poly |
| Vectorized math | vDSP (Accelerate) | ARMPL (opt-in `armpl`) | MKL VML (opt-in `mkl`) |
| MatMul BLAS | Accelerate cblas | OpenBLAS | OpenBLAS |
| Softmax | Fused NEON | Fused NEON | Fused AVX/SSE |
| Median u8 | NEON sort network | NEON sort network | SSE2 sort network |
| Allocator | mimalloc | mimalloc | mimalloc |

All SIMD dispatch paths include scalar fallback for architectures without runtime detection support (e.g., RISC-V, WASM) and for Miri testing.

## Safety invariants

All `unsafe` blocks follow these rules:
1. Every block has a `// SAFETY:` comment explaining why invariants hold
2. `debug_assert!` bounds checks guard FFI boundaries (BLAS sgemm, conv2d direct)
3. `AlignedVec::uninitialized` callers must write every element before read
4. Pointer arithmetic in inner loops is bounded by plan/dimension validation at function entry
5. `#[target_feature]` functions are only called after runtime feature detection
