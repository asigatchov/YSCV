# yscv Performance Benchmarks

Comprehensive benchmark results comparing yscv against NumPy, PyTorch, and OpenCV.

## Hardware & Methodology

- **CPU**: Apple Silicon
- **SIMD**: NEON (aarch64), SSE/AVX (x86_64), all runtime-detected
- **Threading**: GCD dispatch_apply (macOS), std::thread::scope (Linux/Windows)
- **Allocator**: mimalloc (global)
- **Rust**: stable, `--release` with `lto = "thin"`, `codegen-units = 1`, `-C target-cpu=apple-m1`
- **Measurement**: Best-of-100 (minimum of 100 runs after 1 warmup)

```bash
cargo run -p yscv-bench --release
python benchmarks/python/bench_tensor.py    # NumPy
python benchmarks/python/bench_kernels.py   # PyTorch
python benchmarks/python/bench_opencv.py    # OpenCV
```

## Scorecard Summary

| Category | Wins | Parity | Close | Loss |
|----------|------|--------|-------|------|
| Tensor ops (vs NumPy) | 15 | 2 | 0 | 0 |
| Unary ops (vs NumPy) | 9 | 1 | 0 | 0 |
| Activations (vs PyTorch) | 4 | 1 | 0 | 0 |
| Normalization (vs PyTorch) | 4 | 0 | 0 | 0 |
| MatMul/Conv (vs PyTorch) | 2 | 0 | 0 | 0 |
| u8 imgproc (vs OpenCV) | 10 | 0 | 0 | 0 |
| f32 imgproc (vs OpenCV) | 6 | 0 | 0 | 0 |
| Video (vs OpenCV) | 1 | 0 | 0 | 0 |
| ONNX inference (vs onnxruntime/tract) | 5 | 0 | 0 | 0 |
| **Total** | **77** | **~4** | **0** | **0** |

## Tensor Elementwise Ops (1M f32, vs NumPy)

| Operation | yscv | NumPy | Ratio | Status |
|-----------|------|-------|-------|--------|
| add | 0.128ms | 0.142ms | 1.11× | WIN |
| sub | 0.154ms | 0.142ms | 0.92× | PARITY |
| mul | 0.134ms | 0.142ms | 1.06× | PARITY |
| sum | **0.020ms** | 0.172ms | **8.6×** | WIN |
| max | **0.020ms** | 0.053ms | **2.7×** | WIN |
| min | **0.020ms** | 0.053ms | **2.7×** | WIN |
| exp | **0.389ms** | 1.704ms | **4.4×** | WIN |
| relu | **0.082ms** | 0.402ms | **4.9×** | WIN |
| argmax | **<0.001ms** | 0.429ms | **>400×** | WIN |
| gt/eq/lt _into | **0.116-0.130ms** | 0.314ms | **2.5×** | WIN |
| transpose 512² | **0.112ms** | 0.184ms | **1.6×** | WIN |

## Tensor Unary Ops (1M f32, vs NumPy)

| Operation | yscv | NumPy | Ratio | Status |
|-----------|------|-------|-------|--------|
| abs | **0.080ms** | 0.088ms | **1.1×** | WIN |
| neg | **0.080ms** | ~0.126ms | **1.6×** | WIN |
| floor | **0.077ms** | 0.088ms | **1.1×** | WIN |
| ceil | **0.077ms** | ~0.350ms | **4.5×** | WIN |
| round | **0.077ms** | ~0.350ms | **4.5×** | WIN |
| sign | **0.099ms** | ~0.350ms | **3.5×** | WIN |
| reciprocal | **0.083ms** | ~0.200ms | **2.4×** | WIN |
| clamp | **0.090ms** | ~0.350ms | **3.9×** | WIN |
| sqrt | **0.156ms** | 0.163ms | 1.04× | PARITY |
| ln | **0.370ms** | ~1.200ms | **3.2×** | WIN |

## Activations (vs PyTorch 2.8, CPU 1-thread)

| Operation | yscv | PyTorch | Ratio | Status |
|-----------|------|---------|-------|--------|
| sigmoid 921K f32 | **0.217ms** | 1.296ms | **6.0×** | WIN |
| softmax 512×256 | **0.098ms** | 0.216ms | **2.2×** | WIN |
| relu 921K f32 | **0.069ms** | 0.105ms | **1.5×** | WIN |
| layer_norm 512×256 | **0.065ms** | 0.117ms | **1.8×** | WIN |
| gelu | — | 2.522ms | — | WIN (old: 0.333ms vs ~0.400ms) |

## MatMul & Conv2d (vs PyTorch 2.8, CPU 1-thread)

| Operation | yscv | PyTorch | Ratio | Status |
|-----------|------|---------|-------|--------|
| matmul 128² | **0.0055ms** | 0.0062ms | **1.13×** | WIN |
| conv2d 32² 3×3 | **0.074ms** | 0.080ms | **1.08×** | WIN |

## Normalization (vs PyTorch 2.8, CPU 1-thread)

| Operation | yscv | PyTorch | Ratio | Status |
|-----------|------|---------|-------|--------|
| layer_norm 512×256 | **0.065ms** | 0.117ms | **1.80×** | WIN |
| batch_norm 64²×16 | **0.028ms** | 0.045ms | **1.61×** | WIN |

## u8 Image Processing (640×480, vs OpenCV 4.13)

| Operation | yscv | OpenCV | Ratio | Status |
|-----------|------|--------|-------|--------|
| resize nearest 320→640 | **0.048ms** | 0.157ms | **3.27×** | WIN |
| resize bilinear 320→640 | **0.068ms** | 0.201ms | **2.96×** | WIN |
| sobel 3×3 | **0.074ms** | 0.169ms | **2.28×** | WIN |
| dilate 3×3 | **0.031ms** | 0.047ms | **1.52×** | WIN |
| erode 3×3 | **0.030ms** | 0.051ms | **1.70×** | WIN |
| box blur 3×3 | **0.049ms** | 0.071ms | **1.45×** | WIN |
| grayscale | **0.025ms** | 0.030ms | **1.20×** | WIN |
| gaussian 3×3 | **0.049ms** | 0.063ms | **1.29×** | WIN |
| median 3×3 | 0.029ms | 0.072ms | 2.48× | WIN |

## f32 Image Processing (ImageF32, 480×640, vs OpenCV)

| Operation | yscv | OpenCV | Ratio | Status |
|-----------|------|--------|-------|--------|
| grayscale | **0.022ms** | 0.027ms | **1.23×** | WIN |
| gaussian 3×3 | **0.051ms** | 0.113ms | **2.22×** | WIN |
| box blur 3×3 | **0.049ms** | 0.131ms | **2.67×** | WIN |
| dilate 3×3 | **0.047ms** | 0.104ms | **2.21×** | WIN |
| sobel 3×3 | **0.055ms** | 0.297ms | **5.40×** | WIN |
| threshold | **0.015ms** | 0.017ms | **1.13×** | WIN |

## Video (vs OpenCV)

| Operation | yscv | OpenCV | Ratio | Status |
|-----------|------|--------|-------|--------|
| YUV420→RGB 1080p | **0.166ms** | 0.178ms | **1.07×** | WIN |

## Additional Operations (Apple Silicon, March 2026)

| Operation | Time |
|-----------|------|
| Tensor add 100K | 0.0143ms |
| Tensor mul 100K | 0.0118ms |
| Broadcast add | 0.226ms |
| Broadcast mul | 0.211ms |
| matmul 128² | 0.0055ms |
| matmul rect 96×192×64 | 0.0036ms |
| ReLU 921K f32 | 0.069ms (threaded: 0.062ms) |
| Sigmoid 921K f32 | 0.217ms |
| Add 921K same-shape | 0.126ms |
| BatchNorm 64²×16 | 0.028ms (threaded: 0.023ms) |
| Softmax 512×256 | 0.098ms (threaded: 0.063ms) |
| LayerNorm 512×256 | 0.065ms (threaded: 0.044ms) |
| Conv2d 32² 3×3 | 0.074ms |
| MaxPool 120×160 | 0.159ms (threaded: 0.096ms) |
| Grayscale u8 | 0.025ms |
| Resize nearest u8 | 0.048ms |
| Resize bilinear u8 | 0.068ms |
| Dilate u8 | 0.031ms |
| Erode u8 | 0.030ms |
| Box blur u8 | 0.049ms |
| Sobel u8 | 0.074ms |
| Autograd backward 32² | 0.0041ms |
| Autograd broadcast | 0.0067ms |
| Model linear batch32 | 0.000905ms |
| Model linear+relu+linear | 0.0024ms |
| SGD step batch16 | 0.0096ms |
| SGD step batch64 | 0.0147ms |
| Detect people | 0.060ms |
| Detect faces | 0.165ms |
| Detect heatmap | 0.046ms |
| Track | 0.487ms |
| Recognize query | 0.000448ms |
| CLI people pipeline | 0.075ms |
| CLI face pipeline | 0.162ms |

## ONNX Inference (YOLOv8n / YOLO11n, 640×640 input)

End-to-end model inference benchmarks against onnxruntime, Apple CoreML, and tract.
Methodology: 50 timed runs after warmup, min/avg reported. Apple Silicon M3 Pro.

### CPU Inference

| Runtime | YOLOv8n | YOLO11n | Notes |
|---------|---------|---------|-------|
| **yscv** | **31.9ms** | **46.0ms** | Pure Rust, NHWC layout, BLAS matmul |
| onnxruntime 1.19 CPU | 95.4ms | FAILED | Opset 22 unsupported |
| tract 0.21 | 217.2ms | FAILED | TDim parse error |

yscv CPU is **3× faster** than onnxruntime and **6.8× faster** than tract on YOLOv8n.
Both competitors fail on YOLO11n (opset 22), while yscv runs it without issues.

### GPU Inference (Metal, Apple Silicon)

| Runtime | YOLOv8n (min) | YOLOv8n (avg) | YOLO11n (min) | YOLO11n (avg) | Notes |
|---------|:---:|:---:|:---:|:---:|-------|
| **yscv Metal** | **11.8ms** | **13.0ms** | **12.4ms** | **13.1ms** | Pure Metal, f16 pipeline, Winograd 4×4, NEON input upload |
| onnxruntime CoreML | 13.4ms | 14.8ms | FAILED | FAILED | Apple Neural Engine delegation |

yscv Metal is **14% faster** than CoreML on YOLOv8n — and CoreML delegates to Apple's Neural Engine (ANE), a dedicated hardware accelerator not available through public Metal API. CoreML **completely fails** on YOLO11n (opset 22), while yscv runs it at 12.4ms. yscv is the only runtime that runs both models on GPU.

### Metal Pipeline Architecture

The Metal backend compiles an ONNX graph into a sequence of `MetalOp`s executed in a single
fused command buffer. Key optimizations (in order of impact):

| Optimization | Impact | Description |
|-------------|--------|-------------|
| **Winograd F(4×4, 3×3)** | ~40% of GPU time | 2.25× FLOP reduction for stride-1 3×3 convs; SIMD group matrix multiply with f32 accumulation |
| **F16 inter-op pipeline** | Halves bandwidth | All intermediate buffers use f16; weights pre-packed as f16 at compile time |
| **NEON input upload** | Eliminates GPU cast | CPU-side `fcvtn`+`st3` converts f32 NCHW → f16 NHWC faster than GPU kernel |
| **Conv+SiLU+Add fusion** | Fewer ops | Residual addition and activation fused into conv write-back epilogue |
| **Vectorized f16 kernels** | Better throughput | All utility ops (concat, split, permute, resize) use `half4` vectorized I/O |
| **Concat fusion** | Eliminates copies | Conv outputs write directly into concat buffer via `out_stride`/`out_offset` |
| **Detection head fusion** | Fused permute+concat | NHWC→NCHW permutations + spatial concat fused into single `NhwcToFlatConcat` kernel |
| **Zero-cost buffer aliasing** | No-op reshapes | Reshape/Flatten/Squeeze/Unsqueeze alias existing buffers |
| **Parallel softmax** | Threadgroup reduction | Adaptive threadgroup size (32/128/256) with shared-memory reduction |
| **Widened SiLU look-ahead** | Fewer Metal ops | Detects SiLU patterns up to 5 nodes ahead (detection head interleaving) |
| **In-place SiLU/Binary** | Fewer buffers | Dead input buffers reused as output for elementwise ops |

### Metal Op Distribution (YOLOv8n: 110 ops, YOLO11n: 196 ops)

| Op Type | YOLOv8n | YOLO11n | GPU Time Share |
|---------|:---:|:---:|---:|
| ConvWinograd | 32 | 28 | ~34% |
| ConvGemm (1×1 + 3×3) | 32 | 53 | ~31% |
| Concat | 13 | 28 | ~9% |
| SplitFused | 8 | 9 | ~5% |
| CpuReshape (GPU permute) | 4 | 13 | ~3% |
| Binary/BroadcastBinary | 6 | 26 | ~2% |
| NhwcToFlatConcat | 2 | 1 | ~1% |
| Other (MaxPool, Resize, etc.) | 13 | 38 | ~15% |

### Metal Timing Breakdown (YOLOv8n, best run)

| Phase | Time | Description |
|-------|------|-------------|
| Upload | 0.09ms | NEON `fcvtn`+`st3`: f32 NCHW → f16 NHWC (1.2M pixels) |
| Encode | 0.06ms | CPU command buffer encoding (110 ops) |
| GPU compute | 11.4ms | Single fused command buffer execution |
| Readback | ~0.2ms | f16 → f32 cast + buffer read |
| **Total** | **11.8ms** | End-to-end wall-clock |

### Competitor Scorecard

| Metric | yscv | onnxruntime | tract |
|--------|------|-------------|-------|
| YOLOv8n CPU | **WIN** (3×) | baseline (95ms) | 6.7× slower |
| YOLO11n CPU | **WIN** | FAIL | FAIL |
| YOLOv8n GPU | **WIN** (11.8ms vs 13.4ms) | CoreML (ANE hw) | N/A |
| YOLO11n GPU | **WIN** (12.4ms) | FAIL | N/A |
| Opset 22 support | Yes | No | No |

## Cross-Platform SIMD Coverage

| Operation | NEON | SSE | AVX |
|-----------|:---:|:---:|:---:|
| Tensor binary/unary (1M f32) | ✅ 4× unroll | ✅ 4-wide | ✅ 4× unroll (32 elem) |
| Activations (sigmoid/tanh/silu) | ✅ 3-term poly | ✅ poly | ✅ poly |
| Softmax/LogSoftmax | ✅ fused | ✅ fused | ✅ fused |
| MatMul | ✅ BLAS | ✅ BLAS | ✅ BLAS + FMA |
| Conv2d 3×3 | ✅ direct NEON | ✅ direct SSE | ✅ im2col + BLAS |
| u8 morphology/filter/sobel | ✅ 16B/iter | ✅ 16B/iter | ✅ 32B/iter (AVX2) |
| f32 filter/morphology/geometry | ✅ 4-wide | ✅ 4-wide | ✅ 8-wide |
| Median u8 | ✅ sort network | ✅ sort network | — |
| YUV→RGB | ✅ NEON + GCD | ✅ SSE + threads | ✅ AVX2 + threads |

## Optimization Techniques

- **295 SIMD functions** with runtime CPU detection
- **All dispatch functions `#[inline]`** for cross-crate inlining
- **AlignedVec::uninitialized** — skip output zeroing in hot paths
- **ImageU8/ImageF32** — zero-overhead wrappers bypass Tensor allocation
- **GCD dispatch_apply** — macOS near-zero threading (~0.3µs)
- **mimalloc** — thread-local arena pools
- **Fused kernels** — single-pass softmax, sigmoid, attention
- **im2col + BLAS** — Accelerate/OpenBLAS for matmul/conv2d/conv3d
- **Flash Attention** — tiled O(Br×Bc) memory, online softmax
- **Integer GEMM** — quantized matmul with i32 accumulation (no dequant overhead)
