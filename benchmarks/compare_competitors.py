#!/usr/bin/env python3
"""Compare YSCV benchmark results against NumPy, PyTorch, OpenCV, and onnxruntime.

Run with: python3 benchmarks/compare_competitors.py
Requires: pip install numpy torch opencv-python-headless onnxruntime
"""

import time
import sys
import json

def bench(fn, warmup=5, runs=50):
    """Measure median time of fn() in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns -> µs
    times.sort()
    return times[len(times) // 2]

results = {}

# ══════════════════════════════════════════════════════════════════
# yscv-tensor vs NumPy
# ══════════════════════════════════════════════════════════════════
try:
    import numpy as np
    print("=== yscv-tensor vs NumPy", np.__version__, "===")

    # Tensor add (same shape) - 100K elements
    a = np.random.rand(100_000).astype(np.float32)
    b = np.random.rand(100_000).astype(np.float32)
    t = bench(lambda: a + b)
    results["numpy_add_100k"] = t
    print(f"  add 100K f32:          {t:.1f} µs")

    # Tensor mul (same shape)
    t = bench(lambda: a * b)
    results["numpy_mul_100k"] = t
    print(f"  mul 100K f32:          {t:.1f} µs")

    # Sum reduction
    a1m = np.random.rand(1_000_000).astype(np.float32)
    t = bench(lambda: np.sum(a1m))
    results["numpy_sum_1m"] = t
    print(f"  sum 1M f32:            {t:.1f} µs")

    # Argmax
    t = bench(lambda: np.argmax(a1m))
    results["numpy_argmax_1m"] = t
    print(f"  argmax 1M f32:         {t:.1f} µs")

    # Exp
    t = bench(lambda: np.exp(a1m))
    results["numpy_exp_1m"] = t
    print(f"  exp 1M f32:            {t:.1f} µs")

    # ReLU (maximum(0, x))
    t = bench(lambda: np.maximum(a1m, 0))
    results["numpy_relu_1m"] = t
    print(f"  relu 1M f32:           {t:.1f} µs")

    # Matmul 128x128
    m1 = np.random.rand(128, 128).astype(np.float32)
    m2 = np.random.rand(128, 128).astype(np.float32)
    t = bench(lambda: m1 @ m2)
    results["numpy_matmul_128"] = t
    print(f"  matmul 128x128:        {t:.1f} µs")

    # Broadcast add: [1M] + [1]
    scalar = np.float32(0.5)
    t = bench(lambda: a1m + scalar)
    results["numpy_broadcast_add_1m"] = t
    print(f"  broadcast add 1M+1:    {t:.1f} µs")

    # Sort 100K
    a100k = np.random.rand(100_000).astype(np.float32)
    t = bench(lambda: np.sort(a100k), runs=20)
    results["numpy_sort_100k"] = t
    print(f"  sort 100K f32:         {t:.1f} µs")

    # ── Additional elementwise ops (1M) ──
    a_1m = np.random.rand(1_000_000).astype(np.float32)
    b_1m = np.random.rand(1_000_000).astype(np.float32)

    t = bench(lambda: np.sqrt(a_1m))
    results["numpy_sqrt_1m"] = t
    print(f"  sqrt 1M f32:           {t:.1f} µs")

    t = bench(lambda: np.abs(a_1m))
    results["numpy_abs_1m"] = t
    print(f"  abs 1M f32:            {t:.1f} µs")

    t = bench(lambda: np.sin(a_1m))
    results["numpy_sin_1m"] = t
    print(f"  sin 1M f32:            {t:.1f} µs")

    t = bench(lambda: np.floor(a_1m))
    results["numpy_floor_1m"] = t
    print(f"  floor 1M f32:          {t:.1f} µs")

    # ── Additional reductions (1M) ──
    t = bench(lambda: np.min(a_1m))
    results["numpy_min_1m"] = t
    print(f"  min 1M f32:            {t:.1f} µs")

    t = bench(lambda: np.max(a_1m))
    results["numpy_max_1m"] = t
    print(f"  max 1M f32:            {t:.1f} µs")

    t = bench(lambda: np.prod(a_1m))
    results["numpy_prod_1m"] = t
    print(f"  prod 1M f32:           {t:.1f} µs")

    print()
except ImportError:
    print("NumPy not available, skipping\n")

# ══════════════════════════════════════════════════════════════════
# yscv-kernels vs PyTorch (kernel-level ops)
# ══════════════════════════════════════════════════════════════════
try:
    import torch
    print("=== yscv-kernels vs PyTorch", torch.__version__, "(CPU) ===")
    torch.set_num_threads(1)  # single-threaded for fair comparison

    # Tensor add
    ta = torch.rand(100_000, dtype=torch.float32)
    tb = torch.rand(100_000, dtype=torch.float32)
    t = bench(lambda: ta + tb)
    results["pytorch_add_100k"] = t
    print(f"  add 100K f32:          {t:.1f} µs")

    # Softmax
    sm_input = torch.rand(512, 256, dtype=torch.float32)
    t = bench(lambda: torch.softmax(sm_input, dim=-1))
    results["pytorch_softmax_512x256"] = t
    print(f"  softmax 512x256:       {t:.1f} µs")

    # Matmul 128x128
    tm1 = torch.rand(128, 128, dtype=torch.float32)
    tm2 = torch.rand(128, 128, dtype=torch.float32)
    t = bench(lambda: tm1 @ tm2)
    results["pytorch_matmul_128"] = t
    print(f"  matmul 128x128:        {t:.1f} µs")

    # Conv2d
    conv_input = torch.rand(1, 8, 64, 64, dtype=torch.float32)
    conv = torch.nn.Conv2d(8, 16, 3, padding=0, bias=True)
    conv.eval()
    with torch.no_grad():
        t = bench(lambda: conv(conv_input))
    results["pytorch_conv2d_64x64_8to16"] = t
    print(f"  conv2d 64x64 8->16:    {t:.1f} µs")

    # ReLU
    t1m = torch.rand(1_000_000, dtype=torch.float32)
    t = bench(lambda: torch.relu(t1m))
    results["pytorch_relu_1m"] = t
    print(f"  relu 1M f32:           {t:.1f} µs")

    # GELU
    t = bench(lambda: torch.nn.functional.gelu(t1m))
    results["pytorch_gelu_1m"] = t
    print(f"  gelu 1M f32:           {t:.1f} µs")

    # Sigmoid
    t = bench(lambda: torch.sigmoid(t1m))
    results["pytorch_sigmoid_1m"] = t
    print(f"  sigmoid 1M f32:        {t:.1f} µs")

    # LayerNorm
    ln_input = torch.rand(512, 256, dtype=torch.float32)
    ln = torch.nn.LayerNorm(256)
    ln.eval()
    with torch.no_grad():
        t = bench(lambda: ln(ln_input))
    results["pytorch_layernorm_512x256"] = t
    print(f"  layernorm 512x256:     {t:.1f} µs")

    # BatchNorm
    bn_input = torch.rand(2, 16, 64, 64, dtype=torch.float32)
    bn = torch.nn.BatchNorm2d(16)
    bn.eval()
    with torch.no_grad():
        t = bench(lambda: bn(bn_input))
    results["pytorch_batchnorm_64x64x16"] = t
    print(f"  batchnorm 64x64x16:    {t:.1f} µs")

    # MaxPool2d
    pool_input = torch.rand(2, 3, 120, 160, dtype=torch.float32)
    pool = torch.nn.MaxPool2d(2, stride=2)
    with torch.no_grad():
        t = bench(lambda: pool(pool_input))
    results["pytorch_maxpool_120x160"] = t
    print(f"  maxpool 120x160 k2s2:  {t:.1f} µs")

    print()
except ImportError:
    print("PyTorch not available, skipping\n")

# ══════════════════════════════════════════════════════════════════
# yscv-imgproc vs OpenCV
# ══════════════════════════════════════════════════════════════════
try:
    import cv2
    import numpy as np
    print("=== yscv-imgproc vs OpenCV", cv2.__version__, "===")

    # RGB to grayscale
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    t = bench(lambda: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    results["opencv_grayscale_640x480"] = t
    print(f"  grayscale 640x480:     {t:.1f} µs")

    # Sobel
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t = bench(lambda: cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    results["opencv_sobel_640x480"] = t
    print(f"  sobel 640x480:         {t:.1f} µs")

    # Gaussian blur
    t = bench(lambda: cv2.GaussianBlur(gray, (3, 3), 0))
    results["opencv_gaussblur_640x480"] = t
    print(f"  gauss blur 640x480:    {t:.1f} µs")

    # Box blur
    t = bench(lambda: cv2.blur(gray, (3, 3)))
    results["opencv_boxblur_640x480"] = t
    print(f"  box blur 640x480:      {t:.1f} µs")

    # Canny
    t = bench(lambda: cv2.Canny(gray, 50, 150))
    results["opencv_canny_640x480"] = t
    print(f"  canny 640x480:         {t:.1f} µs")

    # Resize nearest
    small = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    t = bench(lambda: cv2.resize(small, (640, 480), interpolation=cv2.INTER_NEAREST))
    results["opencv_resize_nearest_320x240_to_640x480"] = t
    print(f"  resize nearest 320->640:{t:.1f} µs")

    # Resize bilinear
    t = bench(lambda: cv2.resize(small, (640, 480), interpolation=cv2.INTER_LINEAR))
    results["opencv_resize_bilinear_320x240_to_640x480"] = t
    print(f"  resize bilinear 320->640:{t:.1f} µs")

    # Dilate
    kernel = np.ones((3, 3), np.uint8)
    t = bench(lambda: cv2.dilate(gray, kernel))
    results["opencv_dilate_640x480"] = t
    print(f"  dilate 640x480:        {t:.1f} µs")

    # Erode
    t = bench(lambda: cv2.erode(gray, kernel))
    results["opencv_erode_640x480"] = t
    print(f"  erode 640x480:         {t:.1f} µs")

    # FAST feature detection
    fast = cv2.FastFeatureDetector_create(threshold=20)
    t = bench(lambda: fast.detect(gray))
    results["opencv_fast_640x480"] = t
    print(f"  FAST detect 640x480:   {t:.1f} µs")

    # ORB
    orb = cv2.ORB_create(nfeatures=500)
    t = bench(lambda: orb.detectAndCompute(gray, None), runs=20)
    results["opencv_orb_640x480"] = t
    print(f"  ORB detect 640x480:    {t:.1f} µs")

    # ── Extended u8 ops ──
    # Median blur
    t = bench(lambda: cv2.medianBlur(gray, 3))
    results["opencv_medianblur_640x480"] = t
    print(f"  median blur 640x480:   {t:.1f} µs")

    # Gaussian blur (already above, but kept for u8 section completeness)
    t = bench(lambda: cv2.GaussianBlur(gray, (3, 3), 0))
    results["opencv_gaussblur_u8_640x480"] = t
    print(f"  gauss blur u8 640x480: {t:.1f} µs")

    # Morphological opening
    t = bench(lambda: cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel))
    results["opencv_morph_open_640x480"] = t
    print(f"  morph open 640x480:    {t:.1f} µs")

    # Morphological closing
    t = bench(lambda: cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel))
    results["opencv_morph_close_640x480"] = t
    print(f"  morph close 640x480:   {t:.1f} µs")

    print()
except ImportError:
    print("OpenCV not available, skipping\n")

# ══════════════════════════════════════════════════════════════════
# yscv-model vs PyTorch nn
# ══════════════════════════════════════════════════════════════════
try:
    import torch
    print("=== yscv-model vs PyTorch nn", torch.__version__, "(CPU) ===")
    torch.set_num_threads(1)

    # Linear + ReLU + Linear forward
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32)
    )
    model.eval()
    x_model = torch.rand(32, 64)
    with torch.no_grad():
        t = bench(lambda: model(x_model))
    results["pytorch_nn_lin_relu_lin_fwd"] = t
    print(f"  Linear+ReLU+Linear fwd (32x64): {t:.1f} µs")

    # SGD training step
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    target = torch.rand(32, 32)
    def train_step_sgd():
        optimizer.zero_grad()
        pred = model(x_model)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
    t = bench(train_step_sgd)
    results["pytorch_nn_sgd_train_step"] = t
    print(f"  SGD train step (32x64):          {t:.1f} µs")

    print()
except ImportError:
    print("PyTorch not available, skipping\n")

# ══════════════════════════════════════════════════════════════════
# yscv-autograd vs PyTorch autograd
# ══════════════════════════════════════════════════════════════════
try:
    import torch
    print("=== yscv-autograd vs PyTorch autograd", torch.__version__, "(CPU) ===")
    torch.set_num_threads(1)

    # Simple forward + backward timing
    x_ag = torch.rand(32, 64, requires_grad=True)
    w_ag = torch.rand(64, 32, requires_grad=True)

    def autograd_bench():
        y = x_ag @ w_ag
        y = torch.relu(y)
        loss = y.sum()
        loss.backward()
        x_ag.grad = None
        w_ag.grad = None

    t = bench(autograd_bench)
    results["pytorch_autograd_matmul_relu_backward"] = t
    print(f"  matmul+relu+sum backward (32x64 @ 64x32): {t:.1f} µs")

    print()
except ImportError:
    print("PyTorch not available, skipping\n")

# ══════════════════════════════════════════════════════════════════
# yscv-onnx vs onnxruntime
# ══════════════════════════════════════════════════════════════════
try:
    import onnxruntime as ort
    import numpy as np
    print("=== yscv-onnx vs onnxruntime", ort.__version__, "===")

    # Try to create a simple ONNX model via PyTorch export
    _onnx_model_path = "/tmp/_bench_simple_model.onnx"
    _onnx_ready = False
    try:
        import torch
        _m = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        _m.eval()
        _dummy = torch.rand(1, 64)
        torch.onnx.export(_m, _dummy, _onnx_model_path,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                          opset_version=13)
        _onnx_ready = True
    except Exception as e:
        print(f"  (could not export ONNX model: {e})")

    if _onnx_ready:
        sess = ort.InferenceSession(_onnx_model_path,
                                    providers=["CPUExecutionProvider"])
        ort_input = np.random.rand(32, 64).astype(np.float32)

        def ort_infer():
            sess.run(None, {"input": ort_input})

        t = bench(ort_infer)
        results["ort_lin_relu_lin_32x64"] = t
        print(f"  Linear+ReLU+Linear inference (32x64): {t:.1f} µs")

        # Cleanup temp model
        import os
        try:
            os.remove(_onnx_model_path)
        except OSError:
            pass

    print()
except ImportError:
    print("onnxruntime not available, skipping\n")

# ══════════════════════════════════════════════════════════════════
# yscv-detect + yscv-track + yscv-recognize (no direct competitor)
# ══════════════════════════════════════════════════════════════════
print("=== yscv-detect + yscv-track + yscv-recognize ===")
print("  No direct Python competitor — these are integrated Rust pipelines.")
print("  YSCV people detect+track+recognize: 67 µs (from criterion)")
print("  YSCV face detect+track+recognize: 160 µs (from criterion)")
print()

# ── Print JSON for automated comparison ───────────────────────────
print("=== JSON Results ===")
print(json.dumps(results, indent=2))
