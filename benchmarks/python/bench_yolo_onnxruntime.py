"""Benchmark ONNX Runtime on YOLOv8n and YOLO11n — direct comparison target for yscv."""

import numpy as np
import time
import os
import sys

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
except ImportError:
    print("onnxruntime not installed. pip install onnxruntime")
    sys.exit(1)

MODELS = [
    "examples/src/slowwork/yolov8n.onnx",
    "examples/src/slowwork/yolo11n.onnx",
]

# Same input as our Rust benchmark
INPUT_SHAPE = (1, 3, 640, 640)
WARMUP = 3
RUNS = 20


def bench_model(model_path, providers, provider_name):
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found")
        return

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    # intra_op_num_threads: 0 = use all cores, set to match our RAYON_NUM_THREADS=4
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        sess = ort.InferenceSession(model_path, opts, providers=providers)
    except Exception as e:
        print(f"    FAILED to load: {e}")
        return None
    input_name = sess.get_inputs()[0].name

    # Use 0.5 fill like our Rust bench
    dummy = np.full(INPUT_SHAPE, 0.5, dtype=np.float32)

    # Warmup
    for _ in range(WARMUP):
        sess.run(None, {input_name: dummy})

    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        times.append(ms)
        shapes = [o.shape for o in out]
        print(f"    Run {i+1}: {ms:.1f}ms {shapes}")

    avg = sum(times) / len(times)
    mn = min(times)
    print(f"    Avg: {avg:.1f}ms  Min: {mn:.1f}ms")
    return mn


def main():
    # Find repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..", "..")

    print(f"\nInput shape: {INPUT_SHAPE}")
    print(f"Warmup: {WARMUP}, Runs: {RUNS}")
    print(f"intra_op_num_threads: 4")
    print()

    results = {}

    for model_rel in MODELS:
        model_path = os.path.join(repo_root, model_rel)
        model_name = os.path.basename(model_path)
        print(f"{'=' * 20} {model_name} {'=' * 20}")

        # CPU
        print(f"\n  [CPU ExecutionProvider]")
        cpu_min = bench_model(model_path, ["CPUExecutionProvider"], "CPU")
        if cpu_min:
            results[f"{model_name}_cpu"] = cpu_min

        # CoreML (if available)
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            print(f"\n  [CoreML ExecutionProvider]")
            coreml_min = bench_model(
                model_path, ["CoreMLExecutionProvider", "CPUExecutionProvider"], "CoreML"
            )
            if coreml_min:
                results[f"{model_name}_coreml"] = coreml_min
        else:
            print(f"\n  [CoreML not available]")

    print(f"\n{'=' * 60}")
    print("Summary (min times):")
    for k, v in results.items():
        print(f"  {k}: {v:.1f}ms")


if __name__ == "__main__":
    main()
