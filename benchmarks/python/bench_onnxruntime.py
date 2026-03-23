import numpy as np
import time

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
except ImportError:
    print("onnxruntime not installed. pip install onnxruntime")
    exit(1)

def bench(name, func, warmup=3, iters=50):
    for _ in range(warmup):
        func()
    best = float('inf')
    for _ in range(iters):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000.0
        if elapsed < best:
            best = elapsed
    print(f"{name}: {best:.3f} ms")

# Try to download a small ONNX model for benchmarking
import urllib.request
import os

model_path = "/tmp/squeezenet1.0.onnx"
if not os.path.exists(model_path):
    print("Downloading SqueezeNet 1.0 ONNX model...")
    url = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded to {model_path}")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Skipping ONNX inference benchmark")
        exit(0)

# Create session
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"Model input: {input_name} {input_shape}")

# Create random input
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

bench("squeezenet_inference_224x224", lambda: sess.run(None, {input_name: dummy}))
