import numpy as np
import time

def bench(name, func, warmup=1, iters=100):
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

print(f"NumPy version: {np.__version__}")
print()

# --- Tensor elementwise ops (1M elements) ---
a = np.random.randn(1_000_000).astype(np.float32)
b = np.random.randn(1_000_000).astype(np.float32)

bench("add_1M", lambda: np.add(a, b))
bench("mul_1M", lambda: np.multiply(a, b))
bench("sub_1M", lambda: np.subtract(a, b))

# --- Reductions ---
bench("sum_1M", lambda: np.sum(a))
bench("max_1M", lambda: np.max(a))
bench("min_1M", lambda: np.min(a))
bench("mean_1M", lambda: np.mean(a))

# --- Matrix ops ---
m256 = np.random.randn(256, 256).astype(np.float32)
n256 = np.random.randn(256, 256).astype(np.float32)
bench("matmul_256x256", lambda: m256 @ n256)

m512 = np.random.randn(512, 512).astype(np.float32)
n512 = np.random.randn(512, 512).astype(np.float32)
bench("matmul_512x512", lambda: m512 @ n512)

# --- Unary ops ---
bench("exp_1M", lambda: np.exp(a))
bench("sqrt_1M", lambda: np.sqrt(np.abs(a)))
bench("relu_1M", lambda: np.maximum(a, 0))

# --- Broadcasting ---
row = np.random.randn(1000).astype(np.float32)
mat = np.random.randn(1000, 1000).astype(np.float32)
bench("broadcast_add_1000x1000", lambda: mat + row)

# --- Transpose ---
bench("transpose_512x512", lambda: m512.T.copy())

# --- Argmax ---
bench("argmax_1M", lambda: np.argmax(a))

# --- Axis reductions ---
mat1k = np.random.randn(512, 512).astype(np.float32)
bench("sum_axis0_512x512", lambda: np.sum(mat1k, axis=0))
bench("mean_axis1_512x512", lambda: np.mean(mat1k, axis=1))
bench("max_axis0_512x512", lambda: np.max(mat1k, axis=0))
