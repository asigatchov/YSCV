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

a = np.random.randn(1_000_000).astype(np.float32)
b = np.random.randn(1_000_000).astype(np.float32)

# Elementwise
bench("add_1M", lambda: np.add(a, b))
bench("mul_1M", lambda: np.multiply(a, b))
bench("sub_1M", lambda: np.subtract(a, b))

# Unary ops
bench("abs_1M", lambda: np.abs(a))
bench("neg_1M", lambda: np.negative(a))
bench("exp_1M", lambda: np.exp(a))
bench("sqrt_abs_1M", lambda: np.sqrt(np.abs(a)))
bench("sin_1M", lambda: np.sin(a))
bench("cos_1M", lambda: np.cos(a))
bench("log_abs_1M", lambda: np.log(np.abs(a) + 1e-10))
bench("floor_1M", lambda: np.floor(a))
bench("ceil_1M", lambda: np.ceil(a))
bench("round_1M", lambda: np.round(a))
bench("sign_1M", lambda: np.sign(a))
bench("reciprocal_1M", lambda: np.reciprocal(a + 1e-10))
bench("clamp_1M", lambda: np.clip(a, -1.0, 1.0))

# Reductions
bench("sum_1M", lambda: np.sum(a))
bench("max_1M", lambda: np.max(a))
bench("min_1M", lambda: np.min(a))
bench("mean_1M", lambda: np.mean(a))
bench("argmax_1M", lambda: np.argmax(a))
bench("argmin_1M", lambda: np.argmin(a))

# Comparisons
bench("gt_1M", lambda: a > 0)
bench("eq_1M", lambda: a == b)

# Sort/topk
bench("sort_100k", lambda: np.sort(a[:100000]))
bench("argsort_100k", lambda: np.argsort(a[:100000]))

# Axis reductions
mat = np.random.randn(512, 512).astype(np.float32)
bench("sum_axis0_512x512", lambda: np.sum(mat, axis=0))
bench("mean_axis1_512x512", lambda: np.mean(mat, axis=1))
bench("max_axis0_512x512", lambda: np.max(mat, axis=0))

# Shape ops
bench("transpose_512x512", lambda: mat.T.copy())
bench("cat_10x1000", lambda: np.concatenate([np.zeros(1000, dtype=np.float32)] * 10))

# Matmul
m256 = np.random.randn(256, 256).astype(np.float32)
bench("matmul_256x256", lambda: m256 @ m256)
