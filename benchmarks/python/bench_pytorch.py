import torch
import time

torch.set_num_threads(torch.get_num_threads())

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

print(f"PyTorch version: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")
print()

# --- Matmul ---
m256 = torch.randn(256, 256)
n256 = torch.randn(256, 256)
bench("matmul_256x256", lambda: m256 @ n256)

m512 = torch.randn(512, 512)
n512 = torch.randn(512, 512)
bench("matmul_512x512", lambda: m512 @ n512)

m1024 = torch.randn(1024, 1024)
n1024 = torch.randn(1024, 1024)
bench("matmul_1024x1024", lambda: m1024 @ n1024)

# --- Conv2d ---
conv_input = torch.randn(1, 3, 32, 32)
conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
with torch.no_grad():
    bench("conv2d_32x32_3to16", lambda: conv(conv_input))

# --- ReLU ---
relu_input = torch.randn(1_000_000)
bench("relu_1M", lambda: torch.relu(relu_input))

# --- Backward pass ---
def backward_bench():
    a = torch.randn(128, 128, requires_grad=True)
    b = torch.randn(128, 128, requires_grad=True)
    c = (a @ b).relu().mean()
    c.backward()

bench("autograd_matmul_relu_mean_128", backward_bench)

# --- SGD training step ---
model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(16, 64)
y = torch.randn(16, 1)

def train_step():
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

bench("sgd_train_step_16x64", train_step)

# --- Elementwise ---
a1m = torch.randn(1_000_000)
b1m = torch.randn(1_000_000)
bench("add_1M", lambda: a1m + b1m)
bench("mul_1M", lambda: a1m * b1m)
bench("exp_1M", lambda: torch.exp(a1m))
bench("sum_1M", lambda: a1m.sum())

# --- Activations ---
bench("tanh_1M", lambda: torch.tanh(a1m))
bench("sigmoid_1M", lambda: torch.sigmoid(a1m))
bench("gelu_1M", lambda: torch.nn.functional.gelu(a1m))
bench("silu_1M", lambda: torch.nn.functional.silu(a1m))

# --- Normalization ---
ln = torch.nn.LayerNorm(256)
ln_input = torch.randn(32, 256)
with torch.no_grad():
    bench("layer_norm_32x256", lambda: ln(ln_input))

bn = torch.nn.BatchNorm2d(3)
bn.eval()
bn_input = torch.randn(32, 3, 64, 64)
with torch.no_grad():
    bench("batch_norm_32x3x64x64", lambda: bn(bn_input))

# --- Softmax ---
sm_input = torch.randn(32, 1000)
bench("softmax_32x1000", lambda: torch.softmax(sm_input, dim=-1))

# --- Conv2d variants ---
conv_input_64 = torch.randn(1, 3, 64, 64)
conv3x3 = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
with torch.no_grad():
    bench("conv2d_64x64_3to16", lambda: conv3x3(conv_input_64))

dw_conv = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
dw_input = torch.randn(1, 32, 32, 32)
with torch.no_grad():
    bench("depthwise_conv2d_32x32", lambda: dw_conv(dw_input))

# --- Model forward ---
model3 = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
model3.eval()
fwd_input = torch.randn(32, 256)
with torch.no_grad():
    bench("model_forward_3layer_batch32", lambda: model3(fwd_input))

# --- Axis reductions ---
mat512 = torch.randn(512, 512)
bench("sum_axis0_512x512", lambda: mat512.sum(dim=0))
bench("mean_axis1_512x512", lambda: mat512.mean(dim=1))

# --- YUV conversion (OpenCV) ---
try:
    import numpy as np
    import cv2
    yuv_1080 = np.zeros((1620, 1920), dtype=np.uint8)  # YUV420 layout
    bench("yuv420_to_bgr_1080p", lambda: cv2.cvtColor(yuv_1080, cv2.COLOR_YUV2BGR_I420))
except ImportError:
    print("yuv420_to_bgr_1080p: skipped (cv2 not installed)")

# --- Optimizer Steps ---
model_opt = torch.nn.Linear(1024, 1024, bias=False)
sgd_opt = torch.optim.SGD(model_opt.parameters(), lr=0.01, momentum=0.9)
adam_opt = torch.optim.Adam(model_opt.parameters(), lr=0.001)
opt_input = torch.randn(32, 1024)
opt_target = torch.randn(32, 1024)

def sgd_step():
    sgd_opt.zero_grad()
    loss = (model_opt(opt_input) - opt_target).pow(2).mean()
    loss.backward()
    sgd_opt.step()

def adam_step():
    adam_opt.zero_grad()
    loss = (model_opt(opt_input) - opt_target).pow(2).mean()
    loss.backward()
    adam_opt.step()

bench("sgd_step_1024x1024", sgd_step)
bench("adam_step_1024x1024", adam_step)

# --- Attention ---
q = torch.randn(32, 8, 128, 64)  # batch, heads, seq, dim
k = torch.randn(32, 8, 128, 64)
v = torch.randn(32, 8, 128, 64)
with torch.no_grad():
    bench("attention_32x8x128x64", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

# --- Embedding ---
emb = torch.nn.Embedding(10000, 128)
idx = torch.randint(0, 10000, (32, 128))
with torch.no_grad():
    bench("embedding_10kx128_batch32", lambda: emb(idx))

# --- Large matmul ---
m1k = torch.randn(1024, 1024)
n1k = torch.randn(1024, 1024)
bench("matmul_1024x1024", lambda: m1k @ n1k)

m2k = torch.randn(2048, 2048)
n2k = torch.randn(2048, 2048)
bench("matmul_2048x2048", lambda: m2k @ n2k)

# --- Cat/Stack ---
tensors_cat = [torch.randn(1000) for _ in range(10)]
bench("cat_10x1000", lambda: torch.cat(tensors_cat))
bench("stack_10x1000", lambda: torch.stack(tensors_cat))

# --- Topk ---
big = torch.randn(100000)
bench("topk_100k_k100", lambda: torch.topk(big, 100))

# --- Sort ---
bench("sort_100k", lambda: torch.sort(big))
