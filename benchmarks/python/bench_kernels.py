import torch
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

print(f"PyTorch version: {torch.__version__}")
print()

a = torch.randn(1_000_000)

# Activations
bench("relu_1M", lambda: torch.relu(a))
bench("sigmoid_1M", lambda: torch.sigmoid(a))
bench("tanh_1M", lambda: torch.tanh(a))
bench("gelu_1M", lambda: torch.nn.functional.gelu(a))
bench("silu_1M", lambda: torch.nn.functional.silu(a))

# Softmax/norms
sm = torch.randn(32, 1000)
bench("softmax_32x1000", lambda: torch.softmax(sm, dim=-1))
bench("log_softmax_32x1000", lambda: torch.nn.functional.log_softmax(sm, dim=-1))

ln = torch.nn.LayerNorm(256)
ln_in = torch.randn(32, 256)
with torch.no_grad():
    bench("layer_norm_32x256", lambda: ln(ln_in))

bn = torch.nn.BatchNorm2d(3)
bn.eval()
bn_in = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    bench("batch_norm_1x3x64x64", lambda: bn(bn_in))

# Conv
conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
c_in = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    bench("conv2d_32x32_3to16", lambda: conv(c_in))

# Matmul
m = torch.randn(256, 256)
bench("matmul_256x256", lambda: m @ m)
m1k = torch.randn(1024, 1024)
bench("matmul_1024x1024", lambda: m1k @ m1k)

# Optimizer steps
model = torch.nn.Linear(1024, 1024, bias=False)
sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(32, 1024)
t = torch.randn(32, 1024)

def sgd_step():
    sgd.zero_grad()
    (model(x) - t).pow(2).mean().backward()
    sgd.step()

def adam_step():
    adam.zero_grad()
    (model(x) - t).pow(2).mean().backward()
    adam.step()

bench("sgd_step_1024", sgd_step)
bench("adam_step_1024", adam_step)

# Autograd
def autograd_128():
    a = torch.randn(128, 128, requires_grad=True)
    b = torch.randn(128, 128, requires_grad=True)
    (a @ b).relu().mean().backward()

bench("autograd_128x128", autograd_128)

# Sort/topk
big = torch.randn(100000)
bench("sort_100k", lambda: torch.sort(big))
bench("topk_100k_k100", lambda: torch.topk(big, 100))

# Attention
q = torch.randn(32, 8, 128, 64)
k = torch.randn(32, 8, 128, 64)
v = torch.randn(32, 8, 128, 64)
with torch.no_grad():
    bench("attention_32x8x128x64", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
