#!/usr/bin/env python3
"""PyTorch GPU Benchmark — MLP training step counterpart to nabla profile_train.

MLP: 784 -> 256 -> 128 -> 10, leaky_relu(0.01), MSE sum loss, SGD lr=0.001, f32.
Modes: eager / torch.compile / CUDA Graph.

Run: python3 benchmarks/bench_pytorch.py
"""
import math

# Patch triton < 3.5 API mismatch (triton_key removed in 3.x, PyTorch 2.7 still imports it)
try:
    import triton
    import triton.compiler.compiler as _tcc
    if not hasattr(_tcc, "triton_key"):
        _tcc.triton_key = lambda: str(triton.__version__)
except ImportError:
    pass

import torch
import torch.nn as nn

WARMUP = 10
ITERS = 100
BATCH_SIZES = [1, 32, 128, 256, 512, 1024]
LR = 0.001
NEGATIVE_SLOPE = 0.01
DEVICE = torch.device("cuda")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)
        # Kaiming init: w * sqrt(2 / fan_in)
        for layer in [self.fc1, self.fc2, self.fc3]:
            fan_in = layer.weight.shape[1]
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            layer.weight.data.mul_(math.sqrt(2.0 / fan_in))

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=NEGATIVE_SLOPE)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=NEGATIVE_SLOPE)
        return self.fc3(x)


def mse_sum_loss(output, target):
    """MSE sum loss matching nabla: diff.emul(&diff).sum_axis(1).sum_axis(0)."""
    diff = output - target
    return (diff * diff).sum()


def make_data(batch):
    x = torch.randn(batch, 784, device=DEVICE, dtype=torch.float32)
    t = torch.randn(batch, 10, device=DEVICE, dtype=torch.float32) * 0.1
    return x, t


def bench_eager(batch):
    model = MLP().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0, weight_decay=0)
    x, t = make_data(batch)

    def step():
        opt.zero_grad()
        out = model(x)
        loss = mse_sum_loss(out, t)
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bench_compiled(batch):
    model = MLP().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0, weight_decay=0)
    x, t = make_data(batch)

    def step():
        opt.zero_grad()
        out = model(x)
        loss = mse_sum_loss(out, t)
        loss.backward()
        opt.step()

    compiled_step = torch.compile(step, mode="reduce-overhead")

    for _ in range(WARMUP):
        compiled_step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        compiled_step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bench_cuda_graph(batch):
    model = MLP().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0, weight_decay=0)

    # Static buffers for graph capture
    static_x = torch.randn(batch, 784, device=DEVICE, dtype=torch.float32)
    static_t = torch.randn(batch, 10, device=DEVICE, dtype=torch.float32) * 0.1

    # Warmup before capture (side-effect: cuBLAS workspace allocation)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.zero_grad()
            out = model(static_x)
            loss = mse_sum_loss(out, static_t)
            loss.backward()
            opt.step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        out = model(static_x)
        loss = mse_sum_loss(out, static_t)
        loss.backward()
        opt.step()

    # Fill with actual data
    x, t = make_data(batch)
    static_x.copy_(x)
    static_t.copy_(t)

    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bench_matmul(n: int, dtype):
    """Standalone matmul benchmark: C = A @ B, [n,n] x [n,n]."""
    a = torch.randn(n, n, device=DEVICE, dtype=dtype)
    b = torch.randn(n, n, device=DEVICE, dtype=dtype)
    for _ in range(WARMUP):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main():
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
    print(f"allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
    print()

    # ── Matmul benchmark (FP32 TF32=OFF / FP32 TF32=ON / FP16) ──────────────
    print("=== Matmul benchmark (4096x4096) ===")
    for n in [1024, 2048, 4096]:
        torch.backends.cuda.matmul.allow_tf32 = False
        ms_f32 = bench_matmul(n, torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        ms_tf32 = bench_matmul(n, torch.float32)
        ms_f16 = bench_matmul(n, torch.float16)
        ms_bf16 = bench_matmul(n, torch.bfloat16)
        print(f"  {n}x{n}: f32(tf32=off)={ms_f32:.4f}ms  f32(tf32=on)={ms_tf32:.4f}ms  f16={ms_f16:.4f}ms  bf16={ms_bf16:.4f}ms")
    torch.backends.cuda.matmul.allow_tf32 = False  # reset to default
    print()

    print(f"MLP 784->256->128->10, leaky_relu(0.01), MSE sum, SGD lr={LR}, f32")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    results = []

    for batch in BATCH_SIZES:
        print(f"--- batch={batch} ---")
        row = {"batch": batch}

        ms = bench_eager(batch)
        row["eager"] = ms
        print(f"  eager:      {ms:.4f} ms/step")

        try:
            ms = bench_compiled(batch)
            row["compiled"] = ms
            print(f"  compiled:   {ms:.4f} ms/step")
        except Exception as e:
            row["compiled"] = None
            print(f"  compiled:   FAILED ({e})")

        try:
            ms = bench_cuda_graph(batch)
            row["cuda_graph"] = ms
            print(f"  cuda_graph: {ms:.4f} ms/step")
        except Exception as e:
            row["cuda_graph"] = None
            print(f"  cuda_graph: FAILED ({e})")

        results.append(row)
        print()

    # Summary table
    print("=" * 68)
    print(f"{'batch':>6}  {'eager (ms)':>12}  {'compiled (ms)':>14}  {'cuda_graph (ms)':>16}")
    print("-" * 68)
    for r in results:
        eager = f"{r['eager']:.4f}" if r["eager"] is not None else "N/A"
        compiled = f"{r['compiled']:.4f}" if r.get("compiled") is not None else "N/A"
        cuda_graph = f"{r['cuda_graph']:.4f}" if r.get("cuda_graph") is not None else "N/A"
        print(f"{r['batch']:>6}  {eager:>12}  {compiled:>14}  {cuda_graph:>16}")
    print("=" * 68)


if __name__ == "__main__":
    main()
