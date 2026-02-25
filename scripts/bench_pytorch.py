#!/usr/bin/env python3
"""PyTorch GPU Benchmark — comparison counterpart for nabla bench_gpu.

Run: python3 scripts/bench_pytorch.py
"""
import torch
import time

N = 4096
WARMUP = 20
ITERS = 100

device = torch.device("cuda")

def bench(name, fn, read_bytes=None, write_bytes=None):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    per_iter_ms = elapsed / ITERS * 1000
    n_bytes = N * N * 4  # f32
    if read_bytes is None:
        read_bytes = n_bytes
    if write_bytes is None:
        write_bytes = n_bytes
    gbps = (read_bytes + write_bytes) / (elapsed / ITERS) / 1e9
    print(f"{name:<25} {per_iter_ms:>8.3f} ms  {gbps:>8.1f} GB/s")

def main():
    print(f"PyTorch GPU Benchmark — {N}×{N} f32")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 50)

    a = torch.randn(N, N, device=device, dtype=torch.float32)
    b = torch.randn(N, N, device=device, dtype=torch.float32)

    # --- Element-wise unary ---
    print("\n--- Element-wise Unary ---")
    bench("exp",         lambda: torch.exp(a))
    bench("sin",         lambda: torch.sin(a))
    bench("cos",         lambda: torch.cos(a))
    bench("tanh",        lambda: torch.tanh(a))
    bench("sqrt(abs)",   lambda: torch.sqrt(torch.abs(a)))
    bench("ln(abs+1)",   lambda: torch.log1p(torch.abs(a)))
    bench("neg",         lambda: torch.neg(a))
    bench("abs",         lambda: torch.abs(a))

    # --- Element-wise binary ---
    print("\n--- Element-wise Binary ---")
    n_bytes = N * N * 4
    bench("add", lambda: torch.add(a, b), read_bytes=2*n_bytes)
    bench("sub", lambda: torch.sub(a, b), read_bytes=2*n_bytes)
    bench("emul", lambda: torch.mul(a, b), read_bytes=2*n_bytes)

    # --- Fused (torch.compile) ---
    print("\n--- Fused (torch.compile) ---")
    @torch.compile
    def fused_exp_sin(x):
        return torch.sin(torch.exp(x))
    
    @torch.compile
    def fused_4op(x):
        return torch.tanh(torch.cos(torch.sin(torch.exp(x))))
    
    bench("fuse exp+sin", lambda: fused_exp_sin(a))
    bench("fuse 4-op",    lambda: fused_4op(a))

    # --- Reductions ---
    print("\n--- Reductions ---")
    bench("sum_all", lambda: torch.sum(a), write_bytes=4)
    bench("max_all", lambda: torch.max(a), write_bytes=4)

    # --- MatMul ---
    print("\n--- MatMul ---")
    m1 = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    m2 = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    flops_1k = 2 * 1024**3
    bench("matmul 1024", lambda: torch.mm(m1, m2),
          read_bytes=2*1024*1024*4, write_bytes=1024*1024*4)

    m3 = torch.randn(N, N, device=device, dtype=torch.float32)
    m4 = torch.randn(N, N, device=device, dtype=torch.float32)
    bench("matmul 4096", lambda: torch.mm(m3, m4),
          read_bytes=2*N*N*4, write_bytes=N*N*4)

    print("\n" + "=" * 50)
    print("Done.")

if __name__ == "__main__":
    main()
