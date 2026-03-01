#!/usr/bin/env python3
"""PyTorch Dispatch Scaling Benchmark — counterpart to nabla bench_dispatch_scaling.

Measures dispatch overhead across tensor sizes and op counts.
Run: python3 benchmarks/bench_dispatch_scaling.py
"""
import json
import sys
import time
import torch
import torch.nn as nn

WARMUP = 20
SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
CHAIN_OPS = 1000
device = torch.device("cuda")


# ========================================================================
# Experiment 1: Small tensor × many ops (exp chain)
# ========================================================================

def exp1_op_chain():
    print("\n=== Exp1: Small tensor x {} chained exp ops ===".format(CHAIN_OPS), file=sys.stderr)
    print(f"{'size':<8} {'total_ms':>12} {'ops/sec':>14} {'ns/op':>10}", file=sys.stderr)
    print("-" * 48, file=sys.stderr)

    for sz in SIZES:
        a = torch.randn(sz, sz, device=device, dtype=torch.float32)
        for _ in range(WARMUP):
            a = torch.exp(a)
        torch.cuda.synchronize()

        ring = [a.clone() for _ in range(4)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(CHAIN_OPS):
            ring[i % 4] = torch.exp(ring[(i + 3) % 4])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_ms = elapsed * 1000
        ops_per_sec = CHAIN_OPS / elapsed
        ns_per_op = elapsed / CHAIN_OPS * 1e9

        print(json.dumps({"test": "exp1_chain", "size": sz, "ops": CHAIN_OPS,
                           "total_ms": round(total_ms, 3),
                           "ops_per_sec": round(ops_per_sec),
                           "ns_per_op": round(ns_per_op)}))
        print(f"{sz:<8} {total_ms:>12.3f} {ops_per_sec:>14.0f} {ns_per_op:>10.0f}", file=sys.stderr)
        del ring


# ========================================================================
# Experiment 2: MLP forward pass (784→256→128→10)
# ========================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def exp2_mlp_forward():
    model = MLP().to(device)
    model.eval()
    iters = 500

    print("\n=== Exp2: MLP forward (784→256→128→10) ===", file=sys.stderr)
    print(f"{'batch':<12} {'ms/fwd':>10} {'fwd/sec':>12}", file=sys.stderr)
    print("-" * 36, file=sys.stderr)

    for batch in [1, 32, 128, 1024]:
        x = torch.randn(batch, 784, device=device, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(WARMUP):
                model(x)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        ms_per_fwd = elapsed / iters * 1000
        fwd_per_sec = iters / elapsed

        print(json.dumps({"test": "exp2_forward", "batch": batch,
                           "ms_per_fwd": round(ms_per_fwd, 4),
                           "fwd_per_sec": round(fwd_per_sec),
                           "iters": iters}))
        print(f"{batch:<12} {ms_per_fwd:>10.4f} {fwd_per_sec:>12.0f}", file=sys.stderr)


# ========================================================================
# Experiment 3: Training step (forward + backward + SGD step)
# ========================================================================

def exp3_train_step():
    steps = 200

    print("\n=== Exp3: Training step (fwd+bwd+sgd) ===", file=sys.stderr)
    print(f"{'batch':<12} {'ms/step':>12} {'steps/sec':>12}", file=sys.stderr)
    print("-" * 38, file=sys.stderr)

    for batch in [1, 32, 128]:
        model = MLP().to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(batch, 784, device=device, dtype=torch.float32)
        target = torch.randn(batch, 10, device=device, dtype=torch.float32)
        loss_fn = nn.MSELoss()

        # warmup
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        ms_per_step = elapsed / steps * 1000
        steps_per_sec = steps / elapsed

        print(json.dumps({"test": "exp3_train", "batch": batch,
                           "ms_per_step": round(ms_per_step, 4),
                           "steps_per_sec": round(steps_per_sec),
                           "steps": steps}))
        print(f"{batch:<12} {ms_per_step:>12.4f} {steps_per_sec:>12.0f}", file=sys.stderr)


def main():
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}", file=sys.stderr)
    exp1_op_chain()
    exp2_mlp_forward()
    exp3_train_step()


if __name__ == "__main__":
    main()
