#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building nabla benchmarks (CUDA) ==="
cargo build --release --manifest-path "$ROOT_DIR/benchmarks/Cargo.toml" --features cuda

echo ""
echo "=== nabla bench_ops ==="
"$ROOT_DIR/target/release/bench_ops"

echo ""
echo "=== nabla bench_dispatch ==="
"$ROOT_DIR/target/release/bench_dispatch"

echo ""
echo "=== nabla bench_dispatch_scaling ==="
"$ROOT_DIR/target/release/bench_dispatch_scaling"

echo ""
echo "=== nabla MLP training (eager + CUDA Graph) ==="
"$ROOT_DIR/target/release/profile_train_graph"

echo ""
echo "=== PyTorch MLP training (eager + compile + CUDA Graph) ==="
python3 "$SCRIPT_DIR/bench_pytorch.py"

echo ""
echo "=== PyTorch bench_dispatch_scaling ==="
python3 "$SCRIPT_DIR/bench_dispatch_scaling.py"
