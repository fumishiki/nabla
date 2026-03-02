# nabla — Specification

> Ground truth specification for the nabla computation engine.
> Related: [notation.md](notation.md) — API & macro reference | [directory.md](directory.md) — project structure

Legend: ✅ Implemented | ❌ Not possible (language constraint) | 🔲 Not yet implemented

---

## §1 Overview

**Zero-GC, zero-copy, type-safe** Rust linear algebra DSL for researchers who refuse to choose between Python's ease and C++'s speed.
Proc macros (`mat![]`, `map!{}`, `einsum!{}`, `math!`, `fuse!`, `train_step!`) combined with self-contained pure-Rust kernels (CPU) and GPU compute shaders across three backends: wgpu (WGSL), CUDA (nvrtc), HIP (hiprtc).
Exactly one backend is selected via feature flags at build time — no implicit CPU fallback, no cross-backend runtime dispatch.

### Fixed Rule Principle — computation engine, not framework

nabla is a **computation engine**, not a framework. It executes mathematically invariant computation primitives (matmul, conv, softmax, cross_entropy, etc.) at maximum speed on CPU/GPU.

| nabla provides | User decides |
|---|---|
| matmul, conv, bmm, SDPA, embedding | layer stacking, skip connections |
| activations (relu/gelu/silu/softmax) | which activation where |
| loss (cross_entropy/mse/l1/kl_div) | which loss to optimize |
| norm (layer/rms/batch/group) | where to normalize |
| reductions, reshape, gather, scatter | data flow topology |
| AD (reverse/forward), ODE, CAS | what to differentiate, solver choice |
| optimizer, scheduler, trainer, dataloader | model architecture decisions |

Four-layer architecture: `nabla-macros` (notation) -> `nabla-core` (compute) -> `nabla-ml` (application) -> `nabla-train` (training). See [directory.md](directory.md) for details.

---

## §2 Architecture

### 2.1 Four-layer dependency graph

```
nabla-macros  (Layer 1: Notation)    — proc macros: mat!, einsum!, fuse!, sym!, #[derive(Module)]
  ↓
nabla-core    (Layer 2: Compute)     — Tensor<T,B>, Backend trait (6 sub-traits), Scalar, 190+ ops
  ↓
nabla-ml      (Layer 3: Application) — autograd, linalg, CAS (e-graph), ODE/SDE, nn module
  ↓
nabla-train   (Layer 4: Training)    — optimizer, scheduler, dataloader, trainer, checkpoint, profiler, quantize, export
  ↓
nabla-interface (Layer 5: Export+Inference) — GGUF v3 writer, quantization (34 types), llama.cpp FFI, Metal inference
```

Each layer depends only on layers below it. Layer 1 has zero runtime deps. Layer 2 is a pure computation engine. Layer 3 composes Layer 2 primitives into domain-specific APIs. Layer 4 adds training utilities without altering core math semantics.

### 2.2 Backend selection

Exactly one of `{cpu, wgpu, cuda, hip}` via feature flag. All 6 pairwise conflicts -> `compile_error!`.

| Feature | Backend | Scalar Types | Special Capabilities |
|---|---|---|---|
| `cpu` (default) | `Cpu` (pure Rust + rayon) | f32, f64, f16, bf16, c32, c64, Dual | Full linalg, sparse, CAS eval |
| `gpu` | `Gpu` (wgpu/WGSL) | f32 *(f64 unsupported — WGSL core spec has no f64 type)* | BCSR SpMM, register-tile MMA |
| `cuda` | `Cuda` (cuBLAS/nvrtc) | f32, f64 | Tensor cores (WMMA f16 matmul), Graph capture, cublasLt epilogue |
| `hip` | `Hip` (hipBLAS/hiprtc) | f32, f64 | CDNA support, rocWMMA f16 matmul |

All tensors use `Tensor<T>` = `Tensor<T, DefaultBackend>`. Backend trait: all computation methods are **required** (no default body) — no implicit CPU fallback.

### 2.3 Backend trait architecture

The `Backend` trait uses a **sealed sub-trait** pattern — external crate implementations are forbidden via `private::Sealed`.

| Sub-trait | Responsibility | Methods |
|---|---|---|
| `BackendCore` | Storage allocation, arithmetic, element-wise binary | `zeros`, `fill`, `from_fn`, `add`, `sub`, `mul`, `div`, ... |
| `BackendMath` | Element-wise transcendental math | `exp`, `ln`, `sin`, `cos`, `sqrt`, `abs`, `erf`, `powf`, ... |
| `BackendReduce` | Reductions over axes | `sum_all`, `max_all`, `argmax`, `sum_axis`, `cumsum`, ... |
| `BackendBlas` | Matrix multiply, GEMM | `matmul`, `matmul_into`, `bmm`, `addmm`, `baddbmm` |
| `BackendNN` | Activations, loss, normalization | `relu`, `softmax`, `cross_entropy`, `layer_norm`, `conv2d`, ... |
| `BackendFusion` | Fused kernel dispatch | `fuse_elementwise`, `mega_fuse`, JIT codegen |

Blanket impl: `impl<B: BackendCore + BackendMath + BackendReduce + BackendBlas + BackendNN + BackendFusion> Backend for B {}`.

### 2.4 Memory model

- **Zero-GC**: Rust ownership (`Drop`) manages all allocations — no garbage collector.
- **Zero-copy**: `TensorView` borrows data without allocation; `Cow` semantics for lazy cloning.
- **Lazy readback**: GPU storage defers device-to-host transfer until `.get()` / `.to_vec()`.
- **Thread-local RNG**: `set_seed()` / `clear_seed()` for reproducible initialization.

### 2.5 GPU kernel hyperparameters

```
BLOCK_SIZE:          256  (default; tunable per kernel via autotune — do NOT hardcode)
float4 vectorization: all f32 unary/binary kernels (128-bit LDG.E.128 + scalar tail)
warp size:           32  (CUDA/HIP)
workgroup size (wgpu): 256

FlashAttention-2 tile sizes:
  BLOCK_M: 64   BLOCK_N: 64   HEAD_DIM_MAX: 128

im2col+GEMM for conv2d:
  im2col col: [N, C_in*kH*kW, out_H*out_W]  weight: [C_out, C_in*kH*kW]
  GEMM: cublasSgemmStridedBatched per batch

Blelloch parallel scan: O(n) work, O(log n) depth, smem = 2*BLOCK_SIZE*sizeof(T)
```

### 2.6 Pinned dependency versions

```
Rust edition: 2024 | wgpu: 24 | cudarc: 0.19 | rayon: 1
```

### 2.7 Macro dispatch

| Macro | CPU | GPU | Strategy |
|---|---|---|---|
| `einsum!` | ✅ GEMM | ✅ GPU kernel | `matmul_into` (Backend dispatch) |
| `fuse!` | ✅ `from_fn` | ✅ JIT fused kernel | NVRTC/hiprtc codegen |
| `map!` / `stencil!` / `par_*` | ✅ | ❌ compile error | CPU-only (closure / offset access / rayon) |
| `math!` | ✅ | ✅ | Auto-borrow idents in expressions |
| `mat!` / `splat!` / `named!` | ✅ | ✅ | Compile-time expansion |

### 2.8 GPU implementation

```
             ┌───────────────────────────┐
             │  GpuContext trait          │
             │  GpuStorage<T> (lazy D2H) │
             └─────┬──────────┬──────────┘
               wgpu│     cuda │      hip │
         WGSL shaders  CUDA C (nvrtc)  HIP C (hiprtc)
            43 ops      100+ ops       100+ ops
```

**GpuContext trait**: `alloc`, `upload`, `download`, `launch`. Singleton via `OnceLock`.
**GpuStorage**: backend buffer + lazy `Mutex<Option<Vec<T>>>` host cache. Readback on `.get()`/`.to_vec()` only.

### 2.9 Kernel sources — two codebases

74 CUDA/HIP ops + 43 WGSL ops, same semantics, two shader languages:

| Category | CUDA/HIP | wgpu (WGSL) |
|---|---|---|
| Binary (add/sub/mul/div/scale) | float4 vectorized | `elementwise_binary` |
| Unary (15 ops: exp/ln/sin/cos/tanh/sqrt/abs/erf/...) | float4 + fast math | `elementwise_unary` |
| Activation (sigmoid/silu/mish/leaky_relu/elu/hardswish) | float4 vectorized | dedicated shaders |
| Matmul | tiled TILE=16/32, WMMA/MMA tensor cores | tiled TILE=16, register-tile MMA |
| Reduction (sum/max/min/argmax/argmin) | warp shuffle | workgroup reduce |
| Row-wise (softmax/layer_norm/rms_norm) | 3-pass warp shuffle | workgroup reduce |
| Axis reduction (sum_axis/max_axis) | warp shuffle | workgroup reduce |
| Gather (embedding) | thread-per-element | thread-per-element |
| Phase 3 (conv/pool/attn/scan/norm/loss) | 30+ dedicated kernels | CPU defaults |

f32: `float4` 128-bit loads + `__expf`/`__logf` fast-math. f64: scalar (1 elem/thread).
Runtime compilation: nvrtc/hiprtc -> PTX cache (`HashMap<String, KernelEntry>`). Architecture auto-detected.

### 2.10 Kernel fusion

| Level | Scope | Backend |
|---|---|---|
| L1: Element-wise | Consecutive unary/binary -> single JIT kernel | GPU (fuse!) |
| L2: Reduction | Across reduction ops with loop-carried deps | CPU |
| L3: GEMM+pointwise | matmul + activation (cublasLt epilogue) | GPU |
| L4: Map-reduce | Pointwise + axis reduction -> single kernel | GPU |
| L4: Mega-kernel | Shared memory tile reuse for multi-op mega_fuse! | GPU |
| L4: DAG fusion | `prev` keyword for inter-op register pass-through | GPU |
| L5: Loss/optimizer | sub+sq+reduce / multi-param AXPY → single kernel | GPU (cuda) |

Pipeline: `fuse!` AST -> egg EqSat simplify -> `cuda_expr()` -> NVRTC/hiprtc JIT -> FNV-1a hash cache.

### 2.11 GPU advanced features

| Feature | Backend | Approach |
|---|---|---|
| WMMA/MMA tensor cores | cuda/hip | `nvcuda::wmma::mma_sync` (Volta+), `rocwmma` (CDNA2+) |
| Warp shuffle reductions | cuda/hip | `__shfl_down_sync`, 8x reduction |
| Register-tile MMA (wgpu) | wgpu | Software MMA via register tiling |
| Linear Layouts F2 swizzle | all | Bank-conflict-free for any tile size |
| Caching memory allocator | cuda/hip | Best-fit dual-pool, 512B-aligned, GC 0.9 |
| Async execution pipeline | cuda/hip | Defer sync until readback only |
| CUDA Graph capture/replay | cuda | `cuda_graph_capture` API, 1.3–1.6x speedup (MLP training) |
| Fused loss/optimizer kernels | cuda | `k_mse_sum_fwd` (sub+sq+reduce→1), `k_mse_sum_bwd`, `k_multi_axpy3` |
| cuBLAS workspace pre-alloc | cuda | `cublasSetWorkspace_v2` 32MiB |
| Mega-kernel tiled fusion | cuda/hip | Shared memory tile reuse (>=2 ops, >=64K elements) |

---

## §3 Feature Catalog — 190+ ops

nabla covers the mathematically fixed computations provided by PyTorch's `torch.*` / `torch.nn.functional.*` as a computation engine. See [notation.md](notation.md) for API details and argument signatures.

### 3.1 Summary

| Category | Count | Key ops | GPU |
|---|---|---|---|
| **A. Convolution** | 4 | conv1d/2d/3d, conv_transpose2d | ✅ im2col+cuBLAS |
| **B. Pooling** | 4 | max/avg/adaptive_avg pool2d, max_pool1d | ✅ (2d), 🔲 (1d) |
| **C. Normalization** | 4 | layer/rms/batch/group_norm | ✅ GPU kernels |
| **D. Activation** | 10 | relu, gelu, sigmoid, softmax, silu, mish, leaky_relu, elu, hardswish, log_softmax | ✅ float4 + fuse! |
| **E. Loss** | 9 | cross_entropy, mse, mse_sum (fused), l1, smooth_l1, bce_logits, nll, kl_div, cosine_embedding | ✅ fused GPU |
| **F. Attention** | 3 | SDPA (FlashAttention-2), multi_head_attention, embedding | ✅ GPU kernels |
| **G. Manipulation** | 19 | reshape/permute/cat/stack/squeeze/flatten/chunk/pad/gather/scatter/index_select/masked_fill/where/triu/tril/roll/flip/meshgrid/topk/sort | ✅ CPU |
| **H. Batched** | 4 | bmm, baddbmm, addmm, batched reductions | ✅ cuBLAS |
| **I. Construction** | 11 | zeros/ones/full/eye/arange/linspace/rand/randn/empty/clone/contiguous | ✅ |
| **J. Reduction** | 10 | sum/max/min/mean/var/std/argmax/argmin (all + axis), cumsum/cumprod, prod, norm, count_nonzero | ✅ GPU (Blelloch, warp shuffle) |
| **K. Regularization** | 3 | dropout, interpolate_nearest, interpolate_bilinear | ✅ CPU |

### 3.2 Element-wise math (prelude)

`exp` `ln` `log1p` `log2` `log10` `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `asinh` `acosh` `atanh` `sqrt` `abs` `recip` `erf` `ceil` `floor` `round` `powf` `neg` `sign` `rem` — all GPU-accelerated (float4 vectorized).

Operators: `+` `-` `*` `/` (owned + borrowed combos), `scalar * &Tensor`, `epow(&b)`, `hadamard(&rhs)`.

### 3.3 Linear algebra (CPU)

45+ methods via `LinalgExt` trait (f32/f64): `svd`, `qr`, `lu`, `cholesky`, `eig`, `inv`, `solve`, `lstsq`, `det`, `trace`, `rank`, `norm`, `cond`, `pinv`. Structural: `Diagonal`, `Symmetric`, `Triangular`. GPU: `gpu_trsm_lower` (recursive GEMM) only.

### 3.4 Sparse

`BcsrMatrix<T>` BCSR + `WGSL_BCSR_SPMM` kernel (GPU). `mixed_spmm_f64` mixed-precision refinement. CPU: CSC.

### 3.5 CAS (Computer Algebra System)

`egg` 57 unified rules (`cas_rules()`: 33 algebraic + 24 differentiation) + `diff_simplify` (41 rules) + `FuseExpr` 16-node EqSat (18 IEEE-754 safe rules). `gradient`/`jacobian`/`hessian` (auto-simplify). Domain-checked `eval`/`eval_tensor`. Method chain API. See [notation.md](notation.md) §3.10 for details.

### 3.6 ODE/SDE solvers

| Solver | Type | Order |
|---|---|---|
| `euler` | Explicit | 1 |
| `rk4` | Explicit | 4 |
| `dormand_prince` | Adaptive | 5(4) |
| `bdf1` / `bdf2` | Implicit (stiff) | 1 / 2 |
| `if_euler` | Exponential integrator | — |
| `metd` | Matrix exponential | p |
| `stormer_verlet` | Symplectic | 2 |
| `parareal` | Time-parallel | — |
| `euler_maruyama` / `milstein` | SDE | 0.5 / 1.0 |

`OdeProblem<T,B,F>` wrapper. `ensemble_euler_maruyama` (parallel N-trajectory). Backward integration. `OdeSolution::eval(t)` interpolation. See [notation.md](notation.md) §3.10 for details.

### 3.7 Autograd

| Mode | Approach | Status |
|---|---|---|
| Reverse (tape) | `GpuTape<T>`: 14-op enum, backward via GPU kernels | ✅ |
| Forward (dual) | `Dual<T>`: `impl Scalar for Dual<T>` — all ops unchanged | ✅ |
| Source-transform | `#[nabla_grad]` proc macro -> `f_grad(x) -> (T, T)` | ✅ |

NN ops: softmax, reshape, transpose, linear_forward, dropout, clamp, loss ops. Module/Optimizer: `Module` trait, `Sequential`, `AdamW`, `GradScaler`. `impl_var_op!` macro absorbs boilerplate for std::ops trait impls (Add/Sub/Mul x 4 ownership combos). See [notation.md](notation.md) §7 for details.

---

## §4 Design Decisions & Limitations

| Decision | Rationale |
|---|---|
| Direct kernel strings | Fixed-rule ops -> 2 codebases manageable |
| Build-time exclusive backend | CPU fallback is a performance bug source |
| Runtime kernel compilation | nvrtc/hiprtc: no SDK at build time |
| Handle-based GPU storage | Chained ops eliminate host<->device transfer |
| TypeId dispatch | Backend trait sealed + `T: Scalar`, avoids E0276 |
| Embedded kernel strings | WGSL + CUDA/HIP C as `const &str` |
| Native Rust (no C++ wrapper) | Ownership-native tensor > FFI wrapper |
| Recursive GEMM for GPU linalg | Reuse matmul_tiled |
| Einsum canonicalization | 4.7x over JAX |
| Named axes | Compile-time dimension safety |
| `impl Scalar for Dual<T>` | Forward-mode AD as zero-change drop-in |
| Macro DSL absorbs verbosity | Compresses 5-10x LOC gap vs. Julia via macros |

| Limitation | Mitigation |
|---|---|
| No wgpu f64 | WGSL core spec does not define `f64`; all storage buffers are `f32` only. Use `cuda`/`hip` backend |
| No GPU c32/c64 | Compile error (by design) |
| GPU linalg: TRSM only | Full LU/Cholesky/QR CPU only |
| `from_fn` requires host | Use `fuse!` for GPU |
| 2 kernel codebases | WGSL != CUDA/HIP C — fixed ops, rarely changes |
| No REPL | `rust-script` + `cargo watch` |

---

## §5 Performance

### Benchmark (GH200 480GB, 4096x4096 f32, CUDA 12.8, PyTorch 2.7.0)

| Workload | nabla | PyTorch 2.7 | Notes |
|---|---|---|---|
| matmul 4096 (cuBLAS TF32) | 0.358 ms | 2.675 ms | **nabla 7.5x faster** |
| matmul 1024 | 0.036 ms | 0.058 ms | **nabla 1.6x faster** |
| fuse exp+sin | 0.041 ms | 0.081 ms (eager) | **nabla 2.0x faster** |
| sin / cos / tanh | 0.040 ms | 0.041 ms | ~parity |
| add / sub / emul | 0.058 ms | 0.058 ms | ~parity |
| sum_all / max_all | 0.028 ms | 0.026 ms | PyTorch 1.08x |
| fuse 4-op speedup | 3.38x | 1.0x (torch.compile N/A) | nabla fused vs unfused |
| dispatch latency | 42-58 us/op | 40-58 us/op | ~parity |

### MLP training benchmark (GH200 480GB, CUDA 12.8)

MLP 784→256→128→10, leaky_relu(0.01), MSE sum loss, SGD lr=0.001, f32. Warmup=10, iters=100.

| Batch | nabla eager | nabla graph | PyTorch eager | PyTorch graph |
|------:|------------:|------------:|--------------:|--------------:|
| 1     | 0.111 ms    | **0.070 ms** | 0.710 ms     | 0.045 ms      |
| 32    | 0.133 ms    | **0.085 ms** | 0.923 ms     | 0.072 ms      |
| 128   | 0.133 ms    | **0.088 ms** | 0.976 ms     | 0.130 ms      |
| 256   | 0.139 ms    | **0.094 ms** | 0.974 ms     | 0.136 ms      |
| 512   | 0.147 ms    | **0.108 ms** | 0.847 ms     | 0.142 ms      |
| 1024  | 0.170 ms    | **0.130 ms** | 0.966 ms     | 0.160 ms      |

**Key results**: nabla eager 4.2–6.4× faster than PyTorch eager. nabla CUDA Graph wins batch≥128 (1.2–1.5× over PyTorch CUDA Graph). Fused `k_mse_sum_fwd` kernel eliminates 3 kernel launches (sub + emul + sum_axis×2 → 1). Pool bypass during capture + device-only reduction (zero D2H).

Reproducible via `benchmarks/` (Rust + Python side-by-side).

---

## §6 Phase 6 — Profiling, Quantization, Export & Benchmark (Done ✅)

Crate: `nabla-train` (profiler.rs, quantize.rs, onnx.rs, gguf.rs, benchmark.rs)

### 6.1 Profiling

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-PROF-01 | Profiler | CudaEvent-based per-kernel execution time (start/stop/elapsed_ms) | ✅ |
| P6-PROF-02 | Profiler | Per-layer stats (forward/backward time, VRAM, TFLOPS per layer) | ✅ |
| P6-PROF-03 | Profiler | Auto-compute tok/s, latency (ms/token), batch throughput | ✅ |
| P6-PROF-04 | Profiler | TFLOPS: 2MKN/elapsed (matmul), estimated FLOPs per kernel | ✅ |
| P6-PROF-05 | Profiler | Roofline: min(peak_TFLOPS, AI × peak_BW) compute/memory bound classification | ✅ |
| P6-PROF-06 | Profiler | VRAM tracking (peak / current / per-layer breakdown) | ✅ |
| P6-PROF-07 | Profiler | JSON output (kernel breakdown + roofline + per-layer stats) | ✅ |

### 6.2 AWQ INT4 Weight-Only Quantization

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-AWQ-01 | Quantize | Calibration: collect per-channel activation statistics from input data | ✅ |
| P6-AWQ-02 | Quantize | Per-channel scale optimization (activation-aware, grid search) | ✅ |
| P6-AWQ-03 | Quantize | INT4 packing (8 weights → 1 u32, little-endian) | ✅ |
| P6-AWQ-04 | Quantize | CUDA dequant-matmul kernel (INT4 unpack → f16/f32 → GEMM) | ✅ |
| P6-AWQ-05 | Quantize | group_size parameter (default 128) for quantization granularity | ✅ |

### 6.3 ONNX Export

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-ONNX-01 | Export | Module trait walk → ONNX NodeProto DAG construction | ✅ |
| P6-ONNX-02 | Export | Protobuf serialization (minimal encoder, no external deps) | ✅ |
| P6-ONNX-03 | Export | Opset 21 compliance (MatMul, Relu, Conv, LayerNorm, etc.) | ✅ |
| P6-ONNX-04 | Export | Dynamic axes support (batch_size, seq_len) | ✅ |
| P6-ONNX-05 | Export | Inference verification with onnxruntime (numerical match atol=1e-5) | ✅ |

### 6.4 GGUF Export

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-GGUF-01 | Export | GGUF v3 binary writer (magic + version + metadata KV + tensor info + data) | ✅ |
| P6-GGUF-02 | Export | `GgufQuantType` enum — Legacy: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, K-quant: Q2_K/Q3_K_S/M/L/Q4_K_S/M/Q5_K_S/M/Q6_K, IQ: IQ1_S/M/IQ2_XXS/XS/S/M/IQ3_XXS/XS/S/M/IQ4_XS/NL, Ternary: TQ1_0/TQ2_0, Full: F16/BF16 | ✅ |
| P6-GGUF-03 | Export | Legacy block structure (QK=32): Q*_0 = delta(f16)+qs, Q*_1 = delta+min(f16×2)+qs | ✅ |
| P6-GGUF-04 | Export | K-quant block structure (QK_K=256): super-block d/dmin(f16) + sub-block scales + packed qs | ✅ |
| P6-GGUF-05 | Export | IQ block structure (QK_K=256): E8 lattice / non-linear LUT / importance-matrix support | ✅ |
| P6-GGUF-06 | Export | Ternary block structure: TQ1_0 (trit packing 5^5) / TQ2_0 (2bit 3-value) | ✅ |
| P6-GGUF-07 | Export | _S/_M/_L layer mixing strategy (per-tensor quant type, precision allocation for attention/ffn) | ✅ |
| P6-GGUF-08 | Export | imatrix support (importance matrix from calibration data, required for IQ types) | ✅ |
| P6-GGUF-09 | Export | Load and inference verification with llama.cpp (all quant types) | ✅ |

### 6.5 Benchmark Evaluation

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-BENCH-01 | Benchmark | Dataset loader (load user-specified test data) | ✅ |
| P6-BENCH-02 | Benchmark | Perplexity measurement (language model: cross-entropy → exp) | ✅ |
| P6-BENCH-03 | Benchmark | Accuracy / Top-k accuracy measurement (classification tasks) | ✅ |
| P6-BENCH-04 | Benchmark | JSON output (metrics + per-sample scores + summary statistics) | ✅ |

### Acceptance tests

- `nabla-train/tests/profiler.rs`: CudaEvent timing + per-layer stats + roofline classification + JSON output
- `nabla-train/tests/quantize.rs`: AWQ calibration + INT4 pack/unpack round-trip + dequant-matmul accuracy
- `nabla-train/tests/export_onnx.rs`: Module → ONNX → onnxruntime inference → numerical match
- `nabla-train/tests/export_gguf.rs`: Module → GGUF → Q4_0/Q4_K → llama.cpp load verification
- `nabla-train/tests/benchmark.rs`: Dataset load → perplexity/accuracy measurement → JSON output

---

## §7 Phase 7 — nabla-interface (Done ✅)

| Phase | REQ count | Summary |
|---|---|---|
| 7 | 29 | GGUF v3 export + quantization (34 types) + llama.cpp FFI + M-series Metal inference |

Target crate: `nabla-interface` (Layer 5: Export + Inference)
- GGUF v3 binary writer (pure Rust)
- 34-type quantization packing (Legacy 6 + K-quant 9 + IQ 8 + Ternary 2 + Full 4 + Integer 4)
- nabla Module → GGUF tensor name mapping
- llama.cpp C API FFI (system lib via pkg-config, `brew install llama.cpp`)
- Metal inference pipeline (M-series Apple Silicon)

Acceptance tests:
- `nabla-interface/tests/gguf_writer.rs`: GGUF v3 binary write → magic/version/metadata/tensor_info byte verification (5 tests)
- `nabla-interface/tests/quant_roundtrip.rs`: Q4_0/Q8_0/Q4_K_M quantize→dequantize round-trip error bounds (5 tests)
- `nabla-interface/tests/convert.rs`: nabla Tensor → GGUF file output → file size and metadata verification (5 tests)
- `nabla-interface/tests/llama_load.rs`: GGUF → llama.cpp load → tokenize/detokenize (1 test, requires llama.cpp)
- `nabla-interface/tests/llama_inference.rs`: GGUF load → generate → text generation + perf stats (1 test, requires llama.cpp)
