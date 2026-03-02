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
| CUDA Graph capture/replay | cuda | `cuda_graph_capture` API, 1.4–1.6x speedup (MLP training) |
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
| **B. Pooling** | 4 | max/avg/adaptive_avg pool2d, max_pool1d | ✅ (2d/1d) |
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
| matmul 4096 (cuBLAS TF32) | 358 µs | 2675 µs | **nabla 7.5x faster** |
| matmul 1024 | 36 µs | 58 µs | **nabla 1.6x faster** |
| fuse exp+sin | 41 µs | 81 µs (eager) | **nabla 2.0x faster** |
| sin / cos / tanh | 40 µs | 41 µs | ~parity |
| add / sub / emul | 58 µs | 58 µs | ~parity |
| sum_all / max_all | 28 µs | 26 µs | PyTorch 1.08x |
| fuse 4-op speedup | 3.38x | 1.0x (torch.compile N/A) | nabla fused vs unfused |
| dispatch latency | 42-58 us/op | 40-58 us/op | ~parity |

### MLP training benchmark (GH200 480GB, CUDA 12.8)

MLP 784→256→128→10, leaky_relu(0.01), MSE sum loss, SGD lr=0.001, f32. Warmup=10, iters=100.

| Batch | nabla eager | nabla graph | PyTorch eager | PyTorch graph |
|------:|------------:|------------:|--------------:|--------------:|
| 1     | 66 µs    | **48 µs** | 767 µs     | 46 µs      |
| 32    | 96 µs    | **62 µs** | 863 µs     | 72 µs      |
| 128   | 88 µs    | **65 µs** | 867 µs     | 128 µs      |
| 256   | 92 µs    | **69 µs** | 855 µs     | 136 µs      |
| 512   | 98 µs    | **77 µs** | 872 µs     | 143 µs      |
| 1024  | 108 µs    | **86 µs** | 897 µs     | 160 µs      |

**Key results**: nabla eager 8.3–11.6× faster than PyTorch eager. nabla CUDA Graph wins batch≥32 (1.5–2.0× over PyTorch CUDA Graph), matches at batch=1. Optimizations: (1) eliminated redundant memset CUDA graph nodes via `Tensor::empty` for BLAS outputs, (2) GEMV dispatch for batch=1 matmul, (3) owned gradient propagation (`prop_owned`) eliminates D2D memcpy nodes in backward pass.

Reproducible via `benchmarks/` (Rust + Python side-by-side).

---

## §6 Implementation Design Notes

**Named axes**: `Tensor<T,B,Axes=()>` — compile error on axis mismatch. `StaticMatrix` const-generic shape algebra.

**TensorLike unification**: `TensorLike<T,B>` trait abstracts over `Tensor` and `Variable`, enabling generic compute functions. Module layers write math logic once via `tl_add`/`tl_matmul`/`tl_relu`/etc. — the trait dispatches to the correct implementation. `tensor_like_ops!` macro generates the trait definition + both impls from a concise DSL (binary/unary/unary_param categories). Used in `Activation` (7 activation functions) and `Linear` (matmul + bias via `impl_layer!`). Eliminates the Tensor/Variable "two-world" duplication for pure compute logic; parameter wrapping (`tape.variable()`) is auto-generated by `impl_layer!`.

**N-D tensor policy**: `NdTensor<T>` is CPU-only. GPU computation uses 2D (cuBLAS GEMM, FlashAttention, im2col+GEMM). N-D → 2D conversion: `slice_2d` / `into_2d`.

**GPU dispatch**: Switch between CPU and GPU with no code changes: `--features cpu` / `--features cuda`. No PyTorch-style `model.to('cuda')` needed — determined at compile time.

---

## §7 Kernel File Layout

### 7.1 CUDA Dtype Coverage

Full CUDA dtype coverage for `f32`, `f16`, `fp8` (`E4M3`, `E5M2`), and `fp4` (`E2M1`) across the Feature Catalog in §3, with no implicit CPU fallback in the CUDA path.

- Construction / cast / quantization: `fp32↔fp16↔fp8↔fp4`, blockwise fp4 quant/dequant
- Core math and reductions: unary/binary ops, `sum/max/min`, axis reductions, `cumsum/cumprod`, `prod`
- NN ops: activations, softmax/log-softmax, layer/rms/batch/group norm, losses
- Convolution and attention: im2col family, conv transpose 2d, SDPA
- Pooling: `max_pool1d` GPU-path compatible via backend pooling dispatch

### 7.2 Kernel Source Files

CUDA/HIP kernel sources split into focused `.cuh` units, wired through `common/kernels/mod.rs`:

- `k_basic_core.cuh`, `k_basic_math.cuh`, `k_basic_red32.cuh`, `k_basic_red64.cuh`
- `k_norm_softmax.cuh`, `k_norm_group.cuh`, `k_norm_reduce.cuh`, `k_norm_pool.cuh`
- `k_conv_bn_loss.cuh`, `k_attn.cuh`, `k_conv_misc.cuh`

---

## §8 CLI Tool — `nabla`

A standalone binary (`nabla-cli` crate) providing four subcommands for hardware diagnostics, benchmarking, model export, and inference. No Python required.

```
nabla <SUBCOMMAND> [OPTIONS]
```

Install: `cargo install --path nabla-cli` or `cargo install nabla-cli` (crates.io).

---

### 8.1 `nabla info` — Hardware Diagnostics

Detect and display available GPU backends, device properties, and VRAM.

```
nabla info [--json]
```

**Output (human-readable)**:
```
nabla hardware info
───────────────────────────────────────────
Backend : CUDA 12.8
Device  : NVIDIA GH200 480GB (device 0)
VRAM    : 476 GiB total / 473 GiB free
Compute : sm_90 (Hopper)
───────────────────────────────────────────
Backend : CPU
Cores   : 72 (logical)
RAM     : 251 GiB total / 228 GiB free
```

**Behaviour**:
- Probe in order: CUDA → ROCm/HIP → wgpu (Metal/Vulkan/DX12) → CPU
- Print one block per detected backend; skip unavailable backends silently
- `--json` outputs machine-readable JSON (suitable for CI / scripting)
- Exit code 0 if at least one GPU backend is found; 1 if CPU-only

| REQ-ID | Requirement | Status |
|---|---|---|
| CLI-INFO-01 | Detect CUDA devices via `cuDeviceGetCount` + `cuDeviceGetAttribute` | ✅ |
| CLI-INFO-02 | Detect ROCm/HIP devices via `hipGetDeviceCount` + `hipDeviceGetAttribute` | ✅ |
| CLI-INFO-03 | Detect wgpu adapters (Metal/Vulkan/DX12) via `wgpu::Adapter::request_adapter` | ✅ |
| CLI-INFO-04 | Report VRAM total/free (CUDA: `cuMemGetInfo`; HIP: `hipMemGetInfo`) | ✅ |
| CLI-INFO-05 | `--json` flag: emit structured JSON matching the human-readable fields | ✅ |
| CLI-INFO-06 | Exit code 0 = GPU found; exit code 1 = CPU-only | ✅ |

---

### 8.2 `nabla bench` — Benchmark Runner

Run matrix multiply and MLP training-step benchmarks matching the README figures.

```
nabla bench [--workload matmul|mlp|all] [--batch 128,512] [--backend cuda|cpu] [--iters 100] [--warmup 10] [--json]
```

**Examples**:
```bash
nabla bench --workload matmul                   # 4096x4096 f32 matmul, CUDA
nabla bench --workload mlp --batch 1,32,128,512 # MLP training step, all batch sizes
nabla bench --workload all --json               # full suite, JSON output
```

**Output**:
```
nabla bench — MLP 784→256→128→10, leaky_relu, SGD, f32
─────────────────────────────────────────────────────────
 batch │ nabla eager │ nabla graph │ vs PyTorch eager
───────┼─────────────┼─────────────┼──────────────────
   128 │      88 µs  │      65 µs  │ 9.8× faster
   512 │      98 µs  │      77 µs  │ 8.9× faster
```

**Workloads**:

| Workload | Description |
|---|---|
| `matmul` | Square f32 matmul; default sizes 1024, 4096; cuBLAS TF32 |
| `mlp` | MLP 784→256→128→10, leaky_relu(0.01), MSE, SGD; training step time |
| `all` | Both workloads in sequence |

| REQ-ID | Requirement | Status |
|---|---|---|
| CLI-BENCH-01 | `--workload matmul`: run square f32 GEMM at `--sizes` (default `1024,4096`), report µs ± σ | ✅ |
| CLI-BENCH-02 | `--workload mlp`: run MLP training step at each `--batch` size, report µs ± σ | ✅ |
| CLI-BENCH-03 | `--warmup N` / `--iters N` with CudaEvent timing (CUDA backend) or `std::time::Instant` (CPU) | ✅ |
| CLI-BENCH-04 | `--backend` selects feature-gated backend; error if requested backend not compiled in | ✅ |
| CLI-BENCH-05 | `--json` emits per-workload JSON array `[{workload, batch, eager_us, graph_us}, ...]` | ✅ |
| CLI-BENCH-06 | Displayed table matches format of README benchmark table | ✅ |

---

### 8.3 `nabla export` — Model Export & Quantization

Convert a trained nabla model to GGUF or ONNX.

```
nabla export <MODEL_PATH> --format gguf|onnx [--quant Q4_K_M|Q8_0|F16|...] [--out <PATH>] [--imatrix <PATH>]
```

**Examples**:
```bash
nabla export ./checkpoints/model.bin --format gguf --quant Q4_K_M
nabla export ./checkpoints/model.bin --format onnx --out model.onnx
nabla export ./checkpoints/model.bin --format gguf --quant IQ4_XS --imatrix calib.imatrix
```

**Behaviour**:
- `MODEL_PATH`: path to a nabla checkpoint written by `save_tensors` / `Trainer::save`
- `--quant` is GGUF-only; ONNX always exports at the checkpoint's native precision
- `--out` defaults to `<MODEL_PATH_STEM>.<format>` in the same directory
- `--imatrix` enables importance-matrix quantization (required for `IQ*` types)
- Prints export summary: format, quant type, output size, tensor count

| REQ-ID | Requirement | Status |
|---|---|---|
| CLI-EXP-01 | Load nabla checkpoint via `load_tensors` + reconstruct `Module` graph | ✅ |
| CLI-EXP-02 | GGUF export: delegate to `nabla-interface` `GgufWriter`; all 34 quant types supported | ✅ |
| CLI-EXP-03 | ONNX export: delegate to `nabla-train` ONNX export; opset 21 | ✅ |
| CLI-EXP-04 | `--imatrix` path loaded and passed to `GgufWriter` for IQ-type calibration | ✅ |
| CLI-EXP-05 | Print summary: format, quant type, output path, file size, tensor count | ✅ |
| CLI-EXP-06 | Error on `--quant` with `--format onnx` (ONNX has no quant type arg) | ✅ |

---

### 8.4 `nabla run` — GGUF Inference (Streaming)

Run text generation from a GGUF file via `nabla-interface` + llama.cpp.

```
nabla run <GGUF_PATH> --prompt <TEXT> [--max-tokens 256] [--temp 0.8] [--ctx 2048] [--stream]
```

**Examples**:
```bash
nabla run ./model.Q4_K_M.gguf --prompt "Explain nabla in one sentence"
nabla run ./model.gguf --prompt "Hello" --max-tokens 512 --stream
```

**Behaviour**:
- Loads GGUF via `nabla-interface` llama.cpp FFI bridge
- `--stream` (default on) prints tokens as they are generated; `--no-stream` collects and prints at end
- Reports tok/s at completion
- Requires `nabla-cli` compiled with `--features llama` (links `llama.cpp` via pkg-config)

```
> nabla run ./llama3-8b.Q4_K_M.gguf --prompt "What is nabla?"

nabla is a zero-GC Rust tensor engine...
─────────────────────────────────────────
Generated 64 tokens in 1.2 s (53.3 tok/s)
```

| REQ-ID | Requirement | Status |
|---|---|---|
| CLI-RUN-01 | Load GGUF via `nabla-interface` `LlamaModel::from_gguf(path)` | ✅ |
| CLI-RUN-02 | Streaming output: print each token to stdout as generated (`\r\n` flush) | ✅ |
| CLI-RUN-03 | `--no-stream`: buffer all tokens, print complete response at end | ✅ |
| CLI-RUN-04 | Report total tokens and tok/s at completion | ✅ |
| CLI-RUN-05 | `--temp`, `--ctx`, `--max-tokens` forwarded to llama.cpp sampler config | ✅ |
| CLI-RUN-06 | Requires `features = ["llama"]`; error with clear message if llama.cpp not linked | ✅ |

---

### Acceptance Tests

- `nabla-cli/tests/info.rs`: `nabla info` exits 0 on CI (GPU available), JSON output parses correctly
- `nabla-cli/tests/bench.rs`: `nabla bench --workload matmul --iters 5 --warmup 1` completes without error; JSON output schema matches
- `nabla-cli/tests/export.rs`: export small Linear model to GGUF Q4_0 and ONNX; verify file exists and size > 0
- `nabla-cli/tests/run.rs`: skipped unless `features = ["llama"]` and GGUF fixture present
