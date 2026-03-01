# nabla — Specification

> **AI AGENT 必読**: §0 が**唯一の行動命令書**。Part I–IV は参照資料。着手前に §0 を全て確認すること。

**Ground Truth**: `docs/spec.md`（このファイル）が唯一の仕様書。コード・コメントと矛盾する場合、このファイルを優先する。

**完了条件**: 全 REQ 実装 + 全受け入れテスト通過 = Done。1 REQ でも未実装なら done 禁止。

Legend: ✅ Implemented | ❌ Not possible (language constraint) | 🔲 Not yet implemented

---

## §0 Binding Contract

### §0.0 NON-GOALS（実装禁止）

| # | 禁止対象 |
|---|---|
| NG-01 | CPU fallback for any GPU backend operation |
| NG-02 | Multiple active backends at compile time |
| NG-03 | `unwrap()`/`expect()` in library code |
| NG-04 | Architecture decisions (model structure, training loop, optimizer) |
| NG-05 | Internal unit tests (`#[cfg(test)] mod tests`) — use `nabla/tests/*.rs` only |
| NG-06 | CubeCL — use direct WGSL/CUDA/HIP C kernel strings |
| NG-07 | f64 on wgpu backend (WGSL/Metal lacks f64 hardware support) |
| NG-08 | c32/c64 on any GPU backend |
| NG-09 | `unimplemented!()`/`todo!()` in any `pub` function |
| NG-10 | Synchronous GPU→CPU transfer except on `.get()`/`.to_vec()` callsite |

### §0.1 型・形状規約（Type & Shape Conventions）

```rust
// Scalar types
T: f32 | f64 | c32 | c64   // ComplexField + MathOps + ReductionOps + Copy + Send + Sync

// Tensor types
Tensor<T>                    // = Tensor<T, DefaultBackend>; row-major flat storage
StaticMatrix<T, R, C>        // stack-allocated; R, C: const usize
NdTensor<T>                  // N-D CPU-only; flat Vec<T> row-major
DynTensor                    // enum{F32,F64,C32,C64}(Tensor<_>); runtime scalar dispatch

// Indexing: 0-indexed, tuple for multi-dim (Index trait constraint)
a[(i, j)]          // read
a[(i, j)] = v      // write
a.slice(0..3, 1..4) // owned copy via .slice() (Index<Range> impossible — must return &Output)

// Error handling: Result<T, NablaError> — no silent NaN
a.solve(&b)?       // ? propagates NablaError

// Kernel naming convention (CUDA/HIP C strings in cuda_hip/kernels.rs)
k_{op_name}_f32    // e.g. k_conv2d_f32, k_max_pool2d_f32
k_{op_name}_f64    // f64 variant (CUDA/HIP only; not wgpu)

// Conv tensor layout: NCHW (N=batch, C=channels, H=height, W=width)
// H_out = (H + 2*padding - dilation*(kH-1) - 1) / stride + 1
```

### §0.2 REQ表

**Phase 0 (実装済 ✅ — API 契約、変更禁止)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 0 | REQ-B01 | MUST | Exactly one of {cpu,wgpu,cuda,hip} active per build | `nabla-core/src/common/backend.rs` |
| 0 | REQ-B02 | MUST NOT | Multiple features active — `compile_error!` on all 6 pairwise combinations | `nabla-core/src/lib.rs` |
| 0 | REQ-B03 | MUST NOT | CPU fallback path exist for GPU backends | `nabla-core/src/wgpu/ops.rs` |
| 0 | REQ-T01 | MUST | `use nabla::prelude::*;` imports all public types/traits/macros/free-fns | `nabla/src/lib.rs` |
| 0 | REQ-T02 | MUST | `Tensor<T>` aliases to `Tensor<T, DefaultBackend>` | `nabla-core/src/common/tensor/mod.rs` |
| 0 | REQ-T03 | MUST NOT | wgpu backend accept f64 scalar (compile_error!) | `nabla-core/src/common/backend.rs` |
| 0 | REQ-T04 | MUST NOT | Any GPU backend accept c32/c64 scalar (compile_error!) | `nabla-core/src/common/backend.rs` |
| 0 | REQ-T05 | MUST | Backend trait: Phase 0 methods are required (no defaults); Phase 3+ methods have CPU defaults that GPU backends override | `nabla-core/src/common/backend.rs` |

**Phase 3A — GPU Convolution (CUDA/HIP; cross-ref §14.1.A)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3A | REQ-G-CONV-01 | MUST | `conv2d(x,w,bias,stride,padding,dilation,groups)` dispatches to GPU im2col+GEMM kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3A | REQ-G-CONV-02 | MUST | conv2d im2col col shape: `[N, C_in*kH*kW, out_H*out_W]`; weight `[C_out, C_in*kH*kW]`; strided-batched GEMM with batch=N | `cuda_hip/kernels.rs` |
| 3A | REQ-G-CONV-03 | SHOULD | f32 conv2d im2col kernel uses float4 (128-bit) loads (scalar reads currently; performance optimization) | `cuda_hip/kernels.rs` |
| 3A | REQ-G-CONV-04 | MUST | `conv1d(x,w,bias,stride,padding,dilation,groups)` dispatches to GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3A | REQ-G-CONV-05 | MUST | `conv3d(x,w,bias,stride,padding,dilation,groups)` dispatches to GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3A | REQ-G-CONV-06 | MUST | `conv_transpose2d(x,w,bias,stride,padding,output_padding,groups)` dispatches to GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3A | REQ-G-CONV-07 | MUST NOT | conv GPU kernels accept f64 on wgpu backend | `nabla-core/src/common/backend.rs` (wgpu uses CPU default) |

**Phase 3B — GPU Pooling (CUDA/HIP; cross-ref §14.1.B)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3B | REQ-G-POOL-01 | MUST | `max_pool2d(x,kernel_size,stride,padding)` dispatches to GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3B | REQ-G-POOL-02 | MUST | `avg_pool2d(x,kernel_size,stride,padding)` dispatches to GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3B | REQ-G-POOL-03 | MUST | `adaptive_avg_pool2d(x,output_size:[usize;2])` dispatches to GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3B | REQ-G-POOL-04 | MUST | `max_pool2d` kernel stores argmax indices alongside output for backward | `cuda_hip/kernels.rs` |
| 3B | REQ-G-POOL-05 | MUST | All pooling kernels use one-thread-per-output-element parallelism | `cuda_hip/kernels.rs` |

**Phase 3C — GPU Attention & Batched GEMM (CUDA/HIP; cross-ref §14.1.F, §14.1.H)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3C | REQ-G-ATTN-01 | MUST | `sdpa(q,k,v,mask,dropout_p)` dispatches to FlashAttention-2 tiled GPU kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3C | REQ-G-ATTN-02 | MUST | FlashAttention-2 implements online softmax with O(seq_len) HBM memory | `cuda_hip/kernels.rs` |
| 3C | REQ-G-ATTN-03 | MUST NOT | FlashAttention kernel materialise full QK^T matrix in HBM | `cuda_hip/kernels.rs` |
| 3C | REQ-G-ATTN-04 | MUST | `bmm(a,b)` f32 dispatches to `cublasSgemmStridedBatched` on CUDA backend | `cuda_hip/cuda.rs` |
| 3C | REQ-G-ATTN-05 | MUST | `baddbmm(c,a,b,beta,alpha)` dispatches to cuBLAS fused op on CUDA | `cuda_hip/cuda.rs` |
| 3C | REQ-G-ATTN-06 | MUST | `addmm(c,a,b,beta,alpha)` dispatches to cuBLAS fused op on CUDA | `cuda_hip/cuda.rs` |
| 3C | REQ-G-ATTN-07 | MUST | bmm/baddbmm/addmm on HIP/wgpu use native tiled matmul loop (no cuBLAS) | `cuda_hip/hip.rs`, `wgpu/ops.rs` |

**Phase 3D — GPU Reductions (CUDA/HIP; cross-ref §14.1.J)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3D | REQ-G-RED-01 | MUST | `cumsum(x,dim)` dispatches to GPU parallel prefix sum (Blelloch scan) kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3D | REQ-G-RED-02 | MUST | `cumprod(x,dim)` dispatches to GPU parallel prefix (Blelloch scan) kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3D | REQ-G-RED-03 | MUST | `prod_all(x)` dispatches to GPU warp-shuffle reduction kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3D | REQ-G-RED-04 | MUST | `norm(x,p,dim)` Lp-norm dispatches to GPU kernel for p∈{1,2,inf} | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3D | REQ-G-RED-05 | MUST | `count_nonzero(x)` dispatches to GPU reduction kernel | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3D | REQ-G-RED-06 | MUST | cumsum/cumprod kernel output: `out[i,j] == sum/prod(x[i,0..=j])` for dim=1 | `cuda_hip/kernels.rs` |

**Phase 3E — GPU Normalization & Loss (CUDA/HIP; cross-ref §14.1.C, §14.1.E)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3E | REQ-G-NORM-01 | MUST | `batch_norm` GPU kernel updates running_mean/running_var in-place with momentum | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3E | REQ-G-LOSS-01 | MUST | `cross_entropy_loss` GPU: fused log_softmax + nll_loss in single kernel pass | `cuda_hip/cuda.rs`, `cuda_hip/kernels.rs` |
| 3E | REQ-G-LOSS-02 | MUST | `cross_entropy_loss` fused kernel uses online-softmax (numerically stable) | `cuda_hip/kernels.rs` |

### §0.3 受け入れテスト（Acceptance Tests）

テストファイル: `nabla/tests/gpu.rs` (feature-gated)
実行: `cargo test --features cuda -- gpu_phase3`

```rust
// nabla/tests/gpu.rs
#[cfg(feature = "cuda")]
mod gpu_phase3 {
    use nabla::prelude::*;

    // REQ-G-CONV-01, REQ-G-CONV-02
    #[test]
    fn test_conv2d_gpu_shape() {
        // x: [N=2, C_in=3, H=8, W=8], w: [C_out=4, C_in=3, kH=3, kW=3]
        // stride=1, padding=1, dilation=1, groups=1
        // expected output shape: [2, 4, 8, 8]
        let x: Tensor<f32> = randn(2 * 3 * 8 * 8).reshape(2, 3 * 8 * 8); // placeholder
        // conv2d(x, w, None, [1,1], [1,1], [1,1], 1) -> shape [2,4,8,8]
    }

    // REQ-G-POOL-01, REQ-G-POOL-04
    #[test]
    fn test_max_pool2d_gpu_shape() {
        // x: [2, 3, 8, 8], kernel_size=2, stride=2, padding=0
        // expected output shape: [2, 3, 4, 4]
    }

    // REQ-G-ATTN-01, REQ-G-ATTN-02, REQ-G-ATTN-03
    #[test]
    fn test_sdpa_flash_gpu_numeric() {
        // Q,K,V: [batch=2, heads=4, seq=64, head_dim=32]
        // output shape: [2, 4, 64, 32]
        // Compare with naive: atol ≤ 1e-3 (online-softmax numerical equiv)
    }

    // REQ-G-ATTN-04
    #[test]
    fn test_bmm_gpu_shape() {
        // a: [batch=4, m=32, k=16], b: [batch=4, k=16, n=32]
        // expected: [4, 32, 32]
    }

    // REQ-G-RED-01, REQ-G-RED-06
    #[test]
    fn test_cumsum_gpu_correctness() {
        // x: [4, 8] f32
        // cumsum(x, dim=1)[i,j] == (x[i,0] + ... + x[i,j]) for all i,j
    }

    // REQ-G-RED-02
    #[test]
    fn test_cumprod_gpu_correctness() {
        // x: [4, 8] f32; cumprod(x,1)[i,j] == product(x[i,0..=j])
    }

    // REQ-G-NORM-01
    #[test]
    fn test_batch_norm_gpu_running_stats() {
        // Two forward passes; check running_mean shifts toward batch mean
        // momentum=0.1; after 1st pass: running_mean = 0.9*0 + 0.1*batch_mean
    }

    // REQ-G-LOSS-01, REQ-G-LOSS-02
    #[test]
    fn test_cross_entropy_gpu_matches_cpu() {
        // logits: [batch=8, classes=10]; targets: [8] (class indices)
        // GPU result must match CPU result within atol=1e-5
    }
}
```

### §0.4 Ground Truth Parameters

```
GPU kernel hyperparameters (pinned):
  BLOCK_SIZE:          256  (default; tunable per kernel via autotune — do NOT hardcode)
  float4 vectorization: all f32 unary/binary kernels (128-bit LDG.E.128 + scalar tail)
  warp size:           32  (CUDA/HIP)
  workgroup size (wgpu): 256

FlashAttention-2 tile sizes:
  BLOCK_M: 64   (query tile along seq_len)
  BLOCK_N: 64   (key/value tile along seq_len)
  HEAD_DIM_MAX: 128  (current constraint; compile_error! if exceeded)

im2col+GEMM for conv2d:
  im2col col:    [N, C_in * kH * kW, out_H * out_W]  (strided-batched, batch=N)
  weight matrix: [C_out,  C_in * kH * kW]
  GEMM: cublasSgemmStridedBatched(col[b], weight) -> out[b] for b=0..N

Blelloch parallel scan:
  Up-sweep + down-sweep; O(n) work, O(log n) depth
  Shared memory per block: 2 * BLOCK_SIZE * sizeof(T)

Pinned dependency versions:
  Rust edition: 2024
  wgpu:    24
  cudarc:  0.19
  rayon:   1
  faer:    latest (no pinned version)
```

### §0.5 Phase依存グラフ

```
Phase 0 (基盤 ✅ — 変更禁止)
  ├─ Backend trait (sealed) + Tensor<T,B>
  ├─ 74 CUDA/HIP kernels (element-wise, matmul, reduction, activations, softmax, norms)
  ├─ 43 wgpu WGSL kernels (f32 only)
  ├─ GPU caching allocator best-fit dual-pool (W20)
  └─ fuse! L1 JIT fusion via NVRTC/hiprtc (W19)

Phase 3A (GPU Convolution ✅) ← depends on: Phase 0 matmul_into, im2col scratch
  ├─ REQ-G-CONV-01  conv2d
  ├─ REQ-G-CONV-04  conv1d
  ├─ REQ-G-CONV-05  conv3d
  └─ REQ-G-CONV-06  conv_transpose2d

Phase 3B (GPU Pooling ✅) ← depends on: Phase 0 GPU memory allocator
  ├─ REQ-G-POOL-01  max_pool2d  (+ argmax indices)
  ├─ REQ-G-POOL-02  avg_pool2d
  └─ REQ-G-POOL-03  adaptive_avg_pool2d

Phase 3C (GPU Attention ✅) ← depends on: Phase 0 matmul_into + softmax kernel
  ├─ REQ-G-ATTN-01  sdpa / FlashAttention-2
  ├─ REQ-G-ATTN-04  bmm (cuBLAS StridedBatched)
  ├─ REQ-G-ATTN-05  baddbmm
  └─ REQ-G-ATTN-06  addmm

Phase 3D (GPU Reductions ✅) ← depends on: Phase 0 warp-shuffle reduction
  ├─ REQ-G-RED-01  cumsum
  ├─ REQ-G-RED-02  cumprod
  ├─ REQ-G-RED-03  prod_all
  ├─ REQ-G-RED-04  norm Lp
  └─ REQ-G-RED-05  count_nonzero

Phase 3E (GPU Norm/Loss ✅) ← depends on: Phase 0 layer_norm kernel + softmax
  ├─ REQ-G-NORM-01  batch_norm GPU (running stats)
  └─ REQ-G-LOSS-01/02  cross_entropy fused GPU

Phase 4 (performance ✅ W20)
  ├─ Multi-stream async pipeline (D2H overlap + DoubleBuffer)
  ├─ L4 Mega-kernel tiled fusion (shared memory tile reuse)
  ├─ CUDA Graph pointer indirection (PyGraph)
  └─ Conditional Nodes (CUDA 12.4+ IF/WHILE/SWITCH)
```

---

# Part I — Current Implementation

## 1. Overview

**Zero-GC, zero-copy, type-safe** Rust linear algebra DSL for researchers who refuse to choose between Python's ease and C++'s speed.
Proc macros (`mat![]`, `map!{}`, `einsum!{}`) combined with self-contained pure-Rust kernels (CPU) and GPU compute shaders across three backends: wgpu (WGSL), CUDA (nvrtc), HIP (hiprtc).
Exactly one backend is selected via feature flags at build time — no implicit CPU fallback, no cross-backend runtime dispatch.

### Fixed Rule Principle — computation engine, not framework

nabla は **計算エンジン** であり、フレームワークではない。数学的に不変な計算プリミティブ（matmul, conv, softmax, cross_entropy 等）を CPU/GPU で最速に実行する。アーキテクチャ設計・モデル構成・学習ループ・最適化戦略はユーザーの責務。

> **Criterion**: "Is this a fixed mathematical computation, or an architecture decision?"
> Fixed computation → nabla provides. Architecture decision → user composes nabla primitives.

| nabla provides | User decides |
|---|---|
| matmul, conv, bmm, SDPA, embedding | layer stacking, skip connections |
| activations (relu/gelu/silu/softmax) | which activation where |
| loss (cross_entropy/mse/l1/kl_div) | which loss to optimize |
| norm (layer/rms/batch/group) | where to normalize |
| reductions, reshape, gather, scatter | data flow topology |
| AD (reverse/forward), ODE, CAS | what to differentiate, solver choice |
| — | optimizer, training loop, data loading |

### Design principles

1. **Computation engine, not framework** — nabla provides optimized computation primitives; users compose them into any architecture. No opinions on model structure, training loop, or optimizer
2. **Zero-GC, zero-copy** — ownership = automatic memory management without GC. `Drop` = deterministic deallocation. `&` = zero-copy borrow. `_into(out: &mut)` = zero-allocation in-place
3. **Python's ease, C's speed** — PyTorch-familiar API (`loss.backward()`, `.exp()`, `.sum()`) with NumPy-like broadcasting (`map!`), delivered at native speed
4. **Macros = notation layer** — proc macros for concise syntax, type-safe Rust underneath
5. **trait = dispatch** — trait-based multiple dispatch (Python duck-typing composability without runtime cost)
6. **Self-contained LA** — row-major CpuStorage, 9 dense factorizations, CSC sparse. Zero external LA deps
7. **Build-time exclusive backend** — `cpu`/`wgpu`/`cuda`/`hip` feature flags, exactly one active. `compile_error!` on multi-select. No implicit CPU fallback
8. **Two kernel codebases** — WGSL (wgpu) + CUDA/HIP shared C source. ❌CubeCL. Fixed-rule ops → manageable dual maintenance
9. **Fixed-rule principle** — only mathematically invariant computations. Architecture decisions excluded
10. **Adjoint ≠ Transpose** — correct complex LA semantics
11. **Maximum primitive coverage** — every computation a user might need is provided. Edge cases require minimal user-side extension, never reimplementation

---

## 2. Project structure

### 2.1 Three-layer architecture

```
Layer 1: Notation (nabla-macros)     — proc macros for concise DSL syntax
Layer 2: Compute  (nabla-core)       — tensor types + GPU/CPU backends + storage
Layer 3: Application (nabla)         — dense LA + ML training + AD + CAS + ODE + IO
```

**Design principle**: Each layer depends only on layers below it. Layer 1 has zero runtime deps. Layer 2 is a pure computation engine. Layer 3 composes Layer 2 primitives into domain-specific APIs.

### 2.2 Directory layout

```
nabla/                       [workspace root]
├── Cargo.toml               members: nabla-core, nabla-macros, nabla
│
├── nabla-macros/            ━━ Layer 1: Notation ━━
│   └── src/
│       ├── lib.rs           proc-macro entry: re-exports all macros
│       ├── macros/
│       │   ├── mod.rs       re-exports
│       │   ├── fuse.rs      fuse! / mega_fuse! / fuse_! codegen
│       │   ├── einsum.rs    einsum! parser + contraction path + codegen
│       │   ├── mat.rs       mat! / block! matrix literal parsing
│       │   ├── stencil.rs   stencil! offset indexing
│       │   ├── grad.rs      nabla_grad! source-transform AD
│       │   └── sym.rs       sym! Pratt-parser symbolic expression macro
│       └── fusion/
│           ├── mod.rs       fuse! IR entry
│           ├── expr.rs      FuseExpr AST
│           ├── eqsat.rs     e-graph equality saturation
│           └── codegen.rs   CUDA/HIP C kernel string codegen
│
├── nabla-core/              ━━ Layer 2: Compute ━━
│   └── src/
│       ├── lib.rs            #[path] routing + feature-gate compile_error!
│       ├── common/           shared code (cpu + gpu)
│       │   ├── backend.rs    Backend trait (sealed) + DefaultBackend + NablaError + Result (~1179L)
│       │   ├── layout.rs     LinearLayout<N> F₂ binary matrix swizzle (164L)
│       │   ├── scalar/
│       │   │   ├── mod.rs    Scalar/MathOps/ReductionOps traits + f32/f64/half impls (450L)
│       │   │   ├── complex.rs  Complex<T>, c32/c64 (408L)
│       │   │   ├── dual.rs   Dual<T> forward-mode AD (591L)
│       │   │   └── multi_dual.rs  MultiDual<T,N> (420L)
│       │   └── tensor/
│       │       ├── mod.rs    Tensor<T,B> core struct + MatrixLike + trait impls (574L)
│       │       ├── constructors.rs  zeros/ones/identity/rand/from_fn/fill/linspace (449L)
│       │       ├── ops.rs    arithmetic + element-wise + broadcast + operator overloads (781L)
│       │       ├── shape.rs  shape/reshape/broadcast/TensorView/iter/display (639L)
│       │       ├── reductions.rs  sum/max/min/argmax/argmin/norm axis-wise (528L)
│       │       ├── variants.rs  NdTensor/StaticMatrix/DynTensor (769L)
│       │       ├── nn_conv.rs   conv1d/2d/3d/transpose + pooling + config builders (538L)
│       │       └── nn_ops.rs    activations + batch_norm + losses + attention (546L)
│       ├── cpu/
│       │   └── mod.rs        CpuStorage<T> + Cpu struct + impl Backend for Cpu (566L)
│       ├── wgpu/             wgpu backend (feature = "wgpu")
│       │   ├── mod.rs        re-exports
│       │   ├── storage.rs    GpuStorage<T> + GpuContext + wgpu dispatch helpers (338L)
│       │   ├── shaders.rs    WGSL register-tile MMA codegen + kernel strings (898L)
│       │   └── ops.rs        impl Backend for Gpu (wgpu) + all gpu_* fns (1103L)
│       └── cuda_hip/         CUDA + HIP backends (feature = "cuda" | "hip")
│           ├── mod.rs        re-exports common/*
│           ├── common/       shared code (both CUDA and HIP)
│           │   ├── mod.rs    re-exports rtc/pool/fuse
│           │   ├── rtc.rs    RtcStorage + MemoryPool + rtc_backend_impl! macro (520L)
│           │   ├── pool.rs   CUDA/HIP pooling kernels dispatch (585L)
│           │   ├── fuse.rs   fuse! kernel codegen dispatch (537L)
│           │   └── kernels.rs  CUDA/HIP C kernel source strings (NVRTC/hiprtc JIT) (2251L, data file)
│           ├── cuda.rs       CUDA backend — intentionally monolithic (5574L, CudaCtx singleton)
│           └── hip.rs        HIP backend — mirrors CUDA structure (2386L)
│
├── nabla/                   ━━ Layer 3: Application ━━
│   ├── src/
│   │   ├── lib.rs           prelude + macro_rules re-exports (slim entry)
│   │   ├── constructors.rs  free functions: zeros/ones/eye/rand/linspace
│   │   ├── notation.rs      operator shorthands + prelude re-exports
│   │   ├── module.rs        Module<T,B> trait + impl helpers
│   │   ├── optimizer.rs     Optimizer<T,B> trait + AdamW impl
│   │   ├── linalg/
│   │   │   ├── mod.rs       LinalgExt trait + re-exports
│   │   │   ├── lu.rs        LU factorization + solve + det
│   │   │   ├── qr.rs        QR factorization
│   │   │   ├── svd.rs       SVD + pseudoinverse + rank
│   │   │   ├── chol.rs      Cholesky (Llt/Ldlt) + solve
│   │   │   ├── eigen.rs     Schur/Hessenberg/symmetric eigen
│   │   │   ├── francis.rs   Francis QR iteration (implicit double-shift)
│   │   │   ├── solve.rs     direct solve / tridiag / lstsq / det
│   │   │   ├── matrix_fn.rs inv/pinv/logm/sqrtm/expm/matrix_power/polar/balance
│   │   │   ├── equation.rs  sylvester/lyapunov/continuous_riccati
│   │   │   └── structured.rs toeplitz/circulant/vandermonde/frechet_deriv
│   │   ├── nn.rs            activations + norms + embeddings + attention (Layer 3 high-level API)
│   │   ├── optim.rs         adamw_step + LrSchedule + GradScaler
│   │   ├── io.rs            save_tensors/load_tensors (NBLA binary)
│   │   ├── autograd.rs      Reverse-mode AD: Tape + Variable + backward
│   │   ├── cas.rs           Symbolic CAS: Expr tree + diff/simplify/eval (E-graph)
│   │   ├── sparse.rs        SparseMatrix<T> CSC + BcsrMatrix<T> GPU sparse
│   │   └── ode/
│   │       ├── mod.rs       ODE API: euler/rk4/dormand_prince + validate/alloc helpers
│   │       ├── stiff.rs     BDF1/BDF2 stiff solvers + fixed_point_converge
│   │       ├── advanced.rs  Parareal / DAE / Stormer-Verlet
│   │       └── sde.rs       Stochastic DE solvers
│   ├── examples/            cargo run --example <name>
│   │   ├── 01_matrix_ops.rs
│   │   ├── 02_least_squares.rs
│   │   ├── 03_svd_compress.rs
│   │   ├── 04_autograd_mlp.rs
│   │   ├── 05_ode_lorenz.rs
│   │   ├── 06_sparse_solve.rs
│   │   ├── 07_einsum_attention.rs
│   │   ├── 08_cas_symbolic.rs
│   │   ├── 09_dae_pendulum.rs
│   │   └── 10_half_precision.rs
│   └── tests/
│       ├── boundary.rs      CPU boundary tests (231, CUDA-compatible)
│       ├── gpu.rs           GPU backend tests (feature-gated)
│       └── einsum_errors/   trybuild compile-fail fixtures
└── docs/
    └── spec.md              this file
```

Dependencies:
- `nabla-core`: `rayon 1`, `faer`, `wgpu 24` (optional), `pollster 0.4` (optional)
- `nabla-macros`: `syn`, `quote`, `proc-macro2`
- `nabla`: `nabla-core`, `nabla-macros`, `faer`, `rayon 1`
- CUDA/HIP: `libloading` (dlopen libcuda.so/libnvrtc.so, libamdhip64.so/libhiprtc.so at runtime)

### 2.3 File size policy

**Target: 400–800 lines per file. Group by feature, not by responsibility.**

| Rule | Policy |
|---|---|
| Target range | **400–800 lines** per `.rs` file |
| Below 400 lines | Merge into the feature file it belongs to (do NOT create tiny single-purpose files) |
| Above 800 lines | Split by feature group (keep semantically related code together) |
| Module `mod.rs` | Thin re-exports + type aliases only; no logic |
| Grouping principle | Feature-cohesive. `conv.rs` = conv1d+2d+3d+transpose. `scalar.rs` = all scalar types. Never isolate a single type or single method into its own file |
| Max function length | 100 lines → refactor into sub-functions |

**Anti-patterns (禁止)**:
- `activations.rs` (75 lines) — too fine; merge into `nn.rs`
- `utils.rs` (45 lines) — too fine; inline into the calling module
- One type per file (e.g. `complex.rs`, `dual.rs` as separate files) — merge into one `scalar.rs` unless each exceeds 400 lines independently

**Exemptions (400-800L rule does not apply)**:
- `cuda_hip/cuda.rs` (5574L) — intentionally monolithic; `CudaCtx` singleton creates deep coupling. Do NOT split.
- `cuda_hip/hip.rs` (2386L) — mirrors CUDA structure; same reason.
- `cuda_hip/common/kernels.rs` (2251L) — data file (kernel source strings only); splitting adds no value.
- `wgpu/ops.rs` (1103L) — all `impl Backend for Gpu` methods; splitting would scatter a single trait impl.
- `wgpu/shaders.rs` (898L) — WGSL codegen + shader strings tightly coupled; acceptable overshoot.

---

## 3. Notation reference

`use nabla::prelude::*;` — all types, traits, macros, free functions available.

**Design goal: Python の書きやすさ × Julia の数式記法 × Rust のゼロ GC。**
記法 5 原則: (1) free function 型推論 `zeros(m,n)` (2) `Index` trait bracket 記法 `a[(i,j)]` (3) owned/borrowed 両形式 `a*b` / `&a*&b` (4) 短い名前 + `e`-prefix (`emul`/`ediv` = Julia `.* `/`./`) (5) Macro = syntax extension (`fuse!`=Julia `@.` auto-capture, `vcat!`=Julia `[A;B]`)

### 3.1 Quick reference — Math → Python → Julia → nabla

| Math | Python | Julia | nabla | nabla advantage |
|---|---|---|---|---|
| $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ | `np.array([[1,2],[3,4]])` | `[1 2; 3 4]` | `mat![[1, 2], [3, 4]]` | Compile-time shape check |
| $A_{ij}$ | `A[i,j]` | `A[i,j]` | `a[(i, j)]` | Bracket 記法, 0-indexed (`()` = Rust 制約) |
| $A_{ij} = v$ | `A[i,j] = v` | `A[i,j] = v` | `a[(i, j)] = v` | `IndexMut` — 同上 |
| $A_{1:3, 2:4}$ | `A[0:3, 1:4]` | `A[1:3, 2:4]` | `a.slice(0..3, 1..4)` | Owned copy (`Index<Range>` 不可 — Rust 言語制約) |
| $A^\top$ | `A.T` | `A'` | `a.t()` | 3 chars |
| $A^*$ (adjoint) | `A.conj().T` | `A'` (same!) | `a.h()` | **adjoint ≠ transpose, 4 chars** |
| $AB$ | `A @ B` | `A * B` | `a * b` | **Same as Julia** (owned) |
| $A \circ B$ (Hadamard) | `A * B` | `A .* B` | `a.emul(b)` | Julia `A .* B` の `e`-prefix 化 |
| $cA$ | `c * A` | `c * A` | `c * a` | **Same** |
| $Ax = b$ | `np.linalg.solve(A,b)` | `A \ b` | `a.solve(&b)?` | **Result — no silent NaN** |
| $\sin(A)$ element-wise | `np.sin(A)` | `sin.(A)` | `a.sin()` | **Shortest — method chain** |
| $y = \sin(x)^2$ fused | `torch.sin(x)**2` | `@. sin(x)^2` | `fuse!(x.sin().powf(2.0))` | **GPU kernel auto-fusion** (auto-capture) |
| $C = AB$ (einsum) | `np.einsum('ik,kj->ij',A,B)` | `@einsum` | `einsum!(c[i,j] = a[i,k] * b[k,j])` | **7 patterns + spanned errors** |
| $\nabla_x L$ | `loss.backward()` | `gradient(f, x)` | `loss.backward(); x.grad()` | **PyTorch-familiar + zero GC** |
| $\frac{df}{dx}$ symbolic | SymPy `diff(f, x)` | Symbolics.jl | `expr.diff("x").simplify()` | Built-in, single crate |
| $\dot{y} = f(t,y)$ | `scipy.integrate.solve_ivp` | DiffEq.jl | `dormand_prince(f, y0, t, dt)` | Built-in, single crate |
| $\nabla^2 u$ (Laplacian) | manual loop | `@tullio` | `stencil!(out[i,j] = ...)` | **Auto bounds, zero boundary** |
| $[A; B]$ vcat ✅ | `np.vstack([A,B])` | `[A; B]` | `vcat!(a, b, c, ...)` | Julia `[A; B]` style — variadic macro |
| $\langle u,v \rangle$ | `np.dot(u,v)` | `dot(u,v)` | `dot(&u, &v)` | **Scalar return** (not 1×1 matrix) |
| $A \otimes B$ (Kronecker) | `np.kron(A,B)` | `kron(A,B)` | `kron(&a, &b)` | Free function |
| $\det(A)$ | `np.linalg.det(A)` | `det(A)` | `a.det()?` | Via LU, Result |
| $\begin{bmatrix}A&B\\C&D\end{bmatrix}$ | `np.block([[A,B],[C,D]])` | `[A B; C D]` | `block![[a,b],[c,d]]` | Julia `[A B; C D]` style |

### 3.2 Tensor construction

| Operation | nabla | Python | Julia |
|---|---|---|---|
| Matrix literal | `mat![[1, 2], [3, 4]]` or `mat![1, 2; 3, 4]` | `np.array([[1,2],[3,4]])` | `[1 2; 3 4]` |
| Zeros / Ones / Fill | `zeros(m, n)` / `ones(m, n)` / `fill(m, n, val)` | `np.zeros((m,n))` | `zeros(m,n)` |
| Identity | `eye(n)` | `np.eye(n)` | `I(n)` |
| Random | `randn(m, n)` / `rand(m, n)` | `np.random.randn(m,n)` | `randn(m,n)` |
| From function | `from_fn(m, n, \|r, c\| expr)` | `np.fromfunction(f, (m,n))` | — |
| Range / Linspace | `arange(0.0, 1.0, 0.1)` / `linspace(0.0, 1.0, n)` | `np.arange` / `np.linspace` | `0.0:0.1:1.0` |
| Static (stack) | `StaticMatrix::<f64,3,3>::zeros()` | — | `SMatrix{3,3}(...)` |
| N-D | `nd_zeros(&[d0, d1, d2])` | `np.zeros((d0,d1,d2))` | `zeros(d0,d1,d2)` |
| Cat | `vcat!(a, b)` / `hcat!(a, b)` | `np.vstack` / `np.hstack` | `[A; B]` / `[A B]` |
| Reshape | `a.reshape(m, n)` | `A.reshape(m, n)` | `reshape(A, m, n)` |
| Diagonal matrix | `diagm(&v)` | `np.diag(v)` | `diagm(v)` |
| Block matrix | `block![[a,b],[c,d]]` | `np.block(...)` | `[A B; C D]` |

Free functions in `prelude`, type inferred from context. GC なし、即座解放、`StaticMatrix` はスタック配置。

### 3.3 Indexing & slicing

`Index`/`IndexMut` trait で **bracket 記法**（element access のみ）。0-indexed。Range スライスは `.slice()` メソッド（owned copy）。

| Operation | nabla | Python |
|---|---|---|
| Read/Write | `a[(i, j)]` / `a[(i, j)] = v` | `A[i,j]` / `A[i,j] = v` |
| Submatrix | `a.slice(0..3, 1..4)` | `A[0:3, 1:4]` |
| Row/Col slice | `a.slice_rows(0..3)` / `a.slice_cols(1..4)` | `A[0:3, :]` / `A[:, 1:4]` |
| N-D read | `t[&[i,j,k]]` | `T[i,j,k]` |

Rust 制約: `Index` trait は `&Output` を返す → element access `a[(i, j)]` のみ。Range slice は `Index` 不可（owned Tensor を返せない）→ `.slice()` メソッド。対価: 型安全 + 借用チェッカ保証。

### 3.4 Arithmetic

`a * b` = move（消費）、`&a * &b` = borrow（再利用）。

| Math | nabla (owned) | nabla (borrowed) | Python | Julia |
|---|---|---|---|---|
| $A + B$ / $A - B$ | `a + b` / `a - b` | `&a + &b` | `A + B` | `A + B` |
| $AB$ (matmul) | `a * b` | `&a * &b` | `A @ B` | `A * B` |
| $A \circ B$ / $A \oslash B$ | `a.emul(b)` / `a.ediv(b)` | `a.emul(&b)` | `A * B` / `A / B` | `A .* B` |
| $cA$ | `c * a` | `c * &a` | `c * A` | `c * A` |
| $C \mathrel{+}= AB$ | `c.mm_(&a, &b)` | — | `torch.mm(A,B,out=C)` | `mul!(C,A,B)` |
| $A^\top$ / $A^*$ | `a.t()` / `a.h()` | — | `A.T` / `A.conj().T` | `A'` |
| $A \otimes B$ | `a.kron(&b)` | `kron(&a, &b)` | `np.kron(A,B)` | `kron(A,B)` |
| $\langle u,v \rangle$ | `u.dot(&v)` | `dot(&u, &v)` | `np.dot(u,v)` | `dot(u,v)` |
| $u v^\top$ | `u.outer(&v)` | — | `np.outer(u,v)` | `u * v'` |
| $A \mathrel{+}= \alpha x$ | `a.axpy_(α, &x)` | — | — | `axpy!(α, x, y)` |
| $A \mathrel{+}= B$ | `a += &b` | — | `A += B` | `A .+= B` |
| $\alpha A$ (in-place) | `a *= α` | — | `A .*= α` | — |

`emul`/`ediv`: Julia `.*/./` の `e`-prefix 化。`&` = 明示的ゼロコピー + 再利用保証（Python/Julia にはない選択肢）。

**Auto Broadcasting (W27)**: `&a + &b` / `&a - &b` で shape 自動推論。`(m,n) + (1,n)` (row broadcast), `(m,n) + (m,1)` (col broadcast), `(m,n) + (1,1)` (scalar broadcast) を自動展開。owned form (`a + b`) は exact match のみ（move 最適化のため）。

**W28 additions**: `asin` `acos` `atan` `atan2` `sinh` `cosh` `asinh` `acosh` `atanh` `log2` `log10` — all element-wise. `epow(&b)` for tensor-tensor power. Owned operators: `a + b` / `a - b` (move semantics, allocation reuse).

**W29 additions**: `.tan()` element-wise. Commutative scalar*Tensor: `f32 * &Tensor` / `f64 * &Tensor` (+ and -). Owned matmul: `a * b` / `a * &b` / `&a * b` (all ownership combos). `Tensor::from_vec(data, nrows, ncols)` constructor. `.mean()` / `.prod()` global reduction aliases.

**W30 additions**: `Div<T>` operator for `&Tensor / scalar` and `scalar / &Tensor` (f32/f64). `zeros_vec(n)` / `ones_vec(n)` / `rand_vec(n)` / `randn_vec(n)` — single-arg column vector constructors. `prod_axis(axis)` reduction. `var_axis_ddof(axis, ddof)` for Bessel-corrected sample variance.

### 3.5 Broadcasting & fusion

| Level | nabla | PyTorch | GPU | Alloc |
|---|---|---|---|---|
| Single op | `a.sin()` | `torch.sin(a)` | ✅ | 1 |
| Fused chain | `fuse!(x.sin().powf(2.0))` | 2+ kernels | ✅ **1 kernel** | **1** |
| Closure | `map!(\|x\| f(x), &a)` | — | ❌ CPU | 1 |
| In-place | `map_!(a, \|x\| f(x), &b)` | `torch.sin(B, out=A)` | ❌ CPU | **0** |
| Parallel | `par_map!(\|x\| f(x), &a)` | — | ❌ CPU | 1 |
| Auto broadcast | `&a + &bias_row` (shape auto) | `a + bias` | ✅ | 1 |

**原則**: GPU → `fuse!` 一択。CPU closure → `map!`。in-place → `map_!`。並列 → `par_map!`。

**Auto-capture**: `fuse!(expr)` — テンソル変数を AST から自動検出。明示形式 `fuse!(expr; x, y)` も引き続き有効。スカラー変数が式中にある場合のみ明示形式を使用。

Element-wise methods: `.exp()` `.ln()` `.log1p()` `.powf(p)` `.sin()` `.cos()` `.tan()` `.tanh()` `.sqrt()` `.abs()` `.recip()` `.neg()` `.erf()` `.ceil()` `.floor()` `.round()` — all PyTorch-identical。Reduction: `.sum()` `.max()` `.min()` `.argmax()` `.argmin()`。

### 3.6 Linear algebra

CPU dense via `faer`。GPU: Recursive GEMM TRSM (`gpu_trsm_lower`, W15)。

| Math | nabla | NumPy/SciPy | Julia |
|---|---|---|---|
| $Ax = b$ | `a.solve(&b)?` | `np.linalg.solve(A,b)` | `A \ b` |
| $A^{-1}$ | `a.inv()?` | `np.linalg.inv(A)` | `inv(A)` |
| $PA = LU$ | `a.lu()?` | `scipy.linalg.lu(A)` | `lu(A)` |
| $A = QR$ / Chol / LDL | `a.qr()` / `.chol()?` / `.ldl()?` | `np.linalg.qr(A)` | `qr(A)` |
| $A = U\Sigma V^\top$ | `a.svd()?` / `a.svdvals()?` | `np.linalg.svd(A)` | `svd(A)` |
| $\lambda, V$ (sym) | `a.sym(Lower).eigh()?` | `np.linalg.eigh(A)` | `eigen(Sym(A))` |
| $\det(A)$ | `a.det()?` | `np.linalg.det(A)` | `det(A)` |
| $\log|\det(A)|$ | `a.logdet()?` | `np.linalg.slogdet(A)` | `logdet(A)` |
| $A = U\Sigma V^\top$ | `let (u,s,vt) = a.svd_into()?` | `U,S,Vh = svd(A)` | `U,S,Vt = svd(A)` |
| $A = QR$ | `let (q,r) = a.qr_into()` | `Q,R = qr(A)` | `Q,R = qr(A)` |
| $Ax = \lambda Bx$ (gen. eig) | `a.geig(&b)?` | `scipy.linalg.eig(A,B)` | `eigen(A, B)` |
| $\kappa_1(A)$ (cond number) | `a.cond1()?` | `np.linalg.cond(A,1)` | `cond(A, 1)` |
| $\kappa_\infty(A)$ | `a.cond_inf()?` | `np.linalg.cond(A,np.inf)` | `cond(A, Inf)` |
| $\kappa_p(A)$ | `a.cond_p(p)?` | `np.linalg.cond(A,p)` | `cond(A, p)` |
| $A^{-1}$ | `a.inv()?` | `np.linalg.inv(A)` | `inv(A)` |
| null(A) | `a.null_space(tol)?` | — | `nullspace(A)` |
| col space | `a.orth(tol)?` | — | `orth(A)` |
| sign, log\|det\| | `a.slogdet()?` | `np.linalg.slogdet(A)` | `logabsdet(A)` |

Structural: `Diagonal::new(v)`, `Symmetric::new(a, Side::Lower)?`, `Triangular::new(a, TriKind::Lower)?`。Factorization reuse: `lu.solve(&b)`, `lu.inverse()`, `lu.reconstruct()`。`?` = zero-cost `Result`（silent NaN 排除）。Constructors: `vandermonde_rect()` for rectangular Vandermonde matrices (W29).

**W30**: `LinalgExt` now supports `f32` tensors (f32→f64 internal promotion for all 45+ methods). Previously f64-only; f32 tensors required manual cast.

### 3.7 Sparse

```rust
// Julia: S = sparse(I, J, V, m, n)
// SciPy: S = scipy.sparse.csc_matrix((V, (I, J)), shape=(m, n))
let s = sparse(m, n, &[(0,0,1.0), (1,2,3.0)])?;  // free function
let x = s.solve(&b)?;
let x = s.chol_solve(&b)?;    // short name
let c = s * &d;                // SpMM via Mul trait
```

CPU: CSC format via faer. GPU: `BcsrMatrix<T>` BCSR format with WGSL SpMM kernel + `mixed_spmm_f64` mixed-precision refinement (W16).

### 3.8 Einstein summation

```rust
einsum!(c[i,j] = a[i,k] * b[k,j]);       // matmul
einsum!(c[b,i,j] = a[b,i,k] * m[b,k,j]); // batched matmul
einsum!(s = a[i,i]);                       // trace
einsum!(c[i,j] = a[i] * b[j]);            // outer product
```

7 patterns at compile time → auto-selects optimal codegen (GEMM, GEMV, Hadamard, trace, outer, batch GEMM, N-D fallback). GPU dispatch via `matmul_into`. AST-based parsing → タイポはコンパイル時 spanned error。NumPy `einsum` は文字列ベース（実行時エラー）。

### 3.9 Stencil

```rust
stencil!(out[i,j] = -4.0*u[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]); // ∇²u
```

Auto-detects offset bounds, zero boundary condition. CPU only（GPU は `fuse!` で代替）。

### 3.10 Calculus

**Reverse-mode AD** — PyTorch `.backward()` 互換:

```rust
{
    let tape = Tape::new();
    let x = tape.var(tensor_x);
    let w = tape.var(tensor_w);
    let loss = (&x * &w).exp().sum();
    loss.backward();
    let dx = x.grad();
}   // tape, grads, intermediates ALL freed by Drop. PyTorch: GC 待ち
```

**Forward-mode** — `impl Scalar for Dual<T>` で既存コード変更ゼロ:

```rust
let x = Dual::new(2.0, 1.0);   // x = 2, dx/dx = 1
let y = (x * x).sin();          // y = sin(4), dy/dx = 2cos(4)
```

**Symbolic CAS** (W27 method chain):

```rust
let x = var("x");                           // free function (prelude)
let f = (&x * &x).sin_();                   // method chain: .sin_(), .cos_(), .powf(n)
let df = diff(&f, "x");                     // differentiation
let df_simple = simplify(&df);              // e-graph simplification
let val: f64 = 2.0.into();                  // From<f64> for Expr
```

**`sym!` proc macro** — Julia-like symbolic expression syntax at compile time:

```rust
let f = sym!(sin(x^2) + cos(y));            // Pratt parser → Expr::* codegen
let g = sym!(exp(-x) * ln(y + 1.0));        // functions: sin/cos/exp/ln/tanh/sqrt/abs
```

`Expr::sin(&e)` (associated fn) と `e.sin_()` (method chain) の両形式を提供。`var()` free function で `Expr::var()` を簡略化。

**CAS extensions (W28)**: `substitute(expr, var, replacement)` for symbolic substitution. `gradient(expr, vars)`, `jacobian(exprs, vars)`, `hessian(expr, vars)` for multivariate calculus. Inverse trig/hyperbolic in ExprKind: Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh with full differentiation rules.

**CAS extensions (W29)**: `ExprKind::Tan` with full differentiation rules across Backend/Scalar/Tensor/GPU. `eval_tensor` now supports inverse trig/hyperbolic (asin/acos/atan/sinh/cosh/asinh/acosh/atanh). Owned `Expr` operators (Add/Sub/Mul/Div/Neg for all ownership combos). `(&Expr, f64)` mixed Sub/Div operators. `diff`/`simplify` added to prelude.

**ODE solvers**: `euler` (1), `rk4` (4), `dormand_prince` (5(4) adaptive), `bdf1` (implicit), `bdf2` (2nd-order implicit, W27)。全 config に `saveat: Option<Vec<f64>>` オプション (W27) — 指定時刻のみ解を記録（線形補間）。Preparation pattern: `grad_prep(f, &x)` で prep/execute 分離。AD + CAS + ODE が**単一 crate に統合**。

**W28**: `terminate` callback in `AdaptiveConfig` for early stopping. `OdeSolution::eval(t)` linear interpolation. `sol[i]` index access. Backward integration (reversed t_span). `parareal_solve_tensor` for Tensor state. `ensemble_euler_maruyama` for N-trajectory Monte Carlo. Builder methods: `.with_dt()`, `.with_tol()`, `.with_saveat()`.

**W29**: `stormer_verlet` returns `SymplecticSolution` (not raw tuple). `ensemble_euler_maruyama` parallel via `std::thread::scope`. SDE backward integration support (euler_maruyama + milstein). `Bdf2Config` struct (separate from Bdf1Config). `SdeConfig::with_noise_dims()` builder. `euler`/`rk4`/`dormand_prince` added to prelude.

**W30 CAS**: `gradient`/`jacobian`/`hessian` now auto-simplify internally via `diff_simplify`. CAS `eval` domain checks: div-by-zero, ln(x≤0), sqrt(x<0), asin/acos domain (|x|>1), acosh(x<1), atanh domain (|x|≥1) — returns `Err` instead of NaN. `eval`/`eval_tensor` added to prelude.

**W30 ODE**: `OdeProblem<T,B,F>` thin wrapper with `solve_euler`/`solve_rk4`/`solve_adaptive` methods. `EulerConfig`/`Rk4Config` structs with `saveat` field + `euler_with_config`/`rk4_with_config` solver functions.

### 3.11 Utilities

| Math | nabla | Python | Julia |
|---|---|---|---|
| $0 \le x < 1$ | `between!(0.0, x, 1.0)` | `0 <= x < 1` | `0 ≤ x < 1` |
| $0.0, 0.1, \ldots, 1.0$ | `arange(0.0, 1.0, 0.1)` | `np.arange(0,1,0.1)` | `0.0:0.1:1.0` |
| $x \mapsto f \mapsto g$ | `pipe!(x, f, g)` | — | `x \|> f \|> g` |
| $f(a, b, c)$ from tuple | `splat!(f, (a, b, c))` | `f(*args)` | `f(args...)` |
| Named struct | `named!(a: i32 = 1, b: f64 = 2.0)` | `dict(a=1, b=2.0)` | `(a=1, b=2.0)` |
| $0.0, 0.25, \ldots, 1.0$ | `range!(0.0, 0.25, 1.0)` | — | `0.0:0.25:1.0` |

### 3.12 Parallelism & GPU dispatch

| Strategy | nabla | GPU |
|---|---|---|
| Parallel construct | `par_from_fn(m, n, \|r,c\| expr)` | ❌ CPU |
| Parallel map | `a.par_map(\|x\| expr)` | ❌ CPU |
| GPU single op | `a.sin()` | ✅ |
| GPU fused chain | `fuse!(x.sin())` | ✅ **1 kernel** |
| GPU einsum | `einsum!(c[i,j] = a[i,k] * b[k,j])` | ✅ |

コードは 1 文字も変えずに CPU → GPU 切替: `--features cpu` / `--features cuda` / `--features hip` / `--features wgpu`。PyTorch `model.to('cuda')` のようなデバイス移動不要 — **コンパイル時に全て決定**。

### 3.13 Rust friction vs safety tradeoff

| Friction | 原因 | 例 | 対価 |
|---|---|---|---|
| `()` in indexing | `Index` trait = 単一引数 | `a[(i,j)]` vs `A[i,j]` | 型安全 + 寿命保証 |
| `&` for borrow | 所有権モデル | `&a * &b` vs `A * B` | 明示的ゼロコピー |
| `?` for fallibility | `Result<T>` 型 | `a.solve(&b)?` vs `A \ b` | Silent NaN/Inf 排除 |
| `.emul()` | `*` = matmul 占有 | `a.emul(b)` vs `A * B` | matmul と静的区別 |

全て**定数オーバーヘッド** — 式が複雑になるほど相対的影響は減少。4/4 が安全性の対価。

### 3.14 API naming conventions

| Category | Convention | Examples |
|---|---|---|
| Construction | Free function, type-inferred | `zeros`, `eye`, `rand`, `randn`, `ones`, `fill`, `linspace`, `arange` |
| Indexing | `Index` trait / `.slice()` method | `a[(i,j)]`, `a.slice(0..3, 1..4)`, `a.slice_rows(0..3)` |
| Unary op | Method, ≤ 5 chars | `.t()`, `.h()`, `.sin()`, `.exp()`, `.abs()`, `.neg()` |
| Binary op | Operator or short method | `a * b`, `a + b`, `a.emul(b)`, `a.ediv(b)` |
| Factorize | Short method + `?` | `.lu()?`, `.qr()`, `.chol()?`, `.svd()?`, `.ldl()?` |
| Solve | Verb + `?` | `.solve(&b)?`, `.lstsq(&b)?`, `.inv()?` |
| Reduce | Verb | `.sum()`, `.max()`, `.min()`, `.argmax()` |
| In-place | Method + `_` suffix (PyTorch convention) | `.mm_(&a, &b)`, `.add_(&b)` |
| AD | PyTorch-familiar | `tape.var(x)`, `loss.backward()`, `x.grad()` |
| Sparse | Free function | `sparse(m, n, &trips)?` |
| Module | `Module<T,B>` trait | `.forward()`, `.parameters()`, `.named_parameters()` |
| Optimizer | `Optimizer<T,B>` trait | `AdamW::new(lr, shapes).step(params, grads)` |
| Conv config | Builder struct + defaults | `Conv2dConfig::default().stride((2,2)).padding((1,1))` |
| Embedding | Free function | `embedding(&indices, &weight)` |
| Abstract trait | `MatrixLike<T>` | `.nrows()`, `.ncols()`, `.get(r,c)`, `.shape()` |
| View | Zero-copy borrow | `a.view_slice(0..3, 1..4)` → `TensorView<'_,T,B>` |
| Linear layer | Stateful module | `Linear::new(in, out).forward(&x)` |
| Iterator | Element-wise | `t.elements()`, `t.indexed_iter()`, `t.item()`, `t.to_vec()` |
| Scheduler | LR management | `LrScheduler::new(schedule, lr).step()` |

**W29 additions**: `Module::train()`/`eval()` shorthand methods. `state_dict()`/`load_state_dict()` on `Module` trait + `Linear` impl. `save_tensors`/`load_tensors` generic over `T: Scalar` (io.rs). `Optimizer::step_slices()` simplified signature. `AdamW::from_module()` constructor.

**W30 additions**: Autograd NN ops: `Variable::softmax(axis)`, `reshape(m,n)`, `transpose()`, `linear_forward(w,b)`, `dropout(p,training)`, `clamp(lo,hi)`, `mse_loss(target)`, `cross_entropy_indices(targets)`. Module/Autograd bridge: `Module::forward_var()` trait method, `Linear::forward_var` impl, `Tape::track_params()`, `Tape::var()` alias. `Variable` `Div` operator. `AdamW::from_params(&[&Tensor])` constructor. `GradScaler::scale_factor()` accessor. `backward()` NaN/Inf detection (returns `Err`). `clip_grad_norm` vectorized via `Tensor::norm()`. `zero_grad` in-place optimization.

**命名原則**: NumPy/PyTorch と同名なら同名。Julia が短ければ Julia 寄り。どちらでもないなら **最短の明確な名前**。(Julia/Python alignment → §1 Design principles)

---

## 4. Backend architecture

### 4.1 Exclusive backend selection

```rust
// Exactly one of: cpu, wgpu, cuda, hip
#[cfg(not(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "hip")))]
compile_error!("nabla: enable exactly one backend feature (cpu / wgpu / cuda / hip)");

// All 6 pairwise conflicts
#[cfg(all(feature = "cpu", feature = "wgpu"))]
compile_error!("nabla: cpu and wgpu are mutually exclusive");
// ... (cpu+cuda, cpu+hip, wgpu+cuda, wgpu+hip, cuda+hip)
```

| Feature | `DefaultBackend` | `Tensor<f32>` storage | `Tensor<f64>` |
|---|---|---|---|
| `cpu` (default) | `Cpu` | `CpuStorage<f32>` (row-major `Vec<T>`) | ✅ |
| `wgpu` | `Gpu` | `GpuStorage<f32>` (`wgpu::Buffer`) | ❌ compile error |
| `cuda` | `Cuda` | `GpuStorage<f32>` (CUDA `CUdeviceptr`) | ✅ |
| `hip` | `Hip` | `GpuStorage<f32>` (HIP `hipDeviceptr_t`) | ✅ |

All tensors use `Tensor<T>` = `Tensor<T, DefaultBackend>`.

### 4.2 Strict CPU/GPU separation

**Backend trait**: all computation methods are **required** (no default body). Each backend (`Cpu`, `Gpu`, `Cuda`, `Hip`) must provide its own implementation. There is no implicit CPU fallback — attempting to call GPU-only ops on a CPU tensor (or vice versa) is a compile error.

| Prohibited pattern | wgpu | cuda | hip |
|---|---|---|---|
| c32/c64 | ❌ compile error | ❌ compile error | ❌ compile error |
| f64 | ❌ compile error | ✅ native | ✅ native |
| `map!` (closure) | ❌ | ❌ | ❌ |
| `stencil!` (offset access) | ❌ | ❌ | ❌ |
| `par_*` (rayon) | ❌ | ❌ | ❌ |
| linalg/sparse | ❌ | ❌ | ❌ |

### 4.3 Macro GPU dispatch

| Macro | CPU | GPU (all backends) | Strategy |
|---|---|---|---|
| `einsum!` | ✅ GEMM | ✅ GPU kernel | `matmul_into` (Backend dispatch) |
| `fuse!` | ✅ single `from_fn` | ✅ JIT fused kernel | NVRTC/hiprtc codegen (L1), tensor chain (L2/L3) |
| `map!` | ✅ `from_fn` | ❌ compile error | arbitrary closure |
| `stencil!` | ✅ `from_fn` | ❌ compile error | offset access |
| `par_*` | ✅ rayon | ❌ compile error | CPU only |
| `mat!`/`splat!`/`named!`/`generated!` | ✅ | ✅ | Compile-time expansion |

### 4.4 Module visibility

| Module | cpu | wgpu | cuda | hip |
|---|---|---|---|---|
| `tensor` / `backend` | ✅ | ✅ | ✅ | ✅ |
| `wgpu/` | — | ✅ | — | — |
| `cuda_hip/` (cuda) | — | — | ✅ | — |
| `cuda_hip/` (hip) | — | — | — | ✅ |
| `linalg` / `sparse` | ✅ | ❌ | ❌ | ❌ |
| `cas` / `ode` | ✅ | ✅ | ✅ | ✅ |
| `autograd` | ✅ | ✅ | ✅ | ✅ |

---

## 5. GPU implementation

### 5.1 Architecture overview

```
                    ┌─────────────────────────────────┐
                    │      wgpu/storage.rs (dispatch)   │
                    │  GpuContext trait + GpuStorage<T> │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
      wgpu/shaders.rs  cuda_hip/cuda.rs  cuda_hip/hip.rs
        WGSL shaders     CUDA driver API   HIP runtime
        wgpu::Buffer     CUdeviceptr       hipDeviceptr_t
        pollster sync    cuLaunchKernel    hipLaunchKernelGGL
              │                │                 │
              ▼                ▼                 ▼
       wgpu/shaders.rs   cuda_hip/kernels.rs (shared)
        const &str WGSL       const &str CUDA/HIP C
```

### 5.2 GpuContext trait

```rust
pub(crate) trait GpuContext: Send + Sync + 'static {
    type Buffer: Send + Sync;
    fn alloc(&self, size_bytes: usize) -> Self::Buffer;
    fn upload(&self, buf: &Self::Buffer, data: &[u8]);
    fn download(&self, buf: &Self::Buffer, dst: &mut [u8]);
    fn launch(&self, kernel: &str, grid: [u32; 3], block: [u32; 3],
              args: &[&Self::Buffer], scalars: &[u32]);
}
```

Implementations: wgpu (`ComputePipeline` cache), CUDA (`CUmodule` + nvrtc), HIP (`hipModule_t` + hiprtc). Singleton via `OnceLock`.

### 5.3 GpuStorage

`GpuStorage<T>` wraps backend buffer + lazy `Mutex<Option<Vec<T>>>` host cache. Row-major flat array. Readback on first `.get()`/`.to_vec()` call. Chained GPU ops eliminate host↔device transfer.

### 5.4 Kernel sources — two codebases

**74 CUDA/HIP ops + 43 WGSL ops, same semantics, two shader languages:**

| Category | Ops | WGSL (wgpu) | CUDA/HIP C (shared) |
|---|---|---|---|
| Binary | add, sub, mul_elem, div_elem, scale | `elementwise_binary` | `elementwise_binary` (float4) |
| Unary | exp, ln, log1p, sin, cos, tanh, sqrt, abs, recip, erf, ceil, floor, round, powf, neg | `elementwise_unary` | `elementwise_unary` (float4 + fast math) |
| Activation | sigmoid, silu, mish, leaky_relu, elu, hardswish | ✅ WGSL kernel | ✅ float4 vectorized |
| Matmul | tiled matmul (shared memory) | `matmul_tiled` TILE=16 | `matmul_tiled` TILE=16/32 |
| Reduction | sum, max, min | `reduction` | `reduction` (warp shuffle) |
| Arg reduction | argmax, argmin | `reduction_arg` | `reduction_arg` (warp shuffle) |
| Row-wise | softmax, layer_norm, rms_norm | ✅ WGSL workgroup reduce | ✅ warp shuffle (3-pass) |
| Axis reduction | sum_axis1, max_axis1 | ✅ WGSL workgroup reduce | ✅ warp shuffle |
| Gather | embedding | ✅ WGSL thread-per-element | ✅ thread-per-element |
| Construction | zeros, fill, identity | `fill`, `identity` | `fill`, `identity` (float4) |
| Copy | clone, transpose | `copy`, `transpose` | `copy`, `transpose` |

WGSL shaders: embedded as `const &str` in `kernels_wgsl.rs`. Workgroup size: 256.

CUDA/HIP C kernels: embedded as `const &str` in `cuda_hip/kernels.rs`. Block size: 256 (tunable). Compiled at runtime via `nvrtc` / `hiprtc`. **Single source** — HIP C is source-compatible with CUDA C for standard math kernels.

**f32 vectorized memory access** — all f32 unary/binary/scalar kernels use `float4` (128-bit) loads/stores via `LDG.E.128`. Each thread processes 4 elements + scalar tail loop. f32 unary math uses CUDA fast-math intrinsics (`__expf`, `__logf` etc). f64 kernels remain scalar (1 element/thread).

### 5.5 Runtime compilation (CUDA/HIP)

cudarc 0.19 NVRTC API — PTX compiled at runtime, cached in `HashMap<String, KernelEntry>`. HIP path identical (`hiprtcCompileProgram`). Architecture auto-detected via `cuDeviceGetAttribute` (compute_70–90).

**Fusion JIT** — `fuse!` generates CUDA C expression strings at compile time. At runtime, `fuse_launch` builds a complete kernel, JIT compiles via NVRTC/hiprtc, caches by FNV-1a hash + type suffix. Subsequent calls use cached `CUfunction`.

### 5.6 Type support

| Type | cpu | wgpu | cuda | hip | Notes |
|---|---|---|---|---|---|
| `f32` | ✅ | ✅ 32 ops | ✅ 32 ops | ✅ 32 ops | Universal |
| `f64` | ✅ | ❌ compile error | ✅ 32 ops | ✅ 32 ops | WGSL/Metal lacks f64 |
| `c32`/`c64` | ✅ | ❌ compile error | ❌ compile error | ❌ compile error | No GPU complex |

### 5.7 Backend-specific advantages

| Feature | wgpu | cuda | hip |
|---|---|---|---|
| Platform | Vulkan/Metal/DX12 (cross-platform) | NVIDIA only | AMD only |
| f64 | ❌ | ✅ native | ✅ native |
| Tensor cores | ✅ register-tile software MMA (W18) | ✅ WMMA/MMA intrinsics (W15) | ✅ WMMA intrinsics (W15) |
| Warp/wave shuffle | ❌ (WGSL limitation) | ✅ `__shfl_down_sync` (W15) | ✅ `__shfl_*` (W15) |
| Shared memory | ✅ `var<workgroup>` | ✅ `__shared__` | ✅ `__shared__` |
| Max tile size | Limited by WGSL | 32+ (tunable) | 32+ (tunable) |
| Build dependency | `wgpu 24` (Rust crate) | None (runtime dlopen) | None (runtime dlopen) |

---

## 6. einsum! specification

### 6.1 Pattern classification

| Pattern | Math | Codegen |
|---|---|---|
| `c[i,j] = a[i,k] * b[k,j]` | C = AB | `matmul_into` (transpose-aware) |
| `y[i] = a[i,k] * x[k]` | y = Ax | `matmul_into` (Mx1) |
| `c[i,j] = a[i,j] * b[i,j]` | C = A ∘ B | `emul` (element-wise mul) |
| `s = a[i,i]` | tr(A) | diagonal loop |
| `c[i,j] = a[i] * b[j]` | C = ab^T | `from_fn` |
| `c[b,i,j] = a[b,i,k] * m[b,k,j]` | batch matmul | batch loop + inner GEMM |
| General N-D | — | NdTensor + loop codegen |

Compile-time `classify()` auto-detects → optimized path for GEMM/GEMV/Hadamard.

### 6.2 N-D index classification

| Class | Definition | Example |
|---|---|---|
| **Batch** | Present in LHS and all RHS, not contracted | `b` |
| **Free** | Present in LHS, subset of RHS | `i`, `j` |
| **Contraction** | Absent from LHS (RHS only) | `k` |

### 6.3 Spanned errors

`syn::Error::new_spanned` points to the exact token span in the user's `einsum!` expression.

| Error | Span target |
|---|---|
| Unknown index | Ident span of the index |
| Lone contraction index | Ident span + help message |
| Duplicate LHS index | Second occurrence span |
| Duplicate RHS index (non-trace) | Ident span |

---

## 7. Autograd

### 7.1 Tape-based reverse-mode (PyTorch-familiar)

```
tape.var(tensor) → record ops on Tape (Rc<RefCell>) → backward() → gradient tensors
```

14 ops: add, sub, mul, div, matmul, neg, exp, ln, sin, cos, tanh, sqrt, sum, transpose.

| nabla | PyTorch | Memory difference |
|---|---|---|
| `tape.var(x)` | `x.requires_grad_(True)` | nabla: explicit tape ownership. PyTorch: global autograd engine |
| `loss.backward()` | `loss.backward()` | nabla: tape freed by `Drop`. PyTorch: grad_fn chain → GC |
| `x.grad()` → owned `Tensor` | `x.grad` → shared ref | nabla: owned, freed when dropped. PyTorch: accumulates |
| Scope exit = cleanup | `optimizer.zero_grad()` | **nabla: automatic**. PyTorch: manual |

Trade-offs:
- ✅ Simple, correct, composable, **PyTorch-familiar API** (`tape.var()`, `.backward()`, `.grad()`)
- ✅ Tape + grads deterministically freed at scope exit (no GC, no memory leak)
- ❌ Rc overhead per op, dynamic graph allocation, single-thread only (`Rc` not `Send`)

---

## 8. ODE solvers

| Solver | Order | Adaptive | Use case |
|---|---|---|---|
| `euler` | 1 | No | Teaching, fast rough estimate |
| `rk4` | 4 | No | Standard fixed-step |
| `dormand_prince` | 5(4) | Yes | Production adaptive-step |

---

## 9. CAS

`Expr::simplify()` applies rewrite rules destructively in a single pass. Supports `diff`/`simplify`/`eval`/`eval_tensor`.

Current limitation: rule application order determines the result (phase-ordering problem).

---

## 10. Sparse

`SparseMatrix<T>` — Compressed Sparse Column (CSC) format. CPU-only factorization and solve.

---

# Part II — Design Decisions & Constraints

## 11. Design decisions

| Decision | Rationale |
|---|---|
| Direct kernels (no CubeCL) | Fixed-rule 32 ops → 2 codebases manageable |
| Build-time exclusive backend | CPU fallback is a performance bug source |
| Runtime kernel compilation | nvrtc/hiprtc: no SDK at build time |
| Handle-based GPU storage | Chained ops eliminate host↔device transfer |
| TypeId dispatch | Backend trait sealed + `T: Scalar`, avoids E0276 |
| Embedded kernel strings | WGSL + CUDA/HIP C as `const &str` |
| Native Rust (no C++ wrapper) | Ownership-native tensor > FFI wrapper [2505.12425] |
| Recursive GEMM for GPU linalg | Reuse matmul_tiled [2504.13821] |
| Einsum canonicalization | 4.7x over JAX [2601.12220] |
| Named axes (W11) | Compile-time dimension safety [Chiang+ 2021] |
| `impl Scalar for Dual<T>` | Forward-mode AD as zero-change drop-in [2504.15976] |
| Macro DSL absorbs verbosity | Julia 比 5-10x LOC gap を macro で圧縮 |

---

## 12. Known limitations

| Limitation | Mitigation |
|---|---|
| No wgpu f64 | Use `cuda`/`hip` backend |
| No GPU c32/c64 | Compile error (by design) |
| GPU linalg: TRSM only | `gpu_trsm_lower` (W15); full LU/Cholesky/QR CPU only |
| `from_fn` requires host | Use `fuse!` for GPU |
| L2/L3 fuse! on GPU | GPU fused kernels require codegen extension |
| 2 kernel codebases | WGSL ≠ CUDA/HIP C — fixed 32 ops, rarely changes |
| No REPL | `rust-script` + `cargo watch` で即時フィードバック |

---

# Part III — Implementation Notes

## 13. Subsystem implementation

### 13.1 GPU

| Item | Status | Backend | Approach |
|---|---|---|---|
| WMMA/MMA tensor cores | ✅ W15 | cuda/hip | `nvcuda::wmma::mma_sync` (Volta+), `rocwmma` (CDNA2+) |
| Warp shuffle reductions | ✅ W15 | cuda/hip | `__shfl_down_sync`, 8x reduction |
| Subgroup matrix MMA (wgpu) | ✅ W18 | wgpu | Register-tile software MMA |
| Recursive GEMM TRSM | ✅ W15 | cuda/hip | `gpu_trsm_lower`: base n≤32 → CPU, recursive quadrant |
| Linear Layouts F₂ swizzle | ✅ W17 | all | `LinearLayout<N>`: bank-conflict-free for any tile size |
| Deep GEMM+activation fusion | ✅ W17 | all | `detect_gemm_activation`, 10 activations, `matmul_fused` |
| bf16/f16 Scalar | ✅ W7 | all | `f16`/`bf16` types + backend-specific enable |
| f32 float4 vectorization | ✅ W19 | cuda/hip | 128-bit `float4` loads/stores + fast math intrinsics |
| GPU kernel fusion (L1) | ✅ W19 | cuda/hip | `fuse!` → `cuda_expr()` → NVRTC/hiprtc JIT → cache |
| Caching memory allocator | ✅ W20 | cuda/hip | Best-fit dual-pool, 512B-aligned, block splitting, GC 0.9 |
| Vectorized fuse codegen | ✅ W20 | cuda/hip | float4 + `__ldg` prefetch in `fuse_kernel_source()` |
| Async execution pipeline | ✅ W20 | cuda/hip | Defer sync until readback only |
| Single-element D2H readback | ✅ W20 | cuda/hip | `copy_element()` — 4-byte D2H |
| CUDA Graph capture/replay | ✅ W20 | cuda | `TrainingGraph` API, 1.66× speedup (36-launch training step) |
| cuBLAS workspace pre-alloc | ✅ W20 | cuda | `cublasSetWorkspace_v2` 32MiB — prevents internal `cudaMallocAsync` |
| Phase 3 GPU kernels (conv/pool/attn/bmm/scan/norm/loss) | ✅ | cuda/hip | im2col+cuBLAS, FlashAttn-2, Blelloch scan, 2-pass BN, fused cross_entropy |
| Mega-kernel tiled fusion (L4) | ✅ W20 | cuda/hip | Shared memory tile reuse for multi-op mega_fuse! (≥2 ops, ≥64K elements) |

**Kernel fusion levels:**

| Level | Scope | Status |
|---|---|---|
| L1: Element-wise | Fuse consecutive unary/binary ops → single JIT kernel | ✅ W19 GPU |
| L2: Reduction | Fuse across reduction ops with loop-carried deps | ✅ W13 CPU |
| L3: GEMM+pointwise | Fuse matmul + activation (cublasLt epilogue for relu/gelu) | ✅ W17/W20 |
| L4: Map-reduce | Fuse pointwise chain + axis reduction → single kernel | ✅ W20 |
| L4: Mega-kernel tiled | Shared memory tile reuse for multi-op mega_fuse! | ✅ W20 |
| L4: DAG fusion | `prev` keyword for inter-op register pass-through | ✅ W20 |

Fusion pipeline: `fuse!` AST → egg EqSat simplify → `cuda_expr()` → NVRTC/hiprtc JIT → FNV-1a hash cache.

**Benchmark results** (GH200 480GB, 4096×4096 f32, PyTorch 2.7.0):

| Workload | nabla | PyTorch | Gap |
|---|---|---|---|
| exp / sin / cos / tanh | 0.040 ms | 0.040–0.041 ms | **≈ parity** |
| add / emul | 0.058 ms | 0.058 ms | **≈ parity** |
| fuse exp+sin (L1 JIT) | 0.041 ms | 0.081 ms (eager) | **nabla 2× faster** |
| fuse 4-op | 0.050 ms | — | single kernel |
| mega_fuse 4-out | 0.141 ms vs 0.162 ms unfused | — | 1.15× (shared kernel) |
| sum_all / max_all | 0.028 ms | 0.026 ms | PyTorch 1.08× |
| matmul 4096 (cuBLAS TF32) | 0.378 ms | 2.68 ms | **nabla 7× faster** |
| matmul 2048 | 0.069 ms | — | Fixed (was CRASH) |
| CUDA Graph (36-op step) | 56μs/step | — | 1.67× vs eager |

Pre-optimization (W19): exp 2.38ms → **46× gap reduced to ≈ parity** via: async execution, single-element readback, float4 vectorization, caching allocator, fusion cost model. See §14.2 for full optimization history.

---

### 13.2 Autograd

| Mode | Implementation | Status |
|---|---|---|
| Reverse (tape) | `GpuTape<T>`: 12-op enum, buffer registry, backward via GPU kernels | ✅ W15 |
| Forward (dual) | `Dual<T>`: `impl Scalar for Dual<T>` — all tensor ops unchanged | ✅ W18 |
| Source-transform | `#[nabla_grad]` proc macro → `f_grad(x) -> (T, T)` | ✅ W18 |

---

### 13.3 ODE

| Solver | Status | Approach |
|---|---|---|
| DAE (bdf1) | ✅ W7 | Semi-explicit index-1 DAE |
| Parareal | ✅ W18 | Rayon fine propagator, sequential coarse correction |
| Symplectic | ✅ W8 | Störmer-Verlet for Hamiltonian systems |
| IF Euler | ✅ W6 | Exponential integrator: succeeds where RK4 diverges |
| METD | ✅ W8 | Matrix exponential for Lyapunov/Riccati/graph ODE |
| BDF2 (2nd order) | ✅ W27 | BDF1 bootstrap + BDF2 implicit, Newton iteration |
| Euler-Maruyama | ✅ W27 | SDE solver, strong order 0.5, inline Xorshift64 + Box-Muller PRNG |
| Milstein | ✅ W27 | SDE solver, strong order 1.0, noise derivative correction term |
| Ensemble SDE | ✅ W28 | N-trajectory Monte Carlo via ensemble_euler_maruyama |
| Ensemble parallel | ✅ W29 | ensemble_euler_maruyama parallel via std::thread::scope |
| SDE backward | ✅ W29 | euler_maruyama + milstein backward integration support |

---

### 13.4 CAS

All planned: `egg` 32 rules + `CDiff` + `diff_simplify` (41 rules) + `FuseExpr` 16-node EqSat in `fuse!` (15 rules) + multi-var diff (6 rules). No remaining items. W27: Method chain API (`.sin_()`, `.powf(n)`), `From<f64>` implicit conversion, `var()` free function.

---

### 13.5 Sparse

GPU: `BcsrMatrix<T>` BCSR + `WGSL_BCSR_SPMM` kernel (W16)。`mixed_spmm_f64` mixed-precision refinement。CPU: CSC via faer。

---

### 13.6 einsum!

7 patterns: GEMM, GEMV, Hadamard, trace, outer, batch GEMM, N-D fallback。`classify()` + `canonicalize()` で等価式のチューニング再利用。L1 tiled contraction (tile=64, W13)。GPU: `matmul_into` backend dispatch。

---

### 13.7 Notation & DX

Named axes (`Tensor<T,B,Axes=()>` W11): compile error on axis mismatch。StaticMatrix const-generic shape algebra (W12)。Tensor manipulation: reshape/view/permute/cat/stack/squeeze/unsqueeze/flatten/chunk — PyTorch API 名採用。Axis reduction: `.sum_axis(d)` / `.mean_axis(d)` + keepdim variants。

W27 ML abstractions: `Module<T,B>` trait (forward/parameters/named_parameters/parameters_mut), `Optimizer<T,B>` trait + `AdamW` struct, `Conv1dConfig`/`Conv2dConfig`/`Conv3dConfig` builder pattern, `embedding()` free function. Auto broadcasting for `&Tensor + &Tensor` (row/col/scalar). `range!` macro. `view()` copy warning. `sym!` proc macro (Pratt parser → `Expr::*` codegen, supports +/-/*/^/unary/functions). `MatrixLike<T>` trait (read-only common interface for Tensor/StaticMatrix/TensorView). `TensorView<'a,T,B>` zero-copy borrowed slice. SDE solvers: `euler_maruyama` + `milstein`.

**N-D tensor policy (W27 決定)**: `NdTensor<T>` は CPU-only を正式方針とする。GPU 計算の単位は 2D (cuBLAS GEMM, FlashAttention, im*col+GEMM) であり、N-D 次元は shape/indexing の抽象に過ぎない。`Tensor<T,B>` の 2D 設計は GPU ハードウェアと一致しており、統合のメリットがない。N-D → 2D 変換は `slice_2d` / `into_2d` で対応。

W28 Round 2 improvements: Inverse trig/hyperbolic (11 new element-wise ops on Backend+Scalar+CAS), `inv()`/`null_space()`/`orth()`/`slogdet()` linalg, `cond_p(p)` unification, `sum()`/`max()`/`min()` aliases, `epow` tensor power, `linear()` layer + `Linear` struct, CAS `substitute`/`gradient`/`jacobian`/`hessian`, ODE event detection + backward integration + `OdeSolution::eval(t)` + `parareal_solve_tensor` + ensemble SDE, autograd NN ops (relu/sigmoid/gelu/sum_axis/mean/cross_entropy on Variable), `Module` train/eval mode + `forward_with` multi-input + inspection methods, `AdamW` vectorized + `SGD` + `set_lr()` + `LrScheduler`, `elements()`/`indexed_iter()`/`item()`/`to_vec()` iterators, owned `Add/Sub`, sparse transpose/speye/addition, `discrete_lyapunov`/`discrete_sylvester`, `care` alias, `logspace`/`geomspace`, `topk` axis=0, `argsort_by`.

W29 Round 3 improvements: **Math**: `tan()` element-wise across all layers (Backend/Scalar/Tensor/GPU), commutative scalar*Tensor ops (`f32/f64 * &Tensor`, + and -), owned matmul (all ownership combos), `from_vec(data, nrows, ncols)`, `mean()`/`prod()` aliases. **Linalg**: `vandermonde_rect()`. **CAS**: `ExprKind::Tan`, `eval_tensor` inverse trig/hyperbolic fix, owned `Expr` operators, `(&Expr, f64)` Sub/Div, `diff`/`simplify` in prelude. **ODE/SDE**: `stormer_verlet` → `SymplecticSolution`, `ensemble_euler_maruyama` parallel (`std::thread::scope`), SDE backward integration, `Bdf2Config` struct, `SdeConfig::with_noise_dims()`, `euler`/`rk4`/`dormand_prince` in prelude. **Autograd**: `Variable::ediv()`/`epow()`/`abs()`/`log1p()`/`silu()`. **Module/Optimizer**: `Module::train()`/`eval()` shorthand, `state_dict`/`load_state_dict`, io generic scalar, `Optimizer::step_slices()` simplified, `AdamW::from_module()`. **Bug fix**: `reduce_axis` double-count first element (loop start 0→1).

W30 Round 4 improvements (25 items — 2 critical + 10 important + 13 nice-to-have): **Autograd NN ops**: `Variable::softmax(axis)`, `reshape(m,n)`, `transpose()`, `linear_forward(w,b)`, `dropout(p,training)`, `clamp(lo,hi)`, `mse_loss(target)`, `cross_entropy_indices(targets)` — full differentiable NN forward pass in autograd. **Module/Autograd bridge**: `Module::forward_var()` trait method + `Linear::forward_var` impl, `Tape::track_params()`, `Tape::var()` alias. **Math**: `Div<T>` operator (`&a/scalar`, `scalar/&tensor`), Variable `Div`, `prod_axis(axis)`, `var_axis_ddof(axis, ddof)`, `zeros_vec`/`ones_vec`/`rand_vec`/`randn_vec` column constructors. **Linalg**: `LinalgExt` f32 support (f32→f64 promotion, 45+ methods). **CAS**: `gradient`/`jacobian`/`hessian` auto-simplify via `diff_simplify`, `eval` domain checks (div-by-zero, ln≤0, sqrt<0, asin/acos/acosh/atanh), `eval`/`eval_tensor` prelude. **ODE**: `OdeProblem<T,B,F>` thin wrapper, `EulerConfig`/`Rk4Config` with `saveat`. **Training utils**: `GradScaler::scale_factor()`, `AdamW::from_params(&[&Tensor])`, `backward()` NaN/Inf detection (`Err`), vectorized `clip_grad_norm`, in-place `zero_grad`.

---

## 14. Roadmap

### 14.1 PyTorch computational parity — 計算プリミティブの完全網羅

nabla は計算エンジンとして、ユーザーが「この計算がない」と思う場面をゼロにする。PyTorch の `torch.*` / `torch.nn.functional.*` が提供する数学的に固定された計算を、すべてピュア Rust + GPU カーネルで実装する。

**目的**: 汎用性の極限。ユーザーは任意のアーキテクチャを nabla のプリミティブの組み合わせだけで構築できる。エッジケースでも、プリミティブの組み合わせか最小限の拡張で対応可能。

**現状のカバレッジ**: ✅ 190+ ops (element-wise, matmul, reduction, activations, loss, normalization, conv, pooling, attention, manipulation, construction, regularization)
**GPU kernels**: 74 CUDA/HIP + 11 WGSL (activations, softmax, normalization, axis reductions, embedding)
**目標**: CNN / Transformer / GAN / Diffusion — あらゆるアーキテクチャに必要な計算プリミティブを網羅

#### A. Convolution（畳み込み）— ✅ CPU実装済

| Op | 数式 | GPU kernel | AD backward | 優先度 |
|---|---|---|---|---|
| `conv1d(x, w, bias, stride, padding, dilation, groups)` | $(x * w)[n,c_o,l] = \sum_{c_i,k} x[n,c_i,l \cdot s+k \cdot d] \cdot w[c_o,c_i,k]$ | ✅ GPU (im1col+GEMM) | ✅ 必要 | ✅ |
| `conv2d(x, w, bias, stride, padding, dilation, groups)` | $(x * w)[n,c_o,h,w] = \sum_{c_i,kh,kw} x \cdot w$ | ✅ GPU (im2col+GEMM, cuBLAS strided-batched) | ✅ 必要 | ✅ |
| `conv_transpose2d(x, w, ...)` | Fractionally-strided convolution | ✅ GPU (1 thread/output) | ✅ 必要 | ✅ |
| `conv3d` | 3D convolution | ✅ GPU (im3col+GEMM) | ✅ 必要 | ✅ |

#### B. Pooling — ✅ CPU実装済

| Op | 数式 | GPU kernel | AD backward |
|---|---|---|---|
| `max_pool2d(x, kernel_size, stride, padding)` | $y[n,c,h,w] = \max_{kh,kw} x[n,c,h \cdot s+kh, w \cdot s+kw]$ | ✅ GPU (1 thread/output; with_indices variant) | argmax indices for backward |
| `avg_pool2d(x, kernel_size, stride, padding)` | $y = \frac{1}{k^2} \sum x$ | ✅ GPU (1 thread/output) | Uniform gradient distribution |
| `adaptive_avg_pool2d(x, output_size)` | Auto-stride pooling | ✅ GPU (1 thread/output) | Same as avg_pool |
| `max_pool1d` / `avg_pool1d` | 1D variants | 🔲 GPU (future) | — |

#### C. Normalization — ✅ CPU実装済

| Op | 数式 | 現状 | AD backward |
|---|---|---|---|
| `layer_norm(x, shape, weight, bias, eps)` | $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | ✅ CPU | ✅ GPU kernel |
| `rms_norm(x, weight, eps)` | $\frac{x}{\text{RMS}(x)} \cdot \gamma$ | ✅ CPU | ✅ GPU kernel |
| `batch_norm(x, mean, var, weight, bias, training, momentum, eps)` | Running mean/var + affine | ✅ CPU | ✅ GPU kernel (2-pass: stats+normalize, in-place running stats) |
| `group_norm(x, num_groups, weight, bias, eps)` | Group-wise layer norm | ✅ CPU | ✅ GPU kernel |

#### D. Activation functions — ✅ CPU実装済

| Op | 数式 | 現状 | fusable |
|---|---|---|---|
| `relu(x)` | $\max(0, x)$ | ✅ | ✅ `fuse!` |
| `gelu(x)` | $x \cdot \Phi(x)$ | ✅ | ✅ `fuse!` |
| `sigmoid(x)` | $\frac{1}{1+e^{-x}}$ | ✅ | ✅ `fuse!` |
| `softmax(x, dim)` | $\frac{e^{x_i}}{\sum e^{x_j}}$ | ✅ CPU | ✅ GPU kernel |
| `log_softmax(x, dim)` | $x_i - \log \sum e^{x_j}$ | ✅ CPU | ✅ GPU kernel |
| `silu(x)` / swish | $x \cdot \sigma(x)$ | ✅ CPU+GPU kernel | ✅ `fuse!` 可 |
| `mish(x)` | $x \cdot \tanh(\text{softplus}(x))$ | ✅ CPU+GPU kernel | ✅ `fuse!` 可 |
| `leaky_relu(x, α)` | $\max(\alpha x, x)$ | ✅ CPU+GPU kernel | ✅ `fuse!` 可 |
| `elu(x, α)` | $\begin{cases} x & x>0 \\ \alpha(e^x-1) & x \le 0 \end{cases}$ | ✅ CPU+GPU kernel | ✅ `fuse!` 可 |
| `hardswish(x)` | $x \cdot \frac{\text{ReLU6}(x+3)}{6}$ | ✅ CPU+GPU kernel | ✅ `fuse!` 可 |

#### E. Loss functions — ✅ CPU実装済

| Op | 数式 | 現状 | AD backward |
|---|---|---|---|
| `cross_entropy_loss(logits, targets)` | $-\sum y_i \log \text{softmax}(x)_i$ | ✅ CPU | ✅ GPU fused (online softmax + nll in single pass) |
| `mse_loss(pred, target)` | $\frac{1}{n}\sum(y - \hat{y})^2$ | ✅ CPU | ✅ trivial |
| `l1_loss(pred, target)` | $\frac{1}{n}\sum|y - \hat{y}|$ | ✅ CPU | ✅ trivial |
| `smooth_l1_loss(pred, target, beta)` | Huber loss | ✅ CPU | ✅ |
| `binary_cross_entropy_with_logits` | $-[y \log \sigma(x) + (1-y) \log(1-\sigma(x))]$ | ✅ CPU | ✅ |
| `nll_loss(log_probs, targets)` | $-\log p_{y_i}$ | ✅ CPU | ✅ |
| `kl_div(log_p, q)` | $\sum q (\log q - \log p)$ | ✅ CPU | ✅ |
| `cosine_embedding_loss` | $1 - \cos(x_1, x_2)$ | ✅ CPU | ✅ |

#### F. Attention / Transformer primitives — ✅ CPU実装済

| Op | 数式 | GPU kernel | 優先度 |
|---|---|---|---|
| `scaled_dot_product_attention(Q, K, V, mask, dropout_p)` | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$ | ✅ FlashAttention-2 (BLOCK_M=BLOCK_N=64, O(seq_len) HBM) | ✅ |
| `multi_head_attention(Q, K, V, num_heads)` | Reshape → SDPA → concat | ✅ GPU via sdpa dispatch | ✅ |
| `embedding(indices, weight)` | $y_i = W[\text{idx}_i]$ | ✅ GPU kernel | ✅ |

#### G. Tensor manipulation — ✅ CPU実装済

| Op | 現状 | 優先度 | 用途 |
|---|---|---|---|
| `reshape` / `view` | ✅ | — | — |
| `permute` / `transpose` | ✅ | — | — |
| `cat` / `stack` | ✅ | — | — |
| `squeeze` / `unsqueeze` | ✅ | — | — |
| `flatten` / `unflatten` | ✅ / ✅ | — | reshape-based unflatten |
| `chunk` / `split` | ✅ / ✅ | — | Arbitrary split sizes |
| `repeat` / `expand` | ✅ | — | Broadcasting without copy |
| `pad(x, pad, mode, value)` | ✅ | — | Conv padding, sequence padding |
| `gather(x, dim, index)` | ✅ | — | General dim gather |
| `scatter(x, dim, index, src)` | ✅ | — | Embedding backward, sparse update |
| `index_select(x, dim, index)` | ✅ | — | Batch indexing |
| `masked_fill(x, mask, value)` | ✅ | — | Attention mask |
| `where_(cond, x, y)` | ✅ | — | Conditional select |
| `triu` / `tril` | ✅ | — | Causal mask generation |
| `roll` / `flip` | ✅ | — | Shift equivariance |
| `meshgrid` | ✅ | — | Positional encoding |
| `arange` / `linspace` | ✅ | — | Index generation |
| `topk(x, k, dim)` | ✅ | — | Top-k sampling |
| `sort(x, dim)` | ✅ | — | Ranking |

#### H. Batched operations — ✅ CPU実装済

| Op | 数式 | GPU kernel | 用途 |
|---|---|---|---|
| `bmm(A, B)` | Batched matmul: $C_b = A_b B_b$ | ✅ cuBLAS `cublasSgemmStridedBatched` (f32/f64) | Attention, batched linear |
| `baddbmm(C, A, B, β, α)` | $C = \beta C + \alpha A B$ | ✅ cuBLAS fused | Efficient attention |
| `addmm(C, A, B, β, α)` | $C = \beta C + \alpha A B$ | ✅ cuBLAS fused | Linear layer |
| Batched reductions | `sum/max/min` along batch dim | Existing kernels + stride | DataParallel |

#### I. Construction / utility — ✅ CPU実装済

| Op | 現状 | 用途 |
|---|---|---|
| `zeros` / `ones` / `full` | ✅ | — |
| `zeros_like` / `ones_like` / `full_like` | ✅ | — |
| `eye` / `identity` | ✅ | — |
| `arange(start, end, step)` | ✅ | Index tensors, positional encoding |
| `linspace(start, end, steps)` | ✅ | Uniform sampling |
| `rand` / `randn` | ✅ CPU (xorshift64 + Box-Muller) | Weight init, dropout, stochastic |
| `from_numpy` / `to_numpy` | N/A | Rust has no NumPy; use `from_slice` / `to_vec` |
| `empty` (uninitialized) | ✅ | Performance (skip zeroing) |
| `contiguous` | ✅ | Force contiguous layout after permute |
| `clone` / `detach` | ✅ / ✅ | Clone / AD graph detachment |

#### J. Reduction extensions — ✅ CPU実装済

| Op | 数式 | 現状 | GPU |
|---|---|---|---|
| `sum_all` / `max_all` / `min_all` | ✅ | ✅ | ✅ (quad-ILP, mapped host) |
| `sum_axis(d)` / `mean_axis(d)` | ✅ | ✅ CPU | ✅ GPU kernel |
| `var_axis(d)` / `std_axis(d)` | ✅ | ✅ CPU | ✅ GPU kernel |
| `max_axis(d)` / `min_axis(d)` | ✅ | ✅ CPU | ✅ GPU kernel |
| `argmax_axis(d)` / `argmin_axis(d)` | ✅ | ✅ CPU | ✅ GPU kernel |
| `cumsum(x, dim)` | ✅ CPU | — | ✅ GPU (Blelloch scan, O(n) work O(log n) depth) |
| `cumprod(x, dim)` | ✅ CPU | — | ✅ GPU (Blelloch scan) |
| `prod_all` | ✅ CPU | — | ✅ GPU (2-phase shared-memory tree reduction) |
| `norm(x, p, dim)` | ✅ (L2/Linf + Lp axis) | — | ✅ GPU (GPU abs+powf+sum chain; L∞ via max) |
| `count_nonzero` | ✅ CPU | — | ✅ GPU (cast-to-int sum reduction) |

---

#### K. Regularization / Utility — ✅ CPU実装済

| Op | 説明 | 現状 |
|---|---|---|
| `dropout(x, p, training, seed)` | 確率 p で要素をゼロ化、残りを 1/(1-p) でスケール | ✅ CPU |
| `interpolate_nearest(x, h, w, out_h, out_w)` | 最近傍補間（upsample/downsample） | ✅ CPU |
| `interpolate_bilinear(x, h, w, out_h, out_w)` | バイリニア補間（align_corners=false） | ✅ CPU |

---

**Phase 1-2 (CPU完了 ✅):** 190+ ops — conv(1d/2d/3d/transpose), SDPA, bmm/addmm/baddbmm, embedding, softmax/log_softmax, layer/rms/batch/group_norm, pad/repeat/expand, gather/scatter/index_select/masked_fill, arange/linspace, 8 loss functions, 6 activations (silu/mish/leaky_relu/elu/hardswish/sigmoid), multi_head_attention, where/triu/tril, topk/sort, roll/flip/meshgrid, cumsum/cumprod, rand/randn, dropout, interpolate(nearest/bilinear). Phase 3 GPU kernels: 全実装済 ✅ (W22: conv1d/2d/3d/transpose, pool, FlashAttn-2, bmm/addmm/baddbmm, Blelloch scan, batch_norm, cross_entropy).

**Phase 3 (GPU kernels — 既存CPU ops のGPU高速化 ✅ 完了):**

GPU kernels for conv1d/2d/3d/transpose (im*col+cuBLAS StridedBatched), FlashAttention-2 (BLOCK_M=BLOCK_N=64, O(seq_len) HBM), batched GEMM (cuBLAS StridedBatched), max/avg/adaptive_avg pooling (+with_indices), Blelloch parallel prefix scan (cumsum/cumprod), 2-phase batch_norm, fused cross_entropy.

**Phase 3 実装済 GPU kernels:**

| Kernel | Type | CUDA/HIP | wgpu (WGSL) | Status |
|---|---|---|---|---|
| `k_sigmoid_f32/f64` | Activation | float4 vectorized | — (via unary op) | ✅ |
| `k_silu_f32/f64` | Activation | float4 vectorized | ✅ dedicated shader | ✅ |
| `k_mish_f32/f64` | Activation | float4 vectorized | ✅ dedicated shader | ✅ |
| `k_leaky_relu_f32/f64` | Activation | float4 vectorized | ✅ parameterized shader | ✅ |
| `k_elu_f32/f64` | Activation | float4 vectorized | ✅ parameterized shader | ✅ |
| `k_hardswish_f32/f64` | Activation | float4 vectorized | ✅ dedicated shader | ✅ |
| `k_softmax_f32/f64` | Row-wise softmax | 3-pass warp shuffle | ✅ workgroup reduce | ✅ |
| `k_layer_norm_f32/f64` | Fused normalization | fused mean+var+norm | ✅ workgroup reduce | ✅ |
| `k_rms_norm_f32/f64` | Fused normalization | fused RMS+normalize | ✅ workgroup reduce | ✅ |
| `k_sum_axis1_f32/f64` | Axis reduction | one block/row warp shuffle | ✅ workgroup reduce | ✅ |
| `k_max_axis1_f32/f64` | Axis reduction | one block/row warp shuffle | ✅ workgroup reduce | ✅ |
| `k_embedding_f32/f64` | Gather | thread-per-element | ✅ thread-per-element | ✅ |
| `k_max_pool2d_f32/f64` | Max pooling | 1 thread/output, argmax indices | — | ✅ |
| `k_avg_pool2d_f32/f64` | Avg pooling | 1 thread/output | — | ✅ |
| `k_adaptive_avg_pool2d_f32/f64` | Adaptive avg pooling | auto-stride | — | ✅ |
| `k_im2col_f32/f64` | im2col for conv2d | scalar loads; GEMM via cuBLAS | — | ✅ |
| `k_im1col_f32/f64` | im1col for conv1d | scalar loads; GEMM via cuBLAS | — | ✅ |
| `k_im3col_f32/f64` | im3col for conv3d | scalar loads; GEMM via cuBLAS | — | ✅ |
| `k_conv_transpose2d_f32/f64` | Transposed conv | 1 thread/output element | — | ✅ |
| `k_batch_norm_stats_f32/f64` | Batch norm pass 1 | per-feature mean+var | — | ✅ |
| `k_batch_norm_fwd_f32/f64` | Batch norm pass 2 | normalize+affine, running stats | — | ✅ |
| `k_cross_entropy_f32/f64` | Cross-entropy fused | online softmax + nll in 1 pass | — | ✅ |
| `k_cumsum_cumprod_f32/f64` | Prefix scan | Blelloch up/down sweep, smem=2×BLOCK | — | ✅ |
| `k_prod_partial_f32/f64` | Product reduction | 2-phase shared-memory tree | — | ✅ |
| `k_max_pool2d_with_idx_f32/f64` | Max pool + indices | values + flat argmax stored | — | ✅ |
| `k_sdpa_f32/f64` | FlashAttention-2 | online softmax, O(seq) HBM, BLOCK_M=BLOCK_N=64 | — | ✅ |

**Backend coverage:**

| Backend | Kernel count | Approach |
|---|---|---|
| CUDA (nvrtc) | 100+ kernels (f32+f64) | float4 vectorized + warp shuffle + WMMA + Phase 3 kernels |
| HIP (hiprtc) | 100+ kernels (f32+f64) | Source-compatible with CUDA C |
| wgpu (WGSL) | 43 kernels (f32 only) | Workgroup-level reduction, register-tile MMA (Phase 3 uses CPU defaults) |

### 14.2 Performance — W20 全項完了

#### 完了済み最適化

| ID | Item | 概要 |
|---|---|---|
| F-1 | dyn_kernels RwLock 化 | `Mutex<HashMap>` → `RwLock`: cache hit で shared read lock |
| F-2 | cublasLt epilogue fusion | `matmul_epilogue` Backend trait + fuse! L3 emit（f32 relu/gelu → cublasLt 1-kernel） |
| F-3 | mega_fuse! DAG 融合 | `prev` キーワードで op0 レジスタ→op1 直接参照（float4/double2 全パス） |
| F-4 | f64 fuse double2 ベクトル化 | fuse kernel f64 パスを scalar → `double2`（帯域 2×） |
| F-5 | pointwise→reduction fusion | `fuse!(x.exp().sum_axis(1))` → 単一 map-reduce カーネル（warp shuffle + smem、auto-capture） |
| G-1 | capturing scopeguard | `CapGuard` drop guard で panic 時の `capturing` AtomicBool 固着を防止 |
| G-2 | PyGraph pointer indirection | `TrainingGraph` → `PyGraph` 内包、`update_param_ptr()` で 8-byte ポインタ更新 |
| G-5 | Conditional Nodes | `cuda_conditional_set_from_scalar` + `cuda_if_positive`（k_cond_set + CondCmp enum） |
| MS | Multi-stream D2H overlap | `cuda_to_vec_async`（copy_stream D2H）+ `DoubleBuffer<T>`（ping-pong input） |
| L4 | Mega-kernel tiled fusion | `mega_fuse_tiled_kernel_source`: shared memory tile reuse（≥2 ops, ≥64K elements） |

#### W20 基盤（最適化ロードマップ以前）

Auto-fusion cost model, CUDA Graph capture/replay (1.67×), best-fit dual-pool allocator + cross-pool fallback, dispatch KernelId enum + flat array, `__ldg` prefetch, Pool split/coalesce bug fix (matmul ≥2048), cuBLAS workspace 32MiB, malloc_async→sync fallback, boundary.rs CUDA互換化 (231 tests).

#### 残課題（未修正）

| ID | 内容 |
|---|---|
| F-6 | EqSat に分配法則・結合法則なし |
| F-7 | mega_fuse! が全 op に同一 n_inputs を渡す（入力数不一致時のバグ可能性） |
| G-3 | concurrent capture 非対応 — global capturing flag が 1 つ |
| G-4 | end_capture 失敗時にストリーム状態が不明確（abort なし） |
| G-6 | 短い kernel chain でも無条件に graph 化（コストモデルなし） |
| N-1 | ✅ `AddAssign`/`SubAssign`/`MulAssign` 実装済み（alloc-and-swap、Backend in-place は将来課題） |
| N-2 | ✅ `mega_fuse!` auto-capture 実装済み（`inputs:` 省略可、`prev` 自動除外） |
| N-3 | ❌ `Index<(Range, Range)>` は Rust 言語制約で実装不可（`Index::index → &Output`）。`.slice()` が正式 API |
| N-4 | ✅ `linspace<T>(start, stop, n) -> Tensor<T>` prelude export 済み |

#### 不採用（リバート済）

- Auto-tune BLOCK_SIZE (`cuOccupancyMaxPotentialBlockSize` → 2-3× 悪化)
- Persistent grid-stride (grid capping → 2-3× 悪化)

#### ベンチマーク（GH200 480GB, 4096×4096 f32）

| Workload | nabla | PyTorch 2.7 | 備考 |
|---|---|---|---|
| exp / sin / tanh | 0.040 ms | 0.040–0.041 ms | ≈ parity |
| add / emul | 0.058 ms | 0.058 ms | ≈ parity |
| fuse exp+sin | 0.041 ms | 0.081 ms (eager) | **nabla 2× faster** |
| fuse 4-op | 0.050 ms | — | single kernel |
| mega_fuse 4-out | 0.141 ms | — | single kernel, 4 outputs |
| sum_all / max_all | 0.028 ms | 0.026 ms | PyTorch 1.08× |
| matmul 4096 (cuBLAS TF32) | 0.378 ms | 2.68 ms | **nabla 7× faster** |
| Dispatch: fuse! 4-op | — | — | 3.36× vs unfused |
| Dispatch: CUDA Graph 36-op | — | — | 1.67× vs eager (56μs/step) |

#### 参考文献

- [PyGraph] Compiler-level CUDA Graph pointer indirection ([2503.19779](https://arxiv.org/abs/2503.19779))
- [DeepFusionKernel] SwiGLU MLP deep fusion ([2602.11808](https://arxiv.org/abs/2602.11808))
- [FlashMoE] Distributed MoE single persistent kernel ([2506.04667](https://arxiv.org/abs/2506.04667))
- [CUDA 12.6] Constant-time graph launch — 100-node graph CPU overhead 66% reduction

### 14.3 Implementation History

| Wave | Items |
|---|---|
| W3 | Math error messages (M×N format) |
| W4–5 | E-graph CAS (32 rules + CDiff), Einsum greedy contraction path, MultiDual<T,N> |
| W6 | GradPrep AD, Index bracket notation, expm (Padé [7/7]), bdf1, if_euler_scalar |
| W7 | bf16/f16 Scalar, DAE solver, CAS diff_simplify (41 rules) |
| W8 | vcat!/hcat! macros, METD ODE (φ₁ order=8), Störmer-Verlet, StaticMatrix ref ops + Index |
| W9 | fuse! EqSat simplification (6 rules), 10 examples |
| W10 | SVD/lstsq bug fixes, Einsum canonicalization, ML activations (relu/gelu/softmax/layer_norm), reductions (max/min/var/std), broadcast, norms, utils (clamp/trace/diag/one_hot/cumsum/gather…) |
| W11 | Named axes `Tensor<T,B,Axes=()>` + `axis!`/`named_zeros!`, EqSat `fuse!` (FuseExpr 16-node, 15 rules) |
| W12 | CAS multi-var diff (6 rules, IsDifferentVar), StaticMatrix shape-algebra tests |
| W13 | `fuse!` L1 element-wise fusion (single `from_fn` pass), Einsum L1 tiled contraction (tile=64) + path annotation |
| W14 | CUDA backend (cudarc 0.19, 42 kernels f32/f64, NVRTC JIT), HIP backend (hip-runtime-sys 0.1, hiprtc), wgpu multi-versioned kernels (wg 64/128/256, matmul tile 8/16/32) |
| W15 | WMMA tensor cores (`nvcuda::wmma` Volta+, `rocwmma` CDNA2+) + warp shuffle reductions (`__shfl_down_sync`, 8x shared mem reduction), Recursive GEMM TRSM (`gpu_trsm_lower`, base n≤32 → CPU), `GpuTape<T>` GPU-resident AD (12-op enum, buffer tape, backward via existing kernels) |
| W16 | `BcsrMatrix<T>` GPU sparse (BCSR format, `from_sparse`, CPU SpMM, `WGSL_BCSR_SPMM` kernel source), `mixed_spmm_f64` mixed-precision refinement, `StaticMatrix::outer`, Cargo examples (01-10, `cargo run --example`) |
| W17 | `LinearLayout<N>` F₂ binary matrix swizzle (`identity`/`compose`/`apply`/`swizzle_for_tile`/`to_wgsl_swizzle_fn`, LinearLayout16/32/64), `fuse!` L3 GEMM+activation fusion (`detect_gemm_activation`, `GEMM_ACTIVATIONS`, `Tensor::map`, `Tensor::matmul_fused`) |
| W18 | `PararealConfig` + `parareal_solve` CPU parallel-in-time (rayon fine propagator), `#[nabla_grad]` source-transform AD (Dual<T> lifting, forward-mode), `Dual` convenience methods (exp/ln/sin/cos/tanh/sqrt/abs/recip + mixed arithmetic), wgpu 2D register-tile software MMA (`gen_matmul_register_tile`, `select_register_tile_params`, `MatmulRegTile` dispatch ≥64³) |
| W19 | GPU kernel fusion L1 JIT (`cuda_expr()` compile-time codegen → NVRTC/hiprtc runtime JIT → hash-based cache), `Backend::fuse_launch` trait method + `Tensor::__fuse_elementwise`, f32 float4 vectorized memory access in pre-compiled kernels (128-bit `LDG.E.128`, fast math `__expf`/`__logf`/`__sinf`/`__cosf`), GH200 benchmarks (3.8× fusion speedup, 30× gap vs PyTorch identified) |
| W20 | GPU caching memory allocator (best-fit dual-pool, 512B-aligned, block splitting, over-alloc 2/20MB, GC 0.9), float4 vectorized fuse codegen (`fuse_kernel_source()` emits `float4` + `__ldg` prefetch + scalar tail), async execution (removed all unnecessary syncs), single-element D2H readback (`copy_element()` — 4 bytes instead of full tensor), fusion cost model (`estimate_register_pressure()` proc macro + `maxrregcount=120`), CUDA Graph capture/replay (`NablaCudaGraph` API). **Results: sin/tanh/add match PyTorch (0.040ms), fuse exp+sin 2× faster than PyTorch eager, exp+readback 0.053ms ≈ PyTorch 0.053ms, gap reduced from 46× to ≈ parity** |
| W22 | Phase 3 GPU kernels: conv1d/2d/3d im*col+cuBLAS, conv_transpose2d (1-thread/output), max/avg/adaptive_avg_pool2d (+with_indices), FlashAttention-2 sdpa (BLOCK_M=BLOCK_N=64, online softmax, O(seq_len) HBM, no QKᵀ materialization), bmm/addmm/baddbmm cuBLAS StridedBatched, Blelloch parallel prefix scan cumsum/cumprod (smem 2×BLOCK), 2-phase batch_norm_train (in-place running stats), fused cross_entropy (online softmax+nll), prod_all warp-reduce, norm_lp GPU chain, count_nonzero GPU, max_pool2d_with_indices |
| W21 | Strict CPU/GPU separation (Backend trait: 全メソッド required, default body 廃止), `unflatten`/`contiguous`/`detach` 追加, 全 GPU スタブ実装完了 — wgpu: 11 WGSL shaders (activations, softmax, layer_norm, rms_norm, sum/max_axis1, embedding), HIP: 8 launch functions + KERNEL_NAMES 24 追加, CUDA KERNEL_NAMES 24 追加. Backend coverage: CUDA 74, HIP 74, wgpu 43 kernels. `unimplemented!()` ゼロ. `fuse!` auto-capture (`;` 省略可、`collect_all_path_idents` AST walker) |
| W23 | Notation layer: `fuse!`/`mega_fuse!` auto-capture (`;` 省略可), `AddAssign`/`SubAssign`/`MulAssign`, `linspace` prelude free fn, `mat!` semicolon syntax (`mat![1,2;3,4]`), `block!` macro, `dot`/`kron`/`diagm`/`cross`/`norm`/`norm_ord` free functions, `outer`/`eye_like`/`expand_as`/`axpy_`/`sum_dim`/`mean_dim`/`norm_ord` methods, `det`/`logdet`/`svd_into`/`qr_into` linalg, `NdTensor::from_vec`/`reshape_nd`/`into_2d`/`Tensor::into_nd` shape manipulation |
| W24 | R2 notation+linalg: `tr()`/`cast::<U>()`/`argsort()`/`cumsum_dim()`/`cumprod_dim()` (negative index), `slice_set()`, pretty print truncation (>6 rows/cols → 3...3), `cond()`/`rank()`/`pinv()`/`matrix_power()`/`eig_into()` (Francis QR), `schur()`/`logm()`/`sqrtm()`/`sylvester()`/`lyapunov()` (Bartels-Stewart), `xavier_uniform`/`kaiming_normal` init, `approx_eq`/`approx!` macro, `scatter_add_dim0`, `clip_grad_norm`/`zero_grad`/`scale_grad`, LinalgExt trait extended (+5 methods), prelude updated |
| W25 | R3 math+ML+ergonomics: **Linalg**: `solve_tridiag` (Thomas O(n)), `hessenberg` (public API), `polar` (SVD-based), `toeplitz`/`circulant`/`vandermonde` constructors, `balance` (Parlett-Reinsch), `frechet_deriv` (block-triangular), `continuous_riccati` (Hamiltonian Schur). **ML training**: `adamw_step`, `LrSchedule`+`lr_at_step` (cosine/linear/1cycle), `rotary_embedding` (RoPE), `GradScaler` (AMP loss scaling), `kv_cache_append`. **IO**: `save_tensors`/`load_tensors` (NBLA binary format). **Ergonomics**: `eachrow`/`eachcol` iterators + `IntoIterator`, `map_axis`, `similar`/`similar_shape`, `filter_sum`/`count_where`, `backslash` (auto LU/QR), `fuse_!` (in-place), `tmap!` (unified CPU/GPU) |
| W26 | 3-layer architecture restructuring: **Layer 1** (nabla-macros): split lib.rs → fuse.rs + mat.rs + grad.rs modules. **Layer 2** (nabla-core): split tensor.rs → tensor/{mod, constructors, ops, iter, display, ndtensor, static_matrix, dyntensor}, split cuda_backend.rs → cuda/{mod, kernels, graph, pool}. **Layer 3** (nabla): split lib.rs → {constructors, nn, optim, io, macros}.rs (slim prelude), split linalg.rs → linalg/{mod, decompose, solve, matrix_fn, equation, structured}. File size policy: >500 lines → module dir |
| W27 | Notation layer improvements (3-persona evaluation → fixes): **CAS**: method chain `.sin_()/.cos_()/.powf(n)/.powi(n)`, `From<f64/i32> for Expr`, `var()` free fn. **ML abstractions**: `Module<T,B>` trait, `Optimizer<T,B>` trait + `AdamW` struct, `Conv1d/2d/3dConfig` builder pattern, `embedding()` free fn, `view()` copy warning. **ODE**: `saveat` option (all 7 configs), `bdf2` solver (2nd-order implicit). **Linalg**: `geig()` (generalized eigenvalue via Cholesky reduction), `cond1()` (1-norm condition number). **Broadcasting**: auto broadcast for `&Tensor +/- &Tensor` (row/col/scalar shape inference). **Macros**: `range!` alias for `frange!`. **Refactoring**: `einsum.rs` 1392→1178 lines (−15%), `macros.rs` → `notation.rs` rename, `stencil.rs` import fix, unused macro cleanup. **SDE**: `euler_maruyama` (strong order 0.5) + `milstein` (strong order 1.0), `SdeConfig`, inline Xorshift64+Box-Muller PRNG. **sym! macro**: Pratt parser proc macro for `sym!(sin(x^2) + cos(y))` → `Expr::*` codegen. **Traits**: `MatrixLike<T>` read-only common interface (Tensor/StaticMatrix/TensorView), `TensorView<'a,T,B>` zero-copy borrowed slice. **Arch decision**: NdTensor<T> CPU-only を正式方針化 (GPU計算単位=2D、N-D統合は不要) |
| W28 | Round 2 notation improvements (3-persona re-evaluation → 40+ fixes): **Math**: 11 inverse trig/hyperbolic ops (asin/acos/atan/atan2/sinh/cosh/asinh/acosh/atanh/log2/log10) on Backend+Scalar+Dual+CAS+GPU. **Linalg**: `inv()`, `null_space()`, `orth()`, `slogdet()`, `cond_p(p)` unification, `cond_inf()`, `discrete_lyapunov`, `discrete_sylvester`, `care` alias. **CAS**: `substitute`, `gradient`, `jacobian`, `hessian`, 8 new ExprKind variants with diff rules, sym! macro extended. **ODE**: event detection (`terminate` callback), `OdeSolution::eval(t)` + `Index`, backward integration, `parareal_solve_tensor`, `ensemble_euler_maruyama`, builder pattern (`.with_dt()`). **Autograd**: `Variable::relu()/sigmoid()/gelu()/sum_axis_var()/mean_var()/cross_entropy()`, `Tape::no_grad()`. **Module**: train/eval mode, `forward_with` multi-input, inspection methods (children/named_children/buffers/apply), `Linear` layer struct. **Optimizer**: vectorized `AdamW`, `SGD`, `set_lr()`, `LrScheduler`. **Ergonomics**: `sum()/max()/min()` aliases, `epow`, `linear()`, `item()/to_vec()/elements()/indexed_iter()`, `similar_zeros<U>()`, `tensor_range!`, owned `Add/Sub/Neg`, `dropout_auto`, `topk` axis=0, `argsort_by`. **Sparse**: transpose/speye/addition. **Constructors**: `logspace`, `geomspace` |
| W29 | Round 3 improvements (25 items — 14 important + 11 nice-to-have): **Math**: `tan()` across all layers, commutative scalar*Tensor (f32/f64 * &Tensor, + and -), owned matmul (all ownership combos), `from_vec(data,nrows,ncols)`, `mean()`/`prod()` aliases. **Linalg**: `vandermonde_rect()`. **CAS**: `ExprKind::Tan` + diff rules, `eval_tensor` inverse trig/hyperbolic fix, owned `Expr` operators (Add/Sub/Mul/Div/Neg all combos), `(&Expr,f64)` Sub/Div, `diff`/`simplify` prelude. **ODE/SDE**: `stormer_verlet` → `SymplecticSolution`, `ensemble_euler_maruyama` parallel (`std::thread::scope`), SDE backward integration, `Bdf2Config` struct, `SdeConfig::with_noise_dims()`, `euler`/`rk4`/`dormand_prince` prelude. **Autograd**: `Variable::ediv()`/`epow()`/`abs()`/`log1p()`/`silu()`. **Module/Optimizer**: `Module::train()`/`eval()` shorthand, `state_dict`/`load_state_dict` pattern, io generic scalar, `Optimizer::step_slices()` simplified, `AdamW::from_module()`. **Bug fix**: `reduce_axis` double-count first element (loop start 0→1) |
| W30 | Round 4 improvements (25 items — 2 critical + 10 important + 13 nice-to-have): **Autograd NN ops** (critical): `Variable::softmax(axis)/reshape(m,n)/transpose()/linear_forward(w,b)/dropout(p,training)/clamp(lo,hi)/mse_loss(target)/cross_entropy_indices(targets)`. **Module/Autograd bridge** (critical): `Module::forward_var()` trait, `Linear::forward_var`, `Tape::track_params()`, `Tape::var()` alias. **Math**: `Div<T>` operator (Tensor+Variable), `prod_axis`, `var_axis_ddof`, `zeros_vec/ones_vec/rand_vec/randn_vec`. **Linalg**: `LinalgExt` f32 support (f32→f64 promotion, 45+ methods). **CAS**: auto-simplify in `gradient/jacobian/hessian`, `eval` domain checks, `eval/eval_tensor` prelude. **ODE**: `OdeProblem<T,B,F>` wrapper, `EulerConfig/Rk4Config` with `saveat`. **Training**: `GradScaler::scale_factor()`, `AdamW::from_params`, `backward()` NaN/Inf detection, vectorized `clip_grad_norm`, in-place `zero_grad` |

---

# Part IV — Research Foundations

## 15. Papers

### 15.1 Rust LA + Julia→Rust migration

| Paper | ID | Key insight for nabla |
|---|---|---|
| NPB-Rust | [2502.15536] | Rayon shared-mutable state is the parallelism bottleneck → iterator chains |
| Rewrite it in Rust: Physics | [2410.19146] | Ownership-based parallelism can exceed C++ by 5.6x |
| Rust Tensor from Python | [2510.01495] | Rust tensor kernels outperform NumPy/Numba across complexity spectrum |
| kornia-rs | [2505.12425] | Native Rust tensor 3–5x faster than C++ wrapper approach |
| ad-trait | [2504.15976] | Trait-based forward+reverse AD in Rust, SIMD forward mode, ICRA 2025 |
| SGEMM portability | [2507.15277] | Multi-versioned kernels close 10–30% cross-device gap |
| Portable GPU SVD (Julia) | [2508.06339] | Multiple dispatch enables cross-vendor GPU linalg |
| Julia TRMM/TRSM | [2504.13821] | Recursive GEMM decomposition: cuBLAS-comparable in ~300 LOC |
| Batched einsum canon. | [2601.12220] | Graph normalization → 4.7x tuning reuse over JAX |
| Sigma (einsum→dataflow) | [ASPLOS 2023] | Loop ordering + tiling + fusion from einsum notation |
| diffsol (Rust ODE) | [JOSS 2026] | DiffSL DSL + Enzyme AD + JIT: Julia SciML pattern in Rust |
| Rust & Julia comparison | [IEEE CiSE 2024] | Julia = rapid prototype, Rust = maximum performance + safety |

### 15.2 Strictly superior techniques per subsystem

| Paper | ID | Replaces | Improvement |
|---|---|---|---|
| Linear Layouts F₂ | [2505.23819] ASPLOS 2026 | TILE=16 naïve shared memory | Bank-conflict-free swizzle, generic for any tile size |
| Subgroup matrix (wgpu) | gfx-rs/wgpu#5555 | matmul_tiled shared memory | Hardware tensor-core MMA. 2.3–2.9x on Intel |
| Neptune operator fusion | [2510.08726] | Separate kernel dispatch | Fuses across reductions. 1.35x avg over Triton/XLA |
| Liger Kernel fusion | [2410.10989] | fuse! separate ops | SRAM-resident chains. 20% throughput, 60% memory |
| Deep Kernel Fusion | [2602.11808] | GEMM→activation boundary | Fuses matmul + pointwise into single kernel |
| Mirage Persistent Kernel | [2512.22219] | Per-op kernel dispatch | SM-level mega-kernel: cross-op pipelining, 1.7× LLM inference |
| FlashFuser (DSM fusion) | [2512.12949] | Local scratchpad-only fusion | Inter-core DSM fusion: 58% memory reduction, 3.3× kernel speedup (H100) |
| Fused Kernel Library | [2508.07071] | Manual fused kernel dev | C++17 auto HF+VF at compile time: 2×–1000× speedup |
| PyGraph (CUDA Graphs) | [2503.19779] | Per-kernel CPU launch | Compiler-auto CUDA Graph deployment: 2× benefit over PyTorch2 |
| CUDA Graph batching | [2501.09398] PDP 2025 | Sequential iteration dispatch | Iteration unrolling into graphs: >1.4× for iterative solvers |
| FastUSP (launch overhead) | [2602.10940] | Naive USP dispatch | CUDA Graph + comm reorder: 1.12–1.16× (kernel launch = primary bottleneck) |
| STAlloc (memory planning) | [2507.16274] SOSP 2025 | Online caching allocator | Offline spatio-temporal planning: 85% fragmentation reduction, 32.5% throughput |
| VibeTensor | [2601.16238] | — | Stream-ordered caching allocator + CUDA Graphs in standalone Rust/C++ runtime |
| Locality-aware GPU AD | [2509.00406] | CPU Rc tape | On-device gradient, no GPU↔CPU roundtrip |
| IF Euler exponential | [2412.01181] | RK4 on stiff systems | Succeeds where RK4 diverges, no implicit solve |
| METD matrix exponential | [2406.13761] | No matrix ODE solver | Provable order-p for Lyapunov/Riccati/graph ODE |
| Parareal time-parallel | [2510.07672] | Sequential time steps | Parallel time dimension on multi-GPU |
| Acc-SpMM BitTCF | [2501.09251] PPoPP 2025 | CSC CPU-only | 2.52x avg over cuSPARSE (RTX 4090) |
| SMaT BCSR + MMA | [2408.11551] SC 2024 | CSC CPU-only | Up to 125x over cuSPARSE unstructured |
| Mixed-precision sparse | [2412.19322] | FP64-only factorization | 2–4x via FP16 preconditioner + FP64 correction |
| EqSat + MCTS for CAS | [2410.05534] PACT 2024 | Greedy tree rewrite | Stores all equivalents, globally optimal. 11% |
| Slotted E-Graphs | [PLDI 2025] | De Bruijn variable encoding | First-class bound variables, no e-graph explosion |
| EqSat high-level IR | [2502.17075] | Pattern-match proc macro | Algebraic rewrites across control flow |
| egglog (Datalog+EqSat) | [2304.04332] PLDI 2023 | Recursive simplifier | Incremental fixpoint + lattice analyses. Rust crate |
| Contraction path optimizer | [2405.09644] | Fixed 7-pattern order | Optimal path for N-D tensor networks |

### 15.3 Scientist adoption & notation design

| Source | Key insight for nabla |
|---|---|
| Tensor Considered Harmful [Chiang+ 2021] | Named axes eliminate transposition/broadcasting bugs — highest-leverage notation improvement |
| DifferentiationInterface.jl [Blondel+ 2024] | "Preparation pattern" (prep once / execute many) for AD API — amortizes tape/sparsity analysis |
| dfdx compile-time shapes [2302.05727] | Type-level dimension algebra (`Dim<M> * Dim<N>`) catches shape mismatches at compile time |
| ad-trait ICRA [2504.15976] | `impl Scalar for Dual<T>` — forward-mode AD as zero-change drop-in via trait |
| SciML survey 2024 | Julia→Rust = 5-10x LOC increase. Macros+inference must absorb this gap |
| Julia multiple dispatch composability | Scientists value composability (new type + existing functions = just works). nabla trait dispatch must achieve same |
| TTFX (Time-To-First-eXecution) | Julia's #1 complaint. Rust: compile once → instant. Position as advantage |
| rust-script [fornwall] | Single-file `.rs` scripts mitigate "no REPL" complaint. `//! nabla` dep for quick experiments |
| einops [Rogozhnikov 2022] | Readable tensor reshaping notation. Influenced `einsum!` axis naming design |
| ARRAY workshop [PLDI 2024] | DSL design for array languages — rank polymorphism, named dimensions as community direction |
