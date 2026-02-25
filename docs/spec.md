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
a[(0..3, 1..4)]    // zero-copy slice (Range<usize>)

// Error handling: Result<T, NablaError> — no silent NaN
a.solve(&b)?       // ? propagates NablaError

// Kernel naming convention (CUDA/HIP C strings in kernels_cu.rs)
k_{op_name}_f32    // e.g. k_conv2d_f32, k_max_pool2d_f32
k_{op_name}_f64    // f64 variant (CUDA/HIP only; not wgpu)

// Conv tensor layout: NCHW (N=batch, C=channels, H=height, W=width)
// H_out = (H + 2*padding - dilation*(kH-1) - 1) / stride + 1
```

### §0.2 REQ表

**Phase 0 (実装済 ✅ — API 契約、変更禁止)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 0 | REQ-B01 | MUST | Exactly one of {cpu,wgpu,cuda,hip} active per build | `nabla-core/src/backend.rs` |
| 0 | REQ-B02 | MUST NOT | Multiple features active — `compile_error!` on all 6 pairwise combinations | `nabla-core/src/lib.rs` |
| 0 | REQ-B03 | MUST NOT | CPU fallback path exist for GPU backends | `nabla-core/src/gpu.rs` |
| 0 | REQ-T01 | MUST | `use nabla::prelude::*;` imports all public types/traits/macros/free-fns | `nabla/src/lib.rs` |
| 0 | REQ-T02 | MUST | `Tensor<T>` aliases to `Tensor<T, DefaultBackend>` | `nabla-core/src/tensor.rs` |
| 0 | REQ-T03 | MUST NOT | wgpu backend accept f64 scalar (compile_error!) | `nabla-core/src/backend.rs` |
| 0 | REQ-T04 | MUST NOT | Any GPU backend accept c32/c64 scalar (compile_error!) | `nabla-core/src/backend.rs` |
| 0 | REQ-T05 | MUST | Backend trait: Phase 0 methods are required (no defaults); Phase 3+ methods have CPU defaults that GPU backends override | `nabla-core/src/backend.rs` |

**Phase 3A — GPU Convolution (CUDA/HIP; cross-ref §14.1.A)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3A | REQ-G-CONV-01 | MUST | `conv2d(x,w,bias,stride,padding,dilation,groups)` dispatches to GPU im2col+GEMM kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3A | REQ-G-CONV-02 | MUST | conv2d im2col col shape: `[N, C_in*kH*kW, out_H*out_W]`; weight `[C_out, C_in*kH*kW]`; strided-batched GEMM with batch=N | `nabla-core/src/kernels_cu.rs` |
| 3A | REQ-G-CONV-03 | SHOULD | f32 conv2d im2col kernel uses float4 (128-bit) loads (scalar reads currently; performance optimization) | `nabla-core/src/kernels_cu.rs` |
| 3A | REQ-G-CONV-04 | MUST | `conv1d(x,w,bias,stride,padding,dilation,groups)` dispatches to GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3A | REQ-G-CONV-05 | MUST | `conv3d(x,w,bias,stride,padding,dilation,groups)` dispatches to GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3A | REQ-G-CONV-06 | MUST | `conv_transpose2d(x,w,bias,stride,padding,output_padding,groups)` dispatches to GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3A | REQ-G-CONV-07 | MUST NOT | conv GPU kernels accept f64 on wgpu backend | `nabla-core/src/backend.rs` (wgpu uses CPU default) |

**Phase 3B — GPU Pooling (CUDA/HIP; cross-ref §14.1.B)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3B | REQ-G-POOL-01 | MUST | `max_pool2d(x,kernel_size,stride,padding)` dispatches to GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3B | REQ-G-POOL-02 | MUST | `avg_pool2d(x,kernel_size,stride,padding)` dispatches to GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3B | REQ-G-POOL-03 | MUST | `adaptive_avg_pool2d(x,output_size:[usize;2])` dispatches to GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3B | REQ-G-POOL-04 | MUST | `max_pool2d` kernel stores argmax indices alongside output for backward | `nabla-core/src/kernels_cu.rs` |
| 3B | REQ-G-POOL-05 | MUST | All pooling kernels use one-thread-per-output-element parallelism | `nabla-core/src/kernels_cu.rs` |

**Phase 3C — GPU Attention & Batched GEMM (CUDA/HIP; cross-ref §14.1.F, §14.1.H)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3C | REQ-G-ATTN-01 | MUST | `sdpa(q,k,v,mask,dropout_p)` dispatches to FlashAttention-2 tiled GPU kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3C | REQ-G-ATTN-02 | MUST | FlashAttention-2 implements online softmax with O(seq_len) HBM memory | `nabla-core/src/kernels_cu.rs` |
| 3C | REQ-G-ATTN-03 | MUST NOT | FlashAttention kernel materialise full QK^T matrix in HBM | `nabla-core/src/kernels_cu.rs` |
| 3C | REQ-G-ATTN-04 | MUST | `bmm(a,b)` f32 dispatches to `cublasSgemmStridedBatched` on CUDA backend | `nabla-core/src/cuda_backend.rs` |
| 3C | REQ-G-ATTN-05 | MUST | `baddbmm(c,a,b,beta,alpha)` dispatches to cuBLAS fused op on CUDA | `nabla-core/src/cuda_backend.rs` |
| 3C | REQ-G-ATTN-06 | MUST | `addmm(c,a,b,beta,alpha)` dispatches to cuBLAS fused op on CUDA | `nabla-core/src/cuda_backend.rs` |
| 3C | REQ-G-ATTN-07 | MUST | bmm/baddbmm/addmm on HIP/wgpu use native tiled matmul loop (no cuBLAS) | `nabla-core/src/hip_backend.rs`, `gpu_wgpu.rs` |

**Phase 3D — GPU Reductions (CUDA/HIP; cross-ref §14.1.J)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3D | REQ-G-RED-01 | MUST | `cumsum(x,dim)` dispatches to GPU parallel prefix sum (Blelloch scan) kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3D | REQ-G-RED-02 | MUST | `cumprod(x,dim)` dispatches to GPU parallel prefix (Blelloch scan) kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3D | REQ-G-RED-03 | MUST | `prod_all(x)` dispatches to GPU warp-shuffle reduction kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3D | REQ-G-RED-04 | MUST | `norm(x,p,dim)` Lp-norm dispatches to GPU kernel for p∈{1,2,inf} | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3D | REQ-G-RED-05 | MUST | `count_nonzero(x)` dispatches to GPU reduction kernel | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3D | REQ-G-RED-06 | MUST | cumsum/cumprod kernel output: `out[i,j] == sum/prod(x[i,0..=j])` for dim=1 | `nabla-core/src/kernels_cu.rs` |

**Phase 3E — GPU Normalization & Loss (CUDA/HIP; cross-ref §14.1.C, §14.1.E)**

| Phase | REQ-ID | MUST / MUST NOT | 制約（1行） | 実装ファイル |
|---|---|---|---|---|
| 3E | REQ-G-NORM-01 | MUST | `batch_norm` GPU kernel updates running_mean/running_var in-place with momentum | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3E | REQ-G-LOSS-01 | MUST | `cross_entropy_loss` GPU: fused log_softmax + nll_loss in single kernel pass | `nabla-core/src/cuda_backend.rs`, `kernels_cu.rs` |
| 3E | REQ-G-LOSS-02 | MUST | `cross_entropy_loss` fused kernel uses online-softmax (numerically stable) | `nabla-core/src/kernels_cu.rs` |

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

Phase 4 (performance — 🔲 future, not in current REQ scope)
  ├─ Multi-stream async pipeline
  └─ L4 Mega-kernel fusion (SM-level persistent kernel [2512.22219])
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

```
nabla/                       [workspace root]
├── Cargo.toml               members: nabla-core, nabla-macros, nabla
├── nabla-core/              foundation crate — tensor + backend
│   └── src/
│       ├── lib.rs
│       ├── tensor.rs        Tensor<T,B> + StaticMatrix<T,R,C> + NdTensor<T> + DynTensor
│       ├── backend.rs       Backend trait (sealed) + Cpu impl + CpuStorage
│       ├── scalar.rs        Scalar trait + Complex<T> + Dual<T> + MathOps/ReductionOps
│       ├── gpu.rs           GpuStorage<T> + GpuContext trait + wgpu/CUDA/HIP dispatch
│       ├── layout.rs        LinearLayout<N> F₂ binary matrix swizzle (W17)
│       └── wgsl.rs          WGSL register-tile MMA codegen (W18)
├── nabla-macros/            proc-macro crate (syn/quote/proc-macro2)
│   └── src/
│       ├── lib.rs           mat! + fuse! + einsum! + named! + generated! + stencil! + nabla_grad!
│       ├── einsum.rs        einsum parser + codegen
│       └── stencil.rs       stencil! offset indexing
├── nabla/                   facade crate — domain modules + macro_rules!
│   ├── src/
│   │   ├── lib.rs           re-exports + map! + par_map! + map_! + prelude
│   │   ├── linalg.rs        9 dense factorizations + Diagonal/Symmetric/Triangular
│   │   ├── sparse.rs        SparseMatrix<T> CSC + BcsrMatrix<T> GPU sparse (W16)
│   │   ├── autograd.rs      Reverse-mode AD: Tape + Variable + backward
│   │   ├── cas.rs           Symbolic CAS: Expr tree + diff/simplify/eval (E-graph W4-5)
│   │   └── ode.rs           ODE solvers: euler/rk4/dormand_prince/bdf1/if_euler + parareal (W18)
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
│       ├── boundary.rs      CPU boundary tests (157)
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

---

## 3. Notation reference

`use nabla::prelude::*;` — all types, traits, macros, free functions available.

**Design goal: Python の書きやすさ × Julia の数式記法 × Rust のゼロ GC。**
記法 5 原則: (1) free function 型推論 `zeros(m,n)` (2) `Index` trait bracket 記法 `a[(i,j)]` (3) owned/borrowed 両形式 `a*b` / `&a*&b` (4) 短い名前 + `e`-prefix (`emul`/`ediv` = Julia `.* `/`./`) (5) Macro = syntax extension (`fuse!`=Julia `@.`, `vcat!`=Julia `[A;B]`)

### 3.1 Quick reference — Math → Python → Julia → nabla

| Math | Python | Julia | nabla | nabla advantage |
|---|---|---|---|---|
| $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ | `np.array([[1,2],[3,4]])` | `[1 2; 3 4]` | `mat![[1, 2], [3, 4]]` | Compile-time shape check |
| $A_{ij}$ | `A[i,j]` | `A[i,j]` | `a[(i, j)]` | Bracket 記法, 0-indexed (`()` = Rust 制約) |
| $A_{ij} = v$ | `A[i,j] = v` | `A[i,j] = v` | `a[(i, j)] = v` | `IndexMut` — 同上 |
| $A_{1:3, 2:4}$ | `A[0:3, 1:4]` | `A[1:3, 2:4]` | `a[(0..3, 1..4)]` | Zero-copy view |
| $A^\top$ | `A.T` | `A'` | `a.t()` | 3 chars |
| $A^*$ (adjoint) | `A.conj().T` | `A'` (same!) | `a.h()` | **adjoint ≠ transpose, 4 chars** |
| $AB$ | `A @ B` | `A * B` | `a * b` | **Same as Julia** (owned) |
| $A \circ B$ (Hadamard) | `A * B` | `A .* B` | `a.emul(b)` | Julia `A .* B` の `e`-prefix 化 |
| $cA$ | `c * A` | `c * A` | `c * a` | **Same** |
| $Ax = b$ | `np.linalg.solve(A,b)` | `A \ b` | `a.solve(&b)?` | **Result — no silent NaN** |
| $\sin(A)$ element-wise | `np.sin(A)` | `sin.(A)` | `a.sin()` | **Shortest — method chain** |
| $y = \sin(x)^2$ fused | `torch.sin(x)**2` | `@. sin(x)^2` | `fuse!(x.sin().powf(2.0); x)` | **GPU kernel auto-fusion** |
| $C = AB$ (einsum) | `np.einsum('ik,kj->ij',A,B)` | `@einsum` | `einsum!(c[i,j] = a[i,k] * b[k,j])` | **7 patterns + spanned errors** |
| $\nabla_x L$ | `loss.backward()` | `gradient(f, x)` | `loss.backward(); x.grad()` | **PyTorch-familiar + zero GC** |
| $\frac{df}{dx}$ symbolic | SymPy `diff(f, x)` | Symbolics.jl | `expr.diff("x").simplify()` | Built-in, single crate |
| $\dot{y} = f(t,y)$ | `scipy.integrate.solve_ivp` | DiffEq.jl | `dormand_prince(f, y0, t, dt)` | Built-in, single crate |
| $\nabla^2 u$ (Laplacian) | manual loop | `@tullio` | `stencil!(out[i,j] = ...)` | **Auto bounds, zero boundary** |
| $[A; B]$ vcat ✅ | `np.vstack([A,B])` | `[A; B]` | `vcat!(a, b, c, ...)` | Julia `[A; B]` style — variadic macro |

### 3.2 Tensor construction

| Operation | nabla | Python | Julia |
|---|---|---|---|
| Matrix literal | `mat![[1, 2], [3, 4]]` | `np.array([[1,2],[3,4]])` | `[1 2; 3 4]` |
| Zeros / Ones / Fill | `zeros(m, n)` / `ones(m, n)` / `fill(m, n, val)` | `np.zeros((m,n))` | `zeros(m,n)` |
| Identity | `eye(n)` | `np.eye(n)` | `I(n)` |
| Random | `randn(m, n)` / `rand(m, n)` | `np.random.randn(m,n)` | `randn(m,n)` |
| From function | `from_fn(m, n, \|r, c\| expr)` | `np.fromfunction(f, (m,n))` | — |
| Range / Linspace | `arange(0.0, 1.0, 0.1)` / `linspace(0.0, 1.0, n)` | `np.arange` / `np.linspace` | `0.0:0.1:1.0` |
| Static (stack) | `StaticMatrix::<f64,3,3>::zeros()` | — | `SMatrix{3,3}(...)` |
| N-D | `nd_zeros(&[d0, d1, d2])` | `np.zeros((d0,d1,d2))` | `zeros(d0,d1,d2)` |
| Cat | `vcat!(a, b)` / `hcat!(a, b)` | `np.vstack` / `np.hstack` | `[A; B]` / `[A B]` |
| Reshape | `a.reshape(m, n)` | `A.reshape(m, n)` | `reshape(A, m, n)` |

Free functions in `prelude`, type inferred from context. GC なし、即座解放、`StaticMatrix` はスタック配置。

### 3.3 Indexing & slicing

`Index`/`IndexMut` trait で **bracket 記法**。0-indexed。全スライスは **zero-copy** + 借用チェッカによる寿命保証。

| Operation | nabla | Python |
|---|---|---|
| Read/Write | `a[(i, j)]` / `a[(i, j)] = v` | `A[i,j]` / `A[i,j] = v` |
| Submatrix | `a[(0..3, 1..4)]` | `A[0:3, 1:4]` |
| Row/Col slice | `a[(0..3, ..)]` / `a[(.., 1..4)]` | `A[0:3, :]` / `A[:, 1:4]` |
| N-D read | `t[&[i,j,k]]` | `T[i,j,k]` |

Rust 制約: `Index` trait は単一引数 → `a[(i, j)]` とタプルで渡す（+2 chars）。対価: 型安全 + スライス寿命の静的保証。

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

`emul`/`ediv`: Julia `.*/./` の `e`-prefix 化。`&` = 明示的ゼロコピー + 再利用保証（Python/Julia にはない選択肢）。

### 3.5 Broadcasting & fusion

| Level | nabla | PyTorch | GPU | Alloc |
|---|---|---|---|---|
| Single op | `a.sin()` | `torch.sin(a)` | ✅ | 1 |
| Fused chain | `fuse!(x.sin().powf(2.0); x)` | 2+ kernels | ✅ **1 kernel** | **1** |
| Closure | `map!(\|x\| f(x), &a)` | — | ❌ CPU | 1 |
| In-place | `map_!(a, \|x\| f(x), &b)` | `torch.sin(B, out=A)` | ❌ CPU | **0** |
| Parallel | `par_map!(\|x\| f(x), &a)` | — | ❌ CPU | 1 |

**原則**: GPU → `fuse!` 一択。CPU closure → `map!`。in-place → `map_!`。並列 → `par_map!`。

Element-wise methods: `.exp()` `.ln()` `.log1p()` `.powf(p)` `.sin()` `.cos()` `.tanh()` `.sqrt()` `.abs()` `.recip()` `.neg()` `.erf()` `.ceil()` `.floor()` `.round()` — all PyTorch-identical。Reduction: `.sum()` `.max()` `.min()` `.argmax()` `.argmin()`。

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

Structural: `Diagonal::new(v)`, `Symmetric::new(a, Side::Lower)?`, `Triangular::new(a, TriKind::Lower)?`。Factorization reuse: `lu.solve(&b)`, `lu.inverse()`, `lu.reconstruct()`。`?` = zero-cost `Result`（silent NaN 排除）。

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

**Symbolic CAS**: `let f = (x.clone() * x.clone()).sin(); let df = f.diff("x").simplify();`

**ODE solvers**: `euler` (1), `rk4` (4), `dormand_prince` (5(4) adaptive)。Preparation pattern: `grad_prep(f, &x)` で prep/execute 分離。AD + CAS + ODE が**単一 crate に統合**。

### 3.11 Utilities

| Math | nabla | Python | Julia |
|---|---|---|---|
| $0 \le x < 1$ | `between!(0.0, x, 1.0)` | `0 <= x < 1` | `0 ≤ x < 1` |
| $0.0, 0.1, \ldots, 1.0$ | `arange(0.0, 1.0, 0.1)` | `np.arange(0,1,0.1)` | `0.0:0.1:1.0` |
| $x \mapsto f \mapsto g$ | `pipe!(x, f, g)` | — | `x \|> f \|> g` |
| $f(a, b, c)$ from tuple | `splat!(f, (a, b, c))` | `f(*args)` | `f(args...)` |
| Named struct | `named!(a: i32 = 1, b: f64 = 2.0)` | `dict(a=1, b=2.0)` | `(a=1, b=2.0)` |

### 3.12 Parallelism & GPU dispatch

| Strategy | nabla | GPU |
|---|---|---|
| Parallel construct | `par_from_fn(m, n, \|r,c\| expr)` | ❌ CPU |
| Parallel map | `a.par_map(\|x\| expr)` | ❌ CPU |
| GPU single op | `a.sin()` | ✅ |
| GPU fused chain | `fuse!(x.sin(); x)` | ✅ **1 kernel** |
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
| Indexing | `Index` trait | `a[(i,j)]`, `a[(0..3, 1..4)]`, `a[(0..3, ..)]` |
| Unary op | Method, ≤ 5 chars | `.t()`, `.h()`, `.sin()`, `.exp()`, `.abs()`, `.neg()` |
| Binary op | Operator or short method | `a * b`, `a + b`, `a.emul(b)`, `a.ediv(b)` |
| Factorize | Short method + `?` | `.lu()?`, `.qr()`, `.chol()?`, `.svd()?`, `.ldl()?` |
| Solve | Verb + `?` | `.solve(&b)?`, `.lstsq(&b)?`, `.inv()?` |
| Reduce | Verb | `.sum()`, `.max()`, `.min()`, `.argmax()` |
| In-place | Method + `_` suffix (PyTorch convention) | `.mm_(&a, &b)`, `.add_(&b)` |
| AD | PyTorch-familiar | `tape.var(x)`, `loss.backward()`, `x.grad()` |
| Sparse | Free function | `sparse(m, n, &trips)?` |

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
| `gpu` / `gpu_wgpu` | — | ✅ | — | — |
| `gpu` / `gpu_cuda` | — | — | ✅ | — |
| `gpu` / `gpu_hip` | — | — | — | ✅ |
| `linalg` / `sparse` | ✅ | ❌ | ❌ | ❌ |
| `cas` / `ode` | ✅ | ✅ | ✅ | ✅ |
| `autograd` | ✅ | ✅ | ✅ | ✅ |

---

## 5. GPU implementation

### 5.1 Architecture overview

```
                    ┌─────────────────────────────────┐
                    │        gpu.rs (dispatch)         │
                    │  GpuContext trait + GpuStorage<T> │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        wgsl.rs / gpu.rs   cuda_backend.rs   hip_backend.rs
        WGSL shaders     CUDA driver API   HIP runtime
        wgpu::Buffer     CUdeviceptr       hipDeviceptr_t
        pollster sync    cuLaunchKernel    hipLaunchKernelGGL
              │                │                 │
              ▼                ▼                 ▼
        kernels_wgsl.rs       kernels_cu.rs (shared)
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

CUDA/HIP C kernels: embedded as `const &str` in `kernels_cu.rs`. Block size: 256 (tunable). Compiled at runtime via `nvrtc` / `hiprtc`. **Single source** — HIP C is source-compatible with CUDA C for standard math kernels.

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
| CUDA Graph capture/replay | ✅ W20 | cuda | `NablaCudaGraph` API |
| Phase 3 GPU kernels (conv/pool/attn/bmm/scan/norm/loss) | ✅ | cuda/hip | im2col+cuBLAS, FlashAttn-2, Blelloch scan, 2-pass BN, fused cross_entropy |
| Mega-kernel fusion (L4) | 🔲 | cuda/hip | SM-level persistent kernel [2512.22219] |

**Kernel fusion levels:**

| Level | Scope | Status |
|---|---|---|
| L1: Element-wise | Fuse consecutive unary/binary ops → single JIT kernel | ✅ W19 GPU |
| L2: Reduction | Fuse across reduction ops with loop-carried deps | ✅ W13 CPU |
| L3: GEMM+pointwise | Fuse matmul + activation + normalization | ✅ W17 |

Fusion pipeline: `fuse!` AST → egg EqSat simplify → `cuda_expr()` → NVRTC/hiprtc JIT → FNV-1a hash cache.

**Benchmark results** (GH200 480GB, 4096×4096 f32, PyTorch 2.7.0):

| Workload | nabla | PyTorch | Gap |
|---|---|---|---|
| sin / tanh / add | 0.040–0.058 ms | 0.041–0.058 ms | **≈ parity** |
| fuse exp+sin (2-op) | 0.041 ms | 0.081 ms (eager) | **nabla 2× faster** |
| sum_all / max_all | 0.028–0.029 ms | 0.026 ms | PyTorch 1.08–1.12× |
| matmul 1024 | 0.029 ms | 0.057 ms | **nabla 2.0×** |

Pre-optimization (W19): exp 2.38ms → **46× gap reduced to ≈ parity** via: async execution, single-element readback, float4 vectorization, caching allocator, fusion cost model.

**効果なし（リバート済）:** Auto-tune BLOCK_SIZE (`cuOccupancyMaxPotentialBlockSize` → 2–3× 悪化), Persistent grid-stride (Grid capping → 2–3× 悪化).

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

---

### 13.4 CAS

All planned: `egg` 32 rules + `CDiff` + `diff_simplify` (41 rules) + `FuseExpr` 16-node EqSat in `fuse!` (15 rules) + multi-var diff (6 rules). No remaining items.

---

### 13.5 Sparse

GPU: `BcsrMatrix<T>` BCSR + `WGSL_BCSR_SPMM` kernel (W16)。`mixed_spmm_f64` mixed-precision refinement。CPU: CSC via faer。

---

### 13.6 einsum!

7 patterns: GEMM, GEMV, Hadamard, trace, outer, batch GEMM, N-D fallback。`classify()` + `canonicalize()` で等価式のチューニング再利用。L1 tiled contraction (tile=64, W13)。GPU: `matmul_into` backend dispatch。

---

### 13.7 Notation & DX

Named axes (`Tensor<T,B,Axes=()>` W11): compile error on axis mismatch。StaticMatrix const-generic shape algebra (W12)。Tensor manipulation: reshape/view/permute/cat/stack/squeeze/unsqueeze/flatten/chunk — PyTorch API 名採用。Axis reduction: `.sum_axis(d)` / `.mean_axis(d)` + keepdim variants。

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

### 14.2 Performance

| Item | Priority | Status |
|---|---|---|
| Multi-stream pipeline | 🟡 Medium | 🔲 |
| Mega-kernel fusion (L4) | 🔵 Low | 🔲 |

**W20 完了:** Auto-fusion cost model, CUDA Graph capture/replay, best-fit allocator, `__ldg` prefetch.
**効果なし（リバート済）:** Auto-tune BLOCK_SIZE (2-3× 悪化), persistent grid-stride (2-3× 悪化).

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
| W21 | Strict CPU/GPU separation (Backend trait: 全メソッド required, default body 廃止), `unflatten`/`contiguous`/`detach` 追加, 全 GPU スタブ実装完了 — wgpu: 11 WGSL shaders (activations, softmax, layer_norm, rms_norm, sum/max_axis1, embedding), HIP: 8 launch functions + KERNEL_NAMES 24 追加, CUDA KERNEL_NAMES 24 追加. Backend coverage: CUDA 74, HIP 74, wgpu 43 kernels. `unimplemented!()` ゼロ |

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
