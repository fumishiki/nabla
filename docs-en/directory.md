# nabla — Directory & File Structure

> Cross-reference: [spec.md](spec.md) | [notation.md](notation.md) — full specification

## 1. Documentation files

| File | Role |
|---|---|
| [spec.md](spec.md) | Architecture, feature catalog, design decisions, performance benchmarks |
| [notation.md](notation.md) | API reference tables (types, math notation, macros, AD, NN modules) |
| [quick_start.md](quick_start.md) | Tutorial and code examples for all features |
| [directory.md](directory.md) | File structure and contributor guide (this file) |

## 2. Project structure

### 2.1 Five-layer architecture

```
Layer 1: Notation (nabla-macros)     — proc macros for concise DSL syntax
Layer 2: Compute  (nabla-core)       — tensor types + GPU/CPU backends + storage
Layer 3: Application (nabla-ml)      — dense LA + AD + CAS + ODE + IO
Layer 4: Training (nabla-train)      — optimizer + dataloader + trainer + checkpoint
Layer 5: Interface (nabla-interface) — GGUF export + llama.cpp FFI bridge
```

**Design principle**: Each layer depends only on layers below it. Layer 1 has zero runtime deps. Layer 2 is a pure computation engine. Layer 3 composes Layer 2 primitives into domain-specific APIs. Layer 4 adds training utilities without altering core math semantics.

### 2.2 Directory layout

```
nabla/                       [workspace root]
├── Cargo.toml               members: nabla-core, nabla-macros, nabla-ml, nabla-train, nabla-interface
├── .github/
│   └── workflows/
│       └── ci.yml           CI pipeline
├── .jj/                     jj metadata
│
├── nabla-macros/            ━━ Layer 1: Notation ━━
│   └── src/
│       ├── lib.rs           proc-macro entry: re-exports all macros
│       ├── macros/
│       │   ├── mod.rs       re-exports
│       │   ├── fuse.rs      fuse! / mega_fuse! / fuse_! codegen
│       │   ├── einsum.rs    einsum! parser + contraction path + codegen
│       │   ├── attrs.rs     nabla_grad / nabla_main attribute macros
│       │   ├── mat.rs       mat! / block! matrix literal parsing
│       │   ├── stencil.rs   stencil! offset indexing
│       │   ├── sym.rs       sym! Pratt-parser symbolic expression macro
│       │   ├── math.rs      math! auto-borrow expression transform
│       │   └── derive_module.rs  #[derive(Module)] proc macro
│       └── fusion/
│           ├── mod.rs       fuse! IR entry
│           ├── expr.rs      FuseExpr AST
│           ├── eqsat.rs     e-graph equality saturation
│           └── codegen.rs   CUDA/HIP C kernel string codegen
│
├── nabla-train/             ━━ Layer 4: Training ━━
│   └── src/
│       ├── lib.rs           train stack entry (optim/dataloader/trainer/ckpt)
│       ├── optim.rs         Adam/AdamW/SGD + LR schedule glue
│       ├── dataloader.rs    Dataset/Sampler/Batcher/DataLoader
│       ├── trainer.rs       training loop + hooks + AMP
│       ├── checkpoint.rs    save/load for params/optim/scaler
│       └── dist.rs          CPU-only allreduce (2-rank)
│
├── nabla-interface/         ━━ Layer 5: Interface ━━
│   ├── build.rs            llama.cpp pkg-config probe (feature = "llama")
│   └── src/
│       ├── lib.rs          crate entry: re-exports + Error/Result
│       ├── gguf.rs         GGUF v3 binary writer (GgufWriter, MetadataValue, TensorInfo)
│       ├── quant/          quantization packing/unpacking dispatch
│       │   ├── mod.rs      GgufQuantType enum (34 variants) + quantize/dequantize dispatch
│       │   ├── legacy.rs   Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 (QK=32)
│       │   └── kquant.rs   Q2_K/Q3_K/Q4_K/Q5_K/Q6_K (QK_K=256)
│       ├── convert.rs      state_dict → GGUF tensor mapping + export_gguf()
│       ├── llama.rs        llama.cpp FFI (unsafe extern "C", RAII wrappers) [feature = "llama"]
│       └── serve.rs        InferenceEngine + streaming generation [feature = "llama"]
│   └── tests/
│       ├── gguf_writer.rs      GGUF binary format tests (5)
│       ├── quant_roundtrip.rs  quantize/dequantize roundtrip tests (5)
│       ├── convert.rs          export pipeline tests (5)
│       ├── llama_load.rs       llama.cpp load test (#[ignore], needs GGUF file)
│       └── llama_inference.rs  llama.cpp generation test (#[ignore], needs GGUF file)
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
│           │   └── kernels/  CUDA/HIP C kernel sources (NVRTC/hiprtc JIT)
│           │       ├── mod.rs
│           │       ├── k_basic_core.cuh
│           │       ├── k_basic_math.cuh
│           │       ├── k_basic_red32.cuh
│           │       ├── k_basic_red64.cuh
│           │       ├── k_norm_softmax.cuh
│           │       ├── k_norm_group.cuh
│           │       ├── k_norm_reduce.cuh
│           │       ├── k_norm_pool.cuh
│           │       ├── k_conv_bn_loss.cuh
│           │       ├── k_attn.cuh
│           │       ├── k_conv_misc.cuh
│           │       ├── kernels_cond_set.cuh
│           │       ├── kernels_wmma_cuda.cuh
│           │       └── kernels_wmma_hip.cuh
│           ├── cuda/         CUDA backend (split by feature)
│           │   ├── mod.rs
│           │   ├── backend.rs
│           │   ├── core.rs
│           │   ├── conv_ops.rs
│           │   ├── fusion.rs
│           │   ├── graph_runtime.rs
│           │   ├── math_ops.rs
│           │   ├── norm_attn_ops.rs
│           │   ├── reduce_pool_ops.rs
│           │   └── training.rs
│           └── hip/          HIP backend (mirrors CUDA layout)
│               ├── mod.rs
│               ├── backend.rs
│               ├── core.rs
│               ├── conv_ops.rs
│               └── nn_ops.rs
│
├── nabla-ml/                ━━ Layer 3: Application ━━
│   ├── src/
│   │   ├── lib.rs           prelude + macro_rules re-exports (slim entry)
│   │   ├── autograd/        Reverse-mode AD: Tape + Variable + ops
│   │   │   ├── mod.rs
│   │   │   ├── core.rs
│   │   │   ├── ops.rs
│   │   │   └── tensor_like.rs
│   │   ├── cas/             Symbolic CAS: Expr tree + diff/simplify/eval (E-graph)
│   │   │   ├── mod.rs
│   │   │   ├── expr.rs
│   │   │   └── alg.rs
│   │   ├── linalg/          Dense LA: factorizations + solve + matrix functions
│   │   │   ├── mod.rs
│   │   │   ├── core.rs
│   │   │   ├── factor_lu_qr.rs
│   │   │   ├── factor_chol_svd.rs
│   │   │   ├── eigen.rs
│   │   │   ├── matrix_fn.rs
│   │   │   ├── equation.rs
│   │   │   ├── structured.rs
│   │   │   ├── solve_ext.rs
│   │   │   └── solve_types.rs
│   │   ├── module/          Module<T,B> trait + layers + IO helpers
│   │   │   ├── mod.rs
│   │   │   ├── core.rs
│   │   │   └── layers.rs
│   │   ├── ode/             ODE/SDE solvers
│   │   │   ├── mod.rs
│   │   │   ├── advanced.rs
│   │   │   ├── sde.rs
│   │   │   └── stiff.rs
│   │   └── misc/            cross-cutting utilities
│   │       ├── sparse.rs    SparseMatrix<T> CSC + BcsrMatrix<T> GPU sparse
│   │       └── surface.rs   constructors + notation macros surface
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
│   │   ├── 10_half_precision.rs
│   │   ├── bench_dispatch.rs
│   │   └── bench_gpu.rs
│   └── tests/
│       ├── boundary.rs      CPU boundary tests (231, CUDA-compatible)
│       ├── einsum_compile_errors.rs  trybuild harness
│       ├── gpu.rs           GPU backend tests (feature-gated)
│       └── einsum_errors/   trybuild compile-fail fixtures + .stderr
│           ├── duplicate_lhs_index.rs
│           ├── duplicate_lhs_index.stderr
│           ├── lone_contraction_index.rs
│           └── lone_contraction_index.stderr
├── scripts/                 benchmarks + standalone demos
│   ├── 01_solve.rs
│   ├── 02_svd.rs
│   ├── 03_ode.rs
│   ├── cas_simplify.rs
│   ├── linear_regression.rs
│   ├── ode_vanderpol.rs
│   ├── sparse_solve.rs
│   └── bench_pytorch.py
└── docs/
    ├── directory.md         directory structure reference
    ├── history.md           implementation history
    ├── notation.md          DSL notation & macro reference
    └── spec.md              full specification
```

Dependencies:
- `nabla-core`: `rayon 1`, `half 2` (optional), `wgpu 24` (optional), `pollster 0.4` (optional), `cudarc 0.19` (optional), `hip-runtime-sys 0.1` (optional)
- `nabla-macros`: `syn 2`, `quote 1`, `proc-macro2 1`, `egg 0.9`, `ordered-float 4`
- `nabla-ml`: `nabla-core`, `nabla-macros`, `half 2`, `egg 0.9`, `ordered-float 4`, `rayon 1`
- `nabla-train`: `nabla-ml`, `nabla-core`, `nabla-macros`
- `nabla-interface`: `nabla-core`, `nabla-ml` (package="nabla"), `half 2`, `thiserror 2`
- Dev: `trybuild 1` (compile-fail tests)

### 2.3 File size policy

**Target: 400-800 lines per file. Group by feature, not by responsibility.**

| Rule | Policy |
|---|---|
| Target range | **400-800 lines** per `.rs` file |
| Below 400 lines | Merge into the feature file it belongs to (do NOT create tiny single-purpose files) |
| Above 800 lines | Split by feature group (keep semantically related code together) |
| Module `mod.rs` | Thin re-exports + type aliases only; no logic |
| Grouping principle | Feature-cohesive. `conv.rs` = conv1d+2d+3d+transpose. `scalar.rs` = all scalar types. Never isolate a single type or single method into its own file |
| Max function length | 100 lines -> refactor into sub-functions |

**Anti-patterns (prohibited)**:
- `activations.rs` (75 lines) — too fine; merge into `nn.rs`
- `utils.rs` (45 lines) — too fine; inline into the calling module
- One type per file below 400 lines — merge into the feature file unless it exceeds 400 lines independently
