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
├── Cargo.toml               members: nabla-core, nabla-macros, nabla-ml, nabla-train, nabla-interface, benchmarks, nabla-cli
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
│       │   ├── fuse.rs      fuse! / mega_fuse! / fuse_! codegen (484L)
│       │   ├── einsum.rs    einsum! parser + contraction path + codegen (797L)
│       │   ├── attrs.rs     nabla_grad / nabla_main attribute macros (52L)
│       │   ├── mat.rs       mat! / block! matrix literal parsing (299L)
│       │   ├── stencil.rs   stencil! offset indexing (269L)
│       │   ├── sym.rs       sym! Pratt-parser symbolic expression macro (281L)
│       │   ├── math.rs      math! auto-borrow expression transform (79L)
│       │   └── derive_module.rs  #[derive(Module)] proc macro (169L)
│       └── fusion/
│           ├── mod.rs       fuse! IR entry
│           ├── expr.rs      FuseExpr AST
│           ├── eqsat.rs     e-graph equality saturation
│           └── codegen.rs   CUDA/HIP C kernel string codegen
│
├── nabla-train/             ━━ Layer 4: Training ━━
│   ├── src/
│   │   ├── lib.rs           train stack entry (150L)
│   │   ├── optim/           optimizer subsystem
│   │   │   ├── mod.rs       re-exports (12L)
│   │   │   ├── core.rs      Optimizer trait + LR schedulers (413L)
│   │   │   ├── alg.rs       Adam/AdamW/SGD/LAMB algorithms (671L)
│   │   │   └── groups.rs    parameter groups + weight decay (580L)
│   │   ├── dataloader.rs    Dataset/Sampler/Batcher/DataLoader (338L)
│   │   ├── trainer.rs       training loop + hooks + AMP (523L)
│   │   ├── checkpoint.rs    save/load for params/optim/scaler (231L)
│   │   ├── dist.rs          CPU-only allreduce (44L)
│   │   ├── benchmark.rs     training benchmark utilities (262L)
│   │   ├── metrics.rs       training metrics collection (160L)
│   │   ├── profiler.rs      CUDA profiler integration (471L)
│   │   ├── quantize.rs      quantization-aware training (403L)
│   │   ├── gguf.rs          GGUF model export from training (1170L)
│   │   └── onnx.rs          ONNX model export (930L)
│   └── tests/
│       ├── data.rs          dataloader tests (120L)
│       ├── optim.rs         optimizer tests (165L)
│       └── trainer.rs       trainer integration tests (305L)
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
│       ├── imatrix.rs      importance matrix calibration (89L)
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
│       │   ├── backend/      Backend trait module (directory module)
│       │   │   ├── mod.rs    Backend trait (sealed) + DefaultBackend + NablaError + Result (940L)
│       │   │   └── nn.rs     NN-specific backend trait methods (453L)
│       │   ├── layout.rs     LinearLayout<N> F₂ binary matrix swizzle (158L)
│       │   ├── norm_attn_ops.rs  shared norm/attention op implementations (497L)
│       │   ├── scalar/
│       │   │   ├── mod.rs    Scalar/MathOps/ReductionOps traits + f32/f64/half impls (345L)
│       │   │   ├── complex.rs  Complex<T>, c32/c64 (342L)
│       │   │   ├── dual.rs   Dual<T> forward-mode AD (442L)
│       │   │   ├── multi_dual.rs  MultiDual<T,N> (353L)
│       │   │   └── lowp.rs   low-precision scalar types (160L)
│       │   └── tensor/
│       │       ├── mod.rs    Tensor<T,B> core struct + MatrixLike + trait impls (576L)
│       │       ├── constructors.rs  zeros/ones/identity/rand/from_fn/fill/linspace (456L)
│       │       ├── ops.rs    arithmetic + element-wise + broadcast + operator overloads (789L)
│       │       ├── shape.rs  shape/reshape/broadcast/TensorView/iter/display (640L)
│       │       ├── reductions.rs  sum/max/min/argmax/argmin/norm axis-wise (510L)
│       │       ├── variants.rs  NdTensor/StaticMatrix/DynTensor (748L)
│       │       ├── nn_conv.rs   conv1d/2d/3d/transpose + pooling + config builders (596L)
│       │       ├── nn_ops.rs    activations + batch_norm + losses + attention + inplace (604L)
│       │       └── lowp.rs     low-precision tensor operations (109L)
│       ├── cpu/
│       │   └── mod.rs        CpuStorage<T> + Cpu struct + impl Backend for Cpu (1040L)
│       ├── wgpu/             wgpu backend (feature = "wgpu")
│       │   ├── mod.rs        re-exports
│       │   ├── storage.rs    GpuStorage<T> + GpuContext + wgpu dispatch helpers (345L)
│       │   ├── shaders.rs    WGSL register-tile MMA codegen + kernel strings (1116L)
│       │   └── ops.rs        impl Backend for Gpu (wgpu) + all gpu_* fns (4694L)
│       └── cuda_hip/         CUDA + HIP backends (feature = "cuda" | "hip")
│           ├── mod.rs        re-exports common/*
│           ├── common/       shared code (both CUDA and HIP)
│           │   ├── mod.rs    re-exports rtc/pool/fuse
│           │   ├── rtc.rs    RtcStorage + MemoryPool + rtc_backend_impl! macro (549L)
│           │   ├── pool.rs   CUDA/HIP pooling kernels dispatch (786L)
│           │   ├── fuse.rs   fuse! kernel codegen dispatch (482L)
│           │   └── kernels/  CUDA/HIP C kernel sources (NVRTC/hiprtc JIT)
│           │       ├── mod.rs
│           │       ├── k_defs.cuh       type defs + common helpers (806L)
│           │       ├── k_ops.cuh        element-wise + math + reduce ops (762L)
│           │       ├── k_norm.cuh       normalization + attention kernels (441L)
│           │       ├── k_pool.cuh       pooling kernels (578L)
│           │       ├── k_conv.cuh       convolution + loss + QAT kernels (441L)
│           │       ├── k_indexing.cuh   gather/scatter/index kernels (210L)
│           │       ├── kernels_cond_set.cuh   conditional set kernel (68L)
│           │       ├── kernels_wmma_cuda.cuh  CUDA WMMA tensor core (78L)
│           │       └── kernels_wmma_hip.cuh   HIP WMMA tensor core (42L)
│           ├── cuda/         CUDA backend (split by feature)
│           │   ├── mod.rs           re-exports (46L)
│           │   ├── backend.rs       CudaBackend impl + device mgmt (769L)
│           │   ├── core.rs          storage + alloc + memcpy + launch (1044L)
│           │   ├── blas_ops.rs      cuBLAS/cublasLt GEMM + FP8 (824L)
│           │   ├── ops.rs           element-wise + math Backend ops (781L)
│           │   ├── reduce.rs        reductions + pooling dispatch (917L)
│           │   ├── fusion.rs        fuse! runtime dispatch (307L)
│           │   ├── graph.rs         NablaGraph + CUDA graph capture (799L)
│           │   ├── graph_compile.rs GraphCompiler optimization passes (478L)
│           │   ├── norm_attn_ops.rs norm + attention ops (303L)
│           │   ├── indexing_ops.rs  gather/scatter/index ops (494L)
│           │   └── training.rs      AMP + gradient scaling + train ops (629L)
│           └── hip/          HIP backend (mirrors CUDA layout)
│               ├── mod.rs       re-exports (11L)
│               ├── backend.rs   HipBackend impl (927L)
│               ├── core.rs      HIP storage + alloc (650L)
│               ├── conv_ops.rs  convolution ops (417L)
│               └── nn_ops.rs    NN ops dispatch (772L)
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
│   │   ├── 00_demo.rs
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
│       ├── ops_tensor_basic.rs      tensor ops + constructors (559L)
│       ├── ops_tensor_extended.rs   advanced tensor ops (507L)
│       ├── ops_autograd_linalg.rs   autograd + linalg tests (492L)
│       ├── ops_cas_ode.rs           CAS + ODE tests (444L)
│       ├── ops_nn.rs                NN layer tests (601L)
│       ├── ops_random_gpu.rs        random + GPU tests (494L)
│       ├── macros.rs                macro compile tests (441L)
│       ├── gpu_fp16.rs              FP16 GPU tests (242L)
│       ├── gpu_fp8.rs               FP8 GPU tests (256L)
│       ├── gpu_fp4.rs               FP4 GPU tests (245L)
│       ├── gpu_fp_cast.rs           FP cast tests (46L)
│       ├── test_wgpu_all_features.rs      wgpu feature tests (32L)
│       ├── test_wgpu_all_features_fixed.rs  wgpu fixed tests (52L)
│       ├── wgpu_quick_test.rs       wgpu smoke test (50L)
│       └── einsum_errors/   trybuild compile-fail fixtures + .stderr
│           ├── duplicate_lhs_index.rs
│           ├── duplicate_lhs_index.stderr
│           ├── lone_contraction_index.rs
│           └── lone_contraction_index.stderr
├── nabla-cli/               ━━ CLI binary ━━
│   └── src/
│       ├── main.rs          CLI entry + clap dispatch (62L)
│       ├── bench.rs         benchmark subcommand (278L)
│       ├── export.rs        model export subcommand (215L)
│       ├── info.rs          system info subcommand (290L)
│       ├── inspect.rs       model inspection subcommand (209L)
│       ├── run.rs           run/inference subcommand (107L)
│       └── tty.rs           terminal utilities (105L)
│
├── benchmarks/              ━━ Benchmark suite ━━
│   └── src/
│       ├── lib.rs                  benchmark entry (61L)
│       ├── bench_dispatch.rs       dispatch latency benchmark (170L)
│       ├── bench_dispatch_scaling.rs  scaling benchmark (143L)
│       ├── bench_ops.rs            op throughput benchmark (91L)
│       ├── diag_capture.rs         diagnostic capture (225L)
│       ├── profile_train.rs        training profiler (106L)
│       ├── profile_train_graph.rs  graph-mode training profiler (136L)
│       └── verify_train.rs         training verification (232L)
│
├── scripts/                 standalone demos
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
    ├── notation.md          DSL notation & macro reference
    ├── quick_start.md       tutorial and code examples
    ├── spec.md              full specification
    └── specs/               detailed design specifications
        ├── overview.md
        ├── requirements.md
        ├── interfaces.md
        ├── test-plan.md
        └── traceability.md
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
