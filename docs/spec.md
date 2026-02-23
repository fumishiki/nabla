# nabla — Specification

## 1. Overview

Type-safe, zero-copy, compile-time backend-exclusive Rust linear algebra DSL.
Proc macros (`mat![]`, `bcast!{}`, `einsum!{}`) combined with self-contained pure-Rust kernels (CPU) and wgpu + WGSL compute shaders (GPU).
CPU and GPU are exclusively selected via feature flags at build time — no implicit CPU fallback.

### Fixed Rule Principle

nabla's scope is limited to **mathematically invariant rules**. User-customizable domains are never provided.

| Category | nabla provides (CPU/GPU) | User implements |
|---|---|---|
| Tensor ops | matmul, exp, sin, reduction, etc. | — |
| Autodiff | reverse-mode AD (chain rule) | — |
| CAS | diff, simplify, eval | — |
| ODE | euler, rk4, dormand_prince (Butcher tableau) | — |
| Optimizer | — | SGD, Adam, etc. |
| Loss function | — | MSE, cross-entropy, etc. |
| Model architecture | — | layers, forward pass |
| Training loop | — | epoch, batch, logging |

**Criterion**: "Will users need to customize this in the future?" → Yes: not provided. No (mathematically fixed): provided with CPU/GPU support.

Legend: ✅ Implemented | ❌ Not possible (language constraint)

---

## 2. Project structure

```
nabla/
├── Cargo.toml           [workspace] nabla + macros
├── macros/
│   ├── Cargo.toml       proc-macro crate (syn/quote/proc-macro2)
│   └── src/
│       ├── lib.rs       mat! + bcast_all! + named! + generated! + stencil!
│       ├── einsum.rs    einsum parser + codegen
│       └── stencil.rs   stencil! offset indexing
├── src/
│   ├── lib.rs           crate root + prelude + error + util macros
│   ├── tensor.rs        Tensor<T,B> + StaticMatrix<T,R,C> + NdTensor<T> + DynTensor
│   ├── backend.rs       Backend trait (sealed) + Cpu impl + CpuStorage
│   ├── gpu.rs           GpuStorage + wgpu WGSL compute shaders (32 ops)
│   ├── scalar.rs        Scalar trait + Complex<T> + MathOps/ReductionOps
│   ├── linalg.rs        9 dense factorizations + Diagonal/Symmetric/Triangular
│   ├── sparse.rs        SparseMatrix<T> CSC
│   ├── cas.rs           Symbolic CAS: Expr tree + diff/simplify/eval
│   ├── ode.rs           ODE solvers: euler/rk4/dormand_prince
│   └── autograd.rs      Reverse-mode AD: Tape + Variable + backward
├── tests/
│   ├── boundary.rs      CPU boundary tests
│   ├── gpu.rs            GPU backend tests (feature-gated)
│   └── einsum_errors/   trybuild compile-fail fixtures
└── docs/
    └── spec.md          this file
```

Dependencies: `rayon 1` / `nabla-macros` / `wgpu 24` (optional) / `pollster 0.4` (optional)

---

## 3. Feature matrix

### A. Core syntax

| ID | Feature | Syntax | nabla / Rust | Status |
|---|---|---|---|---|
| A1 | Implicit multiplication | `2x` | `2.0 * x` | ❌ Parser limitation |
| A2 | Chained comparison | `0 < x < 1` | `between!(0.0, x, 1.0)` | ✅ |
| A3 | Complex literals | `2 + 3i` | `c32(re, im)` / `c64(re, im)` | ✅ |
| A4 | Rational literals | `3//4` | — | ❌ `//` is comment |
| A5 | Range/Step | `0.0:0.1:1.0` | `frange!()` / `linspace()` | ✅ |
| A6 | Matrix literals | `[1 2; 3 4]` | `mat![[1.0, 2.0], [3.0, 4.0]]` | ✅ |
| A7 | Unicode identifiers | `α`, `β` | Rust native | ✅ |
| A7b | Unicode infix operators | `÷`, `∘` | — | ❌ Fixed ASCII ops only |
| A8 | Transpose / Adjoint | `A'` | `.t()` / `.adjoint()` | ✅ |
| A9 | Pipe | `\|>` | `pipe!(val, f, g)` | ✅ |
| A10 | Splatting | `f(args...)` | `splat!` macro | ✅ |
| A11 | Tuple destructuring | `a, b = f()` | `let (a, b) = f();` | ✅ native |
| A12 | Named Tuple | `(a=1, b=2.0)` | `named!` proc macro | ✅ |

### B. Broadcasting & vectorization

| ID | Feature | Syntax | Strategy | Status |
|---|---|---|---|---|
| B1 | Dot-Call | `f.(x, y)` | `bcast!` (CPU) / `bcast_all!` (GPU) | ✅ |
| B2 | `@.` All | `@. y = sin(x)^2` | `bcast_all!` proc macro — GPU kernel dispatch | ✅ |
| B3 | In-Place `.=` | `A .= B .* C` | `zip_map!(out, f, &a, &b)` | ✅ |

### C. Arrays & collections

| ID | Feature | Strategy | Status |
|---|---|---|---|
| C1 | Comprehensions | `.map().collect()` / `Tensor::from_fn(m, n, f)` | ✅ |
| C2 | Multi-dim indexing | `.slice(rows, cols)` / `.slice_rows()` / `.slice_cols()` | ✅ |
| C3 | `@view` | zero-copy slicing | ✅ |
| C4 | Static matrices | `StaticMatrix<T,R,C>` const generics (stack) | ✅ |

### D–E. Type system & metaprogramming

| ID | Feature | Strategy | Status |
|---|---|---|---|
| D1 | Multiple dispatch (closed) | `DynTensor` enum + `match` | ✅ |
| D2 | Abstract type hierarchy | `trait Matrix: Array` | ✅ |
| D4 | `@generated` | `generated!` proc macro | ✅ |
| E1 | AST Macro | `syn` + `quote!` proc macro | ✅ |
| D1(open)/D5/E2 | Open dispatch / Piracy / eval | — | ❌ Language constraint |

### F. Performance annotations

| ID | Feature | Strategy | Status |
|---|---|---|---|
| F1 | `@inbounds` | Iterator auto-elim (LLVM) | ✅ |
| F2 | `@simd` | LLVM auto-vectorization | ✅ |
| F3 | `@turbo` | `par_from_fn`/`par_map`/`par_bcast!` (rayon, CPU only) | ✅ |
| F4 | `@views` | zero-copy slicing | ✅ |

### G–H. Linear algebra & sparse (CPU only)

| ID | Feature | Strategy | Status |
|---|---|---|---|
| G1 | Structural matrices | `Diagonal` / `Symmetric` / `Triangular` | ✅ |
| G2 | Factorization | 9 types + `.solve()` | ✅ |
| G3 | In-place BLAS | `Tensor::matmul_into(&mut out, &a, &b)` | ✅ |
| H1 | Sparse | `SparseMatrix<T>` CSC + factorization + solve | ✅ |

### I. Symbolic & DSL layer

| ID | Feature | Strategy | Status |
|---|---|---|---|
| I1 | Symbolic CAS | `Expr` tree + `diff`/`simplify`/`eval`/`eval_tensor` | ✅ |
| I2 | ODE DSL | `euler`/`rk4`/`dormand_prince` | ✅ |
| I3 | einsum | `einsum!{}` proc macro (7 patterns, spanned errors) | ✅ |
| I3b | stencil/conv | `stencil!` proc macro (CPU only) | ✅ |
| I4 | GPU Kernels | wgpu + WGSL compute shaders (32 ops) | ✅ |
| I5 | Reverse-mode AD | tape-based autodiff (14 ops + backward) | ✅ |

### J. Backends (exclusive)

| Feature | Backend | Status | Constraint |
|---|---|---|---|
| `cpu` (default) | `Cpu` — pure Rust + rayon | ✅ | Exclusive with `gpu` |
| `gpu` | `Gpu` — wgpu + WGSL (f32 only) | ✅ 32 ops | Exclusive with `cpu` |

---

## 4. Design principles

1. **Zero-copy first** — ownership/borrowing, in-place `_into(out: &mut)` convention
2. **Macros = notation layer** — proc macros for concise syntax, type-safe Rust underneath
3. **trait = dispatch** — trait-based multiple dispatch
4. **Self-contained LA** — row-major CpuStorage, 9 dense factorizations, CSC sparse. Zero external LA deps
5. **Build-time exclusive backend** — `cpu`/`gpu` feature flags, exactly one active. `compile_error!` on multi-select. No implicit CPU fallback
6. **Fixed-rule principle** — only mathematically invariant rules. User-customizable domains excluded
7. **Adjoint ≠ Transpose** — correct complex LA semantics

---

## 5. Backend architecture

### 5.1 Exclusive backend selection

```rust
#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!("nabla: exactly one backend feature must be enabled (cpu / gpu)");
```

| Feature | `DefaultBackend` | `Tensor<f32>` storage |
|---|---|---|
| `cpu` (default) | `Cpu` | `CpuStorage<f32>` (row-major `Vec<T>`) |
| `gpu` | `Gpu` | `GpuStorage<f32>` (`wgpu::Buffer`) |

All tensors use `Tensor<T>` = `Tensor<T, DefaultBackend>`.

### 5.2 CPU fallback prohibition

| Prohibited pattern | Behavior |
|---|---|
| c32/c64 on GPU | **Compile error** (GPU Scalar = f32 only) |
| `bcast!` on GPU | **Compile error** (use `bcast_all!`) |
| `stencil!` on GPU | **Compile error** |
| `par_bcast!`/`par_from_fn`/`par_map` on GPU | **Compile error** (CPU only) |
| linalg/sparse on GPU | **Compile error** (module not exported) |

### 5.3 Macro GPU dispatch

| Macro | CPU | GPU | Strategy |
|---|---|---|---|
| `einsum!` | ✅ GEMM | ✅ GPU kernel | `matmul_into` (Backend dispatch) |
| `bcast_all!` | ✅ CPU chain | ✅ GPU kernel | tensor method chain |
| `bcast!` | ✅ `from_fn` | ❌ compile error | arbitrary closure |
| `stencil!` | ✅ `from_fn` | ❌ compile error | offset access |
| `par_*` | ✅ rayon | ❌ compile error | CPU only |
| `mat!`/`splat!`/`named!`/`generated!` | ✅ | ✅ | Compile-time expansion |

### 5.4 Module visibility

| Module | CPU | GPU | Notes |
|---|---|---|---|
| `tensor` / `backend` | ✅ | ✅ | |
| `gpu` | — | ✅ | GPU only |
| `linalg` / `sparse` | ✅ | ❌ | CPU only |
| `cas` / `ode` | ✅ | ✅ | Backend generic |
| `autograd` | ✅ | ✅ | Rc-based single-thread |

---

## 6. GPU implementation (wgpu + WGSL)

### 6.1 GpuContext (singleton)

```rust
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<HashMap<&'static str, ComputePipeline>>,
}

fn get_context() -> &'static GpuContext {
    static CTX: OnceLock<GpuContext> = OnceLock::new();
    CTX.get_or_init(|| pollster::block_on(init_gpu()))
}
```

### 6.2 GpuStorage

```rust
pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    buffer: wgpu::Buffer,
    host_cache: Mutex<Option<Vec<T>>>,
}
```

- Memory layout: row-major flat array
- `Send`/`Sync`: `unsafe impl` (wgpu::Buffer is Send+Sync)
- Readback: lazy `fill_cache()` via `map_async` + `device.poll(Maintain::Wait)`

```
Host                              Device (GPU)
────                              ──────────────
zeros/fill/identity
  └──── WGSL compute shader ──→ Buffer₁
                                      │
         chained ops (zero transfer)  │
                                      ▼
                                  Buffer₂
                                      │
  .get(r,c)                           │
  ┌──── map_async + poll ◄────────────┘
  ▼
host_cache (lazy, cached on first access)
```

### 6.3 WGSL compute shaders

All shaders embedded as `const &str` in gpu.rs. Workgroup size: 256.

| Category | Shader | Ops |
|---|---|---|
| Binary | `elementwise_binary` | add, sub, mul_elem, div_elem, scale |
| Unary | `elementwise_unary` | exp, ln, log1p, sin, cos, tanh, sqrt, abs, recip, erf, ceil, floor, round, powf, neg |
| Matmul | `matmul_tiled` | tiled matmul (TILE=16, shared memory) |
| Reduction | `reduction` | sum, max, min |
| Arg reduction | `reduction_arg` | argmax, argmin (u32 index) |
| Construction | `fill`, `identity` | zeros, fill, identity |
| Copy | `copy`, `transpose` | clone, transpose |

### 6.4 Type support

| Type | CPU | GPU | Notes |
|---|---|---|---|
| `f32` | ✅ | ✅ all 32 ops | Primary target |
| `f64` | ✅ | ❌ compile error | WGSL/Metal lacks f64 |
| `c32`/`c64` | ✅ | ❌ compile error | WGSL lacks Complex |

---

## 7. einsum! specification

### 7.1 Pattern classification

| Pattern | Math | Codegen |
|---|---|---|
| `c[i,j] = a[i,k] * b[k,j]` | C = AB | `matmul_into` (transpose-aware) |
| `y[i] = a[i,k] * x[k]` | y = Ax | `matmul_into` (Mx1) |
| `c[i,j] = a[i,j] * b[i,j]` | C = A ∘ B | `mul_elem` |
| `s = a[i,i]` | tr(A) | diagonal loop |
| `c[i,j] = a[i] * b[j]` | C = ab^T | `from_fn` |
| `c[b,i,j] = a[b,i,k] * m[b,k,j]` | batch matmul | batch loop + inner GEMM |
| General N-D | — | NdTensor + loop codegen |

Compile-time `classify()` auto-detects → optimized path for GEMM/GEMV/Hadamard.

### 7.2 N-D index classification

| Class | Definition | Example |
|---|---|---|
| **Batch** | Present in LHS and all RHS, not contracted | `b` |
| **Free** | Present in LHS, subset of RHS | `i`, `j` |
| **Contraction** | Absent from LHS (RHS only) | `k` |

### 7.3 Spanned errors

`syn::Error::new_spanned` points to the exact token span in the user's `einsum!` expression.

| Error | Span target |
|---|---|
| Unknown index | Ident span of the index |
| Lone contraction index | Ident span + help message |
| Duplicate LHS index | Second occurrence span |
| Duplicate RHS index (non-trace) | Ident span |

---

## 8. Known limitations

| Limitation | Cause | Mitigation |
|---|---|---|
| No GPU f64 | WGSL/Metal lacks f64 compute | Compile error on `gpu` feature |
| No GPU c32/c64 | WGSL lacks Complex | Excluded from GPU Scalar |
| No GPU linalg/sparse | CPU-only implementation | Module hidden on GPU |
| Tiled matmul TILE=16 | WGSL shared memory constraint | Sufficient for standard ops |
| `from_fn` requires host | Closures cannot run on GPU | Use `bcast_all!` for GPU |

---

## 9. Design decisions

| Decision | Rationale |
|---|---|
| wgpu direct (no CubeCL) | Fixed-rule principle: all 32 ops are standard math → no custom kernel DSL needed |
| 2 backends (cpu/gpu) | wgpu covers Vulkan/Metal/DX12 uniformly |
| Build-time exclusive backend | CPU fallback is a performance bug source |
| wgpu::Buffer-based | Chained ops eliminate host↔device transfer |
| TypeId dispatch | Backend trait sealed + `T: Scalar`. Avoids E0276 |
| GPU f32 only | Eliminates implicit f64/complex fallback |
| Embedded WGSL shaders | Minimizes file count (flat structure) |
| No bytemuck | Self-contained `scalar_to_bytes`/`bytes_to_scalar` |
| `Mutex<Option<Vec<T>>>` cache | Readback is expensive; lazy on get/set |
| pollster for sync | wgpu async → sync bridge |

---

## 10. Future work

| Item | Priority | Notes |
|---|---|---|
| bf16/f16 Scalar types | Medium | Requires Scalar trait extension |
| GPU linalg/sparse | Low | Possible but wgpu lacks solvers |
| Multi-GPU | Future | wgpu multi-adapter not mature |
