# ∇ nabla

[![CI](https://github.com/fumishiki/nabla/actions/workflows/ci.yml/badge.svg)](https://github.com/fumishiki/nabla/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)

**Zero-GC, zero-copy computation engine for Rust** — not a framework. nabla provides every mathematically fixed computation primitive (matmul, conv, softmax, cross_entropy, …) optimized for CPU/GPU. Users compose these primitives into any architecture they want. Edge cases need minimal extension, never reimplementation.

Four backends (cpu / wgpu / cuda / hip), reverse-mode + forward-mode autodiff, symbolic CAS, ODE solvers. Zero external LA dependencies.

| Rust | nabla |
|---|---|
| `faer::linalg::matmul::matmul(&mut c, &a, &b, None, 1.0, Par::Seq)` | `&a * &b` |
| `let lu = a.partial_piv_lu();` `lu.solve_in_place(&mut b);` | `a.solve(&b)?` |
| nested `for i,j,k` loops — runtime shape bug possible | `einsum!(c[i,j]=a[i,k]*b[k,j])` — **compile error** at bad index |
| write CUDA C string → NVRTC compile → launch params | `fuse!(x.sin().powf(2.0); x)` — **1 kernel, 0 intermediates** |
| derive ∂L/∂w by hand, or add `tch-rs` (PyTorch C++ FFI) | `loss.backward()` — pure Rust, no FFI |

---

## Design philosophy

### Fixed-rule principle

nabla's scope is limited to **mathematically invariant rules** — operations whose correct behavior is fully determined by mathematics and will never need user customization.

> **Criterion**: "Will users need to customize this in the future?"
> - **Yes** → not provided (optimizer, loss, architecture, training loop)
> - **No** (mathematically fixed) → provided with CPU/GPU support

| ✅ nabla provides | ❌ User implements | Why the boundary |
|---|---|---|
| Tensor ops (matmul, exp, sin, reduction…) | — | Mathematically fixed |
| Autodiff (reverse + forward + GPU-resident) | — | Chain rule is fixed |
| Symbolic CAS (diff / simplify / eval) | — | Differentiation rules are fixed |
| ODE solvers (Euler, RK4, Dormand-Prince, BDF-1…) | — | Butcher tableaux are fixed |
| GPU sparse (BCSR + WGSL SpMM) | — | SpMM algorithm is fixed |
| — | Optimizer (SGD, Adam, LAMB…) | Update rule is user-defined |
| — | Loss function (MSE, cross-entropy…) | Task-specific |
| — | Model architecture (layers, forward pass) | Domain-specific |
| — | Training loop (epoch, batch, logging) | Workflow-specific |

### Design principles

1. **Zero-GC, zero-copy** — `Drop` = deterministic deallocation. `&` = zero-copy borrow. `_into(out: &mut)` = zero-allocation in-place. No reference counting in the hot path.
2. **Python's ease, C's speed** — PyTorch-familiar API (`loss.backward()`, `.exp()`, `.sum()`) at native Rust speed. Macro layer absorbs the syntax gap vs Julia.
3. **Macros = notation layer** — proc macros (`einsum!`, `fuse!`, `stencil!`) provide concise math notation; type-safe Rust underneath. No runtime overhead.
4. **Build-time exclusive backend** — backend features (`cpu`/`wgpu`/`cuda`/`hip`) are mutually exclusive (`compile_error!` on multi-select). No implicit CPU fallback.
5. **Self-contained LA** — zero external LA deps (no LAPACK, no BLAS, no C++ wrappers). Row-major `CpuStorage`, 9 dense factorizations, CSC sparse — all pure Rust.
6. **Two kernel codebases, not one abstraction** — WGSL (wgpu) + CUDA/HIP shared C source. 190+ fixed ops — dual maintenance is manageable and avoids abstraction overhead (no CubeCL/Triton).
7. **Errors read like math** — shape mismatch says `nabla: matmul 3×2 · 4×2` not `type parameter mismatch`. `einsum!` compile errors point to the exact index character.
8. **Adjoint ≠ Transpose** — `.t()` = transpose, `.h()`/`.adjoint()` = conjugate transpose. Correct complex LA semantics enforced by distinct methods.

---

## Why nabla over raw Rust

Without nabla, GPU linear algebra in Rust means wiring `libcuda` manually, writing C kernel strings, calling cuBLAS through bindgen FFI, and deriving gradients by hand. nabla gives you the same Rust safety and speed with math-notation ergonomics — one consistent API across CPU, Vulkan, CUDA, and AMD.

| Rust | nabla |
|---|---|
| `faer::linalg::matmul::matmul(&mut c, &a, &b, None, 1.0, Par::Seq)` | `&a * &b` |
| `let lu = a.partial_piv_lu(); lu.solve_in_place(&mut b);` | `a.solve(&b)?` |
| nested `for i,j,k` loops — runtime shape bug possible | `einsum!(c[i,j]=a[i,k]*b[k,j])` — **compile error** at bad index |
| write CUDA C string → NVRTC compile → launch params | `fuse!(x.sin().powf(2.0); x)` — **1 kernel, 0 intermediates** |
| derive ∂L/∂w by hand, or add `tch-rs` (PyTorch C++ FFI) | `loss.backward()` — pure Rust, no FFI |

**What nabla does NOT replace** — and is right not to: optimizer update rules, loss function choice, model architecture, training loop. Those are user-defined logic, not fixed mathematics.

**Honest friction** — Rust language constraints, not nabla choices:
- `&a * &b` not `a * b` for re-use — explicit zero-copy borrow semantics
- `a.solve(&b)?` not `A \ b` — `?` propagates `Result`, no silent NaN
- `a.emul(&b)` not `a * b` — `*` is matmul; Hadamard needs a distinct name

---

## Features

| Category | What you get |
|---|---|
| **190+ tensor ops** | matmul, conv1d/conv2d/conv3d/conv\_transpose2d, element-wise (exp/sin/cos/tanh/sqrt/…), reductions (sum/max/min/var/std/argmax/argmin/…), activations (relu/gelu/silu/mish/elu/…), normalization (layer\_norm/rms\_norm/batch\_norm/group\_norm), loss (cross\_entropy/mse/bce/kl\_div/…), pooling (max/avg/adaptive), attention (SDPA/MHA/embedding), manipulation (reshape/permute/cat/pad/gather/scatter/topk/sort/…), random (rand/randn), dropout, interpolate (nearest/bilinear) |
| **13 macros** | `mat!` `einsum!` `fuse!` `map!` `map_!` `par_map!` `stencil!` `named!` `generated!` `splat!` `pipe!` `between!` `frange!` |
| **9 dense factorizations** | LU, QR, Cholesky (LLT), LDL, SVD, Eigendecomposition + structural views (Diagonal, Symmetric, Triangular) |
| **Sparse LA** | CSC (CPU) + `BcsrMatrix` BCSR (GPU WGSL SpMM) + mixed-precision refinement |
| **Symbolic CAS** | `Expr` tree — `diff` / `simplify` (E-graph 32 rules) / `eval` / `eval_tensor` |
| **ODE solvers** | Euler, RK4, Dormand-Prince (adaptive), BDF-1 (stiff), IF-Euler (stiff explicit), METD (matrix exponential), Störmer-Verlet (symplectic), Parareal (parallel-in-time, rayon) |
| **Autodiff** | Reverse-mode tape (`Tape::new`, `.backward()`, `.grad()`), Forward-mode `Dual<T>` + `MultiDual<T,N>`, `#[nabla_grad]` source transform, `GpuTape` GPU-resident AD |
| **Named axes** | `Tensor<T,B,Axes>` — compile-time axis mismatch errors |
| **4 backends** | `cpu` (faer + rayon) · `wgpu` (WGSL, cross-platform) · `cuda` (NVRTC JIT) · `hip` (hiprtc JIT) |
| **No external LA deps** | Pure Rust — no LAPACK, no BLAS, no foreign bindings |

231 boundary tests · MSRV 1.88 (edition 2024) · Apache-2.0 OR MIT

---

## Benchmark — nabla vs PyTorch (GH200 480GB)

4096×4096 f32, 100 iterations, CUDA 12.8, PyTorch 2.7.0.

| Operation | nabla | PyTorch | Winner |
|---|---|---|---|
| matmul 1024² | **0.029 ms** | 0.057 ms | **nabla 2.0×** |
| fuse exp+sin | **0.046 ms** | 0.056 ms | **nabla 1.2×** |
| exp | 0.046 ms | 0.040 ms | PyTorch 1.15× |
| sin / cos / tanh | 0.046 ms | 0.041 ms | PyTorch 1.12× |
| add / sub / emul | 0.063 ms | 0.058 ms | PyTorch 1.09× |
| sum\_all | 0.028 ms | 0.026 ms | PyTorch 1.08× |
| max\_all | 0.029 ms | 0.026 ms | PyTorch 1.12× |

**Key results**: matmul 2× faster (cuBLAS TF32), fused kernels 1.2× faster (zero intermediates), reductions within 8–12% of CUB DeviceReduce. Element-wise ops within 9–15% of PyTorch.

---

## Install

```toml
[dependencies]
# CPU (default — faer + rayon, f32/f64/c32/c64)
nabla = { git = "https://github.com/fumishiki/nabla", features = ["cpu"] }

# wgpu GPU — Vulkan / Metal / DX12 (f32 only)
nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["wgpu"] }

# CUDA GPU — NVRTC JIT, no SDK at build time (f32 + f64)
nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["cuda"] }

# AMD HIP GPU — hiprtc JIT (f32 + f64)
nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["hip"] }

# Backend features are mutually exclusive: choose exactly one of cpu/wgpu/cuda/hip.
```

Backend features are mutually exclusive (`cpu`/`wgpu`/`cuda`/`hip`) at build time:

```rust
// features = ["cuda"]  (single backend per build)
let a: Tensor<f32> = zeros(1024, 1024);  // DefaultBackend = Cuda
let b: Tensor<f32> = zeros(1024, 1024);
let c = &a * &b;
```

---

## Quick start

```rust
use nabla::prelude::*;

// --- Construction ---
let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
let z  = zeros::<f64>(4, 4);
let id = eye::<f64>(4);
let r  = randn::<f32>(100, 100);

// --- Arithmetic ---
let c = &a * &b;              // matmul (owned: a * b, borrowed: &a * &b)
let h = a.emul(&b);           // Hadamard product (A .* B)
let s = &a * 2.0_f64;         // scalar multiply

// --- Indexing (zero-copy) ---
let v  = a[(1, 2)];           // element read
a[(1, 2)] = 5.0;              // element write
let sub = a[(0..2, 1..3)];    // submatrix view

// --- Einstein summation ---
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);       // GEMM
let h: Tensor<f64> = einsum!(h[i,j] = a[i,j] * b[i,j]);       // Hadamard
let t: f64         = einsum!(t = a[i,i]);                       // trace
let batch: NdTensor<f64> = einsum!(c[b,i,j] = a[b,i,k] * m[b,k,j]); // batch GEMM

// --- Broadcasting ---
let y = fuse!(x.sin().powf(2.0); x);     // GPU kernel fusion (Julia @.)
let y = map!(|v| v.ln() + 1.0, &a);     // CPU closure broadcast
map_!(out, |v| v.tanh(), &a);            // in-place, zero alloc
let y = par_map!(|v| v.sqrt(), &a);      // parallel (rayon)

// --- Stencil ---
let lap = stencil!(lap[i,j] = -4.0*u[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]);
```

---

## Linear algebra

```rust
// --- Direct solve ---
let x  = a.solve(&b)?;       // Ax = b
let x  = a.lstsq(&b)?;       // least-squares
let ai = a.inv()?;            // A⁻¹

// --- Factorize once, solve many ---
let lu = a.lu()?;
let x1 = lu.solve(&b1);
let x2 = lu.solve(&b2);      // faer zero-copy reuse

// --- 9 factorizations ---
a.lu()?;    a.qr();    a.chol()?;    a.ldl()?;
a.svd()?;   a.svdvals()?;
a.sym(Side::Lower)?.eigh()?;         // symmetric eigendecomposition

// --- Structural views ---
let d = Diagonal::new(v);
let s = Symmetric::new(a, Side::Lower)?;
let t = Triangular::new(a, TriKind::Lower)?;
```

---

## Sparse

```rust
// CSC — CPU
let s = sparse(4, 4, &[(0,0,4.0), (0,1,-1.0), (1,0,-1.0), (1,1,4.0)])?;
let x = s.solve(&b)?;
let y = &s * &dense;          // SpMM via Mul trait

// BCSR — GPU (wgpu/cuda/hip)
let bs = BcsrMatrix::from_sparse(&s, 16)?;
let y  = bs.spmm(&x);

// Mixed-precision refinement (2–4x speedup)
let y = mixed_spmm_f64(&bs, &x);
```

---

## Symbolic CAS

```rust
use nabla::cas::*;

let x = Expr::var("x");
let f  = (x.clone() * x.clone()).sin();     // sin(x²)
let df = f.diff("x").simplify();            // 2x·cos(x²)  — E-graph 32 rules
let v  = df.eval(&[("x", 1.0)]);           // 2·cos(1)

// Multi-variable
let (fx, fy) = f.diff_multi(&["x", "y"]);
```

---

## ODE solvers

```rust
use nabla::ode::*;

let f = |t: f64, y: f64| -y;   // dy/dt = -y
let y0 = 1.0_f64;

// Fixed-step
let (ts, ys) = euler(f, y0, 0.0, 5.0, 0.01);
let (ts, ys) = rk4(f, y0, 0.0, 5.0, 0.01);

// Adaptive
let (ts, ys) = dormand_prince(f, y0, 0.0, 5.0, 1e-6);

// Stiff — implicit BDF-1
let cfg = Bdf1Config { tol: 1e-8, max_iter: 50 };
let (ts, ys) = bdf1(f, y0, 0.0, 5.0, 0.01, cfg);

// Stiff — IF-Euler (explicit, no Newton)
let (ts, ys) = if_euler_scalar(l, n, y0, 0.0, 5.0, 0.01, cfg);

// Parallel-in-time (rayon)
let cfg = PararealConfig { n_intervals: 8, max_iter: 5, tol: 1e-6 };
let ys = parareal_solve(0.0, 5.0, y0, cfg, coarse_fn, fine_fn);
```

---

## Autodiff

### Reverse-mode (PyTorch-familiar)

```rust
let tape = Tape::new();
let w = tape.var(&weights);
let x = tape.var(&input);

let h = fuse!(v.max(0.0); (&x * &w));   // ReLU, 1 GPU kernel
let loss = fuse!(v.powf(2.0); h).sum();
loss.backward();
let dw = w.grad();          // ∂loss/∂w — owned Tensor, freed on drop
// tape, grads, intermediates ALL freed when scope exits (no GC, no zero_grad())
```

### Forward-mode

```rust
// Dual<T> — drop-in via impl Scalar for Dual<T>
let x = Dual::new(2.0_f64, 1.0);   // value=2, seed=1
let y = (x * x).sin();              // y.value = sin(4), y.deriv = 4·cos(4)

// #[nabla_grad] — source transform, works on any scalar function
#[nabla_grad]
fn energy(x: f64) -> f64 { (x * x).sin() }
let (val, deriv) = energy_grad(2.0);

// MultiDual<T, N> — batch Jacobian (N lanes)
let x = MultiDual::<f64, 4>::seed(2.0, 0);
```

### GPU-resident AD

```rust
// GpuTape — keeps computation graph on device, no GPU↔CPU roundtrip
let tape = GpuTape::<f32>::new();
let xv = tape.var(&gpu_tensor);
let loss = (&xv * &xv).exp().sum();
tape.backward(&loss);
let dx = tape.grad(&xv);
```

---

## Backend system

Compile-time exclusive backend selection — `cpu`/`wgpu`/`cuda`/`hip` are mutually exclusive.

| Feature | Backend | Scalar types | Notes |
|---|---|---|---|
| `cpu` | faer + rayon | f16, bf16, f32, f64, c32, c64 | Default. All modules available |
| `wgpu` | WGSL (Vulkan / Metal / DX12) | f32 | No f64 (WGSL/Metal limitation). No c32/c64 |
| `cuda` | NVRTC JIT (no SDK at build time) | f32, f64 | WMMA tensor cores + warp shuffle. No c32/c64 |
| `hip` | hiprtc JIT | f32, f64 | WMMA intrinsics. No c32/c64 |

**Module availability by backend** — `linalg` and `sparse` (factorizations, CSC solve) are **CPU-only**. `cas`, `ode`, and `autograd` work on all backends.

```bash
cargo build                                              # CPU
cargo build --no-default-features --features wgpu       # wgpu (cross-platform GPU)
cargo build --no-default-features --features cuda        # NVIDIA
cargo build --no-default-features --features hip         # AMD
```

**Prohibited**: multiple backend features enabled simultaneously → `compile_error!`.

**CUDA/HIP runtime requirements** — no CUDA/HIP SDK needed at build time. Libraries are dynamically loaded at runtime via `libloading`:
- CUDA: `libcuda.so` + `libnvrtc.so` (NVIDIA driver ≥ Volta recommended for WMMA)
- HIP: `libamdhip64.so` + `libhiprtc.so` (CDNA2+ recommended for WMMA)

**Macro GPU dispatch** — macros that run on GPU vs CPU:

| Macro | GPU backends | CPU only | Notes |
|---|---|---|---|
| `einsum!` | ✅ | — | Compile-time codegen → GPU kernel |
| `fuse!` | ✅ | — | Fused op chain → 1 GPU kernel (no intermediates) |
| `stencil!` | — | ✅ | Offset-index bounds detection, CPU |
| `map!` / `map_!` | — | ✅ | Arbitrary closures can't run on GPU; use `fuse!` instead |
| `par_map!` | — | ✅ | Rayon parallel, CPU only |

### Code is backend-agnostic

```rust
// Same source, three targets — switch only the feature flag
let a: Tensor<f32> = zeros(1024, 1024);
let b: Tensor<f32> = eye(1024);
let c = &a * &b;     // CPU: faer BLAS | wgpu: WGSL tiled | cuda: WMMA | hip: WMMA
```

---

## Named axes

Eliminate transposition and broadcasting bugs at compile time:

```rust
let x = named_zeros!(batch: 32, seq: 128, dim: 512);
let w = named_zeros!(dim: 512, heads: 8);

// Compile error if axis names don't match
let y = einsum!(y[batch, seq, heads] = x[batch, seq, dim] * w[dim, heads]);
```

`Tensor<T, B, Axes=()>` — `PhantomData<fn() -> Axes>`, zero runtime cost, erased at codegen.

---

## Project structure

```
nabla/                       [workspace root]
├── Cargo.toml               members: nabla-core, nabla-macros, nabla
├── nabla-core/              foundation — tensor + backend
│   └── src/
│       ├── tensor.rs        Tensor<T,B> + StaticMatrix<T,R,C> + NdTensor<T> + DynTensor
│       ├── backend.rs       Backend trait (sealed) + Cpu impl
│       ├── scalar.rs        Scalar + Complex<T> + Dual<T> + f16/bf16
│       ├── gpu.rs           GpuStorage + wgpu dispatch
│       ├── cuda_backend.rs  CUDA NVRTC dispatch + cuBLAS helpers
│       ├── hip_backend.rs   HIP hiprtc dispatch + rocBLAS helpers
│       ├── kernels_cu.rs    CUDA/HIP C kernel source (100+ kernels, f32+f64)
│       ├── layout.rs        LinearLayout<N> F₂ swizzle
│       └── wgsl.rs          register-tile MMA codegen
├── nabla-macros/            proc-macro crate
│   └── src/
│       ├── lib.rs           mat! fuse! einsum! stencil! named! generated! nabla_grad!
│       ├── einsum.rs        7-pattern compile-time codegen
│       └── stencil.rs       offset-index bounds detection
├── nabla/                   facade — domain modules + macro_rules!
│   ├── src/
│   │   ├── linalg.rs        9 dense factorizations + structural views
│   │   ├── sparse.rs        SparseMatrix CSC + BcsrMatrix GPU
│   │   ├── autograd.rs      Tape (reverse) + GpuTape (GPU-resident)
│   │   ├── cas.rs           Expr tree + E-graph 32 rules
│   │   └── ode.rs           Euler/RK4/DP/BDF-1/IF-Euler/METD/Verlet/Parareal
│   ├── examples/            cargo run --example <name> (10 examples)
│   └── tests/               231 boundary + GPU + compile-fail tests
└── docs/spec.md             full specification
```

---

## Examples

```bash
cargo run --example 01_matrix_ops --features cpu        # matrix ops + LU solve
cargo run --example 02_least_squares --features cpu     # QR least-squares
cargo run --example 03_svd_compress --features cpu      # SVD low-rank compression
cargo run --example 04_autograd_mlp --features cpu      # reverse-mode AD
cargo run --example 05_ode_lorenz --features cpu        # Lorenz attractor (Dormand-Prince)
cargo run --example 06_sparse_solve --features cpu      # sparse Poisson FD
cargo run --example 07_einsum_attention --features cpu  # einsum! softmax attention
cargo run --example 08_cas_symbolic --features cpu      # symbolic diff + eval
cargo run --example 09_dae_pendulum --features cpu      # DAE pendulum
cargo run --example 10_half_precision --features cpu    # f16/bf16 arithmetic
```

---

## Building & testing

```bash
cargo test --features cpu                              # all CPU tests
cargo clippy --workspace --all-targets -- -D warnings  # lint
cargo fmt --all -- --check                             # format
cargo doc --workspace --no-deps                        # docs
```

MSRV: **1.88.0** (Rust edition 2024)

---

## Known limitations

| Limitation | Detail |
|---|---|
| No wgpu f64 | WGSL / Metal lacks f64 — use `cuda` or `hip` for f64 |
| No GPU complex | c32 / c64 unsupported on all GPU backends (compile error by design) |
| GPU linalg: TRSM only | `gpu_trsm_lower` available; full LU/Cholesky/QR: use a CPU backend build (`features = ["cpu"]`) |
| `from_fn` requires host | Closures cannot execute on GPU — use `fuse!` for GPU element-wise ops |
| L2/L3 `fuse!` GPU fusion | L1 element-wise fusion on CPU; GPU kernel fusion requires codegen extension |
| Tape AD overhead | `Tape` is Rc + dynamic graph (CPU) — use `GpuTape` for GPU-resident AD |
| No REPL | Compile language — use `cargo watch -x run` or `rust-script` for quick experiments |

---

## Contributing

1. Fork → feature branch
2. `cargo test --features cpu && cargo clippy -- -D warnings && cargo fmt --check`
3. PR against `main`

See [docs/spec.md](docs/spec.md) for the full specification.

---

## License

[Apache-2.0](LICENSE-APACHE) OR [MIT](LICENSE-MIT), at your option.
