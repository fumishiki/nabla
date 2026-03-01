<p align="center">
  <img src="assets/nabla_og.PNG" alt="nabla — GPU math for Rust, no C++ required" width="720">
</p>

<p align="center">
  <a href="https://github.com/fumishiki/nabla/actions/workflows/ci.yml"><img src="https://github.com/fumishiki/nabla/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.88%2B-orange.svg" alt="Rust 1.88+"></a>
</p>

# ∇ nabla

**PyTorch-familiar GPU math for Rust — one pure-Rust ecosystem, zero C++ dependencies.**

**Who is this for**: Researchers and engineers doing ML training, scientific simulation, or numerical computing in Rust who want native GPU acceleration *without* wrestling with `libtorch` or C++ FFI. nabla provides `loss.backward()`, `einsum!`, `fuse!`, Symbolic CAS, ODE solvers, and GGUF export across 4 hardware backends in a tightly integrated suite.

<p align="center">
  <img src="assets/demo_quickstart.gif" alt="nabla demo — solve, SVD, einsum, autodiff, CAS in 30 lines" width="800">
</p>

## The Five-Layer Ecosystem

nabla is more than a tensor library—it's a comprehensive next-generation computation engine.

1. **Layer 1: Notation** (`nabla-macros`) — Python/Julia-level ergonomics via `einsum!`, `fuse!`, `math!`, and `sym!`. 
2. **Layer 2: Compute** (`nabla-core`) — 190+ tensor operations running on CPU, wgpu (Vulkan/Metal), CUDA, and HIP. No implicit CPU fallback; performance is strictly predictable.
3. **Layer 3: Application** (`nabla-ml`) — Reverse/Forward AutoDiff, E-graph based Symbolic CAS, dense/sparse linear algebra, and ODE/SDE solvers.
4. **Layer 4: Training** (`nabla-train`) — Optimizers (AdamW/SGD), LR Schedules, DataLoaders, and AMP for full neural network training.
5. **Layer 5: Interface** (`nabla-interface`) — 34-type GGUF v3 quantization and export + `llama.cpp` inference bridge.

## The pain nabla solves

Write clean math without sacrificing Rust's safety. Nabla turns runtime panics into compile-time checks and recoverable `Result`s.

| Operation | Python / PyTorch / SymPy | nabla (Rust) | Advantage |
|---|---|---|---|
| **GPU Fusion** | `torch.sin(x)**2` (Often 2 kernels) | `fuse!(x.sin().powf(2.0))` | **1 JIT GPU Kernel** via AST EqSat |
| **Solve Ax=b** | `np.linalg.solve(A, b)` | `a.solve(&b)?` | **No silent NaNs** via `Result` |
| **Autodiff** | `loss.backward()` | `loss.backward()?; let dw = w.grad()?;` | Explicit error handling, Zero GC |
| **Einsum** | `np.einsum('ik,kj->ij', a, b)` | `einsum!(c[i,j] = a[i,k] * b[k,j])` | Compile-time pattern matching |
| **SVD** | `np.linalg.svd(A)` | `a.svd()?` | Complete pure-Rust factorization |
| **Symbolic diff**| `diff(x**2 * sin(x), x)` | `diff(&sym!(x^2 * sin(x)), "x")` | Seamless CAS integration |

## Benchmark (GH200 480GB)

**Why is it faster than PyTorch?**
* **Zero FFI Overhead**: PyTorch routes through Python -> ATen C++ -> CUDA. nabla goes straight from native Rust to GPU shaders.
* **Mega-kernel Fusion**: The `fuse!` macro traces the AST and conditionally emits highly optimized NVRTC/hiprtc JIT kernels (e.g., fusing loss + sq + reduce into a single kernel).
* **Graph Replay**: Training networks bypass dispatch latency completely using CUDA Graphs.

**Reproduce locally**: `cd benchmarks && bash run.sh`

<p align="center">
  <img src="assets/demo_benchmark.gif" alt="nabla vs PyTorch benchmark" width="800">
</p>

### MLP training end-to-end — 784→256→128→10, SGD, f32

| Batch | nabla eager | nabla graph | PT eager | PT graph |
|------:|------------:|------------:|---------:|---------:|
| 1     | 0.111 ms    | **0.070 ms**| 0.710 ms | 0.045 ms |
| 128   | 0.133 ms    | **0.088 ms**| 0.976 ms | 0.130 ms |
| 1024  | 0.170 ms    | **0.130 ms**| 0.966 ms | 0.160 ms |

> *nabla eager is 4-6× faster than PyTorch eager. With CUDA Graph (default in `train_step!`), nabla overtakes PyTorch at batch sizes ≥ 128.*

## Install

Backends are **mutually exclusive** at build time to prevent accidental and slow CPU fallbacks. No CUDA or Vulkan SDKs are required to compile nabla (uses dynamic `libloading`).

```toml
[dependencies]
nabla = { git = "https://github.com/fumishiki/nabla", features = ["cpu"] }

# Or choose exactly ONE GPU backend (disabling default CPU):
# nabla = { ..., default-features = false, features = ["cuda"] }  # NVIDIA GPU
# nabla = { ..., default-features = false, features = ["wgpu"] }  # Vulkan/Metal/DX12
# nabla = { ..., default-features = false, features = ["hip"] }   # AMD GPU
```

## Quick Start Highlights

### 1. Unified Math & Linear Algebra
```rust
use nabla::prelude::*;

let a = mat![[1.0, 2.0], [3.0, 4.0]];
let b = mat![[5.0], [6.0]];

let x = a.solve(&b)?;                                      // Ax = b
let (u, s, vt) = a.svd()?;                                 // Pure Rust SVD
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);    // Einstein summation
```

### 2. AutoDiff & Deep Learning
Build models with PyTorch-like semantics, but with Rust's type logic.
```rust
// Reverse-mode autodiff with deterministic memory (No GC)
let tape = Tape::new();
let w = tape.var(mat![[1.0, 2.0]])?;
let loss = (&w * &x).norm_sq();
loss.backward()?;
let dw = w.grad()?;

// Full network training 
let mut model = sequential!(
    Linear::new(784, 128),
    relu(),
    Linear::new(128, 10)
);

let mut opt = AdamW::from_params(1e-3, &model.parameters());
let loss = train_step!(model, opt, tape, |x, out| {
    out.cross_entropy_indices(&targets)
})?;
```

### 3. Symbolic CAS & ODE Solvers
Don't switch to Python for SymPy or SciPy. Do it in the same engine.
```rust
use nabla::cas_prelude::*;

// Computer Algebra System via E-graphs
let f = sym!(x^2 * sin(x));
let df = simplify(&diff(&f, "x")); // x^2 * cos(x) + x * 2 * sin(x)

// ODE solvers (Euler, RK4, Dormand-Prince, BDF, Parareal)
let f_ode = |t, y| -0.5 * y;
let sol = dormand_prince(f_ode, &y0, (0.0, 10.0), &AdaptiveConfig::default())?;
```

### 4. GGUF Export & Quantization
Export your trained `nabla` models to `llama.cpp` using 34 distinct quantization formats.
```rust
use nabla_interface::{export_gguf, GgufQuantType};

// Quantize and export natively to GGUF
export_gguf(
    &model.state_dict(), 
    Path::new("model.gguf"), 
    GgufQuantType::Q4_K_M, 
    &config, 
    &overrides
)?;
```

## Exploring Further

Check out the detailed specs to see limits, performance bounds, and zero-GC principles:
- [Spec](docs-en/spec.md) — Architectural principles & limits
- [Notation](docs-en/notation.md) — DSL, API, and macro reference
- [Quick Start](docs-en/quick_start.md) — Full feature crash course
- [Directory](docs-en/directory.md) — 5-Layer codebase breakdown

**Examples**:
```bash
cargo run --example 01_matrix_ops --features cpu        # matrix ops + LU solve
cargo run --example 04_autograd_mlp --features cpu      # reverse-mode AD
cargo run --example 05_ode_lorenz --features cpu        # Lorenz attractor
cargo run --example 07_einsum_attention --features cpu  # einsum! SDPA
cargo run --example 08_cas_symbolic --features cpu      # symbolic diff
```

## Contributing
Fork → feature branch → `cargo test && cargo clippy && cargo fmt --check` → PR against `main`.

---

**fumishiki** — [GitHub](https://github.com/fumishiki) · [X](https://x.com/fumishiki) · [LinkedIn](https://linkedin.com/in/fumitakamurakami) · [Hugging Face](https://huggingface.co/fumishiki)

[Apache-2.0](LICENSE-APACHE) OR [MIT](LICENSE-MIT), at your option.
