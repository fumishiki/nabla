# nabla

> Inference-specialized Rust linear algebra DSL backed by faer 0.24

nabla provides a macro notation layer (`mat!`, `einsum!`, `bcast!`) over faer's high-performance SIMD kernels, delivering concise mathematical syntax with zero-copy semantics and pluggable GPU backends.

~2,500 lines of src across 5 files. Zero runtime overhead over faer.

---

## Table of contents

- [Why nabla?](#why-nabla)
- [Julia → nabla conversion guide](#julia--nabla-conversion-guide)
- [Install](#install)
- [Quick start](#quick-start)
- [Usage scenarios](#usage-scenarios)
- [API reference](#api-reference)
- [Macros](#macros)
- [Backend system](#backend-system)
- [Project structure](#project-structure)
- [Error handling](#error-handling)
- [Building and testing](#building-and-testing)
- [License](#license)

---

## Why nabla?

### The problem

Rust's linear algebra ecosystem offers powerful low-level primitives (faer, nalgebra, ndarray), but writing mathematical computation remains verbose:

```rust
// faer raw API — a simple matmul requires 6 arguments + trait imports
faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, T::one_impl(), Par::Seq);

// scalar multiply needs a wrapper type
let scaled = mat * faer::Scale(2.0);

// no matrix literals, no einsum, no broadcasting macros
```

For inference-heavy workloads, this verbosity compounds into thousands of lines of boilerplate that obscure the underlying math.

### The solution

nabla wraps faer behind a thin DSL layer:

```rust
use nabla::prelude::*;

let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];

let c = &a * &b;              // matmul — operator syntax
let s = &a * 2.0_f64;         // scalar mul — no wrapper needed
let x = a.solve(&b)?;         // linear solve — one method call
let e: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);  // Einstein notation
```

### Who is it for?

| Audience | Pain point nabla solves |
|---|---|
| **ML inference developers** | Forward-pass computation in Rust without training/autodiff overhead. Concise LA syntax for matrix ops, factorizations, and batched transforms. |
| **Numerical computing practitioners** | Porting math-heavy algorithms from Python/Julia to Rust. `einsum!` and `bcast!` replace NumPy/Julia idioms directly. |
| **Embedded / edge deployment** | Predictable CPU performance with compile-time backend selection. No runtime dispatch, no GC, no dynamic linking. GPU opt-in via feature flags. |
| **Research prototypers** | Quick matrix experiments in Rust with `mat!` literals and operator overloads. Iterate as fast as a scripting language, deploy with systems-language guarantees. |

### What nabla is NOT

- **Not a replacement for faer** — nabla is a DSL layer on top. All heavy computation delegates to faer's SIMD kernels.
- **Not a training framework** — no autodiff, no backpropagation, no optimizer. Use Burn or Candle for training.
- **Not a full NumPy clone** — focused on 2-D linear algebra for inference. No N-D tensor, no dataframe, no plotting.

---

## Julia → nabla conversion guide

**Rule: Rust で数学的行列操作をするときは常に nabla を使用すること。** faer の raw API を直接呼ばず、nabla の DSL 経由で記述する。

### Core syntax

| Julia | nabla / Rust | Notes |
|---|---|---|
| `[1 2; 3 4]` | `mat![[1.0, 2.0], [3.0, 4.0]]` | proc macro |
| `zeros(m, n)` | `Tensor::zeros(m, n)` | |
| `ones(m, n)` | `Tensor::from_fn(m, n, \|_,_\| 1.0)` | |
| `I(n)` / `Matrix{Float64}(I, n, n)` | `Tensor::identity(n)` | |
| `rand(m, n)` | `Tensor::from_fn(m, n, \|_,_\| rng.gen())` | bring your own RNG |
| `A[i, j]` | `a.get(i, j)` | 0-indexed |
| `A[i, j] = v` | `a.set(i, j, v)` | 0-indexed |
| `size(A)` | `a.shape()` → `(usize, usize)` | |
| `size(A, 1)` / `size(A, 2)` | `a.nrows()` / `a.ncols()` | |
| `copy(A)` | `a.clone()` | deep copy |
| `2 + 3im` | `c64(2.0, 3.0)` | `c32` for f32 |
| `0 < x < 1` | `between!(0.0, x, 1.0)` | macro |
| `0.0:0.1:1.0` | `frange!(0.0_f64, 0.1, 1.0)` | `Vec<f64>` |
| `range(0, 1, length=n)` | `linspace(0.0, 1.0, n)` | `Vec<f64>` |
| `let α = 3.14` | `let α = 3.14_f64;` | Rust native Unicode ident |
| `a, b = f()` | `let (a, b) = f();` | native |
| `SMatrix{3,3}(...)` | `StaticMatrix::<f64, 3, 3>::from_fn(\|r,c\| ...)` | stack-allocated |

### Arithmetic & views

| Julia | nabla / Rust | Notes |
|---|---|---|
| `A + B` | `&a + &b` | ref-based operators |
| `A - B` | `&a - &b` | |
| `-A` | `-&a` | |
| `A * B` | `&a * &b` | matmul |
| `A * s` / `s * A` | `&a * s` | scalar mul |
| `mul!(C, A, B)` | `Tensor::matmul_into(&mut c, &a, &b)` | zero-alloc |
| `transpose(A)` | `a.t()` | |
| `A'` / `adjoint(A)` | `a.adjoint()` | conj+transpose for complex |
| `A[1:3, 2:4]` | `a.slice(0..3, 1..4)` | 0-indexed, exclusive end |
| `A[1:3, :]` | `a.slice_rows(0..3)` | |
| `A[:, 2:4]` | `a.slice_cols(1..4)` | |
| `@view A[...]` | faer default (zero-copy) | automatic |

### Broadcasting

| Julia | nabla / Rust | Notes |
|---|---|---|
| `f.(A)` | `bcast!(\|x\| f(x), &a)` | allocating |
| `f.(A, B)` | `bcast!(\|x,y\| f(x,y), &a, &b)` | binary |
| `A .= f.(B)` | `zip_map!(a, \|x\| f(x), &b)` | in-place |
| `A .= f.(B, C)` | `zip_map!(a, \|x,y\| f(x,y), &b, &c)` | in-place binary |
| `@. y = sin(x)^2` | `bcast!(\|x\| x.sin().powi(2), &x)` | manual expansion |

### Linear algebra

| Julia | nabla / Rust | Notes |
|---|---|---|
| `A \ b` | `a.solve(&b)?` | LU internally |
| `A \ b` (overdetermined) | `a.solve_lstsq(&b)?` | thin SVD |
| `b / A` | `a.rsolve(&b)?` | |
| `inv(A)` | `a.inv()?` | |
| `lu(A)` | `a.factorize_lu()?` | partial pivot |
| `qr(A)` | `a.factorize_qr()` | |
| `cholesky(A)` | `a.factorize_llt()?` | LL^T |
| `ldlt(A)` | `a.factorize_ldlt()?` | LDL^T |
| `svd(A)` | `a.factorize_svd()?` | full USV^H |
| `svdvals(A)` | `a.singular_values()?` | values only |
| `eigen(Symmetric(A))` | `Symmetric::new(a, Side::Lower)?.eigen()?` | |
| `F.L`, `F.U` | `lu.reconstruct()` | |
| `F \ b` | `lu.solve(&b)` | reuse factorization |
| `Diagonal(v)` | `Diagonal::new(v)` | |
| `Symmetric(A, :L)` | `Symmetric::new(a, Side::Lower)?` | |
| `LowerTriangular(A)` | `Triangular::new(a, TriKind::Lower)?` | |

### Sparse

| Julia | nabla / Rust | Notes |
|---|---|---|
| `sparse(I, J, V, m, n)` | `SparseMatrix::try_new_from_triplets(m, n, &trips)?` | COO → CSC |
| `nnz(S)` | `s.nnz()` | |
| `size(S)` | `s.shape()` | |
| `S \ b` | `s.solve(&b)?` | sparse LU |
| `S \ b` (lstsq) | `s.solve_lstsq(&b)?` | sparse QR |
| `S * D` | `s.matmul_dense(&d)?` | sparse × dense |
| `cholesky(S)` | `s.cholesky_solve(side, &b)?` | one-shot |

### Einsum / Tullio

| Julia | nabla / Rust | Notes |
|---|---|---|
| `@tullio C[i,j] := A[i,k] * B[k,j]` | `einsum!(c[i,j] = a[i,k] * b[k,j])` | matmul |
| `@tullio C[i,j] := A[i,j] * B[i,j]` | `einsum!(c[i,j] = a[i,j] * b[i,j])` | Hadamard |
| `@tullio s := A[i,i]` | `einsum!(s = a[i,i])` | trace → scalar |
| `[x^2 for x in 1:10]` | `(1..=10).map(\|x\| x * x).collect::<Vec<_>>()` | native Rust |

### Performance annotations

| Julia | nabla / Rust | Notes |
|---|---|---|
| `@inbounds` | iterator auto-elim | LLVM optimizes |
| `@simd` | LLVM auto-vectorization + faer `pulp` | implicit |
| `@views` | faer default views | zero-copy automatic |

---

## Install

```toml
[dependencies]
nabla = { path = "../nabla" }
# GPU backends are opt-in:
# nabla = { path = "../nabla", features = ["cuda"] }
```

---

## Quick start

```rust
use nabla::prelude::*;

fn main() {
    // Matrix literals
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];

    // Arithmetic operators
    let sum    = &a + &b;          // element-wise add
    let diff   = &a - &b;          // element-wise sub
    let prod   = &a * &b;          // matrix multiply
    let scaled = &a * 2.0_f64;     // scalar multiply
    let neg    = -&a;              // negate

    // Construction helpers
    let zeros: Tensor<f64> = Tensor::zeros(3, 3);
    let eye:   Tensor<f64> = Tensor::identity(3);
    let built: Tensor<f64> = Tensor::from_fn(3, 3, |r, c| (r * 3 + c) as f64);

    // Views and slicing
    let at = a.t();                // transpose
    let ah = a.adjoint();          // adjoint (conj + transpose for complex)
    let sub = built.slice(0..2, 1..3);  // submatrix view

    // Linear solve
    let rhs: Tensor<f64> = mat![[10.0_f64], [12.0]];
    let x = a.solve(&rhs).expect("solve failed");

    let _ = (sum, diff, prod, scaled, neg, zeros, eye, at, ah, sub, x);
}
```

---

## Usage scenarios

### Inference pipeline

```rust
use nabla::prelude::*;

fn linear_layer(
    weights: &Tensor<f32>,
    bias: &Tensor<f32>,
    input: &Tensor<f32>,
) -> Tensor<f32> {
    let z = weights * input;   // matmul: W @ x
    &z + bias                  // add bias: z + b
}

fn softmax(logits: &Tensor<f64>) -> Tensor<f64> {
    let max_val = (0..logits.ncols())
        .map(|j| logits.get(0, j))
        .fold(f64::NEG_INFINITY, f64::max);

    let shifted: Tensor<f64> = bcast!(|x| (x - max_val).exp(), logits);
    let sum: f64 = (0..shifted.ncols()).map(|j| shifted.get(0, j)).sum();
    bcast!(|x| x / sum, &shifted)
}
```

### Solving linear systems

```rust
use nabla::prelude::*;

// Ax = b  where A is 3x3, b is 3x1
let a: Tensor<f64> = mat![
    [2.0_f64, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0]
];
let b: Tensor<f64> = mat![[8.0_f64], [-11.0], [-3.0]];

// Direct solve
let x = a.solve(&b).expect("system is singular");

// Explicit factorization for reuse across multiple RHS
let lu = a.factorize_lu().expect("LU failed");
let x1 = lu.solve(&b);
let x2 = lu.solve(&mat![[1.0_f64], [2.0], [3.0]]);
```

### Sparse FEM-style assembly

```rust
use nabla::prelude::*;
use nabla::sparse::{SparseMatrix, Triplet};

// Assemble stiffness matrix from element contributions
let mut triplets = Vec::new();
for elem in 0..100 {
    let (i, j) = (elem, elem + 1);
    triplets.push(Triplet { row: i, col: i, val:  2.0_f64 });
    triplets.push(Triplet { row: i, col: j, val: -1.0 });
    triplets.push(Triplet { row: j, col: i, val: -1.0 });
}
triplets.push(Triplet { row: 100, col: 100, val: 2.0 });

let k = SparseMatrix::try_new_from_triplets(101, 101, &triplets)
    .expect("invalid stiffness matrix");

let force: Tensor<f64> = Tensor::from_fn(101, 1, |i, _| if i == 50 { 1.0 } else { 0.0 });
let displacement = k.solve(&force).expect("singular stiffness matrix");
```

### Einsum for tensor contractions

```rust
use nabla::prelude::*;

let a: Tensor<f64> = Tensor::from_fn(4, 3, |r, c| (r * 3 + c) as f64);
let b: Tensor<f64> = Tensor::from_fn(3, 5, |r, c| (r * 5 + c) as f64);

// Matrix multiply via Einstein notation
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);
assert_eq!(c.shape(), (4, 5));

// Hadamard (element-wise product)
let sq: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
let id: Tensor<f64> = Tensor::identity(2);
let h: Tensor<f64> = einsum!(h[i,j] = sq[i,j] * id[i,j]);

// Trace (scalar reduction)
let tr: f64 = einsum!(tr = sq[i,i]);  // 1 + 4 = 5
```

---

## API reference

### Tensor — construction

| Operation | API |
|---|---|
| Zero matrix | `Tensor::zeros(m, n)` |
| Identity matrix | `Tensor::identity(n)` |
| Fill by closure | `Tensor::from_fn(m, n, \|r, c\| expr)` |
| Matrix literal | `mat![[r0c0, r0c1], [r1c0, r1c1]]` |
| Static matrix (stack) | `StaticMatrix::<f64, R, C>::zeros()` |
| Clone | `.clone()` |

### Tensor — accessors

| Operation | API |
|---|---|
| Shape | `.shape() -> (usize, usize)` |
| Row / col count | `.nrows()` / `.ncols()` |
| Read element | `.get(r, c) -> T` |
| Write element | `.set(r, c, val)` |
| Transpose | `.t()` |
| Adjoint | `.adjoint()` |
| Submatrix | `.slice(row_range, col_range)` |
| Row slice | `.slice_rows(range)` |
| Column slice | `.slice_cols(range)` |

### Tensor — arithmetic

| Operation | Syntax |
|---|---|
| Add | `&a + &b` |
| Sub | `&a - &b` |
| Negate | `-&a` |
| Matmul | `&a * &b` |
| Scalar mul | `&a * scalar` |
| In-place matmul | `Tensor::matmul_into(&mut out, &a, &b)` |

### Dense linear algebra — solve

| Method | Description |
|---|---|
| `.solve(&b)` | Solve Ax = b (LU internally) |
| `.rsolve(&b)` | Solve xA = b |
| `.solve_lstsq(&b)` | Least-squares (thin SVD) |
| `.solve_in_place(&mut b)` | Overwrite b with solution |
| `.inv()` | Matrix inverse |

### Dense linear algebra — factorizations

| Method | Return type | Notes |
|---|---|---|
| `.factorize_lu()` | `Result<PartialPivLu>` | PA = LU |
| `.factorize_full_piv_lu()` | `Result<FullPivLu>` | PAQ^T = LU |
| `.factorize_qr()` | `Qr` | A = QR |
| `.factorize_col_piv_qr()` | `ColPivQr` | AP^T = QR |
| `.factorize_llt()` | `Result<Llt>` | Cholesky LL^T |
| `.factorize_ldlt()` | `Result<Ldlt>` | LDL^T |
| `.factorize_lblt()` | `Lblt` | Bunch-Kaufman (infallible) |
| `.factorize_svd()` | `Result<Svd>` | Full SVD: A = USV^H |
| `.factorize_thin_svd()` | `Result<Svd>` | Compact U/V |
| `.singular_values()` | `Result<Vec<f64>>` | Singular values only |
| `.self_adjoint_eigen()` | `Result<SelfAdjointEigen>` | Symmetric / Hermitian |

All factorization objects share: `.solve(&b)`, `.solve_in_place(&mut b)`, `.solve_transpose(&b)`, `.solve_adjoint(&b)`, `.rsolve(&b)`, `.inverse()`, `.reconstruct()`. QR/ColPivQr/SVD also provide `.solve_lstsq(&b)`.

### Structural matrix types

| Type | Constructor | Key methods |
|---|---|---|
| `Diagonal<T>` | `Diagonal::new(vec)` | `.mul_dense(&t)`, `.to_tensor()` |
| `Symmetric<T>` | `Symmetric::new(t, Side::Lower)?` | `.eigenvalues()`, `.eigen()` |
| `Triangular<T>` | `Triangular::new(t, TriKind::Lower)?` | `.solve_in_place(&mut b)` |

### Sparse matrices

| Method | Description |
|---|---|
| `SparseMatrix::try_new_from_triplets(m, n, &[Triplet])` | Construct from COO triplets (duplicates summed) |
| `.shape()` | Matrix dimensions |
| `.nnz()` | Number of stored nonzeros |
| `.solve(&b)` | Sparse LU solve |
| `.solve_lstsq(&b)` | Sparse least-squares (QR) |
| `.matmul_dense(&dense)` | Sparse x dense matmul |
| `.cholesky_solve(side, &b)?` | Sparse Cholesky solve |
| `.symbolic_factorize_llt()` | Two-phase: symbolic step |
| `.numeric_factorize_llt(symbolic)` | Two-phase: numeric step |

---

## Macros

| Macro | Purpose | Example |
|---|---|---|
| `mat!` | Matrix literal | `mat![[1.0_f64, 2.0], [3.0, 4.0]]` |
| `einsum!` | Einstein summation | `einsum!(c[i,j] = a[i,k] * b[k,j])` |
| `bcast!` | Element-wise broadcast (allocating) | `bcast!(\|x\| x.exp(), &a)` |
| `zip_map!` | Element-wise broadcast (in-place) | `zip_map!(out, \|x, y\| x * y, &a, &b)` |
| `between!` | Chain comparison `lo <= x < hi` | `between!(0.0, x, 1.0)` |
| `frange!` | Float range with step | `frange!(0.0_f64, 0.25, 1.0)` |

### `mat!`

Proc macro — constructs a `Tensor` from nested bracket literals. Validates row/column structure at compile time.

```rust
let m: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
assert_eq!(m.shape(), (2, 3));
```

### `einsum!`

Proc macro — Einstein summation with compile-time index parsing and loop codegen.

```rust
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);  // matmul
let h: Tensor<f64> = einsum!(h[i,j] = a[i,j] * b[i,j]);  // Hadamard
let tr: f64        = einsum!(tr = a[i,i]);                  // trace
```

### `bcast!` / `zip_map!`

`macro_rules!` — element-wise operations over same-shape tensors. `bcast!` allocates a new tensor; `zip_map!` writes into a pre-existing output.

```rust
// Allocating: unary, binary, ternary
let doubled: Tensor<f64> = bcast!(|x| x * 2.0, &a);
let sum:     Tensor<f64> = bcast!(|x, y| x + y, &a, &b);

// In-place
let mut out: Tensor<f64> = Tensor::zeros(2, 2);
zip_map!(out, |x, y| x * y, &a, &b);
```

### `between!` / `frange!`

```rust
assert!(between!(0.0_f64, 0.5, 1.0));  // 0.0 <= 0.5 < 1.0

let v = frange!(0.0_f64, 0.25, 1.0);   // [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

## Backend system

### Architecture

The `Backend` trait is sealed — not implementable outside the crate. All `Tensor<T, B>` operations are generic over `B: Backend`, resolved at compile time.

```
DefaultBackend priority:  cuda > wgpu > cpu
```

### Feature flags

| Feature | Backend | Crates | Status |
|---|---|---|---|
| `cpu` (default) | CPU + rayon | `faer 0.24`, `rayon 1` | Production |
| `cuda` | NVIDIA GPU | `cubecl-cuda 0.9` | Implemented |
| `wgpu` | Vulkan / Metal / DX12 | `cubecl-wgpu 0.9` | Implemented |
| `hip` | AMD GPU | `cubecl-hip 0.9` | Stub (delegates to CPU) |

### Mixed CPU/GPU usage

```rust
use nabla::prelude::*;

// DefaultBackend — GPU if cuda/wgpu enabled, CPU otherwise
let a: Tensor<f32> = Tensor::zeros(128, 128);

// Explicit CPU — always available
let b: Tensor<f32, Cpu> = Tensor::zeros(128, 128);

// Backend conversion
let cpu_copy = a.to_cpu();
// let gpu_copy = b.to_cuda();    // requires `cuda` feature
// let wgpu    = b.to_wgpu();     // requires `wgpu` feature
let generic  = b.to_backend::<Cpu>();
```

Linear algebra and sparse solvers require `Tensor<T, Cpu>` (faer dependency). Convert via `.to_cpu()`, compute, then `.to_cuda()` / `.to_wgpu()` if needed.

### GPU kernels (cubecl)

| Kernel | f32 / f64 | c32 / c64 |
|---|---|---|
| `elementwise_add` | GPU | CPU fallback |
| `elementwise_sub` | GPU | CPU fallback |
| `elementwise_neg` | GPU | CPU fallback |
| `elementwise_scale` | GPU | CPU fallback |
| `transpose` | GPU | CPU fallback |
| `matmul_naive` | GPU | CPU fallback |

---

## Project structure

```
nabla/
├── src/
│   ├── lib.rs        crate root + scalar + error + util modules
│   ├── tensor.rs     Tensor<T,B> + StaticMatrix + Array/Matrix traits
│   ├── backend.rs    Backend trait + Cpu + GPU storage + cubecl kernels
│   ├── linalg.rs     Dense factorization + Diagonal/Symmetric/Triangular
│   └── sparse.rs     SparseMatrix<T> CSC
├── macros/
│   └── src/
│       ├── lib.rs    mat! + einsum! entry points
│       └── einsum.rs einsum parser + codegen
├── tests/
│   ├── basic.rs      integration tests (38)
│   ├── broadcast.rs  broadcast macro tests (7)
│   └── static_mat.rs StaticMatrix tests (18)
└── docs/
    └── spec.md       full specification
```

---

## Error handling

nabla uses a typed `Error` enum with a `Result<T>` alias:

| Variant | When |
|---|---|
| `Error::ShapeMismatch { expected, got }` | Incompatible dimensions in solve, factorize, etc. |
| `Error::InvalidDimension(String)` | Bad input to constructors or structural types |

Fallible operations (`.solve()`, `.factorize_*()`, structural type constructors) return `Result<T>`. Operator overloads (`+`, `-`, `*`) panic on shape mismatch with a descriptive message.

```rust
use nabla::prelude::*;

let a: Tensor<f64> = Tensor::zeros(2, 2);
let b: Tensor<f64> = Tensor::from_fn(2, 1, |_, _| 1.0);

match a.solve(&b) {
    Ok(x)  => println!("solution: {:?}", x),
    Err(e) => eprintln!("error: {e}"),
}
```

---

## Building and testing

```bash
cargo build                                          # build
cargo test --workspace                               # all tests (63 + 9 doc-tests)
cargo clippy --workspace --all-targets -- -D warnings # lint
cargo fmt --all -- --check                           # format check
```

---

## License

Apache-2.0 OR MIT (dual license)
