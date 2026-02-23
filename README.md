# nabla

`nabla` is a compact Rust linear-algebra crate focused on practical workflows for numerical computing.
It provides a Julia-inspired dense tensor API, sparse matrix helpers, and a small macro layer for concise matrix literals, while staying lightweight and performance-conscious.

`nabla` is implemented as:

- `nabla` (core numerical engine)
- `nabla-macros` (`mat!` procedural macro)

It depends on `faer` for dense linear algebra by default and keeps optional GPU paths behind Cargo features.

---

## Table of contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Dense tensors API](#dense-tensors-api)
- [Macros](#macros)
- [Sparse matrices](#sparse-matrices)
- [Linear algebra APIs](#linear-algebra-apis)
- [Backend and features](#backend-and-features)
- [Error handling](#error-handling)
- [Clippy/lints and validation](#validation)
- [Project structure](#project-structure)
- [License](#license)

---

## Highlights

- Dense tensor abstraction `Tensor<T, B>` with shape-safe indexing helpers.
- `mat!` macro for compact matrix construction.
- Operator overloading:
  - `+`, `-` (matrix add/sub)
  - unary `-`
  - `*` (matrix multiplication and scalar multiplication)
- Dense linear algebra pipeline:
  - LU, LDLT, LBLT, QR, SVD
  - eigendecomposition for symmetric/hermitian matrices
  - solve / rsolve / least squares
  - reconstructions and inverse helpers
- Sparse matrix support:
  - `SparseMatrix`
  - triplet constructors
  - symbolic and numeric factorization helpers
  - sparse × dense matmul
- Backend abstraction with CPU default and optional GPU-related feature gates.
- Minimal public surface; internal logic is intentionally narrow and explicit.

---

## Quick start

### 1) Install

```toml
[dependencies]
nabla = { git = "https://github.com/your-org/nabla" }
# or a local path
# nabla = { path = "../nabla" }
```

### 2) Basic dense usage

```rust
use nabla::prelude::*;

fn main() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];

    let c = &a + &b;
    let d = &a * &b;
    let e = &a * 2.0_f64;

    assert_eq!(c.shape(), (2, 2));
    assert_eq!(d.shape(), (2, 2));
    assert_eq!(e.get(0, 0), 2.0);
}
```

### 3) Common linear algebra example

```rust
use nabla::prelude::*;

let a: Tensor<f64> = Tensor::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
let rhs: Tensor<f64> = Tensor::from_fn(2, 1, |_, _| 1.0);

let at = a.t();
let ah = a.adjoint();
let x = a.solve(&rhs).expect("linear solve failed");
let x_ls = a.solve_lstsq(&rhs).expect("least-squares failed");

let _ = ah; // keep example complete if linted in larger contexts
let _ = x;
let _ = x_ls;
let _ = at;
```

### 4) Sparse matrix example

```rust
use nabla::prelude::*;
use nabla::sparse::{SparseMatrix, Triplet};

let entries = [
    Triplet { row: 0, col: 0, val: 10.0_f64 },
    Triplet { row: 1, col: 1, val: 20.0_f64 },
    Triplet { row: 2, col: 2, val: 30.0_f64 },
];

let sp = SparseMatrix::try_new_from_triplets(3, 3, &entries)
    .expect("invalid sparse input");

assert_eq!(sp.shape(), (3, 3));
assert_eq!(sp.nnz(), 3);
```

---

## Core concepts

### Tensor

`Tensor<T, Backend>` is the central dense matrix type.

- Creation:
  - from_fn
  - from_vec / shape-based constructors if exposed by your current version
- Query:
  - `shape()` returns `(rows, cols)`
  - element access via indexing helpers
- Storage model:
  - row-major dense layout in the active backend implementation

Important behavior:
- Shape mismatch in arithmetic is a runtime check and currently panics in this branch.
- This keeps failures explicit for early development and fast debugging.

### Ownership and performance intent

- Keep operations on borrowed inputs where practical.
- Reduce intermediate allocations in internal paths.
- Prefer explicit fallible construction APIs where input dimensions or ranges may be invalid.

---

## Dense tensors API

This crate provides operator-style usage and dedicated methods depending on operation type.

### Construction

- `Tensor::from_fn(rows, cols, f)` builds values by index closure.
- `mat![[...], [...]]` can be used for quick literals.

### Operations

- `+` and `-`: matrix-wise add/sub.
- unary `-`: element-wise negation.
- `*`:
  - matrix × matrix (shape-consistent multiplication)
  - matrix × scalar

### Linalg methods

Most methods are fallible and return `Result`.

- `solve`, `rsolve`
- `solve_lstsq`
- `reconstruct`, `inv`-related helpers
- decomposition methods from LU/QR/SVD families

---

## Macros

### `mat!`

`mat!` is provided by `nabla-macros` and parses nested bracket literals such as:

```rust
let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

Constraints:
- At least one row is required.
- The macro currently validates row/column structure and produces readable compile-time diagnostics when malformed.

---

## Sparse matrices

### Types and conversion

- `SparseMatrix<T>`: sparse storage abstraction.
- `Triplet { row, col, val }`: coordinate format input entry.
- `SparseMatrix::try_new_from_triplets(rows, cols, entries)` is a checked constructor.

### Typical flow

1. Build sparse matrix from triplets.
2. Optionally run symbolic / numeric factorization helpers.
3. Use sparse/dense interop or solver entry points.

Bounds checks exist on shape/index construction to prevent invalid matrix states.

---

## Backend and features

### Features

- `cpu` (default)
- `cuda` (optional): `cubecl-cuda`
- `wgpu` (optional): `cubecl-wgpu`
- `hip` (optional): `cubecl-hip`

### Notes

- Default builds are CPU-only and currently the most stable path.
- GPU features are feature-gated and should be validated with `--all-features` in a network-enabled environment.

### Feature validation

- Offline/CI-limited mode:
  - `cargo test --all --offline`
  - `cargo clippy --offline --all-targets`
- Full feature lint pass requires crates resolvable from registry:
  - `cargo clippy --all-features`

---

## Error handling

`Result`-returning public APIs include explicit error types in their contracts.

- Invalid shapes, bad indices, and construction-time invalid inputs are surfaced as typed errors.
- Favor explicit propagation (`?`) in user code when chaining operations.

Recommended practice:
- Check `shape` before algebraic composition in public-facing code.
- Keep recoverable errors in `Result` and keep programming errors (defensive assertions/panic behavior where decided) documented.

---

## Validation

Current repository checks:

- `cargo test --all --offline`
- `cargo clippy --offline --all-targets`

If you are preparing a broader dependency refresh or enabling all optional backends, run:

- `cargo clippy --all-features`

and review warning set for all feature combinations.

---

## Project structure

```
nabla/
├─ src/
│  ├─ lib.rs
│  ├─ tensor.rs
│  ├─ linalg.rs
│  ├─ sparse.rs
│  └─ backend.rs
├─ macros/
│  └─ src/lib.rs
├─ tests/
├─ README.md
├─ LICENSE
├─ LICENSE-APACHE
├─ LICENSE-MIT
├─ NOTICE
├─ Cargo.toml
└─ Cargo.lock
```

- `src/` contains dense, linalg, sparse, and backend orchestration.
- `macros/` contains `mat!` implementation.
- `tests/` contains behavior checks.

---

## License

This project is available under either:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT (`LICENSE-MIT`)

If you choose one, use:
- Summary file: `LICENSE`
- Third-party attributions: `NOTICE`

