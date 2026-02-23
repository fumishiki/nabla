# ∇ nabla

[![CI](https://github.com/fumishiki/nabla/actions/workflows/ci.yml/badge.svg)](https://github.com/fumishiki/nabla/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**Type-safe linear algebra DSL for Rust** — proc-macro notation, pure-Rust kernels, wgpu GPU backend, reverse-mode autodiff. Zero external LA dependencies.

```rust
use nabla::prelude::*;

let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
let b: Tensor<f64> = mat![[5.0_f64, 6.0], [7.0, 8.0]];

let c = &a * &b;                                         // matmul
let x = a.solve(&b).unwrap();                             // linear solve
let e: Tensor<f64> = einsum!(e[i,j] = a[i,k] * b[k,j]);  // Einstein notation

let tape = Tape::new();
let va = tape.variable(a);
let loss = va.exp().sum_all_var();
let grads = loss.backward();                              // reverse-mode AD
```

---

## Why nabla?

Rust's LA ecosystem is powerful but verbose. nabla fixes that with a DSL layer:

<table>
<tr><th>Operation</th><th>Raw Rust / nalgebra</th><th>nabla</th></tr>
<tr>
<td>Matrix literal</td>
<td>

```rust
let a = DMatrix::from_row_slice(2, 2,
    &[1.0, 2.0, 3.0, 4.0]);
```

</td>
<td>

```rust
let a: Tensor<f64> = mat![
    [1.0_f64, 2.0],
    [3.0, 4.0]
];
```

</td>
</tr>
<tr>
<td>Matmul</td>
<td>

```rust
let c = &a * &b;  // same
```

</td>
<td>

```rust
let c = &a * &b;  // same
```

</td>
</tr>
<tr>
<td>Einsum contraction</td>
<td>

```rust
// manual loop — no einsum in nalgebra
let mut c = DMatrix::zeros(m, n);
for i in 0..m {
  for j in 0..n {
    for k in 0..p {
      c[(i,j)] += a[(i,k)] * b[(k,j)];
    }
  }
}
```

</td>
<td>

```rust
let c: Tensor<f64> = einsum!(
    c[i,j] = a[i,k] * b[k,j]
);
```

</td>
</tr>
<tr>
<td>Batch matmul (N-D)</td>
<td>

```rust
// manual batch loop over 3D array
for batch in 0..b {
    let a_slice = a.slice(..);
    let m_slice = m.slice(..);
    // manual matmul per batch...
}
```

</td>
<td>

```rust
let c: NdTensor<f64> = einsum!(
    c[b,i,j] = a[b,i,k] * m[b,k,j]
);
```

</td>
</tr>
<tr>
<td>Element-wise broadcast</td>
<td>

```rust
let y = a.map(|x| x.sin().powi(2));
```

</td>
<td>

```rust
let y = bcast!(|x| x.sin().powi(2), &a);
// or GPU-aware:
let y = bcast_all!(x.sin().powf(2.0); x);
```

</td>
</tr>
<tr>
<td>In-place broadcast</td>
<td>

```rust
for i in 0..m {
  for j in 0..n {
    out[(i,j)] = a[(i,j)] * b[(i,j)];
  }
}
```

</td>
<td>

```rust
zip_map!(out, |x, y| x * y, &a, &b);
```

</td>
</tr>
<tr>
<td>Stencil (Laplacian)</td>
<td>

```rust
for i in 1..m-1 {
  for j in 1..n-1 {
    out[(i,j)] = -4.0 * a[(i,j)]
      + a[(i-1,j)] + a[(i+1,j)]
      + a[(i,j-1)] + a[(i,j+1)];
  }
}
```

</td>
<td>

```rust
let out = stencil!(out[i,j] =
    -4.0 * a[i,j]
    + a[i-1,j] + a[i+1,j]
    + a[i,j-1] + a[i,j+1]
);
```

</td>
</tr>
<tr>
<td>Autodiff</td>
<td>

```rust
// not available in nalgebra
```

</td>
<td>

```rust
let tape = Tape::new();
let x = tape.variable(tensor);
let loss = x.exp().sum_all_var();
let grads = loss.backward();
let dx = grads.wrt(&x);
```

</td>
</tr>
</table>

### vs Python (NumPy)

| NumPy | nabla |
|---|---|
| `np.array([[1,2],[3,4]])` | `mat![[1.0_f64, 2.0], [3.0, 4.0]]` |
| `np.einsum('ik,kj->ij', a, b)` | `einsum!(c[i,j] = a[i,k] * b[k,j])` |
| `np.sin(x)**2` | `bcast_all!(x.sin().powf(2.0); x)` |
| `np.linalg.solve(a, b)` | `a.solve(&b)?` |
| `np.linalg.svd(a)` | `a.factorize_svd()?` |

### vs Julia

| Julia | nabla |
|---|---|
| `[1 2; 3 4]` | `mat![[1.0_f64, 2.0], [3.0, 4.0]]` |
| `@tullio C[i,j] := A[i,k]*B[k,j]` | `einsum!(c[i,j] = a[i,k] * b[k,j])` |
| `@. y = sin(x)^2` | `bcast_all!(x.sin().powf(2.0); x)` |
| `A \ b` | `a.solve(&b)?` |
| `lu(A)` | `a.factorize_lu()?` |
| `0.0:0.1:1.0` | `frange!(0.0_f64, 0.1, 1.0)` |
| `0 < x < 1` | `between!(0.0, x, 1.0)` |
| `(a=1, b=2)` | `named!(a: i32 = 1, b: i32 = 2)` |

---

## Features at a glance

| Category | What you get |
|---|---|
| **13 macros** | `mat!` `einsum!` `bcast!` `bcast_all!` `zip_map!` `par_bcast!` `stencil!` `named!` `generated!` `splat!` `pipe!` `between!` `frange!` |
| **9 dense factorizations** | LU, Full-Pivot LU, QR, Col-Pivot QR, Cholesky, LDL, Bunch-Kaufman, SVD, Eigen |
| **Sparse LA** | CSC storage, sparse LU/QR/Cholesky solve, sparse × dense matmul |
| **Symbolic CAS** | `Expr` tree — differentiation, simplification, evaluation |
| **ODE solvers** | Euler, RK4, Dormand-Prince (adaptive step) |
| **Reverse-mode AD** | 14 ops, tape-based `backward()`, chain rule |
| **GPU (wgpu)** | 32 WGSL compute shaders, tiled matmul, zero host transfer |
| **Complex** | `c32`/`c64`, proper adjoint ≠ transpose |
| **Zero deps** | Pure Rust LA — no LAPACK, no BLAS, no external LA bindings |

~10,000 lines · 61 tests · MSRV 1.85 · Apache-2.0 OR MIT

---

## Install

```toml
[dependencies]
nabla = { git = "https://github.com/fumishiki/nabla" }

# GPU (exclusive with cpu):
# nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["gpu"] }
```

---

## Quick start

```rust
use nabla::prelude::*;

// Construction
let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
let eye: Tensor<f64> = Tensor::identity(3);
let m: Tensor<f64> = Tensor::from_fn(3, 3, |r, c| (r * 3 + c) as f64);

// Arithmetic
let sum  = &a + &a;          // element-wise add
let prod = &a * &a;          // matmul
let s    = &a * 2.0_f64;     // scalar multiply

// Einsum
let b: Tensor<f64> = Tensor::from_fn(2, 3, |r, c| (r * 3 + c) as f64);
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);

// Solve
let rhs: Tensor<f64> = mat![[5.0_f64], [6.0]];
let x = a.solve(&rhs).unwrap();

// Autodiff
let tape = Tape::new();
let va = tape.variable(a);
let loss = va.exp().sum_all_var();
let grads = loss.backward();
```

---

## Macros

### `mat!` — matrix literal

```rust
let m: Tensor<f64> = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
assert_eq!(m.shape(), (2, 3));  // compile-time row/col validation
```

### `einsum!` — Einstein summation (7 patterns, auto-optimized)

```rust
// GEMM — compiles to matmul_into
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);

// Hadamard — compiles to mul_elem
let h: Tensor<f64> = einsum!(h[i,j] = a[i,j] * b[i,j]);

// Trace — diagonal sum
let tr: f64 = einsum!(tr = a[i,i]);

// Outer product
let o: Tensor<f64> = einsum!(o[i,j] = u[i] * v[j]);

// Batch GEMM (N-D)
let c3: NdTensor<f64> = einsum!(c[b,i,j] = a[b,i,k] * m[b,k,j]);
```

Spanned compile errors point to the exact token in your expression.

### `bcast!` / `bcast_all!` / `zip_map!` — broadcasting

```rust
let y: Tensor<f64> = bcast!(|x| x.exp(), &a);              // allocating
zip_map!(out, |x, y| x * y, &a, &b);                       // in-place
let z: Tensor<f64> = bcast_all!(x.sin().powf(2.0); x);     // GPU-aware
```

### `stencil!` — offset indexing

```rust
let lap = stencil!(lap[i, j] =
    -4.0 * a[i,j] + a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1]
);
```

### More macros

```rust
par_bcast!(|x| x.sqrt(), &a);                // parallel broadcast (rayon)
let p = named!(x: f64 = 1.0, y: f64 = 2.0);  // named tuple
pipe!(x, f, g, h);                            // h(g(f(x)))
splat!(add3, (1.0, 2.0, 3.0));               // tuple splatting
between!(0.0, x, 1.0);                        // chained comparison
frange!(0.0_f64, 0.1, 1.0);                   // float range
```

---

## Dense linear algebra

```rust
// Direct solve
let x = a.solve(&b)?;            // Ax = b  (LU)
let x = a.solve_lstsq(&b)?;     // least-squares (SVD)
let x = a.inv()?;                // inverse

// Factorize once, solve many
let lu = a.factorize_lu()?;
let x1 = lu.solve(&b);
let x2 = lu.solve(&c);

// 9 factorizations available
a.factorize_lu()?;               // PA = LU  (partial pivot)
a.factorize_full_piv_lu()?;      // PAQ^T = LU
a.factorize_qr();                // A = QR
a.factorize_col_piv_qr();        // AP^T = QR
a.factorize_llt()?;              // Cholesky LL^T
a.factorize_ldlt()?;             // LDL^T
a.factorize_lblt();              // Bunch-Kaufman
a.factorize_svd()?;              // full USV^H
a.self_adjoint_eigen()?;         // symmetric eigendecomposition
```

### Structural types

```rust
let d = Diagonal::new(vec![1.0, 2.0, 3.0]);
let s = Symmetric::new(tensor, Side::Lower)?;
let t = Triangular::new(tensor, TriKind::Lower)?;
let eigenvalues = s.eigenvalues()?;
```

---

## Sparse

```rust
use nabla::sparse::{SparseMatrix, Triplet};

let trips = vec![
    Triplet { row: 0, col: 0, val: 4.0_f64 },
    Triplet { row: 0, col: 1, val: -1.0 },
    Triplet { row: 1, col: 0, val: -1.0 },
    Triplet { row: 1, col: 1, val: 4.0 },
];
let s = SparseMatrix::try_new_from_triplets(2, 2, &trips)?;

let x = s.solve(&b)?;             // sparse LU
let x = s.solve_lstsq(&b)?;       // sparse QR
let y = s.matmul_dense(&dense)?;  // sparse × dense
```

---

## Symbolic CAS

```rust
use nabla::cas::Expr;

let x = Expr::var("x");
let f = (&x * &x).sin();          // sin(x²)
let df = f.diff("x").simplify();   // symbolic derivative
let val = df.eval(&[("x", 1.0)]); // evaluate
```

---

## ODE solvers

```rust
use nabla::ode;

let f = |_t: f64, y: &Tensor<f64>| -> Tensor<f64> { /* dy/dt */ };
let y0: Tensor<f64> = mat![[1.0_f64], [0.0]];

let y = ode::euler(f, &y0, 0.0, 1.0, 100);
let y = ode::rk4(f, &y0, 0.0, 1.0, 100);
let y = ode::dormand_prince(f, &y0, 0.0, 1.0, 1e-6, 1e-3);
```

---

## Autodiff

Reverse-mode automatic differentiation — tape-based, 14 ops.

```rust
let tape = Tape::<f64>::new();
let w = tape.variable(weights);
let x = tape.variable(input);

// Forward pass — builds computation graph
let z = w.matmul(&x);
let h = z.exp();
let loss = h.sum_all_var();

// Backward pass — chain rule (fixed mathematical rule)
let grads = loss.backward();
let dw = grads.wrt(&w);  // ∂loss/∂w
let dx = grads.wrt(&x);  // ∂loss/∂x
```

Supported ops: `add`, `sub`, `neg`, `mul_elem`, `matmul`, `scale`, `exp`, `ln`, `sin`, `cos`, `tanh`, `sqrt`, `powf`, `sum_all_var`.

---

## Backend system

Compile-time exclusive selection. One backend per binary — no runtime dispatch.

| Feature | Backend | Types | Status |
|---|---|---|---|
| `cpu` (default) | Pure Rust + rayon | f32, f64, c32, c64 | ✅ |
| `gpu` | wgpu + WGSL compute | f32 only | ✅ 32 ops |

```bash
cargo build                                           # CPU (default)
cargo build --no-default-features --features gpu      # GPU
```

Both enabled → `compile_error!`. No implicit CPU fallback on GPU.

### GPU architecture

```
Host                              Device (GPU)
────                              ──────────────
upload / from_fn
  └──── WGSL compute shader ──→ Buffer₁
                                      │
         chained ops (zero transfer)  │ add/sub/exp/matmul/...
                                      ▼
                                  Buffer₂
                                      │
  .get(r,c)                           │
  ┌──── map_async + poll ◄────────────┘
  ▼
host_cache (lazy, cached on first access)
```

32 WGSL shaders: binary (5) · unary (15) · tiled matmul · reduction (5) · construction (3) · copy/transpose (2) · arg-reduction (2).

---

## Project structure

```
nabla/
├── Cargo.toml           workspace: nabla + macros
├── src/
│   ├── lib.rs            crate root + prelude + error + util
│   ├── tensor.rs         Tensor<T,B> + NdTensor<T> + StaticMatrix + DynTensor
│   ├── backend.rs        Backend trait (sealed) + Cpu + CpuStorage
│   ├── gpu.rs            GpuStorage + wgpu WGSL compute shaders
│   ├── scalar.rs         Scalar trait + Complex<T>
│   ├── linalg.rs         9 dense factorizations + structural types
│   ├── sparse.rs         SparseMatrix<T> CSC
│   ├── cas.rs            Symbolic CAS
│   ├── ode.rs            ODE solvers
│   └── autograd.rs       Reverse-mode AD
├── macros/src/
│   ├── lib.rs            proc macro entries
│   ├── einsum.rs         einsum parser + 7-pattern codegen
│   └── stencil.rs        stencil offset codegen
├── tests/                61 tests (unit + boundary + doc + compile-fail)
└── docs/spec.md          full specification
```

---

## Building & testing

```bash
cargo test                                             # all tests
cargo clippy --workspace --all-targets -- -D warnings  # lint (0 warnings)
cargo fmt --all -- --check                             # format
cargo doc --workspace --no-deps                        # docs
```

MSRV: **1.85.0** (Rust edition 2024)

---

## Contributing

1. Fork → feature branch
2. `cargo test && cargo clippy -- -D warnings && cargo fmt --check`
3. PR against `main`

See [docs/spec.md](docs/spec.md) for the full specification.

---

## License

[Apache-2.0](LICENSE-APACHE) OR [MIT](LICENSE-MIT), at your option.
