# nabla — Specification

Legend: ✅ Implemented | ❌ Not possible (language constraint)

---

# Part I — Current Implementation

## 1. Overview

**Zero-GC, zero-copy, type-safe** Rust linear algebra DSL for researchers who refuse to choose between Python's ease and C++'s speed.
Proc macros (`mat![]`, `map!{}`, `einsum!{}`) combined with self-contained pure-Rust kernels (CPU) and GPU compute shaders across three backends: wgpu (WGSL), CUDA (nvrtc), HIP (hiprtc).
Exactly one backend is selected via feature flags at build time — no implicit CPU fallback, no cross-backend runtime dispatch.

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

### Design principles

1. **Zero-GC, zero-copy** — ownership = automatic memory management without GC. `Drop` = deterministic deallocation. `&` = zero-copy borrow. `_into(out: &mut)` = zero-allocation in-place
2. **Python's ease, C's speed** — PyTorch-familiar API (`loss.backward()`, `.exp()`, `.sum()`) with NumPy-like broadcasting (`map!`), delivered at native speed
3. **Macros = notation layer** — proc macros for concise syntax, type-safe Rust underneath
4. **trait = dispatch** — trait-based multiple dispatch (Python duck-typing composability without runtime cost)
5. **Self-contained LA** — row-major CpuStorage, 9 dense factorizations, CSC sparse. Zero external LA deps
6. **Build-time exclusive backend** — `cpu`/`wgpu`/`cuda`/`hip` feature flags, exactly one active. `compile_error!` on multi-select. No implicit CPU fallback
7. **Two kernel codebases** — WGSL (wgpu) + CUDA/HIP shared C source. ❌CubeCL. Fixed-rule 32 ops → manageable dual maintenance
8. **Fixed-rule principle** — only mathematically invariant rules. User-customizable domains excluded
9. **Adjoint ≠ Transpose** — correct complex LA semantics

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
| $A \circ B$ | `A * B` | `A .* B` | `a.emul(b)` | Julia `A .* B` → `e`-prefix |
| $f(A)$ map | `np.vectorize(f)(A)` | `f.(A)` | `map!(\|v\| f(v), &a)` | Python `map` — CPU |
| $f(A)$ fused GPU | `torch.compile()(f)(A)` | `@. f(A)` | `fuse!(v.f(); a)` | Julia `@.` → GPU 1 kernel |
| $[A; B]$ vcat ✅ | `np.vstack([A,B])` | `[A; B]` | `vcat!(a, b, c, ...)` | Julia `[A; B]` style — variadic macro |

### 3.2 Tensor construction — free functions (型推論に任せる)

```rust
// NumPy: A = np.zeros((3, 3))     Julia: A = zeros(3, 3)
let a: Tensor<f64> = zeros(3, 3);  // explicit type
let a = zeros::<f64>(3, 3);        // turbofish (when inference can't decide)
let b = zeros(3, 3);               // type inferred from later usage

// NumPy: I = np.eye(4)            Julia: I = I(4)
let id = eye(4);

// NumPy: r = np.random.randn(m, n)
let r = randn(100, 100);
```

| Operation | nabla | Python | Julia |
|---|---|---|---|
| Matrix literal | `mat![[1, 2], [3, 4]]` | `np.array([[1,2],[3,4]])` | `[1 2; 3 4]` |
| Zeros | `zeros(m, n)` | `np.zeros((m,n))` | `zeros(m,n)` |
| Ones | `ones(m, n)` | `np.ones((m,n))` | `ones(m,n)` |
| Fill | `fill(m, n, val)` | `np.full((m,n), val)` | `fill(val, m, n)` |
| Identity | `eye(n)` | `np.eye(n)` | `I(n)` |
| Random normal | `randn(m, n)` | `np.random.randn(m,n)` | `randn(m,n)` |
| Random uniform | `rand(m, n)` | `np.random.rand(m,n)` | `rand(m,n)` |
| From function | `from_fn(m, n, \|r, c\| expr)` | `np.fromfunction(f, (m,n))` | — |
| Parallel | `par_from_fn(m, n, \|r, c\| expr)` | — | — |
| Float range | `arange(0.0, 1.0, 0.1)` | `np.arange(0, 1, 0.1)` | `0.0:0.1:1.0` |
| Linspace | `linspace(0.0, 1.0, n)` | `np.linspace(0, 1, n)` | `range(0,1,length=n)` |
| Complex | `c64(2.0, 3.0)` | `2+3j` | `2+3im` |
| Static (stack) | `StaticMatrix::<f64,3,3>::zeros()` | — | `SMatrix{3,3}(...)` |
| N-D | `nd_zeros(&[d0, d1, d2])` | `np.zeros((d0,d1,d2))` | `zeros(d0,d1,d2)` |
| Vertical cat ✅ | `vcat!(a, b, c, ...)` | `np.vstack([A,B,C])` | `[A; B; C]` |
| Horizontal cat ✅ | `hcat!(a, b, c, ...)` | `np.hstack([A,B,C])` | `[A B C]` |
| Reshape ✅ | `a.reshape(m, n)` | `A.reshape(m, n)` | `reshape(A, m, n)` |

Implementation: free functions in `prelude` that delegate to `Tensor::<T>::zeros(m, n)` etc. Type `T` is inferred from context (`let a: Tensor<f64> = zeros(3, 3)` or from downstream usage).

**Python/Julia と同じ文字数。** 違いは: **GC なし、即座解放、`StaticMatrix` はスタック配置（ヒープゼロ）**。

**Julia-inspired naming**: `vcat!`/`hcat!` は Julia `[A; B]` / `[A B]` の意味論を macro で表現。Julia の array literal 構文は Rust で再現不能なため macro で代替。`vcat!` = vertical concatenate (行方向に積む)、`hcat!` = horizontal concatenate (列方向に並べる)。

### 3.3 Indexing & slicing — bracket 記法 (`Index` / `IndexMut` trait)

`Index` / `IndexMut` trait 実装で **Python に近い bracket 記法**。0-indexed。全スライスは **zero-copy**。

**Rust 制約**: Python `A[i,j]` は多引数添字だが、Rust の `Index` trait は単一引数のみ受け取る → `a[(i, j)]` とタプルで渡す必要がある。2 文字 (`()`) の追加コストで、戻り値の型安全性とスライス寿命の静的保証を得る。

```rust
// Python: A[2, 3]                Julia: A[3, 4]  (1-indexed)
let v = a[(2, 3)];               // read — Index<(usize, usize)>
a[(2, 3)] = 5.0;                 // write — IndexMut<(usize, usize)>

// Python: A[0:3, 1:4]           Julia: A[1:3, 2:4]
let sub = a[(0..3, 1..4)];       // zero-copy view — Index<(Range, Range)>

// Python: A[0:3, :]             Julia: A[1:3, :]
let rows = a[(0..3, ..)];        // RangeFull = all columns
let cols = a[(.., 1..4)];        // all rows, columns 1..4
```

| Operation | nabla | Python | Chars saved vs old API |
|---|---|---|---|
| Read element | `a[(i, j)]` | `A[i,j]` | `a.get(i, j)` → **-4 chars** |
| Write element | `a[(i, j)] = v` | `A[i,j] = v` | `a.set(i, j, v)` → **-6 chars** |
| Submatrix | `a[(0..3, 1..4)]` | `A[0:3, 1:4]` | `a.slice(0..3, 1..4)` → **-6 chars** |
| Row slice | `a[(0..3, ..)]` | `A[0:3, :]` | `a.slice_rows(0..3)` → **-9 chars** |
| Col slice | `a[(.., 1..4)]` | `A[:, 1:4]` | `a.slice_cols(1..4)` → **-10 chars** |
| Shape | `a.shape()` | `A.shape` | — |
| Rows / Cols | `a.nrows()` / `a.ncols()` | `A.shape[0/1]` | — |
| N-D read | `t[&[i,j,k]]` | `T[i,j,k]` | `t.get_nd(...)` → **-5 chars** |

**vs Python**: NumPy スライスも zero-copy view だが **view の寿命は GC 任せ** — 元テンソルが GC されると view が dangling する可能性。nabla は **借用チェッカがスライス寿命を静的保証** — dangling view はコンパイルエラー。`()` 2 文字分だけ冗長だが、寿命安全性という対価がある。

### 3.4 Arithmetic — "`&` を付けるか付けないかだけ"

**原則**: `a * b` = move (使い捨て)、`&a * &b` = borrow (再利用)。1 回しか使わないテンソルに `&` は不要。

```rust
// 使い捨て (owned) — Python/Julia と同じ感覚
let c = a * b;              // matmul, a と b は消費される
let d = (a + b).emul(c);     // element-wise, 全て consumed

// 再利用 (borrowed) — Rust ならではの明示的ゼロコピー
let c = &a * &b;            // a, b は後で再利用可能
let d = &a * &b + &c;       // 全て借用

// in-place (zero alloc) — PyTorch out= と同じ
c.mm_(&a, &b);              // C += AB, zero allocation
```

| Math | nabla (owned) | nabla (borrowed) | Python | Julia |
|---|---|---|---|---|
| $A + B$ | `a + b` | `&a + &b` | `A + B` | `A + B` |
| $A - B$ | `a - b` | `&a - &b` | `A - B` | `A - B` |
| $-A$ | `-a` | `-&a` | `-A` | `-A` |
| $AB$ (matmul) | `a * b` | `&a * &b` | `A @ B` | `A * B` |
| $A \circ B$ | `a.emul(b)` | `a.emul(&b)` | `A * B` | `A .* B` |
| $A \oslash B$ | `a.ediv(b)` | `a.ediv(&b)` | `A / B` | `A ./ B` |
| $cA$ | `c * a` | `c * &a` | `c * A` | `c * A` |
| $C \mathrel{+}= AB$ | `c.mm_(&a, &b)` | — | `torch.mm(A,B,out=C)` | `mul!(C,A,B)` |
| $A^\top$ | `a.t()` | `a.t()` | `A.T` | `A'` |
| $A^*$ (adjoint) | `a.h()` | `a.h()` | `A.conj().T` | `A'` |

**owned 列の文字数は Python/Julia とほぼ同じ。** `&` を付けると「再利用可能 + ゼロコピー」という追加の意味が生まれる。Python/Julia にはこの選択肢がない。

**`emul`/`ediv` の命名根拠**: Julia `A .* B` / `A ./ B` の "element-wise" 意味論を `e`-prefix で表現。Julia の `.` (dot broadcasting) は Rust 演算子として不可能なため、`emul` (element multiply) / `ediv` (element divide) という一貫した接頭辞で代替する。`mul_elem` (9 chars) と同長だが `emul` (5 chars) の方が短く、`e`-prefix でシリーズ全体 (`emul`, `ediv`, 将来的に `eadd`, `esub`) の命名一貫性を持つ。`had` (Hadamard の略) より数学知識ゼロでも意味が自明。

**vs Python**: PyTorch `A @ B` は refcount +1 → GC で回収。nabla `a * b` は move → 結果がスコープを抜けたら**即座に解放**。GC なし。
**vs Julia**: Julia `C = A * B` は毎回アロケーション + GC。nabla `c.mm_(&a, &b)` はゼロアロケーション。

### 3.5 Broadcasting — "メソッドチェーン = GPU カーネル"

**単一メソッド** (`.sin()`, `.exp()`) は Python/Julia と同じ。**チェーン** (`fuse!`) は GPU 融合。

```rust
// Level 1: 単一 op — PyTorch/NumPy と完全に同じ
let y = a.sin();                    // PyTorch: torch.sin(a)  Julia: sin.(A)
let y = a.exp();                    // PyTorch: torch.exp(a)
let y = a.tanh();                   // PyTorch: torch.tanh(a)

// Level 2: チェーン — 1 GPU カーネルに融合
let y = fuse!(x.sin().powf(2.0); x);
// PyTorch: torch.sin(x)**2  ← 2 kernels, 1 temp tensor, GC later
// Julia:   @. sin(x)^2       ← fused on CPU, not GPU
// nabla:   1 kernel, 0 temp, 0 GC

// Level 3: in-place — ゼロアロケーション
map_!(a, |b, c| b.sin() * c.cos(), &b, &c);
// PyTorch: torch.sin(B, out=A); A.mul_(torch.cos(C))  ← 2 calls
// nabla:   single expression
```

| Level | nabla | PyTorch | Julia | GPU | Alloc |
|---|---|---|---|---|---|
| Single op | `a.sin()` | `torch.sin(a)` | `sin.(A)` | ✅ | 1 |
| Closure | `map!(\|x\| f(x), &a)` | — | `f.(A)` | ❌ CPU | 1 |
| Fused chain | `fuse!(x.sin().powf(2.0); x)` | 2+ kernels | `@. sin(x)^2` | ✅ **1 kernel** | **1** |
| In-place | `map_!(a, \|x\| f(x), &b)` | `torch.sin(B, out=A)` | `A .= f.(B)` | ❌ CPU | **0** |
| Parallel | `par_map!(\|x\| f(x), &a)` | — | `@turbo f.(A)` | ❌ CPU | 1 |

**vs PyTorch**: `torch.sin(x)**2` は 2 CUDA kernel launch + 1 intermediate tensor (GC 待ち)。nabla `fuse!` は **1 kernel, 0 intermediate, 0 GC**。
**vs Julia**: Julia `@.` は CPU fusion のみ。nabla `fuse!` は **GPU kernel fusion**。

**Which macro? — 判断フロー:**

```
何をしたい?
│
├─ 単一 op (.sin(), .exp(), .tanh())
│   └→ メソッド直接: a.sin()                    GPU ✅  alloc 1
│
├─ 複数 op を 1 GPU kernel に融合したい
│   └→ fuse!(v.sin().powf(2.0); x)         GPU ✅  alloc 1, temp 0
│
├─ 任意の closure を要素に適用したい
│   └→ map!(|v| f(v), &a)                     CPU only, alloc 1
│
├─ 結果を既存テンソルに上書きしたい (0 alloc)
│   └→ map_!(a, |v| f(v), &b)                CPU only, alloc 0
│
└─ CPU 並列で要素適用したい
    └→ par_map!(|v| f(v), &a)                  CPU only (rayon), alloc 1
```

**原則**: GPU なら `fuse!` 一択。CPU で任意 closure なら `map!`。in-place なら `map_!`。並列なら `par_map!`。

**Element-wise methods — PyTorch identical:**

| Category | Methods |
|---|---|
| Exponential | `.exp()` `.ln()` `.log1p()` `.powf(p)` |
| Trigonometric | `.sin()` `.cos()` `.tanh()` |
| Algebraic | `.sqrt()` `.abs()` `.recip()` `.neg()` |
| Special/Rounding | `.erf()` `.ceil()` `.floor()` `.round()` |
| Reduction | `.sum()` `.max()` `.min()` `.argmax()` `.argmin()` |

### 3.6 Linear algebra — 短いメソッド名

CPU dense linalg via `faer`. GPU linalg: Recursive GEMM TRSM (`gpu_trsm_lower`) implemented for CUDA/HIP backend (W15).

```rust
let x = a.solve(&b)?;       // Ax = b  (NumPy: 20 chars → nabla: 16 chars)
let x = a.lstsq(&b)?;       // least squares
let x = a.inv()?;            // A⁻¹

// Factorize once, solve many
let lu = a.lu()?;            // Julia: lu(A)  — same length
let x1 = lu.solve(&b1);
let x2 = lu.solve(&b2);      // reuse — NumPy can't do this easily
```

| Math | nabla | NumPy/SciPy | Julia |
|---|---|---|---|
| $Ax = b$ | `a.solve(&b)?` | `np.linalg.solve(A,b)` | `A \ b` |
| $xA = b$ | `a.rsolve(&b)?` | — | `b / A` |
| $A^{-1}$ | `a.inv()?` | `np.linalg.inv(A)` | `inv(A)` |
| $PA = LU$ | `a.lu()?` | `scipy.linalg.lu(A)` | `lu(A)` |
| $A = QR$ | `a.qr()` | `np.linalg.qr(A)` | `qr(A)` |
| $A = LL^\top$ | `a.chol()?` | `np.linalg.cholesky(A)` | `cholesky(A)` |
| $A = LDL^\top$ | `a.ldl()?` | `scipy.linalg.ldl(A)` | `ldlt(A)` |
| $A = U\Sigma V^\top$ | `a.svd()?` | `np.linalg.svd(A)` | `svd(A)` |
| $\sigma_i$ | `a.svdvals()?` | `np.linalg.svd(A, 0)` | `svdvals(A)` |
| $\lambda, V$ (sym) | `a.sym(Lower).eigh()?` | `np.linalg.eigh(A)` | `eigen(Sym(A))` |

Structural matrices:

| Type | nabla | Julia |
|---|---|---|
| Diagonal | `Diagonal::new(v)` | `Diagonal(v)` |
| Symmetric | `Symmetric::new(a, Side::Lower)?` | `Symmetric(A, :L)` |
| Triangular | `Triangular::new(a, TriKind::Lower)?` | `LowerTriangular(A)` |

Factorization reuse — `lu.solve(&b)`, `lu.inverse()`, `lu.reconstruct()`.

**短さ比較**: nabla `a.lu()?` = 9 chars。Julia `lu(A)` = 5 chars。NumPy `scipy.linalg.lu(A)` = 20 chars。**Julia に迫り、NumPy の半分以下。**

**vs Python**: `?` は zero-cost `Result` — 例外機構なし、GC なし。factorization 再利用も型で保証。
**vs Julia**: `Result<T>` → 特異行列で silent NaN にならない。Julia `A \ b` は `Inf`/`NaN` を黙って返す。

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

### 3.8 Einstein summation — "添字で考える"

```rust
// NumPy: np.einsum('ik,kj->ij', A, B)   ← string parsing, runtime error
// nabla:
einsum!(c[i,j] = a[i,k] * b[k,j]);     // ← compile-time parsing, spanned error

// NumPy: np.einsum('bik,bkj->bij', A, M)
einsum!(c[b,i,j] = a[b,i,k] * m[b,k,j]);

// NumPy: np.einsum('ii->', A)
einsum!(s = a[i,i]);

// NumPy: np.einsum('i,j->ij', a, b)
einsum!(c[i,j] = a[i] * b[j]);
```

7 patterns at compile time → auto-selects optimal codegen (GEMM, GEMV, Hadamard, trace, outer, batch GEMM, N-D fallback). GPU dispatch via `matmul_into`.

**vs Python**: NumPy `einsum` は**文字列ベース** (`'ik,kj->ij'`) — タイポは実行時エラー。nabla は **AST ベース** — タイポはコンパイル時 spanned error で正確な位置を指摘。さらに NumPy einsum は最適化が限定的だが、nabla は 7 パターンを認識して **最適コード生成**。

**vs Julia**: `@einsum` は実行時エラー。nabla は spanned compile errors + `classify()` による自動パターン最適化 + canonicalization (§13.6) で等価式のチューニング再利用。

### 3.9 Stencil — "偏微分を直感的に"

```rust
// ∇²u (2D Laplacian)
stencil!(out[i,j] = -4.0*u[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]);
```

- Auto-detects offset bounds from expressions
- Zero boundary condition (out-of-bounds = 0.0)
- CPU only (GPU は `fuse!` で代替)

**Julia 改良点**: Julia `@tullio` はインデックス範囲を手動指定。nabla `stencil!` はオフセットから**自動で内部範囲を推定**。

### 3.10 Calculus — PyTorch `.backward()` と同じ

**Reverse-mode autodiff** — PyTorch API をほぼそのまま:

```rust
// PyTorch:                              nabla:
// x = torch.tensor(X, requires_grad=T)
// w = torch.tensor(W, requires_grad=T)
// loss = (x @ w).exp().sum()            let loss = (x * w).exp().sum();
// loss.backward()                       loss.backward();
// x.grad                                let dx = x.grad();
// optimizer.zero_grad()                 // ← 不要。スコープ終了で自動クリア
{
    let tape = Tape::new();
    let x = tape.var(tensor_x);   // tape.var() — short for Variable::new
    let w = tape.var(tensor_w);
    let loss = (&x * &w).exp().sum();
    loss.backward();
    let dx = x.grad();
}   // ← tape, grads, intermediates ALL freed. PyTorch: GC 待ち
```

**Forward-mode** ✅ — 既存コード変更ゼロ:

```rust
let x = Dual::new(2.0, 1.0);   // x = 2, dx/dx = 1
let y = (x * x).sin();          // y = sin(4), dy/dx = 2cos(4)
// impl Scalar for Dual<T> → ALL tensor ops work unchanged
```

**Preparation pattern** ✅ (DifferentiationInterface.jl [Blondel+ 2024]):

```rust
let prep = grad_prep(f, &x);         // one-time: sparsity analysis + tape alloc
let g1 = grad(f, &x, &prep);         // reuse prep
let g2 = grad(f, &x2, &prep);        // amortized cost
```

**Symbolic CAS**:

```rust
use nabla::cas::*;
let x = Expr::var("x");
let f = (x.clone() * x.clone()).sin();    // f(x) = sin(x²)
let df = f.diff("x").simplify();          // f'(x) = 2x·cos(x²)
let val = df.eval(&[("x", 1.0)]);        // f'(1) = 2·cos(1)
```

**ODE solvers**:

```rust
// dy/dt = f(t, y), y(0) = y0
let (ts, ys) = dormand_prince(|t, y| /* f(t,y) */, y0, t_span, dt);
```

| Solver | Order | Adaptive |
|---|---|---|
| `euler` | 1 | No |
| `rk4` | 4 | No |
| `dormand_prince` | 5(4) | Yes |

**Julia 改良点**:
- AD + CAS + ODE が**単一 crate に統合**。Julia では DifferentialEquations.jl + Symbolics.jl + Zygote.jl と 3 パッケージ必要
- **Preparation pattern** で prep/execute 分離 → テープ再利用 + スパースパターン解析を1回に償却
- Forward-mode AD は `impl Scalar for Dual<T>` で **既存コード変更ゼロ**。Julia Zygote は forward-mode 非対応（ForwardDiff.jl 別パッケージ）

### 3.11 Utilities

| Math | nabla | Python | Julia |
|---|---|---|---|
| $0 \le x < 1$ | `between!(0.0, x, 1.0)` | `0 <= x < 1` | `0 ≤ x < 1` |
| $0.0, 0.1, \ldots, 1.0$ | `arange(0.0, 1.0, 0.1)` | `np.arange(0,1,0.1)` | `0.0:0.1:1.0` |
| $x \mapsto f \mapsto g$ | `pipe!(x, f, g)` | — | `x \|> f \|> g` |
| $f(a, b, c)$ from tuple | `splat!(f, (a, b, c))` | `f(*args)` | `f(args...)` |
| Named struct | `named!(a: i32 = 1, b: f64 = 2.0)` | `dict(a=1, b=2.0)` | `(a=1, b=2.0)` |

### 3.12 Parallelism & GPU dispatch

| Strategy | nabla | PyTorch | Julia | GPU |
|---|---|---|---|---|
| Parallel construct | `par_from_fn(m, n, \|r,c\| expr)` | — | `@threads` | ❌ CPU |
| Parallel map | `a.par_map(\|x\| expr)` | — | `@turbo` | ❌ CPU |
| GPU single op | `a.sin()` | `a.sin()` | `CUDA.sin.(A)` | ✅ |
| GPU fused chain | `fuse!(x.sin(); x)` | 2+ kernels | `CUDA.@.` | ✅ **1 kernel** |
| GPU einsum | `einsum!(c[i,j] = a[i,k] * b[k,j])` | — | — | ✅ |

**コードは 1 文字も変えずに CPU → GPU を切り替え:**

```rust
// --features cpu  → CPU BLAS
// --features cuda → NVIDIA GPU kernel
// --features hip  → AMD GPU kernel
// --features wgpu → Vulkan/Metal compute shader
let a: Tensor<f32> = zeros(1024, 1024);
let b: Tensor<f32> = eye(1024);
let c = &a * &b;  // same code, different backend
```

PyTorch `model.to('cuda')` のようなデバイス移動不要。**コンパイル時に全て決定**。

### 3.13 Design: "Python の書きやすさ + Julia の数式感 + Rust の速度"

**文字数比較 — 正直な評価:**

| Operation | Python | Julia | nabla | vs Python | vs Julia |
|---|---|---|---|---|---|
| Zeros | `np.zeros((m,n))` (16) | `zeros(m,n)` (10) | `zeros(m, n)` (11) | **69%** 🟢 | 110% |
| Index | `A[i,j]` (6) | `A[i,j]` (6) | `a[(i,j)]` (8) | 133% 🔴 | 133% 🔴 |
| Matmul | `A @ B` (5) | `A * B` (5) | `a * b` (5) | **100%** | **100%** |
| Transpose | `A.T` (3) | `A'` (2) | `a.t()` (5) | 167% 🔴 | 250% 🔴 |
| Sin | `np.sin(A)` (9) | `sin.(A)` (7) | `a.sin()` (7) | **78%** 🟢 | **100%** |
| Solve | `np.linalg.solve(A,b)` (20) | `A \ b` (5) | `a.solve(&b)?` (14) | **70%** 🟢 | 280% 🔴 |
| LU | `scipy.linalg.lu(A)` (19) | `lu(A)` (5) | `a.lu()?` (8) | **42%** 🟢 | 160% 🔴 |
| Backward | `loss.backward()` (15) | — | `loss.backward()` (15) | **100%** | — |
| Hadamard | `A * B` (5) | `A .* B` (6) | `a.emul(b)` (9) | 180% 🔴 | 150% 🔴 |

**勝敗の傾向:**
- 🟢 **nabla が勝つ場所** — Python の冗長な名前空間 (`np.linalg.solve`, `scipy.linalg.lu`) を短縮する演算。PyTorch 互換メソッド (`.sin()`, `.backward()`)
- 🔴 **nabla が負ける場所** — 言語組込み構文 (Python `A[i,j]`, Julia `A'`, `A \ b`) には勝てない。Rust の `Index` trait 制約 (`()`) と `Mul` trait 占有 (`emul`) が定数オーバーヘッド

**核心**: nabla のオーバーヘッドは **定数** (`()`, `&`, `?`, `.emul`) — 式が長くなるほど占有率は下がる。そしてこの定数コストの対価は大きい: GC ゼロ + コンパイル時型検証 + 借用による明示的メモリ制御。

**nabla のポジション**:

```
                  書きやすさ
                     ↑
           Python ●  |
                     |    ● nabla (目標)
            Julia ●  |
                     |
                     +──────────────→ 実行速度 + メモリ制御
                     |           ● C/C++
                     |
```

Python の API 親しみやすさ × Julia の数式記法 × Rust のゼロ GC ゼロコピー = **研究者が GC を意識せず数式を書き、C 並の速度で走る**。

**Rust 記法の honest friction — Python/Julia に勝てない定数コスト:**

| Friction | 原因 | 例 | 対価 |
|---|---|---|---|
| `()` in indexing | `Index` trait = 単一引数 | `a[(i,j)]` vs `A[i,j]` | 型安全 + 寿命保証 |
| `&` for borrow | 所有権モデル | `&a * &b` vs `A * B` | 明示的ゼロコピー + 再利用保証 |
| `?` for fallibility | `Result<T>` 型 | `a.solve(&b)?` vs `A \ b` | Silent NaN/Inf 排除 |
| `.emul()` for Hadamard | `*` = matmul 占有 | `a.emul(b)` vs `A * B` | matmul と静的に区別 |
| `.t()` vs `'` | Rust にポストフィックス演算子なし | `a.t()` vs `A'` | — (言語制約) |
| 型注釈 or turbofish | 型推論の限界 | `zeros::<f64>(3,3)` vs `zeros(3,3)` | コンパイル時型検証 |

**4/6 のフリクションは「安全性の対価」**。残り 2 つ (`.t()`, turbofish) は Rust の言語制約で、対価なしのコスト。全て**定数オーバーヘッド**であり、式が複雑になるほど相対的影響は減少する。

### 3.14 Complete example — PyTorch vs nabla

```python
# PyTorch (31 lines)
import torch                                    # 5 sec import
X = torch.randn(1000, 784, device='cuda')
W1 = torch.randn(784, 256, requires_grad=True, device='cuda')
W2 = torch.randn(256, 10, requires_grad=True, device='cuda')
optimizer = torch.optim.SGD([W1, W2], lr=0.01)

for epoch in range(100):
    H = (X @ W1).relu()                         # 2 ops, 2 allocs
    Y = H @ W2                                  # 1 op, 1 alloc
    loss = Y.pow(2).sum()                        # 2 ops, 2 allocs → GC later
    optimizer.zero_grad()                        # manual grad clear
    loss.backward()                              # grad_fn chain → GC later
    optimizer.step()
    torch.cuda.empty_cache()                     # GC ritual
    # Peak memory: ~3x actual need
    # GC may pause 10-100ms anywhere in this loop
```

```rust
// nabla (25 lines, same logic, zero GC)
use nabla::prelude::*;

let x: Tensor<f32> = randn(1000, 784);
let mut w1: Tensor<f32> = randn(784, 256);
let mut w2: Tensor<f32> = randn(256, 10);
let lr = 0.01_f32;

for _ in 0..100 {
    let tape = Tape::new();
    let xv = tape.var(&x);
    let w1v = tape.var(&w1);
    let w2v = tape.var(&w2);
    let h = fuse!(v.max(0.0); (&xv * &w1v));   // ReLU, 1 GPU kernel
    let y = &h * &w2v;
    let loss = fuse!(v.powf(2.0); y).sum();     // 1 GPU kernel
    loss.backward();
    w1 = &w1 - lr * &w1v.grad();                    // SGD update
    w2 = &w2 - lr * &w2v.grad();
}   // tape, grads, intermediates: ALL freed every iteration by Drop
    // Peak memory = exactly what's alive. Zero GC. Zero pause.
```

**行数**: PyTorch 31 → nabla 25 (−19%)。**GC**: PyTorch = random pauses → nabla = **zero**。**メモリ**: PyTorch ~3x → nabla = **1x (exact)**。

### 3.15 API naming conventions

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
| `cuda` | `Gpu` | `GpuStorage<f32>` (CUDA `CUdeviceptr`) | ✅ |
| `hip` | `Gpu` | `GpuStorage<f32>` (HIP `hipDeviceptr_t`) | ✅ |

All tensors use `Tensor<T>` = `Tensor<T, DefaultBackend>`.

### 4.2 GPU fallback prohibition

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
        gpu_wgpu.rs      gpu_cuda.rs       gpu_hip.rs
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

Backend implementations:
- **wgpu**: `wgpu::Device` + `wgpu::Queue` + `ComputePipeline` cache. `launch` = encode compute pass + submit
- **CUDA**: `CUcontext` + `CUmodule` cache (nvrtc-compiled PTX). `launch` = `cuLaunchKernel`
- **HIP**: `hipCtx_t` + `hipModule_t` cache (hiprtc-compiled). `launch` = `hipModuleLaunchKernel`

Singleton per backend:

```rust
fn get_context() -> &'static impl GpuContext {
    static CTX: OnceLock<ConcreteContext> = OnceLock::new();
    CTX.get_or_init(|| ConcreteContext::init())
}
```

### 5.3 GpuStorage

```rust
pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    buffer: <Context as GpuContext>::Buffer,  // wgpu::Buffer | CUdeviceptr | hipDeviceptr_t
    host_cache: Mutex<Option<Vec<T>>>,
}
```

- Memory layout: row-major flat array (all backends)
- `Send`/`Sync`: `unsafe impl` (all GPU buffer types are thread-safe)
- Readback: lazy `fill_cache()` — `map_async+poll` (wgpu) / `cuMemcpyDtoH` (CUDA) / `hipMemcpyDtoH` (HIP)

```
Host                              Device (GPU)
────                              ──────────────
zeros/fill/identity
  └──── compute kernel ───────→ Buffer₁
                                      │
         chained ops (zero transfer)  │
                                      ▼
                                  Buffer₂
                                      │
  .get(r,c)                           │
  ┌──── readback (lazy) ◄────────────┘
  ▼
host_cache (cached on first access)
```

### 5.4 Kernel sources — two codebases

**32 ops, same semantics, two shader languages:**

| Category | Ops | WGSL (wgpu) | CUDA/HIP C (shared) |
|---|---|---|---|
| Binary | add, sub, mul_elem, div_elem, scale | `elementwise_binary` | `elementwise_binary` (float4) |
| Unary | exp, ln, log1p, sin, cos, tanh, sqrt, abs, recip, erf, ceil, floor, round, powf, neg | `elementwise_unary` | `elementwise_unary` (float4 + fast math) |
| Matmul | tiled matmul (shared memory) | `matmul_tiled` TILE=16 | `matmul_tiled` TILE=16/32 |
| Reduction | sum, max, min | `reduction` | `reduction` (warp shuffle) |
| Arg reduction | argmax, argmin | `reduction_arg` | `reduction_arg` (warp shuffle) |
| Construction | zeros, fill, identity | `fill`, `identity` | `fill`, `identity` (float4) |
| Copy | clone, transpose | `copy`, `transpose` | `copy`, `transpose` |

WGSL shaders: embedded as `const &str` in `kernels_wgsl.rs`. Workgroup size: 256.

CUDA/HIP C kernels: embedded as `const &str` in `kernels_cu.rs`. Block size: 256 (tunable). Compiled at runtime via `nvrtc` / `hiprtc`. **Single source** — HIP C is source-compatible with CUDA C for standard math kernels.

**f32 vectorized memory access** — all f32 unary/binary/scalar kernels use `float4` (128-bit) loads/stores via `LDG.E.128` instructions. Each thread processes 4 elements, with a scalar tail loop for non-aligned remainders. f32 unary math uses CUDA fast-math intrinsics (`__expf`, `__logf`, `__sinf`, `__cosf`, `__fsqrt_rn`).

```c
// Example: k_exp_f32 with float4 vectorization + __expf fast math
extern "C" __global__ void k_exp_f32(const float* in, float* out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = LOAD_F4(in, i4);
        v.x = __expf(v.x); v.y = __expf(v.y); v.z = __expf(v.z); v.w = __expf(v.w);
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = __expf(in[j]); }
}
```

f64 kernels remain scalar (1 element/thread) — `double2` would require `__align__(16)` and shows minimal benefit on GH200.

### 5.5 Runtime compilation (CUDA/HIP)

```rust
// CUDA path (cuda_backend.rs)
fn compile_all_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
    let ptx = nvrtc::compile_ptx_with_opts(
        kernels_cu::KERNELS,
        nvrtc::CompileOptions { arch: Some(arch), ..Default::default() },
    )?;
    let module = cuModuleLoadData(&ptx);
    // cache all kernel functions in HashMap<String, KernelEntry>
}
```

- cudarc 0.19 driver + NVRTC API — no `libloading` needed
- Compiled PTX cached in `HashMap<String, KernelEntry>` (function handle + module)
- HIP path identical: `hiprtcCompileProgram` → `hipModuleLoadData`
- Architecture auto-detected via `cuDeviceGetAttribute` (compute_70–compute_90)

**Fusion JIT** — `fuse!` generates CUDA C expression strings at compile time. At runtime, `fuse_launch` builds a complete kernel source, JIT compiles via NVRTC/hiprtc, and caches by FNV-1a hash + type suffix:

```rust
// Generated by fuse! macro at compile time:
//   gpu_expr = "sqrt(fabs(log(exp(in0[i]))))"
//   kernel_hash = "a3f7b2c1e9d04568"
// At runtime, fuse_launch builds:
//   extern "C" __global__ void k_fused_a3f7b2c1e9d04568_f32(
//       const float* in0, float* out, unsigned n) {
//       unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
//       if (i < n) { out[i] = sqrt(fabs(log(exp(in0[i])))); }
//   }
// Compiled once, cached forever. Subsequent calls use cached CUfunction.
```

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

| Decision | Rationale | Evidence |
|---|---|---|
| Direct kernels (no CubeCL) | Fixed-rule 32 ops → 2 kernel codebases (WGSL + CUDA/HIP C) manageable. CubeCL adds abstraction layer without benefit for fixed ops | — |
| 4 backends (cpu/wgpu/cuda/hip) | wgpu = cross-platform, CUDA = NVIDIA native + f64 + tensor cores, HIP = AMD native + f64 | — |
| Build-time exclusive backend | CPU fallback is a performance bug source | — |
| Runtime kernel compilation | nvrtc/hiprtc: no CUDA/HIP SDK at build time, libloading dlopen | — |
| Handle-based GPU storage | Chained ops eliminate host↔device transfer (all backends) | — |
| TypeId dispatch | Backend trait sealed + `T: Scalar`. Avoids E0276 | — |
| wgpu f32 only, CUDA/HIP f32+f64 | wgpu: WGSL/Metal lacks f64. CUDA/HIP: native f64 | — |
| Embedded kernel strings | WGSL + CUDA/HIP C as `const &str`. Minimizes file count | — |
| No bytemuck | Self-contained `scalar_to_bytes`/`bytes_to_scalar` | — |
| `Mutex<Option<Vec<T>>>` cache | Readback is expensive; lazy on get/set | — |
| pollster for sync | wgpu async → sync bridge | — |
| Native Rust (no C++ wrapper) | Ownership-native tensor > FFI wrapper | kornia-rs 3–5x [2505.12425] |
| Trait-based AD | Drop-in with existing tensor types | ad-trait [2504.15976] |
| Recursive GEMM for GPU linalg | Reuse matmul_tiled, avoid dedicated solvers | Julia TRMM/TRSM [2504.13821] |
| Multi-versioned shaders | Close 10–30% cross-device gap | SGEMM portability [2507.15277] |
| Einsum canonicalization | Tuning reuse across equivalent expressions | 4.7x over JAX [2601.12220] |
| Iterator chains for parallelism | Avoid shared-mutable Rayon bottleneck | NPB-Rust [2502.15536] |
| Named axes (W11) | Eliminate transposition/broadcasting bugs at compile time | Tensor Considered Harmful [Chiang+ 2021] |
| Preparation pattern AD API | prep/execute split — amortize tape/sparsity analysis | DifferentiationInterface.jl [Blondel+ 2024] |
| `impl Scalar for Dual<T>` | Forward-mode AD as zero-change drop-in | ad-trait [2504.15976] ICRA 2025 |
| Mathematical error messages | Dimension errors read like math, not type system noise | dfdx [2302.05727] |
| Macro DSL absorbs verbosity | Julia 比 5-10x LOC gap を macro 記法で圧縮 | SciML survey 2024 |

---

## 12. Known limitations

| Limitation | Cause | Planned mitigation |
|---|---|---|
| No wgpu f64 | WGSL/Metal lacks f64 | Use `cuda` or `hip` backend for f64 |
| No GPU c32/c64 | No GPU backend supports complex | Compile error (by design) |
| GPU linalg: TRSM only | Recursive GEMM covers TRSM; full LU/Cholesky/QR not GPU-accelerated | Use `cuda`/`hip` backend + `gpu_trsm_lower` (W15) |
| Tiled matmul TILE fixed | wgpu: multi-version shaders (W14); CUDA/HIP: WMMA (W15) + register-tile (W18) | ✅ Resolved |
| `from_fn` requires host | Closures cannot run on GPU | Use `fuse!` for GPU |
| L2/L3 fuse! on GPU | L1 element-wise fusion CPU-only; L3 GEMM+activation CPU dispatch | GPU fused kernels require codegen extension |
| 2 kernel codebases | WGSL ≠ CUDA/HIP C → dual maintenance | Fixed 32 ops, rarely changes |
| Rayon shared-mutable bottleneck | Ownership prevents shared `&mut` | Iterator chains [2502.15536] |
| Tape-based AD overhead | Rc + dynamic graph, CPU-only | `GpuTape<T>` GPU-resident AD (W15); `#[nabla_grad]` source-transform (W18) |
| CAS phase-ordering | Greedy rewrite is order-dependent | ✅ E-graph equality saturation (W4–5) |
| Julia比5-10xコード量 | Rust構文の冗長性 (型注釈、借用、エラー処理) | Macro DSL 層で吸収 (§3)。最小限の型注釈で推論に委ねる |
| No REPL/インタラクティブ | コンパイル言語制約 | `rust-script` + `cargo watch` で即時フィードバック |

---

# Part III — Implementation Notes

## 13. Subsystem implementation

### 13.1 GPU

| Item | Status | Backend | Approach | Evidence |
|---|---|---|---|---|
| WMMA/MMA tensor cores | ✅ W15 | cuda/hip | `nvcuda::wmma::mma_sync` (Volta+), `rocwmma` (CDNA2+) | NVIDIA Ampere+, AMD CDNA+ |
| Warp shuffle reductions | ✅ W15 | cuda/hip | `__shfl_down_sync`, 6 warp_reduce functions, 8x reduction | Replaces shared-memory reduction |
| Subgroup matrix MMA (wgpu) | ✅ W18 | wgpu | Register-tile software MMA (`gen_matmul_register_tile`) | wgpu#5555 alternative |
| Recursive GEMM (GPU linalg) | ✅ W15 | cuda/hip | `gpu_trsm_lower`: base n≤32 → CPU, recursive quadrant | cuBLAS-comparable ~300 LOC [2504.13821] |
| Linear Layouts F₂ swizzle | ✅ W17 | all | `LinearLayout<N>`: identity/compose/apply/swizzle_for_tile | Triton-level codegen [2505.23819] |
| Deep GEMM+activation fusion | ✅ W17 | all | `detect_gemm_activation`, 10 activations, `matmul_fused` | [2602.11808] |
| bf16/f16 Scalar | ✅ W7 | all | `f16`/`bf16` types + backend-specific enable | wgpu `FLOAT16`, CUDA `__half`, HIP `__half` |
| f32 float4 vectorization | ✅ W19 | cuda/hip | 128-bit `float4` loads/stores + fast math intrinsics | 4× memory throughput |
| GPU kernel fusion (L1) | ✅ W19 | cuda/hip | `fuse!` → `cuda_expr()` → NVRTC/hiprtc JIT → cache | 3.8× speedup (4-op chain) |
| Caching memory allocator | ✅ W20 | cuda/hip | Best-fit dual-pool (small/large), 512B-aligned, block splitting, GC | PyTorch-equivalent overhead |
| Vectorized fuse codegen | ✅ W20 | cuda/hip | float4 in `fuse_kernel_source()`, `__ldg` prefetch, scalar tail | 3294 GB/s fuse bandwidth |
| Async execution pipeline | ✅ W20 | cuda/hip | Remove per-kernel sync, defer to readback only | sin 0.040ms = PyTorch |
| Single-element D2H readback | ✅ W20 | cuda/hip | `copy_element()` — 4-byte D2H instead of full tensor | exp+get 0.053ms = PyTorch |
| Fusion cost model | ✅ W20 | cuda/hip | `estimate_register_pressure()` in proc macro, `maxrregcount=120` | PyTorch Inductor-inspired |
| CUDA Graph capture/replay | ✅ W20 | cuda | `NablaCudaGraph`: `begin_capture` → `end_capture` → `launch` | 90–95% launch overhead reduction |
| Best-fit allocator + splitting | ✅ W20 | cuda/hip | Dual pools (small <1MB / large ≥1MB), over-alloc 2MB/20MB, GC 0.9 | PyTorch CUDACachingAllocator |
| `__ldg` read-only prefetch | ✅ W20 | cuda | `__ldg()` hints in fuse codegen (float4 + scalar + f64 paths) | 5–15% cache hit improvement |
| Mega-kernel fusion (L4) | 🔲 | cuda/hip | SM-level persistent kernel, cross-op pipelining | MPK [2512.22219], FlashFuser [2512.12949] |

**CUDA/HIP backend** — CUDA C and HIP C are source-compatible for standard math kernels. Single `kernels_cu.rs` contains all 32 ops as `const &str`. At init, `nvrtc`/`hiprtc` compiles to PTX/ISA, cached in `HashMap`. Key CUDA/HIP advantages over wgpu:

| Feature | CUDA/HIP native | wgpu equivalent |
|---|---|---|
| Tensor cores (WMMA) | ✅ `__wmma_*` / `__builtin_amdgcn_wmma_*` (W15) | ✅ register-tile software MMA (W18) |
| Warp shuffle | ✅ `__shfl_down_sync` (W15) | ❌ (WGSL limitation) |
| f64 compute | ✅ native | ❌ |
| Tunable block size | `<<<grid, block>>>` | Workgroup size in shader |
| Occupancy query | `cuOccupancyMaxPotentialBlockSize` | — |

**Kernel fusion** — `fuse!` proc macro has the full op chain at compile time. Generates a single fused CUDA/HIP C kernel (intermediates stay in registers) via NVRTC/hiprtc JIT compilation, with FNV-1a hash-based caching. CPU fallback: single `from_fn` pass.

| Level | Scope | Status | Paper |
|---|---|---|---|
| L1: Element-wise fusion | Fuse consecutive unary/binary ops | ✅ W19 GPU JIT | Liger [2410.10989] |
| L2: Reduction fusion | Fuse across reduction ops with loop-carried deps | ✅ W13 CPU | Neptune [2510.08726] |
| L3: GEMM+pointwise fusion | Fuse matmul + activation + normalization | ✅ W17 | Deep Kernel Fusion [2602.11808] |

**Fusion architecture:**

```
fuse!(c = exp(a) + ln(b))
  ↓ (compile time: nabla-macros)
  1. Parse AST → egg EqSat simplify (16-node FuseExpr, 15 rules)
  2. Check is_elementwise_fusible()
  3. cuda_expr() translates Rust AST → CUDA C: "expf(in0[i]) + logf(in1[i])"
  4. expr_hash() → FNV-1a hash for kernel name deduplication
  5. Emit __fuse_elementwise(inputs, gpu_expr, hash) call
  ↓ (runtime: cuda_backend.rs)
  6. fuse_kernel_source() wraps expression in __global__ kernel
  7. NVRTC compiles to PTX → cuModuleLoadData → cuFuncGetHandle
  8. Cache in HashMap<String, KernelEntry> by "k_fused_{hash}_{type}"
  9. cuLaunchKernel(grid, block, args)
```

**Benchmark results** (GH200 480GB, n=4096×4096, f32, W20 optimized):

| Workload | nabla (kernel-only) | PyTorch (kernel-only) | Gap | nabla (with readback) |
|---|---|---|---|---|
| exp | 0.127 ms (1059 GB/s) | 0.041 ms (3308 GB/s) | 3.1× | 0.053 ms (**≈ PyTorch**) |
| sin | 0.040 ms (3315 GB/s) | 0.041 ms (3295 GB/s) | **≈ equal** | — |
| tanh | 0.040 ms (3317 GB/s) | 0.041 ms (3310 GB/s) | **≈ equal** | — |
| add | 0.058 ms (2309 GB/s) | 0.058 ms (2309 GB/s) | **≈ equal** | — |
| fuse exp+sin (2-op) | 0.041 ms (3291 GB/s) | 0.081 ms (1653 GB/s, eager) | **nabla 2× faster** | — |
| fuse 4-op chain | 0.046 ms (2913 GB/s) | 0.041 ms (3287 GB/s, compile) | 1.1× | — |

Previous (pre-optimization, W19): exp 2.38ms, fuse 4-op 2.43ms → **19–51× improvement from W20 optimizations**. Gap vs PyTorch reduced from **46×** to **≈ parity** for most ops.

**Multi-versioned shaders** — compile workgroup-size variants, select at init:

| Parameter | Current | Target |
|---|---|---|
| Workgroup size (unary/binary) | 256 fixed | 64 / 128 / 256 / 512 |
| Matmul tile | 16 fixed | 8 / 16 / 32 |
| Selection | — | `device.limits().max_compute_workgroup_size_x` |

**Recursive GEMM decomposition** — express triangular ops as recursive GEMM:

```
TRSM(L, B):
  split L into quadrants L₁₁, L₂₁, L₂₂
  → TRSM(L₁₁, B₁)        // small triangular solve
  → GEMM(B₂ -= L₂₁ * B₁) // bulk compute on existing matmul_tiled
  → TRSM(L₂₂, B₂)        // recurse
```

Extends to LU, Cholesky, QR. Apple Silicon support becomes automatic [2504.13821].

**Tensor cores** — CUDA/HIP use WMMA intrinsics (W15); wgpu uses register-tile software MMA (W18):
- CUDA: `nvcuda::wmma::mma_sync` (Volta+), compiled via `compile_wmma_kernels()`
- HIP: `__builtin_amdgcn_wmma_f32_16x16x16_f16` (CDNA2+)
- wgpu: `gen_matmul_register_tile` WGSL generator, dispatched when m/n/k ≥ 64

**Linear Layouts** — model tensor-to-hardware memory mappings as binary matrices over F₂ [2505.23819]. A single generic algorithm derives provably optimal swizzling for any tile size. Also eliminates the quadratic case explosion in einsum codegen — all transpose combinations handled generically.

**Caching memory allocator** — block-based GPU memory pool to eliminate per-op `cudaMalloc`/`cudaFree` overhead, inspired by PyTorch's CUDACachingAllocator and STAlloc's spatio-temporal planning [2507.16274]. VibeTensor [2601.16238] demonstrates that a stream-ordered caching allocator with diagnostics can be built as a standalone module in a tensor runtime.

Design — two-tier pool with Rust ownership semantics:

| Component | Implementation |
|---|---|
| Pool structure | `HashMap<SizeClass, Vec<FreeBlock>>` in `CudaCtx`, guarded by `Mutex` |
| Size classes | Round up to next power-of-2: 256B → 512B → 1KB → … → 512MB |
| Large allocs (>512MB) | Bypass pool, direct `cudaMalloc`/`cudaFree` |
| Allocation | Pool hit → pop `FreeBlock` (O(1)). Miss → `cudaMalloc` |
| Deallocation | `CuBuffer::Drop` returns block to pool (not `cudaFree`) |
| Stream ordering | All pool blocks allocated on default stream; no cross-stream hazards |
| Safety | Rust `Drop` trait = deterministic pool return at scope exit |
| Memory pressure | `trim(target_bytes)` method: release blocks until pool ≤ target |
| HIP mirror | Identical design in `HipCtx` with `hipMalloc`/`hipFree` |
| Diagnostics | Pool stats: `allocated_bytes`, `cached_bytes`, `alloc_count`, `hit_count` |

```rust
struct FreeBlock {
    ptr: CUdeviceptr,
    size: usize,  // actual allocated size (≥ requested)
}

struct MemoryPool {
    bins: HashMap<usize, Vec<FreeBlock>>,  // key = size_class (power-of-2)
    allocated_bytes: usize,
    cached_bytes: usize,
}

impl MemoryPool {
    fn alloc(&mut self, size: usize, stream: &CudaStream) -> CuBuffer { ... }
    fn release(&mut self, ptr: CUdeviceptr, size: usize) { ... }
    fn trim(&mut self, target_bytes: usize) { ... }
}
```

Expected impact: eliminate ~90% of cudaMalloc calls. The GH200 benchmark showed ~46× gap vs PyTorch; allocation overhead is the primary bottleneck (each cudaMalloc ≈ 10–100μs, each kernel ≈ 2–5μs). STAlloc [2507.16274] reports 85% fragmentation reduction + up to 32.5% throughput improvement on LLM training workloads with a similar pool design.

**Vectorized fuse codegen** — extend `fuse_kernel_source()` to generate float4-vectorized fused kernels, matching pre-compiled kernel patterns. The Fused Kernel Library [2508.07071] demonstrates that C++17 metaprogramming can generate optimized fused kernels with vectorized SRAM-resident intermediates at compile time (2×–1000× speedups). For nabla, vectorization is applied at the JIT codegen level:

```c
// Target: vectorized fused kernel (float4 loads/stores, 128-bit LDG.E.128)
extern "C" __global__ void k_fused_abc123_f32(
    const float* in0, float* out, unsigned n) {
    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned i = i4 * 4;
    if (i + 3 < n) {
        float4 v = reinterpret_cast<const float4*>(in0)[i4];
        v.x = sqrtf(fabsf(logf(__expf(v.x))));
        v.y = sqrtf(fabsf(logf(__expf(v.y))));
        v.z = sqrtf(fabsf(logf(__expf(v.z))));
        v.w = sqrtf(fabsf(logf(__expf(v.w))));
        reinterpret_cast<float4*>(out)[i4] = v;
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = ...; }
}
```

| Optimization | Scalar (current) | Vectorized (target) |
|---|---|---|
| Memory access | 32-bit loads `in[i]` | 128-bit `float4` coalesced loads |
| Throughput | 1 element/thread | 4 elements/thread |
| Math intrinsics | Standard `expf` | Fast math `__expf`, `__logf`, `__sinf` |
| Grid size | `(n + 255) / 256` | `((n/4) + 255) / 256` |
| Tail handling | None needed | Scalar fallback for `n % 4 != 0` |

Requires modifying `fuse_kernel_source()` in `cuda_backend.rs` to emit float4 pattern. The macro's `cuda_expr()` already produces per-element expressions — vectorization replicates the expression for `.x`/`.y`/`.z`/`.w` components.

**Async execution & CUDA Graphs** — eliminate host-side kernel launch overhead, which FastUSP [2602.10940] identifies as the primary bottleneck on modern high-bandwidth GPU interconnects (not communication latency). PyGraph [2503.19779] demonstrates that CUDA Graphs can double the performance benefit of kernel fusion for ML workloads, and that cost-benefit analysis should guide selective graph deployment.

| Layer | Current | Target | Paper |
|---|---|---|---|
| Synchronization | `stream.synchronize()` after readback | Defer until `.get()` / `.to_vec()` | — |
| Kernel launch | Individual `cuLaunchKernel` per op | Batch via CUDA Graphs for repeated patterns | PyGraph [2503.19779] |
| Graph capture | Not supported | `cuStreamBeginCapture` → `cuGraphInstantiate` → `cuGraphLaunch` | Ekelund+ [2501.09398] |
| Iteration batching | N/A | Unroll N iterations into single graph for iterative solvers | Ekelund+ [2501.09398] |
| Memory in graphs | N/A | Pool-allocated buffers reused across graph replays | VibeTensor [2601.16238] |

CUDA Graph integration architecture:

```
// Phase 1: Capture
stream.begin_capture();
  kernel_1.launch(stream);
  kernel_2.launch(stream);
  kernel_3.launch(stream);
graph = stream.end_capture();
exec = graph.instantiate();

// Phase 2: Replay (amortized launch overhead)
for _ in 0..1000 {
    exec.launch(stream);  // single dispatch → all 3 kernels
}
```

Ekelund+ [2501.09398] (PDP 2025) shows that iteration batch unrolling into CUDA Graphs yields >1.4× speedup for iterative solvers, with an optimal batch size that is workload-independent per platform. PyGraph [2503.19779] further demonstrates compiler-automated CUDA Graph deployment that doubles benefits over manual usage in PyTorch2.

**Mega-kernel fusion** — long-term target based on Mirage Persistent Kernel (MPK) [2512.22219]. MPK introduces SM-level graph representation for cross-operator software pipelining within a single persistent mega-kernel, achieving up to 1.7× end-to-end LLM inference speedup. FlashFuser [2512.12949] extends fusion to compute-intensive operators via NVIDIA H100 Distributed Shared Memory (DSM), reducing memory access by 58% with 3.3× kernel speedup. For nabla, the relevant insight is that SRAM-resident fusion should not be limited to element-wise ops — inter-SM communication enables fusing across reduction boundaries.

#### 13.1.1 PyTorch optimization techniques (reverse-engineering reference)

PyTorch's GPU performance leadership stems from 5 interlocking subsystems. nabla must replicate or surpass each.

**A. CUDACachingAllocator** — PyTorch's memory pool (c10/cuda/CUDACachingAllocator.cpp):

| Parameter | PyTorch value | nabla current | nabla target |
|---|---|---|---|
| Allocation algorithm | Best-fit `std::set<Block*>` ordered by (stream, size, addr) | ✅ Best-fit dual-pool (W20) | ✅ Done |
| Rounding | 512B default, configurable `roundup_power2_divisions` | ✅ 512B-aligned (W20) | ✅ Done |
| Pool boundary | Small (<1MB) / Large (≥1MB) dual pools | ✅ Dual pool (W20) | ✅ Done |
| New block from CUDA | 2MB (small) / 20MB (large) | ✅ Over-allocate + split (W20) | ✅ Done |
| Block splitting | Threshold: 512B (small pool), 1MB (large pool) | ✅ Split at threshold (W20) | ✅ Done |
| Block coalescing | Address-ordered linked list, merge on free | Size-sorted insert | Coalesce adjacent |
| GC threshold | 0.9 (free when 90% used) | ✅ GC at 0.9 (W20) | ✅ Done |
| Expandable segments | CUDA VMM (`cuMemAddressReserve` + `cuMemMap`) | Not supported | VMM if available |
| Stream ordering | Per-stream free lists, CUDA events for cross-stream | Default stream only | Stream-aware pools |
| Overhead | 2–3% vs raw cudaMalloc | ~10% (mutex + size-class waste) | ≤3% |

Key insight: PyTorch's allocator does **block splitting** (a 4MB cached block can satisfy a 1MB request by splitting, keeping 3MB in pool). nabla's power-of-2 scheme wastes up to 50% memory. Switching to 512B-rounded best-fit with splitting is the highest-impact change.

**B. Kernel launch pipeline** — PyTorch minimizes per-kernel CPU overhead:

| Technique | PyTorch implementation | nabla applicability |
|---|---|---|
| CUDA Graphs | `torch.cuda.CUDAGraph` capture+replay (90–95% launch overhead reduction) | High: repeated fuse patterns in training loops |
| torch.compile | TorchInductor → Triton IR → auto-tuned GPU kernels | N/A (nabla compiles ahead-of-time via NVRTC) |
| Operator batching | `torch._foreach_*` fuses parameter updates into single kernel | High: could batch element-wise ops on multiple tensors |
| ATen dispatch | C++ virtual dispatch, ~0.5μs overhead per op | N/A (nabla uses Rust static dispatch = ~0ns) |
| Autograd engine | Backward ops batched by dependency level | Medium: GpuTape could batch gradient accumulation |

CUDA Graphs: capture a sequence of kernel launches into a graph, then replay with a **single CPU dispatch**. Eliminates 0.5–2μs per-kernel launch overhead. Break-even at ~50 replays. Critical for nabla's training loop optimization.

**C. TorchInductor fusion cost model** — decides which ops to fuse:

| Factor | Threshold / heuristic | nabla implication |
|---|---|---|
| Register pressure | Fuse if total registers ≤ 120/thread (255 max on SM) | Count ops: each transcendental ≈ 8–16 regs |
| Shared memory | Fuse if total ≤ 96KB (SM limit) | Element-wise ops use 0 shared mem → always fusible |
| Memory bandwidth | Fuse saves `(N_intermediate × 2 × sizeof(T) × n)` bytes | 4-op chain saves 6 × 64MB = 384MB → always worth fusing |
| Kernel launch overhead | Fuse if saved launches × 2μs > compute cost | For n≥4096², launch overhead is negligible |
| Fusion boundary | Break at reductions, scatter/gather, dynamic shapes | `fuse!` already limited to element-wise → correct boundary |
| Auto-tuning | 3–15 configs per kernel (BLOCK_SIZE, num_warps, num_stages) | nabla could auto-tune BLOCK_SIZE (64/128/256/512) |

nabla's fuse 4-op (0.080ms) vs PyTorch compile (0.041ms) gap analysis:
- PyTorch Triton uses **auto-tuned tile sizes** and **elements-per-thread** (1/2/4/8)
- nabla uses fixed BLOCK_SIZE=256, 4 elements/thread (float4)
- Adding auto-tuning for BLOCK_SIZE and elements-per-thread could close the 2× gap

**D. Async execution model** — PyTorch never syncs between ops:

| Sync point | PyTorch behavior | nabla current | nabla target |
|---|---|---|---|
| Kernel launch | Async, returns immediately | ✅ Async (W20) | ✅ Done |
| Memory allocation | Stream-ordered `cudaMallocAsync` (no global sync) | ✅ `malloc_async` (W20) | ✅ Done |
| `.item()` / `.get()` | Sync stream + copy **single scalar** (4 bytes) | ✅ Single-element D2H (W20) | ✅ Done |
| `.cpu()` / `.numpy()` | Sync stream + copy full tensor | Full tensor D2H via `ensure_cache` | Same (correct behavior) |
| Between ops | **Never syncs** — stream ordering guarantees correctness | ✅ No inter-op sync (W20) | ✅ Done |
| Backward pass | Async kernel chain, sync only at optimizer step | Not yet applicable | Future: GpuTape batch |
| `non_blocking=True` | H2D transfer on separate stream, overlap with compute | ✅ Multi-stream pipeline | Multi-stream pipeline |

Key insight: PyTorch's `.item()` copies only **1 scalar** (4 bytes), not the entire tensor. nabla's old `cached_get()` copied the **entire tensor** (64MB for 4096²) — 16 million× more data. The `copy_element()` fix eliminates this.

**E. Memory bandwidth optimization** — achieving near-theoretical throughput:

| Technique | PyTorch / Triton | nabla status | Impact |
|---|---|---|---|
| Vectorized loads (float4) | `tl.load` with 128-bit alignment | ✅ fuse! float4 (W20) | 4× memory throughput |
| Coalesced access | Guaranteed by contiguous layout | ✅ Row-major contiguous | Required for bandwidth |
| Persistent kernels | Grid-stride loop, launch ~SM count blocks | ⏸ Tested, reverted (regressed) | Original if/else optimal |
| Memory coalescing | 32-thread warp accesses 128 consecutive bytes | ✅ Implicit (contiguous) | 90% peak bandwidth |
| Shared memory tiling | 32–64KB tiles, bank-conflict-free padding | ✅ matmul only | Element-wise: not needed |
| Register blocking | 2–8 elements/thread in registers | ✅ float4 = 4 elem/thread | 80–90% bandwidth util. |
| `channels_last` layout | NHWC for conv kernels | N/A (no conv) | — |
| Prefetch / eviction | `tl.load(eviction_policy='evict_first')` | ✅ `__ldg` in fuse codegen (W20) | 5–15% cache improvement |

Bandwidth utilization analysis (GH200, theoretical peak ~4000 GB/s):

| Workload | nabla GB/s | PyTorch GB/s | % of peak (nabla) | % of peak (PyTorch) |
|---|---|---|---|---|
| exp | 1059 | 3308 | 26% | 83% |
| sin | 3315 | 3295 | **83%** | 82% |
| tanh | 3317 | 3310 | **83%** | 83% |
| add | 2309 | 2309 | **58%** | 58% |
| fuse exp+sin | 3291 | 1653 (eager) | **82%** | 41% |
| fuse 4-op | 2913 | 3287 (compile) | **73%** | 82% |

nabla now achieves **58–83%** of peak bandwidth (up from 22–46% pre-W20). sin/tanh/add match PyTorch exactly. Fuse exp+sin achieves 82% vs PyTorch eager 41% — **nabla 2× faster**. Remaining exp gap (26% vs 83%) is output allocation overhead.

**F. Remaining gap closure roadmap** (priority-ordered):

| Optimization | Status | Result | Technique source |
|---|---|---|---|
| 1. CUDA Graphs for fuse chains | ✅ W20 | `NablaCudaGraph` capture/replay API | PyTorch `CUDAGraph`, PyGraph [2503.19779] |
| 2. Best-fit allocator with splitting | ✅ W20 | Dual-pool, 512B-aligned, over-alloc, GC 0.9 | PyTorch CUDACachingAllocator |
| 3. Prefetch hints (`__ldg`) | ✅ W20 | `__ldg()` in fuse codegen (float4/scalar/f64) | CUDA `__ldg()` intrinsic |
| 4. Fusion cost model | ✅ W20 | `estimate_register_pressure()`, `maxrregcount=120` | TorchInductor heuristics |
| 5. Multi-stream pipeline | ✅ | `copy_stream` + CUDA events for H2D/compute overlap | PyTorch DataLoader pattern |
| 6. Mega-kernel fusion (L4) | 🔲 | — | MPK [2512.22219], FlashFuser [2512.12949] |

**効果なし（実装・ベンチマーク済、リバート）:**

| Optimization | Result | 理由 |
|---|---|---|
| Auto-tune BLOCK_SIZE | `cuOccupancyMaxPotentialBlockSize` → 2–3× 性能悪化 | GH200では固定 BLOCK_SIZE=256 が最適。Occupancy APIの推奨値は本ワークロード（float4 element-wise）に不適合 |
| Persistent grid-stride kernels | Grid capping (`sm_count×4`) → 2–3× 性能悪化 | n≥4096² では全SM飽和済。Grid-stride ループのオーバーヘッドが if/else 分岐パターンを上回る |

---

### 13.2 Autograd

| Item | Status | Approach | Evidence |
|---|---|---|---|
| GPU-resident AD | ✅ W15 | `GpuTape<T>`: 12-op enum, buffer registry, backward via GPU kernels | [2509.00406] |
| Source-transform AD | ✅ W18 | `#[nabla_grad]` proc macro: Dual<T> lifting, emits `f_grad(x) -> (T, T)` | Enzyme [SC'21] alternative |

**Dual-path AD** — based on ad-trait [2504.15976]:

| Mode | Use case | Implementation |
|---|---|---|
| Forward (dual number) | Few inputs, many outputs (Jacobian columns) | `Dual<T> = (value: T, deriv: T)` overloading `Scalar` trait |
| Reverse (tape) | Many inputs, few outputs (loss → gradients) | Current tape-based approach |
| Forward + SIMD | Batch Jacobian columns | SIMD lanes as independent seed vectors |

Forward-mode requires only trait extension (`impl Scalar for Dual<T>`) — all existing tensor ops work unchanged.

**GPU-resident AD** — based on [2509.00406]. Locality-aware reverse-mode keeps the computation graph on-device, avoiding the GPU→CPU→GPU roundtrip that `Rc<RefCell<Tape>>` forces. Gradient tape should be buffer-based (GPU buffer for op records) with backward pass as GPU compute kernels. Backend-agnostic via `GpuContext` trait.

**Source-transform AD** — `#[nabla_grad]` attribute macro (W18): shadows argument with `Dual::new(x, 1.0)`, re-executes function body at call site, emits `f_grad(x: T) -> (T, T)`. Avoids LLVM-IR dependency; works on stable Rust via proc macro source transformation.

---

### 13.3 ODE

| Item | Status | Approach | Evidence |
|---|---|---|---|
| DAE support | ✅ W7 | `bdf1` implicit solver: semi-explicit index-1 DAE | diffsol [JOSS 2026] |
| Parallel-in-time (Parareal) | ✅ W18 | `parareal_solve`: rayon fine propagator, sequential coarse correction | [2510.07672] |
| Symplectic integrators | ✅ W8 | Störmer-Verlet for Hamiltonian systems | — |

**Exponential integrators** — strictly superior to RK4/DP for stiff systems:

| Method | Paper | Advantage |
|---|---|---|
| IF Euler (Integrating Factor) | [2412.01181] | Succeeds on Van der Pol where RK4 diverges. Explicit → GPU-friendly |
| METD (Matrix Exponential TD) | [2406.13761] | Matrix-valued ODE (Lyapunov, Riccati, graph neural ODE). Provable order-p |

IF Euler: `y_{n+1} = e^{hL} y_n + h φ₁(hL) N(y_n)`. Requires matrix exponential (compose with nabla's dense linalg) but avoids implicit Newton iteration.

Reference architecture: diffsol DiffSL pattern (Julia-inspired DSL → AD Jacobian → JIT). nabla's approach is pure Rust closures + trait-based AD.

---

### 13.4 CAS

All planned CAS items implemented (W4–5/7/11/12). No remaining items.

**E-graph equality saturation** — stores ALL equivalent forms simultaneously, extracts globally optimal:

| Approach | Paper | Implementation |
|---|---|---|
| E-graph + MCTS extraction | [2410.05534] | `egg` 32 rules + `CDiff` + `diff_simplify` (W4–5/7) |
| Slotted E-Graphs | [PLDI 2025] | `CDiff([Id;2])` 2-arg form + `IsDifferentVar` condition (W12) |
| EqSat for high-level IR | [2502.17075] | `FuseExpr` 16-node language, 15 rules in `fuse!` (W11) |
| egglog (Datalog + EqSat) | [2304.04332] | Incremental fixpoint, lattice analyses. Rust crate production-ready |

---

### 13.5 Sparse

| Item | Status | Approach | Evidence |
|---|---|---|---|
| GPU sparse BCSR | ✅ W16 | `BcsrMatrix<T>`: `from_sparse`, CPU SpMM, `WGSL_BCSR_SPMM` kernel source | [2501.09251] 2.52x over cuSPARSE |
| Mixed-precision sparse | ✅ W16 | `mixed_spmm_f64`: f32 preconditioner + f64 residual refinement | [2412.19322] 2–4x |

**GPU sparse formats**:

| Format | Paper | Performance |
|---|---|---|
| BitTCF (Bit Tensor Core Format) | [2501.09251] PPoPP 2025 | 2.52x avg over cuSPARSE (RTX 4090), 1.91x (A800) |
| BCSR (Blocked CSR) + MMA | [2408.11551] SC 2024 | Up to 125x over cuSPARSE for unstructured matrices |

BCSR blocks can target `subgroup_matrix` MMA intrinsics (§13.1) when available. CSC→BCSR preprocessing is one-time cost.

**Mixed-precision iterative refinement**:

```
Preconditioner: FP16 sparse factorization (fast, approximate)
Correction: FP64 residual refinement (accurate)
Iterate until ||r|| < ε
```

API-compatible upgrade to `cholesky_solve()` and `solve()`. 2–4x speedup [2412.19322].

---

### 13.6 einsum!

| Item | Status | Approach | Evidence |
|---|---|---|---|
| Linear Layouts codegen | ✅ W17 | `LinearLayout<N>` F₂ algebra: `to_wgsl_swizzle_fn` for generic transpose | [2505.23819] |

**Canonicalization** — mathematically equivalent einsum expressions map to a canonical form at compile time:

| Benefit | Mechanism |
|---|---|
| Tuning reuse | `c[i,j] = a[i,k] * b[k,j]` ≡ `c[j,i] = b[j,k] * a[k,i]` → same kernel |
| CSE | Repeated subexpressions in compound chains detected at compile time |
| Impact | 4.7x geomean speedup over JAX on TCCG benchmark [2601.12220] |

`classify()` graph-normalization pass sorts indices, canonicalizes operand order, and normalizes batch dims (W10).

**Contraction path** — greedy compile-time path optimizer for NdTensor fallback (W5):

| N-D strategy | Implementation |
|---|---|
| Contraction order | Greedy slot-based optimizer (minimize intermediate index count) |
| Tiling | L1 tiled contraction (tile=64) with path annotation (W13) |

**Einsum-to-dataflow** — long-term target based on Sigma [ASPLOS 2023]. Dataflow graph identifies data reuse per dimension → optimal loop ordering + tiling + fusion. Current 7-pattern `classify()` is tier 1; dataflow is the generalization.

---

### 13.7 Notation & DX (Developer Experience)

| Item | Status | Approach | Evidence |
|---|---|---|---|
| Compile-time shape algebra | ✅ W12 | `StaticMatrix<T,R,C>` const-generic shape checks + test suite | dfdx [2302.05727] compile-time shapes |
| Named axes | ✅ W11 | `Tensor<T,B,Axes=()>` PhantomData, `axis!`/`named_zeros!` macros | Tensor Considered Harmful [Chiang+ 2021] |

**Named axes** (W11) — highest-leverage notation improvement per "Tensor Considered Harmful" (Chiang+ 2021):

| Problem | Positional indexing | Named axes (W11) |
|---|---|---|
| Transposition bugs | Silent wrong result | Compile error on axis mismatch |
| Broadcasting ambiguity | Shape must match positionally | Named dims matched semantically |
| einsum readability | Index letters (`i`, `j`, `k`) | Meaningful names (`batch`, `seq`, `dim`) |

`Tensor<T, B, Axes=()>` — `PhantomData<fn() -> Axes>` (contravariant), zero runtime cost. `with_axes()` attaches axis names; erased at runtime.

```rust
let x = named_zeros!(batch: 32, seq: 128, dim: 512);
let w = named_zeros!(dim: 512, heads: 8);
let y = einsum!(y[batch, seq, heads] = x[batch, seq, dim] * w[dim, heads]);
// Compile error if axis name mismatch
```

**Mathematical error messages** — runtime errors use `"nabla: cannot multiply M×N by P×Q"` format (W3). Compile-time errors from `StaticMatrix` const generics produce `expected StaticMatrix<_, K, _>, found StaticMatrix<_, K2, _>` — maximally precise on stable Rust. Full custom diagnostic (`proc_macro_diagnostic`) requires nightly.

**Tensor manipulation ops** — PyTorch API 名をデファクトとして採用 (W8/W10):

| Operation | nabla API | 設計方針 |
|---|---|---|
| Reshape | `a.reshape([m, n])` | Zero-copy when contiguous, else copy |
| View (alias) | `a.view([m, n])?` | Zero-copy only, fail if non-contiguous |
| Permute | `t.permute([1, 0, 2])` | NdTensor, lazy stride reorder |
| Cat | `cat([&a, &b], 0)` | Free function, alloc 1 |
| Stack | `stack([&a, &b], 0)` | Free function, new dim |
| Squeeze | `t.squeeze(0)` | Remove dim of size 1 |
| Unsqueeze | `t.unsqueeze(0)` | Add dim of size 1 |
| Flatten | `a.flatten()` | Zero-copy if row-major |
| Chunk | `a.chunk(n, 0)` | Zero-copy slices |

**Axis-specific reduction** — `.sum_axis(d)` / `.mean_axis(d)` + keepdim variants (W10):

```rust
let batch_mean = x.sum_axis(0);              // (B, D) → (D,)
let kept = x.sum_axis_keepdim(1);            // (B, D) → (B, 1)
let normed = &x / &x.sum_axis_keepdim(1);   // softmax denominator pattern
```

---

## 14. Roadmap

### 14.1 Remaining

| Item | Priority | Status | Rationale |
|---|---|---|---|
| Multi-stream pipeline | 🟡 Medium | 🔲 | Overlap H2D transfer + compute on separate streams |
| Mega-kernel fusion (L4) | 🔵 Low | 🔲 | SM-level persistent mega-kernel for cross-op pipelining (MPK [2512.22219]) |

**Completed in W20:**
- ✅ Auto-fusion cost model → `estimate_register_pressure()` in proc macro, `maxrregcount=120`
- ✅ CUDA Graph capture/replay → `NablaCudaGraph` API (`begin_capture` / `end_capture` / `launch`)
- ✅ Best-fit allocator + splitting → Dual-pool (small/large), 512B-aligned, over-alloc 2/20MB, GC 0.9
- ✅ `__ldg` prefetch hints → float4 + scalar + f64 paths in fuse codegen

**効果なし（実装・ベンチマーク済、リバート）:**
- ✗ Auto-tune BLOCK_SIZE → `cuOccupancyMaxPotentialBlockSize` が 2–3× 性能悪化。固定256が最適
- ✗ Persistent grid-stride → Grid capping が 2–3× 性能悪化。if/else パターンが n≥4096² で最適

### 14.2 Implemented

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
