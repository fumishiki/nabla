# nabla — Notation Reference

> Ground truth specification: [spec.md](spec.md). Related: [directory.md](directory.md)

`use nabla::prelude::*;` — all types, traits, macros, free functions available.

**CAS symbols**: `diff`, `diff_simplify`, `eval`, `simplify` are included in `nabla::prelude::*`. For additional CAS functions (`gradient`, `jacobian`, `hessian`, `substitute`, `eval_tensor`), use `nabla::cas::*`. `sym!` is a proc macro (also in prelude). Note: `^` in `sym!` means exponentiation (not Rust's bitwise XOR).

**Design goal**: Python's ease of writing × Julia's math notation × Rust's zero-GC safety.

---

## §1 Types & Conventions

```rust
// Scalar types
T: f32 | f64 | c32 | c64   // ComplexField + MathOps + ReductionOps + Copy + Send + Sync

// Tensor types
Tensor<T>                    // = Tensor<T, DefaultBackend>; row-major flat storage
StaticMatrix<T, R, C>        // stack-allocated; R, C: const usize
NdTensor<T>                  // N-D CPU-only; flat Vec<T> row-major
DynTensor                    // enum{F32,F64,C32,C64}(Tensor<_>); runtime scalar dispatch

// Indexing: 0-indexed, tuple for multi-dim
a[(i, j)]  a[(i, j)] = v  a.slice(0..3, 1..4)

// Error handling: Result<T, Error> — no silent NaN
// Example entry point
#[nabla::main(cpu)] fn main() { ... }   // auto #[cfg], Result, fallback
// Kernel naming: k_{op_name}_f32 / k_{op_name}_f64
// Display: `format!("{}", tensor)` default, `format!("{:.4}", tensor)` precision
// Conv layout: NCHW; H_out = (H + 2*padding - dilation*(kH-1) - 1) / stride + 1
```

**Naming conventions**:

| Category | Convention | Examples |
|---|---|---|
| Construction | Free function, type-inferred | `zeros`, `eye`, `rand`, `randn`, `linspace`, `arange` |
| Unary op | Method, ≤ 5 chars | `.t()`, `.h()`, `.sin()`, `.exp()`, `.abs()` |
| Binary op | Operator or short method | `a * b`, `a.emul(b)`, `a.ediv(b)` |
| Factorize | Short method + `?` | `.lu()?`, `.qr()`, `.chol()?`, `.svd()?` |
| Solve | Verb + `?` | `.solve(&b)?`, `.lstsq(&b)?`, `.inv()?` |
| Reduce | Verb | `.sum()`, `.max()`, `.argmax()`, `.sum_axis(d)` |
| In-place | `_` suffix (PyTorch convention) | `.mm_(&a, &b)`, `.add_(&b)` |
| AD | PyTorch-familiar | `tape.var(x)`, `loss.backward()`, `x.grad()` |
| Module | `Module<T,B>` trait | `.forward()`, `.parameters()`, `.state_dict()` |
| Optimizer | `Optimizer<T,B>` trait | `AdamW::new(lr, shapes).step(params, grads)` |

**Rust friction vs safety**:
`a[(i,j)]` (tuple required by `Index` trait), `&a * &b` (explicit zero-copy), `a.solve(&b)?` (Result eliminates silent NaN), `.emul()` (statically distinguishes from matmul). All impose **constant overhead** — the price of safety.

Naming principle: if the name matches NumPy/PyTorch, keep it. If Julia is shorter, lean toward Julia. Otherwise, pick the **shortest unambiguous name**.

---

## §2 Quick Reference — Math → Python → Julia → nabla

| Math | Python | Julia | nabla | nabla advantage |
|---|---|---|---|---|
| $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ | `np.array([[1,2],[3,4]])` | `[1 2; 3 4]` | `mat![[1, 2], [3, 4]]` | Compile-time shape check |
| $A_{ij}$ / $A_{ij} = v$ | `A[i,j]` | `A[i,j]` | `a[(i, j)]` | 0-indexed, type-safe |
| $A_{1:3, 2:4}$ | `A[0:3, 1:4]` | `A[1:3, 2:4]` | `a.slice(0..3, 1..4)` | Owned copy |
| $A^\top$ / $A^*$ | `A.T` / `A.conj().T` | `A'` | `a.t()` / `a.h()` | 3-4 chars |
| $AB$ | `A @ B` | `A * B` | `a * b` | Same as Julia |
| $A \circ B$ | `A * B` | `A .* B` | `a.emul(b)` | `e`-prefix |
| $Ax = b$ | `np.linalg.solve(A,b)` | `A \ b` | `a.solve(&b)?` | Result — no silent NaN |
| $\sin(A)$ | `np.sin(A)` | `sin.(A)` | `a.sin()` | Shortest |
| $y = \sin(x)^2$ fused | `torch.sin(x)**2` | `@. sin(x)^2` | `fuse!(x.sin().powf(2.0))` | GPU auto-fusion |
| $C = AB$ (einsum) | `np.einsum('ik,kj->ij',A,B)` | `@einsum` | `einsum!(c[i,j] = a[i,k] * b[k,j])` | 7 patterns + spanned errors |
| $\nabla_x L$ | `loss.backward()` | `gradient(f, x)` | `loss.backward(); x.grad()` | PyTorch-familiar + zero GC |
| $\frac{df}{dx}$ symbolic | SymPy `diff(f, x)` | Symbolics.jl | `diff(&f, "x")` | Built-in CAS |
| $\dot{y} = f(t,y)$ | `scipy.solve_ivp` | DiffEq.jl | `dormand_prince(f, y0, t, dt)` | Built-in ODE |
| $\nabla^2 u$ | manual loop | `@tullio` | `stencil!(out[i,j] = ...)` | Auto bounds |
| $[A; B]$ vcat | `np.vstack` | `[A; B]` | `vcat!(a, b)` | Julia style |
| $\langle u,v \rangle$ | `np.dot(u,v)` | `dot(u,v)` | `dot(&u, &v)` | Scalar return |
| $\det(A)$ | `np.linalg.det(A)` | `det(A)` | `a.det()?` | Result |

---

## §3 Construction & Indexing

| Operation | nabla | Python | Julia |
|---|---|---|---|
| Matrix literal | `mat![[1, 2], [3, 4]]` or `mat![1, 2; 3, 4]` | `np.array(...)` | `[1 2; 3 4]` |
| Zeros / Ones / Fill | `zeros(m, n)` / `ones(m, n)` / `fill(m, n, val)` | `np.zeros` | `zeros(m,n)` |
| Identity | `eye(n)` | `np.eye(n)` | `I(n)` |
| Random | `randn(m, n)` / `rand(m, n)` | `np.random.randn` | `randn(m,n)` |
| From function | `from_fn(m, n, \|r, c\| expr)` / `par_from_fn(m, n, \|r, c\| expr)` | `np.fromfunction` | — |
| Range / Linspace | `arange(0.0, 1.0, 0.1)` / `linspace(0.0, 1.0, n)` | `np.arange` | `0.0:0.1:1.0` |
| Static (stack) | `StaticMatrix::<f64,3,3>::zeros()` | — | `SMatrix{3,3}` |
| N-D | `nd_zeros(&[d0, d1, d2])` | `np.zeros(shape)` | `zeros(d0,d1,d2)` |
| Cat / Block | `vcat!(a, b)` / `hcat!(a, b)` / `block![[a,b],[c,d]]` | `np.vstack` | `[A; B]` / `[A B; C D]` |
| Diagonal | `diagm(&v)` | `np.diag(v)` | `diagm(v)` |
| Column vector | `zeros_vec(n)` / `ones_vec(n)` / `rand_vec(n)` / `randn_vec(n)` | — | — |
| Seed control | `set_seed(42)` / `clear_seed()` | `np.random.seed` | `Random.seed!` |

**Indexing**: `a[(i,j)]` read/write (0-indexed), `a.slice(r, c)` / `a.slice_rows(r)` / `a.slice_cols(c)` owned copy, `a.view_slice(r, c)` → `TensorView<'_,T,B>` zero-copy borrow. N-D: `t[&[i,j,k]]`.

---

## §4 Arithmetic & Element-wise

`a * b` = move (consumed), `&a * &b` = borrow (reusable). Variable ops also support all 4 ownership combinations: `var + var`, `&var + var`, `var + &var`, `&var + &var`.

| Math | nabla (owned) | nabla (borrowed) |
|---|---|---|
| $A + B$ / $A - B$ | `a + b` / `a - b` | `&a + &b` |
| $AB$ (matmul) | `a * b` | `&a * &b` |
| $A \circ B$ / $A \oslash B$ | `a.emul(&b)` / `a.ediv(&b)` | (always `&self, &Self`) |
| $cA$ / $A/c$ | `c * a` / `a / c` | `c * &a` / `&a / c` |
| $A^\top$ / $A^*$ | `a.t()` / `a.h()` | — |
| $\langle u,v \rangle$ / $uv^\top$ | `u.dot(&v)` / `u.outer(&v)` | `dot(&u, &v)` |
| $A \otimes B$ | `a.kron(&b)` | `kron(&a, &b)` |
| In-place | `a += &b`, `a *= α`, `c.mm_(&a, &b)`, `a.axpy_(α, &x)` | — |

**Broadcasting**: `&a + &b` automatically infers shapes `(m,n)+(1,n)` / `(m,n)+(m,1)` / `(m,n)+(1,1)`. Owned `a + b` requires exact shape match (move optimization).

**Element-wise math** (all GPU float4 vectorized):
`exp` `ln` `log1p` `log2` `log10` `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `asinh` `acosh` `atanh` `sqrt` `abs` `recip` `ceil` `floor` `round` `powf` `neg` `sign` `rem`. `epow(&b)` tensor-tensor power. `hadamard(&rhs)` = `emul` alias (deprecated since 0.1.0, use `emul`).

**Reductions**: `.sum()` `.max()` `.min()` `.mean()` `.prod()` `.argmax()` `.argmin()` + `_axis(d)` variants + `keepdim`. `cumsum(dim)` `cumprod(dim)` (GPU: Blelloch scan). `var_axis_ddof(axis, ddof)`.

---

## §5 Linear Algebra & Sparse

### Dense (CPU)

45+ methods via `LinalgExt` trait (f32/f64, f32→f64 internal promotion):

| Math | nabla | Math | nabla |
|---|---|---|---|
| $Ax = b$ | `a.solve(&b)?` | $A^{-1}$ | `a.inv()?` |
| $PA = LU$ | `a.lu()?` | $A = QR$ | `a.qr()` |
| $A = LL^\top$ | `a.chol()?` | $A = LDL^\top$ | `a.ldl()?` |
| $A = U\Sigma V^\top$ | `a.svd()?` | $\sigma_i$ | `a.svdvals()?` |
| $\lambda, V$ (sym) | `a.sym(Side::Lower)?.eigh()?` | $\det(A)$ | `a.det()?` |
| $\log|\det|$ | `a.logdet()?` | null(A) | `a.null_space(tol)?` |
| $\kappa_p(A)$ | `a.cond_p(p)?` | col space | `a.orth(tol)?` |
| $Ax = \lambda Bx$ | `a.geig(&b)?` | $e^A$ / $\log A$ | `a.expm()?` / `a.logm()?` (2x2 block) |
| $\lambda_i, T, Q$ | `a.eig_into()?` | $A = QTQ^\top$ | `a.schur_decomp()?` |
| $AX + XB = C$ | `sylvester(&a,&b,&c)?` (2x2 block) | $AXB + X = C$ | `discrete_sylvester(&a,&b,&c)?` (2x2 block) |

Additional: `sqrtm` `lyapunov` `discrete_lyapunov` `care`/`continuous_riccati` `balance` `circulant` `toeplitz` `vandermonde`/`vandermonde_rect` `polar` `hessenberg` `frechet_deriv` `solve_tridiag`.

Structural: `Diagonal::new(v)`, `Symmetric::new(a, Side)`, `Triangular::new(a, TriKind)`.
Factorization reuse: `lu.solve(&b)`, `lu.inverse()`, `lu.reconstruct()`.
GPU: `gpu_trsm_lower` (recursive GEMM TRSM) only.

### Sparse

```rust
let s = sparse(m, n, &[(0,0,1.0), (1,2,3.0)])?;
let x = s.solve(&b)?;  let c = s * &d;  // SpMM via Mul trait
let tri = SparseMatrix::tridiag(n, sub, diag, sup)?;  // tridiagonal
let id = speye(n)?;                                    // sparse identity
```

CPU: CSC. GPU: `BcsrMatrix<T>` BCSR + `WGSL_BCSR_SPMM` kernel + `mixed_spmm_f64`.

---

## §6 Macros

### GPU-accelerated

These macros generate GPU kernels or compile to GPU-capable operations:

| Macro | Purpose | Example |
|---|---|---|
| `math!` | Auto-borrow idents/fields/indices | `math!(a * b + self.bias)` |
| `fuse!` | Element-wise fusion → 1 GPU kernel | `fuse!(x.sin().powf(2.0))` |
| `mega_fuse!` | Multi-output DAG fusion | `mega_fuse!(a+b; prev.exp(); inputs: a, b)` |
| `einsum!` | Einstein summation (7 patterns) | `einsum!(c[i,j] = a[i,k] * b[k,j])` |
| `mat!` | Matrix literal | `mat![[1,2],[3,4]]` |
| `vcat!` / `hcat!` / `block!` | Concatenation | `vcat!(a, b)` |
| `sym!` | Symbolic expression | `sym!(sin(x^2) + cos(y))` |

### CPU-only

These macros run on CPU via closure/loop — no GPU kernel:

| Macro | Purpose | Example |
|---|---|---|
| `stencil!` | Finite difference (auto bounds) | `stencil!(out[i,j] = -4*u[i,j] + ...)` |
| `map!` / `map_!` | Closure map / in-place | `map!(\|x\| f(x), &a)` |
| `par_map!` | Rayon parallel map | `par_map!(\|x\| x * 2.0, &a)` |

### Utility / meta

No GPU/CPU distinction — these handle code generation, module wiring, and training ceremony:

| Macro | Purpose | Example |
|---|---|---|
| `ad!` | Autograd forward+backward+grad | `ad!(Cpu; w=init => \|tape\| { ... })` |
| `named!` | Named tuple construction | `named!(x: f64 = 1.0, y: f64 = 2.0)` |
| `vars!` | Batch-create tracked Variables | `vars!(tape; w1 = expr1, w2 = expr2)` |
| `sequential!` | Build Sequential from layers | `sequential!(layer1, layer2)` |
| `cas_vars!` | CAS variable bindings | `cas_vars!{ x: 1.0, y: 2.0 }` |
| `impl_layer!` | Module impl from params + forward | `impl_layer! { Linear { weight; bias } forward(x) { ... } }` |
| `impl_module_params!` | Module boilerplate (7 methods) | `impl_module_params!(weight; bias)` |
| `vec_unpack!` | Column vector destructure | `vec_unpack!(y, x, y_val, z)` |
| `axis!` | Zero-sized marker types for named axes | `axis!(Batch, Seq, Hidden)` |
| `generated!` | Const-generic specialization | `generated!(fn_name, T, R, C)` |
| `named_zeros!` | Typed axis tensor constructor | `named_zeros!(Batch, Seq; 4, 8)` |
| `train_step!` | Training ceremony absorption | `train_step!(model, optimizer, tape, \|x, out\| loss_expr)` |
| `#[derive(Module)]` | Auto-generate Module trait methods | `#[derive(Module)] struct MyLayer { #[param] weight: Tensor<T,B>, ... }` |
| `#[nabla::main]` | Example entry point | `#[nabla::main(cpu)] fn main() { ... }` |

### einsum! pattern classification

| Pattern | Math | Codegen |
|---|---|---|
| `c[i,j] = a[i,k] * b[k,j]` | C = AB | `matmul_into` |
| `y[i] = a[i,k] * x[k]` | y = Ax | `matmul_into` (Mx1) |
| `c[i,j] = a[i,j] * b[i,j]` | C = A ∘ B | `emul` |
| `s = a[i,i]` | tr(A) | diagonal loop |
| `c[i,j] = a[i] * b[j]` | ab^T | `from_fn` |
| `c[b,i,j] = a[b,i,k] * m[b,k,j]` | batch matmul | batch GEMM |
| General N-D | — | NdTensor loop |

Index classes: **Batch** (LHS + all RHS), **Free** (LHS + subset RHS), **Contraction** (RHS only). `syn::Error::new_spanned` for compile-time error spans.

### math! auto-borrow

`math!(a * b + c)` → `&a * &b + &c`. AST walk auto-inserts `&` for `Expr::Path` (simple ident), `Expr::Field` (`self.weight`), and `Expr::Index` (`arr[0]`). `Expr::Reference` is passed through. Closure params are excluded (`math!(a.map(|x| x * 2))` does not borrow `x`).

### fuse! auto-capture

`fuse!(expr)` — tensor variables are auto-detected from the AST. Explicit form `fuse!(expr; x, y)` is also supported. Non-fusible expressions emit a compile-time `eprintln!` warning and fall back to tensor-level evaluation.

GPU: `fuse!` AST → egg EqSat (18 IEEE-754 safe rules) → `cuda_expr()` → NVRTC/hiprtc JIT → FNV-1a cache.
CPU: single `from_fn` loop.

---

## §7 Calculus & AD

### Reverse-mode AD — PyTorch `.backward()` compatible

```rust
let tape = Tape::new();
let x = tape.var(tensor_x);
let loss = (&x * &w).exp().sum();
loss.backward();
let dx = x.grad();  // tape, grads ALL freed by Drop
```

47 ops (21 core + 26 extended). `tape.var()` → record → `backward()` → `x.grad()`.
`backward()` returns `Result` (requires scalar (1,1) output; returns `Err` for non-scalar). For non-scalar outputs use `backward_with(grad_output)` (custom seed gradient, shape must match output). `backward_unchecked()` skips NaN/Inf validation for performance-critical paths.
`grad()` returns `Result<Tensor>` using `Error::NoGradient` (zero-alloc) — forces explicit error handling when gradient is missing. `grad()` / `gradient()` / `gradient_prep()` are generic over `B: Backend` (not CPU-only).
`Variable::retain_grad()` allows inspecting intermediate node gradients after backward (by default only leaf gradients are retained). `variable()` / `var()` now return `Result` (no-grad scope safety).
Trade-offs: ✅ PyTorch-familiar, deterministic memory. ❌ Rc overhead, single-thread.

### Forward-mode AD — `Dual<T>`

```rust
let x = Dual::new(2.0, 1.0);
let y = (x * x).sin();  // y.dual = dy/dx = 2cos(4)
```

`impl Scalar for Dual<T>` — zero changes to existing code. Source-transform: `#[nabla_grad]` proc macro.

### Symbolic CAS

```rust
let x = var("x");
let f = (&x * &x).sin_();              // method chain
let df = diff(&f, "x");                // differentiation
let g = sym!(sin(x^2) + cos(y));       // proc macro syntax
```

`egg` 57 unified rules (`cas_rules()`: 33 algebraic + 24 differentiation). `gradient`/`jacobian`/`hessian` (auto-simplify). Domain-checked `eval`/`eval_tensor`. `Expr` structural `PartialEq`/`Eq`. Precedence-aware Display.

### ODE/SDE solvers

`euler` (1), `rk4` (4), `dormand_prince` (5(4)), `bdf1`/`bdf2` (stiff), `if_euler` (exponential), `metd` (matrix exp), `stormer_verlet` (symplectic), `parareal` (time-parallel), `euler_maruyama`/`milstein` (SDE).

`OdeProblem<T,B,F>` wrapper. `saveat` interpolation. `terminate` callback. `ensemble_euler_maruyama` (parallel N-trajectory). Backward integration. Builder: `.with_dt()`, `.with_tol()`, `.with_saveat()`.

AD + CAS + ODE are **integrated into a single crate**.

---

## §8 NN Module System

### Module & Layers

`Module<T,B>` trait: 2 required methods (`forward`, `set_training`) + 17 optional with sensible defaults. Includes `parameters()` (default empty), `named_parameters()` (default empty), `train()`/`eval()`, `state_dict()`/`load_state_dict()`, `forward_var()` → `Result`, `forward_var_tracked()` → `Result`, `train_forward()` → `Result` (recommended training entry point — convenience alias for `forward_var_tracked`). `From<StateError> for Error` enables `?` propagation across error types.

| Layer | Description |
|---|---|
| `Linear::new(in, out)` | Weight + bias, `forward_var_tracked` |
| `Sequential<T,B>` | `Vec<Box<dyn Module>>`, chained forward; builder: `Sequential::new().with(layer1).with(layer2)` or `.add(impl Module)` (auto-boxes) |
| `Activation` | Generic activation via `ActivationKind` enum (single source of truth — no dual `func`/`kind`; `relu()`/`gelu()`/`sigmoid()`/`silu()`/`tanh()`/`leaky_relu()`/`elu()` constructors, `forward_var` dispatch by enum variant — no string matching) |
| `DropoutLayer` | Inverted dropout, configurable p |
| `EmbeddingLayer` | Embedding table with gradient support |
| `LayerNormModule` | Learnable gamma/beta affine |

`impl_layer!` macro: generates a complete `Module` impl from parameter declarations + a `TensorLike`-generic forward body. Required params before `;`, optional (`Option<Tensor>`) after. Body uses `TensorLike` methods (`tl_matmul`, `tl_t`, `tl_add`, etc.) — works for both `Tensor` (inference) and `Variable` (training). Generates `forward`, `forward_var`, `forward_var_tracked`, and all parameter methods. ~25 lines of boilerplate → ~6 lines:
```rust
impl_layer! {
    Linear { weight; bias }
    forward(x) {
        let wt = weight.tl_t();
        let out = x.tl_matmul(&wt);
        match bias { Some(b) => out.tl_add(b), None => out }
    }
}
```

`impl_module_params!` macro: generates 7 Module trait methods from field annotations — `impl_module_params!(weight; bias)` for required+optional `Tensor<T,B>` fields, `impl_module_params!()` for parameterless modules. Still available for layers needing custom forward logic; `impl_layer!` subsumes it for standard cases.

`#[derive(Module)]` proc macro: auto-generates `training()`/`set_training()`/`parameters()`/`named_parameters()`/`parameters_mut()`/`named_parameters_mut()` from field attributes. Mark fields with `#[param]` (required `Tensor<T,B>`) or `#[param(optional)]` (`Option<Tensor<T,B>>`). Requires `training: bool` field. Use `impl_layer!` for standard forward logic; `#[derive(Module)]` for custom forward with auto parameter management.

`ForwardResult<T,B>`: output Variable + parameter Variables bundled.

### Autograd Variable ops

`Variable::softmax(axis)`, `reshape(m,n)`, `transpose()`, `t()` (alias), `linear_forward(w,b)`, `dropout(p,training)`, `clamp(lo,hi)`, `sum()`, `sum_axis(d)`, `mean()`, `mean_axis(d)`, `mse_loss(target)`, `l1_loss(target)`, `huber_loss(target, delta)`, `binary_cross_entropy(target)`, `cross_entropy(target)` → `Result`, `cross_entropy_indices(targets)` → `Result`, `sign()`, `leaky_relu(alpha)`, `elu(alpha)`, `layer_norm(eps)`, `embedding_lookup(indices)`, `batch_norm(gamma, beta, eps)`, `smooth_l1_loss(target, beta)`, `bce_with_logits(target)`, `nll_loss(targets)` → `Result`, `cosine_embedding_loss(other, y, margin)`, `group_norm(num_groups, weight, bias, eps)`, `slice_rows(range)`, `vcat_var(&[&Self])`, `expand_var(rows, cols)`, `gather_var(axis, &index)`, `index_select_var(axis, &indices)`, `broadcast_mul_cols(&col)`, `broadcast_mul_rows(&row)`, `broadcast_add_rows(&row)`, `bmm_var(&rhs, batch, m, k, n)`, `log_softmax(axis)`, `matmul_tn(&rhs)`, `matmul_nt(&rhs)`, `linear_const(&Tensor)`, `add_const(&Tensor)`, `broadcast_add_rows_const(&Tensor)`, `index_select_const(axis, &Tensor)`, `bmm_const_left(&Tensor, batch, m, k, n)`. `Tape::track_params()`, `Tape::var()` alias. `Variable` owned operators: `Add/Sub/Mul/Neg` (all 4 ownership combos: owned+owned, owned+ref, ref+owned, ref+ref), `Div<T>`/`Mul<T>` scalar ops. `Debug` impl: `Variable(3x4, leaf)` / `Variable(3x4, grad_fn=add)`. `impl_var_op!` internal macro generates all 4 ownership combos for `Add`/`Sub`/`Mul` (ref+ref, owned+owned, mixed).

### TensorLike coverage (✅ registered)

All 13 Variable ops needed by redesign forward are registered in `TensorLike`/`TensorLikeExt`/`TensorLikeMatmulBias`:

| Op | Variable method | Backward | TensorLike |
|---|---|---|---|
| `slice_rows` | `slice_rows(range)` | zero-pad grad to original rows | ✅ `TensorLikeExt` |
| `expand` | `expand_var(rows, cols)` | sum along expanded dims | ✅ `TensorLikeExt` |
| `gather` | `gather_var(axis, &index)` | scatter_add grad | ✅ `TensorLikeExt` |
| `index_select` | `index_select_var(axis, &indices)` | scatter_add grad | ✅ `TensorLikeExt` |
| `broadcast_mul_cols` | `broadcast_mul_cols(&col)` | sum grad×input over axis | ✅ `tensor_like_ops!` binary |
| `broadcast_mul_rows` | `broadcast_mul_rows(&row)` | sum grad×input over axis | ✅ `tensor_like_ops!` binary |
| `broadcast_add_rows` | `broadcast_add_rows(&row)` | sum grad over rows | ✅ `tensor_like_ops!` binary |
| `bmm` | `bmm_var(&rhs, batch, m, k, n)` | batched matmul transposes | ✅ `TensorLikeExt` |
| `log_softmax` | `log_softmax(axis)` | (1-softmax)×grad | ✅ `TensorLikeExt` |
| `matmul_tn` | `matmul_tn(&rhs)` | specialized backward | ✅ `tensor_like_ops!` binary |
| `matmul_nt` | `matmul_nt(&rhs)` | specialized backward | ✅ `tensor_like_ops!` binary |
| `clamp` | `clamp(lo, hi)` | mask where in range | ✅ `TensorLikeExt` |
| `matmul_bias` | `matmul(w).add_var(b)` | fused fwd, split bwd | ✅ `TensorLikeMatmulBias` |
| `vcat` | `vcat_var(&[&Self])` | split grad by row offsets | ✅ `TensorLikeExt` |
| `linear_const` | `linear_const(&Tensor)` | grad @ w^T (input only) | ✅ `TensorLikeExt` |
| `add_const` | `add_const(&Tensor)` | identity (grad pass-through) | ✅ `TensorLikeExt` |
| `broadcast_add_rows_const` | `broadcast_add_rows_const(&Tensor)` | identity | ✅ `TensorLikeExt` |
| `index_select_const` | `index_select_const(axis, &Tensor)` | scatter-add | ✅ `TensorLikeExt` |
| `bmm_const_left` | `bmm_const_left(&Tensor, batch, m, k, n)` | a^T @ grad per batch | ✅ `TensorLikeExt` |

Used by: model, calm, attn, moe, blast, diffusion, optimizer (redesign forward pass).

### Optimizer

`AdamW::new(lr, shapes)` / `Adam::new(lr, shapes)` (no weight decay) / `Sgd::new(lr, shapes).momentum(val)` / `from_params(lr, &[&Tensor])`. `step(params, grads)` / `step_with_vars(module, param_vars)`. `GroupOptimizer::from_module(module, groups, default)` for per-parameter-group optimization. `GradScaler::scale_factor()`. `backward()` NaN/Inf detection (`Err`). Vectorized `clip_grad_norm`. In-place `zero_grad`.

`LrSchedule` enum (`Cosine`/`Linear`/`OneCycle`/`Step`) + `lr_at_step(&schedule, base_lr, step)` + `ScheduleState`.

`save_tensors` / `load_tensors` generic `T: Scalar`.

