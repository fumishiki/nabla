# nabla — Quick Start & API Reference

> See also: [notation.md](notation.md) (naming conventions) | [spec.md](spec.md) (architecture & constraints)

`use nabla::prelude::*;` imports all types, traits, macros, and free functions.
CAS symbols live in a separate prelude: `use nabla::cas_prelude::*;` (or use `sym!` from the main prelude).

---

## 1. Getting Started

### Cargo.toml

```toml
[dependencies]
nabla = { git = "https://github.com/fumishiki/nabla", features = ["cpu"] }
# Training stack (optimizers, dataloader, trainer):
nabla-train = { git = "https://github.com/fumishiki/nabla" }
```

Exactly one backend feature must be enabled — mutual exclusion enforced at compile time:

| Feature | Backend | Storage | Scalar types |
|---|---|---|---|
| `cpu` (default) | `Cpu` | `Vec<T>` row-major | f32, f64, f16, bf16, c32, c64, Dual |
| `wgpu` | `Gpu` | `wgpu::Buffer` | f32 |
| `cuda` | `Cuda` | `CUdeviceptr` | f32, f64, f16, Fp8E4M3, Fp8E5M2, Fp4E2M1 |
| `hip` | `Hip` | `hipDeviceptr_t` | f32, f64 |

### Minimal example

```rust
use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
    let b: Tensor<f64> = mat![[5.0, 6.0], [7.0, 8.0]];
    let c = &a * &b;  // matmul
    println!("{c:.4}");
}
```

`#[nabla::main(cpu)]` sets up the backend, wraps the body in `Result`, and configures `#[cfg]`.

---

## 2. Core Types

```rust
Tensor<T>             // = Tensor<T, DefaultBackend>; 2D row-major
StaticMatrix<T, R, C> // stack-allocated, const-generic shape
NdTensor<T>           // N-D CPU-only, flat Vec<T>
TensorView<'_, T, B>  // zero-copy borrow via view_slice
DynTensor              // runtime scalar dispatch (enum{F32,F64,C32,C64})
Variable<T, B>        // autograd-tracked tensor (on a Tape)
```

**Scalar types**: `f32`, `f64`, `c32`, `c64` (complex, CPU only). All implement `Scalar`.
**Backend trait**: `Backend` — all computation methods are required (no default body, no CPU fallback).

```rust
// Indexing: 0-based, tuple
let v = a[(1, 0)];           // read
a[(1, 0)] = 9.0;             // write
let s = a.slice(0..2, 1..3); // owned copy
let v = a.view_slice(0..2, 1..3); // zero-copy TensorView
```

---

## 3. Constructors

All constructors are free functions in the prelude.

```rust
let z: Tensor<f64> = zeros(3, 4);          // 3x4 zeros
let o: Tensor<f64> = ones(2, 2);           // 2x2 ones
let f: Tensor<f64> = fill(3, 3, 7.0);      // 3x3 filled with 7.0
let i: Tensor<f64> = eye(4);               // 4x4 identity
let r: Tensor<f64> = rand(3, 3);           // uniform [0, 1)
let n: Tensor<f64> = randn(3, 3);          // standard normal
let a: Tensor<f64> = arange(0.0, 1.0, 0.1); // [0.0, 0.1, ..., 0.9]
let l: Tensor<f64> = linspace(0.0, 1.0, 5); // 5 points from 0 to 1
let g: Tensor<f64> = from_fn(3, 3, |r, c| (r * 3 + c) as f64);
```

| Function | Shape | Description |
|---|---|---|
| `zeros(m, n)` | (m, n) | Zero-filled |
| `ones(m, n)` | (m, n) | One-filled |
| `fill(m, n, val)` | (m, n) | Constant-filled |
| `eye(n)` | (n, n) | Identity |
| `rand(m, n)` / `randn(m, n)` | (m, n) | Uniform / normal random |
| `from_fn(m, n, \|r, c\| expr)` | (m, n) | Element-wise closure |
| `arange(start, stop, step)` | (1, k) | Half-open range |
| `linspace(start, stop, n)` | (1, n) | Evenly spaced (inclusive) |
| `logspace(start, stop, n)` | (1, n) | Log-spaced (10^start to 10^stop) |
| `geomspace(start, stop, n)` | (1, n) | Geometrically spaced |
| `zeros_vec(n)` / `ones_vec(n)` | (n, 1) | Column vectors |
| `rand_vec(n)` / `randn_vec(n)` | (n, 1) | Random column vectors |
| `nd_zeros(shape)` | N-D | N-D zero tensor (CPU only) |
| `diagm(&v)` | (n, n) | Diagonal matrix from vector |

**Seed control**: `set_seed(42)` for reproducibility, `clear_seed()` to revert.

**Matrix literal**:
```rust
let m: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
let m: Tensor<f64> = mat![1.0, 2.0; 3.0, 4.0];  // semicolon syntax
```

---

## 4. Arithmetic & Element-wise

`a * b` = matmul (move), `&a * &b` = matmul (borrow). `emul` / `ediv` for element-wise.

```rust
let c = &a * &b;              // matmul
let d = a.emul(&b);           // element-wise multiply (Hadamard)
let e = a.ediv(&b);           // element-wise divide
let f = 2.0 * &a;             // scalar * tensor
let g = a.powf(2.0);          // element-wise power
let h = a.epow(&b);           // tensor-tensor power
```

| Op | Syntax | Notes |
|---|---|---|
| Matmul | `a * b` / `&a * &b` | Move or borrow |
| Hadamard | `a.emul(&b)` | Element-wise multiply |
| Elem div | `a.ediv(&b)` | Element-wise divide |
| Scalar | `c * a`, `a / c` | Scalar-tensor |
| Transpose | `a.t()` | Transpose |
| Conjugate-T | `a.h()` | Hermitian |
| Dot product | `a.dot(&b)` / `dot(&a, &b)` | Scalar return |
| Outer | `a.outer(&b)` | uv^T |
| Kronecker | `a.kron(&b)` / `kron(&a, &b)` | A tensor B |
| In-place | `a += &b`, `a *= s`, `c.mm_(&a, &b)` | Mutable |

**Broadcasting**: `&a + &b` auto-infers `(m,n)+(1,n)`, `(m,n)+(m,1)`, `(m,n)+(1,1)`. Owned ops require exact shapes.

**Element-wise math** (all GPU float4 vectorized):
`exp` `ln` `log1p` `log2` `log10` `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `asinh` `acosh` `atanh` `sqrt` `abs` `recip` `ceil` `floor` `round` `powf` `neg` `sign` `rem`

```rust
let y = a.sin();
let z = a.exp();
let w = fuse!(x.sin().powf(2.0));  // fused into 1 GPU kernel
```

---

## 5. Shape Manipulation

```rust
let r = a.reshape(6, 2);          // reshape to (6, 2)
let t = a.t();                     // transpose
let f = a.flatten();               // flatten to (1, m*n)
let s = a.squeeze();               // remove size-1 dims
let u = a.unsqueeze(0);            // add dim at axis 0
let c = Tensor::cat(&[&a, &b], 0); // concat along axis
let v = vcat!(a, b);               // vertical concat
let h = hcat!(a, b);               // horizontal concat
let blk = block![[a, b], [c, d]];  // block matrix
let chunks = a.chunk(3, 0);        // split into 3 along axis 0
let stk = Tensor::stack(&[&a, &b], 0); // stack along new axis
```

---

## 6. Reductions

```rust
let s = a.sum();                // scalar sum
let m = a.mean();               // scalar mean
let x = a.max();                // scalar max
let i = a.argmax();             // index of max
let v = a.var();                // population variance (ddof=0)
let sd = a.std();               // population std dev

// Axis variants
let row_sum = a.sum_axis(0);    // sum along rows
let col_mean = a.mean_axis(1);  // mean along cols
let cs = a.cumsum(0);           // cumulative sum along axis 0
let cp = a.cumprod(1);          // cumulative product along axis 1
let va = a.var_axis(0);         // variance along axis 0
let sa = a.std_axis(0);         // std dev along axis 0
let vd = a.var_axis_ddof(0, 1); // variance with ddof=1 (sample variance)
let n = a.norm();               // Frobenius norm
let p = a.norm_ord(1.0);        // L1 norm
```

| Reduction | Global | Per-axis |
|---|---|---|
| Sum | `.sum()` | `.sum_axis(d)` |
| Mean | `.mean()` | `.mean_axis(d)` |
| Max / Min | `.max()` / `.min()` | `.max_axis(d)` / `.min_axis(d)` |
| Argmax / Argmin | `.argmax()` | `.argmax_axis(d)` |
| Prod | `.prod()` | `.prod_axis(d)` |
| Cumsum / Cumprod | — | `.cumsum(d)` / `.cumprod(d)` |
| Norm | `.norm()` | — |
| Variance | `.var()` | `.var_axis(d)` / `.var_axis_ddof(d, ddof)` |
| Std Dev | `.std()` | `.std_axis(d)` |

---

## 7. Linear Algebra (CPU)

45+ methods via the `LinalgExt` trait (f32/f64; f32 auto-promotes to f64 internally).

```rust
// Factorizations
let (p, l, u) = a.lu()?;
let (q, r) = a.qr();
let l = a.chol()?;              // lower Cholesky
let (u, s, vt) = a.svd()?;

// Solvers
let x = a.solve(&b)?;           // Ax = b
let x = a.inv()?;               // A^{-1}
let x = a.lstsq(&b)?;          // least squares

// Matrix properties
let d = a.det()?;
let t = a.tr();                  // trace
let k = a.cond_p(2.0)?;         // condition number
let r = a.rank(1e-10)?;

// Eigenvalues
let (eigs, t, q) = a.eig_into()?;
let sym = a.sym(Side::Lower)?;
let (vals, vecs) = sym.eigh()?;

// Matrix functions
let e = a.expm()?;              // matrix exponential
let l = a.logm()?;              // matrix logarithm
let s = a.sqrtm()?;             // matrix square root
let (t, q) = a.schur_decomp()?;
let (u, h) = a.polar()?;

// Equations
let x = sylvester(&a, &b, &c)?;          // AX + XB = C
let x = lyapunov(&a, &c)?;               // AX + XA^T = C
let x = discrete_sylvester(&a, &b, &c)?; // AXB + X = C
```

**Structural types**: `Diagonal::new(v)`, `Symmetric::new(a, Side)`, `Triangular::new(a, TriKind)`.
**Factorization reuse**: `lu.solve(&b)`, `lu.inverse()`, `lu.reconstruct()`.

---

## 8. Automatic Differentiation

### 8.1 Reverse-mode AD (Tape)

```rust
let tape = Tape::new();
let x = tape.var(tensor_x)?;
let w = tape.var(tensor_w)?;
let loss = (&x * &w).exp().sum();
loss.backward()?;
let dx = x.grad()?;   // gradient w.r.t. x
let dw = w.grad()?;   // gradient w.r.t. w
// tape and grads freed by Drop
```

| API | Description |
|---|---|
| `Tape::new()` | Create tape (returns `Rc<Tape>`) |
| `tape.var(tensor)` / `tape.variable(tensor)` | Track a leaf variable |
| `tape.track_params(&[&t1, &t2])` | Batch-track parameters |
| `tape.no_grad(\|\| { ... })` | Disable tracking in scope |
| `loss.backward()` | Backpropagate (scalar output required) |
| `loss.backward_with(grad)` | Custom seed gradient (non-scalar) |
| `x.grad()` | Retrieve gradient (`Result<Tensor>`) |
| `x.retain_grad()` | Keep intermediate node gradients |

**`vars!` macro**: batch-create tracked variables.
```rust
vars!(tape; w1 = weight1, w2 = weight2);
```

**`ad!` macro**: one-shot differentiation block.
```rust
let (loss, dw, db) = ad!(Cpu; w = weights, b = biases => |tape| {
    let y = &(&x * &w) + &b;
    y.sum()
})?;
```

### 8.2 Forward-mode AD (Dual)

```rust
let x = Dual::new(2.0, 1.0);   // value=2, derivative seed=1
let y = (x * x).sin();          // y.dual = dy/dx
```

`Dual<T>` implements `Scalar` — drop-in replacement with zero code changes. `MultiDual<T, N>` for multiple partials. `#[nabla_grad]` proc macro for source-transform AD.

### 8.3 Variable ops (autograd-tracked)

47 ops with backward: `relu`, `gelu`, `sigmoid`, `silu`, `softmax(axis)`, `reshape(m,n)`, `transpose`, `linear_forward(w,b)`, `dropout(p,training)`, `clamp(lo,hi)`, `sum`, `sum_axis(d)`, `mean`, `mean_axis(d)`, `sign`, `leaky_relu(alpha)`, `elu(alpha)`, `layer_norm(eps)`, `embedding_lookup(indices)`, `batch_norm(gamma,beta,eps)`, `mse_loss(target)`, `l1_loss(target)`, `cross_entropy(target)`, `cross_entropy_indices(targets)`, `bce_with_logits(target)`, `nll_loss(targets)`, `smooth_l1_loss(target,beta)`, `cosine_embedding_loss(other,y,margin)`.

Ownership: `Variable` supports all 4 combos for `+`, `-`, `*`: `var+var`, `&var+var`, `var+&var`, `&var+&var`.

---

## 9. Neural Network Modules

### 9.1 Module trait

```rust
pub trait Module<T: Scalar, B: Backend> {
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B>;
    fn set_training(&mut self, training: bool);
    // 17 optional methods with defaults:
    // forward_with, forward_var, forward_var_tracked, train_forward,
    // training, train, eval, parameters, named_parameters,
    // parameters_mut, children, named_children, buffers,
    // state_dict, load_state_dict, named_parameters_mut, apply
}
```

### 9.2 Built-in layers

| Layer | Constructor | Description |
|---|---|---|
| `Linear` | `Linear::new(in, out)` | Fully connected, Xavier init |
| `Sequential` | `Sequential::new().with(l1).with(l2)` | Layer chain |
| `Activation` | `relu()`, `gelu()`, `sigmoid()`, `silu()`, `tanh()` | Enum-dispatched |
| `DropoutLayer` | `DropoutLayer::new(p)` | Inverted dropout |
| `EmbeddingLayer` | `EmbeddingLayer::new(vocab, dim)` | Embedding table |
| `LayerNormModule` | `LayerNormModule::new(dim)` | Learnable affine |

```rust
let model = sequential!(
    Linear::new(784, 128),
    relu(),
    Linear::new(128, 10),
);
let out = model.forward(&input);
```

### 9.3 impl_layer! macro

Generates a complete `Module` impl from parameter declarations and a `TensorLike`-generic forward body (~25 lines of boilerplate to ~6):

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

### 9.4 #[derive(Module)] proc macro

Auto-generates parameter management methods from field attributes:

```rust
#[derive(Module)]
struct MyLayer<T: Scalar, B: Backend> {
    #[param] weight: Tensor<T, B>,
    #[param(optional)] bias: Option<Tensor<T, B>>,
    training: bool,
}
```

### 9.5 Training forward pass

`forward_var_tracked` / `train_forward` return `ForwardResult<T,B>` bundling output `Variable` + parameter `Variable`s for optimizer access:

```rust
let tape = Tape::new();
let x = tape.var(input)?;
let result = model.train_forward(&x, &tape)?;
let loss = result.output.cross_entropy_indices(&targets)?;
loss.backward()?;
// result.param_vars now have gradients
```

### 9.6 Serialization

```rust
save_tensors(&[("weight", &w), ("bias", &b)], Path::new("model.nbla"))?;
let loaded = load_tensors::<f64, Cpu>(Path::new("model.nbla"))?;
```

---

## 10. Optimizers (nabla-train)

Optimizers live in the `nabla-train` crate.

```rust
use nabla_train::prelude::*;
```

| Optimizer | Constructor |
|---|---|
| `AdamW` | `AdamW::new(lr, &shapes)` or `AdamW::from_params(lr, &params)` |
| `Adam` | `Adam::new(lr, &shapes)` or `Adam::from_params(lr, &params)` |
| `Sgd` | `Sgd::new(lr, &shapes)` or `Sgd::from_params(lr, &params)` |

```rust
let params = model.parameters();
let param_refs: Vec<&Tensor<f64>> = params.iter().map(|p| *p).collect();
let mut optimizer = AdamW::from_params(1e-3, &param_refs);
optimizer.step(&mut model.parameters_mut(), &grad_refs);
// Or with ForwardResult:
optimizer.step_with_vars(&mut model, &result.param_vars);
```

**`train_step!` macro** absorbs the tape/backward/grad/optimizer ceremony:

```rust
let loss = train_step!(model, optimizer, tape, |x, out| {
    out.mse_loss(&target)
})?;
```

**Learning rate schedules**: `LrSchedule::Cosine`, `Linear`, `OneCycle`, `Step` + `lr_at_step(&schedule, base_lr, step)`.
**Gradient utilities**: `clip_grad_norm`, `GradScaler`.
**Checkpointing**: `save_checkpoint` / `load_checkpoint`.

---

## 11. Symbolic CAS

CAS symbols (`var`, `diff`, `simplify`, `eval`) are in `nabla::cas_prelude::*` or accessible via `nabla::cas::*`.

```rust
use nabla::cas::*;

let x = var("x");
let f = (&x * &x).sin_();     // sin(x^2), method chain API
let df = diff(&f, "x");        // symbolic differentiation
let val = eval(&df, &[("x", 2.0)].into())?;  // numeric evaluation
```

### sym! proc macro (in main prelude)

```rust
let g = sym!(sin(x^2) + cos(y));  // ^ = exponentiation, not XOR
```

| Function | Description |
|---|---|
| `var("x")` | Create named symbolic variable |
| `diff(&expr, "x")` | Differentiate w.r.t. variable |
| `diff_simplify(&expr, "x")` | Differentiate + simplify |
| `simplify(&expr)` | Algebraic simplification (57 egg rules) |
| `eval(&expr, &vars)` | Numeric evaluation (domain-checked) |
| `eval_tensor(&expr, &vars)` | Evaluate with tensor bindings |
| `gradient(&expr, &["x","y"])` | Gradient vector |
| `jacobian(&exprs, &["x","y"])` | Jacobian matrix |
| `hessian(&expr, &["x","y"])` | Hessian matrix |
| `substitute(&expr, "x", &replacement)` | Variable substitution |

`Expr` supports `+`, `-`, `*`, `/`, `Neg`, structural `PartialEq`/`Eq`, and precedence-aware `Display`.

---

## 12. ODE/SDE Solvers

### Basic usage

```rust
let y0: Tensor<f64> = mat![[1.0]];
let f = |t: f64, y: &Tensor<f64>| -0.5 * y;  // dy/dt = -0.5y

let sol = euler(f, &y0, (0.0, 10.0), 0.01)?;
let sol = rk4(f, &y0, (0.0, 10.0), 0.01)?;
let sol = dormand_prince(f, &y0, (0.0, 10.0), &AdaptiveConfig::default())?;
```

### Solver catalog

| Solver | Type | Signature |
|---|---|---|
| `euler` | Explicit, order 1 | `euler(f, &y0, t_span, dt)` |
| `rk4` | Explicit, order 4 | `rk4(f, &y0, t_span, dt)` |
| `dormand_prince` | Adaptive, order 5(4) | `dormand_prince(f, &y0, t_span, &config)` |
| `bdf1` | Implicit (stiff), order 1 | `bdf1(f, &y0, t_span, &Bdf1Config)` |
| `bdf2` | Implicit (stiff), order 2 | `bdf2(f, &y0, t_span, &Bdf2Config)` |
| `if_euler` | Exponential integrator | CPU only |
| `stormer_verlet` | Symplectic, order 2 | `stormer_verlet(f_q, f_p, &q0, &p0, t_span, dt)` |
| `euler_maruyama` | SDE, order 0.5 | `euler_maruyama(drift, diffusion, &y0, ...)` |
| `milstein` | SDE, order 1.0 | `milstein(drift, diffusion, diff_diffusion, &y0, ...)` |

### OdeSolution

```rust
let state = sol.final_state();     // last state
let y_at = sol.eval(5.0);          // interpolate at t=5
let y3 = &sol[3];                  // index into states
```

### AdaptiveConfig builder

```rust
let config = AdaptiveConfig::default()
    .with_dt(0.001)
    .with_tol(1e-8, 1e-6)
    .with_saveat(vec![0.0, 1.0, 2.0, 5.0, 10.0]);
```

### OdeProblem wrapper

```rust
let prob = OdeProblem { f, y0: y0.clone(), t_span: (0.0, 10.0) };
let sol = prob.solve_euler(0.01)?;
let sol = prob.solve_rk4(0.01)?;
let sol = prob.solve_dormand_prince(&config)?;
```

---

## 13. Macros Reference

| Macro | GPU | Purpose |
|---|---|---|
| `mat![[1,2],[3,4]]` | yes | Matrix literal (compile-time shape check) |
| `block![[a,b],[c,d]]` | yes | Block matrix construction |
| `vcat!(a, b)` / `hcat!(a, b)` | yes | Vertical / horizontal concat |
| `math!(a * b + c)` | yes | Auto-borrow idents/fields/indices |
| `fuse!(x.sin().powf(2.0))` | yes | Element-wise fusion to 1 kernel |
| `mega_fuse!(a+b; prev.exp(); inputs: a, b)` | yes | Multi-output DAG fusion |
| `einsum!(c[i,j] = a[i,k] * b[k,j])` | yes | Einstein summation (7 patterns) |
| `stencil!(out[i,j] = ...)` | CPU | Finite difference (auto bounds) |
| `map!(\|x\| f(x), &a)` | CPU | Element-wise closure map |
| `map_!(out, \|x,y\| x*y, &a, &b)` | CPU | In-place element-wise |
| `par_map!(\|x\| x*2.0, &a)` | CPU | Rayon parallel map |
| `fuse_!(a = expr)` | yes | In-place fused element-wise |
| `sym!(sin(x^2) + cos(y))` | yes | Symbolic expression (^ = power) |
| `ad!(Cpu; w=init => \|tape\| { ... })` | — | Autograd one-shot block |
| `vars!(tape; w1=e1, w2=e2)` | — | Batch-create tracked Variables |
| `sequential!(l1, l2, l3)` | — | Build Sequential model |
| `cas_vars!{ x: 1.0, y: 2.0 }` | — | CAS variable bindings (HashMap) |
| `train_step!(model, optim, tape, \|x,out\| loss)` | — | Training ceremony (nabla-train) |
| `impl_layer! { Name { w; b } forward(x) { ... } }` | — | Module impl from params + forward |
| `#[derive(Module)]` | — | Auto parameter management |
| `approx!(&a, &b)` / `approx!(&a, &b, 1e-6)` | — | Assert approximate equality |
| `vec_unpack!(tensor, x, y, z)` | — | Destructure column vector |

---

## 14. GPU-Specific Features

### Backend selection (compile-time)

```toml
# Pick one:
nabla = { git = "https://github.com/fumishiki/nabla", features = ["cuda"] }
nabla = { git = "https://github.com/fumishiki/nabla", features = ["wgpu"] }
nabla = { git = "https://github.com/fumishiki/nabla", features = ["hip"] }
```

No `model.to('cuda')` — the backend is resolved at compile time.

### CUDA Graph capture/replay

```rust
use nabla::prelude::*;

let graph = TrainingGraph::new();
graph.warmup(|| { /* one forward+backward pass */ });
graph.capture(|| { /* same pass — recorded */ })?;
for _ in 0..1000 {
    graph.replay()?;  // replays captured kernels, ~1.67x speedup
}
```

### cuBLAS epilogue fusion

```rust
cuda_matmul_epilogue(&mut out, &a, &b, Epilogue::GeluBias, Some(bias_ptr));
```

Fuses matmul + activation + bias into a single cuBLAS call.

### Kernel fusion via fuse!

```rust
// CPU: single from_fn loop. GPU: JIT-compiled fused kernel.
let y = fuse!(x.sin().powf(2.0) + x.cos());
// Multi-output DAG fusion:
let (y, z) = mega_fuse!(a + b; prev.exp(); inputs: a, b);
```

Pipeline: AST -> egg EqSat (18 IEEE-754 safe rules) -> NVRTC/hiprtc JIT -> FNV-1a hash cache.

### GEMV auto-dispatch (batch-1 matmul)

When the output has `m=1` (single-row), the CUDA backend automatically dispatches `cuBLAS sgemv` instead of `sgemm`. This gives a significant speedup for inference-style forward passes where batch size = 1.

```rust
// Internally uses sgemv (not sgemm) when a.nrows() == 1
let out = &a * &b;  // a: (1, k), b: (k, n) → dispatches sgemv
```

No API change required — dispatch is automatic based on shape.

### sync()

`sync()` is a GPU stream barrier (no device-to-host transfer). Use before timing or cross-stream dependencies.

### Low-Precision Types (fp8 / fp4)

CUDA backend supports `f16`, `Fp8E4M3`, `Fp8E5M2`, `Fp4E2M1` as first-class `Scalar` types.

```rust
use nabla::prelude::*;

// Cast between precisions
let a: Tensor<f32, Cuda> = randn(64, 64);
let a_f16: Tensor<f16, Cuda>     = a.cast::<f16>();
let a_fp8e4: Tensor<Fp8E4M3, Cuda> = a.quantize_fp8_e4m3();  // = a.cast::<Fp8E4M3>()
let a_fp8e5: Tensor<Fp8E5M2, Cuda> = a.quantize_fp8_e5m2();

// Dequantize back to f32
let a_back: Tensor<f32, Cuda> = a_fp8e4.dequantize_fp8_e4m3();

// Blockwise fp4 (for quantization-aware training)
let (q, scales) = a.quantize_fp4_blockwise(128);   // block_size=128
let a_back: Tensor<f32, Cuda> = q.dequantize_fp4_blockwise(&scales, 128);
```

| Type | Format | Range | Use case |
|---|---|---|---|
| `f16` | IEEE 754 half | ±65504 | General mixed-precision |
| `Fp8E4M3` | E4M3 (OCP) | ±448 | Forward pass activations |
| `Fp8E5M2` | E5M2 (OCP) | ±57344 | Gradient storage |
| `Fp4E2M1` | E2M1 | ±6 | Extreme compression (QAT) |

All low-precision types implement `Scalar`, support `from_fn`, `fill`, `cast`, element-wise ops, and `to_vec`. cuBLAS matmul dispatches `gemm_ex` with the appropriate compute type.

---

## 15. Error Handling

```rust
use nabla::prelude::*;  // imports Error, Result

// nabla::Result<T> = std::result::Result<T, nabla::Error>
// nabla::NablaResult<T> = std::result::Result<T, Box<dyn std::error::Error>>

let x = a.solve(&b)?;      // returns Error on singular matrix
let g = v.grad()?;          // Error::NoGradient if missing
loss.backward()?;           // Error if non-scalar output

#[nabla::main(cpu)]
fn main() {                 // body returns NablaResult
    let a: Tensor<f64> = eye(3);
    let b = a.inv()?;
}
```

No silent NaN — all fallible operations return `Result`.

---

## 16. Complete Training Example

```rust
use nabla::prelude::*;
use nabla_train::prelude::*;

#[nabla::main(cpu)]
fn main() {
    set_seed(42);
    let x: Tensor<f64> = randn(100, 4);
    let target: Tensor<f64> = randn(100, 1);

    let mut model = sequential!(
        Linear::new(4, 32),
        relu(),
        Linear::new(32, 1),
    );
    let params = model.parameters();
    let param_refs: Vec<&Tensor<f64>> = params.iter().map(|p| *p).collect();
    let mut optim = AdamW::from_params(1e-3, &param_refs);

    for epoch in 0..100 {
        let tape = Tape::new();
        let xv = tape.var(x.clone())?;
        let result = model.train_forward(&xv, &tape)?;
        let loss = result.output.mse_loss(&target);
        loss.backward()?;

        let grads: Vec<Tensor<f64>> = result.param_vars.iter()
            .map(|v| v.grad())
            .collect::<Result<_>>()?;
        let grad_refs: Vec<&Tensor<f64>> = grads.iter().collect();
        optim.step(&mut model.parameters_mut(), &grad_refs);

        if epoch % 10 == 0 {
            println!("epoch {epoch}: loss = {:.4}", loss.data().sum());
        }
    }

    save_tensors(
        &model.state_dict().iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
        Path::new("model.nbla"),
    )?;
}
```

## 17. GGUF Export & Inference (`nabla-interface`)

`nabla-interface` bridges nabla's training stack with llama.cpp for deployment on Apple Silicon (Metal) and other platforms.

```toml
# Cargo.toml
[dependencies]
nabla-interface = { git = "https://github.com/fumishiki/nabla" }                    # GGUF export only
nabla-interface = { git = "https://github.com/fumishiki/nabla", features = ["llama"] }  # + llama.cpp inference
```

### 17.1 GGUF Export

Convert a trained `state_dict` to GGUF format for llama.cpp inference:

```rust
use std::path::Path;
use nabla_interface::{export_gguf, GgufArchConfig, GgufQuantType, QuantOverride};

let config = GgufArchConfig {
    architecture: "llama".into(), name: "MyModel-7B".into(),
    context_length: 4096, embedding_length: 4096, block_count: 32,
    head_count: 32, head_count_kv: 8, vocab_size: 32000,
};

// Export with Q4_K_M, override embeddings to F16
let overrides = vec![
    QuantOverride { name: "embedding.weight".into(), qtype: GgufQuantType::F16 },
];
export_gguf(&state_dict, Path::new("model.gguf"), GgufQuantType::Q4_K_M, &config, &overrides)?;
```

Layer names are automatically mapped (e.g. `layers.0.attention.wq.weight` → `blk.0.attn_q.weight`).

### 17.2 Quantization Types

34 GGUF quantization types are supported. Key formats:

| Category | Types | bpw | Quantize/Dequantize |
|---|---|---|---|
| Full precision | F32, F16, BF16, F64 | 32/16/16/64 | ✅ |
| Legacy (QK=32) | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | 4.5–9.0 | ✅ |
| K-quant (QK=256) | Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K | 2.6–6.6 | ✅ |
| IQ (importance) | IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS | 1.6–4.3 | enum only |
| Ternary | TQ1_0, TQ2_0 | 1.7/2.1 | enum only |
| Integer | I8, I16, I32, I64 | 8/16/32/64 | enum only |

```rust
use nabla_interface::quant::{GgufQuantType, quantize, dequantize};

let data = vec![1.0f32, -0.5, 0.25, /* ... 32 elements for Q4_0 */];
let packed = quantize(&data, GgufQuantType::Q4_0)?;    // Vec<u8>
let restored = dequantize(&packed, GgufQuantType::Q4_0)?; // Vec<f32>

assert!(GgufQuantType::Q4_K_M.is_quantizable());   // true
assert!(!GgufQuantType::IQ2_XXS.is_quantizable());  // false — needs importance matrix
```

### 17.3 Low-Level GGUF Writer

Build custom GGUF files directly:

```rust
use nabla_interface::gguf::{GgufWriter, MetadataValue, TensorInfo};
use nabla_interface::quant::GgufQuantType;
use std::io::BufWriter;

let mut writer: GgufWriter<BufWriter<std::fs::File>> = GgufWriter::new();
writer.add_metadata("general.architecture", MetadataValue::String("llama".into()));
writer.add_metadata("general.name", MetadataValue::String("MyModel".into()));
writer.add_metadata("llama.context_length", MetadataValue::U32(4096));

let quantized_data: Vec<u8> = quantize(&f32_data, GgufQuantType::Q4_0)?;
writer.add_tensor(TensorInfo {
    name: "token_embd.weight".into(),
    dims: vec![4096, 32000],
    qtype: GgufQuantType::Q4_0,
    data_size: quantized_data.len(),
}, quantized_data);

let file = std::fs::File::create("model.gguf")?;
let mut buf = BufWriter::new(file);
writer.write_to(&mut buf)?;  // GGUF v3 binary, 32-byte aligned
```

### 17.4 Inference with llama.cpp (requires `llama` feature)

Run inference on exported GGUF models via llama.cpp FFI:

```rust
use nabla_interface::{InferenceEngine, InferenceConfig, SamplingConfig, PerfStats};

// Load model (Metal GPU offload by default on macOS)
let config = InferenceConfig {
    n_ctx: 2048, n_batch: 512,
    n_threads: 8, n_gpu_layers: -1,  // -1 = all layers on GPU
};
let mut engine = InferenceEngine::new("model.gguf", config)?;

// Batch generation
let sampling = SamplingConfig {
    temperature: 0.8, top_k: 40, top_p: 0.95,
    repeat_penalty: 1.1, seed: Some(42),
};
let text = engine.generate("Once upon a time", 128, &sampling)?;
println!("{text}");

// Streaming generation
for token in engine.generate_stream("Hello", 64, &sampling)? {
    print!("{token}");
}

// Performance stats
let stats: PerfStats = engine.perf();
println!("prompt: {:.1} tok/s, gen: {:.1} tok/s, total: {} tokens",
    stats.prompt_tok_per_sec, stats.gen_tok_per_sec, stats.total_tokens);
```

`InferenceConfig::default()` uses `n_ctx=2048, n_batch=512, n_gpu_layers=-1` with auto-detected thread count. `SamplingConfig::default()` uses `temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1`.
