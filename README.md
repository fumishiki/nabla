<p align="center">
  <img src="assets/nabla_og.PNG" alt="nabla — GPU math for Rust, no C++ required" width="720">
</p>

<p align="center">
  <a href="https://github.com/fumishiki/nabla/actions/workflows/ci.yml"><img src="https://github.com/fumishiki/nabla/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.88%2B-orange.svg" alt="Rust 1.88+"></a>
</p>

# ∇ nabla

nabla is a Rust library that provides two things:
- **GPU-accelerated tensor math** — the operations you know from NumPy/PyTorch, running natively on NVIDIA, AMD, or Vulkan/Metal GPUs
- **A complete ML training stack** — model building, autodiff, optimizers, and model export, all in pure Rust with no C++ required

If you know PyTorch, nabla will feel immediately familiar. The difference: you get Rust's safety guarantees, no garbage collector, and in many cases faster GPU execution than PyTorch.

<!-- toc -->

- [More About nabla](#more-about-nabla)
  - [GPU-Ready Tensor Library](#gpu-ready-tensor-library)
  - [Faster Than PyTorch, and Why](#faster-than-pytorch-and-why)
  - [Autodiff: loss.backward() in Rust](#autodiff-lossbackward-in-rust)
  - [Training Stack: nabla-train](#training-stack-nabla-train)
  - [Model Inference: nabla-interface](#model-inference-nabla-interface)
  - [Symbolic CAS and ODE Solvers](#symbolic-cas-and-ode-solvers)
  - [No Silent Errors](#no-silent-errors)
  - [Macro DSL: Write Math, Not Boilerplate](#macro-dsl-write-math-not-boilerplate)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

<!-- tocstop -->

## More About nabla

nabla is split into focused packages — use only what you need:

| Package | What it does |
|---|---|
| [**`nabla-core`**](docs-en/spec.md) | The tensor engine. 190+ operations — slicing, broadcasting, linear algebra, convolutions — on CPU, NVIDIA, AMD, or Vulkan/Metal GPU. Switch GPU with one line in `Cargo.toml`. |
| [**`nabla-macros`**](docs-en/notation.md) | Write math as code. `einsum!` for concise tensor contractions, `fuse!` to merge multiple operations into one GPU kernel automatically, `sym!` for symbolic algebra. |
| [**`nabla-ml`**](docs-en/notation.md) | Automatic gradients (the core of neural-network training), 45+ linear algebra routines, symbolic math, and ODE solvers. |
| [**`nabla-train`**](docs-en/quick_start.md) | Everything you need to train a model: optimizers, learning-rate schedules, data loading, checkpointing, quantization, and ONNX export. |
| [**`nabla-interface`**](docs-en/quick_start.md) | Export a trained model to GGUF and run it locally with llama.cpp, Ollama, or LM Studio — including GPU offload on Apple Silicon. |

### GPU-Ready Tensor Library

If you use NumPy, you have used tensors.

nabla provides tensors that run on CPU or GPU with no code changes — the backend is a compile-time feature flag, so there is no `model.to("cuda")` and no accidental CPU fallback. 190+ operations including:

```rust
use nabla::prelude::*;

let a: Tensor<f64> = mat![[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]];

let b = a.t();                              // transpose
let c = &a * &b;                            // matmul (tensor cores on NVIDIA/AMD)
let d = a.emul(&b);                         // element-wise multiply
let s = a.sum_axis(0);                      // reduce along axis
let (u, sigma, vt) = a.svd()?;             // SVD
let x = a.solve(&rhs)?;                    // Ax = b, returns Err on singular
```

**Convolutions** — conv1d/2d/3d and transposed convolution, GPU-accelerated via im2col + cuBLAS:

```rust
let out = input.conv2d(&weight, &bias, stride, padding, dilation)?;   // NCHW layout
let up  = input.conv_transpose2d(&weight, &bias, stride, padding)?;   // upsampling
let out = input.max_pool2d(kernel_size, stride, padding)?;
let out = input.avg_pool2d(kernel_size, stride, padding)?;
let out = input.adaptive_avg_pool2d((target_h, target_w))?;
```

**Attention / FlashAttention-2** — for transformer models, nabla implements scaled dot-product attention using the FlashAttention-2 algorithm. This avoids materializing the full N×N attention matrix in memory, making it practical for long sequences:

```rust
// Multi-head attention — splits into heads, calls FlashAttention-2 per head, concatenates
// q, k, v: (seq_len, d_model)
let out = Tensor::multi_head_attention(&q, &k, &v, num_heads, mask.as_ref());

// Low-level FlashAttention-2 with explicit shapes (head_dim must be ≤ 128)
let out = Tensor::sdpa(&q, &k, &v, mask.as_ref(), seq_q, seq_k, head_dim, batch_heads);
```

Switch to GPU: change `features = ["cpu"]` to `features = ["cuda"]` in `Cargo.toml`. No other changes.

### Faster Than PyTorch, and Why

<p align="center">
  <img src="assets/demo_benchmark.gif" alt="nabla vs PyTorch benchmark" width="800">
</p>

**Benchmark on GH200 480GB (CUDA 12.8, PyTorch 2.7.0)**

The number that matters for real training workloads — a full step (forward + backward + optimizer):

| Batch size | nabla eager | nabla CUDA Graph | PyTorch eager | PyTorch compile | PyTorch CUDA Graph | nabla eager speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.112 ms | **0.070 ms** | 0.771 ms | 0.598 ms | 0.046 ms | **6.9×** |
| 32 | 0.134 ms | **0.086 ms** | 0.852 ms | 0.592 ms | 0.072 ms | **6.4×** |
| 128 | 0.134 ms | **0.088 ms** | 0.863 ms | 0.592 ms | 0.129 ms | **6.4×** |
| 256 | 0.140 ms | **0.094 ms** | 0.822 ms | 0.582 ms | 0.136 ms | **5.9×** |
| 512 | 0.148 ms | **0.109 ms** | 0.851 ms | 0.583 ms | 0.143 ms | **5.8×** |
| 1024 | 0.170 ms | **0.130 ms** | 0.887 ms | 0.583 ms | 0.168 ms | **5.2×** |

> Model: MLP 784→256→128→10, MSE sum loss, SGD. Same model and loss on both sides — no `allow_tf32` manipulation.
> PyTorch 2.10.0+cu128 / triton 3.6.0, GH200. Scripts: [`benchmarks/bench_pytorch.py`](benchmarks/bench_pytorch.py) vs [`benchmarks/src/profile_train_graph.rs`](benchmarks/src/profile_train_graph.rs).

nabla eager is **5.2–6.9× faster** than PyTorch eager, and **3.5–4.4× faster** than `torch.compile`. nabla CUDA Graph beats PyTorch CUDA Graph by **1.4–1.9×**.

**Single-op and matmul benchmarks (for reference):**

| Workload | nabla | PyTorch 2.10 (default) | PyTorch 2.10 (TF32=ON) | PyTorch 2.10 (FP16) |
|---|---|---|---|---|
| matmul 4096×4096 (f32) | 0.372 ms | 2.664 ms | 0.326 ms — ~parity | — |
| matmul 4096×4096 (f16) | **0.189 ms** | — | — | 0.168 ms |
| matmul 1024×1024 | 0.036 ms | 0.057 ms | — | — |
| `exp` + `sin` fused | **0.041 ms** | 0.081 ms | — | — |
| `sin` / `cos` / `tanh` | 0.040 ms | 0.041 ms | ~parity | — |
| `add` / `sub` / element-wise | 0.058 ms | 0.058 ms | ~parity | — |
| 4-op `fuse!` speedup vs unfused | **3.38×** | — | — | — |

> nabla explicitly enables TF32 Tensor Core math (`CUBLAS_TF32_TENSOR_OP_MATH`) at init.
> PyTorch 2.10.0 ships with `allow_tf32 = False` by default (verified on GH200). When both use TF32=ON, performance is ~equal.
> **FP16 is the fairest single-op comparison:** nabla 0.189 ms vs PyTorch 0.168 ms — ~parity (within measurement noise).
> Single-op comparisons don't show the full picture; the training table above does.

#### Why: Python costs ~7 µs per kernel launch

Every `tensor.exp()` call in PyTorch has to travel through this stack before the GPU sees it:

```
your Python code
  → Python interpreter  (~1 µs)
    → PyTorch Python API  (~1 µs)
      → ATen C++ operator dispatch  (~2 µs)
        → CUDA kernel launch  ← GPU finally starts
```

**Measured on GH200:** small-tensor `exp` chain = **~7 µs per launch** (Python overhead only — the GPU compute is nearly free for small tensors).

One launch is invisible. A training step with 36 kernel launches — measured:

```
PyTorch MLP step (batch=1):   0.670 ms
                              ├─ GPU compute:  ~0.112 ms  (same hardware as nabla)
                              └─ Python overhead: ~0.558 ms  ≈ ~22 launches × 25 µs each
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                              this is the cost you're paying

nabla MLP step (batch=1):     0.112 ms  ← Rust calls CUDA runtime directly, no interpreter
```

The gap is not about GPU speed — it's about how much time Python spends telling the GPU what to do. The bigger your model (more layers, more ops per step), the more this compounds.

```
Training throughput: nabla vs PyTorch eager

Model size (kernel launches per step)
   10 launches:  PyTorch ████░░   nabla ████  (~1.5× faster, measured: small models)
   22 launches:  PyTorch ████████░░░░░   nabla ████  (~6× faster, measured: MLP 784→256→128→10)
  100 launches:  PyTorch ████████████████████░░░░░░░   nabla ████  (~20× faster, extrapolated)
                 ─────── Python overhead grows linearly with op count
                         GPU compute stays constant — nabla is always in this range ────┘
```

#### Second: why plain Rust isn't enough for raw GPU throughput

PyTorch's backend is written in C++ and calls hand-tuned CUDA libraries (cuBLAS, cuDNN) that NVIDIA spends years optimizing. If you just wrote a matrix multiply in ordinary Rust and sent it to the GPU, you would get roughly the same speed as PyTorch — or slower.

nabla has to earn its performance the hard way. Here is what that looks like:

**Reading GPU memory in bulk, not one value at a time**

A GPU has thousands of cores running in parallel. But memory bandwidth is the bottleneck — every time a core reads from GPU memory, it costs time. The trick is to read 128 bits (4 floats) in a single instruction instead of 32 bits (1 float), so you do 4× the math per memory trip.

```
Naive:   core → [reads 1 float] → GPU memory  ←── slow
                [reads 1 float] → GPU memory
                [reads 1 float] → GPU memory
                [reads 1 float] → GPU memory

nabla:   core → [reads 4 floats at once] → GPU memory  ←── 4× fewer memory trips
```

Every element-wise operation in nabla (add, sin, exp, …) uses this 128-bit "float4" loading. PyTorch does too — but nabla enforces it without exception. There is no fallback path.

**Using the GPU's dedicated matrix math hardware**

Modern NVIDIA GPUs have special units called Tensor Cores. They are hardwired to multiply matrices and are dramatically faster than general-purpose floating point for that one job. Using them requires writing WMMA (Warp Matrix Multiply Accumulate) instructions directly in GPU code.

```
General GPU cores:   one multiply-add per cycle per core
Tensor Cores:        an entire 16×16 matrix multiply per cycle  ←── orders of magnitude faster
```

nabla enables TF32 Tensor Core math by default for matmul — PyTorch doesn't. When both use TF32, f32 matmul performance is approximately equal. For FP16, nabla is ~1.1× faster (0.189 ms vs PyTorch's 0.210 ms). The real advantage is in dispatch, not in this single-op comparison.

**No CPU fallback — ever**

In any framework that supports both CPU and GPU, there is always a risk: if an operation isn't implemented on GPU, it silently falls back to CPU. The data has to travel from GPU memory across the PCIe bus to CPU, get computed, then travel back. This round-trip alone takes hundreds of microseconds.

nabla makes CPU fallback a **compile error**. Every operation either has a GPU implementation or the code does not compile. There is no "oops, that quietly ran on CPU" moment.

**Matmul + activation fused at the library level**

A linear layer followed by an activation function (e.g. `linear → ReLU`) is one of the most common patterns in neural networks. Normally this dispatches two GPU kernels: one for the matrix multiply, one for the activation. nabla uses cuBLAS's epilogue API to attach the activation directly to the end of the matmul kernel — one kernel, no intermediate buffer written to GPU memory between them.

```
Normal:  [matmul kernel]──write──►[GPU memory]──read──►[activation kernel]
nabla:   [matmul + activation, single kernel] ← no memory round-trip
```

---

**How nabla keeps the GPU busy**

**1. Rust: zero interpreter overhead**

Rust compiles to native machine code. A tensor call goes straight to the CUDA runtime in a single function call — no interpreter, no Python object overhead, no GIL.

```
PyTorch: ████░░░░████░░░░████░░░░████░░░░ ...
         ████ GPU busy   ░░░░ GPU idle (Python scheduling next op)

nabla:   ████████████████████████████████ ...
         Rust calls CUDA runtime directly — no gaps
```

---

**2. `fuse!` reduces the number of round-trips to the GPU**

Every GPU kernel launch carries a fixed scheduling cost, regardless of how fast the kernel itself runs. In PyTorch, `a.sin().powf(2.0)` launches two separate kernels:

- kernel 1: compute `sin(a)`, write result to a temporary buffer in GPU memory
- kernel 2: read from that buffer, compute `pow(..., 2.0)`, write final result

Two launches. One unnecessary GPU memory round-trip in between.

`fuse!` in nabla analyzes the expression at compile time and emits a **single JIT-compiled kernel** that does both operations in one pass — no intermediate buffer, no second launch:

```rust
let y = fuse!(a.sin().powf(2.0));  // 1 kernel, 0 intermediate buffers
```

---

**3. CUDA Graph replay removes CPU scheduling from the hot path entirely**

A training loop runs the exact same sequence of operations every iteration. CUDA Graph lets nabla record that entire sequence on the first real step, then **replay the recording** on every subsequent step — without the CPU re-issuing each kernel individually.

Replay cost: **~1 μs total per step** instead of the usual hundreds of μs.

```rust
let mut tg = PyGraphTrainingGraph::new(); // warmup 5 iters, then auto-capture
for _ in 0..10_000 {
    tg.step(&mut || {
        // forward + backward + optimizer — identical every iteration
    })?;
    // steps 1-5:  run normally (warmup)
    // step 6:     record the full kernel sequence as a graph
    // steps 7+:   replay the graph — GPU runs uninterrupted
}
```

---

**4. Fused loss and optimizer kernels**

A naive MSE training step dispatches three kernels: `sub → square → sum`. nabla's `k_mse_sum_fwd` does all three in one. Similarly, `k_multi_axpy3` updates all model parameters in a single vectorized GPU pass — PyTorch dispatches one kernel per parameter tensor.

This is why nabla is already faster than PyTorch in eager mode, before CUDA Graph even comes into play.

---

Putting the benchmark numbers in context:

- **5.2–6.9× eager gap** — primarily factors 1 (no interpreter) + 4 (fused kernels)
- **1.0–1.4× CUDA Graph gap** — factors 2 (`fuse!`) + 4 still apply even when Python overhead is gone

Reproduce locally: `cd benchmarks && bash run.sh`

### Autodiff: loss.backward() in Rust

Training a neural network means computing gradients — nabla does this automatically. You write a forward pass normally, call `backward()`, and read the gradients. No manual math required.

```rust
let tape = Tape::new();
let w = tape.var(weights)?;
let b = tape.var(bias)?;

let logits = (&x * &w) + &b;
let loss = logits.cross_entropy_indices(&targets)?;
loss.backward()?;

let dw = w.grad()?;   // Err(NoGradient) if backward wasn't called — no silent None
let db = b.grad()?;
// tape, grads, and intermediates are freed here by Drop
```

All gradient memory is freed immediately when it goes out of scope — no garbage collector, no memory spikes between batches.

For computing how sensitive an output is to each input (Jacobians, sensitivity analysis), **forward-mode AD** is available. You wrap your input in `Dual<T>` and the derivative flows through automatically with zero changes to the math:

```rust
let x = Dual::new(2.0, 1.0);    // value=2, derivative seed=1
let y = (x * x).sin();           // evaluates sin(x²) and d/dx sin(x²) simultaneously
println!("dy/dx = {}", y.dual);  // 2x·cos(x²) ≈ -2.615
```

### Training Stack: nabla-train

`nabla-train` has everything you need to go from a model definition to saved, deployable weights. It is a separate crate so it only adds to your build when you need it.

| Feature | What it does |
|---|---|
| **Optimizers** | AdamW, Adam, SGD — the standard optimizers for deep learning. Set different learning rates per layer group. |
| **LR schedules** | Gradually reduce the learning rate during training. Choose from cosine decay, linear warmup, one-cycle, or step. |
| **DataLoader** | Shuffle, batch, and stream your training data in parallel across CPU threads. |
| **Checkpointing** | Save and restore the full training state — weights plus optimizer momentum — so you can resume after interruption. |
| **Mixed precision** | Train with 16-bit floats for up to 2× speed and half the GPU memory. Automatic overflow detection. |
| **Gradient utilities** | Gradient clipping to prevent exploding gradients; efficient zero-grad across all parameters. |
| **AWQ quantization** | Compress model weights to 4 bits using activation-aware scaling — reduces GPU memory by ~4× with minimal accuracy loss. |
| **GGUF export** | Export trained weights to GGUF with any of the 34 quantization formats (F32/F16/BF16, Q2_K–Q8_0, IQ1–IQ4, TQ1/TQ2). Load directly in llama.cpp, Ollama, or LM Studio. |
| **ONNX export** | Export your trained model to the standard format accepted by TensorFlow, Core ML, ONNX Runtime, and more. |
| **Profiler** | See exactly how long each layer takes, how much GPU memory it uses, and whether it's bottlenecked on compute or memory bandwidth. |

**Training loop** — `train_step!` handles the boilerplate (zero grad, forward, backward, optimizer step) in one call:

```rust
use nabla_train::prelude::*;

let mut optimizer = AdamW::from_params(1e-3, &model.parameters());
for epoch in 0..100 {
    for (x, targets) in &loader {
        let loss = train_step!(model, optimizer, tape, |x, out| {
            out.cross_entropy_indices(&targets)
        })?;
        println!("loss: {:.4}", loss);
    }
}
```

**DataLoader** — wraps any `Dataset` impl, handles shuffling and batching:

```rust
let loader = DataLoader::new(dataset, VecBatcher::default(), 64)
    .shuffle_seed(42);   // shuffle with fixed seed

for (x, y) in &loader {
    // x: Tensor<f32>, y: Tensor<i64> — already batched
}
```

**Checkpointing** — saves weights and optimizer state together so training can resume exactly where it stopped:

```rust
// Save after each epoch
save_checkpoint(&model, &optimizer, Path::new("ckpt.bin"))?;

// Resume on next run — optimizer momentum is restored automatically
load_checkpoint(&mut model, &mut optimizer, Path::new("ckpt.bin"))?;
```

**Profiler** — shows per-layer timing, GPU memory, and whether each layer is bottlenecked on compute or memory:

```rust
let mut prof = Profiler::new();
prof.start();
let _ = train_step!(model, optimizer, tape, |x, out| out.mse_loss(&target))?;
let report = prof.stop();
println!("{}", serde_json::to_string_pretty(&report)?);
```

Example output:
```json
{
  "peak_vram_mb": 1847,
  "total_ms": 0.133,
  "layers": [
    { "name": "Linear(512→256)", "forward_ms": 0.021, "backward_ms": 0.038, "vram_mb": 512, "tflops": 18.4, "bound": "compute" },
    { "name": "LayerNorm",       "forward_ms": 0.004, "backward_ms": 0.006, "vram_mb": 8,   "tflops": 2.1,  "bound": "memory"  }
  ]
}
```

`bound: "memory"` means the layer is limited by how fast data moves between GPU memory and compute units — usually solvable by fusion or larger batch size.

**GGUF export** — export to the format used by llama.cpp, Ollama, and LM Studio. Choose a quantization format to reduce model size:

| Format | Bits/weight | 7B model size | Quality vs F32 |
|---|---|---|---|
| F32 | 32 | 26 GB | Baseline |
| F16 | 16 | 13 GB | Identical |
| Q8_0 | 9.0 | 7.2 GB | Near-identical |
| **Q4_K_M** | **4.8** | **4.1 GB** | **Good — recommended** |
| Q4_0 | 4.5 | 3.8 GB | Good |
| Q3_K_M | 3.9 | 3.3 GB | Acceptable |
| Q2_K | 2.6 | 2.2 GB | Noticeable degradation |

All 34 formats are supported — see the full list in [notation.md](docs-en/notation.md).

```rust
use nabla_train::gguf::{GgufExportConfig, GgufQuantType, export_gguf};
use std::fs::File;

let config = GgufExportConfig {
    base_quant: GgufQuantType::Q4KM,
    model_arch: "llama".into(),
    mixing: None,    // or Some(MixingPreset::...) for per-layer precision
    imatrix: None,   // required for IQ formats — provide calibration importance matrix
    extra_metadata: vec![],
};

// Collect weights as (name, shape, f32_data) tuples
let weights: Vec<_> = model.named_parameters()
    .into_iter()
    .map(|(name, t)| (name, t.shape().to_vec(), t.to_vec()))
    .collect();
let weight_refs: Vec<(&str, &[u64], &[f32])> = weights
    .iter()
    .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
    .collect();

let mut file = File::create("model.gguf")?;
export_gguf(&mut file, &weight_refs, &config)?;
```

**ONNX export** — export to the standard interchange format, runnable in TensorFlow, Core ML, and ONNX Runtime:

```rust
use nabla_train::onnx::export_sequential;
let onnx = export_sequential(&model, input_features as i64, output_features as i64);
onnx.save(Path::new("model.onnx"))?;
```

**AWQ quantization** — compresses model weights to INT4 before export, with activation-aware scaling to minimize accuracy loss:

```rust
use nabla_train::quantize::{CalibrationStats, quantize_awq};

// Build calibration stats from a small representative dataset
let mut calib = CalibrationStats::new(num_channels);
for batch in &calib_loader {
    calib.update(&batch.activations);  // feed activation batches
}
// Quantize individual weight matrix with group_size=128
let qw = quantize_awq(&weight_tensor, &calib, 128);
```

**Benchmark evaluation** — measure perplexity and accuracy of a model against a test dataset:

```rust
use nabla_train::benchmark::{compute_perplexity, compute_accuracy};

let perp = compute_perplexity(&model, &test_dataset, forward_fn)?;
let acc  = compute_accuracy(&model, &test_dataset, forward_fn)?;
println!("perplexity: {:.2}  accuracy: {:.1}%", perp.perplexity, acc.accuracy * 100.0);
```

### Model Inference: nabla-interface

`nabla-interface` loads GGUF files and runs inference via llama.cpp. On Apple Silicon, transformer layers are automatically offloaded to Metal GPU.

```rust
use nabla_interface::{InferenceEngine, InferenceConfig, SamplingConfig};

let engine = InferenceEngine::new(
    "model.gguf",
    InferenceConfig {
        n_ctx: 4096,       // context window size
        n_gpu_layers: 32,  // how many transformer layers to run on GPU (set to 0 for CPU-only)
        ..Default::default()
    },
)?;

// One-shot generation
let text = engine.generate(
    "Explain backpropagation in simple terms:",
    256,  // max tokens to generate
    &SamplingConfig {
        temperature: 0.7,   // higher = more creative, lower = more deterministic
        top_p: 0.9,         // nucleus sampling: consider only top 90% probability mass
        repeat_penalty: 1.1,
        ..Default::default()
    },
)?;

// Streaming — print tokens as they are generated
for token in engine.generate_stream("Hello", 64, &SamplingConfig::default())? {
    print!("{token}");
    std::io::stdout().flush()?;
}

// Performance stats
let stats = engine.perf();
println!("prompt {:.1} tok/s  gen {:.1} tok/s", stats.prompt_tok_per_sec, stats.gen_tok_per_sec);
```

**Full end-to-end: train → export → run locally**

```rust
use nabla_train::prelude::*;
use nabla_train::gguf::{GgufExportConfig, GgufQuantType, export_gguf};
use nabla_train::quantize::{CalibrationStats, quantize_awq};
use nabla_interface::{InferenceEngine, InferenceConfig, SamplingConfig};
use std::fs::File;

// 1. Train
let mut optimizer = AdamW::from_params(1e-4, &model.parameters());
for _ in 0..epochs { train_step!(model, optimizer, tape, |x, out| out.cross_entropy_indices(&y))?; }
save_checkpoint(&model, &optimizer, Path::new("ckpt.bin"))?;

// 2. Export to GGUF (Q4_K_M = recommended quality/size tradeoff)
let weights: Vec<_> = model.named_parameters()
    .into_iter()
    .map(|(n, t)| (n, t.shape().to_vec(), t.to_vec()))
    .collect();
let weight_refs: Vec<_> = weights.iter().map(|(n,s,d)| (n.as_str(), s.as_slice(), d.as_slice())).collect();
let config = GgufExportConfig { base_quant: GgufQuantType::Q4KM, model_arch: "llama".into(), mixing: None, imatrix: None, extra_metadata: vec![] };
export_gguf(&mut File::create("model.gguf")?, &weight_refs, &config)?;

// 3. Run with nabla-interface — or load the .gguf in Ollama / LM Studio
let engine = InferenceEngine::new("model.gguf", InferenceConfig::default())?;
let out = engine.generate("prompt", 128, &SamplingConfig::default())?;
```

### Symbolic Math and ODE Solvers

Sometimes you need exact math, not numerical approximations. nabla includes a symbolic algebra system — you can define expressions with variables, differentiate them analytically, simplify the result, and then evaluate numerically. No need to reach for Python's SymPy.

```rust
use nabla::cas::*;

let f = sym!(x^2 * sin(x));                 // ^ is exponentiation, not XOR
let df = diff_simplify(&f, "x");            // → x²·cos(x) + 2x·sin(x)
let val = eval(&df, &[("x", 1.5)].into())?; // domain-checked numeric evaluation

let grad = gradient(&sym!(x^2 + y^2), &["x", "y"]);   // ∇f = [2x, 2y]
let j = jacobian(&[sym!(x*y), sym!(x+y)], &["x","y"]); // 2×2 Jacobian
```

For simulating physical systems (oscillators, fluid dynamics, orbital mechanics), nabla includes ODE and SDE solvers from simple Euler to adaptive step-size and stiff-system solvers:

| Solver | Good for | Order |
|---|---|---|
| `euler` / `rk4` | Quick experiments | 1 / 4 |
| `dormand_prince` | General use (adaptive step size) | 5(4) |
| `bdf1` / `bdf2` | Stiff systems (e.g. chemical kinetics) | 1 / 2 |
| `stormer_verlet` | Energy-preserving systems (e.g. orbital mechanics) | 2 |
| `parareal` | Long time horizons, parallel across time | — |
| `euler_maruyama` / `milstein` | Stochastic ODEs (SDEs) | 0.5 / 1.0 |

```rust
// Lorenz attractor
let sol = rk4(|_t, y| lorenz(y), &y0, (0.0, 50.0), 0.001)?;
println!("{:.4}", sol.eval(25.0));   // interpolate at t=25
```

### No Silent Errors

PyTorch returns `nan` silently when you invert a singular matrix or take a gradient that doesn't exist. nabla returns `Result`:

| Operation | PyTorch | nabla |
|---|---|---|
| Singular matrix solve | returns `nan` | `Err(SingularMatrix)` |
| Missing gradient | returns `None` | `Err(NoGradient)` |
| Non-scalar `backward()` | raises Python exception | `Err(NonScalarOutput)` |
| Shape mismatch in `einsum!` | runtime error | **compile error** |

This means bugs surface immediately at the call site, with the full Rust error chain, not silently downstream.

### Macro DSL: Write Math, Not Boilerplate

These macros let you express mathematical operations directly in code without ceremony.

**`einsum!`** — express any tensor contraction (matmul, dot product, trace, batched matmul) in one line using index notation. Shape mismatches are caught at **compile time**:

```rust
let c: Tensor<f64> = einsum!(c[i,j] = a[i,k] * b[k,j]);  // matmul
let y: Tensor<f64> = einsum!(y[i]   = a[i,k] * x[k]);     // matrix-vector
let s: f64         = einsum!(s      = a[i,i]);              // trace
let r: Tensor<f64> = einsum!(r[b,i,j] = a[b,i,k] * m[b,k,j]); // batched matmul
```

**`math!`** — write `w * x + bias` without the `&` noise Rust normally requires for tensor operands:

```rust
let out = math!(w * x + bias);   // expands to: &w * &x + &bias
```

**`stencil!`** — write finite-difference stencils (e.g. for PDEs) as a single expression with automatic boundary handling:

```rust
stencil!(laplacian[i,j] = -4.0 * u[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]);
```

**`impl_layer!` and `#[derive(Module)]`** — define a custom neural network layer without writing boilerplate. `impl_layer!` takes a forward function body and generates the full `Module` trait implementation automatically. `#[derive(Module)]` generates `parameters()`, `state_dict()`, `load_state_dict()` from your struct fields:

```rust
impl_layer! {
    MyLinear { weight; bias }
    forward(x) {
        match bias { Some(b) => x.tl_matmul(&weight.tl_t()).tl_add(b),
                     None    => x.tl_matmul(&weight.tl_t()) }
    }
}

#[derive(Module)]
struct Attention<T: Scalar, B: Backend> {
    #[param]           wq: Tensor<T, B>,
    #[param]           wk: Tensor<T, B>,
    #[param]           wv: Tensor<T, B>,
    #[param(optional)] proj_bias: Option<Tensor<T, B>>,
    training: bool,
}
```

---

## Installation

Pick exactly one backend. No CUDA SDK or Vulkan SDK is required to compile nabla — GPU libraries are loaded dynamically at runtime via `libloading`.

```toml
[dependencies]
# CPU (default):
nabla = { git = "https://github.com/fumishiki/nabla", features = ["cpu"] }

# GPU — uncomment exactly one, remove the cpu line:
# nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["cuda"] }  # NVIDIA
# nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["wgpu"] }  # Vulkan/Metal/DX12
# nabla = { git = "https://github.com/fumishiki/nabla", default-features = false, features = ["hip"] }   # AMD

# Training stack:
# nabla-train = { git = "https://github.com/fumishiki/nabla" }

# Model export (GGUF) and inference:
# nabla-interface = { git = "https://github.com/fumishiki/nabla" }
# nabla-interface = { git = "https://github.com/fumishiki/nabla", features = ["llama"] }
```

Switching from CPU to GPU requires **no code changes** — only the feature flag changes.

**CPU**

| Feature | f32 | f64 | f16 / bf16 | Complex |
|---|---|---|---|---|
| `cpu` | ✅ | ✅ | ✅ | ✅ |

**GPU**

| Feature | Hardware | f32 | f64 |
|---|---|---|---|
| `cuda` | NVIDIA GPU | ✅ | ✅ |
| `hip` | AMD GPU | ✅ | ✅ |
| `wgpu` | Vulkan / Metal / DX12 (cross-platform) | ✅ | ❌ |

> **wgpu f64:** WGSL (WebGPU Shading Language) does not include `f64` in its core specification — all storage buffers and arithmetic are `f32` only. Use `cuda`, `hip`, or `cpu` for `f64` workloads.

Complex number support is CPU-only. No CUDA or Vulkan SDK installation required — GPU libraries are loaded at runtime.

---

## Getting Started

Three pointers to get you started:
- [Quick Start](docs-en/quick_start.md) — full API walkthrough with runnable examples
- [Notation reference](docs-en/notation.md) — every macro, type, and method in one place
- [Spec](docs-en/spec.md) — architecture, performance bounds, and design decisions

Run the examples locally:

```bash
cargo run --example 01_matrix_ops    --features cpu   # matrix ops and LU solve
cargo run --example 04_autograd_mlp  --features cpu   # reverse-mode autodiff
cargo run --example 05_ode_lorenz    --features cpu   # Lorenz attractor
cargo run --example 07_einsum_attention --features cpu # self-attention via einsum!
cargo run --example 08_cas_symbolic  --features cpu   # symbolic differentiation
```

---

## Resources

- [Quick Start](docs-en/quick_start.md)
- [Notation reference](docs-en/notation.md)
- [Architecture spec](docs-en/spec.md)
- [Codebase directory](docs-en/directory.md)

---

## Contributing

Fork → feature branch → `cargo test && cargo clippy && cargo fmt --check` → PR against `main`.

Please open an issue before submitting large new features so we can discuss direction first.

---

**fumishiki** — [GitHub](https://github.com/fumishiki) · [X](https://x.com/fumishiki) · [LinkedIn](https://linkedin.com/in/fumitakamurakami) · [Hugging Face](https://huggingface.co/fumishiki)

## License

[Apache-2.0](LICENSE-APACHE) OR [MIT](LICENSE-MIT), at your option.
