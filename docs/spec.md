# nabla — Specification

> **AI AGENT 必読**: §0 が**唯一の行動命令書**。§1–§7 は参照資料。着手前に §0 を全て確認すること。

**Ground Truth**: `docs/spec.md`（このファイル）が唯一の仕様書。コード・コメントと矛盾する場合、このファイルを優先する。
関連ドキュメント: [notation.md](notation.md) — API・記法・型規約・einsum・autograd | [directory.md](directory.md) — プロジェクト構造 | [history.md](history.md) — 実装履歴・最適化ログ

**完了条件**: 全 REQ 実装 + 全受け入れテスト通過 = Done。1 REQ でも未実装なら done 禁止。

Legend: ✅ Implemented | ❌ Not possible (language constraint) | 🔲 Not yet implemented

---

## §0 Binding Contract

### §0.0 NON-GOALS（実装禁止）

| # | 禁止対象 |
|---|---|
| NG-01 | CPU fallback for any GPU backend operation |
| NG-02 | Multiple active backends at compile time |
| NG-03 | `unwrap()`/`expect()` in library code |
| NG-04 | Architecture decisions (model structure) |
| NG-05 | Internal unit tests (`#[cfg(test)] mod tests`) — use crate-level `tests/*.rs` only |
| NG-06 | External GPU frameworks — use direct WGSL/CUDA/HIP C kernel strings |
| NG-07 | f64 on wgpu backend (WGSL/Metal lacks f64 hardware support) |
| NG-08 | c32/c64 on any GPU backend |
| NG-09 | `unimplemented!()`/`todo!()` in any `pub` function |
| NG-10 | Synchronous GPU→CPU transfer except on `.get()`/`.to_vec()` callsite |

### §0.1 REQ（Phase 0/3A–3E 完了）

30 REQs across Phase 0/3A–3E — **全実装・全テスト通過**。変更禁止。
テストファイル: `nabla-ml/tests/gpu.rs` (`cargo test --features cuda -- gpu_phase3`)

| Phase | REQ 数 | 概要 |
|---|---|---|
| 0 | 5 | Backend exclusivity, prelude, Tensor alias, type guards |
| 3A | 7 | GPU conv1d/2d/3d/transpose (im2col+GEMM) |
| 3B | 5 | GPU max/avg/adaptive pooling |
| 3C | 7 | FlashAttention-2, bmm/baddbmm/addmm (cuBLAS) |
| 3D | 6 | GPU cumsum/cumprod (Blelloch), prod_all, norm, count_nonzero |
| 3E | 3 | GPU batch_norm, fused cross_entropy |

### §0.1.1 REQ（Phase 4 — Done ✅）

| Phase | REQ 数 | 概要 |
|---|---|---|
| 4 | 9 | Loss 6種 + group_norm + topk + sort |

受け入れテスト（追加）:
- `nabla-ml/tests/ops_nn.rs`: mse/l1/smooth_l1/bce_logits/nll/cosine_embedding + group_norm の CPU 正答確認
- `nabla-ml/tests/ops_tensor_extended.rs`: topk/sort の CPU 正答確認

### §0.1.2 REQ（Phase 5 — Done ✅）

| Phase | REQ 数 | 概要 |
|---|---|---|
| 5 | 28 | Train stack (optimizer/scheduler/amp/trainer/dataloader/ckpt/dist/logging) |

Phase 5 の対象範囲:
- Optimizer: SGD/Adam/AdamW + schedule + param groups/presets
- Trainer: train/eval/metrics/early-stop/AMP/grad clip + resume
- Dataloader: shuffle/seed/repeat/prefetch + split
- Checkpoint: params/optimizer/scheduler/scaler/RNG
- Dist/Logging: allreduce + stdout/JSON

#### Phase 5 — REQ Table

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P5-OPT-01 | Optimizer | SGD (momentum/weight_decay) を提供 | ✅ |
| P5-OPT-02 | Optimizer | Adam を提供 | ✅ |
| P5-OPT-03 | Optimizer | AdamW 既存を維持し、共通 `Optimizer` trait で統一 | ✅ |
| P5-OPT-04 | Optimizer | `set_lr` と lr schedule を統合（step 単位） | ✅ |
| P5-OPT-05 | Optimizer | Param groups: name-based grouping + per-group lr/weight_decay | ✅ |
| P5-OPT-06 | Optimizer | Weight decay exclusion helper（bias/norm 系） | ✅ |
| P5-OPT-07 | Optimizer | Param group exclusion presets（bias/norm/embedding） | ✅ |
| P5-TRN-01 | Trainer | 1-epoch/step ループ + grad accumulation | ✅ |
| P5-TRN-02 | Trainer | AMP: GradScaler 統合（scale/unscale/overflow） | ✅ |
| P5-TRN-03 | Trainer | metrics/early-stop フック（条件式） | ✅ |
| P5-TRN-04 | Trainer | resume: step/epoch の継続 | ✅ |
| P5-TRN-05 | Trainer | grad clip (global norm) を step 前に適用 | ✅ |
| P5-TRN-06 | Trainer | train/eval 切替 + val ループ | ✅ |
| P5-TRN-07 | Trainer | metrics 集計（avg/last/min/max） | ✅ |
| P5-TRN-08 | Trainer | grad NaN/Inf 検知（skip/stop） | ✅ |
| P5-DL-01 | Dataloader | Dataset trait + index-based access | ✅ |
| P5-DL-02 | Dataloader | Sampler (sequential/shuffle/seeded) | ✅ |
| P5-DL-03 | Dataloader | Batch + drop_last + repeat | ✅ |
| P5-DL-04 | Dataloader | Prefetch (single-thread) | ✅ |
| P5-DL-05 | Dataloader | Subset + split helper (train/val/test, seed) | ✅ |
| P5-DL-06 | Dataloader | seed 制御 API + shuffle 再現性 | ✅ |
| P5-CKPT-01 | Checkpoint | params + optimizer + scheduler + scaler の save/load | ✅ |
| P5-CKPT-02 | Checkpoint | Module::state_dict と互換な key/name で保存 | ✅ |
| P5-CKPT-03 | Checkpoint | RNG state + scheduler state + group optimizer state を保存 | ✅ |
| P5-DIST-01 | Dist | CPU-only 2-rank allreduce (mean) | ✅ |
| P5-LOG-01 | Logging | metrics moving average + stdout logger | ✅ |
| P5-LOG-02 | Logging | logging 出力先切替（stdout / JSON file） | ✅ |

受け入れテスト（追加）:
- `nabla-train/tests/optim.rs`: SGD/Adam/AdamW の CPU 正答確認 + lr schedule + GradScaler + param groups
- `nabla-train/tests/trainer.rs`: train/eval + grad accum + resume + grad clip + metrics + early-stop + checkpoint
- `nabla-train/tests/data.rs`: shuffle/seed/repeat/prefetch + split/Subset + dist allreduce

### §0.1.3 REQ（Phase 6 — Done ✅）

| Phase | REQ 数 | 概要 |
|---|---|---|
| 6 | 30 | Profiling + AWQ INT4 量子化 + ONNX/GGUF エクスポート + ベンチマーク評価 |

対象クレート: `nabla-train`（profiler.rs, quantize.rs, onnx.rs, gguf.rs, benchmark.rs）

#### Profiling

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-PROF-01 | Profiler | CudaEvent ベースの per-kernel 実行時間計測（start/stop/elapsed_ms） | ✅ |
| P6-PROF-02 | Profiler | per-layer 統計（forward/backward 各レイヤーの時間・VRAM・TFLOPS） | ✅ |
| P6-PROF-03 | Profiler | tok/s・latency(ms/token)・batch throughput 自動算出 | ✅ |
| P6-PROF-04 | Profiler | TFLOPS 算出: 2MKN/elapsed（matmul）、各カーネルの FLOPs 推定 | ✅ |
| P6-PROF-05 | Profiler | Roofline 判定: min(peak_TFLOPS, AI × peak_BW) で compute/memory bound 分類 | ✅ |
| P6-PROF-06 | Profiler | VRAM 使用量トラッキング（peak / current / per-layer breakdown） | ✅ |
| P6-PROF-07 | Profiler | JSON 出力（kernel breakdown + roofline + per-layer stats） | ✅ |

#### AWQ INT4 Weight-Only 量子化

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-AWQ-01 | Quantize | キャリブレーション: 入力データから per-channel activation 統計を収集 | ✅ |
| P6-AWQ-02 | Quantize | per-channel scale 最適化（activation-aware、grid search） | ✅ |
| P6-AWQ-03 | Quantize | INT4 パッキング（8 weights → 1 u32、リトルエンディアン） | ✅ |
| P6-AWQ-04 | Quantize | CUDA dequant-matmul カーネル（INT4 unpack → f16/f32 → GEMM） | ✅ |
| P6-AWQ-05 | Quantize | group_size パラメータ（デフォルト 128）で量子化粒度制御 | ✅ |

#### ONNX エクスポート

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-ONNX-01 | Export | Module trait walk → ONNX NodeProto DAG 構築 | ✅ |
| P6-ONNX-02 | Export | protobuf シリアライズ（minimal encoder、外部依存なし） | ✅ |
| P6-ONNX-03 | Export | opset 21 準拠（MatMul, Relu, Conv, LayerNorm 等） | ✅ |
| P6-ONNX-04 | Export | dynamic axes サポート（batch_size, seq_len） | ✅ |
| P6-ONNX-05 | Export | onnxruntime での推論検証（数値一致 atol=1e-5） | ✅ |

#### GGUF エクスポート

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-GGUF-01 | Export | GGUF v3 バイナリライター（magic + version + metadata KV + tensor info + data） | ✅ |
| P6-GGUF-02 | Export | `GgufQuantType` enum — Legacy: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, K-quant: Q2_K/Q3_K_S/M/L/Q4_K_S/M/Q5_K_S/M/Q6_K, IQ: IQ1_S/M/IQ2_XXS/XS/S/M/IQ3_XXS/XS/S/M/IQ4_XS/NL, Ternary: TQ1_0/TQ2_0, Full: F16/BF16 | ✅ |
| P6-GGUF-03 | Export | Legacy block 構造（QK=32）: Q*_0 は delta(f16)+qs, Q*_1 は delta+min(f16×2)+qs | ✅ |
| P6-GGUF-04 | Export | K-quant block 構造（QK_K=256）: super-block d/dmin(f16) + sub-block scales + packed qs | ✅ |
| P6-GGUF-05 | Export | IQ block 構造（QK_K=256）: E8 格子 / 非線形 LUT / importance-matrix 対応 | ✅ |
| P6-GGUF-06 | Export | Ternary block 構造: TQ1_0(trit packing 5^5) / TQ2_0(2bit 3値) | ✅ |
| P6-GGUF-07 | Export | _S/_M/_L レイヤー混合戦略（per-tensor 量子化タイプ選択、attention/ffn で精度配分） | ✅ |
| P6-GGUF-08 | Export | imatrix 対応（キャリブレーションデータから重要度行列算出、IQ 系で必須） | ✅ |
| P6-GGUF-09 | Export | llama.cpp での読み込み・推論検証（全量子化タイプ） | ✅ |

#### ベンチマーク評価

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| P6-BENCH-01 | Benchmark | データセットローダー（ユーザー指定のテストデータを読み込み） | ✅ |
| P6-BENCH-02 | Benchmark | Perplexity 計測（言語モデル向け cross-entropy → exp） | ✅ |
| P6-BENCH-03 | Benchmark | Accuracy / Top-k accuracy 計測（分類タスク向け） | ✅ |
| P6-BENCH-04 | Benchmark | 評価結果の JSON 出力（metrics + per-sample scores + 統計サマリ） | ✅ |

受け入れテスト（追加）:
- `nabla-train/tests/profiler.rs`: CudaEvent timing + per-layer stats + roofline 判定 + JSON 出力
- `nabla-train/tests/quantize.rs`: AWQ calibration + INT4 pack/unpack round-trip + dequant-matmul 精度
- `nabla-train/tests/export_onnx.rs`: Module → ONNX → onnxruntime 推論 → 数値一致
- `nabla-train/tests/export_gguf.rs`: Module → GGUF → Q4_0/Q4_K → llama.cpp 読み込み検証
- `nabla-train/tests/benchmark.rs`: データセット読み込み → perplexity/accuracy 計測 → JSON 出力

### §0.1.4 REQ（Phase 7: nabla-interface — Done ✅）

| Phase | REQ 数 | 概要 |
|---|---|---|
| 7 | 29 | GGUF v3 Export + 34型量子化 + llama.cpp FFI + M-series Metal 推論 |

対象クレート: `nabla-interface`（Layer 5: Export + Inference）

```
nabla-train (学習)
    ↓ state_dict / checkpoint
nabla-interface
    ├── gguf.rs      GGUF v3 writer (pure Rust)
    ├── quant/       量子化パッキング 34型 (pure Rust)
    │   ├── mod.rs   GgufQuantType enum + dispatch
    │   ├── legacy.rs  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1
    │   └── kquant.rs  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
    ├── convert.rs   Module state_dict → GGUF tensor mapping
    ├── llama.rs     llama.cpp FFI (system libllama via pkg-config)
    └── serve.rs     推論パイプライン (tokenize → decode → sample)
```

**依存関係**: `nabla-ml`(Module/Tensor) + `nabla-core`(Scalar/Backend) + System: `libllama`(via `pkg-config --libs llama`) + macOS Apple Silicon + `brew install llama.cpp`

**NON-GOALS**: 独自 Metal/GPU カーネル（llama.cpp に委譲） | llama.cpp ソース同梱（system lib 前提） | tokenizer 独自実装 | CPU fallback for inference

#### GGUF Export

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| IF-GGUF-01 | Writer | GGUF v3 バイナリライター: magic(`GGUF`) + version(3) + tensor_count + metadata_kv_count + metadata KV pairs + tensor_info array + padding(32B alignment) + tensor data | ✅ |
| IF-GGUF-02 | Writer | metadata KV: string/u32/f32/u64/array 型。必須キー: `general.architecture`, `general.name`, `general.file_type`, `{arch}.context_length`, `{arch}.embedding_length`, `{arch}.block_count`, `{arch}.attention.head_count`, `{arch}.attention.head_count_kv` | ✅ |
| IF-GGUF-03 | Writer | tensor_info: name(string) + n_dims(u32) + dims([u64]) + type(GgufType) + offset(u64)。offset は data セクション先頭からの相対位置、32B aligned | ✅ |

#### Quantization

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| IF-QUANT-01 | Quant | `GgufQuantType` enum: 34 variants matching ggml — F32, F16, BF16, F64, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K_{S,M,L}, Q4_K_{S,M}, Q5_K_{S,M}, Q6_K, IQ1_{S,M}, IQ2_{XXS,XS,S}, IQ3_{XXS,S}, IQ4_{NL,XS}, I8, I16, I32, I64, TQ1_0, TQ2_0 | ✅ |
| IF-QUANT-02 | Quant | Legacy Q4_0: block_size=32, delta=absmax/15(f16) + 32 nibbles packed to 16 bytes | ✅ |
| IF-QUANT-03 | Quant | Legacy Q8_0: block_size=32, delta=absmax/127(f16) + 32 i8 values | ✅ |
| IF-QUANT-04 | Quant | K-quant Q4_K_M: QK_K=256, super-block d/dmin(f16) + 12-byte sub-block scales + 128-byte packed qs | ✅ |
| IF-QUANT-05 | Quant | quantize/dequantize round-trip: `dequant(quant(tensor))` の最大絶対誤差が Q4_0 で ≤ 2*absmax/15, Q8_0 で ≤ 2*absmax/127 | ✅ |
| IF-QUANT-06 | Quant | _S/_M strategy: attention weights に高精度(Q6_K/Q5_K)、FFN に低精度(Q4_K) を自動割当 | ✅ |
| IF-QUANT-07 | Quant | Legacy Q4_1/Q5_0/Q5_1/Q8_1 + K-quant Q2_K/Q3_K/Q5_K/Q6_K の quantize/dequantize 実装。IQ/Ternary/Integer は enum のみ（`is_quantizable()` = false） | ✅ |

#### Convert (nabla → GGUF)

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| IF-CONV-01 | Convert | `Module::state_dict()` → GGUF tensor name mapping。規則: `layers.{i}.attention.wq.weight` → `blk.{i}.attn_q.weight` 等 | ✅ |
| IF-CONV-02 | Convert | `export_gguf(state_dict, path, quant_type, config, overrides)` 一発変換 API | ✅ |
| IF-CONV-03 | Convert | `GgufArchConfig` struct で architecture/context_length/embedding_length/block_count/head_count/head_count_kv/vocab_size を指定 | ✅ |
| IF-CONV-04 | Convert | per-tensor 量子化オーバーライド: `QuantOverride` で特定テンソルの量子化タイプを個別指定可能 | ✅ |

#### llama.cpp FFI

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| IF-FFI-01 | FFI | `LlamaBackend` RAII guard（init/free ライフタイム管理） | ✅ |
| IF-FFI-02 | FFI | `LlamaModel::load(path, n_gpu_layers)` → RAII, Drop で `llama_model_free` | ✅ |
| IF-FFI-03 | FFI | `LlamaContext::new(model, n_ctx, n_batch, n_threads)` → RAII | ✅ |
| IF-FFI-04 | FFI | `tokenize(&str, add_bos)` / `detokenize(&[i32])` safe wrapper | ✅ |
| IF-FFI-05 | FFI | `llama_decode` safe wrapper（`LlamaBatch`） | ✅ |
| IF-FFI-06 | FFI | `SamplerChain::new().temperature(t).top_k(k).top_p(p).build()` builder API | ✅ |
| IF-FFI-07 | FFI | `sampler.sample(&mut ctx, idx)` safe wrapper | ✅ |
| IF-FFI-08 | FFI | Metal backend 自動検出（`n_gpu_layers = -1` で全レイヤー GPU offload） | ✅ |
| IF-FFI-09 | FFI | build.rs: `pkg-config` で libllama を検出（homebrew は llama.pc に全ライブラリを含む） | ✅ |

#### Inference Pipeline

| REQ-ID | Unit | Requirement | Status |
|---|---|---|---|
| IF-SERVE-01 | Serve | `InferenceEngine::new(gguf_path, config)` → GGUF ロード + コンテキスト初期化 | ✅ |
| IF-SERVE-02 | Serve | `engine.generate(prompt, max_tokens, sampling)` → `String` テキスト生成 | ✅ |
| IF-SERVE-03 | Serve | `engine.generate_stream(prompt, max_tokens, sampling)` → `impl Iterator<Item = String>` | ✅ |
| IF-SERVE-04 | Serve | `InferenceConfig`: n_ctx(2048), n_batch(512), n_threads(num_cpus), n_gpu_layers(-1=all) | ✅ |
| IF-SERVE-05 | Serve | `SamplingConfig`: temperature(0.8), top_k(40), top_p(0.95), repeat_penalty(1.1), seed(Option) | ✅ |
| IF-SERVE-06 | Serve | `engine.perf()` → `PerfStats { prompt_tok_per_sec, gen_tok_per_sec, total_tokens }` | ✅ |

#### Layer Name Mapping (nabla → GGUF)

| nabla state_dict key | GGUF tensor name |
|---|---|
| `embedding.weight` | `token_embd.weight` |
| `layers.{i}.attention.wq.weight` | `blk.{i}.attn_q.weight` |
| `layers.{i}.attention.wk.weight` | `blk.{i}.attn_k.weight` |
| `layers.{i}.attention.wv.weight` | `blk.{i}.attn_v.weight` |
| `layers.{i}.attention.wo.weight` | `blk.{i}.attn_output.weight` |
| `layers.{i}.attention_norm.weight` | `blk.{i}.attn_norm.weight` |
| `layers.{i}.ffn.w1.weight` / `w2` / `w3` | `blk.{i}.ffn_gate` / `ffn_down` / `ffn_up.weight` |
| `layers.{i}.ffn_norm.weight` | `blk.{i}.ffn_norm.weight` |
| `norm.weight` | `output_norm.weight` |
| `output.weight` | `output.weight` |

#### Quantization Block Structures

| Type | Category | QK | Bytes/block | Layout |
|---|---|---|---|---|
| Q4_0 | Legacy | 32 | 18 | d(f16) + qs([u8;16] nibbles, centered at 8) |
| Q4_1 | Legacy | 32 | 20 | d(f16) + min(f16) + qs([u8;16]) |
| Q5_0 | Legacy | 32 | 22 | d(f16) + qh([u8;4] high bits) + qs([u8;16]) |
| Q5_1 | Legacy | 32 | 24 | d(f16) + min(f16) + qh([u8;4]) + qs([u8;16]) |
| Q8_0 | Legacy | 32 | 34 | d(f16) + qs([i8;32]) |
| Q8_1 | Legacy | 32 | 36 | d(f16) + s(f16 sum) + qs([i8;32]) |
| Q2_K | K-quant | 256 | 84 | d/dmin(f16) + scales + qs(2-bit packed) |
| Q3_K | K-quant | 256 | 110 | d(f16) + hmask + qs + scales |
| Q4_K | K-quant | 256 | 144 | d/dmin(f16) + scales([u8;12] 6-bit packed) + qs([u8;128]) |
| Q5_K | K-quant | 256 | 176 | d/dmin(f16) + scales + qh + qs |
| Q6_K | K-quant | 256 | 210 | d(f16) + ql + qh + scales |

受け入れテスト:
- `nabla-interface/tests/gguf_writer.rs`: GGUF v3 バイナリ書き出し → magic/version/metadata/tensor_info のバイト列検証 (5 tests)
- `nabla-interface/tests/quant_roundtrip.rs`: Q4_0/Q8_0/Q4_K_M の quantize→dequantize round-trip 誤差検証 (5 tests)
- `nabla-interface/tests/convert.rs`: nabla Tensor → GGUF ファイル出力 → ファイルサイズ・metadata 検証 (5 tests)
- `nabla-interface/tests/llama_load.rs`: GGUF → llama.cpp load → tokenize/detokenize 検証 (1 test, requires llama.cpp)
- `nabla-interface/tests/llama_inference.rs`: GGUF load → generate → テキスト生成 + perf stats (1 test, requires llama.cpp)

### §0.2 Ground Truth Parameters

```
GPU kernel hyperparameters (pinned):
  BLOCK_SIZE:          256  (default; tunable per kernel via autotune — do NOT hardcode)
  float4 vectorization: all f32 unary/binary kernels (128-bit LDG.E.128 + scalar tail)
  warp size:           32  (CUDA/HIP)
  workgroup size (wgpu): 256

FlashAttention-2 tile sizes:
  BLOCK_M: 64   BLOCK_N: 64   HEAD_DIM_MAX: 128

im2col+GEMM for conv2d:
  im2col col: [N, C_in*kH*kW, out_H*out_W]  weight: [C_out, C_in*kH*kW]
  GEMM: cublasSgemmStridedBatched per batch

Blelloch parallel scan: O(n) work, O(log n) depth, smem = 2*BLOCK_SIZE*sizeof(T)

Pinned dependency versions:
  Rust edition: 2024 | wgpu: 24 | cudarc: 0.19 | rayon: 1
```

### §0.3 Phase（全完了 ✅）

```
Phase 0 (基盤) → 3A (Conv) → 3B (Pool) → 3C (Attention) → 3D (Reduction) → 3E (Norm/Loss) → Phase 4 (Performance)
```

---

## §1 Overview

**Zero-GC, zero-copy, type-safe** Rust linear algebra DSL for researchers who refuse to choose between Python's ease and C++'s speed.
Proc macros (`mat![]`, `map!{}`, `einsum!{}`, `math!`, `fuse!`, `train_step!`) combined with self-contained pure-Rust kernels (CPU) and GPU compute shaders across three backends: wgpu (WGSL), CUDA (nvrtc), HIP (hiprtc).
Exactly one backend is selected via feature flags at build time — no implicit CPU fallback, no cross-backend runtime dispatch.

### Fixed Rule Principle — computation engine, not framework

nabla は **計算エンジン** であり、フレームワークではない。数学的に不変な計算プリミティブ（matmul, conv, softmax, cross_entropy 等）を CPU/GPU で最速に実行する。

| nabla provides | User decides |
|---|---|
| matmul, conv, bmm, SDPA, embedding | layer stacking, skip connections |
| activations (relu/gelu/silu/softmax) | which activation where |
| loss (cross_entropy/mse/l1/kl_div) | which loss to optimize |
| norm (layer/rms/batch/group) | where to normalize |
| reductions, reshape, gather, scatter | data flow topology |
| AD (reverse/forward), ODE, CAS | what to differentiate, solver choice |
| optimizer, scheduler, trainer, dataloader | model architecture decisions |

Four-layer architecture: `nabla-macros` (notation) → `nabla-core` (compute) → `nabla-ml` (application) → `nabla-train` (training). 詳細は [directory.md](directory.md) 参照。

---

## §2 Architecture

### 2.1 Backend selection

Exactly one of `{cpu, wgpu, cuda, hip}` via feature flag. All 6 pairwise conflicts → `compile_error!`.

| Feature | `DefaultBackend` | Storage | f64 | c32/c64 |
|---|---|---|---|---|
| `cpu` (default) | `Cpu` | `Vec<T>` row-major | ✅ | ✅ |
| `wgpu` | `Gpu` | `wgpu::Buffer` | ❌ | ❌ |
| `cuda` | `Cuda` | `CUdeviceptr` | ✅ | ❌ |
| `hip` | `Hip` | `hipDeviceptr_t` | ✅ | ❌ |

All tensors use `Tensor<T>` = `Tensor<T, DefaultBackend>`. Backend trait: all computation methods are **required** (no default body) — no implicit CPU fallback.

### 2.2 Macro dispatch

| Macro | CPU | GPU | Strategy |
|---|---|---|---|
| `einsum!` | ✅ GEMM | ✅ GPU kernel | `matmul_into` (Backend dispatch) |
| `fuse!` | ✅ `from_fn` | ✅ JIT fused kernel | NVRTC/hiprtc codegen |
| `map!` / `stencil!` / `par_*` | ✅ | ❌ compile error | CPU-only (closure / offset access / rayon) |
| `math!` | ✅ | ✅ | Auto-borrow idents in expressions |
| `mat!` / `splat!` / `named!` | ✅ | ✅ | Compile-time expansion |

### 2.3 GPU implementation

```
             ┌───────────────────────────┐
             │  GpuContext trait          │
             │  GpuStorage<T> (lazy D2H) │
             └─────┬──────────┬──────────┘
               wgpu│     cuda │      hip │
         WGSL shaders  CUDA C (nvrtc)  HIP C (hiprtc)
            43 ops      100+ ops       100+ ops
```

**GpuContext trait**: `alloc`, `upload`, `download`, `launch`. Singleton via `OnceLock`.
**GpuStorage**: backend buffer + lazy `Mutex<Option<Vec<T>>>` host cache. Readback on `.get()`/`.to_vec()` only.

### 2.4 Kernel sources — two codebases

74 CUDA/HIP ops + 43 WGSL ops, same semantics, two shader languages:

| Category | CUDA/HIP | wgpu (WGSL) |
|---|---|---|
| Binary (add/sub/mul/div/scale) | float4 vectorized | `elementwise_binary` |
| Unary (15 ops: exp/ln/sin/cos/tanh/sqrt/abs/erf/...) | float4 + fast math | `elementwise_unary` |
| Activation (sigmoid/silu/mish/leaky_relu/elu/hardswish) | float4 vectorized | dedicated shaders |
| Matmul | tiled TILE=16/32, WMMA/MMA tensor cores | tiled TILE=16, register-tile MMA |
| Reduction (sum/max/min/argmax/argmin) | warp shuffle | workgroup reduce |
| Row-wise (softmax/layer_norm/rms_norm) | 3-pass warp shuffle | workgroup reduce |
| Axis reduction (sum_axis/max_axis) | warp shuffle | workgroup reduce |
| Gather (embedding) | thread-per-element | thread-per-element |
| Phase 3 (conv/pool/attn/scan/norm/loss) | 30+ dedicated kernels | CPU defaults |

f32: `float4` 128-bit loads + `__expf`/`__logf` fast-math. f64: scalar (1 elem/thread).
Runtime compilation: nvrtc/hiprtc → PTX cache (`HashMap<String, KernelEntry>`). Architecture auto-detected.

### 2.5 Kernel fusion

| Level | Scope | Backend |
|---|---|---|
| L1: Element-wise | Consecutive unary/binary → single JIT kernel | GPU (fuse!) |
| L2: Reduction | Across reduction ops with loop-carried deps | CPU |
| L3: GEMM+pointwise | matmul + activation (cublasLt epilogue) | GPU |
| L4: Map-reduce | Pointwise + axis reduction → single kernel | GPU |
| L4: Mega-kernel | Shared memory tile reuse for multi-op mega_fuse! | GPU |
| L4: DAG fusion | `prev` keyword for inter-op register pass-through | GPU |

Pipeline: `fuse!` AST → egg EqSat simplify → `cuda_expr()` → NVRTC/hiprtc JIT → FNV-1a hash cache.

### 2.6 GPU advanced features

| Feature | Backend | Approach |
|---|---|---|
| WMMA/MMA tensor cores | cuda/hip | `nvcuda::wmma::mma_sync` (Volta+), `rocwmma` (CDNA2+) |
| Warp shuffle reductions | cuda/hip | `__shfl_down_sync`, 8x reduction |
| Register-tile MMA (wgpu) | wgpu | Software MMA via register tiling |
| Linear Layouts F₂ swizzle | all | Bank-conflict-free for any tile size |
| Caching memory allocator | cuda/hip | Best-fit dual-pool, 512B-aligned, GC 0.9 |
| Async execution pipeline | cuda/hip | Defer sync until readback only |
| CUDA Graph capture/replay | cuda | `TrainingGraph` API, 1.67× speedup |
| cuBLAS workspace pre-alloc | cuda | `cublasSetWorkspace_v2` 32MiB |
| Mega-kernel tiled fusion | cuda/hip | Shared memory tile reuse (≥2 ops, ≥64K elements) |

---

## §3 Feature Catalog — 190+ ops

nabla は計算エンジンとして PyTorch の `torch.*` / `torch.nn.functional.*` が提供する数学的に固定された計算を網羅する。API 詳細・引数シグネチャは [notation.md](notation.md) 参照。

### 3.1 Summary

| Category | Count | Key ops | GPU |
|---|---|---|---|
| **A. Convolution** | 4 | conv1d/2d/3d, conv_transpose2d | ✅ im2col+cuBLAS |
| **B. Pooling** | 4 | max/avg/adaptive_avg pool2d, max_pool1d | ✅ (2d), 🔲 (1d) |
| **C. Normalization** | 4 | layer/rms/batch/group_norm | ✅ GPU kernels |
| **D. Activation** | 10 | relu, gelu, sigmoid, softmax, silu, mish, leaky_relu, elu, hardswish, log_softmax | ✅ float4 + fuse! |
| **E. Loss** | 8 | cross_entropy, mse, l1, smooth_l1, bce_logits, nll, kl_div, cosine_embedding | ✅ fused GPU |
| **F. Attention** | 3 | SDPA (FlashAttention-2), multi_head_attention, embedding | ✅ GPU kernels |
| **G. Manipulation** | 19 | reshape/permute/cat/stack/squeeze/flatten/chunk/pad/gather/scatter/index_select/masked_fill/where/triu/tril/roll/flip/meshgrid/topk/sort | ✅ CPU |
| **H. Batched** | 4 | bmm, baddbmm, addmm, batched reductions | ✅ cuBLAS |
| **I. Construction** | 11 | zeros/ones/full/eye/arange/linspace/rand/randn/empty/clone/contiguous | ✅ |
| **J. Reduction** | 10 | sum/max/min/mean/var/std/argmax/argmin (all + axis), cumsum/cumprod, prod, norm, count_nonzero | ✅ GPU (Blelloch, warp shuffle) |
| **K. Regularization** | 3 | dropout, interpolate_nearest, interpolate_bilinear | ✅ CPU |

### 3.2 Element-wise math (prelude)

`exp` `ln` `log1p` `log2` `log10` `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `asinh` `acosh` `atanh` `sqrt` `abs` `recip` `erf` `ceil` `floor` `round` `powf` `neg` `sign` `rem` — all GPU-accelerated (float4 vectorized).

Operators: `+` `-` `*` `/` (owned + borrowed combos), `scalar * &Tensor`, `epow(&b)`, `hadamard(&rhs)`.

### 3.3 Linear algebra (CPU)

45+ methods via `LinalgExt` trait (f32/f64): `svd`, `qr`, `lu`, `cholesky`, `eig`, `inv`, `solve`, `lstsq`, `det`, `trace`, `rank`, `norm`, `cond`, `pinv`. Structural: `Diagonal`, `Symmetric`, `Triangular`. GPU: `gpu_trsm_lower` (recursive GEMM) only.

### 3.4 Sparse

`BcsrMatrix<T>` BCSR + `WGSL_BCSR_SPMM` kernel (GPU). `mixed_spmm_f64` mixed-precision refinement. CPU: CSC.

### 3.5 CAS (Computer Algebra System)

`egg` 57 unified rules (`cas_rules()`: 33 algebraic + 24 differentiation) + `diff_simplify` (41 rules) + `FuseExpr` 16-node EqSat (18 IEEE-754 safe rules). `gradient`/`jacobian`/`hessian` (auto-simplify). Domain-checked `eval`/`eval_tensor`. Method chain API. 詳細は [notation.md](notation.md) §3.10 参照。

### 3.6 ODE/SDE solvers

| Solver | Type | Order |
|---|---|---|
| `euler` | Explicit | 1 |
| `rk4` | Explicit | 4 |
| `dormand_prince` | Adaptive | 5(4) |
| `bdf1` / `bdf2` | Implicit (stiff) | 1 / 2 |
| `if_euler` | Exponential integrator | — |
| `metd` | Matrix exponential | p |
| `stormer_verlet` | Symplectic | 2 |
| `parareal` | Time-parallel | — |
| `euler_maruyama` / `milstein` | SDE | 0.5 / 1.0 |

`OdeProblem<T,B,F>` wrapper. `ensemble_euler_maruyama` (parallel N-trajectory). Backward integration. `OdeSolution::eval(t)` interpolation. 詳細は [notation.md](notation.md) §3.10 参照。

### 3.7 Autograd

| Mode | Approach | Status |
|---|---|---|
| Reverse (tape) | `GpuTape<T>`: 14-op enum, backward via GPU kernels | ✅ |
| Forward (dual) | `Dual<T>`: `impl Scalar for Dual<T>` — all ops unchanged | ✅ |
| Source-transform | `#[nabla_grad]` proc macro → `f_grad(x) -> (T, T)` | ✅ |

NN ops: softmax, reshape, transpose, linear_forward, dropout, clamp, loss ops. Module/Optimizer: `Module` trait, `Sequential`, `AdamW`, `GradScaler`. `impl_var_op!` macro で std::ops trait impls のボイラープレートを吸収（Add/Sub/Mul × 4 ownership combos）。詳細は [notation.md](notation.md) §7 参照。

---

## §4 Design Decisions & Limitations

| Decision | Rationale |
|---|---|
| Direct kernel strings | Fixed-rule ops → 2 codebases manageable |
| Build-time exclusive backend | CPU fallback is a performance bug source |
| Runtime kernel compilation | nvrtc/hiprtc: no SDK at build time |
| Handle-based GPU storage | Chained ops eliminate host↔device transfer |
| TypeId dispatch | Backend trait sealed + `T: Scalar`, avoids E0276 |
| Embedded kernel strings | WGSL + CUDA/HIP C as `const &str` |
| Native Rust (no C++ wrapper) | Ownership-native tensor > FFI wrapper |
| Recursive GEMM for GPU linalg | Reuse matmul_tiled |
| Einsum canonicalization | 4.7x over JAX |
| Named axes | Compile-time dimension safety |
| `impl Scalar for Dual<T>` | Forward-mode AD as zero-change drop-in |
| Macro DSL absorbs verbosity | Julia 比 5-10x LOC gap を macro で圧縮 |

| Limitation | Mitigation |
|---|---|
| No wgpu f64 | Use `cuda`/`hip` backend |
| No GPU c32/c64 | Compile error (by design) |
| GPU linalg: TRSM only | Full LU/Cholesky/QR CPU only |
| `from_fn` requires host | Use `fuse!` for GPU |
| 2 kernel codebases | WGSL ≠ CUDA/HIP C — fixed ops, rarely changes |
| No REPL | `rust-script` + `cargo watch` |

---

## §5 Performance

### Benchmark (GH200 480GB, 4096×4096 f32)

| Workload | nabla | PyTorch 2.7 | 備考 |
|---|---|---|---|
| exp / sin / tanh | 0.040 ms | 0.040–0.041 ms | ≈ parity |
| add / emul | 0.058 ms | 0.058 ms | ≈ parity |
| fuse exp+sin | 0.041 ms | 0.081 ms (eager) | **nabla 2× faster** |
| fuse 4-op | 0.050 ms | — | single kernel |
| mega_fuse 4-out | 0.141 ms | — | single kernel, 4 outputs |
| sum_all / max_all | 0.028 ms | 0.026 ms | PyTorch 1.08× |
| matmul 4096 (cuBLAS TF32) | 0.378 ms | 2.68 ms | **nabla 7× faster** |
| CUDA Graph 36-op | 56 μs/step | — | 1.67× vs eager |

完了済み最適化の詳細は [history.md](history.md) 参照。

---
