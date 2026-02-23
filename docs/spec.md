# nabla — Rust Linear Algebra DSL 統合仕様書

## 概要

推論特化の条件下で、型安全・ゼロコピー・マルチバックエンドを備えた Rust 線形代数 DSL を提供する。
proc macro による記法レイヤ（`mat![]`, `bcast!{}`, `einsum!{}`）と faer ベースの高性能カーネルを組み合わせ、
簡潔な数理記述と C-ABI FFI 互換を両立する。

凡例: ✅ 実装済み | 🔧 未実装(実装可能) | ❌ Impossible(スキップ)

---

## 1. プロジェクト構造

```
~/Desktop/GitHub/nabla/
├── Cargo.toml           [workspace] nabla + macros
├── macros/
│   ├── Cargo.toml       proc-macro crate (syn/quote/proc-macro2)
│   └── src/
│       ├── lib.rs       mat![] + einsum![] entry points
│       └── einsum.rs    einsum parser + codegen
├── src/
│   ├── lib.rs           crate root + pub mod scalar + pub mod error + pub mod util (307 lines)
│   ├── tensor.rs        Tensor<T,B> + StaticMatrix<T,R,C> + Array/Matrix traits (626 lines)
│   ├── backend.rs       Backend trait + Cpu impl + struct defs + DefaultBackend (273 lines)
│   ├── gpu.rs           GpuStorage + CubeCL kernels + TypeId dispatch (646 lines)
│   ├── linalg.rs        Dense factorization + Diagonal/Symmetric/Triangular wrappers (535 lines)
│   └── sparse.rs        SparseMatrix<T> CSC (339 lines)
├── tests/
│   ├── basic.rs               integration tests (61)
│   ├── broadcast.rs           broadcast macro tests (7)
│   ├── static_mat.rs          StaticMatrix + hierarchy trait tests (18)
│   ├── gpu.rs                 GPU backend tests, wgpu feature-gated (13)
│   ├── einsum_compile_errors.rs  trybuild compile-fail runner (1)
│   └── einsum_errors/         spanned error .rs + .stderr fixtures (3)
└── docs/
    └── spec.md          本ファイル
```

Tests: 87 CPU + 1 trybuild + 13 GPU + 10 doc-tests | Clippy: clean

### 依存関係

- `faer = "0.24"` / `faer-traits = "0.24"` / `rayon = "1"`
- `nabla-macros = { path = "macros" }`
- `cubecl-core / cubecl-cuda / cubecl-wgpu / cubecl-runtime` (optional, version 0.9)

---

## 2. 機能仕様 + 実装状況

### A. コア構文

| ID | 機能 | 記法 | nabla / Rust | 状態 |
|---|---|---|---|---|
| A1 | 暗黙乗算 | `2x` | `2.0 * x` | ❌ パーサ制約 |
| A2 | 連鎖比較 | `0 < x < 1` | `between!(0.0, x, 1.0)` macro_rules! | ✅ lib.rs (util module) |
| A3 | 複素数リテラル | `2 + 3i` | `c32(re, im)` / `c64(re, im)` helper | ✅ lib.rs (util module) |
| A4 | 有理数リテラル | `3//4` | `Ratio::new(3, 4)` | ❌ `//` はコメント |
| A5 | Range/Step | `0.0:0.1:1.0` | `frange!()` / `linspace()` | ✅ lib.rs (util module) |
| A6 | 行列リテラル | `[1 2; 3 4]` | `mat![[1.0, 2.0], [3.0, 4.0]]` | ✅ proc macro |
| A7 | Unicode識別子 | `α`, `β` | Rust native `let α = 3.14_f64;` | ✅ native |
| A7 | Unicode中置演算子 | `÷`, `∘` | — | ❌ 固定ASCII演算子のみ |
| A8 | Transpose | `transpose(A)` | `.t()` | ✅ tensor.rs |
| A8 | Adjoint | `A'` | `.adjoint()` (IS_REAL分岐+conj) | ✅ tensor.rs |
| A8 | 後置構文 | `A'` | — | ❌ 後置演算子不可 |
| A9 | パイプ | `\|>` | `pipe!(val, f, g)` macro (`#[macro_export]`) | ✅ pipe! macro |
| A10 | Splatting | `f(args...)` | `itertools::chain!` / tuple展開 | 🔧 Hard |
| A11 | タプル分割代入 | `a, b = f()` | `let (a, b) = f();` | ✅ native |
| A12 | Named Tuple | `(a=1, b=2.0)` | struct宣言必須 | 🔧 Hard |

### B. Broadcasting・ベクトル化

| ID | 機能 | 記法 | nabla 戦略 | 状態 |
|---|---|---|---|---|
| B1 | Dot-Call | `f.(x, y)` | `bcast!(f, &a, &b)` macro_rules! | ✅ lib.rs (util module) |
| B2 | `@.` All | `@. y = sin(x)^2` | `bcast!{}` 統合 (全関数dot挿入) | 🔧 Hard |
| B3 | In-Place `.=` | `A .= B .* C` | `zip_map!(out, f, &a, &b)` macro_rules! | ✅ lib.rs (util module) |

### C. 配列・コレクション

| ID | 機能 | 記法 | nabla / Rust | 状態 |
|---|---|---|---|---|
| C1 | 配列内包 | `[x^2 for x in 1:10]` | `.map().filter().collect()` | ✅ native |
| C1 | 多次元 | `Array(f, m, n)` | `Tensor::from_fn(m, n, f)` | ✅ tensor.rs |
| C2 | 多次元Indexing | `A[1:3, 2:4]` | `.slice(rows, cols)` / `.slice_rows()` / `.slice_cols()` | ✅ tensor.rs |
| C3 | `@view` | `@view A[1:3, :]` | faer default view (zero-copy) | ✅ native |
| C4 | Static行列 | `SMatrix{3,3}` | `StaticMatrix<T,R,C>` const generics (stack-allocated) | ✅ tensor.rs |

### D. 型システム・ディスパッチ

| ID | 機能 | 記法 | nabla / Rust | 状態 |
|---|---|---|---|---|
| D1 | Multiple Dispatch (open) | 全引数型選択 | — | ❌ Orphan rule |
| D1 | Multiple Dispatch (closed) | — | `DynTensor` enum + `match` dispatch | ✅ tensor.rs DynTensor |
| D2 | Abstract型階層 | `abstract type` | `trait Matrix: Array` | ✅ tensor.rs (全Backend対応, `t_dyn`/`matmul_dyn`→`Tensor<T,Cpu>`) |
| D3 | Parametric Types | `Point{T<:Real}` | `struct Point<T: Real>` | ✅ native |
| D4 | `@generated` | コンパイル時関数本体生成 | const generics + proc macro | 🔧 Hard |
| D5 | Type Piracy | open method追加 | Newtype pattern | ❌ 設計上 |

### E. メタプログラミング

| ID | 機能 | 記法 | nabla / Rust | 状態 |
|---|---|---|---|---|
| E1 | AST Macro | `quote`/`$` | `syn` + `quote!` proc macro (基盤あり) | ✅ 基盤 / 🔧 拡張 |
| E2 | `eval` / Runtime CG | `eval(:(x+1))` | — | ❌ AOTコンパイル |

### F. パフォーマンスアノテーション

| ID | 機能 | 記法 | nabla / Rust | 状態 |
|---|---|---|---|---|
| F1 | `@inbounds` | bounds check除去 | iterator-based auto-elim / `get_unchecked` | ✅ native |
| F2 | `@simd` | SIMD code gen | LLVM auto-vec + `pulp` (faer internal) | ✅ implicit |
| F3 | `@turbo` | SIMD+MT schedule | faer `pulp` + `rayon` 組合せ | 🔧 Hard |
| F4 | `@views` | block view化 | faer default view | ✅ native |

### G. 線形代数stdlib

| ID | 機能 | 記法 | nabla 戦略 | 状態 |
|---|---|---|---|---|
| G1 | 構造行列型 | `Symmetric(A)`, `Diagonal(v)` | `Diagonal` / `Symmetric` / `Triangular` wrapper | ✅ linalg.rs |
| G2 | Factorization | `factorize(A)`, `F \ b` | faer `.partial_piv_lu()` 等 wrap + `.solve()` | ✅ linalg.rs |
| G3 | In-Place BLAS | `mul!(C, A, B)` | `Tensor::matmul_into(&mut out, &a, &b)` | ✅ tensor.rs |

### H. スパース配列

| ID | 機能 | 記法 | nabla 戦略 | 状態 |
|---|---|---|---|---|
| H1 | SparseArrays | CSCデフォルト | `SparseMatrix<T>` (CSC wrap + factorization + solve) | ✅ sparse.rs |

### I. 記号・DSLレイヤ

| ID | 機能 | 記法 | nabla 戦略 | 状態 |
|---|---|---|---|---|
| I1 | Symbolics CAS | `@variables x y` | `symbolica` (別crate検討) | 🔧 Hard (外部) |
| I2 | ODE DSL | `D(x) ~ σ*(y-x)` | `diffsol` | 🔧 Hard (外部) |
| I3 | Einsum (standard) | `@tullio` | `einsum!{}` proc macro | ✅ macros/einsum.rs |
| I3a | Einsum → GEMM 最適化 | `c[i,j]=a[i,k]*b[k,j]` → matmul | コンパイル時パターン認識 → faer/gpu_matmul dispatch | ✅ Wave 9 |
| I3b | N-D einsum 拡張 | `c[b,i,j]=a[b,i,k]*m[b,k,j]` | N 階テンソル縮約 + batch/free/contraction 3 分類 (パーサ拡張済み, N-D codegen は 2-D 制約) | ✅ Wave 9 Phase 1 |
| I3c | einsum! spanned エラー | `syn::Error::new_spanned` | ユーザコードへの精密エラー表示 | ✅ Wave 9 |
| I3 | Tullio (full) | stencil/conv | offset indexing proc macro | 🔧 Hard |
| I4 | GPU Kernels | `KernelAbstractions` | `cubecl` `#[cube(launch)]` | ✅ gpu.rs |
| I5 | Units | `Unitful.jl` | `uom` crate | 🔧 Easy (外部) |

### J. バックエンド

| ID | 機能 | 状態 | 備考 |
|---|---|---|---|
| J | Backend trait (sealed) | ✅ | `Backend<Storage<T>>` GAT, 14 methods |
| J | Cpu impl | ✅ | `faer::Mat<T>` + 全メソッド委譲 (backend.rs) |
| J | Cuda impl | ✅ | `GpuStorage<T>` Handle-based + cubecl kernels (gpu.rs) |
| J | Wgpu impl | ✅ | `GpuStorage<T>` Handle-based + cubecl kernels (gpu.rs) |
| J | Hip GPU backend | ✅ | cubecl_hip::HipRuntime via impl_gpu_backend! (same kernel path as Cuda/Wgpu) |
| J | DefaultBackend cfg | ✅ | cuda > wgpu > hip > cpu 優先 |

---

## 3. faer 0.24 API リファレンス

### 3.1 spec.md からの主な乖離

| spec 記載 | faer 0.24 実態 |
|---|---|
| `faer::Entity` trait | **存在しない** → `faer_traits::ComplexField` のみ |
| `faer::ComplexField` direct | private → `faer-traits` crate 経由で import |
| `faer::complex_native::c32` | `faer::c32` (num_complex::Complex<f32> alias) |
| `mat.read(row, col)` | `*mat.get(row, col)` (参照返し) |
| matmul alpha/beta | `matmul(dst, Accum::Replace, lhs, rhs, T::one_impl(), Par::Seq)` |
| scalar multiply | `mat * Scale(s)` (Scale wrapper 必須) |
| Symmetric/Triangular view | **wrapper 型なし** — `Side` enum + メソッドベース |
| faer-sparse 別 crate | `faer::sparse` に統合済み |
| faer 0.20 | 実際は **0.24** を使用 |
| cubecl 0.8 | 実際は **0.9** を使用 |

### 3.2 Dense Matrix — `faer::Mat<T>`

構築: `Mat::zeros(nrows, ncols)` / `Mat::from_fn(nrows, ncols, |r,c| ...)` / `Mat::identity(n,n)`

アクセス: `.nrows()` / `.ncols()` / `*mat.get(row, col)` / `.as_ref()` (MatRef, zero-copy) / `.as_mut()` (MatMut, zero-copy)

演算: `&a + &b` / `&a - &b` / `-&a` / `&a * Scale(s)` / `faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, T::one_impl(), Par::Seq)`

型定数: `T::one_impl()` / `T::zero_impl()` / `T::IS_REAL`

Transpose/Adjoint (zero-copy): `mat.as_ref().transpose()` / `mat.as_ref().adjoint()`

### 3.3 Factorizations

全て `MatRef` 上のメソッド。`use faer::prelude::*` で `Solve` trait を import。

| メソッド | 戻り値 | 備考 |
|---|---|---|
| `.partial_piv_lu()` | `PartialPivLu<T>` | PA = LU |
| `.full_piv_lu()` | `FullPivLu<T>` | PAQ^T = LU |
| `.qr()` | `Qr<T>` | A = QR |
| `.col_piv_qr()` | `ColPivQr<T>` | AP^T = QR |
| `.llt(side)?` | `Result<Llt<T>>` | A = LL^T |
| `.ldlt(side)?` | `Result<Ldlt<T>>` | LDL^T |
| `.lblt(side)` | `Lblt<T>` | Bunch-Kaufman, infallible |
| `.svd()?` | `Result<Svd<T>>` | A = USV^H |
| `.thin_svd()?` | `Result<Svd<T>>` | compact U/V |
| `.singular_values()?` | `Result<Vec<Real<T>>>` | — |
| `.self_adjoint_eigen(side)?` | `Result<SelfAdjointEigen<T>>` | — |

Solve trait (全 factorization 共通): `.solve(&b)` / `.solve_in_place(&mut b)` / `.solve_transpose(&b)` / `.solve_adjoint(&b)` / `.rsolve(&b)` / `.inverse()` / `.reconstruct()` / `.solve_lstsq(&b)` (Qr/ColPivQr/Svd のみ)

### 3.4 Structural Types

faer 0.24 に Symmetric/Triangular の **wrapper view 型は存在しない**。

**Diagonal** — `Diag<T>` / `DiagRef` / `DiagMut`: `mat.as_ref().diagonal()` で zero-copy 取得。`.column_vector()` / `.nrows()` / `d[i]`

**Triangular** — メソッドベース: `.copy_from_triangular_lower/upper()` / `.solve_lower/upper_triangular_in_place()` / `.solve_unit_lower/upper_triangular_in_place()`

**Symmetric** — `Side` enum で伝達: `.llt(Side::Lower)` / `.self_adjoint_eigen(Side::Lower)`

### 3.5 Slicing / Submatrix (全て zero-copy)

`view.get(r, c)` / `view.get(1..3, 1..3)` / `view.get(1, ..)` (RowRef) / `view.get(.., 2)` (ColRef) / `.submatrix(rs, cs, nr, nc)` / `.subrows()` / `.subcols()` / `.row(i)` / `.col(j)` / `.diagonal()` / `.split_at(row, col)` → `(TL, TR, BL, BR)`

### 3.6 Sparse — `faer::sparse`

**Format**: `SparseColMat<I, T>` (CSC) / `SparseRowMat<I, T>` (CSR)

構築: `SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &entries)` (triplet重複は合算)

アクセス: `.as_ref().parts()` → `(symbolic, &[T])` / `symbolic.col_ptrs()` / `.row_indices()` / `.nnz()`

Sparse × Dense: `sparse_dense_matmul(dst.as_mut(), Accum::Replace, sp.as_ref(), dense.as_ref(), 1.0, Par::Seq)`

Sparse Solve (two-phase: symbolic + numeric): `SymbolicLlt/Lu/Qr::try_new(sp.symbolic(), ...)` → `Llt/Lu/Qr::try_new_with_symbolic(sym, sp.as_ref(), ...)` → `.solve(&b)` / `.solve_lstsq(&b)`

---

## 4. 設計原則

1. **Zero-copy優先** — 所有権・借用を最大限活用し、in-place操作を実現。`_into(out: &mut)` 規約
2. **Macro = 記法レイヤ** — `bcast!{}`, `einsum!{}` 等のproc macroで簡潔な記法を提供。内部は型安全Rust
3. **trait = dispatch** — Multiple dispatchの必要十分をtraitで表現
4. **faer中心** — lazy Adjoint・構造型・in-place操作を備えた高性能 Rust 線形代数ライブラリを基盤として採用
5. **Backend generic** — 全演算が `B: Backend` でgeneric。`to_backend::<B>()`/`to_cpu()`/`to_wgpu()`/`to_cuda()` で相互変換可能。feature flagでCPU/GPU切替、ランタイムコストゼロ
6. **推論特化** — autodiff不要、forward passの数理記述に集中
7. **Adjoint ≠ Transpose** — 複素数LAでの区別を正確に扱う

---

## 5. バックエンド構成

| Feature Flag | バックエンド | Crate | 状態 |
|---|---|---|---|
| `cpu` (default) | CPU + rayon並列 | faer + rayon | ✅ 完全実装 |
| `cuda` | NVIDIA GPU | cubecl-cuda 0.9 | ✅ GpuStorage + 6 kernels |
| `wgpu` | Vulkan/Metal/DX12 | cubecl-wgpu 0.9 | ✅ GpuStorage + 6 kernels |
| `hip` | AMD GPU | cubecl-hip | ✅ impl_gpu_backend!(Hip, HipRuntime) |

DefaultBackend 優先: cuda > wgpu > cpu

### 5.1 CPU/GPU 混在利用

GPU featureが有効な場合でも CPU と GPU を同時に使用できる。

**型の使い分け:**

| 型 | 説明 |
|---|---|
| `Tensor<f32>` | DefaultBackend（cuda/wgpu有効時はGPU, それ以外はCPU） |
| `Tensor<f32, Cpu>` | 明示的CPUテンソル（常に利用可能） |
| `Tensor<f32, Wgpu>` | 明示的wgpuテンソル（`wgpu` feature必須） |
| `Tensor<f32, Cuda>` | 明示的CUDAテンソル（`cuda` feature必須） |

**バックエンド変換メソッド:**

| メソッド | 説明 |
|---|---|
| `.to_backend::<B2>()` | 任意バックエンドへコピー |
| `.to_cpu()` | CPUテンソルへコピー（常に利用可能） |
| `.to_wgpu()` | wgpuテンソルへコピー（`wgpu` feature） |
| `.to_cuda()` | CUDAテンソルへコピー（`cuda` feature） |

**制約事項（faer依存）:**
- `linalg.rs`の密行列ソルバ（LU/QR/SVD/Cholesky）は `Tensor<T, Cpu>` のみ対応
- `sparse.rs`のスパースソルバも `Tensor<T, Cpu>` のみ対応
- GPU → CPU変換: `.to_cpu()` → linalg/sparse操作 → 必要に応じて `.to_gpu()`

### 5.2 CubeCL 0.9.0 Complete API Reference

Section 3（faer）と対をなす GPU 側の完全 API リファレンス。ソースコード調査に基づく。

#### 5.2.1 クレートエコシステム

| クレート | バージョン | 状態 | 用途 |
|---|---|---|---|
| `cubecl` | 0.9.0 | ✅ stable | facade（全 re-export） |
| `cubecl-core` | 0.9.0 | ✅ stable | カーネル DSL, 型, launch, intrinsics |
| `cubecl-runtime` | 0.9.0 | ✅ stable | ComputeClient, Handle, メモリ管理 |
| `cubecl-std` | 0.9.0 | ✅ stable | TensorHandle, contiguous 変換, identity, FastDivmod |
| `cubecl-wgpu` | 0.9.0 | ✅ stable | wgpu backend (Vulkan/Metal/DX12) |
| `cubecl-cuda` | 0.9.0 | ✅ stable | CUDA backend |
| `cubecl-hip` | 0.9.0 | ✅ stable | HIP/ROCm backend |
| `cubecl-matmul` | 0.9.0-pre.5 | ⚠️ pre のみ | 高性能 matmul (TensorCore) — **stable 未リリース** |
| `cubecl-zspace` | 0.9.0 | experimental | stride/layout ユーティリティ |

**cubecl-std 0.9.0 に含まれないもの（重要）**:
❌ reduction (sum/mean/max/min/argmax/argmin) | ❌ sort | ❌ scan/prefix sum | ❌ Flash Attention

#### 5.2.2 ランタイムモデル

```
Runtime trait
├── type Compiler: Compiler           // カーネルコンパイラ
├── type Server: ComputeServer        // GPU サーバー
├── type Device: Device               // デバイス識別子
├── fn client(device) -> ComputeClient<Self>  // クライアント取得
├── fn name(client) -> &str           // バックエンド名
├── fn supported_line_sizes() -> &[LineSize]  // ベクトル化サイズ
├── fn max_cube_count() -> (u32,u32,u32)      // 最大グリッド
└── fn target_properties() -> TargetProperties
```

| 概念 | CubeCL 型 | 備考 |
|---|---|---|
| ランタイム | `R: Runtime` | 全 GPU API の型パラメータ |
| デバイス | `R::Device` | `Default` trait 経由でデフォルトデバイス取得 |
| クライアント | `ComputeClient<R>` | GPU 操作の主エントリポイント (Arc-backed, Clone = cheap) |
| ハンドル | `server::Handle` | GPU メモリ参照 (RAII, Clone = ref-count++) |
| バインディング | `server::Binding` | launch 用の不透明参照 |
| アロケーション | `server::Allocation` | Handle + strides |

クライアント取得: `R::client(&R::Device::default())` — 毎回呼んでよい（内部 Arc 共有）

#### 5.2.3 GPU メモリ管理 — `ComputeClient<R>` 全メソッド

**メモリ確保:**

| メソッド | 戻り値 | 説明 |
|---|---|---|
| `create(bytes)` | `Handle` | Bytes → デバイス upload |
| `create_from_slice(&[u8])` | `Handle` | スライス → デバイス upload |
| `empty(size_bytes)` | `Handle` | 未初期化デバイスメモリ確保 |
| `create_tensor(&[u8], &[usize], elem_size)` | `Allocation` | stride 付きテンソル確保 |
| `empty_tensor(&[usize], elem_size)` | `Allocation` | stride 付き空テンソル確保 |
| `create_tensors(descriptors)` | `Vec<Allocation>` | バッチ確保 |
| `empty_tensors(descriptors)` | `Vec<Allocation>` | バッチ空確保 |

**読み出し:**

| メソッド | 戻り値 | 説明 |
|---|---|---|
| `read_one(handle)` | `Bytes` | デバイス → ホスト readback (blocking) |
| `read(handles)` | `Vec<Bytes>` | バッチ readback |
| `read_async(handles)` | `impl Future<Output=Result<Vec<Bytes>>>` | 非同期 readback |
| `read_tensor(descriptors)` | `Vec<Bytes>` | stride 付き readback |
| `read_one_tensor(descriptor)` | `Bytes` | stride 付き単一 readback |
| `read_tensor_async(descriptors)` | `impl Future<...>` | stride 付き非同期 readback |

**実行・同期:**

| メソッド | 説明 |
|---|---|
| `launch(kernel, count, bindings)` | カーネル実行 → `Result<(), LaunchError>` |
| `unsafe launch_unchecked(...)` | 境界チェックなし実行 |
| `sync()` | デバイス同期 (async) |
| `flush()` | コマンドバッファフラッシュ |

**メモリ管理・情報:**

| メソッド | 説明 |
|---|---|
| `memory_usage()` | `MemoryUsage { number_allocs, bytes_in_use, bytes_padding, bytes_reserved }` |
| `memory_cleanup()` | 未使用メモリ解放 |
| `unsafe allocation_mode(mode)` | `Auto` / `Persistent` 切替 |
| `info()` | サーバー情報取得 |
| `properties()` | デバイスプロパティ |
| `profile(func, name)` | プロファイリング → `Result<(O, ProfileDuration)>` |

`Handle` のライフサイクル: `create/empty` で生成 → kernel launch で参照借用 → 最終参照 Drop でデバイスメモリ自動解放。pool allocator が内部で再利用。

#### 5.2.4 カーネル DSL

**マクロ:**

| マクロ | 用途 |
|---|---|
| `#[cube(launch)]` | launchable カーネル定義 |
| `#[cube]` | GPU サブ関数定義 |
| `comptime!(expr)` | JIT 時定数式 |
| `#[derive(CubeType)]` | CubeType 自動導出 |
| `#[derive(CubeLaunch)]` | LaunchArg 自動導出 |

**コンテナ型:**

| 型 | 用途 | GPU メソッド |
|---|---|---|
| `Array<F>` | 1D GPU バッファ | `len()`, index `[]` |
| `Tensor<T>` | N-D GPU テンソル | `stride(dim)`, `shape(dim)`, `coordinate(idx, dim)` |
| `SharedMemory<E>` | ブロック共有メモリ | `new(#[comptime] size)`, `new_lined(size, line_size)`, `len()` |
| `Line<P>` | SIMD ベクトル | `new(val)`, `fill(val)`, `empty(size)`, `line_size()` |
| `Slice<E, IO>` | バッファスライス | `into_lined()`, `try_cast_unchecked::<T>()`, Iterator |
| `Sequence<T>` | コンパイル時シーケンス | `new()`, `push()`, `len()`, `index()` |
| `Atomic<T>` | アトミック変数 | 下記 5.2.8 参照 |

**カーネル定義例:**
```
#[cube(launch)]
fn my_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = input[i] * F::new(2.0_f32);
    }
}
```

**launch 呼び出し（自動生成）:**
```
my_kernel::launch::<F, R>(
    &client, CubeCount::Static(grid_x, 1, 1), CubeDim::new_1d(256),
    ArrayArg::<R>::from_raw_parts::<F>(h_in, n, 1),
    ArrayArg::<R>::from_raw_parts::<F>(&h_out, n, 1),
) -> Result<(), LaunchError>
```

#### 5.2.5 要素型とトレイト

**トレイト階層:**
```
CubeElement (POD: bytemuck::Pod + Send + Sync)
  └── CubePrimitive (CubeType + PartialEq + Send + Sync + Clone + Copy)
        ├── Numeric (Arith + PartialOrd)
        │     ├── Float (Numeric + transcendental + special)
        │     └── Int (Numeric + bitwise + saturating)
        └── Bool
```

**Float 実装型:**

| 型 | サイズ | wgpu 対応 | CUDA 対応 | 備考 |
|---|---|---|---|---|
| `f32` | 32bit | ✅ | ✅ | 主要ターゲット |
| `f64` | 64bit | ❌ Metal FLOAT64 なし | ✅ | wgpu では CPU fallback |
| `f16` | 16bit | ✅ | ✅ | half precision |
| `bf16` | 16bit | ✅ | ✅ | bfloat16 |
| `tf32` | 19bit | — | ✅ | TensorFloat-32 (CUDA only) |
| `flex32` | 32bit | ✅ | ✅ | flexible precision |
| `e5m2` | 8bit | — | ✅ | FP8 E5M2 |
| `e4m3` | 8bit | — | ✅ | FP8 E4M3 |
| `e3m2` | 5bit | — | ✅ | MX formats |
| `e2m3` | 5bit | — | ✅ | MX formats |
| `e2m1x2` | 4bit | — | ✅ | sub-byte |
| `e2m1` | 3bit | — | ✅ | sub-byte |

**Int 実装型:** `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`

#### 5.2.6 Float GPU Intrinsics（全量）

`F: Float` trait が提供する GPU 組み込み関数。全てシェーダコード生成に展開。

**算術:**

| 操作 | trait bound | 説明 |
|---|---|---|
| `+`, `-`, `*`, `/`, `%` | Numeric | 四則演算+剰余 |
| `-x` | Neg | 符号反転 |
| `recip()` | Recip | 逆数 1/x |

**指数・対数:**

| 操作 | trait bound | 説明 |
|---|---|---|
| `exp()` | Exp | e^x |
| `log()` | Log | ln(x) |
| `log1p()` | Log1p | ln(1+x) |
| `powf(y)` | Powf | x^y |
| `powi(n)` | Powi<i32> | x^n |
| `sqrt()` | Sqrt | √x |
| `inverse_sqrt()` | InverseSqrt | 1/√x (rsqrt) |

**三角関数:**

| 操作 | trait bound | 説明 |
|---|---|---|
| `sin()`, `cos()`, `tan()` | Sin, Cos, Tan | 基本三角関数 |
| `sinh()`, `cosh()`, `tanh()` | Sinh, Cosh, Tanh | 双曲線関数 |
| `asin()`, `acos()`, `atan()` | ArcSin, ArcCos, ArcTan | 逆三角関数 |
| `asinh()`, `acosh()`, `atanh()` | ArcSinh, ArcCosh, ArcTanh | 逆双曲線 |
| `atan2(y)` | ArcTan2 | 2引数 atan |

**丸め・判定:**

| 操作 | trait bound | 説明 |
|---|---|---|
| `round()` | Round | 四捨五入 |
| `floor()` | Floor | 切り捨て |
| `ceil()` | Ceil | 切り上げ |
| `trunc()` | Trunc | 0方向切り捨て |
| `is_nan()` | IsNan | NaN 判定 |
| `is_inf()` | IsInf | Inf 判定 |

**特殊関数:**

| 操作 | trait bound | 説明 |
|---|---|---|
| `erf()` | Erf | 誤差関数 |
| `hypot(y)` | Hypot | √(x²+y²) |
| `rhypot(y)` | Rhypot | 1/√(x²+y²) |
| `magnitude()` | Magnitude | ベクトル長 |
| `normalize()` | Normalize | 正規化 |
| `dot(other)` | Dot | 内積 |

**FloatOps:**

| 操作 | 説明 |
|---|---|
| `min(other)` | 最小値 |
| `max(other)` | 最大値 |
| `clamp(min, max)` | クランプ |

**Float 定数:** `DIGITS`, `EPSILON`, `INFINITY`, `NAN`, `NEG_INFINITY`, `MIN_POSITIVE`, `RADIX`, etc.

**コンストラクタ:** `F::new(val: f32)` — コンパイル時リテラル生成

#### 5.2.7 Launch 型

**グリッド・ブロック:**

| 型 | コンストラクタ | 説明 |
|---|---|---|
| `CubeCount` | `Static(x, y, z)` / `Dynamic(Binding)` | グリッドサイズ |
| `CubeCount` | `new_single()` / `new_1d(x)` / `new_2d(x,y)` / `new_3d(x,y,z)` | ショートカット |
| `CubeDim` | `new_1d(x)` / `new_2d(x,y)` / `new_3d(x,y,z)` / `new_single()` | ブロックサイズ |
| `CubeDim` | `new::<R>(client, working_units)` | 自動最適化 |
| `CubeDim` | `num_elems()` | 総スレッド数 |

**カーネル引数:**

| 型 | 構築 | 用途 |
|---|---|---|
| `ArrayArg<'a, R>` | `unsafe { ArrayArg::<R>::from_raw_parts::<E>(handle, n, line_size) }` | 1D GPU バッファ。**R=Runtime, E はターボフィッシュ** |
| `TensorArg<'a, R>` | `unsafe { TensorArg::<R>::from_raw_parts::<E>(handle, strides, shape, line_size) }` | N-D テンソル (stride付き) |
| `ScalarArg` | `ScalarArg::new(value)` | スカラー定数 |

**重要**: `ArrayArg`/`TensorArg` の型パラメータは `R: Runtime`（❌ 要素型 E ではない）。E は `from_raw_parts` のターボフィッシュで指定。

**Alias** (in-place 操作用): `ArrayArg::Alias { input_pos }` / `TensorArg::alias(pos)`

#### 5.2.8 Atomic 操作

```rust
Atomic<Inner: Numeric> {
    load() -> Inner
    store(value)
    swap(value) -> Inner
    fetch_add(value) -> Inner
    fetch_sub(value) -> Inner
    fetch_and(value) -> Inner      // Int only
    fetch_or(value) -> Inner       // Int only
    fetch_xor(value) -> Inner      // Int only
    fetch_min(value) -> Inner
    fetch_max(value) -> Inner
    compare_exchange(expected, new) -> (Inner, bool)
}
```

#### 5.2.9 Plane 操作（Warp-level）

**シャッフル:**

| 操作 | シグネチャ | 説明 |
|---|---|---|
| `plane_broadcast` | `(value: E, index: u32) -> E` | lane から broadcast |
| `plane_shuffle` | `(value: E, src_lane: u32) -> E` | lane 間データ交換 |
| `plane_shuffle_xor` | `(value: E, mask: u32) -> E` | XOR lane交換 |
| `plane_shuffle_up` | `(value: E, delta: u32) -> E` | 上方向シフト |
| `plane_shuffle_down` | `(value: E, delta: u32) -> E` | 下方向シフト |

**リダクション:**

| 操作 | シグネチャ | 説明 |
|---|---|---|
| `plane_sum` | `(value: E) -> E` | warp 内合計 |
| `plane_prod` | `(value: E) -> E` | warp 内積 |
| `plane_max` | `(value: E) -> E` | warp 内最大 |
| `plane_min` | `(value: E) -> E` | warp 内最小 |

**スキャン (prefix):**

| 操作 | シグネチャ | 説明 |
|---|---|---|
| `plane_inclusive_sum` | `(value: E) -> E` | inclusive prefix sum |
| `plane_exclusive_sum` | `(value: E) -> E` | exclusive prefix sum |
| `plane_inclusive_prod` | `(value: E) -> E` | inclusive prefix product |
| `plane_exclusive_prod` | `(value: E) -> E` | exclusive prefix product |

**ブーリアン:**

| 操作 | シグネチャ | 説明 |
|---|---|---|
| `plane_all` | `(value: bool) -> bool` | 全 lane true か |
| `plane_any` | `(value: bool) -> bool` | いずれかの lane true か |
| `plane_ballot` | `(value: bool) -> Line<u32>` | bitmask |
| `plane_elect` | `() -> bool` | 1 lane のみ true |

#### 5.2.10 同期

| 関数 | 説明 |
|---|---|
| `sync_cube()` | ブロック内全スレッド同期 + shared memory 可視性 |
| `sync_plane()` | warp 内同期 |
| `sync_storage()` | ブロック内同期 + global memory 可視性 |

#### 5.2.11 CubeCL 用語マッピング

| CubeCL | CUDA | 説明 |
|---|---|---|
| Unit | Thread | 最小実行単位 |
| Cube | Block | スレッドグループ |
| CubeCount | Grid | ブロック数 |
| Plane | Warp | SIMD グループ (32 threads on NVIDIA) |
| `ABSOLUTE_POS` | `blockIdx*blockDim+threadIdx` | グローバルスレッド ID |

#### 5.2.12 cubecl-std 0.9.0 ユーティリティ

| 機能 | 型/関数 | 説明 |
|---|---|---|
| GPU テンソルハンドル | `TensorHandle<R>` | `empty()`, `zeros()`, `as_ref()`, `as_arg()` |
| contiguous 変換 | `into_contiguous_ref()` | 非連続 → 連続レイアウト変換 |
| コピー | `copy_into()` | テンソル間コピー |
| identity 行列 | `identity::launch()` | 正方行列を identity に初期化 |
| 高速除算 | `FastDivmod<I>` | GPU 上の高速整数除算/modulo |
| GPU Optional | `CubeOption<T>` | `Some(T)` / `None` GPU 安全型 |
| Swizzle | `Swizzle` | 共有メモリバンクコンフリクト回避 |
| レイアウト | `LinearLayout` | `Plain` / `Strided` / `Permuted` |
| View システム | `View<E, C, IO>` | 抽象インデクシング |
| 量子化 | `QuantizedView<Q,S,F,C>` | 量子化テンソルの透過的読み出し |
| 型再解釈 | `ReinterpretSlice<S,T>` | バイトレベル型変換 (wgpu 非対応) |

#### 5.2.13 cubecl-matmul 0.9.0-pre.5（pre-release）

**⚠️ 0.9.0 stable には含まれない。0.9.0-pre.5 のみ。**

| API | シグネチャ | 説明 |
|---|---|---|
| `launch` | `(strategy, client, lhs, rhs, out, dtypes) -> Result<(), MatmulSetupError>` | 高性能 matmul |
| `launch_ref` | `(strategy, client, lhs_ref, rhs_ref, out_ref, dtypes) -> Result<(), MatmulSetupError>` | 参照版 |

**Strategy 一覧:**

| Strategy | 説明 | TensorCore |
|---|---|---|
| `Auto` (default) | Simple → SimpleUnit fallback | 可能なら使用 |
| `Simple` | 単一バッファ、最も互換性高い | CMMA/MMA |
| `DoubleBuffering` | 二重バッファリングでレイテンシ隠蔽 | CMMA/MMA |
| `Specialized` | 非同期読み込み (TMA 対応) | CMMA/MMA |
| `SimpleUnit` | Unit-level (TensorCore なし) | ❌ |
| `Naive` | 素朴実装 | ❌ |

**入力型:** `MatmulInputHandle<R>` — `Normal(TensorHandle)` / `Quantized { data, scale, shape, scheme }`

**MatmulProblem:** `{ m, n, k, lhs_batches, rhs_batches, out_batches, strides, layout }`

**MatmulKind:** `General` / `MatVec` / `VecMat` / `InnerProduct` / `OuterProduct` / `ScalarVec` / `VecScalar`

#### 5.2.14 エラー型

| 型 | バリアント | 説明 |
|---|---|---|
| `LaunchError` | `CompilationError` | カーネルコンパイル失敗 |
| | `OutOfMemory { reason }` | VRAM 不足 |
| | `Unknown { reason }` | 不明なエラー |
| | `IoError` | I/O エラー |
| `IoError` | `BufferTooBig { size }` | バッファサイズ超過 |
| | `UnsupportedStrides` | stride 非対応 |
| | `InvalidHandle` | 無効なハンドル |
| | `Execution(ExecutionError)` | 実行時エラー |
| `MatmulSetupError` | `Unavailable(...)` | matmul 戦略非対応 |
| (pre.5 only) | `InvalidConfig(...)` | 設定不正 |
| | `Launch(LaunchError)` | launch 失敗 |

#### 5.2.15 nabla でのパス解決

`use cubecl_core as cubecl` — `#[cube(launch)]` proc macro が `cubecl::` プレフィックスを期待するため必須。

---

### 5.3 GpuStorage<T> — GPU ストレージ

Section 3.2（`faer::Mat<T>`）と対をなす GPU 側のストレージ仕様。

#### 5.3.1 構造

```rust
pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    handle: cubecl_runtime::server::Handle,       // GPU メモリ (RAII)
    host_cache: Mutex<Option<Vec<T>>>,             // lazy readback cache
}
```

メモリレイアウト: **row-major** flat array (`nrows * ncols` 要素)

Send/Sync: `unsafe impl` — Handle は Arc-backed、Mutex は T: Send+Sync（Scalar が保証）

#### 5.3.2 構築

| 内部メソッド | 入力 | 動作 | cache 状態 |
|---|---|---|---|
| `from_handle(r, c, handle)` | Handle | Handle をそのまま保持 | None (lazy) |
| `from_handle_cached(r, c, handle, data)` | Handle + Vec<T> | Handle + ホストコピー保持 | Some(data) |
| `upload::<R>(r, c, data)` | Vec<T> | `scalar_to_bytes` → `client.create_from_slice` | Some(data) |

Backend trait 経由の公開 API:

| 関数 | 対応 Backend method | 動作 |
|---|---|---|
| `gpu_zeros::<T, R>(r, c)` | `Backend::zeros` | ホストで `vec![T::zero; n]` → upload |
| `gpu_from_fn::<T, R>(r, c, f)` | `Backend::from_fn` | ホストで closure 実行 → Vec → upload |

#### 5.3.3 アクセス

| 関数 | 対応 Backend method | 動作 |
|---|---|---|
| `gpu_get::<T, R>(s, r, c)` | `Backend::get` | `fill_cache` → `cache[r*ncols+c]` |
| `gpu_set::<T, R>(s, r, c, v)` | `Backend::set` | `fill_cache` → cache 更新 → `scalar_to_bytes` → re-upload → handle 更新 |

**readback パス**: `fill_cache::<R>()` → cache が None なら `client.read_one(handle.clone())` → `bytes_to_scalar` → Some(Vec)。以降 cache hit。

**set 後の状態**: cache は最新、handle も re-upload 済み。次の GPU op は新 handle を参照。

#### 5.3.4 download / clone

| 関数 | 対応 Backend method | 動作 |
|---|---|---|
| `download::<R>()` | (内部) | cache hit → clone / miss → readback |
| `gpu_clone::<T, R>(s)` | `Backend::clone_storage` | cache 有 → re-upload from cache / 無 → readback → re-upload。**独立な新 Handle を返す** |

Clone は独立コピー（Handle 共有しない）。元 storage の変更は clone に影響しない。

#### 5.3.5 バイト変換ヘルパー

| 関数 | 方向 | SAFETY 前提 |
|---|---|---|
| `scalar_to_bytes::<T>(&[T]) -> &[u8]` | Scalar → bytes | T は POD (f32/f64/c32/c64) で安定レイアウト |
| `bytes_to_scalar::<T>(&[u8]) -> Vec<T>` | bytes → Scalar | bytes は元が [T] スライスで正しいアラインメントと長さ |

bytemuck 非依存。`core::slice::from_raw_parts` による直接 reinterpret。

#### 5.3.6 メモリフロー図

```
Host                              Device (GPU)
────                              ──────────────
from_fn/zeros
  │ Vec<T>
  │ scalar_to_bytes
  └──── client.create ───────────→ Handle₁
                                      │
           (GPU kernels)              │ add/sub/neg/scale/t/matmul
           handle→handle              │ (zero host transfer)
                                      ▼
                                  Handle₂
                                      │
  get(r,c) / to_cpu()                │
  ┌──── client.read_one ◄────────────┘
  │ bytes_to_scalar
  ▼
host_cache: Some(Vec<T>)
  │
  │ (cache hit: 次回 get は即座)
```

---

### 5.4 GPU カーネル仕様

Section 3.3（Factorizations）と対をなす GPU 演算の詳細仕様。

#### 5.4.1 カーネル一覧

| カーネル | DSL 定義 | 型 bound | 入力 | 出力 | スレッドマッピング |
|---|---|---|---|---|---|
| `elementwise_add_kernel` | `#[cube(launch)]` | `F: Float` | `lhs, rhs: &Array<F>` | `out: &mut Array<F>` | 1 thread = 1 element |
| `elementwise_sub_kernel` | `#[cube(launch)]` | `F: Float` | `lhs, rhs: &Array<F>` | `out: &mut Array<F>` | 1 thread = 1 element |
| `elementwise_neg_kernel` | `#[cube(launch)]` | `F: Float` | `input: &Array<F>` | `out: &mut Array<F>` | 1 thread = 1 element |
| `elementwise_scale_kernel` | `#[cube(launch)]` | `F: Float + CubeElement` | `input: &Array<F>, scalar: F` | `out: &mut Array<F>` | 1 thread = 1 element |
| `transpose_kernel` | `#[cube(launch)]` | `F: Float` | `input: &Array<F>, rows, cols: usize` | `out: &mut Array<F>` | 1 thread = 1 element |
| `matmul_naive_kernel` | `#[cube(launch)]` | `F: Float` | `a, b: &Array<F>, k_dim, n_dim: usize` | `out: &mut Array<F>` | 1 thread = 1 output element |

全カーネル共通: `ABSOLUTE_POS` で境界チェック → 範囲外スレッドは early return。

#### 5.4.2 グリッド/ブロック構成

```rust
fn cube_count(n: usize) -> CubeCount {
    CubeCount::Static(n.div_ceil(256) as u32, 1, 1)
}
```

| パラメータ | 値 | 備考 |
|---|---|---|
| CubeDim (block size) | 256 | 全カーネル共通、1D |
| CubeCount (grid size) | `⌈n/256⌉` | n = 総要素数 (matmul は m*n) |

#### 5.4.3 各カーネルのアルゴリズム

**elementwise_add/sub/neg/scale**: `out[i] = f(input[i])` — O(n) 要素並列

**transpose**: row-major → row-major 転置
```
input[row * cols + col] → out[col * rows + row]
row = i / cols, col = i % cols
```

**matmul_naive**: O(m·n·k) 素朴実装
```
out[row * n + col] = Σ_{k=0..K} a[row * K + k] * b[k * N + col]
row = i / N, col = i % N
```
1 output element = 1 thread。TensorCore/共有メモリ未使用（cubecl-matmul 統合は deferred）。

#### 5.4.4 カーネルヘルパー層

カーネル DSL と dispatch 関数の間にある型付きヘルパー。ArrayArg 構築と launch を隠蔽。

| ヘルパー | シグネチャ | 用途 |
|---|---|---|
| `gpu_binary_kernel` | `<E: Float+CubeElement, R: Runtime>(client, h_a, h_b, n, is_sub) -> Result<Handle, Error>` | add/sub 共用。`is_sub` フラグで分岐 |
| `gpu_neg_kernel` | `<E, R>(client, h_in, n) -> Result<Handle, Error>` | neg |
| `gpu_scale_kernel` | `<E, R>(client, h_in, n, scalar) -> Result<Handle, Error>` | scale |
| `gpu_transpose_kernel` | `<E, R>(client, h_in, rows, cols) -> Result<Handle, Error>` | transpose |
| `gpu_matmul_kernel` | `<E, R>(client, h_a, h_b, m, k, n) -> Result<Handle, Error>` | matmul |

共通パターン:
1. `client.empty(n * size_of::<E>())` で出力バッファ確保
2. `unsafe { ArrayArg::<R>::from_raw_parts::<E>(handle, n, 1) }` で引数構築
3. `kernel::launch::<E, R>(client, cube_count, cube_dim, args...) -> Result<(), LaunchError>`
4. `Ok(h_out)` or `Err(Error::invalid(...))`

---

### 5.5 GPU dispatch 層

Section 3.2-3.3 の solve/factorize と対をなす dispatch 関数群。

#### 5.5.1 TypeId dispatch パターン

全 `gpu_*` 関数が同一パターン:

```
pub(crate) fn gpu_OP<T: Scalar, R: Runtime>(args...) -> GpuStorage<T> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // GPU kernel via gpu_OP_kernel::<f32, R>
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        // GPU kernel via gpu_OP_kernel::<f64, R>
    } else {
        // CPU fallback: download → compute → upload
    }
}
```

**設計理由**: Backend trait は `T: Scalar` bound のみ。trait impl に追加の `where T: GpuOps` を付けると E0276 (impl has stricter requirements than trait) になる。TypeId ランタイム dispatch で回避。

#### 5.5.2 dispatch 関数一覧

| 関数 | 対応 Backend method | 入出力 | GPU カーネル | CPU fallback |
|---|---|---|---|---|
| `gpu_zeros` | `zeros` | `(r, c) -> GpuStorage` | — (ホスト生成 → upload) | — |
| `gpu_from_fn` | `from_fn` | `(r, c, f) -> GpuStorage` | — (ホスト生成 → upload) | — |
| `gpu_get` | `get` | `(s, r, c) -> T` | — (readback) | — |
| `gpu_set` | `set` | `(s, r, c, v)` | — (re-upload) | — |
| `gpu_clone` | `clone_storage` | `(s) -> GpuStorage` | — (readback → re-upload) | — |
| `gpu_add` | `add` | `(a, b) -> GpuStorage` | `gpu_binary_kernel(is_sub=false)` | `zip + add` |
| `gpu_sub` | `sub` | `(a, b) -> GpuStorage` | `gpu_binary_kernel(is_sub=true)` | `zip + sub` |
| `gpu_neg` | `neg` | `(a) -> GpuStorage` | `gpu_neg_kernel` | `map + neg` |
| `gpu_scale` | `scale` | `(a, s) -> GpuStorage` | `gpu_scale_kernel` + `downcast_ref` | `map + mul` |
| `gpu_transpose` | `transpose` | `(a) -> GpuStorage` | `gpu_transpose_kernel` | nested loop |
| `gpu_matmul` | `matmul_into` | `(out, a, b)` | `gpu_matmul_kernel` | triple loop |

**gpu_scale の特殊処理**: `T` → `f32/f64` の変換に `(&s as &dyn Any).downcast_ref::<f32>()` を使用。TypeId で型一致を確認済みのため安全。

**gpu_matmul の特殊処理**: 他の dispatch 関数は `-> GpuStorage<T>` を返すが、matmul は `out: &mut GpuStorage<T>` を受け取り、handle/nrows/ncols を上書き＋cache 無効化する（Backend trait の `matmul_into` シグネチャに合わせるため）。

#### 5.5.3 型別対応表

| 型 | GPU カーネル | CPU fallback | 備考 |
|---|---|---|---|
| `f32` | ✅ 全 6 カーネル | — | 主要ターゲット |
| `f64` | ✅ 全 6 カーネル | — | wgpu/Metal は FLOAT64 非対応（5.7 参照） |
| `c32` | — | ✅ download→compute→upload | CubeCL `Float` trait に Complex 型なし |
| `c64` | — | ✅ download→compute→upload | 同上 |

---

### 5.6 impl_gpu_backend! マクロ

Backend trait の 14 メソッドを GPU dispatch 関数にマッピングする macro_rules。

```rust
macro_rules! impl_gpu_backend {
    ($Backend:ty, $Runtime:path) => {
        impl Backend for $Backend {
            type Storage<T: Scalar> = GpuStorage<T>;
            fn zeros<T: Scalar>(r, c) -> GpuStorage<T>       { gpu_zeros::<T, $Runtime>(r, c) }
            fn from_fn<T: Scalar>(r, c, f) -> GpuStorage<T>  { gpu_from_fn::<T, $Runtime>(r, c, f) }
            fn nrows<T: Scalar>(s) -> usize                   { s.nrows }
            fn ncols<T: Scalar>(s) -> usize                   { s.ncols }
            fn get<T: Scalar>(s, r, c) -> T                   { gpu_get::<T, $Runtime>(s, r, c) }
            fn set<T: Scalar>(s, r, c, v)                     { gpu_set::<T, $Runtime>(s, r, c, v) }
            fn matmul_into<T: Scalar>(out, a, b)              { gpu_matmul::<T, $Runtime>(out, a, b) }
            fn add<T: Scalar>(a, b) -> GpuStorage<T>          { gpu_add::<T, $Runtime>(a, b) }
            fn sub<T: Scalar>(a, b) -> GpuStorage<T>          { gpu_sub::<T, $Runtime>(a, b) }
            fn neg<T: Scalar>(a) -> GpuStorage<T>             { gpu_neg::<T, $Runtime>(a) }
            fn transpose<T: Scalar>(a) -> GpuStorage<T>       { gpu_transpose::<T, $Runtime>(a) }
            fn scale<T: Scalar>(a, s) -> GpuStorage<T>        { gpu_scale::<T, $Runtime>(a, s) }
            fn clone_storage<T: Scalar>(s) -> GpuStorage<T>   { gpu_clone::<T, $Runtime>(s) }
        }
    };
}
```

**マクロ展開**:

| 呼び出し | Backend 型 | Runtime 型 |
|---|---|---|
| `impl_gpu_backend!(Cuda, cubecl_cuda::CudaRuntime)` | `crate::backend::Cuda` | `cubecl_cuda::CudaRuntime` |
| `impl_gpu_backend!(Wgpu, cubecl_wgpu::WgpuRuntime)` | `crate::backend::Wgpu` | `cubecl_wgpu::WgpuRuntime` |

**重要**: メソッドに追加の `where` bound なし（`T: Scalar` のみ）。TypeId dispatch は各 `gpu_*` 関数内部で処理。

---

### 5.7 制限事項と設計判断

#### 既知の制限

| 制限 | 原因 | 対処 |
|---|---|---|
| wgpu/Metal で f64 GPU カーネル不可 | Metal Shading Language に FLOAT64 なし | f64 テストは `Tensor<f64, Cpu>` で実行 |
| c32/c64 GPU カーネルなし | CubeCL `Float` trait に Complex 型がない | CPU fallback (download→compute→upload) |
| linalg/sparse GPU 不可 | faer 依存（CPU SIMD 前提） | `.to_cpu()` → linalg → `.to_wgpu()` パターン |
| matmul が naive O(mnk) | cubecl-matmul 0.9 stable 未リリース | deferred（5.8 参照） |
| 構築が常にホスト経由 | `from_fn`/`zeros` はホストで Vec 生成 → upload | GPU 初期化カーネルは将来検討 |

#### 設計判断

| 判断 | 理由 |
|---|---|
| Handle-based（❌ Vec<T> 直接保持） | chained ops で host↔device 転送を排除。旧実装は全 op で往復していた |
| TypeId dispatch（❌ trait dispatch） | Backend trait は sealed + `T: Scalar` のみ。追加 bound は E0276 |
| 直接 launch 呼び出し（❌ closure ヘルパー） | `#[cube(launch)]` 生成関数は全 ArrayArg を同一ライフタイムに束縛。closure 経由は HRTB 不一致 |
| bytemuck 非依存 | `scalar_to_bytes`/`bytes_to_scalar` を 2 関数で自前実装。POD 保証は Scalar trait bound |
| `Mutex<Option<Vec<T>>>` lazy cache | readback は高コスト。get/set 時のみ実行し cache hit で再利用 |
| gpu_scale の Any downcast | TypeId で型一致確認後の `downcast_ref` は安全。Scalar→Float 変換の汎用パスがないため |

#### GPU crate 選定

| 候補 | 採否 | 理由 |
|---|---|---|
| cubecl | ✅ | proc macro で 1 カーネル = 全バックエンド。Burn 実績 |
| cudarc | 不要 | cubecl-cuda 経由 |
| rust-gpu | ✗ | nightly 必須、実験段階 |

---

### 5.8 GPU 対応ギャップ分析

#### 5.8.1 nabla CPU → GPU 対応状況

**Backend trait 30 メソッド (Wave 5 で 14→30 に拡張):**

| Backend method | CPU (faer) | GPU (CubeCL) | 備考 |
|---|---|---|---|
| `zeros` | ✅ `Mat::zeros` | ✅ ホスト生成→upload | GPU zero-fill kernel で改善可 |
| `from_fn` | ✅ `Mat::from_fn` | ✅ ホスト生成→upload | closure はホスト実行 |
| `nrows` / `ncols` | ✅ `.nrows()` | ✅ `s.nrows` | メタデータのみ |
| `get` | ✅ `*mat.get(r,c)` | ✅ lazy readback cache | |
| `set` | ✅ `*mat.get_mut(r,c)` | ✅ cache更新+re-upload | |
| `add` / `sub` / `neg` | ✅ `&a+&b` etc. | ✅ GPU kernel (f32/f64) + CPU fallback (c32/c64) | |
| `scale` | ✅ `mat * Scale(s)` | ✅ GPU kernel (f32/f64) + CPU fallback (c32/c64) | |
| `transpose` | ✅ `.transpose()` | ✅ GPU kernel (f32/f64) + CPU fallback (c32/c64) | |
| `matmul_into` | ✅ faer matmul | ✅ GPU naive O(mnk) | cubecl-matmul で高速化可 |
| `clone_storage` | ✅ `.clone()` | ✅ readback→re-upload | |
| `exp`/`ln`/`log1p` | ✅ MathOps trait | ✅ CubeCL Float intrinsics | Wave 5 追加 |
| `sin`/`cos`/`tanh` | ✅ MathOps trait | ✅ CubeCL Float intrinsics | Wave 5 追加 |
| `sqrt`/`abs`/`recip` | ✅ MathOps trait | ✅ CubeCL Float intrinsics | Wave 5 追加 |
| `erf`/`ceil`/`floor`/`round` | ✅ MathOps trait | ✅ CubeCL Float intrinsics | Wave 5 追加 |
| `powf` | ✅ MathOps trait | ✅ CubeCL Float::powf | Wave 5 追加 |
| `mul_elem` / `div_elem` | ✅ MathOps trait | ✅ GPU kernel (f32/f64) | Wave 5 追加 |
| `sum_all` | ✅ ReductionOps | ✅ download + CPU fold | Wave 6 追加 |
| `max_all` / `min_all` | ✅ ReductionOps | ✅ download + CPU fold | Wave 6 追加 |
| `argmax_all` / `argmin_all` | ✅ ReductionOps (reduction_gt) | ✅ download + index tracking | Wave 6 追加 |

**linalg.rs — 密行列 factorization (CPU only):**

| 操作 | CPU | GPU | CubeCL 提供 | 対応方針 |
|---|---|---|---|---|
| LU / Full-piv LU | ✅ faer | ❌ | ❌ solver なし | `.to_cpu()` 経由 |
| QR / Col-piv QR | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |
| Cholesky (LLT) | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |
| LDLt | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |
| SVD / Thin SVD | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |
| Eigenvalue | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |
| solve / inv | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |
| solve_lstsq | ✅ faer | ❌ | ❌ | `.to_cpu()` 経由 |

**sparse.rs (CPU only):**

| 操作 | CPU | GPU | 対応方針 |
|---|---|---|---|
| CSC 構築 | ✅ faer | ❌ | `.to_cpu()` 経由 |
| Sparse solve/lstsq | ✅ faer | ❌ | `.to_cpu()` 経由 |
| SpMV (sparse × dense) | ✅ faer | ❌ | 将来 GPU SpMV kernel |

**構造行列型 (CPU only):**

| 型 | CPU | GPU | 対応方針 |
|---|---|---|---|
| Diagonal | ✅ faer | ❌ | GPU diagonal kernel 可能 |
| Symmetric | ✅ faer | ❌ | `.to_cpu()` 経由 (eigenvalue 必要) |
| Triangular | ✅ faer | ❌ | `.to_cpu()` 経由 (solve 必要) |

#### 5.8.2 CubeCL が提供するが nabla が未使用の機能

| CubeCL 機能 | nabla 活用先 | 優先度 | 備考 |
|---|---|---|---|
| ~~**Float intrinsics** (exp/log/sin/sqrt/erf etc.)~~ | ~~`bcast!` GPU 版~~ | ~~**High**~~ | ✅ Wave 5 で 13 unary + powf を GPU 実装済み |
| ~~**Plane reduction** (plane_sum/max/min)~~ | ~~GPU sum/max/min~~ | ~~**High**~~ | ✅ Wave 6 で download + CPU fallback として実装済み (plane kernel は Wave 6.1 deferred) |
| **Atomic** (fetch_add etc.) | GPU reduction | **High** | global reduction の実装に必要 |
| **SharedMemory** | tiled matmul | **High** | cubecl-matmul 待ち or 自前 |
| **TensorArg** (stride 付き) | non-contiguous GPU tensor | Medium | 現在は ArrayArg (flat) のみ |
| **cubecl-std TensorHandle** | GPU メモリ管理改善 | Medium | zeros(), empty() 直接 |
| **cubecl-std identity** | `Tensor::identity()` GPU 版 | Medium | 現在はホスト生成→upload |
| **cubecl-std contiguous** | レイアウト変換 | Medium | transpose 最適化 |
| **bf16/f16** | 低精度推論 | Medium | CubeCL 対応済み、nabla Scalar 拡張必要 |
| **CMMA** (TensorCore) | 高速 matmul | Medium | cubecl-matmul 経由 |
| **Quantization** | INT8/FP8 推論 | Low | cubecl-std QuantizedView |
| **read_tensor_async** | 非同期 readback | Low | パイプライン最適化 |
| **FastDivmod** | index 計算最適化 | Low | 大規模テンソルのみ効果 |
| **Line vectorization** | SIMD 幅最適化 | Low | line_size > 1 |
| **profile()** | パフォーマンス計測 | Low | デバッグ用 |

#### 5.8.3 GPU 全対応に向けた実装計画

**Wave 5: GPU elementwise 拡張 (Float intrinsics) ✅**

| タスク | 詳細 | ファイル | 状態 |
|---|---|---|---|
| Backend trait 16 math methods | MathOps private trait + Cpu impl + Hip delegate | backend.rs, tensor.rs | ✅ |
| 13 unary GPU kernels | exp/ln/log1p/sin/cos/tanh/sqrt/abs/recip/erf/ceil/floor/round | gpu.rs | ✅ |
| powf GPU kernel | Float::powf with scalar arg | gpu.rs | ✅ |
| mul_elem / div_elem GPU kernels | Hadamard product + element-wise division | gpu.rs | ✅ |
| GPU integration tests | 17 new tests (30 total), GPU↔CPU comparison | tests/gpu.rs | ✅ |

**Wave 6: GPU reduction ✅**

| タスク | 詳細 | ファイル | 状態 |
|---|---|---|---|
| gpu_sum_all | download → CPU fold (scalar result is always transferred) | gpu.rs | ✅ |
| gpu_max_all / gpu_min_all | download → CPU fold | gpu.rs | ✅ |
| gpu_argmax_all / gpu_argmin_all | download → index tracking reduction | gpu.rs | ✅ |
| Backend trait 拡張 | `argmax_all`, `argmin_all` メソッド追加 | backend.rs | ✅ |
| Tensor API 拡張 | `.argmax()` / `.argmin()` メソッド | tensor.rs | ✅ |
| CPU integration tests | 9 new tests (sum_all/max_all/min_all/argmax/argmin) | tests/basic.rs | ✅ |

Note: GPU side uses CPU fallback (download + reduce). Proper GPU kernels (plane_sum + SharedMemory) are deferred to Wave 6.1 once CubeCL subgroup portability across CUDA/wgpu/HIP is confirmed.

**Wave 7: GPU matmul 高速化 ✅**

| タスク | 詳細 | ファイル | 状態 |
|---|---|---|---|
| SharedMemory tiled matmul | 自前実装 TILE=16, 1D CubeCount, `sync_cube()` | gpu.rs | ✅ |
| cubecl-matmul 統合 | Strategy::Auto, TensorCore | gpu.rs, Cargo.toml | ⏳ cubecl-matmul stable 待ち |

**Wave 8: GPU 構築最適化 ✅**

| タスク | 詳細 | ファイル | 状態 |
|---|---|---|---|
| GPU zeros kernel | zero-fill kernel (ホスト経由排除) | gpu.rs | ✅ |
| GPU identity kernel | fill_identity_kernel 自前実装 | gpu.rs | ✅ |
| GPU fill kernel | 定数値充填カーネル | gpu.rs | ✅ |
| Backend::fill / ::identity デフォルト実装 | CPU fallback default + GPU override | backend.rs | ✅ |
| Tensor::fill / ::identity API | Wave 8 public API | tensor.rs | ✅ |

**Deferred (ブロッカーあり):**

| 項目 | ブロッカー | 優先度 |
|---|---|---|
| GPU linalg (LU/QR/SVD) | CubeCL に solver なし | Low — faer 委譲が妥当 |
| GPU sparse | CubeCL に sparse なし | Low |
| bf16/f16 Scalar 型追加 | nabla Scalar trait 拡張必要 | Medium |
| Multi-GPU (NCCL 相当) | CubeCL 未成熟 | Future |
| GPU bcast! macro 統合 | proc macro 拡張 (Backend-aware codegen) | Medium |

---

## 5.9 einsum! 次世代仕様 — 縮約最適化・N-D 拡張・エラー人間化

Julia の `@tullio` / `@einsum` が持つ「直感的な添字記法で書くだけで高速コードが出る」体験を
Rust proc macro で完全に再現する。数理記述の簡潔さと計算性能を両立させるための 3 軸拡張。

### 5.9.1 縮約を BLAS/GEMM に落とすパス（I3a）

**現状**: `einsum!(c[i,j] = a[i,k] * b[k,j])` は `Tensor::from_fn` + 3 重ループに展開される。
Julia では `@tullio` がこれを自動的に BLAS 呼び出しに置換し、ハードウェアのピーク FLOPS を引き出す。
nabla でも同等のコンパイル時最適化を行う。

**目標**: proc macro の codegen 段階で AST をパターン認識し、既存の `Tensor::matmul`（faer GEMM）
や `mul_elem` 等の最適化済み関数呼び出しに**置換**する。ユーザは添字記法のまま、裏側で
faer の SIMD/AMX 最適化や GPU tiled matmul が走る。

#### パターン分類テーブル

| パターン | 数学的意味 | 検出条件 | 生成コード (CPU) | 生成コード (GPU) |
|---|---|---|---|---|
| **GEMM** `c[i,j] = a[i,k] * b[k,j]` | 行列積 $C = AB$ | 2 テンソル積、縮約 1 本、free 2 本 | `Tensor::matmul(&a, &b)` → faer GEMM | `gpu_matmul` (tiled) |
| **GEMV** `y[i] = a[i,k] * x[k]` | 行列ベクトル積 $y = Ax$ | 行列×ベクトル、縮約 1 本、free 1 本 | `Tensor::matmul(&a, &x_col)` | `gpu_matmul` (M×1) |
| **Outer** `c[i,j] = a[i] * b[j]` | 外積 $C = ab^T$ | 縮約なし、2 ベクトル積 | `from_fn` (ループ) | `from_fn` (ループ) |
| **Trace** `s = a[i,i]` | トレース $\mathrm{tr}(A)$ | 同一テンソル内で同一 idx が 2 位置 | 対角ループ | 対角ループ |
| **Hadamard** `c[i,j] = a[i,j] * b[i,j]` | アダマール積 $C = A \circ B$ | 縮約なし、全 idx が free | `mul_elem` | `gpu_mul_elem` |
| **Fallback** | 一般 N 階縮約 | 上記いずれにも該当しない | 現行ループ codegen | 現行ループ codegen |

#### コンパイル時判定アルゴリズム

```
fn classify(input: &EinsumInput) -> ContractionKind {
    let free = &input.output_indices;          // LHS に現れるインデックス
    let contraction = rhs_only_indices(input);  // RHS のみに現れるインデックス
    let terms = &input.rhs_terms;

    // GEMM: 2 terms × 2 indices each, 1 contraction index
    if terms.len() == 2
        && terms[0].indices.len() == 2
        && terms[1].indices.len() == 2
        && free.len() == 2
        && contraction.len() == 1
    {
        let k = &contraction[0];
        // a[i,k] * b[k,j] — k が a の dim1, b の dim0
        if terms[0].indices[1] == *k && terms[1].indices[0] == *k {
            return ContractionKind::Gemm;           // 標準配置
        }
        // a[k,i] * b[j,k] 等 — transpose フラグを付けて GEMM
        return ContractionKind::GemmTransposed { transpose_a: ..., transpose_b: ... };
    }

    // GEMV: matrix[i,k] * vector[k] → vector[i]
    if terms.len() == 2
        && free.len() == 1
        && contraction.len() == 1
        && ((terms[0].indices.len() == 2 && terms[1].indices.len() == 1)
         || (terms[0].indices.len() == 1 && terms[1].indices.len() == 2))
    {
        return ContractionKind::Gemv;
    }

    // Hadamard: all RHS indices are free, no contraction
    if contraction.is_empty() && terms.len() == 2 {
        return ContractionKind::Hadamard;
    }

    ContractionKind::Fallback
}
```

#### 生成コード例 (GEMM)

```rust
// einsum!(c[i,j] = a[i,k] * b[k,j])
// ↓ コンパイル時展開
{
    let __a = &a;
    let __b = &b;
    Tensor::matmul(__a, __b)       // → faer GEMM (CPU) or gpu_matmul (GPU)
}
```

**transpose 配置の場合** (例: `c[i,j] = a[k,i] * b[k,j]` → $C = A^T B$):
```rust
{
    let __a = &a;
    let __b = &b;
    Tensor::matmul(&__a.t(), __b)  // transpose は zero-copy (faer MatRef)
}
```

### 5.9.2 N 次元 einsum! 拡張（I3b）

Julia の `@tullio` は任意階数のテンソルを自然に扱える。nabla の einsum! も
Einstein 縮約規約に忠実な一般 N 階テンソル記法をサポートする。

**動機**: 物理学・数値計算で頻出する 3 階以上の縮約を直感的に記述する。

```rust
// 例: 3階テンソルと行列の縮約  T_{ijk} M_{kl} → R_{ijl}
let r = einsum!(r[i,j,l] = t[i,j,k] * m[k,l]);

// 例: テンソル縮約 (2本同時縮約)  A_{ijkl} B_{klmn} → C_{ijmn}
let c = einsum!(c[i,j,m,n] = a[i,j,k,l] * b[k,l,m,n]);

// 例: バッチ行列積  C_{bij} = A_{bik} B_{bkj}  (b はバッチ次元)
let c = einsum!(c[b,i,j] = a[b,i,k] * q[b,k,j]);
```

#### インデックス 3 分類

現行の 2 分類（free / contraction）を 3 分類に一般化:

| 分類 | 定義 | 例 (`r[b,i,j] = a[b,i,k] * m[b,k,j]`) |
|---|---|---|
| **Batch** | LHS と RHS の全テンソルに共通出現し、縮約されない | `b` |
| **Free** | LHS に出現するが、一部の RHS テンソルにのみ出現 | `i` (a のみ), `j` (m のみ) |
| **Contraction** | LHS に出現しない（RHS のみ） | `k` |

```
fn classify_indices(input: &EinsumInput) -> (Vec<Ident>, Vec<Ident>, Vec<Ident>) {
    let lhs_set = set(&input.output_indices);
    let rhs_all = all_rhs_indices(input);
    let contraction: Vec<_> = rhs_all.iter()
        .filter(|idx| !lhs_set.contains(idx))
        .collect();

    let lhs_indices = &input.output_indices;
    let mut batch = vec![];
    let mut free = vec![];
    for idx in lhs_indices {
        if input.rhs_terms.iter().all(|t| t.indices.contains(idx)) {
            batch.push(idx.clone());
        } else {
            free.push(idx.clone());
        }
    }
    (batch, free, contraction)
}
```

#### パーサ拡張

| 変更点 | 現行 | Wave 9 |
|---|---|---|
| テンソルあたりの最大インデックス数 | 2 | **N** (制限なし) |
| 出力の最大 free インデックス数 | 2 | **N** (制限なし) |
| `tensor_element_access` | `.get(i, j)` (2D) | N-D: batch 次元はループ変数、inner 2D は `.get(i, j)` |
| `dim_expr_for_index` | `.nrows()` / `.ncols()` | 一般化: `.dim(axis)` |

#### N-D 縮約のコード生成戦略

batch 次元をループで剥がし、innermost の 2D 縮約が GEMM パターンに合致する場合は
§5.9.1 の最適化を適用する。

```rust
// einsum!(r[b, i, j] = a[b, i, k] * m[b, k, j])
// ↓ コンパイル時展開（batch=[b], free=[i,j], contraction=[k]）
{
    let __a = &a;
    let __m = &m;
    // batch ループ → inner GEMM dispatch
    for b in 0..__a.dim(0) {
        let a_slice = __a.slice(b);  // [M, K] view
        let m_slice = __m.slice(b);  // [K, N] view
        let c_slice = Tensor::matmul(&a_slice, &m_slice);  // GEMM!
        out.set_slice(b, c_slice);
    }
}
```

**段階的実装**:
- **Phase 1**: パーサの N-D 拡張 + batch/free/contraction 3 分類（一般ループ生成）
- **Phase 2**: inner 2D が GEMM パターンに合致する場合、§5.9.1 の GEMM 最適化を適用
- **Phase 3**: `Tensor` に stride-based N-D view を追加し、`slice` をゼロコピー化

#### 利用イメージ

```rust
// 応力テンソルの縮約: σ_{ij} = C_{ijkl} ε_{kl}
let σ = einsum!(σ[i,j] = c_tensor[i,j,k,l] * ε[k,l]);

// バッチ行列積
let y = einsum!(y[b,i] = a[b,i,k] * x[b,k]);

// 3階テンソルのトレース的縮約
let v = einsum!(v[i] = t[i,j,j]);
```

### 5.9.3 エラーメッセージの人間化（I3c）

#### 現状の問題

```rust
einsum!(c[i,j] = a[i,k] * b[x,j]);
//                              ^
// 現行: "einsum! index `x` not found in any RHS tensor" (Span::call_site)
// → エラー位置がマクロ呼び出し行全体を指し、どの文字が問題か不明
```

#### 改善方針: `syn::Error::new_spanned` の全面採用

`Span::call_site()` を廃止し、**ユーザが書いたトークンの Span** を使って精密なエラーを出す。

| エラー種別 | メッセージ例 | Span 対象 |
|---|---|---|
| 未知のインデックス | `` einsum!: index `x` is not defined on the LHS and does not appear in any other RHS tensor `` | `x` の Ident span |
| 次元不整合 | `` einsum!: index `k` binds to dim 1 of `a` (ncols) and dim 0 of `b` (nrows); these must be equal at runtime `` | `k` の Ident span (警告/note) |
| 空の RHS | `` einsum!: at least one tensor term is required on the right-hand side `` | `=` トークンの span |
| テンソルインデックス数超過 | `` einsum!: tensor `a` has 5 indices, but the current Tensor type supports at most N `` | `a[...]` の bracket span |
| 縮約インデックス未使用 | `` einsum!: index `k` appears only once on the RHS; contraction indices must appear in at least 2 terms (did you mean to include it on the LHS?) `` | `k` の Ident span |
| LHS インデックス重複 | `` einsum!: index `i` appears twice on the LHS `` | 2 番目の `i` の span |
| RHS テンソル内インデックス重複 (trace以外) | `` einsum!: index `i` appears twice in tensor `a`; this is only valid for trace (scalar output) `` | 該当 Ident span |

#### 実装パターン

```rust
// Before (現行)
Err(Error::new(Span::call_site(), format!("einsum! index `{idx}` not found ...")))

// After (Wave 9)
Err(Error::new_spanned(idx, format!("einsum!: index `{idx}` is not defined on the LHS ...")))
//                     ^^^ ユーザが書いた `idx` トークンの位置に赤線
```

#### AST に Span を保持

`IndexedTensor` 構造体に bracket の Span を保持し、テンソル全体や bracket 範囲を指すエラーにも対応:

```rust
struct IndexedTensor {
    name: Ident,
    indices: Vec<Ident>,
    bracket_span: Option<Span>,  // `[i, k]` 全体の Span (new_spanned 用)
}
```

#### コンパイラ出力例 (改善後)

```
error: einsum!: index `x` is not defined on the LHS and does not appear
       in any other RHS tensor
  --> src/main.rs:42:38
   |
42 |     let c = einsum!(c[i,j] = a[i,k] * b[x,j]);
   |                                          ^ unknown index
```

```
error: einsum!: index `k` appears only once on the RHS; contraction
       indices must appear in at least 2 terms
  --> src/main.rs:10:30
   |
10 |     let y = einsum!(y[i] = a[i,k]);
   |                                ^ used in only one term
   |
   = help: did you mean to include `k` on the LHS? e.g., `y[i,k] = a[i,k]`
```

---

## 6. 実装ロードマップ

### Wave 1 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| G2: Factorization wrappers | src/linalg.rs | ✅ |
| G1: Structural matrix types | src/linalg.rs | ✅ |
| A2+A5: between!() + linspace() + frange!() | src/lib.rs (util module) | ✅ |

### Wave 2 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| B1+B3: bcast! + zip_map! broadcasting | src/lib.rs (util module) | ✅ |
| C2: Slicing API (slice/slice_rows/slice_cols/set) | src/tensor.rs | ✅ |
| A3: Complex helpers (c32/c64) | src/lib.rs (util module) | ✅ |

### Wave 3 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| H1: Sparse matrix support | src/sparse.rs | ✅ |
| I3: einsum!{} macro | macros/src/einsum.rs | ✅ |

### Wave 4 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| C4: Const generic small matrices | src/tensor.rs | ✅ |
| D2: Type hierarchy traits | src/tensor.rs | ✅ |
| I4: cubecl GPU kernel実装 (Handle-based) | src/gpu.rs | ✅ |

### Wave 5 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| Backend trait 16 math methods (MathOps + Cpu impl) | src/backend.rs, src/tensor.rs | ✅ |
| 16 CubeCL GPU kernels (exp/ln/sin/cos/tanh/sqrt/abs/recip/erf/ceil/floor/round/log1p/powf/mul_elem/div_elem) | src/gpu.rs | ✅ |
| GPU integration tests (17 new, 30 total) | tests/gpu.rs | ✅ |

### Wave 6 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| GPU sum_all / max_all / min_all (download + CPU fold) | src/gpu.rs, src/backend.rs | ✅ |
| GPU argmax_all / argmin_all (reduction_gt + CPU fold) | src/gpu.rs, src/backend.rs | ✅ |
| Tensor::argmax / Tensor::argmin | src/tensor.rs | ✅ |
| Reduction tests (9 new) | tests/basic.rs | ✅ |

### Wave 7 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| SharedMemory tiled matmul kernel (TILE=16) | src/gpu.rs | ✅ |
| gpu_matmul → tiled kernel dispatch | src/gpu.rs | ✅ |
| GPU matmul tests (3 new: square/non-square/large) | tests/gpu.rs | ✅ |

### Wave 8 ✅

| タスク | ファイル | 状態 |
|---|---|---|
| fill_zeros_kernel / fill_scalar_kernel / fill_identity_kernel | src/gpu.rs | ✅ |
| gpu_zeros → GPU kernel dispatch | src/gpu.rs | ✅ |
| gpu_fill / gpu_identity dispatch functions | src/gpu.rs | ✅ |
| Backend::fill / Backend::identity (default + GPU override) | src/backend.rs, src/gpu.rs | ✅ |
| Tensor::fill / Tensor::identity (GPU-accelerated API) | src/tensor.rs | ✅ |
| GPU construction tests (zeros/fill/identity) + CPU tests | tests/gpu.rs, tests/basic.rs | ✅ |

ファイル数: src/ 6 (FLAT構造維持)

### Wave 9 ✅ — einsum! 縮約最適化・N-D 拡張・エラー人間化

Julia `@tullio` 相当の「添字記法で書くだけで高速コードが出る」体験の完成。詳細仕様は §5.9 参照。

| タスク | 詳細 | ファイル | 状態 |
|---|---|---|---|
| I3c: `new_spanned` エラー人間化 | 全 `Span::call_site()` → `new_spanned(token)` 置換、bracket_span 保持、lint 追加 (孤立縮約idx警告等) | macros/src/einsum.rs | ✅ |
| I3a: ContractionKind 分類器 | GEMM/GEMV/Hadamard/Trace/Outer/Fallback パターン認識 | macros/src/einsum.rs | ✅ |
| I3a: GEMM codegen | `classify()` → `Tensor::matmul` / `.t()` + matmul 生成、transpose 配置対応 | macros/src/einsum.rs | ✅ |
| I3a: GEMV codegen | matrix × vector → `Tensor::matmul` (Mx1 column vector) | macros/src/einsum.rs | ✅ |
| I3a: Hadamard codegen | 全 idx free + 2 terms → `mul_elem` dispatch | macros/src/einsum.rs | ✅ |
| I3b Phase 1: N-D パーサ拡張 | テンソルあたり N indices、出力 N free indices、batch/free/contraction 3 分類 | macros/src/einsum.rs | ✅ |
| I3b Phase 2: N-D 縮約 codegen | batch ループ + inner GEMM dispatch (§5.9.2 参照) | macros/src/einsum.rs | 🔧 (N-D Tensor 型待ち) |
| I3b Phase 3: stride-based N-D view | stride-based view + `slice` ゼロコピー | src/tensor.rs | 🔧 (N-D Tensor 型待ち) |
| einsum! GEMM テスト | naive vs GEMM 結果一致、transpose 配置、大行列ベンチ | tests/basic.rs | ✅ |
| einsum! N-D テスト | 3階テンソル縮約 `r[i,j,l]=t[i,j,k]*m[k,l]`、バッチ行列積 | tests/basic.rs | 🔧 (N-D Tensor 型待ち) |
| einsum! エラーテスト | `trybuild` or compile_fail テストで spanned エラー位置を検証 | tests/einsum_errors/ | ✅ |

---

## 7. 再現度サマリ

### Easy（直接等価 or 実装済み）
| 機能 | Rust / nabla | 状態 |
|---|---|---|
| Parametric types | `struct Point<T>` | ✅ |
| タプル分割代入 | `let (a, b) = f();` | ✅ |
| スライス | faer `MatRef` default | ✅ |
| bounds check除去 | iterator auto-elim | ✅ |
| In-place BLAS `mul!` | `Tensor::matmul_into` | ✅ |
| Unicode識別子 | native | ✅ |
| block view | faer default | ✅ |
| Factorization objects | faer `.partial_piv_lu()` 等 | ✅ linalg.rs |
| イテレータ内包 | `.map().filter().collect()` | ✅ |

### Medium（macro/traitで達成可能）
| 機能 | nabla 戦略 | 状態 |
|---|---|---|
| 行列リテラル `[1 2; 3 4]` | `mat![]` proc macro | ✅ |
| Adjoint / Transpose | `.adjoint()` / `.t()` | ✅ |
| `@tullio` 標準einsum | `einsum!{}` proc macro | ✅ |
| einsum → GEMM 最適化 | コンパイル時パターン認識 → matmul dispatch | ✅ Wave 9 |
| N-D einsum 拡張 | N 階テンソル縮約 + batch 次元自動認識 | ✅ Wave 9 Phase 1 (パーサ) |
| Broadcasting `f.(x)` | `bcast!{}` macro | ✅ bcast! macro |
| Float range | `linspace()` / `frange!()` | ✅ linspace/frange |
| 構造行列型 | nabla wrapper types | ✅ Diagonal/Symmetric/Triangular |
| `\` solve | `.solve()` method | ✅ .solve() method |
| Macro system | `syn` + `quote` (基盤あり) | ✅/🔧 |

### Hard（本格的DSL作業）
| 機能 | 困難の理由 |
|---|---|
| broadcast-all | 式AST全書き換え proc macro |
| 自動ループ融合 | 構文変換等価なし |
| LoopVectorization | 単一マクロSIMDスケジューラなし |
| Tullio stencil/convolution | オフセットインデクス proc macro |
| Full multiple dispatch (open) | Orphan rule |
| Symbolics.jl CAS | symbolica は統合度低い |
| ModelingToolkit DSL | 記号前処理なし |

### Impossible（言語根本差異）
暗黙乗算 `2x`, 連鎖比較中置 `0<x<1`, 後置 `A'`, `//` 有理数, Unicode中置演算子,
`\` solve演算子, Open dispatch, Runtime `eval`, `end` indexing
