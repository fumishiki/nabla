// tensor/mod.rs — Tensor<T, B> core struct, accessors, fuse internals, and axis compat.
//
// Design notes:
// - Operator overloads are defined on references to avoid moves (Julia semantics).
// - Shape mismatches in Add/Sub/Mul panic with a descriptive message (Option A).
// - Adjoint uses `T::IS_REAL` const to choose between transpose and conj+transpose.
// - `adjoint` delegates element-wise conjugation via `scalar::math_utils::conj`.

/// Tensor constructors: zeros, ones, identity, rand, fill, linspace, etc.
pub mod constructors;
/// Debug/Display formatting for tensors.
pub mod display;
/// Array/Matrix traits and DynTensor enum for runtime dispatch.
pub mod dyntensor;
/// Row and column iterators over tensors.
pub mod iter;
/// N-dimensional tensor stored as a flat Vec<T>.
pub mod ndtensor;
/// Neural network operations: activations, normalization, loss, convolution, pooling, attention.
pub mod nn;
/// Tensor arithmetic: element access, element-wise ops, broadcast, transpose, matmul, overloads.
pub mod ops;
/// Reduction operations: sum, mean, norm, min/max, argmin/argmax, variance, cumsum, cumprod.
pub mod reductions;
/// Shape manipulation: reshape, concat, stack, gather, scatter, sort, topk, CPU-gated impls.
pub mod shape;
/// Stack-allocated fixed-size matrix.
pub mod static_matrix;
/// Zero-copy read-only view into a tensor subregion.
pub mod view;

pub use dyntensor::{Array, DynTensor, Matrix};
pub use iter::{ColIter, RowIter};
pub use ndtensor::NdTensor;
pub use static_matrix::StaticMatrix;
pub use view::TensorView;

use core::marker::PhantomData;
use core::ops::{Bound, RangeBounds};

use crate::backend::{Backend, DefaultBackend};
use crate::scalar::Scalar;

/// A 2-D dense matrix backed by a pluggable [`Backend`].
///
/// The default backend is [`crate::backend::Cpu`], which uses nabla's CPU kernels.
/// The optional `Axes` parameter carries named axis types at zero cost.
pub struct Tensor<T: Scalar, B: Backend = DefaultBackend, Axes = ()> {
    pub(super) storage: B::Storage<T>,
    _axes: PhantomData<fn() -> Axes>,
}

impl<T: Scalar, B: Backend, Axes> Clone for Tensor<T, B, Axes> {
    fn clone(&self) -> Self {
        Self {
            storage: B::clone_storage(&self.storage),
            _axes: PhantomData,
        }
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Returns an iterator over the rows of this tensor.
    pub fn iter(&self) -> RowIter<'_, T, B> {
        self.eachrow()
    }
}

/// `for row in &tensor` iterates rows (same as `tensor.eachrow()`).
impl<'a, T: Scalar, B: Backend> IntoIterator for &'a Tensor<T, B> {
    type Item = Tensor<T, B>;
    type IntoIter = RowIter<'a, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.eachrow()
    }
}

// Named axes: zero-cost axis reinterpretation + accessors for any Axes.
impl<T: Scalar, B: Backend, Axes> Tensor<T, B, Axes> {
    #[inline]
    fn cast_axes<NewAxes>(storage: B::Storage<T>) -> Tensor<T, B, NewAxes> {
        Tensor {
            storage,
            _axes: PhantomData,
        }
    }

    /// Reinterpret the phantom axis type (zero-cost, compile-time only).
    #[inline]
    #[must_use]
    pub fn with_axes<NewAxes>(self) -> Tensor<T, B, NewAxes> {
        Self::cast_axes(self.storage)
    }

    /// Erase axis types back to untyped `()`.
    #[inline]
    #[must_use]
    pub fn erase_axes(self) -> Tensor<T, B> {
        Self::cast_axes(self.storage)
    }

    /// Number of rows.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        B::nrows(&self.storage)
    }

    /// Number of columns.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        B::ncols(&self.storage)
    }

    /// Shape as `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Read element at `(row, col)`.
    #[must_use]
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        B::get(&self.storage, row, col)
    }

    /// Block until all pending backend operations (e.g. GPU kernels) have completed.
    /// On GPU backends this flushes the command stream; on CPU this is a no-op.
    /// Use this for accurate throughput benchmarks instead of `get()` to avoid
    /// including the device-to-host transfer cost in measurements.
    pub fn sync(&self) {
        B::sync(&self.storage);
    }

    /// Dimension along a given axis: 0 -> nrows, 1 -> ncols.
    #[must_use]
    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        match axis {
            0 => self.nrows(),
            1 => self.ncols(),
            _ => panic!("Tensor is 2-D: axis must be 0 or 1, got {axis}"),
        }
    }

    /// Opaque pointer to internal storage (for fuse! macro GPU codegen).
    #[doc(hidden)]
    #[inline]
    pub fn __storage_ptr(&self) -> *const u8 {
        (&raw const self.storage).cast::<u8>()
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Fused element-wise kernel launch (for fuse! macro codegen).
    ///
    /// GPU backends JIT-compile the expression; CPU backends use the closure.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn __fuse_elementwise(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reg_estimate: usize,
    ) -> Self {
        Self::from_storage(B::fuse_launch(
            inputs,
            nrows,
            ncols,
            cpu_fn,
            gpu_expr,
            kernel_hash,
            n_inputs,
            reg_estimate,
        ))
    }

    /// Execute multiple fused element-wise operations as a **single** GPU
    /// kernel launch (mega-kernel fusion).
    ///
    /// All operations must have the same tensor dimensions and element type.
    /// On GPU backends this emits one mega-kernel that processes every op
    /// sequentially within a single launch, eliminating all inter-op kernel
    /// launch overhead.  On the CPU backend each operation is executed
    /// independently via `from_fn`.
    ///
    /// # Arguments
    ///
    /// * `ops` -- per-op descriptors: `(input_ptrs, gpu_expr, n_inputs, uses_prev)`.
    ///   `uses_prev` signals DAG register pass-through: the op reads the preceding
    ///   op's output register instead of a global-memory buffer for its first input.
    /// * `nrows`, `ncols` -- shared dimensions for all ops.
    /// * `cpu_fns` -- per-op CPU closures (used only on CPU backend).
    ///   For `uses_prev` ops the macro inlines the previous body directly into the
    ///   closure, so the CPU implementation requires no special handling here.
    /// * `kernel_hash` -- cache key for the compiled mega-kernel.
    #[doc(hidden)]
    #[inline]
    pub fn __mega_fuse_elementwise<'a>(
        ops: &[(Vec<*const u8>, String, usize, bool)],
        nrows: usize,
        ncols: usize,
        cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T + 'a>>,
        kernel_hash: &str,
    ) -> Vec<Self> {
        B::mega_fuse_launch::<T>(ops, nrows, ncols, cpu_fns, kernel_hash)
            .into_iter()
            .map(Self::from_storage)
            .collect()
    }

    /// Fused GEMM + epilogue activation dispatched through the backend.
    ///
    /// On CUDA with `f32`, this calls cublasLt's epilogue fusion to perform the
    /// activation inside the GEMM kernel (one launch instead of two).
    /// All other backends and scalar types fall back to two passes: GEMM then
    /// an element-wise activation kernel.
    ///
    /// `epilogue_id`:
    /// - `0` -> ReLU
    /// - `1` -> GELU (tanh approximation)
    ///
    /// Emitted by the `fuse!` macro for `(A * B).relu()` and `(A * B).gelu()`
    /// patterns when the `L3` GEMM+activation fusion tier is active.
    #[doc(hidden)]
    #[inline]
    pub fn __matmul_epilogue(a: &Self, b: &Self, epilogue_id: u8) -> Self {
        Self::from_storage(B::matmul_epilogue::<T>(&a.storage, &b.storage, epilogue_id))
    }

    /// Fused map-reduce: apply a pointwise expression then reduce along `axis`.
    ///
    /// `reduce_op`:
    /// - `0` -> sum
    /// - `3` -> mean
    ///
    /// `axis`: `0` -> reduce rows (output `1xncols`); `1` -> reduce columns (output `nrowsx1`).
    ///
    /// On CUDA this compiles to a single JIT kernel; other backends use a two-pass fallback.
    /// Emitted by the `fuse!` macro for patterns like `x.exp().sum_axis(1)`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn __fuse_reduce(
        inputs: &[*const u8],
        nrows: usize,
        ncols: usize,
        cpu_fn: impl FnMut(usize, usize) -> T,
        gpu_expr: &str,
        kernel_hash: &str,
        n_inputs: usize,
        reduce_op: u8,
        axis: u8,
    ) -> Self {
        Self::from_storage(B::fuse_reduce_launch::<T>(
            inputs,
            nrows,
            ncols,
            cpu_fn,
            gpu_expr,
            kernel_hash,
            n_inputs,
            reduce_op,
            axis,
        ))
    }

    /// Create a tensor from a preallocated storage.
    #[inline]
    /// Construct from raw backend storage.
    pub fn from_storage(storage: B::Storage<T>) -> Self {
        Self {
            storage,
            _axes: PhantomData,
        }
    }

    /// Consume this tensor and return its backing storage.
    pub fn into_storage(self) -> B::Storage<T> {
        self.storage
    }

    /// Borrow the backing storage.
    pub fn storage(&self) -> &B::Storage<T> {
        &self.storage
    }
}

/// Sealed module for `MatmulCompat`.
mod matmul_compat_seal {
    pub trait Sealed {}
    impl Sealed for () {}
    impl<A, K> Sealed for (A, K) {}
}

/// Compile-time axis compatibility for matrix multiplication.
///
/// `(A, K) * (K, B) -> (A, B)` when axes are named.
/// Untyped `()` is compatible with everything.
pub trait MatmulCompat<Rhs, Out>: matmul_compat_seal::Sealed {}

impl MatmulCompat<(), ()> for () {}
impl<A, K> MatmulCompat<(), ()> for (A, K) {}
impl<K, B> MatmulCompat<(K, B), ()> for () {}
impl<A, K, B> MatmulCompat<(K, B), (A, B)> for (A, K) {}

/// Resolve a `RangeBounds` into `(start, end)` given a dimension length.
pub(super) fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    (start, end)
}

/// Generate element-wise math forwarding methods on Tensor.
macro_rules! impl_tensor_math {
    ($($(#[$meta:meta])* $name:ident);+ $(;)?) => {
        impl<T: Scalar, B: Backend> Tensor<T, B> {
            $(
                $(#[$meta])*
                #[must_use]
                #[inline]
                pub fn $name(&self) -> Self {
                    Self::from_storage(B::$name(&self.storage))
                }
            )+
        }
    };
}

impl_tensor_math! {
    /// Element-wise `e^x`.
    exp;
    /// Element-wise natural logarithm `ln(x)`.
    ln;
    /// Element-wise `ln(1 + x)`.
    log1p;
    /// Element-wise `sin(x)`.
    sin;
    /// Element-wise `cos(x)`.
    cos;
    /// Element-wise `tanh(x)`.
    tanh;
    /// Element-wise `sqrt(x)`.
    sqrt;
    /// Element-wise absolute value.
    abs;
    /// Element-wise reciprocal `1/x`.
    recip;
    /// Element-wise error function.
    erf;
    /// Element-wise `ceil(x)`.
    ceil;
    /// Element-wise `floor(x)`.
    floor;
    /// Element-wise `round(x)`.
    round;
    /// Element-wise `asin(x)`.
    asin;
    /// Element-wise `acos(x)`.
    acos;
    /// Element-wise `atan(x)`.
    atan;
    /// Element-wise `sinh(x)`.
    sinh;
    /// Element-wise `cosh(x)`.
    cosh;
    /// Element-wise `asinh(x)`.
    asinh;
    /// Element-wise `acosh(x)`.
    acosh;
    /// Element-wise `atanh(x)`.
    atanh;
    /// Element-wise `log2(x)`.
    log2;
    /// Element-wise `log10(x)`.
    log10;
}

/// Helper: `T::one() + T::one()` (avoids repeating the pattern).
#[inline]
pub(super) fn two<T: Scalar>() -> T {
    T::one() + T::one()
}
