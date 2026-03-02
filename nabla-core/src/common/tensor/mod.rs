/// Tensor constructors and iterators.
pub mod constructors;
/// Convolution operations.
pub mod nn_conv;
/// Neural-network operations (activation, norm, pooling).
pub mod nn_ops;
pub use nn_conv::*;
#[allow(unused_imports)]
pub use nn_ops::*;
/// Element-wise and binary tensor operations.
pub mod ops;
/// Reduction operations (sum, mean, max, etc.).
pub mod reductions;
/// Shape manipulation (reshape, transpose, broadcast).
pub mod shape;
/// Tensor variants: `NdTensor`, `StaticMatrix`, `DynTensor`, and related traits.
pub mod variants;
/// Low-precision (fp8/fp4) quantize/dequantize helpers.
pub mod lowp;

pub use constructors::{ColIter, RowIter, TensorView};
pub use variants::{Array, NdTensor, StaticMatrix};
#[cfg(feature = "cpu")]
pub use variants::{DynTensor, Matrix};

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Bound, RangeBounds};

use crate::backend::{Backend, DefaultBackend};
use crate::scalar::Scalar;

/// Dense 2-D matrix with pluggable backend and optional phantom axis types.
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

impl<'a, T: Scalar, B: Backend> IntoIterator for &'a Tensor<T, B> {
    type Item = Tensor<T, B>;
    type IntoIter = RowIter<'a, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.eachrow()
    }
}

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

    /// Fused `(self @ b) + bias` — single cublasLt dispatch on CUDA f32, fallback elsewhere.
    #[inline]
    pub fn matmul_bias(a: &Self, b: &Self, bias: &Self) -> Self {
        Self::from_storage(B::matmul_bias::<T>(&a.storage, &b.storage, &bias.storage))
    }


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

    /// Construct from raw backend storage.
    #[inline]
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

mod matmul_compat_seal {
    pub trait Sealed {}
    impl Sealed for () {}
    impl<A, K> Sealed for (A, K) {}
}

/// Sealed trait constraining valid axis combinations for matrix multiplication.
pub trait MatmulCompat<Rhs, Out>: matmul_compat_seal::Sealed {}

impl MatmulCompat<(), ()> for () {}
impl<A, K> MatmulCompat<(), ()> for (A, K) {}
impl<K, B> MatmulCompat<(K, B), ()> for () {}
impl<A, K, B> MatmulCompat<(K, B), (A, B)> for (A, K) {}

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
    /// Element-wise `tan(x)`.
    tan;
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

#[inline]
pub(super) fn two<T: Scalar>() -> T {
    T::one() + T::one()
}

fn display_indices(len: usize) -> Vec<Option<usize>> {
    if len > 6 {
        let mut v: Vec<Option<usize>> = (0..3).map(Some).collect();
        v.push(None);
        v.extend((len - 3..len).map(Some));
        v
    } else {
        (0..len).map(Some).collect()
    }
}

pub(crate) fn fmt_matrix(
    rows: usize,
    cols: usize,
    mut elem: impl FnMut(usize, usize, &mut fmt::Formatter<'_>) -> fmt::Result,
    f: &mut fmt::Formatter<'_>,
    prefix: Option<&str>,
) -> fmt::Result {
    // Build the sequence of row / column indices to display.
    // When a dimension exceeds 6, show indices 0,1,2 and (n-3),(n-2),(n-1).
    let row_indices = display_indices(rows);
    let col_indices = display_indices(cols);

    if let Some(p) = prefix {
        write!(f, "{p}")?;
    }
    write!(f, "[")?;
    let mut first_row = true;
    for row_slot in row_indices {
        if !first_row {
            if prefix.is_some() {
                write!(f, ", ")?;
            } else {
                writeln!(f)?;
                write!(f, " ")?;
            }
        }
        first_row = false;
        match row_slot {
            None => {
                // Row ellipsis: emit a placeholder row
                write!(f, "[...]")?;
            }
            Some(r) => {
                write!(f, "[")?;
                let mut first_col = true;
                for col_slot in col_indices.iter().copied() {
                    if !first_col {
                        write!(f, ", ")?;
                    }
                    first_col = false;
                    match col_slot {
                        None => write!(f, "...")?,
                        Some(c) => elem(r, c, f)?,
                    }
                }
                write!(f, "]")?;
            }
        }
    }
    write!(f, "]")
}

impl<T: Scalar + fmt::Display, B: Backend> fmt::Display for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        let prec = f.precision();
        fmt_matrix(
            rows,
            cols,
            |r, c, fmt| match prec {
                Some(p) => write!(fmt, "{:.prec$}", self.get(r, c).to_f64(), prec = p),
                None => write!(fmt, "{}", self.get(r, c)),
            },
            f,
            None,
        )
    }
}

impl<T: Scalar + fmt::Debug, B: Backend> fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        let prefix = format!("Tensor({rows}x{cols})");
        fmt_matrix(
            rows,
            cols,
            |r, c, f| write!(f, "{:?}", self.get(r, c)),
            f,
            Some(&prefix),
        )
    }
}

/// Read-only matrix interface for shape and element access.
pub trait MatrixLike<T: Scalar> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Shape as `(rows, cols)`.
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    /// Element access (read-only).
    fn get(&self, row: usize, col: usize) -> T;
    /// Total number of elements.
    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }
    /// Whether the matrix is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Scalar, B: Backend, Axes> MatrixLike<T> for Tensor<T, B, Axes> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        self.get(row, col)
    }
}

impl<T: Scalar, const R: usize, const C: usize> MatrixLike<T> for StaticMatrix<T, R, C> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        self.get(row, col)
    }
}
