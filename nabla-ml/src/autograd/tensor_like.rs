use std::ops::Neg;

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::core::Variable;

/// Generates the `TensorLike` trait + impls for `Tensor` and `Variable`.
macro_rules! tensor_like_ops {
    (
        binary { $($bin_name:ident => ($tensor_op:expr, $var_method:ident)),* $(,)? }
        unary { $($un_name:ident => ($tensor_method:ident, $var_un_method:ident)),* $(,)? }
        unary_param($param_ty:ty) {
            $($up_name:ident ($param:ident) => ($tensor_param_op:expr, $var_param_op:expr)),* $(,)?
        }
    ) => {
        /// Trait abstracting over `Tensor` and `Variable` for generic compute functions.
        ///
        /// Allows writing math logic once that works in both inference (`Tensor`) and
        /// training (`Variable`) paths, eliminating the "two-world" duplication in
        /// `Module::forward` vs `Module::forward_var`.
        pub trait TensorLike<T: Scalar, B: Backend>: Sized {
            $(
                /// Binary operation (see concrete impls for semantics).
                fn $bin_name(&self, rhs: &Self) -> Self;
            )*
            $(
                /// Unary operation (see concrete impls for semantics).
                fn $un_name(&self) -> Self;
            )*
            $(
                /// Parameterised unary operation (see concrete impls for semantics).
                fn $up_name(&self, $param: $param_ty) -> Self;
            )*
        }

        impl<T: Scalar, B: Backend> TensorLike<T, B> for Tensor<T, B> {
            $(
                #[inline]
                fn $bin_name(&self, rhs: &Self) -> Self { ($tensor_op)(self, rhs) }
            )*
            $(
                #[inline]
                fn $un_name(&self) -> Self { self.$tensor_method() }
            )*
            $(
                #[inline]
                fn $up_name(&self, $param: $param_ty) -> Self { ($tensor_param_op)(self, $param) }
            )*
        }

        impl<T: Scalar, B: Backend> TensorLike<T, B> for Variable<T, B> {
            $(
                #[inline]
                fn $bin_name(&self, rhs: &Self) -> Self { self.$var_method(rhs) }
            )*
            $(
                #[inline]
                fn $un_name(&self) -> Self { self.$var_un_method() }
            )*
            $(
                #[inline]
                fn $up_name(&self, $param: $param_ty) -> Self { ($var_param_op)(self, $param) }
            )*
        }
    };
}

tensor_like_ops! {
    binary {
        tl_add    => (|a, b| a + b, add_var),
        tl_sub    => (|a, b| a - b, sub_var),
        tl_matmul => (|a, b| a * b, matmul),
        tl_matmul_tn => (|a: &Tensor<T, B>, b| a.matmul_tn(b), matmul_tn),
        tl_matmul_nt => (|a: &Tensor<T, B>, b| a.matmul_nt(b), matmul_nt),
        tl_emul   => (|a: &Tensor<T, B>, b| a.emul(b), emul),
        tl_ediv   => (|a: &Tensor<T, B>, b| a.ediv(b), ediv),
        tl_broadcast_mul_cols => (|a: &Tensor<T, B>, b| a.broadcast_mul_cols(b), broadcast_mul_cols),
        tl_broadcast_mul_rows => (|a: &Tensor<T, B>, b| a.broadcast_mul_rows(b), broadcast_mul_rows),
        tl_broadcast_add_rows => (|a: &Tensor<T, B>, b| a.broadcast_add_rows(b), broadcast_add_rows),
    }
    unary {
        tl_t       => (t, transpose),
        tl_relu    => (relu, relu),
        tl_gelu    => (gelu, gelu),
        tl_sigmoid => (sigmoid, sigmoid),
        tl_silu    => (silu, silu),
        tl_tanh    => (tanh, tanh),
        tl_neg     => (neg, neg_var),
        tl_exp     => (exp, exp),
        tl_ln      => (ln, ln),
        tl_abs     => (abs, abs),
        tl_sqrt    => (sqrt, sqrt),
    }
    unary_param(f64) {
        tl_leaky_relu(alpha) => (|s: &Tensor<T, B>, a| s.leaky_relu(T::from_f64(a)), |s: &Variable<T, B>, a| s.leaky_relu(a)),
        tl_elu(alpha)        => (|s: &Tensor<T, B>, a| s.elu(T::from_f64(a)), |s: &Variable<T, B>, a| s.elu(a)),
    }
}

/// Extension trait for ops with non-standard signatures (usize params, Range, etc.).
pub trait TensorLikeExt<T: Scalar, B: Backend>: TensorLike<T, B> {
    /// Multiply every element by `s`.
    fn tl_scale(&self, s: T) -> Self;
    /// Raise every element to the scalar power `p`.
    fn tl_powf(&self, p: T) -> Self;
    /// Reduce along `axis` with a sum.
    fn tl_sum_axis(&self, axis: usize) -> Self;
    /// Reduce along `axis` with a mean.
    fn tl_mean_axis(&self, axis: usize) -> Self;
    /// Compute the mean over all elements.
    fn tl_mean(&self) -> Self;
    /// Apply softmax along `axis`.
    fn tl_softmax(&self, axis: usize) -> Self;
    /// Apply log-softmax along `axis`.
    fn tl_log_softmax(&self, axis: usize) -> Self;
    /// Reshape into `(rows, cols)`.
    fn tl_reshape(&self, rows: usize, cols: usize) -> Self;
    /// Broadcast to `(rows, cols)`.
    fn tl_expand(&self, rows: usize, cols: usize) -> Self;
    /// Slice a row range.
    fn tl_slice_rows(&self, range: std::ops::Range<usize>) -> Self;
    /// Gather along `axis` using `index`.
    fn tl_gather(&self, axis: usize, index: &Tensor<T, B>) -> Self;
    /// Select entries along `axis` using `indices`.
    fn tl_index_select(&self, axis: usize, indices: &Tensor<T, B>) -> Self;
    /// Batched matrix multiply with explicit shapes.
    fn tl_bmm(&self, rhs: &Self, batch: usize, m: usize, k: usize, n: usize) -> Self;
    /// Clamp values into `[min, max]`.
    fn tl_clamp(&self, min: T, max: T) -> Self;
    /// Vertically concatenate `slices`.
    fn tl_vcat(slices: &[&Self]) -> Self;
    /// Apply a constant linear projection.
    fn tl_linear_const(&self, w: &Tensor<T, B>) -> Self;
    /// Add a constant tensor.
    fn tl_add_const(&self, c: &Tensor<T, B>) -> Self;
    /// Add a constant row bias with broadcast semantics.
    fn tl_broadcast_add_rows_const(&self, bias: &Tensor<T, B>) -> Self;
    /// Constant-tensor variant of [`Self::tl_index_select`].
    fn tl_index_select_const(&self, axis: usize, indices: &Tensor<T, B>) -> Self;
    /// Batched matrix multiply with a constant left operand.
    fn tl_bmm_const_left(
        &self,
        left: &Tensor<T, B>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self;
    /// Return the number of rows.
    fn tl_nrows(&self) -> usize;
    /// Return the number of columns.
    fn tl_ncols(&self) -> usize;
    /// Return `(rows, cols)`.
    fn tl_shape(&self) -> (usize, usize);
}

impl<T: Scalar, B: Backend> TensorLikeExt<T, B> for Tensor<T, B> {
    #[inline]
    fn tl_scale(&self, s: T) -> Self {
        self * s
    }
    #[inline]
    fn tl_powf(&self, p: T) -> Self {
        self.powf(p)
    }
    #[inline]
    fn tl_sum_axis(&self, axis: usize) -> Self {
        self.sum_axis(axis)
    }
    #[inline]
    fn tl_mean_axis(&self, axis: usize) -> Self {
        self.mean_axis(axis)
    }
    #[inline]
    fn tl_mean(&self) -> Self {
        Tensor::fill(1, 1, self.mean())
    }
    #[inline]
    fn tl_softmax(&self, axis: usize) -> Self {
        self.softmax(axis)
    }
    #[inline]
    fn tl_log_softmax(&self, axis: usize) -> Self {
        self.log_softmax(axis)
    }
    #[inline]
    fn tl_reshape(&self, rows: usize, cols: usize) -> Self {
        self.reshape(rows, cols)
    }
    #[inline]
    fn tl_expand(&self, rows: usize, cols: usize) -> Self {
        self.expand(rows, cols)
    }
    #[inline]
    fn tl_slice_rows(&self, range: std::ops::Range<usize>) -> Self {
        self.slice_rows(range)
    }
    #[inline]
    fn tl_gather(&self, axis: usize, index: &Self) -> Self {
        self.gather(axis, index)
    }
    #[inline]
    fn tl_index_select(&self, axis: usize, indices: &Self) -> Self {
        self.index_select(axis, indices)
    }
    #[inline]
    fn tl_bmm(&self, rhs: &Self, batch: usize, m: usize, k: usize, n: usize) -> Self {
        self.bmm(rhs, batch, m, k, n)
    }
    #[inline]
    fn tl_clamp(&self, min: T, max: T) -> Self {
        self.clamp(min, max)
    }
    #[inline]
    fn tl_vcat(slices: &[&Self]) -> Self {
        Tensor::vcat(slices)
    }
    #[inline]
    fn tl_linear_const(&self, w: &Tensor<T, B>) -> Self {
        self * w
    }
    #[inline]
    fn tl_add_const(&self, c: &Tensor<T, B>) -> Self {
        self + c
    }
    #[inline]
    fn tl_broadcast_add_rows_const(&self, bias: &Tensor<T, B>) -> Self {
        self.broadcast_add_rows(bias)
    }
    #[inline]
    fn tl_index_select_const(&self, axis: usize, indices: &Tensor<T, B>) -> Self {
        self.index_select(axis, indices)
    }
    #[inline]
    fn tl_bmm_const_left(
        &self,
        left: &Tensor<T, B>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self {
        left.bmm(self, batch, m, k, n)
    }
    #[inline]
    fn tl_nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn tl_ncols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn tl_shape(&self) -> (usize, usize) {
        self.shape()
    }
}

impl<T: Scalar, B: Backend> TensorLikeExt<T, B> for Variable<T, B> {
    #[inline]
    fn tl_scale(&self, s: T) -> Self {
        self.scale(s)
    }
    #[inline]
    fn tl_powf(&self, p: T) -> Self {
        self.powf(p)
    }
    #[inline]
    fn tl_sum_axis(&self, axis: usize) -> Self {
        self.sum_axis(axis)
    }
    #[inline]
    fn tl_mean_axis(&self, axis: usize) -> Self {
        self.mean_axis(axis)
    }
    #[inline]
    fn tl_mean(&self) -> Self {
        self.mean_var()
    }
    #[inline]
    fn tl_softmax(&self, axis: usize) -> Self {
        self.softmax(axis)
    }
    #[inline]
    fn tl_log_softmax(&self, axis: usize) -> Self {
        self.log_softmax(axis)
    }
    #[inline]
    fn tl_reshape(&self, rows: usize, cols: usize) -> Self {
        self.reshape(rows, cols)
    }
    #[inline]
    fn tl_expand(&self, rows: usize, cols: usize) -> Self {
        self.expand_var(rows, cols)
    }
    #[inline]
    fn tl_slice_rows(&self, range: std::ops::Range<usize>) -> Self {
        self.slice_rows(range)
    }
    #[inline]
    fn tl_gather(&self, axis: usize, index: &Tensor<T, B>) -> Self {
        self.gather_var(axis, index)
    }
    #[inline]
    fn tl_index_select(&self, axis: usize, indices: &Tensor<T, B>) -> Self {
        self.index_select_var(axis, indices)
    }
    #[inline]
    fn tl_bmm(&self, rhs: &Self, batch: usize, m: usize, k: usize, n: usize) -> Self {
        self.bmm_var(rhs, batch, m, k, n)
    }
    #[inline]
    fn tl_clamp(&self, min: T, max: T) -> Self {
        self.clamp(min, max)
    }
    #[inline]
    fn tl_vcat(slices: &[&Self]) -> Self {
        Variable::vcat_var(slices)
    }
    #[inline]
    fn tl_linear_const(&self, w: &Tensor<T, B>) -> Self {
        self.linear_const(w)
    }
    #[inline]
    fn tl_add_const(&self, c: &Tensor<T, B>) -> Self {
        self.add_const(c)
    }
    #[inline]
    fn tl_broadcast_add_rows_const(&self, bias: &Tensor<T, B>) -> Self {
        self.broadcast_add_rows_const(bias)
    }
    #[inline]
    fn tl_index_select_const(&self, axis: usize, indices: &Tensor<T, B>) -> Self {
        self.index_select_const(axis, indices)
    }
    #[inline]
    fn tl_bmm_const_left(
        &self,
        left: &Tensor<T, B>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self {
        self.bmm_const_left(left, batch, m, k, n)
    }
    #[inline]
    fn tl_nrows(&self) -> usize {
        self.data().nrows()
    }
    #[inline]
    fn tl_ncols(&self) -> usize {
        self.data().ncols()
    }
    #[inline]
    fn tl_shape(&self) -> (usize, usize) {
        self.data().shape()
    }
}

/// Extension for fused matmul+bias — cannot be expressed in the binary/unary macro.
pub trait TensorLikeMatmulBias<T: Scalar, B: Backend>: Sized {
    /// Compute `self @ weight + bias` in a single fused dispatch where possible.
    fn tl_matmul_bias(&self, weight: &Self, bias: &Self) -> Self;
}

impl<T: Scalar, B: Backend> TensorLikeMatmulBias<T, B> for Tensor<T, B> {
    #[inline]
    fn tl_matmul_bias(&self, weight: &Self, bias: &Self) -> Self {
        Tensor::matmul_bias(self, weight, bias)
    }
}

impl<T: Scalar, B: Backend> TensorLikeMatmulBias<T, B> for Variable<T, B> {
    #[inline]
    fn tl_matmul_bias(&self, weight: &Self, bias: &Self) -> Self {
        // Two autograd nodes — fused kernel fires only in the Tensor (inference) path.
        self.matmul(weight).add_var(bias)
    }
}
