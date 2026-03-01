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
        tl_matmul => (|a, b| a * b, matmul),
        tl_emul   => (|a: &Tensor<T, B>, b| a.emul(b), emul),
    }
    unary {
        tl_t       => (t, transpose),
        tl_relu    => (relu, relu),
        tl_gelu    => (gelu, gelu),
        tl_sigmoid => (sigmoid, sigmoid),
        tl_silu    => (silu, silu),
        tl_tanh    => (tanh, tanh),
    }
    unary_param(f64) {
        tl_leaky_relu(alpha) => (|s: &Tensor<T, B>, a| s.leaky_relu(T::from_f64(a)), |s: &Variable<T, B>, a| s.leaky_relu(a)),
        tl_elu(alpha)        => (|s: &Tensor<T, B>, a| s.elu(T::from_f64(a)), |s: &Variable<T, B>, a| s.elu(a)),
    }
}
