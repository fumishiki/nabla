/// Generates `Module` trait boilerplate (training, parameters, forward_var delegation).
///
/// # With parameters
/// ```ignore
/// impl_module_params!(required1, required2; optional1, optional2);
/// ```
/// Required fields = `Tensor<T,B>`, optional fields = `Option<Tensor<T,B>>`.
///
/// # Without parameters (training/set_training only)
/// ```ignore
/// impl_module_params!();
/// ```
macro_rules! impl_module_params {
    () => {
        fn training(&self) -> bool { self.training }
        fn set_training(&mut self, training: bool) { self.training = training; }
    };

    ($($req:ident),+ $(;)?) => {
        impl_module_params!(@inner [$($req),+] []);
    };

    ($($req:ident),+ ; $($opt:ident),+) => {
        impl_module_params!(@inner [$($req),+] [$($opt),+]);
    };

    (@inner [$($req:ident),+] []) => {
        fn training(&self) -> bool { self.training }
        fn set_training(&mut self, training: bool) { self.training = training; }

        fn forward_var(&self, x: &Variable<T, B>, tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
            Ok(self.forward_var_tracked(x, tape)?.output)
        }

        fn parameters(&self) -> Vec<&Tensor<T, B>> {
            vec![$(&self.$req),+]
        }

        fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)> {
            vec![$((stringify!($req), &self.$req)),+]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
            vec![$(&mut self.$req),+]
        }

        fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)> {
            vec![$((stringify!($req), &mut self.$req)),+]
        }
    };

    (@inner [$($req:ident),+] [$($opt:ident),+]) => {
        fn training(&self) -> bool { self.training }
        fn set_training(&mut self, training: bool) { self.training = training; }

        fn forward_var(&self, x: &Variable<T, B>, tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
            Ok(self.forward_var_tracked(x, tape)?.output)
        }

        fn parameters(&self) -> Vec<&Tensor<T, B>> {
            let mut params = vec![$(&self.$req),+];
            $(if let Some(ref v) = self.$opt { params.push(v); })*
            params
        }

        fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)> {
            let mut params = vec![$((stringify!($req), &self.$req)),+];
            $(if let Some(ref v) = self.$opt { params.push((stringify!($opt), v)); })*
            params
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
            let mut params: Vec<&mut Tensor<T, B>> = vec![$(&mut self.$req),+];
            $(if let Some(ref mut v) = self.$opt { params.push(v); })*
            params
        }

        fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)> {
            let mut params: Vec<(&str, &mut Tensor<T, B>)> = vec![$((stringify!($req), &mut self.$req)),+];
            $(if let Some(ref mut v) = self.$opt { params.push((stringify!($opt), v)); })*
            params
        }
    };
}

/// Generates a complete `Module` impl for a struct with a `TensorLike`-generic forward body.
///
/// # Syntax
/// ```ignore
/// impl_layer! {
///     StructName { required1, required2; optional1, optional2 }
///     forward(x) { body }
/// }
/// ```
/// Required fields = `Tensor<T,B>`, optional = `Option<Tensor<T,B>>`.
/// The body uses `TensorLike` methods (`tl_add`, `tl_matmul`, etc.) so it
/// works for both `Tensor` (inference) and `Variable` (training).
macro_rules! impl_layer {
    // Required + optional params
    ($name:ident { $($req:ident),+ ; $($opt:ident),+ } forward($x:ident) $body:block) => {
        impl_layer!(@impl $name [$($req),+] [$($opt),+] $x $body);
    };
    // Required only
    ($name:ident { $($req:ident),+ } forward($x:ident) $body:block) => {
        impl_layer!(@impl $name [$($req),+] [] $x $body);
    };
    // Optional only (empty required before semicolon)
    ($name:ident { ; $($opt:ident),+ } forward($x:ident) $body:block) => {
        impl_layer!(@impl $name [] [$($opt),+] $x $body);
    };
    // No params
    ($name:ident { } forward($x:ident) $body:block) => {
        impl_layer!(@impl $name [] [] $x $body);
    };

    (@impl $name:ident [$($req:ident),*] [$($opt:ident),*] $x:ident $body:block) => {
        const _: () = {
            use crate::autograd::{Tape, TensorLike, TensorLikeMatmulBias, Variable};
            use nabla_core::backend::Backend;
            use nabla_core::error::Result;
            use nabla_core::scalar::Scalar;
            use nabla_core::tensor::Tensor;
            use std::rc::Rc;
            use super::{ForwardResult, Module};

            #[allow(clippy::needless_pass_by_value)]
            fn __compute<T: Scalar, B: Backend, X: TensorLike<T, B> + TensorLikeMatmulBias<T, B>>(
                $x: &X, $($req: &X,)* $($opt: Option<&X>,)*
            ) -> X
            $body

            impl<T: Scalar, B: Backend> Module<T, B> for $name<T, B> {
                fn forward(&self, $x: &Tensor<T, B>) -> Tensor<T, B> {
                    __compute($x, $(&self.$req,)* $(self.$opt.as_ref(),)*)
                }

                fn forward_var_tracked(
                    &self,
                    $x: &Variable<T, B>,
                    tape: &Rc<Tape<T, B>>,
                ) -> Result<ForwardResult<T, B>> {
                    $(let $req = tape.variable(self.$req.clone())?;)*
                    $(let $opt = self.$opt.as_ref()
                        .map(|p| tape.variable(p.clone())).transpose()?;)*
                    let output = __compute($x, $(&$req,)* $($opt.as_ref(),)*);
                    let mut param_vars = Vec::new();
                    $(param_vars.push($req);)*
                    $(if let Some(v) = $opt { param_vars.push(v); })*
                    Ok(ForwardResult { output, param_vars })
                }

                fn training(&self) -> bool { self.training }
                fn set_training(&mut self, training: bool) { self.training = training; }

                fn forward_var(&self, x: &Variable<T, B>, tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
                    Ok(self.forward_var_tracked(x, tape)?.output)
                }

                fn parameters(&self) -> Vec<&Tensor<T, B>> {
                    let mut params: Vec<&Tensor<T, B>> = Vec::new();
                    $(params.push(&self.$req);)*
                    $(if let Some(ref v) = self.$opt { params.push(v); })*
                    params
                }

                fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)> {
                    let mut params: Vec<(&str, &Tensor<T, B>)> = Vec::new();
                    $(params.push((stringify!($req), &self.$req));)*
                    $(if let Some(ref v) = self.$opt { params.push((stringify!($opt), v)); })*
                    params
                }

                fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
                    let mut params: Vec<&mut Tensor<T, B>> = Vec::new();
                    $(params.push(&mut self.$req);)*
                    $(if let Some(ref mut v) = self.$opt { params.push(v); })*
                    params
                }

                fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)> {
                    let mut params: Vec<(&str, &mut Tensor<T, B>)> = Vec::new();
                    $(params.push((stringify!($req), &mut self.$req));)*
                    $(if let Some(ref mut v) = self.$opt { params.push((stringify!($opt), v)); })*
                    params
                }
            }
        };
    };
}

mod core;
mod layers;

pub use core::{ForwardResult, Linear, Module, Sequential, StateError, load_tensors, save_tensors};
pub use layers::{
    Activation, ActivationKind, DropoutLayer, EmbeddingLayer, LayerNormModule, embedding,
    kaiming_normal, kv_cache_append, linear, rotary_embedding, xavier_uniform,
};

#[cfg(feature = "cpu")]
pub use layers::backslash;
