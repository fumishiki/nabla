//! nabla-macros — Layer 1: Notation proc macros for concise DSL syntax.
//!
//! All `#[proc_macro]` and `#[proc_macro_attribute]` entry points live here.
//! Implementation logic is delegated to submodules.

use proc_macro::TokenStream;

mod fusion;
mod macros;

fn expand<F>(input: TokenStream, f: F) -> TokenStream
where
    F: FnOnce(proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream>,
{
    match f(input.into()) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error(),
    }
    .into()
}

#[proc_macro]
pub fn mat(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::mat_impl)
}

#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    expand(input, macros::einsum::einsum_impl)
}

#[proc_macro]
pub fn fuse(input: TokenStream) -> TokenStream {
    expand(input, macros::fuse::fuse_impl)
}

#[proc_macro]
pub fn mega_fuse(input: TokenStream) -> TokenStream {
    expand(input, macros::fuse::mega_fuse_impl)
}

#[proc_macro]
pub fn stencil(input: TokenStream) -> TokenStream {
    expand(input, macros::stencil::stencil_impl)
}

#[proc_macro]
pub fn named(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::named_impl)
}

#[proc_macro]
pub fn generated(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::generated_impl)
}

#[proc_macro]
pub fn axis(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::axis_impl)
}

#[proc_macro]
pub fn named_zeros(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::named_zeros_impl)
}

#[proc_macro_attribute]
pub fn nabla_grad(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand(item, macros::attrs::nabla_grad_impl)
}

#[proc_macro]
pub fn block(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::block_impl)
}

#[proc_macro]
pub fn sym(input: TokenStream) -> TokenStream {
    expand(input, macros::sym::sym_impl)
}

#[proc_macro]
pub fn math(input: TokenStream) -> TokenStream {
    expand(input, macros::math::math_impl)
}

#[proc_macro_derive(Module, attributes(param))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    expand(input, macros::derive_module::derive_module_impl)
}

#[proc_macro_attribute]
pub fn nabla_main(attr: TokenStream, item: TokenStream) -> TokenStream {
    match macros::attrs::nabla_main_impl(attr.into(), item.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
