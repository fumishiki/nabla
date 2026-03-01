//! `#[derive(Module)]` — auto-generate `Module` trait parameter boilerplate.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, GenericParam, Meta, TypeParam, parse2};

pub fn derive_module_impl(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = parse2(input)?;
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &input,
                    "Module derive requires named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input,
                "Module derive only supports structs",
            ));
        }
    };

    let has_training = fields
        .iter()
        .any(|f| f.ident.as_ref().is_some_and(|id| id == "training"));
    if !has_training {
        return Err(syn::Error::new_spanned(
            &input,
            "Module derive requires a `training: bool` field",
        ));
    }

    let mut required = Vec::new();
    let mut optional = Vec::new();

    for field in fields {
        let ident = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "unnamed field"))?;
        for attr in &field.attrs {
            if !attr.path().is_ident("param") {
                continue;
            }
            match &attr.meta {
                Meta::Path(_) => required.push(ident.clone()),
                Meta::List(list) => {
                    let kw: syn::Ident = list.parse_args()?;
                    if kw == "optional" {
                        optional.push(ident.clone());
                    } else {
                        return Err(syn::Error::new_spanned(&kw, "expected `optional`"));
                    }
                }
                Meta::NameValue(_) => {
                    return Err(syn::Error::new_spanned(attr, "unexpected #[param = ...]"));
                }
            }
        }
    }

    let (t_param, b_param) = extract_tb_params(&input)?;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let param_count = required.len() + optional.len();

    let req_push: Vec<_> = required
        .iter()
        .map(|id| quote! { params.push(&self.#id); })
        .collect();
    let opt_push: Vec<_> = optional
        .iter()
        .map(|id| quote! { if let Some(ref v) = self.#id { params.push(v); } })
        .collect();

    let req_named_push: Vec<_> = required
        .iter()
        .map(|id| {
            let s = id.to_string();
            quote! { params.push((#s, &self.#id)); }
        })
        .collect();
    let opt_named_push: Vec<_> = optional
        .iter()
        .map(|id| {
            let s = id.to_string();
            quote! { if let Some(ref v) = self.#id { params.push((#s, v)); } }
        })
        .collect();

    let req_mut_push: Vec<_> = required
        .iter()
        .map(|id| quote! { params.push(&mut self.#id); })
        .collect();
    let opt_mut_push: Vec<_> = optional
        .iter()
        .map(|id| quote! { if let Some(ref mut v) = self.#id { params.push(v); } })
        .collect();

    let req_named_mut_push: Vec<_> = required
        .iter()
        .map(|id| {
            let s = id.to_string();
            quote! { params.push((#s, &mut self.#id)); }
        })
        .collect();
    let opt_named_mut_push: Vec<_> = optional
        .iter()
        .map(|id| {
            let s = id.to_string();
            quote! { if let Some(ref mut v) = self.#id { params.push((#s, v)); } }
        })
        .collect();

    let struct_name_str = name.to_string();

    Ok(quote! {
        impl #impl_generics crate::module::Module<#t_param, #b_param> for #name #ty_generics #where_clause {
            fn forward(&self, _x: &nabla_core::tensor::Tensor<#t_param, #b_param>) -> nabla_core::tensor::Tensor<#t_param, #b_param> {
                ::core::panic!("{}: forward() not provided by derive(Module) — use impl_layer! or implement Module manually", #struct_name_str)
            }

            fn training(&self) -> bool { self.training }
            fn set_training(&mut self, training: bool) { self.training = training; }

            fn parameters(&self) -> Vec<&nabla_core::tensor::Tensor<#t_param, #b_param>> {
                let mut params = Vec::with_capacity(#param_count);
                #(#req_push)* #(#opt_push)*
                params
            }

            fn named_parameters(&self) -> Vec<(&str, &nabla_core::tensor::Tensor<#t_param, #b_param>)> {
                let mut params = Vec::with_capacity(#param_count);
                #(#req_named_push)* #(#opt_named_push)*
                params
            }

            fn parameters_mut(&mut self) -> Vec<&mut nabla_core::tensor::Tensor<#t_param, #b_param>> {
                let mut params = Vec::with_capacity(#param_count);
                #(#req_mut_push)* #(#opt_mut_push)*
                params
            }

            fn named_parameters_mut(&mut self) -> Vec<(&str, &mut nabla_core::tensor::Tensor<#t_param, #b_param>)> {
                let mut params = Vec::with_capacity(#param_count);
                #(#req_named_mut_push)* #(#opt_named_mut_push)*
                params
            }
        }
    })
}

fn extract_tb_params(input: &DeriveInput) -> syn::Result<(syn::Ident, TokenStream)> {
    let type_params: Vec<&TypeParam> = input
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let GenericParam::Type(tp) = p {
                Some(tp)
            } else {
                None
            }
        })
        .collect();

    let t = type_params
        .iter()
        .find(|tp| {
            tp.bounds
                .iter()
                .any(|b| matches!(b, syn::TypeParamBound::Trait(t) if t.path.is_ident("Scalar")))
        })
        .ok_or_else(|| {
            syn::Error::new_spanned(input, "Module derive requires a `T: Scalar` type parameter")
        })?;

    let b = type_params.iter().find(|tp| {
        tp.bounds
            .iter()
            .any(|b| matches!(b, syn::TypeParamBound::Trait(t) if t.path.is_ident("Backend")))
    });

    let t_ident = t.ident.clone();
    let b_tokens = b.map_or_else(
        || quote! { nabla_core::backend::DefaultBackend },
        |bp| {
            let id = &bp.ident;
            quote! { #id }
        },
    );
    Ok((t_ident, b_tokens))
}
