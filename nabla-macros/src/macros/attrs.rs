use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::quote;
use syn::Result;

pub(crate) fn nabla_grad_impl(item: TokenStream2) -> Result<TokenStream2> {
    let func: syn::ItemFn = syn::parse2(item)?;
    let sig = &func.sig;
    if sig.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            sig,
            "#[nabla_grad] requires exactly one scalar argument",
        ));
    }
    let syn::FnArg::Typed(pat_ty) = &sig.inputs[0] else {
        return Err(syn::Error::new_spanned(
            &sig.inputs[0],
            "#[nabla_grad] does not support self",
        ));
    };
    let (arg_name, arg_ty) = (&pat_ty.pat, &pat_ty.ty);
    let fn_name = &sig.ident;
    let grad_name = Ident::new(&format!("{fn_name}_grad"), fn_name.span());
    let vis = &func.vis;
    let body = &func.block;

    Ok(quote! {
        #func

        #vis fn #grad_name(#arg_name: #arg_ty) -> (#arg_ty, #arg_ty) {
            let #arg_name: nabla::scalar::Dual<#arg_ty> = nabla::scalar::Dual::new(
                #arg_name,
                <#arg_ty as nabla::scalar::Scalar>::from_f64(1.0),
            );
            let __nabla_result: nabla::scalar::Dual<#arg_ty> = #body;
            (__nabla_result.value, __nabla_result.deriv)
        }
    })
}

pub(crate) fn nabla_main_impl(attr: TokenStream2, item: TokenStream2) -> Result<TokenStream2> {
    let feature: syn::Ident = syn::parse2(attr)?;
    let func: syn::ItemFn = syn::parse2(item)?;
    let body = &func.block;
    let feature_str = feature.to_string();

    Ok(quote! {
        #[cfg(feature = #feature_str)]
        fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
            #body
            Ok(())
        }

        #[cfg(not(feature = #feature_str))]
        fn main() {
            eprintln!(concat!("this example requires --features ", #feature_str));
        }
    })
}
