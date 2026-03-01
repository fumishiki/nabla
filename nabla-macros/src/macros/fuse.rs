//! fuse! / mega_fuse! proc macro implementations.
//!
//! Generates fused element-wise GPU kernels with multiple fusion levels:
//! - L1: element-wise fusion (single from_fn pass)
//! - L3: GEMM + activation fusion (cublasLt epilogue)
//! - L4: map-reduce fusion (fused reduction kernel)
//! - mega_fuse!: multi-output DAG fusion with register pass-through

use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    Error, Expr, ExprBinary, Result,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Comma,
};

use crate::fusion::codegen::{
    MAX_FUSE_REGISTERS, cuda_expr, cuda_expr_mega, estimate_register_pressure, expr_hash,
    lift_expr, scalar_expr, scalar_expr_mega,
};
use crate::fusion::eqsat::eqsat_simplify;
use crate::fusion::expr::{
    collect_all_path_idents, collect_tensor_idents, expr_references_prev, is_elementwise_fusible,
};

fn err(msg: impl std::fmt::Display) -> Error {
    Error::new(Span::call_site(), msg)
}

fn let_bindings(tensors: &[Ident]) -> Vec<TokenStream2> {
    tensors
        .iter()
        .map(|t| {
            let var = Ident::new(&format!("__fuse_v_{t}"), t.span());
            quote! { let #var = #t.get(__fuse_r, __fuse_c); }
        })
        .collect()
}

fn parse_input_idents(input: ParseStream<'_>, err_msg: &'static str) -> Result<Vec<Ident>> {
    let parsed: Vec<Ident> = Punctuated::<Ident, Comma>::parse_terminated(input)?
        .into_iter()
        .collect();
    if parsed.is_empty() {
        return Err(err(err_msg));
    }
    Ok(parsed)
}

fn storage_ptrs(tensors: &[Ident]) -> Vec<TokenStream2> {
    tensors
        .iter()
        .map(|t| quote! { #t.__storage_ptr() })
        .collect()
}

fn emit_shape_checks(tensors: &[Ident], msg: &str) -> Vec<TokenStream2> {
    let Some(first) = tensors.first() else {
        return Vec::new();
    };
    tensors
        .iter()
        .skip(1)
        .map(|t| quote! { assert_eq!(#first.shape(), #t.shape(), #msg); })
        .collect()
}

struct MapReduceInfo {
    pointwise_expr: Expr,
    reduce_op: &'static str,
    reduce_op_id: u8,
    axis: usize,
    inputs: Vec<Ident>,
}

fn try_extract_map_reduce(expr: &Expr, tensor_names: &[String]) -> Option<MapReduceInfo> {
    let Expr::MethodCall(mc) = expr else {
        return None;
    };

    let method_name = mc.method.to_string();
    let (reduce_op, reduce_op_id) = match method_name.as_str() {
        "sum_axis" => ("sum", 0),
        "mean_axis" => ("mean", 3),
        _ => return None,
    };
    if mc.args.len() != 1 {
        return None;
    }

    let axis = match mc.args.iter().next()? {
        Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(i),
            ..
        }) => match i.base10_parse::<usize>().ok()? {
            v @ (0 | 1) => v,
            _ => return None,
        },
        _ => return None,
    };

    let inner = &mc.receiver;
    if !is_elementwise_fusible(inner) {
        return None;
    }

    let mut inputs: Vec<Ident> = Vec::new();
    collect_tensor_idents(inner, tensor_names, &mut inputs);
    if inputs.is_empty() {
        return None;
    }

    Some(MapReduceInfo {
        pointwise_expr: *inner.clone(),
        reduce_op,
        reduce_op_id,
        axis,
        inputs,
    })
}

fn emit_map_reduce_fuse(mr: MapReduceInfo) -> Result<TokenStream2> {
    let MapReduceInfo {
        pointwise_expr,
        reduce_op,
        reduce_op_id,
        axis,
        inputs,
    } = mr;

    let input_names: Vec<String> = inputs.iter().map(|i| i.to_string()).collect();
    let axis_u8 = axis as u8;

    let scalar_body = scalar_expr(&pointwise_expr, &input_names)?;
    let gpu_expr_str = cuda_expr(&pointwise_expr, &input_names)?;
    let kernel_hash = expr_hash(&format!("{gpu_expr_str}_reduce{axis}_{reduce_op}"));
    let n_inputs = inputs.len();

    let let_bindings = let_bindings(&inputs);
    let storage_ptrs = storage_ptrs(&inputs);

    let first_input = &inputs[0];
    let shape_checks = emit_shape_checks(&inputs, "fuse! map-reduce: shape mismatch");

    Ok(quote! {{
        #(#shape_checks)*
        use nabla::scalar::MathOps as _;
        let __fuse_inputs: &[*const u8] = &[#(#storage_ptrs),*];
        nabla::tensor::Tensor::__fuse_reduce(
            __fuse_inputs,
            #first_input.nrows(),
            #first_input.ncols(),
            |__fuse_r, __fuse_c| {
                #(#let_bindings)*
                #scalar_body
            },
            #gpu_expr_str,
            #kernel_hash,
            #n_inputs,
            #reduce_op_id,
            #axis_u8,
        )
    }})
}

const GEMM_ACTIVATIONS: &[&str] = &[
    "sigmoid", "relu", "tanh", "gelu", "exp", "ln", "sqrt", "abs", "neg", "recip",
];

fn detect_gemm_activation(expr: &Expr) -> Option<(Expr, Expr, Ident)> {
    if let Expr::MethodCall(mc) = expr
        && mc.args.is_empty()
        && GEMM_ACTIVATIONS.contains(&mc.method.to_string().as_str())
    {
        let inner = match &*mc.receiver {
            Expr::Paren(ep) => &ep.expr,
            other => other,
        };
        if let Expr::Binary(ExprBinary {
            left,
            op: syn::BinOp::Mul(_),
            right,
            ..
        }) = inner
        {
            let lhs = strip_ref(left);
            let rhs = strip_ref(right);
            return Some((lhs.clone(), rhs.clone(), mc.method.clone()));
        }
    }
    None
}

fn strip_ref(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(er) => &er.expr,
        other => other,
    }
}

fn emit_gemm_activation(lhs: Expr, rhs: Expr, act: Ident) -> TokenStream2 {
    let act_str = act.to_string();
    match act_str.as_str() {
        "relu" => quote! {{
            nabla::tensor::Tensor::__matmul_epilogue(&#lhs, &#rhs, 0)
        }},
        "gelu" => quote! {{
            nabla::tensor::Tensor::__matmul_epilogue(&#lhs, &#rhs, 1)
        }},
        _ => quote! {{
            let __fuse_c = &#lhs * &#rhs;
            __fuse_c.#act()
        }},
    }
}

pub(crate) struct FuseInput {
    pub(crate) body: Expr,
    pub(crate) tensors: Vec<Ident>,
}

impl Parse for FuseInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let body: Expr = input.parse()?;

        let tensors = if input.peek(syn::Token![;]) {
            input.parse::<syn::Token![;]>()?;
            parse_input_idents(
                input,
                "fuse!: empty tensor list after `;` — either list tensors or omit `;`",
            )?
        } else {
            let mut auto = Vec::new();
            collect_all_path_idents(&body, &mut auto);
            if auto.is_empty() {
                return Err(err("fuse!: no tensor variables detected in expression"));
            }
            auto
        };

        Ok(FuseInput { body, tensors })
    }
}

pub(crate) fn fuse_impl(input: TokenStream2) -> Result<TokenStream2> {
    let FuseInput { body, tensors } = syn::parse2(input)?;
    let tensor_names: Vec<String> = tensors.iter().map(|t| t.to_string()).collect();
    let body = eqsat_simplify(&body);

    // L4: Map-reduce fusion
    if let Some(mr) = try_extract_map_reduce(&body, &tensor_names) {
        return emit_map_reduce_fuse(mr);
    }

    // L3: GEMM + activation fusion
    if let Some((lhs, rhs, act)) = detect_gemm_activation(&body) {
        return Ok(emit_gemm_activation(lhs, rhs, act));
    }

    let first = &tensors[0];
    let shape_checks = emit_shape_checks(&tensors, "fuse!: shape mismatch");

    let fused_body = if is_elementwise_fusible(&body) {
        // L1: Fused path — single kernel pass
        let scalar_body = scalar_expr(&body, &tensor_names)?;
        let let_bindings = let_bindings(&tensors);

        let gpu_expr_str = cuda_expr(&body, &tensor_names)?;
        let kernel_hash = expr_hash(&gpu_expr_str);
        let n_inputs = tensors.len();

        let reg_estimate = estimate_register_pressure(&body, &tensor_names);
        if reg_estimate > MAX_FUSE_REGISTERS {
            eprintln!(
                "warning: fuse! estimated register pressure {reg_estimate} exceeds \
                 threshold {MAX_FUSE_REGISTERS} — kernel may spill to local memory"
            );
        }
        let reg_est_lit = reg_estimate;

        let storage_ptrs = storage_ptrs(&tensors);

        quote! {{
            use nabla::scalar::MathOps as _;
            let __fuse_inputs: &[*const u8] = &[#(#storage_ptrs),*];
            nabla::tensor::Tensor::__fuse_elementwise(
                __fuse_inputs,
                #first.nrows(),
                #first.ncols(),
                |__fuse_r, __fuse_c| {
                    #(#let_bindings)*
                    #scalar_body
                },
                #gpu_expr_str,
                #kernel_hash,
                #n_inputs,
                #reg_est_lit,
            )
        }}
    } else {
        // Fallback: tensor-level chained ops
        eprintln!(
            "warning: fuse! expression is not element-wise fusible — falling back to tensor-level ops. \
                   Only +, -, *, /, exp, ln, log1p, sin, cos, tanh, sqrt, abs, recip, erf, ceil, floor, round, neg, powf are fusible."
        );
        let lifted = lift_expr(&body, &tensor_names);
        quote! { #lifted }
    };

    Ok(quote! {{
        #(#shape_checks)*
        #fused_body
    }})
}

pub(crate) struct MegaFuseInput {
    pub(crate) bodies: Vec<Expr>,
    pub(crate) uses_prev: Vec<bool>,
    pub(crate) tensors: Vec<Ident>,
}

impl Parse for MegaFuseInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut bodies = Vec::new();
        let mut uses_prev = Vec::new();

        loop {
            if input.is_empty() {
                break;
            }
            if input.peek(syn::Ident) {
                let fork = input.fork();
                let ident: Ident = fork.parse()?;
                if ident == "inputs" && fork.peek(syn::Token![:]) {
                    let _: Ident = input.parse()?;
                    let _: syn::Token![:] = input.parse()?;
                    let tensors = parse_input_idents(
                        input,
                        "mega_fuse! needs at least one tensor after `inputs:`",
                    )?;
                    if bodies.is_empty() {
                        return Err(err(
                            "mega_fuse! needs at least one expression before `inputs:`",
                        ));
                    }
                    return Ok(MegaFuseInput {
                        bodies,
                        uses_prev,
                        tensors,
                    });
                }
            }
            let body: Expr = input.parse()?;
            let has_prev = expr_references_prev(&body);
            if has_prev && bodies.is_empty() {
                return Err(err(
                    "mega_fuse!: `prev` cannot appear in the first operation — \
                     there is no preceding op to reference",
                ));
            }
            uses_prev.push(has_prev);
            bodies.push(body);
            if input.peek(syn::Token![;]) {
                let _: syn::Token![;] = input.parse()?;
            } else {
                break;
            }
        }

        if bodies.is_empty() {
            return Err(err("mega_fuse! needs at least one expression"));
        }

        // Auto-capture: collect all path idents, excluding `prev`
        let mut tensors: Vec<Ident> = Vec::new();
        for body in &bodies {
            collect_all_path_idents(body, &mut tensors);
        }
        tensors.retain(|id| id != "prev");
        if tensors.is_empty() {
            return Err(err(
                "mega_fuse!: no tensor variables detected in expressions",
            ));
        }

        Ok(MegaFuseInput {
            bodies,
            uses_prev,
            tensors,
        })
    }
}

pub(crate) fn mega_fuse_impl(input: TokenStream2) -> Result<TokenStream2> {
    let MegaFuseInput {
        bodies,
        uses_prev,
        tensors,
    } = syn::parse2(input)?;
    let tensor_names: Vec<String> = tensors.iter().map(|t| t.to_string()).collect();

    for (i, body) in bodies.iter().enumerate() {
        if !is_elementwise_fusible(body) {
            return Err(Error::new(
                Span::call_site(),
                format!(
                    "mega_fuse! expression {i} is not element-wise fusible — \
                     only element-wise ops (exp, sin, tanh, +, -, *, /) are supported"
                ),
            ));
        }
    }

    let first = &tensors[0];
    let shape_checks = emit_shape_checks(&tensors, "mega_fuse!: shape mismatch");

    let let_bindings = let_bindings(&tensors);

    let mut gpu_exprs: Vec<String> = Vec::new();
    let mut cpu_closures: Vec<TokenStream2> = Vec::new();
    let mut combined_hash = String::new();
    let mut prev_scalar_body: Option<TokenStream2> = None;
    let mut op_inputs: Vec<Vec<Ident>> = Vec::new();

    for body in &bodies {
        let simplified = eqsat_simplify(body);
        let mut inputs = Vec::new();
        collect_tensor_idents(&simplified, &tensor_names, &mut inputs);
        let op_tensor_names: Vec<String> = inputs.iter().map(|i| i.to_string()).collect();

        let gpu_str = cuda_expr_mega(&simplified, &op_tensor_names)?;
        combined_hash.push_str(&gpu_str);
        combined_hash.push(';');
        gpu_exprs.push(gpu_str);

        let scalar_body =
            scalar_expr_mega(&simplified, &op_tensor_names, prev_scalar_body.as_ref())?;

        cpu_closures.push(quote! {
            Box::new(|__fuse_r: usize, __fuse_c: usize| {
                #(#let_bindings)*
                #scalar_body
            }) as Box<dyn FnMut(usize, usize) -> _>
        });

        prev_scalar_body = Some(scalar_body);
        op_inputs.push(inputs);
    }

    let kernel_hash = expr_hash(&combined_hash);
    let gpu_expr_lits: Vec<TokenStream2> = gpu_exprs.iter().map(|e| quote! { #e }).collect();
    let uses_prev_lits: Vec<TokenStream2> = uses_prev
        .iter()
        .map(|&b| {
            if b {
                quote! { true }
            } else {
                quote! { false }
            }
        })
        .collect();
    let op_ptrs: Vec<Vec<TokenStream2>> = op_inputs.iter().map(|v| storage_ptrs(v)).collect();
    let op_ptr_lists: Vec<TokenStream2> = op_ptrs
        .iter()
        .map(|ptrs| quote! { vec![#(#ptrs),*] })
        .collect();
    let op_input_counts: Vec<TokenStream2> = op_inputs
        .iter()
        .map(|inputs| {
            let n = inputs.len();
            quote! { #n }
        })
        .collect();

    Ok(quote! {{
        #(#shape_checks)*
        use nabla::scalar::MathOps as _;
        let __mega_ops: Vec<(Vec<*const u8>, String, usize, bool)> = vec![
            #( (#op_ptr_lists, #gpu_expr_lits.to_string(), #op_input_counts, #uses_prev_lits) ),*
        ];
        let __mega_cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> _>> = vec![
            #(#cpu_closures),*
        ];
        nabla::tensor::Tensor::__mega_fuse_elementwise(
            &__mega_ops,
            #first.nrows(),
            #first.ncols(),
            __mega_cpu_fns,
            #kernel_hash,
        )
    }})
}
