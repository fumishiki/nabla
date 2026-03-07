//! Expression codegen for fuse!/mega_fuse! — scalar, CUDA, and tensor-level rewriting.
//!
//! Transforms `syn::Expr` ASTs into:
//! - Scalar closures for CPU from_fn paths
//! - CUDA C expression strings for GPU JIT kernels
//! - Tensor-level operations for non-fusible fallback

use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::{Error, Expr, ExprBinary, ExprMethodCall, ExprPath, ExprUnary, Result};

use super::expr::{contains_tensor, scalar_method_name};

const CUDA_UNARY_MAP: &[(&str, &str)] = &[
    ("exp", "exp"), ("ln", "log"), ("log1p", "log1p"), ("sin", "sin"),
    ("cos", "cos"), ("tanh", "tanh"), ("sqrt", "sqrt"), ("abs", "fabs"),
    ("erf", "erf"), ("ceil", "ceil"), ("floor", "floor"), ("round", "round"),
];

fn cuda_method_expr(method: &str, recv: &str) -> Option<String> {
    if method == "recip" { return Some(format!("(1.0/({recv}))")); }
    if method == "neg" { return Some(format!("(-{recv})")); }
    CUDA_UNARY_MAP
        .iter()
        .find(|(m, _)| *m == method)
        .map(|(_, c)| format!("{c}({recv})"))
}

fn cuda_binop(op: &syn::BinOp, ctx: &str) -> Result<&'static str> {
    match op {
        syn::BinOp::Add(_) => Ok("+"),
        syn::BinOp::Sub(_) => Ok("-"),
        syn::BinOp::Mul(_) => Ok("*"),
        syn::BinOp::Div(_) => Ok("/"),
        _ => Err(Error::new(
            Span::call_site(),
            format!("{ctx}: unsupported binary op for GPU codegen"),
        )),
    }
}

fn cast_int_lit(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(i),
            ..
        }) => i.base10_parse::<f64>().ok(),
        _ => None,
    }
}

pub(crate) fn scalar_expr(expr: &Expr, tensor_names: &[String]) -> Result<TokenStream2> {
    scalar_expr_inner(expr, tensor_names, None, None)
}

pub(crate) fn scalar_expr_mega(
    expr: &Expr,
    tensor_names: &[String],
    prev_body: Option<&TokenStream2>,
) -> Result<TokenStream2> {
    scalar_expr_inner(
        expr,
        tensor_names,
        prev_body,
        Some("mega_fuse!: `prev` referenced but no preceding operation exists"),
    )
}

fn scalar_expr_inner(
    expr: &Expr,
    tensor_names: &[String],
    prev_body: Option<&TokenStream2>,
    prev_err: Option<&'static str>,
) -> Result<TokenStream2> {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let name = path.segments[0].ident.to_string();
            if name == "prev" {
                if let Some(prev) = prev_body {
                    return Ok(prev.clone());
                }
                if let Some(msg) = prev_err {
                    return Err(Error::new(Span::call_site(), msg));
                }
            }
            if tensor_names.contains(&name) {
                let fused = Ident::new(&format!("__fuse_v_{name}"), path.segments[0].ident.span());
                Ok(quote! { #fused })
            } else {
                Ok(expr.to_token_stream())
            }
        }
        Expr::Lit(_) => Ok(expr.to_token_stream()),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l = scalar_expr_inner(left, tensor_names, prev_body, prev_err)?;
            let r = scalar_expr_inner(right, tensor_names, prev_body, prev_err)?;
            Ok(quote! { (#l #op #r) })
        }
        Expr::Unary(ExprUnary {
            op, expr: inner, ..
        }) => {
            let i = scalar_expr_inner(inner, tensor_names, prev_body, prev_err)?;
            Ok(quote! { (#op #i) })
        }
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let recv = scalar_expr_inner(receiver, tensor_names, prev_body, prev_err)?;
            let rewritten_args: Vec<_> = args
                .iter()
                .map(|a| scalar_expr_inner(a, tensor_names, prev_body, prev_err))
                .collect::<Result<Vec<_>>>()?;
            let method_name = method.to_string();
            if method_name == "neg" {
                return Ok(quote! { (-#recv) });
            }
            if let Some(sn) = scalar_method_name(&method_name) {
                let sm = Ident::new(sn, method.span());
                Ok(quote! { #recv.#sm(#(#rewritten_args),*) })
            } else {
                Ok(quote! { #recv.#method(#(#rewritten_args),*) })
            }
        }
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                if let Some(sn) = scalar_method_name(&fname) {
                    let arg = scalar_expr_inner(&ec.args[0], tensor_names, prev_body, prev_err)?;
                    let method = Ident::new(sn, path.segments[0].ident.span());
                    return Ok(quote! { #arg.#method() });
                }
            }
            let func = &ec.func;
            let rewritten_args: Vec<_> = ec
                .args
                .iter()
                .map(|a| scalar_expr_inner(a, tensor_names, prev_body, prev_err))
                .collect::<Result<Vec<_>>>()?;
            Ok(quote! { #func(#(#rewritten_args),*) })
        }
        Expr::Paren(ep) => {
            let inner = scalar_expr_inner(&ep.expr, tensor_names, prev_body, prev_err)?;
            Ok(quote! { (#inner) })
        }
        Expr::Cast(ec) => {
            if let Some(v) = cast_int_lit(ec.expr.as_ref()) {
                return Ok(quote! { #v });
            }
            let inner = scalar_expr_inner(&ec.expr, tensor_names, prev_body, prev_err)?;
            let ty = &ec.ty;
            Ok(quote! { (#inner as #ty) })
        }
        _ => Ok(expr.to_token_stream()),
    }
}

pub(crate) fn cuda_expr(expr: &Expr, tensor_names: &[String]) -> Result<String> {
    cuda_expr_inner(expr, tensor_names, false)
}

pub(crate) fn cuda_expr_mega(expr: &Expr, tensor_names: &[String]) -> Result<String> {
    cuda_expr_inner(expr, tensor_names, true)
}

fn cuda_expr_inner(expr: &Expr, tensor_names: &[String], mega_mode: bool) -> Result<String> {
    let ctx = if mega_mode { "mega_fuse!" } else { "fuse!" };
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let name = path.segments[0].ident.to_string();
            if mega_mode && name == "prev" {
                return Ok("__NABLA_PREV__".to_string());
            }
            if let Some(idx) = tensor_names.iter().position(|n| n == &name) {
                Ok(format!("in{idx}[i]"))
            } else {
                Ok(name)
            }
        }
        Expr::Lit(syn::ExprLit { lit, .. }) => match lit {
            syn::Lit::Float(f) => Ok(f.to_string()),
            syn::Lit::Int(i) => Ok(format!("(double)({})", i.base10_digits())),
            _ => Ok(lit.to_token_stream().to_string()),
        },
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l = cuda_expr_inner(left, tensor_names, mega_mode)?;
            let r = cuda_expr_inner(right, tensor_names, mega_mode)?;
            let op_str = cuda_binop(op, ctx)?;
            Ok(format!("({l} {op_str} {r})"))
        }
        Expr::Unary(ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: inner,
            ..
        }) => {
            let i = cuda_expr_inner(inner, tensor_names, mega_mode)?;
            Ok(format!("(-{i})"))
        }
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let recv = cuda_expr_inner(receiver, tensor_names, mega_mode)?;
            let method_name = method.to_string();
            if let Some(result) = cuda_method_expr(&method_name, &recv) {
                Ok(result)
            } else if method_name == "powf" && args.len() == 1 {
                let p = cuda_expr_inner(&args[0], tensor_names, mega_mode)?;
                Ok(format!("pow({recv}, {p})"))
            } else {
                Err(Error::new_spanned(
                    method,
                    format!("{ctx}: unsupported GPU method: {method_name}"),
                ))
            }
        }
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                let arg = cuda_expr_inner(&ec.args[0], tensor_names, mega_mode)?;
                if let Some(result) = cuda_method_expr(&fname, &arg) {
                    return Ok(result);
                }
            }
            Err(Error::new_spanned(
                &ec.func,
                format!("{ctx}: unsupported GPU function call"),
            ))
        }
        Expr::Paren(ep) => {
            let inner = cuda_expr_inner(&ep.expr, tensor_names, mega_mode)?;
            Ok(format!("({inner})"))
        }
        Expr::Cast(ec) => {
            if let Some(v) = cast_int_lit(ec.expr.as_ref()) {
                return Ok(format!("{v}"));
            }
            cuda_expr_inner(&ec.expr, tensor_names, mega_mode)
        }
        _ => Err(Error::new(
            Span::call_site(),
            format!("{ctx}: unsupported expression for GPU codegen"),
        )),
    }
}

pub(crate) fn lift_expr(expr: &Expr, tensor_names: &[String]) -> TokenStream2 {
    match expr {
        Expr::Path(_) => expr.to_token_stream(),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l_has = contains_tensor(left, tensor_names);
            let r_has = contains_tensor(right, tensor_names);
            let l = lift_expr(left, tensor_names);
            let r = lift_expr(right, tensor_names);
            lift_binop(op, l, r, l_has, r_has)
        }
        Expr::Unary(ExprUnary {
            op, expr: inner, ..
        }) => {
            let i = lift_expr(inner, tensor_names);
            lift_unary(op, i, contains_tensor(inner, tensor_names))
        }
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let recv = lift_expr(receiver, tensor_names);
            let rewritten_args: Vec<_> = args.iter().map(|a| lift_expr(a, tensor_names)).collect();
            quote! { #recv.#method(#(#rewritten_args),*) }
        }
        Expr::Call(ec) => {
            let func = &ec.func;
            let rewritten_args: Vec<_> =
                ec.args.iter().map(|a| lift_expr(a, tensor_names)).collect();
            quote! { #func(#(#rewritten_args),*) }
        }
        Expr::Paren(ep) => {
            let inner = lift_expr(&ep.expr, tensor_names);
            quote! { (#inner) }
        }
        Expr::Reference(er) => {
            let inner = lift_expr(&er.expr, tensor_names);
            if er.mutability.is_some() {
                quote! { &mut #inner }
            } else {
                quote! { &#inner }
            }
        }
        _ => expr.to_token_stream(),
    }
}

fn lift_binop(
    op: &syn::BinOp,
    l: TokenStream2,
    r: TokenStream2,
    l_has: bool,
    r_has: bool,
) -> TokenStream2 {
    match (l_has, r_has, op) {
        (true, true, syn::BinOp::Mul(_)) => quote! { (#l).emul(&#r) },
        (true, true, syn::BinOp::Div(_)) => quote! { (#l).ediv(&#r) },
        (true, true, syn::BinOp::Add(_) | syn::BinOp::Sub(_)) => quote! { (&#l #op &#r) },
        (true, false, syn::BinOp::Mul(_)) => quote! { (&#l * #r) },
        (false, true, syn::BinOp::Mul(_)) => quote! { (&#r * #l) },
        _ => quote! { (#l #op #r) },
    }
}

fn lift_unary(op: &syn::UnOp, inner: TokenStream2, has_tensor: bool) -> TokenStream2 {
    if has_tensor && matches!(op, syn::UnOp::Neg(_)) {
        quote! { (-&#inner) }
    } else {
        quote! { (#op #inner) }
    }
}

pub(crate) const MAX_FUSE_REGISTERS: usize = 120;

pub(crate) fn estimate_register_pressure(expr: &Expr, tensor_names: &[String]) -> usize {
    let mut transcendental = 0usize;
    let mut arithmetic = 0usize;
    let mut inputs = std::collections::HashSet::new();
    count_ops(
        expr,
        tensor_names,
        &mut transcendental,
        &mut arithmetic,
        &mut inputs,
    );
    let input_regs = inputs.len() * 4;
    let output_regs = 4;
    input_regs + transcendental * 12 + arithmetic * 2 + output_regs
}

fn count_ops(
    expr: &Expr,
    tensor_names: &[String],
    transcendental: &mut usize,
    arithmetic: &mut usize,
    inputs: &mut std::collections::HashSet<String>,
) {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let name = path.segments[0].ident.to_string();
            if tensor_names.contains(&name) {
                inputs.insert(name);
            }
        }
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            match op {
                syn::BinOp::Add(_)
                | syn::BinOp::Sub(_)
                | syn::BinOp::Mul(_)
                | syn::BinOp::Div(_) => {
                    *arithmetic += 1;
                }
                _ => {}
            }
            count_ops(left, tensor_names, transcendental, arithmetic, inputs);
            count_ops(right, tensor_names, transcendental, arithmetic, inputs);
        }
        Expr::Unary(ExprUnary { expr: inner, .. }) => {
            count_ops(inner, tensor_names, transcendental, arithmetic, inputs);
        }
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let name = method.to_string();
            let (t_ops, a_ops) = op_cost(&name);
            *transcendental += t_ops;
            *arithmetic += a_ops;
            count_ops(receiver, tensor_names, transcendental, arithmetic, inputs);
            for a in args {
                count_ops(a, tensor_names, transcendental, arithmetic, inputs);
            }
        }
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                let (t_ops, a_ops) = op_cost(&fname);
                *transcendental += t_ops;
                *arithmetic += a_ops;
            }
            for a in &ec.args {
                count_ops(a, tensor_names, transcendental, arithmetic, inputs);
            }
        }
        Expr::Paren(ep) => {
            count_ops(&ep.expr, tensor_names, transcendental, arithmetic, inputs);
        }
        Expr::Cast(ec) => {
            count_ops(&ec.expr, tensor_names, transcendental, arithmetic, inputs);
        }
        _ => {}
    }
}

const TRANSCENDENTAL_OPS: &[&str] = &["exp", "ln", "sqrt", "sin", "cos", "tanh", "erf", "log1p"];
const CHEAP_OPS: &[&str] = &["abs", "ceil", "floor", "round", "neg", "recip"];

fn op_cost(name: &str) -> (usize, usize) {
    if TRANSCENDENTAL_OPS.contains(&name) || name == "powf" { (1, 0) }
    else if CHEAP_OPS.contains(&name) { (0, 1) }
    else { (0, 0) }
}

pub(crate) fn expr_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}
