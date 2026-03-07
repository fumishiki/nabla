//! Shared expression analysis utilities for fuse!/mega_fuse!.
//!
//! Provides expression analysis (fusibility, tensor detection, ident collection)
//! and method name mapping. Codegen lives in `codegen.rs`.

use proc_macro2::Ident;
use syn::{Expr, ExprBinary, ExprMethodCall, ExprPath, ExprUnary};

const ELEMENTWISE_UNARY: &[&str] = &[
    "exp", "ln", "log1p", "sin", "cos", "tanh", "sqrt", "abs", "recip", "erf", "ceil", "floor",
    "round", "neg",
];

const ELEMENTWISE_UNARY_ARG: &[&str] = &["powf"];

fn single_ident(expr: &Expr) -> Option<&Ident> {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            Some(&path.segments[0].ident)
        }
        _ => None,
    }
}

fn expr_any(expr: &Expr, pred: &mut impl FnMut(&Expr) -> bool) -> bool {
    if pred(expr) {
        return true;
    }
    match expr {
        Expr::Binary(ExprBinary { left, right, .. }) => {
            expr_any(left, pred) || expr_any(right, pred)
        }
        Expr::Unary(ExprUnary { expr: inner, .. }) => expr_any(inner, pred),
        Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
            expr_any(receiver, pred) || args.iter().any(|a| expr_any(a, pred))
        }
        Expr::Call(ec) => ec.args.iter().any(|a| expr_any(a, pred)),
        Expr::Paren(ep) => expr_any(&ep.expr, pred),
        Expr::Reference(er) => expr_any(&er.expr, pred),
        Expr::Cast(ec) => expr_any(&ec.expr, pred),
        _ => false,
    }
}

pub(crate) fn is_elementwise_fusible(expr: &Expr) -> bool {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => true,
        Expr::Lit(_) => true,
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            matches!(
                op,
                syn::BinOp::Add(_) | syn::BinOp::Sub(_) | syn::BinOp::Mul(_) | syn::BinOp::Div(_)
            ) && is_elementwise_fusible(left)
                && is_elementwise_fusible(right)
        }
        Expr::Unary(ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: inner,
            ..
        }) => is_elementwise_fusible(inner),
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let name = method.to_string();
            let recv_ok = is_elementwise_fusible(receiver);
            is_elementwise_method(&name, args.len())
                && recv_ok
                && args.first().is_none_or(is_elementwise_fusible)
        }
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                is_elementwise_free_fn(&fname, ec.args.len()) && is_elementwise_fusible(&ec.args[0])
            } else {
                false
            }
        }
        Expr::Paren(ep) => is_elementwise_fusible(&ep.expr),
        Expr::Cast(ec) => is_elementwise_fusible(&ec.expr),
        _ => false,
    }
}

pub(crate) fn contains_tensor(expr: &Expr, tensor_names: &[String]) -> bool {
    let mut pred = |e: &Expr| {
        single_ident(e).is_some_and(|ident| tensor_names.contains(&ident.to_string()))
    };
    expr_any(expr, &mut pred)
}

fn collect_idents(
    expr: &Expr,
    out: &mut Vec<Ident>,
    accept: &mut impl FnMut(&Ident) -> bool,
    dedup_by_name: bool,
) {
    if let Some(ident) = single_ident(expr) {
        let is_dup = if dedup_by_name {
            let name = ident.to_string();
            out.iter().any(|i| *i == name)
        } else {
            out.iter().any(|i| i == ident)
        };
        if accept(ident) && !is_dup {
            out.push(ident.clone());
        }
        return;
    }
    match expr {
        Expr::Binary(ExprBinary { left, right, .. }) => {
            collect_idents(left, out, accept, dedup_by_name);
            collect_idents(right, out, accept, dedup_by_name);
        }
        Expr::Unary(ExprUnary { expr: inner, .. }) => {
            collect_idents(inner, out, accept, dedup_by_name);
        }
        Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
            collect_idents(receiver, out, accept, dedup_by_name);
            for a in args {
                collect_idents(a, out, accept, dedup_by_name);
            }
        }
        Expr::Call(ec) => {
            for a in &ec.args {
                collect_idents(a, out, accept, dedup_by_name);
            }
        }
        Expr::Paren(ep) => collect_idents(&ep.expr, out, accept, dedup_by_name),
        Expr::Reference(er) => collect_idents(&er.expr, out, accept, dedup_by_name),
        _ => {}
    }
}

pub(crate) fn collect_tensor_idents(expr: &Expr, tensor_names: &[String], out: &mut Vec<Ident>) {
    let mut accept = |ident: &Ident| tensor_names.contains(&ident.to_string());
    collect_idents(expr, out, &mut accept, true);
}

pub(crate) fn collect_all_path_idents(expr: &Expr, out: &mut Vec<Ident>) {
    let mut accept = |_: &Ident| true;
    collect_idents(expr, out, &mut accept, false);
}

const SCALAR_METHOD_MAP: &[(&str, &str)] = &[
    ("exp", "math_exp"), ("ln", "math_ln"), ("log1p", "math_log1p"),
    ("sin", "math_sin"), ("cos", "math_cos"), ("tanh", "math_tanh"),
    ("sqrt", "math_sqrt"), ("abs", "math_abs"), ("recip", "math_recip"),
    ("erf", "math_erf"), ("ceil", "math_ceil"), ("floor", "math_floor"),
    ("round", "math_round"), ("powf", "math_powf"),
];

pub(crate) fn scalar_method_name(method: &str) -> Option<&'static str> {
    SCALAR_METHOD_MAP
        .iter()
        .find(|(m, _)| *m == method)
        .map(|(_, s)| *s)
}

pub(crate) fn expr_references_prev(expr: &Expr) -> bool {
    let mut pred = |e: &Expr| single_ident(e).is_some_and(|ident| ident == "prev");
    expr_any(expr, &mut pred)
}

fn is_elementwise_method(name: &str, arg_count: usize) -> bool {
    (arg_count == 0 && ELEMENTWISE_UNARY.contains(&name))
        || (arg_count == 1 && ELEMENTWISE_UNARY_ARG.contains(&name))
}

fn is_elementwise_free_fn(name: &str, arg_count: usize) -> bool {
    arg_count == 1 && ELEMENTWISE_UNARY.contains(&name)
}
