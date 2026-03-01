//! Equality saturation (egg) for compile-time algebraic simplification.
//!
//! Converts `syn::Expr` to an e-graph representation, runs rewrite rules,
//! and extracts the smallest equivalent expression back into `syn::Expr`.

use egg::{AstSize, Id, RecExpr, Rewrite, Runner, Symbol, define_language, rewrite};
use ordered_float::OrderedFloat;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{Expr, ExprBinary, ExprPath, ExprUnary};

define_language! {
    pub(crate) enum FuseExpr {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Sub([Id; 2]),
        "/" = Div([Id; 2]),
        "neg" = Neg([Id; 1]),
        "exp" = Exp([Id; 1]),
        "ln"  = Ln([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "abs" = Abs([Id; 1]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "tanh" = Tanh([Id; 1]),
        "recip" = Recip([Id; 1]),
        "pow" = Pow([Id; 2]),
        Num(OrderedFloat<f64>),
        Symbol(Symbol),
    }
}

fn fuse_rules() -> Vec<Rewrite<FuseExpr, ()>> {
    vec![
        rewrite!("add-zero-r"; "(+ ?x 0)"  => "?x"),
        rewrite!("add-zero-l"; "(+ 0 ?x)"  => "?x"),
        rewrite!("add-comm";   "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("add-assoc";  "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-one-r";  "(* ?x 1)"  => "?x"),
        rewrite!("mul-one-l";  "(* 1 ?x)"  => "?x"),
        rewrite!("mul-zero-r"; "(* ?x 0)"  => "0"),
        rewrite!("mul-zero-l"; "(* 0 ?x)"  => "0"),
        rewrite!("mul-comm";   "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("mul-assoc";  "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("distrib-factor-l"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rewrite!("distrib-factor-r"; "(+ (* ?b ?a) (* ?c ?a))" => "(* (+ ?b ?c) ?a)"),
        rewrite!("double-neg"; "(neg (neg ?x))" => "?x"),
        rewrite!("ln-exp";     "(ln (exp ?x))"  => "?x"),
        rewrite!("pow-zero";   "(pow ?x 0)" => "1"),
        rewrite!("pow-one";    "(pow ?x 1)" => "?x"),
        rewrite!("abs-abs";    "(abs (abs ?x))" => "(abs ?x)"),
        rewrite!("sqrt-pow2";  "(pow (sqrt ?x) 2)" => "(abs ?x)"),
    ]
}

fn add_opaque_symbol(
    expr: &Expr,
    rec: &mut RecExpr<FuseExpr>,
    sym_map: &mut Vec<(String, Expr)>,
) -> Id {
    let name = format!("__sym{}", sym_map.len());
    let id = rec.add(FuseExpr::Symbol(Symbol::from(&*name)));
    sym_map.push((name, expr.clone()));
    id
}

fn as_numeric_node(lit: &syn::Lit) -> Option<FuseExpr> {
    match lit {
        syn::Lit::Float(f) => f
            .base10_parse::<f64>()
            .ok()
            .map(|v| FuseExpr::Num(OrderedFloat(v))),
        syn::Lit::Int(i) => i
            .base10_parse::<i64>()
            .ok()
            .map(|v| FuseExpr::Num(OrderedFloat(v as f64))),
        _ => None,
    }
}

fn method_to_expr(method: &str, recv: Id, rhs_arg: Option<Id>) -> Option<FuseExpr> {
    if let Some(arg) = rhs_arg {
        return (method == "powf").then_some(FuseExpr::Pow([recv, arg]));
    }
    match method {
        "exp" => Some(FuseExpr::Exp([recv])),
        "ln" => Some(FuseExpr::Ln([recv])),
        "sqrt" => Some(FuseExpr::Sqrt([recv])),
        "abs" => Some(FuseExpr::Abs([recv])),
        "sin" => Some(FuseExpr::Sin([recv])),
        "cos" => Some(FuseExpr::Cos([recv])),
        "tanh" => Some(FuseExpr::Tanh([recv])),
        "recip" => Some(FuseExpr::Recip([recv])),
        "neg" => Some(FuseExpr::Neg([recv])),
        _ => None,
    }
}

fn expr_to_egg(expr: &Expr, rec: &mut RecExpr<FuseExpr>, sym_map: &mut Vec<(String, Expr)>) -> Id {
    match expr {
        // Literal numbers
        Expr::Lit(syn::ExprLit { lit, .. }) => {
            if let Some(node) = as_numeric_node(lit) {
                rec.add(node)
            } else {
                add_opaque_symbol(expr, rec, sym_map)
            }
        }

        // Binary ops
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l = expr_to_egg(left, rec, sym_map);
            let r = expr_to_egg(right, rec, sym_map);
            match op {
                syn::BinOp::Add(_) => rec.add(FuseExpr::Add([l, r])),
                syn::BinOp::Sub(_) => rec.add(FuseExpr::Sub([l, r])),
                syn::BinOp::Mul(_) => rec.add(FuseExpr::Mul([l, r])),
                syn::BinOp::Div(_) => rec.add(FuseExpr::Div([l, r])),
                _ => add_opaque_symbol(expr, rec, sym_map),
            }
        }

        // Unary neg
        Expr::Unary(ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: inner,
            ..
        }) => {
            let child = expr_to_egg(inner, rec, sym_map);
            rec.add(FuseExpr::Neg([child]))
        }

        // Method calls: x.exp(), x.ln(), x.sin(), etc.
        Expr::MethodCall(mc) => {
            let method = mc.method.to_string();
            let recv = expr_to_egg(&mc.receiver, rec, sym_map);
            if let Some(node) = method_to_expr(
                method.as_str(),
                recv,
                if mc.args.len() == 1 {
                    Some(expr_to_egg(&mc.args[0], rec, sym_map))
                } else {
                    None
                },
            ) {
                rec.add(node)
            } else {
                add_opaque_symbol(expr, rec, sym_map)
            }
        }

        // Free function calls: exp(x), ln(x), sin(x), etc.
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                let arg = expr_to_egg(&ec.args[0], rec, sym_map);
                if let Some(node) = method_to_expr(fname.as_str(), arg, None) {
                    return rec.add(node);
                }
            }
            add_opaque_symbol(expr, rec, sym_map)
        }

        // Paren
        Expr::Paren(ep) => expr_to_egg(&ep.expr, rec, sym_map),

        // Simple ident — use its name directly
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let ident = path.segments[0].ident.to_string();
            rec.add(FuseExpr::Symbol(Symbol::from(ident)))
        }

        // Anything else — opaque symbol
        _ => add_opaque_symbol(expr, rec, sym_map),
    }
}

fn egg_to_expr(rec: &RecExpr<FuseExpr>, id: Id, sym_map: &[(String, Expr)]) -> Expr {
    let node = &rec[id];
    match node {
        FuseExpr::Num(n) => {
            let v = n.into_inner();
            if v.fract() == 0.0 && v.abs() < i64::MAX as f64 {
                let iv = v as i64;
                syn::parse_quote!(#iv as _)
            } else {
                syn::parse_quote!(#v)
            }
        }
        FuseExpr::Symbol(s) => {
            let name = s.as_str();
            for (sname, original) in sym_map {
                if sname == name {
                    return original.clone();
                }
            }
            let ident = Ident::new(name, Span::call_site());
            syn::parse_quote!(#ident)
        }
        FuseExpr::Add([a, b]) => bin_expr(rec, sym_map, *a, *b, quote!(+)),
        FuseExpr::Sub([a, b]) => bin_expr(rec, sym_map, *a, *b, quote!(-)),
        FuseExpr::Mul([a, b]) => bin_expr(rec, sym_map, *a, *b, quote!(*)),
        FuseExpr::Div([a, b]) => bin_expr(rec, sym_map, *a, *b, quote!(/)),
        FuseExpr::Neg([a]) => unary_call(rec, sym_map, *a, "neg"),
        FuseExpr::Exp([a]) => unary_call(rec, sym_map, *a, "exp"),
        FuseExpr::Ln([a]) => unary_call(rec, sym_map, *a, "ln"),
        FuseExpr::Sqrt([a]) => unary_call(rec, sym_map, *a, "sqrt"),
        FuseExpr::Abs([a]) => unary_call(rec, sym_map, *a, "abs"),
        FuseExpr::Sin([a]) => unary_call(rec, sym_map, *a, "sin"),
        FuseExpr::Cos([a]) => unary_call(rec, sym_map, *a, "cos"),
        FuseExpr::Tanh([a]) => unary_call(rec, sym_map, *a, "tanh"),
        FuseExpr::Recip([a]) => unary_call(rec, sym_map, *a, "recip"),
        FuseExpr::Pow([a, b]) => pow_expr(rec, sym_map, *a, *b),
    }
}

fn bin_expr(
    rec: &RecExpr<FuseExpr>,
    sym_map: &[(String, Expr)],
    a: Id,
    b: Id,
    op: proc_macro2::TokenStream,
) -> Expr {
    let la = egg_to_expr(rec, a, sym_map);
    let lb = egg_to_expr(rec, b, sym_map);
    syn::parse_quote!((#la #op #lb))
}

fn unary_call(rec: &RecExpr<FuseExpr>, sym_map: &[(String, Expr)], a: Id, method: &str) -> Expr {
    let la = egg_to_expr(rec, a, sym_map);
    let m = Ident::new(method, Span::call_site());
    syn::parse_quote!(#la.#m())
}

fn pow_expr(rec: &RecExpr<FuseExpr>, sym_map: &[(String, Expr)], a: Id, b: Id) -> Expr {
    let la = egg_to_expr(rec, a, sym_map);
    let lb = egg_to_expr(rec, b, sym_map);
    syn::parse_quote!(#la.powf(#lb))
}

pub(crate) fn eqsat_simplify(expr: &Expr) -> Expr {
    let mut rec = RecExpr::default();
    let mut sym_map = Vec::new();
    let _root = expr_to_egg(expr, &mut rec, &mut sym_map);

    let runner = Runner::<FuseExpr, ()>::default()
        .with_iter_limit(8)
        .with_node_limit(5_000)
        .with_expr(&rec)
        .run(&fuse_rules());

    let extractor = egg::Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);

    let best_root = Id::from(best.as_ref().len() - 1);
    egg_to_expr(&best, best_root, &sym_map)
}
