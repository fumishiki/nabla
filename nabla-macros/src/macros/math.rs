use std::collections::HashSet;

use proc_macro2::TokenStream;
use syn::{Expr, ExprPath};
use quote::quote;

fn transform(expr: Expr, exclude: &HashSet<String>) -> Expr {
    match expr {
        Expr::Path(ref p) if is_simple_ident(p) && !exclude.contains(&ident_name(p)) => {
            syn::parse_quote!(&#expr)
        }
        Expr::Binary(mut b) => {
            *b.left = transform(*b.left, exclude);
            *b.right = transform(*b.right, exclude);
            Expr::Binary(b)
        }
        Expr::MethodCall(mut m) => {
            *m.receiver = transform(*m.receiver, exclude);
            m.args = m.args.into_iter().map(|a| transform(a, exclude)).collect();
            Expr::MethodCall(m)
        }
        Expr::Paren(mut p) => {
            *p.expr = transform(*p.expr, exclude);
            Expr::Paren(p)
        }
        Expr::Unary(mut u) => {
            *u.expr = transform(*u.expr, exclude);
            Expr::Unary(u)
        }
        Expr::Call(mut c) => {
            c.args = c.args.into_iter().map(|a| transform(a, exclude)).collect();
            Expr::Call(c)
        }
        Expr::Closure(mut c) => {
            let mut inner_exclude = exclude.clone();
            for pat in &c.inputs {
                collect_pat_idents(pat, &mut inner_exclude);
            }
            *c.body = transform(*c.body, &inner_exclude);
            Expr::Closure(c)
        }
        Expr::Field(f) => {
            let inner = Expr::Field(f);
            syn::parse_quote!(&#inner)
        }
        Expr::Index(idx) => {
            let inner = Expr::Index(idx);
            syn::parse_quote!(&#inner)
        }
        Expr::Reference(_) => expr,
        _ => expr,
    }
}

fn is_simple_ident(p: &ExprPath) -> bool {
    p.qself.is_none() && p.path.segments.len() == 1 && p.path.leading_colon.is_none()
}

fn ident_name(p: &ExprPath) -> String {
    p.path.segments[0].ident.to_string()
}

fn collect_pat_idents(pat: &syn::Pat, set: &mut HashSet<String>) {
    match pat {
        syn::Pat::Ident(pi) => { set.insert(pi.ident.to_string()); }
        syn::Pat::Type(pt) => collect_pat_idents(&pt.pat, set),
        syn::Pat::Reference(pr) => collect_pat_idents(&pr.pat, set),
        _ => {}
    }
}

/// Auto-borrow simple ident paths in math expressions.
pub fn math_impl(input: TokenStream) -> syn::Result<TokenStream> {
    let expr: Expr = syn::parse2(input)?;
    let result = transform(expr, &HashSet::new());
    Ok(quote!(#result))
}
