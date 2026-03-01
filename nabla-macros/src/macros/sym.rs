//! `sym!` — Natural math notation for CAS symbolic expressions.
//!
//! Parses mathematical expressions and generates `Expr::*` construction code.
//!
//! # Examples
//!
//! ```rust,ignore
//! let f = sym!(sin(x^2));
//! let g = sym!(x * y + cos(x));
//! let h = sym!(exp(x) / (1 + x^2));
//! ```

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote_spanned;
use syn::{
    Ident, LitFloat, LitInt, Result, Token,
    parse::{Parse, ParseStream},
};


const KNOWN_FNS: &[&str] = &[
    "sin", "cos", "exp", "ln", "tanh", "sqrt", "abs",
    "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh",
];

fn is_known_fn(name: &str) -> bool {
    KNOWN_FNS.contains(&name)
}


enum SymExpr {
    /// Named variable: `x` → `Expr::var("x")`.
    Var(Ident),
    /// Numeric literal: `2.0` or `2` → `Expr::lit(2.0)`.
    Lit(f64, Span),
    /// Unary negation: `-e`.
    Neg(Box<SymExpr>, Span),
    /// Binary addition: `a + b`.
    Add(Box<SymExpr>, Box<SymExpr>, Span),
    /// Binary subtraction: `a - b`.
    Sub(Box<SymExpr>, Box<SymExpr>, Span),
    /// Binary multiplication: `a * b`.
    Mul(Box<SymExpr>, Box<SymExpr>, Span),
    /// Binary division: `a / b`.
    Div(Box<SymExpr>, Box<SymExpr>, Span),
    /// Power: `a ^ b`.
    Pow(Box<SymExpr>, Box<SymExpr>, Span),
    /// Function call: `sin(e)`, `cos(e)`, etc.
    Fn(String, Box<SymExpr>, Span),
}


impl SymExpr {
    /// Lower the AST into a `TokenStream2` that constructs `nabla::cas::Expr`.
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            Self::Var(ident) => {
                let name = ident.to_string();
                let sp = ident.span();
                quote_spanned! {sp=> nabla::cas::Expr::var(#name) }
            }
            Self::Lit(val, sp) => {
                let sp = *sp;
                quote_spanned! {sp=> nabla::cas::Expr::lit(#val) }
            }
            Self::Neg(inner, sp) => {
                let sp = *sp;
                let inner_ts = inner.to_tokens();
                quote_spanned! {sp=> -&#inner_ts }
            }
            Self::Add(lhs, rhs, sp) => {
                let sp = *sp;
                let l = lhs.to_tokens();
                let r = rhs.to_tokens();
                quote_spanned! {sp=> &(#l) + &(#r) }
            }
            Self::Sub(lhs, rhs, sp) => {
                let sp = *sp;
                let l = lhs.to_tokens();
                let r = rhs.to_tokens();
                quote_spanned! {sp=> &(#l) - &(#r) }
            }
            Self::Mul(lhs, rhs, sp) => {
                let sp = *sp;
                let l = lhs.to_tokens();
                let r = rhs.to_tokens();
                quote_spanned! {sp=> &(#l) * &(#r) }
            }
            Self::Div(lhs, rhs, sp) => {
                let sp = *sp;
                let l = lhs.to_tokens();
                let r = rhs.to_tokens();
                quote_spanned! {sp=> &(#l) / &(#r) }
            }
            Self::Pow(base, exp, sp) => {
                let sp = *sp;
                let b = base.to_tokens();
                let e = exp.to_tokens();
                quote_spanned! {sp=> nabla::cas::Expr::pow(&(#b), &(#e)) }
            }
            Self::Fn(name, arg, sp) => {
                let sp = *sp;
                let a = arg.to_tokens();
                let fn_ident = Ident::new(name, sp);
                quote_spanned! {sp=> nabla::cas::Expr::#fn_ident(&(#a)) }
            }
        }
    }
}


#[derive(Clone, Copy)]
struct Bp(u8);

#[derive(Clone, Copy)]
enum InfixOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl InfixOp {
    /// Left and right binding power.
    /// Right-associative ops have right bp < left bp.
    fn bp(self) -> (Bp, Bp) {
        match self {
            Self::Add | Self::Sub => (Bp(1), Bp(2)),
            Self::Mul | Self::Div => (Bp(3), Bp(4)),
            // Right-associative: right bp == left bp (so right recurse at same level).
            Self::Pow => (Bp(6), Bp(5)),
        }
    }
}

fn peek_infix(input: ParseStream<'_>) -> Option<InfixOp> {
    if input.peek(Token![+]) {
        Some(InfixOp::Add)
    } else if input.peek(Token![-]) {
        Some(InfixOp::Sub)
    } else if input.peek(Token![*]) {
        Some(InfixOp::Mul)
    } else if input.peek(Token![/]) {
        Some(InfixOp::Div)
    } else if input.peek(Token![^]) {
        Some(InfixOp::Pow)
    } else {
        None
    }
}

fn consume_infix(input: ParseStream<'_>, op: InfixOp) -> Result<Span> {
    match op {
        InfixOp::Add => input.parse::<Token![+]>().map(|t| t.span),
        InfixOp::Sub => input.parse::<Token![-]>().map(|t| t.span),
        InfixOp::Mul => input.parse::<Token![*]>().map(|t| t.span),
        InfixOp::Div => input.parse::<Token![/]>().map(|t| t.span),
        InfixOp::Pow => input.parse::<Token![^]>().map(|t| t.span),
    }
}

fn make_binop(op: InfixOp, lhs: SymExpr, rhs: SymExpr, sp: Span) -> SymExpr {
    let (l, r) = (Box::new(lhs), Box::new(rhs));
    match op {
        InfixOp::Add => SymExpr::Add(l, r, sp),
        InfixOp::Sub => SymExpr::Sub(l, r, sp),
        InfixOp::Mul => SymExpr::Mul(l, r, sp),
        InfixOp::Div => SymExpr::Div(l, r, sp),
        InfixOp::Pow => SymExpr::Pow(l, r, sp),
    }
}

fn parse_expr(input: ParseStream<'_>, min_bp: Bp) -> Result<SymExpr> {
    // Parse prefix / atom.
    let mut lhs = parse_prefix(input)?;

    // Parse infix operators using Pratt algorithm.
    while let Some(op) = peek_infix(input) {
        let (l_bp, r_bp) = op.bp();
        if l_bp.0 < min_bp.0 {
            break;
        }
        let sp = consume_infix(input, op)?;
        let rhs = parse_expr(input, r_bp)?;
        lhs = make_binop(op, lhs, rhs, sp);
    }

    Ok(lhs)
}

fn parse_prefix(input: ParseStream<'_>) -> Result<SymExpr> {
    // Unary minus.
    if input.peek(Token![-]) {
        let minus: Token![-] = input.parse()?;
        let sp = minus.span;
        // Unary minus binds tighter than +/- but looser than * / ^.
        // BP 5 is between Mul(3,4) and Pow(6,5), matching standard math precedence.
        let inner = parse_prefix(input)?;
        return Ok(SymExpr::Neg(Box::new(inner), sp));
    }

    // Parenthesized group.
    if input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        return parse_expr(&content, Bp(0));
    }

    // Identifier: variable or function call.
    if input.peek(Ident) {
        let ident: Ident = input.parse()?;
        let name = ident.to_string();

        // Function call: `sin(e)`.
        if is_known_fn(&name) {
            if !input.peek(syn::token::Paren) {
                return Err(syn::Error::new(
                    ident.span(),
                    format!("expected `(` after function `{name}`"),
                ));
            }
            let content;
            syn::parenthesized!(content in input);
            let arg = parse_expr(&content, Bp(0))?;
            if !content.is_empty() {
                return Err(syn::Error::new(
                    content.span(),
                    format!("function `{name}` takes exactly one argument"),
                ));
            }
            return Ok(SymExpr::Fn(name, Box::new(arg), ident.span()));
        }

        // Variable.
        return Ok(SymExpr::Var(ident));
    }

    // Float literal.
    if input.peek(LitFloat) {
        let lit: LitFloat = input.parse()?;
        let val: f64 = lit.base10_parse().map_err(|e| {
            syn::Error::new(lit.span(), format!("invalid float literal: {e}"))
        })?;
        return Ok(SymExpr::Lit(val, lit.span()));
    }

    // Integer literal.
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let val: u64 = lit.base10_parse().map_err(|e| {
            syn::Error::new(lit.span(), format!("invalid integer literal: {e}"))
        })?;
        #[allow(clippy::cast_precision_loss)]
        let fval = val as f64;
        return Ok(SymExpr::Lit(fval, lit.span()));
    }

    Err(syn::Error::new(
        input.span(),
        "expected variable, number, function call, or parenthesized expression",
    ))
}


struct SymInput {
    expr: SymExpr,
}

impl Parse for SymInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let expr = parse_expr(input, Bp(0))?;
        if !input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "unexpected tokens after expression",
            ));
        }
        Ok(Self { expr })
    }
}


pub(crate) fn sym_impl(input: TokenStream2) -> Result<TokenStream2> {
    let parsed: SymInput = syn::parse2(input)?;
    Ok(parsed.expr.to_tokens())
}
