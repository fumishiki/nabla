//! Stencil/convolution proc macro — Julia `@tullio` offset-indexing equivalent.
//!
//! `stencil!(out[i,j] = -4.0*a[i,j] + a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1])`
//!
//! Generates bounds-checked interior iteration over a 2-D tensor.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{Error, Expr, Ident, Result, Token, parse::ParseStream};

// ── AST types ────────────────────────────────────────────────────────────────

/// A single index expression: `i`, `i+1`, `i-1`, `j`, etc.
#[derive(Clone)]
pub(crate) enum IdxExpr {
    /// Plain index variable, e.g. `i`
    Plain(String),
    /// Index variable + constant offset, e.g. `i+1` or `j-2`
    Offset(String, i64),
}

impl IdxExpr {
    fn var_name(&self) -> &str {
        match self {
            IdxExpr::Plain(v) | IdxExpr::Offset(v, _) => v,
        }
    }
    fn offset(&self) -> i64 {
        match self {
            IdxExpr::Plain(_) => 0,
            IdxExpr::Offset(_, o) => *o,
        }
    }
}

/// A tensor access: `a[i-1, j+2]`
struct TensorAccess {
    name: Ident,
    indices: Vec<IdxExpr>,
}

/// An RHS term: `coeff * tensor_access` or just `tensor_access`
struct RhsTerm {
    coeff: Option<Expr>,
    negate: bool,
    access: TensorAccess,
}

/// Full stencil: `out[i,j] = sum_of_terms`
pub(crate) struct StencilInput {
    out_name: Ident,
    out_indices: Vec<Ident>,
    terms: Vec<RhsTerm>,
}

// ── Parser ───────────────────────────────────────────────────────────────────

fn parse_idx_expr(input: ParseStream<'_>) -> Result<IdxExpr> {
    let var: Ident = input.parse()?;
    if input.peek(Token![+]) {
        input.parse::<Token![+]>()?;
        let lit: syn::LitInt = input.parse()?;
        Ok(IdxExpr::Offset(var.to_string(), lit.base10_parse()?))
    } else if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        let lit: syn::LitInt = input.parse()?;
        let v: i64 = lit.base10_parse()?;
        Ok(IdxExpr::Offset(var.to_string(), -v))
    } else {
        Ok(IdxExpr::Plain(var.to_string()))
    }
}

fn parse_tensor_access(input: ParseStream<'_>) -> Result<TensorAccess> {
    let name: Ident = input.parse()?;
    let inner;
    syn::bracketed!(inner in input);
    let mut indices = Vec::new();
    loop {
        indices.push(parse_idx_expr(&inner)?);
        if inner.is_empty() {
            break;
        }
        inner.parse::<Token![,]>()?;
    }
    Ok(TensorAccess { name, indices })
}

impl syn::parse::Parse for StencilInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        // out[i, j] = ...
        let out_name: Ident = input.parse()?;
        let inner;
        syn::bracketed!(inner in input);
        let out_indices: syn::punctuated::Punctuated<Ident, Token![,]> =
            inner.parse_terminated(Ident::parse, Token![,])?;
        let out_indices: Vec<Ident> = out_indices.into_iter().collect();

        input.parse::<Token![=]>()?;

        // Parse RHS terms: optional coeff * access, separated by + or -
        let mut terms = Vec::new();
        let mut leading_negate = false;

        // Check for leading minus
        if input.peek(Token![-]) {
            input.parse::<Token![-]>()?;
            leading_negate = true;
        }

        // First term
        let first_term = parse_rhs_term(input, leading_negate)?;
        terms.push(first_term);

        // Subsequent terms
        while !input.is_empty() {
            let negate = if input.peek(Token![+]) {
                input.parse::<Token![+]>()?;
                false
            } else if input.peek(Token![-]) {
                input.parse::<Token![-]>()?;
                true
            } else {
                break;
            };
            terms.push(parse_rhs_term(input, negate)?);
        }

        Ok(StencilInput {
            out_name,
            out_indices,
            terms,
        })
    }
}

fn parse_rhs_term(input: ParseStream<'_>, negate: bool) -> Result<RhsTerm> {
    // Try: literal_coeff * tensor_access
    let fork = input.fork();
    if let Ok(lit) = fork.parse::<syn::Lit>() {
        if fork.peek(Token![*]) {
            // coeff * access — parse the literal and the `*`
            let coeff_lit: syn::Lit = input.parse()?;
            let coeff_expr: Expr = Expr::Lit(syn::ExprLit {
                attrs: vec![],
                lit: coeff_lit,
            });
            input.parse::<Token![*]>()?;
            let access = parse_tensor_access(input)?;
            return Ok(RhsTerm {
                coeff: Some(coeff_expr),
                negate,
                access,
            });
        }
        let _ = lit;
    }
    // Just tensor_access
    let access = parse_tensor_access(input)?;
    Ok(RhsTerm {
        coeff: None,
        negate,
        access,
    })
}

// ── Codegen ──────────────────────────────────────────────────────────────────

pub(crate) fn stencil_impl(input: TokenStream2) -> Result<TokenStream2> {
    let stencil: StencilInput = syn::parse2(input)?;

    if stencil.out_indices.len() != 2 {
        return Err(Error::new(
            Span::call_site(),
            "stencil! currently supports 2-D stencils only",
        ));
    }

    // Determine index variable names
    let idx_i = &stencil.out_indices[0];
    let idx_j = &stencil.out_indices[1];

    // Compute min/max offsets for each dimension
    let mut min_off_i: i64 = 0;
    let mut max_off_i: i64 = 0;
    let mut min_off_j: i64 = 0;
    let mut max_off_j: i64 = 0;

    let idx_i_str = idx_i.to_string();
    let idx_j_str = idx_j.to_string();

    for term in &stencil.terms {
        for idx in &term.access.indices {
            let var = idx.var_name();
            let off = idx.offset();
            if var == idx_i_str {
                min_off_i = min_off_i.min(off);
                max_off_i = max_off_i.max(off);
            } else if var == idx_j_str {
                min_off_j = min_off_j.min(off);
                max_off_j = max_off_j.max(off);
            }
        }
    }

    // Interior bounds: iterate from -min_off_i to nrows-max_off_i
    let start_i = (-min_off_i) as usize;
    let start_j = (-min_off_j) as usize;
    let end_off_i = max_off_i as usize;
    let end_off_j = max_off_j as usize;

    // First input tensor for shape reference
    let first_tensor = &stencil.terms[0].access.name;
    let out_name = &stencil.out_name;

    // Generate the expression for each term
    let term_tokens: Vec<TokenStream2> = stencil
        .terms
        .iter()
        .enumerate()
        .map(|(t_idx, term)| {
            let access_name = &term.access.name;
            let idx_exprs: Vec<TokenStream2> = term
                .access
                .indices
                .iter()
                .zip([idx_i, idx_j].iter())
                .map(|(idx, loop_var)| {
                    let off = idx.offset();
                    if off == 0 {
                        quote! { #loop_var }
                    } else if off > 0 {
                        let off_u = off as usize;
                        quote! { (#loop_var + #off_u) }
                    } else {
                        let off_u = (-off) as usize;
                        quote! { (#loop_var - #off_u) }
                    }
                })
                .collect();

            let get_expr = quote! { #access_name.get(#(#idx_exprs),*) };

            let valued = if let Some(ref coeff) = term.coeff {
                quote! { (#coeff) * #get_expr }
            } else {
                get_expr
            };

            if term.negate {
                quote! { -(#valued) }
            } else if t_idx == 0 {
                valued
            } else {
                quote! { + (#valued) }
            }
        })
        .collect();

    Ok(quote! {{
        let (__rows, __cols) = #first_tensor.shape();
        let mut #out_name = nabla::tensor::Tensor::zeros(__rows, __cols);
        for #idx_i in #start_i..(__rows - #end_off_i) {
            for #idx_j in #start_j..(__cols - #end_off_j) {
                #out_name.set(#idx_i, #idx_j, #(#term_tokens)*);
            }
        }
        #out_name
    }})
}
