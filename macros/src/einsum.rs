// einsum.rs — Einstein summation proc macro implementation.
//
// Parses `output[free_indices] = term1 * term2 * ...` and generates
// a Rust block that computes the contraction using nabla::tensor::Tensor::from_fn
// or a scalar accumulation loop.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::HashSet;
use syn::{
    Error, Ident, Result,
    parse::{Parse, ParseStream},
};

// ---------------------------------------------------------------------------
// AST types
// ---------------------------------------------------------------------------

/// A single tensor reference in the expression: `name[i, j]` or `name[i]`.
struct IndexedTensor {
    name: Ident,
    /// Index names in order (0, 1, or 2 entries for scalars/vectors/matrices).
    indices: Vec<Ident>,
}

/// Parsed representation of a full einsum expression.
///
/// Examples:
///   `c[i,j] = a[i,k] * b[k,j]`  → output indices `[i,j]`, contraction `[k]`
///   `s = a[i,i]`                 → scalar output (no free indices)
pub(crate) struct EinsumInput {
    /// Free indices on the LHS (in declaration order).
    output_indices: Vec<Ident>,
    /// RHS tensor references in order.
    rhs_terms: Vec<IndexedTensor>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

impl Parse for EinsumInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        // Parse the LHS: either `name[i,j]` or bare `name` (output name consumed, not stored).
        let _output_name: Ident = input.parse()?;

        let output_indices = if input.peek(syn::token::Bracket) {
            let inner;
            syn::bracketed!(inner in input);
            parse_index_list(&inner)?
        } else {
            vec![]
        };

        // Consume `=`.
        input.parse::<syn::Token![=]>()?;

        // Parse RHS: one or more `IndexedTensor` separated by `*`.
        let mut rhs_terms = vec![parse_indexed_tensor(input)?];
        while input.peek(syn::Token![*]) {
            input.parse::<syn::Token![*]>()?;
            rhs_terms.push(parse_indexed_tensor(input)?);
        }

        Ok(EinsumInput {
            output_indices,
            rhs_terms,
        })
    }
}

/// Parse a comma-separated list of index identifiers inside already-consumed brackets.
fn parse_index_list(inner: ParseStream<'_>) -> Result<Vec<Ident>> {
    let mut indices = vec![];
    while !inner.is_empty() {
        indices.push(inner.parse::<Ident>()?);
        if inner.peek(syn::Token![,]) {
            inner.parse::<syn::Token![,]>()?;
        }
    }
    Ok(indices)
}

/// Parse `name` or `name[i]` or `name[i,j]`.
fn parse_indexed_tensor(input: ParseStream<'_>) -> Result<IndexedTensor> {
    let name: Ident = input.parse()?;
    let indices = if input.peek(syn::token::Bracket) {
        let inner;
        syn::bracketed!(inner in input);
        parse_index_list(&inner)?
    } else {
        vec![]
    };
    Ok(IndexedTensor { name, indices })
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/// Entry point called from lib.rs.
pub(crate) fn einsum_impl(input: TokenStream2) -> Result<TokenStream2> {
    let parsed: EinsumInput = syn::parse2(input)?;
    codegen_einsum(&parsed)
}

fn codegen_einsum(input: &EinsumInput) -> Result<TokenStream2> {
    let free_indices = &input.output_indices;
    let rhs_terms = &input.rhs_terms;

    // Collect all indices that appear on the RHS.
    let all_rhs_indices: Vec<&Ident> = rhs_terms.iter().flat_map(|t| t.indices.iter()).collect();

    // Contraction indices = appear on RHS but NOT in output free indices.
    // Use string comparison for deduplication; Ident equality ignores span.
    let free_names: Vec<String> = free_indices.iter().map(Ident::to_string).collect();
    let mut contraction_indices: Vec<&Ident> = vec![];
    let mut seen_contraction: HashSet<String> = HashSet::new();
    for idx in &all_rhs_indices {
        let s = idx.to_string();
        if !free_names.contains(&s) && seen_contraction.insert(s) {
            contraction_indices.push(idx);
        }
    }

    // Emit `let __name = &name;` for each unique tensor name.
    let mut seen_names: HashSet<String> = HashSet::new();
    let tensor_bindings: Vec<TokenStream2> = rhs_terms
        .iter()
        .filter_map(|t| {
            if seen_names.insert(t.name.to_string()) {
                let name = &t.name;
                let binding = Ident::new(&format!("__{name}"), name.span());
                Some(quote! { let #binding = &#name; })
            } else {
                None
            }
        })
        .collect();

    // Build the product expression for a single iteration step.
    let product_expr = build_product_expr(rhs_terms)?;

    // Build the accumulator body (inner loops over contraction indices + product).
    let acc_body = build_accumulator(&contraction_indices, rhs_terms, &product_expr)?;

    let result = if free_indices.is_empty() {
        // Scalar output: plain accumulation, no from_fn wrapper.
        // `faer_traits::math_utils::zero::<_>()` infers T from the add expression below.
        quote! {
            {
                #(#tensor_bindings)*
                let mut __acc = faer_traits::math_utils::zero::<_>();
                #acc_body
                __acc
            }
        }
    } else if free_indices.len() == 1 {
        // 1-D output vector (n x 1 column vector).
        let i_idx = &free_indices[0];
        let nrows_expr = dim_expr_for_index(i_idx, rhs_terms, 0)?;
        quote! {
            {
                #(#tensor_bindings)*
                nabla::tensor::Tensor::from_fn(#nrows_expr, 1, |#i_idx, _| {
                    let mut __acc = faer_traits::math_utils::zero::<_>();
                    #acc_body
                    __acc
                })
            }
        }
    } else if free_indices.len() == 2 {
        // 2-D output matrix.
        let i_idx = &free_indices[0];
        let j_idx = &free_indices[1];
        let nrows_expr = dim_expr_for_index(i_idx, rhs_terms, 0)?;
        // Use any-position search so ncols works for cases like b[k,j] where j is at pos 1.
        let ncols_expr = dim_expr_for_index_any_pos(j_idx, rhs_terms)?;
        quote! {
            {
                #(#tensor_bindings)*
                nabla::tensor::Tensor::from_fn(#nrows_expr, #ncols_expr, |#i_idx, #j_idx| {
                    let mut __acc = faer_traits::math_utils::zero::<_>();
                    #acc_body
                    __acc
                })
            }
        }
    } else {
        return Err(Error::new(
            Span::call_site(),
            "einsum! supports at most 2 free indices (2-D output tensors)",
        ));
    };

    Ok(result)
}

/// Build the inner accumulator (loops over contraction indices, then accumulate).
///
/// Uses `faer_traits::math_utils::add` which dispatches through `AddByRef`.
fn build_accumulator(
    contraction_indices: &[&Ident],
    rhs_terms: &[IndexedTensor],
    product_expr: &TokenStream2,
) -> Result<TokenStream2> {
    let step = quote! {
        let __prod = #product_expr;
        __acc = faer_traits::math_utils::add(&__acc, &__prod);
    };

    if contraction_indices.is_empty() {
        return Ok(step);
    }

    // Wrap in loops from innermost to outermost.
    let mut loops = step;
    for &cidx in contraction_indices.iter().rev() {
        let dim = dim_expr_for_index_any_pos(cidx, rhs_terms)?;
        loops = quote! {
            for #cidx in 0..#dim {
                #loops
            }
        };
    }

    Ok(loops)
}

/// Build the multiplication expression for all RHS terms at the current index values.
///
/// Uses `faer_traits::math_utils::mul` which dispatches through `MulByRef`.
fn build_product_expr(rhs_terms: &[IndexedTensor]) -> Result<TokenStream2> {
    if rhs_terms.is_empty() {
        return Err(Error::new(
            Span::call_site(),
            "einsum! requires at least one RHS term",
        ));
    }

    // Build individual element access for each term.
    let accesses: Vec<TokenStream2> = rhs_terms
        .iter()
        .map(tensor_element_access)
        .collect::<Result<Vec<_>>>()?;

    // Chain with math_utils::mul.
    let mut expr = accesses[0].clone();
    for access in &accesses[1..] {
        expr = quote! {
            faer_traits::math_utils::mul(&#expr, &#access)
        };
    }

    Ok(expr)
}

/// Generate element access expression for a single tensor term.
///
/// - 0 indices: just the variable itself (scalar).
/// - 1 index: column-vector convention `__name.get(idx, 0)`.
/// - 2 indices: `__name.get(idx0, idx1)`.
fn tensor_element_access(term: &IndexedTensor) -> Result<TokenStream2> {
    let binding = Ident::new(&format!("__{}", term.name), term.name.span());
    match term.indices.len() {
        0 => Ok(quote! { #binding }),
        1 => {
            let i = &term.indices[0];
            Ok(quote! { #binding.get(#i, 0) })
        }
        2 => {
            let i = &term.indices[0];
            let j = &term.indices[1];
            Ok(quote! { #binding.get(#i, #j) })
        }
        n => Err(Error::new(
            term.name.span(),
            format!("einsum! supports at most 2 indices per tensor, got {n}"),
        )),
    }
}

/// Find the dimension expression for a given index at a preferred position (0=nrows, 1=ncols).
///
/// Searches rhs_terms for the first tensor where `idx` appears at `preferred_pos`.
fn dim_expr_for_index(
    idx: &Ident,
    rhs_terms: &[IndexedTensor],
    preferred_pos: usize,
) -> Result<TokenStream2> {
    for term in rhs_terms {
        for (pos, tidx) in term.indices.iter().enumerate() {
            if tidx == idx && pos == preferred_pos {
                let binding = Ident::new(&format!("__{}", term.name), term.name.span());
                return Ok(match pos {
                    0 => quote! { #binding.nrows() },
                    1 => quote! { #binding.ncols() },
                    _ => unreachable!(),
                });
            }
        }
    }
    // Fall back to any position.
    dim_expr_for_index_any_pos(idx, rhs_terms)
}

/// Find a dimension expression for a given index at any position in any tensor.
fn dim_expr_for_index_any_pos(idx: &Ident, rhs_terms: &[IndexedTensor]) -> Result<TokenStream2> {
    for term in rhs_terms {
        for (pos, tidx) in term.indices.iter().enumerate() {
            if tidx == idx {
                let binding = Ident::new(&format!("__{}", term.name), term.name.span());
                return Ok(match pos {
                    0 => quote! { #binding.nrows() },
                    1 => quote! { #binding.ncols() },
                    _ => unreachable!(),
                });
            }
        }
    }
    Err(Error::new(
        idx.span(),
        format!("einsum! index `{idx}` not found in any RHS tensor"),
    ))
}
