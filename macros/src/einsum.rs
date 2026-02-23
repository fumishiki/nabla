// einsum.rs — Einstein summation proc macro implementation.
//
// Parses `output[free_indices] = term1 * term2 * ...` and generates
// optimised Rust code:
//   - GEMM pattern  → `Tensor::matmul_into`  (faer BLAS / GPU tiled)
//   - GEMV pattern  → `Tensor::matmul_into`  (M×1 column vector)
//   - Hadamard      → `mul_elem`
//   - Trace         → diagonal sum loop
//   - Outer         → `from_fn` outer product
//   - N-D Fallback  → `Tensor::from_fn` + general accumulation loops
//
// All compile errors use `syn::Error::new_spanned` for precise diagnostics.

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
    /// Index names in order (N-D: no limit on count).
    indices: Vec<Ident>,
    /// Span covering the `[i, j, ...]` brackets (for precise error reporting).
    #[allow(dead_code)]
    bracket_span: Option<Span>,
}

/// Parsed representation of a full einsum expression.
pub(crate) struct EinsumInput {
    /// The LHS output name (e.g. `c` in `c[i,j] = ...`).
    output_name: Ident,
    /// Free indices on the LHS (in declaration order).
    output_indices: Vec<Ident>,
    /// The `=` token (kept for spanned errors).
    eq_token: syn::Token![=],
    /// RHS tensor references in order.
    rhs_terms: Vec<IndexedTensor>,
}

/// Compile-time classification of the contraction pattern.
#[derive(Debug)]
enum ContractionKind {
    /// C = A B  (standard layout: a[i,k] * b[k,j])
    Gemm,
    /// C = op(A) op(B) where op is identity or transpose
    GemmTransposed { transpose_a: bool, transpose_b: bool },
    /// y = A x  (matrix-vector product)
    Gemv { mat_first: bool },
    /// C = A ∘ B  (element-wise / Hadamard product, no contraction)
    Hadamard,
    /// tr(A) — scalar output from diagonal: s = a[i,i]
    Trace,
    /// Outer product: c[i,j] = a[i] * b[j] (no contraction, vector × vector)
    Outer,
    /// General loop-based contraction (fallback)
    Fallback,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

impl Parse for EinsumInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let output_name: Ident = input.parse()?;

        let output_indices = if input.peek(syn::token::Bracket) {
            let inner;
            syn::bracketed!(inner in input);
            parse_index_list(&inner)?
        } else {
            vec![]
        };

        let eq_token: syn::Token![=] = input.parse()?;

        let mut rhs_terms = vec![parse_indexed_tensor(input)?];
        while input.peek(syn::Token![*]) {
            input.parse::<syn::Token![*]>()?;
            rhs_terms.push(parse_indexed_tensor(input)?);
        }

        Ok(EinsumInput {
            output_name,
            output_indices,
            eq_token,
            rhs_terms,
        })
    }
}

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

fn parse_indexed_tensor(input: ParseStream<'_>) -> Result<IndexedTensor> {
    let name: Ident = input.parse()?;
    let (indices, bracket_span) = if input.peek(syn::token::Bracket) {
        let inner;
        let bracket = syn::bracketed!(inner in input);
        let indices = parse_index_list(&inner)?;
        (indices, Some(bracket.span.join()))
    } else {
        (vec![], None)
    };
    Ok(IndexedTensor {
        name,
        indices,
        bracket_span,
    })
}

// ---------------------------------------------------------------------------
// Validation (I3c: spanned errors)
// ---------------------------------------------------------------------------

fn validate(input: &EinsumInput) -> Result<()> {
    // Check: at least one RHS term.
    if input.rhs_terms.is_empty() {
        return Err(Error::new_spanned(
            input.eq_token,
            "einsum!: at least one tensor term is required on the right-hand side",
        ));
    }

    // Check: no duplicate indices on LHS (except trace pattern with empty LHS).
    {
        let mut seen: HashSet<String> = HashSet::new();
        for idx in &input.output_indices {
            if !seen.insert(idx.to_string()) {
                return Err(Error::new_spanned(
                    idx,
                    format!("einsum!: index `{idx}` appears twice on the LHS"),
                ));
            }
        }
    }

    // Collect free names for later checks.
    let free_names: HashSet<String> =
        input.output_indices.iter().map(Ident::to_string).collect();

    // Check: contraction indices must appear in at least 2 terms.
    // (An index on the RHS only, appearing in just 1 term, is likely a typo.)
    {
        let mut idx_term_count: Vec<(String, usize, &Ident)> = vec![];
        for term in &input.rhs_terms {
            for idx in &term.indices {
                let s = idx.to_string();
                if !free_names.contains(&s) {
                    if let Some(entry) = idx_term_count.iter_mut().find(|(n, _, _)| n == &s) {
                        entry.1 += 1;
                    } else {
                        idx_term_count.push((s, 1, idx));
                    }
                }
            }
        }
        for (name, count, ident) in &idx_term_count {
            // Allow trace pattern: same index twice in one term (e.g. a[i,i])
            if *count == 1 {
                // Check if it appears twice within a single term (trace).
                let is_trace = input.rhs_terms.iter().any(|t| {
                    t.indices.iter().filter(|i| i.to_string() == *name).count() >= 2
                });
                if !is_trace {
                    return Err(Error::new_spanned(
                        ident,
                        format!(
                            "einsum!: index `{name}` appears only once on the RHS; \
                             contraction indices must appear in at least 2 terms \
                             (did you mean to include it on the LHS?)"
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Contraction classifier (I3a)
// ---------------------------------------------------------------------------

fn classify(input: &EinsumInput) -> ContractionKind {
    let free = &input.output_indices;
    let rhs = &input.rhs_terms;

    let free_names: HashSet<String> = free.iter().map(Ident::to_string).collect();
    let mut contraction: Vec<String> = vec![];
    let mut seen: HashSet<String> = HashSet::new();
    for term in rhs {
        for idx in &term.indices {
            let s = idx.to_string();
            if !free_names.contains(&s) && seen.insert(s.clone()) {
                contraction.push(s);
            }
        }
    }

    // Trace: scalar output, single term, same index appears twice (e.g. s = a[i,i])
    if free.is_empty() && rhs.len() == 1 && contraction.is_empty() {
        let term = &rhs[0];
        if term.indices.len() == 2 && term.indices[0] == term.indices[1] {
            return ContractionKind::Trace;
        }
    }

    // GEMM: 2 terms × 2 indices each, 2 free, 1 contraction
    if rhs.len() == 2
        && rhs[0].indices.len() == 2
        && rhs[1].indices.len() == 2
        && free.len() == 2
        && contraction.len() == 1
    {
        let k = &contraction[0];
        let a0 = rhs[0].indices[0].to_string();
        let a1 = rhs[0].indices[1].to_string();
        let b0 = rhs[1].indices[0].to_string();
        let b1 = rhs[1].indices[1].to_string();

        // Standard: a[i,k] * b[k,j]
        if a1 == *k && b0 == *k {
            return ContractionKind::Gemm;
        }

        // Determine transpose flags.
        let transpose_a = a0 == *k;
        let transpose_b = b1 == *k;

        if transpose_a || transpose_b {
            return ContractionKind::GemmTransposed {
                transpose_a,
                transpose_b,
            };
        }
    }

    // GEMV: matrix × vector (or vector × matrix)
    if rhs.len() == 2 && free.len() == 1 && contraction.len() == 1 {
        if rhs[0].indices.len() == 2 && rhs[1].indices.len() == 1 {
            return ContractionKind::Gemv { mat_first: true };
        }
        if rhs[0].indices.len() == 1 && rhs[1].indices.len() == 2 {
            return ContractionKind::Gemv { mat_first: false };
        }
    }

    // Outer product: 2 terms, each with 1 index, no contraction, 2 free indices
    if rhs.len() == 2
        && contraction.is_empty()
        && free.len() == 2
        && rhs[0].indices.len() == 1
        && rhs[1].indices.len() == 1
    {
        return ContractionKind::Outer;
    }

    // Hadamard: 2 terms with matching dimensionality, no contraction, all indices are free
    if rhs.len() == 2 && contraction.is_empty() && !free.is_empty() {
        let all_rhs_indices_are_free = rhs.iter().all(|t| {
            t.indices.iter().all(|idx| free_names.contains(&idx.to_string()))
        });
        if all_rhs_indices_are_free
            && rhs[0].indices.len() == rhs[1].indices.len()
            && rhs[0].indices.len() == free.len()
        {
            return ContractionKind::Hadamard;
        }
    }

    ContractionKind::Fallback
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

pub(crate) fn einsum_impl(input: TokenStream2) -> Result<TokenStream2> {
    let parsed: EinsumInput = syn::parse2(input)?;
    validate(&parsed)?;
    codegen_einsum(&parsed)
}

fn codegen_einsum(input: &EinsumInput) -> Result<TokenStream2> {
    let kind = classify(input);

    match kind {
        ContractionKind::Gemm => codegen_gemm(input, false, false),
        ContractionKind::GemmTransposed {
            transpose_a,
            transpose_b,
        } => codegen_gemm(input, transpose_a, transpose_b),
        ContractionKind::Gemv { mat_first } => codegen_gemv(input, mat_first),
        ContractionKind::Hadamard => codegen_hadamard(input),
        ContractionKind::Trace => codegen_trace(input),
        ContractionKind::Outer => codegen_outer(input),
        ContractionKind::Fallback => codegen_fallback(input),
    }
}

/// GEMM: emit `Tensor::matmul_into` with optional transposes.
fn codegen_gemm(
    input: &EinsumInput,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let b_name = &input.rhs_terms[1].name;
    let a_bind = Ident::new(&format!("__{a_name}"), a_name.span());
    let b_bind = Ident::new(&format!("__{b_name}"), b_name.span());

    let a_expr = if transpose_a {
        quote! { #a_bind.t() }
    } else {
        quote! { #a_bind.clone() }
    };
    let b_expr = if transpose_b {
        quote! { #b_bind.t() }
    } else {
        quote! { #b_bind.clone() }
    };

    Ok(quote! {
        {
            let #a_bind = &#a_name;
            let #b_bind = &#b_name;
            let __a_op = #a_expr;
            let __b_op = #b_expr;
            let __m = __a_op.nrows();
            let __n = __b_op.ncols();
            let mut __out = nabla::tensor::Tensor::from_fn(__m, __n, |_, _| {
                faer_traits::math_utils::zero::<_>()
            });
            nabla::tensor::Tensor::matmul_into(&mut __out, &__a_op, &__b_op);
            __out
        }
    })
}

/// GEMV: matrix × vector → column vector, via matmul_into with Mx1 / 1xN.
fn codegen_gemv(input: &EinsumInput, mat_first: bool) -> Result<TokenStream2> {
    let (mat_term, vec_term) = if mat_first {
        (&input.rhs_terms[0], &input.rhs_terms[1])
    } else {
        (&input.rhs_terms[1], &input.rhs_terms[0])
    };

    let mat_name = &mat_term.name;
    let vec_name = &vec_term.name;
    let mat_bind = Ident::new(&format!("__{mat_name}"), mat_name.span());
    let vec_bind = Ident::new(&format!("__{vec_name}"), vec_name.span());

    // Check if matrix needs transpose: free idx should be at row position.
    let transpose_mat = mat_term.indices[0] != input.output_indices[0];

    let mat_expr = if transpose_mat {
        quote! { #mat_bind.t() }
    } else {
        quote! { #mat_bind.clone() }
    };

    Ok(quote! {
        {
            let #mat_bind = &#mat_name;
            let #vec_bind = &#vec_name;
            let __mat = #mat_expr;
            // Vector is already a column vector (nrows × 1).
            let __m = __mat.nrows();
            let mut __out = nabla::tensor::Tensor::from_fn(__m, 1, |_, _| {
                faer_traits::math_utils::zero::<_>()
            });
            nabla::tensor::Tensor::matmul_into(&mut __out, &__mat, #vec_bind);
            __out
        }
    })
}

/// Hadamard: element-wise product via `mul_elem`.
fn codegen_hadamard(input: &EinsumInput) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let b_name = &input.rhs_terms[1].name;

    // Check if index order matches (both same order → direct mul_elem).
    // If b has reversed indices, we need to transpose b.
    let a_indices: Vec<String> = input.rhs_terms[0].indices.iter().map(Ident::to_string).collect();
    let b_indices: Vec<String> = input.rhs_terms[1].indices.iter().map(Ident::to_string).collect();

    if a_indices == b_indices {
        Ok(quote! {
            {
                (#a_name).mul_elem(&#b_name)
            }
        })
    } else {
        // b has reversed index order → transpose b
        Ok(quote! {
            {
                (#a_name).mul_elem(&(#b_name).t())
            }
        })
    }
}

/// Trace: diagonal sum → scalar output (e.g. `s = a[i,i]`).
fn codegen_trace(input: &EinsumInput) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let a_bind = Ident::new(&format!("__{a_name}"), a_name.span());

    Ok(quote! {
        {
            let #a_bind = &#a_name;
            let __n = #a_bind.nrows().min(#a_bind.ncols());
            let mut __acc = faer_traits::math_utils::zero::<_>();
            for __diag_idx in 0..__n {
                __acc = faer_traits::math_utils::add(
                    &__acc,
                    &#a_bind.get(__diag_idx, __diag_idx),
                );
            }
            __acc
        }
    })
}

/// Outer product: c[i,j] = a[i] * b[j] → `from_fn` with direct element access.
fn codegen_outer(input: &EinsumInput) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let b_name = &input.rhs_terms[1].name;
    let a_bind = Ident::new(&format!("__{a_name}"), a_name.span());
    let b_bind = Ident::new(&format!("__{b_name}"), b_name.span());

    // Determine which free index comes from which term.
    let free0 = input.output_indices[0].to_string();
    let a_idx = input.rhs_terms[0].indices[0].to_string();

    let (row_src, col_src) = if a_idx == free0 {
        // a provides rows, b provides cols: c[i,j] = a[i] * b[j]
        (quote! { #a_bind }, quote! { #b_bind })
    } else {
        // b provides rows, a provides cols: c[i,j] = a[j] * b[i]
        (quote! { #b_bind }, quote! { #a_bind })
    };

    Ok(quote! {
        {
            let #a_bind = &#a_name;
            let #b_bind = &#b_name;
            let __m = #row_src.nrows();
            let __n = #col_src.nrows();
            nabla::tensor::Tensor::from_fn(__m, __n, |__i, __j| {
                faer_traits::math_utils::mul(
                    &#row_src.get(__i, 0),
                    &#col_src.get(__j, 0),
                )
            })
        }
    })
}

/// Fallback: general loop-based contraction (supports N-D indices).
fn codegen_fallback(input: &EinsumInput) -> Result<TokenStream2> {
    let free_indices = &input.output_indices;
    let rhs_terms = &input.rhs_terms;

    let all_rhs_indices: Vec<&Ident> = rhs_terms.iter().flat_map(|t| t.indices.iter()).collect();

    let free_names: Vec<String> = free_indices.iter().map(Ident::to_string).collect();
    let mut contraction_indices: Vec<&Ident> = vec![];
    let mut seen_contraction: HashSet<String> = HashSet::new();
    for idx in &all_rhs_indices {
        let s = idx.to_string();
        if !free_names.contains(&s) && seen_contraction.insert(s) {
            contraction_indices.push(idx);
        }
    }

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

    let product_expr = build_product_expr(rhs_terms)?;
    let acc_body = build_accumulator(&contraction_indices, rhs_terms, &product_expr)?;

    let result = if free_indices.is_empty() {
        // Scalar output
        quote! {
            {
                #(#tensor_bindings)*
                let mut __acc = faer_traits::math_utils::zero::<_>();
                #acc_body
                __acc
            }
        }
    } else if free_indices.len() == 1 {
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
        let i_idx = &free_indices[0];
        let j_idx = &free_indices[1];
        let nrows_expr = dim_expr_for_index(i_idx, rhs_terms, 0)?;
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
        // N-D output (>2 free indices): generate nested loops over batch dims
        // with inner 2D `from_fn`. For now, fall back to a flat loop approach
        // that constructs the 2D output from the last two free indices.
        return Err(Error::new_spanned(
            &input.output_name,
            "einsum!: N-D output (>2 free indices) requires N-D tensor support \
             which is not yet available; consider restructuring as batch loops \
             over 2-D operations",
        ));
    };

    Ok(result)
}

// ---------------------------------------------------------------------------
// Fallback helpers (accumulator / product / access / dim)
// ---------------------------------------------------------------------------

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

fn build_product_expr(rhs_terms: &[IndexedTensor]) -> Result<TokenStream2> {
    if rhs_terms.is_empty() {
        return Err(Error::new(
            proc_macro2::Span::call_site(),
            "einsum! requires at least one RHS term",
        ));
    }

    let accesses: Vec<TokenStream2> = rhs_terms
        .iter()
        .map(tensor_element_access)
        .collect::<Result<Vec<_>>>()?;

    let mut expr = accesses[0].clone();
    for access in &accesses[1..] {
        expr = quote! {
            faer_traits::math_utils::mul(&#expr, &#access)
        };
    }

    Ok(expr)
}

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
        _ => {
            // N-D: currently Tensor is 2-D, so N>2 indices require future N-D support.
            Err(Error::new_spanned(
                &term.name,
                format!(
                    "einsum!: tensor `{}` has {} indices, but only 2-D tensors are \
                     currently supported; N-D tensor support is planned for a future release",
                    term.name,
                    term.indices.len()
                ),
            ))
        }
    }
}

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
                    n => quote! { #binding.dim(#n) },
                });
            }
        }
    }
    dim_expr_for_index_any_pos(idx, rhs_terms)
}

fn dim_expr_for_index_any_pos(idx: &Ident, rhs_terms: &[IndexedTensor]) -> Result<TokenStream2> {
    for term in rhs_terms {
        for (pos, tidx) in term.indices.iter().enumerate() {
            if tidx == idx {
                let binding = Ident::new(&format!("__{}", term.name), term.name.span());
                return Ok(match pos {
                    0 => quote! { #binding.nrows() },
                    1 => quote! { #binding.ncols() },
                    n => quote! { #binding.dim(#n) },
                });
            }
        }
    }
    Err(Error::new_spanned(
        idx,
        format!("einsum!: index `{idx}` not found in any RHS tensor"),
    ))
}
