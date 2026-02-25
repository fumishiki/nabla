// einsum.rs — Einstein summation proc macro implementation.
//
// Parses `output[free_indices] = term1 * term2 * ...` and generates
// optimised Rust code:
//   - GEMM pattern  → `Tensor::matmul_into`  (nabla CPU / GPU tiled)
//   - GEMV pattern  → `Tensor::matmul_into`  (M×1 column vector)
//   - Hadamard      → `emul`
//   - Trace         → diagonal sum loop
//   - Outer         → `from_fn` outer product
//   - N-D Fallback  → `Tensor::from_fn` + general accumulation loops
//
// All compile errors use `syn::Error::new_spanned` for precise diagnostics.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::{HashMap, HashSet};
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
    _output_name: Ident,
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
    GemmTransposed {
        transpose_a: bool,
        transpose_b: bool,
    },
    /// y = A x  (matrix-vector product)
    Gemv { mat_first: bool },
    /// C = A ∘ B  (element-wise / Hadamard product, no contraction)
    Hadamard,
    /// tr(A) — scalar output from diagonal: s = a[i,i]
    Trace,
    /// Outer product: c[i,j] = a[i] * b[j] (no contraction, vector × vector)
    Outer,
    /// Batch GEMM: c[b..,i,j] = a[b..,i,k] * m[b..,k,j]
    /// batch_count = number of batch dimensions (leading shared dims)
    BatchGemm { batch_count: usize },
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
            _output_name: output_name,
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
    let free_names: HashSet<String> = input.output_indices.iter().map(Ident::to_string).collect();

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
                let is_trace = input
                    .rhs_terms
                    .iter()
                    .any(|t| t.indices.iter().filter(|i| i.to_string() == *name).count() >= 2);
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
// Canonicalization (alpha-equivalence + term ordering)
// ---------------------------------------------------------------------------

/// Rename indices to canonical names and sort 2-term RHS by tensor name.
///
/// Operand sort FIRST (commutativity), THEN index rename (alpha-equivalence).
/// This ensures that swapping operands + renaming indices produces the same
/// canonical form regardless of the original spelling.
///
/// Output indices → i, j, k, l, m, n (in LHS order).
/// Contraction indices → p, q, r, s (in first-appearance order on sorted RHS).
fn canonicalize(input: &mut EinsumInput) {
    // Sort 2-term RHS alphabetically by tensor name BEFORE renaming,
    // so first-appearance order of contraction indices is deterministic.
    if input.rhs_terms.len() == 2 && input.rhs_terms[0].name > input.rhs_terms[1].name {
        input.rhs_terms.swap(0, 1);
    }

    const FREE_NAMES: &[&str] = &["i", "j", "k", "l", "m", "n"];
    const CONTRACTION_NAMES: &[&str] = &["p", "q", "r", "s"];

    let mut rename: HashMap<String, Ident> = HashMap::new();

    // Assign canonical names to output (free) indices
    for (pos, idx) in input.output_indices.iter().enumerate() {
        let old = idx.to_string();
        if pos < FREE_NAMES.len() && !rename.contains_key(&old) {
            rename.insert(old, Ident::new(FREE_NAMES[pos], idx.span()));
        }
    }

    // Collect contraction indices (RHS-only, not in output) in first-appearance order
    let free_names: HashSet<String> = input.output_indices.iter().map(Ident::to_string).collect();
    let mut contraction_pos = 0;
    let mut seen: HashSet<String> = HashSet::new();
    for term in &input.rhs_terms {
        for idx in &term.indices {
            let old = idx.to_string();
            if !free_names.contains(&old)
                && seen.insert(old.clone())
                && !rename.contains_key(&old)
                && contraction_pos < CONTRACTION_NAMES.len()
            {
                rename.insert(
                    old,
                    Ident::new(CONTRACTION_NAMES[contraction_pos], idx.span()),
                );
                contraction_pos += 1;
            }
        }
    }

    // Apply renames
    for idx in &mut input.output_indices {
        if let Some(new) = rename.get(&idx.to_string()) {
            *idx = new.clone();
        }
    }
    for term in &mut input.rhs_terms {
        for idx in &mut term.indices {
            if let Some(new) = rename.get(&idx.to_string()) {
                *idx = new.clone();
            }
        }
    }
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
            t.indices
                .iter()
                .all(|idx| free_names.contains(&idx.to_string()))
        });
        if all_rhs_indices_are_free
            && rhs[0].indices.len() == rhs[1].indices.len()
            && rhs[0].indices.len() == free.len()
        {
            return ContractionKind::Hadamard;
        }
    }

    // Batch GEMM: 2 terms, each with ≥3 indices, exactly 1 contraction index,
    // leading indices are shared batch dims, inner 2 dims form a GEMM.
    // Pattern: c[b..,i,j] = a[b..,i,k] * m[b..,k,j]
    if rhs.len() == 2 && contraction.len() == 1 && free.len() >= 3 {
        let a = &rhs[0];
        let b = &rhs[1];
        if a.indices.len() >= 3 && b.indices.len() >= 3 {
            let batch_count = a.indices.len() - 2;
            // Check that leading indices are the same batch dims in both terms
            if b.indices.len() - 2 == batch_count {
                let batch_match = (0..batch_count).all(|d| a.indices[d] == b.indices[d]);
                // Check that inner 2 dims form standard GEMM: a[..,i,k] * b[..,k,j]
                let k = &contraction[0];
                let a_inner1 = a.indices[batch_count + 1].to_string();
                let b_inner0 = b.indices[batch_count].to_string();
                if batch_match && a_inner1 == *k && b_inner0 == *k {
                    // Check output is [b..,i,j]
                    let out_batch_match = (0..batch_count).all(|d| free[d] == a.indices[d]);
                    if out_batch_match && free.len() == batch_count + 2 {
                        return ContractionKind::BatchGemm { batch_count };
                    }
                }
            }
        }
    }

    ContractionKind::Fallback
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

pub(crate) fn einsum_impl(input: TokenStream2) -> Result<TokenStream2> {
    let mut parsed: EinsumInput = syn::parse2(input)?;
    validate(&parsed)?;
    canonicalize(&mut parsed);
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
        ContractionKind::BatchGemm { batch_count } => codegen_batch_gemm(input, batch_count),
        ContractionKind::Fallback => codegen_fallback(input),
    }
}

/// GEMM: emit `Tensor::matmul_into` with optional transposes.
///
/// When no transpose is needed, passes references directly (zero-copy).
fn codegen_gemm(input: &EinsumInput, transpose_a: bool, transpose_b: bool) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let b_name = &input.rhs_terms[1].name;
    let a_bind = Ident::new(&format!("__{a_name}"), a_name.span());
    let b_bind = Ident::new(&format!("__{b_name}"), b_name.span());

    // When transposed, we must materialise; otherwise borrow directly.
    let (a_prep, a_ref) = if transpose_a {
        (quote! { let __a_t = #a_bind.t(); }, quote! { (&__a_t) })
    } else {
        (quote! {}, quote! { #a_bind })
    };
    let (b_prep, b_ref) = if transpose_b {
        (quote! { let __b_t = #b_bind.t(); }, quote! { (&__b_t) })
    } else {
        (quote! {}, quote! { #b_bind })
    };

    Ok(quote! {
        {
            let #a_bind = &#a_name;
            let #b_bind = &#b_name;
            #a_prep
            #b_prep
            let __m = #a_ref.nrows();
            let __n = #b_ref.ncols();
            let mut __out = nabla::tensor::Tensor::zeros(__m, __n);
            nabla::tensor::Tensor::matmul_into(&mut __out, #a_ref, #b_ref);
            __out
        }
    })
}

/// GEMV: matrix × vector → column vector, via matmul_into with Mx1 / 1xN.
///
/// When no transpose is needed, passes references directly (zero-copy).
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

    let (mat_prep, mat_ref) = if transpose_mat {
        (
            quote! { let __mat_t = #mat_bind.t(); },
            quote! { (&__mat_t) },
        )
    } else {
        (quote! {}, quote! { #mat_bind })
    };

    Ok(quote! {
        {
            let #mat_bind = &#mat_name;
            let #vec_bind = &#vec_name;
            #mat_prep
            let __m = #mat_ref.nrows();
            let mut __out = nabla::tensor::Tensor::zeros(__m, 1);
            nabla::tensor::Tensor::matmul_into(&mut __out, #mat_ref, #vec_bind);
            __out
        }
    })
}

/// Hadamard: element-wise product via `emul`.
fn codegen_hadamard(input: &EinsumInput) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let b_name = &input.rhs_terms[1].name;

    // Check if index order matches (both same order → direct emul).
    // If b has reversed indices, we need to transpose b.
    let a_indices: Vec<String> = input.rhs_terms[0]
        .indices
        .iter()
        .map(Ident::to_string)
        .collect();
    let b_indices: Vec<String> = input.rhs_terms[1]
        .indices
        .iter()
        .map(Ident::to_string)
        .collect();

    if a_indices == b_indices {
        Ok(quote! {
            {
                (#a_name).emul(&#b_name)
            }
        })
    } else {
        // b has reversed index order → transpose b
        Ok(quote! {
            {
                (#a_name).emul(&(#b_name).t())
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
            let mut __acc = nabla::scalar::math_utils::zero::<_>();
            for __diag_idx in 0..__n {
                __acc = nabla::scalar::math_utils::add(
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
                nabla::scalar::math_utils::mul(
                    &#row_src.get(__i, 0),
                    &#col_src.get(__j, 0),
                )
            })
        }
    })
}

/// Batch GEMM: c[b..,i,j] = a[b..,i,k] * m[b..,k,j]
///
/// Generates nested batch loops, extracts 2-D slices via `slice_2d`, calls
/// `Tensor::matmul`, and writes back via `set_slice_2d`.
fn codegen_batch_gemm(input: &EinsumInput, batch_count: usize) -> Result<TokenStream2> {
    let a_name = &input.rhs_terms[0].name;
    let b_name = &input.rhs_terms[1].name;
    let a_bind = Ident::new(&format!("__{a_name}"), a_name.span());
    let b_bind = Ident::new(&format!("__{b_name}"), b_name.span());

    // Batch loop variables
    let batch_vars: Vec<Ident> = (0..batch_count)
        .map(|d| Ident::new(&format!("__b{d}"), proc_macro2::Span::call_site()))
        .collect();

    // Shape expressions for each output dimension
    let mut shape_exprs = Vec::new();
    for d in 0..batch_count {
        shape_exprs.push(quote! { #a_bind.dim(#d) });
    }
    // Inner matrix dims: rows from a, cols from b
    let inner_row_dim = batch_count;
    let inner_col_dim = batch_count + 1;
    shape_exprs.push(quote! { #a_bind.dim(#inner_row_dim) });
    shape_exprs.push(quote! { #b_bind.dim(#inner_col_dim) });

    // Build nested batch loops (outermost → innermost)
    let batch_idx_array = quote! { &[#(#batch_vars),*] };
    let inner_body = quote! {
        let __a_slice = #a_bind.slice_2d(#batch_idx_array);
        let __b_slice = #b_bind.slice_2d(#batch_idx_array);
        let __c_slice = &__a_slice * &__b_slice;
        __out.set_slice_2d(#batch_idx_array, &__c_slice);
    };

    let mut loops = inner_body;
    for d in (0..batch_count).rev() {
        let bv = &batch_vars[d];
        let dim = quote! { #a_bind.dim(#d) };
        loops = quote! {
            for #bv in 0..#dim {
                #loops
            }
        };
    }

    Ok(quote! {
        {
            let #a_bind = &#a_name;
            let #b_bind = &#b_name;
            let mut __out = nabla::tensor::NdTensor::zeros(
                &[#(#shape_exprs),*]
            );
            #loops
            __out
        }
    })
}

/// Fallback: general loop-based contraction (supports N-D indices).
///
/// For 3+ terms, generates sequential binary contractions instead of one big
/// nested loop. E.g. `c[i,j] = a[i,k] * b[k,l] * d[l,j]` becomes:
///   1. `tmp[i,l] = a[i,k] * b[k,l]`  (contract k)
///   2. `c[i,j] = tmp[i,l] * d[l,j]`  (contract l)
fn codegen_fallback(input: &EinsumInput) -> Result<TokenStream2> {
    if input.rhs_terms.len() >= 3 {
        return codegen_sequential_contraction(input);
    }
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
                let mut __acc = nabla::scalar::math_utils::zero::<_>();
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
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
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
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
                    #acc_body
                    __acc
                })
            }
        }
    } else {
        // N-D output (>2 free indices): generate `NdTensor::from_fn`.
        let mut shape_exprs = Vec::new();
        for fidx in free_indices {
            let dim = dim_expr_for_index_any_pos(fidx, rhs_terms)?;
            shape_exprs.push(dim);
        }
        // Map free index idents to closure parameter via `__idx[n]`
        let idx_bindings: Vec<TokenStream2> = free_indices
            .iter()
            .enumerate()
            .map(|(n, fi)| quote! { let #fi = __idx[#n]; })
            .collect();
        quote! {
            {
                #(#tensor_bindings)*
                nabla::tensor::NdTensor::from_fn(
                    &[#(#shape_exprs),*],
                    |__idx| {
                        #(#idx_bindings)*
                        let mut __acc = nabla::scalar::math_utils::zero::<_>();
                        #acc_body
                        __acc
                    },
                )
            }
        }
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
        __acc = nabla::scalar::math_utils::add(&__acc, &__prod);
    };

    if contraction_indices.is_empty() {
        return Ok(step);
    }

    // L1 tiling: tile the innermost (last) contraction index for cache locality.
    // Outer contraction indices use plain loops; the innermost gets step_by tiling.
    let mut loops = step;
    let n = contraction_indices.len();
    for (loop_i, &cidx) in contraction_indices.iter().rev().enumerate() {
        let dim = dim_expr_for_index_any_pos(cidx, rhs_terms)?;
        if loop_i == 0 && n >= 1 {
            // Innermost contraction: tile with block size 64
            let block_var = Ident::new(
                &format!("__{}_blk", cidx),
                cidx.span(),
            );
            let end_var = Ident::new(
                &format!("__{}_end", cidx),
                cidx.span(),
            );
            loops = quote! {
                {
                    let __dim = #dim;
                    let mut #block_var: usize = 0;
                    while #block_var < __dim {
                        let #end_var = if #block_var + 64 < __dim { #block_var + 64 } else { __dim };
                        for #cidx in #block_var..#end_var {
                            #loops
                        }
                        #block_var += 64;
                    }
                }
            };
        } else {
            loops = quote! {
                for #cidx in 0..#dim {
                    #loops
                }
            };
        }
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
            nabla::scalar::math_utils::mul(&#expr, &#access)
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
            // N-D: generate `.get_nd(&[i, j, k, ...])`
            let indices = &term.indices;
            Ok(quote! { #binding.get_nd(&[#(#indices),*]) })
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
                return Ok(dim_expr_at(&binding, pos, term.indices.len()));
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
                return Ok(dim_expr_at(&binding, pos, term.indices.len()));
            }
        }
    }
    Err(Error::new_spanned(
        idx,
        format!("einsum!: index `{idx}` not found in any RHS tensor"),
    ))
}

/// Generate a dimension expression for a tensor at a given axis position.
///
/// For 2-D tensors (≤2 indices), uses `.nrows()` / `.ncols()` for readability.
/// For N-D tensors (>2 indices), always uses `.dim(pos)` because `.nrows()` /
/// `.ncols()` on `NdTensor` refer to the *last* two dimensions, not axis 0/1.
fn dim_expr_at(binding: &Ident, pos: usize, ndim: usize) -> TokenStream2 {
    if ndim > 2 {
        quote! { #binding.dim(#pos) }
    } else {
        match pos {
            0 => quote! { #binding.nrows() },
            1 => quote! { #binding.ncols() },
            n => quote! { #binding.dim(#n) },
        }
    }
}

// ---------------------------------------------------------------------------
// Greedy contraction order optimizer for 3+ term einsums
// ---------------------------------------------------------------------------

/// Compute the greedy contraction order for N terms.
///
/// At each step, pick the pair of remaining terms whose binary contraction
/// produces the smallest intermediate (fewest unique output indices). Since
/// actual tensor sizes are unknown at compile time, we use index count as a
/// proxy (equivalent to assuming each dimension has the same size).
///
/// Returns a sequence of `(left_idx, right_idx)` pairs into the original
/// `rhs_terms` array, plus the resulting index set for each step. The caller
/// should execute contractions in this order.
fn greedy_contraction_order(
    terms: &[IndexedTensor],
    output_indices: &[Ident],
) -> Vec<(usize, usize)> {
    let output_names: HashSet<String> = output_indices.iter().map(Ident::to_string).collect();

    // Track each "slot": initially each slot is one original term.
    // As we contract pairs, we replace two slots with one merged slot.
    // slot_indices[i] = current index set for slot i
    let mut slot_indices: Vec<HashSet<String>> = terms
        .iter()
        .map(|t| t.indices.iter().map(Ident::to_string).collect())
        .collect();
    // slot_alive[i] = whether slot i is still available
    let mut slot_alive: Vec<bool> = vec![true; terms.len()];
    let mut order: Vec<(usize, usize)> = Vec::new();

    let n = terms.len();
    for _ in 0..n - 1 {
        let alive: Vec<usize> = (0..slot_indices.len())
            .filter(|&i| slot_alive[i])
            .collect();

        let mut best_pair = (alive[0], alive[1]);
        let mut best_cost = usize::MAX;

        // Enumerate all pairs of alive slots, pick the one that minimizes
        // the number of unique indices in the result.
        for i in 0..alive.len() {
            for j in (i + 1)..alive.len() {
                let li = alive[i];
                let ri = alive[j];
                let left = &slot_indices[li];
                let right = &slot_indices[ri];

                // Indices in the remaining alive slots (excluding li and ri).
                let remaining: HashSet<String> = alive
                    .iter()
                    .filter(|&&k| k != li && k != ri)
                    .flat_map(|&k| slot_indices[k].iter().cloned())
                    .collect();

                // Contracted = shared indices NOT in output, NOT in remaining.
                let contracted: HashSet<&String> = left
                    .intersection(right)
                    .filter(|idx| !output_names.contains(*idx) && !remaining.contains(*idx))
                    .collect();

                // Result indices = union minus contracted.
                let result_count = left
                    .union(right)
                    .filter(|idx| !contracted.contains(idx))
                    .count();

                if result_count < best_cost {
                    best_cost = result_count;
                    best_pair = (li, ri);
                }
            }
        }

        let (li, ri) = best_pair;
        order.push((li, ri));

        // Merge: compute the result index set for this contraction.
        let left = &slot_indices[li];
        let right = &slot_indices[ri];
        let remaining: HashSet<String> = (0..slot_indices.len())
            .filter(|&k| slot_alive[k] && k != li && k != ri)
            .flat_map(|k| slot_indices[k].iter().cloned())
            .collect();
        let contracted: HashSet<String> = left
            .intersection(right)
            .filter(|idx| !output_names.contains(*idx) && !remaining.contains(*idx))
            .cloned()
            .collect();
        let merged: HashSet<String> = left
            .union(right)
            .filter(|idx| !contracted.contains(*idx))
            .cloned()
            .collect();

        // Kill the two slots, add a new slot with the merged indices.
        slot_alive[li] = false;
        slot_alive[ri] = false;
        slot_indices.push(merged);
        slot_alive.push(true);
    }

    order
}

// ---------------------------------------------------------------------------
// Sequential binary contraction for 3+ term einsums
// ---------------------------------------------------------------------------

/// Represents an intermediate tensor in the sequential contraction pipeline.
/// Can be either an original user tensor (referenced by name) or a generated
/// intermediate (referenced by tmp variable name).
struct IntermediateTensor {
    /// Variable name to use in generated code (e.g. `__a` or `__tmp_0`).
    binding: Ident,
    /// Index labels carried by this tensor.
    indices: Vec<Ident>,
    /// Whether this is an original (needs `.get(i,j)`) or intermediate (needs `.get_nd`).
    is_original: bool,
}

/// Generate sequential binary contractions for 3+ term einsum expressions.
///
/// Uses greedy contraction ordering to minimize intermediate tensor sizes.
/// At each step, contracts the pair of remaining tensors whose binary
/// contraction produces the fewest intermediate indices.
fn codegen_sequential_contraction(input: &EinsumInput) -> Result<TokenStream2> {
    let output_names: HashSet<String> = input
        .output_indices
        .iter()
        .map(Ident::to_string)
        .collect();

    // Compute greedy contraction order.
    let contraction_order = greedy_contraction_order(&input.rhs_terms, &input.output_indices);

    // Build contraction path annotation for generated code readability.
    let path_desc: String = contraction_order
        .iter()
        .enumerate()
        .map(|(step, &(li, ri))| {
            let l_name = if li < input.rhs_terms.len() {
                input.rhs_terms[li].name.to_string()
            } else {
                format!("tmp_{}", li - input.rhs_terms.len())
            };
            let r_name = if ri < input.rhs_terms.len() {
                input.rhs_terms[ri].name.to_string()
            } else {
                format!("tmp_{}", ri - input.rhs_terms.len())
            };
            format!("step {step}: {l_name} x {r_name}")
        })
        .collect::<Vec<_>>()
        .join(", ");

    // Collect all original tensor bindings (deduplicated).
    let mut seen_names: HashSet<String> = HashSet::new();
    let tensor_bindings: Vec<TokenStream2> = input
        .rhs_terms
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

    // Build a map from slot index to IntermediateTensor.
    // Original terms get slots 0..n_terms; intermediates get slots n_terms+.
    let n_terms = input.rhs_terms.len();
    let mut slots: HashMap<usize, IntermediateTensor> = HashMap::new();
    for (i, term) in input.rhs_terms.iter().enumerate() {
        slots.insert(
            i,
            IntermediateTensor {
                binding: Ident::new(&format!("__{}", term.name), term.name.span()),
                indices: term.indices.clone(),
                is_original: true,
            },
        );
    }

    let mut intermediate_stmts: Vec<TokenStream2> = Vec::new();
    let n_steps = contraction_order.len();
    let mut next_slot = n_terms;

    for (step_idx, &(li, ri)) in contraction_order.iter().enumerate() {
        // Take left and right from slots.
        let left = slots.remove(&li);
        let right = slots.remove(&ri);
        let (acc, right_tensor) = match (left, right) {
            (Some(l), Some(r)) => (l, r),
            _ => {
                return Err(Error::new(
                    Span::call_site(),
                    "einsum!: internal error in contraction order",
                ));
            }
        };

        let right_indices = &right_tensor.indices;

        // Collect indices in remaining (unconsumed) slots.
        let remaining_indices: HashSet<String> = slots
            .values()
            .flat_map(|t| t.indices.iter())
            .map(Ident::to_string)
            .collect();

        // Determine which indices to contract: those that appear in BOTH
        // left (acc) and right, but NOT in the output and NOT in remaining terms.
        let left_set: HashSet<String> = acc.indices.iter().map(Ident::to_string).collect();
        let right_set: HashSet<String> = right_indices.iter().map(Ident::to_string).collect();

        let contracted: HashSet<String> = left_set
            .intersection(&right_set)
            .filter(|idx| !output_names.contains(*idx) && !remaining_indices.contains(*idx))
            .cloned()
            .collect();

        // Output indices of this step = union of left and right indices, minus contracted,
        // preserving order (left first, then right-only).
        let mut step_output_indices: Vec<Ident> = Vec::new();
        let mut step_output_seen: HashSet<String> = HashSet::new();
        for idx in &acc.indices {
            let s = idx.to_string();
            if !contracted.contains(&s) && step_output_seen.insert(s) {
                step_output_indices.push(idx.clone());
            }
        }
        for idx in right_indices {
            let s = idx.to_string();
            if !contracted.contains(&s) && step_output_seen.insert(s) {
                step_output_indices.push(idx.clone());
            }
        }

        // Contraction index idents (for loop generation).
        let contraction_idents: Vec<&Ident> = acc
            .indices
            .iter()
            .chain(right_indices.iter())
            .filter(|idx| contracted.contains(&idx.to_string()))
            .collect::<Vec<_>>();
        // Deduplicate contraction idents (they appear in both sides).
        let mut contraction_dedup: Vec<&Ident> = Vec::new();
        {
            let mut seen_c: HashSet<String> = HashSet::new();
            for ci in &contraction_idents {
                if seen_c.insert(ci.to_string()) {
                    contraction_dedup.push(ci);
                }
            }
        }

        // Build element access for left operand.
        let left_access = intermediate_element_access(&acc)?;
        // Build element access for right operand.
        let right_access = intermediate_element_access(&right_tensor)?;

        // Product expression.
        let product = quote! {
            nabla::scalar::math_utils::mul(&#left_access, &#right_access)
        };

        // Build contraction loops around the product + accumulate.
        let step_stmt = quote! {
            let __prod = #product;
            __acc = nabla::scalar::math_utils::add(&__acc, &__prod);
        };

        let mut loops = step_stmt;
        let n_contr = contraction_dedup.len();
        for (loop_i, &cidx) in contraction_dedup.iter().rev().enumerate() {
            let dim = dim_for_binary_index(cidx, &acc, &right_tensor)?;
            if loop_i == 0 && n_contr >= 1 {
                // L1 tiling on innermost contraction index (block size 64)
                let block_var = Ident::new(
                    &format!("__{}_blk", cidx),
                    cidx.span(),
                );
                let end_var = Ident::new(
                    &format!("__{}_end", cidx),
                    cidx.span(),
                );
                loops = quote! {
                    {
                        let __dim = #dim;
                        let mut #block_var: usize = 0;
                        while #block_var < __dim {
                            let #end_var = if #block_var + 64 < __dim { #block_var + 64 } else { __dim };
                            for #cidx in #block_var..#end_var {
                                #loops
                            }
                            #block_var += 64;
                        }
                    }
                };
            } else {
                loops = quote! {
                    for #cidx in 0..#dim {
                        #loops
                    }
                };
            }
        }

        let is_final_step = step_idx == n_steps - 1;

        if is_final_step {
            // Final step: produce the actual output type matching `input.output_indices`.
            // The step_output_indices should match output_indices (possibly reordered).
            let free_indices = &input.output_indices;

            let result = if free_indices.is_empty() {
                // Scalar output
                quote! {
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
                    #loops
                }
            } else if free_indices.len() == 1 {
                let i_idx = &free_indices[0];
                let nrows_expr = dim_for_binary_index(i_idx, &acc, &right_tensor)?;
                quote! {
                    let __final = nabla::tensor::Tensor::from_fn(#nrows_expr, 1, |#i_idx, _| {
                        let mut __acc = nabla::scalar::math_utils::zero::<_>();
                        #loops
                        __acc
                    });
                }
            } else if free_indices.len() == 2 {
                let i_idx = &free_indices[0];
                let j_idx = &free_indices[1];
                let nrows_expr = dim_for_binary_index(i_idx, &acc, &right_tensor)?;
                let ncols_expr = dim_for_binary_index(j_idx, &acc, &right_tensor)?;
                quote! {
                    let __final = nabla::tensor::Tensor::from_fn(#nrows_expr, #ncols_expr, |#i_idx, #j_idx| {
                        let mut __acc = nabla::scalar::math_utils::zero::<_>();
                        #loops
                        __acc
                    });
                }
            } else {
                // N-D output
                let mut shape_exprs = Vec::new();
                for fidx in free_indices {
                    shape_exprs.push(dim_for_binary_index(fidx, &acc, &right_tensor)?);
                }
                let idx_bindings: Vec<TokenStream2> = free_indices
                    .iter()
                    .enumerate()
                    .map(|(n, fi)| quote! { let #fi = __idx[#n]; })
                    .collect();
                quote! {
                    let __final = nabla::tensor::NdTensor::from_fn(
                        &[#(#shape_exprs),*],
                        |__idx| {
                            #(#idx_bindings)*
                            let mut __acc = nabla::scalar::math_utils::zero::<_>();
                            #loops
                            __acc
                        },
                    );
                }
            };

            intermediate_stmts.push(result);
        } else {
            // Intermediate step: produce NdTensor for the next step.
            let tmp_name = Ident::new(&format!("__tmp_{step_idx}"), Span::call_site());

            let ndim = step_output_indices.len();
            if ndim == 0 {
                // Scalar intermediate (unlikely but handle it).
                intermediate_stmts.push(quote! {
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
                    #loops
                    let #tmp_name = __acc;
                });
            } else {
                let mut shape_exprs = Vec::new();
                for fidx in &step_output_indices {
                    shape_exprs.push(dim_for_binary_index(fidx, &acc, &right_tensor)?);
                }
                let idx_bindings: Vec<TokenStream2> = step_output_indices
                    .iter()
                    .enumerate()
                    .map(|(n, fi)| quote! { let #fi = __idx[#n]; })
                    .collect();
                intermediate_stmts.push(quote! {
                    let #tmp_name = nabla::tensor::NdTensor::from_fn(
                        &[#(#shape_exprs),*],
                        |__idx| {
                            #(#idx_bindings)*
                            let mut __acc = nabla::scalar::math_utils::zero::<_>();
                            #loops
                            __acc
                        },
                    );
                });
            }

            // Insert the intermediate result as a new slot.
            slots.insert(
                next_slot,
                IntermediateTensor {
                    binding: tmp_name,
                    indices: step_output_indices,
                    is_original: false,
                },
            );
            next_slot += 1;
        }
    }

    // Assemble the final block.
    let final_expr = if input.output_indices.is_empty() {
        quote! { __acc }
    } else {
        quote! { __final }
    };

    // Emit contraction path as a no-op let binding (visible in expanded code).
    let path_comment = Ident::new("__einsum_contraction_path", Span::call_site());

    Ok(quote! {
        {
            #[allow(unused_variables)]
            let #path_comment: &str = #path_desc;
            #(#tensor_bindings)*
            #(#intermediate_stmts)*
            #final_expr
        }
    })
}

/// Generate element access for an intermediate tensor.
fn intermediate_element_access(tensor: &IntermediateTensor) -> Result<TokenStream2> {
    let binding = &tensor.binding;
    match tensor.indices.len() {
        0 => Ok(quote! { #binding }),
        1 => {
            let i = &tensor.indices[0];
            if tensor.is_original {
                Ok(quote! { #binding.get(#i, 0) })
            } else {
                Ok(quote! { #binding.get_nd(&[#i]) })
            }
        }
        2 => {
            let i = &tensor.indices[0];
            let j = &tensor.indices[1];
            if tensor.is_original {
                Ok(quote! { #binding.get(#i, #j) })
            } else {
                Ok(quote! { #binding.get_nd(&[#i, #j]) })
            }
        }
        _ => {
            let indices = &tensor.indices;
            Ok(quote! { #binding.get_nd(&[#(#indices),*]) })
        }
    }
}

/// Find the dimension expression for an index, searching both tensors in a
/// binary contraction step. Handles both original tensors (`.nrows()`/`.ncols()`)
/// and intermediate NdTensors (`.dim(pos)`).
fn dim_for_binary_index(
    idx: &Ident,
    left: &IntermediateTensor,
    right: &IntermediateTensor,
) -> Result<TokenStream2> {
    let s = idx.to_string();

    for tensor in [left, right] {
        for (pos, tidx) in tensor.indices.iter().enumerate() {
            if *tidx == s {
                let binding = &tensor.binding;
                if tensor.is_original {
                    return Ok(dim_expr_at(binding, pos, tensor.indices.len()));
                }
                return Ok(quote! { #binding.dim(#pos) });
            }
        }
    }

    Err(Error::new_spanned(
        idx,
        format!("einsum!: index `{idx}` not found in contraction operands"),
    ))
}
