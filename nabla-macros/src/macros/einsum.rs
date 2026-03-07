use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::{HashMap, HashSet};
use syn::{
    Error, Ident, Result,
    parse::{Parse, ParseStream},
};

struct IndexedTensor {
    name: Ident,
    indices: Vec<Ident>,
}

pub(crate) struct EinsumInput {
    _output_name: Ident,
    output_indices: Vec<Ident>,
    eq_token: syn::Token![=],
    rhs_terms: Vec<IndexedTensor>,
}

#[derive(Debug)]
enum ContractionKind {
    Gemm,
    GemmTransposed { transpose_a: bool, transpose_b: bool },
    Gemv { mat_first: bool },
    Hadamard,
    Trace,
    Outer,
    BatchGemm { batch_count: usize },
    Fallback,
}

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
    Ok(inner
        .parse_terminated(Ident::parse, syn::Token![,])?
        .into_iter()
        .collect())
}

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

fn name_set(indices: &[Ident]) -> HashSet<String> {
    indices.iter().map(Ident::to_string).collect()
}

fn name_list(indices: &[Ident]) -> Vec<String> {
    indices.iter().map(Ident::to_string).collect()
}

fn validate(input: &EinsumInput) -> Result<()> {
    if input.rhs_terms.is_empty() {
        return Err(Error::new_spanned(
            input.eq_token,
            "einsum!: at least one tensor term is required on the right-hand side",
        ));
    }

    // No duplicate indices on LHS.
    let mut seen: HashSet<String> = HashSet::new();
    for idx in &input.output_indices {
        if !seen.insert(idx.to_string()) {
            return Err(Error::new_spanned(
                idx,
                format!("einsum!: index `{idx}` appears twice on the LHS"),
            ));
        }
    }

    let free_names = name_set(&input.output_indices);
    for (name, (count, ident)) in &contraction_counts(input, &free_names) {
        if *count == 1 && !is_trace_index(input, name) {
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

    Ok(())
}

fn contraction_counts<'a>(
    input: &'a EinsumInput,
    free_names: &HashSet<String>,
) -> HashMap<String, (usize, &'a Ident)> {
    let mut counts = HashMap::new();
    for term in &input.rhs_terms {
        for idx in &term.indices {
            let name = idx.to_string();
            if !free_names.contains(&name) {
                counts.entry(name).and_modify(|(c, _)| *c += 1).or_insert((1, idx));
            }
        }
    }
    counts
}

fn is_trace_index(input: &EinsumInput, name: &str) -> bool {
    input
        .rhs_terms
        .iter()
        .any(|t| t.indices.iter().filter(|i| *i == name).count() >= 2)
}

fn canonicalize(input: &mut EinsumInput) {
    // Sort 2-term RHS alphabetically by tensor name BEFORE renaming.
    if input.rhs_terms.len() == 2 && input.rhs_terms[0].name > input.rhs_terms[1].name {
        input.rhs_terms.swap(0, 1);
    }

    const FREE_NAMES: &[&str] = &["i", "j", "k", "l", "m", "n"];
    const CONTRACTION_NAMES: &[&str] = &["p", "q", "r", "s"];

    let mut rename: HashMap<String, Ident> = HashMap::new();

    for (pos, idx) in input.output_indices.iter().enumerate() {
        if pos < FREE_NAMES.len() {
            rename
                .entry(idx.to_string())
                .or_insert_with(|| Ident::new(FREE_NAMES[pos], idx.span()));
        }
    }

    let free_names = name_set(&input.output_indices);
    let mut contraction_pos = 0;
    let mut seen: HashSet<String> = HashSet::new();
    for term in &input.rhs_terms {
        for idx in &term.indices {
            let old = idx.to_string();
            if !free_names.contains(&old)
                && seen.insert(old.clone())
                && contraction_pos < CONTRACTION_NAMES.len()
            {
                rename.entry(old).or_insert_with(|| {
                    let ident = Ident::new(CONTRACTION_NAMES[contraction_pos], idx.span());
                    contraction_pos += 1;
                    ident
                });
            }
        }
    }

    for idx in input.output_indices.iter_mut()
        .chain(input.rhs_terms.iter_mut().flat_map(|t| t.indices.iter_mut()))
    {
        if let Some(new) = rename.get(&idx.to_string()) { *idx = new.clone(); }
    }
}

fn collect_non_free<'a>(terms: &'a [IndexedTensor], free_names: &HashSet<String>) -> (Vec<String>, Vec<&'a Ident>) {
    let (mut names, mut idents) = (Vec::new(), Vec::new());
    let mut seen = HashSet::new();
    for term in terms {
        for idx in &term.indices {
            let s = idx.to_string();
            if !free_names.contains(&s) && seen.insert(s.clone()) {
                names.push(s);
                idents.push(idx);
            }
        }
    }
    (names, idents)
}

fn classify(input: &EinsumInput) -> ContractionKind {
    let free = &input.output_indices;
    let rhs = &input.rhs_terms;
    let free_names = name_set(free);
    let (contraction, _) = collect_non_free(rhs, &free_names);

    // Trace: scalar output, single 2D term, repeated index.
    if free.is_empty() && rhs.len() == 1 && contraction.is_empty() {
        let t = &rhs[0];
        if t.indices.len() == 2 && t.indices[0] == t.indices[1] {
            return ContractionKind::Trace;
        }
    }

    // Two-term patterns.
    if rhs.len() == 2 {
        let (a, b) = (&rhs[0], &rhs[1]);

        // GEMM / GemmTransposed: 2x2D, 2 free, 1 contraction.
        if a.indices.len() == 2 && b.indices.len() == 2 && free.len() == 2 && contraction.len() == 1
        {
            let k = &contraction[0];
            let (a0, a1) = (a.indices[0].to_string(), a.indices[1].to_string());
            let (b0, b1) = (b.indices[0].to_string(), b.indices[1].to_string());

            if a1 == *k && b0 == *k {
                return ContractionKind::Gemm;
            }
            let (ta, tb) = (a0 == *k, b1 == *k);
            if ta || tb {
                return ContractionKind::GemmTransposed {
                    transpose_a: ta,
                    transpose_b: tb,
                };
            }
        }

        // GEMV: 1 free, 1 contraction, one 2D + one 1D.
        if free.len() == 1 && contraction.len() == 1 {
            if a.indices.len() == 2 && b.indices.len() == 1 {
                return ContractionKind::Gemv { mat_first: true };
            }
            if a.indices.len() == 1 && b.indices.len() == 2 {
                return ContractionKind::Gemv { mat_first: false };
            }
        }

        // Outer product: 2 vectors, no contraction, 2 free.
        if contraction.is_empty() && free.len() == 2 && a.indices.len() == 1 && b.indices.len() == 1
        {
            return ContractionKind::Outer;
        }

        // Hadamard: matching dimensionality, no contraction, all free.
        if contraction.is_empty()
            && !free.is_empty()
            && a.indices.len() == b.indices.len()
            && a.indices.len() == free.len()
        {
            if rhs.iter().all(|t| t.indices.iter().all(|i| free_names.contains(&i.to_string()))) {
                return ContractionKind::Hadamard;
            }
        }

        // Batch GEMM: both >=3D, 1 contraction, matching leading batch dims.
        if contraction.len() == 1 && free.len() >= 3 && a.indices.len() >= 3 && b.indices.len() >= 3
        {
            let bc = a.indices.len() - 2;
            if b.indices.len() - 2 == bc {
                let batch_ok = (0..bc).all(|d| a.indices[d] == b.indices[d]);
                let k = &contraction[0];
                let a_inner1 = a.indices[bc + 1].to_string();
                let b_inner0 = b.indices[bc].to_string();
                if batch_ok && a_inner1 == *k && b_inner0 == *k {
                    let out_ok = (0..bc).all(|d| free[d] == a.indices[d]) && free.len() == bc + 2;
                    if out_ok {
                        return ContractionKind::BatchGemm { batch_count: bc };
                    }
                }
            }
        }
    }

    ContractionKind::Fallback
}

pub(crate) fn einsum_impl(input: TokenStream2) -> Result<TokenStream2> {
    let mut parsed: EinsumInput = syn::parse2(input)?;
    validate(&parsed)?;
    canonicalize(&mut parsed);
    codegen_einsum(&parsed)
}

fn codegen_einsum(input: &EinsumInput) -> Result<TokenStream2> {
    use ContractionKind::*;
    match classify(input) {
        Gemm => codegen_gemm(input, false, false),
        GemmTransposed { transpose_a, transpose_b } => codegen_gemm(input, transpose_a, transpose_b),
        Gemv { mat_first } => codegen_gemv(input, mat_first),
        Hadamard => codegen_hadamard(input),
        Trace => codegen_trace(input),
        Outer => codegen_outer(input),
        BatchGemm { batch_count } => codegen_batch_gemm(input, batch_count),
        Fallback => codegen_fallback(input),
    }
}

fn element_access(binding: &Ident, indices: &[Ident], is_original: bool) -> TokenStream2 {
    match (indices.len(), is_original) {
        (0, _) => quote! { #binding },
        (1, true) => { let i = &indices[0]; quote! { #binding.get(#i, 0) } }
        (2, true) => { let (i, j) = (&indices[0], &indices[1]); quote! { #binding.get(#i, #j) } }
        _ => quote! { #binding.get_nd(&[#(#indices),*]) },
    }
}

fn dim_expr_at(binding: &Ident, pos: usize, ndim: usize) -> TokenStream2 {
    if ndim > 2 { return quote! { #binding.dim(#pos) }; }
    match pos {
        0 => quote! { #binding.nrows() },
        1 => quote! { #binding.ncols() },
        n => quote! { #binding.dim(#n) },
    }
}

fn find_dim_expr(
    idx: &Ident,
    tensors: &[(&Ident, &[Ident], bool)],
    preferred_pos: Option<usize>,
) -> Result<TokenStream2> {
    let s = idx.to_string();
    let mut fallback = None;
    for &(binding, indices, is_orig) in tensors {
        for (pos, tidx) in indices.iter().enumerate() {
            if *tidx == s {
                let expr = if is_orig { dim_expr_at(binding, pos, indices.len()) }
                           else { quote! { #binding.dim(#pos) } };
                if preferred_pos == Some(pos) { return Ok(expr); }
                if fallback.is_none() { fallback = Some(expr); }
            }
        }
    }
    fallback.ok_or_else(|| Error::new_spanned(idx, format!("einsum!: index `{idx}` not found in any tensor")))
}

fn rhs_descriptors(terms: &[IndexedTensor]) -> Vec<(Ident, &[Ident], bool)> {
    terms
        .iter()
        .map(|t| {
            let binding = Ident::new(&format!("__{}", t.name), t.name.span());
            (binding, t.indices.as_slice(), true)
        })
        .collect()
}

fn build_tiled_loops<F>(
    contraction_indices: &[&Ident],
    dim_lookup: F,
    inner_step: TokenStream2,
) -> Result<TokenStream2>
where
    F: Fn(&Ident) -> Result<TokenStream2>,
{
    if contraction_indices.is_empty() {
        return Ok(inner_step);
    }

    let mut loops = inner_step;
    for (loop_i, &cidx) in contraction_indices.iter().rev().enumerate() {
        let dim = dim_lookup(cidx)?;
        if loop_i == 0 {
            let blk = Ident::new(&format!("__{}_blk", cidx), cidx.span());
            let end = Ident::new(&format!("__{}_end", cidx), cidx.span());
            loops = quote! {
                {
                    let __dim = #dim;
                    let mut #blk: usize = 0;
                    while #blk < __dim {
                        let #end = if #blk + 64 < __dim { #blk + 64 } else { __dim };
                        for #cidx in #blk..#end {
                            #loops
                        }
                        #blk += 64;
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

fn emit_output_tensor<F>(
    free_indices: &[Ident],
    dim_lookup: F,
    loops: &TokenStream2,
    wrap: bool,
) -> Result<TokenStream2>
where
    F: Fn(&Ident) -> Result<TokenStream2>,
{
    let wrap_or_bare = |body: TokenStream2| -> TokenStream2 {
        if wrap { quote! { let __final = #body; } } else { body }
    };

    match free_indices.len() {
        0 => {
            let core = quote! { let mut __acc = nabla::scalar::math_utils::zero::<_>(); #loops };
            Ok(if wrap { core } else { quote! { #core __acc } })
        }
        1 => {
            let fi = &free_indices[0];
            let nr = dim_lookup(fi)?;
            Ok(wrap_or_bare(quote! {
                nabla::tensor::Tensor::from_fn(#nr, 1, |#fi, _| {
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
                    #loops
                    __acc
                })
            }))
        }
        2 => {
            let (fi, fj) = (&free_indices[0], &free_indices[1]);
            let (nr, nc) = (dim_lookup(fi)?, dim_lookup(fj)?);
            Ok(wrap_or_bare(quote! {
                nabla::tensor::Tensor::from_fn(#nr, #nc, |#fi, #fj| {
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
                    #loops
                    __acc
                })
            }))
        }
        _ => {
            let shape_exprs: Vec<TokenStream2> = free_indices.iter()
                .map(|fi| dim_lookup(fi)).collect::<Result<_>>()?;
            let idx_bindings: Vec<TokenStream2> = free_indices.iter().enumerate()
                .map(|(n, fi)| quote! { let #fi = __idx[#n]; }).collect();
            Ok(wrap_or_bare(quote! {
                nabla::tensor::NdTensor::from_fn(
                    &[#(#shape_exprs),*],
                    |__idx| { #(#idx_bindings)* let mut __acc = nabla::scalar::math_utils::zero::<_>(); #loops __acc },
                )
            }))
        }
    }
}

fn bind_ref(name: &Ident) -> (Ident, TokenStream2) {
    let bind = Ident::new(&format!("__{name}"), name.span());
    (bind.clone(), quote! { let #bind = &#name; })
}

fn transpose_ref(bind: &Ident, label: &str, transpose: bool) -> (TokenStream2, TokenStream2) {
    if !transpose { return (quote! {}, quote! { #bind }); }
    let t_bind = Ident::new(&format!("__{label}_t"), bind.span());
    (quote! { let #t_bind = #bind.t(); }, quote! { (&#t_bind) })
}

fn emit_matmul_into(
    a_ref: &TokenStream2,
    b_ref: &TokenStream2,
    out_cols: TokenStream2,
) -> TokenStream2 {
    quote! {
        let __m = #a_ref.nrows();
        let __n = #out_cols;
        let mut __out = nabla::tensor::Tensor::zeros(__m, __n);
        nabla::tensor::Tensor::matmul_into(&mut __out, #a_ref, #b_ref);
        __out
    }
}

fn codegen_gemm(input: &EinsumInput, transpose_a: bool, transpose_b: bool) -> Result<TokenStream2> {
    let (a_bind, a_stmt) = bind_ref(&input.rhs_terms[0].name);
    let (b_bind, b_stmt) = bind_ref(&input.rhs_terms[1].name);
    let (a_prep, a_ref) = transpose_ref(&a_bind, "a", transpose_a);
    let (b_prep, b_ref) = transpose_ref(&b_bind, "b", transpose_b);
    let body = emit_matmul_into(&a_ref, &b_ref, quote! { #b_ref.ncols() });
    Ok(quote! {{ #a_stmt #b_stmt #a_prep #b_prep #body }})
}

fn codegen_gemv(input: &EinsumInput, mat_first: bool) -> Result<TokenStream2> {
    let (mat_term, vec_term) = if mat_first { (&input.rhs_terms[0], &input.rhs_terms[1]) }
                                else { (&input.rhs_terms[1], &input.rhs_terms[0]) };
    let (mat_bind, mat_stmt) = bind_ref(&mat_term.name);
    let (vec_bind, vec_stmt) = bind_ref(&vec_term.name);
    let transpose_mat = mat_term.indices[0] != input.output_indices[0];
    let (mat_prep, mat_ref) = transpose_ref(&mat_bind, "mat", transpose_mat);
    let body = emit_matmul_into(&mat_ref, &quote! { #vec_bind }, quote! { 1 });
    Ok(quote! {{ #mat_stmt #vec_stmt #mat_prep #body }})
}

fn codegen_hadamard(input: &EinsumInput) -> Result<TokenStream2> {
    let (a, b) = (&input.rhs_terms[0].name, &input.rhs_terms[1].name);
    let same_order = name_list(&input.rhs_terms[0].indices) == name_list(&input.rhs_terms[1].indices);
    Ok(if same_order { quote! { { (#a).emul(&#b) } } }
       else { quote! { { (#a).emul(&(#b).t()) } } })
}

fn codegen_trace(input: &EinsumInput) -> Result<TokenStream2> {
    let (a_bind, a_stmt) = bind_ref(&input.rhs_terms[0].name);
    Ok(quote! {{
        #a_stmt
        let __n = #a_bind.nrows().min(#a_bind.ncols());
        let mut __acc = nabla::scalar::math_utils::zero::<_>();
        for __d in 0..__n {
            __acc = nabla::scalar::math_utils::add(&__acc, &#a_bind.get(__d, __d));
        }
        __acc
    }})
}

fn codegen_outer(input: &EinsumInput) -> Result<TokenStream2> {
    let (a_bind, a_stmt) = bind_ref(&input.rhs_terms[0].name);
    let (b_bind, b_stmt) = bind_ref(&input.rhs_terms[1].name);
    let (row, col) = if input.rhs_terms[0].indices[0].to_string() == input.output_indices[0].to_string() {
        (quote! { #a_bind }, quote! { #b_bind })
    } else {
        (quote! { #b_bind }, quote! { #a_bind })
    };
    Ok(quote! {{ #a_stmt #b_stmt
        nabla::tensor::Tensor::from_fn(#row.nrows(), #col.nrows(), |__i, __j| {
            nabla::scalar::math_utils::mul(&#row.get(__i, 0), &#col.get(__j, 0))
        })
    }})
}

fn codegen_batch_gemm(input: &EinsumInput, batch_count: usize) -> Result<TokenStream2> {
    let (a_bind, a_stmt) = bind_ref(&input.rhs_terms[0].name);
    let (b_bind, b_stmt) = bind_ref(&input.rhs_terms[1].name);
    let batch_vars: Vec<Ident> = (0..batch_count)
        .map(|d| Ident::new(&format!("__b{d}"), Span::call_site())).collect();
    let (rd, cd) = (batch_count, batch_count + 1);
    let mut shape_exprs: Vec<TokenStream2> = (0..batch_count)
        .map(|d| quote! { #a_bind.dim(#d) }).collect();
    shape_exprs.extend([quote! { #a_bind.dim(#rd) }, quote! { #b_bind.dim(#cd) }]);

    let batch_idx = quote! { &[#(#batch_vars),*] };
    let mut loops = quote! {
        let __a_slice = #a_bind.slice_2d(#batch_idx);
        let __b_slice = #b_bind.slice_2d(#batch_idx);
        __out.set_slice_2d(#batch_idx, &(&__a_slice * &__b_slice));
    };
    for d in (0..batch_count).rev() {
        let (bv, dim) = (&batch_vars[d], quote! { #a_bind.dim(#d) });
        loops = quote! { for #bv in 0..#dim { #loops } };
    }
    Ok(quote! {{ #a_stmt #b_stmt let mut __out = nabla::tensor::NdTensor::zeros(&[#(#shape_exprs),*]); #loops __out }})
}

fn build_accumulate_step(terms: &[IndexedTensor]) -> TokenStream2 {
    let accesses: Vec<TokenStream2> = terms.iter().map(|t| {
        let binding = Ident::new(&format!("__{}", t.name), t.name.span());
        element_access(&binding, &t.indices, true)
    }).collect();
    let product_expr = accesses.iter().skip(1).fold(accesses[0].clone(), |acc, a| {
        quote! { nabla::scalar::math_utils::mul(&#acc, &#a) }
    });
    quote! {
        let __prod = #product_expr;
        __acc = nabla::scalar::math_utils::add(&__acc, &__prod);
    }
}

fn codegen_fallback(input: &EinsumInput) -> Result<TokenStream2> {
    if input.rhs_terms.len() >= 3 {
        return codegen_sequential_contraction(input);
    }

    let rhs = &input.rhs_terms;
    let free_names = name_set(&input.output_indices);
    let (_, contraction_indices) = collect_non_free(rhs, &free_names);
    let tensor_bindings = dedup_bindings(rhs);
    let descs = rhs_descriptors(rhs);
    let desc_refs: Vec<(&Ident, &[Ident], bool)> =
        descs.iter().map(|(b, i, o)| (b, *i, *o)).collect();

    let step = build_accumulate_step(rhs);
    let loops = build_tiled_loops(
        &contraction_indices,
        |ci| find_dim_expr(ci, &desc_refs, None),
        step,
    )?;
    let output = emit_output_tensor(
        &input.output_indices,
        |fi| find_dim_expr(fi, &desc_refs, Some(0)),
        &loops,
        false,
    )?;

    Ok(quote! { { #(#tensor_bindings)* #output } })
}

fn dedup_bindings(terms: &[IndexedTensor]) -> Vec<TokenStream2> {
    let mut seen: HashSet<String> = HashSet::new();
    terms
        .iter()
        .filter_map(|t| {
            if seen.insert(t.name.to_string()) {
                let name = &t.name;
                let binding = Ident::new(&format!("__{name}"), name.span());
                Some(quote! { let #binding = &#name; })
            } else {
                None
            }
        })
        .collect()
}

fn compute_contracted<'a>(
    left: &'a HashSet<String>,
    right: &'a HashSet<String>,
    output_names: &HashSet<String>,
    remaining: &HashSet<String>,
) -> HashSet<String> {
    left.intersection(right)
        .filter(|idx| !output_names.contains(*idx) && !remaining.contains(*idx))
        .cloned().collect()
}

fn greedy_contraction_order(
    terms: &[IndexedTensor],
    output_indices: &[Ident],
) -> Vec<(usize, usize)> {
    let output_names: HashSet<String> = output_indices.iter().map(Ident::to_string).collect();
    let mut slot_indices: Vec<HashSet<String>> = terms.iter()
        .map(|t| t.indices.iter().map(Ident::to_string).collect()).collect();
    let mut slot_alive: Vec<bool> = vec![true; terms.len()];
    let mut order: Vec<(usize, usize)> = Vec::new();

    for _ in 0..terms.len() - 1 {
        let alive: Vec<usize> = (0..slot_indices.len()).filter(|&i| slot_alive[i]).collect();
        let mut best = (alive[0], alive[1]);
        let mut best_cost = usize::MAX;

        for i in 0..alive.len() {
            for j in (i + 1)..alive.len() {
                let (li, ri) = (alive[i], alive[j]);
                let remaining: HashSet<String> = alive.iter()
                    .filter(|&&k| k != li && k != ri)
                    .flat_map(|&k| slot_indices[k].iter().cloned()).collect();
                let contracted = compute_contracted(&slot_indices[li], &slot_indices[ri], &output_names, &remaining);
                let cost = slot_indices[li].union(&slot_indices[ri])
                    .filter(|idx| !contracted.contains(*idx)).count();
                if cost < best_cost { best_cost = cost; best = (li, ri); }
            }
        }

        let (li, ri) = best;
        order.push(best);

        let remaining: HashSet<String> = (0..slot_indices.len())
            .filter(|&k| slot_alive[k] && k != li && k != ri)
            .flat_map(|k| slot_indices[k].iter().cloned()).collect();
        let contracted = compute_contracted(&slot_indices[li], &slot_indices[ri], &output_names, &remaining);
        let merged: HashSet<String> = slot_indices[li].union(&slot_indices[ri])
            .filter(|idx| !contracted.contains(*idx)).cloned().collect();

        slot_alive[li] = false;
        slot_alive[ri] = false;
        slot_indices.push(merged);
        slot_alive.push(true);
    }

    order
}

struct IntermediateTensor {
    binding: Ident,
    indices: Vec<Ident>,
    is_original: bool,
}

fn codegen_sequential_contraction(input: &EinsumInput) -> Result<TokenStream2> {
    let output_names: HashSet<String> = input.output_indices.iter().map(Ident::to_string).collect();
    let contraction_order = greedy_contraction_order(&input.rhs_terms, &input.output_indices);
    let n_terms = input.rhs_terms.len();
    let n_steps = contraction_order.len();

    let slot_name = |idx: usize| -> String {
        if idx < n_terms { input.rhs_terms[idx].name.to_string() }
        else { format!("tmp_{}", idx - n_terms) }
    };
    let path_desc: String = contraction_order.iter().enumerate()
        .map(|(step, &(li, ri))| format!("step {step}: {} x {}", slot_name(li), slot_name(ri)))
        .collect::<Vec<_>>().join(", ");

    let tensor_bindings = dedup_bindings(&input.rhs_terms);

    let mut slots: HashMap<usize, IntermediateTensor> = input.rhs_terms.iter().enumerate()
        .map(|(i, t)| (i, IntermediateTensor {
            binding: Ident::new(&format!("__{}", t.name), t.name.span()),
            indices: t.indices.clone(), is_original: true,
        })).collect();

    let mut intermediate_stmts: Vec<TokenStream2> = Vec::new();
    let mut next_slot = n_terms;

    for (step_idx, &(li, ri)) in contraction_order.iter().enumerate() {
        let (acc, rhs_t) = match (slots.remove(&li), slots.remove(&ri)) {
            (Some(l), Some(r)) => (l, r),
            _ => return Err(Error::new(Span::call_site(), "einsum!: internal error in contraction order")),
        };

        let remaining: HashSet<String> = slots.values()
            .flat_map(|t| t.indices.iter()).map(Ident::to_string).collect();
        let left_set: HashSet<String> = acc.indices.iter().map(Ident::to_string).collect();
        let right_set: HashSet<String> = rhs_t.indices.iter().map(Ident::to_string).collect();
        let contracted: HashSet<String> = left_set.intersection(&right_set)
            .filter(|idx| !output_names.contains(*idx) && !remaining.contains(*idx))
            .cloned().collect();

        let all_indices = acc.indices.iter().chain(rhs_t.indices.iter());
        let (mut step_out, mut contr_dedup) = (Vec::new(), Vec::new());
        let (mut seen_out, mut seen_c) = (HashSet::new(), HashSet::new());
        for idx in all_indices {
            let s = idx.to_string();
            if contracted.contains(&s) {
                if seen_c.insert(s) { contr_dedup.push(idx); }
            } else if seen_out.insert(s) {
                step_out.push(idx.clone());
            }
        }

        let l_access = element_access(&acc.binding, &acc.indices, acc.is_original);
        let r_access = element_access(&rhs_t.binding, &rhs_t.indices, rhs_t.is_original);
        let step_stmt = quote! {
            let __prod = nabla::scalar::math_utils::mul(&#l_access, &#r_access);
            __acc = nabla::scalar::math_utils::add(&__acc, &__prod);
        };

        let pair: Vec<(&Ident, &[Ident], bool)> = vec![
            (&acc.binding, acc.indices.as_slice(), acc.is_original),
            (&rhs_t.binding, rhs_t.indices.as_slice(), rhs_t.is_original),
        ];
        let loops = build_tiled_loops(&contr_dedup, |ci| find_dim_expr(ci, &pair, None), step_stmt)?;

        if step_idx == n_steps - 1 {
            intermediate_stmts.push(emit_output_tensor(
                &input.output_indices, |fi| find_dim_expr(fi, &pair, None), &loops, true,
            )?);
        } else {
            let tmp_name = Ident::new(&format!("__tmp_{step_idx}"), Span::call_site());
            if step_out.is_empty() {
                intermediate_stmts.push(quote! {
                    let mut __acc = nabla::scalar::math_utils::zero::<_>();
                    #loops
                    let #tmp_name = __acc;
                });
            } else {
                let shape_exprs: Vec<TokenStream2> = step_out.iter()
                    .map(|fi| find_dim_expr(fi, &pair, None)).collect::<Result<_>>()?;
                let idx_bindings: Vec<TokenStream2> = step_out.iter().enumerate()
                    .map(|(n, fi)| quote! { let #fi = __idx[#n]; }).collect();
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
            slots.insert(next_slot, IntermediateTensor {
                binding: tmp_name, indices: step_out, is_original: false,
            });
            next_slot += 1;
        }
    }

    let final_expr = if input.output_indices.is_empty() { quote! { __acc } }
                     else { quote! { __final } };

    Ok(quote! {{
        #[allow(unused_variables)]
        let __einsum_contraction_path: &str = #path_desc;
        #(#tensor_bindings)*
        #(#intermediate_stmts)*
        #final_expr
    }})
}
