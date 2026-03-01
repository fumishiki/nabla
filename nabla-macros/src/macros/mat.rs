//! Matrix and construction literal macros.
//!
//! - `mat!` — Julia `[1 2; 3 4]` equivalent
//! - `block!` — block matrix construction via hcat/vcat
//! - `named!` — named tuple construction
//! - `generated!` — const-generic specialization
//! - `axis!` — zero-sized marker types for named tensor axes
//! - `named_zeros!` — typed axis tensor constructor

use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    Error, Expr, Result,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Comma,
};

fn err(msg: impl std::fmt::Display) -> Error {
    Error::new(Span::call_site(), msg)
}


pub(crate) struct MatInput {
    pub(crate) rows: Vec<Vec<Expr>>,
    pub(crate) type_prefix: Option<syn::Type>,
}

impl Parse for MatInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        // Optional type prefix: `f64:` or `f32:` etc.
        let type_prefix = if input.peek(syn::Ident) && input.peek2(syn::Token![:]) {
            let ty: syn::Type = input.parse()?;
            input.parse::<syn::Token![:]>()?;
            Some(ty)
        } else {
            None
        };

        if input.peek(syn::token::Bracket) {
            let rows = input
                .parse_terminated(RowExprs::parse, Comma)?
                .into_iter()
                .map(|r| r.elems)
                .collect();
            Ok(Self { rows, type_prefix })
        } else {
            let mut mi = parse_semicolon_rows(input)?;
            mi.type_prefix = type_prefix;
            Ok(mi)
        }
    }
}

fn parse_semicolon_rows(input: ParseStream<'_>) -> Result<MatInput> {
    let mut rows = Vec::new();
    while !input.is_empty() {
        let mut elems = Vec::new();
        elems.push(input.parse::<Expr>()?);
        while input.peek(Comma) {
            input.parse::<Comma>()?;
            if input.is_empty() || input.peek(syn::Token![;]) {
                break;
            }
            elems.push(input.parse::<Expr>()?);
        }
        rows.push(elems);
        if input.peek(syn::Token![;]) {
            input.parse::<syn::Token![;]>()?;
        }
    }
    if rows.is_empty() {
        return Err(err("mat![] requires at least one row"));
    }
    Ok(MatInput { rows, type_prefix: None })
}

struct RowExprs {
    elems: Vec<Expr>,
}

impl Parse for RowExprs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let inner;
        syn::bracketed!(inner in input);
        Ok(RowExprs {
            elems: inner
                .parse_terminated(Expr::parse, Comma)?
                .into_iter()
                .collect(),
        })
    }
}

pub(crate) fn mat_impl(input: TokenStream2) -> Result<TokenStream2> {
    let mi: MatInput = syn::parse2(input)?;

    let first_row = mi.rows
        .first()
        .ok_or_else(|| err("mat![] requires at least one row"))?;
    let ncols = first_row.len();
    if ncols == 0 {
        return Err(err("mat![] requires at least one column"));
    }

    for (idx, row) in mi.rows.iter().enumerate() {
        if row.len() != ncols {
            return Err(err(format!(
                "mat![] row {} has {} columns, expected {}",
                idx, row.len(), ncols
            )));
        }
    }

    let nrows = mi.rows.len();
    let row_tokens: Vec<TokenStream2> = mi.rows
        .iter()
        .map(|row| {
            let elems = row.iter();
            quote! { [#(#elems),*] }
        })
        .collect();

    let array_type = match &mi.type_prefix {
        Some(ty) => quote! { [[#ty; COLS]; ROWS] },
        None => quote! { [[_; COLS]; ROWS] },
    };

    Ok(quote! {
        {
            const ROWS: usize = #nrows;
            const COLS: usize = #ncols;
            let data: #array_type = [#(#row_tokens),*];
            nabla::tensor::Tensor::from_fn(ROWS, COLS, |i, j| data[i][j])
        }
    })
}


pub(crate) fn block_impl(input: TokenStream2) -> Result<TokenStream2> {
    let outer: syn::ExprArray = syn::parse2(input)?;

    if outer.elems.is_empty() {
        return Err(err("block! requires at least one row"));
    }

    let row_exprs: Vec<TokenStream2> = outer
        .elems
        .iter()
        .map(|elem| {
            let Expr::Array(row) = elem else {
                return Err(Error::new_spanned(
                    elem,
                    "block! rows must be array literals: [A, B, ...]",
                ));
            };
            if row.elems.is_empty() {
                return Err(Error::new_spanned(row, "block! rows must not be empty"));
            }
            if row.elems.len() == 1 {
                let single = &row.elems[0];
                Ok(quote! { #single })
            } else {
                let elems = row.elems.iter();
                Ok(quote! { nabla::hcat!(#(#elems),*) })
            }
        })
        .collect::<Result<Vec<_>>>()?;

    if row_exprs.len() == 1 {
        return row_exprs
            .into_iter()
            .next()
            .ok_or_else(|| err("block! requires at least one row"));
    }
    Ok(quote! { nabla::vcat!(#(#row_exprs),*) })
}


struct NamedField {
    name: Ident,
    ty: syn::Type,
    value: Expr,
}

impl Parse for NamedField {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<syn::Token![:]>()?;
        let ty: syn::Type = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let value: Expr = input.parse()?;
        Ok(NamedField { name, ty, value })
    }
}

struct NamedInput {
    fields: Vec<NamedField>,
}

impl Parse for NamedInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        Ok(NamedInput {
            fields: input
                .parse_terminated(NamedField::parse, Comma)?
                .into_iter()
                .collect(),
        })
    }
}

pub(crate) fn named_impl(input: TokenStream2) -> Result<TokenStream2> {
    let NamedInput { fields } = syn::parse2(input)?;
    if fields.is_empty() {
        return Err(err("named!() requires at least one field"));
    }

    let field_names: Vec<&Ident> = fields.iter().map(|f| &f.name).collect();
    let field_types: Vec<&syn::Type> = fields.iter().map(|f| &f.ty).collect();
    let field_values: Vec<&Expr> = fields.iter().map(|f| &f.value).collect();

    Ok(quote! {{
        #[derive(Debug, Clone, PartialEq)]
        struct __NamedTuple {
            #( pub #field_names: #field_types ),*
        }
        __NamedTuple {
            #( #field_names: #field_values ),*
        }
    }})
}


pub(crate) fn generated_impl(input: TokenStream2) -> Result<TokenStream2> {
    let func: syn::ItemFn = syn::parse2(input)?;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    Ok(quote! {
        #[inline]
        #vis #sig #block
    })
}


pub(crate) fn axis_impl(input: TokenStream2) -> Result<TokenStream2> {
    let names: Punctuated<Ident, Comma> =
        syn::parse::Parser::parse2(Punctuated::<Ident, Comma>::parse_terminated, input)?;
    if names.is_empty() {
        return Err(err("axis!() requires at least one identifier"));
    }
    let structs = names.iter().map(|name| {
        quote! { pub struct #name; }
    });
    Ok(quote! { #(#structs)* })
}


struct NamedZerosInput {
    axes: Vec<Ident>,
    dims: Vec<Expr>,
}

impl Parse for NamedZerosInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let axes: Vec<Ident> = Punctuated::<Ident, Comma>::parse_separated_nonempty(input)?
            .into_iter()
            .collect();
        input.parse::<syn::Token![;]>()?;
        let dims: Vec<Expr> = Punctuated::<Expr, Comma>::parse_separated_nonempty(input)?
            .into_iter()
            .collect();
        if axes.len() != dims.len() {
            return Err(err(format!(
                "named_zeros!: {} axes but {} dimensions",
                axes.len(),
                dims.len()
            )));
        }
        if axes.len() != 2 {
            return Err(err(
                "named_zeros! requires exactly 2 axes (row, col) for a 2-D tensor",
            ));
        }
        Ok(NamedZerosInput { axes, dims })
    }
}

pub(crate) fn named_zeros_impl(input: TokenStream2) -> Result<TokenStream2> {
    let NamedZerosInput { axes, dims } = syn::parse2(input)?;
    let ax0 = &axes[0];
    let ax1 = &axes[1];
    let d0 = &dims[0];
    let d1 = &dims[1];
    Ok(quote! {
        nabla::tensor::Tensor::zeros(#d0, #d1).with_axes::<(#ax0, #ax1)>()
    })
}
