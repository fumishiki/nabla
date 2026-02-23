use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Error, Expr, Result,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Comma,
};

mod einsum;

// Parse `[[e00, e01], [e10, e11], ...]`.
struct MatInput {
    rows: Vec<Vec<Expr>>,
}

impl Parse for MatInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let rows = input
            .parse_terminated(RowExprs::parse, Comma)?
            .into_iter()
            .map(|r| r.elems)
            .collect();
        Ok(Self { rows })
    }
}

struct RowExprs {
    elems: Vec<Expr>,
}

impl Parse for RowExprs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let inner;
        syn::bracketed!(inner in input);
        let elems: Punctuated<Expr, Comma> = inner.parse_terminated(Expr::parse, Comma)?;
        Ok(RowExprs {
            elems: elems.into_iter().collect(),
        })
    }
}

/// Matrix literal macro — Julia `[1 2; 3 4]` equivalent.
///
/// # Example
///
/// ```rust
/// # mod nabla {
/// #    pub mod tensor {
/// #        pub struct Tensor;
/// #        impl Tensor {
/// #            pub fn from_fn<F, T>(_: usize, _: usize, _: F) -> Self
/// #            where
/// #                F: FnMut(usize, usize) -> T,
/// #            {
/// #                Self
/// #            }
/// #        }
/// #    }
/// # }
/// use nabla_macros::mat;
/// let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
/// ```
///
/// Expands to a `nabla::tensor::Tensor` constructed via `from_fn`.
#[proc_macro]
pub fn mat(input: TokenStream) -> TokenStream {
    match mat_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Einstein summation macro — Julia `@tullio` / `@einsum` equivalent.
///
/// Computes Einstein summation over 2-D tensors. Free indices appear on the
/// LHS; contraction indices appear only on the RHS and are summed over.
///
/// # Examples
///
/// ```rust,ignore
/// // Matrix multiply: C = A * B
/// let c = einsum!(c[i,j] = a[i,k] * b[k,j]);
///
/// // Matrix-vector multiply (column vector output)
/// let y = einsum!(y[i] = a[i,j] * x[j]);
///
/// // Trace (scalar output)
/// let s: f64 = einsum!(s = a[i,i]);
///
/// // Hadamard product
/// let c = einsum!(c[i,j] = a[i,j] * b[i,j]);
/// ```
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    match einsum::einsum_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn mat_impl(input: TokenStream2) -> Result<TokenStream2> {
    let MatInput { rows } = syn::parse2(input)?;

    if rows.is_empty() {
        return Err(Error::new(
            proc_macro2::Span::call_site(),
            "mat![] requires at least one row",
        ));
    }

    let ncols = rows[0].len();

    if ncols == 0 {
        return Err(Error::new(
            proc_macro2::Span::call_site(),
            "mat![] requires at least one column",
        ));
    }

    for (idx, row) in rows.iter().enumerate() {
        if row.len() != ncols {
            return Err(Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "mat![] row {} has {} columns, expected {}",
                    idx,
                    row.len(),
                    ncols
                ),
            ));
        }
    }

    let nrows = rows.len();

    let row_tokens: Vec<TokenStream2> = rows
        .iter()
        .map(|row| {
            let elems = row.iter();
            quote! { [#(#elems),*] }
        })
        .collect();

    Ok(quote! {
        {
            const ROWS: usize = #nrows;
            const COLS: usize = #ncols;
            let data: [[_; COLS]; ROWS] = [#(#row_tokens),*];
            nabla::tensor::Tensor::from_fn(ROWS, COLS, |i, j| data[i][j])
        }
    })
}
