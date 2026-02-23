use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::{
    Error, Expr, ExprBinary, ExprMethodCall, ExprPath, ExprUnary, Result,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Comma,
};

mod einsum;
mod stencil;

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

/// Broadcast-all macro — Julia `@.` equivalent.
///
/// Automatically lifts a scalar expression into an element-wise tensor
/// operation.  Every occurrence of the tensor variables is replaced with
/// `var.get(__r, __c)`, and the whole expression is wrapped in `from_fn`.
///
/// # Syntax
///
/// ```rust,ignore
/// // Single tensor
/// let y: Tensor<f64> = bcast_all!(sin(x).powf(2.0); x);
///
/// // Multiple tensors
/// let z: Tensor<f64> = bcast_all!(x * y + x.sin(); x, y);
/// ```
///
/// The last part (after `;`) lists the tensor variables that should be
/// element-wise broadcast.
#[proc_macro]
pub fn bcast_all(input: TokenStream) -> TokenStream {
    match bcast_all_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

// ── bcast_all! implementation ────────────────────────────────────────────────

struct BcastAllInput {
    body: Expr,
    tensors: Vec<Ident>,
}

impl Parse for BcastAllInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let body: Expr = input.parse()?;
        input.parse::<syn::Token![;]>()?;
        let vars: Punctuated<Ident, Comma> = input.parse_terminated(Ident::parse, Comma)?;
        let tensors: Vec<_> = vars.into_iter().collect();
        if tensors.is_empty() {
            return Err(Error::new(
                Span::call_site(),
                "bcast_all! needs at least one tensor variable after `;`",
            ));
        }
        Ok(BcastAllInput { body, tensors })
    }
}

/// Check if an expression references any tensor variable.
fn contains_tensor(expr: &Expr, tensor_names: &[String]) -> bool {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            tensor_names.contains(&path.segments[0].ident.to_string())
        }
        Expr::Binary(ExprBinary { left, right, .. }) => {
            contains_tensor(left, tensor_names) || contains_tensor(right, tensor_names)
        }
        Expr::Unary(ExprUnary { expr: inner, .. }) => contains_tensor(inner, tensor_names),
        Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
            contains_tensor(receiver, tensor_names)
                || args.iter().any(|a| contains_tensor(a, tensor_names))
        }
        Expr::Call(ec) => ec.args.iter().any(|a| contains_tensor(a, tensor_names)),
        Expr::Paren(ep) => contains_tensor(&ep.expr, tensor_names),
        Expr::Reference(er) => contains_tensor(&er.expr, tensor_names),
        _ => false,
    }
}

/// Rewrite expression to **tensor-level** operations (GPU-compatible).
///
/// Tensor variables stay as-is; binary ops become tensor ops;
/// method calls on tensor expressions remain tensor methods.
/// This ensures GPU backends dispatch kernel calls, not scalar readbacks.
fn lift_expr(expr: &Expr, tensor_names: &[String]) -> TokenStream2 {
    match expr {
        // Tensor variable — emit as-is (no .get()!)
        Expr::Path(_) => expr.to_token_stream(),

        // Binary op: tensor OP tensor → tensor method, tensor OP scalar → tensor method
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l_has = contains_tensor(left, tensor_names);
            let r_has = contains_tensor(right, tensor_names);
            let l = lift_expr(left, tensor_names);
            let r = lift_expr(right, tensor_names);

            if l_has && r_has {
                // tensor OP tensor
                match op {
                    syn::BinOp::Mul(_) => quote! { (#l).mul_elem(&#r) },
                    syn::BinOp::Div(_) => quote! { (#l).div_elem(&#r) },
                    syn::BinOp::Add(_) => quote! { (&#l + &#r) },
                    syn::BinOp::Sub(_) => quote! { (&#l - &#r) },
                    _ => quote! { (#l #op #r) },
                }
            } else if l_has {
                // tensor OP scalar
                match op {
                    syn::BinOp::Mul(_) => quote! { (&#l * #r) },
                    syn::BinOp::Add(_) | syn::BinOp::Sub(_) => quote! { (#l #op #r) },
                    _ => quote! { (#l #op #r) },
                }
            } else if r_has {
                // scalar OP tensor
                match op {
                    syn::BinOp::Mul(_) => quote! { (&#r * #l) },
                    _ => quote! { (#l #op #r) },
                }
            } else {
                // scalar OP scalar — pass through
                quote! { (#l #op #r) }
            }
        }

        // Unary: -tensor → (-&tensor)
        Expr::Unary(ExprUnary {
            op, expr: inner, ..
        }) => {
            let i = lift_expr(inner, tensor_names);
            if contains_tensor(inner, tensor_names) {
                match op {
                    syn::UnOp::Neg(_) => quote! { (-&#i) },
                    _ => quote! { (#op #i) },
                }
            } else {
                quote! { (#op #i) }
            }
        }

        // Method call: tensor.sin() → tensor.sin() (Backend-dispatched)
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let recv = lift_expr(receiver, tensor_names);
            let rewritten_args: Vec<_> = args.iter().map(|a| lift_expr(a, tensor_names)).collect();
            quote! { #recv.#method(#(#rewritten_args),*) }
        }

        // Function call
        Expr::Call(ec) => {
            let func = &ec.func;
            let rewritten_args: Vec<_> =
                ec.args.iter().map(|a| lift_expr(a, tensor_names)).collect();
            quote! { #func(#(#rewritten_args),*) }
        }

        Expr::Paren(ep) => {
            let inner = lift_expr(&ep.expr, tensor_names);
            quote! { (#inner) }
        }

        Expr::Reference(er) => {
            let inner = lift_expr(&er.expr, tensor_names);
            if er.mutability.is_some() {
                quote! { &mut #inner }
            } else {
                quote! { &#inner }
            }
        }

        // Literal / other — pass through
        _ => expr.to_token_stream(),
    }
}

fn bcast_all_impl(input: TokenStream2) -> Result<TokenStream2> {
    let BcastAllInput { body, tensors } = syn::parse2(input)?;
    let tensor_names: Vec<String> = tensors.iter().map(|t| t.to_string()).collect();
    let lifted = lift_expr(&body, &tensor_names);

    let first = &tensors[0];
    let shape_checks: Vec<TokenStream2> = tensors
        .iter()
        .skip(1)
        .map(|t| {
            quote! {
                assert_eq!(#first.shape(), #t.shape(), "bcast_all!: shape mismatch");
            }
        })
        .collect();

    Ok(quote! {{
        #(#shape_checks)*
        #lifted
    }})
}

// ── named! (A12 Named Tuple) ─────────────────────────────────────────────────

/// Named tuple macro — Julia `(a=1, b=2.0)` equivalent.
///
/// Generates an anonymous struct with the specified fields and returns an
/// instance.  The struct derives `Debug`, `Clone`, and `PartialEq`.
///
/// # Examples
///
/// ```rust,ignore
/// let p = named!(x: f64 = 1.0, y: f64 = 2.0);
/// assert_eq!(p.x, 1.0);
/// assert_eq!(p.y, 2.0);
/// ```
#[proc_macro]
pub fn named(input: TokenStream) -> TokenStream {
    match named_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
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
        let fields: Punctuated<NamedField, Comma> =
            input.parse_terminated(NamedField::parse, Comma)?;
        Ok(NamedInput {
            fields: fields.into_iter().collect(),
        })
    }
}

fn named_impl(input: TokenStream2) -> Result<TokenStream2> {
    let NamedInput { fields } = syn::parse2(input)?;
    if fields.is_empty() {
        return Err(Error::new(
            Span::call_site(),
            "named!() requires at least one field",
        ));
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

// ── generated! (D4 @generated) ───────────────────────────────────────────────

/// Compile-time specialization macro — Julia `@generated` equivalent.
///
/// Generates a function with a `match` on a const generic parameter,
/// allowing different implementations per value.
///
/// # Syntax
///
/// ```rust,ignore
/// generated! {
///     fn det<const N: usize>(vals: &[f64; N]) -> f64 {
///         match N {
///             1 => vals[0],
///             2 => vals[0]*vals[3] - vals[1]*vals[2],
///             _ => unimplemented!()
///         }
///     }
/// }
/// ```
///
/// Expands to a regular function containing the match expression.
#[proc_macro]
pub fn generated(input: TokenStream) -> TokenStream {
    // Pass through: the match-on-const-generic is already valid Rust.
    // This macro validates and emits the function with #[inline] added.
    match generated_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generated_impl(input: TokenStream2) -> Result<TokenStream2> {
    let func: syn::ItemFn = syn::parse2(input)?;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    Ok(quote! {
        #[inline]
        #vis #sig #block
    })
}

/// Stencil macro — Julia `@tullio` offset-indexing equivalent.
///
/// Generates bounds-checked interior iteration over a 2-D tensor with
/// offset indexing (e.g. `a[i-1, j+1]`).
///
/// # Examples
///
/// ```rust,ignore
/// // 5-point Laplacian stencil
/// let out = stencil!(out[i,j] = -4.0*a[i,j] + a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1]);
/// ```
#[proc_macro]
pub fn stencil(input: TokenStream) -> TokenStream {
    match stencil::stencil_impl(input.into()) {
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
