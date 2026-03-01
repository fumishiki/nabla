//! nabla-macros — Layer 1: Notation proc macros for concise DSL syntax.
//!
//! All `#[proc_macro]` and `#[proc_macro_attribute]` entry points live here.
//! Implementation logic is delegated to submodules.

use proc_macro::TokenStream;

mod fusion;
mod macros;

fn expand<F>(input: TokenStream, f: F) -> TokenStream
where
    F: FnOnce(proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream>,
{
    match f(input.into()) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error(),
    }
    .into()
}

// ── mat! ────────────────────────────────────────────────────────────────────

/// Matrix literal macro — Julia `[1 2; 3 4]` equivalent.
///
/// Expands to a `nabla::tensor::Tensor` constructed via `from_fn`.
///
/// # Examples
///
/// ```rust,ignore
/// let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
/// ```
#[proc_macro]
pub fn mat(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::mat_impl)
}

// ── einsum! ─────────────────────────────────────────────────────────────────

/// Einstein summation macro.
///
/// # Examples
///
/// ```rust,ignore
/// let c = einsum!(c[i,j] = a[i,k] * b[k,j]);
/// ```
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    expand(input, macros::einsum::einsum_impl)
}

// ── fuse! ───────────────────────────────────────────────────────────────────

/// Fused broadcast macro — Julia `@.` equivalent.
///
/// Lifts a scalar expression into an element-wise tensor operation with
/// multi-level fusion (L1 element-wise, L3 GEMM+activation, L4 map-reduce).
///
/// # Examples
///
/// ```rust,ignore
/// let y = fuse!(sin(x).powf(2.0); x);
/// let z = fuse!(x * y + x.sin(); x, y);
/// ```
#[proc_macro]
pub fn fuse(input: TokenStream) -> TokenStream {
    expand(input, macros::fuse::fuse_impl)
}

// ── mega_fuse! ──────────────────────────────────────────────────────────────

/// Mega-fused element-wise macro — multiple ops as a single GPU kernel launch.
///
/// # Examples
///
/// ```rust,ignore
/// let (y, z) = mega_fuse!(
///     a.exp().sin();
///     b.tanh() + a;
///     inputs: a, b
/// );
/// ```
#[proc_macro]
pub fn mega_fuse(input: TokenStream) -> TokenStream {
    expand(input, macros::fuse::mega_fuse_impl)
}

// ── stencil! ────────────────────────────────────────────────────────────────

/// Stencil macro — Julia `@tullio` offset-indexing equivalent.
///
/// # Examples
///
/// ```rust,ignore
/// let out = stencil!(out[i,j] = -4.0*a[i,j] + a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1]);
/// ```
#[proc_macro]
pub fn stencil(input: TokenStream) -> TokenStream {
    expand(input, macros::stencil::stencil_impl)
}

// ── named! ──────────────────────────────────────────────────────────────────

/// Named tuple macro — Julia `(a=1, b=2.0)` equivalent.
///
/// # Examples
///
/// ```rust,ignore
/// let p = named!(x: f64 = 1.0, y: f64 = 2.0);
/// assert_eq!(p.x, 1.0);
/// ```
#[proc_macro]
pub fn named(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::named_impl)
}

// ── generated! ──────────────────────────────────────────────────────────────

/// Compile-time specialization macro — Julia `@generated` equivalent.
///
/// # Examples
///
/// ```rust,ignore
/// generated! {
///     fn det<const N: usize>(vals: &[f64; N]) -> f64 {
///         match N { 1 => vals[0], _ => unimplemented!() }
///     }
/// }
/// ```
#[proc_macro]
pub fn generated(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::generated_impl)
}

// ── axis! ───────────────────────────────────────────────────────────────────

/// Declare zero-sized marker types for named tensor axes.
///
/// # Examples
///
/// ```rust,ignore
/// axis!(Batch, Seq, Hidden);
/// ```
#[proc_macro]
pub fn axis(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::axis_impl)
}

// ── named_zeros! ────────────────────────────────────────────────────────────

/// Construct a zero-filled tensor with named axes.
///
/// # Examples
///
/// ```rust,ignore
/// let t = named_zeros!(Batch, Hidden; 32, 768);
/// ```
#[proc_macro]
pub fn named_zeros(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::named_zeros_impl)
}

// ── #[nabla_grad] ───────────────────────────────────────────────────────────

/// Source-transform AD: lifts `fn f(x: T) -> T` to also generate `fn f_grad(x: T) -> (T, T)`.
///
/// # Examples
///
/// ```rust,ignore
/// #[nabla_grad]
/// fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
/// let (val, grad) = sigmoid_grad(0.0);
/// ```
#[proc_macro_attribute]
pub fn nabla_grad(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand(item, macros::grad::nabla_grad_impl)
}

// ── block! ──────────────────────────────────────────────────────────────────

/// Block matrix construction: `block![[A, B], [C, D]]`.
///
/// # Examples
///
/// ```rust,ignore
/// let m = block![[a, b], [c, d]];
/// ```
#[proc_macro]
pub fn block(input: TokenStream) -> TokenStream {
    expand(input, macros::mat::block_impl)
}

// ── sym! ───────────────────────────────────────────────────────────────────

/// Symbolic CAS expression macro — natural math notation.
///
/// Parses mathematical expressions and generates `nabla::cas::Expr` construction
/// code. Supports `+`, `-`, `*`, `/`, `^` (power), unary `-`, parentheses, and
/// functions: `sin`, `cos`, `exp`, `ln`, `tanh`, `sqrt`, `abs`.
///
/// # Examples
///
/// ```rust,ignore
/// let f = sym!(sin(x^2));
/// let g = sym!(x * y + cos(x));
/// let h = sym!(exp(x) / (1 + x^2));
/// ```
#[proc_macro]
pub fn sym(input: TokenStream) -> TokenStream {
    expand(input, macros::sym::sym_impl)
}
