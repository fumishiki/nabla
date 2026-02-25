use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::{
    Error, Expr, ExprBinary, ExprMethodCall, ExprPath, ExprUnary, Result,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Comma,
};

use egg::{AstSize, Id, RecExpr, Rewrite, Runner, Symbol, define_language, rewrite};
use ordered_float::OrderedFloat;

mod einsum;
mod stencil;

// ── EqSat (egg) for fuse! compile-time simplification ──────────────────────

define_language! {
    enum FuseExpr {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Sub([Id; 2]),
        "/" = Div([Id; 2]),
        "neg" = Neg([Id; 1]),
        "exp" = Exp([Id; 1]),
        "ln"  = Ln([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "abs" = Abs([Id; 1]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "tanh" = Tanh([Id; 1]),
        "recip" = Recip([Id; 1]),
        "pow" = Pow([Id; 2]),
        Num(OrderedFloat<f64>),
        Symbol(Symbol),
    }
}

fn fuse_rules() -> Vec<Rewrite<FuseExpr, ()>> {
    vec![
        rewrite!("add-zero-r"; "(+ ?x 0)"  => "?x"),
        rewrite!("add-zero-l"; "(+ 0 ?x)"  => "?x"),
        rewrite!("mul-one-r";  "(* ?x 1)"  => "?x"),
        rewrite!("mul-one-l";  "(* 1 ?x)"  => "?x"),
        rewrite!("mul-zero-r"; "(* ?x 0)"  => "0"),
        rewrite!("mul-zero-l"; "(* 0 ?x)"  => "0"),
        rewrite!("double-neg"; "(neg (neg ?x))" => "?x"),
        rewrite!("exp-ln";     "(exp (ln ?x))"  => "?x"),
        rewrite!("ln-exp";     "(ln (exp ?x))"  => "?x"),
        rewrite!("pow-zero";   "(pow ?x 0)" => "1"),
        rewrite!("pow-one";    "(pow ?x 1)" => "?x"),
        rewrite!("sub-self";   "(- ?x ?x)"  => "0"),
        rewrite!("div-self";   "(/ ?x ?x)"  => "1"),
        rewrite!("abs-abs";    "(abs (abs ?x))" => "(abs ?x)"),
        rewrite!("sqrt-pow2";  "(pow (sqrt ?x) 2)" => "(abs ?x)"),
    ]
}

// Store an opaque sub-expression as a unique symbol, avoiding separate name.clone().
fn add_opaque_symbol(
    expr: &Expr,
    rec: &mut RecExpr<FuseExpr>,
    sym_map: &mut Vec<(String, Expr)>,
) -> Id {
    let name = format!("__sym{}", sym_map.len());
    let id = rec.add(FuseExpr::Symbol(Symbol::from(&*name)));
    sym_map.push((name, expr.clone()));
    id
}

// syn::Expr → RecExpr<FuseExpr>, returning root Id. Opaque sub-exprs get unique symbols.
fn expr_to_egg(
    expr: &Expr,
    rec: &mut RecExpr<FuseExpr>,
    sym_map: &mut Vec<(String, Expr)>,
) -> Id {
    match expr {
        // Literal numbers
        Expr::Lit(syn::ExprLit { lit: syn::Lit::Float(f), .. }) => {
            if let Ok(v) = f.base10_parse::<f64>() {
                rec.add(FuseExpr::Num(OrderedFloat(v)))
            } else {
                add_opaque_symbol(expr, rec, sym_map)
            }
        }
        Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) => {
            if let Ok(v) = i.base10_parse::<i64>() {
                rec.add(FuseExpr::Num(OrderedFloat(v as f64)))
            } else {
                add_opaque_symbol(expr, rec, sym_map)
            }
        }

        // Binary ops
        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let l = expr_to_egg(left, rec, sym_map);
            let r = expr_to_egg(right, rec, sym_map);
            match op {
                syn::BinOp::Add(_) => rec.add(FuseExpr::Add([l, r])),
                syn::BinOp::Sub(_) => rec.add(FuseExpr::Sub([l, r])),
                syn::BinOp::Mul(_) => rec.add(FuseExpr::Mul([l, r])),
                syn::BinOp::Div(_) => rec.add(FuseExpr::Div([l, r])),
                _ => add_opaque_symbol(expr, rec, sym_map),
            }
        }

        // Unary neg
        Expr::Unary(ExprUnary { op: syn::UnOp::Neg(_), expr: inner, .. }) => {
            let child = expr_to_egg(inner, rec, sym_map);
            rec.add(FuseExpr::Neg([child]))
        }

        // Method calls: x.exp(), x.ln(), x.sin(), etc.
        Expr::MethodCall(mc) => {
            let method = mc.method.to_string();
            let recv = expr_to_egg(&mc.receiver, rec, sym_map);
            match (method.as_str(), mc.args.len()) {
                ("exp", 0)   => rec.add(FuseExpr::Exp([recv])),
                ("ln", 0)    => rec.add(FuseExpr::Ln([recv])),
                ("sqrt", 0)  => rec.add(FuseExpr::Sqrt([recv])),
                ("abs", 0)   => rec.add(FuseExpr::Abs([recv])),
                ("sin", 0)   => rec.add(FuseExpr::Sin([recv])),
                ("cos", 0)   => rec.add(FuseExpr::Cos([recv])),
                ("tanh", 0)  => rec.add(FuseExpr::Tanh([recv])),
                ("recip", 0) => rec.add(FuseExpr::Recip([recv])),
                ("neg", 0)   => rec.add(FuseExpr::Neg([recv])),
                ("powf", 1)  => {
                    let arg = expr_to_egg(&mc.args[0], rec, sym_map);
                    rec.add(FuseExpr::Pow([recv, arg]))
                }
                _ => add_opaque_symbol(expr, rec, sym_map),
            }
        }

        // Free function calls: exp(x), ln(x), sin(x), etc.
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                let arg = expr_to_egg(&ec.args[0], rec, sym_map);
                match fname.as_str() {
                    "exp"  => return rec.add(FuseExpr::Exp([arg])),
                    "ln"   => return rec.add(FuseExpr::Ln([arg])),
                    "sqrt" => return rec.add(FuseExpr::Sqrt([arg])),
                    "abs"  => return rec.add(FuseExpr::Abs([arg])),
                    "sin"  => return rec.add(FuseExpr::Sin([arg])),
                    "cos"  => return rec.add(FuseExpr::Cos([arg])),
                    "tanh" => return rec.add(FuseExpr::Tanh([arg])),
                    _ => {}
                }
            }
            add_opaque_symbol(expr, rec, sym_map)
        }

        // Paren
        Expr::Paren(ep) => expr_to_egg(&ep.expr, rec, sym_map),

        // Simple ident — use its name directly
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let ident = path.segments[0].ident.to_string();
            rec.add(FuseExpr::Symbol(Symbol::from(ident)))
        }

        // Anything else — opaque symbol
        _ => add_opaque_symbol(expr, rec, sym_map),
    }
}

// RecExpr<FuseExpr> → syn::Expr, resolving opaque symbols back to original expressions.
fn egg_to_expr(rec: &RecExpr<FuseExpr>, id: Id, sym_map: &[(String, Expr)]) -> Expr {
    let node = &rec[id];
    match node {
        FuseExpr::Num(n) => {
            let v = n.into_inner();
            // Emit integer if exact
            if v.fract() == 0.0 && v.abs() < i64::MAX as f64 {
                let iv = v as i64;
                syn::parse_quote!(#iv as _)
            } else {
                syn::parse_quote!(#v)
            }
        }
        FuseExpr::Symbol(s) => {
            let name = s.as_str();
            // Check if it's an opaque symbol
            for (sname, original) in sym_map {
                if sname == name {
                    return original.clone();
                }
            }
            // Regular variable ident
            let ident = Ident::new(name, Span::call_site());
            syn::parse_quote!(#ident)
        }
        FuseExpr::Add([a, b]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            let lb = egg_to_expr(rec, *b, sym_map);
            syn::parse_quote!((#la + #lb))
        }
        FuseExpr::Sub([a, b]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            let lb = egg_to_expr(rec, *b, sym_map);
            syn::parse_quote!((#la - #lb))
        }
        FuseExpr::Mul([a, b]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            let lb = egg_to_expr(rec, *b, sym_map);
            syn::parse_quote!((#la * #lb))
        }
        FuseExpr::Div([a, b]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            let lb = egg_to_expr(rec, *b, sym_map);
            syn::parse_quote!((#la / #lb))
        }
        FuseExpr::Neg([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.neg())
        }
        FuseExpr::Exp([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.exp())
        }
        FuseExpr::Ln([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.ln())
        }
        FuseExpr::Sqrt([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.sqrt())
        }
        FuseExpr::Abs([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.abs())
        }
        FuseExpr::Sin([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.sin())
        }
        FuseExpr::Cos([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.cos())
        }
        FuseExpr::Tanh([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.tanh())
        }
        FuseExpr::Recip([a]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            syn::parse_quote!(#la.recip())
        }
        FuseExpr::Pow([a, b]) => {
            let la = egg_to_expr(rec, *a, sym_map);
            let lb = egg_to_expr(rec, *b, sym_map);
            syn::parse_quote!(#la.powf(#lb))
        }
    }
}

// Run egg EqSat on an expression, returning the simplified syn::Expr.
fn eqsat_simplify(expr: &Expr) -> Expr {
    let mut rec = RecExpr::default();
    let mut sym_map = Vec::new();
    let _root = expr_to_egg(expr, &mut rec, &mut sym_map);

    let runner = Runner::<FuseExpr, ()>::default()
        .with_iter_limit(8)
        .with_node_limit(5_000)
        .with_expr(&rec)
        .run(&fuse_rules());

    let extractor = egg::Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);

    let best_root = Id::from(best.as_ref().len() - 1);
    egg_to_expr(&best, best_root, &sym_map)
}

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
        Ok(RowExprs {
            elems: inner
                .parse_terminated(Expr::parse, Comma)?
                .into_iter()
                .collect(),
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

/// Fused broadcast macro — Julia `@.` equivalent.
///
/// Automatically lifts a scalar expression into an element-wise tensor
/// operation.  Every occurrence of the tensor variables is replaced with
/// tensor-level operations, and the whole expression dispatches through
/// the backend (GPU-compatible).
///
/// # Syntax
///
/// ```rust,ignore
/// // Single tensor
/// let y: Tensor<f64> = fuse!(sin(x).powf(2.0); x);
///
/// // Multiple tensors
/// let z: Tensor<f64> = fuse!(x * y + x.sin(); x, y);
/// ```
///
/// The last part (after `;`) lists the tensor variables that should be
/// element-wise broadcast.
#[proc_macro]
pub fn fuse(input: TokenStream) -> TokenStream {
    match fuse_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

// ── fuse! implementation ─────────────────────────────────────────────────────

struct FuseInput {
    body: Expr,
    tensors: Vec<Ident>,
}

impl Parse for FuseInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let body: Expr = input.parse()?;
        input.parse::<syn::Token![;]>()?;
        let tensors: Vec<Ident> = input
            .parse_terminated(Ident::parse, Comma)?
            .into_iter()
            .collect();
        if tensors.is_empty() {
            return Err(Error::new(
                Span::call_site(),
                "fuse! needs at least one tensor variable after `;`",
            ));
        }
        Ok(FuseInput { body, tensors })
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
                    syn::BinOp::Mul(_) => quote! { (#l).emul(&#r) },
                    syn::BinOp::Div(_) => quote! { (#l).ediv(&#r) },
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

// Element-wise unary methods that operate per-element (fusible as scalar ops).
const ELEMENTWISE_UNARY: &[&str] = &[
    "exp", "ln", "log1p", "sin", "cos", "tanh", "sqrt", "abs", "recip", "erf", "ceil", "floor",
    "round", "neg",
];

// Element-wise unary methods that take one scalar arg (fusible as scalar ops).
const ELEMENTWISE_UNARY_ARG: &[&str] = &["powf"];

/// Check if the entire expression tree can be fused into a single from_fn pass.
/// Returns true only when all ops are element-wise (no matmul, reduction, slicing, etc.).
fn is_elementwise_fusible(expr: &Expr) -> bool {
    match expr {
        // Tensor variable or plain ident — always fusible
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => true,

        // Literals — always fusible
        Expr::Lit(_) => true,

        // Binary +, -, *, / — fusible if both sides are
        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            matches!(
                op,
                syn::BinOp::Add(_)
                    | syn::BinOp::Sub(_)
                    | syn::BinOp::Mul(_)
                    | syn::BinOp::Div(_)
            ) && is_elementwise_fusible(left)
                && is_elementwise_fusible(right)
        }

        // Unary neg — fusible
        Expr::Unary(ExprUnary { op: syn::UnOp::Neg(_), expr: inner, .. }) => {
            is_elementwise_fusible(inner)
        }

        // Method calls: only known element-wise methods
        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let name = method.to_string();
            let recv_ok = is_elementwise_fusible(receiver);
            if args.is_empty() && ELEMENTWISE_UNARY.contains(&name.as_str()) {
                recv_ok
            } else if args.len() == 1 && ELEMENTWISE_UNARY_ARG.contains(&name.as_str()) {
                recv_ok && is_elementwise_fusible(&args[0])
            } else {
                false
            }
        }

        // Free function calls: known element-wise functions
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                ELEMENTWISE_UNARY.contains(&fname.as_str())
                    && is_elementwise_fusible(&ec.args[0])
            } else {
                false
            }
        }

        Expr::Paren(ep) => is_elementwise_fusible(&ep.expr),

        // Cast expressions from eqsat integer rendering (e.g. `2 as _`)
        Expr::Cast(ec) => is_elementwise_fusible(&ec.expr),

        _ => false,
    }
}

/// Rewrite expression to scalar-level ops for use inside `from_fn`.
/// Tensor variables become `__fuse_v_<name>` (scalar values read via .get()).
/// Binary * between two tensor-containing exprs becomes plain scalar *.
fn scalar_expr(expr: &Expr, tensor_names: &[String]) -> TokenStream2 {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let name = path.segments[0].ident.to_string();
            if tensor_names.contains(&name) {
                let fused = Ident::new(&format!("__fuse_v_{name}"), path.segments[0].ident.span());
                quote! { #fused }
            } else {
                expr.to_token_stream()
            }
        }

        Expr::Lit(_) => expr.to_token_stream(),

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let l = scalar_expr(left, tensor_names);
            let r = scalar_expr(right, tensor_names);
            quote! { (#l #op #r) }
        }

        Expr::Unary(ExprUnary { op, expr: inner, .. }) => {
            let i = scalar_expr(inner, tensor_names);
            quote! { (#op #i) }
        }

        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let recv = scalar_expr(receiver, tensor_names);
            let rewritten_args: Vec<_> = args.iter().map(|a| scalar_expr(a, tensor_names)).collect();
            let method_name = method.to_string();
            // Map tensor-level method names to MathOps scalar trait methods
            let scalar_method = match method_name.as_str() {
                "exp" => quote! { math_exp },
                "ln" => quote! { math_ln },
                "log1p" => quote! { math_log1p },
                "sin" => quote! { math_sin },
                "cos" => quote! { math_cos },
                "tanh" => quote! { math_tanh },
                "sqrt" => quote! { math_sqrt },
                "abs" => quote! { math_abs },
                "recip" => quote! { math_recip },
                "erf" => quote! { math_erf },
                "ceil" => quote! { math_ceil },
                "floor" => quote! { math_floor },
                "round" => quote! { math_round },
                "powf" => quote! { math_powf },
                "neg" => return quote! { (-#recv) },
                _ => quote! { #method },
            };
            quote! { #recv.#scalar_method(#(#rewritten_args),*) }
        }

        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                let arg = scalar_expr(&ec.args[0], tensor_names);
                match fname.as_str() {
                    "exp" => return quote! { #arg.math_exp() },
                    "ln" => return quote! { #arg.math_ln() },
                    "sqrt" => return quote! { #arg.math_sqrt() },
                    "abs" => return quote! { #arg.math_abs() },
                    "sin" => return quote! { #arg.math_sin() },
                    "cos" => return quote! { #arg.math_cos() },
                    "tanh" => return quote! { #arg.math_tanh() },
                    _ => {}
                }
            }
            let func = &ec.func;
            let rewritten_args: Vec<_> =
                ec.args.iter().map(|a| scalar_expr(a, tensor_names)).collect();
            quote! { #func(#(#rewritten_args),*) }
        }

        Expr::Paren(ep) => {
            let inner = scalar_expr(&ep.expr, tensor_names);
            quote! { (#inner) }
        }

        // Eqsat renders exact integers as `2i64 as _`. Inside from_fn
        // the `_` can't be inferred, so emit as float literal instead.
        Expr::Cast(ec) => {
            if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) = &*ec.expr
                && let Ok(v) = i.base10_parse::<f64>()
            {
                return quote! { #v };
            }
            let inner = scalar_expr(&ec.expr, tensor_names);
            let ty = &ec.ty;
            quote! { (#inner as #ty) }
        }

        _ => expr.to_token_stream(),
    }
}

/// Generate a CUDA C expression string from the AST for GPU fused kernels.
/// Tensor variables map to `inN[i]` where N is the tensor's index.
/// Math ops map to CUDA C++ overloaded functions (sin, cos, exp, etc.).
fn cuda_expr(expr: &Expr, tensor_names: &[String]) -> String {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let name = path.segments[0].ident.to_string();
            if let Some(idx) = tensor_names.iter().position(|n| n == &name) {
                format!("in{idx}[i]")
            } else {
                name
            }
        }

        Expr::Lit(syn::ExprLit { lit, .. }) => {
            match lit {
                syn::Lit::Float(f) => f.to_string(),
                syn::Lit::Int(i) => format!("(double)({})", i.base10_digits()),
                _ => lit.to_token_stream().to_string(),
            }
        }

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let l = cuda_expr(left, tensor_names);
            let r = cuda_expr(right, tensor_names);
            let op_str = match op {
                syn::BinOp::Add(_) => "+",
                syn::BinOp::Sub(_) => "-",
                syn::BinOp::Mul(_) => "*",
                syn::BinOp::Div(_) => "/",
                _ => panic!("fuse!: unsupported binary op for GPU"),
            };
            format!("({l} {op_str} {r})")
        }

        Expr::Unary(ExprUnary { op: syn::UnOp::Neg(_), expr: inner, .. }) => {
            let i = cuda_expr(inner, tensor_names);
            format!("(-{i})")
        }

        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let recv = cuda_expr(receiver, tensor_names);
            let method_name = method.to_string();
            match method_name.as_str() {
                "exp" => format!("exp({recv})"),
                "ln" => format!("log({recv})"),
                "log1p" => format!("log1p({recv})"),
                "sin" => format!("sin({recv})"),
                "cos" => format!("cos({recv})"),
                "tanh" => format!("tanh({recv})"),
                "sqrt" => format!("sqrt({recv})"),
                "abs" => format!("fabs({recv})"),
                "recip" => format!("(1.0/({recv}))"),
                "erf" => format!("erf({recv})"),
                "ceil" => format!("ceil({recv})"),
                "floor" => format!("floor({recv})"),
                "round" => format!("round({recv})"),
                "neg" => format!("(-{recv})"),
                "powf" if args.len() == 1 => {
                    let p = cuda_expr(&args[0], tensor_names);
                    format!("pow({recv}, {p})")
                }
                _ => panic!("fuse!: unsupported GPU method: {method_name}"),
            }
        }

        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func
                && path.segments.len() == 1
                && ec.args.len() == 1
            {
                let fname = path.segments[0].ident.to_string();
                let arg = cuda_expr(&ec.args[0], tensor_names);
                match fname.as_str() {
                    "exp" => return format!("exp({arg})"),
                    "ln" => return format!("log({arg})"),
                    "sqrt" => return format!("sqrt({arg})"),
                    "abs" => return format!("fabs({arg})"),
                    "sin" => return format!("sin({arg})"),
                    "cos" => return format!("cos({arg})"),
                    "tanh" => return format!("tanh({arg})"),
                    _ => {}
                }
            }
            panic!("fuse!: unsupported GPU function call")
        }

        Expr::Paren(ep) => {
            let inner = cuda_expr(&ep.expr, tensor_names);
            format!("({inner})")
        }

        Expr::Cast(ec) => {
            if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) = &*ec.expr
                && let Ok(v) = i.base10_parse::<f64>()
            {
                return format!("{v}");
            }
            cuda_expr(&ec.expr, tensor_names)
        }

        _ => panic!("fuse!: unsupported expression for GPU codegen"),
    }
}

// ── Fusion cost model (register pressure heuristic) ─────────────────────────

/// Maximum recommended registers per thread before spill risk.
const MAX_FUSE_REGISTERS: usize = 120;

/// Walk the expression tree and estimate register usage for a fused kernel.
///
/// Heuristic costs (approximate GPU register usage):
/// - Each input tensor: 4 registers (float4 vector load)
/// - Each transcendental op: +12 registers
/// - Each arithmetic op: +2 registers
/// - Output: 4 registers (float4 store)
fn estimate_register_pressure(expr: &Expr, tensor_names: &[String]) -> usize {
    let mut transcendental = 0usize;
    let mut arithmetic = 0usize;
    let mut inputs = std::collections::HashSet::new();
    count_ops(expr, tensor_names, &mut transcendental, &mut arithmetic, &mut inputs);
    let input_regs = inputs.len() * 4;
    let output_regs = 4;
    input_regs + transcendental * 12 + arithmetic * 2 + output_regs
}

fn count_ops(
    expr: &Expr,
    tensor_names: &[String],
    transcendental: &mut usize,
    arithmetic: &mut usize,
    inputs: &mut std::collections::HashSet<String>,
) {
    match expr {
        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
            let name = path.segments[0].ident.to_string();
            if tensor_names.contains(&name) {
                inputs.insert(name);
            }
        }
        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            match op {
                syn::BinOp::Add(_) | syn::BinOp::Sub(_) | syn::BinOp::Mul(_) | syn::BinOp::Div(_) => {
                    *arithmetic += 1;
                }
                _ => {}
            }
            count_ops(left, tensor_names, transcendental, arithmetic, inputs);
            count_ops(right, tensor_names, transcendental, arithmetic, inputs);
        }
        Expr::Unary(ExprUnary { expr: inner, .. }) => {
            count_ops(inner, tensor_names, transcendental, arithmetic, inputs);
        }
        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let name = method.to_string();
            match name.as_str() {
                "exp" | "sin" | "cos" | "ln" | "tanh" | "sqrt" | "erf" | "log1p" => {
                    *transcendental += 1;
                }
                "powf" => *transcendental += 1,
                "abs" | "ceil" | "floor" | "round" | "neg" | "recip" => {
                    *arithmetic += 1;
                }
                _ => {}
            }
            count_ops(receiver, tensor_names, transcendental, arithmetic, inputs);
            for a in args {
                count_ops(a, tensor_names, transcendental, arithmetic, inputs);
            }
        }
        Expr::Call(ec) => {
            if let Expr::Path(ExprPath { path, .. }) = &*ec.func {
                if path.segments.len() == 1 {
                    let fname = path.segments[0].ident.to_string();
                    match fname.as_str() {
                        "exp" | "sin" | "cos" | "ln" | "tanh" | "sqrt" => {
                            *transcendental += 1;
                        }
                        _ => {}
                    }
                }
            }
            for a in &ec.args {
                count_ops(a, tensor_names, transcendental, arithmetic, inputs);
            }
        }
        Expr::Paren(ep) => {
            count_ops(&ep.expr, tensor_names, transcendental, arithmetic, inputs);
        }
        Expr::Cast(ec) => {
            count_ops(&ec.expr, tensor_names, transcendental, arithmetic, inputs);
        }
        _ => {}
    }
}

/// Compute a simple FNV-1a hash of the GPU expression for kernel name deduplication.
fn expr_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

// Activation methods recognized for GEMM fusion.
const GEMM_ACTIVATIONS: &[&str] = &[
    "sigmoid", "relu", "tanh", "gelu", "exp", "ln", "sqrt", "abs", "neg", "recip",
];

/// Detect `(A * B).activation()` pattern for L3 GEMM+activation fusion.
/// Returns `Some((lhs_expr, rhs_expr, activation_ident))`.
fn detect_gemm_activation(expr: &Expr) -> Option<(Expr, Expr, Ident)> {
    // Pattern: method_call.activation() where receiver is (A * B)
    if let Expr::MethodCall(mc) = expr
        && mc.args.is_empty()
        && GEMM_ACTIVATIONS.contains(&mc.method.to_string().as_str())
    {
        // Receiver should be a paren around binary mul, or just binary mul
        let inner = match &*mc.receiver {
            Expr::Paren(ep) => &ep.expr,
            other => other,
        };
        if let Expr::Binary(ExprBinary { left, op: syn::BinOp::Mul(_), right, .. }) = inner {
            // Strip reference wrappers from operands
            let lhs = strip_ref(left);
            let rhs = strip_ref(right);
            return Some((lhs.clone(), rhs.clone(), mc.method.clone()));
        }
    }
    None
}

fn strip_ref(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(er) => &er.expr,
        other => other,
    }
}

fn fuse_impl(input: TokenStream2) -> Result<TokenStream2> {
    let FuseInput { body, tensors } = syn::parse2(input)?;
    let tensor_names: Vec<String> = tensors.iter().map(|t| t.to_string()).collect();
    let body = eqsat_simplify(&body);

    // L3: GEMM + activation fusion — emit matmul then tensor activation
    if let Some((lhs, rhs, act)) = detect_gemm_activation(&body) {
        return Ok(quote! {{
            let __fuse_c = &#lhs * &#rhs;
            __fuse_c.#act()
        }});
    }

    let first = &tensors[0];
    let shape_checks: Vec<TokenStream2> = tensors
        .iter()
        .skip(1)
        .map(|t| {
            quote! {
                assert_eq!(#first.shape(), #t.shape(), "fuse!: shape mismatch");
            }
        })
        .collect();

    if is_elementwise_fusible(&body) {
        // Fused path: single kernel pass over all elements
        let scalar_body = scalar_expr(&body, &tensor_names);
        let let_bindings: Vec<TokenStream2> = tensors
            .iter()
            .map(|t| {
                let var = Ident::new(&format!("__fuse_v_{t}"), t.span());
                quote! { let #var = #t.get(__fuse_r, __fuse_c); }
            })
            .collect();

        // GPU codegen: CUDA/HIP C expression + kernel hash
        let gpu_expr_str = cuda_expr(&body, &tensor_names);
        let kernel_hash = expr_hash(&gpu_expr_str);
        let n_inputs = tensors.len();

        // Register pressure estimate (compile-time heuristic)
        let reg_estimate = estimate_register_pressure(&body, &tensor_names);
        if reg_estimate > MAX_FUSE_REGISTERS {
            eprintln!(
                "warning: fuse! estimated register pressure {reg_estimate} exceeds \
                 threshold {MAX_FUSE_REGISTERS} — kernel may spill to local memory"
            );
        }
        let reg_est_lit = reg_estimate;

        let storage_ptrs: Vec<TokenStream2> = tensors
            .iter()
            .map(|t| quote! { #t.__storage_ptr() })
            .collect();

        Ok(quote! {{
            #(#shape_checks)*
            use nabla::scalar::MathOps as _;
            let __fuse_inputs: &[*const u8] = &[#(#storage_ptrs),*];
            nabla::tensor::Tensor::__fuse_elementwise(
                __fuse_inputs,
                #first.nrows(),
                #first.ncols(),
                |__fuse_r, __fuse_c| {
                    #(#let_bindings)*
                    #scalar_body
                },
                #gpu_expr_str,
                #kernel_hash,
                #n_inputs,
                #reg_est_lit,
            )
        }})
    } else {
        // Fallback: tensor-level chained ops (existing behavior)
        let lifted = lift_expr(&body, &tensor_names);
        Ok(quote! {{
            #(#shape_checks)*
            #lifted
        }})
    }
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
        Ok(NamedInput {
            fields: input
                .parse_terminated(NamedField::parse, Comma)?
                .into_iter()
                .collect(),
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

// ── axis! (Named Axis marker types) ─────────────────────────────────────────

/// Declare zero-sized marker types for named tensor axes.
///
/// # Examples
///
/// ```rust,ignore
/// axis!(Batch, Seq, Hidden);
/// // expands to: pub struct Batch; pub struct Seq; pub struct Hidden;
/// ```
#[proc_macro]
pub fn axis(input: TokenStream) -> TokenStream {
    match axis_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn axis_impl(input: TokenStream2) -> Result<TokenStream2> {
    let names: Punctuated<Ident, Comma> =
        syn::parse::Parser::parse2(Punctuated::<Ident, Comma>::parse_terminated, input)?;
    if names.is_empty() {
        return Err(Error::new(
            Span::call_site(),
            "axis!() requires at least one identifier",
        ));
    }
    let structs = names.iter().map(|name| {
        quote! { pub struct #name; }
    });
    Ok(quote! { #(#structs)* })
}

// ── named_zeros! (Typed axis constructor) ────────────────────────────────────

/// Construct a zero-filled tensor with named axes.
///
/// # Syntax
///
/// ```rust,ignore
/// // named_zeros!(AxisRow, AxisCol; rows, cols)
/// let t = named_zeros!(Batch, Hidden; 32, 768);
/// // type: Tensor<f64, DefaultBackend, (Batch, Hidden)>
/// ```
#[proc_macro]
pub fn named_zeros(input: TokenStream) -> TokenStream {
    match named_zeros_impl(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
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
            return Err(Error::new(
                Span::call_site(),
                format!(
                    "named_zeros!: {} axes but {} dimensions",
                    axes.len(),
                    dims.len()
                ),
            ));
        }
        if axes.len() != 2 {
            return Err(Error::new(
                Span::call_site(),
                "named_zeros! requires exactly 2 axes (row, col) for a 2-D tensor",
            ));
        }
        Ok(NamedZerosInput { axes, dims })
    }
}

fn named_zeros_impl(input: TokenStream2) -> Result<TokenStream2> {
    let NamedZerosInput { axes, dims } = syn::parse2(input)?;
    let ax0 = &axes[0];
    let ax1 = &axes[1];
    let d0 = &dims[0];
    let d1 = &dims[1];
    Ok(quote! {
        nabla::tensor::Tensor::zeros(#d0, #d1).with_axes::<(#ax0, #ax1)>()
    })
}

// ── #[nabla_grad] attribute macro ────────────────────────────────────────────

/// Source-transform AD: lifts `fn f(x: T) -> T` to also generate `fn f_grad(x: T) -> (T, T)`.
///
/// The generated `_grad` function evaluates `f` using `Dual<T>` to compute both
/// the value and the first derivative in a single forward pass.
///
/// # Examples
///
/// ```rust,ignore
/// #[nabla_grad]
/// fn sigmoid(x: f64) -> f64 {
///     1.0 / (1.0 + (-x).exp())
/// }
///
/// let (val, grad) = sigmoid_grad(0.0);
/// // val ≈ 0.5, grad ≈ 0.25
/// ```
#[proc_macro_attribute]
pub fn nabla_grad(_attr: TokenStream, item: TokenStream) -> TokenStream {
    match nabla_grad_impl(item.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn nabla_grad_impl(item: TokenStream2) -> Result<TokenStream2> {
    let func: syn::ItemFn = syn::parse2(item)?;

    // Validate: exactly one argument
    if func.sig.inputs.len() != 1 {
        return Err(Error::new_spanned(
            &func.sig,
            "#[nabla_grad] requires exactly one scalar argument",
        ));
    }

    let fn_name = &func.sig.ident;
    let grad_name = Ident::new(&format!("{fn_name}_grad"), fn_name.span());
    let vis = &func.vis;

    // Extract the argument name and type
    let arg = &func.sig.inputs[0];
    let (arg_name, arg_ty) = match arg {
        syn::FnArg::Typed(pat_ty) => (&pat_ty.pat, &pat_ty.ty),
        syn::FnArg::Receiver(_) => {
            return Err(Error::new_spanned(arg, "#[nabla_grad] does not support self"));
        }
    };

    let body = &func.block;

    // Emit original function unchanged + generated _grad function.
    // The _grad function re-executes the function body with a Dual-seeded input,
    // exploiting Dual<T>'s operator overloads for forward-mode AD.
    Ok(quote! {
        #func

        #vis fn #grad_name(#arg_name: #arg_ty) -> (#arg_ty, #arg_ty) {
            let #arg_name: nabla::scalar::Dual<#arg_ty> = nabla::scalar::Dual::new(
                #arg_name,
                <#arg_ty as nabla::scalar::Scalar>::from_f64(1.0),
            );
            let __nabla_result: nabla::scalar::Dual<#arg_ty> = #body;
            (__nabla_result.value, __nabla_result.deriv)
        }
    })
}
