// cas.rs — Symbolic Computer Algebra System (CAS) for nabla.
//
// Design:
//   - Expr is a newtype wrapper around Arc<ExprKind> for structural sharing.
//   - Sub is not a variant: a - b ≡ Add(a, Neg(b)) — halves simplification rules.
//   - diff/simplify/eval are pure recursive functions (no mutation).

// CAS literal comparison is intentionally exact (0.0, 1.0 are constructed, not computed).
#![allow(clippy::float_cmp, clippy::too_many_lines, clippy::single_match_else)]
#![allow(clippy::missing_errors_doc, clippy::implicit_hasher, clippy::many_single_char_names)]

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::sync::Arc;

use egg::{define_language, rewrite, Analysis, DidMerge, EGraph, Id, RecExpr, Rewrite, Runner, Symbol, Subst, Var};
use nabla_core::backend::Backend;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;
use ordered_float::NotNan;

// ---------------------------------------------------------------------------
// ExprKind — the inner enum
// ---------------------------------------------------------------------------

/// Symbolic expression node variants.
///
/// Pattern-match through [`Expr`]'s `Deref` impl: `match &*expr { ... }`.
#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Named variable.
    Var(String),
    /// Numeric literal.
    Lit(f64),
    /// Arithmetic negation.
    Neg(Expr),
    /// Addition.
    Add(Expr, Expr),
    /// Multiplication.
    Mul(Expr, Expr),
    /// Division.
    Div(Expr, Expr),
    /// Exponentiation.
    Pow(Expr, Expr),
    /// Sine.
    Sin(Expr),
    /// Cosine.
    Cos(Expr),
    /// Natural exponential.
    Exp(Expr),
    /// Natural logarithm.
    Ln(Expr),
    /// Hyperbolic tangent.
    Tanh(Expr),
    /// Square root.
    Sqrt(Expr),
    /// Absolute value.
    Abs(Expr),
}

// ---------------------------------------------------------------------------
// Expr — newtype wrapper with Arc
// ---------------------------------------------------------------------------

/// A symbolic expression with cheap cloning via `Arc`.
///
/// Construct via [`Expr::var`], [`Expr::lit`], unary helpers (`.sin()` etc.),
/// and standard operator overloads (`&a + &b`, `&a * 2.0`, …).
///
/// Pattern-match through `Deref`: `match &*expr { ExprKind::Var(s) => … }`.
#[derive(Debug, Clone)]
pub struct Expr(Arc<ExprKind>);

impl Deref for Expr {
    type Target = ExprKind;
    #[inline]
    fn deref(&self) -> &ExprKind {
        &self.0
    }
}

impl Expr {
    #[inline]
    fn wrap(kind: ExprKind) -> Self {
        Self(Arc::new(kind))
    }

    /// Variable node.
    #[must_use]
    pub fn var(name: &str) -> Self {
        Self::wrap(ExprKind::Var(name.to_owned()))
    }

    /// Numeric literal node.
    #[must_use]
    pub fn lit(val: f64) -> Self {
        Self::wrap(ExprKind::Lit(val))
    }

    /// `base ^ exp`.
    #[must_use]
    pub fn pow(base: &Self, exp: &Self) -> Self {
        Self::wrap(ExprKind::Pow(base.clone(), exp.clone()))
    }
}

/// Generate unary math constructor methods on `Expr`.
macro_rules! cas_unary {
    ($($name:ident => $variant:ident),+ $(,)?) => {
        impl Expr {
            $(
                #[doc = concat!("`", stringify!($name), "(e)`.")]
                #[must_use]
                pub fn $name(e: &Self) -> Self { Self::wrap(ExprKind::$variant(e.clone())) }
            )+
        }
    };
}
cas_unary!(sin => Sin, cos => Cos, exp => Exp, ln => Ln, tanh => Tanh, sqrt => Sqrt, abs => Abs);

// ---------------------------------------------------------------------------
// Operator overloads
// ---------------------------------------------------------------------------

macro_rules! impl_expr_binop {
    ($($trait:ident, $method:ident, $variant:ident);+ $(;)?) => {
        $(
            impl $trait for &Expr {
                type Output = Expr;
                fn $method(self, rhs: &Expr) -> Expr {
                    Expr::wrap(ExprKind::$variant(self.clone(), rhs.clone()))
                }
            }
        )+
    };
}
impl_expr_binop!(Add, add, Add; Mul, mul, Mul; Div, div, Div);

impl Neg for &Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::wrap(ExprKind::Neg(self.clone()))
    }
}

impl Sub for &Expr {
    type Output = Expr;
    fn sub(self, rhs: &Expr) -> Expr {
        self + &(-rhs)
    }
}

// Mixed: &Expr op f64, f64 op &Expr

macro_rules! impl_expr_f64_binop {
    ($($trait:ident, $method:ident);+ $(;)?) => {
        $(
            impl $trait<f64> for &Expr {
                type Output = Expr;
                fn $method(self, rhs: f64) -> Expr {
                    $trait::$method(self, &Expr::lit(rhs))
                }
            }
            impl $trait<&Expr> for f64 {
                type Output = Expr;
                fn $method(self, rhs: &Expr) -> Expr {
                    $trait::$method(&Expr::lit(self), rhs)
                }
            }
        )+
    };
}
impl_expr_f64_binop!(Add, add; Mul, mul);

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for ExprKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(s) => write!(f, "{s}"),
            Self::Lit(v) => {
                if v.fract() == 0.0 && v.is_finite() {
                    write!(f, "{v:.0}")
                } else {
                    write!(f, "{v}")
                }
            }
            Self::Neg(a) => write!(f, "(-{a})"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Div(a, b) => write!(f, "({a} / {b})"),
            Self::Pow(a, b) => write!(f, "({a} ^ {b})"),
            Self::Sin(a) => write!(f, "sin({a})"),
            Self::Cos(a) => write!(f, "cos({a})"),
            Self::Exp(a) => write!(f, "exp({a})"),
            Self::Ln(a) => write!(f, "ln({a})"),
            Self::Tanh(a) => write!(f, "tanh({a})"),
            Self::Sqrt(a) => write!(f, "sqrt({a})"),
            Self::Abs(a) => write!(f, "abs({a})"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

// ---------------------------------------------------------------------------
// Symbolic differentiation
// ---------------------------------------------------------------------------

/// Symbolically differentiate `expr` with respect to `var`.
///
/// Returns a new (unsimplified) expression. Call [`simplify`] for a cleaner form.
#[must_use]
pub fn diff(expr: &Expr, var: &str) -> Expr {
    match &**expr {
        ExprKind::Var(s) => {
            if s == var {
                Expr::lit(1.0)
            } else {
                Expr::lit(0.0)
            }
        }
        ExprKind::Lit(_) => Expr::lit(0.0),
        ExprKind::Neg(a) => -&diff(a, var),
        ExprKind::Add(a, b) => &diff(a, var) + &diff(b, var),
        // Product rule: (ab)' = ab' + a'b
        ExprKind::Mul(a, b) => &(a * &diff(b, var)) + &(&diff(a, var) * b),
        // Quotient rule: (a/b)' = (a'b − ab') / b²
        ExprKind::Div(a, b) => {
            let num = &(&diff(a, var) * b) - &(a * &diff(b, var));
            let den = b * b;
            &num / &den
        }
        // Power rule
        ExprKind::Pow(a, b) => {
            if let ExprKind::Lit(n) = &**b {
                // n * a^(n-1) * a'
                &(&Expr::lit(*n) * &Expr::pow(a, &Expr::lit(n - 1.0))) * &diff(a, var)
            } else {
                // a^b * (b'·ln(a) + b·a'/a)
                let ab = Expr::pow(a, b);
                let t1 = &diff(b, var) * &Expr::ln(a);
                let t2 = b * &(&diff(a, var) / a);
                &ab * &(&t1 + &t2)
            }
        }
        ExprKind::Sin(a) => &Expr::cos(a) * &diff(a, var),
        ExprKind::Cos(a) => -&(&Expr::sin(a) * &diff(a, var)),
        ExprKind::Exp(a) => &Expr::exp(a) * &diff(a, var),
        ExprKind::Ln(a) => &diff(a, var) / a,
        ExprKind::Tanh(a) => {
            let th = Expr::tanh(a);
            let one_minus_th2 = &Expr::lit(1.0) - &Expr::pow(&th, &Expr::lit(2.0));
            &one_minus_th2 * &diff(a, var)
        }
        ExprKind::Sqrt(a) => &diff(a, var) / &(2.0 * &Expr::sqrt(a)),
        ExprKind::Abs(a) => &(a / &Expr::abs(a)) * &diff(a, var),
    }
}

// ---------------------------------------------------------------------------
// E-graph CAS — equality saturation via egg
// ---------------------------------------------------------------------------

define_language! {
    enum CasLang {
        Num(NotNan<f64>),
        "neg" = CNeg([Id; 1]),
        "+" = CAdd([Id; 2]),
        "*" = CMul([Id; 2]),
        "/" = CDiv([Id; 2]),
        "^" = CPow([Id; 2]),
        "sin" = CSin([Id; 1]),
        "cos" = CCos([Id; 1]),
        "exp" = CExp([Id; 1]),
        "ln"  = CLn([Id; 1]),
        "tanh" = CTanh([Id; 1]),
        "sqrt" = CSqrt([Id; 1]),
        "abs"  = CAbs([Id; 1]),
        "diff" = CDiff([Id; 2]),
        Symbol(Symbol),
    }
}

#[derive(Default)]
struct ConstFold;

impl Analysis<CasLang> for ConstFold {
    type Data = Option<NotNan<f64>>;

    fn make(egraph: &EGraph<CasLang, Self>, enode: &CasLang) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        match enode {
            CasLang::Num(n) => Some(*n),
            CasLang::CNeg([a]) => Some(-x(a)?),
            CasLang::CAdd([a, b]) => Some(x(a)? + x(b)?),
            CasLang::CMul([a, b]) => Some(x(a)? * x(b)?),
            CasLang::CDiv([a, b]) => {
                let bv = x(b)?;
                if *bv == 0.0 { None } else { Some(x(a)? / bv) }
            }
            CasLang::CPow([a, b]) => NotNan::new(x(a)?.powf(*x(b)?)).ok(),
            CasLang::CSin([a]) => NotNan::new(x(a)?.sin()).ok(),
            CasLang::CCos([a]) => NotNan::new(x(a)?.cos()).ok(),
            CasLang::CExp([a]) => NotNan::new(x(a)?.exp()).ok(),
            CasLang::CLn([a]) => {
                let av = x(a)?;
                if *av > 0.0 { NotNan::new(av.ln()).ok() } else { None }
            }
            CasLang::CTanh([a]) => NotNan::new(x(a)?.tanh()).ok(),
            CasLang::CSqrt([a]) => {
                let av = x(a)?;
                if *av >= 0.0 { NotNan::new(av.sqrt()).ok() } else { None }
            }
            CasLang::CAbs([a]) => NotNan::new(x(a)?.abs()).ok(),
            CasLang::CDiff(_) | CasLang::Symbol(_) => None,
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            debug_assert_eq!(*a, b);
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph<CasLang, Self>, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = egraph.add(CasLang::Num(c));
            egraph.union(id, added);
        }
    }
}

// Condition: ?v is a numeric constant (has ConstFold data).
struct IsConst {
    v: Var,
}

impl egg::Condition<CasLang, ConstFold> for IsConst {
    fn check(&self, egraph: &mut EGraph<CasLang, ConstFold>, _: Id, subst: &Subst) -> bool {
        egraph[subst[self.v]].data.is_some()
    }
}

// Condition: ?y is a symbol different from ?x (and not a constant).
struct IsDifferentVar {
    y: Var,
    x: Var,
}

impl egg::Condition<CasLang, ConstFold> for IsDifferentVar {
    fn check(&self, egraph: &mut EGraph<CasLang, ConstFold>, _: Id, subst: &Subst) -> bool {
        let y_class = &egraph[subst[self.y]];
        let x_class = &egraph[subst[self.x]];
        // y must be a symbol
        let y_sym = y_class.nodes.iter().find_map(|n| match n {
            CasLang::Symbol(s) => Some(*s),
            _ => None,
        });
        let x_sym = x_class.nodes.iter().find_map(|n| match n {
            CasLang::Symbol(s) => Some(*s),
            _ => None,
        });
        matches!((y_sym, x_sym), (Some(ys), Some(xs)) if ys != xs)
    }
}

fn cas_rules() -> Vec<Rewrite<CasLang, ConstFold>> {
    vec![
        rewrite!("add-zero-r"; "(+ ?a 0)" => "?a"),
        rewrite!("add-zero-l"; "(+ 0 ?a)" => "?a"),
        rewrite!("mul-one-r";  "(* ?a 1)" => "?a"),
        rewrite!("mul-one-l";  "(* 1 ?a)" => "?a"),
        rewrite!("mul-zero-r"; "(* ?a 0)" => "0"),
        rewrite!("mul-zero-l"; "(* 0 ?a)" => "0"),
        rewrite!("double-neg"; "(neg (neg ?a))" => "?a"),
        rewrite!("exp-ln";     "(exp (ln ?a))" => "?a"),
        rewrite!("ln-exp";     "(ln (exp ?a))" => "?a"),
        rewrite!("pow-zero";   "(^ ?a 0)" => "1"),
        rewrite!("pow-one";    "(^ ?a 1)" => "?a"),
        rewrite!("div-one";    "(/ ?a 1)" => "?a"),
        rewrite!("zero-div";   "(/ 0 ?a)" => "0"),
        rewrite!("neg-zero";   "(neg 0)" => "0"),
        rewrite!("add-neg";    "(+ ?a (neg ?a))" => "0"),
        rewrite!("add-comm";   "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("mul-comm";   "(* ?a ?b)" => "(* ?b ?a)"),
        // x / x => 1
        rewrite!("div-self"; "(/ ?x ?x)" => "1"),
        // x^a * x^b => x^(a+b)
        rewrite!("pow-add"; "(* (^ ?x ?a) (^ ?x ?b))" => "(^ ?x (+ ?a ?b))"),
        // ln(x) + ln(y) => ln(x*y)
        rewrite!("ln-prod"; "(+ (ln ?x) (ln ?y))" => "(ln (* ?x ?y))"),
        // (x^a)^b => x^(a*b)
        rewrite!("pow-pow"; "(^ (^ ?x ?a) ?b)" => "(^ ?x (* ?a ?b))"),
        // (a*b)/b => a
        rewrite!("mul-div-cancel"; "(/ (* ?a ?b) ?b)" => "?a"),
        // a * (b/a) => b
        rewrite!("mul-div-cancel2"; "(* ?a (/ ?b ?a))" => "?b"),
        // -(x+y) => (-x)+(-y)
        rewrite!("neg-add"; "(neg (+ ?x ?y))" => "(+ (neg ?x) (neg ?y))"),
        // ln(1) => 0
        rewrite!("ln-one"; "(ln 1)" => "0"),
        // sqrt(x) => x^0.5
        rewrite!("sqrt-to-pow"; "(sqrt ?x)" => "(^ ?x 0.5)"),
        // exp(a) * exp(b) => exp(a+b)
        rewrite!("exp-add"; "(* (exp ?a) (exp ?b))" => "(exp (+ ?a ?b))"),
        // ln(x) - ln(y) => ln(x/y)  [sub = (+ (ln x) (neg (ln y)))]
        rewrite!("ln-quot"; "(+ (ln ?x) (neg (ln ?y)))" => "(ln (/ ?x ?y))"),
        // a * ln(x) => ln(x^a)
        rewrite!("ln-pow"; "(* ?a (ln ?x))" => "(ln (^ ?x ?a))"),
        // 0 + (neg x) => neg x  [0 - x = -x, sub encoded as add-neg; add-zero-l handles it]
        // x * (1/y) => x/y
        rewrite!("mul-recip"; "(* ?x (/ 1 ?y))" => "(/ ?x ?y)"),
        // (a+b)*c => a*c + b*c  (distribute, forward only to avoid loop)
        rewrite!("dist-r"; "(* (+ ?a ?b) ?c)" => "(+ (* ?a ?c) (* ?b ?c))"),
        // Differentiation rules
        rewrite!("diff-const"; "(diff ?c ?x)" => "0"
            if IsConst { v: "?c".parse().expect("var") }),
        rewrite!("diff-var-same"; "(diff ?x ?x)" => "1"),
        rewrite!("diff-var-other"; "(diff ?y ?x)" => "0"
            if IsDifferentVar {
                y: "?y".parse().expect("var"),
                x: "?x".parse().expect("var"),
            }),
        // d[a+b]/dx = d[a]/dx + d[b]/dx
        rewrite!("diff-add"; "(diff (+ ?a ?b) ?x)" => "(+ (diff ?a ?x) (diff ?b ?x))"),
        // d[a*b]/dx = a'*b + a*b'
        rewrite!("diff-mul"; "(diff (* ?a ?b) ?x)" =>
            "(+ (* (diff ?a ?x) ?b) (* ?a (diff ?b ?x)))"),
        // d[-a]/dx = -d[a]/dx
        rewrite!("diff-neg"; "(diff (neg ?a) ?x)" => "(neg (diff ?a ?x))"),
        // d[exp(a)]/dx = exp(a) * a'
        rewrite!("diff-exp"; "(diff (exp ?a) ?x)" => "(* (exp ?a) (diff ?a ?x))"),
        // d[ln(a)]/dx = a' / a
        rewrite!("diff-ln"; "(diff (ln ?a) ?x)" => "(/ (diff ?a ?x) ?a)"),
        // d[sin(a)]/dx = cos(a) * a'
        rewrite!("diff-sin"; "(diff (sin ?a) ?x)" => "(* (cos ?a) (diff ?a ?x))"),
        // d[cos(a)]/dx = -(sin(a) * a')
        rewrite!("diff-cos"; "(diff (cos ?a) ?x)" => "(neg (* (sin ?a) (diff ?a ?x)))"),
        // d[a/b]/dx = (a'*b - a*b') / b^2
        rewrite!("diff-div"; "(diff (/ ?a ?b) ?x)" =>
            "(/ (+ (* (diff ?a ?x) ?b) (neg (* ?a (diff ?b ?x)))) (^ ?b 2))"),
        // d[a^n]/dx = n * a^(n-1) * a'  (n constant)
        rewrite!("diff-pow"; "(diff (^ ?a ?n) ?x)" =>
            "(* (* ?n (^ ?a (+ ?n -1))) (diff ?a ?x))"
            if IsConst { v: "?n".parse().expect("var") }),
        // d[tanh(a)]/dx = (1 - tanh(a)^2) * a'
        rewrite!("diff-tanh"; "(diff (tanh ?a) ?x)" =>
            "(* (+ 1 (neg (^ (tanh ?a) 2))) (diff ?a ?x))"),
        // d[sqrt(a)]/dx = a' / (2 * sqrt(a))
        rewrite!("diff-sqrt"; "(diff (sqrt ?a) ?x)" =>
            "(/ (diff ?a ?x) (* 2 (sqrt ?a)))"),
        // d[abs(a)]/dx = (a / abs(a)) * a'
        rewrite!("diff-abs"; "(diff (abs ?a) ?x)" =>
            "(* (/ ?a (abs ?a)) (diff ?a ?x))"),
    ]
}

fn to_recexpr(e: &Expr, rec: &mut RecExpr<CasLang>) -> Id {
    match &**e {
        ExprKind::Lit(v) => {
            let n = NotNan::new(*v).unwrap_or_else(|_| NotNan::new(0.0).expect("0.0 is not NaN"));
            rec.add(CasLang::Num(n))
        }
        ExprKind::Var(s) => rec.add(CasLang::Symbol(s.as_str().into())),
        ExprKind::Neg(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CNeg([i])) }
        ExprKind::Add(a, b) => { let i = to_recexpr(a, rec); let j = to_recexpr(b, rec); rec.add(CasLang::CAdd([i, j])) }
        ExprKind::Mul(a, b) => { let i = to_recexpr(a, rec); let j = to_recexpr(b, rec); rec.add(CasLang::CMul([i, j])) }
        ExprKind::Div(a, b) => { let i = to_recexpr(a, rec); let j = to_recexpr(b, rec); rec.add(CasLang::CDiv([i, j])) }
        ExprKind::Pow(a, b) => { let i = to_recexpr(a, rec); let j = to_recexpr(b, rec); rec.add(CasLang::CPow([i, j])) }
        ExprKind::Sin(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CSin([i])) }
        ExprKind::Cos(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CCos([i])) }
        ExprKind::Exp(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CExp([i])) }
        ExprKind::Ln(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CLn([i])) }
        ExprKind::Tanh(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CTanh([i])) }
        ExprKind::Sqrt(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CSqrt([i])) }
        ExprKind::Abs(a) => { let i = to_recexpr(a, rec); rec.add(CasLang::CAbs([i])) }
    }
}

fn from_recexpr(rec: &RecExpr<CasLang>, id: Id) -> Expr {
    match &rec[id] {
        CasLang::Num(n) => Expr::lit(**n),
        CasLang::Symbol(s) => Expr::var(s.as_str()),
        CasLang::CNeg([a]) => -&from_recexpr(rec, *a),
        CasLang::CAdd([a, b]) => &from_recexpr(rec, *a) + &from_recexpr(rec, *b),
        CasLang::CMul([a, b]) => &from_recexpr(rec, *a) * &from_recexpr(rec, *b),
        CasLang::CDiv([a, b]) => &from_recexpr(rec, *a) / &from_recexpr(rec, *b),
        CasLang::CPow([a, b]) => Expr::pow(&from_recexpr(rec, *a), &from_recexpr(rec, *b)),
        CasLang::CSin([a]) => Expr::sin(&from_recexpr(rec, *a)),
        CasLang::CCos([a]) => Expr::cos(&from_recexpr(rec, *a)),
        CasLang::CExp([a]) => Expr::exp(&from_recexpr(rec, *a)),
        CasLang::CLn([a]) => Expr::ln(&from_recexpr(rec, *a)),
        CasLang::CTanh([a]) => Expr::tanh(&from_recexpr(rec, *a)),
        CasLang::CSqrt([a]) => Expr::sqrt(&from_recexpr(rec, *a)),
        CasLang::CAbs([a]) => Expr::abs(&from_recexpr(rec, *a)),
        // Residual diff node: fall back to tree-based differentiation
        CasLang::CDiff([a, x]) => {
            let inner = from_recexpr(rec, *a);
            let var_name = match &rec[*x] {
                CasLang::Symbol(s) => s.as_str().to_owned(),
                _ => "x".to_owned(),
            };
            diff(&inner, &var_name)
        }
    }
}

// ---------------------------------------------------------------------------
// Simplification — equality saturation via egg
// ---------------------------------------------------------------------------

/// Differentiate and simplify in a single e-graph saturation pass.
///
/// Wraps `expr` in a `Diff(expr, var)` node, then runs all rules
/// (algebraic + differentiation) via equality saturation.
#[must_use]
pub fn diff_simplify(expr: &Expr, var: &str) -> Expr {
    let mut rec = RecExpr::default();
    let expr_id = to_recexpr(expr, &mut rec);
    let var_id = rec.add(CasLang::Symbol(var.into()));
    rec.add(CasLang::CDiff([expr_id, var_id]));
    let runner = Runner::<CasLang, ConstFold, ()>::default()
        .with_expr(&rec)
        .run(&cas_rules());
    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize);
    let (_cost, best) = extractor.find_best(runner.roots[0]);
    from_recexpr(&best, Id::from(best.as_ref().len() - 1))
}

/// Simplify an expression via equality saturation (e-graph rewriting).
#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
    let mut rec = RecExpr::default();
    to_recexpr(expr, &mut rec);
    let runner = Runner::<CasLang, ConstFold, ()>::default()
        .with_expr(&rec)
        .run(&cas_rules());
    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize);
    let (_cost, best) = extractor.find_best(runner.roots[0]);
    from_recexpr(&best, Id::from(best.as_ref().len() - 1))
}

// ---------------------------------------------------------------------------
// Scalar evaluation
// ---------------------------------------------------------------------------

/// Evaluate `expr` to `f64` given variable bindings.
pub fn eval(expr: &Expr, vars: &HashMap<&str, f64>) -> Result<f64> {
    match &**expr {
        ExprKind::Var(s) => vars
            .get(s.as_str())
            .copied()
            .ok_or_else(|| Error::eval(format!("unbound variable: {s}"))),
        ExprKind::Lit(v) => Ok(*v),
        ExprKind::Neg(a) => Ok(-eval(a, vars)?),
        ExprKind::Add(a, b) => Ok(eval(a, vars)? + eval(b, vars)?),
        ExprKind::Mul(a, b) => Ok(eval(a, vars)? * eval(b, vars)?),
        ExprKind::Div(a, b) => Ok(eval(a, vars)? / eval(b, vars)?),
        ExprKind::Pow(a, b) => Ok(eval(a, vars)?.powf(eval(b, vars)?)),
        ExprKind::Sin(a) => Ok(eval(a, vars)?.sin()),
        ExprKind::Cos(a) => Ok(eval(a, vars)?.cos()),
        ExprKind::Exp(a) => Ok(eval(a, vars)?.exp()),
        ExprKind::Ln(a) => Ok(eval(a, vars)?.ln()),
        ExprKind::Tanh(a) => Ok(eval(a, vars)?.tanh()),
        ExprKind::Sqrt(a) => Ok(eval(a, vars)?.sqrt()),
        ExprKind::Abs(a) => Ok(eval(a, vars)?.abs()),
    }
}

// ---------------------------------------------------------------------------
// Tensor evaluation
// ---------------------------------------------------------------------------

/// Evaluate `expr` element-wise over tensor-valued variables.
///
/// Literal nodes broadcast to the shape inferred from the first variable.
pub fn eval_tensor<T: Scalar, B: Backend>(
    expr: &Expr,
    vars: &HashMap<&str, &Tensor<T, B>>,
) -> Result<Tensor<T, B>> {
    match &**expr {
        ExprKind::Var(s) => vars
            .get(s.as_str())
            .map(|t| (*t).clone())
            .ok_or_else(|| Error::eval(format!("unbound variable: {s}"))),

        ExprKind::Lit(v) => {
            let val = T::from_f64(*v);
            let (r, c) = infer_shape(expr, vars)?;
            Ok(Tensor::fill(r, c, val))
        }

        ExprKind::Neg(a) => Ok(-&eval_tensor(a, vars)?),
        ExprKind::Add(a, b) => {
            let ta = eval_tensor(a, vars)?;
            let tb = eval_tensor(b, vars)?;
            Ok(&ta + &tb)
        }
        ExprKind::Mul(a, b) => {
            let ta = eval_tensor(a, vars)?;
            let tb = eval_tensor(b, vars)?;
            Ok(ta.emul(&tb))
        }
        ExprKind::Div(a, b) => {
            let ta = eval_tensor(a, vars)?;
            let tb = eval_tensor(b, vars)?;
            Ok(ta.ediv(&tb))
        }
        ExprKind::Pow(a, b) => {
            let ta = eval_tensor(a, vars)?;
            match &**b {
                ExprKind::Lit(n) => Ok(ta.powf(T::from_f64(*n))),
                _ => {
                    let tb = eval_tensor(b, vars)?;
                    Ok(ta.ln().emul(&tb).exp())
                }
            }
        }
        ExprKind::Sin(a) => Ok(eval_tensor(a, vars)?.sin()),
        ExprKind::Cos(a) => Ok(eval_tensor(a, vars)?.cos()),
        ExprKind::Exp(a) => Ok(eval_tensor(a, vars)?.exp()),
        ExprKind::Ln(a) => Ok(eval_tensor(a, vars)?.ln()),
        ExprKind::Tanh(a) => Ok(eval_tensor(a, vars)?.tanh()),
        ExprKind::Sqrt(a) => Ok(eval_tensor(a, vars)?.sqrt()),
        ExprKind::Abs(a) => Ok(eval_tensor(a, vars)?.abs()),
    }
}

/// Walk the tree to find the shape of the first tensor variable.
fn infer_shape<T: Scalar, B: Backend>(
    expr: &Expr,
    vars: &HashMap<&str, &Tensor<T, B>>,
) -> Result<(usize, usize)> {
    match &**expr {
        ExprKind::Var(s) => vars
            .get(s.as_str())
            .map(|t| t.shape())
            .ok_or_else(|| Error::eval(format!("unbound variable: {s}"))),

        ExprKind::Lit(_) => vars
            .values()
            .next()
            .map(|t| t.shape())
            .ok_or_else(|| Error::eval("cannot infer shape: no tensor variables")),

        ExprKind::Neg(a)
        | ExprKind::Sin(a)
        | ExprKind::Cos(a)
        | ExprKind::Exp(a)
        | ExprKind::Ln(a)
        | ExprKind::Tanh(a)
        | ExprKind::Sqrt(a)
        | ExprKind::Abs(a) => infer_shape(a, vars),

        ExprKind::Add(a, b) | ExprKind::Mul(a, b) | ExprKind::Div(a, b) | ExprKind::Pow(a, b) => {
            infer_shape(a, vars).or_else(|_| infer_shape(b, vars))
        }
    }
}
