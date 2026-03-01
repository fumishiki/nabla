use std::collections::HashMap;

use egg::{
    Analysis, DidMerge, EGraph, Id, RecExpr, Rewrite, Runner, Subst, Symbol, Var, define_language,
    rewrite,
};
use nabla_core::backend::Backend;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;
use ordered_float::NotNan;

use super::expr::{Expr, ExprKind};

/// Symbolic differentiation of an expression with respect to a variable.
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
        // d/dx asin(a) = a' / sqrt(1 - a^2)
        ExprKind::Asin(a) => {
            &diff(a, var) / &Expr::sqrt(&(&Expr::lit(1.0) - &Expr::pow(a, &Expr::lit(2.0))))
        }
        // d/dx acos(a) = -a' / sqrt(1 - a^2)
        ExprKind::Acos(a) => {
            -&(&diff(a, var) / &Expr::sqrt(&(&Expr::lit(1.0) - &Expr::pow(a, &Expr::lit(2.0)))))
        }
        // d/dx atan(a) = a' / (1 + a^2)
        ExprKind::Atan(a) => &diff(a, var) / &(&Expr::lit(1.0) + &Expr::pow(a, &Expr::lit(2.0))),
        // d/dx sinh(a) = a' * cosh(a)
        ExprKind::Sinh(a) => &Expr::cosh(a) * &diff(a, var),
        // d/dx cosh(a) = a' * sinh(a)
        ExprKind::Cosh(a) => &Expr::sinh(a) * &diff(a, var),
        // d/dx asinh(a) = a' / sqrt(a^2 + 1)
        ExprKind::Asinh(a) => {
            &diff(a, var) / &Expr::sqrt(&(&Expr::pow(a, &Expr::lit(2.0)) + &Expr::lit(1.0)))
        }
        // d/dx acosh(a) = a' / sqrt(a^2 - 1)
        ExprKind::Acosh(a) => {
            &diff(a, var) / &Expr::sqrt(&(&Expr::pow(a, &Expr::lit(2.0)) - &Expr::lit(1.0)))
        }
        // d/dx atanh(a) = a' / (1 - a^2)
        ExprKind::Atanh(a) => &diff(a, var) / &(&Expr::lit(1.0) - &Expr::pow(a, &Expr::lit(2.0))),
        // d/dx tan(a) = a' * (1 + tan(a)^2)
        ExprKind::Tan(a) => {
            let ta = Expr::tan(a);
            let sec2 = &Expr::lit(1.0) + &Expr::pow(&ta, &Expr::lit(2.0));
            &sec2 * &diff(a, var)
        }
    }
}

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
        "asin" = CAsin([Id; 1]),
        "acos" = CAcos([Id; 1]),
        "atan" = CAtan([Id; 1]),
        "sinh" = CSinh([Id; 1]),
        "cosh" = CCosh([Id; 1]),
        "asinh" = CAsinh([Id; 1]),
        "acosh" = CAcosh([Id; 1]),
        "atanh" = CAtanh([Id; 1]),
        "tan" = CTan([Id; 1]),
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
                if *av > 0.0 {
                    NotNan::new(av.ln()).ok()
                } else {
                    None
                }
            }
            CasLang::CTanh([a]) => NotNan::new(x(a)?.tanh()).ok(),
            CasLang::CSqrt([a]) => {
                let av = x(a)?;
                if *av >= 0.0 {
                    NotNan::new(av.sqrt()).ok()
                } else {
                    None
                }
            }
            CasLang::CAbs([a]) => NotNan::new(x(a)?.abs()).ok(),
            CasLang::CAsin([a]) => NotNan::new(x(a)?.asin()).ok(),
            CasLang::CAcos([a]) => NotNan::new(x(a)?.acos()).ok(),
            CasLang::CAtan([a]) => NotNan::new(x(a)?.atan()).ok(),
            CasLang::CSinh([a]) => NotNan::new(x(a)?.sinh()).ok(),
            CasLang::CCosh([a]) => NotNan::new(x(a)?.cosh()).ok(),
            CasLang::CAsinh([a]) => NotNan::new(x(a)?.asinh()).ok(),
            CasLang::CAcosh([a]) => {
                let av = x(a)?;
                if *av >= 1.0 {
                    NotNan::new(av.acosh()).ok()
                } else {
                    None
                }
            }
            CasLang::CAtanh([a]) => {
                let av = x(a)?;
                if av.abs() < 1.0 {
                    NotNan::new(av.atanh()).ok()
                } else {
                    None
                }
            }
            CasLang::CTan([a]) => NotNan::new(x(a)?.tan()).ok(),
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

struct IsConst {
    v: Var,
}

impl egg::Condition<CasLang, ConstFold> for IsConst {
    fn check(&self, egraph: &mut EGraph<CasLang, ConstFold>, _: Id, subst: &Subst) -> bool {
        egraph[subst[self.v]].data.is_some()
    }
}

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
        // d[asin(a)]/dx = a' / sqrt(1 - a^2)
        rewrite!("diff-asin"; "(diff (asin ?a) ?x)" =>
            "(/ (diff ?a ?x) (sqrt (+ 1 (neg (^ ?a 2)))))"),
        // d[acos(a)]/dx = -a' / sqrt(1 - a^2)
        rewrite!("diff-acos"; "(diff (acos ?a) ?x)" =>
            "(neg (/ (diff ?a ?x) (sqrt (+ 1 (neg (^ ?a 2))))))"),
        // d[atan(a)]/dx = a' / (1 + a^2)
        rewrite!("diff-atan"; "(diff (atan ?a) ?x)" =>
            "(/ (diff ?a ?x) (+ 1 (^ ?a 2)))"),
        // d[sinh(a)]/dx = cosh(a) * a'
        rewrite!("diff-sinh"; "(diff (sinh ?a) ?x)" =>
            "(* (cosh ?a) (diff ?a ?x))"),
        // d[cosh(a)]/dx = sinh(a) * a'
        rewrite!("diff-cosh"; "(diff (cosh ?a) ?x)" =>
            "(* (sinh ?a) (diff ?a ?x))"),
        // d[asinh(a)]/dx = a' / sqrt(a^2 + 1)
        rewrite!("diff-asinh"; "(diff (asinh ?a) ?x)" =>
            "(/ (diff ?a ?x) (sqrt (+ (^ ?a 2) 1)))"),
        // d[acosh(a)]/dx = a' / sqrt(a^2 - 1)
        rewrite!("diff-acosh"; "(diff (acosh ?a) ?x)" =>
            "(/ (diff ?a ?x) (sqrt (+ (^ ?a 2) -1)))"),
        // d[atanh(a)]/dx = a' / (1 - a^2)
        rewrite!("diff-atanh"; "(diff (atanh ?a) ?x)" =>
            "(/ (diff ?a ?x) (+ 1 (neg (^ ?a 2))))"),
        // tan simplifications
        rewrite!("tan-zero"; "(tan 0)" => "0"),
        // tan(a) = sin(a) / cos(a)
        rewrite!("tan-def"; "(tan ?a)" => "(/ (sin ?a) (cos ?a))"),
        // d[tan(a)]/dx = (1 + tan(a)^2) * a'
        rewrite!("diff-tan"; "(diff (tan ?a) ?x)" =>
            "(* (+ 1 (^ (tan ?a) 2)) (diff ?a ?x))"),
    ]
}

fn recexpr_unary(rec: &mut RecExpr<CasLang>, a: &Expr, ctor: fn([Id; 1]) -> CasLang) -> Id {
    let i = to_recexpr(a, rec);
    rec.add(ctor([i]))
}

fn recexpr_binary(
    rec: &mut RecExpr<CasLang>,
    a: &Expr,
    b: &Expr,
    ctor: fn([Id; 2]) -> CasLang,
) -> Id {
    let i = to_recexpr(a, rec);
    let j = to_recexpr(b, rec);
    rec.add(ctor([i, j]))
}

fn to_recexpr(e: &Expr, rec: &mut RecExpr<CasLang>) -> Id {
    match &**e {
        ExprKind::Lit(v) => {
            let n = NotNan::new(*v).unwrap_or_else(|_| NotNan::new(0.0).expect("0.0 is not NaN"));
            rec.add(CasLang::Num(n))
        }
        ExprKind::Var(s) => rec.add(CasLang::Symbol(s.as_str().into())),
        ExprKind::Neg(a) => recexpr_unary(rec, a, CasLang::CNeg),
        ExprKind::Add(a, b) => recexpr_binary(rec, a, b, CasLang::CAdd),
        ExprKind::Mul(a, b) => recexpr_binary(rec, a, b, CasLang::CMul),
        ExprKind::Div(a, b) => recexpr_binary(rec, a, b, CasLang::CDiv),
        ExprKind::Pow(a, b) => recexpr_binary(rec, a, b, CasLang::CPow),
        ExprKind::Sin(a) => recexpr_unary(rec, a, CasLang::CSin),
        ExprKind::Cos(a) => recexpr_unary(rec, a, CasLang::CCos),
        ExprKind::Exp(a) => recexpr_unary(rec, a, CasLang::CExp),
        ExprKind::Ln(a) => recexpr_unary(rec, a, CasLang::CLn),
        ExprKind::Tanh(a) => recexpr_unary(rec, a, CasLang::CTanh),
        ExprKind::Sqrt(a) => recexpr_unary(rec, a, CasLang::CSqrt),
        ExprKind::Abs(a) => recexpr_unary(rec, a, CasLang::CAbs),
        ExprKind::Asin(a) => recexpr_unary(rec, a, CasLang::CAsin),
        ExprKind::Acos(a) => recexpr_unary(rec, a, CasLang::CAcos),
        ExprKind::Atan(a) => recexpr_unary(rec, a, CasLang::CAtan),
        ExprKind::Sinh(a) => recexpr_unary(rec, a, CasLang::CSinh),
        ExprKind::Cosh(a) => recexpr_unary(rec, a, CasLang::CCosh),
        ExprKind::Asinh(a) => recexpr_unary(rec, a, CasLang::CAsinh),
        ExprKind::Acosh(a) => recexpr_unary(rec, a, CasLang::CAcosh),
        ExprKind::Atanh(a) => recexpr_unary(rec, a, CasLang::CAtanh),
        ExprKind::Tan(a) => recexpr_unary(rec, a, CasLang::CTan),
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
        CasLang::CAsin([a]) => Expr::asin(&from_recexpr(rec, *a)),
        CasLang::CAcos([a]) => Expr::acos(&from_recexpr(rec, *a)),
        CasLang::CAtan([a]) => Expr::atan(&from_recexpr(rec, *a)),
        CasLang::CSinh([a]) => Expr::sinh(&from_recexpr(rec, *a)),
        CasLang::CCosh([a]) => Expr::cosh(&from_recexpr(rec, *a)),
        CasLang::CAsinh([a]) => Expr::asinh(&from_recexpr(rec, *a)),
        CasLang::CAcosh([a]) => Expr::acosh(&from_recexpr(rec, *a)),
        CasLang::CAtanh([a]) => Expr::atanh(&from_recexpr(rec, *a)),
        CasLang::CTan([a]) => Expr::tan(&from_recexpr(rec, *a)),
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

fn saturate(rec: &RecExpr<CasLang>) -> Expr {
    let runner = Runner::<CasLang, ConstFold, ()>::default()
        .with_expr(rec)
        .run(&cas_rules());
    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize);
    let (_cost, best) = extractor.find_best(runner.roots[0]);
    from_recexpr(&best, Id::from(best.as_ref().len() - 1))
}

/// Differentiate and simplify via e-graph rewriting.
#[must_use]
pub fn diff_simplify(expr: &Expr, var: &str) -> Expr {
    let mut rec = RecExpr::default();
    let expr_id = to_recexpr(expr, &mut rec);
    let var_id = rec.add(CasLang::Symbol(var.into()));
    rec.add(CasLang::CDiff([expr_id, var_id]));
    saturate(&rec)
}

/// Simplify an expression via e-graph equality saturation.
#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
    let mut rec = RecExpr::default();
    to_recexpr(expr, &mut rec);
    saturate(&rec)
}

/// Evaluate a symbolic expression with concrete variable bindings.
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
        ExprKind::Div(a, b) => {
            let denom = eval(b, vars)?;
            if denom.abs() < f64::EPSILON {
                return Err(Error::invalid("division by zero"));
            }
            Ok(eval(a, vars)? / denom)
        }
        ExprKind::Pow(a, b) => Ok(eval(a, vars)?.powf(eval(b, vars)?)),
        ExprKind::Sin(a) => Ok(eval(a, vars)?.sin()),
        ExprKind::Cos(a) => Ok(eval(a, vars)?.cos()),
        ExprKind::Exp(a) => Ok(eval(a, vars)?.exp()),
        ExprKind::Ln(a) => {
            let v = eval(a, vars)?;
            if v <= 0.0 {
                return Err(Error::invalid("ln: argument must be positive"));
            }
            Ok(v.ln())
        }
        ExprKind::Tanh(a) => Ok(eval(a, vars)?.tanh()),
        ExprKind::Sqrt(a) => {
            let v = eval(a, vars)?;
            if v < 0.0 {
                return Err(Error::invalid("sqrt: argument must be non-negative"));
            }
            Ok(v.sqrt())
        }
        ExprKind::Abs(a) => Ok(eval(a, vars)?.abs()),
        ExprKind::Asin(a) => {
            let v = eval(a, vars)?;
            if !(-1.0..=1.0).contains(&v) {
                return Err(Error::invalid("asin: argument must be in [-1, 1]"));
            }
            Ok(v.asin())
        }
        ExprKind::Acos(a) => {
            let v = eval(a, vars)?;
            if !(-1.0..=1.0).contains(&v) {
                return Err(Error::invalid("acos: argument must be in [-1, 1]"));
            }
            Ok(v.acos())
        }
        ExprKind::Atan(a) => Ok(eval(a, vars)?.atan()),
        ExprKind::Sinh(a) => Ok(eval(a, vars)?.sinh()),
        ExprKind::Cosh(a) => Ok(eval(a, vars)?.cosh()),
        ExprKind::Asinh(a) => Ok(eval(a, vars)?.asinh()),
        ExprKind::Acosh(a) => {
            let v = eval(a, vars)?;
            if v < 1.0 {
                return Err(Error::invalid("acosh: argument must be >= 1"));
            }
            Ok(v.acosh())
        }
        ExprKind::Atanh(a) => {
            let v = eval(a, vars)?;
            if v.abs() >= 1.0 {
                return Err(Error::invalid("atanh: argument must be in (-1, 1)"));
            }
            Ok(v.atanh())
        }
        ExprKind::Tan(a) => Ok(eval(a, vars)?.tan()),
    }
}

/// Evaluate a symbolic expression with tensor-valued variable bindings.
pub fn eval_tensor<T: Scalar, B: Backend>(
    expr: &Expr,
    vars: &HashMap<&str, &Tensor<T, B>>,
) -> Result<Tensor<T, B>> {
    match &**expr {
        ExprKind::Var(s) => vars
            .get(s.as_str())
            .copied()
            .cloned()
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
        ExprKind::Asin(a) => Ok(eval_tensor(a, vars)?.asin()),
        ExprKind::Acos(a) => Ok(eval_tensor(a, vars)?.acos()),
        ExprKind::Atan(a) => Ok(eval_tensor(a, vars)?.atan()),
        ExprKind::Sinh(a) => Ok(eval_tensor(a, vars)?.sinh()),
        ExprKind::Cosh(a) => Ok(eval_tensor(a, vars)?.cosh()),
        ExprKind::Asinh(a) => Ok(eval_tensor(a, vars)?.asinh()),
        ExprKind::Acosh(a) => Ok(eval_tensor(a, vars)?.acosh()),
        ExprKind::Atanh(a) => Ok(eval_tensor(a, vars)?.atanh()),
        ExprKind::Tan(a) => Ok(eval_tensor(a, vars)?.tan()),
    }
}

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
        | ExprKind::Abs(a)
        | ExprKind::Asin(a)
        | ExprKind::Acos(a)
        | ExprKind::Atan(a)
        | ExprKind::Sinh(a)
        | ExprKind::Cosh(a)
        | ExprKind::Asinh(a)
        | ExprKind::Acosh(a)
        | ExprKind::Atanh(a)
        | ExprKind::Tan(a) => infer_shape(a, vars),

        ExprKind::Add(a, b) | ExprKind::Mul(a, b) | ExprKind::Div(a, b) | ExprKind::Pow(a, b) => {
            infer_shape(a, vars).or_else(|_| infer_shape(b, vars))
        }
    }
}

/// Replace all occurrences of a variable with a sub-expression.
#[must_use]
pub fn substitute(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    match &**expr {
        ExprKind::Var(s) if s == var => replacement.clone(),
        ExprKind::Var(_) | ExprKind::Lit(_) => expr.clone(),
        ExprKind::Neg(a) => -&substitute(a, var, replacement),
        ExprKind::Add(a, b) => &substitute(a, var, replacement) + &substitute(b, var, replacement),
        ExprKind::Mul(a, b) => &substitute(a, var, replacement) * &substitute(b, var, replacement),
        ExprKind::Div(a, b) => &substitute(a, var, replacement) / &substitute(b, var, replacement),
        ExprKind::Pow(a, b) => Expr::pow(
            &substitute(a, var, replacement),
            &substitute(b, var, replacement),
        ),
        ExprKind::Sin(a) => Expr::sin(&substitute(a, var, replacement)),
        ExprKind::Cos(a) => Expr::cos(&substitute(a, var, replacement)),
        ExprKind::Exp(a) => Expr::exp(&substitute(a, var, replacement)),
        ExprKind::Ln(a) => Expr::ln(&substitute(a, var, replacement)),
        ExprKind::Tanh(a) => Expr::tanh(&substitute(a, var, replacement)),
        ExprKind::Sqrt(a) => Expr::sqrt(&substitute(a, var, replacement)),
        ExprKind::Abs(a) => Expr::abs(&substitute(a, var, replacement)),
        ExprKind::Asin(a) => Expr::asin(&substitute(a, var, replacement)),
        ExprKind::Acos(a) => Expr::acos(&substitute(a, var, replacement)),
        ExprKind::Atan(a) => Expr::atan(&substitute(a, var, replacement)),
        ExprKind::Sinh(a) => Expr::sinh(&substitute(a, var, replacement)),
        ExprKind::Cosh(a) => Expr::cosh(&substitute(a, var, replacement)),
        ExprKind::Asinh(a) => Expr::asinh(&substitute(a, var, replacement)),
        ExprKind::Acosh(a) => Expr::acosh(&substitute(a, var, replacement)),
        ExprKind::Atanh(a) => Expr::atanh(&substitute(a, var, replacement)),
        ExprKind::Tan(a) => Expr::tan(&substitute(a, var, replacement)),
    }
}

/// Compute the symbolic gradient of an expression with respect to multiple variables.
#[must_use]
pub fn gradient(expr: &Expr, vars: &[&str]) -> Vec<Expr> {
    vars.iter().map(|v| diff_simplify(expr, v)).collect()
}

/// Compute the symbolic Jacobian matrix of multiple expressions.
#[must_use]
pub fn jacobian(exprs: &[Expr], vars: &[&str]) -> Vec<Vec<Expr>> {
    exprs
        .iter()
        .map(|e| vars.iter().map(|v| diff_simplify(e, v)).collect())
        .collect()
}

/// Compute the symbolic Hessian matrix of a scalar expression.
#[must_use]
pub fn hessian(expr: &Expr, vars: &[&str]) -> Vec<Vec<Expr>> {
    vars.iter()
        .map(|vi| {
            let di = diff_simplify(expr, vi);
            vars.iter().map(|vj| diff_simplify(&di, vj)).collect()
        })
        .collect()
}
