// cas.rs — Symbolic Computer Algebra System (CAS) for nabla.
//
// Design:
//   - Expr is a newtype wrapper around Arc<ExprKind> for structural sharing.
//   - Sub is not a variant: a - b ≡ Add(a, Neg(b)) — halves simplification rules.
//   - diff/simplify/eval are pure recursive functions (no mutation).

// CAS literal comparison is intentionally exact (0.0, 1.0 are constructed, not computed).
#![allow(clippy::float_cmp, clippy::too_many_lines, clippy::single_match_else)]
#![allow(clippy::missing_errors_doc, clippy::implicit_hasher)]

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::scalar::Scalar;
use crate::tensor::Tensor;

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

    /// `sin(e)`.
    #[must_use]
    pub fn sin(e: &Self) -> Self {
        Self::wrap(ExprKind::Sin(e.clone()))
    }

    /// `cos(e)`.
    #[must_use]
    pub fn cos(e: &Self) -> Self {
        Self::wrap(ExprKind::Cos(e.clone()))
    }

    /// `exp(e)`.
    #[must_use]
    pub fn exp(e: &Self) -> Self {
        Self::wrap(ExprKind::Exp(e.clone()))
    }

    /// `ln(e)`.
    #[must_use]
    pub fn ln(e: &Self) -> Self {
        Self::wrap(ExprKind::Ln(e.clone()))
    }

    /// `tanh(e)`.
    #[must_use]
    pub fn tanh(e: &Self) -> Self {
        Self::wrap(ExprKind::Tanh(e.clone()))
    }

    /// `sqrt(e)`.
    #[must_use]
    pub fn sqrt(e: &Self) -> Self {
        Self::wrap(ExprKind::Sqrt(e.clone()))
    }

    /// `abs(e)`.
    #[must_use]
    pub fn abs(e: &Self) -> Self {
        Self::wrap(ExprKind::Abs(e.clone()))
    }

    /// `base ^ exp`.
    #[must_use]
    pub fn pow(base: &Self, exp: &Self) -> Self {
        Self::wrap(ExprKind::Pow(base.clone(), exp.clone()))
    }
}

// ---------------------------------------------------------------------------
// Operator overloads
// ---------------------------------------------------------------------------

impl Add for &Expr {
    type Output = Expr;
    fn add(self, rhs: &Expr) -> Expr {
        Expr::wrap(ExprKind::Add(self.clone(), rhs.clone()))
    }
}

impl Mul for &Expr {
    type Output = Expr;
    fn mul(self, rhs: &Expr) -> Expr {
        Expr::wrap(ExprKind::Mul(self.clone(), rhs.clone()))
    }
}

impl Div for &Expr {
    type Output = Expr;
    fn div(self, rhs: &Expr) -> Expr {
        Expr::wrap(ExprKind::Div(self.clone(), rhs.clone()))
    }
}

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

// Mixed: &Expr op f64

impl Add<f64> for &Expr {
    type Output = Expr;
    fn add(self, rhs: f64) -> Expr {
        self + &Expr::lit(rhs)
    }
}

impl Mul<f64> for &Expr {
    type Output = Expr;
    fn mul(self, rhs: f64) -> Expr {
        self * &Expr::lit(rhs)
    }
}

// Mixed: f64 op &Expr

impl Add<&Expr> for f64 {
    type Output = Expr;
    fn add(self, rhs: &Expr) -> Expr {
        &Expr::lit(self) + rhs
    }
}

impl Mul<&Expr> for f64 {
    type Output = Expr;
    fn mul(self, rhs: &Expr) -> Expr {
        &Expr::lit(self) * rhs
    }
}

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
// Simplification
// ---------------------------------------------------------------------------

/// Simplify an expression bottom-up: constant folding + algebraic identities.
#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
    match &**expr {
        ExprKind::Var(_) | ExprKind::Lit(_) => expr.clone(),

        ExprKind::Neg(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(-v),
                ExprKind::Neg(inner) => inner.clone(),
                _ => Expr::wrap(ExprKind::Neg(a)),
            }
        }

        ExprKind::Add(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&*a, &*b) {
                (ExprKind::Lit(x), ExprKind::Lit(y)) => Expr::lit(x + y),
                (_, ExprKind::Lit(v)) if *v == 0.0 => a,
                (ExprKind::Lit(v), _) if *v == 0.0 => b,
                _ => Expr::wrap(ExprKind::Add(a, b)),
            }
        }

        ExprKind::Mul(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&*a, &*b) {
                (ExprKind::Lit(x), ExprKind::Lit(y)) => Expr::lit(x * y),
                (_, ExprKind::Lit(v)) if *v == 1.0 => a,
                (ExprKind::Lit(v), _) if *v == 1.0 => b,
                (_, ExprKind::Lit(v)) if *v == 0.0 => Expr::lit(0.0),
                (ExprKind::Lit(v), _) if *v == 0.0 => Expr::lit(0.0),
                _ => Expr::wrap(ExprKind::Mul(a, b)),
            }
        }

        ExprKind::Div(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&*a, &*b) {
                (ExprKind::Lit(x), ExprKind::Lit(y)) => Expr::lit(x / y),
                (_, ExprKind::Lit(v)) if *v == 1.0 => a,
                (ExprKind::Lit(v), _) if *v == 0.0 => Expr::lit(0.0),
                _ => Expr::wrap(ExprKind::Div(a, b)),
            }
        }

        ExprKind::Pow(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&*a, &*b) {
                (ExprKind::Lit(x), ExprKind::Lit(y)) => Expr::lit(x.powf(*y)),
                (_, ExprKind::Lit(v)) if *v == 0.0 => Expr::lit(1.0),
                (_, ExprKind::Lit(v)) if *v == 1.0 => a,
                _ => Expr::wrap(ExprKind::Pow(a, b)),
            }
        }

        ExprKind::Sin(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.sin()),
                _ => Expr::wrap(ExprKind::Sin(a)),
            }
        }

        ExprKind::Cos(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.cos()),
                _ => Expr::wrap(ExprKind::Cos(a)),
            }
        }

        ExprKind::Exp(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.exp()),
                ExprKind::Ln(inner) => inner.clone(),
                _ => Expr::wrap(ExprKind::Exp(a)),
            }
        }

        ExprKind::Ln(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.ln()),
                ExprKind::Exp(inner) => inner.clone(),
                _ => Expr::wrap(ExprKind::Ln(a)),
            }
        }

        ExprKind::Tanh(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.tanh()),
                _ => Expr::wrap(ExprKind::Tanh(a)),
            }
        }

        ExprKind::Sqrt(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.sqrt()),
                _ => Expr::wrap(ExprKind::Sqrt(a)),
            }
        }

        ExprKind::Abs(a) => {
            let a = simplify(a);
            match &*a {
                ExprKind::Lit(v) => Expr::lit(v.abs()),
                _ => Expr::wrap(ExprKind::Abs(a)),
            }
        }
    }
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
            Ok(ta.mul_elem(&tb))
        }
        ExprKind::Div(a, b) => {
            let ta = eval_tensor(a, vars)?;
            let tb = eval_tensor(b, vars)?;
            Ok(ta.div_elem(&tb))
        }
        ExprKind::Pow(a, b) => {
            let ta = eval_tensor(a, vars)?;
            match &**b {
                ExprKind::Lit(n) => Ok(ta.powf(T::from_f64(*n))),
                _ => {
                    let tb = eval_tensor(b, vars)?;
                    Ok(ta.ln().mul_elem(&tb).exp())
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
