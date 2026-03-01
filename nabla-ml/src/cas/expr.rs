use std::fmt;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::sync::Arc;

fn expr_eq(a: &ExprKind, b: &ExprKind) -> bool {
    match (a, b) {
        (ExprKind::Var(sa), ExprKind::Var(sb)) => sa == sb,
        (ExprKind::Lit(va), ExprKind::Lit(vb)) => va.to_bits() == vb.to_bits(),
        (ExprKind::Neg(ea), ExprKind::Neg(eb))
        | (ExprKind::Sin(ea), ExprKind::Sin(eb))
        | (ExprKind::Cos(ea), ExprKind::Cos(eb))
        | (ExprKind::Tan(ea), ExprKind::Tan(eb))
        | (ExprKind::Exp(ea), ExprKind::Exp(eb))
        | (ExprKind::Ln(ea), ExprKind::Ln(eb))
        | (ExprKind::Tanh(ea), ExprKind::Tanh(eb))
        | (ExprKind::Sqrt(ea), ExprKind::Sqrt(eb))
        | (ExprKind::Abs(ea), ExprKind::Abs(eb))
        | (ExprKind::Asin(ea), ExprKind::Asin(eb))
        | (ExprKind::Acos(ea), ExprKind::Acos(eb))
        | (ExprKind::Atan(ea), ExprKind::Atan(eb))
        | (ExprKind::Sinh(ea), ExprKind::Sinh(eb))
        | (ExprKind::Cosh(ea), ExprKind::Cosh(eb))
        | (ExprKind::Asinh(ea), ExprKind::Asinh(eb))
        | (ExprKind::Acosh(ea), ExprKind::Acosh(eb))
        | (ExprKind::Atanh(ea), ExprKind::Atanh(eb)) => expr_eq(ea, eb),
        (ExprKind::Add(la, ra), ExprKind::Add(lb, rb))
        | (ExprKind::Mul(la, ra), ExprKind::Mul(lb, rb))
        | (ExprKind::Div(la, ra), ExprKind::Div(lb, rb))
        | (ExprKind::Pow(la, ra), ExprKind::Pow(lb, rb)) => expr_eq(la, lb) && expr_eq(ra, rb),
        _ => false,
    }
}

/// Node kind for a symbolic expression tree.
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
    /// Inverse sine (arcsin).
    Asin(Expr),
    /// Inverse cosine (arccos).
    Acos(Expr),
    /// Inverse tangent (arctan).
    Atan(Expr),
    /// Hyperbolic sine.
    Sinh(Expr),
    /// Hyperbolic cosine.
    Cosh(Expr),
    /// Inverse hyperbolic sine.
    Asinh(Expr),
    /// Inverse hyperbolic cosine.
    Acosh(Expr),
    /// Inverse hyperbolic tangent.
    Atanh(Expr),
    /// Tangent.
    Tan(Expr),
}

impl PartialEq for ExprKind {
    fn eq(&self, other: &Self) -> bool {
        expr_eq(self, other)
    }
}
impl Eq for ExprKind {}

/// Symbolic mathematical expression (Arc-wrapped).
#[derive(Debug, Clone)]
pub struct Expr(Arc<ExprKind>);

impl Deref for Expr {
    type Target = ExprKind;
    #[inline]
    fn deref(&self) -> &ExprKind {
        &self.0
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        expr_eq(self, other)
    }
}
impl Eq for Expr {}

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
cas_unary!(
    sin => Sin, cos => Cos, tan => Tan, exp => Exp, ln => Ln, tanh => Tanh, sqrt => Sqrt, abs => Abs,
    asin => Asin, acos => Acos, atan => Atan,
    sinh => Sinh, cosh => Cosh,
    asinh => Asinh, acosh => Acosh, atanh => Atanh,
);

macro_rules! cas_method {
    ($($method:ident => $assoc:ident),+ $(,)?) => {
        impl Expr {
            $(
                #[doc = concat!("Method form of [`Expr::", stringify!($assoc), "`].")]
                #[inline]
                #[must_use]
                pub fn $method(&self) -> Self { Self::$assoc(self) }
            )+
        }
    };
}
cas_method!(
    sin_ => sin, cos_ => cos, tan_ => tan, exp_ => exp, ln_ => ln, tanh_ => tanh, sqrt_ => sqrt, abs_ => abs,
    asin_ => asin, acos_ => acos, atan_ => atan,
    sinh_ => sinh, cosh_ => cosh,
    asinh_ => asinh, acosh_ => acosh, atanh_ => atanh,
);

impl Expr {
    /// `self ^ n` where `n` is an `f64` literal.
    #[inline]
    #[must_use]
    pub fn powf(&self, n: f64) -> Self {
        Self::pow(self, &Self::lit(n))
    }

    /// `self ^ n` where `n` is an `i32` literal.
    #[inline]
    #[must_use]
    pub fn powi(&self, n: i32) -> Self {
        Self::pow(self, &Self::lit(f64::from(n)))
    }
}

impl From<f64> for Expr {
    #[inline]
    fn from(v: f64) -> Self {
        Self::lit(v)
    }
}

impl From<i32> for Expr {
    #[inline]
    fn from(v: i32) -> Self {
        Self::lit(f64::from(v))
    }
}

/// Create a named symbolic variable.
#[inline]
#[must_use]
pub fn var(name: &str) -> Expr {
    Expr::var(name)
}

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
impl_expr_f64_binop!(Add, add; Mul, mul; Sub, sub; Div, div);

macro_rules! impl_owned_binop {
    ($($trait:ident, $method:ident);+ $(;)?) => {
        $(
            // owned + owned
            impl $trait for Expr {
                type Output = Expr;
                #[inline]
                fn $method(self, rhs: Expr) -> Expr { $trait::$method(&self, &rhs) }
            }
            // owned + ref
            impl $trait<&Expr> for Expr {
                type Output = Expr;
                #[inline]
                fn $method(self, rhs: &Expr) -> Expr { $trait::$method(&self, rhs) }
            }
            // ref + owned
            impl $trait<Expr> for &Expr {
                type Output = Expr;
                #[inline]
                fn $method(self, rhs: Expr) -> Expr { $trait::$method(self, &rhs) }
            }
        )+
    };
}
impl_owned_binop!(Add, add; Sub, sub; Mul, mul; Div, div);

impl Neg for Expr {
    type Output = Expr;
    #[inline]
    fn neg(self) -> Expr {
        Neg::neg(&self)
    }
}

macro_rules! impl_owned_f64_binop {
    ($($trait:ident, $method:ident);+ $(;)?) => {
        $(
            impl $trait<f64> for Expr {
                type Output = Expr;
                #[inline]
                fn $method(self, rhs: f64) -> Expr { $trait::$method(&self, rhs) }
            }
            impl $trait<Expr> for f64 {
                type Output = Expr;
                #[inline]
                fn $method(self, rhs: Expr) -> Expr { $trait::$method(self, &rhs) }
            }
        )+
    };
}
impl_owned_f64_binop!(Add, add; Sub, sub; Mul, mul; Div, div);

fn precedence(e: &ExprKind) -> u8 {
    match e {
        ExprKind::Add(_, _) => 0,
        ExprKind::Mul(_, _) | ExprKind::Div(_, _) => 1,
        ExprKind::Pow(_, _) => 2,
        ExprKind::Var(_)
        | ExprKind::Lit(_)
        | ExprKind::Neg(_)
        | ExprKind::Sin(_)
        | ExprKind::Cos(_)
        | ExprKind::Tan(_)
        | ExprKind::Exp(_)
        | ExprKind::Ln(_)
        | ExprKind::Tanh(_)
        | ExprKind::Sqrt(_)
        | ExprKind::Abs(_)
        | ExprKind::Asin(_)
        | ExprKind::Acos(_)
        | ExprKind::Atan(_)
        | ExprKind::Sinh(_)
        | ExprKind::Cosh(_)
        | ExprKind::Asinh(_)
        | ExprKind::Acosh(_)
        | ExprKind::Atanh(_) => 3,
    }
}

fn fmt_child(f: &mut fmt::Formatter<'_>, child: &Expr, parent_prec: u8) -> fmt::Result {
    if precedence(child) < parent_prec {
        write!(f, "({child})")
    } else {
        write!(f, "{child}")
    }
}

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
            Self::Neg(a) => {
                write!(f, "-")?;
                fmt_child(f, a, 3)
            }
            Self::Add(a, b) => {
                fmt_child(f, a, 0)?;
                write!(f, " + ")?;
                fmt_child(f, b, 0)
            }
            Self::Mul(a, b) => {
                fmt_child(f, a, 1)?;
                write!(f, " * ")?;
                fmt_child(f, b, 1)
            }
            Self::Div(a, b) => {
                fmt_child(f, a, 1)?;
                write!(f, " / ")?;
                // Right operand of div needs strictly higher precedence to avoid ambiguity
                fmt_child(f, b, 2)
            }
            Self::Pow(a, b) => {
                // Base needs strictly higher precedence (atoms/functions ok, binops need parens)
                fmt_child(f, a, 3)?;
                write!(f, " ^ ")?;
                // Exponent: right-associative, so same-prec is ok
                fmt_child(f, b, 2)
            }
            Self::Sin(a) => write!(f, "sin({a})"),
            Self::Cos(a) => write!(f, "cos({a})"),
            Self::Exp(a) => write!(f, "exp({a})"),
            Self::Ln(a) => write!(f, "ln({a})"),
            Self::Tanh(a) => write!(f, "tanh({a})"),
            Self::Sqrt(a) => write!(f, "sqrt({a})"),
            Self::Abs(a) => write!(f, "abs({a})"),
            Self::Asin(a) => write!(f, "asin({a})"),
            Self::Acos(a) => write!(f, "acos({a})"),
            Self::Atan(a) => write!(f, "atan({a})"),
            Self::Sinh(a) => write!(f, "sinh({a})"),
            Self::Cosh(a) => write!(f, "cosh({a})"),
            Self::Asinh(a) => write!(f, "asinh({a})"),
            Self::Acosh(a) => write!(f, "acosh({a})"),
            Self::Atanh(a) => write!(f, "atanh({a})"),
            Self::Tan(a) => write!(f, "tan({a})"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}
