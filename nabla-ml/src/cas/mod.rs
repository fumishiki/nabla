#![allow(clippy::float_cmp, clippy::too_many_lines, clippy::single_match_else)]
#![allow(
    clippy::missing_errors_doc,
    clippy::implicit_hasher,
    clippy::many_single_char_names
)]

mod alg;
mod expr;

pub use alg::{
    diff, diff_simplify, eval, eval_tensor, gradient, hessian, jacobian, simplify, substitute,
};
pub use expr::{Expr, ExprKind, var};
