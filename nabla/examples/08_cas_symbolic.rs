//! # Symbolic CAS — nabla vs Julia conciseness comparison
//!
//! nabla LOC: ~20 | Julia LOC: ~20 (Symbolics.jl)
//! Julia advantage: `@variables x` macro, operator overloading feels native
//! nabla advantage: e-graph simplification (egg), guaranteed optimal form
//!
//! Run: cargo run --example 08_cas_symbolic --features cpu

use nabla::cas::{diff, diff_simplify, eval, simplify};
use nabla::prelude::*;
use std::collections::HashMap;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let x = Expr::var("x");

    // f(x) = x^3 + sin(x) * exp(x)
    let f = &Expr::pow(&x, &Expr::lit(3.0)) + &(&Expr::sin(&x) * &Expr::exp(&x));
    println!("f(x) = {f}");

    // Symbolic differentiation
    let df = diff(&f, "x");
    println!("f'(x) = {df}");

    // Simplify via e-graph
    let df_simple = simplify(&df);
    println!("f'(x) simplified = {df_simple}");

    // One-step diff+simplify
    let df2 = diff_simplify(&f, "x");
    println!("diff_simplify(f, x) = {df2}");

    // Evaluate at x = 1.0
    let vars = HashMap::from([("x", 1.0)]);
    let val = eval(&f, &vars)?;
    let dval = eval(&df_simple, &vars)?;
    println!("f(1.0) = {val:.6}");
    println!("f'(1.0) = {dval:.6}");

    Ok(())
}
