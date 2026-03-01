//! # Symbolic CAS -- algebra in a few lines
//! Run: cargo run --example 08_cas_symbolic --features cpu

use nabla::prelude::*;

#[nabla::main(cpu)]
fn main() {
    let f = sym!(x ^ 3 + sin(x) * exp(x));
    println!("f(x) = {f}");

    let df = diff(&f, "x");
    println!("f'(x) = {df}");

    let df_simple = simplify(&df);
    println!("f'(x) simplified = {df_simple}");

    let df2 = diff_simplify(&f, "x");
    println!("diff_simplify(f, x) = {df2}");

    let vars = cas_vars! { x: 1.0 };
    let val = eval(&f, &vars)?;
    let dval = eval(&df_simple, &vars)?;
    println!("f(1.0) = {val:.6}");
    println!("f'(1.0) = {dval:.6}");
}
