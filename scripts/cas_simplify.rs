#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nabla = { path = "../nabla", features = ["cpu"] }
//! ```

use nabla::cas::{Expr, diff_simplify, simplify};
use nabla::prelude::*;

fn main() {
    let x = Expr::var("x");

    // d/dx(sin(x)^2 + cos(x)^2) should simplify to 0
    let sin2_cos2 = &Expr::pow(&Expr::sin(&x), &Expr::lit(2.0))
        + &Expr::pow(&Expr::cos(&x), &Expr::lit(2.0));
    let deriv = diff_simplify(&sin2_cos2, "x");
    println!("d/dx(sin(x)^2 + cos(x)^2) = {deriv}");

    // Simplify sin(x)^2 + cos(x)^2 -> 1
    let simplified = simplify(&sin2_cos2);
    println!("simplify(sin(x)^2 + cos(x)^2) = {simplified}");

    // d/dx(exp(2*x)) = 2*exp(2*x)
    let exp_2x = Expr::exp(&(&Expr::lit(2.0) * &x));
    let d_exp = diff_simplify(&exp_2x, "x");
    println!("d/dx(exp(2x)) = {d_exp}");

    // fuse! demo: element-wise chain on tensor
    let t: Tensor<f64> = mat![[0.0_f64, 1.0], [2.0, 3.0]];
    let fused: Tensor<f64> = fuse!(t.exp().ln(); t);
    println!("fuse!(t.exp().ln()) = [{:.4}, {:.4}; {:.4}, {:.4}]",
        fused.get(0, 0), fused.get(0, 1), fused.get(1, 0), fused.get(1, 1));
}
