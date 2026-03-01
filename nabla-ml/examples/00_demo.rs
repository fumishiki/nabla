//! nabla showcase — matmul, solve, SVD, einsum, autodiff in 20 lines.
//! Run: cargo run --example 00_demo --features cpu

use nabla::prelude::*;

fn main() {
    let a = mat![[3.0_f64, 1.0], [1.0, 2.0]];
    let b = mat![[9.0_f64], [8.0]];

    // Solve Ax = b
    let x = a.solve(&b).expect("solve failed");
    println!("Ax = b  =>  x = [{:.2}, {:.2}]", x.get(0, 0), x.get(1, 0));

    // SVD
    let svd = a.svd().expect("svd failed");
    println!("SVD     =>  singular values = [{:.4}, {:.4}]", svd.s()[0], svd.s()[1]);

    // Einsum — batch matmul
    let m1 = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let m2 = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let prod: Tensor<f64> = einsum!(c[i,j] = m1[i,k] * m2[k,j]);
    println!("einsum  =>  (1,2;3,4) @ (5,6;7,8) = [{:.0}, {:.0}; {:.0}, {:.0}]",
        prod.get(0, 0), prod.get(0, 1), prod.get(1, 0), prod.get(1, 1));

    // Autodiff — ∂/∂w of ||Wx||²
    let tape: std::rc::Rc<Tape<f64, DefaultBackend>> = Tape::new();
    let w = tape.variable(mat![[1.0_f64, 2.0], [3.0, 4.0]]).expect("var");
    let inp = tape.variable(mat![[1.0_f64], [1.0]]).expect("var");
    let out = w.matmul(&inp);
    let loss = out.emul(&out).sum_axis(1).sum_axis(0);
    println!("loss    =>  {:.4}", loss.data().get(0, 0));
    loss.backward().expect("backward");
    let dw = w.grad().expect("grad");
    println!("dL/dW   =>  [{:.1}, {:.1}; {:.1}, {:.1}]",
        dw.get(0, 0), dw.get(0, 1), dw.get(1, 0), dw.get(1, 1));

    // Symbolic CAS
    let f = sym!(x^2 * sin(x));
    let df = diff(&f, "x");
    println!("d/dx(x²sin(x)) = {df}");
}
