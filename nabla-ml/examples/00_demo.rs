//! nabla showcase — REPL-style demo with colored output.
//! Run: cargo run --example 00_demo --features cpu

use nabla::prelude::*;
use std::io::{self, Write};
use std::thread::sleep;
use std::time::Duration;

const BOLD: &str = "\x1b[1m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

macro_rules! code {
    ($($line:expr),+ $(,)?) => {
        $( println!("{CYAN}{BOLD}  {}{RESET}", $line); )+
        io::stdout().flush().unwrap();
        sleep(Duration::from_millis(600));
    };
}
macro_rules! out {
    ($fmt:expr $(, $arg:expr)*) => {
        println!("{GREEN}  ➜ {}{RESET}", format!($fmt $(, $arg)*));
        io::stdout().flush().unwrap();
        sleep(Duration::from_millis(1500));
    };
}

#[cfg(not(feature = "cpu"))]
fn main() {
    println!("00_demo is CPU-only. Re-run with --features cpu.");
}

#[cfg(feature = "cpu")]
fn main() {
    println!("\n{BOLD}  ∇ nabla{RESET} {DIM}— GPU math for Rust, no C++ required{RESET}\n");
    sleep(Duration::from_millis(500));

    // 1. Solve Ax = b
    println!("{YELLOW}  ── Solve Ax = b ──{RESET}");
    code!(
        "let a = mat![[3.0, 1.0], [1.0, 2.0]];",
        "let b = mat![[9.0], [8.0]];",
        "a.solve(&b)?"
    );
    let a = mat![[3.0_f64, 1.0], [1.0, 2.0]];
    let b = mat![[9.0_f64], [8.0]];
    let x = a.solve(&b).expect("solve");
    out!("x = [{:.1}, {:.1}]", x.get(0, 0), x.get(1, 0));

    // 2. Einsum
    println!("\n{YELLOW}  ── Einsum ──{RESET}");
    code!("einsum!(c[i,j] = a[i,k] * b[k,j])");
    let m1 = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let m2 = mat![[5.0_f64, 6.0], [7.0, 8.0]];
    let prod: Tensor<f64> = einsum!(c[i,j] = m1[i,k] * m2[k,j]);
    out!(
        "[{:.0}, {:.0}; {:.0}, {:.0}]",
        prod.get(0, 0),
        prod.get(0, 1),
        prod.get(1, 0),
        prod.get(1, 1)
    );

    // 3. Kernel Fusion
    println!("\n{YELLOW}  ── GPU Kernel Fusion ──{RESET}");
    code!("fuse!(x.sin().powf(2.0) + x.cos())");
    let x: Tensor<f64> = mat![[1.0_f64, 2.0]];
    let y: Tensor<f64> = fuse!(x.sin().powf(2.0) + x.cos());
    out!("y = [{:.2}, {:.2}]", y.get(0, 0), y.get(0, 1));

    // 4. Autodiff
    println!("\n{YELLOW}  ── Autodiff ──{RESET}");
    code!(
        "let tape = Tape::new();",
        "let w = tape.var(mat![[1, 2], [3, 4]]);",
        "let loss = (&w * &x).norm_sq();",
        "loss.backward(); let dw = w.grad();"
    );
    let tape: std::rc::Rc<Tape<f64, DefaultBackend>> = Tape::new();
    let w = tape
        .variable(mat![[1.0_f64, 2.0], [3.0, 4.0]])
        .expect("var");
    let inp = tape.variable(mat![[1.0_f64], [1.0]]).expect("var");
    let o = w.matmul(&inp);
    let loss = o.emul(&o).sum_axis(1).sum_axis(0);
    loss.backward().expect("backward");
    let dw = w.grad().expect("grad");
    out!(
        "∂L/∂W = [{:.0}, {:.0}; {:.0}, {:.0}]",
        dw.get(0, 0),
        dw.get(0, 1),
        dw.get(1, 0),
        dw.get(1, 1)
    );

    // 5. Symbolic CAS
    println!("\n{YELLOW}  ── Symbolic CAS ──{RESET}");
    code!("simplify(&diff(&sym!(x^2 * sin(x)), \"x\"))");
    let f = sym!(x ^ 2 * sin(x));
    let df = simplify(&diff(&f, "x"));
    out!("{df}");

    println!();
}
