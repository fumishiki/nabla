//! Verify: autograd matmul + leaky_relu gradient flow + training convergence.
//! Run: cargo run --bin verify_train --features cpu

use nabla::prelude::*;
use nabla_bench::{rand_tensor, kaiming};
use nabla_train::prelude::{Optimizer, Sgd};

type B = DefaultBackend;

fn grad_norm(g: &Tensor<f32>) -> f32 {
    let (m, n) = g.shape();
    let mut s = 0.0f32;
    for r in 0..m { for c in 0..n { let v = g.get(r, c); s += v * v; } }
    s.sqrt()
}

fn print_stats(name: &str, t: &Tensor<f32>) {
    let (m, n) = t.shape();
    let (mut min, mut max, mut sum, mut zeros) = (f32::MAX, f32::MIN, 0.0f32, 0usize);
    for r in 0..m { for c in 0..n {
        let v = t.get(r, c);
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v;
        if v == 0.0 { zeros += 1; }
    }}
    println!("  {name}: min={min:.4} max={max:.4} mean={:.4} zeros={zeros}/{}", sum / (m * n) as f32, m * n);
}

fn check_grad(name: &str, var: &Variable<f32, B>) {
    match var.grad() {
        Ok(g) => {
            let n = grad_norm(&g);
            println!("  {name} grad norm = {n:.6}  {}", if n > 0.0 { "PASS" } else { "FAIL" });
        }
        Err(e) => println!("  {name} grad FAILED: {e:?}"),
    }
}

fn main() {
    println!("=== Test 1: Single matmul, no activation ===");
    {
        let tape = Tape::new();
        let x_var = tape.variable(rand_tensor(2, 3)).ok().unwrap();
        let w_var = tape.variable(rand_tensor(3, 2).map(|v| v * 0.1)).ok().unwrap();
        let out = x_var.matmul(&w_var);
        let loss = out.emul(&out).sum_axis(1).sum_axis(0);
        println!("  loss = {:.6}", loss.data().get(0, 0));
        let _ = loss.backward();
        check_grad("w", &w_var);
    }

    println!("\n=== Test 2: Matmul + leaky_relu ===");
    {
        let tape = Tape::new();
        let x_var = tape.variable(rand_tensor(2, 3)).ok().unwrap();
        let w_var = tape.variable(rand_tensor(3, 4).map(|v| v * 0.5)).ok().unwrap();
        let h = x_var.matmul(&w_var).leaky_relu(0.01);
        let loss = h.emul(&h).sum_axis(1).sum_axis(0);
        println!("  loss = {:.6}", loss.data().get(0, 0));
        let _ = loss.backward();
        check_grad("w", &w_var);
    }

    println!("\n=== Test 3: Two matmuls + leaky_relu (small MLP 4→8→2) ===");
    {
        let tape = Tape::new();
        let x_var = tape.variable(rand_tensor(2, 4)).ok().unwrap();
        let w1_var = tape.variable(rand_tensor(4, 8).map(|v| v * 0.3)).ok().unwrap();
        let w2_var = tape.variable(rand_tensor(8, 2).map(|v| v * 0.3)).ok().unwrap();
        let t_var = tape.variable(rand_tensor(2, 2).map(|v| v * 0.1)).ok().unwrap();
        let h1 = x_var.matmul(&w1_var).leaky_relu(0.01);
        let out = h1.matmul(&w2_var);
        let diff = out.sub_var(&t_var);
        let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
        println!("  loss = {:.6}", loss.data().get(0, 0));
        let _ = loss.backward();
        check_grad("w1", &w1_var);
        check_grad("w2", &w2_var);
    }

    println!("\n=== Test 4: Full MLP 784→256→128→10, batch=1 ===");
    {
        let tape = Tape::new();
        let x_var = tape.variable(rand_tensor(1, 784)).ok().unwrap();
        let w1_var = tape.variable(kaiming(784, 256)).ok().unwrap();
        let w2_var = tape.variable(kaiming(256, 128)).ok().unwrap();
        let w3_var = tape.variable(kaiming(128, 10)).ok().unwrap();
        let t_var = tape.variable(rand_tensor(1, 10).map(|v| v * 0.1)).ok().unwrap();
        let h1 = x_var.matmul(&w1_var).leaky_relu(0.01);
        print_stats("h1_act", h1.data());
        let h2 = h1.matmul(&w2_var).leaky_relu(0.01);
        print_stats("h2_act", h2.data());
        let out = h2.matmul(&w3_var);
        print_stats("out", out.data());
        let diff = out.sub_var(&t_var);
        let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
        println!("  loss = {:.6}", loss.data().get(0, 0));
        let _ = loss.backward();
        check_grad("w1", &w1_var);
        check_grad("w2", &w2_var);
        check_grad("w3", &w3_var);
    }

    println!("\n=== Test 5: Training loop convergence (4→8→2, 50 steps) ===");
    {
        let mut w1 = kaiming(4, 8);
        let mut w2 = kaiming(8, 2);
        let x = rand_tensor(4, 4);
        let target = rand_tensor(4, 2).map(|v| v * 0.5);
        let params = vec![&w1, &w2];
        let mut sgd: Sgd<f32, B> = Sgd::from_params(0.01, &params);
        drop(params);
        let mut losses = Vec::new();
        for step in 0..50 {
            let tape = Tape::new();
            let x_var = tape.variable(x.clone()).ok().unwrap();
            let w1_var = tape.variable(w1.clone()).ok().unwrap();
            let w2_var = tape.variable(w2.clone()).ok().unwrap();
            let t_var = tape.variable(target.clone()).ok().unwrap();
            let h = x_var.matmul(&w1_var).leaky_relu(0.01);
            let out = h.matmul(&w2_var);
            let diff = out.sub_var(&t_var);
            let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
            let l = loss.data().get(0, 0);
            if step % 10 == 0 || step == 49 { println!("  step {step:3}: loss = {l:.6}"); }
            losses.push(l);
            let _ = loss.backward();
            let g1 = w1_var.grad().ok().unwrap();
            let g2 = w2_var.grad().ok().unwrap();
            sgd.step(&mut [&mut w1, &mut w2], &[&g1, &g2]);
        }
        let converged = losses.last().unwrap() < &(losses[0] * 0.5);
        println!("  initial={:.6} final={:.6}  {}", losses[0], losses.last().unwrap(),
            if converged { "PASS (>50% reduction)" } else { "FAIL" });
    }

    println!("\n=== Test 6: Full MLP training 784→256→128→10, 30 steps ===");
    {
        let mut w1 = kaiming(784, 256);
        let mut w2 = kaiming(256, 128);
        let mut w3 = kaiming(128, 10);
        let x = rand_tensor(2, 784);
        let target = rand_tensor(2, 10).map(|v| v * 0.1);
        let params = vec![&w1, &w2, &w3];
        let mut sgd: Sgd<f32, B> = Sgd::from_params(0.001, &params);
        drop(params);
        let mut losses = Vec::new();
        for step in 0..30 {
            let tape = Tape::new();
            let x_var = tape.variable(x.clone()).ok().unwrap();
            let w1_var = tape.variable(w1.clone()).ok().unwrap();
            let w2_var = tape.variable(w2.clone()).ok().unwrap();
            let w3_var = tape.variable(w3.clone()).ok().unwrap();
            let t_var = tape.variable(target.clone()).ok().unwrap();
            let h1 = x_var.matmul(&w1_var).leaky_relu(0.01);
            let h2 = h1.matmul(&w2_var).leaky_relu(0.01);
            let out = h2.matmul(&w3_var);
            let diff = out.sub_var(&t_var);
            let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
            let l = loss.data().get(0, 0);
            if step % 5 == 0 || step == 29 { println!("  step {step:3}: loss = {l:.6}"); }
            losses.push(l);
            let _ = loss.backward();
            let g1 = w1_var.grad().ok().unwrap();
            let g2 = w2_var.grad().ok().unwrap();
            let g3 = w3_var.grad().ok().unwrap();
            sgd.step(&mut [&mut w1, &mut w2, &mut w3], &[&g1, &g2, &g3]);
        }
        let converged = losses.last().unwrap() < &(losses[0] * 0.5);
        println!("  initial={:.6} final={:.6}  {}", losses[0], losses.last().unwrap(),
            if converged { "PASS (>50% reduction)" } else { "FAIL" });
    }
}
