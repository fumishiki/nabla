//! Profile: break down training step into phases.
//! Run: cargo run --release --bin profile_train --features cuda

use nabla::prelude::*;
use nabla_bench::{rand_tensor, kaiming, gpu_sync, must};
use nabla_train::prelude::{Optimizer, Sgd};
use std::time::Instant;

type B = DefaultBackend;

fn main() {
    let mut w1 = kaiming(784, 256);
    let mut w2 = kaiming(256, 128);
    let mut w3 = kaiming(128, 10);
    let params = vec![&w1, &w2, &w3];
    let mut sgd: Sgd<f32, B> = Sgd::from_params(0.001, &params);
    drop(params);

    for &batch in &[1usize, 32, 128] {
        let x = rand_tensor(batch, 784);
        let target = rand_tensor(batch, 10).map(|v| v * 0.1);

        for _ in 0..3 {
            let tape = Tape::new();
            let xv = must!(tape.variable(x.clone()));
            let w1v = must!(tape.variable(w1.clone()));
            let w2v = must!(tape.variable(w2.clone()));
            let w3v = must!(tape.variable(w3.clone()));
            let tv = must!(tape.variable(target.clone()));
            let h1 = xv.matmul(&w1v).leaky_relu(0.01);
            let h2 = h1.matmul(&w2v).leaky_relu(0.01);
            let out = h2.matmul(&w3v);
            let diff = out.sub_var(&tv);
            let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
            let _ = loss.backward_unchecked();
            let g1 = must!(w1v.grad_ref());
            let g2 = must!(w2v.grad_ref());
            let g3 = must!(w3v.grad_ref());
            sgd.step(&mut [&mut w1, &mut w2, &mut w3], &[&*g1, &*g2, &*g3]);
        }

        let steps = if batch <= 1 { 20 } else if batch <= 32 { 10 } else { 5 };
        let (mut t_tape, mut t_fwd, mut t_bwd, mut t_grad, mut t_sgd) = (0u128, 0u128, 0u128, 0u128, 0u128);

        for _ in 0..steps {
            gpu_sync();
            let t0 = Instant::now();
            let tape = Tape::new();
            let xv = must!(tape.variable(x.clone()));
            let w1v = must!(tape.variable(w1.clone()));
            let w2v = must!(tape.variable(w2.clone()));
            let w3v = must!(tape.variable(w3.clone()));
            let tv = must!(tape.variable(target.clone()));
            gpu_sync(); let t1 = Instant::now(); t_tape += (t1 - t0).as_nanos();

            let h1 = xv.matmul(&w1v).leaky_relu(0.01);
            let h2 = h1.matmul(&w2v).leaky_relu(0.01);
            let out = h2.matmul(&w3v);
            let diff = out.sub_var(&tv);
            let loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
            gpu_sync(); let t2 = Instant::now(); t_fwd += (t2 - t1).as_nanos();

            let _ = loss.backward_unchecked();
            gpu_sync(); let t3 = Instant::now(); t_bwd += (t3 - t2).as_nanos();

            let g1 = must!(w1v.grad_ref());
            let g2 = must!(w2v.grad_ref());
            let g3 = must!(w3v.grad_ref());
            gpu_sync(); let t4 = Instant::now(); t_grad += (t4 - t3).as_nanos();

            sgd.step(&mut [&mut w1, &mut w2, &mut w3], &[&*g1, &*g2, &*g3]);
            gpu_sync(); let t5 = Instant::now(); t_sgd += (t5 - t4).as_nanos();
        }

        let s = steps as f64;
        let to_ms = |ns: u128| ns as f64 / s / 1_000_000.0;
        println!("batch={batch}  steps={steps}");
        println!("  tape_setup: {:.4} ms", to_ms(t_tape));
        println!("  forward:    {:.4} ms", to_ms(t_fwd));
        println!("  backward:   {:.4} ms", to_ms(t_bwd));
        println!("  grad_read:  {:.4} ms", to_ms(t_grad));
        println!("  sgd_step:   {:.4} ms", to_ms(t_sgd));
        println!("  TOTAL:      {:.4} ms", to_ms(t_tape + t_fwd + t_bwd + t_grad + t_sgd));
        println!();
    }
}
