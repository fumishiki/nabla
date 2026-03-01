//! Benchmark: nabla eager vs CUDA Graph — MLP training step.
//! Run: cargo run --release --bin profile_train_graph --features cuda

use nabla::prelude::*;
use nabla_bench::{rand_tensor, kaiming, bench_ms, must};
#[cfg(feature = "cuda")]
use nabla_bench::gpu_sync;
use nabla_train::prelude::{Optimizer, Sgd};

type B = DefaultBackend;

fn train_step(
    x: &Tensor<f32, B>, target: &Tensor<f32, B>,
    w1: &mut Tensor<f32, B>, w2: &mut Tensor<f32, B>, w3: &mut Tensor<f32, B>,
    sgd: &mut Sgd<f32, B>,
) {
    let tape = Tape::new();
    let xv = must!(tape.variable(x.clone()));
    let w1v = must!(tape.variable(w1.clone()));
    let w2v = must!(tape.variable(w2.clone()));
    let w3v = must!(tape.variable(w3.clone()));
    let tv = must!(tape.variable(target.clone()));
    let h1 = xv.matmul(&w1v).leaky_relu(0.01);
    let h2 = h1.matmul(&w2v).leaky_relu(0.01);
    let out = h2.matmul(&w3v);
    let loss = out.mse_sum_loss(&tv);
    let _ = loss.backward_unchecked();
    let g1 = must!(w1v.grad_ref());
    let g2 = must!(w2v.grad_ref());
    let g3 = must!(w3v.grad_ref());
    sgd.step(&mut vec![w1, w2, w3], &vec![&*g1, &*g2, &*g3]);
}

const WARMUP: usize = 10;
const ITERS: usize = 100;
const BATCH_SIZES: &[usize] = &[1, 32, 128, 256, 512, 1024];

fn main() {
    println!("nabla MLP benchmark  784->256->128->10, leaky_relu(0.01), MSE sum, SGD lr=0.001, f32");
    println!("warmup={WARMUP}, iters={ITERS}\n");

    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    for &batch in BATCH_SIZES {
        let (mut w1, mut w2, mut w3) = (kaiming(784, 256), kaiming(256, 128), kaiming(128, 10));
        let params = vec![&w1, &w2, &w3];
        let mut sgd: Sgd<f32, B> = Sgd::from_params(0.001, &params);
        drop(params);
        let x = rand_tensor(batch, 784);
        let target = rand_tensor(batch, 10).map(|v| v * 0.1);

        let eager_ms = bench_ms(WARMUP, ITERS, || train_step(&x, &target, &mut w1, &mut w2, &mut w3, &mut sgd));

        #[cfg(feature = "cuda")]
        let graph_ms = {
            use nabla::cuda_graph_capture;
            for _ in 0..WARMUP { train_step(&x, &target, &mut w1, &mut w2, &mut w3, &mut sgd); }
            gpu_sync();
            match cuda_graph_capture(|| train_step(&x, &target, &mut w1, &mut w2, &mut w3, &mut sgd)) {
                Ok(graph) => {
                    gpu_sync();
                    for _ in 0..WARMUP { let _ = graph.launch(); }
                    gpu_sync();
                    let start = std::time::Instant::now();
                    let mut ok = true;
                    for i in 0..ITERS {
                        if let Err(e) = graph.launch() {
                            eprintln!("  batch={batch} replay {i} failed: {e}");
                            ok = false; break;
                        }
                    }
                    gpu_sync();
                    if ok { start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64 } else { f64::NAN }
                }
                Err(e) => { eprintln!("  batch={batch} capture failed: {e}"); f64::NAN }
            }
        };
        #[cfg(not(feature = "cuda"))]
        let graph_ms = f64::NAN;

        println!("--- batch={batch} ---");
        println!("  eager:      {eager_ms:.4} ms/step");
        if graph_ms.is_finite() {
            println!("  cuda_graph: {graph_ms:.4} ms/step  ({:.2}x)", eager_ms / graph_ms);
        } else {
            println!("  cuda_graph: N/A");
        }
        println!();
        results.push((batch, eager_ms, graph_ms));
    }

    println!("{}", "=".repeat(56));
    println!("{:>6}  {:>14}  {:>14}  {:>10}", "batch", "eager (ms)", "graph (ms)", "speedup");
    println!("{}", "-".repeat(56));
    for &(batch, eager, graph) in &results {
        let g = if graph.is_finite() { format!("{graph:.4}") } else { "N/A".to_string() };
        let s = if graph.is_finite() { format!("{:.2}x", eager / graph) } else { "-".to_string() };
        println!("{batch:>6}  {eager:>14.4}  {g:>14}  {s:>10}");
    }
    println!("{}", "=".repeat(56));
}
