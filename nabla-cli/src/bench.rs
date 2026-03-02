//! `nabla bench` — matmul and MLP training-step benchmarks.
//!
//! Usage: nabla bench [--workload matmul|mlp|all] [--batch 128,512]
//!                    [--sizes 1024,4096] [--iters 100] [--warmup 10]
//!                    [--backend cuda|hip|wgpu|cpu] [--json]

use std::error::Error;
use std::time::Instant;

use nabla_core::backend::DefaultBackend;
use nabla_core::tensor::Tensor;
use nabla_train::prelude::{Optimizer, Sgd};

// Re-export the standard Tape/Variable via nabla crate (not the prelude to avoid Result alias clash).
use nabla::autograd::Tape;

/// Return the name of the compiled-in default backend.
fn compiled_backend() -> &'static str {
    #[cfg(feature = "cuda")] { return "cuda"; }
    #[cfg(feature = "hip")]  { return "hip"; }
    #[cfg(feature = "wgpu")] { return "wgpu"; }
    "cpu"
}

pub fn run(args: &[String]) -> std::result::Result<(), Box<dyn Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: nabla bench [OPTIONS]\n\n\
            Options:\n  \
            --workload matmul|mlp|all  (default: all)\n  \
            --batch    <n,n,...>        MLP batch sizes (default: 128,512)\n  \
            --sizes    <n,n,...>        Matmul matrix sizes (default: 1024,4096)\n  \
            --iters    <n>              Iterations per measurement (default: 100)\n  \
            --warmup   <n>              Warmup iterations (default: 10)\n  \
            --backend  cuda|hip|wgpu|cpu  Select backend (must match compiled features)\n  \
            --json                     Emit JSON output"
        );
        return Ok(());
    }

    // Validate --backend if provided.
    if let Some(requested) = flag_str(args, "--backend") {
        let available = compiled_backend();
        if requested != available {
            return Err(format!(
                "backend `{requested}` is not compiled in (this binary has `{available}`); \
                 rebuild with --features {requested}"
            ).into());
        }
    }

    let workload = flag_str(args, "--workload").unwrap_or("all");
    let batches  = parse_usize_list(flag_str(args, "--batch").unwrap_or("128,512"));
    let sizes    = parse_usize_list(flag_str(args, "--sizes").unwrap_or("1024,4096"));
    let iters    = flag_usize(args, "--iters").unwrap_or(100);
    let warmup   = flag_usize(args, "--warmup").unwrap_or(10);
    let json     = args.iter().any(|a| a == "--json");

    let backend = compiled_backend().to_uppercase();
    if !json {
        println!("nabla bench  [{backend}]");
        println!("{:<26} {}", "Workload", "Time (µs)");
        println!("{}", "─".repeat(36));
    }

    let mut records: Vec<BenchRecord> = Vec::new();

    if workload == "matmul" || workload == "all" {
        for &n in &sizes {
            records.push(bench_matmul(n, warmup, iters));
        }
    }
    if workload == "mlp" || workload == "all" {
        for &b in &batches {
            records.push(bench_mlp(b, warmup, iters));
        }
    }

    if records.is_empty() {
        return Err(format!("unknown workload `{workload}` — use matmul, mlp, or all").into());
    }

    if json { print_json(&records); } else { print_rows(&records); }
    Ok(())
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

struct BenchRecord {
    workload: &'static str,
    label:    String,
    eager_us: f64,
}

// ---------------------------------------------------------------------------
// Matmul benchmark
// ---------------------------------------------------------------------------

fn bench_matmul(n: usize, warmup: usize, iters: usize) -> BenchRecord {
    let a: Tensor<f32, DefaultBackend> = Tensor::randn(n, n, 42);
    let b: Tensor<f32, DefaultBackend> = Tensor::randn(n, n, 43);

    for _ in 0..warmup { let _ = &a * &b; }
    gpu_sync();

    let t = Instant::now();
    for _ in 0..iters { let _ = &a * &b; }
    gpu_sync();
    let us = t.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    BenchRecord { workload: "matmul", label: format!("{n}x{n} f32"), eager_us: us }
}

// ---------------------------------------------------------------------------
// MLP benchmark — MLP 784→256→128→10, leaky_relu(0.01), MSE, SGD
// ---------------------------------------------------------------------------

fn bench_mlp(batch: usize, warmup: usize, iters: usize) -> BenchRecord {
    let mut w1: Tensor<f32, DefaultBackend> = kaiming(784, 256);
    let mut w2: Tensor<f32, DefaultBackend> = kaiming(256, 128);
    let mut w3: Tensor<f32, DefaultBackend> = kaiming(128, 10);
    let params = vec![&w1, &w2, &w3];
    let mut sgd: Sgd<f32, DefaultBackend> = Sgd::from_params(0.001, &params);
    drop(params);

    let x      = Tensor::<f32, DefaultBackend>::randn(batch, 784, 1);
    let target = Tensor::<f32, DefaultBackend>::randn(batch, 10, 2).map(|v| v * 0.1);

    for _ in 0..warmup { mlp_step(&mut w1, &mut w2, &mut w3, &mut sgd, &x, &target); }
    gpu_sync();

    let t = Instant::now();
    for _ in 0..iters { mlp_step(&mut w1, &mut w2, &mut w3, &mut sgd, &x, &target); }
    gpu_sync();
    let us = t.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    BenchRecord { workload: "mlp", label: format!("batch={batch}"), eager_us: us }
}

fn mlp_step(
    w1: &mut Tensor<f32, DefaultBackend>,
    w2: &mut Tensor<f32, DefaultBackend>,
    w3: &mut Tensor<f32, DefaultBackend>,
    sgd: &mut Sgd<f32, DefaultBackend>,
    x: &Tensor<f32, DefaultBackend>,
    target: &Tensor<f32, DefaultBackend>,
) {
    let tape = Tape::new();
    let xv  = tape.variable(x.clone()).expect("tape var");
    let w1v = tape.variable(w1.clone()).expect("tape var");
    let w2v = tape.variable(w2.clone()).expect("tape var");
    let w3v = tape.variable(w3.clone()).expect("tape var");
    let tv  = tape.variable(target.clone()).expect("tape var");

    let h1  = xv.matmul(&w1v).leaky_relu(0.01);
    let h2  = h1.matmul(&w2v).leaky_relu(0.01);
    let out = h2.matmul(&w3v);
    let d   = out.sub_var(&tv);
    let loss = d.emul(&d).sum_axis(1).sum_axis(0);
    let _ = loss.backward_unchecked();

    let g1 = w1v.grad_ref().expect("grad");
    let g2 = w2v.grad_ref().expect("grad");
    let g3 = w3v.grad_ref().expect("grad");
    sgd.step(&mut [w1, w2, w3], &[&*g1, &*g2, &*g3]);
}

fn kaiming(rows: usize, cols: usize) -> Tensor<f32, DefaultBackend> {
    Tensor::randn(rows, cols, 42).map(|x| x * (2.0_f32 / rows as f32).sqrt())
}

// ---------------------------------------------------------------------------
// GPU sync
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[inline] fn gpu_sync() { nabla_core::cuda_synchronize(); }

#[cfg(not(feature = "cuda"))]
#[inline] fn gpu_sync() {}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_rows(records: &[BenchRecord]) {
    for r in records {
        println!(
            "{:<26} {:>9.1} µs",
            format!("{} {}", r.workload, r.label),
            r.eager_us
        );
    }
}

fn print_json(records: &[BenchRecord]) {
    print!("[");
    for (i, r) in records.iter().enumerate() {
        if i > 0 { print!(","); }
        print!(
            "{{\"workload\":{:?},\"label\":{:?},\"eager_us\":{:.2}}}",
            r.workload, r.label, r.eager_us
        );
    }
    println!("]");
}

// ---------------------------------------------------------------------------
// Arg helpers
// ---------------------------------------------------------------------------

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())
}

fn flag_usize(args: &[String], flag: &str) -> Option<usize> {
    flag_str(args, flag)?.parse().ok()
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}
