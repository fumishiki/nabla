//! Diagnose which operation breaks CUDA Graph capture.
//! Run: cargo run --release --bin diag_capture --features cuda

#[cfg(feature = "cuda")]
use nabla::prelude::*;
#[cfg(feature = "cuda")]
use nabla_bench::{rand_tensor, kaiming, must};

#[cfg(feature = "cuda")]
fn gpu_sync() { nabla::cuda_synchronize(); }

#[cfg(feature = "cuda")]
macro_rules! test_capture {
    ($name:expr, $body:expr) => {
        match nabla::cuda_graph_capture(|| { $body }) {
            Ok(g) => { eprintln!("PASS: {}", $name); Some(g) }
            Err(e) => { eprintln!("FAIL: {} - {e}", $name); None }
        }
    };
}

#[cfg(feature = "cuda")]
fn main() {
    use nabla::cuda_graph_capture;

    let batch = 32usize;
    let x = rand_tensor(batch, 784);
    let w1 = kaiming(784, 256);
    let w2 = kaiming(256, 128);
    let w3 = kaiming(128, 10);
    let target = rand_tensor(batch, 10).map(|v| v * 0.1);

    // Warmup: compile all kernels
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
    }
    gpu_sync();
    eprintln!("warmup done");

    // Individual op tests
    test_capture!("matmul", { let _ = &x * &w1; });
    gpu_sync();
    test_capture!("clone", { let _ = x.clone(); });
    gpu_sync();
    test_capture!("zeros", { let _: Tensor<f32> = Tensor::zeros(batch, 256); });
    gpu_sync();
    test_capture!("fill", { let _: Tensor<f32> = Tensor::fill(batch, 256, 1.0f32); });
    gpu_sync();

    let h = &x * &w1;
    test_capture!("leaky_relu fwd", { let _ = h.leaky_relu(0.01); });
    gpu_sync();
    test_capture!("neg", { let _ = -&x; });
    gpu_sync();

    let a = rand_tensor(batch, 10);
    let b = rand_tensor(batch, 10);
    test_capture!("emul", { let _ = a.emul(&b); });
    gpu_sync();
    test_capture!("sum_axis", { let _ = a.sum_axis(1); });
    gpu_sync();

    let small = Tensor::<f32>::fill(1, 10, 1.0);
    test_capture!("expand", { let _ = small.expand(batch, 10); });
    gpu_sync();

    let grad = rand_tensor(batch, 256);
    let input = rand_tensor(batch, 256);
    test_capture!("leaky_relu_bwd", { let _ = grad.leaky_relu_backward(&input, 0.01f32); });
    gpu_sync();

    let g11 = rand_tensor(784, 10);
    test_capture!("matmul_tn", { let _ = w1.matmul_tn(&g11); });
    gpu_sync();

    let g12 = rand_tensor(batch, 10);
    let w_nt = rand_tensor(256, 10);
    test_capture!("matmul_nt", { let _ = g12.matmul_nt(&w_nt); });
    gpu_sync();

    let mut param = w1.clone();
    let grad_w = rand_tensor(784, 256);
    test_capture!("axpy_inplace", { param.axpy_inplace(-0.001f32, &grad_w); });
    gpu_sync();

    // Composite tests
    test_capture!("full forward", {
        let h1 = (&x * &w1).leaky_relu(0.01);
        let h2 = (&h1 * &w2).leaky_relu(0.01);
        let _out = &h2 * &w3;
    });
    gpu_sync();

    test_capture!("fwd+tape", {
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
        let _loss = diff.emul(&diff).sum_axis(1).sum_axis(0);
    });
    gpu_sync();

    test_capture!("fwd+bwd", {
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
    });
    gpu_sync();

    // Replay tests
    match cuda_graph_capture(|| { let _ = &x * &w1; }) {
        Ok(g) => {
            eprintln!("PASS: capture matmul");
            match g.launch() {
                Ok(_) => { gpu_sync(); eprintln!("PASS: replay matmul"); }
                Err(e) => eprintln!("FAIL: replay matmul - {e}"),
            }
        }
        Err(e) => eprintln!("FAIL: capture matmul for replay - {e}"),
    }
    gpu_sync();

    match cuda_graph_capture(|| {
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
    }) {
        Ok(g) => {
            eprintln!("PASS: capture fwd+bwd");
            for i in 0..5 {
                match g.launch() {
                    Ok(_) => { gpu_sync(); eprintln!("PASS: replay fwd+bwd #{i}"); }
                    Err(e) => { eprintln!("FAIL: replay fwd+bwd #{i} - {e}"); break; }
                }
            }
        }
        Err(e) => eprintln!("FAIL: capture fwd+bwd for replay - {e}"),
    }
    gpu_sync();
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("Requires --features cuda"); }
