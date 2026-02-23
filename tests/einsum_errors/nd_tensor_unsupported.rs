// N-D tensor (>2 indices) should produce a clear error.
use nabla::prelude::*;

fn main() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let m: Tensor<f64> = Tensor::zeros(2, 2);
    let _r: Tensor<f64> = einsum!(r[i, j] = a[i, j, k] * m[k]);
}
