// Duplicate index on LHS should fail.
use nabla::prelude::*;

fn main() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let _c: Tensor<f64> = einsum!(c[i, i] = a[i, i]);
}
