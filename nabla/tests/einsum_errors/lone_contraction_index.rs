// Contraction index appearing only once should fail.
use nabla::prelude::*;

fn main() {
    let a: Tensor<f64> = Tensor::zeros(2, 2);
    let _y: Tensor<f64> = einsum!(y[i] = a[i, k]);
}
