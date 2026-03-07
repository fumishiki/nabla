#![allow(clippy::missing_errors_doc)]

mod core;
mod ops;
mod tensor_like;

pub use core::{Tape, Variable};
pub use ops::{GradPrep, grad, gradient, gradient_prep};
pub use ops::{clip_grad_norm, scale_grad, zero_grad};
pub use tensor_like::{TensorLike, TensorLikeExt, TensorLikeMatmulBias};
