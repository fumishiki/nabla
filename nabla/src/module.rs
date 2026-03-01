//! Neural network module trait (PyTorch `nn.Module` equivalent).
//!
//! Provides a common interface for neural network layers with trainable
//! parameters, enabling generic training loops and model composition.

use std::rc::Rc;

use nabla_core::backend::{Backend, DefaultBackend};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use crate::autograd::{Tape, Variable};

/// Trait for neural network modules with trainable parameters.
///
/// Implement this on layer structs (e.g. `Linear`, `Conv2d`) to enable
/// generic forward passes, parameter collection, and optimizer integration.
pub trait Module<T: Scalar, B: Backend> {
    /// Run the forward pass on input `x`.
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B>;

    /// Forward pass with additional inputs beyond the primary tensor.
    ///
    /// Default delegates to [`Module::forward`], ignoring `extra`.
    /// Override for modules that require multiple inputs (e.g. cross-attention).
    fn forward_with(&self, x: &Tensor<T, B>, _extra: &[&Tensor<T, B>]) -> Tensor<T, B> {
        self.forward(x)
    }

    /// Run forward pass with autograd tracking.
    ///
    /// Default implementation wraps parameters as `Variable`s on the given tape,
    /// runs `forward` on the underlying tensor data, and returns the result as
    /// a `Variable`.  Override for proper gradient flow through layer operations.
    fn forward_var(
        &self,
        x: &Variable<T, B>,
        tape: &Rc<Tape<T, B>>,
    ) -> Variable<T, B> {
        // Fallback: run forward on raw data, wrap output as a leaf.
        // Subclasses should override for proper gradient tracking.
        let out = self.forward(x.data());
        tape.variable(out)
    }

    /// Whether the module is in training mode.
    ///
    /// Default returns `true`. Implementors should store a `training: bool` field.
    fn training(&self) -> bool {
        true
    }

    /// Set training/evaluation mode.
    fn set_training(&mut self, training: bool);

    /// Switch to training mode (shorthand for `set_training(true)`).
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Switch to evaluation mode (shorthand for `set_training(false)`).
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Collect all trainable parameters (immutable references).
    fn parameters(&self) -> Vec<&Tensor<T, B>>;

    /// Collect all trainable parameters with human-readable names.
    ///
    /// Names follow the `"layer.weight"` / `"layer.bias"` convention.
    fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)>;

    /// Mutable parameter access for in-place optimizer updates.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>>;

    /// Return child sub-modules (empty by default for leaf modules).
    fn children(&self) -> Vec<&dyn Module<T, B>> {
        vec![]
    }

    /// Return named child sub-modules (empty by default for leaf modules).
    fn named_children(&self) -> Vec<(&str, &dyn Module<T, B>)> {
        vec![]
    }

    /// Return non-parameter persistent state tensors (e.g. running stats).
    fn buffers(&self) -> Vec<&Tensor<T, B>> {
        vec![]
    }

    /// Return a snapshot of all named parameters (state dictionary).
    ///
    /// Default implementation delegates to [`Module::named_parameters`].
    fn state_dict(&self) -> Vec<(&str, &Tensor<T, B>)> {
        self.named_parameters()
    }

    /// Load parameters from a state dictionary, matching by name.
    ///
    /// Default implementation matches names from [`Module::named_parameters`]
    /// and copies data from the provided dictionary entries.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter name in the dictionary does not match
    /// any known parameter, or if shapes are incompatible.
    fn load_state_dict(
        &mut self,
        dict: &[(&str, &Tensor<T, B>)],
    ) -> Result<(), crate::io::StateError> {
        let mut params = self.named_parameters_mut();
        for (name, src) in dict {
            let dst = params
                .iter_mut()
                .find(|(n, _)| n == name)
                .ok_or_else(|| crate::io::StateError::MissingKey((*name).to_owned()))?;
            let (sr, sc) = src.shape();
            let (dr, dc) = dst.1.shape();
            if sr != dr || sc != dc {
                return Err(crate::io::StateError::ShapeMismatch {
                    key: (*name).to_owned(),
                    expected: (dr, dc),
                    got: (sr, sc),
                });
            }
            *dst.1 = Tensor::from_fn(sr, sc, |r, c| src.get(r, c));
        }
        Ok(())
    }

    /// Mutable named parameter access for [`Module::load_state_dict`].
    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)>;

    /// Apply a function to every mutable parameter in-place.
    fn apply(&mut self, f: &dyn Fn(&mut Tensor<T, B>)) {
        for p in &mut self.parameters_mut() {
            f(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Linear layer
// ---------------------------------------------------------------------------

/// Fully-connected linear layer: `y = x @ weight^T + bias`.
///
/// Weight shape: `(out_features, in_features)`.
/// Bias shape (when present): `(1, out_features)`.
pub struct Linear<T: Scalar, B: Backend = DefaultBackend> {
    /// Weight matrix `(out_features, in_features)`.
    pub weight: Tensor<T, B>,
    /// Optional bias vector `(1, out_features)`.
    pub bias: Option<Tensor<T, B>>,
    training: bool,
}

impl<T: Scalar> Linear<T> {
    /// Create a new `Linear` layer with Xavier uniform weight init and zero bias.
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = crate::nn::xavier_uniform::<T>(in_features, out_features);
        let bias = Tensor::zeros(1, out_features);
        Self {
            weight,
            bias: Some(bias),
            training: true,
        }
    }

    /// Create a new `Linear` layer without bias, Xavier uniform weight init.
    #[must_use]
    pub fn without_bias(in_features: usize, out_features: usize) -> Self {
        let weight = crate::nn::xavier_uniform::<T>(in_features, out_features);
        Self {
            weight,
            bias: None,
            training: true,
        }
    }
}

impl<T: Scalar, B: Backend> Module<T, B> for Linear<T, B> {
    /// Compute `x @ weight^T + bias`.
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B> {
        let wt = self.weight.t();
        let out = x * &wt;
        match &self.bias {
            Some(b) => &out + b,
            None => out,
        }
    }

    /// Autograd-tracked forward: `x @ weight^T + bias` using Variable ops.
    fn forward_var(
        &self,
        x: &Variable<T, B>,
        tape: &Rc<Tape<T, B>>,
    ) -> Variable<T, B> {
        // Wrap transposed weight as a tracked variable.
        // Variable::transpose() is now available, but using pre-transposed data
        // avoids an extra op on the tape.
        let wt_data = self.weight.t();
        let wt_var = tape.variable(wt_data);
        // x @ weight^T via matmul
        let out = x.matmul(&wt_var);
        // Add bias if present.
        match &self.bias {
            Some(b) => {
                let bias_var = tape.variable(b.clone());
                out.add_var(&bias_var)
            }
            None => out,
        }
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)> {
        let mut params = vec![("weight", &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias", b));
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        let mut params: Vec<&mut Tensor<T, B>> = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)> {
        let mut params: Vec<(&str, &mut Tensor<T, B>)> = vec![("weight", &mut self.weight)];
        if let Some(ref mut b) = self.bias {
            params.push(("bias", b));
        }
        params
    }
}
