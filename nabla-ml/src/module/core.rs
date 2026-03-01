//! Neural network module trait (PyTorch `nn.Module` equivalent).
//!
//! Provides a common interface for neural network layers with trainable
//! parameters, enabling generic training loops and model composition.

use std::rc::Rc;
use std::{
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::Path,
};

use nabla_core::backend::{Backend, DefaultBackend};
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use crate::autograd::{Tape, Variable};

const NBLA_MAGIC: &[u8; 4] = b"NBLA";

/// Error returned when loading a state dictionary fails.
#[derive(Debug)]
pub enum StateError {
    /// A key in the provided dictionary does not match any parameter.
    MissingKey(String),
    /// Shape mismatch between source and destination tensors.
    ShapeMismatch {
        /// Parameter name.
        key: String,
        /// Expected `(rows, cols)`.
        expected: (usize, usize),
        /// Actual `(rows, cols)` from the source tensor.
        got: (usize, usize),
    },
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKey(k) => write!(f, "unknown parameter key: {k}"),
            Self::ShapeMismatch { key, expected, got } => {
                write!(
                    f,
                    "shape mismatch for '{key}': expected ({}, {}), got ({}, {})",
                    expected.0, expected.1, got.0, got.1
                )
            }
        }
    }
}

impl std::error::Error for StateError {}

impl From<StateError> for nabla_core::error::Error {
    fn from(e: StateError) -> Self {
        match e {
            StateError::ShapeMismatch { expected, got, .. } => Self::mismatch(expected, got),
            StateError::MissingKey(k) => Self::invalid(format!("missing parameter key: {k}")),
        }
    }
}

/// Serialize named tensors to a binary file in nabla format.
pub fn save_tensors<T: Scalar, B: Backend>(
    tensors: &[(&str, &Tensor<T, B>)],
    path: &Path,
) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(NBLA_MAGIC)?;
    write_u32(&mut file, tensors.len() as u32)?;
    for (name, t) in tensors {
        let name_bytes = name.as_bytes();
        write_u32(&mut file, name_bytes.len() as u32)?;
        file.write_all(name_bytes)?;
        let (m, n) = t.shape();
        write_u32(&mut file, m as u32)?;
        write_u32(&mut file, n as u32)?;
        for r in 0..m {
            for c in 0..n {
                write_f64(&mut file, t.get(r, c).to_f64())?;
            }
        }
    }
    Ok(())
}

/// Deserialize named tensors from a binary nabla file.
pub fn load_tensors<T: Scalar, B: Backend>(
    path: &Path,
) -> io::Result<Vec<(String, Tensor<T, B>)>> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != NBLA_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a nabla tensor file",
        ));
    }
    let count = read_u32(&mut file)? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u32(&mut file)? as usize;
        let mut name_buf = vec![0u8; name_len];
        file.read_exact(&mut name_buf)?;
        let name =
            String::from_utf8(name_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let nrows = read_u32(&mut file)? as usize;
        let ncols = read_u32(&mut file)? as usize;
        let total = nrows * ncols;
        let mut data = vec![0.0f64; total];
        for v in &mut data {
            *v = read_f64(&mut file)?;
        }
        let t = Tensor::from_fn(nrows, ncols, |r, c| T::from_f64(data[r * ncols + c]));
        result.push((name, t));
    }
    Ok(result)
}

#[inline]
fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> io::Result<()> {
    writer.write_all(bytes)
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> io::Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}

#[inline]
fn read_bytes<R: Read, const N: usize>(reader: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(reader)?))
}

fn read_f64<R: Read>(reader: &mut R) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_bytes(reader)?))
}

/// Result of a training forward pass with tracked parameter variables.
pub struct ForwardResult<T: Scalar, B: Backend> {
    /// Output of the forward pass.
    pub output: Variable<T, B>,
    /// Tracked parameter Variables (gradients accessible after backward).
    pub param_vars: Vec<Variable<T, B>>,
}

/// Neural network module trait.
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
    /// a `Variable`. Override for proper gradient flow through layer operations.
    fn forward_var(&self, x: &Variable<T, B>, tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
        let out = self.forward(x.data());
        tape.variable(out)
    }

    /// Forward pass with autograd tracking that returns tracked parameter Variables.
    ///
    /// Unlike [`Module::forward_var`], the returned [`ForwardResult`] exposes leaf
    /// parameter Variables whose gradients remain accessible after backward.
    /// Default delegates to `forward_var` with an empty `param_vars` vec.
    fn forward_var_tracked(
        &self,
        x: &Variable<T, B>,
        tape: &Rc<Tape<T, B>>,
    ) -> Result<ForwardResult<T, B>> {
        Ok(ForwardResult {
            output: self.forward_var(x, tape)?,
            param_vars: vec![],
        })
    }

    /// Recommended training forward pass (alias for [`Module::forward_var_tracked`]).
    ///
    /// Tracks autograd and exposes parameter Variables for optimizer updates.
    fn train_forward(&self, x: &Variable<T, B>, tape: &Rc<Tape<T, B>>) -> Result<ForwardResult<T, B>> {
        self.forward_var_tracked(x, tape)
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
    ///
    /// Default returns empty vec for parameterless layers.
    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        vec![]
    }

    /// Collect all trainable parameters with human-readable names.
    ///
    /// Names follow the "layer.weight" / "layer.bias" convention.
    /// Default returns empty vec for parameterless layers.
    fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)> {
        vec![]
    }

    /// Mutable parameter access for in-place optimizer updates.
    ///
    /// Default returns empty vec for parameterless layers.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        vec![]
    }

    /// Return child sub-modules (empty by default for leaf modules).
    fn children(&self) -> Vec<&dyn Module<T, B>> {
        vec![]
    }

    /// Return named child sub-modules (empty by default for leaf modules).
    fn named_children(&self) -> Vec<(String, &dyn Module<T, B>)> {
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
    fn load_state_dict(&mut self, dict: &[(&str, &Tensor<T, B>)]) -> std::result::Result<(), StateError> {
        let mut params = self.named_parameters_mut();
        for (name, src) in dict {
            let dst = params
                .iter_mut()
                .find(|(n, _)| n == name)
                .ok_or_else(|| StateError::MissingKey((*name).to_owned()))?;
            let (sr, sc) = src.shape();
            let (dr, dc) = dst.1.shape();
            if sr != dr || sc != dc {
                return Err(StateError::ShapeMismatch {
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
    ///
    /// Default returns empty vec for parameterless layers.
    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)> {
        vec![]
    }

    /// Apply a function to every mutable parameter in-place.
    fn apply(&mut self, f: &dyn Fn(&mut Tensor<T, B>)) {
        for p in &mut self.parameters_mut() {
            f(p);
        }
    }
}


/// Fully connected linear layer `y = x @ W^T + b`.
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

impl_layer! {
    Linear { weight; bias }
    forward(x) {
        let wt = weight.tl_t();
        let out = x.tl_matmul(&wt);
        match bias { Some(b) => out.tl_add(b), None => out }
    }
}


/// Sequential container that chains modules in order.
pub struct Sequential<T: Scalar, B: Backend> {
    layers: Vec<Box<dyn Module<T, B>>>,
    /// Precomputed parameter names: "0.weight", "0.bias", "1.weight", etc.
    param_names: Vec<String>,
    training: bool,
}

impl<T: Scalar, B: Backend> Default for Sequential<T, B> {
    fn default() -> Self { Self::new() }
}

impl<T: Scalar, B: Backend> Sequential<T, B> {
    /// Create a new empty `Sequential`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            param_names: Vec::new(),
            training: true,
        }
    }

    /// Create a `Sequential` from a vec of boxed modules.
    #[must_use]
    pub fn from_layers(layers: Vec<Box<dyn Module<T, B>>>) -> Self {
        let param_names = Self::build_param_names(&layers);
        Self {
            layers,
            param_names,
            training: true,
        }
    }

    /// Append a module (auto-boxed) and return self for owned builder chains.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, layer: impl Module<T, B> + 'static) -> Self {
        self.with(Box::new(layer))
    }

    /// Append a module and return self for chaining.
    pub fn push(&mut self, layer: Box<dyn Module<T, B>>) -> &mut Self {
        self.layers.push(layer);
        self.param_names = Self::build_param_names(&self.layers);
        self
    }

    /// Append a module, consuming and returning self for owned builder chains.
    #[must_use]
    pub fn with(mut self, layer: Box<dyn Module<T, B>>) -> Self {
        self.push(layer);
        self
    }

    /// Number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the container is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Build prefixed parameter names for all layers: "0.weight", "1.bias", etc.
    fn build_param_names(layers: &[Box<dyn Module<T, B>>]) -> Vec<String> {
        let mut names = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            for (name, _) in layer.named_parameters() {
                names.push(format!("{i}.{name}"));
            }
        }
        names
    }
}

impl<T: Scalar, B: Backend> Module<T, B> for Sequential<T, B> {
    fn forward(&self, x: &Tensor<T, B>) -> Tensor<T, B> {
        self.layers
            .iter()
            .fold(x.clone(), |acc, layer| layer.forward(&acc))
    }

    fn forward_var(&self, x: &Variable<T, B>, tape: &Rc<Tape<T, B>>) -> Result<Variable<T, B>> {
        Ok(self.forward_var_tracked(x, tape)?.output)
    }

    fn forward_var_tracked(
        &self,
        x: &Variable<T, B>,
        tape: &Rc<Tape<T, B>>,
    ) -> Result<ForwardResult<T, B>> {
        let mut current = match self.layers.first() {
            None => ForwardResult {
                output: tape.variable(x.data().clone())?,
                param_vars: vec![],
            },
            Some(first) => first.forward_var_tracked(x, tape)?,
        };
        let mut all_params = current.param_vars;
        for layer in self.layers.iter().skip(1) {
            let result = layer.forward_var_tracked(&current.output, tape)?;
            current.output = result.output;
            all_params.extend(result.param_vars);
        }
        Ok(ForwardResult {
            output: current.output,
            param_vars: all_params,
        })
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor<T, B>)> {
        let params: Vec<&Tensor<T, B>> = self.parameters();
        self.param_names
            .iter()
            .zip(params)
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<T, B>)> {
        let mut result = Vec::new();
        let mut name_idx = 0;
        for layer in &mut self.layers {
            for (_, tensor) in layer.named_parameters_mut() {
                result.push((self.param_names[name_idx].as_str(), tensor));
                name_idx += 1;
            }
        }
        result
    }

    fn children(&self) -> Vec<&dyn Module<T, B>> {
        self.layers.iter().map(AsRef::as_ref).collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T, B>)> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| (i.to_string(), l.as_ref()))
            .collect()
    }
}
