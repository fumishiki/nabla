// tensor/nn.rs — Neural network operations: activations, normalization, loss,
//                 convolution, pooling, attention, dropout, interpolation.

mod activations;
mod attention;
mod conv;
mod losses;
mod norm;
mod pooling;

// Re-export config structs for backward compatibility.
pub use conv::{Conv1dConfig, Conv2dConfig, Conv3dConfig};
