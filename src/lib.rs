#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]

//! nabla — Julia math notation in Rust.
//!
//! Import [`prelude`] for the most common types and traits.

/// Error types for nabla operations.
pub mod error;

/// Scalar numeric types supported by nabla.
pub mod scalar;

/// Compute backends (CPU + future GPU stubs).
pub mod backend;

/// 2-D dense tensor type with operator overloads.
pub mod tensor;

/// Dense and sparse linear algebra helpers.
pub mod linalg;

/// Sparse matrix support.
pub mod sparse;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::backend::{Backend, Cpu, DefaultBackend};
    pub use crate::error::{Error, Result};
    pub use crate::scalar::Scalar;
    pub use crate::tensor::Tensor;
    pub use crate::sparse::*;
    pub use nabla_macros::mat;
}
