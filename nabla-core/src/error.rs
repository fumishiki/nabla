use core::{fmt, result};

/// Convenience `Result` alias for nabla operations.
pub type Result<T> = result::Result<T, Error>;

/// Errors that can occur in nabla operations.
#[derive(Debug)]
pub enum Error {
    /// Matrix shape was not compatible with the operation.
    ShapeMismatch {
        /// Expected shape `(rows, cols)`.
        expected: (usize, usize),
        /// Actual shape `(rows, cols)`.
        got: (usize, usize),
    },
    /// A dimension value was invalid for the given context.
    InvalidDimension(String),
    /// GPU kernel launch or execution failed.
    GpuKernelFailed(String),
    /// Expression evaluation failed (unbound variable, empty context, etc.).
    EvalError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => write!(
                f,
                "shape mismatch: expected ({}, {}), got ({}, {})",
                expected.0, expected.1, got.0, got.1
            ),
            Self::InvalidDimension(msg) => write!(f, "invalid dimension: {msg}"),
            Self::GpuKernelFailed(msg) => write!(f, "GPU kernel failed: {msg}"),
            Self::EvalError(msg) => write!(f, "eval error: {msg}"),
        }
    }
}

impl Error {
    /// Create a shape mismatch error.
    #[inline]
    pub fn mismatch(expected: (usize, usize), got: (usize, usize)) -> Self {
        Self::ShapeMismatch { expected, got }
    }

    /// Create an invalid dimension error.
    #[inline]
    pub fn invalid<T: core::fmt::Display>(msg: T) -> Self {
        Self::InvalidDimension(msg.to_string())
    }

    /// Create an eval error.
    #[inline]
    pub fn eval<T: core::fmt::Display>(msg: T) -> Self {
        Self::EvalError(msg.to_string())
    }
}

impl std::error::Error for Error {}
