/// Convenience alias for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

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
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => write!(
                f,
                "shape mismatch: expected ({}, {}), got ({}, {})",
                expected.0, expected.1, got.0, got.1
            ),
            Self::InvalidDimension(msg) => write!(f, "invalid dimension: {msg}"),
        }
    }
}

impl Error {
    #[inline]
    pub(crate) fn mismatch(expected: (usize, usize), got: (usize, usize)) -> Self {
        Self::ShapeMismatch { expected, got }
    }

    #[inline]
    pub(crate) fn invalid<T: core::fmt::Display>(msg: T) -> Self {
        Self::InvalidDimension(msg.to_string())
    }
}

impl std::error::Error for Error {}
