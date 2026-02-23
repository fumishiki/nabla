// scalar.rs — Scalar trait bridging faer's type system.
//
// faer 0.24 uses `ComplexField` (from faer-traits) as the primary numeric
// constraint. There is no `Entity` trait in this version; `ComplexField` is
// the sole bound required for matrix construction and arithmetic.

use faer_traits::ComplexField;

/// Marker trait for numeric types supported by nabla.
///
/// Implemented for `f32`, `f64`, `c32`, and `c64` — the four types that faer
/// provides native SIMD kernels for.
pub trait Scalar: ComplexField + Copy + Send + Sync + 'static {}

macro_rules! impl_scalar {
    ($($ty:ty),* $(,)?) => {
        $(impl Scalar for $ty {})*
    };
}

impl_scalar!(f32, f64, faer::c32, faer::c64);
