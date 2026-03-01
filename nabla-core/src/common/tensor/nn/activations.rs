// tensor/nn/activations.rs — ML activation functions.

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::{two, Tensor};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ---- ML activation functions ----

    /// Element-wise ReLU: `max(x, 0)`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let two = two::<T>();
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            (x + x.math_abs()) / two
        })
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            T::one() / (T::one() + (T::zero() - x).math_exp())
        })
    }

    /// Element-wise GELU (tanh approximation):
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    #[must_use]
    pub fn gelu(&self) -> Self {
        let half = T::from_f64(0.5);
        let k = T::from_f64(0.797_884_560_8); // sqrt(2/pi)
        let c = T::from_f64(0.044_715);
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, col| {
            let x = self.get(r, col);
            let inner = k * (x + c * x * x * x);
            half * x * (T::one() + inner.math_tanh())
        })
    }

    /// Element-wise SiLU (Swish): `x * sigmoid(x)`.
    #[must_use]
    pub fn silu(&self) -> Self {
        Self::from_storage(B::silu(&self.storage))
    }

    /// Element-wise Mish: `x * tanh(softplus(x))` where softplus(x) = ln(1 + exp(x)).
    #[must_use]
    pub fn mish(&self) -> Self {
        Self::from_storage(B::mish(&self.storage))
    }

    /// Element-wise Leaky ReLU: `max(alpha * x, x)`.
    #[must_use]
    pub fn leaky_relu(&self, alpha: T) -> Self {
        Self::from_storage(B::leaky_relu(&self.storage, alpha))
    }

    /// Element-wise ELU: `x if x > 0, alpha * (exp(x) - 1) otherwise`.
    #[must_use]
    pub fn elu(&self, alpha: T) -> Self {
        Self::from_storage(B::elu(&self.storage, alpha))
    }

    /// Element-wise HardSwish: `x * relu6(x + 3) / 6`.
    #[must_use]
    pub fn hardswish(&self) -> Self {
        Self::from_storage(B::hardswish(&self.storage))
    }
}
