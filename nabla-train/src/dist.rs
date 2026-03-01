use std::sync::mpsc::{Receiver, Sender, channel};

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

pub struct CpuAllReduce<T: Scalar, B: Backend> {
    rank: usize,
    tx: Sender<Tensor<T, B>>,
    rx: Receiver<Tensor<T, B>>,
}

impl<T: Scalar, B: Backend> CpuAllReduce<T, B> {
    #[must_use]
    pub fn pair() -> (Self, Self) {
        let (tx0, rx0) = channel();
        let (tx1, rx1) = channel();
        let r0 = Self { rank: 0, tx: tx0, rx: rx1 };
        let r1 = Self { rank: 1, tx: tx1, rx: rx0 };
        (r0, r1)
    }

    #[must_use]
    pub fn rank(&self) -> usize { self.rank }

    pub fn allreduce_mean(&self, t: &Tensor<T, B>) -> Result<Tensor<T, B>, String> {
        self.tx.send(t.clone()).map_err(|_| "send failed".to_owned())?;
        let other = self.rx.recv().map_err(|_| "recv failed".to_owned())?;
        let half = T::from_f64(0.5);
        Ok((&other + t).map(|x| x * half))
    }
}
