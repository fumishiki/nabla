
use crate::backend::Backend;
use crate::scalar::Scalar;

use super::{Tensor, two};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    fn axis_len(&self, axis: usize, op: &str) -> usize {
        match axis {
            0 => self.nrows(),
            1 => self.ncols(),
            _ => panic!("nabla: {op} axis must be 0 or 1, got {axis}"),
        }
    }

    fn resolve_axis(axis: isize, op: &str) -> usize {
        let ndim = 2isize;
        let resolved = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        assert!(
            resolved < 2,
            "nabla: {op} axis {axis} out of range for 2-D tensor"
        );
        resolved
    }

    /// Reduce along axis with a custom fold function, seeded from the first element.
    fn reduce_axis<F: Fn(T, T) -> T>(&self, axis: usize, f: F) -> Self {
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                // Start from index 1 to avoid double-counting the first element.
                (1..self.nrows())
                    .map(|r| self.get(r, c))
                    .fold(self.get(0, c), &f)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                (1..self.ncols())
                    .map(|c| self.get(r, c))
                    .fold(self.get(r, 0), &f)
            }),
            _ => panic!("nabla: reduce_axis axis must be 0 or 1, got {axis}"),
        }
    }

    // ---- Scalar reductions ----

    /// Sum of all elements.
    #[must_use]
    #[inline]
    pub fn sum_all(&self) -> T {
        B::sum_all(&self.storage)
    }

    /// Sum of all elements (alias for [`sum_all`](Self::sum_all)).
    #[must_use]
    #[inline]
    pub fn sum(&self) -> T {
        self.sum_all()
    }

    /// Mean of all elements: `sum / count`.
    #[must_use]
    #[inline]
    pub fn mean(&self) -> T {
        let (m, n) = self.shape();
        let count = T::from_f64((m * n) as f64);
        self.sum_all() / count
    }

    /// Product of all elements (alias for [`prod_all`](Self::prod_all)).
    #[must_use]
    #[inline]
    pub fn prod(&self) -> T {
        self.prod_all()
    }

    /// Maximum element (alias for [`max_all`](Self::max_all)).
    #[must_use]
    #[inline]
    pub fn max(&self) -> T {
        self.max_all()
    }

    /// Minimum element (alias for [`min_all`](Self::min_all)).
    #[must_use]
    #[inline]
    pub fn min(&self) -> T {
        self.min_all()
    }

    /// Frobenius norm squared: sum of squares of all elements.
    #[must_use]
    pub fn norm_sq(&self) -> T {
        self.emul(self).sum_all()
    }

    /// Frobenius norm: `sqrt(sum of squares)`. For vectors this is the L2 norm.
    #[must_use]
    pub fn norm(&self) -> T {
        self.norm_sq().math_sqrt()
    }

    /// L1 norm: sum of absolute values.
    #[must_use]
    pub fn l1_norm(&self) -> T {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| self.get(r, c).math_abs()).sum_all()
    }

    /// L-infinity norm: maximum absolute value.
    #[must_use]
    pub fn linf_norm(&self) -> T {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| self.get(r, c).math_abs()).max_all()
    }

    /// Lp norm: `(sum |x_i|^p)^(1/p)`, or `max|x_i|` for p=inf.
    #[must_use]
    pub fn norm_lp(&self, p: T) -> T {
        B::norm_lp(&self.storage, p)
    }

    /// Unified norm: `norm_ord(1)` = L1, `norm_ord(2)` = L2/Frobenius, `norm_ord(inf)` = Linf.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn norm_ord(&self, p: f64) -> T {
        if p == 1.0 {
            self.l1_norm()
        } else if p == 2.0 {
            self.norm()
        } else if p.is_infinite() {
            self.linf_norm()
        } else {
            self.norm_lp(T::from_f64(p))
        }
    }

    /// Inner product (dot product): `sum(self .* other)`. Both must be same shape.
    #[must_use]
    pub fn dot(&self, other: &Self) -> T {
        let (m, n) = self.shape();
        let (p, q) = other.shape();
        assert!(
            m == p && n == q,
            "nabla: dot ({m}x{n}) vs ({p}x{q}) -- shapes must match"
        );
        self.emul(other).sum_all()
    }

    /// Outer product: `u * v^T`. Self must be (n,1) or (1,n), other must be (m,1) or (1,m).
    #[must_use]
    pub fn outer(&self, other: &Self) -> Self {
        let n = self.nrows() * self.ncols();
        let m = other.nrows() * other.ncols();
        Self::from_fn(n, m, |i, j| {
            let a = if self.ncols() == 1 {
                self.get(i, 0)
            } else {
                self.get(0, i)
            };
            let b = if other.ncols() == 1 {
                other.get(j, 0)
            } else {
                other.get(0, j)
            };
            a * b
        })
    }

    /// Kronecker product: `A (x) B`. Returns (m*p, n*q) tensor.
    #[must_use]
    pub fn kron(&self, other: &Self) -> Self {
        let (m, n) = self.shape();
        let (p, q) = other.shape();
        Self::from_fn(m * p, n * q, |i, j| {
            self.get(i / p, j / q) * other.get(i % p, j % q)
        })
    }

    /// Normalize to unit vector (L2 norm = 1). Returns clone if norm is zero.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == T::zero() {
            self.clone()
        } else {
            self * (T::one() / n)
        }
    }

    /// Element with the maximum value (or maximum magnitude for complex types).
    #[must_use]
    #[inline]
    pub fn max_all(&self) -> T {
        B::max_all(&self.storage)
    }

    /// Element with the minimum value (or minimum magnitude for complex types).
    #[must_use]
    #[inline]
    pub fn min_all(&self) -> T {
        B::min_all(&self.storage)
    }

    /// `(row, col)` of the element with the maximum value.
    #[must_use]
    #[inline]
    pub fn argmax(&self) -> (usize, usize) {
        B::argmax_all(&self.storage)
    }

    /// `(row, col)` of the element with the minimum value.
    #[must_use]
    #[inline]
    pub fn argmin(&self) -> (usize, usize) {
        B::argmin_all(&self.storage)
    }

    /// Extract the diagonal as an `nx1` column vector, where `n = min(rows, cols)`.
    #[must_use]
    pub fn diag(&self) -> Self {
        let n = self.nrows().min(self.ncols());
        Self::from_fn(n, 1, |i, _| self.get(i, i))
    }

    /// Sum of diagonal elements.
    #[must_use]
    pub fn trace(&self) -> T {
        let n = self.nrows().min(self.ncols());
        (0..n).map(|i| self.get(i, i)).fold(T::zero(), |a, b| a + b)
    }

    /// Short alias for [`Tensor::trace`].
    #[must_use]
    #[inline]
    pub fn tr(&self) -> T {
        self.trace()
    }

    // ---- Axis reductions ----

    /// Sum along axis. axis=0 -> (1, ncols), axis=1 -> (nrows, 1).
    #[must_use]
    pub fn sum_axis(&self, axis: usize) -> Self {
        match axis {
            0 => {
                let t = self.t();
                let sum_t = Self::from_storage(B::sum_axis1(&t.storage));
                sum_t.t()
            }
            1 => Self::from_storage(B::sum_axis1(&self.storage)),
            _ => panic!("sum_axis: axis {axis} out of bounds"),
        }
    }

    /// Mean along axis.
    #[must_use]
    pub fn mean_axis(&self, axis: usize) -> Self {
        let n = self.axis_len(axis, "mean_axis");
        let sum = self.sum_axis(axis);
        let inv_n = T::from_f64(1.0 / n as f64);
        &sum * inv_n
    }

    /// Sum along axis with signed indexing: `sum_dim(-1)` = last axis.
    #[deprecated(since = "0.1.0", note = "use sum_axis() instead")]
    #[must_use]
    pub fn sum_dim(&self, axis: isize) -> Self {
        self.sum_axis(Self::resolve_axis(axis, "sum_dim"))
    }

    /// Mean along axis with signed indexing: `mean_dim(-1)` = last axis.
    #[deprecated(since = "0.1.0", note = "use mean_axis() instead")]
    #[must_use]
    pub fn mean_dim(&self, axis: isize) -> Self {
        self.mean_axis(Self::resolve_axis(axis, "mean_dim"))
    }

    /// Sum along axis with keepdim semantics (for 2-D, identical to `sum_axis`).
    #[must_use]
    pub fn sum_axis_keepdim(&self, axis: usize) -> Self {
        self.sum_axis(axis)
    }

    /// Mean along axis with keepdim semantics (for 2-D, identical to `mean_axis`).
    #[must_use]
    pub fn mean_axis_keepdim(&self, axis: usize) -> Self {
        self.mean_axis(axis)
    }

    /// Maximum along axis: axis 0 -> 1xn (column-wise), axis 1 -> mx1 (row-wise).
    #[must_use]
    pub fn max_axis(&self, axis: usize) -> Self {
        self.reduce_axis(axis, |a, b| if b.reduction_gt(a) { b } else { a })
    }

    /// Minimum along axis.
    #[must_use]
    pub fn min_axis(&self, axis: usize) -> Self {
        self.reduce_axis(axis, |a, b| if a.reduction_gt(b) { b } else { a })
    }

    /// Maximum along axis with keepdim semantics.
    #[must_use]
    pub fn max_axis_keepdim(&self, axis: usize) -> Self {
        self.max_axis(axis)
    }

    /// Minimum along axis with keepdim semantics.
    #[must_use]
    pub fn min_axis_keepdim(&self, axis: usize) -> Self {
        self.min_axis(axis)
    }

    /// Population variance along axis: `E[X^2] - E[X]^2`.
    #[must_use]
    pub fn var_axis(&self, axis: usize) -> Self {
        let mean = self.mean_axis(axis);
        let (mr, mc) = mean.shape();
        let sq = Self::from_fn(self.nrows(), self.ncols(), |r, c| {
            let x = self.get(r, c);
            x * x
        });
        let mean_sq = sq.mean_axis(axis);
        Self::from_fn(mr, mc, |r, c| {
            let m = mean.get(r, c);
            mean_sq.get(r, c) - m * m
        })
    }

    /// Variance along axis with degrees-of-freedom correction.
    ///
    /// `ddof=0` gives population variance (same as `var_axis`).
    /// `ddof=1` gives sample (unbiased) variance with `N-1` denominator.
    #[must_use]
    pub fn var_axis_ddof(&self, axis: usize, ddof: usize) -> Self {
        let n = self.axis_len(axis, "var_axis_ddof");
        assert!(
            n > ddof,
            "nabla: var_axis_ddof requires axis length ({n}) > ddof ({ddof})"
        );
        let mean = self.mean_axis(axis);
        let (mr, mc) = mean.shape();
        // Compute sum of squared deviations, then divide by (n - ddof).
        let sq_dev = match axis {
            0 => Self::from_fn(mr, mc, |_, c| {
                let mu = mean.get(0, c);
                (0..self.nrows()).fold(T::zero(), |acc, r| {
                    let d = self.get(r, c) - mu;
                    acc + d * d
                })
            }),
            _ => Self::from_fn(mr, mc, |r, _| {
                let mu = mean.get(r, 0);
                (0..self.ncols()).fold(T::zero(), |acc, c| {
                    let d = self.get(r, c) - mu;
                    acc + d * d
                })
            }),
        };
        #[allow(clippy::cast_precision_loss)]
        let inv_denom = T::from_f64(1.0 / (n - ddof) as f64);
        &sq_dev * inv_denom
    }

    /// Population standard deviation along axis: `sqrt(var_axis)`.
    #[must_use]
    pub fn std_axis(&self, axis: usize) -> Self {
        let v = self.var_axis(axis);
        Self::from_fn(v.nrows(), v.ncols(), |r, c| v.get(r, c).math_sqrt())
    }

    /// Variance along axis with keepdim semantics.
    #[must_use]
    pub fn var_axis_keepdim(&self, axis: usize) -> Self {
        self.var_axis(axis)
    }

    /// Std deviation along axis with keepdim semantics.
    #[must_use]
    pub fn std_axis_keepdim(&self, axis: usize) -> Self {
        self.std_axis(axis)
    }

    /// Product along axis: axis 0 -> (1, ncols), axis 1 -> (nrows, 1).
    #[must_use]
    pub fn prod_axis(&self, axis: usize) -> Self {
        self.reduce_axis(axis, |a, b| a * b)
    }

    /// Product of all elements.
    #[must_use]
    pub fn prod_all(&self) -> T {
        B::prod_all(&self.storage)
    }

    /// Count of non-zero elements.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        B::count_nonzero(&self.storage)
    }

    /// Cumulative sum along axis (0 = column-wise, 1 = row-wise).
    #[must_use]
    pub fn cumsum(&self, axis: usize) -> Self {
        match axis {
            1 => Self::from_storage(B::cumsum_axis1(&self.storage)),
            0 => {
                let t = Self::from_storage(B::transpose(&self.storage));
                let cs = Self::from_storage(B::cumsum_axis1(&t.storage));
                Self::from_storage(B::transpose(&cs.storage))
            }
            _ => panic!("nabla: cumsum axis must be 0 or 1, got {axis}"),
        }
    }

    /// Cumulative product along axis.
    #[must_use]
    pub fn cumprod(&self, axis: usize) -> Self {
        match axis {
            1 => Self::from_storage(B::cumprod_axis1(&self.storage)),
            0 => {
                let t = Self::from_storage(B::transpose(&self.storage));
                let cp = Self::from_storage(B::cumprod_axis1(&t.storage));
                Self::from_storage(B::transpose(&cp.storage))
            }
            _ => panic!("nabla: cumprod axis must be 0 or 1, got {axis}"),
        }
    }

    /// Cumulative sum along `dim` with signed (negative) index support.
    #[must_use]
    pub fn cumsum_dim(&self, dim: i64) -> Self {
        let axis = if dim < 0 {
            (2i64 + dim) as isize
        } else {
            dim as isize
        };
        self.cumsum(Self::resolve_axis(axis, "cumsum_dim"))
    }

    /// Cumulative product along `dim` with signed (negative) index support.
    #[must_use]
    pub fn cumprod_dim(&self, dim: i64) -> Self {
        let axis = if dim < 0 {
            (2i64 + dim) as isize
        } else {
            dim as isize
        };
        self.cumprod(Self::resolve_axis(axis, "cumprod_dim"))
    }

    /// Lp-norm along axis.
    #[must_use]
    pub fn norm_axis(&self, p: T, axis: usize) -> Self {
        let inv_p = T::from_f64(1.0 / p.to_f64());
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                let sum = (0..self.nrows()).fold(T::zero(), |acc, r| {
                    acc + self.get(r, c).math_abs().math_powf(p)
                });
                sum.math_powf(inv_p)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                let sum = (0..self.ncols()).fold(T::zero(), |acc, c| {
                    acc + self.get(r, c).math_abs().math_powf(p)
                });
                sum.math_powf(inv_p)
            }),
            _ => panic!("nabla: norm_axis axis must be 0 or 1, got {axis}"),
        }
    }

    /// Argmax along axis. Returns indices tensor.
    #[must_use]
    pub fn argmax_axis(&self, axis: usize) -> Self {
        let two = two::<T>();
        match axis {
            0 => Self::from_fn(1, self.ncols(), |_, c| {
                let mut best_idx = 0usize;
                let mut best_val = self.get(0, c);
                for r in 1..self.nrows() {
                    let v = self.get(r, c);
                    let diff = v - best_val;
                    let is_gt = (diff + diff.math_abs()) / two;
                    if is_gt.to_f64() > 0.0 {
                        best_val = v;
                        best_idx = r;
                    }
                }
                T::from_f64(best_idx as f64)
            }),
            1 => Self::from_fn(self.nrows(), 1, |r, _| {
                let mut best_idx = 0usize;
                let mut best_val = self.get(r, 0);
                for c in 1..self.ncols() {
                    let v = self.get(r, c);
                    let diff = v - best_val;
                    let is_gt = (diff + diff.math_abs()) / two;
                    if is_gt.to_f64() > 0.0 {
                        best_val = v;
                        best_idx = c;
                    }
                }
                T::from_f64(best_idx as f64)
            }),
            _ => panic!("nabla: argmax_axis axis must be 0 or 1, got {axis}"),
        }
    }

    /// Argmin along axis. Returns indices tensor.
    #[must_use]
    pub fn argmin_axis(&self, axis: usize) -> Self {
        let neg = -self;
        neg.argmax_axis(axis)
    }

    /// Sum of all elements satisfying `pred`.
    #[must_use]
    pub fn filter_sum(&self, pred: impl Fn(T) -> bool) -> T {
        let (m, n) = self.shape();
        let mut acc = T::zero();
        for r in 0..m {
            for c in 0..n {
                let v = self.get(r, c);
                if pred(v) {
                    acc = acc + v;
                }
            }
        }
        acc
    }

    /// Count of elements satisfying `pred`.
    #[must_use]
    pub fn count_where(&self, pred: impl Fn(T) -> bool) -> usize {
        let (m, n) = self.shape();
        let mut count = 0usize;
        for r in 0..m {
            for c in 0..n {
                if pred(self.get(r, c)) {
                    count += 1;
                }
            }
        }
        count
    }
}
