use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, RangeBounds, Sub, SubAssign};
use std::any::TypeId;

use crate::backend::Backend;
use crate::scalar::Scalar;

use super::{Tensor, resolve_range, two};

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Validate two tensors have the same shape.
    fn assert_same_shape(&self, other: &Self, op: &str) {
        assert!(
            self.nrows() == other.nrows() && self.ncols() == other.ncols(),
            "{op}: shape mismatch ({},{}) vs ({},{})",
            self.nrows(),
            self.ncols(),
            other.nrows(),
            other.ncols()
        );
    }

    /// Validate vector matches row/column dimension for broadcast.
    fn assert_broadcast_shape(&self, vec: &Self, axis: usize, op: &str) {
        if axis == 0 {
            assert!(
                vec.nrows() == 1 && vec.ncols() == self.ncols(),
                "{op}: expected (1,{}) got ({},{})",
                self.ncols(),
                vec.nrows(),
                vec.ncols()
            );
        } else {
            assert!(
                vec.ncols() == 1 && vec.nrows() == self.nrows(),
                "{op}: expected ({},1) got ({},{})",
                self.nrows(),
                vec.nrows(),
                vec.ncols()
            );
        }
    }

    /// Expand a row/column vector to match self.
    fn expand_broadcast_vec(&self, vec: &Self, axis: usize, op: &str) -> Self {
        self.assert_broadcast_shape(vec, axis, op);
        let (m, n) = self.shape();
        vec.expand(m, n)
    }

    #[inline]
    pub(super) fn check_range(name: &str, index: usize, bound: usize) {
        assert!(
            index < bound,
            "{name} index {index} out of bounds for {bound}"
        );
    }

    #[inline]
    pub(super) fn check_half_open(name: &str, start: usize, end: usize, bound: usize) {
        assert!(
            start <= end,
            "{name} start ({start}) must be <= end ({end})"
        );
        assert!(end <= bound, "{name} end ({end}) must be <= {bound}");
    }

    // ---- Element access / slicing ----

    /// Set element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        Self::check_range("set row", row, self.nrows());
        Self::check_range("set col", col, self.ncols());
        B::set(&mut self.storage, row, col, val);
    }

    /// Copy a submatrix into a fresh tensor.
    #[must_use]
    pub fn submatrix(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        Self::check_half_open("submatrix row", row_start, row_end, self.nrows());
        Self::check_half_open("submatrix col", col_start, col_end, self.ncols());
        let nrows = row_end - row_start;
        let ncols = col_end - col_start;
        Self::from_storage(B::submatrix(
            &self.storage,
            row_start,
            col_start,
            nrows,
            ncols,
        ))
    }

    /// Slice by row and column ranges, returning a new `Tensor` (copy).
    #[must_use]
    pub fn slice(&self, rows: impl RangeBounds<usize>, cols: impl RangeBounds<usize>) -> Self {
        let (rs, re) = resolve_range(rows, self.nrows());
        let (cs, ce) = resolve_range(cols, self.ncols());
        self.submatrix(rs, re, cs, ce)
    }

    /// Slice rows, all columns.
    #[must_use]
    pub fn slice_rows(&self, rows: impl RangeBounds<usize>) -> Self {
        self.slice(rows, ..)
    }

    /// Slice columns, all rows.
    #[must_use]
    pub fn slice_cols(&self, cols: impl RangeBounds<usize>) -> Self {
        self.slice(.., cols)
    }

    /// Extract a row vector tensor with shape `1 x ncols`.
    #[must_use]
    pub fn row(&self, row: usize) -> Self {
        Self::check_range("row", row, self.nrows());
        self.slice_rows(row..=row)
    }

    /// Extract a column vector tensor with shape `nrows x 1`.
    #[must_use]
    pub fn col(&self, col: usize) -> Self {
        Self::check_range("col", col, self.ncols());
        self.slice_cols(col..=col)
    }

    /// Split into four quadrants: `(top-left, top-right, bottom-left, bottom-right)`.
    #[must_use]
    pub fn split_at(&self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        Self::check_half_open("split row", 0, row, self.nrows());
        Self::check_half_open("split col", 0, col, self.ncols());
        let top_left = self.submatrix(0, row, 0, col);
        let top_right = self.submatrix(0, row, col, self.ncols());
        let bottom_left = self.submatrix(row, self.nrows(), 0, col);
        let bottom_right = self.submatrix(row, self.nrows(), col, self.ncols());
        (top_left, top_right, bottom_left, bottom_right)
    }

    /// Write values from `src` into the sub-region defined by `rows` and `cols`.
    pub fn slice_set(
        &mut self,
        rows: impl RangeBounds<usize>,
        cols: impl RangeBounds<usize>,
        src: &Self,
    ) {
        let (rs, re) = resolve_range(rows, self.nrows());
        let (cs, ce) = resolve_range(cols, self.ncols());
        let h = re - rs;
        let w = ce - cs;
        assert_eq!(
            src.nrows(),
            h,
            "slice_set: src nrows {srn} != region height {h}",
            srn = src.nrows()
        );
        assert_eq!(
            src.ncols(),
            w,
            "slice_set: src ncols {scn} != region width {w}",
            scn = src.ncols()
        );
        B::slice_set(&mut self.storage, rs, cs, &src.storage);
    }

    // ---- Scalar extraction ----

    /// Extract the single element from a `(1, 1)` tensor.
    ///
    /// Panics with a descriptive message if the tensor is not scalar-shaped.
    #[must_use]
    #[inline]
    pub fn item(&self) -> T {
        let (m, n) = self.shape();
        assert!(
            m == 1 && n == 1,
            "nabla: item() requires shape (1, 1), got ({m}, {n})"
        );
        self.get(0, 0)
    }

    /// Pre-fetch data to host-side cache for fast element access.
    /// On GPU backends this triggers a single bulk device-to-host transfer.
    /// On CPU this is a no-op.
    pub fn prefetch(&self) {
        B::prefetch(&self.storage);
    }

    /// Collect all elements into a flat `Vec<T>` in row-major order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        B::to_vec_async(&self.storage)
    }

    // ---- Element-wise operations ----

    /// Element-wise `x^p` for scalar exponent `p`.
    #[must_use]
    #[inline]
    pub fn powf(&self, p: T) -> Self {
        Self::from_storage(B::powf(&self.storage, p))
    }

    /// Element-wise multiplication `self[i,j] * other[i,j]`.
    #[must_use]
    pub fn emul(&self, other: &Self) -> Self {
        self.assert_same_shape(other, "nabla: emul");
        Self::from_storage(B::emul(&self.storage, &other.storage))
    }

    /// Element-wise division `self[i,j] / other[i,j]`.
    #[must_use]
    pub fn ediv(&self, other: &Self) -> Self {
        self.assert_same_shape(other, "nabla: ediv");
        Self::from_storage(B::ediv(&self.storage, &other.storage))
    }

    /// Element-wise `atan2(self, other)`.
    #[must_use]
    pub fn atan2(&self, other: &Self) -> Self {
        self.assert_same_shape(other, "nabla: atan2");
        Self::from_storage(B::atan2(&self.storage, &other.storage))
    }

    /// Element-wise power `self[i,j] ^ other[i,j]`.
    #[must_use]
    pub fn epow(&self, other: &Self) -> Self {
        self.assert_same_shape(other, "nabla: epow");
        let (m, n) = self.shape();
        let inputs = [self.__storage_ptr(), other.__storage_ptr()];
        let expr = if TypeId::of::<T>() == TypeId::of::<f32>() {
            "powf(in0[i], in1[i])"
        } else {
            "pow(in0[i], in1[i])"
        };
        Self::__fuse_elementwise(
            &inputs,
            m,
            n,
            |r, c| self.get(r, c).math_powf(other.get(r, c)),
            expr,
            "epow",
            2,
            0,
        )
    }

    /// Element-wise clamp: values below `lo` become `lo`, above `hi` become `hi`.
    #[must_use]
    pub fn clamp(&self, lo: T, hi: T) -> Self {
        let two = two::<T>();
        let (m, n) = self.shape();
        let lo_t = Tensor::fill(1, 1, lo).expand(m, n);
        let hi_t = Tensor::fill(1, 1, hi).expand(m, n);
        let clamped_lo = (self + &lo_t + (self - &lo_t).abs()) / two;
        (&clamped_lo + &hi_t - (&clamped_lo - &hi_t).abs()) / two
    }

    /// Element-wise sign: returns -1, 0, or 1 for each element.
    ///
    /// Uses `T::zero()` and `T::one_impl()` for type-generic comparison.
    #[must_use]
    pub fn sign(&self) -> Self {
        let (m, n) = self.shape();
        let inputs = [self.__storage_ptr()];
        Self::__fuse_elementwise(
            &inputs,
            m,
            n,
            |r, c| {
                let v = self.get(r, c).to_f64();
                if v > 0.0 {
                    T::one_impl()
                } else if v < 0.0 {
                    T::zero() - T::one_impl()
                } else {
                    T::zero()
                }
            },
            "((in0[i] > 0) ? 1 : ((in0[i] < 0) ? -1 : 0))",
            "sign",
            1,
            0,
        )
    }

    /// Element-wise remainder: `self % rhs` (each element).
    #[must_use]
    pub fn rem(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs, "rem");
        let (m, n) = self.shape();
        if TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>() {
            let inputs = [self.__storage_ptr(), rhs.__storage_ptr()];
            let expr = if TypeId::of::<T>() == TypeId::of::<f32>() {
                "fmodf(in0[i], in1[i])"
            } else {
                "fmod(in0[i], in1[i])"
            };
            return Self::__fuse_elementwise(
                &inputs,
                m,
                n,
                |r, c| {
                    let a = self.get(r, c).to_f64();
                    let b = rhs.get(r, c).to_f64();
                    T::from_f64(a % b)
                },
                expr,
                "rem",
                2,
                0,
            );
        }
        #[cfg(feature = "cpu")]
        {
            Self::from_fn(m, n, |r, c| {
                let a = self.get(r, c).to_f64();
                let b = rhs.get(r, c).to_f64();
                T::from_f64(a % b)
            })
        }
        #[cfg(not(feature = "cpu"))]
        {
            panic!("nabla: rem is CPU-only for non-f32/f64 scalars");
        }
    }

    /// Element-wise remainder with a scalar divisor.
    #[must_use]
    pub fn rem_scalar(&self, rhs: T) -> Self {
        let (m, n) = self.shape();
        if TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>() {
            let rhs_t = Tensor::fill(1, 1, rhs).expand(m, n);
            return self.rem(&rhs_t);
        }
        #[cfg(feature = "cpu")]
        {
            let b = rhs.to_f64();
            Self::from_fn(m, n, |r, c| T::from_f64(self.get(r, c).to_f64() % b))
        }
        #[cfg(not(feature = "cpu"))]
        {
            let _ = rhs;
            panic!("nabla: rem_scalar is CPU-only for non-f32/f64 scalars");
        }
    }

    /// Hadamard (element-wise) product. Alias for [`Tensor::emul`].
    #[deprecated(since = "0.1.0", note = "use emul() instead")]
    #[must_use]
    #[inline]
    pub fn hadamard(&self, rhs: &Self) -> Self {
        self.emul(rhs)
    }

    /// Replace elements where `mask` is non-zero with `value`.
    #[must_use]
    pub fn masked_fill(&self, mask: &Self, value: T) -> Self {
        Tensor::from_storage(B::masked_fill(&self.storage, &mask.storage, value))
    }

    /// Element-wise conditional: `where cond != 0, pick self, else pick other`.
    #[must_use]
    pub fn where_cond(&self, cond: &Self, other: &Self) -> Self {
        Tensor::from_storage(B::where_cond(&self.storage, &cond.storage, &other.storage))
    }

    // ---- Broadcast operations ----

    fn broadcast_op(
        &self,
        vec: &Self,
        axis: usize,
        op: &str,
        f: impl FnOnce(&B::Storage<T>, &B::Storage<T>) -> B::Storage<T>,
    ) -> Self {
        let expanded = self.expand_broadcast_vec(vec, axis, op);
        Tensor::from_storage(f(&self.storage, &expanded.storage))
    }

    /// Add a row vector `(1xn)` to every row of `self (mxn)`.
    #[must_use]
    pub fn broadcast_add_rows(&self, row: &Self) -> Self {
        self.broadcast_op(row, 0, "nabla: broadcast_add_rows", B::add)
    }

    /// Add a column vector `(mx1)` to every column of `self (mxn)`.
    #[must_use]
    pub fn broadcast_add_cols(&self, col: &Self) -> Self {
        self.broadcast_op(col, 1, "nabla: broadcast_add_cols", B::add)
    }

    /// Element-wise multiply each row by a row vector `(1xn)`.
    #[must_use]
    pub fn broadcast_mul_rows(&self, row: &Self) -> Self {
        self.broadcast_op(row, 0, "nabla: broadcast_mul_rows", B::emul)
    }

    /// Element-wise multiply each column by a column vector `(mx1)`.
    #[must_use]
    pub fn broadcast_mul_cols(&self, col: &Self) -> Self {
        self.broadcast_op(col, 1, "nabla: broadcast_mul_cols", B::emul)
    }

    /// In-place add a row vector `(1xn)` to every row.
    pub fn broadcast_add_rows_(&mut self, row: &Self) {
        self.assert_broadcast_shape(row, 0, "nabla: broadcast_add_rows_");
        let (m, n) = self.shape();
        let row_exp = row.expand(m, n);
        B::axpy_inplace(&mut self.storage, T::one_impl(), &row_exp.storage);
    }

    /// In-place add a column vector `(mx1)` to every column.
    pub fn broadcast_add_cols_(&mut self, col: &Self) {
        self.assert_broadcast_shape(col, 1, "nabla: broadcast_add_cols_");
        let (m, n) = self.shape();
        let col_exp = col.expand(m, n);
        B::axpy_inplace(&mut self.storage, T::one_impl(), &col_exp.storage);
    }

    // ---- Transpose / permute ----

    /// Return the transpose: a new `Tensor` of shape `(ncols, nrows)`.
    #[must_use]
    pub fn t(&self) -> Self {
        Self::from_storage(B::transpose(&self.storage))
    }

    /// Return the conjugate transpose (adjoint / Hermitian transpose).
    #[must_use]
    pub fn adjoint(&self) -> Self {
        if T::IS_REAL {
            return Self::from_storage(B::transpose(&self.storage));
        }
        #[cfg(feature = "cpu")]
        {
            let (r, c) = self.shape();
            Self::from_storage(B::from_fn(c, r, |i, j| {
                crate::scalar::math_utils::conj(&self.get(j, i))
            }))
        }
        #[cfg(not(feature = "cpu"))]
        {
            panic!("nabla: adjoint for complex scalars is CPU-only");
        }
    }

    /// Short alias for conjugate transpose.
    #[must_use]
    #[inline]
    pub fn h(&self) -> Self {
        self.adjoint()
    }

    /// Permute axes of a 2-D tensor.
    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            2,
            "nabla: Tensor is 2-D -- permute axes must have length 2"
        );
        assert!(
            axes[0] < 2 && axes[1] < 2 && axes[0] != axes[1],
            "nabla: permute axes must be a permutation of {{0, 1}}, got [{}, {}]",
            axes[0],
            axes[1]
        );
        match (axes[0], axes[1]) {
            (0, 1) => self.clone(),
            _ => self.t(),
        }
    }

    // ---- Matrix multiply ----

    /// Compute `out = a * b` (matrix multiply), overwriting `out`.
    pub fn matmul_into(out: &mut Self, a: &Self, b: &Self) {
        let (m, k_a) = a.shape();
        let (k_b, n) = b.shape();
        assert_eq!(
            k_a, k_b,
            "matmul inner dimension mismatch: a is ({m}, {k_a}), b is ({k_b}, {n})"
        );
        let (out_r, out_c) = out.shape();
        assert!(
            out_r == m && out_c == n,
            "matmul output shape mismatch: expected ({m}, {n}), got ({out_r}, {out_c})"
        );
        B::matmul_into(&mut out.storage, &a.storage, &b.storage);
    }

    /// Compute `a^T @ b` without materializing the transpose.
    #[must_use]
    pub fn matmul_tn(&self, rhs: &Self) -> Self {
        let (k, m) = self.shape();
        let (k2, n) = rhs.shape();
        assert_eq!(k, k2, "matmul_tn: a rows {k} != b rows {k2}");
        let mut out = Self::empty(m, n);
        B::matmul_tn_into(&mut out.storage, &self.storage, &rhs.storage);
        out
    }

    /// Compute `a @ b^T` without materializing the transpose.
    #[must_use]
    pub fn matmul_nt(&self, rhs: &Self) -> Self {
        let (m, k) = self.shape();
        let (n, k2) = rhs.shape();
        assert_eq!(k, k2, "matmul_nt: a cols {k} != b cols {k2}");
        let mut out = Self::empty(m, n);
        B::matmul_nt_into(&mut out.storage, &self.storage, &rhs.storage);
        out
    }

    /// Newton-Schulz orthogonalization toward nearest orthogonal matrix.
    ///
    /// Pre-scales input by `1 / ||X||_F` for convergence, then iterates
    /// the polar Newton-Schulz iteration: `X_{k+1} = X @ (1.5*I - 0.5*X^T@X)`.
    /// Works for both square and rectangular matrices.
    #[must_use]
    pub fn newton_schulz_ortho(&self, iters: usize) -> Self {
        let (rows, cols) = (self.nrows(), self.ncols());
        assert!(rows > 0 && cols > 0, "newton_schulz_ortho: empty matrix");
        let n = rows.min(cols);
        let fnorm = self.norm();
        assert!(
            fnorm.to_f64() > 0.0,
            "newton_schulz_ortho: zero-norm matrix"
        );
        let mut x = self * fnorm.math_recip();
        let eye = Self::from_storage(B::identity(n));
        let a_coeff = T::from_f64(1.5);
        let b_coeff = T::from_f64(-0.5);
        for _ in 0..iters {
            let g = x.matmul_tn(&x);
            let poly = &(&eye * a_coeff) + &(&g * b_coeff);
            x = &x * &poly;
        }
        x
    }

    /// Fused matmul + element-wise activation.
    #[must_use]
    #[cfg(feature = "cpu")]
    pub fn matmul_fused<F>(a: &Self, b: &Self, act: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let c = a * b;
        let (m, n) = c.shape();
        Self::from_storage(B::from_fn(m, n, |r, col| act(c.get(r, col))))
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Apply a closure element-wise, returning a new tensor.
    #[must_use]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let (m, n) = self.shape();
        Self::from_storage(B::from_fn(m, n, |r, c| f(self.get(r, c))))
    }
}

macro_rules! impl_broadcast_binop {
    ($trait:ident, $method:ident, $backend_fn:ident, $name:literal) => {
        impl<T: Scalar, B: Backend> $trait for &Tensor<T, B> {
            type Output = Tensor<T, B>;

            fn $method(self, rhs: Self) -> Self::Output {
                let (m, n) = self.shape();
                let (p, q) = rhs.shape();
                if m == p && n == q {
                    return Tensor::from_storage(B::$backend_fn(&self.storage, &rhs.storage));
                }
                // rhs is broadcastable to self's shape
                if (p == 1 || p == m) && (q == 1 || q == n) && (p, q) != (m, n) {
                    let rhs_exp = rhs.expand(m, n);
                    return Tensor::from_storage(B::$backend_fn(&self.storage, &rhs_exp.storage));
                }
                // self is broadcastable to rhs's shape
                if (m == 1 || m == p) && (n == 1 || n == q) && (m, n) != (p, q) {
                    let self_exp = self.expand(p, q);
                    return Tensor::from_storage(B::$backend_fn(&self_exp.storage, &rhs.storage));
                }
                panic!(
                    "nabla: {} ({}x{}) vs ({}x{}) -- shapes are not broadcast-compatible",
                    $name, m, n, p, q
                );
            }
        }
    };
}

impl_broadcast_binop!(Add, add, add, "add");
impl_broadcast_binop!(Sub, sub, sub, "sub");

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $backend_fn:ident, $sym:literal) => {
        impl<T: Scalar, B: Backend> $trait<&Tensor<T, B>> for Tensor<T, B> {
            fn $method(&mut self, rhs: &Tensor<T, B>) {
                let (m, n) = self.shape();
                let (p, q) = rhs.shape();
                assert!(
                    m == p && n == q,
                    "nabla: {} ({}x{}) vs ({}x{}) -- shapes must match",
                    $sym,
                    m,
                    n,
                    p,
                    q
                );
                self.storage = B::$backend_fn(&self.storage, &rhs.storage);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, add, "+=");
impl_assign_op!(SubAssign, sub_assign, sub, "-=");

impl<T: Scalar, B: Backend> MulAssign<T> for Tensor<T, B> {
    fn mul_assign(&mut self, rhs: T) {
        self.storage = B::scale(&self.storage, rhs);
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// In-place axpy: `self[i] += alpha * x[i]`. Zero allocation, single kernel.
    #[inline]
    pub fn axpy_inplace(&mut self, alpha: T, x: &Self) {
        let (m, n) = self.shape();
        let (p, q) = x.shape();
        assert!(
            m == p && n == q,
            "nabla: axpy_inplace ({m}x{n}) vs ({p}x{q}) -- shapes must match"
        );
        B::axpy_inplace(&mut self.storage, alpha, &x.storage);
    }

    /// In-place add: `self[i] += other[i]`. Zero allocation.
    #[inline]
    pub fn add_assign_inplace(&mut self, other: &Self) {
        self.axpy_inplace(T::one_impl(), other);
    }

    /// In-place scale: `self[i] *= scalar`. Replaces self with scaled result.
    #[inline]
    pub fn scale_assign(&mut self, scalar: T) {
        self.storage = B::scale(&self.storage, scalar);
    }
}

impl<T: Scalar, B: Backend> Neg for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn neg(self) -> Self::Output {
        Tensor::from_storage(B::neg(&self.storage))
    }
}

impl<T: Scalar, B: Backend> Mul for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (m, k_a) = self.shape();
        let (k_b, n) = rhs.shape();
        assert_eq!(
            k_a, k_b,
            "nabla: matmul ({m}x{k_a}) x ({k_b}x{n}) -- inner dims {k_a} != {k_b}"
        );
        let mut out = Tensor::<T, B>::empty(m, n);
        B::matmul_into(&mut out.storage, &self.storage, &rhs.storage);
        out
    }
}

impl<T: Scalar, B: Backend> Mul<T> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_storage(B::scale(&self.storage, rhs))
    }
}

impl<T: Scalar, B: Backend> Div<T> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self * rhs.math_recip()
    }
}

impl<T: Scalar, B: Backend> Div<T> for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

// Owned-variant forwarding: all 3 combos delegate to &T op &T.
macro_rules! impl_owned_binop_fwd {
    ($trait:ident, $method:ident) => {
        impl<T: Scalar, B: Backend> $trait for Tensor<T, B> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
        impl<T: Scalar, B: Backend> $trait<Tensor<T, B>> for &Tensor<T, B> {
            type Output = Tensor<T, B>;
            #[inline]
            fn $method(self, rhs: Tensor<T, B>) -> Self::Output {
                self.$method(&rhs)
            }
        }
        impl<T: Scalar, B: Backend> $trait<&Tensor<T, B>> for Tensor<T, B> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &Self) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

impl_owned_binop_fwd!(Add, add);
impl_owned_binop_fwd!(Sub, sub);
impl_owned_binop_fwd!(Mul, mul);

impl<T: Scalar, B: Backend> Neg for Tensor<T, B> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: Scalar, B: Backend> Mul<T> for Tensor<T, B> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}

macro_rules! impl_scalar_lhs_ops {
    ($($t:ty),*) => { $(
        impl<B: Backend> Mul<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn mul(self, rhs: &Tensor<$t, B>) -> Self::Output { rhs * self }
        }
        impl<B: Backend> Mul<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn mul(self, rhs: Tensor<$t, B>) -> Self::Output { self * &rhs }
        }
        impl<B: Backend> Add<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn add(self, rhs: &Tensor<$t, B>) -> Self::Output {
                let (m, n) = rhs.shape();
                let s = Tensor::<$t, B>::fill(1, 1, self).expand(m, n);
                Tensor::from_storage(B::add(&s.storage, &rhs.storage))
            }
        }
        impl<B: Backend> Add<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn add(self, rhs: Tensor<$t, B>) -> Self::Output { self + &rhs }
        }
        impl<B: Backend> Sub<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn sub(self, rhs: &Tensor<$t, B>) -> Self::Output {
                let (m, n) = rhs.shape();
                let s = Tensor::<$t, B>::fill(1, 1, self).expand(m, n);
                Tensor::from_storage(B::sub(&s.storage, &rhs.storage))
            }
        }
        impl<B: Backend> Sub<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn sub(self, rhs: Tensor<$t, B>) -> Self::Output { self - &rhs }
        }
        impl<B: Backend> Div<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn div(self, rhs: &Tensor<$t, B>) -> Self::Output {
                let (m, n) = rhs.shape();
                let s = Tensor::<$t, B>::fill(1, 1, self).expand(m, n);
                Tensor::from_storage(B::ediv(&s.storage, &rhs.storage))
            }
        }
        impl<B: Backend> Div<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;
            #[inline]
            fn div(self, rhs: Tensor<$t, B>) -> Self::Output { self / &rhs }
        }
    )* };
}

impl_scalar_lhs_ops!(f32, f64);
