// tensor/ops.rs — Tensor arithmetic: element access, element-wise ops, broadcast,
//                  transpose/permute, matrix multiply, and operator overloads.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, RangeBounds, Sub, SubAssign};

use crate::backend::Backend;
use crate::scalar::Scalar;

use super::{resolve_range, two, Tensor};

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

    /// Apply a binary op between self and a broadcast vector along the given axis.
    fn apply_broadcast_op<F: Fn(T, T) -> T>(
        &self,
        vec: &Self,
        axis: usize,
        op: &str,
        f: F,
    ) -> Self {
        self.assert_broadcast_shape(vec, axis, op);
        let (m, n) = self.shape();
        match axis {
            0 => Self::from_fn(m, n, |r, c| f(self.get(r, c), vec.get(0, c))),
            _ => Self::from_fn(m, n, |r, c| f(self.get(r, c), vec.get(r, 0))),
        }
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
        Self::from_storage(B::from_fn(nrows, ncols, |r, c| {
            self.get(row_start + r, col_start + c)
        }))
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
            src.nrows(), h,
            "slice_set: src nrows {srn} != region height {h}",
            srn = src.nrows()
        );
        assert_eq!(
            src.ncols(), w,
            "slice_set: src ncols {scn} != region width {w}",
            scn = src.ncols()
        );
        for r in 0..h {
            for c in 0..w {
                self.set(rs + r, cs + c, src.get(r, c));
            }
        }
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

    /// Collect all elements into a flat `Vec<T>` in row-major order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let (m, n) = self.shape();
        let len = m * n;
        let mut v = Vec::with_capacity(len);
        for r in 0..m {
            for c in 0..n {
                v.push(self.get(r, c));
            }
        }
        v
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
        Self::from_fn(m, n, |r, c| self.get(r, c).math_powf(other.get(r, c)))
    }

    /// Apply a closure element-wise, returning a new tensor.
    #[must_use]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| f(self.get(r, c)))
    }

    /// Element-wise clamp: values below `lo` become `lo`, above `hi` become `hi`.
    #[must_use]
    pub fn clamp(&self, lo: T, hi: T) -> Self {
        let two = two::<T>();
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            let x = self.get(r, c);
            let clamped_lo = (x + lo + (x - lo).math_abs()) / two;
            (clamped_lo + hi - (clamped_lo - hi).math_abs()) / two
        })
    }

    /// Replace elements where `mask` is non-zero with `value`.
    #[must_use]
    pub fn masked_fill(&self, mask: &Self, value: T) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if mask.get(r, c).to_f64() == 0.0 {
                self.get(r, c)
            } else {
                value
            }
        })
    }

    /// Element-wise conditional: `where cond != 0, pick self, else pick other`.
    #[must_use]
    pub fn where_cond(&self, cond: &Self, other: &Self) -> Self {
        let (m, n) = self.shape();
        Self::from_fn(m, n, |r, c| {
            if cond.get(r, c).to_f64() == 0.0 {
                other.get(r, c)
            } else {
                self.get(r, c)
            }
        })
    }

    // ---- Broadcast operations ----

    /// Add a row vector `(1xn)` to every row of `self (mxn)`.
    #[must_use]
    pub fn broadcast_add_rows(&self, row: &Self) -> Self {
        self.apply_broadcast_op(row, 0, "nabla: broadcast_add_rows", |a, b| a + b)
    }

    /// Add a column vector `(mx1)` to every column of `self (mxn)`.
    #[must_use]
    pub fn broadcast_add_cols(&self, col: &Self) -> Self {
        self.apply_broadcast_op(col, 1, "nabla: broadcast_add_cols", |a, b| a + b)
    }

    /// Element-wise multiply each row by a row vector `(1xn)`.
    #[must_use]
    pub fn broadcast_mul_rows(&self, row: &Self) -> Self {
        self.apply_broadcast_op(row, 0, "nabla: broadcast_mul_rows", |a, b| a * b)
    }

    /// Element-wise multiply each column by a column vector `(mx1)`.
    #[must_use]
    pub fn broadcast_mul_cols(&self, col: &Self) -> Self {
        self.apply_broadcast_op(col, 1, "nabla: broadcast_mul_cols", |a, b| a * b)
    }

    /// In-place add a row vector `(1xn)` to every row.
    pub fn broadcast_add_rows_(&mut self, row: &Self) {
        self.assert_broadcast_shape(row, 0, "nabla: broadcast_add_rows_");
        let (m, n) = self.shape();
        for r in 0..m {
            for c in 0..n {
                self.set(r, c, self.get(r, c) + row.get(0, c));
            }
        }
    }

    /// In-place add a column vector `(mx1)` to every column.
    pub fn broadcast_add_cols_(&mut self, col: &Self) {
        self.assert_broadcast_shape(col, 1, "nabla: broadcast_add_cols_");
        let (m, n) = self.shape();
        for r in 0..m {
            for c in 0..n {
                self.set(r, c, self.get(r, c) + col.get(r, 0));
            }
        }
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
        let (r, c) = self.shape();
        Self::from_storage(if T::IS_REAL {
            B::from_fn(c, r, |i, j| self.get(j, i))
        } else {
            B::from_fn(c, r, |i, j| {
                crate::scalar::math_utils::conj(&self.get(j, i))
            })
        })
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
        assert_eq!(axes.len(), 2, "nabla: Tensor is 2-D -- permute axes must have length 2");
        assert!(
            axes[0] < 2 && axes[1] < 2 && axes[0] != axes[1],
            "nabla: permute axes must be a permutation of {{0, 1}}, got [{}, {}]",
            axes[0], axes[1]
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

    /// Fused matmul + element-wise activation.
    #[must_use]
    pub fn matmul_fused<F>(a: &Self, b: &Self, act: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let c = a * b;
        c.map(act)
    }
}

// ---- Operator overloads ----

// Auto-broadcasting Add: exact match, row/col/scalar broadcast.
impl<T: Scalar, B: Backend> Add for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn add(self, rhs: Self) -> Self::Output {
        let (m, n) = self.shape();
        let (p, q) = rhs.shape();
        // Exact match — fast path via backend.
        if m == p && n == q {
            return Tensor::from_storage(B::add(&self.storage, &rhs.storage));
        }
        // (m,n) + (1,n) → broadcast row vector across rows
        if p == 1 && q == n {
            return self.apply_broadcast_op(rhs, 0, "nabla: add broadcast rows", |a, b| a + b);
        }
        // (m,n) + (m,1) → broadcast column vector across columns
        if p == m && q == 1 {
            return self.apply_broadcast_op(rhs, 1, "nabla: add broadcast cols", |a, b| a + b);
        }
        // (m,n) + (1,1) → broadcast scalar
        if p == 1 && q == 1 {
            let s = rhs.get(0, 0);
            return Tensor::from_fn(m, n, |r, c| self.get(r, c) + s);
        }
        // (1,n) + (m,n) → commutative, swap operands
        if m == 1 && n == q {
            return rhs.apply_broadcast_op(self, 0, "nabla: add broadcast rows", |a, b| a + b);
        }
        // (m,1) + (m,n) → commutative, swap operands
        if m == p && n == 1 {
            return rhs.apply_broadcast_op(self, 1, "nabla: add broadcast cols", |a, b| a + b);
        }
        // (1,1) + (m,n) → broadcast scalar
        if m == 1 && n == 1 {
            let s = self.get(0, 0);
            return Tensor::from_fn(p, q, |r, c| s + rhs.get(r, c));
        }
        panic!(
            "nabla: add ({m}x{n}) vs ({p}x{q}) -- shapes are not broadcast-compatible"
        );
    }
}

// Auto-broadcasting Sub: exact match, row/col/scalar broadcast.
impl<T: Scalar, B: Backend> Sub for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn sub(self, rhs: Self) -> Self::Output {
        let (m, n) = self.shape();
        let (p, q) = rhs.shape();
        // Exact match — fast path via backend.
        if m == p && n == q {
            return Tensor::from_storage(B::sub(&self.storage, &rhs.storage));
        }
        // (m,n) - (1,n) → broadcast row vector across rows
        if p == 1 && q == n {
            return self.apply_broadcast_op(rhs, 0, "nabla: sub broadcast rows", |a, b| a - b);
        }
        // (m,n) - (m,1) → broadcast column vector across columns
        if p == m && q == 1 {
            return self.apply_broadcast_op(rhs, 1, "nabla: sub broadcast cols", |a, b| a - b);
        }
        // (m,n) - (1,1) → broadcast scalar
        if p == 1 && q == 1 {
            let s = rhs.get(0, 0);
            return Tensor::from_fn(m, n, |r, c| self.get(r, c) - s);
        }
        // (1,n) - (m,n) → row_vec - matrix = -(matrix - row_vec)
        if m == 1 && n == q {
            return Tensor::from_fn(p, q, |r, c| self.get(0, c) - rhs.get(r, c));
        }
        // (m,1) - (m,n) → col_vec - matrix
        if m == p && n == 1 {
            return Tensor::from_fn(p, q, |r, c| self.get(r, 0) - rhs.get(r, c));
        }
        // (1,1) - (m,n) → scalar broadcast
        if m == 1 && n == 1 {
            let s = self.get(0, 0);
            return Tensor::from_fn(p, q, |r, c| s - rhs.get(r, c));
        }
        panic!(
            "nabla: sub ({m}x{n}) vs ({p}x{q}) -- shapes are not broadcast-compatible"
        );
    }
}

/// In-place element-wise addition: `tensor += &other`.
impl<T: Scalar, B: Backend> AddAssign<&Tensor<T, B>> for Tensor<T, B> {
    fn add_assign(&mut self, rhs: &Tensor<T, B>) {
        let (m, n) = self.shape();
        let (p, q) = rhs.shape();
        assert!(m == p && n == q, "nabla: += ({m}x{n}) vs ({p}x{q}) -- shapes must match");
        self.storage = B::add(&self.storage, &rhs.storage);
    }
}

/// In-place element-wise subtraction: `tensor -= &other`.
impl<T: Scalar, B: Backend> SubAssign<&Tensor<T, B>> for Tensor<T, B> {
    fn sub_assign(&mut self, rhs: &Tensor<T, B>) {
        let (m, n) = self.shape();
        let (p, q) = rhs.shape();
        assert!(m == p && n == q, "nabla: -= ({m}x{n}) vs ({p}x{q}) -- shapes must match");
        self.storage = B::sub(&self.storage, &rhs.storage);
    }
}

/// In-place scalar scaling: `tensor *= scalar`.
impl<T: Scalar, B: Backend> MulAssign<T> for Tensor<T, B> {
    fn mul_assign(&mut self, rhs: T) {
        self.storage = B::scale(&self.storage, rhs);
    }
}

impl<T: Scalar, B: Backend> Neg for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn neg(self) -> Self::Output {
        Tensor::from_storage(B::neg(&self.storage))
    }
}

/// Matrix multiply (`*`). Panics if inner dimensions do not match.
impl<T: Scalar, B: Backend> Mul for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (m, k_a) = self.shape();
        let (k_b, n) = rhs.shape();
        assert_eq!(
            k_a, k_b,
            "nabla: matmul ({m}x{k_a}) x ({k_b}x{n}) -- inner dims {k_a} != {k_b}"
        );
        let mut out = Tensor::<T, B>::zeros(m, n);
        B::matmul_into(&mut out.storage, &self.storage, &rhs.storage);
        out
    }
}

/// Scalar multiply: `&tensor * scalar`.
impl<T: Scalar, B: Backend> Mul<T> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_storage(B::scale(&self.storage, rhs))
    }
}

/// Scalar division: `&tensor / scalar`.
impl<T: Scalar, B: Backend> Div<T> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self * rhs.math_recip()
    }
}

/// Scalar division: `tensor / scalar` (owned).
impl<T: Scalar, B: Backend> Div<T> for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

// ---- Owned / mixed-ref operator overloads ----

/// `Tensor + Tensor` (owned + owned) — delegates to `&lhs + &rhs`.
impl<T: Scalar, B: Backend> Add for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

/// `&Tensor + Tensor` (ref + owned) — delegates to `&lhs + &rhs`.
impl<T: Scalar, B: Backend> Add<Tensor<T, B>> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    #[inline]
    fn add(self, rhs: Tensor<T, B>) -> Self::Output {
        self + &rhs
    }
}

/// `Tensor + &Tensor` (owned + ref) — delegates to `&lhs + rhs`.
impl<T: Scalar, B: Backend> Add<&Tensor<T, B>> for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

/// `Tensor - Tensor` (owned - owned) — delegates to `&lhs - &rhs`.
impl<T: Scalar, B: Backend> Sub for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

/// `&Tensor - Tensor` (ref - owned) — delegates to `&lhs - &rhs`.
impl<T: Scalar, B: Backend> Sub<Tensor<T, B>> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    #[inline]
    fn sub(self, rhs: Tensor<T, B>) -> Self::Output {
        self - &rhs
    }
}

/// `Tensor - &Tensor` (owned - ref) — delegates to `&lhs - rhs`.
impl<T: Scalar, B: Backend> Sub<&Tensor<T, B>> for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

/// `-Tensor` (owned negation) — delegates to `-&self`.
impl<T: Scalar, B: Backend> Neg for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        -&self
    }
}

// ---- Owned matmul operator overloads (I-3) ----

/// `Tensor * Tensor` (owned * owned) — delegates to `&lhs * &rhs`.
impl<T: Scalar, B: Backend> Mul for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

/// `Tensor * &Tensor` (owned * ref) — delegates to `&lhs * rhs`.
impl<T: Scalar, B: Backend> Mul<&Tensor<T, B>> for Tensor<T, B> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

/// `&Tensor * Tensor` (ref * owned) — delegates to `self * &rhs`.
impl<T: Scalar, B: Backend> Mul<Tensor<T, B>> for &Tensor<T, B> {
    type Output = Tensor<T, B>;

    #[inline]
    fn mul(self, rhs: Tensor<T, B>) -> Self::Output {
        self * &rhs
    }
}

// ---- Commutative scalar operators (I-2) ----

/// Generate `scalar * &Tensor`, `scalar * Tensor`, `scalar + &Tensor`, `scalar + Tensor`,
/// `scalar - &Tensor`, `scalar - Tensor` for concrete scalar types.
macro_rules! impl_scalar_lhs_ops {
    ($($t:ty),*) => { $(
        /// `scalar * &Tensor` — scalar scaling (commutative).
        impl<B: Backend> Mul<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn mul(self, rhs: &Tensor<$t, B>) -> Self::Output {
                rhs * self
            }
        }

        /// `scalar * Tensor` — scalar scaling (commutative, owned).
        impl<B: Backend> Mul<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn mul(self, rhs: Tensor<$t, B>) -> Self::Output {
                &rhs * self
            }
        }

        /// `scalar + &Tensor` — broadcast scalar add (commutative).
        impl<B: Backend> Add<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn add(self, rhs: &Tensor<$t, B>) -> Self::Output {
                let (m, n) = rhs.shape();
                Tensor::from_fn(m, n, |r, c| self + rhs.get(r, c))
            }
        }

        /// `scalar + Tensor` — broadcast scalar add (commutative, owned).
        impl<B: Backend> Add<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn add(self, rhs: Tensor<$t, B>) -> Self::Output {
                self + &rhs
            }
        }

        /// `scalar - &Tensor` — broadcast scalar sub.
        impl<B: Backend> Sub<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn sub(self, rhs: &Tensor<$t, B>) -> Self::Output {
                let (m, n) = rhs.shape();
                Tensor::from_fn(m, n, |r, c| self - rhs.get(r, c))
            }
        }

        /// `scalar - Tensor` — broadcast scalar sub (owned).
        impl<B: Backend> Sub<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn sub(self, rhs: Tensor<$t, B>) -> Self::Output {
                self - &rhs
            }
        }

        /// `scalar / &Tensor` — element-wise reciprocal scaled by scalar.
        impl<B: Backend> Div<&Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn div(self, rhs: &Tensor<$t, B>) -> Self::Output {
                let (m, n) = rhs.shape();
                Tensor::from_fn(m, n, |r, c| self / rhs.get(r, c))
            }
        }

        /// `scalar / Tensor` — element-wise reciprocal scaled by scalar (owned).
        impl<B: Backend> Div<Tensor<$t, B>> for $t {
            type Output = Tensor<$t, B>;

            #[inline]
            fn div(self, rhs: Tensor<$t, B>) -> Self::Output {
                self / &rhs
            }
        }
    )* };
}

impl_scalar_lhs_ops!(f32, f64);
