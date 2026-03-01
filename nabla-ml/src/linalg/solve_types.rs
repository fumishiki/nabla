use nabla_core::backend::Cpu;
use nabla_core::error::{Error, Result};
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use super::chol::{Lblt, Ldlt, Llt};
use super::eigen::SelfAdjointEigen;
use super::lu::{FullPivLu, PartialPivLu};
use super::qr::{ColPivQr, Qr};
use super::solve_ext::LinalgExt;
use super::svd::Svd;
use super::{Side, from_f64_buf, impl_factorization_methods, require_square, to_f64_buf};

impl_factorization_methods!(general PartialPivLu);
impl_factorization_methods!(symmetric Llt);
impl_factorization_methods!(symmetric Ldlt);

impl Lblt<f64> {
    /// Reconstruct the stored matrix.
    #[must_use]
    pub fn reconstruct(&self) -> Tensor<f64, Cpu> {
        self.reconstruct_impl()
    }
}

impl Qr<f64> {
    /// Least-squares solve `A·x ≈ b`.
    #[must_use]
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        self.solve_lstsq_impl(rhs)
    }

    /// In-place least-squares solve.
    pub fn solve_lstsq_in_place(&self, rhs: &mut Tensor<f64, Cpu>) {
        *rhs = self.solve_lstsq_impl(rhs);
    }

    /// Extract Q matrix (`m × m`).
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn q_matrix(&self) -> Tensor<f64, Cpu> {
        let m = self.m;
        let mut eye = vec![0.0f64; m * m];
        for i in 0..m {
            eye[i * m + i] = 1.0;
        }
        let qt = self.apply_qt(&eye, m);
        let mut q = vec![0.0f64; m * m];
        for r in 0..m {
            for c in 0..m {
                q[r * m + c] = qt[c * m + r];
            }
        }
        from_f64_buf(q, m, m)
    }

    /// Extract thin Q matrix (`m × k`, first k columns).
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn q_matrix_thin(&self, k: usize) -> Tensor<f64, Cpu> {
        let m = self.m;
        let k = k.min(m);
        let mut basis = vec![0.0f64; m * k];
        for i in 0..k {
            basis[i * k + i] = 1.0;
        }
        let q_cols = self.apply_q(&basis, k);
        from_f64_buf(q_cols, m, k)
    }

    /// Extract R matrix (`min(m,n) × n`).
    #[must_use]
    pub fn r_matrix(&self) -> Tensor<f64, Cpu> {
        let (m, n) = (self.m, self.n);
        let k = m.min(n);
        let qr_buf = to_f64_buf(&self.qr);
        let mut r = vec![0.0f64; k * n];
        for i in 0..k {
            for j in i..n {
                r[i * n + j] = super::buf_get(&qr_buf, n, i, j);
            }
        }
        from_f64_buf(r, k, n)
    }

    /// Unpack into `(Q, R)` matrices.
    #[must_use]
    pub fn into_parts(self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>) {
        let q = self.q_matrix();
        let r = self.r_matrix();
        (q, r)
    }
}

impl ColPivQr<f64> {
    /// Least-squares solve `A·x ≈ b`.
    #[must_use]
    pub fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Tensor<f64, Cpu> {
        self.solve_lstsq_impl(rhs)
    }
}

pub struct Diagonal<T: Scalar> {
    diag: Vec<T>,
}

impl<T: Scalar> Diagonal<T> {
    /// Create from a vector of diagonal elements.
    #[must_use]
    pub fn new(diag: Vec<T>) -> Self {
        Self { diag }
    }

    /// Side length for the square matrix.
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.diag.len()
    }

    /// Diagonal element at index `i`.
    #[must_use]
    #[inline]
    pub fn get(&self, i: usize) -> T {
        self.diag[i]
    }

    /// Convert to a dense `n × n` tensor.
    #[must_use]
    pub fn to_tensor(&self) -> Tensor<T> {
        let n = self.size();
        Tensor::from_fn(
            n,
            n,
            |r, c| if r == c { self.diag[r] } else { T::zero_impl() },
        )
    }

    /// Diagonal-times-dense multiplication.
    pub fn mul_dense(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        let n = self.size();
        if rhs.nrows() != n {
            return Err(Error::mismatch((n, rhs.ncols()), rhs.shape()));
        }
        Ok(Tensor::from_fn(n, rhs.ncols(), |r, c| {
            self.diag[r] * rhs.get(r, c)
        }))
    }
}

pub struct Symmetric<T: Scalar> {
    tensor: Tensor<T, Cpu>,
    side: Side,
}

impl<T: Scalar> Symmetric<T> {
    /// Wrap `tensor` as symmetric, reading from `side`.
    pub fn new(tensor: Tensor<T, Cpu>, side: Side) -> Result<Self> {
        require_square(tensor.shape(), "Symmetric::new")?;
        Ok(Self { tensor, side })
    }

    /// Borrow the underlying tensor.
    #[must_use]
    #[inline]
    pub fn as_tensor(&self) -> &Tensor<T, Cpu> {
        &self.tensor
    }

    /// Which triangle is authoritative.
    #[must_use]
    #[inline]
    pub fn side(&self) -> Side {
        self.side
    }
}

impl Symmetric<f64> {
    /// Cholesky factorization.
    pub fn llt(&self) -> Result<Llt<f64>> {
        self.tensor.llt(self.side)
    }

    /// Self-adjoint eigendecomposition.
    pub fn eigen(&self) -> Result<SelfAdjointEigen<f64>> {
        self.tensor.self_adjoint_eigen(self.side)
    }

    /// Alias for `eigen`.
    pub fn eigh(&self) -> Result<SelfAdjointEigen<f64>> {
        self.eigen()
    }

    /// Eigenvalues only.
    pub fn eigenvalues(&self) -> Result<Vec<f64>> {
        self.tensor.self_adjoint_eigenvalues(self.side)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriKind {
    /// Lower-triangular (diagonal and below).
    Lower,
    /// Upper-triangular (diagonal and above).
    Upper,
    /// Unit lower-triangular (implicit 1 on diagonal).
    UnitLower,
    /// Unit upper-triangular (implicit 1 on diagonal).
    UnitUpper,
}

pub struct Triangular<T: Scalar> {
    tensor: Tensor<T, Cpu>,
    kind: TriKind,
}

impl<T: Scalar> Triangular<T> {
    /// Wrap `tensor` as triangular with the given `kind`.
    pub fn new(tensor: Tensor<T, Cpu>, kind: TriKind) -> Result<Self> {
        require_square(tensor.shape(), "Triangular::new")?;
        Ok(Self { tensor, kind })
    }

    /// Borrow the underlying tensor.
    #[must_use]
    #[inline]
    pub fn as_tensor(&self) -> &Tensor<T, Cpu> {
        &self.tensor
    }

    /// The triangular kind of this wrapper.
    #[must_use]
    #[inline]
    pub fn kind(&self) -> &TriKind {
        &self.kind
    }
}

impl Triangular<f64> {
    /// Solve `T·x = b` in place.
    pub fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        match self.kind {
            TriKind::Lower => self.tensor.solve_lower_triangular_in_place(rhs),
            TriKind::Upper => self.tensor.solve_upper_triangular_in_place(rhs),
            TriKind::UnitLower => self.tensor.solve_unit_lower_triangular_in_place(rhs),
            TriKind::UnitUpper => self.tensor.solve_unit_upper_triangular_in_place(rhs),
        }
    }
}

#[inline]
fn promote_f32(t: &Tensor<f32, Cpu>) -> Tensor<f64, Cpu> {
    let (m, n) = t.shape();
    Tensor::<f64, Cpu>::from_fn(m, n, |r, c| f64::from(t.get(r, c)))
}

impl LinalgExt for Tensor<f32, Cpu> {
    fn partial_piv_lu(&self) -> Result<PartialPivLu<f64>> {
        promote_f32(self).partial_piv_lu()
    }

    fn full_piv_lu(&self) -> Result<FullPivLu<f64>> {
        promote_f32(self).full_piv_lu()
    }

    fn qr(&self) -> Qr<f64> {
        promote_f32(self).qr()
    }

    fn col_piv_qr(&self) -> ColPivQr<f64> {
        promote_f32(self).col_piv_qr()
    }

    fn llt(&self, side: Side) -> Result<Llt<f64>> {
        promote_f32(self).llt(side)
    }

    fn ldlt(&self, side: Side) -> Result<Ldlt<f64>> {
        promote_f32(self).ldlt(side)
    }

    fn lblt(&self, side: Side) -> Lblt<f64> {
        promote_f32(self).lblt(side)
    }

    fn svd(&self) -> Result<Svd<f64>> {
        promote_f32(self).svd()
    }

    fn thin_svd(&self) -> Result<Svd<f64>> {
        promote_f32(self).thin_svd()
    }

    fn singular_values(&self) -> Result<Vec<f64>> {
        promote_f32(self).singular_values()
    }

    fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<f64>> {
        promote_f32(self).self_adjoint_eigen(side)
    }

    fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<f64>> {
        promote_f32(self).self_adjoint_eigenvalues(side)
    }

    fn sym(&self, side: Side) -> Result<Symmetric<f64>> {
        promote_f32(self).sym(side)
    }

    fn solve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).solve(rhs)
    }

    fn solve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).solve_transpose(rhs)
    }

    fn solve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).solve_adjoint(rhs)
    }

    fn rsolve(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).rsolve(rhs)
    }

    fn rsolve_transpose(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).rsolve_transpose(rhs)
    }

    fn rsolve_adjoint(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).rsolve_adjoint(rhs)
    }

    fn solve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_in_place(rhs)
    }

    fn solve_transpose_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_transpose_in_place(rhs)
    }

    fn solve_adjoint_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_adjoint_in_place(rhs)
    }

    fn rsolve_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).rsolve_in_place(rhs)
    }

    fn solve_lstsq(&self, rhs: &Tensor<f64, Cpu>) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).solve_lstsq(rhs)
    }

    fn solve_lstsq_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_lstsq_in_place(rhs)
    }

    fn solve_lower_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_lower_triangular_in_place(rhs)
    }

    fn solve_upper_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_upper_triangular_in_place(rhs)
    }

    fn solve_unit_lower_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_unit_lower_triangular_in_place(rhs)
    }

    fn solve_unit_upper_triangular_in_place(&self, rhs: &mut Tensor<f64, Cpu>) -> Result<()> {
        promote_f32(self).solve_unit_upper_triangular_in_place(rhs)
    }

    fn partial_piv_lu_reconstruct(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).partial_piv_lu_reconstruct()
    }

    fn partial_piv_lu_inverse(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).partial_piv_lu_inverse()
    }

    fn llt_reconstruct(&self, side: Side) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).llt_reconstruct(side)
    }

    fn llt_inverse(&self, side: Side) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).llt_inverse(side)
    }

    fn ldlt_reconstruct(&self, side: Side) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).ldlt_reconstruct(side)
    }

    fn ldlt_inverse(&self, side: Side) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).ldlt_inverse(side)
    }

    fn lblt_reconstruct(&self, side: Side) -> Tensor<f64, Cpu> {
        promote_f32(self).lblt_reconstruct(side)
    }

    fn det(&self) -> Result<f64> {
        promote_f32(self).det()
    }

    fn logdet(&self) -> Result<f64> {
        promote_f32(self).logdet()
    }

    fn svd_into(&self) -> Result<(Tensor<f64, Cpu>, Vec<f64>, Tensor<f64, Cpu>)> {
        promote_f32(self).svd_into()
    }

    fn qr_into(&self) -> (Tensor<f64, Cpu>, Tensor<f64, Cpu>) {
        promote_f32(self).qr_into()
    }

    fn cond(&self) -> Result<f64> {
        promote_f32(self).cond()
    }

    fn rank(&self, tol: f64) -> Result<usize> {
        promote_f32(self).rank(tol)
    }

    fn pinv(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).pinv()
    }

    fn matrix_power(&self, n: i32) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).matrix_power(n)
    }

    fn eig_into(&self) -> Result<(Vec<(f64, f64)>, Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        promote_f32(self).eig_into()
    }

    fn geig(&self, b: &Self) -> Result<Vec<(f64, f64)>> {
        let a64 = promote_f32(self);
        let b64 = promote_f32(b);
        a64.geig(&b64)
    }

    fn cond1(&self) -> Result<f64> {
        promote_f32(self).cond1()
    }

    fn cond_inf(&self) -> Result<f64> {
        promote_f32(self).cond_inf()
    }

    fn cond_p(&self, p: f64) -> Result<f64> {
        promote_f32(self).cond_p(p)
    }

    fn inv(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).inv()
    }

    fn null_space(&self, tol: f64) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).null_space(tol)
    }

    fn orth(&self, tol: f64) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).orth(tol)
    }

    fn slogdet(&self) -> Result<(f64, f64)> {
        promote_f32(self).slogdet()
    }

    fn expm(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).expm()
    }

    fn logm(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).logm()
    }

    fn sqrtm(&self) -> Result<Tensor<f64, Cpu>> {
        promote_f32(self).sqrtm()
    }

    fn schur_decomp(&self) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        promote_f32(self).schur_decomp()
    }

    fn polar_decomp(&self) -> Result<(Tensor<f64, Cpu>, Tensor<f64, Cpu>)> {
        promote_f32(self).polar_decomp()
    }
}
