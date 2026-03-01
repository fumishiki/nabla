
#[macro_use]
mod core;
mod factor_lu_qr;
mod factor_chol_svd;
mod solve_ext;
mod solve_types;

pub mod eigen;
pub mod equation;
pub mod matrix_fn;
pub mod structured;

pub use core::Side;
pub(crate) use core::*;
pub use solve_ext::LinalgExt;

pub use crate::module::backslash;
pub use solve_types::{Diagonal, Symmetric, TriKind, Triangular};

pub use factor_lu_qr::lu;
pub use factor_lu_qr::qr;
pub use factor_chol_svd::chol;
pub use factor_chol_svd::svd;

pub use {
    chol::{Lblt, Ldlt, Llt},
    eigen::SelfAdjointEigen,
    equation::{discrete_lyapunov, discrete_sylvester, lyapunov, solve_tridiag, sylvester},
    lu::{FullPivLu, PartialPivLu},
    matrix_fn::{expm, logm, schur, sqrtm},
    qr::{ColPivQr, Qr},
    structured::{
        balance, care, circulant, continuous_riccati, frechet_deriv, hessenberg, polar, toeplitz,
        vandermonde, vandermonde_rect,
    },
    svd::{RandomizedSvd, Svd, SvdParams},
};
