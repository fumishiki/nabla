#![cfg(feature = "cpu")]
#![allow(unused_imports)]

use nabla::cas::{Expr, diff, eval, eval_tensor, simplify};
use nabla::ode::{AdaptiveConfig, dormand_prince, rk4};
use nabla::prelude::*;
use nabla::{between, frange};
use std::collections::HashMap;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

fn linear_f64(rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::from_fn(rows, cols, |i, j| (i * cols + j + 1) as f64)
}

fn assert_approx_grid(got: &Tensor<f64>, expected: &Tensor<f64>, tol: f64) {
    assert_eq!(got.shape(), expected.shape(), "shape mismatch");
    let (r, c) = got.shape();
    for i in 0..r {
        for j in 0..c {
            assert!(
                (got.get(i, j) - expected.get(i, j)).abs() < tol,
                "mismatch at ({i},{j}): got {}, expected {}",
                got.get(i, j),
                expected.get(i, j)
            );
        }
    }
}


#[nabla_grad]
#[allow(dead_code)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn autograd_simple_backward() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let x = tape.variable(linear_f64(2, 2)).expect("variable");
    let y = x.emul(&x);
    y.backward().expect("backward failed");
    let grad = x.grad().expect("grad failed");
    assert!((grad.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((grad.get(0, 1) - 4.0).abs() < 1e-10);
    assert!((grad.get(1, 0) - 6.0).abs() < 1e-10);
    assert!((grad.get(1, 1) - 8.0).abs() < 1e-10);
}


#[test]
fn autograd_chain_rule() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let x = tape.variable(Tensor::from_fn(1, 1, |_, _| 1.0_f64)).expect("variable");
    let x2 = x.emul(&x);
    let y = x2.sin();
    y.backward().expect("backward failed");
    let grad = x.grad().expect("grad failed");
    let expected = 2.0_f64 * 1.0_f64.cos();
    assert!((grad.get(0, 0) - expected).abs() < 1e-10);
}


#[test]
fn autograd_matmul_backward() {
    use nabla::autograd::Tape;
    let tape = Tape::<f64, Cpu>::new();
    let a = tape.variable(mat![[1.0_f64, 2.0], [3.0, 4.0]]).expect("variable");
    let b = tape.variable(mat![[5.0_f64, 6.0], [7.0, 8.0]]).expect("variable");
    let c = &a * &b;
    c.backward().expect("backward failed");
    let grad_a = a.grad().expect("grad_a failed");
    let grad_b = b.grad().expect("grad_b failed");
    assert!((grad_a.get(0, 0) - 11.0).abs() < 1e-10);
    assert!((grad_a.get(0, 1) - 15.0).abs() < 1e-10);
    assert!((grad_a.get(1, 0) - 11.0).abs() < 1e-10);
    assert!((grad_a.get(1, 1) - 15.0).abs() < 1e-10);
    assert!((grad_b.get(0, 0) - 4.0).abs() < 1e-10);
    assert!((grad_b.get(0, 1) - 4.0).abs() < 1e-10);
    assert!((grad_b.get(1, 0) - 6.0).abs() < 1e-10);
    assert!((grad_b.get(1, 1) - 6.0).abs() < 1e-10);
}


#[test]
fn dual_exp_ln() {
    // f(x) = exp(ln(x)) = x, f'(x) = 1
    use nabla::prelude::Dual;
    use nabla::scalar::MathOps;
    let x = Dual::new(3.0_f64, 1.0);
    let y = x.math_ln().math_exp();
    assert!((y.value - 3.0).abs() < 1e-12);
    assert!((y.deriv - 1.0).abs() < 1e-12);
}


#[test]
fn grad_quadratic() {
    // f(x) = sum(x^2), df/dx = 2x; at x=[[3.0]] => grad = [[6.0]]
    let x = mat![[3.0_f64]];
    let g =
        grad(|xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var(), &x).expect("grad returned None");
    assert!((g[(0, 0)] - 6.0).abs() < 1e-10);
}


#[test]
fn gradient_prep_reuse() {
    let f = |xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var();
    let x1 = mat![[2.0_f64]];
    let x2 = mat![[5.0_f64]];
    let prep = gradient_prep(&f, &x1);
    let g1 = gradient(&f, &x1, &prep).expect("gradient returned None");
    let g2 = gradient(&f, &x2, &prep).expect("gradient returned None");
    assert!((g1[(0, 0)] - 4.0).abs() < 1e-10);
    assert!((g2[(0, 0)] - 10.0).abs() < 1e-10);
}


#[test]
fn nabla_grad_sigmoid() {
    let (val, grad) = sigmoid_grad(0.0);
    // sigmoid(0) = 0.5
    assert!(
        (val - 0.5).abs() < 1e-12,
        "sigmoid(0) = {val}, expected 0.5"
    );
    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25
    assert!(
        (grad - 0.25).abs() < 1e-12,
        "sigmoid'(0) = {grad}, expected 0.25"
    );
}

#[nabla_grad]
#[allow(dead_code)]
fn poly(x: f64) -> f64 {
    x * x + 2.0 * x
}


#[test]
fn nabla_grad_chain() {
    let (val, grad) = poly_grad(3.0);
    // poly(3) = 9 + 6 = 15
    assert!((val - 15.0).abs() < 1e-12, "poly(3) = {val}, expected 15");
    // poly'(x) = 2x + 2, poly'(3) = 8
    assert!((grad - 8.0).abs() < 1e-12, "poly'(3) = {grad}, expected 8");
}

// wgpu register-tile software MMA shader generation


#[test]
fn gradient_prep_x_squared() {
    // f(x) = sum(x^2), grad = 2x
    let f = |x: &Variable<f64, Cpu>| x.emul(x).sum_all_var();
    let x: Tensor<f64> = mat![[2.0_f64, 3.0]];
    let prep = gradient_prep(&f, &x);
    let g = gradient(&f, &x, &prep).expect("gradient returned None");
    assert!(approx_eq(g.get(0, 0), 4.0));
    assert!(approx_eq(g.get(0, 1), 6.0));
}


#[test]
fn grad_single_use() {
    let x: Tensor<f64> = mat![[3.0_f64, 4.0]];
    let g =
        grad(|xv: &Variable<f64, Cpu>| xv.emul(xv).sum_all_var(), &x).expect("grad returned None");
    assert!(approx_eq(g.get(0, 0), 6.0));
    assert!(approx_eq(g.get(0, 1), 8.0));
}



#[test]
fn symmetric_eigen() {
    let a: Tensor<f64, Cpu> = Tensor::from_fn(2, 2, |i, j| [[4.0_f64, 2.0], [2.0, 3.0]][i][j]);
    let sym = Symmetric::new(a, nabla::linalg::Side::Lower).expect("Symmetric::new failed");
    let evals = sym.eigenvalues().expect("eigenvalues failed");
    assert_eq!(evals.len(), 2);
    assert!(evals[0] > 0.0);
    assert!(evals[1] > 0.0);
}


#[test]
fn expm_diagonal() {
    // For diagonal A = diag(1.0, 2.0), exp(A) = diag(e, e^2)
    let mut a = Tensor::<f64>::zeros(2, 2);
    a[(0, 0)] = 1.0;
    a[(1, 1)] = 2.0;
    let e = expm(&a).expect("expm failed");
    assert!(
        (e[(0, 0)] - std::f64::consts::E).abs() < 1e-4,
        "expm diag (0,0): got {}, expected {}",
        e[(0, 0)],
        std::f64::consts::E
    );
    assert!(
        (e[(1, 1)] - std::f64::consts::E.powi(2)).abs() < 1e-4,
        "expm diag (1,1): got {}, expected {}",
        e[(1, 1)],
        std::f64::consts::E.powi(2)
    );
    assert!(e[(0, 1)].abs() < 1e-6);
    assert!(e[(1, 0)].abs() < 1e-6);
}


#[test]
fn static_matrix_matmul_shape() {
    let a: StaticMatrix<f64, 2, 3> = StaticMatrix::from_fn(|r, c| (r * 3 + c) as f64);
    let b: StaticMatrix<f64, 3, 2> = StaticMatrix::from_fn(|r, c| (r * 2 + c) as f64);
    let c: StaticMatrix<f64, 2, 2> = a * b;
    // row0 of a = [0,1,2], col0 of b = [0,2,4] => 0+2+8 = 10
    assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
}


#[test]
fn static_matrix_add_and_neg() {
    let a: StaticMatrix<f64, 2, 2> = StaticMatrix::from_fn(|r, c| (r + c) as f64);
    let b = a + a;
    assert!((b[(0, 1)] - 2.0).abs() < 1e-10);
    let neg = -&a;
    assert!((neg[(0, 1)] - (-1.0)).abs() < 1e-10);
}


#[test]
fn static_matrix_typed_matmul() {
    // 3x4 * 4x2 -> 3x2, compile-time shape checked
    let a = StaticMatrix::<f64, 3, 4>::from_fn(|r, c| (r * 4 + c) as f64);
    let b = StaticMatrix::<f64, 4, 2>::from_fn(|r, c| (r * 2 + c) as f64);
    let c: StaticMatrix<f64, 3, 2> = a * b;
    assert_eq!(c.shape(), (3, 2));
    // (0,0): row0=[0,1,2,3] dot col0=[0,2,4,6] = 0+2+8+18 = 28
    assert!((c.get(0, 0) - 28.0).abs() < 1e-10);
}


#[test]
fn static_matrix_typed_transpose() {
    let a = StaticMatrix::<f64, 3, 4>::from_fn(|r, c| (r * 4 + c) as f64);
    let at: StaticMatrix<f64, 4, 3> = a.t();
    assert_eq!(at.shape(), (4, 3));
    assert!((at.get(2, 1) - a.get(1, 2)).abs() < 1e-10);
}


#[test]
fn static_matrix_sub_ref() {
    let a = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c + 1) as f64);
    let b = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c) as f64);
    let d = a - b;
    // Every element should be 1.0
    for r in 0..2 {
        for c in 0..3 {
            assert!((d.get(r, c) - 1.0).abs() < 1e-10);
        }
    }
}


#[test]
fn solve_lstsq_overdetermined_exact() {
    // A*x = b where b is in range(A): x should be recovered exactly.
    let a = mat![[1.0_f64, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let b = mat![[1.0_f64], [2.0], [3.0]];
    let x = a.solve_lstsq(&b).expect("solve_lstsq failed");
    assert!(
        (x.get(0, 0) - 0.0).abs() < 1e-9,
        "intercept: expected 0, got {}",
        x.get(0, 0)
    );
    assert!(
        (x.get(1, 0) - 1.0).abs() < 1e-9,
        "slope: expected 1, got {}",
        x.get(1, 0)
    );
}


#[test]
fn solve_lstsq_overdetermined_approximate() {
    // A*x ≈ b where b is NOT in range(A): least-squares fit.
    let a = mat![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let b = mat![[1.0_f64], [2.0], [4.0]];
    // Normal equations: A^T A x = A^T b
    // A^T A = [[2,1],[1,2]], A^T b = [[5],[6]]
    // x = [[4/3],[7/3]]
    let x = a.solve_lstsq(&b).expect("solve_lstsq failed");
    assert!(
        (x.get(0, 0) - 4.0 / 3.0).abs() < 1e-9,
        "x0: expected {}, got {}",
        4.0 / 3.0,
        x.get(0, 0)
    );
    assert!(
        (x.get(1, 0) - 7.0 / 3.0).abs() < 1e-9,
        "x1: expected {}, got {}",
        7.0 / 3.0,
        x.get(1, 0)
    );
}


#[test]
fn svd_tall_3x2_reconstruction() {
    let a = mat![[3.0_f64, 2.0], [2.0, 3.0], [1.0, 1.0]];
    let svd = a.svd().expect("SVD of 3x2 failed");
    let s = svd.s();
    let u = svd.u();
    let vt = svd.vt();
    assert_eq!(s.len(), 2);
    let (m, n) = a.shape();
    let recon = Tensor::from_fn(m, n, |i, j| {
        (0..s.len())
            .map(|r| u.get(i, r) * s[r] * vt.get(r, j))
            .sum::<f64>()
    });
    let err = (&a - &recon).abs().sum_all();
    assert!(err < 1e-12, "reconstruction error: {err}");
}


#[test]
fn svd_rank_deficient() {
    // Rank-1 matrix: outer product
    let a = mat![[1.0_f64, 2.0], [2.0, 4.0]];
    let svd = a.svd().expect("SVD of rank-deficient failed");
    let s = svd.s();
    assert_eq!(s.len(), 2);
    assert!(s[0] > 1e-10, "s[0] should be nonzero: {}", s[0]);
    assert!(s[1] < 1e-10, "s[1] should be ~0: {}", s[1]);
}


#[test]
fn svd_singular_values_descending() {
    let a = mat![
        [4.0_f64, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0],
        [0.0, 1.0, 0.0]
    ];
    let svd = a.svd().expect("SVD failed");
    let s = svd.s();
    for w in s.windows(2) {
        assert!(w[0] >= w[1], "not descending: {} < {}", w[0], w[1]);
    }
}


#[test]
fn svd_reconstruct_rank_1_reduces_error() {
    let a = mat![
        [1.0_f64, 2.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ];
    let svd = a.svd().expect("SVD failed");
    let rank1 = svd.reconstruct_rank(1);
    let rank2 = svd.reconstruct_rank(2);
    assert_eq!(rank1.shape(), a.shape());
    // rank2 error < rank1 error (Eckart-Young)
    let e1 = (&a - &rank1).norm();
    let e2 = (&a - &rank2).norm();
    assert!(e2 < e1, "rank2 error ({e2}) should be < rank1 error ({e1})");
}


#[test]
fn norm_3_4_5_triangle() {
    let a: Tensor<f64> = mat![[3.0_f64, 4.0]];
    assert!((a.norm() - 5.0).abs() < 1e-10, "norm={}", a.norm());
    assert!(
        (a.norm_sq() - 25.0).abs() < 1e-10,
        "norm_sq={}",
        a.norm_sq()
    );
}


#[test]
fn norm_matrix_frobenius() {
    let a: Tensor<f64> = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    // Frobenius: sqrt(1+4+9+16) = sqrt(30)
    let expected = 30.0_f64.sqrt();
    assert!((a.norm() - expected).abs() < 1e-10, "norm={}", a.norm());
}


#[test]
fn static_matrix_outer_product() {
    let u: [f64; 3] = [1.0, 2.0, 3.0];
    let v: [f64; 2] = [4.0, 5.0];
    let m: StaticMatrix<f64, 3, 2> = StaticMatrix::outer(&u, &v);
    assert_eq!(m.shape(), (3, 2));
    // u[i] * v[j]
    assert!(approx_eq(m.get(0, 0), 4.0));
    assert!(approx_eq(m.get(0, 1), 5.0));
    assert!(approx_eq(m.get(1, 0), 8.0));
    assert!(approx_eq(m.get(1, 1), 10.0));
    assert!(approx_eq(m.get(2, 0), 12.0));
    assert!(approx_eq(m.get(2, 1), 15.0));
}


#[test]
fn static_matrix_data_access() {
    let m: StaticMatrix<f64, 2, 2> = StaticMatrix::from_fn(|r, c| (r * 2 + c + 1) as f64);
    let d = m.data();
    assert!(approx_eq(d[0][0], 1.0));
    assert!(approx_eq(d[0][1], 2.0));
    assert!(approx_eq(d[1][0], 3.0));
    assert!(approx_eq(d[1][1], 4.0));
}


#[test]
fn linear_layout_identity() {
    let id = LinearLayout16::identity();
    for v in 0..16u64 {
        assert_eq!(id.apply(v), v, "identity failed for v={v}");
    }
}


#[test]
fn linear_layout_swizzle_no_conflict() {
    let sw = LinearLayout16::swizzle_for_tile(16, 16, 32);
    // For each row, the 16 column addresses must map to distinct bank slots
    for row in 0..16u64 {
        let mut banks = std::collections::HashSet::new();
        for col in 0..16u64 {
            let addr = (row << 4) | col;
            let swizzled = sw.apply(addr);
            let bank = swizzled & 0x1F; // low 5 bits = bank index (mod 32)
            banks.insert(bank);
        }
        assert_eq!(banks.len(), 16, "bank conflict in row {row}");
    }
}


#[test]
fn linear_layout_compose() {
    let a = LinearLayout16::swizzle_for_tile(16, 16, 32);
    let b = LinearLayout16::identity();
    let ab = a.compose(&b);
    // compose(A, identity) == A
    for v in 0..16u64 {
        assert_eq!(ab.apply(v), a.apply(v), "compose(A,I) != A for v={v}");
    }
    // compose(A, B).apply(v) == A.apply(B.apply(v))
    let ba = b.compose(&a);
    for v in 0..32u64 {
        assert_eq!(
            ba.apply(v),
            b.apply(a.apply(v)),
            "compose(B,A).apply != B(A(v)) for v={v}"
        );
    }
}


    #[test]
    fn h_alias_matches_adjoint() {
        let a: Tensor<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
        let h = a.h();
        let adj = a.adjoint();
        assert_approx_grid(&h, &adj, 1e-12);

        let s = StaticMatrix::<f64, 2, 3>::from_fn(|r, c| (r * 3 + c + 1) as f64);
        let sh = s.h();
        let sadj = s.adjoint();
        let (rows, cols) = sh.shape();
        assert_eq!((rows, cols), sadj.shape());
        for r in 0..rows {
            for c in 0..cols {
                assert!(approx_eq(sh.get(r, c), sadj.get(r, c)));
            }
        }
    }


    #[test]
    fn linalg_short_aliases() -> Result<()> {
        let a: Tensor<f64> = mat![[4.0, 1.0], [1.0, 3.0]];
        let b: Tensor<f64> = mat![[1.0], [2.0]];

        let _ = a.lu()?;
        let _ = a.chol()?;
        let _ = a.ldl()?;

        let x_short = a.lstsq(&b)?;
        let x_long = a.solve_lstsq(&b)?;
        assert_approx_grid(&x_short, &x_long, 1e-10);

        let sv_short = a.svdvals()?;
        let sv_long = a.singular_values()?;
        assert_eq!(sv_short.len(), sv_long.len());
        for i in 0..sv_short.len() {
            assert!(approx_eq(sv_short[i], sv_long[i]));
        }

        let eig = a.sym(Side::Lower)?.eigh()?;
        assert_eq!(eig.values().len(), 2);
        Ok(())
    }
