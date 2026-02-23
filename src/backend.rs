// backend.rs — Sealed Backend trait + Cpu implementation backed by faer 0.24.
//
// API adaptations vs. spec (faer 0.24 differences):
//   - No `faer::Entity`; `faer::ComplexField` is the sole trait bound.
//   - `Mat::from_fn` closure receives `usize` indices for dynamic matrices.
//   - Element access uses `mat.get(row, col)` returning `&T`.
//   - matmul: `faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, one, Par::Seq)`.
//   - Scalar multiply: `mat * faer::Scale(s)`.
//   - Transpose collects via `Mat::from_fn` over the transposed view.

use faer::{Accum, Mat, Par, Scale, linalg::matmul::matmul};

use crate::scalar::Scalar;

mod private {
    pub trait Sealed {}
}

/// Compute backend abstraction (sealed — not implementable outside this crate).
pub trait Backend: private::Sealed + Send + Sync + 'static {
    /// Owned storage for a 2-D matrix of element type `T`.
    type Storage<T: Scalar>: Send + Sync;

    /// Allocate a zero-filled matrix.
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> Self::Storage<T>;

    /// Allocate a matrix and fill it by calling `f(row, col)`.
    fn from_fn<T: Scalar>(
        nrows: usize,
        ncols: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> Self::Storage<T>;

    /// Row count of `storage`.
    fn nrows<T: Scalar>(storage: &Self::Storage<T>) -> usize;

    /// Column count of `storage`.
    fn ncols<T: Scalar>(storage: &Self::Storage<T>) -> usize;

    /// Read element at `(row, col)`.
    fn get<T: Scalar>(storage: &Self::Storage<T>, row: usize, col: usize) -> T;

    /// Write element at `(row, col)`.
    fn set<T: Scalar>(storage: &mut Self::Storage<T>, row: usize, col: usize, val: T);

    /// Compute `out = a * b`, overwriting `out`.
    fn matmul_into<T: Scalar>(
        out: &mut Self::Storage<T>,
        a: &Self::Storage<T>,
        b: &Self::Storage<T>,
    );

    /// Element-wise addition.
    fn add<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise subtraction.
    fn sub<T: Scalar>(a: &Self::Storage<T>, b: &Self::Storage<T>) -> Self::Storage<T>;

    /// Element-wise negation.
    fn neg<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Transpose: result has shape `(ncols(a), nrows(a))`.
    fn transpose<T: Scalar>(a: &Self::Storage<T>) -> Self::Storage<T>;

    /// Scalar multiply: every element of `a` multiplied by `s`.
    fn scale<T: Scalar>(a: &Self::Storage<T>, s: T) -> Self::Storage<T>;

    /// Clone storage.
    fn clone_storage<T: Scalar>(storage: &Self::Storage<T>) -> Self::Storage<T>;
}

/// CPU backend using faer's native SIMD kernels.
pub struct Cpu;

impl private::Sealed for Cpu {}

impl Backend for Cpu {
    type Storage<T: Scalar> = Mat<T>;

    #[inline]
    fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> Mat<T> {
        Mat::zeros(nrows, ncols)
    }

    #[inline]
    fn from_fn<T: Scalar>(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> Mat<T> {
        Mat::from_fn(nrows, ncols, f)
    }

    #[inline]
    fn nrows<T: Scalar>(storage: &Mat<T>) -> usize {
        storage.nrows()
    }

    #[inline]
    fn ncols<T: Scalar>(storage: &Mat<T>) -> usize {
        storage.ncols()
    }

    #[inline]
    fn get<T: Scalar>(storage: &Mat<T>, row: usize, col: usize) -> T {
        *storage.get(row, col)
    }

    #[inline]
    fn set<T: Scalar>(storage: &mut Mat<T>, row: usize, col: usize, val: T) {
        *storage.get_mut(row, col) = val;
    }

    #[inline]
    fn matmul_into<T: Scalar>(out: &mut Mat<T>, a: &Mat<T>, b: &Mat<T>) {
        matmul(out, Accum::Replace, a, b, T::one_impl(), Par::Seq);
    }

    #[inline]
    fn add<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
        a + b
    }

    #[inline]
    fn sub<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
        a - b
    }

    #[inline]
    fn neg<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        -a
    }

    #[inline]
    fn transpose<T: Scalar>(a: &Mat<T>) -> Mat<T> {
        let t = a.as_ref().transpose();
        Mat::from_fn(t.nrows(), t.ncols(), |r, c| *t.get(r, c))
    }

    #[inline]
    fn scale<T: Scalar>(a: &Mat<T>, s: T) -> Mat<T> {
        a * Scale(s)
    }

    #[inline]
    fn clone_storage<T: Scalar>(storage: &Mat<T>) -> Mat<T> {
        storage.clone()
    }
}

// Hip still delegates to Cpu; Cuda and Wgpu use real GPU storage via gpu.rs.
#[cfg(feature = "hip")]
type MatStorage<T> = Mat<T>;

#[cfg(feature = "cuda")]
/// CUDA backend — uses cubecl-cuda kernels for f32/f64, CPU fallback for c32/c64.
pub struct Cuda;

#[cfg(feature = "cuda")]
impl private::Sealed for Cuda {}

#[cfg(feature = "wgpu")]
/// wgpu backend — uses cubecl-wgpu kernels for f32/f64, CPU fallback for c32/c64.
pub struct Wgpu;

#[cfg(feature = "wgpu")]
impl private::Sealed for Wgpu {}

#[cfg(feature = "hip")]
/// HIP backend stub — currently delegates all operations to Cpu.
pub struct Hip;

#[cfg(feature = "hip")]
impl private::Sealed for Hip {}

// Macro is only used for Hip (delegates everything to Cpu).
#[cfg(feature = "hip")]
macro_rules! delegate_backend {
    ($backend:ty) => {
        impl Backend for $backend {
            type Storage<T: Scalar> = MatStorage<T>;

            #[inline]
            fn zeros<T: Scalar>(nrows: usize, ncols: usize) -> MatStorage<T> {
                Cpu::zeros(nrows, ncols)
            }

            #[inline]
            fn from_fn<T: Scalar>(
                nrows: usize,
                ncols: usize,
                f: impl FnMut(usize, usize) -> T,
            ) -> MatStorage<T> {
                Cpu::from_fn(nrows, ncols, f)
            }

            #[inline]
            fn nrows<T: Scalar>(storage: &MatStorage<T>) -> usize {
                Cpu::nrows(storage)
            }

            #[inline]
            fn ncols<T: Scalar>(storage: &MatStorage<T>) -> usize {
                Cpu::ncols(storage)
            }

            #[inline]
            fn get<T: Scalar>(storage: &MatStorage<T>, row: usize, col: usize) -> T {
                Cpu::get(storage, row, col)
            }

            #[inline]
            fn set<T: Scalar>(storage: &mut MatStorage<T>, row: usize, col: usize, val: T) {
                Cpu::set(storage, row, col, val)
            }

            #[inline]
            fn matmul_into<T: Scalar>(
                out: &mut MatStorage<T>,
                a: &MatStorage<T>,
                b: &MatStorage<T>,
            ) {
                Cpu::matmul_into(out, a, b)
            }

            #[inline]
            fn add<T: Scalar>(a: &MatStorage<T>, b: &MatStorage<T>) -> MatStorage<T> {
                Cpu::add(a, b)
            }

            #[inline]
            fn sub<T: Scalar>(a: &MatStorage<T>, b: &MatStorage<T>) -> MatStorage<T> {
                Cpu::sub(a, b)
            }

            #[inline]
            fn neg<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::neg(a)
            }

            #[inline]
            fn transpose<T: Scalar>(a: &MatStorage<T>) -> MatStorage<T> {
                Cpu::transpose(a)
            }

            #[inline]
            fn scale<T: Scalar>(a: &MatStorage<T>, s: T) -> MatStorage<T> {
                Cpu::scale(a, s)
            }

            #[inline]
            fn clone_storage<T: Scalar>(storage: &MatStorage<T>) -> MatStorage<T> {
                Cpu::clone_storage(storage)
            }
        }
    };
}

#[cfg(feature = "hip")]
delegate_backend!(Hip);

#[allow(unused_macros)]
macro_rules! impl_gpu_backend {
    ($Backend:ty, $Runtime:path) => {
        impl Backend for $Backend {
            type Storage<T: Scalar> = GpuStorage<T>;
            #[inline]
            fn zeros<T: Scalar>(r: usize, c: usize) -> GpuStorage<T> {
                GpuStorage::zeros(r, c)
            }
            #[inline]
            fn from_fn<T: Scalar>(
                r: usize,
                c: usize,
                f: impl FnMut(usize, usize) -> T,
            ) -> GpuStorage<T> {
                GpuStorage::from_fn(r, c, f)
            }
            #[inline]
            fn nrows<T: Scalar>(s: &GpuStorage<T>) -> usize {
                s.nrows
            }
            #[inline]
            fn ncols<T: Scalar>(s: &GpuStorage<T>) -> usize {
                s.ncols
            }
            #[inline]
            fn get<T: Scalar>(s: &GpuStorage<T>, r: usize, c: usize) -> T {
                s.get(r, c)
            }
            #[inline]
            fn set<T: Scalar>(s: &mut GpuStorage<T>, r: usize, c: usize, v: T) {
                s.set(r, c, v)
            }
            #[inline]
            fn matmul_into<T: Scalar>(o: &mut GpuStorage<T>, a: &GpuStorage<T>, b: &GpuStorage<T>) {
                gpu_matmul::<T, $Runtime>(o, a, b)
            }
            #[inline]
            fn add<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
                gpu_add::<T, $Runtime>(a, b)
            }
            #[inline]
            fn sub<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
                gpu_sub::<T, $Runtime>(a, b)
            }
            #[inline]
            fn neg<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
                gpu_neg::<T, $Runtime>(a)
            }
            #[inline]
            fn transpose<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
                gpu_transpose::<T, $Runtime>(a)
            }
            #[inline]
            fn scale<T: Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
                gpu_scale::<T, $Runtime>(a, s)
            }
            #[inline]
            fn clone_storage<T: Scalar>(s: &GpuStorage<T>) -> GpuStorage<T> {
                GpuStorage {
                    nrows: s.nrows,
                    ncols: s.ncols,
                    data: s.data.clone(),
                }
            }
        }
    };
}

#[cfg(feature = "cuda")]
impl_gpu_backend!(Cuda, cubecl_cuda::CudaRuntime);
#[cfg(feature = "wgpu")]
impl_gpu_backend!(Wgpu, cubecl_wgpu::WgpuRuntime);

#[cfg(feature = "cuda")]
/// Default backend: CUDA (highest priority when enabled).
pub type DefaultBackend = Cuda;

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "hip")))]
/// Default backend: wgpu (used when cuda is not enabled).
pub type DefaultBackend = Wgpu;

#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
/// Default backend: HIP (used when cuda and wgpu are not enabled).
pub type DefaultBackend = Hip;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "hip")))]
/// Default backend: CPU (fallback when no GPU feature is enabled).
pub type DefaultBackend = Cpu;

// GPU implementation (formerly gpu.rs): compiled only when cuda or wgpu feature is enabled.
// The `use cubecl_core as cubecl` alias is required so that #[cube(launch)]
// macro-generated paths like `cubecl::prelude::*` resolve correctly.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use cubecl::prelude::*;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use cubecl_core as cubecl;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::any::TypeId;

/// Row-major, CPU-mirrored storage for a GPU-backed matrix.
///
/// Data lives in a `Vec<T>` on the host.  When a GPU operation is requested
/// the slice is uploaded to device memory, the kernel is launched, and the
/// result is read back synchronously.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) data: Vec<T>,
}

// SAFETY: T: Send + Sync (via Scalar bound) and Vec<T> is Send + Sync.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
unsafe impl<T: Scalar> Send for GpuStorage<T> {}
#[cfg(any(feature = "cuda", feature = "wgpu"))]
unsafe impl<T: Scalar> Sync for GpuStorage<T> {}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
impl<T: Scalar> GpuStorage<T> {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: vec![T::zero_impl(); nrows * ncols],
        }
    }

    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let data = (0..nrows * ncols)
            .map(|i| f(i / ncols, i % ncols))
            .collect();
        Self { nrows, ncols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> T {
        self.data[r * self.ncols + c]
    }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: T) {
        self.data[r * self.ncols + c] = v;
    }
}

// SAFETY: cast_slice/cast_vec — sound only when TypeId::of::<T>() == TypeId::of::<U>() (same layout).
#[cfg(any(feature = "cuda", feature = "wgpu"))]
unsafe fn cast_slice<T, U>(s: &[T]) -> &[U] {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<U>(), s.len()) }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
unsafe fn cast_vec<T, U>(mut v: Vec<T>) -> Vec<U> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    let (ptr, len, cap) = (v.as_mut_ptr().cast::<U>(), v.len(), v.capacity());
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn elementwise_add_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] + rhs[i];
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn elementwise_sub_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] - rhs[i];
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn elementwise_neg_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = -input[i];
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn elementwise_scale_f32_kernel(input: &Array<f32>, out: &mut Array<f32>, scalar: f32) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = input[i] * scalar;
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn elementwise_scale_f64_kernel(input: &Array<f64>, out: &mut Array<f64>, scalar: f64) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = input[i] * scalar;
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn transpose_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>, rows: usize, cols: usize) {
    let i = ABSOLUTE_POS;
    if i < rows * cols {
        let row = i / cols;
        let col = i % cols;
        out[col * rows + row] = input[i];
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch)]
fn matmul_naive_kernel<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    out: &mut Array<F>,
    k_dim: usize,
    n_dim: usize,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let row = i / n_dim;
        let col = i % n_dim;
        let mut acc = F::new(0.0_f32);
        for k in 0..k_dim {
            acc += a[row * k_dim + k] * b[k * n_dim + col];
        }
        out[i] = acc;
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn cube_count(n: usize) -> CubeCount {
    CubeCount::Static(((n + 255) / 256) as u32, 1, 1)
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
macro_rules! run_elementwise_binary {
    ($kernel:path, $fn_name:ident) => {
        fn $fn_name<E: Float + CubeElement, R: Runtime>(a: &[E], b: &[E]) -> Vec<E>
        where
            R::Device: Default,
        {
            let client = R::client(&R::Device::default());
            let n = a.len();
            let h_a = client.create_from_slice(E::as_bytes(a));
            let h_b = client.create_from_slice(E::as_bytes(b));
            let h_out = client.empty(n * std::mem::size_of::<E>());
            if let Err(e) = $kernel::<E, R>(
                &client,
                cube_count(n),
                CubeDim::new_1d(256),
                unsafe { ArrayArg::from_raw_parts::<E>(&h_a, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<E>(&h_b, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<E>(&h_out, n, 1) },
            ) {
                panic!("GPU kernel launch failed: {e}");
            }
            E::from_bytes(&client.read_one(h_out)).to_vec()
        }
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
macro_rules! run_elementwise_unary {
    ($kernel:path, $fn_name:ident) => {
        fn $fn_name<E: Float + CubeElement, R: Runtime>(a: &[E]) -> Vec<E>
        where
            R::Device: Default,
        {
            let client = R::client(&R::Device::default());
            let n = a.len();
            let h_in = client.create_from_slice(E::as_bytes(a));
            let h_out = client.empty(n * std::mem::size_of::<E>());
            if let Err(e) = $kernel::<E, R>(
                &client,
                cube_count(n),
                CubeDim::new_1d(256),
                unsafe { ArrayArg::from_raw_parts::<E>(&h_in, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<E>(&h_out, n, 1) },
            ) {
                panic!("GPU kernel launch failed: {e}");
            }
            E::from_bytes(&client.read_one(h_out)).to_vec()
        }
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
run_elementwise_binary!(elementwise_add_kernel::launch, run_add);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
run_elementwise_binary!(elementwise_sub_kernel::launch, run_sub);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
run_elementwise_unary!(elementwise_neg_kernel::launch, run_neg);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
macro_rules! run_scale {
    ($ty:ty, $kernel:path, $fn_name:ident) => {
        fn $fn_name<R: Runtime>(a: &[$ty], s: $ty) -> Vec<$ty>
        where
            R::Device: Default,
        {
            let client = R::client(&R::Device::default());
            let n = a.len();
            let h_in = client.create_from_slice(<$ty>::as_bytes(a));
            let h_out = client.empty(n * std::mem::size_of::<$ty>());
            if let Err(e) = $kernel::<R>(
                &client,
                cube_count(n),
                CubeDim::new_1d(256),
                unsafe { ArrayArg::from_raw_parts::<$ty>(&h_in, n, 1) },
                unsafe { ArrayArg::from_raw_parts::<$ty>(&h_out, n, 1) },
                ScalarArg::new(s),
            ) {
                panic!("GPU kernel launch failed: {e}");
            }
            <$ty>::from_bytes(&client.read_one(h_out)).to_vec()
        }
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
run_scale!(f32, elementwise_scale_f32_kernel::launch, run_scale_f32);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
run_scale!(f64, elementwise_scale_f64_kernel::launch, run_scale_f64);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn run_transpose<E: Float + CubeElement, R: Runtime>(a: &[E], rows: usize, cols: usize) -> Vec<E>
where
    R::Device: Default,
{
    let client = R::client(&R::Device::default());
    let n = rows * cols;
    let h_in = client.create_from_slice(E::as_bytes(a));
    let h_out = client.empty(n * std::mem::size_of::<E>());
    if let Err(e) = transpose_kernel::launch::<E, R>(
        &client,
        cube_count(n),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts::<E>(&h_in, n, 1) },
        unsafe { ArrayArg::from_raw_parts::<E>(&h_out, n, 1) },
        ScalarArg::new(rows),
        ScalarArg::new(cols),
    ) {
        panic!("GPU kernel launch failed: {e}");
    }
    E::from_bytes(&client.read_one(h_out)).to_vec()
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn run_matmul<E: Float + CubeElement, R: Runtime>(
    a: &[E],
    b: &[E],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<E>
where
    R::Device: Default,
{
    let client = R::client(&R::Device::default());
    let out_n = m * n;
    let h_a = client.create_from_slice(E::as_bytes(a));
    let h_b = client.create_from_slice(E::as_bytes(b));
    let h_out = client.empty(out_n * std::mem::size_of::<E>());
    if let Err(e) = matmul_naive_kernel::launch::<E, R>(
        &client,
        cube_count(out_n),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts::<E>(&h_a, m * k, 1) },
        unsafe { ArrayArg::from_raw_parts::<E>(&h_b, k * n, 1) },
        unsafe { ArrayArg::from_raw_parts::<E>(&h_out, out_n, 1) },
        ScalarArg::new(k),
        ScalarArg::new(n),
    ) {
        panic!("GPU kernel launch failed: {e}");
    }
    E::from_bytes(&client.read_one(h_out)).to_vec()
}

// Dispatch helper: run f32 or f64 GPU path, fall back to CPU for other types.
// SAFETY: callers guarantee TypeId::of::<T>() == TypeId::of::<f32/f64>() before cast.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn typed_run<T: Scalar>(
    data: &[T],
    f32_fn: impl FnOnce(&[f32]) -> Vec<f32>,
    f64_fn: impl FnOnce(&[f64]) -> Vec<f64>,
    fallback: impl FnOnce(&[T]) -> Vec<T>,
) -> Vec<T> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe { cast_vec(f32_fn(cast_slice(data))) }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe { cast_vec(f64_fn(cast_slice(data))) }
    } else {
        fallback(data)
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn typed_run2<T: Scalar>(
    a: &[T],
    b: &[T],
    f32_fn: impl FnOnce(&[f32], &[f32]) -> Vec<f32>,
    f64_fn: impl FnOnce(&[f64], &[f64]) -> Vec<f64>,
    fallback: impl FnOnce(&[T], &[T]) -> Vec<T>,
) -> Vec<T> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe { cast_vec(f32_fn(cast_slice(a), cast_slice(b))) }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe { cast_vec(f64_fn(cast_slice(a), cast_slice(b))) }
    } else {
        fallback(a, b)
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
macro_rules! gpu_elementwise_binary {
    ($run_f32:expr, $run_f64:expr, $fallback:expr, $fn_name:ident) => {
        pub(crate) fn $fn_name<T: Scalar, R: Runtime>(
            a: &GpuStorage<T>,
            b: &GpuStorage<T>,
        ) -> GpuStorage<T>
        where
            R::Device: Default,
        {
            GpuStorage {
                nrows: a.nrows,
                ncols: a.ncols,
                data: typed_run2(&a.data, &b.data, $run_f32, $run_f64, $fallback),
            }
        }
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
macro_rules! gpu_elementwise_unary {
    ($run_f32:expr, $run_f64:expr, $fallback:expr, $fn_name:ident) => {
        pub(crate) fn $fn_name<T: Scalar, R: Runtime>(a: &GpuStorage<T>) -> GpuStorage<T>
        where
            R::Device: Default,
        {
            GpuStorage {
                nrows: a.nrows,
                ncols: a.ncols,
                data: typed_run(&a.data, $run_f32, $run_f64, $fallback),
            }
        }
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
gpu_elementwise_binary!(
    run_add::<f32, R>,
    run_add::<f64, R>,
    |a, b| a.iter().zip(b).map(|(&x, &y)| x + y).collect(),
    gpu_add
);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
gpu_elementwise_binary!(
    run_sub::<f32, R>,
    run_sub::<f64, R>,
    |a, b| a.iter().zip(b).map(|(&x, &y)| x - y).collect(),
    gpu_sub
);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
gpu_elementwise_unary!(
    run_neg::<f32, R>,
    run_neg::<f64, R>,
    |a| a.iter().map(|&x| -x).collect(),
    gpu_neg
);

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn gpu_scale<T: Scalar, R: Runtime>(a: &GpuStorage<T>, s: T) -> GpuStorage<T>
where
    R::Device: Default,
{
    let data = if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: TypeId proves T == f32.
        unsafe {
            let s_f32 = *(&s as *const T as *const f32);
            cast_vec(run_scale_f32::<R>(cast_slice(&a.data), s_f32))
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            let s_f64 = *(&s as *const T as *const f64);
            cast_vec(run_scale_f64::<R>(cast_slice(&a.data), s_f64))
        }
    } else {
        a.data.iter().map(|&x| x * s).collect()
    };
    GpuStorage {
        nrows: a.nrows,
        ncols: a.ncols,
        data,
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn gpu_transpose<T: Scalar, R: Runtime>(a: &GpuStorage<T>) -> GpuStorage<T>
where
    R::Device: Default,
{
    let (rows, cols) = (a.nrows, a.ncols);
    GpuStorage {
        nrows: cols,
        ncols: rows,
        data: typed_run(
            &a.data,
            |d| run_transpose::<f32, R>(d, rows, cols),
            |d| run_transpose::<f64, R>(d, rows, cols),
            |_| {
                let mut buf = vec![T::zero_impl(); rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        buf[c * rows + r] = a.data[r * cols + c];
                    }
                }
                buf
            },
        ),
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn gpu_matmul<T: Scalar, R: Runtime>(
    out: &mut GpuStorage<T>,
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) where
    R::Device: Default,
{
    let (m, k, n) = (a.nrows, a.ncols, b.ncols);
    out.data = typed_run2(
        &a.data,
        &b.data,
        |a, b| run_matmul::<f32, R>(a, b, m, k, n),
        |a, b| run_matmul::<f64, R>(a, b, m, k, n),
        |a, b| {
            let mut buf = vec![T::zero_impl(); m * n];
            for i in 0..m {
                for j in 0..n {
                    buf[i * n + j] =
                        (0..k).fold(T::zero_impl(), |s, l| s + a[i * k + l] * b[l * n + j]);
                }
            }
            buf
        },
    );
    (out.nrows, out.ncols) = (m, n);
}
