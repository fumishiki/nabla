// gpu.rs — Handle-based GPU storage and CubeCL kernel dispatch.
//
// Design:
//   - GpuStorage<T> owns a cubecl_runtime Handle (RAII GPU memory, Arc-backed Clone).
//   - GPU ops dispatch via TypeId: f32/f64 → GPU kernels, c32/c64 → CPU fallback.
//   - Byte conversion via raw slice reinterpretation (no bytemuck dependency).
//   - get/set use a lazy Mutex<Option<Vec<T>>> host_cache (readback on first access).

// CubeCL proc-macro (#[cube]) generates code that appends format!(..) to String;
// suppress the resulting false-positive lint for this entire module.
#![allow(clippy::format_push_string)]

use cubecl::prelude::*;
use cubecl_core as cubecl;

use std::sync::Mutex;

use crate::error::Error;
use crate::scalar::Scalar;

// ── Byte conversion helpers ───────────────────────────────────────────────────

// SAFETY: all Scalar types (f32/f64/c32/c64) are POD with stable layout.
unsafe fn scalar_to_bytes<T: Scalar>(data: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data))
    }
}

// SAFETY: bytes originated from a [T] slice with correct alignment and length.
unsafe fn bytes_to_scalar<T: Scalar>(bytes: &[u8]) -> Vec<T> {
    unsafe {
        let len = bytes.len() / core::mem::size_of::<T>();
        core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len).to_vec()
    }
}

// ── GpuStorage ───────────────────────────────────────────────────────────────

/// Row-major GPU-backed matrix.
///
/// `handle` owns device memory (RAII via Arc-backed ref-count).
/// `host_cache` is populated lazily on the first `get` call and invalidated on `set`.
pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    handle: cubecl_runtime::server::Handle,
    host_cache: Mutex<Option<Vec<T>>>,
}

// SAFETY: cubecl_runtime::server::Handle is Send+Sync (Arc-backed). Mutex<Option<Vec<T>>> is
// Send+Sync when T: Send+Sync, which the Scalar bound guarantees.
unsafe impl<T: Scalar> Send for GpuStorage<T> {}
unsafe impl<T: Scalar> Sync for GpuStorage<T> {}

impl<T: Scalar> GpuStorage<T> {
    fn from_handle(nrows: usize, ncols: usize, handle: cubecl_runtime::server::Handle) -> Self {
        Self { nrows, ncols, handle, host_cache: Mutex::new(None) }
    }

    fn from_handle_cached(
        nrows: usize,
        ncols: usize,
        handle: cubecl_runtime::server::Handle,
        cache: Vec<T>,
    ) -> Self {
        Self { nrows, ncols, handle, host_cache: Mutex::new(Some(cache)) }
    }

    fn upload<R: Runtime>(nrows: usize, ncols: usize, data: Vec<T>) -> Self
    where
        R::Device: Default,
    {
        let client = R::client(&R::Device::default());
        // SAFETY: T is a Scalar POD type; reinterpreted as bytes for GPU upload.
        let handle = client.create_from_slice(unsafe { scalar_to_bytes(&data) });
        Self::from_handle_cached(nrows, ncols, handle, data)
    }

    fn download<R: Runtime>(&self) -> Vec<T>
    where
        R::Device: Default,
    {
        let guard = self.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(ref v) = *guard {
            return v.clone();
        }
        drop(guard);
        let client = R::client(&R::Device::default());
        let bytes = client.read_one(self.handle.clone());
        // SAFETY: bytes originated from a [T] slice via scalar_to_bytes.
        unsafe { bytes_to_scalar::<T>(&bytes) }
    }

    fn fill_cache<R: Runtime>(&self) -> std::sync::MutexGuard<'_, Option<Vec<T>>>
    where
        R::Device: Default,
    {
        let mut guard = self.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_none() {
            let client = R::client(&R::Device::default());
            let bytes = client.read_one(self.handle.clone());
            // SAFETY: bytes originated from a [T] slice via scalar_to_bytes.
            *guard = Some(unsafe { bytes_to_scalar::<T>(&bytes) });
        }
        guard
    }
}

// ── CubeCL kernels ────────────────────────────────────────────────────────────
// #[allow] attrs suppress false-positive lints from CubeCL proc-macro expansion.

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_add_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] + rhs[i];
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_sub_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] - rhs[i];
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_neg_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = -input[i];
    }
}

/// Unified scale kernel — scalar arg requires [`CubeElement`] bound.
#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_scale_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    out: &mut Array<F>,
    scalar: F,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = input[i] * scalar;
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn transpose_kernel<F: Float>(
    input: &Array<F>,
    out: &mut Array<F>,
    rows: usize,
    cols: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows * cols {
        let row = i / cols;
        let col = i % cols;
        out[col * rows + row] = input[i];
    }
}

#[allow(clippy::format_push_string, clippy::many_single_char_names)]
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

// ── Elementwise math kernels (unary) ─────────────────────────────────────────

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_exp_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Exp::exp(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_ln_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Log::ln(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_log1p_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Log1p::log1p(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_sin_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Sin::sin(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_cos_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Cos::cos(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_tanh_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Tanh::tanh(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_sqrt_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Sqrt::sqrt(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_abs_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Abs::abs(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_recip_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Recip::recip(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_erf_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Erf::erf(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_ceil_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Ceil::ceil(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_floor_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Floor::floor(input[i]);
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_round_kernel<F: Float>(input: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Round::round(input[i]);
    }
}

// powf: scalar power exponent requires CubeElement bound
#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_powf_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    out: &mut Array<F>,
    power: F,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = Powf::powf(input[i], power);
    }
}

// ── Elementwise math kernels (binary) ────────────────────────────────────────

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_mul_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] * rhs[i];
    }
}

#[allow(clippy::format_push_string)]
#[cube(launch)]
fn elementwise_div_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = lhs[i] / rhs[i];
    }
}

// ── cube_count ────────────────────────────────────────────────────────────────

#[inline]
fn cube_count(n: usize) -> CubeCount {
    #[allow(clippy::cast_possible_truncation)]
    CubeCount::Static(n.div_ceil(256) as u32, 1, 1)
}

// ── GPU kernel helpers ────────────────────────────────────────────────────────

fn gpu_binary_kernel<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_a: &cubecl_runtime::server::Handle,
    h_b: &cubecl_runtime::server::Handle,
    n: usize,
    is_sub: bool,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_a, h_b, h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let la = ArrayArg::<R>::from_raw_parts::<E>(h_a, n, 1);
        let lb = ArrayArg::<R>::from_raw_parts::<E>(h_b, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        let res = if is_sub {
            elementwise_sub_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), la, lb, lout)
        } else {
            elementwise_add_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), la, lb, lout)
        };
        res.map(|()| h_out).map_err(|e| Error::invalid(format!("GPU binary kernel failed: {e}")))
    }
}

fn gpu_neg_kernel<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_in: &cubecl_runtime::server::Handle,
    n: usize,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_in and h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let lin = ArrayArg::<R>::from_raw_parts::<E>(h_in, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        elementwise_neg_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), lin, lout)
            .map(|()| h_out)
            .map_err(|e| Error::invalid(format!("GPU neg kernel failed: {e}")))
    }
}

fn gpu_scale_kernel<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_in: &cubecl_runtime::server::Handle,
    n: usize,
    scalar: E,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_in and h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let lin = ArrayArg::<R>::from_raw_parts::<E>(h_in, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        elementwise_scale_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), lin, lout, ScalarArg::new(scalar))
            .map(|()| h_out)
            .map_err(|e| Error::invalid(format!("GPU scale kernel failed: {e}")))
    }
}

fn gpu_transpose_kernel<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_in: &cubecl_runtime::server::Handle,
    rows: usize,
    cols: usize,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let n = rows * cols;
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_in and h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let lin = ArrayArg::<R>::from_raw_parts::<E>(h_in, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        transpose_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), lin, lout, ScalarArg::new(rows), ScalarArg::new(cols))
            .map(|()| h_out)
            .map_err(|e| Error::invalid(format!("GPU transpose kernel failed: {e}")))
    }
}

fn gpu_matmul_kernel<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_a: &cubecl_runtime::server::Handle,
    h_b: &cubecl_runtime::server::Handle,
    m: usize,
    k: usize,
    n: usize,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let out_n = m * n;
    let h_out = client.empty(out_n * std::mem::size_of::<E>());
    // SAFETY: h_a, h_b are valid GPU allocations for m*k and k*n elements of type E.
    // h_out is valid for m*n elements of type E.
    unsafe {
        let la = ArrayArg::<R>::from_raw_parts::<E>(h_a, m * k, 1);
        let lb = ArrayArg::<R>::from_raw_parts::<E>(h_b, k * n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, out_n, 1);
        matmul_naive_kernel::launch::<E, R>(client, cube_count(out_n), CubeDim::new_1d(256), la, lb, lout, ScalarArg::new(k), ScalarArg::new(n))
            .map(|()| h_out)
            .map_err(|e| Error::invalid(format!("GPU matmul kernel failed: {e}")))
    }
}

// ── Unary math kernel helpers (macro to reduce boilerplate) ──────────────────

macro_rules! impl_gpu_unary_math_helper {
    ($name:ident, $kernel:ident) => {
        fn $name<E: Float + CubeElement, R: Runtime>(
            client: &ComputeClient<R>,
            h_in: &cubecl_runtime::server::Handle,
            n: usize,
        ) -> Result<cubecl_runtime::server::Handle, Error> {
            let h_out = client.empty(n * std::mem::size_of::<E>());
            // SAFETY: h_in and h_out are valid GPU allocations for exactly `n` elements of type E.
            unsafe {
                let lin = ArrayArg::<R>::from_raw_parts::<E>(h_in, n, 1);
                let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
                $kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), lin, lout)
                    .map(|()| h_out)
                    .map_err(|e| {
                        Error::invalid(format!("GPU {} failed: {e}", stringify!($name)))
                    })
            }
        }
    };
}

impl_gpu_unary_math_helper!(gpu_exp_helper, elementwise_exp_kernel);
impl_gpu_unary_math_helper!(gpu_ln_helper, elementwise_ln_kernel);
impl_gpu_unary_math_helper!(gpu_log1p_helper, elementwise_log1p_kernel);
impl_gpu_unary_math_helper!(gpu_sin_helper, elementwise_sin_kernel);
impl_gpu_unary_math_helper!(gpu_cos_helper, elementwise_cos_kernel);
impl_gpu_unary_math_helper!(gpu_tanh_helper, elementwise_tanh_kernel);
impl_gpu_unary_math_helper!(gpu_sqrt_helper, elementwise_sqrt_kernel);
impl_gpu_unary_math_helper!(gpu_abs_helper, elementwise_abs_kernel);
impl_gpu_unary_math_helper!(gpu_recip_helper, elementwise_recip_kernel);
impl_gpu_unary_math_helper!(gpu_erf_helper, elementwise_erf_kernel);
impl_gpu_unary_math_helper!(gpu_ceil_helper, elementwise_ceil_kernel);
impl_gpu_unary_math_helper!(gpu_floor_helper, elementwise_floor_kernel);
impl_gpu_unary_math_helper!(gpu_round_helper, elementwise_round_kernel);

// ── Binary math kernel helpers ────────────────────────────────────────────────

fn gpu_mul_elem_helper<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_a: &cubecl_runtime::server::Handle,
    h_b: &cubecl_runtime::server::Handle,
    n: usize,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_a, h_b, h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let la = ArrayArg::<R>::from_raw_parts::<E>(h_a, n, 1);
        let lb = ArrayArg::<R>::from_raw_parts::<E>(h_b, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        elementwise_mul_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), la, lb, lout)
            .map(|()| h_out)
            .map_err(|e| Error::invalid(format!("GPU mul_elem failed: {e}")))
    }
}

fn gpu_div_elem_helper<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_a: &cubecl_runtime::server::Handle,
    h_b: &cubecl_runtime::server::Handle,
    n: usize,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_a, h_b, h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let la = ArrayArg::<R>::from_raw_parts::<E>(h_a, n, 1);
        let lb = ArrayArg::<R>::from_raw_parts::<E>(h_b, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        elementwise_div_kernel::launch::<E, R>(client, cube_count(n), CubeDim::new_1d(256), la, lb, lout)
            .map(|()| h_out)
            .map_err(|e| Error::invalid(format!("GPU div_elem failed: {e}")))
    }
}

fn gpu_powf_helper<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    h_in: &cubecl_runtime::server::Handle,
    n: usize,
    power: E,
) -> Result<cubecl_runtime::server::Handle, Error> {
    let h_out = client.empty(n * std::mem::size_of::<E>());
    // SAFETY: h_in and h_out are valid GPU allocations for exactly `n` elements of type E.
    unsafe {
        let lin = ArrayArg::<R>::from_raw_parts::<E>(h_in, n, 1);
        let lout = ArrayArg::<R>::from_raw_parts::<E>(&h_out, n, 1);
        elementwise_powf_kernel::launch::<E, R>(
            client,
            cube_count(n),
            CubeDim::new_1d(256),
            lin,
            lout,
            ScalarArg::new(power),
        )
        .map(|()| h_out)
        .map_err(|e| Error::invalid(format!("GPU powf failed: {e}")))
    }
}

// ── Unary math dispatch macro ─────────────────────────────────────────────────
// TypeId dispatch: f32/f64 → GPU kernels, c32/c64 → CPU fallback via MathOps.

macro_rules! impl_gpu_unary_math_dispatch {
    ($name:ident, $helper:ident, $cpu_method:ident) => {
        pub(crate) fn $name<T: Scalar, R: Runtime>(a: &GpuStorage<T>) -> GpuStorage<T>
        where
            R::Device: Default,
        {
            let n = a.nrows * a.ncols;
            let client = R::client(&R::Device::default());
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let h = $helper::<f32, R>(&client, &a.handle, n)
                    .unwrap_or_else(|e| panic!("{e}"));
                GpuStorage::from_handle(a.nrows, a.ncols, h)
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let h = $helper::<f64, R>(&client, &a.handle, n)
                    .unwrap_or_else(|e| panic!("{e}"));
                GpuStorage::from_handle(a.nrows, a.ncols, h)
            } else {
                // c32/c64: CPU fallback via MathOps
                let ha = a.download::<R>();
                let data: Vec<T> = ha.iter()
                    .map(|&x| crate::backend::MathOps::$cpu_method(x))
                    .collect();
                GpuStorage::upload::<R>(a.nrows, a.ncols, data)
            }
        }
    };
}

impl_gpu_unary_math_dispatch!(gpu_exp, gpu_exp_helper, math_exp);
impl_gpu_unary_math_dispatch!(gpu_ln, gpu_ln_helper, math_ln);
impl_gpu_unary_math_dispatch!(gpu_log1p, gpu_log1p_helper, math_log1p);
impl_gpu_unary_math_dispatch!(gpu_sin, gpu_sin_helper, math_sin);
impl_gpu_unary_math_dispatch!(gpu_cos, gpu_cos_helper, math_cos);
impl_gpu_unary_math_dispatch!(gpu_tanh, gpu_tanh_helper, math_tanh);
impl_gpu_unary_math_dispatch!(gpu_sqrt, gpu_sqrt_helper, math_sqrt);
impl_gpu_unary_math_dispatch!(gpu_abs, gpu_abs_helper, math_abs);
impl_gpu_unary_math_dispatch!(gpu_recip, gpu_recip_helper, math_recip);
impl_gpu_unary_math_dispatch!(gpu_erf, gpu_erf_helper, math_erf);
impl_gpu_unary_math_dispatch!(gpu_ceil, gpu_ceil_helper, math_ceil);
impl_gpu_unary_math_dispatch!(gpu_floor, gpu_floor_helper, math_floor);
impl_gpu_unary_math_dispatch!(gpu_round, gpu_round_helper, math_round);

// ── Binary math dispatch functions ───────────────────────────────────────────

pub(crate) fn gpu_mul_elem<T: Scalar, R: Runtime>(
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_mul_elem_helper::<f32, R>(&client, &a.handle, &b.handle, n)
            .unwrap_or_else(|e| panic!("{e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_mul_elem_helper::<f64, R>(&client, &a.handle, &b.handle, n)
            .unwrap_or_else(|e| panic!("{e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let hb = b.download::<R>();
        let data: Vec<T> = ha.iter()
            .zip(hb.iter())
            .map(|(&x, &y)| crate::backend::MathOps::math_mul(x, y))
            .collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

pub(crate) fn gpu_div_elem<T: Scalar, R: Runtime>(
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_div_elem_helper::<f32, R>(&client, &a.handle, &b.handle, n)
            .unwrap_or_else(|e| panic!("{e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_div_elem_helper::<f64, R>(&client, &a.handle, &b.handle, n)
            .unwrap_or_else(|e| panic!("{e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let hb = b.download::<R>();
        let data: Vec<T> = ha.iter()
            .zip(hb.iter())
            .map(|(&x, &y)| crate::backend::MathOps::math_div(x, y))
            .collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

pub(crate) fn gpu_powf<T: Scalar, R: Runtime>(a: &GpuStorage<T>, p: T) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let power = (&p as &dyn std::any::Any)
            .downcast_ref::<f32>()
            .copied()
            .expect("T is f32");
        let h = gpu_powf_helper::<f32, R>(&client, &a.handle, n, power)
            .unwrap_or_else(|e| panic!("{e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let power = (&p as &dyn std::any::Any)
            .downcast_ref::<f64>()
            .copied()
            .expect("T is f64");
        let h = gpu_powf_helper::<f64, R>(&client, &a.handle, n, power)
            .unwrap_or_else(|e| panic!("{e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let data: Vec<T> = ha.iter()
            .map(|&x| crate::backend::MathOps::math_powf(x, p))
            .collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

// ── Public gpu_* dispatch functions ──────────────────────────────────────────
// TypeId dispatch: f32/f64 -> GPU kernels, c32/c64 -> CPU fallback.

pub(crate) fn gpu_zeros<T: Scalar, R: Runtime>(nrows: usize, ncols: usize) -> GpuStorage<T>
where
    R::Device: Default,
{
    let data = vec![<T as faer_traits::ComplexField>::zero_impl(); nrows * ncols];
    GpuStorage::upload::<R>(nrows, ncols, data)
}

pub(crate) fn gpu_from_fn<T: Scalar, R: Runtime>(
    nrows: usize,
    ncols: usize,
    mut f: impl FnMut(usize, usize) -> T,
) -> GpuStorage<T>
where
    R::Device: Default,
{
    let data: Vec<T> = (0..nrows * ncols).map(|i| f(i / ncols, i % ncols)).collect();
    GpuStorage::upload::<R>(nrows, ncols, data)
}

pub(crate) fn gpu_get<T: Scalar, R: Runtime>(s: &GpuStorage<T>, r: usize, c: usize) -> T
where
    R::Device: Default,
{
    let guard = s.fill_cache::<R>();
    guard.as_ref().expect("cache populated")[r * s.ncols + c]
}

pub(crate) fn gpu_set<T: Scalar, R: Runtime>(
    s: &mut GpuStorage<T>,
    r: usize,
    c: usize,
    v: T,
) where
    R::Device: Default,
{
    {
        let mut guard = s.fill_cache::<R>();
        guard.as_mut().expect("cache populated")[r * s.ncols + c] = v;
    }
    let guard = s.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let data = guard.as_ref().expect("cache populated");
    let client = R::client(&R::Device::default());
    // SAFETY: data is a valid [T] slice; reinterpreted as bytes for upload.
    s.handle = client.create_from_slice(unsafe { scalar_to_bytes(data) });
}

pub(crate) fn gpu_clone<T: Scalar, R: Runtime>(s: &GpuStorage<T>) -> GpuStorage<T>
where
    R::Device: Default,
{
    let guard = s.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let client = R::client(&R::Device::default());
    let new_handle = if let Some(ref data) = *guard {
        // SAFETY: data is a valid [T] slice.
        client.create_from_slice(unsafe { scalar_to_bytes(data) })
    } else {
        drop(guard);
        let bytes = client.read_one(s.handle.clone());
        client.create_from_slice(&bytes)
    };
    GpuStorage::from_handle(s.nrows, s.ncols, new_handle)
}

pub(crate) fn gpu_add<T: Scalar, R: Runtime>(
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_binary_kernel::<f32, R>(&client, &a.handle, &b.handle, n, false)
            .unwrap_or_else(|e| panic!("GPU add failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_binary_kernel::<f64, R>(&client, &a.handle, &b.handle, n, false)
            .unwrap_or_else(|e| panic!("GPU add failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let hb = b.download::<R>();
        let data: Vec<T> = ha.iter().zip(hb.iter()).map(|(&x, &y)| x + y).collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

pub(crate) fn gpu_sub<T: Scalar, R: Runtime>(
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_binary_kernel::<f32, R>(&client, &a.handle, &b.handle, n, true)
            .unwrap_or_else(|e| panic!("GPU sub failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_binary_kernel::<f64, R>(&client, &a.handle, &b.handle, n, true)
            .unwrap_or_else(|e| panic!("GPU sub failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let hb = b.download::<R>();
        let data: Vec<T> = ha.iter().zip(hb.iter()).map(|(&x, &y)| x - y).collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

pub(crate) fn gpu_neg<T: Scalar, R: Runtime>(a: &GpuStorage<T>) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_neg_kernel::<f32, R>(&client, &a.handle, n)
            .unwrap_or_else(|e| panic!("GPU neg failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_neg_kernel::<f64, R>(&client, &a.handle, n)
            .unwrap_or_else(|e| panic!("GPU neg failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let data: Vec<T> = ha.iter().map(|&x| -x).collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

pub(crate) fn gpu_scale<T: Scalar, R: Runtime>(a: &GpuStorage<T>, s: T) -> GpuStorage<T>
where
    R::Device: Default,
{
    let n = a.nrows * a.ncols;
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let scalar = (&s as &dyn std::any::Any)
            .downcast_ref::<f32>()
            .copied()
            .expect("T is f32");
        let h = gpu_scale_kernel::<f32, R>(&client, &a.handle, n, scalar)
            .unwrap_or_else(|e| panic!("GPU scale failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let scalar = (&s as &dyn std::any::Any)
            .downcast_ref::<f64>()
            .copied()
            .expect("T is f64");
        let h = gpu_scale_kernel::<f64, R>(&client, &a.handle, n, scalar)
            .unwrap_or_else(|e| panic!("GPU scale failed: {e}"));
        GpuStorage::from_handle(a.nrows, a.ncols, h)
    } else {
        let ha = a.download::<R>();
        let data: Vec<T> = ha.iter().map(|&x| x * s).collect();
        GpuStorage::upload::<R>(a.nrows, a.ncols, data)
    }
}

pub(crate) fn gpu_transpose<T: Scalar, R: Runtime>(a: &GpuStorage<T>) -> GpuStorage<T>
where
    R::Device: Default,
{
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_transpose_kernel::<f32, R>(&client, &a.handle, a.nrows, a.ncols)
            .unwrap_or_else(|e| panic!("GPU transpose failed: {e}"));
        GpuStorage::from_handle(a.ncols, a.nrows, h)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_transpose_kernel::<f64, R>(&client, &a.handle, a.nrows, a.ncols)
            .unwrap_or_else(|e| panic!("GPU transpose failed: {e}"));
        GpuStorage::from_handle(a.ncols, a.nrows, h)
    } else {
        let (rows, cols) = (a.nrows, a.ncols);
        let host = a.download::<R>();
        let zero = <T as faer_traits::ComplexField>::zero_impl();
        let mut buf = vec![zero; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                buf[c * rows + r] = host[r * cols + c];
            }
        }
        GpuStorage::upload::<R>(cols, rows, buf)
    }
}

pub(crate) fn gpu_matmul<T: Scalar, R: Runtime>(
    out: &mut GpuStorage<T>,
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) where
    R::Device: Default,
{
    let (rows, kdim, cols) = (a.nrows, a.ncols, b.ncols);
    let client = R::client(&R::Device::default());
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let h = gpu_matmul_kernel::<f32, R>(&client, &a.handle, &b.handle, rows, kdim, cols)
            .unwrap_or_else(|e| panic!("GPU matmul failed: {e}"));
        out.handle = h;
        out.nrows = rows;
        out.ncols = cols;
        let mut guard = out.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = None;
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let h = gpu_matmul_kernel::<f64, R>(&client, &a.handle, &b.handle, rows, kdim, cols)
            .unwrap_or_else(|e| panic!("GPU matmul failed: {e}"));
        out.handle = h;
        out.nrows = rows;
        out.ncols = cols;
        let mut guard = out.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = None;
    } else {
        let ha = a.download::<R>();
        let hb = b.download::<R>();
        let zero = <T as faer_traits::ComplexField>::zero_impl();
        let mut buf = vec![zero; rows * cols];
        for ri in 0..rows {
            for ci in 0..cols {
                buf[ri * cols + ci] = (0..kdim)
                    .fold(zero, |acc, l| acc + ha[ri * kdim + l] * hb[l * cols + ci]);
            }
        }
        let new = GpuStorage::upload::<R>(rows, cols, buf);
        out.handle = new.handle;
        out.nrows = rows;
        out.ncols = cols;
        let mut guard = out.host_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = new.host_cache.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

// ── GPU reduction dispatch functions ─────────────────────────────────────────
// Wave 6: sum/max/min/argmax/argmin — CPU fallback via download.
// The result is always a host scalar, so full download cost is paid regardless.
// Proper GPU-side partial reduction kernels (plane_sum + SharedMemory) can be
// added as an optimization once CubeCL subgroup size portability is confirmed.

pub(crate) fn gpu_sum_all<T: Scalar, R: Runtime>(s: &GpuStorage<T>) -> T
where
    R::Device: Default,
{
    let data = s.download::<R>();
    data.iter().fold(T::reduction_zero(), |acc, &x| acc.reduction_add(x))
}

pub(crate) fn gpu_max_all<T: Scalar, R: Runtime>(s: &GpuStorage<T>) -> T
where
    R::Device: Default,
{
    assert!(s.nrows > 0 && s.ncols > 0, "max_all: matrix must be non-empty");
    let data = s.download::<R>();
    data[1..].iter().fold(data[0], |acc, &x| acc.reduction_max(x))
}

pub(crate) fn gpu_min_all<T: Scalar, R: Runtime>(s: &GpuStorage<T>) -> T
where
    R::Device: Default,
{
    assert!(s.nrows > 0 && s.ncols > 0, "min_all: matrix must be non-empty");
    let data = s.download::<R>();
    data[1..].iter().fold(data[0], |acc, &x| acc.reduction_min(x))
}

pub(crate) fn gpu_argmax_all<T: Scalar, R: Runtime>(s: &GpuStorage<T>) -> (usize, usize)
where
    R::Device: Default,
{
    assert!(s.nrows > 0 && s.ncols > 0, "argmax_all: matrix must be non-empty");
    let data = s.download::<R>();
    let ncols = s.ncols;
    let mut best_idx = 0usize;
    for i in 1..(s.nrows * s.ncols) {
        if data[i].reduction_gt(data[best_idx]) {
            best_idx = i;
        }
    }
    (best_idx / ncols, best_idx % ncols)
}

pub(crate) fn gpu_argmin_all<T: Scalar, R: Runtime>(s: &GpuStorage<T>) -> (usize, usize)
where
    R::Device: Default,
{
    assert!(s.nrows > 0 && s.ncols > 0, "argmin_all: matrix must be non-empty");
    let data = s.download::<R>();
    let ncols = s.ncols;
    let mut best_idx = 0usize;
    for i in 1..(s.nrows * s.ncols) {
        if data[best_idx].reduction_gt(data[i]) {
            best_idx = i;
        }
    }
    (best_idx / ncols, best_idx % ncols)
}

// ── impl_gpu_backend! macro + invocations ────────────────────────────────────

macro_rules! impl_gpu_backend {
    ($Backend:ty, $Runtime:path) => {
        impl crate::backend::Backend for $Backend {
            type Storage<T: crate::scalar::Scalar> = crate::gpu::GpuStorage<T>;

            #[inline]
            fn zeros<T: crate::scalar::Scalar>(r: usize, c: usize) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_zeros::<T, $Runtime>(r, c)
            }

            #[inline]
            fn from_fn<T: crate::scalar::Scalar>(
                r: usize,
                c: usize,
                f: impl FnMut(usize, usize) -> T,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_from_fn::<T, $Runtime>(r, c, f)
            }

            #[inline]
            fn nrows<T: crate::scalar::Scalar>(s: &crate::gpu::GpuStorage<T>) -> usize {
                s.nrows
            }

            #[inline]
            fn ncols<T: crate::scalar::Scalar>(s: &crate::gpu::GpuStorage<T>) -> usize {
                s.ncols
            }

            #[inline]
            fn get<T: crate::scalar::Scalar>(
                s: &crate::gpu::GpuStorage<T>,
                r: usize,
                c: usize,
            ) -> T {
                crate::gpu::gpu_get::<T, $Runtime>(s, r, c)
            }

            #[inline]
            fn set<T: crate::scalar::Scalar>(
                s: &mut crate::gpu::GpuStorage<T>,
                r: usize,
                c: usize,
                v: T,
            ) {
                crate::gpu::gpu_set::<T, $Runtime>(s, r, c, v)
            }

            #[inline]
            fn matmul_into<T: crate::scalar::Scalar>(
                out: &mut crate::gpu::GpuStorage<T>,
                a: &crate::gpu::GpuStorage<T>,
                b: &crate::gpu::GpuStorage<T>,
            ) {
                crate::gpu::gpu_matmul::<T, $Runtime>(out, a, b)
            }

            #[inline]
            fn add<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
                b: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_add::<T, $Runtime>(a, b)
            }

            #[inline]
            fn sub<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
                b: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_sub::<T, $Runtime>(a, b)
            }

            #[inline]
            fn neg<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_neg::<T, $Runtime>(a)
            }

            #[inline]
            fn transpose<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_transpose::<T, $Runtime>(a)
            }

            #[inline]
            fn scale<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
                s: T,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_scale::<T, $Runtime>(a, s)
            }

            #[inline]
            fn clone_storage<T: crate::scalar::Scalar>(
                s: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_clone::<T, $Runtime>(s)
            }

            #[inline]
            fn exp<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_exp::<T, $Runtime>(a)
            }

            #[inline]
            fn ln<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_ln::<T, $Runtime>(a)
            }

            #[inline]
            fn log1p<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_log1p::<T, $Runtime>(a)
            }

            #[inline]
            fn sin<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_sin::<T, $Runtime>(a)
            }

            #[inline]
            fn cos<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_cos::<T, $Runtime>(a)
            }

            #[inline]
            fn tanh<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_tanh::<T, $Runtime>(a)
            }

            #[inline]
            fn sqrt<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_sqrt::<T, $Runtime>(a)
            }

            #[inline]
            fn abs<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_abs::<T, $Runtime>(a)
            }

            #[inline]
            fn recip<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_recip::<T, $Runtime>(a)
            }

            #[inline]
            fn erf<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_erf::<T, $Runtime>(a)
            }

            #[inline]
            fn ceil<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_ceil::<T, $Runtime>(a)
            }

            #[inline]
            fn floor<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_floor::<T, $Runtime>(a)
            }

            #[inline]
            fn round<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_round::<T, $Runtime>(a)
            }

            #[inline]
            fn powf<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
                p: T,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_powf::<T, $Runtime>(a, p)
            }

            #[inline]
            fn mul_elem<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
                b: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_mul_elem::<T, $Runtime>(a, b)
            }

            #[inline]
            fn div_elem<T: crate::scalar::Scalar>(
                a: &crate::gpu::GpuStorage<T>,
                b: &crate::gpu::GpuStorage<T>,
            ) -> crate::gpu::GpuStorage<T> {
                crate::gpu::gpu_div_elem::<T, $Runtime>(a, b)
            }

            #[inline]
            fn sum_all<T: crate::scalar::Scalar>(s: &crate::gpu::GpuStorage<T>) -> T {
                crate::gpu::gpu_sum_all::<T, $Runtime>(s)
            }

            #[inline]
            fn max_all<T: crate::scalar::Scalar>(s: &crate::gpu::GpuStorage<T>) -> T {
                crate::gpu::gpu_max_all::<T, $Runtime>(s)
            }

            #[inline]
            fn min_all<T: crate::scalar::Scalar>(s: &crate::gpu::GpuStorage<T>) -> T {
                crate::gpu::gpu_min_all::<T, $Runtime>(s)
            }

            #[inline]
            fn argmax_all<T: crate::scalar::Scalar>(
                s: &crate::gpu::GpuStorage<T>,
            ) -> (usize, usize) {
                crate::gpu::gpu_argmax_all::<T, $Runtime>(s)
            }

            #[inline]
            fn argmin_all<T: crate::scalar::Scalar>(
                s: &crate::gpu::GpuStorage<T>,
            ) -> (usize, usize) {
                crate::gpu::gpu_argmin_all::<T, $Runtime>(s)
            }
        }
    };
}

#[cfg(feature = "cuda")]
impl_gpu_backend!(crate::backend::Cuda, cubecl_cuda::CudaRuntime);

#[cfg(feature = "wgpu")]
impl_gpu_backend!(crate::backend::Wgpu, cubecl_wgpu::WgpuRuntime);
