use std::ffi::c_void;

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

/// Launch a unary kernel with a scalar parameter: kernel(input, scalar, output, n).
pub(super) fn launch_unary_scalar<T: Scalar>(a: &CudaStorage<T>, s: T, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, op, type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    let grid = cuda_grid_1d::<T>(n);
    // SAFETY: launching elementwise kernel with valid buffer pointers and scalar arg.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &s as *const T as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA kernel launch",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_scale<T: Scalar>(a: &CudaStorage<T>, s: T) -> CudaStorage<T> {
    launch_unary_scalar(a, s, "scale")
}

pub(crate) fn cuda_scale_inplace<T: Scalar>(a: &mut CudaStorage<T>, s: T) {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "scale", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let n_u32 = n as u32;
    let grid = cuda_grid_1d::<T>(n);
    // SAFETY: scale kernel is elementwise; output ptr == input ptr is safe.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &s as *const T as *mut c_void,
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch scale_inplace",
        );
    }
    a.invalidate_cache();
}

pub(crate) fn cuda_cast<T: Scalar, U: Scalar>(a: &CudaStorage<T>) -> CudaStorage<U> {
    use std::any::TypeId;

    if TypeId::of::<T>() == TypeId::of::<U>() {
        let cloned = cuda_clone(a);
        // SAFETY: T == U, identical layout for CudaStorage.
        return unsafe { std::mem::transmute::<CudaStorage<T>, CudaStorage<U>>(cloned) };
    }

    fn cast_to_f32<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<f32> {
        use std::any::TypeId;
        let ctx = get_ctx();
        let n = a.n();
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let cloned = cuda_clone(a);
            // SAFETY: T is f32.
            return unsafe { std::mem::transmute::<CudaStorage<T>, CudaStorage<f32>>(cloned) };
        }
        let name = if TypeId::of::<T>() == TypeId::of::<half::f16>() {
            "k_cast_f16_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<half::bf16>() {
            "k_cast_bf16_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            "k_cast_f64_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E4M3>() {
            "k_cast_fp8e4m3_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp8E5M2>() {
            "k_cast_fp8e5m2_to_f32"
        } else if TypeId::of::<T>() == TypeId::of::<crate::scalar::Fp4E2M1>() {
            "k_cast_fp4e2m1_to_f32"
        } else {
            panic!("CUDA cast: unsupported source type");
        };
        let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
        let out_buf = alloc_out::<f32>(ctx, n);
        let n_u32 = n as u32;
        unsafe {
            expect_ok(
                result::launch_kernel(
                    func,
                    (grid_1d(n), 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &n_u32 as *const u32 as *mut c_void,
                    ],
                ),
                "CUDA launch cast to f32",
            );
        }
        CudaStorage::new(a.nrows, a.ncols, out_buf)
    }

    fn cast_from_f32<U: Scalar>(a: &CudaStorage<f32>) -> CudaStorage<U> {
        use std::any::TypeId;
        let ctx = get_ctx();
        let n = a.n();
        if TypeId::of::<U>() == TypeId::of::<f32>() {
            let cloned = cuda_clone(a);
            // SAFETY: U is f32.
            return unsafe { std::mem::transmute::<CudaStorage<f32>, CudaStorage<U>>(cloned) };
        }
        let name = if TypeId::of::<U>() == TypeId::of::<half::f16>() {
            "k_cast_f32_to_f16"
        } else if TypeId::of::<U>() == TypeId::of::<half::bf16>() {
            "k_cast_f32_to_bf16"
        } else if TypeId::of::<U>() == TypeId::of::<f64>() {
            "k_cast_f32_to_f64"
        } else if TypeId::of::<U>() == TypeId::of::<crate::scalar::Fp8E4M3>() {
            "k_cast_f32_to_fp8e4m3"
        } else if TypeId::of::<U>() == TypeId::of::<crate::scalar::Fp8E5M2>() {
            "k_cast_f32_to_fp8e5m2"
        } else if TypeId::of::<U>() == TypeId::of::<crate::scalar::Fp4E2M1>() {
            "k_cast_f32_to_fp4e2m1"
        } else {
            panic!("CUDA cast: unsupported destination type");
        };
        let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
        let out_buf = alloc_out::<U>(ctx, n);
        let n_u32 = n as u32;
        unsafe {
            expect_ok(
                result::launch_kernel(
                    func,
                    (grid_1d(n), 1, 1),
                    (BLOCK_SIZE, 1, 1),
                    0,
                    ctx.stream.cu_stream(),
                    &mut [
                        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                        &n_u32 as *const u32 as *mut c_void,
                    ],
                ),
                "CUDA launch cast from f32",
            );
        }
        CudaStorage::new(a.nrows, a.ncols, out_buf)
    }

    let tmp = cast_to_f32(a);
    cast_from_f32::<U>(&tmp)
}

pub(crate) fn cuda_masked_fill<T: Scalar>(
    a: &CudaStorage<T>,
    mask: &CudaStorage<T>,
    value: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "masked_fill", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &mask.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &value as *const T as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch masked_fill",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_where<T: Scalar>(
    a: &CudaStorage<T>,
    cond: &CudaStorage<T>,
    b: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "where", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &cond.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch where",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_axpy_inplace<T: Scalar>(y: &mut CudaStorage<T>, alpha: T, x: &CudaStorage<T>) {
    let ctx = get_ctx();
    let n = y.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "axpy", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let n_u32 = n as u32;
    let grid = cuda_grid_1d::<T>(n);
    // SAFETY: kernel writes y in-place; x is read-only.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &y.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &alpha as *const T as *mut c_void,
                    &x.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch axpy",
        );
    }
    y.invalidate_cache();
}

pub(crate) fn cuda_powf<T: Scalar>(a: &CudaStorage<T>, p: T) -> CudaStorage<T> {
    launch_unary_scalar(a, p, "powf")
}

pub(crate) fn cuda_expand<T: Scalar>(
    out: &mut CudaStorage<T>,
    src: &CudaStorage<T>,
    src_rows: usize,
    src_cols: usize,
) {
    let ctx = get_ctx();
    let dst_rows = out.nrows;
    let dst_cols = out.ncols;
    let n = dst_rows * dst_cols;
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "expand", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let src_rows_u32 = src_rows as u32;
    let src_cols_u32 = src_cols as u32;
    let dst_rows_u32 = dst_rows as u32;
    let dst_cols_u32 = dst_cols as u32;
    // SAFETY: launching expand kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &out.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src_rows_u32 as *const u32 as *mut c_void,
                    &src_cols_u32 as *const u32 as *mut c_void,
                    &dst_rows_u32 as *const u32 as *mut c_void,
                    &dst_cols_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch expand",
        );
    }
    out.invalidate_cache();
}

pub(crate) fn cuda_diag<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    cuda_zeros(a.nrows.min(a.ncols), 1)
}

pub(crate) fn cuda_mse_sum_fwd<T: Scalar>(
    pred: &CudaStorage<T>,
    target: &CudaStorage<T>,
) -> CudaStorage<T> {
    use crate::kernels_cu::{REDUCE_BLOCK, REDUCE_GRID_CAP};
    let ctx = get_ctx();
    let n = pred.n();
    assert_eq!(n, target.n(), "mse_sum_fwd: pred/target size mismatch");
    if n == 0 {
        return cuda_zeros(1, 1);
    }
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "mse_sum_fwd", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA mse_sum_fwd kernel");
    let grid1 = REDUCE_GRID_CAP.min(((n as u32) + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
    let scratch = ctx.reduce_scratch;
    let out_buf = alloc_out::<T>(ctx, 1);
    let n_u32 = n as u32;
    // SAFETY: launching reduction kernel; scratch is pre-allocated, out_buf is freshly allocated.
    unsafe {
        // SAFETY: zero counter at partial[grid1] before launch to prevent race
        cudarc::driver::sys::cuMemsetD32Async(
            scratch + (grid1 as usize * std::mem::size_of::<f32>()) as u64,
            0,
            1,
            ctx.stream.cu_stream(),
        );
        cudarc::driver::sys::cuLaunchKernel(
            func,
            grid1,
            1,
            1,
            REDUCE_BLOCK,
            1,
            1,
            0,
            ctx.stream.cu_stream(),
            [
                &pred.buf.ptr as *const CUdeviceptr as *mut c_void,
                &target.buf.ptr as *const CUdeviceptr as *mut c_void,
                &scratch as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
            ]
            .as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }
    CudaStorage::new(1, 1, out_buf)
}

pub(crate) fn cuda_mse_sum_bwd<T: Scalar>(
    pred: &CudaStorage<T>,
    target: &CudaStorage<T>,
    grad: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = pred.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "mse_sum_bwd", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA mse_sum_bwd kernel");
    let out_buf = alloc_out::<T>(ctx, n);
    let n_u32 = n as u32;
    let grid = cuda_grid_1d::<T>(n);
    // SAFETY: launching elementwise kernel with valid buffer pointers.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &pred.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &target.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &grad.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u32 as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch mse_sum_bwd",
        );
    }
    CudaStorage::new(pred.nrows, pred.ncols, out_buf)
}

pub(crate) fn cuda_add_bias_row<T: Scalar>(
    a: &CudaStorage<T>,
    bias: &CudaStorage<T>,
) -> CudaStorage<T> {
    let mut bias_expanded = cuda_zeros::<T>(a.nrows, a.ncols);
    cuda_expand(&mut bias_expanded, bias, 1, bias.ncols);
    launch_binary(a, &bias_expanded, "add")
}

pub(crate) fn cuda_copy_reshape<T: Scalar>(
    a: &CudaStorage<T>,
    out_rows: usize,
    out_cols: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = out_rows * out_cols;
    assert_eq!(n, a.n(), "cuda_copy_reshape: size mismatch");
    let bytes = n * std::mem::size_of::<T>();
    let out_buf = alloc_out::<T>(ctx, n);
    if bytes > 0 {
        // SAFETY: device-to-device copy of same-sized buffers.
        unsafe {
            let _ = cudarc::driver::result::memcpy_dtod_async(
                out_buf.ptr,
                a.buf.ptr,
                bytes,
                ctx.stream.cu_stream(),
            );
        }
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_repeat<T: Scalar>(
    a: &CudaStorage<T>,
    row_reps: usize,
    col_reps: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let out_rows = a.nrows * row_reps;
    let out_cols = a.ncols * col_reps;
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "repeat", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let in_rows = a.nrows as u32;
    let in_cols = a.ncols as u32;
    let out_rows_u = out_rows as u32;
    let out_cols_u = out_cols as u32;
    // SAFETY: launching repeat kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &in_rows as *const u32 as *mut c_void,
                    &in_cols as *const u32 as *mut c_void,
                    &out_rows_u as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch repeat",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_pad<T: Scalar>(
    a: &CudaStorage<T>,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
    value: T,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let out_rows = a.nrows + top + bottom;
    let out_cols = a.ncols + left + right;
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "pad", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let in_rows = a.nrows as u32;
    let in_cols = a.ncols as u32;
    let out_cols_u = out_cols as u32;
    let left_u = left as u32;
    let top_u = top as u32;
    let right_u = right as u32;
    let bottom_u = bottom as u32;
    // SAFETY: launching pad kernel with valid buffer pointers, dimensions, and scalar value.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &in_rows as *const u32 as *mut c_void,
                    &in_cols as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                    &left_u as *const u32 as *mut c_void,
                    &right_u as *const u32 as *mut c_void,
                    &top_u as *const u32 as *mut c_void,
                    &bottom_u as *const u32 as *mut c_void,
                    &value as *const T as *mut c_void,
                ],
            ),
            "CUDA launch pad",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

fn cuda_tri_impl<T: Scalar>(a: &CudaStorage<T>, diagonal: isize, mode: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, mode, type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let diag = diagonal as i32;
    // SAFETY: launching triu/tril kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows as *const u32 as *mut c_void,
                    &cols as *const u32 as *mut c_void,
                    &diag as *const i32 as *mut c_void,
                ],
            ),
            "CUDA launch tri",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_triu<T: Scalar>(a: &CudaStorage<T>, diagonal: isize) -> CudaStorage<T> {
    cuda_tri_impl(a, diagonal, "triu")
}

pub(crate) fn cuda_tril<T: Scalar>(a: &CudaStorage<T>, diagonal: isize) -> CudaStorage<T> {
    cuda_tri_impl(a, diagonal, "tril")
}

pub(crate) fn cuda_roll<T: Scalar>(
    a: &CudaStorage<T>,
    shift: isize,
    axis: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "roll", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let axis_u = axis as u32;
    let shift_i = shift as i32;
    // SAFETY: launching roll kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows as *const u32 as *mut c_void,
                    &cols as *const u32 as *mut c_void,
                    &shift_i as *const i32 as *mut c_void,
                    &axis_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch roll",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_flip<T: Scalar>(a: &CudaStorage<T>, axis: usize) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "flip", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let axis_u = axis as u32;
    // SAFETY: launching flip kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows as *const u32 as *mut c_void,
                    &cols as *const u32 as *mut c_void,
                    &axis_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch flip",
        );
    }
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_from_diag<T: Scalar>(v: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = v.nrows.max(v.ncols);
    let out_buf = alloc_out::<T>(ctx, n * n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "from_diag", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let n_u = n as u32;
    // SAFETY: launching from_diag kernel with valid buffer pointers.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n * n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &v.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &n_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch from_diag",
        );
    }
    CudaStorage::new(n, n, out_buf)
}

pub(crate) fn cuda_meshgrid<T: Scalar>(
    x: &CudaStorage<T>,
    y: &CudaStorage<T>,
) -> (CudaStorage<T>, CudaStorage<T>) {
    let ctx = get_ctx();
    let nx = x.nrows * x.ncols;
    let ny = y.nrows * y.ncols;
    let out_rows = ny;
    let out_cols = nx;
    let n = out_rows * out_cols;
    let out_x = alloc_out::<T>(ctx, n);
    let out_y = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "meshgrid", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let nx_u = nx as u32;
    let ny_u = ny as u32;
    let out_cols_u = out_cols as u32;
    // SAFETY: launching meshgrid kernel with valid buffer pointers.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &x.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &y.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_x.ptr as *const CUdeviceptr as *mut c_void,
                    &out_y.ptr as *const CUdeviceptr as *mut c_void,
                    &nx_u as *const u32 as *mut c_void,
                    &ny_u as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch meshgrid",
        );
    }
    (
        CudaStorage::new(out_rows, out_cols, out_x),
        CudaStorage::new(out_rows, out_cols, out_y),
    )
}

pub(crate) fn cuda_kron<T: Scalar>(
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    m: usize,
    n: usize,
    p: usize,
    q: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let out_rows = m * p;
    let out_cols = n * q;
    let n_out = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n_out);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "kron", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let m_u = m as u32;
    let n_u = n as u32;
    let p_u = p as u32;
    let q_u = q as u32;
    let out_cols_u = out_cols as u32;
    // SAFETY: launching kron kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n_out), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &b.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &m_u as *const u32 as *mut c_void,
                    &n_u as *const u32 as *mut c_void,
                    &p_u as *const u32 as *mut c_void,
                    &q_u as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch kron",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}
