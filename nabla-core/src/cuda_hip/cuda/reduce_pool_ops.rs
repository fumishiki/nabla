use std::ffi::c_void;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::result;

use crate::gpu_common::{grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

pub(super) fn cuda_max_pool2d<T: Scalar>(
    a: &CudaStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let name = format!("k_max_pool2d_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>())
        .or_panic("CUDA alloc");
    let (h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, out_h_u, out_w_u, nc_u) = (
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    unsafe {
        result::launch_kernel(
            func,
            ((total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &h_u as *const u32 as *mut c_void,
                &w_u as *const u32 as *mut c_void,
                &kh_u as *const u32 as *mut c_void,
                &kw_u as *const u32 as *mut c_void,
                &sh_u as *const u32 as *mut c_void,
                &sw_u as *const u32 as *mut c_void,
                &ph_u as *const u32 as *mut c_void,
                &pw_u as *const u32 as *mut c_void,
                &out_h_u as *const u32 as *mut c_void,
                &out_w_u as *const u32 as *mut c_void,
                &nc_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(nc, out_h * out_w, out_buf)
}

pub(super) fn cuda_avg_pool2d<T: Scalar>(
    a: &CudaStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let name = format!("k_avg_pool2d_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>())
        .or_panic("CUDA alloc");
    let (h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, out_h_u, out_w_u, nc_u) = (
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    unsafe {
        result::launch_kernel(
            func,
            ((total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &h_u as *const u32 as *mut c_void,
                &w_u as *const u32 as *mut c_void,
                &kh_u as *const u32 as *mut c_void,
                &kw_u as *const u32 as *mut c_void,
                &sh_u as *const u32 as *mut c_void,
                &sw_u as *const u32 as *mut c_void,
                &ph_u as *const u32 as *mut c_void,
                &pw_u as *const u32 as *mut c_void,
                &out_h_u as *const u32 as *mut c_void,
                &out_w_u as *const u32 as *mut c_void,
                &nc_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(nc, out_h * out_w, out_buf)
}

pub(super) fn cuda_adaptive_avg_pool2d<T: Scalar>(
    a: &CudaStorage<T>,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let total = nc * out_h * out_w;
    let name = format!("k_adaptive_avg_pool2d_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>())
        .or_panic("CUDA alloc");
    let (in_h_u, in_w_u, out_h_u, out_w_u, nc_u) = (
        in_h as u32,
        in_w as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    unsafe {
        result::launch_kernel(
            func,
            ((total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &in_h_u as *const u32 as *mut c_void,
                &in_w_u as *const u32 as *mut c_void,
                &out_h_u as *const u32 as *mut c_void,
                &out_w_u as *const u32 as *mut c_void,
                &nc_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(nc, out_h * out_w, out_buf)
}

pub(super) fn cuda_softmax<T: Scalar>(a: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let name = format!("k_softmax_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        result::launch_kernel(
            func,
            (rows as u32, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &rows_u32 as *const u32 as *mut c_void,
                &cols_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_axis_reduce<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, rows * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        result::launch_kernel(
            func,
            (rows as u32, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &rows_u32 as *const u32 as *mut c_void,
                &cols_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(rows, 1, out_buf)
}

pub(super) fn cuda_axis_same_shape<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        result::launch_kernel(
            func,
            (grid_1d(rows), 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &rows_u32 as *const u32 as *mut c_void,
                &cols_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(super) fn cuda_cumsum_cumprod<T: Scalar>(a: &CudaStorage<T>, op: &str) -> CudaStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let name = format!("k_{op}_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let shared_mem = (2 * BLOCK_SIZE as usize * std::mem::size_of::<T>()) as u32;
    unsafe {
        result::launch_kernel(
            func,
            (rows as u32, 1, 1),
            (BLOCK_SIZE, 1, 1),
            shared_mem,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &rows_u32 as *const u32 as *mut c_void,
                &cols_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    CudaStorage::new(rows, cols, out_buf)
}

pub(crate) fn cuda_prod_all<T: Scalar>(a: &CudaStorage<T>) -> T {
    let ctx = get_ctx();
    let n = a.n();
    if n == 0 {
        return T::one();
    }
    let tsuf = type_suffix::<T>();
    let func_name = format!("k_prod_partial_{tsuf}");
    let func = expect_ok(get_kernel(ctx, &func_name), "CUDA kernel lookup");
    let grid1 = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let partial_buf =
        CuBuffer::alloc_async(&ctx.stream, grid1 as usize * std::mem::size_of::<T>())
            .or_panic("CUDA alloc partial");
    let n_u32 = n as u32;
    unsafe {
        result::launch_kernel(
            func,
            (grid1, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &partial_buf.ptr as *const CUdeviceptr as *mut c_void,
                &n_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA prod phase1");
    }
    if grid1 == 1 {
        let partial_storage = CudaStorage::new(1, 1, partial_buf);
        return cuda_get(&partial_storage, 0, 0);
    }
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, std::mem::size_of::<T>()).or_panic("CUDA alloc out");
    let grid1_u32 = grid1;
    unsafe {
        result::launch_kernel(
            func,
            (1, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &partial_buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &grid1_u32 as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA prod phase2");
    }
    cuda_get(&CudaStorage::new(1, 1, out_buf), 0, 0)
}

pub(super) fn cuda_max_pool2d_with_idx<T: Scalar>(
    a: &CudaStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> (CudaStorage<T>, CudaStorage<T>) {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let name = format!("k_max_pool2d_with_idx_{}", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, &name), "CUDA kernel lookup");
    let out_buf = CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>())
        .or_panic("CUDA alloc");
    let idx_buf = CuBuffer::alloc_async(&ctx.stream, total * std::mem::size_of::<T>())
        .or_panic("CUDA alloc idx");
    let (h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, out_h_u, out_w_u, nc_u) = (
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    unsafe {
        result::launch_kernel(
            func,
            (grid_1d(total), 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
                &h_u as *const u32 as *mut c_void,
                &w_u as *const u32 as *mut c_void,
                &kh_u as *const u32 as *mut c_void,
                &kw_u as *const u32 as *mut c_void,
                &sh_u as *const u32 as *mut c_void,
                &sw_u as *const u32 as *mut c_void,
                &ph_u as *const u32 as *mut c_void,
                &pw_u as *const u32 as *mut c_void,
                &out_h_u as *const u32 as *mut c_void,
                &out_w_u as *const u32 as *mut c_void,
                &nc_u as *const u32 as *mut c_void,
            ],
        )
        .or_panic("CUDA launch {name}");
    }
    (
        CudaStorage::new(nc, out_h * out_w, out_buf),
        CudaStorage::new(nc, out_h * out_w, idx_buf),
    )
}
