use std::ffi::c_void;

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

pub(super) fn cuda_type_name<T: Scalar>() -> &'static str {
    match type_suffix::<T>() {
        "f32" => "float",
        "f64" => "double",
        _ => panic!("cuda indexing ops: only f32/f64 supported"),
    }
}

pub(crate) fn cuda_submatrix<T: Scalar>(
    a: &CudaStorage<T>,
    row_start: usize,
    col_start: usize,
    out_rows: usize,
    out_cols: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "submatrix", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let src_rows = a.nrows as u32;
    let src_cols = a.ncols as u32;
    let rs = row_start as u32;
    let cs = col_start as u32;
    let orows = out_rows as u32;
    let ocols = out_cols as u32;
    // SAFETY: launching submatrix kernel with valid buffer pointers and dimensions.
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
                    &src_rows as *const u32 as *mut c_void,
                    &src_cols as *const u32 as *mut c_void,
                    &rs as *const u32 as *mut c_void,
                    &cs as *const u32 as *mut c_void,
                    &orows as *const u32 as *mut c_void,
                    &ocols as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch submatrix",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_slice_set<T: Scalar>(
    dst: &mut CudaStorage<T>,
    row_start: usize,
    col_start: usize,
    src: &CudaStorage<T>,
) {
    dst.invalidate_cache();
    let ctx = get_ctx();
    let n = src.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "slice_set", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let src_rows = src.nrows as u32;
    let src_cols = src.ncols as u32;
    let dst_rows = dst.nrows as u32;
    let dst_cols = dst.ncols as u32;
    let rs = row_start as u32;
    let cs = col_start as u32;
    // SAFETY: launching slice_set kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &src.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &dst.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src_rows as *const u32 as *mut c_void,
                    &src_cols as *const u32 as *mut c_void,
                    &dst_rows as *const u32 as *mut c_void,
                    &dst_cols as *const u32 as *mut c_void,
                    &rs as *const u32 as *mut c_void,
                    &cs as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch slice_set",
        );
    }
}

pub(crate) fn cuda_gather_rows<T: Scalar>(a: &CudaStorage<T>, indices: &[usize]) -> CudaStorage<T> {
    let idx_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let idx_buf = cuda_upload_u32(&idx_u32);
    let ctx = get_ctx();
    let out_rows = indices.len();
    let out_cols = a.ncols;
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "gather_rows_u32idx", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_cols_u = out_cols as u32;
    let src_rows_u = a.nrows as u32;
    let src_cols_u = a.ncols as u32;
    let out_rows_u = out_rows as u32;
    // SAFETY: launching gather_rows kernel with valid buffer pointers and dimensions.
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
                    &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_rows_u as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                    &src_rows_u as *const u32 as *mut c_void,
                    &src_cols_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch gather_rows",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_gather<T: Scalar>(
    a: &CudaStorage<T>,
    axis: usize,
    index: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let out_rows = index.nrows;
    let out_cols = index.ncols;
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "gather", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let in_rows = a.nrows as u32;
    let in_cols = a.ncols as u32;
    let axis_u = axis as u32;
    let out_rows_u = out_rows as u32;
    let out_cols_u = out_cols as u32;
    // SAFETY: launching gather kernel with valid buffer pointers and dimensions.
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
                    &index.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_rows_u as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                    &in_rows as *const u32 as *mut c_void,
                    &in_cols as *const u32 as *mut c_void,
                    &axis_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch gather",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_scatter<T: Scalar>(
    a: &CudaStorage<T>,
    axis: usize,
    index: &CudaStorage<T>,
    src: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let mut out = cuda_clone(a);
    let n = index.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "scatter", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let out_rows_u = out.nrows as u32;
    let in_cols = out.ncols as u32;
    let axis_u = axis as u32;
    let src_cols = src.ncols as u32;
    let src_rows = src.nrows as u32;
    // SAFETY: launching scatter kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &src.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &index.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src_rows as *const u32 as *mut c_void,
                    &src_cols as *const u32 as *mut c_void,
                    &out_rows_u as *const u32 as *mut c_void,
                    &in_cols as *const u32 as *mut c_void,
                    &axis_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch scatter",
        );
    }
    out.invalidate_cache();
    out
}

pub(crate) fn cuda_index_select<T: Scalar>(
    a: &CudaStorage<T>,
    axis: usize,
    index: &CudaStorage<T>,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let k = index.nrows * index.ncols;
    let (out_rows, out_cols) = if axis == 0 {
        (k, a.ncols)
    } else {
        (a.nrows, k)
    };
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "index_select", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let in_rows = a.nrows as u32;
    let in_cols = a.ncols as u32;
    let axis_u = axis as u32;
    let out_rows_u = out_rows as u32;
    let out_cols_u = out_cols as u32;
    let k_u = k as u32;
    // SAFETY: launching index_select kernel with valid buffer pointers and dimensions.
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
                    &index.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_rows_u as *const u32 as *mut c_void,
                    &out_cols_u as *const u32 as *mut c_void,
                    &in_rows as *const u32 as *mut c_void,
                    &in_cols as *const u32 as *mut c_void,
                    &axis_u as *const u32 as *mut c_void,
                    &k_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch index_select",
        );
    }
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_sort_rows<T: Scalar>(
    a: &CudaStorage<T>,
    descending: bool,
) -> (CudaStorage<T>, CudaStorage<T>) {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let idx_buf = alloc_out::<T>(ctx, n);
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "sort_rows", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let desc_u = u32::from(descending);
    // SAFETY: launching sort_rows kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(rows), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &a.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &out_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &rows_u as *const u32 as *mut c_void,
                    &cols_u as *const u32 as *mut c_void,
                    &desc_u as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch sort_rows",
        );
    }
    (
        CudaStorage::new(rows, cols, out_buf),
        CudaStorage::new(rows, cols, idx_buf),
    )
}

pub(crate) fn cuda_topk_rows<T: Scalar>(
    a: &CudaStorage<T>,
    k: usize,
) -> (CudaStorage<T>, CudaStorage<T>) {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let out_n = rows * k;
    let out_buf = alloc_out::<T>(ctx, out_n);
    let idx_buf = alloc_out::<T>(ctx, out_n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_topk_rows_{tsuf}");
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let k_u = k as u32;
    // O(n*k) per-thread insertion sort with register-local k-buffer (not global memory)
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* in, {type_name}* out_val, {type_name}* out_idx, unsigned rows, unsigned cols, unsigned k) {{\n\
            unsigned row = blockIdx.x * blockDim.x + threadIdx.x;\n\
            if (row >= rows) return;\n\
            unsigned base_in = row * cols;\n\
            unsigned base_out = row * k;\n\
            {type_name} top_val[{k}];\n\
            {type_name} top_idx[{k}];\n\
            for (unsigned i = 0; i < {k}; ++i) {{\n\
                top_val[i] = in[base_in + i];\n\
                top_idx[i] = ({type_name})i;\n\
            }}\n\
            for (unsigned i = 1; i < {k}; ++i) {{\n\
                {type_name} tv = top_val[i];\n\
                {type_name} ti = top_idx[i];\n\
                int j = (int)i - 1;\n\
                while (j >= 0 && top_val[j] < tv) {{\n\
                    top_val[j + 1] = top_val[j];\n\
                    top_idx[j + 1] = top_idx[j];\n\
                    j--;\n\
                }}\n\
                top_val[j + 1] = tv;\n\
                top_idx[j + 1] = ti;\n\
            }}\n\
            for (unsigned c = {k}; c < cols; ++c) {{\n\
                {type_name} val = in[base_in + c];\n\
                if (val > top_val[{k} - 1]) {{\n\
                    int j = (int){k} - 2;\n\
                    while (j >= 0 && top_val[j] < val) {{\n\
                        top_val[j + 1] = top_val[j];\n\
                        top_idx[j + 1] = top_idx[j];\n\
                        j--;\n\
                    }}\n\
                    top_val[j + 1] = val;\n\
                    top_idx[j + 1] = ({type_name})c;\n\
                }}\n\
            }}\n\
            for (unsigned i = 0; i < {k}; ++i) {{\n\
                out_val[base_out + i] = top_val[i];\n\
                out_idx[base_out + i] = top_idx[i];\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
        &rows_u as *const u32 as *mut c_void,
        &cols_u as *const u32 as *mut c_void,
        &k_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(rows), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    (
        CudaStorage::new(rows, k, out_buf),
        CudaStorage::new(rows, k, idx_buf),
    )
}

pub(crate) fn cuda_scatter_add_dim0<T: Scalar>(
    dst: &mut CudaStorage<T>,
    indices: &[usize],
    src: &CudaStorage<T>,
) {
    let idx_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let idx_buf = cuda_upload_u32(&idx_u32);
    let ctx = get_ctx();
    let n = src.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "scatter_add_dim0_u32idx", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let src_cols = src.ncols as u32;
    let dst_rows_u = dst.nrows as u32;
    let dst_cols = dst.ncols as u32;
    let src_rows = src.nrows as u32;
    // SAFETY: launching scatter_add_dim0 kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &src.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &dst.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src_rows as *const u32 as *mut c_void,
                    &src_cols as *const u32 as *mut c_void,
                    &dst_rows_u as *const u32 as *mut c_void,
                    &dst_cols as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch scatter_add_dim0",
        );
    }
    dst.invalidate_cache();
}

pub(crate) fn cuda_scatter_add<T: Scalar>(
    dst: &mut CudaStorage<T>,
    axis: usize,
    indices: &[usize],
    src: &CudaStorage<T>,
) {
    if axis == 0 {
        cuda_scatter_add_dim0(dst, indices, src);
        return;
    }
    let idx_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let idx_buf = cuda_upload_u32(&idx_u32);
    let ctx = get_ctx();
    let n = src.n();
    let mut nbuf = [0u8; 64];
    let name = kernel_name_buf(&mut nbuf, "scatter_add_dim1_u32idx", type_suffix::<T>());
    let func = expect_ok(get_kernel(ctx, name), "CUDA kernel lookup");
    let src_rows = src.nrows as u32;
    let src_cols = src.ncols as u32;
    let dst_cols = dst.ncols as u32;
    // SAFETY: launching scatter_add_dim1 kernel with valid buffer pointers and dimensions.
    unsafe {
        expect_ok(
            result::launch_kernel(
                func,
                (grid_1d(n), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut [
                    &src.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
                    &dst.buf.ptr as *const CUdeviceptr as *mut c_void,
                    &src_rows as *const u32 as *mut c_void,
                    &src_cols as *const u32 as *mut c_void,
                    &dst_cols as *const u32 as *mut c_void,
                ],
            ),
            "CUDA launch scatter_add_dim1",
        );
    }
    dst.invalidate_cache();
}
