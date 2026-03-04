use std::ffi::c_void;

use cudarc::driver::sys::CUdeviceptr;

use crate::gpu_common::{grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

fn cuda_type_name<T: Scalar>() -> &'static str {
    let tsuf = type_suffix::<T>();
    if tsuf == "f32" {
        "float"
    } else if tsuf == "f64" {
        "double"
    } else {
        panic!("cuda shape ops: only f32/f64 supported");
    }
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_submatrix_{tsuf}");
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned src_rows, unsigned src_cols, unsigned row_start, unsigned col_start, unsigned out_rows, unsigned out_cols) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned n = out_rows * out_cols;\n\
            if (i >= n) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            unsigned src_r = row_start + r;\n\
            unsigned src_c = col_start + c;\n\
            out[i] = a[src_r * src_cols + src_c];\n\
        }}\n"
    );
    let src_rows = a.nrows as u32;
    let src_cols = a.ncols as u32;
    let rs = row_start as u32;
    let cs = col_start as u32;
    let orows = out_rows as u32;
    let ocols = out_cols as u32;
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &src_rows as *const u32 as *mut c_void,
        &src_cols as *const u32 as *mut c_void,
        &rs as *const u32 as *mut c_void,
        &cs as *const u32 as *mut c_void,
        &orows as *const u32 as *mut c_void,
        &ocols as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_slice_set_{tsuf}");
    let src_rows = src.nrows as u32;
    let src_cols = src.ncols as u32;
    let dst_cols = dst.ncols as u32;
    let rs = row_start as u32;
    let cs = col_start as u32;
    let src_code = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* src, {type_name}* dst, unsigned src_rows, unsigned src_cols, unsigned dst_cols, unsigned row_start, unsigned col_start) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned n = src_rows * src_cols;\n\
            if (i >= n) return;\n\
            unsigned r = i / src_cols;\n\
            unsigned c = i - r * src_cols;\n\
            unsigned dr = row_start + r;\n\
            unsigned dc = col_start + c;\n\
            dst[dr * dst_cols + dc] = src[i];\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &src.buf.ptr as *const CUdeviceptr as *mut c_void,
        &dst.buf.ptr as *const CUdeviceptr as *mut c_void,
        &src_rows as *const u32 as *mut c_void,
        &src_cols as *const u32 as *mut c_void,
        &dst_cols as *const u32 as *mut c_void,
        &rs as *const u32 as *mut c_void,
        &cs as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src_code,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    ctx.stream.synchronize().ok();
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_repeat_{tsuf}");
    let in_rows = a.nrows as u32;
    let in_cols = a.ncols as u32;
    let out_rows_u = out_rows as u32;
    let out_cols_u = out_cols as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned in_rows, unsigned in_cols, unsigned out_rows, unsigned out_cols) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = out_rows * out_cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            unsigned src_r = r % in_rows;\n\
            unsigned src_c = c % in_cols;\n\
            out[i] = a[src_r * in_cols + src_c];\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &in_rows as *const u32 as *mut c_void,
        &in_cols as *const u32 as *mut c_void,
        &out_rows_u as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_pad_{tsuf}");
    let in_rows = a.nrows as u32;
    let in_cols = a.ncols as u32;
    let out_cols_u = out_cols as u32;
    let left_u = left as u32;
    let top_u = top as u32;
    let right_u = right as u32;
    let bottom_u = bottom as u32;
    let val = value;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned in_rows, unsigned in_cols, unsigned out_cols, unsigned left, unsigned right, unsigned top, unsigned bottom, {type_name} val) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned out_rows = in_rows + top + bottom;\n\
            unsigned out_cols_l = out_cols;\n\
            unsigned total = out_rows * out_cols_l;\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols_l;\n\
            unsigned c = i - r * out_cols_l;\n\
            if (r >= top && r < top + in_rows && c >= left && c < left + in_cols) {{\n\
                unsigned src_r = r - top;\n\
                unsigned src_c = c - left;\n\
                out[i] = a[src_r * in_cols + src_c];\n\
            }} else {{\n\
                out[i] = val;\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &in_rows as *const u32 as *mut c_void,
        &in_cols as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
        &left_u as *const u32 as *mut c_void,
        &right_u as *const u32 as *mut c_void,
        &top_u as *const u32 as *mut c_void,
        &bottom_u as *const u32 as *mut c_void,
        &val as *const T as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_triu<T: Scalar>(a: &CudaStorage<T>, diagonal: isize) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_triu_{tsuf}");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let diag = diagonal as i32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned rows, unsigned cols, int diag) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = rows * cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / cols;\n\
            unsigned c = i - r * cols;\n\
            if ((int)c >= (int)r + diag) {{\n\
                out[i] = a[i];\n\
            }} else {{\n\
                out[i] = ({type_name})0;\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &rows as *const u32 as *mut c_void,
        &cols as *const u32 as *mut c_void,
        &diag as *const i32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_tril<T: Scalar>(a: &CudaStorage<T>, diagonal: isize) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_tril_{tsuf}");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let diag = diagonal as i32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned rows, unsigned cols, int diag) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = rows * cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / cols;\n\
            unsigned c = i - r * cols;\n\
            if ((int)c <= (int)r + diag) {{\n\
                out[i] = a[i];\n\
            }} else {{\n\
                out[i] = ({type_name})0;\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &rows as *const u32 as *mut c_void,
        &cols as *const u32 as *mut c_void,
        &diag as *const i32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_roll<T: Scalar>(
    a: &CudaStorage<T>,
    shift: isize,
    axis: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_roll_{tsuf}");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let axis_u = axis as u32;
    let shift_i = shift as i32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned rows, unsigned cols, int shift, unsigned axis) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = rows * cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / cols;\n\
            unsigned c = i - r * cols;\n\
            if (axis == 0) {{\n\
                int dim = (int)rows;\n\
                int s = shift % dim;\n\
                if (s < 0) s += dim;\n\
                unsigned src_r = (unsigned)((((int)r - s) % dim + dim) % dim);\n\
                out[i] = a[src_r * cols + c];\n\
            }} else {{\n\
                int dim = (int)cols;\n\
                int s = shift % dim;\n\
                if (s < 0) s += dim;\n\
                unsigned src_c = (unsigned)((((int)c - s) % dim + dim) % dim);\n\
                out[i] = a[r * cols + src_c];\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &rows as *const u32 as *mut c_void,
        &cols as *const u32 as *mut c_void,
        &shift_i as *const i32 as *mut c_void,
        &axis_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_flip<T: Scalar>(a: &CudaStorage<T>, axis: usize) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = a.n();
    let out_buf = alloc_out::<T>(ctx, n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_flip_{tsuf}");
    let rows = a.nrows as u32;
    let cols = a.ncols as u32;
    let axis_u = axis as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, {type_name}* out, unsigned rows, unsigned cols, unsigned axis) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = rows * cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / cols;\n\
            unsigned c = i - r * cols;\n\
            if (axis == 0) {{\n\
                unsigned src_r = rows - 1 - r;\n\
                out[i] = a[src_r * cols + c];\n\
            }} else {{\n\
                unsigned src_c = cols - 1 - c;\n\
                out[i] = a[r * cols + src_c];\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &rows as *const u32 as *mut c_void,
        &cols as *const u32 as *mut c_void,
        &axis_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(a.nrows, a.ncols, out_buf)
}

pub(crate) fn cuda_from_diag<T: Scalar>(v: &CudaStorage<T>) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = v.nrows.max(v.ncols);
    let out_buf = alloc_out::<T>(ctx, n * n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_from_diag_{tsuf}");
    let n_u = n as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* v, {type_name}* out, unsigned n) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = n * n;\n\
            if (i >= total) return;\n\
            unsigned r = i / n;\n\
            unsigned c = i - r * n;\n\
            out[i] = (r == c) ? v[r] : ({type_name})0;\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &v.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &n_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n * n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(n, n, out_buf)
}

pub(crate) fn cuda_gather_rows<T: Scalar>(a: &CudaStorage<T>, indices: &[usize]) -> CudaStorage<T> {
    let idx: Vec<T> = indices.iter().map(|&i| T::from_f64(i as f64)).collect();
    let idx_storage = cuda_from_vec_async(indices.len(), 1, idx);
    let ctx = get_ctx();
    let out_rows = indices.len();
    let out_cols = a.ncols;
    let n = out_rows * out_cols;
    let out_buf = alloc_out::<T>(ctx, n);
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_gather_rows_{tsuf}");
    let out_cols_u = out_cols as u32;
    let src_cols_u = a.ncols as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, const {type_name}* idx, {type_name}* out, unsigned out_rows, unsigned out_cols, unsigned src_cols) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = out_rows * out_cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            unsigned src_r = (unsigned)idx[r];\n\
            out[i] = a[src_r * src_cols + c];\n\
        }}\n"
    );
    let out_rows_u = out_rows as u32;
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &idx_storage.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_rows_u as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
        &src_cols_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_gather_{tsuf}");
    let in_cols = a.ncols as u32;
    let axis_u = axis as u32;
    let out_rows_u = out_rows as u32;
    let out_cols_u = out_cols as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, const {type_name}* idx, {type_name}* out, unsigned out_rows, unsigned out_cols, unsigned in_cols, unsigned axis) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = out_rows * out_cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            unsigned index_val = (unsigned)idx[i];\n\
            if (axis == 0) {{\n\
                out[i] = a[index_val * in_cols + c];\n\
            }} else {{\n\
                out[i] = a[r * in_cols + index_val];\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &index.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_rows_u as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
        &in_cols as *const u32 as *mut c_void,
        &axis_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(out_rows, out_cols, out_buf)
}

pub(crate) fn cuda_scatter<T: Scalar>(
    a: &CudaStorage<T>,
    axis: usize,
    index: &CudaStorage<T>,
    src: &CudaStorage<T>,
) -> CudaStorage<T> {
    let mut out = cuda_clone(a);
    let n = index.n();
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_scatter_{tsuf}");
    let in_cols = out.ncols as u32;
    let axis_u = axis as u32;
    let src_cols = src.ncols as u32;
    let src_rows = src.nrows as u32;
    let src_code = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* src, const {type_name}* idx, {type_name}* out, unsigned src_rows, unsigned src_cols, unsigned out_cols, unsigned axis) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = src_rows * src_cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / src_cols;\n\
            unsigned c = i - r * src_cols;\n\
            unsigned index_val = (unsigned)idx[i];\n\
            if (axis == 0) {{\n\
                out[index_val * out_cols + c] = src[i];\n\
            }} else {{\n\
                out[r * out_cols + index_val] = src[i];\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &src.buf.ptr as *const CUdeviceptr as *mut c_void,
        &index.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out.buf.ptr as *const CUdeviceptr as *mut c_void,
        &src_rows as *const u32 as *mut c_void,
        &src_cols as *const u32 as *mut c_void,
        &in_cols as *const u32 as *mut c_void,
        &axis_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src_code,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_index_select_{tsuf}");
    let in_cols = a.ncols as u32;
    let axis_u = axis as u32;
    let out_rows_u = out_rows as u32;
    let out_cols_u = out_cols as u32;
    let k_u = k as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, const {type_name}* idx, {type_name}* out, unsigned out_rows, unsigned out_cols, unsigned in_cols, unsigned axis, unsigned k) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = out_rows * out_cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            if (axis == 0) {{\n\
                unsigned src_r = (unsigned)idx[r];\n\
                out[i] = a[src_r * in_cols + c];\n\
            }} else {{\n\
                unsigned src_c = (unsigned)idx[c];\n\
                out[i] = a[r * in_cols + src_c];\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &index.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_rows_u as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
        &in_cols as *const u32 as *mut c_void,
        &axis_u as *const u32 as *mut c_void,
        &k_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_sort_rows_{tsuf}");
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let desc_u = if descending { 1u32 } else { 0u32 };
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* in, {type_name}* out, {type_name}* idx, unsigned rows, unsigned cols, unsigned desc) {{\n\
            unsigned row = blockIdx.x * blockDim.x + threadIdx.x;\n\
            if (row >= rows) return;\n\
            unsigned base = row * cols;\n\
            for (unsigned c = 0; c < cols; ++c) {{\n\
                out[base + c] = in[base + c];\n\
                idx[base + c] = ({type_name})c;\n\
            }}\n\
            for (unsigned i = 0; i < cols; ++i) {{\n\
                for (unsigned j = 0; j + 1 < cols - i; ++j) {{\n\
                    unsigned a_idx = base + j;\n\
                    unsigned b_idx = base + j + 1;\n\
                    {type_name} va = out[a_idx];\n\
                    {type_name} vb = out[b_idx];\n\
                    int cmp = desc ? (va < vb) : (va > vb);\n\
                    if (cmp) {{\n\
                        out[a_idx] = vb;\n\
                        out[b_idx] = va;\n\
                        {type_name} ia = idx[a_idx];\n\
                        idx[a_idx] = idx[b_idx];\n\
                        idx[b_idx] = ia;\n\
                    }}\n\
                }}\n\
            }}\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &idx_buf.ptr as *const CUdeviceptr as *mut c_void,
        &rows_u as *const u32 as *mut c_void,
        &cols_u as *const u32 as *mut c_void,
        &desc_u as *const u32 as *mut c_void,
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
        CudaStorage::new(rows, cols, out_buf),
        CudaStorage::new(rows, cols, idx_buf),
    )
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_meshgrid_{tsuf}");
    let nx_u = nx as u32;
    let ny_u = ny as u32;
    let out_cols_u = out_cols as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* x, const {type_name}* y, {type_name}* out_x, {type_name}* out_y, unsigned nx, unsigned ny, unsigned out_cols) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = nx * ny;\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            out_x[i] = x[c];\n\
            out_y[i] = y[r];\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &x.buf.ptr as *const CUdeviceptr as *mut c_void,
        &y.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_x.ptr as *const CUdeviceptr as *mut c_void,
        &out_y.ptr as *const CUdeviceptr as *mut c_void,
        &nx_u as *const u32 as *mut c_void,
        &ny_u as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    (
        CudaStorage::new(out_rows, out_cols, out_x),
        CudaStorage::new(out_rows, out_cols, out_y),
    )
}

pub(crate) fn cuda_scatter_add_dim0<T: Scalar>(
    dst: &mut CudaStorage<T>,
    indices: &[usize],
    src: &CudaStorage<T>,
) {
    let idx: Vec<T> = indices.iter().map(|&i| T::from_f64(i as f64)).collect();
    let idx_storage = cuda_from_vec_async(indices.len(), 1, idx);
    let n = src.n();
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_scatter_add_dim0_{tsuf}");
    let src_cols = src.ncols as u32;
    let dst_cols = dst.ncols as u32;
    let src_rows = src.nrows as u32;
    let src_code = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* src, const {type_name}* idx, {type_name}* dst, unsigned src_rows, unsigned src_cols, unsigned dst_cols) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = src_rows * src_cols;\n\
            if (i >= total) return;\n\
            unsigned r = i / src_cols;\n\
            unsigned c = i - r * src_cols;\n\
            unsigned dst_r = (unsigned)idx[r];\n\
            atomicAdd(&dst[dst_r * dst_cols + c], src[i]);\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &src.buf.ptr as *const CUdeviceptr as *mut c_void,
        &idx_storage.buf.ptr as *const CUdeviceptr as *mut c_void,
        &dst.buf.ptr as *const CUdeviceptr as *mut c_void,
        &src_rows as *const u32 as *mut c_void,
        &src_cols as *const u32 as *mut c_void,
        &dst_cols as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src_code,
        (grid_1d(n), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    dst.invalidate_cache();
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
    let type_name = cuda_type_name::<T>();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_kron_{tsuf}");
    let m_u = m as u32;
    let n_u = n as u32;
    let p_u = p as u32;
    let q_u = q as u32;
    let out_cols_u = out_cols as u32;
    let src = format!(
        "extern \"C\" __global__ void {kernel_name}(const {type_name}* a, const {type_name}* b, {type_name}* out, unsigned m, unsigned n, unsigned p, unsigned q, unsigned out_cols) {{\n\
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n\
            unsigned total = (m * p) * (n * q);\n\
            if (i >= total) return;\n\
            unsigned r = i / out_cols;\n\
            unsigned c = i - r * out_cols;\n\
            unsigned ar = r / p;\n\
            unsigned ac = c / q;\n\
            unsigned br = r - ar * p;\n\
            unsigned bc = c - ac * q;\n\
            out[i] = a[ar * n + ac] * b[br * q + bc];\n\
        }}\n"
    );
    let mut args: Vec<*mut c_void> = vec![
        &a.buf.ptr as *const CUdeviceptr as *mut c_void,
        &b.buf.ptr as *const CUdeviceptr as *mut c_void,
        &out_buf.ptr as *const CUdeviceptr as *mut c_void,
        &m_u as *const u32 as *mut c_void,
        &n_u as *const u32 as *mut c_void,
        &p_u as *const u32 as *mut c_void,
        &q_u as *const u32 as *mut c_void,
        &out_cols_u as *const u32 as *mut c_void,
    ];
    cuda_launch_kernel_src(
        &kernel_name,
        &src,
        (grid_1d(n_out), 1, 1),
        (BLOCK_SIZE, 1, 1),
        0,
        &mut args,
    );
    CudaStorage::new(out_rows, out_cols, out_buf)
}
