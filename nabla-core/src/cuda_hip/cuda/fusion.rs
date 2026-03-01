use std::ffi::{CString, c_void};

use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::nvrtc;

use crate::gpu_common::{self, grid_1d, type_suffix};
use crate::kernels_cu::BLOCK_SIZE;
use crate::scalar::Scalar;

use super::*;

pub(super) fn cuda_fuse_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
    reg_estimate: usize,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_fused_{kernel_hash}_{tsuf}");

    {
        let map = ctx
            .kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let src = gpu_common::fuse_kernel_source(
                gpu_expr,
                n_inputs,
                type_name,
                &kernel_name,
                reg_estimate,
                true,
            );
            let (major, minor) = query_compute_capability();
            let arch: &'static str = nvrtc_arch(major, minor);
            let maxreg = (reg_estimate > 80).then_some(120);
            let ptx = nvrtc::compile_ptx_with_opts(
                &src,
                nvrtc::CompileOptions {
                    arch: Some(arch),
                    maxrregcount: maxreg,
                    ..Default::default()
                },
            )
            .or_panic("NVRTC fuse compile failed");
            let ptx_src = ptx.to_src();
            let c_ptx = CString::new(ptx_src).unwrap_or_else(|_| panic!("null in PTX"));
            // SAFETY: loading compiled PTX as a CUDA module.
            let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>()) }
                .or_panic("CUDA module load");
            let c_fn = CString::new(kernel_name.as_str())
                .unwrap_or_else(|_| panic!("null in kernel name"));
            // SAFETY: getting function handle from loaded module.
            let func =
                unsafe { result::module::get_function(module, c_fn) }.or_panic("CUDA get_function");
            let mut map = ctx
                .kernels
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(
                kernel_name.clone(),
                KernelEntry {
                    func,
                    _module: module,
                },
            );
        }
    }

    let func = expect_ok(get_kernel(ctx, &kernel_name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc");
    let n_u32 = n as u32;
    let grid = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        grid_1d((n + 3) / 4)
    } else {
        grid_1d(n)
    };

    // SAFETY: input pointers are valid CudaStorage<T> — cast back to extract .buf.ptr
    let input_ptrs: Vec<CUdeviceptr> = inputs
        .iter()
        .map(|&p| {
            let storage = unsafe { &*(p as *const CudaStorage<T>) };
            storage.buf.ptr
        })
        .collect();

    let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + 2);
    for ptr in &input_ptrs {
        args.push(ptr as *const CUdeviceptr as *mut c_void);
    }
    args.push(&out_buf.ptr as *const CUdeviceptr as *mut c_void);
    args.push(&n_u32 as *const u32 as *mut c_void);

    // SAFETY: launching fused kernel with correct argument layout.
    unsafe {
        result::launch_kernel(
            func,
            (grid, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        )
        .or_panic("CUDA launch {kernel_name}");
    }
    CudaStorage::new(nrows, ncols, out_buf)
}

#[inline]
fn use_tiled_mega(n_ops: usize, n: usize) -> bool {
    n_ops >= 2 && n >= 65_536
}

fn ops_all_share_inputs(ops: &[MegaFuseOp]) -> bool {
    let Some(first) = ops.first() else {
        return false;
    };
    ops.iter()
        .all(|op| op.n_inputs == first.n_inputs && op.inputs == first.inputs)
}

pub(crate) struct MegaFuseOp {
    pub inputs: Vec<*const u8>,
    pub gpu_expr: String,
    pub n_inputs: usize,
    pub uses_prev: bool,
}

pub(crate) fn cuda_mega_fuse_launch<T: Scalar>(
    ops: &[MegaFuseOp],
    nrows: usize,
    ncols: usize,
    kernel_hash: &str,
) -> Vec<CudaStorage<T>> {
    let ctx = get_ctx();
    let n = nrows * ncols;
    let tsuf = type_suffix::<T>();
    let any_dag = ops.iter().any(|op| op.uses_prev);
    let use_tiled = !any_dag && use_tiled_mega(ops.len(), n) && ops_all_share_inputs(ops);
    let kernel_name = if use_tiled {
        format!("k_mega_tile_{kernel_hash}_{tsuf}")
    } else {
        format!("k_mega_{kernel_hash}_{tsuf}")
    };

    {
        let map = ctx
            .kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let op_descs: Vec<(String, usize)> = ops
                .iter()
                .map(|op| (op.gpu_expr.clone(), op.n_inputs))
                .collect();
            let op_uses_prev: Vec<bool> = ops.iter().map(|op| op.uses_prev).collect();
            let src = if use_tiled {
                gpu_common::mega_fuse_tiled_kernel_source(
                    &op_descs,
                    type_name,
                    &kernel_name,
                    true, // use_ldg
                )
            } else {
                gpu_common::mega_fuse_kernel_source(
                    &op_descs,
                    &op_uses_prev,
                    type_name,
                    &kernel_name,
                    true,
                )
            };
            let (major, minor) = query_compute_capability();
            let arch: &'static str = nvrtc_arch(major, minor);
            let ptx = nvrtc::compile_ptx_with_opts(
                &src,
                nvrtc::CompileOptions {
                    arch: Some(arch),
                    ..Default::default()
                },
            )
            .or_panic("NVRTC mega-fuse compile failed");
            let ptx_src = ptx.to_src();
            let c_ptx = CString::new(ptx_src).unwrap_or_else(|_| panic!("null in PTX"));
            let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>()) }
                .or_panic("CUDA module load");
            let c_fn = CString::new(kernel_name.as_str())
                .unwrap_or_else(|_| panic!("null in kernel name"));
            let func =
                unsafe { result::module::get_function(module, c_fn) }.or_panic("CUDA get_function");
            let mut map = ctx
                .kernels
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(
                kernel_name.clone(),
                KernelEntry {
                    func,
                    _module: module,
                },
            );
        }
    }

    let func = expect_ok(get_kernel(ctx, &kernel_name), "CUDA kernel lookup");

    let out_bufs: Vec<CuBuffer> = (0..ops.len())
        .map(|_| {
            CuBuffer::alloc_async(&ctx.stream, n * std::mem::size_of::<T>()).or_panic("CUDA alloc")
        })
        .collect();

    let n_u32 = n as u32;

    if use_tiled {
        let n_inputs = ops[0].n_inputs;
        // SAFETY: each *const u8 was cast from a &CudaStorage<T> by the macro;
        let shared_input_ptrs: Vec<CUdeviceptr> = ops[0]
            .inputs
            .iter()
            .map(|&p| {
                let storage = unsafe { &*(p as *const CudaStorage<T>) };
                storage.buf.ptr
            })
            .collect();

        let tile_size: usize = if tsuf == "f32" { 1024 } else { 512 };
        let grid = grid_1d(n.div_ceil(tile_size));

        let total_args = n_inputs + ops.len() + 1;
        let mut args: Vec<*mut c_void> = Vec::with_capacity(total_args);
        for j in 0..n_inputs {
            args.push(&shared_input_ptrs[j] as *const CUdeviceptr as *mut c_void);
        }
        for op_idx in 0..ops.len() {
            args.push(&out_bufs[op_idx].ptr as *const CUdeviceptr as *mut c_void);
        }
        args.push(&n_u32 as *const u32 as *mut c_void);

        // SAFETY: launching tiled mega-kernel; argument layout matches the
        unsafe {
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut args,
            )
            .or_panic("CUDA launch {kernel_name}");
        }
    } else {
        let grid = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            grid_1d((n + 3) / 4)
        } else {
            grid_1d(n)
        };

        let input_ptrs: Vec<Vec<CUdeviceptr>> = ops
            .iter()
            .map(|op| {
                op.inputs
                    .iter()
                    .map(|&p| {
                        // SAFETY: raw pointer cast back to CudaStorage<T>; caller
                        let storage = unsafe { &*(p as *const CudaStorage<T>) };
                        storage.buf.ptr
                    })
                    .collect()
            })
            .collect();

        let total_args = ops.iter().map(|op| op.n_inputs + 1).sum::<usize>() + 1;
        let mut args: Vec<*mut c_void> = Vec::with_capacity(total_args);
        for (op_idx, op) in ops.iter().enumerate() {
            for j in 0..op.n_inputs {
                args.push(&input_ptrs[op_idx][j] as *const CUdeviceptr as *mut c_void);
            }
            args.push(&out_bufs[op_idx].ptr as *const CUdeviceptr as *mut c_void);
        }
        args.push(&n_u32 as *const u32 as *mut c_void);

        // SAFETY: launching standard mega-fused kernel with correct argument
        unsafe {
            result::launch_kernel(
                func,
                (grid, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                ctx.stream.cu_stream(),
                &mut args,
            )
            .or_panic("CUDA launch {kernel_name}");
        }
    }

    out_bufs
        .into_iter()
        .map(|buf| CudaStorage::new(nrows, ncols, buf))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_fuse_reduce_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    kernel_hash: &str,
    n_inputs: usize,
    reduce_op: u8,
    axis: u8,
) -> CudaStorage<T> {
    let ctx = get_ctx();
    let tsuf = type_suffix::<T>();
    let kernel_name = format!("k_fuse_reduce_{kernel_hash}_{tsuf}");

    {
        let map = ctx
            .kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !map.contains_key(&kernel_name) {
            drop(map);
            let type_name = if tsuf == "f32" { "float" } else { "double" };
            let src = gpu_common::fuse_reduce_kernel_source(
                gpu_expr,
                n_inputs,
                type_name,
                &kernel_name,
                axis,
                true,
            );
            let (major, minor) = query_compute_capability();
            let arch: &'static str = nvrtc_arch(major, minor);
            let ptx = nvrtc::compile_ptx_with_opts(
                &src,
                nvrtc::CompileOptions {
                    arch: Some(arch),
                    ..Default::default()
                },
            )
            .or_panic("NVRTC fuse_reduce compile failed");
            let ptx_src = ptx.to_src();
            let c_ptx = CString::new(ptx_src).unwrap_or_else(|_| panic!("null in PTX"));
            // SAFETY: loading compiled PTX as a CUDA module.
            let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>()) }
                .or_panic("CUDA module load (fuse_reduce)");
            let c_fn = CString::new(kernel_name.as_str())
                .unwrap_or_else(|_| panic!("null in kernel name"));
            // SAFETY: getting function handle from loaded module.
            let func = unsafe { result::module::get_function(module, c_fn) }
                .or_panic("CUDA get_function (fuse_reduce)");
            let mut map = ctx
                .kernels
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(
                kernel_name.clone(),
                KernelEntry {
                    func,
                    _module: module,
                },
            );
        }
    }

    let (out_rows, out_cols, grid_dim) = if axis == 1 {
        (nrows, 1usize, nrows as u32)
    } else {
        (1usize, ncols, ncols as u32)
    };

    let func = expect_ok(get_kernel(ctx, &kernel_name), "CUDA kernel lookup");
    let out_buf =
        CuBuffer::alloc_async(&ctx.stream, out_rows * out_cols * std::mem::size_of::<T>())
            .or_panic("CUDA alloc (fuse_reduce)");

    let rows_u32 = nrows as u32;
    let cols_u32 = ncols as u32;

    // SAFETY: each *const u8 in `inputs` is a valid &CudaStorage<T> cast.
    let input_ptrs: Vec<CUdeviceptr> = inputs
        .iter()
        .map(|&p| {
            let storage = unsafe { &*(p as *const CudaStorage<T>) };
            storage.buf.ptr
        })
        .collect();

    let mut args: Vec<*mut c_void> = Vec::with_capacity(n_inputs + 3);
    for ptr in &input_ptrs {
        args.push(ptr as *const CUdeviceptr as *mut c_void);
    }
    args.push(&out_buf.ptr as *const CUdeviceptr as *mut c_void);
    args.push(&rows_u32 as *const u32 as *mut c_void);
    args.push(&cols_u32 as *const u32 as *mut c_void);

    // SAFETY: launching fused map-reduce kernel with correct argument layout.
    unsafe {
        result::launch_kernel(
            func,
            (grid_dim, 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut args,
        )
        .or_panic("CUDA launch {kernel_name}");
    }

    let summed = CudaStorage::new(out_rows, out_cols, out_buf);

    if reduce_op == 3 {
        let count = if axis == 1 { ncols } else { nrows };
        let inv_n = T::from_f64(1.0 / count as f64);
        cuda_scale(&summed, inv_n)
    } else {
        summed
    }
}
