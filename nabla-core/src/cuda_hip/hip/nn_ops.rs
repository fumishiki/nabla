use crate::gpu_common;

use super::*;

pub(super) fn hip_max_pool2d<T: Scalar>(
    a: &HipStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("max_pool2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
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
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(nc, out_h * out_w, out_buf)
}

pub(super) fn hip_avg_pool2d<T: Scalar>(
    a: &HipStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("avg_pool2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
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
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(nc, out_h * out_w, out_buf)
}

pub(super) fn hip_adaptive_avg_pool2d<T: Scalar>(
    a: &HipStorage<T>,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let nc = a.nrows;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("adaptive_avg_pool2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let (in_h_u, in_w_u, out_h_u, out_w_u, nc_u) = (
        in_h as u32,
        in_w as u32,
        out_h as u32,
        out_w as u32,
        nc as u32,
    );
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&in_h_u as *const u32).cast_mut().cast(),
            (&in_w_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(nc, out_h * out_w, out_buf)
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_softmax<T: Scalar>(a: &HipStorage<T>) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("softmax"));
    let out_buf = HipBuffer::alloc_zeros(rows * cols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    hip_launch(
        func,
        [rows as u32, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, cols, out_buf)
}

pub(super) fn hip_axis_reduce<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = HipBuffer::alloc_zeros(rows * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    hip_launch(
        func,
        [rows as u32, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, 1, out_buf)
}

pub(super) fn hip_axis_same_shape<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    hip_launch(
        func,
        [grid_1d(rows), 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, cols, out_buf)
}

pub(super) fn hip_cumsum_cumprod<T: Scalar>(a: &HipStorage<T>, op: &str) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let n = rows * cols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>(op));
    let out_buf = HipBuffer::alloc_zeros(n * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let shared_mem = (2 * BLOCK_SIZE as usize * core::mem::size_of::<T>()) as u32;
    hip_launch_smem(
        func,
        [rows as u32, 1, 1],
        [BLOCK_SIZE, 1, 1],
        shared_mem,
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&rows_u32 as *const u32).cast_mut().cast(),
            (&cols_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(rows, cols, out_buf)
}

pub(crate) fn hip_prod_all<T: Scalar>(a: &HipStorage<T>) -> T {
    gpu_common::rtc_fold_first_prod(a)
}

pub(super) fn hip_max_pool2d_with_idx<T: Scalar>(
    a: &HipStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> (HipStorage<T>, HipStorage<T>) {
    let ctx = get_ctx();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("max_pool2d_with_idx"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let idx_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc idx: {e}"));
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
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&idx_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
            (&nc_u as *const u32).cast_mut().cast(),
        ],
    );
    (
        HipStorage::new(nc, out_h * out_w, out_buf),
        HipStorage::new(nc, out_h * out_w, idx_buf),
    )
}

pub(super) fn hip_layer_norm<T: Scalar>(
    a: &HipStorage<T>,
    gamma: &HipStorage<T>,
    beta: &HipStorage<T>,
    eps: T,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("layer_norm"));
    let out_buf = HipBuffer::alloc_zeros(rows * cols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let eps_f = eps.to_f64();
    if type_suffix::<T>() == "f32" {
        let eps_val = eps_f as f32;
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_val as *const f32).cast_mut().cast(),
            ],
        );
    } else {
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_f as *const f64).cast_mut().cast(),
            ],
        );
    }
    HipStorage::new(rows, cols, out_buf)
}

pub(super) fn hip_rms_norm<T: Scalar>(
    a: &HipStorage<T>,
    gamma: &HipStorage<T>,
    eps: T,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let rows = a.nrows;
    let cols = a.ncols;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("rms_norm"));
    let out_buf = HipBuffer::alloc_zeros(rows * cols * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let eps_f = eps.to_f64();
    if type_suffix::<T>() == "f32" {
        let eps_val = eps_f as f32;
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_val as *const f32).cast_mut().cast(),
            ],
        );
    } else {
        hip_launch(
            func,
            [rows as u32, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&rows_u32 as *const u32).cast_mut().cast(),
                (&cols_u32 as *const u32).cast_mut().cast(),
                (&eps_f as *const f64).cast_mut().cast(),
            ],
        );
    }
    HipStorage::new(rows, cols, out_buf)
}

pub(super) fn hip_batch_norm_train<T: Scalar>(
    a: &HipStorage<T>,
    gamma: &HipStorage<T>,
    beta: &HipStorage<T>,
    running_mean: &mut HipStorage<T>,
    running_var: &mut HipStorage<T>,
    eps: T,
    momentum: T,
    training: bool,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = a.nrows;
    let c = a.ncols;
    let total = n * c;
    let sz = core::mem::size_of::<T>();
    let eps_f = eps.to_f64();
    let total_u32 = total as u32;
    let c_u32 = c as u32;
    let fwd_func = get_kernel_by_id(ctx, kernel_id::<T>("batch_norm_fwd"));
    let out_buf = HipBuffer::alloc_zeros(total * sz)
        .unwrap_or_else(|e| panic!("HIP alloc batch_norm out: {e}"));
    let fwd_grid = (total_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if training {
        let stats_func = get_kernel_by_id(ctx, kernel_id::<T>("batch_norm_stats"));
        let mean_buf = HipBuffer::alloc_zeros(c * sz)
            .unwrap_or_else(|e| panic!("HIP alloc batch_norm mean: {e}"));
        let var_buf = HipBuffer::alloc_zeros(c * sz)
            .unwrap_or_else(|e| panic!("HIP alloc batch_norm var: {e}"));
        let stats_grid = (c_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n_u32 = n as u32;
        hip_launch(
            stats_func,
            [stats_grid, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &mut [
                (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&mean_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&var_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&n_u32 as *const u32).cast_mut().cast(),
                (&c_u32 as *const u32).cast_mut().cast(),
            ],
        );
        let mean_s = HipStorage::new(1, c, mean_buf);
        let var_s = HipStorage::new(1, c, var_buf);
        let one_minus = T::from_f64(1.0) - momentum;
        for i in 0..c {
            let m = hip_get(&mean_s, 0, i);
            let v = hip_get(&var_s, 0, i);
            let rm = hip_get(running_mean, 0, i);
            let rv = hip_get(running_var, 0, i);
            hip_set(running_mean, 0, i, one_minus * rm + momentum * m);
            hip_set(running_var, 0, i, one_minus * rv + momentum * v);
        }
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&mean_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&var_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_val as *const f32).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        } else {
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&mean_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&var_s.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_f as *const f64).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        }
    } else {
        // Eval mode: use running_mean/running_var directly.
        if type_suffix::<T>() == "f32" {
            let eps_val = eps_f as f32;
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&running_mean.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&running_var.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_val as *const f32).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        } else {
            hip_launch(
                fwd_func,
                [fwd_grid, 1, 1],
                [BLOCK_SIZE, 1, 1],
                &mut [
                    (&a.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&gamma.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&beta.buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&running_mean.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&running_var.buf.ptr as *const *mut c_void)
                        .cast_mut()
                        .cast(),
                    (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                    (&eps_f as *const f64).cast_mut().cast(),
                    (&total_u32 as *const u32).cast_mut().cast(),
                    (&c_u32 as *const u32).cast_mut().cast(),
                ],
            );
        }
    }
    HipStorage::new(n, c, out_buf)
}

pub(super) fn hip_cross_entropy_fused<T: Scalar>(
    input: &HipStorage<T>,
    target: &HipStorage<T>,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let n = input.nrows;
    let c = input.ncols;
    let sz = core::mem::size_of::<T>();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("cross_entropy"));
    let loss_buf = HipBuffer::alloc_zeros(n * sz)
        .unwrap_or_else(|e| panic!("HIP alloc cross_entropy loss: {e}"));
    let n_u32 = n as u32;
    let c_u32 = c as u32;
    let grid = (n_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hip_launch(
        func,
        [grid, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&target.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&loss_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u32 as *const u32).cast_mut().cast(),
            (&c_u32 as *const u32).cast_mut().cast(),
        ],
    );
    let loss_s = HipStorage::new(n, 1, loss_buf);
    let total = (0..n).fold(T::zero(), |acc, i| acc + hip_get(&loss_s, i, 0));
    let mean = total / T::from_f64(n as f64);
    let out_buf = HipBuffer::alloc_zeros(sz)
        .unwrap_or_else(|e| panic!("HIP alloc cross_entropy result: {e}"));
    let mut out_s = HipStorage::new(1, 1, out_buf);
    hip_set(&mut out_s, 0, 0, mean);
    out_s
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_sdpa<T: Scalar>(
    q: &HipStorage<T>,
    k: &HipStorage<T>,
    v: &HipStorage<T>,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    batch_heads: usize,
) -> HipStorage<T> {
    const FA_BLOCK_M: u32 = 64;
    const FA_BLOCK_N: u32 = 64;
    let sz = core::mem::size_of::<T>();
    let out_buf = HipBuffer::alloc_zeros(batch_heads * seq_q * head_dim * sz)
        .unwrap_or_else(|e| panic!("HIP alloc sdpa: {e}"));
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("sdpa"));
    let num_q_tiles = seq_q.div_ceil(FA_BLOCK_M as usize) as u32;
    let grid = batch_heads as u32 * num_q_tiles;
    let smem = 2 * FA_BLOCK_N as usize * head_dim * sz;
    let seq_q_u = seq_q as u32;
    let seq_k_u = seq_k as u32;
    let head_dim_u = head_dim as u32;
    let bh_u = batch_heads as u32;
    let scale_f64 = 1.0_f64 / (head_dim as f64).sqrt();
    if type_suffix::<T>() == "f32" {
        let scale = scale_f64 as f32;
        hip_launch_smem(
            func,
            [grid, 1, 1],
            [FA_BLOCK_M, 1, 1],
            smem as u32,
            &mut [
                (&q.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&k.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&v.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&seq_q_u as *const u32).cast_mut().cast(),
                (&seq_k_u as *const u32).cast_mut().cast(),
                (&head_dim_u as *const u32).cast_mut().cast(),
                (&bh_u as *const u32).cast_mut().cast(),
                (&scale as *const f32).cast_mut().cast(),
            ],
        );
    } else {
        hip_launch_smem(
            func,
            [grid, 1, 1],
            [FA_BLOCK_M, 1, 1],
            smem as u32,
            &mut [
                (&q.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&k.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&v.buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
                (&seq_q_u as *const u32).cast_mut().cast(),
                (&seq_k_u as *const u32).cast_mut().cast(),
                (&head_dim_u as *const u32).cast_mut().cast(),
                (&bh_u as *const u32).cast_mut().cast(),
                (&scale_f64 as *const f64).cast_mut().cast(),
            ],
        );
    }
    HipStorage::new(batch_heads * seq_q, head_dim, out_buf)
}

pub(super) fn hip_embedding<T: Scalar>(
    indices: &HipStorage<T>,
    weight: &HipStorage<T>,
) -> HipStorage<T> {
    let ctx = get_ctx();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = weight.ncols;
    let total = n_tokens * embed_dim;
    let func = get_kernel_by_id(ctx, kernel_id::<T>("embedding"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc: {e}"));
    let n_tokens_u32 = n_tokens as u32;
    let embed_dim_u32 = embed_dim as u32;
    hip_launch(
        func,
        [((total as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&indices.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&weight.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_tokens_u32 as *const u32).cast_mut().cast(),
            (&embed_dim_u32 as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n_tokens, embed_dim, out_buf)
}
