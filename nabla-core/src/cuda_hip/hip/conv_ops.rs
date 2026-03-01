use super::*;

pub(super) fn hip_im2col<T: Scalar>(
    input: &HipStorage<T>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    out_h: usize,
    out_w: usize,
) -> HipStorage<T> {
    let k_cols = c_in * kh * kw;
    let out_spatial = out_h * out_w;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("im2col"));
    let col_buf = HipBuffer::alloc_zeros(n * k_cols * out_spatial * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc im2col: {e}"));
    let col_elem = k_cols * out_spatial;
    let grid_x = ((col_elem as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, h_u, w_u, kh_u, kw_u, sh_u, sw_u, ph_u, pw_u, dh_u, dw_u, out_h_u, out_w_u) = (
        c_in as u32,
        h as u32,
        w as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
        ph as u32,
        pw as u32,
        dh as u32,
        dw as u32,
        out_h as u32,
        out_w as u32,
    );
    hip_launch(
        func,
        [grid_x, n as u32, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&col_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&c_in_u as *const u32).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&dh_u as *const u32).cast_mut().cast(),
            (&dw_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n * k_cols, out_spatial, col_buf)
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_im1col<T: Scalar>(
    input: &HipStorage<T>,
    n: usize,
    c_in: usize,
    l: usize,
    kl: usize,
    sl: usize,
    pl: usize,
    dl: usize,
    out_l: usize,
) -> HipStorage<T> {
    let k_cols = c_in * kl;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("im1col"));
    let col_buf = HipBuffer::alloc_zeros(n * k_cols * out_l * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc im1col: {e}"));
    let col_elem = k_cols * out_l;
    let grid_x = ((col_elem as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, l_u, kl_u, sl_u, pl_u, dl_u, out_l_u) = (
        c_in as u32,
        l as u32,
        kl as u32,
        sl as u32,
        pl as u32,
        dl as u32,
        out_l as u32,
    );
    hip_launch(
        func,
        [grid_x, n as u32, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&col_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&c_in_u as *const u32).cast_mut().cast(),
            (&l_u as *const u32).cast_mut().cast(),
            (&kl_u as *const u32).cast_mut().cast(),
            (&sl_u as *const u32).cast_mut().cast(),
            (&pl_u as *const u32).cast_mut().cast(),
            (&dl_u as *const u32).cast_mut().cast(),
            (&out_l_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n * k_cols, out_l, col_buf)
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_im3col<T: Scalar>(
    input: &HipStorage<T>,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    out_d: usize,
    out_h: usize,
    out_w: usize,
) -> HipStorage<T> {
    let k_vol = c_in * kd * kh * kw;
    let out_vol = out_d * out_h * out_w;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("im3col"));
    let col_buf = HipBuffer::alloc_zeros(n * k_vol * out_vol * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc im3col: {e}"));
    let col_elem = k_vol * out_vol;
    let grid_x = ((col_elem as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let (c_in_u, d_u, h_u, w_u) = (c_in as u32, d as u32, h as u32, w as u32);
    let (kd_u, kh_u, kw_u) = (kd as u32, kh as u32, kw as u32);
    let (sd_u, sh_u, sw_u) = (sd as u32, sh as u32, sw as u32);
    let (pd_u, ph_u, pw_u) = (pd as u32, ph as u32, pw as u32);
    let (dd_u, dh_u, dw_u) = (dd as u32, dh as u32, dw as u32);
    let (out_d_u, out_h_u, out_w_u) = (out_d as u32, out_h as u32, out_w as u32);
    hip_launch(
        func,
        [grid_x, n as u32, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&col_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&c_in_u as *const u32).cast_mut().cast(),
            (&d_u as *const u32).cast_mut().cast(),
            (&h_u as *const u32).cast_mut().cast(),
            (&w_u as *const u32).cast_mut().cast(),
            (&kd_u as *const u32).cast_mut().cast(),
            (&kh_u as *const u32).cast_mut().cast(),
            (&kw_u as *const u32).cast_mut().cast(),
            (&sd_u as *const u32).cast_mut().cast(),
            (&sh_u as *const u32).cast_mut().cast(),
            (&sw_u as *const u32).cast_mut().cast(),
            (&pd_u as *const u32).cast_mut().cast(),
            (&ph_u as *const u32).cast_mut().cast(),
            (&pw_u as *const u32).cast_mut().cast(),
            (&dd_u as *const u32).cast_mut().cast(),
            (&dh_u as *const u32).cast_mut().cast(),
            (&dw_u as *const u32).cast_mut().cast(),
            (&out_d_u as *const u32).cast_mut().cast(),
            (&out_h_u as *const u32).cast_mut().cast(),
            (&out_w_u as *const u32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n * k_vol, out_vol, col_buf)
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_conv1d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
    n: usize,
    c_in: usize,
    l: usize,
    c_out: usize,
    kl: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> HipStorage<T> {
    assert!(
        groups == 1,
        "GPU conv1d: groups > 1 not supported; use CPU backend"
    );
    let out_l = (l + 2 * padding - dilation * (kl - 1) - 1) / stride + 1;
    let k_cols = c_in * kl;

    let col = hip_im1col(input, n, c_in, l, kl, stride, padding, dilation, out_l);

    let out_buf = HipBuffer::alloc_zeros(n * c_out * out_l * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv1d out: {e}"));
    let mut out = HipStorage::new(n * c_out, out_l, out_buf);
    for bi in 0..n {
        let col_off = bi * k_cols * out_l * core::mem::size_of::<T>();
        let out_off = bi * c_out * out_l * core::mem::size_of::<T>();
        // SAFETY: offsets are within allocated buffers; borrow_ptr creates non-owning views.
        let col_ptr = unsafe { col.buf.ptr.byte_add(col_off) };
        let out_ptr = unsafe { out.buf.ptr.byte_add(out_off) };
        let col_slice = HipStorage::new(k_cols, out_l, unsafe {
            HipBuffer::borrow_ptr(col_ptr, k_cols * out_l * core::mem::size_of::<T>())
        });
        let mut out_slice = HipStorage::new(c_out, out_l, unsafe {
            HipBuffer::borrow_ptr(out_ptr, c_out * out_l * core::mem::size_of::<T>())
        });
        hip_matmul(&mut out_slice, weight, &col_slice);
        core::mem::forget(col_slice);
        core::mem::forget(out_slice);
    }
    out.invalidate_cache();
    out
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_conv3d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> HipStorage<T> {
    assert!(
        groups == 1,
        "GPU conv3d: groups > 1 not supported; use CPU backend"
    );
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;
    let (dd, dh, dw) = dilation;
    let out_d = (d + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let out_vol = out_d * out_h * out_w;
    let k_vol = c_in * kd * kh * kw;

    let col = hip_im3col(
        input, n, c_in, d, h, w, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, out_d, out_h,
        out_w,
    );

    let out_buf = HipBuffer::alloc_zeros(n * c_out * out_vol * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv3d out: {e}"));
    let mut out = HipStorage::new(n * c_out, out_vol, out_buf);
    for bi in 0..n {
        let col_off = bi * k_vol * out_vol * core::mem::size_of::<T>();
        let out_off = bi * c_out * out_vol * core::mem::size_of::<T>();
        // SAFETY: offsets are within allocated buffers; borrow_ptr creates non-owning views.
        let col_ptr = unsafe { col.buf.ptr.byte_add(col_off) };
        let out_ptr = unsafe { out.buf.ptr.byte_add(out_off) };
        let col_slice = HipStorage::new(k_vol, out_vol, unsafe {
            HipBuffer::borrow_ptr(col_ptr, k_vol * out_vol * core::mem::size_of::<T>())
        });
        let mut out_slice = HipStorage::new(c_out, out_vol, unsafe {
            HipBuffer::borrow_ptr(out_ptr, c_out * out_vol * core::mem::size_of::<T>())
        });
        hip_matmul(&mut out_slice, weight, &col_slice);
        core::mem::forget(col_slice);
        core::mem::forget(out_slice);
    }
    out.invalidate_cache();
    out
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_conv2d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> HipStorage<T> {
    assert!(
        groups == 1,
        "GPU conv2d: groups > 1 not supported; use CPU backend"
    );
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let out_spatial = out_h * out_w;
    let k_cols = c_in * kh * kw;

    // Step 1: im2col → col: (N*k_cols, out_spatial)
    let col = hip_im2col(
        input, n, c_in, h, w, kh, kw, sh, sw, ph, pw, dh, dw, out_h, out_w,
    );

    // Step 2: for each sample, GEMM weight (c_out x k_cols) @ col[b] (k_cols x out_spatial).
    let out_buf = HipBuffer::alloc_zeros(n * c_out * out_spatial * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv2d out: {e}"));
    let mut out = HipStorage::new(n * c_out, out_spatial, out_buf);
    for bi in 0..n {
        let col_off = bi * k_cols * out_spatial * core::mem::size_of::<T>();
        let out_off = bi * c_out * out_spatial * core::mem::size_of::<T>();
        // SAFETY: offsets are within allocated buffers; borrow_ptr creates non-owning views.
        let col_ptr = unsafe { col.buf.ptr.byte_add(col_off) };
        let out_ptr = unsafe { out.buf.ptr.byte_add(out_off) };
        let col_slice = HipStorage::new(k_cols, out_spatial, unsafe {
            HipBuffer::borrow_ptr(col_ptr, k_cols * out_spatial * core::mem::size_of::<T>())
        });
        let mut out_slice = HipStorage::new(c_out, out_spatial, unsafe {
            HipBuffer::borrow_ptr(out_ptr, c_out * out_spatial * core::mem::size_of::<T>())
        });
        hip_matmul(&mut out_slice, weight, &col_slice);
        // Prevent borrowed buffers from being freed on drop.
        core::mem::forget(col_slice);
        core::mem::forget(out_slice);
    }
    out.invalidate_cache();
    out
}

#[allow(clippy::too_many_arguments)]

pub(super) fn hip_conv_transpose2d<T: Scalar>(
    input: &HipStorage<T>,
    weight: &HipStorage<T>,
    n_batch: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> HipStorage<T> {
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let out_h = (h - 1) * sh + kh - 2 * ph + oph;
    let out_w = (w - 1) * sw + kw - 2 * pw + opw;
    let total = n_batch * c_out * out_h * out_w;
    let ctx = get_ctx();
    let func = get_kernel_by_id(ctx, kernel_id::<T>("conv_transpose2d"));
    let out_buf = HipBuffer::alloc_zeros(total * core::mem::size_of::<T>())
        .unwrap_or_else(|e| panic!("HIP alloc conv_transpose2d: {e}"));
    let (n_u, c_in_u, h_u, w_u, c_out_u, kh_u, kw_u, out_h_u, out_w_u, sh_u, sw_u, ph_u, pw_u) = (
        n_batch as i32,
        c_in as i32,
        h as i32,
        w as i32,
        c_out as i32,
        kh as i32,
        kw as i32,
        out_h as i32,
        out_w as i32,
        sh as i32,
        sw as i32,
        ph as i32,
        pw as i32,
    );
    hip_launch(
        func,
        [(total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1],
        [BLOCK_SIZE, 1, 1],
        &mut [
            (&input.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&weight.buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&out_buf.ptr as *const *mut c_void).cast_mut().cast(),
            (&n_u as *const i32).cast_mut().cast(),
            (&c_in_u as *const i32).cast_mut().cast(),
            (&h_u as *const i32).cast_mut().cast(),
            (&w_u as *const i32).cast_mut().cast(),
            (&c_out_u as *const i32).cast_mut().cast(),
            (&kh_u as *const i32).cast_mut().cast(),
            (&kw_u as *const i32).cast_mut().cast(),
            (&out_h_u as *const i32).cast_mut().cast(),
            (&out_w_u as *const i32).cast_mut().cast(),
            (&sh_u as *const i32).cast_mut().cast(),
            (&sw_u as *const i32).cast_mut().cast(),
            (&ph_u as *const i32).cast_mut().cast(),
            (&pw_u as *const i32).cast_mut().cast(),
        ],
    );
    HipStorage::new(n_batch * c_out, out_h * out_w, out_buf)
}
