use std::any::TypeId;

use wgpu::util::DeviceExt;

use crate::scalar::Scalar;

use super::shaders::*;
use super::storage::*;

fn params_buf(data: &[u32]) -> wgpu::Buffer {
    let ctx = get_context();
    let bytes: &[u8] = bytemuck_cast_u32(data);
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
}

fn bytemuck_cast_u32(data: &[u32]) -> &[u8] {
    // SAFETY: u32 is 4 bytes, plain old data.
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) }
}

#[inline]
fn workgroups(n: usize, wg_size: u32) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        n.div_ceil(wg_size as usize) as u32
    }
}

#[allow(dead_code)]
fn run_custom_shader(
    ctx: &GpuContext,
    shader_src: &str,
    buffers: &[(&wgpu::Buffer, bool)],
    workgroups_x: u32,
) {
    let pipeline = compile_pipeline(ctx, shader_src);
    let bg = bind_group(ctx, &pipeline, buffers);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups_x);
}

#[allow(clippy::cast_possible_truncation)]
fn run_1in(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    n: usize,
    params: &[u32],
) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey { op, wg_size: wg };
    let p = params_buf(params);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

#[allow(clippy::cast_possible_truncation)]
fn run_2in(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n: usize,
    params: &[u32],
) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey { op, wg_size: wg };
    let p = params_buf(params);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[(a, true), (b, true), (&out, false), (&p, true)],
        );
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

#[allow(clippy::cast_possible_truncation)]
fn run_binary_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n: usize,
    op_type: u32,
) -> wgpu::Buffer {
    run_2in(ctx, ShaderOp::Binary, a, b, n, &[n as u32, op_type])
}

#[allow(clippy::cast_possible_truncation)]
fn run_unary_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, op_type: u32) -> wgpu::Buffer {
    run_1in(ctx, ShaderOp::Unary, a, n, &[n as u32, op_type])
}

#[allow(clippy::cast_possible_truncation)]
fn run_scale_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, scalar: f32) -> wgpu::Buffer {
    run_1in(ctx, ShaderOp::Scale, a, n, &[n as u32, scalar.to_bits()])
}

#[allow(clippy::cast_possible_truncation)]
fn run_powf_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, power: f32) -> wgpu::Buffer {
    run_1in(ctx, ShaderOp::Powf, a, n, &[n as u32, power.to_bits()])
}

#[allow(clippy::cast_possible_truncation)]
fn run_transpose_f32(ctx: &GpuContext, a: &wgpu::Buffer, rows: usize, cols: usize) -> wgpu::Buffer {
    run_1in(
        ctx,
        ShaderOp::Transpose,
        a,
        rows * cols,
        &[rows as u32, cols as u32],
    )
}

#[allow(clippy::cast_possible_truncation)]
fn run_copy_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    run_1in(ctx, ShaderOp::Copy, a, n, &[n as u32])
}

fn run_matmul_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> wgpu::Buffer {
    // Large matrices: register-tile software MMA
    if m >= 64 && n >= 64 && k >= 64 {
        return run_matmul_register_tile_f32(ctx, a, b, m, k, n);
    }
    let tile = select_matmul_tile(m, k, n);
    let tile_usize = tile as usize;
    let grid_cols = n.div_ceil(tile_usize);
    let grid_rows = m.div_ceil(tile_usize);
    let grid_size = grid_rows * grid_cols;
    let key = PipelineKey {
        op: ShaderOp::Matmul { tile },
        wg_size: tile,
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[m as u32, k as u32, n as u32, grid_cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((m * n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[(a, true), (b, true), (&out, false), (&params, true)],
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            #[allow(clippy::cast_possible_truncation)]
            pass.dispatch_workgroups(grid_size as u32, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.device.poll(wgpu::MaintainBase::Wait).panic_on_timeout();
    });
    out
}

fn run_matmul_register_tile_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> wgpu::Buffer {
    let (tr, tc, bm, bn, bk) = select_register_tile_params(m, n, k);
    let bm_u = bm as usize;
    let bn_u = bn as usize;
    let grid_cols = n.div_ceil(bn_u);
    let grid_rows = m.div_ceil(bm_u);
    let grid_size = grid_rows * grid_cols;
    let wg_x = bn / tc;
    let key = PipelineKey {
        op: ShaderOp::MatmulRegTile { tr, tc, bm, bn, bk },
        wg_size: wg_x,
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[m as u32, k as u32, n as u32, grid_cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((m * n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[(a, true), (b, true), (&out, false), (&params, true)],
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            #[allow(clippy::cast_possible_truncation)]
            pass.dispatch_workgroups(grid_size as u32, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.device.poll(wgpu::MaintainBase::Wait).panic_on_timeout();
    });
    out
}

fn run_fill_zeros_f32(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::FillZeros,
        wg_size: wg,
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(&out, false), (&params, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

fn run_fill_scalar_f32(ctx: &GpuContext, n: usize, val: f32) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::FillScalar,
        wg_size: wg,
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, val.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(&out, false), (&params, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

fn run_fill_identity_f32(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::FillIdentity,
        wg_size: wg,
    };
    let total = n * n;
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(&out, false), (&params, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(total, wg));
    });
    out
}

fn run_reduce_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, op: ShaderOp) -> Vec<f32> {
    let wg = ctx.wg_size;
    let num_blocks = n.div_ceil(wg as usize);
    let key = PipelineKey { op, wg_size: wg };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((num_blocks * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&params, true)]);
        #[allow(clippy::cast_possible_truncation)]
        dispatch_and_wait(ctx, pipeline, &bg, num_blocks as u32);
    });
    let bytes = readback(ctx, &out, (num_blocks * 4) as u64);
    // SAFETY: bytes from f32 buffer.
    unsafe { bytes_to_scalar::<f32>(&bytes) }
}

fn run_argreduce_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    n: usize,
    op: ShaderOp,
) -> (Vec<f32>, Vec<u32>) {
    let wg = ctx.wg_size;
    let num_blocks = n.div_ceil(wg as usize);
    let key = PipelineKey { op, wg_size: wg };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out_vals = GpuStorage::<f32>::empty_buf((num_blocks * 4) as u64);
    let out_idxs = GpuStorage::<f32>::empty_buf((num_blocks * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[
                (a, true),
                (&out_vals, false),
                (&out_idxs, false),
                (&params, true),
            ],
        );
        #[allow(clippy::cast_possible_truncation)]
        dispatch_and_wait(ctx, pipeline, &bg, num_blocks as u32);
    });
    let vbytes = readback(ctx, &out_vals, (num_blocks * 4) as u64);
    let ibytes = readback(ctx, &out_idxs, (num_blocks * 4) as u64);
    // SAFETY: bytes from valid f32 buffer.
    let vals: Vec<f32> = unsafe { bytes_to_scalar::<f32>(&vbytes) };
    let idxs: Vec<u32> = bytes_to_u32(&ibytes);
    (vals, idxs)
}

fn run_activation_1in(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    n: usize,
    extra_params: &[u32],
) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey { op, wg_size: wg };
    let mut pdata = vec![n as u32];
    pdata.extend_from_slice(extra_params);
    let p = params_buf(&pdata);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

fn run_rowwise_1in(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey { op, wg_size: wg };
    let p = params_buf(&[rows as u32, cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((rows * cols * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, rows as u32);
    });
    out
}

fn run_rowwise_reduce(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey { op, wg_size: wg };
    let p = params_buf(&[rows as u32, cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((rows * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, rows as u32);
    });
    out
}

pub(crate) fn gpu_silu<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_activation_1in(
        ctx,
        ShaderOp::ActivationSilu,
        &a.buffer,
        a.nrows * a.ncols,
        &[],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_mish<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_activation_1in(
        ctx,
        ShaderOp::ActivationMish,
        &a.buffer,
        a.nrows * a.ncols,
        &[],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_leaky_relu<T: Scalar>(a: &GpuStorage<T>, slope: T) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let slope_bits = (slope.to_f64() as f32).to_bits();
    let buf = run_activation_1in(
        ctx,
        ShaderOp::ActivationLeakyRelu,
        &a.buffer,
        a.nrows * a.ncols,
        &[slope_bits],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_elu<T: Scalar>(a: &GpuStorage<T>, alpha: T) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let alpha_bits = (alpha.to_f64() as f32).to_bits();
    let buf = run_activation_1in(
        ctx,
        ShaderOp::ActivationElu,
        &a.buffer,
        a.nrows * a.ncols,
        &[alpha_bits],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_hardswish<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_activation_1in(
        ctx,
        ShaderOp::ActivationHardswish,
        &a.buffer,
        a.nrows * a.ncols,
        &[],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_softmax<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_rowwise_1in(ctx, ShaderOp::Softmax, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_layer_norm<T: Scalar>(
    a: &GpuStorage<T>,
    gamma: &GpuStorage<T>,
    beta: &GpuStorage<T>,
    eps: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::LayerNorm,
        wg_size: wg,
    };
    let eps_bits = (eps.to_f64() as f32).to_bits();
    let p = params_buf(&[a.nrows as u32, a.ncols as u32, eps_bits]);
    let out = GpuStorage::<f32>::empty_buf((a.nrows * a.ncols * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[
                (&a.buffer, true),
                (&gamma.buffer, true),
                (&beta.buffer, true),
                (&out, false),
                (&p, true),
            ],
        );
        dispatch_and_wait(ctx, pipeline, &bg, a.nrows as u32);
    });
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_rms_norm<T: Scalar>(
    a: &GpuStorage<T>,
    gamma: &GpuStorage<T>,
    eps: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::RmsNorm,
        wg_size: wg,
    };
    let eps_bits = (eps.to_f64() as f32).to_bits();
    let p = params_buf(&[a.nrows as u32, a.ncols as u32, eps_bits]);
    let out = GpuStorage::<f32>::empty_buf((a.nrows * a.ncols * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[
                (&a.buffer, true),
                (&gamma.buffer, true),
                (&out, false),
                (&p, true),
            ],
        );
        dispatch_and_wait(ctx, pipeline, &bg, a.nrows as u32);
    });
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_sum_axis1<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_rowwise_reduce(ctx, ShaderOp::SumAxis1, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.nrows, 1, buf)
}

pub(crate) fn gpu_max_axis1<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_rowwise_reduce(ctx, ShaderOp::MaxAxis1, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.nrows, 1, buf)
}

pub(crate) fn gpu_embedding<T: Scalar>(
    indices: &GpuStorage<T>,
    weight: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = weight.ncols;
    let total = n_tokens * embed_dim;
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::Embedding,
        wg_size: wg,
    };
    let p = params_buf(&[n_tokens as u32, embed_dim as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[
                (&indices.buffer, true),
                (&weight.buffer, true),
                (&out, false),
                (&p, true),
            ],
        );
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(total, wg));
    });
    GpuStorage::from_buffer(n_tokens, embed_dim, out)
}

pub(crate) fn gpu_zeros<T: Scalar>(nrows: usize, ncols: usize) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_fill_zeros_f32(ctx, nrows * ncols);
    GpuStorage::from_buffer(nrows, ncols, buf)
}

pub(crate) fn gpu_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    // SAFETY: TypeId confirmed T == f32; reinterpret bits via pointer cast.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let f32_val: f32 = unsafe { *std::ptr::from_ref(&val).cast::<f32>() };
    let buf = run_fill_scalar_f32(ctx, nrows * ncols, f32_val);
    GpuStorage::from_buffer(nrows, ncols, buf)
}

pub(crate) fn gpu_identity<T: Scalar>(n: usize) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_fill_identity_f32(ctx, n);
    GpuStorage::from_buffer(n, n, buf)
}

pub(crate) fn gpu_from_fn<T: Scalar>(
    _nrows: usize,
    _ncols: usize,
    _f: impl FnMut(usize, usize) -> T,
) -> GpuStorage<T> {
    panic!("nabla: Tensor::from_fn is CPU-only; WGPU fallback is forbidden");
}

pub(crate) fn gpu_get<T: Scalar>(s: &GpuStorage<T>, r: usize, c: usize) -> T {
    assert_is_f32::<T>();
    let guard = s.fill_cache_mut();
    cache_ref(&guard, "cache populated")[r * s.ncols + c]
}

pub(crate) fn gpu_set<T: Scalar>(s: &mut GpuStorage<T>, r: usize, c: usize, v: T) {
    assert_is_f32::<T>();
    {
        let mut guard = s.fill_cache_mut();
        cache_mut(&mut guard, "cache populated")[r * s.ncols + c] = v;
    }
    let guard = lock_or_recover(&s.host_cache);
    let data = cache_ref(&guard, "cache populated");
    let ctx = get_context();
    // SAFETY: data is a valid [T] slice; reinterpreted as bytes for upload.
    let bytes = unsafe { scalar_to_bytes(data) };
    s.buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
}

pub(crate) fn gpu_clone<T: Scalar>(s: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let buf = run_copy_f32(ctx, &s.buffer, n);
    GpuStorage::from_buffer(s.nrows, s.ncols, buf)
}

macro_rules! impl_gpu_binary {
    ($name:ident, $op:expr) => {
        pub(crate) fn $name<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
            assert_is_f32::<T>();
            let ctx = get_context();
            let n = a.nrows * a.ncols;
            let buf = run_binary_f32(ctx, &a.buffer, &b.buffer, n, $op);
            GpuStorage::from_buffer(a.nrows, a.ncols, buf)
        }
    };
}

impl_gpu_binary!(gpu_add, 0);
impl_gpu_binary!(gpu_sub, 1);
impl_gpu_binary!(gpu_emul, 2);
impl_gpu_binary!(gpu_ediv, 3);
impl_gpu_binary!(gpu_atan2, 4);

pub(crate) fn gpu_neg<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let buf = run_unary_f32(ctx, &a.buffer, n, 0); // op 0 = neg
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_scale<T: Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let scalar: f32 = unsafe { *std::ptr::from_ref(&s).cast::<f32>() };
    let buf = run_scale_f32(ctx, &a.buffer, n, scalar);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn gpu_axpy_inplace<T: Scalar>(y: &mut GpuStorage<T>, alpha: T, x: &GpuStorage<T>) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = y.nrows * y.ncols;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let alpha_f32: f32 = unsafe { *std::ptr::from_ref(&alpha).cast::<f32>() };
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::Axpy,
        wg_size: wg,
    };
    let p = params_buf(&[n as u32, alpha_f32.to_bits()]);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(
            ctx,
            pipeline,
            &[(&y.buffer, false), (&x.buffer, true), (&p, true)],
        );
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    *lock_or_recover(&y.host_cache) = None;
}

pub(crate) fn gpu_transpose<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_transpose_f32(ctx, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.ncols, a.nrows, buf)
}

#[allow(dead_code)]
pub(crate) fn gpu_reshape_copy<T: Scalar>(
    a: &GpuStorage<T>,
    out_rows: usize,
    out_cols: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = out_rows * out_cols;
    let buf = run_copy_f32(ctx, &a.buffer, n);
    GpuStorage::from_buffer(out_rows, out_cols, buf)
}

#[allow(dead_code)]
pub(crate) fn gpu_submatrix<T: Scalar>(
    a: &GpuStorage<T>,
    row_start: usize,
    col_start: usize,
    out_rows: usize,
    out_cols: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let total = out_rows * out_cols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let out_rows = params[0];
    let out_cols = params[1];
    let src_cols = params[2];
    let row_start = params[3];
    let col_start = params[4];
    if i >= out_rows * out_cols {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    let src_r = r + row_start;
    let src_c = c + col_start;
    out[i] = a[src_r * src_cols + src_c];
}}
"
    );
    let params = params_buf(&[
        out_rows as u32,
        out_cols as u32,
        a.ncols as u32,
        row_start as u32,
        col_start as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

#[allow(dead_code)]
pub(crate) fn gpu_slice_set<T: Scalar>(
    dst: &mut GpuStorage<T>,
    row_start: usize,
    col_start: usize,
    src: &GpuStorage<T>,
) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let total = src.nrows * src.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let src_rows = params[0];
    let src_cols = params[1];
    let dst_cols = params[2];
    let row_start = params[3];
    let col_start = params[4];
    if i >= src_rows * src_cols {{ return; }}
    let r = i / src_cols;
    let c = i - r * src_cols;
    let dst_r = r + row_start;
    let dst_c = c + col_start;
    dst[dst_r * dst_cols + dst_c] = src[i];
}}
"
    );
    let params = params_buf(&[
        src.nrows as u32,
        src.ncols as u32,
        dst.ncols as u32,
        row_start as u32,
        col_start as u32,
    ]);
    run_custom_shader(
        ctx,
        &shader,
        &[(&src.buffer, true), (&dst.buffer, false), (&params, true)],
        workgroups(total, wg),
    );
    *lock_or_recover(&dst.host_cache) = None;
}

#[allow(dead_code)]
pub(crate) fn gpu_repeat<T: Scalar>(
    a: &GpuStorage<T>,
    row_reps: usize,
    col_reps: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let out_rows = a.nrows * row_reps;
    let out_cols = a.ncols * col_reps;
    let total = out_rows * out_cols;
    let ctx = get_context();
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let out_rows = params[0];
    let out_cols = params[1];
    let src_rows = params[2];
    let src_cols = params[3];
    if i >= out_rows * out_cols {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    let src_r = r % src_rows;
    let src_c = c % src_cols;
    out[i] = a[src_r * src_cols + src_c];
}}
"
    );
    let params = params_buf(&[
        out_rows as u32,
        out_cols as u32,
        a.nrows as u32,
        a.ncols as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

#[allow(dead_code)]
pub(crate) fn gpu_pad<T: Scalar>(
    a: &GpuStorage<T>,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
    value: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let out_rows = a.nrows + top + bottom;
    let out_cols = a.ncols + left + right;
    let total = out_rows * out_cols;
    let ctx = get_context();
    let wg = ctx.wg_size;
    // SAFETY: T is f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let val_f32: f32 = unsafe { *std::ptr::from_ref(&value).cast::<f32>() };
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let out_rows = params[0];
    let out_cols = params[1];
    let src_rows = params[2];
    let src_cols = params[3];
    let left = params[4];
    let top = params[5];
    let fill = bitcast<f32>(params[6]);
    if i >= out_rows * out_cols {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    if r >= top && r < top + src_rows && c >= left && c < left + src_cols {{
        let src_r = r - top;
        let src_c = c - left;
        out[i] = a[src_r * src_cols + src_c];
    }} else {{
        out[i] = fill;
    }}
}}
"
    );
    let params = params_buf(&[
        out_rows as u32,
        out_cols as u32,
        a.nrows as u32,
        a.ncols as u32,
        left as u32,
        top as u32,
        val_f32.to_bits(),
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

#[allow(dead_code)]
pub(crate) fn gpu_triu<T: Scalar>(a: &GpuStorage<T>, diagonal: isize) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let total = a.nrows * a.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let rows = params[0];
    let cols = params[1];
    let diag = i32(bitcast<i32>(params[2]));
    if i >= rows * cols {{ return; }}
    let r = i / cols;
    let c = i - r * cols;
    let keep = (i32(c) - i32(r)) >= diag;
    out[i] = select(0.0, a[i], keep);
}}
"
    );
    let diag_bits = (diagonal as i32) as u32;
    let params = params_buf(&[a.nrows as u32, a.ncols as u32, diag_bits]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_matmul<T: Scalar>(out: &mut GpuStorage<T>, a: &GpuStorage<T>, b: &GpuStorage<T>) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let (rows, kdim, cols) = (a.nrows, a.ncols, b.ncols);
    let buf = run_matmul_f32(ctx, &a.buffer, &b.buffer, rows, kdim, cols);
    out.buffer = buf;
    out.nrows = rows;
    out.ncols = cols;
    *lock_or_recover(&out.host_cache) = None;
}

pub(crate) fn gpu_powf<T: Scalar>(a: &GpuStorage<T>, p: T) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let power: f32 = unsafe { *std::ptr::from_ref(&p).cast::<f32>() };
    let buf = run_powf_f32(ctx, &a.buffer, n, power);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

macro_rules! impl_gpu_unary {
    ($name:ident, $op:expr) => {
        pub(crate) fn $name<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
            assert_is_f32::<T>();
            let ctx = get_context();
            let n = a.nrows * a.ncols;
            let buf = run_unary_f32(ctx, &a.buffer, n, $op);
            GpuStorage::from_buffer(a.nrows, a.ncols, buf)
        }
    };
}

impl_gpu_unary!(gpu_exp, 1);
impl_gpu_unary!(gpu_ln, 2);
impl_gpu_unary!(gpu_log1p, 3);
impl_gpu_unary!(gpu_sin, 4);
impl_gpu_unary!(gpu_cos, 5);
impl_gpu_unary!(gpu_tanh, 6);
impl_gpu_unary!(gpu_sqrt, 7);
impl_gpu_unary!(gpu_abs, 8);
impl_gpu_unary!(gpu_recip, 9);
impl_gpu_unary!(gpu_erf, 10);
impl_gpu_unary!(gpu_ceil, 11);
impl_gpu_unary!(gpu_floor, 12);
impl_gpu_unary!(gpu_round, 13);
impl_gpu_unary!(gpu_tan, 14);
impl_gpu_unary!(gpu_asin, 15);
impl_gpu_unary!(gpu_acos, 16);
impl_gpu_unary!(gpu_atan, 17);
impl_gpu_unary!(gpu_sinh, 18);
impl_gpu_unary!(gpu_cosh, 19);
impl_gpu_unary!(gpu_asinh, 20);
impl_gpu_unary!(gpu_acosh, 21);
impl_gpu_unary!(gpu_atanh, 22);
impl_gpu_unary!(gpu_log2, 23);
impl_gpu_unary!(gpu_log10, 24);

pub(crate) fn gpu_sum_all<T: Scalar>(s: &GpuStorage<T>) -> T {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let partials = run_reduce_f32(ctx, &s.buffer, n, ShaderOp::ReduceSum);
    let total: f32 = partials.iter().copied().sum();
    // SAFETY: T == f32 confirmed by assert_is_f32; same size/align.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    unsafe {
        *std::ptr::from_ref(&total).cast::<T>()
    }
}

pub(crate) fn gpu_max_all<T: Scalar>(s: &GpuStorage<T>) -> T {
    assert!(
        s.nrows > 0 && s.ncols > 0,
        "max_all: matrix must be non-empty"
    );
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let partials = run_reduce_f32(ctx, &s.buffer, n, ShaderOp::ReduceMax);
    let total: f32 = partials.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // SAFETY: T == f32 confirmed by assert_is_f32; same size/align.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    unsafe {
        *std::ptr::from_ref(&total).cast::<T>()
    }
}

pub(crate) fn gpu_min_all<T: Scalar>(s: &GpuStorage<T>) -> T {
    assert!(
        s.nrows > 0 && s.ncols > 0,
        "min_all: matrix must be non-empty"
    );
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let partials = run_reduce_f32(ctx, &s.buffer, n, ShaderOp::ReduceMin);
    let total: f32 = partials.iter().copied().fold(f32::INFINITY, f32::min);
    // SAFETY: T == f32 confirmed by assert_is_f32; same size/align.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    unsafe {
        *std::ptr::from_ref(&total).cast::<T>()
    }
}

pub(crate) fn gpu_argmax_all<T: Scalar>(s: &GpuStorage<T>) -> (usize, usize) {
    assert!(
        s.nrows > 0 && s.ncols > 0,
        "argmax_all: matrix must be non-empty"
    );
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let ncols = s.ncols;
    let (vals, idxs) = run_argreduce_f32(ctx, &s.buffer, n, ShaderOp::Argmax);
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (v, i) in vals.iter().zip(idxs.iter()) {
        // Exact equality is intentional: tie-break by index from GPU partial results.
        #[allow(clippy::float_cmp)]
        let tie = *v == best_v;
        if *v > best_v || (tie && *i < best_i) {
            best_v = *v;
            best_i = *i;
        }
    }
    let flat = best_i as usize;
    (flat / ncols, flat % ncols)
}

pub(crate) fn gpu_argmin_all<T: Scalar>(s: &GpuStorage<T>) -> (usize, usize) {
    assert!(
        s.nrows > 0 && s.ncols > 0,
        "argmin_all: matrix must be non-empty"
    );
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let ncols = s.ncols;
    let (vals, idxs) = run_argreduce_f32(ctx, &s.buffer, n, ShaderOp::Argmin);
    let mut best_v = f32::INFINITY;
    let mut best_i = 0u32;
    for (v, i) in vals.iter().zip(idxs.iter()) {
        // Exact equality is intentional: tie-break by index from GPU partial results.
        #[allow(clippy::float_cmp)]
        let tie = *v == best_v;
        if *v < best_v || (tie && *i < best_i) {
            best_v = *v;
            best_i = *i;
        }
    }
    let flat = best_i as usize;
    (flat / ncols, flat % ncols)
}

#[inline]
fn assert_is_f32<T: Scalar>() {
    assert!(
        TypeId::of::<T>() == TypeId::of::<f32>(),
        "GPU backend only supports f32; got a different scalar type"
    );
}

macro_rules! gpu_fwd_unary {
    ($($trait_fn:ident => $impl_fn:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $trait_fn<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
                $impl_fn::<T>(a)
            }
        )+
    };
}

macro_rules! gpu_fwd_binary {
    ($($trait_fn:ident => $impl_fn:ident),+ $(,)?) => {
        $(
            #[inline]
            fn $trait_fn<T: crate::scalar::Scalar>(
                a: &GpuStorage<T>,
                b: &GpuStorage<T>,
            ) -> GpuStorage<T> {
                $impl_fn::<T>(a, b)
            }
        )+
    };
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendCore for crate::backend::Gpu {
    type Storage<T: crate::scalar::Scalar> = GpuStorage<T>;

    #[inline]
    fn zeros<T: crate::scalar::Scalar>(r: usize, c: usize) -> GpuStorage<T> {
        gpu_zeros::<T>(r, c)
    }

    #[inline]
    fn fill<T: crate::scalar::Scalar>(r: usize, c: usize, val: T) -> GpuStorage<T> {
        gpu_fill::<T>(r, c, val)
    }

    #[inline]
    fn identity<T: crate::scalar::Scalar>(n: usize) -> GpuStorage<T> {
        gpu_identity::<T>(n)
    }

    #[inline]
    fn from_fn<T: crate::scalar::Scalar>(
        r: usize,
        c: usize,
        f: impl FnMut(usize, usize) -> T,
    ) -> GpuStorage<T> {
        gpu_from_fn::<T>(r, c, f)
    }

    #[inline]
    fn nrows<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> usize {
        s.nrows
    }

    #[inline]
    fn ncols<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> usize {
        s.ncols
    }

    #[inline]
    fn get<T: crate::scalar::Scalar>(s: &GpuStorage<T>, r: usize, c: usize) -> T {
        gpu_get::<T>(s, r, c)
    }

    #[inline]
    fn set<T: crate::scalar::Scalar>(s: &mut GpuStorage<T>, r: usize, c: usize, v: T) {
        gpu_set::<T>(s, r, c, v);
    }

    #[inline]
    fn prefetch<T: crate::scalar::Scalar>(storage: &GpuStorage<T>) {
        let _cache_guard = storage.fill_cache_mut();
    }

    #[inline]
    fn clone_storage<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_clone::<T>(s)
    }

    gpu_fwd_unary!(neg => gpu_neg, transpose => gpu_transpose);
    gpu_fwd_binary!(add => gpu_add, sub => gpu_sub);

    #[inline]
    fn scale<T: crate::scalar::Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
        gpu_scale::<T>(a, s)
    }

    #[inline]
    fn axpy_inplace<T: crate::scalar::Scalar>(y: &mut GpuStorage<T>, alpha: T, x: &GpuStorage<T>) {
        gpu_axpy_inplace::<T>(y, alpha, x);
    }
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendMath for crate::backend::Gpu {
    gpu_fwd_unary!(
        exp => gpu_exp, ln => gpu_ln, log1p => gpu_log1p,
        sin => gpu_sin, cos => gpu_cos, tan => gpu_tan, tanh => gpu_tanh,
        sqrt => gpu_sqrt, abs => gpu_abs, recip => gpu_recip,
        erf => gpu_erf, ceil => gpu_ceil, floor => gpu_floor, round => gpu_round,
        asin => gpu_asin, acos => gpu_acos, atan => gpu_atan,
        sinh => gpu_sinh, cosh => gpu_cosh, asinh => gpu_asinh, acosh => gpu_acosh, atanh => gpu_atanh,
        log2 => gpu_log2, log10 => gpu_log10,
    );
    gpu_fwd_binary!(emul => gpu_emul, ediv => gpu_ediv, atan2 => gpu_atan2);

    #[inline]
    fn powf<T: crate::scalar::Scalar>(a: &GpuStorage<T>, p: T) -> GpuStorage<T> {
        gpu_powf::<T>(a, p)
    }
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendReduce for crate::backend::Gpu {
    #[inline]
    fn sum_all<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> T {
        gpu_sum_all::<T>(s)
    }

    #[inline]
    fn max_all<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> T {
        gpu_max_all::<T>(s)
    }

    #[inline]
    fn min_all<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> T {
        gpu_min_all::<T>(s)
    }

    #[inline]
    fn argmax_all<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> (usize, usize) {
        gpu_argmax_all::<T>(s)
    }

    #[inline]
    fn argmin_all<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> (usize, usize) {
        gpu_argmin_all::<T>(s)
    }

    fn sum_axis1<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_sum_axis1(a)
    }

    fn max_axis1<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_max_axis1(a)
    }
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendShape for crate::backend::Gpu {}

#[cfg(feature = "gpu")]
impl crate::backend::BackendBlas for crate::backend::Gpu {
    #[inline]
    fn matmul_into<T: crate::scalar::Scalar>(
        out: &mut GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
    ) {
        gpu_matmul::<T>(out, a, b);
    }
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendNN for crate::backend::Gpu {
    fn silu<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_silu(a)
    }
    fn mish<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_mish(a)
    }
    fn leaky_relu<T: crate::scalar::Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
        gpu_leaky_relu(a, s)
    }
    fn elu<T: crate::scalar::Scalar>(a: &GpuStorage<T>, alpha: T) -> GpuStorage<T> {
        gpu_elu(a, alpha)
    }
    fn hardswish<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_hardswish(a)
    }
    fn softmax<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_softmax(a)
    }
    fn layer_norm<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        g: &GpuStorage<T>,
        b: &GpuStorage<T>,
        eps: T,
    ) -> GpuStorage<T> {
        gpu_layer_norm(a, g, b, eps)
    }
    fn rms_norm<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        g: &GpuStorage<T>,
        eps: T,
    ) -> GpuStorage<T> {
        gpu_rms_norm(a, g, eps)
    }
    fn embedding<T: crate::scalar::Scalar>(i: &GpuStorage<T>, w: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_embedding(i, w)
    }
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendFusion for crate::backend::Gpu {
    fn fuse_launch<T: crate::scalar::Scalar>(
        _inputs: &[*const u8],
        _nrows: usize,
        _ncols: usize,
        _cpu_fn: impl FnMut(usize, usize) -> T,
        _gpu_expr: &str,
        _kernel_hash: &str,
        _n_inputs: usize,
        _reg_estimate: usize,
    ) -> GpuStorage<T> {
        panic!("fuse_launch: not supported on wgpu backend; use cpu/cuda/hip backend")
    }

    fn mega_fuse_launch<'a, T: crate::scalar::Scalar>(
        _ops: &[(Vec<*const u8>, String, usize, bool)],
        _nrows: usize,
        _ncols: usize,
        _cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T + 'a>>,
        _kernel_hash: &str,
    ) -> Vec<GpuStorage<T>> {
        panic!("mega_fuse_launch: not supported on wgpu backend; use cpu/cuda/hip backend")
    }
}

#[allow(dead_code)]
pub(crate) const WGSL_BCSR_SPMM: &str = r"
@group(0) @binding(0) var<storage, read> row_ptrs: array<u32>;
@group(0) @binding(1) var<storage, read> col_idxs: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> params: array<u32>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {
    let nrows = params[0];
    let ncols = params[1];
    let k = params[2];
    let bs = params[3];
    let bs2 = bs * bs;
    let br = wg_id.x;
    let lr = gid.x % bs;
    let row = br * bs + lr;
    if row >= nrows { return; }

    let start = row_ptrs[br];
    let end = row_ptrs[br + 1u];

    for var c: u32 = 0u; c < k; c = c + 1u {
        var acc: f32 = 0.0;
        for var p: u32 = start; p < end; p = p + 1u {
            let bc = col_idxs[p];
            let block_base = p * bs2;
            for var lc: u32 = 0u; lc < bs; lc = lc + 1u {
                let col = bc * bs + lc;
                if col < ncols {
                    let a_val = values[block_base + lr * bs + lc];
                    acc = acc + a_val * x[col * k + c];
                }
            }
        }
        out[row * k + c] = acc;
    }
}
";
