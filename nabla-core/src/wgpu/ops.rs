use std::any::TypeId;

use wgpu::util::DeviceExt;

use crate::scalar::Scalar;

use super::shaders::*;
use super::storage::*;

#[cfg(feature = "wgpu-f16")]
use half::f16;

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
fn is_f32<T: Scalar>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f16<T: Scalar>() -> bool {
    #[cfg(feature = "wgpu-f16")]
    {
        TypeId::of::<T>() == TypeId::of::<f16>()
    }
    #[cfg(not(feature = "wgpu-f16"))]
    {
        false
    }
}

fn ensure_f16_supported() {
    #[cfg(feature = "wgpu-f16")]
    {
        let ctx = get_context();
        if !ctx.device.features().contains(wgpu::Features::SHADER_F16) {
            panic!("nabla: wgpu shader-f16 not supported on this device");
        }
    }
}

#[inline]
fn assert_is_f32_or_f16<T: Scalar>() {
    if is_f32::<T>() {
        return;
    }
    if is_f16::<T>() {
        ensure_f16_supported();
        return;
    }
    panic!("nabla: wgpu backend supports only f32 (and f16 with wgpu-f16)");
}

#[inline]
fn scalar_kind<T: Scalar>() -> ScalarKind {
    if is_f16::<T>() {
        ScalarKind::F16
    } else {
        ScalarKind::F32
    }
}

#[inline]
fn elem_size_bytes<T: Scalar>() -> u64 {
    core::mem::size_of::<T>() as u64
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
fn run_1in<T: Scalar>(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    n: usize,
    params: &[u32],
) -> wgpu::Buffer {
    assert_is_f32_or_f16::<T>();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let p = params_buf(params);
    let out = GpuStorage::<T>::empty_buf(n as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

#[allow(clippy::cast_possible_truncation)]
fn run_2in<T: Scalar>(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n: usize,
    params: &[u32],
) -> wgpu::Buffer {
    assert_is_f32_or_f16::<T>();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let p = params_buf(params);
    let out = GpuStorage::<T>::empty_buf(n as u64 * elem_size_bytes::<T>());
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
fn run_binary<T: Scalar>(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n: usize,
    op_type: u32,
) -> wgpu::Buffer {
    run_2in::<T>(ctx, ShaderOp::Binary, a, b, n, &[n as u32, op_type])
}

#[allow(clippy::cast_possible_truncation)]
fn run_unary<T: Scalar>(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    n: usize,
    op_type: u32,
) -> wgpu::Buffer {
    run_1in::<T>(ctx, ShaderOp::Unary, a, n, &[n as u32, op_type])
}

#[allow(clippy::cast_possible_truncation)]
fn run_scale<T: Scalar>(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, scalar: f32) -> wgpu::Buffer {
    run_1in::<T>(ctx, ShaderOp::Scale, a, n, &[n as u32, scalar.to_bits()])
}

#[allow(clippy::cast_possible_truncation)]
fn run_powf<T: Scalar>(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, power: f32) -> wgpu::Buffer {
    run_1in::<T>(ctx, ShaderOp::Powf, a, n, &[n as u32, power.to_bits()])
}

#[allow(clippy::cast_possible_truncation)]
fn run_transpose<T: Scalar>(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> wgpu::Buffer {
    run_1in::<T>(
        ctx,
        ShaderOp::Transpose,
        a,
        rows * cols,
        &[rows as u32, cols as u32],
    )
}

#[allow(clippy::cast_possible_truncation)]
fn run_copy<T: Scalar>(ctx: &GpuContext, a: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    run_1in::<T>(ctx, ShaderOp::Copy, a, n, &[n as u32])
}

fn run_matmul<T: Scalar>(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> wgpu::Buffer {
    // Large or moderately large matrices: register-tile software MMA
    let mn = m * n;
    if (m >= 64 && n >= 64 && k >= 32) || (mn >= 16_384 && k >= 64) {
        return run_matmul_register_tile::<T>(ctx, a, b, m, k, n);
    }
    let tile = select_matmul_tile(m, k, n);
    let tile_usize = tile as usize;
    let grid_cols = n.div_ceil(tile_usize);
    let grid_rows = m.div_ceil(tile_usize);
    let grid_size = grid_rows * grid_cols;
    let key = PipelineKey {
        op: ShaderOp::Matmul { tile },
        wg_size: tile,
        scalar: scalar_kind::<T>(),
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[m as u32, k as u32, n as u32, grid_cols as u32]);
    let out = GpuStorage::<T>::empty_buf(m as u64 * n as u64 * elem_size_bytes::<T>());
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

fn run_matmul_register_tile<T: Scalar>(
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
        scalar: scalar_kind::<T>(),
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[m as u32, k as u32, n as u32, grid_cols as u32]);
    let out = GpuStorage::<T>::empty_buf(m as u64 * n as u64 * elem_size_bytes::<T>());
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

fn run_fill_zeros<T: Scalar>(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::FillZeros,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<T>::empty_buf(n as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(&out, false), (&params, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

fn run_fill_scalar<T: Scalar>(ctx: &GpuContext, n: usize, val: f32) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::FillScalar,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, val.to_bits()]);
    let out = GpuStorage::<T>::empty_buf(n as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(&out, false), (&params, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

fn run_fill_identity<T: Scalar>(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::FillIdentity,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let total = n * n;
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<T>::empty_buf(total as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(&out, false), (&params, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(total, wg));
    });
    out
}

fn run_reduce_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, op: ShaderOp) -> Vec<f32> {
    let wg = ctx.wg_size;
    let num_blocks = n.div_ceil(wg as usize);
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: ScalarKind::F32,
    };
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
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: ScalarKind::F32,
    };
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

fn run_activation_1in<T: Scalar>(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    n: usize,
    extra_params: &[u32],
) -> wgpu::Buffer {
    assert_is_f32_or_f16::<T>();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let mut pdata = vec![n as u32];
    pdata.extend_from_slice(extra_params);
    let p = params_buf(&pdata);
    let out = GpuStorage::<T>::empty_buf(n as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

fn run_rowwise_1in<T: Scalar>(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> wgpu::Buffer {
    assert_is_f32_or_f16::<T>();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let p = params_buf(&[rows as u32, cols as u32]);
    let out = GpuStorage::<T>::empty_buf(rows as u64 * cols as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, rows as u32);
    });
    out
}

fn run_rowwise_reduce<T: Scalar>(
    ctx: &GpuContext,
    op: ShaderOp,
    a: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> wgpu::Buffer {
    assert_is_f32_or_f16::<T>();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let p = params_buf(&[rows as u32, cols as u32]);
    let out = GpuStorage::<T>::empty_buf(rows as u64 * elem_size_bytes::<T>());
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, rows as u32);
    });
    out
}

pub(crate) fn gpu_silu<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_activation_1in::<T>(
        ctx,
        ShaderOp::ActivationSilu,
        &a.buffer,
        a.nrows * a.ncols,
        &[],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_mish<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_activation_1in::<T>(
        ctx,
        ShaderOp::ActivationMish,
        &a.buffer,
        a.nrows * a.ncols,
        &[],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_leaky_relu<T: Scalar>(a: &GpuStorage<T>, slope: T) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let slope_bits = (slope.to_f64() as f32).to_bits();
    let buf = run_activation_1in::<T>(
        ctx,
        ShaderOp::ActivationLeakyRelu,
        &a.buffer,
        a.nrows * a.ncols,
        &[slope_bits],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_elu<T: Scalar>(a: &GpuStorage<T>, alpha: T) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let alpha_bits = (alpha.to_f64() as f32).to_bits();
    let buf = run_activation_1in::<T>(
        ctx,
        ShaderOp::ActivationElu,
        &a.buffer,
        a.nrows * a.ncols,
        &[alpha_bits],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_hardswish<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_activation_1in::<T>(
        ctx,
        ShaderOp::ActivationHardswish,
        &a.buffer,
        a.nrows * a.ncols,
        &[],
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_softmax<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_rowwise_1in::<T>(ctx, ShaderOp::Softmax, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_layer_norm<T: Scalar>(
    a: &GpuStorage<T>,
    gamma: &GpuStorage<T>,
    beta: &GpuStorage<T>,
    eps: T,
) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::LayerNorm,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let eps_bits = (eps.to_f64() as f32).to_bits();
    let p = params_buf(&[a.nrows as u32, a.ncols as u32, eps_bits]);
    let out = GpuStorage::<T>::empty_buf(a.nrows as u64 * a.ncols as u64 * elem_size_bytes::<T>());
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
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::RmsNorm,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let eps_bits = (eps.to_f64() as f32).to_bits();
    let p = params_buf(&[a.nrows as u32, a.ncols as u32, eps_bits]);
    let out = GpuStorage::<T>::empty_buf(a.nrows as u64 * a.ncols as u64 * elem_size_bytes::<T>());
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
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_rowwise_reduce::<T>(ctx, ShaderOp::SumAxis1, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.nrows, 1, buf)
}

pub(crate) fn gpu_max_axis1<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_rowwise_reduce::<T>(ctx, ShaderOp::MaxAxis1, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.nrows, 1, buf)
}

pub(crate) fn gpu_embedding<T: Scalar>(
    indices: &GpuStorage<T>,
    weight: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = weight.ncols;
    let total = n_tokens * embed_dim;
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::Embedding,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
    };
    let p = params_buf(&[n_tokens as u32, embed_dim as u32]);
    let out = GpuStorage::<T>::empty_buf(total as u64 * elem_size_bytes::<T>());
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
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_fill_zeros::<T>(ctx, nrows * ncols);
    GpuStorage::from_buffer(nrows, ncols, buf)
}

pub(crate) fn gpu_empty<T: Scalar>(nrows: usize, ncols: usize) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let n_bytes = (nrows * ncols * core::mem::size_of::<T>()) as u64;
    let buf = GpuStorage::<T>::empty_buf(n_bytes);
    GpuStorage::from_buffer(nrows, ncols, buf)
}

pub(crate) fn gpu_fill<T: Scalar>(nrows: usize, ncols: usize, val: T) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let f32_val = val.to_f64() as f32;
    let buf = run_fill_scalar::<T>(ctx, nrows * ncols, f32_val);
    GpuStorage::from_buffer(nrows, ncols, buf)
}

pub(crate) fn gpu_from_vec<T: Scalar>(nrows: usize, ncols: usize, data: Vec<T>) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    // SAFETY: data is a valid [T] slice; reinterpreted as bytes for upload.
    let bytes = unsafe { scalar_to_bytes(&data) };
    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    let storage = GpuStorage::from_buffer(nrows, ncols, buffer);
    *lock_or_recover(&storage.host_cache) = Some(data);
    storage
}

pub(crate) fn gpu_identity<T: Scalar>(n: usize) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_fill_identity::<T>(ctx, n);
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
    assert_is_f32_or_f16::<T>();
    let guard = s.fill_cache_mut();
    cache_ref(&guard, "cache populated")[r * s.ncols + c]
}

pub(crate) fn gpu_set<T: Scalar>(s: &mut GpuStorage<T>, r: usize, c: usize, v: T) {
    assert_is_f32_or_f16::<T>();
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
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let buf = run_copy::<T>(ctx, &s.buffer, n);
    GpuStorage::from_buffer(s.nrows, s.ncols, buf)
}

macro_rules! impl_gpu_binary {
    ($name:ident, $op:expr) => {
        pub(crate) fn $name<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
            assert_is_f32_or_f16::<T>();
            let ctx = get_context();
            let n = a.nrows * a.ncols;
            let buf = run_binary::<T>(ctx, &a.buffer, &b.buffer, n, $op);
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
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let buf = run_unary::<T>(ctx, &a.buffer, n, 0); // op 0 = neg
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_scale<T: Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let scalar = s.to_f64() as f32;
    let buf = run_scale::<T>(ctx, &a.buffer, n, scalar);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn gpu_axpy_inplace<T: Scalar>(y: &mut GpuStorage<T>, alpha: T, x: &GpuStorage<T>) {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n = y.nrows * y.ncols;
    let alpha_f32 = alpha.to_f64() as f32;
    let wg = ctx.wg_size;
    let key = PipelineKey {
        op: ShaderOp::Axpy,
        wg_size: wg,
        scalar: scalar_kind::<T>(),
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
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let buf = run_transpose::<T>(ctx, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.ncols, a.nrows, buf)
}

#[allow(dead_code)]
pub(crate) fn gpu_reshape_copy<T: Scalar>(
    a: &GpuStorage<T>,
    out_rows: usize,
    out_cols: usize,
) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n = out_rows * out_cols;
    let buf = run_copy::<T>(ctx, &a.buffer, n);
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

pub(crate) fn gpu_expand_into<T: Scalar>(
    out: &mut GpuStorage<T>,
    src: &GpuStorage<T>,
    src_rows: usize,
    src_cols: usize,
) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let total = out.nrows * out.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> src: array<f32>;
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
    let sr = select(r, 0u, src_rows == 1u);
    let sc = select(c, 0u, src_cols == 1u);
    out[i] = src[sr * src_cols + sc];
}}
"
    );
    let params = params_buf(&[
        out.nrows as u32,
        out.ncols as u32,
        src_rows as u32,
        src_cols as u32,
    ]);
    run_custom_shader(
        ctx,
        &shader,
        &[(&src.buffer, true), (&out.buffer, false), (&params, true)],
        workgroups(total, wg),
    );
    *lock_or_recover(&out.host_cache) = None;
}

pub(crate) fn gpu_one_hot_from_indices<T: Scalar>(
    indices: &GpuStorage<T>,
    n_classes: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let rows = indices.nrows;
    let cols = n_classes;
    let total = rows * cols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> idx: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let rows = params[0];
    let cols = params[1];
    let idx_cols = params[2];
    if i >= rows * cols {{ return; }}
    let r = i / cols;
    let c = i - r * cols;
    let idx_f = idx[r * idx_cols];
    let idx_u = u32(max(idx_f, 0.0));
    out[i] = select(0.0, 1.0, c == idx_u);
}}
"
    );
    let params = params_buf(&[rows as u32, cols as u32, indices.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&indices.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(rows, cols, out)
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

pub(crate) fn gpu_tril<T: Scalar>(a: &GpuStorage<T>, diagonal: isize) -> GpuStorage<T> {
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
    let keep = (i32(c) - i32(r)) <= diag;
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

pub(crate) fn gpu_roll<T: Scalar>(a: &GpuStorage<T>, shift: isize, axis: usize) -> GpuStorage<T> {
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
    let shift = i32(bitcast<i32>(params[2]));
    let axis = params[3];
    if i >= rows * cols {{ return; }}
    let r = i / cols;
    let c = i - r * cols;
    var sr: i32 = i32(r);
    var sc: i32 = i32(c);
    if axis == 0u {{
        sr = i32(r) - shift;
        let dim = i32(rows);
        sr = sr % dim;
        if sr < 0 {{ sr = sr + dim; }}
    }} else {{
        sc = i32(c) - shift;
        let dim = i32(cols);
        sc = sc % dim;
        if sc < 0 {{ sc = sc + dim; }}
    }}
    let src_idx = u32(sr) * cols + u32(sc);
    out[i] = a[src_idx];
}}
"
    );
    let shift_bits = (shift as i32) as u32;
    let params = params_buf(&[a.nrows as u32, a.ncols as u32, shift_bits, axis as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_flip<T: Scalar>(a: &GpuStorage<T>, axis: usize) -> GpuStorage<T> {
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
    let axis = params[2];
    if i >= rows * cols {{ return; }}
    let r = i / cols;
    let c = i - r * cols;
    var sr = r;
    var sc = c;
    if axis == 0u {{
        sr = rows - 1u - r;
    }} else {{
        sc = cols - 1u - c;
    }}
    out[i] = a[sr * cols + sc];
}}
"
    );
    let params = params_buf(&[a.nrows as u32, a.ncols as u32, axis as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_from_diag<T: Scalar>(v: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = v.nrows.max(v.ncols);
    let total = n * n;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> v: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    let v_rows = params[1];
    let v_cols = params[2];
    if i >= n * n {{ return; }}
    let r = i / n;
    let c = i - r * n;
    if r == c {{
        let idx = select(r, c, v_rows == 1u);
        out[i] = v[idx];
    }} else {{
        out[i] = 0.0;
    }}
}}
"
    );
    let params = params_buf(&[n as u32, v.nrows as u32, v.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&v.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(n, n, out)
}

pub(crate) fn gpu_gather_rows<T: Scalar>(a: &GpuStorage<T>, indices: &[usize]) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let idx: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
    let idx_storage = GpuStorage::<f32>::upload(indices.len(), 1, idx);
    let ctx = get_context();
    let out_rows = indices.len();
    let out_cols = a.ncols;
    let total = out_rows * out_cols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let out_rows = params[0];
    let out_cols = params[1];
    let src_cols = params[2];
    if i >= out_rows * out_cols {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    let src_r = u32(round(idx[r]));
    out[i] = a[src_r * src_cols + c];
}}
"
    );
    let params = params_buf(&[out_rows as u32, out_cols as u32, a.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&idx_storage.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

pub(crate) fn gpu_gather<T: Scalar>(
    a: &GpuStorage<T>,
    axis: usize,
    index: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let out_rows = index.nrows;
    let out_cols = index.ncols;
    let total = out_rows * out_cols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let out_rows = params[0];
    let out_cols = params[1];
    let in_cols = params[2];
    let axis = params[3];
    if i >= out_rows * out_cols {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    let index_val = u32(round(idx[i]));
    if axis == 0u {{
        out[i] = a[index_val * in_cols + c];
    }} else {{
        out[i] = a[r * in_cols + index_val];
    }}
}}
"
    );
    let params = params_buf(&[
        out_rows as u32,
        out_cols as u32,
        a.ncols as u32,
        axis as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&index.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

pub(crate) fn gpu_scatter<T: Scalar>(
    a: &GpuStorage<T>,
    axis: usize,
    index: &GpuStorage<T>,
    src: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let out = gpu_clone(a);
    let total = src.nrows * src.ncols;
    let ctx = get_context();
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let src_rows = params[0];
    let src_cols = params[1];
    let out_cols = params[2];
    let axis = params[3];
    if i >= src_rows * src_cols {{ return; }}
    let r = i / src_cols;
    let c = i - r * src_cols;
    let index_val = u32(round(idx[i]));
    if axis == 0u {{
        out[index_val * out_cols + c] = src[i];
    }} else {{
        out[r * out_cols + index_val] = src[i];
    }}
}}
"
    );
    let params = params_buf(&[
        src.nrows as u32,
        src.ncols as u32,
        out.ncols as u32,
        axis as u32,
    ]);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&src.buffer, true),
            (&index.buffer, true),
            (&out.buffer, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    *lock_or_recover(&out.host_cache) = None;
    out
}

pub(crate) fn gpu_index_select<T: Scalar>(
    a: &GpuStorage<T>,
    axis: usize,
    index: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let k = index.nrows * index.ncols;
    let (out_rows, out_cols) = if axis == 0 {
        (k, a.ncols)
    } else {
        (a.nrows, k)
    };
    let total = out_rows * out_cols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let out_rows = params[0];
    let out_cols = params[1];
    let in_cols = params[2];
    let axis = params[3];
    if i >= out_rows * out_cols {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    if axis == 0u {{
        let src_r = u32(round(idx[r]));
        out[i] = a[src_r * in_cols + c];
    }} else {{
        let src_c = u32(round(idx[c]));
        out[i] = a[r * in_cols + src_c];
    }}
}}
"
    );
    let params = params_buf(&[
        out_rows as u32,
        out_cols as u32,
        a.ncols as u32,
        axis as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&index.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

pub(crate) fn gpu_sort_rows<T: Scalar>(
    a: &GpuStorage<T>,
    descending: bool,
) -> (GpuStorage<T>, GpuStorage<T>) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let rows = a.nrows;
    let cols = a.ncols;
    let total = rows * cols;
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    let idx = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    let desc_u = if descending { 1u32 } else { 0u32 };
    let shader = r#"
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = params[0];
    let cols = params[1];
    let desc = params[2];
    if row >= rows { return; }
    let base = row * cols;
    for (var c: u32 = 0u; c < cols; c++) {
        out[base + c] = inp[base + c];
        idx[base + c] = f32(c);
    }
    for (var i: u32 = 0u; i < cols; i++) {
        for (var j: u32 = 0u; j + 1u < cols - i; j++) {
            let a_idx = base + j;
            let b_idx = base + j + 1u;
            let va = out[a_idx];
            let vb = out[b_idx];
            let swap = select(va > vb, va < vb, desc == 1u);
            if swap {
                out[a_idx] = vb;
                out[b_idx] = va;
                let ia = idx[a_idx];
                idx[a_idx] = idx[b_idx];
                idx[b_idx] = ia;
            }
        }
    }
}
"#;
    let params = params_buf(&[rows as u32, cols as u32, desc_u]);
    run_custom_shader(
        ctx,
        shader,
        &[
            (&a.buffer, true),
            (&out, false),
            (&idx, false),
            (&params, true),
        ],
        rows as u32,
    );
    (
        GpuStorage::from_buffer(rows, cols, out),
        GpuStorage::from_buffer(rows, cols, idx),
    )
}

pub(crate) fn gpu_topk_rows<T: Scalar>(
    a: &GpuStorage<T>,
    k: usize,
) -> (GpuStorage<T>, GpuStorage<T>) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let rows = a.nrows;
    let cols = a.ncols;
    let out_n = rows * k;
    let out = GpuStorage::<f32>::empty_buf((out_n * 4) as u64);
    let idx = GpuStorage::<f32>::empty_buf((out_n * 4) as u64);
    // O(n·k) insertion-based topk per row
    let shader = r#"
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_idx: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = params[0];
    let cols = params[1];
    let k = params[2];
    if row >= rows { return; }
    let base_in = row * cols;
    let base_out = row * k;
    for (var i: u32 = 0u; i < k; i++) {
        out_val[base_out + i] = inp[base_in + i];
        out_idx[base_out + i] = f32(i);
    }
    for (var i: u32 = 1u; i < k; i++) {
        let tv = out_val[base_out + i];
        let ti = out_idx[base_out + i];
        var j: i32 = i32(i) - 1;
        while j >= 0 && out_val[base_out + u32(j)] < tv {
            out_val[base_out + u32(j + 1)] = out_val[base_out + u32(j)];
            out_idx[base_out + u32(j + 1)] = out_idx[base_out + u32(j)];
            j--;
        }
        out_val[base_out + u32(j + 1)] = tv;
        out_idx[base_out + u32(j + 1)] = ti;
    }
    for (var c: u32 = k; c < cols; c++) {
        let val = inp[base_in + c];
        if val > out_val[base_out + k - 1u] {
            var j: i32 = i32(k) - 2;
            while j >= 0 && out_val[base_out + u32(j)] < val {
                out_val[base_out + u32(j + 1)] = out_val[base_out + u32(j)];
                out_idx[base_out + u32(j + 1)] = out_idx[base_out + u32(j)];
                j--;
            }
            out_val[base_out + u32(j + 1)] = val;
            out_idx[base_out + u32(j + 1)] = f32(c);
        }
    }
}
"#;
    let params = params_buf(&[rows as u32, cols as u32, k as u32]);
    run_custom_shader(
        ctx,
        shader,
        &[
            (&a.buffer, true),
            (&out, false),
            (&idx, false),
            (&params, true),
        ],
        rows as u32,
    );
    (
        GpuStorage::from_buffer(rows, k, out),
        GpuStorage::from_buffer(rows, k, idx),
    )
}

pub(crate) fn gpu_meshgrid<T: Scalar>(
    x: &GpuStorage<T>,
    y: &GpuStorage<T>,
) -> (GpuStorage<T>, GpuStorage<T>) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let nx = x.nrows * x.ncols;
    let ny = y.nrows * y.ncols;
    let out_rows = ny;
    let out_cols = nx;
    let total = out_rows * out_cols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_y: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nx = params[0];
    let ny = params[1];
    let out_cols = params[2];
    if i >= nx * ny {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    out_x[i] = x[c];
    out_y[i] = y[r];
}}
"
    );
    let params = params_buf(&[nx as u32, ny as u32, out_cols as u32]);
    let out_x = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    let out_y = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&x.buffer, true),
            (&y.buffer, true),
            (&out_x, false),
            (&out_y, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    (
        GpuStorage::from_buffer(out_rows, out_cols, out_x),
        GpuStorage::from_buffer(out_rows, out_cols, out_y),
    )
}

pub(crate) fn gpu_scatter_add_dim0<T: Scalar>(
    dst: &mut GpuStorage<T>,
    indices: &[usize],
    src: &GpuStorage<T>,
) {
    assert_is_f32::<T>();
    let idx: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
    let idx_storage = GpuStorage::<f32>::upload(indices.len(), 1, idx);
    let ctx = get_context();
    let total = dst.nrows * dst.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let dst_rows = params[0];
    let dst_cols = params[1];
    let src_rows = params[2];
    let src_cols = params[3];
    if i >= dst_rows * dst_cols {{ return; }}
    let r = i / dst_cols;
    let c = i - r * dst_cols;
    var acc: f32 = dst[i];
    for (var sr: u32 = 0u; sr < src_rows; sr++) {{
        let idx_r = u32(round(idx[sr]));
        if idx_r == r {{
            acc = acc + src[sr * src_cols + c];
        }}
    }}
    dst[i] = acc;
}}
"
    );
    let params = params_buf(&[
        dst.nrows as u32,
        dst.ncols as u32,
        src.nrows as u32,
        src.ncols as u32,
    ]);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&src.buffer, true),
            (&idx_storage.buffer, true),
            (&dst.buffer, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    *lock_or_recover(&dst.host_cache) = None;
}

pub(crate) fn gpu_kron<T: Scalar>(
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
    m: usize,
    n: usize,
    p: usize,
    q: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let out_rows = m * p;
    let out_cols = n * q;
    let total = out_rows * out_cols;
    let ctx = get_context();
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let m = params[0];
    let n = params[1];
    let p = params[2];
    let q = params[3];
    let out_cols = params[4];
    if i >= (m * p) * (n * q) {{ return; }}
    let r = i / out_cols;
    let c = i - r * out_cols;
    let ar = r / p;
    let ac = c / q;
    let br = r - ar * p;
    let bc = c - ac * q;
    out[i] = a[ar * n + ac] * b[br * q + bc];
}}
"
    );
    let params = params_buf(&[m as u32, n as u32, p as u32, q as u32, out_cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&b.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(out_rows, out_cols, out)
}

pub(crate) fn gpu_diag<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows.min(a.ncols);
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    let cols = params[1];
    if i >= n {{ return; }}
    out[i] = a[i * cols + i];
}}
"
    );
    let params = params_buf(&[n as u32, a.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(n, 1, out)
}

pub(crate) fn gpu_cumsum_axis1<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let shader = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = params[0];
    let cols = params[1];
    if row >= rows { return; }
    var acc: f32 = 0.0;
    let base = row * cols;
    for (var c: u32 = 0u; c < cols; c++) {
        acc = acc + a[base + c];
        out[base + c] = acc;
    }
}
"#;
    let params = params_buf(&[a.nrows as u32, a.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((a.nrows * a.ncols * 4) as u64);
    run_custom_shader(
        ctx,
        shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        a.nrows as u32,
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_cumprod_axis1<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let shader = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = params[0];
    let cols = params[1];
    if row >= rows { return; }
    var acc: f32 = 1.0;
    let base = row * cols;
    for (var c: u32 = 0u; c < cols; c++) {
        acc = acc * a[base + c];
        out[base + c] = acc;
    }
}
"#;
    let params = params_buf(&[a.nrows as u32, a.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((a.nrows * a.ncols * 4) as u64);
    run_custom_shader(
        ctx,
        shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        a.nrows as u32,
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_matmul<T: Scalar>(out: &mut GpuStorage<T>, a: &GpuStorage<T>, b: &GpuStorage<T>) {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let (rows, kdim, cols) = (a.nrows, a.ncols, b.ncols);
    let buf = run_matmul::<T>(ctx, &a.buffer, &b.buffer, rows, kdim, cols);
    out.buffer = buf;
    out.nrows = rows;
    out.ncols = cols;
    *lock_or_recover(&out.host_cache) = None;
}

pub(crate) fn gpu_matmul_tn<T: Scalar>(
    out: &mut GpuStorage<T>,
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) {
    let a_t = gpu_transpose(a);
    gpu_matmul(out, &a_t, b);
}

pub(crate) fn gpu_matmul_nt<T: Scalar>(
    out: &mut GpuStorage<T>,
    a: &GpuStorage<T>,
    b: &GpuStorage<T>,
) {
    let b_t = gpu_transpose(b);
    gpu_matmul(out, a, &b_t);
}

pub(crate) fn gpu_add_bias_row<T: Scalar>(
    a: &GpuStorage<T>,
    bias: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let rows = params[0];
    let cols = params[1];
    if i >= rows * cols {{ return; }}
    let r = i / cols;
    let c = i - r * cols;
    out[i] = a[i] + bias[c];
}}
"
    );
    let params = params_buf(&[a.nrows as u32, a.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&bias.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_gelu_tanh_approx<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let half = T::from_f64(0.5);
    let k = T::from_f64(0.797_884_560_8);
    let c = T::from_f64(0.044_715);
    let x2 = gpu_emul(a, a);
    let x3 = gpu_emul(&x2, a);
    let cx3 = gpu_scale(&x3, c);
    let x_plus = gpu_add(a, &cx3);
    let t = gpu_scale(&x_plus, k);
    let tanh = gpu_tanh(&t);
    let ones = gpu_fill(a.nrows, a.ncols, T::one());
    let one_plus = gpu_add(&ones, &tanh);
    let x_half = gpu_scale(a, half);
    gpu_emul(&x_half, &one_plus)
}

pub(crate) fn gpu_group_norm<T: Scalar>(
    a: &GpuStorage<T>,
    gamma: &GpuStorage<T>,
    beta: &GpuStorage<T>,
    groups: usize,
    eps: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let rows = a.nrows;
    let cols = a.ncols;
    let g_size = cols / groups;
    let total = rows * cols;
    let wg = ctx.wg_size;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let eps_f32: f32 = unsafe { *std::ptr::from_ref(&eps).cast::<f32>() };
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let rows = params[0];
    let cols = params[1];
    let g_size = params[2];
    let eps = bitcast<f32>(params[3]);
    if i >= rows * cols {{ return; }}
    let r = i / cols;
    let c = i - r * cols;
    let g = c / g_size;
    let g_start = g * g_size;
    var sum: f32 = 0.0;
    var sumsq: f32 = 0.0;
    for (var j: u32 = 0u; j < g_size; j = j + 1u) {{
        let v = a[r * cols + g_start + j];
        sum = sum + v;
        sumsq = sumsq + v * v;
    }}
    let inv = 1.0 / f32(g_size);
    let mean = sum * inv;
    let var = sumsq * inv - mean * mean;
    let x = a[i];
    let norm = (x - mean) / sqrt(var + eps);
    out[i] = norm * gamma[c] + beta[c];
}}
"
    );
    let params = params_buf(&[rows as u32, cols as u32, g_size as u32, eps_f32.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&gamma.buffer, true),
            (&beta.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(rows, cols, out)
}

pub(crate) fn gpu_batch_norm_train<T: Scalar>(
    a: &GpuStorage<T>,
    gamma: &GpuStorage<T>,
    beta: &GpuStorage<T>,
    running_mean: &mut GpuStorage<T>,
    running_var: &mut GpuStorage<T>,
    eps: T,
    momentum: T,
    training: bool,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let rows = a.nrows;
    let cols = a.ncols;
    let inv_rows = T::from_f64(1.0 / rows as f64);
    let (mean_row, var_row) = if training {
        let a_t = gpu_transpose(a);
        let sum_cols = gpu_sum_axis1(&a_t);
        let mean_col = gpu_scale(&sum_cols, inv_rows);
        let mean_row = gpu_transpose(&mean_col);
        let mut mean_exp = gpu_zeros(rows, cols);
        gpu_expand_into(&mut mean_exp, &mean_row, 1, cols);
        let diff = gpu_sub(a, &mean_exp);
        let diff_sq = gpu_emul(&diff, &diff);
        let diff_t = gpu_transpose(&diff_sq);
        let sum_var = gpu_sum_axis1(&diff_t);
        let var_col = gpu_scale(&sum_var, inv_rows);
        let var_row = gpu_transpose(&var_col);
        let one = T::one();
        let one_minus = one - momentum;
        let rm_scaled = gpu_scale(running_mean, one_minus);
        let mean_scaled = gpu_scale(&mean_row, momentum);
        *running_mean = gpu_add(&rm_scaled, &mean_scaled);
        let rv_scaled = gpu_scale(running_var, one_minus);
        let var_scaled = gpu_scale(&var_row, momentum);
        *running_var = gpu_add(&rv_scaled, &var_scaled);
        (mean_row, var_row)
    } else {
        (gpu_clone(running_mean), gpu_clone(running_var))
    };
    let mut mean_exp = gpu_zeros(rows, cols);
    gpu_expand_into(&mut mean_exp, &mean_row, 1, cols);
    let mut var_exp = gpu_zeros(rows, cols);
    gpu_expand_into(&mut var_exp, &var_row, 1, cols);
    let eps_mat = gpu_fill(rows, cols, eps);
    let var_eps = gpu_add(&var_exp, &eps_mat);
    let denom = gpu_sqrt(&var_eps);
    let diff = gpu_sub(a, &mean_exp);
    let norm = gpu_ediv(&diff, &denom);
    let mut gamma_exp = gpu_zeros(rows, cols);
    gpu_expand_into(&mut gamma_exp, gamma, 1, cols);
    let mut beta_exp = gpu_zeros(rows, cols);
    gpu_expand_into(&mut beta_exp, beta, 1, cols);
    let scaled = gpu_emul(&norm, &gamma_exp);
    gpu_add(&scaled, &beta_exp)
}

pub(crate) fn gpu_cross_entropy_fused<T: Scalar>(
    input: &GpuStorage<T>,
    target: &GpuStorage<T>,
    n: usize,
    c: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> target: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    let n = params[0];
    let c = params[1];
    let tcols = params[2];
    if row >= n {{ return; }}
    var maxv: f32 = -3.4028235e38;
    let base = row * c;
    for (var j: u32 = 0u; j < c; j = j + 1u) {{
        let v = input[base + j];
        if v > maxv {{ maxv = v; }}
    }}
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < c; j = j + 1u) {{
        sum = sum + exp(input[base + j] - maxv);
    }}
    let t = u32(max(target[row * tcols], 0.0));
    let logit = input[base + t];
    out[row] = -(logit - maxv - log(sum));
}}
"
    );
    let params = params_buf(&[n as u32, c as u32, target.ncols as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&input.buffer, true),
            (&target.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    let losses = GpuStorage::<T>::from_buffer(n, 1, out);
    let total = gpu_sum_all(&losses);
    let mean = total / T::from_f64(n as f64);
    gpu_fill(1, 1, mean)
}

pub(crate) fn gpu_embedding_backward<T: Scalar>(
    indices: &GpuStorage<T>,
    grad: &GpuStorage<T>,
    vocab: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n_tokens = indices.nrows * indices.ncols;
    let embed_dim = grad.ncols;
    let total = vocab * embed_dim;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> idx: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let vocab = params[0];
    let embed = params[1];
    let n_tokens = params[2];
    let idx_cols = params[3];
    if i >= vocab * embed {{ return; }}
    let row = i / embed;
    let col = i - row * embed;
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < n_tokens; t = t + 1u) {{
        let id = u32(max(idx[t * idx_cols], 0.0));
        if id == row {{
            acc = acc + grad[t * embed + col];
        }}
    }}
    out[i] = acc;
}}
"
    );
    let params = params_buf(&[
        vocab as u32,
        embed_dim as u32,
        n_tokens as u32,
        indices.ncols as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&indices.buffer, true),
            (&grad.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(vocab, embed_dim, out)
}

pub(crate) fn gpu_relu_backward<T: Scalar>(
    grad: &GpuStorage<T>,
    input: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = grad.nrows * grad.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    if i >= total {{ return; }}
    out[i] = select(0.0, grad[i], input[i] > 0.0);
}}
"
    );
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&grad.buffer, true),
            (&input.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(grad.nrows, grad.ncols, out)
}

pub(crate) fn gpu_leaky_relu_backward<T: Scalar>(
    grad: &GpuStorage<T>,
    input: &GpuStorage<T>,
    alpha: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = grad.nrows * grad.ncols;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let alpha_f32: f32 = unsafe { *std::ptr::from_ref(&alpha).cast::<f32>() };
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    let alpha = bitcast<f32>(params[1]);
    if i >= total {{ return; }}
    let g = grad[i];
    out[i] = select(alpha * g, g, input[i] > 0.0);
}}
"
    );
    let params = params_buf(&[n as u32, alpha_f32.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&grad.buffer, true),
            (&input.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(grad.nrows, grad.ncols, out)
}

pub(crate) fn gpu_elu_backward<T: Scalar>(
    grad: &GpuStorage<T>,
    input: &GpuStorage<T>,
    alpha: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = grad.nrows * grad.ncols;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let alpha_f32: f32 = unsafe { *std::ptr::from_ref(&alpha).cast::<f32>() };
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    let alpha = bitcast<f32>(params[1]);
    if i >= total {{ return; }}
    let g = grad[i];
    let x = input[i];
    out[i] = select(g * alpha * exp(x), g, x > 0.0);
}}
"
    );
    let params = params_buf(&[n as u32, alpha_f32.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&grad.buffer, true),
            (&input.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(grad.nrows, grad.ncols, out)
}

pub(crate) fn gpu_gelu_backward<T: Scalar>(
    grad: &GpuStorage<T>,
    input: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = grad.nrows * grad.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    if i >= total {{ return; }}
    let g = grad[i];
    let x = input[i];
    let k = 0.7978845608;
    let c = 0.044715;
    let x2 = x * x;
    let x3 = x2 * x;
    let t = k * (x + c * x3);
    let th = tanh(t);
    let sech2 = 1.0 - th * th;
    let dt = k * (1.0 + 3.0 * c * x2);
    let dx = 0.5 * (1.0 + th) + 0.5 * x * sech2 * dt;
    out[i] = g * dx;
}}
"
    );
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&grad.buffer, true),
            (&input.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(grad.nrows, grad.ncols, out)
}

pub(crate) fn gpu_abs_backward<T: Scalar>(
    grad: &GpuStorage<T>,
    input: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = grad.nrows * grad.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    if i >= total {{ return; }}
    let g = grad[i];
    let x = input[i];
    if x > 0.0 {{
        out[i] = g;
    }} else if x < 0.0 {{
        out[i] = -g;
    }} else {{
        out[i] = 0.0;
    }}
}}
"
    );
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&grad.buffer, true),
            (&input.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(grad.nrows, grad.ncols, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_max_pool2d<T: Scalar>(
    a: &GpuStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nc = params[0];
    let h = params[1];
    let w = params[2];
    let kh = params[3];
    let kw = params[4];
    let sh = params[5];
    let sw = params[6];
    let ph = params[7];
    let pw = params[8];
    let out_h = params[9];
    let out_w = params[10];
    if i >= nc * out_h * out_w {{ return; }}
    let n = i / (out_h * out_w);
    let op = i - n * out_h * out_w;
    let oh = op / out_w;
    let ow = op - oh * out_w;
    var best: f32 = -3.4028235e38;
    var found: bool = false;
    for (var khr: u32 = 0u; khr < kh; khr = khr + 1u) {{
        for (var kwc: u32 = 0u; kwc < kw; kwc = kwc + 1u) {{
            let ih = oh * sh + khr;
            let iw = ow * sw + kwc;
            if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {{
                let v = a[n * h * w + (ih - ph) * w + (iw - pw)];
                if !found || v > best {{
                    best = v;
                    found = true;
                }}
            }}
        }}
    }}
    out[i] = select(0.0, best, found);
}}
"
    );
    let params = params_buf(&[
        nc as u32,
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
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(nc, out_h * out_w, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_max_pool2d_with_indices<T: Scalar>(
    a: &GpuStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> (GpuStorage<T>, GpuStorage<T>) {
    assert_is_f32::<T>();
    let ctx = get_context();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idxs: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nc = params[0];
    let h = params[1];
    let w = params[2];
    let kh = params[3];
    let kw = params[4];
    let sh = params[5];
    let sw = params[6];
    let ph = params[7];
    let pw = params[8];
    let out_h = params[9];
    let out_w = params[10];
    if i >= nc * out_h * out_w {{ return; }}
    let n = i / (out_h * out_w);
    let op = i - n * out_h * out_w;
    let oh = op / out_w;
    let ow = op - oh * out_w;
    var best: f32 = -3.4028235e38;
    var best_flat: u32 = 0u;
    var found: bool = false;
    for (var khr: u32 = 0u; khr < kh; khr = khr + 1u) {{
        for (var kwc: u32 = 0u; kwc < kw; kwc = kwc + 1u) {{
            let ih = oh * sh + khr;
            let iw = ow * sw + kwc;
            if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {{
                let fi = n * h * w + (ih - ph) * w + (iw - pw);
                let v = a[fi];
                if !found || v > best {{
                    best = v;
                    best_flat = fi;
                    found = true;
                }}
            }}
        }}
    }}
    out[i] = select(0.0, best, found);
    idxs[i] = select(0.0, f32(best_flat), found);
}}
"
    );
    let params = params_buf(&[
        nc as u32,
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
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    let idxs = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&out, false),
            (&idxs, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    (
        GpuStorage::from_buffer(nc, out_h * out_w, out),
        GpuStorage::from_buffer(nc, out_h * out_w, idxs),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_avg_pool2d<T: Scalar>(
    a: &GpuStorage<T>,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let nc = a.nrows;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let total = nc * out_h * out_w;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nc = params[0];
    let h = params[1];
    let w = params[2];
    let kh = params[3];
    let kw = params[4];
    let sh = params[5];
    let sw = params[6];
    let ph = params[7];
    let pw = params[8];
    let out_h = params[9];
    let out_w = params[10];
    if i >= nc * out_h * out_w {{ return; }}
    let n = i / (out_h * out_w);
    let op = i - n * out_h * out_w;
    let oh = op / out_w;
    let ow = op - oh * out_w;
    var sum: f32 = 0.0;
    var cnt: f32 = 0.0;
    for (var khr: u32 = 0u; khr < kh; khr = khr + 1u) {{
        for (var kwc: u32 = 0u; kwc < kw; kwc = kwc + 1u) {{
            let ih = oh * sh + khr;
            let iw = ow * sw + kwc;
            if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {{
                sum = sum + a[n * h * w + (ih - ph) * w + (iw - pw)];
                cnt = cnt + 1.0;
            }}
        }}
    }}
    out[i] = select(0.0, sum / cnt, cnt > 0.0);
}}
"
    );
    let params = params_buf(&[
        nc as u32,
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
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(nc, out_h * out_w, out)
}

pub(crate) fn gpu_adaptive_avg_pool2d<T: Scalar>(
    a: &GpuStorage<T>,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let nc = a.nrows;
    let total = nc * out_h * out_w;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nc = params[0];
    let in_h = params[1];
    let in_w = params[2];
    let out_h = params[3];
    let out_w = params[4];
    if i >= nc * out_h * out_w {{ return; }}
    let n = i / (out_h * out_w);
    let op = i - n * out_h * out_w;
    let oh = op / out_w;
    let ow = op - oh * out_w;
    let ih_start = oh * in_h / out_h;
    let ih_end = (oh + 1u) * in_h / out_h;
    let iw_start = ow * in_w / out_w;
    let iw_end = (ow + 1u) * in_w / out_w;
    var sum: f32 = 0.0;
    var cnt: f32 = 0.0;
    for (var ih: u32 = ih_start; ih < ih_end; ih = ih + 1u) {{
        for (var iw: u32 = iw_start; iw < iw_end; iw = iw + 1u) {{
            sum = sum + a[n * in_h * in_w + ih * in_w + iw];
            cnt = cnt + 1.0;
        }}
    }}
    out[i] = select(0.0, sum / cnt, cnt > 0.0);
}}
"
    );
    let params = params_buf(&[
        nc as u32,
        in_h as u32,
        in_w as u32,
        out_h as u32,
        out_w as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[(&a.buffer, true), (&out, false), (&params, true)],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(nc, out_h * out_w, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_conv2d<T: Scalar>(
    input: &GpuStorage<T>,
    weight: &GpuStorage<T>,
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
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let c_in_g = c_in / groups;
    let c_out_g = c_out / groups;
    let out_h = (h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
    let out_spatial = out_h * out_w;
    let total = n * c_out * out_spatial;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    let c_in = params[1];
    let h = params[2];
    let w = params[3];
    let c_out = params[4];
    let kh = params[5];
    let kw = params[6];
    let sh = params[7];
    let sw = params[8];
    let ph = params[9];
    let pw = params[10];
    let dh = params[11];
    let dw = params[12];
    let groups = params[13];
    let c_in_g = params[14];
    let c_out_g = params[15];
    let out_h = params[16];
    let out_w = params[17];
    let out_spatial = out_h * out_w;
    if i >= n * c_out * out_spatial {{ return; }}
    let row = i / out_spatial;
    let col = i - row * out_spatial;
    let b = row / c_out;
    let oc = row - b * c_out;
    let g = oc / c_out_g;
    let oh = col / out_w;
    let ow = col - oh * out_w;
    var acc: f32 = 0.0;
    for (var ic: u32 = 0u; ic < c_in_g; ic = ic + 1u) {{
        for (var khr: u32 = 0u; khr < kh; khr = khr + 1u) {{
            for (var kwc: u32 = 0u; kwc < kw; kwc = kwc + 1u) {{
                let ih = oh * sh + khr * dh;
                let iw = ow * sw + kwc * dw;
                if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {{
                    let in_row = b * c_in + g * c_in_g + ic;
                    let in_col = (ih - ph) * w + (iw - pw);
                    let x = input[in_row * h * w + in_col];
                    let w_idx = oc * (c_in_g * kh * kw) + ic * (kh * kw) + khr * kw + kwc;
                    let wt = weight[w_idx];
                    acc = acc + x * wt;
                }}
            }}
        }}
    }}
    out[i] = acc;
}}
"
    );
    let params = params_buf(&[
        n as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        stride.0 as u32,
        stride.1 as u32,
        padding.0 as u32,
        padding.1 as u32,
        dilation.0 as u32,
        dilation.1 as u32,
        groups as u32,
        c_in_g as u32,
        c_out_g as u32,
        out_h as u32,
        out_w as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&input.buffer, true),
            (&weight.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(n * c_out, out_spatial, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_conv1d<T: Scalar>(
    input: &GpuStorage<T>,
    weight: &GpuStorage<T>,
    n_batch: usize,
    c_in: usize,
    length: usize,
    c_out: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let c_in_g = c_in / groups;
    let out_len = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let total = n_batch * c_out * out_len;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n_batch = params[0];
    let c_in = params[1];
    let length = params[2];
    let c_out = params[3];
    let k = params[4];
    let stride = params[5];
    let pad = params[6];
    let dil = params[7];
    let groups = params[8];
    let c_in_g = params[9];
    let out_len = params[10];
    if i >= n_batch * c_out * out_len {{ return; }}
    let row = i / out_len;
    let col = i - row * out_len;
    let b = row / c_out;
    let oc = row - b * c_out;
    let g = oc / (c_out / groups);
    var acc: f32 = 0.0;
    for (var ic: u32 = 0u; ic < c_in_g; ic = ic + 1u) {{
        for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {{
            let il = col * stride + kk * dil;
            if il >= pad && il < length + pad {{
                let in_row = b * c_in + g * c_in_g + ic;
                let in_col = il - pad;
                let x = input[in_row * length + in_col];
                let w_idx = oc * (c_in_g * k) + ic * k + kk;
                let wt = weight[w_idx];
                acc = acc + x * wt;
            }}
        }}
    }}
    out[i] = acc;
}}
"
    );
    let params = params_buf(&[
        n_batch as u32,
        c_in as u32,
        length as u32,
        c_out as u32,
        kernel_size as u32,
        stride as u32,
        padding as u32,
        dilation as u32,
        groups as u32,
        c_in_g as u32,
        out_len as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&input.buffer, true),
            (&weight.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(n_batch * c_out, out_len, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_conv3d<T: Scalar>(
    input: &GpuStorage<T>,
    weight: &GpuStorage<T>,
    n_batch: usize,
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
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let c_in_g = c_in / groups;
    let c_out_g = c_out / groups;
    let out_d = (d + 2 * padding.0 - dilation.0 * (kd - 1) - 1) / stride.0 + 1;
    let out_h = (h + 2 * padding.1 - dilation.1 * (kh - 1) - 1) / stride.1 + 1;
    let out_w = (w + 2 * padding.2 - dilation.2 * (kw - 1) - 1) / stride.2 + 1;
    let out_spatial = out_d * out_h * out_w;
    let total = n_batch * c_out * out_spatial;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n_batch = params[0];
    let c_in = params[1];
    let d = params[2];
    let h = params[3];
    let w = params[4];
    let c_out = params[5];
    let kd = params[6];
    let kh = params[7];
    let kw = params[8];
    let sd = params[9];
    let sh = params[10];
    let sw = params[11];
    let pd = params[12];
    let ph = params[13];
    let pw = params[14];
    let dd = params[15];
    let dh = params[16];
    let dw = params[17];
    let c_in_g = params[18];
    let c_out_g = params[19];
    let out_d = params[20];
    let out_h = params[21];
    let out_w = params[22];
    let out_spatial = out_d * out_h * out_w;
    if i >= n_batch * c_out * out_spatial {{ return; }}
    let row = i / out_spatial;
    let col = i - row * out_spatial;
    let b = row / c_out;
    let oc = row - b * c_out;
    let g = oc / c_out_g;
    let od = col / (out_h * out_w);
    let oh = (col / out_w) % out_h;
    let ow = col - (col / out_w) * out_w;
    var acc: f32 = 0.0;
    for (var ic: u32 = 0u; ic < c_in_g; ic = ic + 1u) {{
        for (var kdr: u32 = 0u; kdr < kd; kdr = kdr + 1u) {{
            for (var khr: u32 = 0u; khr < kh; khr = khr + 1u) {{
                for (var kwc: u32 = 0u; kwc < kw; kwc = kwc + 1u) {{
                    let id = od * sd + kdr * dd;
                    let ih = oh * sh + khr * dh;
                    let iw = ow * sw + kwc * dw;
                    if id >= pd && id < d + pd && ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {{
                        let in_row = b * c_in + g * c_in_g + ic;
                        let in_col = (id - pd) * h * w + (ih - ph) * w + (iw - pw);
                        let x = input[in_row * d * h * w + in_col];
                        let w_idx = oc * (c_in_g * kd * kh * kw)
                            + ic * (kd * kh * kw)
                            + kdr * (kh * kw)
                            + khr * kw
                            + kwc;
                        let wt = weight[w_idx];
                        acc = acc + x * wt;
                    }}
                }}
            }}
        }}
    }}
    out[i] = acc;
}}
"
    );
    let params = params_buf(&[
        n_batch as u32,
        c_in as u32,
        d as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kd as u32,
        kh as u32,
        kw as u32,
        stride.0 as u32,
        stride.1 as u32,
        stride.2 as u32,
        padding.0 as u32,
        padding.1 as u32,
        padding.2 as u32,
        dilation.0 as u32,
        dilation.1 as u32,
        dilation.2 as u32,
        c_in_g as u32,
        c_out_g as u32,
        out_d as u32,
        out_h as u32,
        out_w as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&input.buffer, true),
            (&weight.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(n_batch * c_out, out_spatial, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_conv_transpose2d<T: Scalar>(
    input: &GpuStorage<T>,
    weight: &GpuStorage<T>,
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
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let out_h = (h - 1) * stride.0 - 2 * padding.0 + kh + output_padding.0;
    let out_w = (w - 1) * stride.1 - 2 * padding.1 + kw + output_padding.1;
    let out_spatial = out_h * out_w;
    let total = n_batch * c_out * out_spatial;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n_batch = params[0];
    let c_in = params[1];
    let h = params[2];
    let w = params[3];
    let c_out = params[4];
    let kh = params[5];
    let kw = params[6];
    let sh = params[7];
    let sw = params[8];
    let ph = params[9];
    let pw = params[10];
    let out_h = params[11];
    let out_w = params[12];
    let out_spatial = out_h * out_w;
    if i >= n_batch * c_out * out_spatial {{ return; }}
    let row = i / out_spatial;
    let col = i - row * out_spatial;
    let b = row / c_out;
    let oc = row - b * c_out;
    let oh = col / out_w;
    let ow = col - oh * out_w;
    var acc: f32 = 0.0;
    for (var ic: u32 = 0u; ic < c_in; ic = ic + 1u) {{
        for (var khr: u32 = 0u; khr < kh; khr = khr + 1u) {{
            for (var kwc: u32 = 0u; kwc < kw; kwc = kwc + 1u) {{
                let ih_pad = oh + ph;
                let iw_pad = ow + pw;
                if ih_pad >= khr && iw_pad >= kwc {{
                    let ih = ih_pad - khr;
                    let iw = iw_pad - kwc;
                    if (ih % sh) == 0u && (iw % sw) == 0u {{
                        let ihs = ih / sh;
                        let iws = iw / sw;
                        if ihs < h && iws < w {{
                            let x = input[(b * c_in + ic) * h * w + ihs * w + iws];
                            let w_idx = ic * (c_out * kh * kw) + oc * (kh * kw) + khr * kw + kwc;
                            let wt = weight[w_idx];
                            acc = acc + x * wt;
                        }}
                    }}
                }}
            }}
        }}
    }}
    out[i] = acc;
}}
"
    );
    let params = params_buf(&[
        n_batch as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        stride.0 as u32,
        stride.1 as u32,
        padding.0 as u32,
        padding.1 as u32,
        out_h as u32,
        out_w as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&input.buffer, true),
            (&weight.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(n_batch * c_out, out_spatial, out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_sdpa<T: Scalar>(
    q: &GpuStorage<T>,
    k: &GpuStorage<T>,
    v: &GpuStorage<T>,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    batch_heads: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let total = batch_heads * seq_q * head_dim;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let seq_q = params[0];
    let seq_k = params[1];
    let head_dim = params[2];
    let batch_heads = params[3];
    let rows = batch_heads * seq_q;
    if i >= rows * head_dim {{ return; }}
    let row = i / head_dim;
    let d = i - row * head_dim;
    let bh = row / seq_q;
    let qi = row - bh * seq_q;
    let scale = 1.0 / sqrt(f32(head_dim));
    var maxv: f32 = -3.4028235e38;
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {{
        var dot: f32 = 0.0;
        let q_base = (bh * seq_q + qi) * head_dim;
        let k_base = (bh * seq_k + j) * head_dim;
        for (var dd: u32 = 0u; dd < head_dim; dd = dd + 1u) {{
            dot = dot + q[q_base + dd] * k[k_base + dd];
        }}
        let score = dot * scale;
        if score > maxv {{ maxv = score; }}
    }}
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {{
        var dot: f32 = 0.0;
        let q_base = (bh * seq_q + qi) * head_dim;
        let k_base = (bh * seq_k + j) * head_dim;
        for (var dd: u32 = 0u; dd < head_dim; dd = dd + 1u) {{
            dot = dot + q[q_base + dd] * k[k_base + dd];
        }}
        sum = sum + exp(dot * scale - maxv);
    }}
    let inv = 1.0 / sum;
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < seq_k; j = j + 1u) {{
        var dot: f32 = 0.0;
        let q_base = (bh * seq_q + qi) * head_dim;
        let k_base = (bh * seq_k + j) * head_dim;
        for (var dd: u32 = 0u; dd < head_dim; dd = dd + 1u) {{
            dot = dot + q[q_base + dd] * k[k_base + dd];
        }}
        let w = exp(dot * scale - maxv) * inv;
        let v_base = (bh * seq_k + j) * head_dim;
        acc = acc + w * v[v_base + d];
    }}
    out[i] = acc;
}}
"
    );
    let params = params_buf(&[
        seq_q as u32,
        seq_k as u32,
        head_dim as u32,
        batch_heads as u32,
    ]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&q.buffer, true),
            (&k.buffer, true),
            (&v.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(total, wg),
    );
    GpuStorage::from_buffer(batch_heads * seq_q, head_dim, out)
}

pub(crate) fn gpu_powf<T: Scalar>(a: &GpuStorage<T>, p: T) -> GpuStorage<T> {
    assert_is_f32_or_f16::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let power = p.to_f64() as f32;
    let buf = run_powf::<T>(ctx, &a.buffer, n, power);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_masked_fill<T: Scalar>(
    a: &GpuStorage<T>,
    mask: &GpuStorage<T>,
    value: T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    // SAFETY: TypeId confirmed T == f32.
    #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
    let val_f32: f32 = unsafe { *std::ptr::from_ref(&value).cast::<f32>() };
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    let val = bitcast<f32>(params[1]);
    if i >= total {{ return; }}
    let m = mask[i];
    out[i] = select(a[i], val, m != 0.0);
}}
"
    );
    let params = params_buf(&[n as u32, val_f32.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&mask.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

pub(crate) fn gpu_where_cond<T: Scalar>(
    a: &GpuStorage<T>,
    cond: &GpuStorage<T>,
    b: &GpuStorage<T>,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let wg = ctx.wg_size;
    let shader = format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let total = params[0];
    if i >= total {{ return; }}
    let c = cond[i];
    out[i] = select(b[i], a[i], c != 0.0);
}}
"
    );
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    run_custom_shader(
        ctx,
        &shader,
        &[
            (&a.buffer, true),
            (&cond.buffer, true),
            (&b.buffer, true),
            (&out, false),
            (&params, true),
        ],
        workgroups(n, wg),
    );
    GpuStorage::from_buffer(a.nrows, a.ncols, out)
}

macro_rules! impl_gpu_unary {
    ($name:ident, $op:expr) => {
        pub(crate) fn $name<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
            assert_is_f32_or_f16::<T>();
            let ctx = get_context();
            let n = a.nrows * a.ncols;
            let buf = run_unary::<T>(ctx, &a.buffer, n, $op);
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

fn translate_fuse_expr(expr: &str) -> String {
    let mut out = expr.to_string();
    out = out.replace("fmodf", "fmod");
    out = out.replace("fmod", "fmod");
    out = out.replace("powf", "pow");
    out = out.replace("expf", "exp");
    out = out.replace("logf", "log");
    out = out.replace("tanhf", "tanh");
    out = out.replace("sinf", "sin");
    out = out.replace("cosf", "cos");
    out = out.replace("sqrtf", "sqrt");
    out = out.replace("fabsf", "abs");
    out = out.replace("atan2f", "atan2");
    out = out.replace("fmaxf", "max");
    out = out.replace("fminf", "min");
    out
}

fn build_fuse_shader(n_inputs: usize, wg: u32, expr: &str) -> String {
    let mut src = String::new();
    src.push_str("fn fmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }\n");
    for i in 0..n_inputs {
        src.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> in{i}: array<f32>;\n"
        ));
    }
    let out_binding = n_inputs;
    let params_binding = n_inputs + 1;
    src.push_str(&format!(
        "@group(0) @binding({out_binding}) var<storage, read_write> out: array<f32>;\n"
    ));
    src.push_str(&format!(
        "@group(0) @binding({params_binding}) var<storage, read> params: array<u32>;\n"
    ));
    src.push_str(&format!(
        "@compute @workgroup_size({wg})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
    ));
    src.push_str("    let i = gid.x;\n    let n = params[0];\n    if i >= n { return; }\n");
    src.push_str("    let v = ");
    src.push_str(expr);
    src.push_str(";\n    out[i] = v;\n}\n");
    src
}

fn gpu_fuse_launch<T: Scalar>(
    inputs: &[*const u8],
    nrows: usize,
    ncols: usize,
    gpu_expr: &str,
    n_inputs: usize,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = nrows * ncols;
    let wg = ctx.wg_size;
    let expr = translate_fuse_expr(gpu_expr);
    let shader = build_fuse_shader(n_inputs, wg, &expr);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let params = params_buf(&[n as u32]);
    let mut buffers: Vec<(&wgpu::Buffer, bool)> = Vec::with_capacity(n_inputs + 2);
    for &ptr in inputs.iter().take(n_inputs) {
        // SAFETY: inputs are storage pointers from Tensor::__storage_ptr for this backend.
        let storage = unsafe { &*(ptr.cast::<GpuStorage<T>>()) };
        buffers.push((&storage.buffer, true));
    }
    buffers.push((&out, false));
    buffers.push((&params, true));
    run_custom_shader(ctx, &shader, &buffers, workgroups(n, wg));
    GpuStorage::from_buffer(nrows, ncols, out)
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
    fn empty<T: crate::scalar::Scalar>(r: usize, c: usize) -> GpuStorage<T> {
        gpu_empty::<T>(r, c)
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
    fn from_vec<T: crate::scalar::Scalar>(r: usize, c: usize, data: Vec<T>) -> GpuStorage<T> {
        gpu_from_vec::<T>(r, c, data)
    }

    #[inline]
    fn from_vec_async<T: crate::scalar::Scalar>(r: usize, c: usize, data: Vec<T>) -> GpuStorage<T> {
        gpu_from_vec::<T>(r, c, data)
    }

    #[inline]
    fn to_vec_async<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> Vec<T> {
        assert_is_f32_or_f16::<T>();
        let ctx = get_context();
        let n = a.nrows * a.ncols;
        let bytes = readback(ctx, &a.buffer, n as u64 * elem_size_bytes::<T>());
        // SAFETY: bytes are a valid [T] buffer from GPU readback.
        unsafe { bytes_to_scalar::<T>(&bytes) }
    }

    #[inline]
    fn cast<T: crate::scalar::Scalar, U: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
    ) -> GpuStorage<U> {
        if is_f32::<T>() && is_f32::<U>() {
        } else if is_f16::<T>() && is_f16::<U>() {
            ensure_f16_supported();
        } else {
            panic!("nabla: wgpu cast supports only f32<->f32 or f16<->f16");
        }
        let ctx = get_context();
        let n = a.nrows * a.ncols;
        let buf = run_copy::<T>(ctx, &a.buffer, n);
        GpuStorage::from_buffer(a.nrows, a.ncols, buf)
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
    fn one_hot_from_indices<T: crate::scalar::Scalar>(
        indices: &GpuStorage<T>,
        n_classes: usize,
    ) -> GpuStorage<T> {
        gpu_one_hot_from_indices(indices, n_classes)
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

    #[inline]
    fn expand_into<T: crate::scalar::Scalar>(
        out: &mut GpuStorage<T>,
        src: &GpuStorage<T>,
        src_rows: usize,
        src_cols: usize,
    ) {
        gpu_expand_into(out, src, src_rows, src_cols);
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

    #[inline]
    fn masked_fill<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        mask: &GpuStorage<T>,
        value: T,
    ) -> GpuStorage<T> {
        gpu_masked_fill(a, mask, value)
    }

    #[inline]
    fn where_cond<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        cond: &GpuStorage<T>,
        b: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_where_cond(a, cond, b)
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

    fn diag<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_diag(a)
    }

    fn trace<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> T {
        let d = gpu_diag(a);
        gpu_sum_all(&d)
    }

    fn prod_all<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> T {
        assert_is_f32::<T>();
        let ctx = get_context();
        let n = a.nrows * a.ncols;
        let partial = run_reduce_f32(ctx, &a.buffer, n, ShaderOp::ReduceProd);
        let mut acc = 1.0f32;
        for v in partial {
            acc *= v;
        }
        // SAFETY: T is f32; reinterpret f32 bits as T.
        #[allow(clippy::borrow_as_ptr, clippy::ptr_cast_constness)]
        unsafe {
            *std::ptr::from_ref(&acc).cast::<T>()
        }
    }

    fn count_nonzero<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> usize {
        assert_is_f32::<T>();
        let ctx = get_context();
        let n = a.nrows * a.ncols;
        let partial = run_reduce_f32(ctx, &a.buffer, n, ShaderOp::ReduceCountNonzero);
        let mut acc = 0.0f32;
        for v in partial {
            acc += v;
        }
        acc as usize
    }

    fn sum_axis1<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_sum_axis1(a)
    }

    fn max_axis1<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_max_axis1(a)
    }

    fn cumsum_axis1<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_cumsum_axis1(a)
    }

    fn cumprod_axis1<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_cumprod_axis1(a)
    }

    fn mse_sum_fwd<T: crate::scalar::Scalar>(
        pred: &GpuStorage<T>,
        target: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        let diff = gpu_sub(pred, target);
        let sq = gpu_emul(&diff, &diff);
        let sum = gpu_sum_all(&sq);
        gpu_fill(1, 1, sum)
    }

    fn mse_sum_bwd<T: crate::scalar::Scalar>(
        pred: &GpuStorage<T>,
        target: &GpuStorage<T>,
        grad: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        let diff = gpu_sub(pred, target);
        let g = gpu_get(grad, 0, 0);
        let two = T::from_f64(2.0);
        let scale = two * g;
        gpu_scale(&diff, scale)
    }

    fn norm_lp<T: crate::scalar::Scalar>(a: &GpuStorage<T>, p: T) -> T {
        let p_f64 = p.to_f64();
        if p_f64.is_infinite() && p_f64 > 0.0 {
            let abs = gpu_abs(a);
            return gpu_max_all(&abs);
        }
        if p_f64 == 1.0 {
            let abs = gpu_abs(a);
            return gpu_sum_all(&abs);
        }
        if p_f64 == 2.0 {
            let abs = gpu_abs(a);
            let sq = gpu_emul(&abs, &abs);
            let sum = gpu_sum_all(&sq);
            return sum.math_sqrt();
        }
        let abs = gpu_abs(a);
        let pow = gpu_powf(&abs, p);
        let sum = gpu_sum_all(&pow);
        let inv = T::from_f64(1.0 / p_f64);
        sum.math_powf(inv)
    }
}

#[cfg(feature = "gpu")]
impl crate::backend::BackendShape for crate::backend::Gpu {
    fn reshape_copy<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        out_rows: usize,
        out_cols: usize,
    ) -> GpuStorage<T> {
        gpu_reshape_copy(a, out_rows, out_cols)
    }

    fn submatrix<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        row_start: usize,
        col_start: usize,
        out_rows: usize,
        out_cols: usize,
    ) -> GpuStorage<T> {
        gpu_submatrix(a, row_start, col_start, out_rows, out_cols)
    }

    fn slice_set<T: crate::scalar::Scalar>(
        dst: &mut GpuStorage<T>,
        row_start: usize,
        col_start: usize,
        src: &GpuStorage<T>,
    ) {
        gpu_slice_set(dst, row_start, col_start, src);
    }

    fn repeat<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        row_reps: usize,
        col_reps: usize,
    ) -> GpuStorage<T> {
        gpu_repeat(a, row_reps, col_reps)
    }

    fn pad<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        left: usize,
        right: usize,
        top: usize,
        bottom: usize,
        value: T,
    ) -> GpuStorage<T> {
        gpu_pad(a, left, right, top, bottom, value)
    }

    fn triu<T: crate::scalar::Scalar>(a: &GpuStorage<T>, diagonal: isize) -> GpuStorage<T> {
        gpu_triu(a, diagonal)
    }

    fn tril<T: crate::scalar::Scalar>(a: &GpuStorage<T>, diagonal: isize) -> GpuStorage<T> {
        gpu_tril(a, diagonal)
    }

    fn roll<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        shift: isize,
        axis: usize,
    ) -> GpuStorage<T> {
        gpu_roll(a, shift, axis)
    }

    fn flip<T: crate::scalar::Scalar>(a: &GpuStorage<T>, axis: usize) -> GpuStorage<T> {
        gpu_flip(a, axis)
    }

    fn from_diag<T: crate::scalar::Scalar>(v: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_from_diag(v)
    }

    fn gather_rows<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        indices: &[usize],
    ) -> GpuStorage<T> {
        gpu_gather_rows(a, indices)
    }

    fn gather<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        axis: usize,
        index: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_gather(a, axis, index)
    }

    fn scatter<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        axis: usize,
        index: &GpuStorage<T>,
        src: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_scatter(a, axis, index, src)
    }

    fn index_select<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        axis: usize,
        index: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_index_select(a, axis, index)
    }

    fn sort_rows<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        descending: bool,
    ) -> (GpuStorage<T>, GpuStorage<T>) {
        gpu_sort_rows(a, descending)
    }

    fn topk_rows<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        k: usize,
    ) -> (GpuStorage<T>, GpuStorage<T>) {
        gpu_topk_rows(a, k)
    }

    fn meshgrid<T: crate::scalar::Scalar>(
        x: &GpuStorage<T>,
        y: &GpuStorage<T>,
    ) -> (GpuStorage<T>, GpuStorage<T>) {
        gpu_meshgrid(x, y)
    }

    fn scatter_add_dim0<T: crate::scalar::Scalar>(
        dst: &mut GpuStorage<T>,
        indices: &[usize],
        src: &GpuStorage<T>,
    ) {
        gpu_scatter_add_dim0(dst, indices, src);
    }

    fn kron<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
        m: usize,
        n: usize,
        p: usize,
        q: usize,
    ) -> GpuStorage<T> {
        gpu_kron(a, b, m, n, p, q)
    }
}

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

    #[inline]
    fn matmul_tn_into<T: crate::scalar::Scalar>(
        out: &mut GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
    ) {
        gpu_matmul_tn::<T>(out, a, b);
    }

    #[inline]
    fn matmul_nt_into<T: crate::scalar::Scalar>(
        out: &mut GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
    ) {
        gpu_matmul_nt::<T>(out, a, b);
    }

    fn matmul_epilogue<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
        epilogue_id: u8,
    ) -> GpuStorage<T> {
        let mut out = gpu_zeros(a.nrows, b.ncols);
        gpu_matmul(&mut out, a, b);
        match epilogue_id {
            0 => gpu_leaky_relu(&out, T::zero()),
            1 => gpu_gelu_tanh_approx(&out),
            _ => out,
        }
    }

    fn matmul_bias<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
        bias: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        let mut out = gpu_zeros(a.nrows, b.ncols);
        gpu_matmul(&mut out, a, b);
        gpu_add_bias_row(&out, bias)
    }

    fn bmm<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> GpuStorage<T> {
        let mut out = gpu_zeros(batch * m, n);
        for bi in 0..batch {
            let a_i = gpu_submatrix(a, bi * m, 0, m, k);
            let b_i = gpu_submatrix(b, bi * k, 0, k, n);
            let mut tmp = gpu_zeros(m, n);
            gpu_matmul(&mut tmp, &a_i, &b_i);
            gpu_slice_set(&mut out, bi * m, 0, &tmp);
        }
        out
    }

    fn addmm<T: crate::scalar::Scalar>(
        c: &GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
        beta: T,
        alpha: T,
    ) -> GpuStorage<T> {
        let mut ab = gpu_zeros(a.nrows, b.ncols);
        gpu_matmul(&mut ab, a, b);
        let ab_scaled = gpu_scale(&ab, alpha);
        let c_scaled = gpu_scale(c, beta);
        gpu_add(&c_scaled, &ab_scaled)
    }

    fn baddbmm<T: crate::scalar::Scalar>(
        c: &GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        beta: T,
        alpha: T,
    ) -> GpuStorage<T> {
        let mut out = gpu_zeros(batch * m, n);
        for bi in 0..batch {
            let a_i = gpu_submatrix(a, bi * m, 0, m, k);
            let b_i = gpu_submatrix(b, bi * k, 0, k, n);
            let c_i = gpu_submatrix(c, bi * m, 0, m, n);
            let mut tmp = gpu_zeros(m, n);
            gpu_matmul(&mut tmp, &a_i, &b_i);
            let ab_scaled = gpu_scale(&tmp, alpha);
            let c_scaled = gpu_scale(&c_i, beta);
            let out_i = gpu_add(&c_scaled, &ab_scaled);
            gpu_slice_set(&mut out, bi * m, 0, &out_i);
        }
        out
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

    fn group_norm<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        g: &GpuStorage<T>,
        b: &GpuStorage<T>,
        groups: usize,
        eps: T,
    ) -> GpuStorage<T> {
        gpu_group_norm(a, g, b, groups, eps)
    }

    fn batch_norm_train<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        gamma: &GpuStorage<T>,
        beta: &GpuStorage<T>,
        running_mean: &mut GpuStorage<T>,
        running_var: &mut GpuStorage<T>,
        eps: T,
        momentum: T,
        training: bool,
    ) -> GpuStorage<T> {
        gpu_batch_norm_train(
            a,
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
            momentum,
            training,
        )
    }

    fn cross_entropy_fused<T: crate::scalar::Scalar>(
        input: &GpuStorage<T>,
        target: &GpuStorage<T>,
        n: usize,
        c: usize,
    ) -> GpuStorage<T> {
        gpu_cross_entropy_fused(input, target, n, c)
    }

    fn embedding_backward<T: crate::scalar::Scalar>(
        indices: &GpuStorage<T>,
        grad: &GpuStorage<T>,
        vocab: usize,
    ) -> GpuStorage<T> {
        gpu_embedding_backward(indices, grad, vocab)
    }

    fn max_pool2d<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> GpuStorage<T> {
        gpu_max_pool2d(a, h, w, kh, kw, sh, sw, ph, pw)
    }

    fn max_pool2d_with_indices<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> (GpuStorage<T>, GpuStorage<T>) {
        gpu_max_pool2d_with_indices(a, h, w, kh, kw, sh, sw, ph, pw)
    }

    fn avg_pool2d<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> GpuStorage<T> {
        gpu_avg_pool2d(a, h, w, kh, kw, sh, sw, ph, pw)
    }

    fn adaptive_avg_pool2d<T: crate::scalar::Scalar>(
        a: &GpuStorage<T>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> GpuStorage<T> {
        gpu_adaptive_avg_pool2d(a, in_h, in_w, out_h, out_w)
    }

    fn conv2d<T: crate::scalar::Scalar>(
        input: &GpuStorage<T>,
        weight: &GpuStorage<T>,
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
    ) -> GpuStorage<T> {
        gpu_conv2d(
            input, weight, n, c_in, h, w, c_out, kh, kw, stride, padding, dilation, groups,
        )
    }

    fn conv1d<T: crate::scalar::Scalar>(
        input: &GpuStorage<T>,
        weight: &GpuStorage<T>,
        n_batch: usize,
        c_in: usize,
        length: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> GpuStorage<T> {
        gpu_conv1d(
            input,
            weight,
            n_batch,
            c_in,
            length,
            c_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        )
    }

    fn conv3d<T: crate::scalar::Scalar>(
        input: &GpuStorage<T>,
        weight: &GpuStorage<T>,
        n_batch: usize,
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
    ) -> GpuStorage<T> {
        gpu_conv3d(
            input, weight, n_batch, c_in, d, h, w, c_out, kd, kh, kw, stride, padding, dilation,
            groups,
        )
    }

    fn conv_transpose2d<T: crate::scalar::Scalar>(
        input: &GpuStorage<T>,
        weight: &GpuStorage<T>,
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
    ) -> GpuStorage<T> {
        gpu_conv_transpose2d(
            input,
            weight,
            n_batch,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            stride,
            padding,
            output_padding,
        )
    }

    fn relu_backward<T: crate::scalar::Scalar>(
        grad: &GpuStorage<T>,
        input: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_relu_backward(grad, input)
    }

    fn leaky_relu_backward<T: crate::scalar::Scalar>(
        grad: &GpuStorage<T>,
        input: &GpuStorage<T>,
        alpha: T,
    ) -> GpuStorage<T> {
        gpu_leaky_relu_backward(grad, input, alpha)
    }

    fn elu_backward<T: crate::scalar::Scalar>(
        grad: &GpuStorage<T>,
        input: &GpuStorage<T>,
        alpha: T,
    ) -> GpuStorage<T> {
        gpu_elu_backward(grad, input, alpha)
    }

    fn gelu_backward<T: crate::scalar::Scalar>(
        grad: &GpuStorage<T>,
        input: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_gelu_backward(grad, input)
    }

    fn abs_backward<T: crate::scalar::Scalar>(
        grad: &GpuStorage<T>,
        input: &GpuStorage<T>,
    ) -> GpuStorage<T> {
        gpu_abs_backward(grad, input)
    }

    fn sdpa<T: crate::scalar::Scalar>(
        q: &GpuStorage<T>,
        k: &GpuStorage<T>,
        v: &GpuStorage<T>,
        _mask: Option<&GpuStorage<T>>,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        batch_heads: usize,
    ) -> GpuStorage<T> {
        gpu_sdpa(q, k, v, seq_q, seq_k, head_dim, batch_heads)
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
        gpu_fuse_launch(_inputs, _nrows, _ncols, _gpu_expr, _n_inputs)
    }

    fn mega_fuse_launch<'a, T: crate::scalar::Scalar>(
        _ops: &[(Vec<*const u8>, String, usize, bool)],
        _nrows: usize,
        _ncols: usize,
        _cpu_fns: Vec<Box<dyn FnMut(usize, usize) -> T + 'a>>,
        _kernel_hash: &str,
    ) -> Vec<GpuStorage<T>> {
        let mut out = Vec::with_capacity(_ops.len());
        for (inputs, expr, n_inputs, _uses_prev) in _ops {
            out.push(gpu_fuse_launch(inputs, _nrows, _ncols, expr, *n_inputs));
        }
        out
    }

    fn fuse_reduce_launch<T: crate::scalar::Scalar>(
        _inputs: &[*const u8],
        _nrows: usize,
        _ncols: usize,
        _cpu_fn: impl FnMut(usize, usize) -> T,
        _gpu_expr: &str,
        _kernel_hash: &str,
        _n_inputs: usize,
        _reduce_op: u8,
        _axis: u8,
    ) -> GpuStorage<T> {
        let fused = gpu_fuse_launch(_inputs, _nrows, _ncols, _gpu_expr, _n_inputs);
        match _reduce_op {
            0 => {
                if _axis == 1 {
                    gpu_sum_axis1(&fused)
                } else {
                    let s = gpu_sum_all(&fused);
                    gpu_fill(1, 1, s)
                }
            }
            1 => {
                if _axis == 1 {
                    gpu_max_axis1(&fused)
                } else {
                    let m = gpu_max_all(&fused);
                    gpu_fill(1, 1, m)
                }
            }
            _ => fused,
        }
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
