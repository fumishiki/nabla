use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use wgpu::util::DeviceExt;

use crate::scalar::Scalar;

use super::shaders::generate_shader;

// SAFETY: Scalar types used by WGPU (f32/f16) are POD with stable layout.
pub(super) unsafe fn scalar_to_bytes<T: Scalar>(data: &[T]) -> &[u8] {
    // SAFETY: T is a POD type; reinterpreted as bytes.
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data)) }
}

// SAFETY: bytes originated from a [T] slice with correct alignment and length.
pub(super) unsafe fn bytes_to_scalar<T: Scalar>(bytes: &[u8]) -> Vec<T> {
    // SAFETY: bytes came from a valid [T] allocation.
    unsafe {
        let len = bytes.len() / core::mem::size_of::<T>();
        core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len).to_vec()
    }
}

pub(super) fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(super) struct GpuContext {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) wg_size: u32,
    pipelines: Mutex<HashMap<PipelineKey, wgpu::ComputePipeline>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(super) enum ShaderOp {
    Binary,
    Scale,
    Unary,
    Powf,
    Transpose,
    Copy,
    FillZeros,
    FillScalar,
    FillIdentity,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    ReduceCountNonzero,
    Argmax,
    Argmin,
    Matmul {
        tile: u32,
    },
    MatmulRegTile {
        tr: u32,
        tc: u32,
        bm: u32,
        bn: u32,
        bk: u32,
    },
    // Activation / composite ops
    ActivationSilu,
    ActivationMish,
    ActivationLeakyRelu,
    ActivationElu,
    ActivationHardswish,
    ActivationSigmoid,
    Softmax,
    LayerNorm,
    RmsNorm,
    SumAxis1,
    MaxAxis1,
    Embedding,
    Axpy,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ScalarKind {
    F32,
    F16,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PipelineKey {
    pub(super) op: ShaderOp,
    pub(super) wg_size: u32,
    pub(super) scalar: ScalarKind,
}

pub(super) fn get_context() -> &'static GpuContext {
    static CTX: OnceLock<GpuContext> = OnceLock::new();
    CTX.get_or_init(|| pollster::block_on(init_gpu()))
}

async fn init_gpu() -> GpuContext {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap_or_else(|| panic!("no GPU adapter found"));
    #[cfg(feature = "wgpu-f16")]
    let required_features = {
        let available = adapter.features();
        let mut required = wgpu::Features::empty();
        if available.contains(wgpu::Features::SHADER_F16) {
            required |= wgpu::Features::SHADER_F16;
        }
        required
    };
    #[cfg(not(feature = "wgpu-f16"))]
    let required_features = wgpu::Features::empty();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nabla"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .unwrap_or_else(|e| panic!("failed to create wgpu device: {e}"));
    let max_wg = device.limits().max_compute_workgroup_size_x;
    let wg_size = if max_wg >= 512 {
        256
    } else if max_wg >= 256 {
        128
    } else {
        64
    };
    GpuContext {
        device,
        queue,
        wg_size,
        pipelines: Mutex::new(HashMap::new()),
    }
}

pub(super) fn compile_pipeline(ctx: &GpuContext, shader_src: &str) -> wgpu::ComputePipeline {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

pub(super) fn with_pipeline<R>(
    ctx: &GpuContext,
    key: PipelineKey,
    f: impl FnOnce(&wgpu::ComputePipeline) -> R,
) -> R {
    let mut cache = lock_or_recover(&ctx.pipelines);
    let pipeline = cache.entry(key).or_insert_with(|| {
        let src = generate_shader(key);
        compile_pipeline(ctx, &src)
    });
    f(pipeline)
}

pub(super) fn bind_group(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(&wgpu::Buffer, bool)],
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, (buf, _ro))| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    })
}

pub(super) fn dispatch_and_wait(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    ctx.device.poll(wgpu::MaintainBase::Wait).panic_on_timeout();
}

pub(super) fn readback(ctx: &GpuContext, buf: &wgpu::Buffer, size_bytes: u64) -> Vec<u8> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size_bytes);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::MaintainBase::Wait).panic_on_timeout();
    let data = slice.get_mapped_range().to_vec();
    drop(staging);
    data
}

pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(super) buffer: wgpu::Buffer,
    pub(super) host_cache: Mutex<Option<Vec<T>>>,
}

// SAFETY: wgpu::Buffer is Send+Sync (Arc-backed). Mutex<Option<Vec<T>>> is
unsafe impl<T: Scalar> Send for GpuStorage<T> {}
unsafe impl<T: Scalar> Sync for GpuStorage<T> {}

#[inline]
pub(super) fn lock_or_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[inline]
pub(super) fn cache_ref<'a, T>(cache: &'a Option<Vec<T>>, msg: &str) -> &'a [T] {
    match cache {
        Some(v) => v,
        None => panic!("{msg}"),
    }
}

#[inline]
pub(super) fn cache_mut<'a, T>(cache: &'a mut Option<Vec<T>>, msg: &str) -> &'a mut [T] {
    match cache {
        Some(v) => v,
        None => panic!("{msg}"),
    }
}

impl<T: Scalar> GpuStorage<T> {
    pub(super) fn from_buffer(nrows: usize, ncols: usize, buffer: wgpu::Buffer) -> Self {
        Self {
            nrows,
            ncols,
            buffer,
            host_cache: Mutex::new(None),
        }
    }

    #[allow(dead_code)]
    pub(super) fn from_buffer_cached(
        nrows: usize,
        ncols: usize,
        buffer: wgpu::Buffer,
        cache: Vec<T>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            buffer,
            host_cache: Mutex::new(Some(cache)),
        }
    }

    #[allow(dead_code)]
    pub(super) fn upload(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        let ctx = get_context();
        // SAFETY: T is a Scalar POD type; reinterpreted as bytes for GPU upload.
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
        Self::from_buffer_cached(nrows, ncols, buffer, data)
    }

    pub(super) fn empty_buf(n_bytes: u64) -> wgpu::Buffer {
        let ctx = get_context();
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(super) fn fill_cache_mut(&self) -> std::sync::MutexGuard<'_, Option<Vec<T>>> {
        let mut guard = lock_or_recover(&self.host_cache);
        if guard.is_none() {
            let ctx = get_context();
            let size_bytes = (self.nrows * self.ncols * core::mem::size_of::<T>()) as u64;
            let bytes = readback(ctx, &self.buffer, size_bytes);
            // SAFETY: bytes originated from a [T] slice via scalar_to_bytes.
            *guard = Some(unsafe { bytes_to_scalar::<T>(&bytes) });
        }
        guard
    }
}
