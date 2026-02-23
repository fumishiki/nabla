// gpu.rs — wgpu-backed GPU storage and WGSL compute shader dispatch.
//
// Design:
//   - GpuContext (OnceLock singleton) owns wgpu::Device + wgpu::Queue.
//   - GpuStorage<T> owns a wgpu::Buffer (RAII, Arc-backed via wgpu internals).
//   - GPU ops are f32-only; TypeId dispatch guards all entry points.
//   - host_cache: Mutex<Option<Vec<T>>> — lazily populated on first get/readback.
//   - Pipeline cache: Mutex<HashMap<&'static str, wgpu::ComputePipeline>>.

use std::any::TypeId;
use std::sync::{Mutex, OnceLock};

use wgpu::util::DeviceExt;

use crate::scalar::Scalar;

// ── Byte conversion helpers ───────────────────────────────────────────────────

// SAFETY: all Scalar types (f32) are POD with stable layout.
unsafe fn scalar_to_bytes<T: Scalar>(data: &[T]) -> &[u8] {
    // SAFETY: T is a POD type; reinterpreted as bytes.
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data)) }
}

// SAFETY: bytes originated from a [T] slice with correct alignment and length.
unsafe fn bytes_to_scalar<T: Scalar>(bytes: &[u8]) -> Vec<T> {
    // SAFETY: bytes came from a valid [T] allocation.
    unsafe {
        let len = bytes.len() / core::mem::size_of::<T>();
        core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len).to_vec()
    }
}

fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── GpuContext singleton ──────────────────────────────────────────────────────

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn get_context() -> &'static GpuContext {
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
        .expect("no GPU adapter found");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nabla"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .expect("failed to create wgpu device");
    GpuContext { device, queue }
}

fn create_pipeline(ctx: &GpuContext, shader_src: &str) -> wgpu::ComputePipeline {
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

// ── Dispatch helper ───────────────────────────────────────────────────────────

/// Bind group entries builder: storage buffers (read-only or read-write).
fn bind_group(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(&wgpu::Buffer, bool)], // (buffer, read_only)
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

/// Submit a compute pass and wait for completion.
fn dispatch_and_wait(
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

/// Read a GPU buffer back to host as bytes.
fn readback(ctx: &GpuContext, buf: &wgpu::Buffer, size_bytes: u64) -> Vec<u8> {
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

// ── WGSL shader sources ───────────────────────────────────────────────────────

// Uniform struct for passing parameters to kernels.
// All parameter buffers use a simple u32 array layout.

// Binary ops: op_type 0=add 1=sub 2=mul_elem 3=div_elem
const SHADER_BINARY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>; // [n, op_type]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params[0];
    let op = params[1];
    if i >= n { return; }
    if op == 0u { out[i] = a[i] + b[i]; }
    else if op == 1u { out[i] = a[i] - b[i]; }
    else if op == 2u { out[i] = a[i] * b[i]; }
    else { out[i] = a[i] / b[i]; }
}
"#;

// Scale: out[i] = a[i] * scalar
// params: [n] as u32, scalar passed as bitcast u32
const SHADER_SCALE: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n, scalar_bits]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params[0];
    if i >= n { return; }
    let scalar = bitcast<f32>(params[1]);
    out[i] = a[i] * scalar;
}
"#;

// Unary ops: op_type 0=neg 1=exp 2=ln 3=log1p 4=sin 5=cos 6=tanh 7=sqrt 8=abs 9=recip
//            10=erf 11=ceil 12=floor 13=round
const SHADER_UNARY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n, op_type]

// Abramowitz & Stegun polynomial erf (max error ~1.5e-7)
fn erf_approx(x: f32) -> f32 {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let r = 1.0 - poly * exp(-ax * ax);
    return select(-r, r, x >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params[0];
    let op = params[1];
    if i >= n { return; }
    let v = a[i];
    if op == 0u { out[i] = -v; }
    else if op == 1u { out[i] = exp(v); }
    else if op == 2u { out[i] = log(v); }
    else if op == 3u { out[i] = log(1.0 + v); }
    else if op == 4u { out[i] = sin(v); }
    else if op == 5u { out[i] = cos(v); }
    else if op == 6u { out[i] = tanh(v); }
    else if op == 7u { out[i] = sqrt(v); }
    else if op == 8u { out[i] = abs(v); }
    else if op == 9u { out[i] = 1.0 / v; }
    else if op == 10u { out[i] = erf_approx(v); }
    else if op == 11u { out[i] = ceil(v); }
    else if op == 12u { out[i] = floor(v); }
    else { out[i] = round(v); }
}
"#;

// powf: out[i] = a[i]^p
const SHADER_POWF: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n, power_bits]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params[0];
    if i >= n { return; }
    let p = bitcast<f32>(params[1]);
    out[i] = pow(a[i], p);
}
"#;

// Transpose: out[col*rows + row] = a[row*cols + col]
const SHADER_TRANSPOSE: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [rows, cols]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let rows = params[0];
    let cols = params[1];
    if i >= rows * cols { return; }
    let row = i / cols;
    let col = i % cols;
    out[col * rows + row] = a[i];
}
"#;

// Copy: out[i] = a[i]
const SHADER_COPY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params[0] { return; }
    out[i] = a[i];
}
"#;

// Tiled matmul: TILE=16, workgroup 16×16
const SHADER_MATMUL: &str = r#"
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>; // [m, k, n, grid_cols]

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = params[0];
    let k = params[1];
    let n = params[2];
    let grid_cols = params[3];

    let block_row = wgid.x / grid_cols;
    let block_col = wgid.x % grid_cols;
    let ty = lid.y;
    let tx = lid.x;
    let row = block_row * 16u + ty;
    let col = block_col * 16u + tx;

    var sum: f32 = 0.0;
    let n_tiles = (k + 15u) / 16u;

    for (var t: u32 = 0u; t < n_tiles; t++) {
        let a_col = t * 16u + tx;
        if row < m && a_col < k {
            tile_a[ty * 16u + tx] = a[row * k + a_col];
        } else {
            tile_a[ty * 16u + tx] = 0.0;
        }
        let b_row = t * 16u + ty;
        if b_row < k && col < n {
            tile_b[ty * 16u + tx] = b[b_row * n + col];
        } else {
            tile_b[ty * 16u + tx] = 0.0;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk++) {
            sum += tile_a[ty * 16u + kk] * tile_b[kk * 16u + tx];
        }
        workgroupBarrier();
    }
    if row < m && col < n {
        out[row * n + col] = sum;
    }
}
"#;

// Fill zeros
const SHADER_FILL_ZEROS: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params[0] { return; }
    out[gid.x] = 0.0;
}
"#;

// Fill scalar
const SHADER_FILL_SCALAR: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>; // [n, val_bits]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params[0] { return; }
    out[gid.x] = bitcast<f32>(params[1]);
}
"#;

// Fill identity matrix
const SHADER_FILL_IDENTITY: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params[0];
    if i >= n * n { return; }
    let row = i / n;
    let col = i % n;
    out[i] = select(0.0, 1.0, row == col);
}
"#;

// Reduction sum: each workgroup (256 threads) writes partial sum to out[workgroup_id]
const SHADER_REDUCE_SUM: &str = r#"
var<workgroup> shared: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    shared[pos] = select(0.0, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < 256u; k++) { acc += shared[k]; }
        out[wgid.x] = acc;
    }
}
"#;

// Reduction max
const SHADER_REDUCE_MAX: &str = r#"
var<workgroup> shared: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    shared[pos] = select(-3.4028235e+38, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {
        var acc: f32 = shared[0];
        for (var k: u32 = 1u; k < 256u; k++) {
            if shared[k] > acc { acc = shared[k]; }
        }
        out[wgid.x] = acc;
    }
}
"#;

// Reduction min
const SHADER_REDUCE_MIN: &str = r#"
var<workgroup> shared: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    shared[pos] = select(3.4028235e+38, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {
        var acc: f32 = shared[0];
        for (var k: u32 = 1u; k < 256u; k++) {
            if shared[k] < acc { acc = shared[k]; }
        }
        out[wgid.x] = acc;
    }
}
"#;

// Argmax: each workgroup writes (best_val, best_idx) to vals[wgid] / idxs[wgid]
const SHADER_ARGMAX: &str = r#"
var<workgroup> shared_v: array<f32, 256>;
var<workgroup> shared_i: array<u32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> vals: array<f32>;
@group(0) @binding(2) var<storage, read_write> idxs: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    if i < n {
        shared_v[pos] = input[i];
        shared_i[pos] = i;
    } else {
        shared_v[pos] = -3.4028235e+38;
        shared_i[pos] = 0xFFFFFFFFu;
    }
    workgroupBarrier();
    if pos == 0u {
        var bv: f32 = shared_v[0];
        var bi: u32 = shared_i[0];
        for (var k: u32 = 1u; k < 256u; k++) {
            let v = shared_v[k];
            let ki = shared_i[k];
            if v > bv || (v == bv && ki < bi) { bv = v; bi = ki; }
        }
        vals[wgid.x] = bv;
        idxs[wgid.x] = bi;
    }
}
"#;

// Argmin
const SHADER_ARGMIN: &str = r#"
var<workgroup> shared_v: array<f32, 256>;
var<workgroup> shared_i: array<u32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> vals: array<f32>;
@group(0) @binding(2) var<storage, read_write> idxs: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>; // [n]

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    if i < n {
        shared_v[pos] = input[i];
        shared_i[pos] = i;
    } else {
        shared_v[pos] = 3.4028235e+38;
        shared_i[pos] = 0xFFFFFFFFu;
    }
    workgroupBarrier();
    if pos == 0u {
        var bv: f32 = shared_v[0];
        var bi: u32 = shared_i[0];
        for (var k: u32 = 1u; k < 256u; k++) {
            let v = shared_v[k];
            let ki = shared_i[k];
            if v < bv || (v == bv && ki < bi) { bv = v; bi = ki; }
        }
        vals[wgid.x] = bv;
        idxs[wgid.x] = bi;
    }
}
"#;

// ── GpuStorage ───────────────────────────────────────────────────────────────

/// Row-major GPU-backed matrix.
///
/// `buffer` owns device memory (RAII via wgpu's Arc-backed buffer).
/// `host_cache` is populated lazily on the first `get` call and invalidated on `set`.
pub struct GpuStorage<T: Scalar> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    buffer: wgpu::Buffer,
    host_cache: Mutex<Option<Vec<T>>>,
}

// SAFETY: wgpu::Buffer is Send+Sync (Arc-backed). Mutex<Option<Vec<T>>> is
// Send+Sync when T: Send+Sync, which the Scalar bound guarantees.
unsafe impl<T: Scalar> Send for GpuStorage<T> {}
unsafe impl<T: Scalar> Sync for GpuStorage<T> {}

/// Lock a mutex, recovering from poisoned state.
#[inline]
fn lock_or_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<T: Scalar> GpuStorage<T> {
    fn from_buffer(nrows: usize, ncols: usize, buffer: wgpu::Buffer) -> Self {
        Self {
            nrows,
            ncols,
            buffer,
            host_cache: Mutex::new(None),
        }
    }

    fn from_buffer_cached(nrows: usize, ncols: usize, buffer: wgpu::Buffer, cache: Vec<T>) -> Self {
        Self {
            nrows,
            ncols,
            buffer,
            host_cache: Mutex::new(Some(cache)),
        }
    }

    fn upload(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
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

    fn empty_buf(n_bytes: u64) -> wgpu::Buffer {
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

    fn fill_cache_mut(&self) -> std::sync::MutexGuard<'_, Option<Vec<T>>> {
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

// ── Params buffer helpers ─────────────────────────────────────────────────────

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
fn workgroups(n: usize) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        n.div_ceil(256) as u32
    }
}

// ── Core dispatch operations ──────────────────────────────────────────────────

fn run_binary_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n: usize,
    op_type: u32,
) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, op_type]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_BINARY);
    let bg = bind_group(
        ctx,
        &pipeline,
        &[(a, true), (b, true), (&out, false), (&params, true)],
    );
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_scale_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, scalar: f32) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, scalar.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_SCALE);
    let bg = bind_group(ctx, &pipeline, &[(a, true), (&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_unary_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, op_type: u32) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, op_type]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_UNARY);
    let bg = bind_group(ctx, &pipeline, &[(a, true), (&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_powf_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, power: f32) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, power.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_POWF);
    let bg = bind_group(ctx, &pipeline, &[(a, true), (&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_transpose_f32(ctx: &GpuContext, a: &wgpu::Buffer, rows: usize, cols: usize) -> wgpu::Buffer {
    let n = rows * cols;
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[rows as u32, cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_TRANSPOSE);
    let bg = bind_group(ctx, &pipeline, &[(a, true), (&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_copy_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_COPY);
    let bg = bind_group(ctx, &pipeline, &[(a, true), (&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_matmul_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> wgpu::Buffer {
    let grid_cols = n.div_ceil(16);
    let grid_rows = m.div_ceil(16);
    let grid_size = grid_rows * grid_cols;
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[m as u32, k as u32, n as u32, grid_cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((m * n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_MATMUL);
    let bg = bind_group(
        ctx,
        &pipeline,
        &[(a, true), (b, true), (&out, false), (&params, true)],
    );
    // Matmul uses workgroup_size(16,16); dispatch grid_size workgroups
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        #[allow(clippy::cast_possible_truncation)]
        pass.dispatch_workgroups(grid_size as u32, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    ctx.device.poll(wgpu::MaintainBase::Wait).panic_on_timeout();
    out
}

fn run_fill_zeros_f32(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_FILL_ZEROS);
    let bg = bind_group(ctx, &pipeline, &[(&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_fill_scalar_f32(ctx: &GpuContext, n: usize, val: f32) -> wgpu::Buffer {
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32, val.to_bits()]);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_FILL_SCALAR);
    let bg = bind_group(ctx, &pipeline, &[(&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(n));
    out
}

fn run_fill_identity_f32(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    let total = n * n;
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((total * 4) as u64);
    let pipeline = create_pipeline(ctx, SHADER_FILL_IDENTITY);
    let bg = bind_group(ctx, &pipeline, &[(&out, false), (&params, true)]);
    dispatch_and_wait(ctx, &pipeline, &bg, workgroups(total));
    out
}

// Reduction helpers — return raw bytes of partial results

fn run_reduce_f32(ctx: &GpuContext, a: &wgpu::Buffer, n: usize, shader: &str) -> Vec<f32> {
    let num_blocks = n.div_ceil(256);
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out = GpuStorage::<f32>::empty_buf((num_blocks * 4) as u64);
    let pipeline = create_pipeline(ctx, shader);
    let bg = bind_group(ctx, &pipeline, &[(a, true), (&out, false), (&params, true)]);
    #[allow(clippy::cast_possible_truncation)]
    dispatch_and_wait(ctx, &pipeline, &bg, num_blocks as u32);
    let bytes = readback(ctx, &out, (num_blocks * 4) as u64);
    // SAFETY: bytes from f32 buffer.
    unsafe { bytes_to_scalar::<f32>(&bytes) }
}

fn run_argreduce_f32(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    n: usize,
    shader: &str,
) -> (Vec<f32>, Vec<u32>) {
    let num_blocks = n.div_ceil(256);
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[n as u32]);
    let out_vals = GpuStorage::<f32>::empty_buf((num_blocks * 4) as u64);
    let out_idxs = GpuStorage::<f32>::empty_buf((num_blocks * 4) as u64);
    let pipeline = create_pipeline(ctx, shader);
    let bg = bind_group(
        ctx,
        &pipeline,
        &[
            (a, true),
            (&out_vals, false),
            (&out_idxs, false),
            (&params, true),
        ],
    );
    #[allow(clippy::cast_possible_truncation)]
    dispatch_and_wait(ctx, &pipeline, &bg, num_blocks as u32);
    let vbytes = readback(ctx, &out_vals, (num_blocks * 4) as u64);
    let ibytes = readback(ctx, &out_idxs, (num_blocks * 4) as u64);
    // SAFETY: bytes from valid f32 buffer.
    let vals: Vec<f32> = unsafe { bytes_to_scalar::<f32>(&vbytes) };
    let idxs: Vec<u32> = bytes_to_u32(&ibytes);
    (vals, idxs)
}

// ── pub(crate) GPU dispatch functions ────────────────────────────────────────

// TypeId guard: GPU only supports f32 (f64/c32/c64 → unreachable!())

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
    nrows: usize,
    ncols: usize,
    mut f: impl FnMut(usize, usize) -> T,
) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let data: Vec<T> = (0..nrows * ncols)
        .map(|i| f(i / ncols, i % ncols))
        .collect();
    GpuStorage::upload(nrows, ncols, data)
}

pub(crate) fn gpu_get<T: Scalar>(s: &GpuStorage<T>, r: usize, c: usize) -> T {
    assert_is_f32::<T>();
    let guard = s.fill_cache_mut();
    guard.as_ref().expect("cache populated")[r * s.ncols + c]
}

pub(crate) fn gpu_set<T: Scalar>(s: &mut GpuStorage<T>, r: usize, c: usize, v: T) {
    assert_is_f32::<T>();
    {
        let mut guard = s.fill_cache_mut();
        guard.as_mut().expect("cache populated")[r * s.ncols + c] = v;
    }
    let guard = lock_or_recover(&s.host_cache);
    let data = guard.as_ref().expect("cache populated");
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

pub(crate) fn gpu_add<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let buf = run_binary_f32(ctx, &a.buffer, &b.buffer, n, 0);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_sub<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let buf = run_binary_f32(ctx, &a.buffer, &b.buffer, n, 1);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

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

pub(crate) fn gpu_transpose<T: Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let buf = run_transpose_f32(ctx, &a.buffer, a.nrows, a.ncols);
    GpuStorage::from_buffer(a.ncols, a.nrows, buf)
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

pub(crate) fn gpu_mul_elem<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let buf = run_binary_f32(ctx, &a.buffer, &b.buffer, n, 2);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
}

pub(crate) fn gpu_div_elem<T: Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = a.nrows * a.ncols;
    let buf = run_binary_f32(ctx, &a.buffer, &b.buffer, n, 3);
    GpuStorage::from_buffer(a.nrows, a.ncols, buf)
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

// Unary math ops: indexed by op_type matching SHADER_UNARY
// 0=neg(already covered), 1=exp, 2=ln, 3=log1p, 4=sin, 5=cos, 6=tanh, 7=sqrt,
// 8=abs, 9=recip, 10=erf, 11=ceil, 12=floor, 13=round

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

// ── Reduction ops ─────────────────────────────────────────────────────────────

pub(crate) fn gpu_sum_all<T: Scalar>(s: &GpuStorage<T>) -> T {
    assert_is_f32::<T>();
    let ctx = get_context();
    let n = s.nrows * s.ncols;
    let partials = run_reduce_f32(ctx, &s.buffer, n, SHADER_REDUCE_SUM);
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
    let partials = run_reduce_f32(ctx, &s.buffer, n, SHADER_REDUCE_MAX);
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
    let partials = run_reduce_f32(ctx, &s.buffer, n, SHADER_REDUCE_MIN);
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
    let (vals, idxs) = run_argreduce_f32(ctx, &s.buffer, n, SHADER_ARGMAX);
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
    let (vals, idxs) = run_argreduce_f32(ctx, &s.buffer, n, SHADER_ARGMIN);
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

// ── TypeId guard ──────────────────────────────────────────────────────────────

#[inline]
fn assert_is_f32<T: Scalar>() {
    assert!(
        TypeId::of::<T>() == TypeId::of::<f32>(),
        "GPU backend only supports f32; got a different scalar type"
    );
}

// ── Backend impl for Gpu ──────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl crate::backend::Backend for crate::backend::Gpu {
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
        gpu_set::<T>(s, r, c, v)
    }

    #[inline]
    fn matmul_into<T: crate::scalar::Scalar>(
        out: &mut GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
    ) {
        gpu_matmul::<T>(out, a, b)
    }

    #[inline]
    fn add<T: crate::scalar::Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_add::<T>(a, b)
    }

    #[inline]
    fn sub<T: crate::scalar::Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_sub::<T>(a, b)
    }

    #[inline]
    fn neg<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_neg::<T>(a)
    }

    #[inline]
    fn transpose<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_transpose::<T>(a)
    }

    #[inline]
    fn scale<T: crate::scalar::Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
        gpu_scale::<T>(a, s)
    }

    #[inline]
    fn clone_storage<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_clone::<T>(s)
    }

    #[inline]
    fn exp<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_exp::<T>(a)
    }

    #[inline]
    fn ln<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_ln::<T>(a)
    }

    #[inline]
    fn log1p<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_log1p::<T>(a)
    }

    #[inline]
    fn sin<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_sin::<T>(a)
    }

    #[inline]
    fn cos<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_cos::<T>(a)
    }

    #[inline]
    fn tanh<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_tanh::<T>(a)
    }

    #[inline]
    fn sqrt<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_sqrt::<T>(a)
    }

    #[inline]
    fn abs<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_abs::<T>(a)
    }

    #[inline]
    fn recip<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_recip::<T>(a)
    }

    #[inline]
    fn erf<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_erf::<T>(a)
    }

    #[inline]
    fn ceil<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_ceil::<T>(a)
    }

    #[inline]
    fn floor<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_floor::<T>(a)
    }

    #[inline]
    fn round<T: crate::scalar::Scalar>(a: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_round::<T>(a)
    }

    #[inline]
    fn powf<T: crate::scalar::Scalar>(a: &GpuStorage<T>, p: T) -> GpuStorage<T> {
        gpu_powf::<T>(a, p)
    }

    #[inline]
    fn mul_elem<T: crate::scalar::Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_mul_elem::<T>(a, b)
    }

    #[inline]
    fn div_elem<T: crate::scalar::Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_div_elem::<T>(a, b)
    }

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
}
