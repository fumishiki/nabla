// gpu.rs — wgpu-backed GPU storage and WGSL compute shader dispatch.
//
// Design:
//   - GpuContext (OnceLock singleton) owns wgpu::Device + wgpu::Queue.
//   - GpuStorage<T> owns a wgpu::Buffer (RAII, Arc-backed via wgpu internals).
//   - GPU ops are f32-only; TypeId dispatch guards all entry points.
//   - host_cache: Mutex<Option<Vec<T>>> — lazily populated on first get/readback.
//   - Pipeline cache: Mutex<HashMap<&'static str, wgpu::ComputePipeline>>.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use wgpu::util::DeviceExt;

use crate::scalar::Scalar;

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
    wg_size: u32,
    pipelines: Mutex<HashMap<PipelineKey, wgpu::ComputePipeline>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ShaderOp {
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
    Argmax,
    Argmin,
    Matmul { tile: u32 },
    MatmulRegTile { tr: u32, tc: u32, bm: u32, bn: u32, bk: u32 },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    op: ShaderOp,
    wg_size: u32,
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
    let max_wg = device.limits().max_compute_workgroup_size_x;
    let wg_size = if max_wg >= 512 { 256 } else if max_wg >= 256 { 128 } else { 64 };
    GpuContext {
        device,
        queue,
        wg_size,
        pipelines: Mutex::new(HashMap::new()),
    }
}

fn compile_pipeline(ctx: &GpuContext, shader_src: &str) -> wgpu::ComputePipeline {
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

fn with_pipeline<R>(ctx: &GpuContext, key: PipelineKey, f: impl FnOnce(&wgpu::ComputePipeline) -> R) -> R {
    let mut cache = lock_or_recover(&ctx.pipelines);
    cache.entry(key).or_insert_with(|| {
        let src = generate_shader(key);
        compile_pipeline(ctx, &src)
    });
    f(cache.get(&key).expect("pipeline just inserted"))
}

// ── Dispatch helper ───────────────────────────────────────────────────────────

fn bind_group(
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

// ── WGSL shader generation ───────────────────────────────────────────────────

fn generate_shader(key: PipelineKey) -> String {
    let wg = key.wg_size;
    match key.op {
        ShaderOp::Binary => gen_binary(wg),
        ShaderOp::Scale => gen_scale(wg),
        ShaderOp::Unary => gen_unary(wg),
        ShaderOp::Powf => gen_powf(wg),
        ShaderOp::Transpose => gen_transpose(wg),
        ShaderOp::Copy => gen_copy(wg),
        ShaderOp::FillZeros => gen_fill_zeros(wg),
        ShaderOp::FillScalar => gen_fill_scalar(wg),
        ShaderOp::FillIdentity => gen_fill_identity(wg),
        ShaderOp::ReduceSum => gen_reduce_sum(wg),
        ShaderOp::ReduceMax => gen_reduce_max(wg),
        ShaderOp::ReduceMin => gen_reduce_min(wg),
        ShaderOp::Argmax => gen_argmax(wg),
        ShaderOp::Argmin => gen_argmin(wg),
        ShaderOp::Matmul { tile } => gen_matmul(tile),
        ShaderOp::MatmulRegTile { tr, tc, bm, bn, bk } => gen_matmul_register_tile(tr, tc, bm, bn, bk),
    }
}

fn gen_binary(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    let op = params[1];
    if i >= n {{ return; }}
    if op == 0u {{ out[i] = a[i] + b[i]; }}
    else if op == 1u {{ out[i] = a[i] - b[i]; }}
    else if op == 2u {{ out[i] = a[i] * b[i]; }}
    else {{ out[i] = a[i] / b[i]; }}
}}
")
}

fn gen_scale(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let scalar = bitcast<f32>(params[1]);
    out[i] = a[i] * scalar;
}}
")
}

fn gen_unary(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
fn erf_approx(x: f32) -> f32 {{
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let r = 1.0 - poly * exp(-ax * ax);
    return select(-r, r, x >= 0.0);
}}
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    let op = params[1];
    if i >= n {{ return; }}
    let v = a[i];
    if op == 0u {{ out[i] = -v; }}
    else if op == 1u {{ out[i] = exp(v); }}
    else if op == 2u {{ out[i] = log(v); }}
    else if op == 3u {{ out[i] = log(1.0 + v); }}
    else if op == 4u {{ out[i] = sin(v); }}
    else if op == 5u {{ out[i] = cos(v); }}
    else if op == 6u {{ out[i] = tanh(v); }}
    else if op == 7u {{ out[i] = sqrt(v); }}
    else if op == 8u {{ out[i] = abs(v); }}
    else if op == 9u {{ out[i] = 1.0 / v; }}
    else if op == 10u {{ out[i] = erf_approx(v); }}
    else if op == 11u {{ out[i] = ceil(v); }}
    else if op == 12u {{ out[i] = floor(v); }}
    else {{ out[i] = round(v); }}
}}
")
}

fn gen_powf(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let p = bitcast<f32>(params[1]);
    out[i] = pow(a[i], p);
}}
")
}

fn gen_transpose(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let rows = params[0];
    let cols = params[1];
    if i >= rows * cols {{ return; }}
    let row = i / cols;
    let col = i % cols;
    out[col * rows + row] = a[i];
}}
")
}

fn gen_copy(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if i >= params[0] {{ return; }}
    out[i] = a[i];
}}
")
}

fn gen_matmul(tile: u32) -> String {
    let tile_sq = tile * tile;
    let tile_m1 = tile - 1;
    format!(r"
var<workgroup> tile_a: array<f32, {tile_sq}>;
var<workgroup> tile_b: array<f32, {tile_sq}>;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({tile}, {tile})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let m = params[0];
    let k = params[1];
    let n = params[2];
    let grid_cols = params[3];
    let block_row = wgid.x / grid_cols;
    let block_col = wgid.x % grid_cols;
    let ty = lid.y;
    let tx = lid.x;
    let row = block_row * {tile}u + ty;
    let col = block_col * {tile}u + tx;
    var sum: f32 = 0.0;
    let n_tiles = (k + {tile_m1}u) / {tile}u;
    for (var t: u32 = 0u; t < n_tiles; t++) {{
        let a_col = t * {tile}u + tx;
        if row < m && a_col < k {{
            tile_a[ty * {tile}u + tx] = a[row * k + a_col];
        }} else {{
            tile_a[ty * {tile}u + tx] = 0.0;
        }}
        let b_row = t * {tile}u + ty;
        if b_row < k && col < n {{
            tile_b[ty * {tile}u + tx] = b[b_row * n + col];
        }} else {{
            tile_b[ty * {tile}u + tx] = 0.0;
        }}
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < {tile}u; kk++) {{
            sum += tile_a[ty * {tile}u + kk] * tile_b[kk * {tile}u + tx];
        }}
        workgroupBarrier();
    }}
    if row < m && col < n {{
        out[row * n + col] = sum;
    }}
}}
")
}

fn gen_matmul_register_tile(tr: u32, tc: u32, bm: u32, bn: u32, bk: u32) -> String {
    crate::wgsl::gen_matmul_register_tile(tr, tc, bm, bn, bk)
}

fn select_register_tile_params(m: usize, n: usize, k: usize) -> (u32, u32, u32, u32, u32) {
    crate::wgsl::select_register_tile_params(m, n, k)
}

fn gen_fill_zeros(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params[0] {{ return; }}
    out[gid.x] = 0.0;
}}
")
}

fn gen_fill_scalar(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params[0] {{ return; }}
    out[gid.x] = bitcast<f32>(params[1]);
}}
")
}

fn gen_fill_identity(wg: u32) -> String {
    format!(r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n * n {{ return; }}
    let row = i / n;
    let col = i % n;
    out[i] = select(0.0, 1.0, row == col);
}}
")
}

fn gen_reduce_sum(wg: u32) -> String {
    format!(r"
var<workgroup> shared: array<f32, {wg}>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    shared[pos] = select(0.0, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {{
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < {wg}u; k++) {{ acc += shared[k]; }}
        out[wgid.x] = acc;
    }}
}}
")
}

fn gen_reduce_max(wg: u32) -> String {
    format!(r"
var<workgroup> shared: array<f32, {wg}>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    shared[pos] = select(-3.4028235e+38, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {{
        var acc: f32 = shared[0];
        for (var k: u32 = 1u; k < {wg}u; k++) {{
            if shared[k] > acc {{ acc = shared[k]; }}
        }}
        out[wgid.x] = acc;
    }}
}}
")
}

fn gen_reduce_min(wg: u32) -> String {
    format!(r"
var<workgroup> shared: array<f32, {wg}>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    shared[pos] = select(3.4028235e+38, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {{
        var acc: f32 = shared[0];
        for (var k: u32 = 1u; k < {wg}u; k++) {{
            if shared[k] < acc {{ acc = shared[k]; }}
        }}
        out[wgid.x] = acc;
    }}
}}
")
}

fn gen_argmax(wg: u32) -> String {
    format!(r"
var<workgroup> shared_v: array<f32, {wg}>;
var<workgroup> shared_i: array<u32, {wg}>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> vals: array<f32>;
@group(0) @binding(2) var<storage, read_write> idxs: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    if i < n {{
        shared_v[pos] = input[i];
        shared_i[pos] = i;
    }} else {{
        shared_v[pos] = -3.4028235e+38;
        shared_i[pos] = 0xFFFFFFFFu;
    }}
    workgroupBarrier();
    if pos == 0u {{
        var bv: f32 = shared_v[0];
        var bi: u32 = shared_i[0];
        for (var k: u32 = 1u; k < {wg}u; k++) {{
            let v = shared_v[k];
            let ki = shared_i[k];
            if v > bv || (v == bv && ki < bi) {{ bv = v; bi = ki; }}
        }}
        vals[wgid.x] = bv;
        idxs[wgid.x] = bi;
    }}
}}
")
}

fn gen_argmin(wg: u32) -> String {
    format!(r"
var<workgroup> shared_v: array<f32, {wg}>;
var<workgroup> shared_i: array<u32, {wg}>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> vals: array<f32>;
@group(0) @binding(2) var<storage, read_write> idxs: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let n = params[0];
    let i = gid.x;
    let pos = lid.x;
    if i < n {{
        shared_v[pos] = input[i];
        shared_i[pos] = i;
    }} else {{
        shared_v[pos] = 3.4028235e+38;
        shared_i[pos] = 0xFFFFFFFFu;
    }}
    workgroupBarrier();
    if pos == 0u {{
        var bv: f32 = shared_v[0];
        var bi: u32 = shared_i[0];
        for (var k: u32 = 1u; k < {wg}u; k++) {{
            let v = shared_v[k];
            let ki = shared_i[k];
            if v < bv || (v == bv && ki < bi) {{ bv = v; bi = ki; }}
        }}
        vals[wgid.x] = bv;
        idxs[wgid.x] = bi;
    }}
}}
")
}

// Select matmul tile size based on matrix dimensions
fn select_matmul_tile(m: usize, k: usize, n: usize) -> u32 {
    let max_dim = m.max(k).max(n);
    if max_dim < 64 { 8 } else if max_dim >= 256 { 32 } else { 16 }
}

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
fn workgroups(n: usize, wg_size: u32) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        n.div_ceil(wg_size as usize) as u32
    }
}

// ── Core dispatch operations ──────────────────────────────────────────────────

/// Single-input GPU dispatch: input → params → output.
#[allow(clippy::cast_possible_truncation)]
fn run_1in(ctx: &GpuContext, op: ShaderOp, a: &wgpu::Buffer, n: usize, params: &[u32]) -> wgpu::Buffer {
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

/// Two-input GPU dispatch: a, b → params → output.
#[allow(clippy::cast_possible_truncation)]
fn run_2in(ctx: &GpuContext, op: ShaderOp, a: &wgpu::Buffer, b: &wgpu::Buffer, n: usize, params: &[u32]) -> wgpu::Buffer {
    let wg = ctx.wg_size;
    let key = PipelineKey { op, wg_size: wg };
    let p = params_buf(params);
    let out = GpuStorage::<f32>::empty_buf((n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (b, true), (&out, false), (&p, true)]);
        dispatch_and_wait(ctx, pipeline, &bg, workgroups(n, wg));
    });
    out
}

#[allow(clippy::cast_possible_truncation)]
fn run_binary_f32(ctx: &GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer, n: usize, op_type: u32) -> wgpu::Buffer {
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
    run_1in(ctx, ShaderOp::Transpose, a, rows * cols, &[rows as u32, cols as u32])
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
    let key = PipelineKey { op: ShaderOp::Matmul { tile }, wg_size: tile };
    #[allow(clippy::cast_possible_truncation)]
    let params = params_buf(&[m as u32, k as u32, n as u32, grid_cols as u32]);
    let out = GpuStorage::<f32>::empty_buf((m * n * 4) as u64);
    with_pipeline(ctx, key, |pipeline| {
        let bg = bind_group(ctx, pipeline, &[(a, true), (b, true), (&out, false), (&params, true)]);
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
        let bg = bind_group(ctx, pipeline, &[(a, true), (b, true), (&out, false), (&params, true)]);
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
    let key = PipelineKey { op: ShaderOp::FillZeros, wg_size: wg };
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
    let key = PipelineKey { op: ShaderOp::FillScalar, wg_size: wg };
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
    let key = PipelineKey { op: ShaderOp::FillIdentity, wg_size: wg };
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

// Reduction helpers — return raw bytes of partial results

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
            &[(a, true), (&out_vals, false), (&out_idxs, false), (&params, true)],
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
        gpu_set::<T>(s, r, c, v);
    }

    #[inline]
    fn matmul_into<T: crate::scalar::Scalar>(
        out: &mut GpuStorage<T>,
        a: &GpuStorage<T>,
        b: &GpuStorage<T>,
    ) {
        gpu_matmul::<T>(out, a, b);
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
                fn $trait_fn<T: crate::scalar::Scalar>(a: &GpuStorage<T>, b: &GpuStorage<T>) -> GpuStorage<T> {
                    $impl_fn::<T>(a, b)
                }
            )+
        };
    }

    gpu_fwd_unary!(
        neg => gpu_neg, transpose => gpu_transpose,
        exp => gpu_exp, ln => gpu_ln, log1p => gpu_log1p,
        sin => gpu_sin, cos => gpu_cos, tanh => gpu_tanh,
        sqrt => gpu_sqrt, abs => gpu_abs, recip => gpu_recip,
        erf => gpu_erf, ceil => gpu_ceil, floor => gpu_floor, round => gpu_round,
    );
    gpu_fwd_binary!(add => gpu_add, sub => gpu_sub, emul => gpu_emul, ediv => gpu_ediv);

    #[inline]
    fn scale<T: crate::scalar::Scalar>(a: &GpuStorage<T>, s: T) -> GpuStorage<T> {
        gpu_scale::<T>(a, s)
    }

    #[inline]
    fn clone_storage<T: crate::scalar::Scalar>(s: &GpuStorage<T>) -> GpuStorage<T> {
        gpu_clone::<T>(s)
    }

    #[inline]
    fn powf<T: crate::scalar::Scalar>(a: &GpuStorage<T>, p: T) -> GpuStorage<T> {
        gpu_powf::<T>(a, p)
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

/// WGSL compute shader for BCSR SpMM (Block Compressed Sparse Row × Dense).
///
/// Bindings:
///   0: row_ptrs (array<u32>)     — block-row pointers
///   1: col_idxs (array<u32>)     — block-column indices
///   2: values   (array<f32>)     — B×B dense blocks, row-major
///   3: x        (array<f32>)     — dense RHS matrix, row-major (ncols × k)
///   4: out      (array<f32>)     — output matrix, row-major (nrows × k)
///   5: params   (array<u32>)     — [nrows, ncols, k, block_size, nblock_rows]
///
/// Dispatch: one workgroup per block-row. Each thread handles one output row
/// within the block-row, iterating over all non-zero blocks in that row.
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
