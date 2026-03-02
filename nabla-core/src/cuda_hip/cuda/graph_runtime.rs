use std::cell::Cell;
use std::ffi::{CString, c_void};
use std::sync::Arc;

use cudarc::driver::sys::{CUdeviceptr, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph as CudarcCudaGraph, result};
use cudarc::nvrtc;

use crate::gpu_common::{lock_or_recover, type_suffix};
use crate::kernels_cu;
use crate::scalar::Scalar;

use super::*;

thread_local! {
    static CAPTURE_DEPTH: Cell<usize> = Cell::new(0);
}

pub(super) fn cuda_graph_is_capturing() -> bool {
    CAPTURE_DEPTH.with(|depth| depth.get() > 0)
}

struct CaptureGuard;

impl CaptureGuard {
    fn new() -> Self {
        CAPTURE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURE_DEPTH.with(|depth| {
            let v = depth.get();
            depth.set(v.saturating_sub(1));
        });
    }
}

fn clear_stream_capture(stream: cudarc::driver::sys::CUstream) {
    let status = unsafe { result::stream::is_capturing(stream) };
    if let Ok(status) = status {
        if status != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            if let Ok(graph) = unsafe { result::stream::end_capture(stream) } {
                if !graph.is_null() {
                    unsafe { cudarc::driver::sys::cuGraphDestroy(graph) };
                }
            }
        }
    }
}

pub struct NablaCudaGraph {
    inner: CudarcCudaGraph,
}

// SAFETY: CudarcCudaGraph holds Arc<CudaStream> which is Send+Sync.
unsafe impl Send for NablaCudaGraph {}
unsafe impl Sync for NablaCudaGraph {}

impl NablaCudaGraph {
    fn begin_capture() -> CudaResult<()> {
        let ctx = get_ctx();
        ctx.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        Ok(())
    }

    fn end_capture() -> CudaResult<Self> {
        let ctx = get_ctx();
        let flags: cudarc::driver::sys::CUgraphInstantiate_flags =
            unsafe { std::mem::transmute(0u32) };
        let graph = match ctx.stream.end_capture(flags) {
            Ok(g) => g.ok_or(CudaError::NullPtr)?,
            Err(e) => {
                clear_stream_capture(ctx.stream.cu_stream());
                return Err(CudaError::Driver(e));
            }
        };
        Ok(Self { inner: graph })
    }

    pub fn launch(&self) -> CudaResult<()> {
        self.inner.launch()?;
        Ok(())
    }
}

pub fn cuda_graph_capture<F: FnOnce()>(f: F) -> CudaResult<NablaCudaGraph> {
    NablaCudaGraph::begin_capture()?;
    let _guard = CaptureGuard::new();
    f();
    NablaCudaGraph::end_capture()
}

pub fn cuda_graph_capture_cached<F: FnOnce()>(name: &str, f: F) -> CudaResult<Arc<NablaCudaGraph>> {
    let ctx = get_ctx();
    {
        let cache: std::sync::MutexGuard<
            '_,
            std::collections::HashMap<String, Arc<NablaCudaGraph>>,
        > = lock_or_recover(&ctx.graphs);
        if let Some(g) = cache.get(name) {
            return Ok(Arc::clone(g));
        }
    }
    let graph = cuda_graph_capture(f)?;
    let graph = Arc::new(graph);
    let mut cache: std::sync::MutexGuard<
        '_,
        std::collections::HashMap<String, Arc<NablaCudaGraph>>,
    > = lock_or_recover(&ctx.graphs);
    cache.insert(name.to_string(), Arc::clone(&graph));
    Ok(graph)
}

pub fn cuda_to_vec_async<T: Scalar>(storage: &CudaStorage<T>) -> CudaResult<Vec<T>> {
    if cuda_graph_is_capturing() {
        panic!("CUDA Graph capture forbids D2H readback; avoid to_vec() during capture.");
    }
    let ctx = get_ctx();
    let n = storage.n();
    let mut host = vec![T::zero(); n];
    if n == 0 {
        return Ok(host);
    }
    // Serialize concurrent D2H calls: shared streams require exclusive access.
    // Also ensures the CUDA context is current on this thread before raw driver calls.
    let _d2h_guard = ctx.d2h_mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    ctx.stream.context().bind_to_thread().map_err(CudaError::Driver)?;
    // SAFETY: event lifecycle is contained within this function; the event is
    unsafe {
        let event =
            result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(CudaError::Driver)?;
        result::event::record(event, ctx.stream.cu_stream()).map_err(CudaError::Driver)?;
        result::stream::wait_event(
            ctx.copy_stream.cu_stream(),
            event,
            cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )
        .map_err(CudaError::Driver)?;
        result::memcpy_dtoh_async(&mut host, storage.buf.ptr, ctx.copy_stream.cu_stream())
            .map_err(CudaError::Driver)?;
        ctx.copy_stream.synchronize().map_err(CudaError::Driver)?;
        result::event::destroy(event).map_err(CudaError::Driver)?;
    }
    Ok(host)
}

pub fn cuda_copy_from_host<T: Scalar>(storage: &CudaStorage<T>, data: &[T]) {
    assert_eq!(
        storage.n(),
        data.len(),
        "cuda_copy_from_host: size mismatch (buffer={}, data={})",
        storage.n(),
        data.len()
    );
    let ctx = get_ctx();
    // SAFETY: copying from host slice to pre-allocated GPU buffer of matching size.
    unsafe {
        result::memcpy_htod_async(storage.buf.ptr, data, ctx.stream.cu_stream())
            .or_panic("CUDA copy_from_host");
    }
    if let Ok(mut cache) = storage.host_cache.lock() {
        *cache = None;
    }
}

pub struct KernelNodeState {
    node: cudarc::driver::sys::CUgraphNode,
    params: cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS,
    arg_bytes: Vec<u64>,
    arg_ptrs: Vec<*mut c_void>,
}

// SAFETY: arg_ptrs are raw pointers into arg_bytes which we own exclusively.
unsafe impl Send for KernelNodeState {}

pub struct PyGraph {
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    cu_stream: cudarc::driver::sys::CUstream,
    pub kernel_nodes: Vec<KernelNodeState>,
}

// SAFETY: PyGraph is used from a single training thread; raw handles are valid
unsafe impl Send for PyGraph {}

impl Drop for PyGraph {
    fn drop(&mut self) {
        // SAFETY: both handles are non-null (set in PyGraph::capture) and we
        unsafe {
            cudarc::driver::sys::cuGraphExecDestroy(self.cu_graph_exec);
            cudarc::driver::sys::cuGraphDestroy(self.cu_graph);
        }
    }
}

impl PyGraph {
    pub fn capture<F: FnOnce()>(f: F) -> CudaResult<Self> {
        let ctx = get_ctx();
        let cu_stream = ctx.stream.cu_stream();

        ctx.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let _guard = CaptureGuard::new();

        f();
        drop(_guard);

        let cu_graph = match unsafe { result::stream::end_capture(cu_stream) } {
            Ok(g) => g,
            Err(e) => {
                clear_stream_capture(cu_stream);
                return Err(CudaError::Driver(e));
            }
        };
        if cu_graph.is_null() {
            return Err(CudaError::NullPtr);
        }

        let cu_graph_exec = unsafe {
            // SAFETY: cu_graph is a valid non-null graph from cuStreamEndCapture.
            let mut exec = std::mem::MaybeUninit::uninit();
            if let Err(e) =
                cudarc::driver::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), cu_graph, 0)
                    .result()
            {
                cudarc::driver::sys::cuGraphDestroy(cu_graph);
                return Err(CudaError::Driver(e));
            }
            exec.assume_init()
        };
        if cu_graph_exec.is_null() {
            unsafe {
                cudarc::driver::sys::cuGraphDestroy(cu_graph);
            }
            return Err(CudaError::NullPtr);
        }

        let kernel_nodes = Self::collect_kernel_nodes(cu_graph)?;

        Ok(Self {
            cu_graph,
            cu_graph_exec,
            cu_stream,
            kernel_nodes,
        })
    }

    fn collect_kernel_nodes(
        cu_graph: cudarc::driver::sys::CUgraph,
    ) -> CudaResult<Vec<KernelNodeState>> {
        use cudarc::driver::sys::{CUDA_KERNEL_NODE_PARAMS, CUgraphNodeType};

        let node_count = unsafe {
            // SAFETY: cu_graph is valid; passing null nodes ptr queries the count.
            let mut n: usize = 0;
            cudarc::driver::sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n)
                .result()
                .map_err(CudaError::Driver)?;
            n
        };

        if node_count == 0 {
            return Ok(Vec::new());
        }

        let mut nodes = vec![std::ptr::null_mut(); node_count];
        unsafe {
            // SAFETY: nodes is sized to node_count from the count query above.
            let mut n = node_count;
            cudarc::driver::sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut n)
                .result()
                .map_err(CudaError::Driver)?;
        }

        let mut kernel_nodes = Vec::new();

        for node in nodes {
            let node_type = unsafe {
                // SAFETY: node is a valid graph node handle from cuGraphGetNodes.
                let mut t: CUgraphNodeType = std::mem::zeroed();
                cudarc::driver::sys::cuGraphNodeGetType(node, &mut t)
                    .result()
                    .map_err(CudaError::Driver)?;
                t
            };

            if node_type != CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                continue;
            }

            let raw_params = unsafe {
                // SAFETY: node is a kernel node (type checked above).
                let mut p: CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
                cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(node, &mut p)
                    .result()
                    .map_err(CudaError::Driver)?;
                p
            };

            let n_args = if raw_params.kernelParams.is_null() {
                0
            } else {
                let mut count = 0usize;
                // SAFETY: kernelParams is a null-terminated *mut *mut c_void array
                unsafe {
                    while !(*raw_params.kernelParams.add(count)).is_null() {
                        count += 1;
                    }
                }
                count
            };

            let mut arg_bytes: Vec<u64> = vec![0u64; n_args];
            if n_args > 0 {
                // SAFETY: kernelParams[i] points to the snapshotted arg value
                unsafe {
                    for i in 0..n_args {
                        let src = (*raw_params.kernelParams.add(i)) as *const u64;
                        arg_bytes[i] = src.read_unaligned();
                    }
                }
            }

            let mut arg_ptrs: Vec<*mut c_void> = arg_bytes
                .iter_mut()
                .map(|v| (v as *mut u64).cast::<c_void>())
                .collect();
            arg_ptrs.push(std::ptr::null_mut());

            let mut params = raw_params;
            params.kernelParams = arg_ptrs.as_mut_ptr();
            params.extra = std::ptr::null_mut();

            kernel_nodes.push(KernelNodeState {
                node,
                params,
                arg_bytes,
                arg_ptrs,
            });
        }

        Ok(kernel_nodes)
    }

    pub fn launch(&self) -> CudaResult<()> {
        // SAFETY: cu_graph_exec and cu_stream are valid for the lifetime of self.
        unsafe {
            cudarc::driver::sys::cuGraphLaunch(self.cu_graph_exec, self.cu_stream)
                .result()
                .map_err(CudaError::Driver)
        }
    }

    pub fn update_node_param_ptr(
        &mut self,
        node_idx: usize,
        param_idx: usize,
        new_ptr: CUdeviceptr,
    ) -> CudaResult<()> {
        let state = &mut self.kernel_nodes[node_idx];
        state.arg_bytes[param_idx] = new_ptr;

        // SAFETY: cu_graph_exec is valid; state.node is a kernel node in this graph;
        unsafe {
            cudarc::driver::sys::cuGraphExecKernelNodeSetParams_v2(
                self.cu_graph_exec,
                state.node,
                &state.params,
            )
            .result()
            .map_err(CudaError::Driver)
        }
    }

    #[must_use]
    pub fn kernel_node_count(&self) -> usize {
        self.kernel_nodes.len()
    }

    #[must_use]
    pub fn arg_count(&self, node_idx: usize) -> usize {
        self.kernel_nodes[node_idx].arg_bytes.len()
    }

    #[must_use]
    pub fn get_param(&self, node_idx: usize, param_idx: usize) -> CUdeviceptr {
        self.kernel_nodes[node_idx].arg_bytes[param_idx]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalKind {
    If,
    While,
    Switch { num_cases: u32 },
}

struct ConditionalHandle {
    raw: cudarc::driver::sys::CUgraphConditionalHandle,
}

// SAFETY: CUgraphConditionalHandle is an opaque u64 token; all mutation goes
unsafe impl Send for ConditionalHandle {}
unsafe impl Sync for ConditionalHandle {}

impl ConditionalHandle {
    unsafe fn new(
        graph: cudarc::driver::sys::CUgraph,
        cu_ctx: cudarc::driver::sys::CUcontext,
        default_value: u32,
    ) -> CudaResult<Self> {
        let mut raw: cudarc::driver::sys::CUgraphConditionalHandle = 0;
        // SAFETY: all pointers valid; pHandle_out points to a local stack slot.
        let r = unsafe {
            cudarc::driver::sys::cuGraphConditionalHandleCreate(
                &mut raw,
                graph,
                cu_ctx,
                default_value,
                cudarc::driver::sys::CU_GRAPH_COND_ASSIGN_DEFAULT,
            )
        };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::Driver(cudarc::driver::DriverError(r)));
        }
        Ok(Self { raw })
    }

    fn raw(&self) -> cudarc::driver::sys::CUgraphConditionalHandle {
        self.raw
    }
}

pub struct ConditionalGraph {
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_exec: cudarc::driver::sys::CUgraphExec,
    handle: ConditionalHandle,
}

// SAFETY: CUgraph / CUgraphExec are process-global opaque handles. We never
unsafe impl Send for ConditionalGraph {}
unsafe impl Sync for ConditionalGraph {}

impl Drop for ConditionalGraph {
    fn drop(&mut self) {
        // SAFETY: each pointer is destroyed exactly once; fields are set to
        if !self.cu_exec.is_null() {
            // SAFETY: cu_exec is a valid, live executable graph handle.
            unsafe { cudarc::driver::sys::cuGraphExecDestroy(self.cu_exec) };
            self.cu_exec = std::ptr::null_mut();
        }
        if !self.cu_graph.is_null() {
            // SAFETY: cu_graph is a valid, live graph handle.
            unsafe { cudarc::driver::sys::cuGraphDestroy(self.cu_graph) };
            self.cu_graph = std::ptr::null_mut();
        }
    }
}

impl ConditionalGraph {
    pub fn new<F>(kind: ConditionalKind, default_value: u32, bodies: &[F]) -> CudaResult<Self>
    where
        F: Fn(cudarc::driver::sys::CUgraph) -> CudaResult<()>,
    {
        let ctx = get_ctx();
        let cu_ctx = ctx.stream.context().cu_ctx();

        let mut cu_graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
        // SAFETY: cuGraphCreate writes a valid handle into cu_graph on success.
        let r = unsafe { cudarc::driver::sys::cuGraphCreate(&mut cu_graph, 0) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::Driver(cudarc::driver::DriverError(r)));
        }

        macro_rules! bail {
            ($e:expr) => {{
                // SAFETY: cu_graph is valid at this point; destroyed at most once.
                unsafe { cudarc::driver::sys::cuGraphDestroy(cu_graph) };
                return Err($e);
            }};
        }

        // SAFETY: cu_graph is valid and will outlive the handle.
        let handle = match unsafe { ConditionalHandle::new(cu_graph, cu_ctx, default_value) } {
            Ok(h) => h,
            Err(e) => bail!(e),
        };

        let (cond_type, size) = match kind {
            ConditionalKind::If => (
                cudarc::driver::sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
                bodies.len() as u32,
            ),
            ConditionalKind::While => (
                cudarc::driver::sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
                1u32,
            ),
            ConditionalKind::Switch { .. } => {
                bail!(CudaError::KernelNotFound(
                    "CUDA conditional switch not supported by this driver".to_string(),
                ));
            }
        };

        let mut body_graphs: Vec<cudarc::driver::sys::CUgraph> =
            vec![std::ptr::null_mut(); size as usize];

        // SAFETY: CUgraphNodeParams_st is a C struct; zeroing padding/reserved
        let mut node_params: cudarc::driver::sys::CUgraphNodeParams = unsafe { std::mem::zeroed() };
        node_params.type_ = cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        // SAFETY: type_ discriminant set above; writing the matching union arm
        unsafe {
            node_params.__bindgen_anon_1.conditional =
                cudarc::driver::sys::CUDA_CONDITIONAL_NODE_PARAMS {
                    handle: handle.raw(),
                    type_: cond_type,
                    size,
                    phGraph_out: body_graphs.as_mut_ptr(),
                    ctx: cu_ctx,
                };
        }

        let mut cond_node: cudarc::driver::sys::CUgraphNode = std::ptr::null_mut();
        // SAFETY: cu_graph, node_params, body_graphs are all valid.
        let r = unsafe {
            cudarc::driver::sys::cuGraphAddNode(
                &mut cond_node,
                cu_graph,
                std::ptr::null(),
                0,
                &mut node_params,
            )
        };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            bail!(CudaError::Driver(cudarc::driver::DriverError(r)));
        }

        for (i, body_fn) in bodies.iter().enumerate() {
            let body = body_graphs[i];
            if body.is_null() {
                bail!(CudaError::NullPtr);
            }
            if let Err(e) = body_fn(body) {
                bail!(e);
            }
        }

        let mut cu_exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: cu_graph is fully constructed; cu_exec receives the exec handle.
        let r = unsafe {
            cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut cu_exec, cu_graph, 0u64)
        };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            bail!(CudaError::Driver(cudarc::driver::DriverError(r)));
        }

        Ok(Self {
            cu_graph,
            cu_exec,
            handle,
        })
    }

    pub fn new_if<F>(body: F) -> CudaResult<Self>
    where
        F: Fn(cudarc::driver::sys::CUgraph) -> CudaResult<()>,
    {
        Self::new(ConditionalKind::If, 0, &[body])
    }

    pub fn new_if_else<F>(true_body: F, false_body: F) -> CudaResult<Self>
    where
        F: Fn(cudarc::driver::sys::CUgraph) -> CudaResult<()>,
    {
        Self::new(ConditionalKind::If, 0, &[true_body, false_body])
    }

    pub fn new_while<F>(body: F) -> CudaResult<Self>
    where
        F: Fn(cudarc::driver::sys::CUgraph) -> CudaResult<()>,
    {
        Self::new(ConditionalKind::While, 0, &[body])
    }

    pub fn launch(&self) -> CudaResult<()> {
        let ctx = get_ctx();
        // SAFETY: cu_exec and the stream are valid for this graph's lifetime.
        let r = unsafe { cudarc::driver::sys::cuGraphLaunch(self.cu_exec, ctx.stream.cu_stream()) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaError::Driver(cudarc::driver::DriverError(r)));
        }
        Ok(())
    }

    pub fn device_handle(&self) -> cudarc::driver::sys::CUgraphConditionalHandle {
        self.handle.raw()
    }
}

const COND_SET_KERNEL_NAMES: &[&str] = &["k_cond_set_f32", "k_cond_set_f16", "k_cond_set_f64"];

fn compile_cond_set_kernels(ctx: &CudaCtx, arch: &'static str) -> CudaResult<()> {
    {
        let map = ctx
            .kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.contains_key("k_cond_set_f32") {
            return Ok(());
        }
    }

    let ptx = nvrtc::compile_ptx_with_opts(
        kernels_cu::COND_SET_KERNELS,
        nvrtc::CompileOptions {
            arch: Some(arch),
            include_paths: nvrtc_include_paths(),
            ..Default::default()
        },
    )?;

    let ptx_src = ptx.to_src();
    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled PTX as a CUDA module.
    let module = unsafe { result::module::load_data(c_ptx.as_ptr().cast::<c_void>())? };

    let mut map = ctx
        .kernels
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &name in COND_SET_KERNEL_NAMES {
        let c_fn = CString::new(name).map_err(|_| CudaError::NullPtr)?;
        // SAFETY: retrieving function handle from the freshly loaded module.
        let func = unsafe { result::module::get_function(module, c_fn)? };
        map.insert(
            name.to_owned(),
            KernelEntry {
                func,
                _module: module,
            },
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCmp {
    Positive,
    Zero,
    LessThan,
}

pub fn cuda_conditional_set_from_scalar<T: Scalar>(
    handle: cudarc::driver::sys::CUgraphConditionalHandle,
    scalar_storage: &CudaStorage<T>,
    cmp: CondCmp,
    threshold: f32,
) -> CudaResult<()> {
    let ctx = get_ctx();
    let (major, minor) = query_compute_capability();
    let arch = nvrtc_arch(major, minor);
    compile_cond_set_kernels(ctx, arch)?;

    let suffix = type_suffix::<T>();
    let kernel_name = format!("k_cond_set_{suffix}");
    let func = get_kernel(ctx, &kernel_name)?;

    let handle_u64: u64 = handle;
    let cmp_u32: u32 = match cmp {
        CondCmp::Positive => 0,
        CondCmp::Zero => 1,
        CondCmp::LessThan => 2,
    };

    // SAFETY: all pointers are valid device pointers; scalar_storage contains
    unsafe {
        result::launch_kernel(
            func,
            (1, 1, 1),
            (1, 1, 1),
            0,
            ctx.stream.cu_stream(),
            &mut [
                &handle_u64 as *const u64 as *mut c_void,
                &scalar_storage.buf.ptr as *const CUdeviceptr as *mut c_void,
                &cmp_u32 as *const u32 as *mut c_void,
                &threshold as *const f32 as *mut c_void,
            ],
        )
        .map_err(CudaError::Driver)?;
    }
    Ok(())
}

pub fn cuda_if_positive<T: Scalar, F>(
    condition_storage: &CudaStorage<T>,
    body: F,
) -> CudaResult<ConditionalGraph>
where
    F: Fn(cudarc::driver::sys::CUgraph) -> CudaResult<()>,
{
    let cond_graph = ConditionalGraph::new_if(body)?;

    cuda_conditional_set_from_scalar(
        cond_graph.device_handle(),
        condition_storage,
        CondCmp::Positive,
        0.0_f32,
    )?;

    cond_graph.launch()?;

    Ok(cond_graph)
}

// ── PyGraphTrainingGraph: warmup → capture → replay wrapper ─────────────────

/// High-level training-loop wrapper that warms up, captures a CUDA graph, then replays it.
pub struct PyGraphTrainingGraph {
    graph: Option<PyGraph>,
    warmup_iters: usize,
    iter_count: usize,
}

impl PyGraphTrainingGraph {
    /// Create with default warmup (5 iterations).
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: None,
            warmup_iters: 5,
            iter_count: 0,
        }
    }

    /// Create with custom warmup count.
    #[must_use]
    pub fn with_warmup(warmup_iters: usize) -> Self {
        Self {
            graph: None,
            warmup_iters,
            iter_count: 0,
        }
    }

    /// Mutable access to the captured PyGraph (None before capture completes).
    pub fn graph(&mut self) -> Option<&mut PyGraph> {
        self.graph.as_mut()
    }

    /// Execute one training step (warmup / capture / replay).
    pub fn step<F: FnMut()>(&mut self, f: &mut F) -> CudaResult<()> {
        self.iter_count += 1;

        if self.iter_count <= self.warmup_iters {
            f();
            cuda_synchronize();
            Ok(())
        } else if self.graph.is_none() {
            let graph = PyGraph::capture(|| f())?;
            self.graph = Some(graph);
            Ok(())
        } else {
            self.graph.as_ref().ok_or(CudaError::NullPtr)?.launch()
        }
    }

    /// Reset -- force re-capture on next step.
    pub fn reset(&mut self) {
        self.graph = None;
        self.iter_count = 0;
    }

    /// True once the graph has been captured.
    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }
}

impl Default for PyGraphTrainingGraph {
    fn default() -> Self {
        Self::new()
    }
}
