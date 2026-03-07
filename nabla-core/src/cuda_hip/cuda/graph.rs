// ── Graph Analysis ───────────────────────────────────────────────────────────

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUgraph, CUgraphNode, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph as CudarcCudaGraph, result};
use crate::gpu_common::{lock_or_recover, round_size, type_suffix};
use crate::kernels_cu;
use crate::scalar::Scalar;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelClass {
    UnaryElementwise,
    BinaryElementwise,
    Reduction,
    Matmul,
    Norm,
    Conv,
    Fused,
    Other,
}

#[derive(Debug)]
pub struct AnalyzedNode {
    pub node: CUgraphNode,
    pub func: CUfunction,
    pub kernel_name: Option<String>,
    pub class: KernelClass,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub n_args: usize,
    pub arg_values: Vec<u64>,
    pub deps: Vec<usize>,
    pub dependents: Vec<usize>,
}

#[derive(Debug)]
pub struct FusionCandidate {
    pub node_indices: Vec<usize>,
    pub ops: Vec<String>,
    pub estimated_speedup: f32,
}

#[derive(Debug)]
pub struct EpilogueCandidate {
    pub matmul_idx: usize,
    pub activation_idx: usize,
    pub activation: String,
}

#[derive(Debug)]
pub struct TransposeElimCandidate {
    pub transpose_idx: usize,
    pub matmul_idx: usize,
    pub variant: &'static str,
}

pub struct OptimizationReport {
    pub total_nodes: usize,
    pub kernel_nodes: usize,
    pub elementwise_count: usize,
    pub fusion_candidates: Vec<FusionCandidate>,
    pub epilogue_candidates: Vec<EpilogueCandidate>,
    pub transpose_elim_candidates: Vec<TransposeElimCandidate>,
}

impl OptimizationReport {
    pub fn estimated_node_reduction(&self) -> usize {
        let fusion: usize = self
            .fusion_candidates
            .iter()
            .map(|c| c.node_indices.len() - 1)
            .sum();
        fusion + self.epilogue_candidates.len() + self.transpose_elim_candidates.len()
    }
}

impl std::fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== NablaGraph Optimization Report ===")?;
        writeln!(f, "Total graph nodes: {}, kernel nodes: {}", self.total_nodes, self.kernel_nodes)?;
        writeln!(f, "Elementwise ops: {}", self.elementwise_count)?;
        if !self.fusion_candidates.is_empty() {
            let saved: usize = self.fusion_candidates.iter().map(|c| c.node_indices.len() - 1).sum();
            writeln!(f, "Fusion candidates: {} (saves {saved} launches)", self.fusion_candidates.len())?;
            for (i, c) in self.fusion_candidates.iter().enumerate() {
                writeln!(f, "  [{i}] {} ops: {} (est. {:.1}x)", c.node_indices.len(), c.ops.join(" -> "), c.estimated_speedup)?;
            }
        }
        if !self.epilogue_candidates.is_empty() {
            writeln!(f, "Epilogue candidates: {} (matmul+activation -> cublasLt)", self.epilogue_candidates.len())?;
            for c in &self.epilogue_candidates {
                writeln!(f, "  matmul[{}] + {}[{}]", c.matmul_idx, c.activation, c.activation_idx)?;
            }
        }
        if !self.transpose_elim_candidates.is_empty() {
            let v = self.transpose_elim_candidates.first().map_or("tn", |c| c.variant);
            writeln!(f, "Transpose elimination: {} (transpose+matmul -> matmul_{v})", self.transpose_elim_candidates.len())?;
        }
        let total = self.estimated_node_reduction();
        if total > 0 {
            writeln!(f, "Estimated reduction: {} -> {} kernel launches ({total} eliminated)", self.kernel_nodes, self.kernel_nodes.saturating_sub(total))?;
        } else {
            writeln!(f, "No optimization opportunities detected.")?;
        }
        Ok(())
    }
}

pub(super) fn build_reverse_kernel_map(ctx: &CudaCtx) -> HashMap<usize, String> {
    let map = ctx.kernels.read().unwrap_or_else(std::sync::PoisonError::into_inner);
    map.iter().map(|(name, entry)| (entry.func as usize, name.clone())).collect()
}

pub(super) fn classify_kernel(name: &str) -> KernelClass {
    const UNARY: &[&str] = &[
        "neg", "recip", "exp", "ln", "log1p", "log2", "log10", "sin", "cos", "tan", "tanh",
        "sqrt", "abs", "ceil", "floor", "round", "erf", "asin", "acos", "atan", "sinh", "cosh",
        "asinh", "acosh", "atanh", "sigmoid", "silu", "mish", "hardswish", "sign",
    ];
    const BINARY: &[&str] = &[
        "add", "sub", "emul", "ediv", "powf", "atan2", "scale", "axpy", "fill",
    ];
    const ACT_BWD: &[&str] = &[
        "relu_bwd", "leaky_relu_bwd", "elu_bwd", "sigmoid_bwd", "tanh_bwd", "silu_bwd", "gelu_bwd",
    ];

    let base = name.strip_prefix("k_").unwrap_or(name);
    let op = base.rsplit_once('_').map_or(base, |(op, _)| op);

    if UNARY.contains(&op) { return KernelClass::UnaryElementwise; }
    if BINARY.contains(&op) || ACT_BWD.contains(&op) { return KernelClass::BinaryElementwise; }
    if op.contains("pool") { return KernelClass::Other; }

    if ["sum", "max", "min", "prod"].iter().any(|p| op.starts_with(p))
        || matches!(op, "softmax" | "log_softmax")
    { return KernelClass::Reduction; }
    if op.contains("matmul") || op.contains("gemm") { return KernelClass::Matmul; }
    if op.contains("norm") || op.contains("batch_norm") { return KernelClass::Norm; }
    if ["conv", "im2col", "im1col", "im3col"].iter().any(|p| op.contains(p)) { return KernelClass::Conv; }
    if op.starts_with("fused_") || op.starts_with("mega_") || op.starts_with("fuse_reduce_") { return KernelClass::Fused; }
    KernelClass::Other
}

pub(super) fn node_dependencies(node: CUgraphNode) -> CudaResult<Vec<CUgraphNode>> {
    let count = unsafe {
        let mut n: usize = 0;
        // SAFETY: node is valid; null deps ptr queries count only.
        cudarc::driver::sys::cuGraphNodeGetDependencies(node, std::ptr::null_mut(), &mut n)
            .result()
            .map_err(CudaError::Driver)?;
        n
    };
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut deps = vec![std::ptr::null_mut(); count];
    unsafe {
        let mut n = count;
        // SAFETY: deps is sized to count from the count query above.
        cudarc::driver::sys::cuGraphNodeGetDependencies(node, deps.as_mut_ptr(), &mut n)
            .result()
            .map_err(CudaError::Driver)?;
    }
    Ok(deps)
}

fn count_kernel_params(p: &cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS) -> usize {
    if p.kernelParams.is_null() { return 0; }
    let mut count = 0usize;
    // SAFETY: kernelParams is a null-terminated *mut *mut c_void array.
    unsafe { while !(*p.kernelParams.add(count)).is_null() { count += 1; } }
    count
}

// SAFETY: caller must ensure kernelParams has at least `n` valid entries.
unsafe fn read_kernel_arg_values(p: &cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS, n: usize) -> Vec<u64> {
    let mut v = vec![0u64; n];
    for i in 0..n {
        // SAFETY: kernelParams[i] points to the snapshotted arg value.
        unsafe { v[i] = ((*p.kernelParams.add(i)) as *const u64).read_unaligned(); }
    }
    v
}

/// Fetch all graph node handles from a CUDA graph via count-then-fetch.
pub(super) fn fetch_graph_nodes(cu_graph: CUgraph) -> CudaResult<Vec<CUgraphNode>> {
    let count = unsafe {
        let mut n: usize = 0;
        // SAFETY: cu_graph is valid; null nodes ptr queries count.
        cudarc::driver::sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n)
            .result()
            .map_err(CudaError::Driver)?;
        n
    };
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    unsafe {
        let mut n = count;
        // SAFETY: nodes is sized to count from the count query above.
        cudarc::driver::sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut n)
            .result()
            .map_err(CudaError::Driver)?;
    }
    Ok(nodes)
}

/// Extract kernel node params if `node` is a kernel node, else return None.
pub(super) fn extract_kernel_params(
    node: CUgraphNode,
) -> CudaResult<Option<cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS>> {
    use cudarc::driver::sys::{CUDA_KERNEL_NODE_PARAMS, CUgraphNodeType};
    let node_type = unsafe {
        let mut t: CUgraphNodeType = std::mem::zeroed();
        // SAFETY: node is a valid graph node handle.
        cudarc::driver::sys::cuGraphNodeGetType(node, &mut t)
            .result()
            .map_err(CudaError::Driver)?;
        t
    };
    if node_type != CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
        return Ok(None);
    }
    let p = unsafe {
        let mut p: CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
        // SAFETY: node is a kernel node (type checked above).
        cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(node, &mut p)
            .result()
            .map_err(CudaError::Driver)?;
        p
    };
    Ok(Some(p))
}

pub fn analyze_graph(cu_graph: CUgraph) -> CudaResult<(Vec<AnalyzedNode>, OptimizationReport)> {
    let ctx = get_ctx();
    let reverse_map = build_reverse_kernel_map(ctx);
    let all_nodes = fetch_graph_nodes(cu_graph)?;
    let total_nodes = all_nodes.len();

    let handle_to_idx: HashMap<usize, usize> = all_nodes.iter().enumerate().map(|(i, &n)| (n as usize, i)).collect();

    let mut analyzed: Vec<AnalyzedNode> = Vec::with_capacity(total_nodes);
    let mut kernel_idx_map: HashMap<usize, usize> = HashMap::with_capacity(total_nodes);

    for (all_idx, &node) in all_nodes.iter().enumerate() {
        let Some(p) = extract_kernel_params(node)? else { continue; };
        let kernel_name = reverse_map.get(&(p.func as usize)).cloned();
        let class = kernel_name.as_deref().map_or(KernelClass::Other, classify_kernel);
        let n_args = count_kernel_params(&p);
        // SAFETY: kernelParams validated by count_kernel_params.
        let arg_values = unsafe { read_kernel_arg_values(&p, n_args) };
        kernel_idx_map.insert(all_idx, analyzed.len());
        analyzed.push(AnalyzedNode {
            node, func: p.func, kernel_name, class, n_args, arg_values,
            grid: (p.gridDimX, p.gridDimY, p.gridDimZ), block: (p.blockDimX, p.blockDimY, p.blockDimZ),
            deps: Vec::new(), dependents: Vec::new(),
        });
    }

    for (&all_idx, &analyzed_idx) in &kernel_idx_map {
        analyzed[analyzed_idx].deps = node_dependencies(all_nodes[all_idx])?
            .iter()
            .filter_map(|&dep| handle_to_idx.get(&(dep as usize)))
            .filter_map(|&aidx| kernel_idx_map.get(&aidx).copied())
            .collect();
    }
    for i in 0..analyzed.len() {
        let deps = analyzed[i].deps.clone();
        for dep_idx in deps {
            analyzed[dep_idx].dependents.push(i);
        }
    }

    let elementwise_count = analyzed.iter().filter(|n| is_elementwise(n.class)).count();
    let fusion_candidates = detect_fusion_chains(&analyzed);
    let epilogue_candidates = detect_epilogue_patterns(&analyzed);
    let transpose_elim_candidates = detect_transpose_elim(&analyzed);

    let report = OptimizationReport { total_nodes, kernel_nodes: analyzed.len(), elementwise_count, fusion_candidates, epilogue_candidates, transpose_elim_candidates };
    Ok((analyzed, report))
}

fn is_elementwise(class: KernelClass) -> bool {
    matches!(class, KernelClass::UnaryElementwise | KernelClass::BinaryElementwise)
}

fn detect_fusion_chains(nodes: &[AnalyzedNode]) -> Vec<FusionCandidate> {
    let mut visited = vec![false; nodes.len()];
    let mut candidates = Vec::new();

    for start in 0..nodes.len() {
        if visited[start] || !is_elementwise(nodes[start].class) { continue; }
        let mut chain = vec![start];
        let mut chain_set: HashSet<usize> = HashSet::new();
        chain_set.insert(start);
        visited[start] = true;
        let mut current = start;

        loop {
            if nodes[current].dependents.len() != 1 { break; }
            let next = nodes[current].dependents[0];
            if visited[next] || !is_elementwise(nodes[next].class) { break; }
            let non_chain_deps = nodes[next].deps.iter().filter(|&&d| !chain_set.contains(&d)).count();
            if nodes[next].class == KernelClass::BinaryElementwise && non_chain_deps > 1 { break; }
            if nodes[next].class == KernelClass::UnaryElementwise && non_chain_deps > 0 { break; }
            chain.push(next);
            chain_set.insert(next);
            visited[next] = true;
            current = next;
        }

        if chain.len() >= 2 {
            let ops: Vec<String> = chain
                .iter()
                .filter_map(|&i| nodes[i].kernel_name.as_ref())
                .map(|n| super::graph_compile::extract_op(n).to_string())
                .collect();
            let speedup = 1.0 + 0.5 * (chain.len() as f32 - 1.0).min(4.0);
            candidates.push(FusionCandidate {
                node_indices: chain,
                ops,
                estimated_speedup: speedup,
            });
        }
    }
    candidates
}

fn detect_epilogue_patterns(nodes: &[AnalyzedNode]) -> Vec<EpilogueCandidate> {
    let mut candidates = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if node.class != KernelClass::Matmul || node.dependents.len() != 1 { continue; }
        let next = node.dependents[0];
        let op = super::graph_compile::extract_op(nodes[next].kernel_name.as_deref().unwrap_or(""));
        let activation = match op {
            name if name.contains("relu") && !name.contains("leaky") => "relu",
            name if name.contains("gelu") => "gelu",
            _ => continue,
        };
        if nodes[next].deps.len() == 1 && nodes[next].deps[0] == i {
            candidates.push(EpilogueCandidate { matmul_idx: i, activation_idx: next, activation: activation.to_string() });
        }
    }
    candidates
}

fn detect_transpose_elim(nodes: &[AnalyzedNode]) -> Vec<TransposeElimCandidate> {
    let mut candidates = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let name = node.kernel_name.as_deref().unwrap_or("");
        if !name.contains("transpose") || node.dependents.len() != 1 { continue; }
        let next = node.dependents[0];
        if nodes[next].class != KernelClass::Matmul { continue; }
        let variant = if nodes[next].deps.first() == Some(&i) { "tn" } else { "nt" };
        candidates.push(TransposeElimCandidate { transpose_idx: i, matmul_idx: next, variant });
    }
    candidates
}

pub struct AllocationProfile {
    pub buffer_sizes: Vec<usize>,
    pub peak_bytes: usize,
}

pub(super) fn dtype_bytes(kernel_name: &str) -> usize {
    if kernel_name.ends_with("_f64") { 8 }
    else if kernel_name.ends_with("_f16") || kernel_name.ends_with("_bf16") { 2 }
    else if kernel_name.contains("fp8") || kernel_name.contains("fp4") { 1 }
    else { 4 }
}

pub fn extract_allocation_profile(analyzed: &[AnalyzedNode]) -> AllocationProfile {
    let mut ptr_max_size: HashMap<u64, usize> = HashMap::with_capacity(analyzed.len() * 2);
    for node in analyzed {
        let (ptrs, n_idx) = match node.class {
            KernelClass::UnaryElementwise if node.arg_values.len() >= 3 => {
                (&node.arg_values[..2], 2)
            }
            KernelClass::BinaryElementwise if node.arg_values.len() >= 4 => {
                (&node.arg_values[..3], 3)
            }
            _ => continue,
        };
        let n = (node.arg_values[n_idx] & 0xFFFF_FFFF) as usize;
        let dbytes = node.kernel_name.as_deref().map_or(4, dtype_bytes);
        let buf_size = round_size(n * dbytes);
        for &ptr in ptrs {
            let entry = ptr_max_size.entry(ptr).or_insert(0);
            *entry = (*entry).max(buf_size);
        }
    }
    let mut buffer_sizes: Vec<usize> = ptr_max_size.into_values().collect();
    buffer_sizes.sort_unstable();
    let peak_bytes = buffer_sizes.iter().sum();
    AllocationProfile { buffer_sizes, peak_bytes }
}

// ── Graph Runtime ────────────────────────────────────────────────────────────

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
    let Ok(status) = (unsafe { result::stream::is_capturing(stream) }) else { return };
    if status == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE { return; }
    if let Ok(graph) = unsafe { result::stream::end_capture(stream) } {
        if !graph.is_null() { unsafe { cudarc::driver::sys::cuGraphDestroy(graph) }; }
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
        get_ctx().stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        Ok(())
    }

    fn end_capture() -> CudaResult<Self> {
        let ctx = get_ctx();
        // SAFETY: transmuting 0u32 to flags enum (no flags set).
        let flags: cudarc::driver::sys::CUgraphInstantiate_flags = unsafe { std::mem::transmute(0u32) };
        let graph = ctx.stream.end_capture(flags).map_err(|e| { clear_stream_capture(ctx.stream.cu_stream()); CudaError::Driver(e) })?.ok_or(CudaError::NullPtr)?;
        Ok(Self { inner: graph })
    }

    pub fn launch(&self) -> CudaResult<()> { self.inner.launch()?; Ok(()) }
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
        let cache = lock_or_recover(&ctx.graphs);
        if let Some(g) = cache.get(name) {
            return Ok(Arc::clone(g));
        }
    }
    let graph = Arc::new(cuda_graph_capture(f)?);
    let mut cache = lock_or_recover(&ctx.graphs);
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
    let _d2h_guard = ctx.d2h_mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    ctx.stream.context().bind_to_thread().map_err(CudaError::Driver)?;
    // SAFETY: event lifecycle is contained within this function.
    unsafe {
        let event = result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).map_err(CudaError::Driver)?;
        result::event::record(event, ctx.stream.cu_stream()).map_err(CudaError::Driver)?;
        result::stream::wait_event(ctx.copy_stream.cu_stream(), event, cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT).map_err(CudaError::Driver)?;
        result::memcpy_dtoh_async(&mut host, storage.buf.ptr, ctx.copy_stream.cu_stream()).map_err(CudaError::Driver)?;
        ctx.copy_stream.synchronize().map_err(CudaError::Driver)?;
        result::event::destroy(event).map_err(CudaError::Driver)?;
    }
    Ok(host)
}

pub fn cuda_copy_from_host<T: Scalar>(storage: &CudaStorage<T>, data: &[T]) {
    assert_eq!(storage.n(), data.len(), "cuda_copy_from_host: size mismatch (buffer={}, data={})", storage.n(), data.len());
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
    pub(crate) arg_bytes: Vec<u64>,
    _arg_ptrs: Vec<*mut c_void>,
}

// SAFETY: arg_ptrs are raw pointers into arg_bytes which we own exclusively.
unsafe impl Send for KernelNodeState {}

pub struct PyGraph {
    pub(super) cu_graph: CUgraph,
    pub(super) cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    cu_stream: cudarc::driver::sys::CUstream,
    pub kernel_nodes: Vec<KernelNodeState>,
}

// SAFETY: PyGraph is used from a single training thread; raw handles are valid
unsafe impl Send for PyGraph {}

impl Drop for PyGraph {
    fn drop(&mut self) {
        // SAFETY: both handles are non-null (set in PyGraph::capture) and we
        unsafe { cudarc::driver::sys::cuGraphExecDestroy(self.cu_graph_exec); cudarc::driver::sys::cuGraphDestroy(self.cu_graph); }
    }
}

impl PyGraph {
    pub fn capture<F: FnOnce()>(f: F) -> CudaResult<Self> {
        let ctx = get_ctx();
        let cu_stream = ctx.stream.cu_stream();

        ctx.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let _guard = CaptureGuard::new();

        f();
        drop(_guard);

        let cu_graph = unsafe { result::stream::end_capture(cu_stream) }
            .map_err(|e| { clear_stream_capture(cu_stream); CudaError::Driver(e) })?;
        if cu_graph.is_null() { return Err(CudaError::NullPtr); }

        let cu_graph_exec = unsafe {
            // SAFETY: cu_graph is a valid non-null graph from cuStreamEndCapture.
            let mut exec = std::mem::MaybeUninit::uninit();
            if let Err(e) = cudarc::driver::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), cu_graph, 0).result() {
                cudarc::driver::sys::cuGraphDestroy(cu_graph);
                return Err(CudaError::Driver(e));
            }
            exec.assume_init()
        };
        if cu_graph_exec.is_null() { unsafe { cudarc::driver::sys::cuGraphDestroy(cu_graph) }; return Err(CudaError::NullPtr); }

        Ok(Self { cu_graph, cu_graph_exec, cu_stream, kernel_nodes: Self::collect_kernel_nodes(cu_graph)? })
    }

    pub(super) fn collect_kernel_nodes(cu_graph: CUgraph) -> CudaResult<Vec<KernelNodeState>> {
        let mut kernel_nodes = Vec::new();
        for node in fetch_graph_nodes(cu_graph)? {
            let Some(raw_params) = extract_kernel_params(node)? else { continue; };
            let n_args = count_kernel_params(&raw_params);
            // SAFETY: kernelParams validated by count_kernel_params above.
            let mut arg_bytes = unsafe { read_kernel_arg_values(&raw_params, n_args) };
            let mut arg_ptrs: Vec<*mut c_void> = arg_bytes.iter_mut().map(|v| (v as *mut u64).cast::<c_void>()).collect();
            arg_ptrs.push(std::ptr::null_mut());
            let mut params = raw_params;
            params.kernelParams = arg_ptrs.as_mut_ptr();
            params.extra = std::ptr::null_mut();
            kernel_nodes.push(KernelNodeState { node, params, arg_bytes, _arg_ptrs: arg_ptrs });
        }
        Ok(kernel_nodes)
    }

    pub fn launch(&self) -> CudaResult<()> {
        // SAFETY: cu_graph_exec and cu_stream are valid for the lifetime of self.
        unsafe { cudarc::driver::sys::cuGraphLaunch(self.cu_graph_exec, self.cu_stream).result().map_err(CudaError::Driver) }
    }

    pub fn update_node_param_ptr(&mut self, node_idx: usize, param_idx: usize, new_ptr: CUdeviceptr) -> CudaResult<()> {
        let state = &mut self.kernel_nodes[node_idx];
        state.arg_bytes[param_idx] = new_ptr;
        // SAFETY: cu_graph_exec is valid; state.node is a kernel node in this graph.
        unsafe { cudarc::driver::sys::cuGraphExecKernelNodeSetParams_v2(self.cu_graph_exec, state.node, &state.params).result().map_err(CudaError::Driver) }
    }

    #[must_use] pub fn kernel_node_count(&self) -> usize { self.kernel_nodes.len() }
    #[must_use] pub fn arg_count(&self, node_idx: usize) -> usize { self.kernel_nodes[node_idx].arg_bytes.len() }
    #[must_use] pub fn get_param(&self, node_idx: usize, param_idx: usize) -> CUdeviceptr { self.kernel_nodes[node_idx].arg_bytes[param_idx] }
}

pub struct PyGraphTrainingGraph {
    graph: Option<PyGraph>,
    warmup_iters: usize,
    iter_count: usize,
}

impl PyGraphTrainingGraph {
    #[must_use]
    pub fn new() -> Self { Self { graph: None, warmup_iters: 5, iter_count: 0 } }

    #[must_use]
    pub fn with_warmup(warmup_iters: usize) -> Self { Self { graph: None, warmup_iters, iter_count: 0 } }

    pub fn graph(&mut self) -> Option<&mut PyGraph> { self.graph.as_mut() }

    pub fn step<F: FnMut()>(&mut self, f: &mut F) -> CudaResult<()> {
        self.iter_count += 1;
        if self.iter_count <= self.warmup_iters { f(); cuda_synchronize(); Ok(()) }
        else if self.graph.is_none() { self.graph = Some(PyGraph::capture(|| f())?); Ok(()) }
        else { self.graph.as_ref().ok_or(CudaError::NullPtr)?.launch() }
    }

    pub fn reset(&mut self) { self.graph = None; self.iter_count = 0; }
    #[must_use] pub fn is_captured(&self) -> bool { self.graph.is_some() }
}

impl Default for PyGraphTrainingGraph {
    fn default() -> Self {
        Self::new()
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
    unsafe fn new(graph: CUgraph, cu_ctx: cudarc::driver::sys::CUcontext, default_value: u32) -> CudaResult<Self> {
        use cudarc::driver::sys as cds;
        let mut raw: cds::CUgraphConditionalHandle = 0;
        // SAFETY: all pointers valid; pHandle_out points to a local stack slot.
        let r = unsafe { cds::cuGraphConditionalHandleCreate(&mut raw, graph, cu_ctx, default_value, cds::CU_GRAPH_COND_ASSIGN_DEFAULT) };
        if r != cds::CUresult::CUDA_SUCCESS { return Err(CudaError::Driver(cudarc::driver::DriverError(r))); }
        Ok(Self { raw })
    }
    fn raw(&self) -> cudarc::driver::sys::CUgraphConditionalHandle { self.raw }
}

pub struct ConditionalGraph {
    cu_graph: CUgraph,
    cu_exec: cudarc::driver::sys::CUgraphExec,
    handle: ConditionalHandle,
}

// SAFETY: CUgraph / CUgraphExec are process-global opaque handles. We never
unsafe impl Send for ConditionalGraph {}
unsafe impl Sync for ConditionalGraph {}

impl Drop for ConditionalGraph {
    fn drop(&mut self) {
        // SAFETY: each pointer is destroyed exactly once.
        if !self.cu_exec.is_null() { unsafe { cudarc::driver::sys::cuGraphExecDestroy(self.cu_exec) }; }
        if !self.cu_graph.is_null() { unsafe { cudarc::driver::sys::cuGraphDestroy(self.cu_graph) }; }
    }
}

impl ConditionalGraph {
    pub fn new<F: Fn(CUgraph) -> CudaResult<()>>(kind: ConditionalKind, default_value: u32, bodies: &[F]) -> CudaResult<Self> {
        use cudarc::driver::sys as cds;
        let cu_ctx = get_ctx().stream.context().cu_ctx();
        let mut cu_graph: CUgraph = std::ptr::null_mut();
        // SAFETY: cuGraphCreate writes a valid handle into cu_graph on success.
        let r = unsafe { cds::cuGraphCreate(&mut cu_graph, 0) };
        if r != cds::CUresult::CUDA_SUCCESS { return Err(CudaError::Driver(cudarc::driver::DriverError(r))); }

        macro_rules! bail {
            ($e:expr) => {{ unsafe { cds::cuGraphDestroy(cu_graph) }; return Err($e); }};
        }

        // SAFETY: cu_graph is valid and will outlive the handle.
        let handle = match unsafe { ConditionalHandle::new(cu_graph, cu_ctx, default_value) } { Ok(h) => h, Err(e) => bail!(e) };
        let (cond_type, size) = match kind {
            ConditionalKind::If => (cds::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF, bodies.len() as u32),
            ConditionalKind::While => (cds::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE, 1u32),
            ConditionalKind::Switch { .. } => bail!(CudaError::KernelNotFound("CUDA conditional switch unsupported".into())),
        };

        let mut body_graphs: Vec<CUgraph> = vec![std::ptr::null_mut(); size as usize];
        // SAFETY: CUgraphNodeParams_st is a C struct; zeroing and setting the conditional union arm.
        let mut node_params: cds::CUgraphNodeParams = unsafe { std::mem::zeroed() };
        node_params.type_ = cds::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        node_params.__bindgen_anon_1.conditional = cds::CUDA_CONDITIONAL_NODE_PARAMS { handle: handle.raw(), type_: cond_type, size, phGraph_out: body_graphs.as_mut_ptr(), ctx: cu_ctx };

        let mut cond_node: CUgraphNode = std::ptr::null_mut();
        // SAFETY: cu_graph, node_params, body_graphs are all valid.
        let r = unsafe { cds::cuGraphAddNode(&mut cond_node, cu_graph, std::ptr::null(), 0, &mut node_params) };
        if r != cds::CUresult::CUDA_SUCCESS { bail!(CudaError::Driver(cudarc::driver::DriverError(r))); }

        for (i, body_fn) in bodies.iter().enumerate() {
            let body = body_graphs[i];
            if body.is_null() { bail!(CudaError::NullPtr); }
            if let Err(e) = body_fn(body) { bail!(e); }
        }

        let mut cu_exec: cds::CUgraphExec = std::ptr::null_mut();
        // SAFETY: cu_graph is fully constructed; cu_exec receives the exec handle.
        let r = unsafe { cds::cuGraphInstantiateWithFlags(&mut cu_exec, cu_graph, 0u64) };
        if r != cds::CUresult::CUDA_SUCCESS { bail!(CudaError::Driver(cudarc::driver::DriverError(r))); }

        Ok(Self { cu_graph, cu_exec, handle })
    }

    pub fn new_if<F: Fn(CUgraph) -> CudaResult<()>>(body: F) -> CudaResult<Self> { Self::new(ConditionalKind::If, 0, &[body]) }
    pub fn new_if_else<F: Fn(CUgraph) -> CudaResult<()>>(t: F, f: F) -> CudaResult<Self> { Self::new(ConditionalKind::If, 0, &[t, f]) }
    pub fn new_while<F: Fn(CUgraph) -> CudaResult<()>>(body: F) -> CudaResult<Self> { Self::new(ConditionalKind::While, 0, &[body]) }

    pub fn launch(&self) -> CudaResult<()> {
        // SAFETY: cu_exec and the stream are valid for this graph's lifetime.
        let r = unsafe { cudarc::driver::sys::cuGraphLaunch(self.cu_exec, get_ctx().stream.cu_stream()) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(CudaError::Driver(cudarc::driver::DriverError(r))); }
        Ok(())
    }

    pub fn device_handle(&self) -> cudarc::driver::sys::CUgraphConditionalHandle { self.handle.raw() }
}

const COND_SET_KERNEL_NAMES: &[&str] = &["k_cond_set_f32", "k_cond_set_f16", "k_cond_set_f64"];

fn compile_cond_set_kernels(ctx: &CudaCtx) -> CudaResult<()> {
    {
        let map = ctx.kernels.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.contains_key("k_cond_set_f32") { return Ok(()); }
    }
    for &name in COND_SET_KERNEL_NAMES {
        super::graph_compile::compile_and_cache_kernel(ctx, name, kernels_cu::COND_SET_KERNELS)?;
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
    handle: cudarc::driver::sys::CUgraphConditionalHandle, scalar_storage: &CudaStorage<T>,
    cmp: CondCmp, threshold: f32,
) -> CudaResult<()> {
    let ctx = get_ctx();
    compile_cond_set_kernels(ctx)?;
    let func = get_kernel(ctx, &format!("k_cond_set_{}", type_suffix::<T>()))?;
    let handle_u64: u64 = handle;
    let cmp_u32: u32 = match cmp { CondCmp::Positive => 0, CondCmp::Zero => 1, CondCmp::LessThan => 2 };
    // SAFETY: all pointers are valid device pointers; scalar_storage is valid for the kernel call.
    unsafe {
        result::launch_kernel(func, (1, 1, 1), (1, 1, 1), 0, ctx.stream.cu_stream(), &mut [
            &handle_u64 as *const u64 as *mut c_void, &scalar_storage.buf.ptr as *const CUdeviceptr as *mut c_void,
            &cmp_u32 as *const u32 as *mut c_void, &threshold as *const f32 as *mut c_void,
        ]).map_err(CudaError::Driver)?;
    }
    Ok(())
}

pub fn cuda_if_positive<T: Scalar, F: Fn(CUgraph) -> CudaResult<()>>(
    condition_storage: &CudaStorage<T>,
    body: F,
) -> CudaResult<ConditionalGraph> {
    let cond_graph = ConditionalGraph::new_if(body)?;
    cuda_conditional_set_from_scalar(cond_graph.device_handle(), condition_storage, CondCmp::Positive, 0.0_f32)?;
    cond_graph.launch()?;
    Ok(cond_graph)
}
