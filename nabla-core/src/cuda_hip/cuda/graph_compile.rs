use std::collections::{HashMap, HashSet};
use std::ffi::{CString, c_void};
use std::fmt::Write as _;
use std::path::PathBuf;

use cudarc::driver::sys::{CUfunction, CUgraph, CUgraphNode};
use cudarc::nvrtc;

use super::core::{CudaCtx, CudaError, CudaResult, KernelEntry};
use super::get_ctx;
use super::graph::{AnalyzedNode, FusionCandidate, KernelClass, OptimizationReport, analyze_graph};

fn elementwise_out_idx(class: KernelClass) -> usize {
    if class == KernelClass::UnaryElementwise {
        1
    } else {
        2
    }
}

struct ChainDataFlow {
    external_inputs: Vec<u64>,
    final_output: u64,
    n_val: u64,
}

fn op_to_cuda_f32(op: &str, x: &str) -> Option<String> {
    let expr = match op {
        "neg" => format!("-({x})"),
        "exp" => format!("expf({x})"),
        "sin" => format!("sinf({x})"),
        "cos" => format!("cosf({x})"),
        "tan" => format!("tanf({x})"),
        "tanh" => format!("tanhf({x})"),
        "sqrt" => format!("sqrtf({x})"),
        "abs" => format!("fabsf({x})"),
        "sigmoid" => format!("(1.0f / (1.0f + expf(-({x}))))"),
        "silu" => format!("(({x}) / (1.0f + expf(-({x}))))"),
        "ln" => format!("logf({x})"),
        "recip" => format!("(1.0f / ({x}))"),
        "log2" => format!("log2f({x})"),
        "log10" => format!("log10f({x})"),
        "erf" => format!("erff({x})"),
        "ceil" => format!("ceilf({x})"),
        "floor" => format!("floorf({x})"),
        "round" => format!("roundf({x})"),
        "sign" => format!("(((({x}) > 0.0f) ? 1.0f : 0.0f) - ((({x}) < 0.0f) ? 1.0f : 0.0f))"),
        "asin" => format!("asinf({x})"),
        "acos" => format!("acosf({x})"),
        "atan" => format!("atanf({x})"),
        "sinh" => format!("sinhf({x})"),
        "cosh" => format!("coshf({x})"),
        "asinh" => format!("asinhf({x})"),
        "acosh" => format!("acoshf({x})"),
        "atanh" => format!("atanhf({x})"),
        "log1p" => format!("log1pf({x})"),
        "mish" => format!("(({x}) * tanhf(logf(1.0f + expf({x}))))"),
        "hardswish" => format!(
            "((({x}) <= -3.0f) ? 0.0f : ((({x}) >= 3.0f) ? ({x}) : ({x}) * (({x}) + 3.0f) / 6.0f))"
        ),
        _ => return None,
    };
    Some(expr)
}

fn op_to_cuda_binary_f32(op: &str, a: &str, b: &str) -> Option<String> {
    let expr = match op {
        "add" => format!("(({a}) + ({b}))"),
        "sub" => format!("(({a}) - ({b}))"),
        "emul" => format!("(({a}) * ({b}))"),
        "ediv" => format!("(({a}) / ({b}))"),
        _ => return None,
    };
    Some(expr)
}

/// Extract the base op name from a kernel name like "k_neg_f32" -> "neg".
pub(super) fn extract_op(name: &str) -> &str {
    let base = name.strip_prefix("k_").unwrap_or(name);
    base.rsplit_once('_').map_or(base, |(op, _)| op)
}

fn trace_chain_dataflow(nodes: &[AnalyzedNode], chain: &[usize]) -> ChainDataFlow {
    let mut chain_outputs: HashSet<u64> = HashSet::new();
    for &idx in chain {
        let node = &nodes[idx];
        let out_idx = elementwise_out_idx(node.class);
        if out_idx < node.arg_values.len() {
            chain_outputs.insert(node.arg_values[out_idx]);
        }
    }

    let mut external_inputs: Vec<u64> = Vec::new();
    for &idx in chain {
        let node = &nodes[idx];
        let n_inputs = elementwise_out_idx(node.class);
        for arg_i in 0..n_inputs {
            if arg_i < node.arg_values.len() {
                let ptr = node.arg_values[arg_i];
                if !chain_outputs.contains(&ptr) && !external_inputs.contains(&ptr) {
                    external_inputs.push(ptr);
                }
            }
        }
    }

    let last = &nodes[chain[chain.len() - 1]];
    let last_out_idx = elementwise_out_idx(last.class);
    let final_output = last.arg_values.get(last_out_idx).copied().unwrap_or(0);
    let n_val = nodes[chain[0]].arg_values.last().copied().unwrap_or(0) & 0xFFFF_FFFF;
    ChainDataFlow {
        external_inputs,
        final_output,
        n_val,
    }
}

fn generate_fused_source(
    chain: &[usize],
    nodes: &[AnalyzedNode],
    dataflow: &ChainDataFlow,
    kernel_name: &str,
) -> Option<String> {
    let n_ext = dataflow.external_inputs.len();
    let mut src = String::with_capacity(1024);

    let _ = write!(src, "extern \"C\" __global__ void {kernel_name}(");
    for i in 0..n_ext {
        let _ = write!(src, "const float* __restrict__ in{i}, ");
    }
    let _ = write!(src, "float* __restrict__ out, unsigned int n) {{\n");
    let _ = write!(
        src,
        "  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    );
    let _ = write!(src, "  if (i >= n) return;\n");

    for i in 0..n_ext {
        let _ = write!(src, "  float ext{i} = in{i}[i];\n");
    }

    let mut ptr_expr: HashMap<u64, String> = HashMap::new();
    for (i, &ptr) in dataflow.external_inputs.iter().enumerate() {
        ptr_expr.insert(ptr, format!("ext{i}"));
    }

    fn lookup(map: &HashMap<u64, String>, ptr: u64) -> String {
        map.get(&ptr).cloned().unwrap_or_else(|| "0.0f".to_string())
    }

    for (step, &idx) in chain.iter().enumerate() {
        let node = &nodes[idx];
        let op = node.kernel_name.as_deref().map(extract_op).unwrap_or("?");
        let out_idx = elementwise_out_idx(node.class);
        let out_ptr = *node.arg_values.get(out_idx)?;

        let expr = if node.class == KernelClass::UnaryElementwise {
            let in_ptr = node.arg_values.first().copied().unwrap_or(0);
            op_to_cuda_f32(op, &lookup(&ptr_expr, in_ptr))?
        } else {
            let a_ptr = node.arg_values.first().copied().unwrap_or(0);
            let b_ptr = *node.arg_values.get(1)?;
            op_to_cuda_binary_f32(op, &lookup(&ptr_expr, a_ptr), &lookup(&ptr_expr, b_ptr))?
        };

        let var = format!("t{step}");
        let _ = write!(src, "  float {var} = {expr};\n");
        ptr_expr.insert(out_ptr, var);
    }

    let last = &nodes[chain[chain.len() - 1]];
    let last_out_ptr = last.arg_values[elementwise_out_idx(last.class)];
    let final_expr = lookup(&ptr_expr, last_out_ptr);
    let _ = write!(src, "  out[i] = {final_expr};\n}}\n");

    Some(src)
}

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn ptx_cache_path(source: &str, arch: &str) -> Option<PathBuf> {
    let mut buf = Vec::with_capacity(source.len() + arch.len());
    buf.extend_from_slice(source.as_bytes());
    buf.extend_from_slice(arch.as_bytes());
    let hash = fnv1a_hash(&buf);
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(format!(".cache/nabla/ptx/{hash:016x}.ptx")))
}

fn plan_cache_path(key: u64) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(format!(".cache/nabla/plans/{key:016x}.plan")))
}

fn plan_cache_key(nodes: &[AnalyzedNode]) -> u64 {
    let mut data = Vec::new();
    for node in nodes {
        data.extend_from_slice(node.kernel_name.as_deref().unwrap_or("?").as_bytes());
        data.push(0);
    }
    fnv1a_hash(&data)
}

fn save_plan(key: u64, entries: &[(String, String)]) {
    let Some(path) = plan_cache_path(key) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut content = String::new();
    for (name, source) in entries {
        let _ = write!(content, "KERNEL:{name}\n{source}ENDKERNEL\n");
    }
    let _ = std::fs::write(path, content);
}

fn load_plan(key: u64) -> Option<Vec<(String, String)>> {
    let path = plan_cache_path(key)?;
    let content = std::fs::read_to_string(path).ok()?;
    let mut entries = Vec::new();
    let mut lines = content.lines().peekable();
    while let Some(header) = lines.next() {
        let name = header.strip_prefix("KERNEL:")?.to_string();
        let mut source = String::new();
        for line in lines.by_ref() {
            if line == "ENDKERNEL" {
                break;
            }
            source.push_str(line);
            source.push('\n');
        }
        entries.push((name, source));
    }
    Some(entries)
}

pub(super) fn compile_and_cache_kernel(
    ctx: &CudaCtx,
    name: &str,
    source: &str,
) -> CudaResult<CUfunction> {
    {
        let map = ctx
            .kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = map.get(name) {
            return Ok(entry.func);
        }
    }

    let (major, minor) = super::core::query_compute_capability();
    let arch = super::core::nvrtc_arch(major, minor);
    let cache_path = ptx_cache_path(source, arch);

    let ptx_src = if let Some(cached) = cache_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
    {
        cached
    } else {
        let ptx = nvrtc::compile_ptx_with_opts(
            source,
            nvrtc::CompileOptions {
                arch: Some(arch),
                include_paths: super::nvrtc_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|e| CudaError::KernelNotFound(format!("NVRTC compile: {e}")))?;
        let src = ptx.to_src();
        if let Some(ref path) = cache_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &src);
        }
        src
    };

    let c_ptx = CString::new(ptx_src).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: loading compiled PTX as a CUDA module.
    let module =
        unsafe { cudarc::driver::result::module::load_data(c_ptx.as_ptr().cast::<c_void>()) }
            .map_err(CudaError::Driver)?;

    let c_fn = CString::new(name).map_err(|_| CudaError::NullPtr)?;
    // SAFETY: retrieving function handle from freshly loaded module.
    let func = unsafe { cudarc::driver::result::module::get_function(module, c_fn) }
        .map_err(CudaError::Driver)?;

    let mut map = ctx
        .kernels
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.insert(
        name.to_owned(),
        KernelEntry {
            func,
            _module: module,
        },
    );
    Ok(func)
}

/// Collect predecessor/successor graph node handles for a chain, excluding chain-internal edges.
fn collect_chain_edges(
    nodes: &[AnalyzedNode],
    chain: &[usize],
) -> (Vec<CUgraphNode>, Vec<CUgraphNode>) {
    let chain_set: HashSet<usize> = chain.iter().copied().collect();
    let first_idx = chain[0];
    let last_idx = chain[chain.len() - 1];

    let predecessors: Vec<CUgraphNode> = nodes[first_idx]
        .deps
        .iter()
        .filter(|&&d| !chain_set.contains(&d))
        .map(|&d| nodes[d].node)
        .collect();
    let successors: Vec<CUgraphNode> = nodes[last_idx]
        .dependents
        .iter()
        .filter(|&&d| !chain_set.contains(&d))
        .map(|&d| nodes[d].node)
        .collect();

    let mut extra_preds: Vec<CUgraphNode> = Vec::new();
    for &idx in &chain[1..] {
        for &dep in &nodes[idx].deps {
            if !chain_set.contains(&dep)
                && !predecessors.contains(&nodes[dep].node)
                && !extra_preds.contains(&nodes[dep].node)
            {
                extra_preds.push(nodes[dep].node);
            }
        }
    }

    let mut all_preds = predecessors;
    all_preds.extend(extra_preds);
    (all_preds, successors)
}

fn ops_chain_hash(ops: &[String]) -> u64 {
    ops.iter().fold(0u64, |h, op| {
        h.wrapping_mul(31).wrapping_add(
            op.bytes()
                .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
        )
    })
}

/// Rewrite a single fusion candidate in the CUDA graph in-place.
pub fn apply_fusion(
    cu_graph: CUgraph,
    analyzed: &[AnalyzedNode],
    candidate: &FusionCandidate,
) -> CudaResult<bool> {
    if candidate.node_indices.len() < 2 {
        return Ok(false);
    }

    let dataflow = trace_chain_dataflow(analyzed, &candidate.node_indices);
    let n_ext = dataflow.external_inputs.len();

    let ops_hash = ops_chain_hash(&candidate.ops);
    let kernel_name = format!("k_opt_fuse_{ops_hash:016x}");

    let Some(source) =
        generate_fused_source(&candidate.node_indices, analyzed, &dataflow, &kernel_name)
    else {
        return Ok(false);
    };

    let ctx = get_ctx();
    let func = compile_and_cache_kernel(ctx, &kernel_name, &source)?;

    let mut arg_values: Vec<u64> = Vec::with_capacity(n_ext + 2);
    arg_values.extend_from_slice(&dataflow.external_inputs);
    arg_values.extend([dataflow.final_output, dataflow.n_val]);

    let mut arg_ptrs: Vec<*mut c_void> = arg_values
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect();
    arg_ptrs.push(std::ptr::null_mut());

    let n = dataflow.n_val as u32;
    let block_size = 256u32;
    let grid_size = n.div_ceil(block_size);

    let mut params: cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    params.func = func;
    params.gridDimX = grid_size;
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = block_size;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = arg_ptrs.as_mut_ptr();

    let (predecessors, successors) = collect_chain_edges(analyzed, &candidate.node_indices);

    let mut fused_node: CUgraphNode = std::ptr::null_mut();
    // SAFETY: cu_graph is valid; predecessors are valid node handles from the same graph.
    unsafe {
        cudarc::driver::sys::cuGraphAddKernelNode_v2(
            &mut fused_node,
            cu_graph,
            if predecessors.is_empty() {
                std::ptr::null()
            } else {
                predecessors.as_ptr()
            },
            predecessors.len(),
            &params,
        )
        .result()
        .map_err(CudaError::Driver)?;
    }

    if !successors.is_empty() {
        let from_nodes: Vec<CUgraphNode> = vec![fused_node; successors.len()];
        // SAFETY: fused_node and successors are valid handles in cu_graph.
        unsafe {
            cudarc::driver::sys::cuGraphAddDependencies(
                cu_graph,
                from_nodes.as_ptr(),
                successors.as_ptr(),
                successors.len(),
            )
            .result()
            .map_err(CudaError::Driver)?;
        }
    }

    for &idx in candidate.node_indices.iter().rev() {
        // SAFETY: node handle is valid; cuGraphDestroyNode auto-removes edges.
        unsafe {
            cudarc::driver::sys::cuGraphDestroyNode(analyzed[idx].node)
                .result()
                .map_err(CudaError::Driver)?;
        }
    }

    Ok(true)
}

/// Apply all fusion candidates to a CUDA graph. Returns the number successfully applied.
pub fn apply_all_fusions(
    cu_graph: CUgraph,
    analyzed: &[AnalyzedNode],
    candidates: &[FusionCandidate],
) -> CudaResult<usize> {
    let mut sorted: Vec<usize> = (0..candidates.len()).collect();
    sorted.sort_by(|&a, &b| {
        let max_idx = |c: &FusionCandidate| c.node_indices.iter().max().copied().unwrap_or(0);
        max_idx(&candidates[b]).cmp(&max_idx(&candidates[a]))
    });
    let mut applied = 0usize;
    for idx in sorted {
        if apply_fusion(cu_graph, analyzed, &candidates[idx])? {
            applied += 1;
        }
    }
    Ok(applied)
}

/// Run optimization with disk-cached plans.
pub fn optimize_with_cache(cu_graph: CUgraph) -> CudaResult<(OptimizationReport, usize, bool)> {
    let ctx = get_ctx();
    let (analyzed, report) = analyze_graph(cu_graph)?;
    let key = plan_cache_key(&analyzed);

    if let Some(entries) = load_plan(key) {
        for (name, source) in &entries {
            let _ = compile_and_cache_kernel(ctx, name, source);
        }
        let applied = apply_all_fusions(cu_graph, &analyzed, &report.fusion_candidates)?;
        return Ok((report, applied, true));
    }

    let applied = apply_all_fusions(cu_graph, &analyzed, &report.fusion_candidates)?;

    if applied > 0 {
        let plan_entries: Vec<(String, String)> = report
            .fusion_candidates
            .iter()
            .filter(|c| c.node_indices.len() >= 2)
            .filter_map(|c| {
                let dataflow = trace_chain_dataflow(&analyzed, &c.node_indices);
                let name = format!("k_opt_fuse_{:016x}", ops_chain_hash(&c.ops));
                generate_fused_source(&c.node_indices, &analyzed, &dataflow, &name)
                    .map(|s| (name, s))
            })
            .collect();
        save_plan(key, &plan_entries);
    }

    Ok((report, applied, false))
}
