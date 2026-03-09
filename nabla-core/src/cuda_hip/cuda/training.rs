use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;
use std::collections::HashMap;

use crate::scalar::Scalar;

use super::*;

pub struct TrainingGraph {
    graph: Option<PyGraph>,
    warmup_iters: usize,
    iter_count: usize,
    min_nodes: usize,
    capture_disabled: bool,
}

impl TrainingGraph {
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: None,
            warmup_iters: 5,
            iter_count: 0,
            min_nodes: 3,
            capture_disabled: false,
        }
    }

    #[must_use]
    pub fn with_warmup(warmup_iters: usize) -> Self {
        Self {
            warmup_iters,
            ..Self::new()
        }
    }

    #[must_use]
    pub fn with_min_nodes(min_nodes: usize) -> Self {
        Self {
            min_nodes: min_nodes.max(1),
            ..Self::new()
        }
    }

    pub fn step<F: FnMut()>(&mut self, f: &mut F) -> CudaResult<()> {
        if self.capture_disabled {
            f();
            return Ok(());
        }
        self.iter_count += 1;

        if self.iter_count <= self.warmup_iters {
            f();
            cuda_synchronize();
            Ok(())
        } else if self.graph.is_none() {
            let captured = PyGraph::capture(|| f())?;
            if captured.kernel_node_count() < self.min_nodes {
                self.capture_disabled = true;
                return Ok(());
            }
            self.graph = Some(captured);
            Ok(())
        } else {
            self.graph.as_ref().ok_or(CudaError::NullPtr)?.launch()
        }
    }

    pub fn reset(&mut self) {
        self.graph = None;
        self.iter_count = 0;
        self.capture_disabled = false;
    }

    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }

    #[must_use]
    pub fn kernel_node_count(&self) -> usize {
        self.graph.as_ref().map_or(0, |g| g.kernel_node_count())
    }

    #[must_use]
    pub fn arg_count(&self, node_idx: usize) -> usize {
        self.graph.as_ref().map_or(0, |g| g.arg_count(node_idx))
    }

    #[must_use]
    pub fn get_param(&self, node_idx: usize, param_idx: usize) -> CUdeviceptr {
        self.graph
            .as_ref()
            .map_or(0, |g| g.get_param(node_idx, param_idx))
    }

    pub fn update_param_ptr(
        &mut self,
        node_idx: usize,
        param_idx: usize,
        new_ptr: CUdeviceptr,
    ) -> CudaResult<()> {
        self.graph
            .as_mut()
            .ok_or(CudaError::NullPtr)?
            .update_node_param_ptr(node_idx, param_idx, new_ptr)
    }
}

impl Default for TrainingGraph {
    fn default() -> Self {
        Self::new()
    }
}

struct ParamBinding {
    original_ptr: u64,
    refs: Vec<(usize, usize)>,
}

fn build_bindings(tracked_ptrs: &[u64], kernel_nodes: &[KernelNodeState]) -> Vec<ParamBinding> {
    tracked_ptrs
        .iter()
        .map(|&ptr| {
            let refs = kernel_nodes
                .iter()
                .enumerate()
                .flat_map(|(ni, node)| {
                    node.arg_bytes
                        .iter()
                        .enumerate()
                        .filter(move |&(_, &v)| v == ptr)
                        .map(move |(ai, _)| (ni, ai))
                })
                .collect();
            ParamBinding {
                original_ptr: ptr,
                refs,
            }
        })
        .collect()
}

/// Fused CUDA Graph with automatic parameter pointer rebinding.
/// Tracks registered parameter device pointers, captures a training step
/// into a CUDA Graph, and on replay automatically updates any kernel nodes
/// whose arguments match relocated parameters.
pub struct NablaGraph {
    inner: Option<PyGraph>,
    tracked_ptrs: Vec<u64>,
    bindings: Vec<ParamBinding>,
    warmup_iters: usize,
    iter_count: usize,
    capture_disabled: bool,
    last_profile: Option<Vec<usize>>,
}

impl NablaGraph {
    #[must_use]
    pub fn with_warmup(warmup: usize) -> Self {
        Self {
            inner: None,
            tracked_ptrs: Vec::new(),
            bindings: Vec::new(),
            warmup_iters: warmup.max(1),
            iter_count: 0,
            capture_disabled: false,
            last_profile: None,
        }
    }

    /// Execute one training step. During warmup, runs eagerly. On the first
    /// post-warmup call, captures into a CUDA Graph and auto-binds parameter
    /// pointers. Subsequent calls update changed pointers and replay the graph.
    ///
    /// `current_ptrs` must match the order of the initial `current_ptrs` from
    /// the capture call (i.e., same parameters in the same order).
    pub fn step<F: FnMut()>(&mut self, f: &mut F, current_ptrs: &[u64]) -> CudaResult<()> {
        if self.capture_disabled {
            f();
            return Ok(());
        }
        self.iter_count += 1;

        if self.iter_count <= self.warmup_iters {
            if self.iter_count == 1 {
                if let Some(ref sizes) = self.last_profile {
                    let _ = cuda_pre_warm_pool(sizes);
                }
            }
            f();
            cuda_synchronize();
            Ok(())
        } else if self.inner.is_none() {
            self.tracked_ptrs = current_ptrs.to_vec();
            let captured = PyGraph::capture(|| f())?;
            if captured.kernel_node_count() < 3 {
                self.capture_disabled = true;
                return Ok(());
            }
            self.bindings = self.auto_bind(&captured.kernel_nodes);
            self.inner = Some(captured);
            Ok(())
        } else {
            self.rebind_changed(current_ptrs)?;
            self.inner.as_ref().ok_or(CudaError::NullPtr)?.launch()
        }
    }

    fn auto_bind(&self, kernel_nodes: &[KernelNodeState]) -> Vec<ParamBinding> {
        build_bindings(&self.tracked_ptrs, kernel_nodes)
    }

    fn rebind_changed(&mut self, current_ptrs: &[u64]) -> CudaResult<()> {
        let graph = self.inner.as_mut().ok_or(CudaError::NullPtr)?;
        for (i, &new_ptr) in current_ptrs.iter().enumerate() {
            if i < self.bindings.len() && new_ptr != self.bindings[i].original_ptr {
                for &(node_idx, arg_idx) in &self.bindings[i].refs {
                    graph.update_node_param_ptr(node_idx, arg_idx, new_ptr)?;
                }
                self.bindings[i].original_ptr = new_ptr;
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.inner = None;
        self.bindings.clear();
        self.tracked_ptrs.clear();
        self.iter_count = 0;
        self.capture_disabled = false;
    }

    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.inner.is_some()
    }

    #[must_use]
    pub fn kernel_node_count(&self) -> usize {
        self.inner.as_ref().map_or(0, |g| g.kernel_node_count())
    }

    /// Analyze the captured graph for optimization opportunities.
    pub fn analyze(&self) -> CudaResult<OptimizationReport> {
        let graph = self.inner.as_ref().ok_or(CudaError::NullPtr)?;
        let (_, report) = analyze_graph(graph.cu_graph)?;
        Ok(report)
    }

    /// Analyze and apply elementwise fusion rewrites to the captured graph.
    /// Re-instantiates the graph executable and rebinds parameters after rewriting.
    pub fn optimize(&mut self) -> CudaResult<OptimizationReport> {
        let graph = self.inner.as_mut().ok_or(CudaError::NullPtr)?;
        let (report, applied, _cache_hit) = optimize_with_cache(graph.cu_graph)?;

        if applied > 0 {
            // SAFETY: cu_graph_exec is valid; destroying before re-instantiation.
            unsafe {
                cudarc::driver::sys::cuGraphExecDestroy(graph.cu_graph_exec);
            }

            let new_exec = unsafe {
                let mut exec = std::mem::MaybeUninit::uninit();
                // SAFETY: cu_graph is valid after node rewrites; instantiating a new executable.
                cudarc::driver::sys::cuGraphInstantiateWithFlags(
                    exec.as_mut_ptr(),
                    graph.cu_graph,
                    0,
                )
                .result()
                .map_err(CudaError::Driver)?;
                exec.assume_init()
            };
            graph.cu_graph_exec = new_exec;
            graph.kernel_nodes = PyGraph::collect_kernel_nodes(graph.cu_graph)?;

            let kernel_nodes = &self.inner.as_ref().ok_or(CudaError::NullPtr)?.kernel_nodes;
            self.bindings = build_bindings(&self.tracked_ptrs, kernel_nodes);
        }

        // Extract allocation profile for pool pre-warming on next reset cycle
        let graph = self.inner.as_ref().ok_or(CudaError::NullPtr)?;
        let (analyzed_nodes, _) = analyze_graph(graph.cu_graph)?;
        let profile = extract_allocation_profile(&analyzed_nodes);
        let _ = cuda_pre_warm_pool(&profile.buffer_sizes);
        self.last_profile = Some(profile.buffer_sizes);

        Ok(report)
    }
}

pub struct DoubleBuffer<T: Scalar> {
    buffers: [CudaStorage<T>; 2],
    active: usize,
}

impl<T: Scalar> DoubleBuffer<T> {
    pub fn new(nrows: usize, ncols: usize) -> CudaResult<Self> {
        let ctx = get_ctx();
        let bytes = nrows * ncols * std::mem::size_of::<T>();
        let buf0 = CuBuffer::alloc_async(&ctx.stream, bytes)?;
        let buf1 = CuBuffer::alloc_async(&ctx.stream, bytes)?;
        Ok(Self {
            buffers: [
                CudaStorage::new(nrows, ncols, buf0),
                CudaStorage::new(nrows, ncols, buf1),
            ],
            active: 0,
        })
    }

    #[must_use]
    pub fn active(&self) -> &CudaStorage<T> {
        &self.buffers[self.active]
    }

    #[must_use]
    pub fn active_ptr(&self) -> CUdeviceptr {
        self.buffers[self.active].buf.ptr
    }

    #[must_use]
    pub fn inactive_ptr(&self) -> CUdeviceptr {
        self.buffers[1 - self.active].buf.ptr
    }

    pub fn upload_next(&self, data: &[T]) -> CudaResult<()> {
        let ctx = get_ctx();
        let inactive = &self.buffers[1 - self.active];
        assert_eq!(
            inactive.n(),
            data.len(),
            "DoubleBuffer::upload_next: data.len()={} != buffer size={}",
            data.len(),
            inactive.n()
        );
        // SAFETY: copying from host slice to pre-allocated GPU buffer of matching size.
        unsafe {
            result::memcpy_htod_async(inactive.buf.ptr, data, ctx.copy_stream.cu_stream())
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    pub fn swap(&mut self) -> CUdeviceptr {
        self.active = 1 - self.active;
        self.buffers[self.active].buf.ptr
    }
}

// SAFETY: CudaStorage<T> is Send+Sync when T: Scalar (see impl above).
unsafe impl<T: Scalar> Send for DoubleBuffer<T> {}
unsafe impl<T: Scalar> Sync for DoubleBuffer<T> {}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) enum GpuOp {
    Add {
        a_id: usize,
        b_id: usize,
        out_id: usize,
    },
    Sub {
        a_id: usize,
        b_id: usize,
        out_id: usize,
    },
    Neg {
        a_id: usize,
        out_id: usize,
    },
    Scale {
        a_id: usize,
        s_idx: usize,
        out_id: usize,
    },
    Emul {
        a_id: usize,
        b_id: usize,
        out_id: usize,
    },
    Matmul {
        a_id: usize,
        b_id: usize,
        out_id: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    Exp {
        a_id: usize,
        out_id: usize,
    },
    Ln {
        a_id: usize,
        out_id: usize,
    },
    Sin {
        a_id: usize,
        out_id: usize,
    },
    Cos {
        a_id: usize,
        out_id: usize,
    },
    Tanh {
        a_id: usize,
        out_id: usize,
    },
    SumAll {
        a_id: usize,
        out_id: usize,
        rows: usize,
        cols: usize,
    },
}

#[allow(dead_code)]
pub(crate) struct GpuTape<T: Scalar> {
    ops: Vec<GpuOp>,
    buffers: HashMap<usize, CudaStorage<T>>,
    grads: HashMap<usize, CudaStorage<T>>,
    next_id: usize,
}

#[allow(dead_code)]
impl<T: Scalar> GpuTape<T> {
    pub(crate) fn new() -> Self {
        Self {
            ops: Vec::new(),
            buffers: HashMap::new(),
            grads: HashMap::new(),
            next_id: 0,
        }
    }

    pub(crate) fn register(&mut self, storage: CudaStorage<T>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.buffers.insert(id, storage);
        id
    }

    pub(crate) fn record(&mut self, op: GpuOp, out: CudaStorage<T>) -> usize {
        let out_id = self.register(out);
        macro_rules! patch {
            ($variant:ident { $($field:ident),+ }) => {
                GpuOp::$variant { $($field,)+ out_id }
            };
        }
        let patched = match op {
            GpuOp::Add { a_id, b_id, .. } => patch!(Add { a_id, b_id }),
            GpuOp::Sub { a_id, b_id, .. } => patch!(Sub { a_id, b_id }),
            GpuOp::Neg { a_id, .. } => patch!(Neg { a_id }),
            GpuOp::Scale { a_id, s_idx, .. } => patch!(Scale { a_id, s_idx }),
            GpuOp::Emul { a_id, b_id, .. } => patch!(Emul { a_id, b_id }),
            GpuOp::Matmul {
                a_id,
                b_id,
                m,
                k,
                n,
                ..
            } => patch!(Matmul {
                a_id,
                b_id,
                m,
                k,
                n
            }),
            GpuOp::Exp { a_id, .. } => patch!(Exp { a_id }),
            GpuOp::Ln { a_id, .. } => patch!(Ln { a_id }),
            GpuOp::Sin { a_id, .. } => patch!(Sin { a_id }),
            GpuOp::Cos { a_id, .. } => patch!(Cos { a_id }),
            GpuOp::Tanh { a_id, .. } => patch!(Tanh { a_id }),
            GpuOp::SumAll {
                a_id, rows, cols, ..
            } => patch!(SumAll { a_id, rows, cols }),
        };
        self.ops.push(patched);
        out_id
    }

    fn accum_grad(&mut self, id: usize, delta: CudaStorage<T>) {
        if let Some(existing) = self.grads.get(&id) {
            let sum = launch_binary(existing, &delta, "add");
            self.grads.insert(id, sum);
        } else {
            self.grads.insert(id, delta);
        }
    }

    pub(crate) fn backward(&mut self, loss_id: usize) {
        let loss_buf = self
            .buffers
            .get(&loss_id)
            .unwrap_or_else(|| panic!("GpuTape::backward: loss_id {loss_id} not found"));
        let seed = cuda_fill(loss_buf.nrows, loss_buf.ncols, T::one_impl());
        self.grads.insert(loss_id, seed);

        for i in (0..self.ops.len()).rev() {
            let op = self.ops[i].clone();
            match op {
                GpuOp::Add { a_id, b_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let ga = cuda_clone(g);
                        let gb = cuda_clone(g);
                        self.accum_grad(a_id, ga);
                        self.accum_grad(b_id, gb);
                    }
                }
                GpuOp::Sub { a_id, b_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let ga = cuda_clone(g);
                        let neg_g = launch_unary(g, "neg");
                        self.accum_grad(a_id, ga);
                        self.accum_grad(b_id, neg_g);
                    }
                }
                GpuOp::Neg { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let neg_g = launch_unary(g, "neg");
                        self.accum_grad(a_id, neg_g);
                    }
                }
                GpuOp::Scale {
                    a_id,
                    s_idx,
                    out_id,
                } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let s_val = cuda_get(
                            self.buffers
                                .get(&s_idx)
                                .unwrap_or_else(|| panic!("GpuTape: scalar {s_idx} missing")),
                            0,
                            0,
                        );
                        let da = cuda_scale(g, s_val);
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Emul { a_id, b_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let b_buf = self
                            .buffers
                            .get(&b_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {b_id} missing"));
                        let a_buf = self
                            .buffers
                            .get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let da = launch_binary(g, b_buf, "emul");
                        let db = launch_binary(g, a_buf, "emul");
                        self.accum_grad(a_id, da);
                        self.accum_grad(b_id, db);
                    }
                }
                GpuOp::Matmul {
                    a_id,
                    b_id,
                    out_id,
                    m,
                    k: _,
                    n,
                } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let b_buf = self
                            .buffers
                            .get(&b_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {b_id} missing"));
                        let a_buf = self
                            .buffers
                            .get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let bt = cuda_transpose(b_buf);
                        let at = cuda_transpose(a_buf);
                        let mut da = cuda_zeros::<T>(m, bt.ncols);
                        cuda_matmul(&mut da, g, &bt);
                        let mut db = cuda_zeros::<T>(at.nrows, n);
                        cuda_matmul(&mut db, &at, g);
                        self.accum_grad(a_id, da);
                        self.accum_grad(b_id, db);
                    }
                }
                GpuOp::Exp { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let out_buf = self
                            .buffers
                            .get(&out_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {out_id} missing"));
                        let da = launch_binary(g, out_buf, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Ln { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let a_buf = self
                            .buffers
                            .get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let da = launch_binary(g, a_buf, "ediv");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Sin { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let a_buf = self
                            .buffers
                            .get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let cos_a = launch_unary(a_buf, "cos");
                        let da = launch_binary(g, &cos_a, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Cos { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let a_buf = self
                            .buffers
                            .get(&a_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {a_id} missing"));
                        let sin_a = launch_unary(a_buf, "sin");
                        let neg_sin = launch_unary(&sin_a, "neg");
                        let da = launch_binary(g, &neg_sin, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::Tanh { a_id, out_id } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let out_buf = self
                            .buffers
                            .get(&out_id)
                            .unwrap_or_else(|| panic!("GpuTape: buffer {out_id} missing"));
                        let out_sq = launch_binary(out_buf, out_buf, "emul");
                        let ones = cuda_fill(out_sq.nrows, out_sq.ncols, T::one_impl());
                        let sech2 = launch_binary(&ones, &out_sq, "sub");
                        let da = launch_binary(g, &sech2, "emul");
                        self.accum_grad(a_id, da);
                    }
                }
                GpuOp::SumAll {
                    a_id,
                    out_id,
                    rows,
                    cols,
                } => {
                    if let Some(g) = self.grads.get(&out_id) {
                        let g_val = cuda_get(g, 0, 0);
                        let da = cuda_fill(rows, cols, g_val);
                        self.accum_grad(a_id, da);
                    }
                }
            }
        }
    }

    pub(crate) fn grad(&self, id: usize) -> Option<&CudaStorage<T>> {
        self.grads.get(&id)
    }
}
