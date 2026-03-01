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
            graph: None,
            warmup_iters,
            iter_count: 0,
            min_nodes: 3,
            capture_disabled: false,
        }
    }

    #[must_use]
    pub fn with_min_nodes(min_nodes: usize) -> Self {
        Self {
            graph: None,
            warmup_iters: 5,
            iter_count: 0,
            min_nodes: min_nodes.max(1),
            capture_disabled: false,
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

pub(crate) struct GpuTape<T: Scalar> {
    ops: Vec<GpuOp>,
    buffers: HashMap<usize, CudaStorage<T>>,
    grads: HashMap<usize, CudaStorage<T>>,
    next_id: usize,
}

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
        let patched = match op {
            GpuOp::Add { a_id, b_id, .. } => GpuOp::Add { a_id, b_id, out_id },
            GpuOp::Sub { a_id, b_id, .. } => GpuOp::Sub { a_id, b_id, out_id },
            GpuOp::Neg { a_id, .. } => GpuOp::Neg { a_id, out_id },
            GpuOp::Scale { a_id, s_idx, .. } => GpuOp::Scale {
                a_id,
                s_idx,
                out_id,
            },
            GpuOp::Emul { a_id, b_id, .. } => GpuOp::Emul { a_id, b_id, out_id },
            GpuOp::Matmul {
                a_id,
                b_id,
                m,
                k,
                n,
                ..
            } => GpuOp::Matmul {
                a_id,
                b_id,
                out_id,
                m,
                k,
                n,
            },
            GpuOp::Exp { a_id, .. } => GpuOp::Exp { a_id, out_id },
            GpuOp::Ln { a_id, .. } => GpuOp::Ln { a_id, out_id },
            GpuOp::Sin { a_id, .. } => GpuOp::Sin { a_id, out_id },
            GpuOp::Cos { a_id, .. } => GpuOp::Cos { a_id, out_id },
            GpuOp::Tanh { a_id, .. } => GpuOp::Tanh { a_id, out_id },
            GpuOp::SumAll {
                a_id, rows, cols, ..
            } => GpuOp::SumAll {
                a_id,
                out_id,
                rows,
                cols,
            },
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
