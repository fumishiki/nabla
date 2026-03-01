use std::fmt::Write;

pub(crate) fn fuse_kernel_source(
    gpu_expr: &str,
    n_inputs: usize,
    type_name: &str,
    kernel_name: &str,
    reg_estimate: usize,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(if is_f32 { 1536 } else { 512 });

    let _ = write!(src, "// estimated registers: {reg_estimate}\n");

    if is_f32 {
        let scalar_expr = gpu_expr.to_string();

        src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const float* __restrict__ in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str("float* __restrict__ out, unsigned n) {\n");
        src.push_str("    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i4 * 4;\n");
        src.push_str("    if (i + 3 < n) {\n");
        for j in 0..n_inputs {
            if use_ldg {
                let _ = write!(
                    src,
                    "        float4 v{j} = __ldg(reinterpret_cast<const float4*>(in{j}) + i4);\n"
                );
            } else {
                let _ = write!(
                    src,
                    "        float4 v{j} = reinterpret_cast<const float4*>(in{j})[i4];\n"
                );
            }
        }
        src.push_str("        float4 r;\n");
        for comp in &["x", "y", "z", "w"] {
            let mut comp_expr = scalar_expr.clone();
            for j in (0..n_inputs).rev() {
                comp_expr = comp_expr.replace(&format!("in{j}[i]"), &format!("v{j}.{comp}"));
            }
            src.push_str(&format!("        r.{comp} = {comp_expr};\n"));
        }
        src.push_str("        reinterpret_cast<float4*>(out)[i4] = r;\n");
        src.push_str("    } else {\n");
        src.push_str("        for (unsigned j = i; j < n && j < i + 4; j++) {\n");
        let mut tail_expr = scalar_expr;
        for j in (0..n_inputs).rev() {
            if use_ldg {
                tail_expr = tail_expr.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[j])"));
            } else {
                tail_expr = tail_expr.replace(&format!("in{j}[i]"), &format!("in{j}[j]"));
            }
        }
        src.push_str(&format!("            out[j] = {tail_expr};\n"));
        src.push_str("        }\n");
        src.push_str("    }\n}\n");
    } else {
        // double2 vectorized path for f64 (2 elements per thread)
        let scalar_expr = gpu_expr.to_string();

        src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
        src.push_str(kernel_name);
        src.push('(');
        for i in 0..n_inputs {
            src.push_str("const ");
            src.push_str(type_name);
            src.push_str("* __restrict__ in");
            src.push_str(&i.to_string());
            src.push_str(", ");
        }
        src.push_str(type_name);
        src.push_str("* __restrict__ out, unsigned n) {\n");
        src.push_str("    unsigned i2 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i2 * 2;\n");
        src.push_str("    if (i + 1 < n) {\n");
        for j in 0..n_inputs {
            if use_ldg {
                src.push_str(&format!(
                    "        double2 v{j} = __ldg(reinterpret_cast<const double2*>(in{j}) + i2);\n"
                ));
            } else {
                src.push_str(&format!(
                    "        double2 v{j} = reinterpret_cast<const double2*>(in{j})[i2];\n"
                ));
            }
        }
        src.push_str("        double2 r;\n");
        for comp in &["x", "y"] {
            let mut comp_expr = scalar_expr.clone();
            for j in (0..n_inputs).rev() {
                comp_expr = comp_expr.replace(&format!("in{j}[i]"), &format!("v{j}.{comp}"));
            }
            src.push_str(&format!("        r.{comp} = {comp_expr};\n"));
        }
        src.push_str("        reinterpret_cast<double2*>(out)[i2] = r;\n");
        src.push_str("    } else if (i < n) {\n");
        // Scalar tail for odd element count
        let tail_expr = if use_ldg {
            let mut e = scalar_expr;
            for j in (0..n_inputs).rev() {
                e = e.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            e
        } else {
            scalar_expr
        };
        src.push_str(&format!("        out[i] = {tail_expr};\n"));
        src.push_str("    }\n}\n");
    }
    src
}

pub(crate) fn fuse_reduce_kernel_source(
    gpu_expr: &str,
    n_inputs: usize,
    type_name: &str,
    kernel_name: &str,
    axis: u8,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    let zero_init = if is_f32 { "0.0f" } else { "0.0" };
    let t = type_name;
    let mut src = String::with_capacity(2048);

    // Parameter list: const T* in0, ..., T* out, unsigned rows, unsigned cols
    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');
    for i in 0..n_inputs {
        src.push_str(&format!("const {t}* __restrict__ in{i}, "));
    }
    src.push_str(&format!(
        "{t}* __restrict__ out, unsigned rows, unsigned cols) {{\n"
    ));

    if axis == 1 {
        // axis=1: each block handles one row → output shape (rows, 1).
        src.push_str("    unsigned row = blockIdx.x;\n");
        src.push_str("    if (row >= rows) return;\n");
        src.push_str(&format!("    {t} acc = {zero_init};\n"));
        src.push_str("    for (unsigned col = threadIdx.x; col < cols; col += blockDim.x) {\n");
        src.push_str("        unsigned i = row * cols + col;\n");
        // Apply pointwise expression using `in0[i], in1[i], ...` placeholders.
        let load_expr = gpu_expr.to_string();
        let load_str = if use_ldg {
            let mut e = load_expr.clone();
            for j in (0..n_inputs).rev() {
                e = e.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            e
        } else {
            load_expr
        };
        src.push_str(&format!("        {t} v = {load_str};\n"));
        src.push_str("        acc += v;\n");
        src.push_str("    }\n");
    } else {
        // axis=0: each block handles one column → output shape (1, cols).
        src.push_str("    unsigned col = blockIdx.x;\n");
        src.push_str("    if (col >= cols) return;\n");
        src.push_str(&format!("    {t} acc = {zero_init};\n"));
        src.push_str("    for (unsigned row = threadIdx.x; row < rows; row += blockDim.x) {\n");
        src.push_str("        unsigned i = row * cols + col;\n");
        let load_expr = gpu_expr.to_string();
        let load_str = if use_ldg {
            let mut e = load_expr.clone();
            for j in (0..n_inputs).rev() {
                e = e.replace(&format!("in{j}[i]"), &format!("__ldg(&in{j}[i])"));
            }
            e
        } else {
            load_expr
        };
        src.push_str(&format!("        {t} v = {load_str};\n"));
        src.push_str("        acc += v;\n");
        src.push_str("    }\n");
    }

    // Warp shuffle reduction
    src.push_str("    // warp-level reduction\n");
    src.push_str("    for (int offset = 16; offset > 0; offset >>= 1)\n");
    src.push_str("        acc += __shfl_down_sync(0xffffffff, acc, offset);\n");

    // Block-level reduction via shared memory (handles blockDim > 32)
    src.push_str("    __shared__ ");
    src.push_str(t);
    src.push_str(" smem[32];\n");
    src.push_str("    unsigned lane = threadIdx.x & 31u;\n");
    src.push_str("    unsigned wid  = threadIdx.x >> 5u;\n");
    src.push_str("    if (lane == 0) smem[wid] = acc;\n");
    src.push_str("    __syncthreads();\n");
    src.push_str("    if (threadIdx.x < (blockDim.x >> 5u)) {\n");
    src.push_str("        acc = smem[threadIdx.x];\n");
    src.push_str("        for (int offset = 16; offset > 0; offset >>= 1)\n");
    src.push_str("            acc += __shfl_down_sync(0xffffffff, acc, offset);\n");
    src.push_str("    }\n");

    if axis == 1 {
        src.push_str("    if (threadIdx.x == 0) out[row] = acc;\n");
    } else {
        src.push_str("    if (threadIdx.x == 0) out[col] = acc;\n");
    }

    src.push_str("}\n");
    src
}

pub(crate) fn mega_fuse_kernel_source(
    ops: &[(String, usize)], // (gpu_expr, n_inputs)
    uses_prev: &[bool],
    type_name: &str,
    kernel_name: &str,
    use_ldg: bool,
) -> String {
    debug_assert_eq!(
        ops.len(),
        uses_prev.len(),
        "mega_fuse_kernel_source: ops and uses_prev must have equal length"
    );
    let is_f32 = type_name == "float";
    let mut src = String::with_capacity(2048);

    // ── Kernel signature ──────────────────────────────────────────────────────
    // All input pointers are emitted regardless of uses_prev.
    // The `__NABLA_PREV__` sentinel in the GPU expr maps to a register reference,
    // not to any inN pointer — it is completely orthogonal to the inN[i] mapping.
    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');
    let mut first_param = true;
    for (op_idx, (_expr, n_in)) in ops.iter().enumerate() {
        for j in 0..*n_in {
            if !first_param {
                src.push_str(", ");
            }
            first_param = false;
            src.push_str(&format!("const {type_name}* __restrict__ op{op_idx}_in{j}"));
        }
        if !first_param {
            src.push_str(", ");
        }
        first_param = false;
        src.push_str(&format!("{type_name}* __restrict__ op{op_idx}_out"));
    }
    src.push_str(", unsigned n) {\n");

    if is_f32 {
        // ── float4 vectorized main path ───────────────────────────────────────
        src.push_str("    unsigned i4 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i4 * 4;\n");
        src.push_str("    if (i + 3 < n) {\n");

        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            src.push_str(&format!("        // Op {op_idx}\n"));
            // Load all global inputs.  For uses_prev ops the expr may also reference
            // __NABLA_PREV__ (replaced below) alongside normal inN[i] references.
            for j in 0..*n_in {
                if use_ldg {
                    src.push_str(&format!(
                        "        float4 op{op_idx}_v{j} = __ldg(reinterpret_cast<const float4*>(op{op_idx}_in{j}) + i4);\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "        float4 op{op_idx}_v{j} = reinterpret_cast<const float4*>(op{op_idx}_in{j})[i4];\n"
                    ));
                }
            }
            src.push_str(&format!("        float4 op{op_idx}_r;\n"));
            for comp in &["x", "y", "z", "w"] {
                let mut comp_expr = gpu_expr.clone();
                // Replace DAG sentinel with the previous op's register component.
                if op_uses_prev {
                    let prev_reg = format!("op{}_r.{}", op_idx - 1, comp);
                    comp_expr = comp_expr.replace("__NABLA_PREV__", &prev_reg);
                }
                // Replace inN[i] placeholders with loaded float4 components.
                for j in (0..*n_in).rev() {
                    comp_expr =
                        comp_expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_v{j}.{comp}"));
                }
                src.push_str(&format!("        op{op_idx}_r.{comp} = {comp_expr};\n"));
            }
            src.push_str(&format!(
                "        reinterpret_cast<float4*>(op{op_idx}_out)[i4] = op{op_idx}_r;\n"
            ));
        }

        // ── f32 scalar tail (remainder < 4 elements) ─────────────────────────
        src.push_str("    } else {\n");
        src.push_str("        for (unsigned j = i; j < n && j < i + 4; j++) {\n");
        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            let mut tail_expr = gpu_expr.clone();
            // DAG sentinel → previous op's output element (already written in this loop iteration).
            if op_uses_prev {
                tail_expr =
                    tail_expr.replace("__NABLA_PREV__", &format!("op{}_out[j]", op_idx - 1));
            }
            for j in (0..*n_in).rev() {
                if use_ldg {
                    tail_expr = tail_expr.replace(
                        &format!("in{j}[i]"),
                        &format!("__ldg(&op{op_idx}_in{j}[j])"),
                    );
                } else {
                    tail_expr =
                        tail_expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_in{j}[j]"));
                }
            }
            src.push_str(&format!("            op{op_idx}_out[j] = {tail_expr};\n"));
        }
        src.push_str("        }\n");
        src.push_str("    }\n}\n");
    } else {
        // ── double2 vectorized main path ──────────────────────────────────────
        src.push_str("    unsigned i2 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        src.push_str("    unsigned i = i2 * 2;\n");
        src.push_str("    if (i + 1 < n) {\n");

        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            src.push_str(&format!("        // Op {op_idx}\n"));
            for j in 0..*n_in {
                if use_ldg {
                    src.push_str(&format!(
                        "        double2 op{op_idx}_v{j} = __ldg(reinterpret_cast<const double2*>(op{op_idx}_in{j}) + i2);\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "        double2 op{op_idx}_v{j} = reinterpret_cast<const double2*>(op{op_idx}_in{j})[i2];\n"
                    ));
                }
            }
            src.push_str(&format!("        double2 op{op_idx}_r;\n"));
            for comp in &["x", "y"] {
                let mut comp_expr = gpu_expr.clone();
                // Replace DAG sentinel with the previous op's register component.
                if op_uses_prev {
                    let prev_reg = format!("op{}_r.{}", op_idx - 1, comp);
                    comp_expr = comp_expr.replace("__NABLA_PREV__", &prev_reg);
                }
                for j in (0..*n_in).rev() {
                    comp_expr =
                        comp_expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_v{j}.{comp}"));
                }
                src.push_str(&format!("        op{op_idx}_r.{comp} = {comp_expr};\n"));
            }
            src.push_str(&format!(
                "        reinterpret_cast<double2*>(op{op_idx}_out)[i2] = op{op_idx}_r;\n"
            ));
        }

        // ── f64 scalar tail (odd element count) ──────────────────────────────
        src.push_str("    } else if (i < n) {\n");
        for (op_idx, (gpu_expr, n_in)) in ops.iter().enumerate() {
            let op_uses_prev = uses_prev[op_idx];
            let mut tail_expr = gpu_expr.clone();
            // DAG sentinel → previous op's output element (already written).
            if op_uses_prev {
                tail_expr =
                    tail_expr.replace("__NABLA_PREV__", &format!("op{}_out[i]", op_idx - 1));
            }
            for j in (0..*n_in).rev() {
                if use_ldg {
                    tail_expr = tail_expr.replace(
                        &format!("in{j}[i]"),
                        &format!("__ldg(&op{op_idx}_in{j}[i])"),
                    );
                } else {
                    tail_expr =
                        tail_expr.replace(&format!("in{j}[i]"), &format!("op{op_idx}_in{j}[i]"));
                }
            }
            src.push_str(&format!("        op{op_idx}_out[i] = {tail_expr};\n"));
        }
        src.push_str("    }\n}\n");
    }
    src
}

pub(crate) fn mega_fuse_tiled_kernel_source(
    ops: &[(String, usize)], // (gpu_expr, n_inputs) — all must have equal n_inputs
    type_name: &str,
    kernel_name: &str,
    use_ldg: bool,
) -> String {
    let is_f32 = type_name == "float";
    // Tile sizes chosen so smem[tile_size] fits in typical 48 KiB L1/smem.
    // f32: 256 threads × 4 elems = 1024 floats  = 4 KiB per input.
    // f64: 256 threads × 2 elems = 512  doubles = 4 KiB per input.
    let tile_size: usize = if is_f32 { 1024 } else { 512 };
    let elems_per_thread: usize = if is_f32 { 4 } else { 2 };
    let n_inputs = ops.first().map(|(_, n)| *n).unwrap_or(1);

    let mut src = String::with_capacity(4096);

    // Kernel signature: all shared inputs once, then one output per op, then n.
    src.push_str("extern \"C\" __global__ __launch_bounds__(256) void ");
    src.push_str(kernel_name);
    src.push('(');

    // Shared inputs appear once at the front (they are the same across all ops).
    for j in 0..n_inputs {
        src.push_str(&format!("const {type_name}* __restrict__ in{j}, "));
    }
    // One output per op.
    for (op_idx, _) in ops.iter().enumerate() {
        src.push_str(&format!("{type_name}* __restrict__ op{op_idx}_out, "));
    }
    src.push_str("unsigned n) {\n");

    // Shared memory: one smem slot per input, each of tile_size elements.
    for j in 0..n_inputs {
        src.push_str(&format!(
            "    __shared__ {type_name} s_in{j}[{tile_size}];\n"
        ));
    }

    src.push_str(&format!(
        "    unsigned tile_base = blockIdx.x * {tile_size}u;\n"
    ));
    src.push_str("    unsigned tid = threadIdx.x;\n\n");

    // Cooperative load: 256 threads load tile_size elements (elems_per_thread per thread).
    src.push_str("    // Phase 1: cooperative load of shared inputs into smem\n");
    src.push_str(&format!(
        "    #pragma unroll\n    for (unsigned k = 0; k < {elems_per_thread}u; k++) {{\n"
    ));
    src.push_str(&format!(
        "        unsigned smem_idx = tid * {elems_per_thread}u + k;\n"
    ));
    src.push_str("        unsigned glob_idx = tile_base + smem_idx;\n");
    src.push_str("        if (glob_idx < n) {\n");
    for j in 0..n_inputs {
        if use_ldg {
            src.push_str(&format!(
                "            s_in{j}[smem_idx] = __ldg(&in{j}[glob_idx]);\n"
            ));
        } else {
            src.push_str(&format!(
                "            s_in{j}[smem_idx] = in{j}[glob_idx];\n"
            ));
        }
    }
    src.push_str("        }\n    }\n");
    src.push_str("    __syncthreads();\n\n");

    // Phase 2: each thread processes elems_per_thread elements from smem.
    src.push_str("    // Phase 2: apply all ops reading from smem\n");
    src.push_str(&format!(
        "    #pragma unroll\n    for (unsigned k = 0; k < {elems_per_thread}u; k++) {{\n"
    ));
    src.push_str(&format!(
        "        unsigned smem_idx = tid * {elems_per_thread}u + k;\n"
    ));
    src.push_str("        unsigned glob_idx = tile_base + smem_idx;\n");
    src.push_str("        if (glob_idx < n) {\n");

    // Emit each op: replace `in{j}[i]` placeholders with smem references.
    for (op_idx, (gpu_expr, _n_in)) in ops.iter().enumerate() {
        let mut expr = gpu_expr.clone();
        // Replace placeholders in reverse order to avoid partial matches.
        for j in (0..n_inputs).rev() {
            expr = expr.replace(&format!("in{j}[i]"), &format!("s_in{j}[smem_idx]"));
        }
        src.push_str(&format!("            op{op_idx}_out[glob_idx] = {expr};\n"));
    }

    src.push_str("        }\n    }\n}\n");
    src
}
