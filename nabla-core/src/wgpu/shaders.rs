use super::storage::{PipelineKey, ScalarKind, ShaderOp};

pub(super) fn generate_shader(key: PipelineKey) -> String {
    let wg = key.wg_size;
    let scalar = key.scalar;
    let mut src = match key.op {
        ShaderOp::Binary => gen_binary(wg, scalar),
        ShaderOp::Scale => gen_scale(wg, scalar),
        ShaderOp::Unary => gen_unary(wg, scalar),
        ShaderOp::Powf => gen_powf(wg, scalar),
        ShaderOp::Transpose => gen_transpose(wg, scalar),
        ShaderOp::Copy => gen_copy(wg, scalar),
        ShaderOp::FillZeros => gen_fill_zeros(wg, scalar),
        ShaderOp::FillScalar => gen_fill_scalar(wg, scalar),
        ShaderOp::FillIdentity => gen_fill_identity(wg, scalar),
        ShaderOp::ReduceSum => {
            assert_f32(scalar, "reduce_sum");
            gen_reduce_sum(wg)
        }
        ShaderOp::ReduceMax => {
            assert_f32(scalar, "reduce_max");
            gen_reduce_max(wg)
        }
        ShaderOp::ReduceMin => {
            assert_f32(scalar, "reduce_min");
            gen_reduce_min(wg)
        }
        ShaderOp::ReduceProd => {
            assert_f32(scalar, "reduce_prod");
            gen_reduce_prod(wg)
        }
        ShaderOp::ReduceCountNonzero => {
            assert_f32(scalar, "reduce_count_nonzero");
            gen_reduce_count_nonzero(wg)
        }
        ShaderOp::Argmax => {
            assert_f32(scalar, "argmax");
            gen_argmax(wg)
        }
        ShaderOp::Argmin => {
            assert_f32(scalar, "argmin");
            gen_argmin(wg)
        }
        ShaderOp::Matmul { tile } => gen_matmul(tile, scalar),
        ShaderOp::MatmulRegTile { tr, tc, bm, bn, bk } => {
            gen_matmul_register_tile(tr, tc, bm, bn, bk, scalar)
        }
        ShaderOp::ActivationSilu => gen_activation_silu(wg, scalar),
        ShaderOp::ActivationMish => gen_activation_mish(wg, scalar),
        ShaderOp::ActivationLeakyRelu => gen_activation_leaky_relu(wg, scalar),
        ShaderOp::ActivationElu => gen_activation_elu(wg, scalar),
        ShaderOp::ActivationHardswish => gen_activation_hardswish(wg, scalar),
        ShaderOp::ActivationSigmoid => gen_activation_sigmoid(wg, scalar),
        ShaderOp::Softmax => gen_softmax(wg, scalar),
        ShaderOp::LayerNorm => gen_layer_norm(wg, scalar),
        ShaderOp::RmsNorm => gen_rms_norm(wg, scalar),
        ShaderOp::SumAxis1 => gen_sum_axis1(wg, scalar),
        ShaderOp::MaxAxis1 => gen_max_axis1(wg, scalar),
        ShaderOp::Embedding => gen_embedding(wg, scalar),
        ShaderOp::Axpy => gen_axpy(wg, scalar),
    };
    if matches!(scalar, ScalarKind::F16) {
        src = format!("enable f16;\n{src}");
    }
    src
}

fn assert_f32(scalar: ScalarKind, op: &str) {
    if !matches!(scalar, ScalarKind::F32) {
        panic!("nabla: wgpu {op} shader supports only f32 for now");
    }
}

fn scalar_ty(scalar: ScalarKind) -> &'static str {
    match scalar {
        ScalarKind::F32 => "f32",
        ScalarKind::F16 => "f16",
    }
}

fn scalar_from_f32_bits(scalar: ScalarKind, expr: &str) -> String {
    match scalar {
        ScalarKind::F32 => format!("bitcast<f32>({expr})"),
        ScalarKind::F16 => format!("f16(bitcast<f32>({expr}))"),
    }
}

fn scalar_min_value(scalar: ScalarKind) -> &'static str {
    match scalar {
        ScalarKind::F32 => "-3.402823e+38",
        ScalarKind::F16 => "-65504.0",
    }
}

fn gen_binary(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> b: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;
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
    else if op == 3u {{ out[i] = a[i] / b[i]; }}
    else {{ out[i] = atan2(a[i], b[i]); }}
}}
",
        ty = ty
    )
}

fn gen_scale(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let val = scalar_from_f32_bits(scalar, "params[1]");
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let scalar = {val};
    out[i] = a[i] * scalar;
}}
",
        ty = ty,
        val = val
    )
}

fn gen_unary(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
fn erf_approx(x: {ty}) -> {ty} {{
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
    else if op == 13u {{ out[i] = round(v); }}
    else if op == 14u {{ out[i] = tan(v); }}
    else if op == 15u {{ out[i] = asin(v); }}
    else if op == 16u {{ out[i] = acos(v); }}
    else if op == 17u {{ out[i] = atan(v); }}
    else if op == 18u {{ out[i] = sinh(v); }}
    else if op == 19u {{ out[i] = cosh(v); }}
    else if op == 20u {{ out[i] = asinh(v); }}
    else if op == 21u {{ out[i] = acosh(v); }}
    else if op == 22u {{ out[i] = atanh(v); }}
    else if op == 23u {{ out[i] = log2(v); }}
    else {{ out[i] = log(v) / log(10.0); }}
}}
",
        ty = ty
    )
}

fn gen_powf(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let p = scalar_from_f32_bits(scalar, "params[1]");
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let p = {p};
    out[i] = pow(a[i], p);
}}
",
        ty = ty,
        p = p
    )
}

fn gen_transpose(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
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
",
        ty = ty
    )
}

fn gen_copy(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if i >= params[0] {{ return; }}
    out[i] = a[i];
}}
",
        ty = ty
    )
}

fn gen_matmul(tile: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let tile_sq = tile * tile;
    let tile_m1 = tile - 1;
    format!(
        r"
var<workgroup> tile_a: array<{ty}, {tile_sq}>;
var<workgroup> tile_b: array<{ty}, {tile_sq}>;
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> b: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;
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
    var sum: {ty} = 0.0;
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
",
        ty = ty
    )
}

fn gen_matmul_register_tile(
    tr: u32,
    tc: u32,
    bm: u32,
    bn: u32,
    bk: u32,
    scalar: ScalarKind,
) -> String {
    let ty = scalar_ty(scalar);
    let wg_x = bn / tc;
    let wg_y = bm / tr;
    let smem_a = bm * bk;
    let smem_b = bk * bn;
    let reg_count = tr * tc;
    let total_threads = wg_x * wg_y;
    let a_loads = smem_a.div_ceil(total_threads);
    let b_loads = smem_b.div_ceil(total_threads);
    format!(
        r"
var<workgroup> smem_a: array<{ty}, {smem_a}>;
var<workgroup> smem_b: array<{ty}, {smem_b}>;
@group(0) @binding(0) var<storage, read> mat_a: array<{ty}>;
@group(0) @binding(1) var<storage, read> mat_b: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> mat_c: array<{ty}>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg_x}, {wg_y}, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {{
    let m = params[0];
    let k = params[1];
    let n = params[2];
    let grid_cols = params[3];
    let block_row = wgid.x / grid_cols;
    let block_col = wgid.x % grid_cols;
    let tx = lid.x;
    let ty = lid.y;
    var regs: array<{ty}, {reg_count}>;
    for (var i: u32 = 0u; i < {reg_count}u; i++) {{
        regs[i] = 0.0;
    }}
    let n_tiles = (k + {bk}u - 1u) / {bk}u;
    for (var t: u32 = 0u; t < n_tiles; t++) {{
        for (var ld: u32 = 0u; ld < {a_loads}u; ld++) {{
            let idx = li + ld * {total_threads}u;
            if idx < {smem_a}u {{
                let sr = idx / {bk}u;
                let sc = idx % {bk}u;
                let gr = block_row * {bm}u + sr;
                let gc = t * {bk}u + sc;
                if gr < m && gc < k {{
                    smem_a[idx] = mat_a[gr * k + gc];
                }} else {{
                    smem_a[idx] = 0.0;
                }}
            }}
        }}
        for (var ld: u32 = 0u; ld < {b_loads}u; ld++) {{
            let idx = li + ld * {total_threads}u;
            if idx < {smem_b}u {{
                let sr = idx / {bn}u;
                let sc = idx % {bn}u;
                let gr = t * {bk}u + sr;
                let gc = block_col * {bn}u + sc;
                if gr < k && gc < n {{
                    smem_b[idx] = mat_b[gr * n + gc];
                }} else {{
                    smem_b[idx] = 0.0;
                }}
            }}
        }}
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < {bk}u; kk++) {{
            for (var ri: u32 = 0u; ri < {tr}u; ri++) {{
                let a_val = smem_a[(ty * {tr}u + ri) * {bk}u + kk];
                for (var ci: u32 = 0u; ci < {tc}u; ci++) {{
                    regs[ri * {tc}u + ci] += a_val * smem_b[kk * {bn}u + tx * {tc}u + ci];
                }}
            }}
        }}
        workgroupBarrier();
    }}
    for (var ri: u32 = 0u; ri < {tr}u; ri++) {{
        let grow = block_row * {bm}u + ty * {tr}u + ri;
        if grow < m {{
            for (var ci: u32 = 0u; ci < {tc}u; ci++) {{
                let gcol = block_col * {bn}u + tx * {tc}u + ci;
                if gcol < n {{
                    mat_c[grow * n + gcol] = regs[ri * {tc}u + ci];
                }}
            }}
        }}
    }}
}}
",
        ty = ty
    )
}

pub fn select_register_tile_params(m: usize, n: usize, _k: usize) -> (u32, u32, u32, u32, u32) {
    let size = m.max(n);
    if size <= 64 {
        (2, 2, 16, 16, 8)
    } else if size <= 128 {
        (4, 4, 32, 32, 8)
    } else {
        (4, 4, 64, 64, 16)
    }
}

fn gen_fill_zeros(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params[0] {{ return; }}
    out[gid.x] = 0.0;
}}
",
        ty = ty
    )
}

fn gen_fill_scalar(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let val = scalar_from_f32_bits(scalar, "params[1]");
    format!(
        r"
@group(0) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params[0] {{ return; }}
    out[gid.x] = {val};
}}
",
        ty = ty,
        val = val
    )
}

fn gen_fill_identity(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read_write> out: array<{ty}>;
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
",
        ty = ty
    )
}

fn gen_reduce_sum(wg: u32) -> String {
    format!(
        r"
var<workgroup> wg_mem: array<f32, {wg}>;
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
    wg_mem[pos] = select(0.0, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {{
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < {wg}u; k++) {{ acc += wg_mem[k]; }}
        out[wgid.x] = acc;
    }}
}}
"
    )
}

fn gen_reduce_max(wg: u32) -> String {
    format!(
        r"
var<workgroup> wg_mem: array<f32, {wg}>;
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
    wg_mem[pos] = select(-3.4028235e+38, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {{
        var acc: f32 = wg_mem[0];
        for (var k: u32 = 1u; k < {wg}u; k++) {{
            if wg_mem[k] > acc {{ acc = wg_mem[k]; }}
        }}
        out[wgid.x] = acc;
    }}
}}
"
    )
}

fn gen_reduce_min(wg: u32) -> String {
    format!(
        r"
var<workgroup> wg_mem: array<f32, {wg}>;
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
    wg_mem[pos] = select(3.4028235e+38, input[i], i < n);
    workgroupBarrier();
    if pos == 0u {{
        var acc: f32 = wg_mem[0];
        for (var k: u32 = 1u; k < {wg}u; k++) {{
            if wg_mem[k] < acc {{ acc = wg_mem[k]; }}
        }}
        out[wgid.x] = acc;
    }}
}}
"
    )
}

fn gen_reduce_prod(wg: u32) -> String {
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
var<workgroup> scratch: array<f32, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let n = params[0];
    let i = gid.x;
    let li = lid.x;
    var v: f32 = 1.0;
    if i < n {{ v = a[i]; }}
    scratch[li] = v;
    workgroupBarrier();
    var stride = {wg}u / 2u;
    while (stride > 0u) {{
        if li < stride {{
            scratch[li] = scratch[li] * scratch[li + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    if li == 0u {{
        out[wg_id.x] = scratch[0];
    }}
}}
"
    )
}

fn gen_reduce_count_nonzero(wg: u32) -> String {
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
var<workgroup> scratch: array<f32, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let n = params[0];
    let i = gid.x;
    let li = lid.x;
    var v: f32 = 0.0;
    if i < n {{
        let x = a[i];
        v = select(0.0, 1.0, x != 0.0);
    }}
    scratch[li] = v;
    workgroupBarrier();
    var stride = {wg}u / 2u;
    while (stride > 0u) {{
        if li < stride {{
            scratch[li] = scratch[li] + scratch[li + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    if li == 0u {{
        out[wg_id.x] = scratch[0];
    }}
}}
"
    )
}

fn gen_argmax(wg: u32) -> String {
    format!(
        r"
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
"
    )
}

fn gen_argmin(wg: u32) -> String {
    format!(
        r"
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
"
    )
}

fn gen_activation_silu(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let x = a[i];
    out[i] = x / (1.0 + exp(-x));
}}
",
        ty = ty
    )
}

fn gen_activation_mish(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let x = a[i];
    let sp = log(1.0 + exp(x));
    out[i] = x * tanh(sp);
}}
",
        ty = ty
    )
}

fn gen_activation_leaky_relu(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let slope = scalar_from_f32_bits(scalar, "params[1]");
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let x = a[i];
    let slope = {slope};
    out[i] = select(slope * x, x, x >= 0.0);
}}
",
        ty = ty,
        slope = slope
    )
}

fn gen_activation_elu(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let alpha = scalar_from_f32_bits(scalar, "params[1]");
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let x = a[i];
    let alpha = {alpha};
    out[i] = select(alpha * (exp(x) - 1.0), x, x >= 0.0);
}}
",
        ty = ty,
        alpha = alpha
    )
}

fn gen_activation_hardswish(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let x = a[i];
    let v = clamp(x + 3.0, 0.0, 6.0);
    out[i] = x * v / 6.0;
}}
",
        ty = ty
    )
}

fn gen_activation_sigmoid(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let x = a[i];
    out[i] = 1.0 / (1.0 + exp(-x));
}}
",
        ty = ty
    )
}

fn gen_softmax(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let min_val = scalar_min_value(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
var<workgroup> sdata: array<{ty}, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let row = wg_id.x;
    let rows = params[0];
    let cols = params[1];
    if row >= rows {{ return; }}
    let tid = lid.x;
    let base = row * cols;

    // Pass 1: find max
    var m: {ty} = {min_val};
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        m = max(m, a[base + j]);
    }}
    sdata[tid] = m;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = max(sdata[tid], sdata[tid + s]); }}
        workgroupBarrier();
    }}
    let row_max = sdata[0];
    workgroupBarrier();

    // Pass 2: sum exp
    var sum_val: {ty} = 0.0;
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        sum_val = sum_val + exp(a[base + j] - row_max);
    }}
    sdata[tid] = sum_val;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = sdata[tid] + sdata[tid + s]; }}
        workgroupBarrier();
    }}
    let row_sum = sdata[0];
    workgroupBarrier();

    // Pass 3: write output
    let inv = 1.0 / row_sum;
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        out[base + j] = exp(a[base + j] - row_max) * inv;
    }}
}}
",
        ty = ty,
        min_val = min_val
    )
}

fn gen_layer_norm(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let eps = scalar_from_f32_bits(scalar, "params[2]");
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> gamma: array<{ty}>;
@group(0) @binding(2) var<storage, read> beta: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
var<workgroup> sdata: array<{ty}, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let row = wg_id.x;
    let cols = params[1];
    let eps = {eps};
    let tid = lid.x;
    let base = row * cols;

    // Mean
    var sum_val: {ty} = 0.0;
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        sum_val = sum_val + a[base + j];
    }}
    sdata[tid] = sum_val;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = sdata[tid] + sdata[tid + s]; }}
        workgroupBarrier();
    }}
    let mean = sdata[0] / {ty}(cols);
    workgroupBarrier();

    // Variance
    var var_val: {ty} = 0.0;
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        let d = a[base + j] - mean;
        var_val = var_val + d * d;
    }}
    sdata[tid] = var_val;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = sdata[tid] + sdata[tid + s]; }}
        workgroupBarrier();
    }}
    let inv_std = 1.0 / sqrt(sdata[0] / {ty}(cols) + eps);
    workgroupBarrier();

    // Normalize + affine
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        out[base + j] = (a[base + j] - mean) * inv_std * gamma[j] + beta[j];
    }}
}}
",
        ty = ty,
        eps = eps
    )
}

fn gen_rms_norm(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let eps = scalar_from_f32_bits(scalar, "params[2]");
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> gamma: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
var<workgroup> sdata: array<{ty}, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let row = wg_id.x;
    let cols = params[1];
    let eps = {eps};
    let tid = lid.x;
    let base = row * cols;

    // Sum of squares
    var sq_sum: {ty} = 0.0;
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        let v = a[base + j];
        sq_sum = sq_sum + v * v;
    }}
    sdata[tid] = sq_sum;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = sdata[tid] + sdata[tid + s]; }}
        workgroupBarrier();
    }}
    let inv_rms = 1.0 / sqrt(sdata[0] / {ty}(cols) + eps);
    workgroupBarrier();

    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        out[base + j] = a[base + j] * inv_rms * gamma[j];
    }}
}}
",
        ty = ty,
        eps = eps
    )
}

fn gen_sum_axis1(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
var<workgroup> sdata: array<{ty}, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let row = wg_id.x;
    let cols = params[1];
    let tid = lid.x;
    let base = row * cols;
    var acc: {ty} = 0.0;
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        acc = acc + a[base + j];
    }}
    sdata[tid] = acc;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = sdata[tid] + sdata[tid + s]; }}
        workgroupBarrier();
    }}
    if tid == 0u {{ out[row] = sdata[0]; }}
}}
",
        ty = ty
    )
}

fn gen_max_axis1(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let min_val = scalar_min_value(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
var<workgroup> sdata: array<{ty}, {wg}>;
@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {{
    let row = wg_id.x;
    let cols = params[1];
    let tid = lid.x;
    let base = row * cols;
    var acc: {ty} = {min_val};
    for (var j: u32 = tid; j < cols; j = j + {wg}u) {{
        acc = max(acc, a[base + j]);
    }}
    sdata[tid] = acc;
    workgroupBarrier();
    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{
        if tid < s {{ sdata[tid] = max(sdata[tid], sdata[tid + s]); }}
        workgroupBarrier();
    }}
    if tid == 0u {{ out[row] = sdata[0]; }}
}}
",
        ty = ty,
        min_val = min_val
    )
}

fn gen_embedding(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    format!(
        r"
@group(0) @binding(0) var<storage, read> indices: array<{ty}>;
@group(0) @binding(1) var<storage, read> weight: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let n_tokens = params[0];
    let embed_dim = params[1];
    let total = n_tokens * embed_dim;
    if tid >= total {{ return; }}
    let token = tid / embed_dim;
    let dim = tid % embed_dim;
    let idx = u32(indices[token]);
    out[tid] = weight[idx * embed_dim + dim];
}}
",
        ty = ty
    )
}

fn gen_axpy(wg: u32, scalar: ScalarKind) -> String {
    let ty = scalar_ty(scalar);
    let alpha = scalar_from_f32_bits(scalar, "params[1]");
    format!(
        r"
@group(0) @binding(0) var<storage, read_write> y: array<{ty}>;
@group(0) @binding(1) var<storage, read> x: array<{ty}>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params[0];
    if i >= n {{ return; }}
    let alpha = {alpha};
    y[i] = y[i] + alpha * x[i];
}}
",
        ty = ty,
        alpha = alpha
    )
}

pub(super) fn select_matmul_tile(m: usize, k: usize, n: usize) -> u32 {
    let max_dim = m.max(k).max(n);
    if max_dim < 48 {
        8
    } else if max_dim >= 192 {
        32
    } else {
        16
    }
}
