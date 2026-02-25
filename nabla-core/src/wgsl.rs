//! WGSL shader generators — pure string ops, always compiled (no wgpu dependency).

/// Generate WGSL software MMA shader: each thread computes TR x TC output tile.
/// Workgroup: (BN/TC) x (BM/TR) threads, each holding TR x TC registers.
pub fn gen_matmul_register_tile(
    tr: u32,
    tc: u32,
    bm: u32,
    bn: u32,
    bk: u32,
) -> String {
    let wg_x = bn / tc;
    let wg_y = bm / tr;
    let smem_a = bm * bk;
    let smem_b = bk * bn;
    let reg_count = tr * tc;
    let total_threads = wg_x * wg_y;
    let a_loads = smem_a.div_ceil(total_threads);
    let b_loads = smem_b.div_ceil(total_threads);
    format!(r"
var<workgroup> smem_a: array<f32, {smem_a}>;
var<workgroup> smem_b: array<f32, {smem_b}>;
@group(0) @binding(0) var<storage, read> mat_a: array<f32>;
@group(0) @binding(1) var<storage, read> mat_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mat_c: array<f32>;
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
    var regs: array<f32, {reg_count}>;
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
")
}

/// Select register tile params based on matrix dimensions.
/// Returns (tr, tc, bm, bn, bk).
pub fn select_register_tile_params(m: usize, n: usize, _k: usize) -> (u32, u32, u32, u32, u32) {
    match m.max(n) {
        0..=64 => (2, 2, 16, 16, 8),
        65..=128 => (4, 4, 32, 32, 8),
        _ => (4, 4, 64, 64, 16),
    }
}
