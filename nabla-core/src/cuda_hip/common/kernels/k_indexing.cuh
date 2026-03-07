// k_indexing.cuh -- Indexing/slicing kernels (submatrix, slice_set, gather, scatter, index_select, scatter_add)
// Depends on k_defs.cuh (THREAD_ID, type helpers)

// ============================================================
// === Macro-based kernel generators for f32/f64 ===
// ============================================================

#define SUBMATRIX_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_submatrix_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, \
    unsigned src_rows, unsigned src_cols, unsigned row_start, unsigned col_start, \
    unsigned out_rows, unsigned out_cols) { \
    unsigned i = THREAD_ID; \
    unsigned n = out_rows * out_cols; \
    if (i >= n) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    unsigned src_r = row_start + r; \
    unsigned src_c = col_start + c; \
    if (src_r >= src_rows || src_c >= src_cols) return; \
    out[i] = a[src_r * src_cols + src_c]; \
}

#define SLICE_SET_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_slice_set_##suffix( \
    const T* __restrict__ src, T* __restrict__ dst, \
    unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols, \
    unsigned row_start, unsigned col_start) { \
    unsigned i = THREAD_ID; \
    unsigned n = src_rows * src_cols; \
    if (i >= n) return; \
    unsigned r = i / src_cols; \
    unsigned c = i - r * src_cols; \
    unsigned dr = row_start + r; \
    unsigned dc = col_start + c; \
    if (dr >= dst_rows || dc >= dst_cols) return; \
    dst[dr * dst_cols + dc] = src[i]; \
}

#define GATHER_ROWS_U32IDX_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_gather_rows_u32idx_##suffix( \
    const T* __restrict__ a, const unsigned* __restrict__ idx, T* __restrict__ out, \
    unsigned out_rows, unsigned out_cols, unsigned src_rows, unsigned src_cols) { \
    unsigned i = THREAD_ID; \
    unsigned total = out_rows * out_cols; \
    if (i >= total) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    unsigned src_r = idx[r]; \
    if (src_r >= src_rows || c >= src_cols) return; \
    out[i] = a[src_r * src_cols + c]; \
}

#define GATHER_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_gather_##suffix( \
    const T* __restrict__ a, const T* __restrict__ idx, T* __restrict__ out, \
    unsigned out_rows, unsigned out_cols, unsigned in_rows, unsigned in_cols, unsigned axis) { \
    unsigned i = THREAD_ID; \
    unsigned total = out_rows * out_cols; \
    if (i >= total) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    unsigned index_val = (unsigned)idx[i]; \
    if (axis == 0) { \
        if (index_val >= in_rows || c >= in_cols) return; \
        out[i] = a[index_val * in_cols + c]; \
    } else { \
        if (r >= in_rows || index_val >= in_cols) return; \
        out[i] = a[r * in_cols + index_val]; \
    } \
}

#define SCATTER_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_scatter_##suffix( \
    const T* __restrict__ src, const T* __restrict__ idx, T* __restrict__ out, \
    unsigned src_rows, unsigned src_cols, unsigned out_rows, unsigned out_cols, unsigned axis) { \
    unsigned i = THREAD_ID; \
    unsigned total = src_rows * src_cols; \
    if (i >= total) return; \
    unsigned r = i / src_cols; \
    unsigned c = i - r * src_cols; \
    unsigned index_val = (unsigned)idx[i]; \
    if (axis == 0) { \
        if (index_val >= out_rows || c >= out_cols) return; \
        out[index_val * out_cols + c] = src[i]; \
    } else { \
        if (r >= out_rows || index_val >= out_cols) return; \
        out[r * out_cols + index_val] = src[i]; \
    } \
}

#define INDEX_SELECT_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_index_select_##suffix( \
    const T* __restrict__ a, const T* __restrict__ idx, T* __restrict__ out, \
    unsigned out_rows, unsigned out_cols, unsigned in_rows, unsigned in_cols, \
    unsigned axis, unsigned k) { \
    unsigned i = THREAD_ID; \
    unsigned total = out_rows * out_cols; \
    if (i >= total) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    if (axis == 0) { \
        unsigned src_r = (unsigned)idx[r]; \
        if (src_r >= in_rows || c >= in_cols) return; \
        out[i] = a[src_r * in_cols + c]; \
    } else { \
        unsigned src_c = (unsigned)idx[c]; \
        if (r >= in_rows || src_c >= in_cols) return; \
        out[i] = a[r * in_cols + src_c]; \
    } \
}

#define SCATTER_ADD_DIM0_U32IDX_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_scatter_add_dim0_u32idx_##suffix( \
    const T* __restrict__ src, const unsigned* __restrict__ idx, T* __restrict__ dst, \
    unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) { \
    unsigned i = THREAD_ID; \
    unsigned total = src_rows * src_cols; \
    if (i >= total) return; \
    unsigned r = i / src_cols; \
    unsigned c = i - r * src_cols; \
    unsigned dst_r = idx[r]; \
    if (dst_r >= dst_rows || c >= dst_cols) return; \
    atomicAdd(&dst[dst_r * dst_cols + c], src[i]); \
}

#define SORT_ROWS_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_sort_rows_##suffix( \
    const T* __restrict__ in, T* __restrict__ out, T* __restrict__ idx, \
    unsigned rows, unsigned cols, unsigned desc) { \
    unsigned row = THREAD_ID; \
    if (row >= rows) return; \
    unsigned base = row * cols; \
    for (unsigned c = 0; c < cols; ++c) { \
        out[base + c] = in[base + c]; \
        idx[base + c] = (T)c; \
    } \
    for (unsigned i = 0; i < cols; ++i) { \
        for (unsigned j = 0; j + 1 < cols - i; ++j) { \
            unsigned a_idx = base + j; \
            unsigned b_idx = base + j + 1; \
            T va = out[a_idx]; \
            T vb = out[b_idx]; \
            int cmp = desc ? (va < vb) : (va > vb); \
            if (cmp) { \
                out[a_idx] = vb; \
                out[b_idx] = va; \
                T ia = idx[a_idx]; \
                idx[a_idx] = idx[b_idx]; \
                idx[b_idx] = ia; \
            } \
        } \
    } \
}

#define SCATTER_ADD_DIM1_U32IDX_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_scatter_add_dim1_u32idx_##suffix( \
    const T* __restrict__ src, const unsigned* __restrict__ idx, T* __restrict__ dst, \
    unsigned src_rows, unsigned src_cols, unsigned dst_cols) { \
    unsigned i = THREAD_ID; \
    unsigned total = src_rows * src_cols; \
    if (i >= total) return; \
    unsigned r = i / src_cols; \
    unsigned c = i - r * src_cols; \
    unsigned dst_c = idx[c]; \
    if (dst_c >= dst_cols) return; \
    atomicAdd(&dst[r * dst_cols + dst_c], src[i]); \
}

// ============================================================
// === Instantiate for f32 and f64 ===
// ============================================================

SUBMATRIX_KERNEL(f32, float)
SUBMATRIX_KERNEL(f64, double)

SLICE_SET_KERNEL(f32, float)
SLICE_SET_KERNEL(f64, double)

GATHER_ROWS_U32IDX_KERNEL(f32, float)
GATHER_ROWS_U32IDX_KERNEL(f64, double)

GATHER_KERNEL(f32, float)
GATHER_KERNEL(f64, double)

SCATTER_KERNEL(f32, float)
SCATTER_KERNEL(f64, double)

INDEX_SELECT_KERNEL(f32, float)
INDEX_SELECT_KERNEL(f64, double)

SCATTER_ADD_DIM0_U32IDX_KERNEL(f32, float)
SCATTER_ADD_DIM0_U32IDX_KERNEL(f64, double)

SCATTER_ADD_DIM1_U32IDX_KERNEL(f32, float)
SCATTER_ADD_DIM1_U32IDX_KERNEL(f64, double)

SORT_ROWS_KERNEL(f32, float)
SORT_ROWS_KERNEL(f64, double)

// Cleanup indexing-local macros
#undef SUBMATRIX_KERNEL
#undef SLICE_SET_KERNEL
#undef GATHER_ROWS_U32IDX_KERNEL
#undef GATHER_KERNEL
#undef SCATTER_KERNEL
#undef INDEX_SELECT_KERNEL
#undef SCATTER_ADD_DIM0_U32IDX_KERNEL
#undef SCATTER_ADD_DIM1_U32IDX_KERNEL
#undef SORT_ROWS_KERNEL
