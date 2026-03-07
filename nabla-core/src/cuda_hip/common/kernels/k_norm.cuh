// === k_norm.cuh ===
// Softmax, layer_norm, rms_norm, group_norm, SDPA (FlashAttention-2)
// All use f32 accumulation with dtype conversion via LOAD/STORE macros

// ---- softmax ----

#define DEFINE_SOFTMAX_KERNEL(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME(const CTYPE* __restrict__ in, \
                                 CTYPE* __restrict__ out, \
                                 unsigned rows, unsigned cols) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    const CTYPE* x = in + row * cols; \
    CTYPE* y = out + row * cols; \
    unsigned tid = threadIdx.x; \
    float m = _NEG_INF_F32; \
    for (unsigned i = tid; i < cols; i += blockDim.x) \
        m = fmaxf(m, (float)LOAD(x[i])); \
    __shared__ float smax[32]; \
    BLOCK_REDUCE_BCAST_F32(m, warp_reduce_max_f32, _NEG_INF_F32, smax, tid); \
    float s = 0.0f; \
    for (unsigned i = tid; i < cols; i += blockDim.x) \
        s += __expf((float)LOAD(x[i]) - m); \
    __shared__ float ssum[32]; \
    BLOCK_REDUCE_BCAST_F32(s, warp_reduce_sum_f32, 0.0f, ssum, tid); \
    float inv_s = 1.0f / s; \
    for (unsigned i = tid; i < cols; i += blockDim.x) { \
        float v = __expf((float)LOAD(x[i]) - m) * inv_s; \
        y[i] = STORE(v); \
    } \
}

DEFINE_SOFTMAX_KERNEL(k_softmax_f32, float, _IDENTITY, _IDENTITY)
DEFINE_SOFTMAX_KERNEL(k_softmax_f16, __half, from_half, to_half)
DEFINE_SOFTMAX_KERNEL(k_softmax_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_SOFTMAX_KERNEL(k_softmax_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_SOFTMAX_KERNEL(k_softmax_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_SOFTMAX_KERNEL(k_softmax_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_softmax_f64(const double* __restrict__ in,
                                          double* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const double* x = in + row * cols;
    double* y = out + row * cols;
    unsigned tid = threadIdx.x;

    double m = _NEG_INF_F64;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmax(m, x[i]);
    __shared__ double smax[32];
    BLOCK_REDUCE_BCAST_F64(m, warp_reduce_max_f64, _NEG_INF_F64, smax, tid);

    double s = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += exp(x[i] - m);
    __shared__ double ssum[32];
    BLOCK_REDUCE_BCAST_F64(s, warp_reduce_sum_f64, 0.0, ssum, tid);

    double inv_s = 1.0 / s;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = exp(x[i] - m) * inv_s;
}

// ---- layer_norm ----

#define DEFINE_LAYER_NORM_KERNEL(NAME, CTYPE, EPS_TYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, \
    const CTYPE* __restrict__ gamma, \
    const CTYPE* __restrict__ beta, \
    CTYPE* __restrict__ out, \
    unsigned rows, unsigned cols, EPS_TYPE eps) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    const CTYPE* x = in + row * cols; \
    CTYPE* y = out + row * cols; \
    unsigned tid = threadIdx.x; \
    float sum = 0.0f; \
    for (unsigned i = tid; i < cols; i += blockDim.x) \
        sum += (float)LOAD(x[i]); \
    __shared__ float sdata[32]; \
    BLOCK_REDUCE_BCAST_F32(sum, warp_reduce_sum_f32, 0.0f, sdata, tid); \
    float mean = sum / (float)cols; \
    float var_sum = 0.0f; \
    for (unsigned i = tid; i < cols; i += blockDim.x) { \
        float d = (float)LOAD(x[i]) - mean; \
        var_sum += d * d; \
    } \
    BLOCK_REDUCE_BCAST_F32(var_sum, warp_reduce_sum_f32, 0.0f, sdata, tid); \
    float inv_std = 1.0f / sqrtf(var_sum / (float)cols + (float)eps); \
    for (unsigned i = tid; i < cols; i += blockDim.x) { \
        float xv = (float)LOAD(x[i]); \
        float gv = (float)LOAD(gamma[i]); \
        float bv = (float)LOAD(beta[i]); \
        y[i] = STORE((xv - mean) * inv_std * gv + bv); \
    } \
}

DEFINE_LAYER_NORM_KERNEL(k_layer_norm_f32, float, float, _IDENTITY, _IDENTITY)
DEFINE_LAYER_NORM_KERNEL(k_layer_norm_f16, __half, double, from_half, to_half)
DEFINE_LAYER_NORM_KERNEL(k_layer_norm_bf16, __nv_bfloat16, double, from_bf16, to_bf16)
DEFINE_LAYER_NORM_KERNEL(k_layer_norm_fp8e4m3, uint8_t, double, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_LAYER_NORM_KERNEL(k_layer_norm_fp8e5m2, uint8_t, double, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_LAYER_NORM_KERNEL(k_layer_norm_fp4e2m1, uint8_t, double, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_layer_norm_f64(
    const double* __restrict__ in,
    const double* __restrict__ gamma,
    const double* __restrict__ beta,
    double* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const double* x = in + row * cols;
    double* y = out + row * cols;
    unsigned tid = threadIdx.x;

    double sum = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x) sum += x[i];
    __shared__ double sdata[32];
    BLOCK_REDUCE_BCAST_F64(sum, warp_reduce_sum_f64, 0.0, sdata, tid);
    double mean = sum / (double)cols;

    double var_sum = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        double d = x[i] - mean;
        var_sum += d * d;
    }
    BLOCK_REDUCE_BCAST_F64(var_sum, warp_reduce_sum_f64, 0.0, sdata, tid);
    double inv_std = 1.0 / sqrt(var_sum / (double)cols + eps);

    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

// ---- rms_norm ----

#define DEFINE_RMS_NORM_KERNEL(NAME, CTYPE, EPS_TYPE, LOAD, STORE, RSQRT_FN) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, \
    const CTYPE* __restrict__ gamma, \
    CTYPE* __restrict__ out, \
    unsigned rows, unsigned cols, EPS_TYPE eps) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    const CTYPE* x = in + row * cols; \
    CTYPE* y = out + row * cols; \
    unsigned tid = threadIdx.x; \
    float sq_sum = 0.0f; \
    for (unsigned i = tid; i < cols; i += blockDim.x) { \
        float v = (float)LOAD(x[i]); \
        sq_sum += v * v; \
    } \
    __shared__ float sdata[32]; \
    BLOCK_REDUCE_BCAST_F32(sq_sum, warp_reduce_sum_f32, 0.0f, sdata, tid); \
    float inv_rms = RSQRT_FN(sq_sum / (float)cols + (float)eps); \
    for (unsigned i = tid; i < cols; i += blockDim.x) { \
        float xv = (float)LOAD(x[i]); \
        float gv = (float)LOAD(gamma[i]); \
        y[i] = STORE(xv * inv_rms * gv); \
    } \
}

DEFINE_RMS_NORM_KERNEL(k_rms_norm_f32, float, float, _IDENTITY, _IDENTITY, _INV_SQRT_F32)
DEFINE_RMS_NORM_KERNEL(k_rms_norm_f16, __half, double, from_half, to_half, _RSQRT_F32)
DEFINE_RMS_NORM_KERNEL(k_rms_norm_bf16, __nv_bfloat16, double, from_bf16, to_bf16, _RSQRT_F32)
DEFINE_RMS_NORM_KERNEL(k_rms_norm_fp8e4m3, uint8_t, double, fp8e4m3_to_f32, fp8e4m3_from_f32, _RSQRT_F32)
DEFINE_RMS_NORM_KERNEL(k_rms_norm_fp8e5m2, uint8_t, double, fp8e5m2_to_f32, fp8e5m2_from_f32, _RSQRT_F32)
DEFINE_RMS_NORM_KERNEL(k_rms_norm_fp4e2m1, uint8_t, double, fp4e2m1_to_f32, fp4e2m1_from_f32, _RSQRT_F32)

extern "C" __global__ void k_rms_norm_f64(
    const double* __restrict__ in,
    const double* __restrict__ gamma,
    double* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const double* x = in + row * cols;
    double* y = out + row * cols;
    unsigned tid = threadIdx.x;

    double sq_sum = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sq_sum += x[i] * x[i];
    __shared__ double sdata[32];
    BLOCK_REDUCE_BCAST_F64(sq_sum, warp_reduce_sum_f64, 0.0, sdata, tid);
    double inv_rms = 1.0 / sqrt(sq_sum / (double)cols + eps);

    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = x[i] * inv_rms * gamma[i];
}

// ---- group_norm ----

#define DEFINE_GROUP_NORM_KERNEL(NAME, CTYPE, EPS_TYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, \
    const CTYPE* __restrict__ gamma, \
    const CTYPE* __restrict__ beta, \
    CTYPE* __restrict__ out, \
    unsigned rows, unsigned cols, unsigned groups, EPS_TYPE eps) { \
    unsigned gid = blockIdx.x; \
    unsigned row = gid / groups; \
    if (row >= rows) return; \
    unsigned g = gid % groups; \
    unsigned g_size = cols / groups; \
    unsigned g_start = g * g_size; \
    const CTYPE* x = in + row * cols + g_start; \
    CTYPE* y = out + row * cols + g_start; \
    unsigned tid = threadIdx.x; \
    float sum = 0.0f; \
    for (unsigned i = tid; i < g_size; i += blockDim.x) \
        sum += (float)LOAD(x[i]); \
    __shared__ float sdata[32]; \
    BLOCK_REDUCE_BCAST_F32(sum, warp_reduce_sum_f32, 0.0f, sdata, tid); \
    float mean = sum / (float)g_size; \
    float var_sum = 0.0f; \
    for (unsigned i = tid; i < g_size; i += blockDim.x) { \
        float d = (float)LOAD(x[i]) - mean; \
        var_sum += d * d; \
    } \
    BLOCK_REDUCE_BCAST_F32(var_sum, warp_reduce_sum_f32, 0.0f, sdata, tid); \
    float inv_std = rsqrtf(var_sum / (float)g_size + (float)eps); \
    for (unsigned i = tid; i < g_size; i += blockDim.x) { \
        unsigned ci = g_start + i; \
        float xv = (float)LOAD(x[i]); \
        float gv = (float)LOAD(gamma[ci]); \
        float bv = (float)LOAD(beta[ci]); \
        y[i] = STORE((xv - mean) * inv_std * gv + bv); \
    } \
}

DEFINE_GROUP_NORM_KERNEL(k_group_norm_f32, float, float, _IDENTITY, _IDENTITY)
DEFINE_GROUP_NORM_KERNEL(k_group_norm_f16, __half, double, from_half, to_half)
DEFINE_GROUP_NORM_KERNEL(k_group_norm_bf16, __nv_bfloat16, double, from_bf16, to_bf16)
DEFINE_GROUP_NORM_KERNEL(k_group_norm_fp8e4m3, uint8_t, double, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_GROUP_NORM_KERNEL(k_group_norm_fp8e5m2, uint8_t, double, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_GROUP_NORM_KERNEL(k_group_norm_fp4e2m1, uint8_t, double, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_group_norm_f64(
    const double* __restrict__ in,
    const double* __restrict__ gamma,
    const double* __restrict__ beta,
    double* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, double eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const double* x = in + row * cols + g_start;
    double* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    double sum = 0.0;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
        sum += x[i];
    __shared__ double sdata[32];
    BLOCK_REDUCE_BCAST_F64(sum, warp_reduce_sum_f64, 0.0, sdata, tid);
    double mean = sum / (double)g_size;

    double var_sum = 0.0;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        double d = x[i] - mean;
        var_sum += d * d;
    }
    BLOCK_REDUCE_BCAST_F64(var_sum, warp_reduce_sum_f64, 0.0, sdata, tid);
    double inv_std = 1.0 / sqrt(var_sum / (double)g_size + eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        y[i] = (x[i] - mean) * inv_std * gamma[ci] + beta[ci];
    }
}

// === Attention ===
// FlashAttention-2 SDPA kernel (online softmax, O(seq_len) HBM)

#define FA2_SCORE_AND_UPDATE(FTYPE, D, smem_k, smem_v, qi, oi, mi, li, k_start, seq_k, scale, EXPF, NEGINF) \
    do { \
        FTYPE sij[FA_BLOCK_N]; \
        for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) { \
            unsigned k_row = k_start + kj; \
            FTYPE dot = 0; \
            if (k_row < seq_k) { \
                for (unsigned d = 0; d < D; d++) \
                    dot += qi[d] * smem_k[kj * D + d]; \
                sij[kj] = dot * scale; \
            } else { \
                sij[kj] = NEGINF; \
            } \
        } \
        FTYPE mij = sij[0]; \
        for (unsigned kj = 1; kj < FA_BLOCK_N; kj++) \
            if (sij[kj] > mij) mij = sij[kj]; \
        FTYPE pij[FA_BLOCK_N]; \
        FTYPE lij = 0; \
        for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) { \
            pij[kj] = EXPF(sij[kj] - mij); \
            lij += pij[kj]; \
        } \
        FTYPE mi_new     = (mi > mij) ? mi : mij; \
        FTYPE scale_old  = EXPF(mi - mi_new); \
        FTYPE scale_new  = EXPF(mij - mi_new); \
        for (unsigned d = 0; d < D; d++) { \
            FTYPE pv = 0; \
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) \
                pv += pij[kj] * smem_v[kj * D + d]; \
            oi[d] = scale_old * oi[d] + scale_new * pv; \
        } \
        li = scale_old * li + scale_new * lij; \
        mi = mi_new; \
    } while (0)

#define DEFINE_SDPA_F32_KERNEL(NAME, CTYPE, SMEM_TYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ Q, \
    const CTYPE* __restrict__ K, \
    const CTYPE* __restrict__ V, \
    CTYPE* __restrict__ Out, \
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH, \
    float scale \
) { \
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M; \
    unsigned bh      = blockIdx.x / num_q_tiles; \
    unsigned qi_tile = blockIdx.x % num_q_tiles; \
    unsigned tid     = threadIdx.x; \
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid; \
    extern __shared__ SMEM_TYPE smem_sdpa[]; \
    SMEM_TYPE* smem_k = smem_sdpa; \
    SMEM_TYPE* smem_v = smem_k + FA_BLOCK_N * D; \
    float qi[FA_HEAD_DIM_MAX]; \
    float oi[FA_HEAD_DIM_MAX]; \
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0f; oi[d] = 0.0f; } \
    float mi = _NEG_INF_F32; \
    float li = 0.0f; \
    if (qi_row < seq_q) { \
        unsigned base = bh * seq_q * D + qi_row * D; \
        for (unsigned d = 0; d < D; d++) qi[d] = LOAD(Q[base + d]); \
    } \
    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N; \
    for (unsigned ki = 0; ki < num_k_tiles; ki++) { \
        unsigned k_start = ki * FA_BLOCK_N; \
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) { \
            unsigned kj = j / D, d = j % D; \
            unsigned k_row = k_start + kj; \
            smem_k[j] = (k_row < seq_k && bh < BH) \
                ? (SMEM_TYPE)LOAD(K[bh * seq_k * D + k_row * D + d]) : (SMEM_TYPE)0; \
        } \
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) { \
            unsigned kj = j / D, d = j % D; \
            unsigned k_row = k_start + kj; \
            smem_v[j] = (k_row < seq_k && bh < BH) \
                ? (SMEM_TYPE)LOAD(V[bh * seq_k * D + k_row * D + d]) : (SMEM_TYPE)0; \
        } \
        __syncthreads(); \
        if (qi_row < seq_q) { \
            FA2_SCORE_AND_UPDATE(float, D, smem_k, smem_v, qi, oi, mi, li, k_start, seq_k, scale, __expf, _NEG_INF_F32); \
        } \
        __syncthreads(); \
    } \
    if (qi_row < seq_q) { \
        float inv_li = (li > 0.0f) ? 1.0f / li : 0.0f; \
        unsigned base = bh * seq_q * D + qi_row * D; \
        for (unsigned d = 0; d < D; d++) \
            Out[base + d] = STORE(oi[d] * inv_li); \
    } \
}

DEFINE_SDPA_F32_KERNEL(k_sdpa_f32, float, float, _IDENTITY, _IDENTITY)
DEFINE_SDPA_F32_KERNEL(k_sdpa_f16, __half, float, from_half, to_half)
DEFINE_SDPA_F32_KERNEL(k_sdpa_bf16, __nv_bfloat16, float, from_bf16, to_bf16)
DEFINE_SDPA_F32_KERNEL(k_sdpa_fp8e4m3, uint8_t, float, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_SDPA_F32_KERNEL(k_sdpa_fp8e5m2, uint8_t, float, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_SDPA_F32_KERNEL(k_sdpa_fp4e2m1, uint8_t, float, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_sdpa_f64(
    const double* __restrict__ Q,
    const double* __restrict__ K,
    const double* __restrict__ V,
    double* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    double scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ double smem_f64[];
    double* smem_k = smem_f64;
    double* smem_v = smem_k + FA_BLOCK_N * D;

    double qi[FA_HEAD_DIM_MAX];
    double oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0; oi[d] = 0.0; }
    double mi = _NEG_INF_F64;
    double li = 0.0;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = Q[base + d];
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            smem_k[j] = (k_row < seq_k && bh < BH)
                ? K[bh * seq_k * D + k_row * D + d] : 0.0;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            smem_v[j] = (k_row < seq_k && bh < BH)
                ? V[bh * seq_k * D + k_row * D + d] : 0.0;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            FA2_SCORE_AND_UPDATE(double, D, smem_k, smem_v, qi, oi, mi, li, k_start, seq_k, scale, exp, _NEG_INF_F64);
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        double inv_li = (li > 0.0) ? 1.0 / li : 0.0;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = oi[d] * inv_li;
    }
}
