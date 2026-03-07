// === k_pool.cuh ===
// Pooling: max/avg/adaptive pool2d, prod_partial
// === k_reduce_axis.cuh ===
// Axis reductions: sum/max axis1, embedding, cumsum/cumprod

// Pool2d position decode: (pos, outH, outW) -> (ow, oh, n)
#define POOL2D_DECODE(pos, outH, outW, ow, oh, n) \
    unsigned ow = (pos) % (outW); \
    unsigned oh = ((pos) / (outW)) % (outH); \
    unsigned n  = (pos) / ((outH) * (outW))

// Pool2d kernel window loop with bounds check
#define POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) \
    for (unsigned kh = 0; kh < kH; kh++) \
        for (unsigned kw = 0; kw < kW; kw++) { \
            int ih = (int)((oh) * (sH) + kh) - (int)(pH); \
            int iw = (int)((ow) * (sW) + kw) - (int)(pW); \
            if (ih >= 0 && ih < (int)(H) && iw >= 0 && iw < (int)(W))

// ---- prod_partial: shared-memory tree reduction for product ----

#define DEFINE_PROD_PARTIAL(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, CTYPE* __restrict__ partial, unsigned N \
) { \
    __shared__ float smem[256]; \
    unsigned tid = threadIdx.x; \
    unsigned idx = blockIdx.x * blockDim.x + tid; \
    smem[tid] = (idx < N) ? (float)LOAD(in[idx]) : 1.0f; \
    __syncthreads(); \
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) { \
        if (tid < s) smem[tid] *= smem[tid + s]; \
        __syncthreads(); \
    } \
    if (tid == 0) partial[blockIdx.x] = STORE(smem[0]); \
}

DEFINE_PROD_PARTIAL(k_prod_partial_f32, float, _IDENTITY, _IDENTITY)
DEFINE_PROD_PARTIAL(k_prod_partial_f16, __half, from_half, to_half)
DEFINE_PROD_PARTIAL(k_prod_partial_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_PROD_PARTIAL(k_prod_partial_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_PROD_PARTIAL(k_prod_partial_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_PROD_PARTIAL(k_prod_partial_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_prod_partial_f64(
    const double* __restrict__ in, double* __restrict__ partial, unsigned N
) {
    __shared__ double smem[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? in[idx] : 1.0;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] *= smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem[0];
}

// ---- max_pool2d_with_idx ----

#define DEFINE_MAX_POOL2D_WITH_IDX(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, CTYPE* __restrict__ out, CTYPE* __restrict__ idx_out, \
    unsigned H, unsigned W, unsigned kH, unsigned kW, \
    unsigned sH, unsigned sW, unsigned pH, unsigned pW, \
    unsigned outH, unsigned outW, unsigned NC \
) { \
    unsigned pos = THREAD_ID; \
    unsigned total = NC * outH * outW; \
    if (pos >= total) return; \
    POOL2D_DECODE(pos, outH, outW, ow, oh, n); \
    float max_val = -3.402823466e+38f; \
    unsigned best_idx = 0; \
    POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) { \
                unsigned flat = n * H * W + (unsigned)ih * W + (unsigned)iw; \
                float v = (float)LOAD(in[flat]); \
                if (v > max_val) { max_val = v; best_idx = flat; } \
            } \
        } \
    out[pos] = STORE(max_val); \
    idx_out[pos] = STORE((float)best_idx); \
}

DEFINE_MAX_POOL2D_WITH_IDX(k_max_pool2d_with_idx_f32, float, _IDENTITY, _IDENTITY)
DEFINE_MAX_POOL2D_WITH_IDX(k_max_pool2d_with_idx_f16, __half, from_half, to_half)
DEFINE_MAX_POOL2D_WITH_IDX(k_max_pool2d_with_idx_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_MAX_POOL2D_WITH_IDX(k_max_pool2d_with_idx_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_MAX_POOL2D_WITH_IDX(k_max_pool2d_with_idx_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_MAX_POOL2D_WITH_IDX(k_max_pool2d_with_idx_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_max_pool2d_with_idx_f64(
    const double* __restrict__ in, double* __restrict__ out, double* __restrict__ idx_out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned pos = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (pos >= total) return;
    POOL2D_DECODE(pos, outH, outW, ow, oh, n);
    double max_val = -1.7976931348623157e+308;
    unsigned best_idx = 0;
    POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) {
                unsigned flat = n * H * W + (unsigned)ih * W + (unsigned)iw;
                double v = in[flat];
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    out[pos] = max_val;
    idx_out[pos] = (double)best_idx;
}

// ---- max_pool2d (no idx) ----

#define DEFINE_MAX_POOL2D(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, CTYPE* __restrict__ out, \
    unsigned H, unsigned W, unsigned kH, unsigned kW, \
    unsigned sH, unsigned sW, unsigned pH, unsigned pW, \
    unsigned outH, unsigned outW, unsigned NC \
) { \
    unsigned pos = THREAD_ID; \
    unsigned total = NC * outH * outW; \
    if (pos >= total) return; \
    POOL2D_DECODE(pos, outH, outW, ow, oh, n); \
    float max_val = -3.402823466e+38f; \
    POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) { \
                float v = (float)LOAD(in[n * H * W + ih * W + iw]); \
                if (v > max_val) max_val = v; \
            } \
        } \
    out[pos] = STORE(max_val); \
}

DEFINE_MAX_POOL2D(k_max_pool2d_f32, float, _IDENTITY, _IDENTITY)
DEFINE_MAX_POOL2D(k_max_pool2d_f16, __half, from_half, to_half)
DEFINE_MAX_POOL2D(k_max_pool2d_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_MAX_POOL2D(k_max_pool2d_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_MAX_POOL2D(k_max_pool2d_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_MAX_POOL2D(k_max_pool2d_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_max_pool2d_f64(
    const double* __restrict__ in, double* __restrict__ out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned pos = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (pos >= total) return;
    POOL2D_DECODE(pos, outH, outW, ow, oh, n);
    double max_val = -1.7976931348623157e+308;
    POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) {
                double v = in[n * H * W + ih * W + iw];
                if (v > max_val) max_val = v;
            }
        }
    out[pos] = max_val;
}

// ---- avg_pool2d ----

#define DEFINE_AVG_POOL2D(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, CTYPE* __restrict__ out, \
    unsigned H, unsigned W, unsigned kH, unsigned kW, \
    unsigned sH, unsigned sW, unsigned pH, unsigned pW, \
    unsigned outH, unsigned outW, unsigned NC \
) { \
    unsigned pos = THREAD_ID; \
    unsigned total = NC * outH * outW; \
    if (pos >= total) return; \
    POOL2D_DECODE(pos, outH, outW, ow, oh, n); \
    float sum = 0.0f; unsigned cnt = 0; \
    POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) { \
                sum += (float)LOAD(in[n * H * W + ih * W + iw]); cnt++; \
            } \
        } \
    out[pos] = STORE(cnt > 0 ? sum / (float)cnt : 0.0f); \
}

DEFINE_AVG_POOL2D(k_avg_pool2d_f32, float, _IDENTITY, _IDENTITY)
DEFINE_AVG_POOL2D(k_avg_pool2d_f16, __half, from_half, to_half)
DEFINE_AVG_POOL2D(k_avg_pool2d_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_AVG_POOL2D(k_avg_pool2d_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_AVG_POOL2D(k_avg_pool2d_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_AVG_POOL2D(k_avg_pool2d_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_avg_pool2d_f64(
    const double* __restrict__ in, double* __restrict__ out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned pos = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (pos >= total) return;
    POOL2D_DECODE(pos, outH, outW, ow, oh, n);
    double sum = 0.0; unsigned cnt = 0;
    POOL2D_FOR_WINDOW(kH, kW, oh, ow, sH, sW, pH, pW, H, W) {
                sum += in[n * H * W + ih * W + iw]; cnt++;
            }
        }
    out[pos] = cnt > 0 ? sum / (double)cnt : 0.0;
}

// ---- adaptive_avg_pool2d ----

#define DEFINE_ADAPTIVE_AVG_POOL2D(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ in, CTYPE* __restrict__ out, \
    unsigned inH, unsigned inW, unsigned outH, unsigned outW, unsigned NC \
) { \
    unsigned pos = THREAD_ID; \
    unsigned total = NC * outH * outW; \
    if (pos >= total) return; \
    POOL2D_DECODE(pos, outH, outW, ow, oh, n); \
    unsigned ih_start = oh * inH / outH; \
    unsigned ih_end = (oh + 1) * inH / outH; if (ih_end <= ih_start) ih_end = ih_start + 1; \
    unsigned iw_start = ow * inW / outW; \
    unsigned iw_end = (ow + 1) * inW / outW; if (iw_end <= iw_start) iw_end = iw_start + 1; \
    float sum = 0.0f; unsigned cnt = 0; \
    for (unsigned ih = ih_start; ih < ih_end && ih < inH; ih++) { \
        for (unsigned iw = iw_start; iw < iw_end && iw < inW; iw++) { \
            sum += (float)LOAD(in[n * inH * inW + ih * inW + iw]); cnt++; \
        } \
    } \
    out[pos] = STORE(cnt > 0 ? sum / (float)cnt : 0.0f); \
}

DEFINE_ADAPTIVE_AVG_POOL2D(k_adaptive_avg_pool2d_f32, float, _IDENTITY, _IDENTITY)
DEFINE_ADAPTIVE_AVG_POOL2D(k_adaptive_avg_pool2d_f16, __half, from_half, to_half)
DEFINE_ADAPTIVE_AVG_POOL2D(k_adaptive_avg_pool2d_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_ADAPTIVE_AVG_POOL2D(k_adaptive_avg_pool2d_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_ADAPTIVE_AVG_POOL2D(k_adaptive_avg_pool2d_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_ADAPTIVE_AVG_POOL2D(k_adaptive_avg_pool2d_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_adaptive_avg_pool2d_f64(
    const double* __restrict__ in, double* __restrict__ out,
    unsigned inH, unsigned inW, unsigned outH, unsigned outW, unsigned NC
) {
    unsigned pos = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (pos >= total) return;
    POOL2D_DECODE(pos, outH, outW, ow, oh, n);
    unsigned ih_start = oh * inH / outH;
    unsigned ih_end = (oh + 1) * inH / outH; if (ih_end <= ih_start) ih_end = ih_start + 1;
    unsigned iw_start = ow * inW / outW;
    unsigned iw_end = (ow + 1) * inW / outW; if (iw_end <= iw_start) iw_end = iw_start + 1;
    double sum = 0.0; unsigned cnt = 0;
    for (unsigned ih = ih_start; ih < ih_end && ih < inH; ih++) {
        for (unsigned iw = iw_start; iw < iw_end && iw < inW; iw++) {
            sum += in[n * inH * inW + ih * inW + iw]; cnt++;
        }
    }
    out[pos] = cnt > 0 ? sum / (double)cnt : 0.0;
}

// ---- sum_axis1: row-wise sum with warp+block reduce ----

#define DEFINE_SUM_AXIS1(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME(const CTYPE* __restrict__ in, \
                                CTYPE* __restrict__ out, \
                                unsigned rows, unsigned cols) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    unsigned tid = threadIdx.x; \
    float acc = 0.0f; \
    for (unsigned i = tid; i < cols; i += blockDim.x) \
        acc += (float)LOAD(in[row * cols + i]); \
    __shared__ float sdata[32]; \
    BLOCK_REDUCE_F32(acc, warp_reduce_sum_f32, 0.0f, sdata, tid); \
    if (tid == 0) out[row] = STORE(acc); \
}

// f32 kept separate: float4 vectorized path
extern "C" __global__ void k_sum_axis1_f32(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = 0.0f;
    const float* row_ptr = in + row * cols;
    if ((cols >= 4) && (((unsigned long long)row_ptr & 15ull) == 0) && ((cols & 3u) == 0)) {
        const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
        unsigned n4 = cols >> 2;
        #pragma unroll 4
        for (unsigned i = tid; i < n4; i += blockDim.x) {
            float4 v = row4[i];
            acc += v.x + v.y + v.z + v.w;
        }
    } else {
        for (unsigned i = tid; i < cols; i += blockDim.x)
            acc += row_ptr[i];
    }
    __shared__ float sdata[32];
    BLOCK_REDUCE_F32(acc, warp_reduce_sum_f32, 0.0f, sdata, tid);
    if (tid == 0) out[row] = acc;
}

DEFINE_SUM_AXIS1(k_sum_axis1_f16, __half, from_half, to_half)
DEFINE_SUM_AXIS1(k_sum_axis1_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_SUM_AXIS1(k_sum_axis1_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_SUM_AXIS1(k_sum_axis1_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_SUM_AXIS1(k_sum_axis1_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_sum_axis1_f64(const double* __restrict__ in,
                                            double* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    double acc = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += in[row * cols + i];
    __shared__ double sdata[32];
    BLOCK_REDUCE_F64(acc, warp_reduce_sum_f64, 0.0, sdata, tid);
    if (tid == 0) out[row] = acc;
}

// ---- max_axis1: row-wise max with warp+block reduce ----

#define DEFINE_MAX_AXIS1(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME(const CTYPE* __restrict__ in, \
                                CTYPE* __restrict__ out, \
                                unsigned rows, unsigned cols) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    unsigned tid = threadIdx.x; \
    float acc = _NEG_INF_F32; \
    for (unsigned i = tid; i < cols; i += blockDim.x) \
        acc = fmaxf(acc, (float)LOAD(in[row * cols + i])); \
    __shared__ float sdata[32]; \
    BLOCK_REDUCE_F32(acc, warp_reduce_max_f32, _NEG_INF_F32, sdata, tid); \
    if (tid == 0) out[row] = STORE(acc); \
}

DEFINE_MAX_AXIS1(k_max_axis1_f32, float, _IDENTITY, _IDENTITY)
DEFINE_MAX_AXIS1(k_max_axis1_f16, __half, from_half, to_half)
DEFINE_MAX_AXIS1(k_max_axis1_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_MAX_AXIS1(k_max_axis1_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_MAX_AXIS1(k_max_axis1_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_MAX_AXIS1(k_max_axis1_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

extern "C" __global__ void k_max_axis1_f64(const double* __restrict__ in,
                                            double* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    double acc = _NEG_INF_F64;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmax(acc, in[row * cols + i]);
    __shared__ double sdata[32];
    BLOCK_REDUCE_F64(acc, warp_reduce_max_f64, _NEG_INF_F64, sdata, tid);
    if (tid == 0) out[row] = acc;
}

// ---- embedding: lookup table ----

#define DEFINE_EMBEDDING(NAME, CTYPE, LOAD_IDX) \
extern "C" __global__ void NAME( \
    const CTYPE* __restrict__ indices, \
    const CTYPE* __restrict__ weight, \
    CTYPE* __restrict__ out, \
    unsigned n_tokens, unsigned embed_dim) { \
    unsigned tid = THREAD_ID; \
    unsigned total = n_tokens * embed_dim; \
    if (tid >= total) return; \
    unsigned tok = tid / embed_dim; \
    unsigned dim = tid % embed_dim; \
    unsigned idx = (unsigned)LOAD_IDX(indices[tok]); \
    out[tid] = weight[idx * embed_dim + dim]; \
}

DEFINE_EMBEDDING(k_embedding_f32, float, _IDENTITY)
DEFINE_EMBEDDING(k_embedding_f16, __half, from_half)
DEFINE_EMBEDDING(k_embedding_bf16, __nv_bfloat16, from_bf16)
DEFINE_EMBEDDING(k_embedding_fp8e4m3, uint8_t, fp8e4m3_to_f32)
DEFINE_EMBEDDING(k_embedding_fp8e5m2, uint8_t, fp8e5m2_to_f32)
DEFINE_EMBEDDING(k_embedding_fp4e2m1, uint8_t, fp4e2m1_to_f32)

extern "C" __global__ void k_embedding_f64(
    const double* __restrict__ indices,
    const double* __restrict__ weight,
    double* __restrict__ out,
    unsigned n_tokens, unsigned embed_dim) {
    unsigned tid = THREAD_ID;
    unsigned total = n_tokens * embed_dim;
    if (tid >= total) return;
    unsigned tok = tid / embed_dim;
    unsigned dim = tid % embed_dim;
    unsigned idx = (unsigned)indices[tok];
    out[tid] = weight[idx * embed_dim + dim];
}

// ---- Blelloch exclusive prefix scan -> inclusive (f32/f64) ----
// OP_ACCUM: += or *=, IDENTITY: 0.0f/1.0f, COMBINE: + or *

#define DEFINE_BLELLOCH_PREFIX_F32(NAME, SMEM_NAME, OP_ACCUM, IDENTITY, COMBINE) \
extern "C" __global__ void NAME(const float* in, float* out, unsigned rows, unsigned cols) { \
    extern __shared__ float SMEM_NAME[]; \
    unsigned r = blockIdx.x; \
    if (r >= rows) return; \
    unsigned bx = blockDim.x; \
    unsigned n = cols; \
    if (n <= 2 * bx) { \
        unsigned i1 = 2 * threadIdx.x; \
        unsigned i2 = 2 * threadIdx.x + 1; \
        SMEM_NAME[i1] = (i1 < n) ? in[r * n + i1] : (IDENTITY); \
        SMEM_NAME[i2] = (i2 < n) ? in[r * n + i2] : (IDENTITY); \
        __syncthreads(); \
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx < 2 * bx) SMEM_NAME[idx] OP_ACCUM SMEM_NAME[idx - stride]; \
            __syncthreads(); \
        } \
        if (threadIdx.x == 0) SMEM_NAME[2 * bx - 1] = (IDENTITY); \
        __syncthreads(); \
        for (unsigned stride = bx; stride >= 1; stride >>= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx < 2 * bx) { \
                float t = SMEM_NAME[idx - stride]; \
                SMEM_NAME[idx - stride] = SMEM_NAME[idx]; \
                SMEM_NAME[idx] OP_ACCUM t; \
            } \
            __syncthreads(); \
        } \
        if (i1 < n) out[r * n + i1] = SMEM_NAME[i1] COMBINE in[r * n + i1]; \
        if (i2 < n) out[r * n + i2] = SMEM_NAME[i2] COMBINE in[r * n + i2]; \
    } else { \
        if (threadIdx.x == 0) { \
            float acc = (IDENTITY); \
            for (unsigned c = 0; c < n; c++) { \
                acc OP_ACCUM in[r * n + c]; \
                out[r * n + c] = acc; \
            } \
        } \
    } \
}

#define DEFINE_BLELLOCH_PREFIX_F64(NAME, SMEM_NAME, OP_ACCUM, IDENTITY, COMBINE) \
extern "C" __global__ void NAME(const double* in, double* out, unsigned rows, unsigned cols) { \
    extern __shared__ double SMEM_NAME[]; \
    unsigned r = blockIdx.x; \
    if (r >= rows) return; \
    unsigned bx = blockDim.x; \
    unsigned n = cols; \
    if (n <= 2 * bx) { \
        unsigned i1 = 2 * threadIdx.x; \
        unsigned i2 = 2 * threadIdx.x + 1; \
        SMEM_NAME[i1] = (i1 < n) ? in[r * n + i1] : (IDENTITY); \
        SMEM_NAME[i2] = (i2 < n) ? in[r * n + i2] : (IDENTITY); \
        __syncthreads(); \
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx < 2 * bx) SMEM_NAME[idx] OP_ACCUM SMEM_NAME[idx - stride]; \
            __syncthreads(); \
        } \
        if (threadIdx.x == 0) SMEM_NAME[2 * bx - 1] = (IDENTITY); \
        __syncthreads(); \
        for (unsigned stride = bx; stride >= 1; stride >>= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx < 2 * bx) { \
                double t = SMEM_NAME[idx - stride]; \
                SMEM_NAME[idx - stride] = SMEM_NAME[idx]; \
                SMEM_NAME[idx] OP_ACCUM t; \
            } \
            __syncthreads(); \
        } \
        if (i1 < n) out[r * n + i1] = SMEM_NAME[i1] COMBINE in[r * n + i1]; \
        if (i2 < n) out[r * n + i2] = SMEM_NAME[i2] COMBINE in[r * n + i2]; \
    } else { \
        if (threadIdx.x == 0) { \
            double acc = (IDENTITY); \
            for (unsigned c = 0; c < n; c++) { \
                acc OP_ACCUM in[r * n + c]; \
                out[r * n + c] = acc; \
            } \
        } \
    } \
}

DEFINE_BLELLOCH_PREFIX_F32(k_cumsum_axis1_f32, smem_cs_f32, +=, 0.0f, +)
DEFINE_BLELLOCH_PREFIX_F64(k_cumsum_axis1_f64, smem_cs_f64, +=, 0.0, +)
DEFINE_BLELLOCH_PREFIX_F32(k_cumprod_axis1_f32, smem_cp_f32, *=, 1.0f, *)
DEFINE_BLELLOCH_PREFIX_F64(k_cumprod_axis1_f64, smem_cp_f64, *=, 1.0, *)

// Hillis-Steele inclusive prefix for convert types (cumsum)
#define DEFINE_HILLIS_STEELE_SUM(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME(const CTYPE* in, CTYPE* out, unsigned rows, unsigned cols) { \
    extern __shared__ float smem_cs[]; \
    unsigned r = blockIdx.x; \
    if (r >= rows) return; \
    unsigned bx = blockDim.x; \
    unsigned n = cols; \
    if (n <= 2 * bx) { \
        unsigned i1 = 2 * threadIdx.x; \
        unsigned i2 = 2 * threadIdx.x + 1; \
        smem_cs[i1] = (i1 < n) ? (float)LOAD(in[r * n + i1]) : 0.0f; \
        smem_cs[i2] = (i2 < n) ? (float)LOAD(in[r * n + i2]) : 0.0f; \
        __syncthreads(); \
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx < 2 * bx) smem_cs[idx] += smem_cs[idx - stride]; \
            __syncthreads(); \
        } \
        for (unsigned stride = bx; stride > 0; stride >>= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx + stride < 2 * bx) smem_cs[idx + stride] += smem_cs[idx]; \
            __syncthreads(); \
        } \
        if (i1 < n) out[r * n + i1] = STORE(smem_cs[i1]); \
        if (i2 < n) out[r * n + i2] = STORE(smem_cs[i2]); \
    } else { \
        if (threadIdx.x == 0) { \
            float acc = 0.0f; \
            for (unsigned c = 0; c < n; c++) { \
                acc += (float)LOAD(in[r * n + c]); \
                out[r * n + c] = STORE(acc); \
            } \
        } \
    } \
}

// Hillis-Steele inclusive prefix for convert types (cumprod)
#define DEFINE_HILLIS_STEELE_PROD(NAME, CTYPE, LOAD, STORE) \
extern "C" __global__ void NAME(const CTYPE* in, CTYPE* out, unsigned rows, unsigned cols) { \
    extern __shared__ float smem_cp[]; \
    unsigned r = blockIdx.x; \
    if (r >= rows) return; \
    unsigned bx = blockDim.x; \
    unsigned n = cols; \
    if (n <= 2 * bx) { \
        unsigned i1 = 2 * threadIdx.x; \
        unsigned i2 = 2 * threadIdx.x + 1; \
        smem_cp[i1] = (i1 < n) ? (float)LOAD(in[r * n + i1]) : 1.0f; \
        smem_cp[i2] = (i2 < n) ? (float)LOAD(in[r * n + i2]) : 1.0f; \
        __syncthreads(); \
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx < 2 * bx) smem_cp[idx] *= smem_cp[idx - stride]; \
            __syncthreads(); \
        } \
        for (unsigned stride = bx; stride > 0; stride >>= 1) { \
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1; \
            if (idx + stride < 2 * bx) smem_cp[idx + stride] *= smem_cp[idx]; \
            __syncthreads(); \
        } \
        if (i1 < n) out[r * n + i1] = STORE(smem_cp[i1]); \
        if (i2 < n) out[r * n + i2] = STORE(smem_cp[i2]); \
    } else { \
        if (threadIdx.x == 0) { \
            float acc = 1.0f; \
            for (unsigned c = 0; c < n; c++) { \
                acc *= (float)LOAD(in[r * n + c]); \
                out[r * n + c] = STORE(acc); \
            } \
        } \
    } \
}

DEFINE_HILLIS_STEELE_SUM(k_cumsum_axis1_f16, __half, from_half, to_half)
DEFINE_HILLIS_STEELE_SUM(k_cumsum_axis1_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_HILLIS_STEELE_SUM(k_cumsum_axis1_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_HILLIS_STEELE_SUM(k_cumsum_axis1_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_HILLIS_STEELE_SUM(k_cumsum_axis1_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

DEFINE_HILLIS_STEELE_PROD(k_cumprod_axis1_f16, __half, from_half, to_half)
DEFINE_HILLIS_STEELE_PROD(k_cumprod_axis1_bf16, __nv_bfloat16, from_bf16, to_bf16)
DEFINE_HILLIS_STEELE_PROD(k_cumprod_axis1_fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
DEFINE_HILLIS_STEELE_PROD(k_cumprod_axis1_fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
DEFINE_HILLIS_STEELE_PROD(k_cumprod_axis1_fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

#undef POOL2D_DECODE
#undef POOL2D_FOR_WINDOW
