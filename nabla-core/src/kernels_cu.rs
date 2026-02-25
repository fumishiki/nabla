// kernels_cu.rs — CUDA/HIP C kernel source strings compiled at runtime via nvrtc/hiprtc.
//
// All kernels are plain C compatible with both CUDA nvcc and HIP hipcc.
// Type-suffixed kernel names: k_{op}_f32, k_{op}_f64.
// THREAD_ID = global thread index, BLOCK_SIZE = 256.

#![allow(dead_code)]

pub(crate) const BLOCK_SIZE: u32 = 256;
pub(crate) const REDUCE_BLOCK: u32 = 256;
// Max blocks for first-pass reduction; last block aggregates.
pub(crate) const REDUCE_GRID_CAP: u32 = 256;

// Combined kernel source — unary, binary, scalar, reduction, transpose, matmul for f32+f64.
pub(crate) const KERNELS: &str = r#"
#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define TILE 16

// float4 is CUDA built-in with __attribute__((aligned(16))) → emits LDG.E.128
#define LOAD_F4(ptr, i)  (((const float4*)(ptr))[i])
#define STORE_F4(ptr, i, v) (((float4*)(ptr))[i] = (v))
#define VEC4_IDX (blockIdx.x * blockDim.x + threadIdx.x)

// ── Kernel generator macros ──────────────────────────────────────────────

#define _NEG(x) (-(x))
#define _RECIP_F(x) (1.0f/(x))
#define _RECIP_D(x) (1.0/(x))
#define _LOG1P_FAST(x) (__logf(1.0f+(x)))

#define UNARY_F32(name, vop, sop) \
extern "C" __global__ void k_##name##_f32(const float* in, float* out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 v = LOAD_F4(in, i4); \
        v.x = vop(v.x); v.y = vop(v.y); v.z = vop(v.z); v.w = vop(v.w); \
        STORE_F4(out, i4, v); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = sop(in[j]); } \
}

#define UNARY_F64(name, op) \
extern "C" __global__ void k_##name##_f64(const double* in, double* out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = op(in[i]); \
}

#define BINARY_F32(name, op) \
extern "C" __global__ void k_##name##_f32(const float* a, const float* b, float* out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4); \
        float4 vo = make_float4(va.x op vb.x, va.y op vb.y, va.z op vb.z, va.w op vb.w); \
        STORE_F4(out, i4, vo); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = a[j] op b[j]; } \
}

#define BINARY_F64(name, op) \
extern "C" __global__ void k_##name##_f64(const double* a, const double* b, double* out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = a[i] op b[i]; \
}

// ── Device helpers (erf A&S polynomial, max error ~1.5e-7) ──────────────

__device__ float erf_approx_f32(float x) {
    float ax = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = t * (0.254829592f + t * (-0.284496736f +
              t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    float r = 1.0f - p * __expf(-x * x);
    return (x >= 0.0f) ? r : -r;
}

__device__ double erf_approx_f64(double x) {
    double ax = fabs(x);
    double t = 1.0 / (1.0 + 0.3275911 * ax);
    double p = t * (0.254829592 + t * (-0.284496736 +
               t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    double r = 1.0 - p * exp(-x * x);
    return (x >= 0.0) ? r : -r;
}

// ── Unary f32 (float4 + fast math) ──────────────────────────────────────

UNARY_F32(neg,   _NEG,            _NEG)
UNARY_F32(recip, _RECIP_F,        _RECIP_F)
UNARY_F32(exp,   __expf,          __expf)
UNARY_F32(ln,    __logf,          __logf)
UNARY_F32(log1p, _LOG1P_FAST,     log1pf)
UNARY_F32(sin,   __sinf,          __sinf)
UNARY_F32(cos,   __cosf,          __cosf)
UNARY_F32(tanh,  tanhf,           tanhf)
UNARY_F32(sqrt,  __fsqrt_rn,      sqrtf)
UNARY_F32(abs,   fabsf,           fabsf)
UNARY_F32(ceil,  ceilf,           ceilf)
UNARY_F32(floor, floorf,          floorf)
UNARY_F32(round, roundf,          roundf)
UNARY_F32(erf,   erf_approx_f32,  erf_approx_f32)

// ── Activation device helpers ────────────────────────────────────────────

__device__ float sigmoid_f32(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ double sigmoid_f64(double x) { return 1.0 / (1.0 + exp(-x)); }

__device__ float silu_f32(float x) { return x * sigmoid_f32(x); }
__device__ double silu_f64(double x) { return x * sigmoid_f64(x); }

__device__ float mish_f32(float x) {
    float sp = __logf(1.0f + __expf(x)); // softplus
    return x * tanhf(sp);
}
__device__ double mish_f64(double x) {
    double sp = log(1.0 + exp(x));
    return x * tanh(sp);
}

__device__ float leaky_relu_f32(float x) { return (x >= 0.0f) ? x : 0.01f * x; }
__device__ double leaky_relu_f64(double x) { return (x >= 0.0) ? x : 0.01 * x; }

__device__ float elu_f32(float x) { return (x >= 0.0f) ? x : (__expf(x) - 1.0f); }
__device__ double elu_f64(double x) { return (x >= 0.0) ? x : (exp(x) - 1.0); }

__device__ float hardswish_f32(float x) {
    float v = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * v * (1.0f / 6.0f);
}
__device__ double hardswish_f64(double x) {
    double v = fmin(fmax(x + 3.0, 0.0), 6.0);
    return x * v * (1.0 / 6.0);
}

// ── Activation kernels f32 (float4) ──────────────────────────────────────

UNARY_F32(silu,       silu_f32,       silu_f32)
UNARY_F32(mish,       mish_f32,       mish_f32)
UNARY_F32(leaky_relu, leaky_relu_f32, leaky_relu_f32)
UNARY_F32(elu,        elu_f32,        elu_f32)
UNARY_F32(hardswish,  hardswish_f32,  hardswish_f32)

// ── Binary f32 (float4) ─────────────────────────────────────────────────

BINARY_F32(add,  +)
BINARY_F32(sub,  -)
BINARY_F32(emul, *)
BINARY_F32(ediv, /)

// ── Scalar ops f32 (float4 + fast math) ─────────────────────────────────

extern "C" __global__ void k_scale_f32(const float* in, float s, float* out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = LOAD_F4(in, i4);
        v.x *= s; v.y *= s; v.z *= s; v.w *= s;
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = in[j]*s; }
}
extern "C" __global__ void k_powf_f32(const float* in, float p, float* out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = LOAD_F4(in, i4);
        v.x = __expf(p*__logf(v.x)); v.y = __expf(p*__logf(v.y));
        v.z = __expf(p*__logf(v.z)); v.w = __expf(p*__logf(v.w));
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = powf(in[j], p); }
}
extern "C" __global__ void k_fill_f32(float* out, float val, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = make_float4(val, val, val, val);
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = val; }
}

// ── Transpose f32 ──────────────────────────────────────────────────────────

extern "C" __global__ void k_transpose_f32(const float* in, float* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}

// ── Tiled matmul f32 ───────────────────────────────────────────────────────

extern "C" __global__ void k_matmul_f32(const float* A, const float* B, float* C,
                                         unsigned M, unsigned K, unsigned N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    unsigned row = blockIdx.y * TILE + threadIdx.y;
    unsigned col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned ak = t * TILE + threadIdx.x;
        unsigned bk = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ── Warp shuffle device functions f32 ─────────────────────────────────────

#ifdef __HIP_PLATFORM_AMD__
#define SHFL_DOWN_F32(val, offset) __shfl_down(val, offset)
#define SHFL_DOWN_F64(val, offset) __shfl_down(val, offset)
#else
#define SHFL_DOWN_F32(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#define SHFL_DOWN_F64(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#endif

__device__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += SHFL_DOWN_F32(val, offset);
    return val;
}

__device__ float warp_reduce_max_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, SHFL_DOWN_F32(val, offset));
    return val;
}

__device__ float warp_reduce_min_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, SHFL_DOWN_F32(val, offset));
    return val;
}

// ── Reduction f32 (grid-stride + vectorized float4) ──────────────────────

// Single-kernel sum: grid-stride + float4 + last-block final reduction
extern "C" __global__ void __launch_bounds__(256) k_sum_f32(
    const float* __restrict__ in,
    float* __restrict__ partial,
    unsigned n,
    float* __restrict__ out) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    unsigned n4 = n / 4;
    const float4* in4 = (const float4*)in;
    unsigned i = blockIdx.x * blockDim.x + tid;
    unsigned stride4 = grid_stride * 4;
    // Quad-accumulator loop for maximum ILP
    for (; i + grid_stride * 3 < n4; i += stride4) {
        float4 v0 = in4[i];
        float4 v1 = in4[i + grid_stride];
        float4 v2 = in4[i + grid_stride * 2];
        float4 v3 = in4[i + grid_stride * 3];
        acc0 += v0.x + v0.y + v0.z + v0.w;
        acc1 += v1.x + v1.y + v1.z + v1.w;
        acc2 += v2.x + v2.y + v2.z + v2.w;
        acc3 += v3.x + v3.y + v3.z + v3.w;
    }
    for (; i < n4; i += grid_stride) {
        float4 v = in4[i];
        acc0 += v.x + v.y + v.z + v.w;
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (unsigned j = n4 * 4 + blockIdx.x * blockDim.x + tid; j < n; j += grid_stride)
        acc += in[j];

    acc = warp_reduce_sum_f32(acc);

    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
    }
    if (tid == 0) partial[blockIdx.x] = acc;

    // Last-block aggregation: use counter at partial[gridDim.x] (cast to uint)
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        unsigned ticket = atomicInc(counter, gridDim.x);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
                val = warp_reduce_sum_f32(val);
            }
        }
        if (tid == 0) out[0] = val;
    }
}

// Single-kernel max: grid-stride + float4 + last-block aggregation
extern "C" __global__ void __launch_bounds__(256) k_max_f32(
    const float* __restrict__ in,
    float* __restrict__ partial,
    unsigned n,
    float* __restrict__ out) {
    float neg_inf = -__int_as_float(0x7f800000);
    float acc0 = neg_inf, acc1 = neg_inf, acc2 = neg_inf, acc3 = neg_inf;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    const float4* in4 = (const float4*)in;
    unsigned n4 = n / 4;
    unsigned i = blockIdx.x * blockDim.x + tid;
    unsigned stride4 = grid_stride * 4;
    for (; i + grid_stride * 3 < n4; i += stride4) {
        float4 v0 = in4[i];
        float4 v1 = in4[i + grid_stride];
        float4 v2 = in4[i + grid_stride * 2];
        float4 v3 = in4[i + grid_stride * 3];
        acc0 = fmaxf(acc0, fmaxf(fmaxf(v0.x, v0.y), fmaxf(v0.z, v0.w)));
        acc1 = fmaxf(acc1, fmaxf(fmaxf(v1.x, v1.y), fmaxf(v1.z, v1.w)));
        acc2 = fmaxf(acc2, fmaxf(fmaxf(v2.x, v2.y), fmaxf(v2.z, v2.w)));
        acc3 = fmaxf(acc3, fmaxf(fmaxf(v3.x, v3.y), fmaxf(v3.z, v3.w)));
    }
    for (; i < n4; i += grid_stride) {
        float4 v = in4[i];
        acc0 = fmaxf(acc0, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    float acc = fmaxf(fmaxf(acc0, acc1), fmaxf(acc2, acc3));
    for (unsigned j = n4 * 4 + blockIdx.x * blockDim.x + tid; j < n; j += grid_stride)
        acc = fmaxf(acc, in[j]);

    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : neg_inf;
        acc = warp_reduce_max_f32(acc);
    }
    if (tid == 0) partial[blockIdx.x] = acc;

    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        unsigned ticket = atomicInc(counter, gridDim.x);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : neg_inf;
        val = warp_reduce_max_f32(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : neg_inf;
                val = warp_reduce_max_f32(val);
            }
        }
        if (tid == 0) out[0] = val;
    }
}

extern "C" __global__ void k_min_f32(const float* __restrict__ in,
                                      float* __restrict__ partial,
                                      unsigned n,
                                      float* __restrict__ out) {
    float acc = __int_as_float(0x7f800000); // +INFINITY
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    const float4* in4 = (const float4*)in;
    unsigned n4 = n / 4;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n4; i += grid_stride) {
        float4 v = in4[i];
        acc = fminf(acc, fminf(fminf(v.x, v.y), fminf(v.z, v.w)));
    }
    for (unsigned i = n4 * 4 + blockIdx.x * blockDim.x + tid; i < n; i += grid_stride)
        acc = fminf(acc, in[i]);

    acc = warp_reduce_min_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : __int_as_float(0x7f800000);
        acc = warp_reduce_min_f32(acc);
    }
    if (tid == 0) partial[blockIdx.x] = acc;

    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        unsigned ticket = atomicInc(counter, gridDim.x);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        float pos_inf = __int_as_float(0x7f800000);
        float val = (tid < gridDim.x) ? partial[tid] : pos_inf;
        val = warp_reduce_min_f32(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : pos_inf;
                val = warp_reduce_min_f32(val);
            }
        }
        if (tid == 0) out[0] = val;
    }
}

// ── Unary f64 ──────────────────────────────────────────────────────────────

UNARY_F64(neg,   _NEG)
UNARY_F64(recip, _RECIP_D)
UNARY_F64(exp,   exp)
UNARY_F64(ln,    log)
UNARY_F64(log1p, log1p)
UNARY_F64(sin,   sin)
UNARY_F64(cos,   cos)
UNARY_F64(tanh,  tanh)
UNARY_F64(sqrt,  sqrt)
UNARY_F64(abs,   fabs)
UNARY_F64(ceil,  ceil)
UNARY_F64(floor, floor)
UNARY_F64(round, round)
UNARY_F64(erf,   erf_approx_f64)

// ── Activation kernels f64 ───────────────────────────────────────────────

UNARY_F64(silu,       silu_f64)
UNARY_F64(mish,       mish_f64)
UNARY_F64(leaky_relu, leaky_relu_f64)
UNARY_F64(elu,        elu_f64)
UNARY_F64(hardswish,  hardswish_f64)

// ── Binary f64 ─────────────────────────────────────────────────────────────

BINARY_F64(add,  +)
BINARY_F64(sub,  -)
BINARY_F64(emul, *)
BINARY_F64(ediv, /)

// ── Scalar ops f64 ─────────────────────────────────────────────────────────

extern "C" __global__ void k_scale_f64(const double* in, double s, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = in[i] * s;
}
extern "C" __global__ void k_powf_f64(const double* in, double p, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = pow(in[i], p);
}
extern "C" __global__ void k_fill_f64(double* out, double val, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = val;
}

// ── Transpose f64 ──────────────────────────────────────────────────────────

extern "C" __global__ void k_transpose_f64(const double* in, double* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}

// ── Tiled matmul f64 ───────────────────────────────────────────────────────

extern "C" __global__ void k_matmul_f64(const double* A, const double* B, double* C,
                                         unsigned M, unsigned K, unsigned N) {
    __shared__ double sA[TILE][TILE], sB[TILE][TILE];
    unsigned row = blockIdx.y * TILE + threadIdx.y;
    unsigned col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned ak = t * TILE + threadIdx.x;
        unsigned bk = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0;
        __syncthreads();
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ── Warp shuffle device functions f64 ─────────────────────────────────────

__device__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += SHFL_DOWN_F64(val, offset);
    return val;
}

__device__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmax(val, SHFL_DOWN_F64(val, offset));
    return val;
}

__device__ double warp_reduce_min_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmin(val, SHFL_DOWN_F64(val, offset));
    return val;
}

// ── Reduction f64 (grid-stride + vectorized double2) ─────────────────────

// Single-kernel sum with last-block aggregation
extern "C" __global__ void k_sum_f64(const double* __restrict__ in,
                                      double* __restrict__ partial, unsigned n,
                                      double* __restrict__ out) {
    double acc = 0.0;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    unsigned n2 = n / 2;
    const double2* in2 = (const double2*)in;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n2; i += grid_stride) {
        double2 v = in2[i];
        acc += v.x + v.y;
    }
    for (unsigned i = n2 * 2 + blockIdx.x * blockDim.x + tid; i < n; i += grid_stride)
        acc += in[i];

    acc = warp_reduce_sum_f64(acc);

    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
        acc = warp_reduce_sum_f64(acc);
    }
    if (tid == 0) partial[blockIdx.x] = acc;

    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        unsigned ticket = atomicInc(counter, gridDim.x);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        double val = (tid < gridDim.x) ? partial[tid] : 0.0;
        val = warp_reduce_sum_f64(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
                val = warp_reduce_sum_f64(val);
            }
        }
        if (tid == 0) out[0] = val;
    }
}

// Single-kernel max with last-block aggregation
extern "C" __global__ void k_max_f64(const double* __restrict__ in,
                                      double* __restrict__ partial,
                                      unsigned n,
                                      double* __restrict__ out) {
    double acc = __longlong_as_double(0xFFF0000000000000LL); // -INFINITY
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    unsigned n2 = n / 2;
    const double2* in2 = (const double2*)in;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n2; i += grid_stride) {
        double2 v = in2[i];
        acc = fmax(acc, fmax(v.x, v.y));
    }
    for (unsigned i = n2 * 2 + blockIdx.x * blockDim.x + tid; i < n; i += grid_stride)
        acc = fmax(acc, in[i]);

    acc = warp_reduce_max_f64(acc);
    __shared__ double sdata[32];
    double neg_inf = __longlong_as_double(0xFFF0000000000000LL);
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : neg_inf;
        acc = warp_reduce_max_f64(acc);
    }
    if (tid == 0) partial[blockIdx.x] = acc;

    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        unsigned ticket = atomicInc(counter, gridDim.x);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        double val = (tid < gridDim.x) ? partial[tid] : neg_inf;
        val = warp_reduce_max_f64(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : neg_inf;
                val = warp_reduce_max_f64(val);
            }
        }
        if (tid == 0) out[0] = val;
    }
}

extern "C" __global__ void k_min_f64(const double* __restrict__ in,
                                      double* __restrict__ partial,
                                      unsigned n,
                                      double* __restrict__ out) {
    double acc = __longlong_as_double(0x7FF0000000000000LL); // +INFINITY
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    unsigned n2 = n / 2;
    const double2* in2 = (const double2*)in;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n2; i += grid_stride) {
        double2 v = in2[i];
        acc = fmin(acc, fmin(v.x, v.y));
    }
    for (unsigned i = n2 * 2 + blockIdx.x * blockDim.x + tid; i < n; i += grid_stride)
        acc = fmin(acc, in[i]);

    acc = warp_reduce_min_f64(acc);
    __shared__ double sdata[32];
    double pos_inf = __longlong_as_double(0x7FF0000000000000LL);
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : pos_inf;
        acc = warp_reduce_min_f64(acc);
    }
    if (tid == 0) partial[blockIdx.x] = acc;

    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        unsigned ticket = atomicInc(counter, gridDim.x);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last) {
        double val = (tid < gridDim.x) ? partial[tid] : pos_inf;
        val = warp_reduce_min_f64(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : pos_inf;
                val = warp_reduce_min_f64(val);
            }
        }
        if (tid == 0) out[0] = val;
    }
}

// ── Online Softmax (row-wise, one block per row) f32 ─────────────────────

extern "C" __global__ void k_softmax_f32(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + row * cols;
    float* y = out + row * cols;
    unsigned tid = threadIdx.x;

    // Pass 1: find row max
    float m = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmaxf(m, x[i]);
    m = warp_reduce_max_f32(m);
    __shared__ float smax[32];
    if (tid % 32 == 0) smax[tid / 32] = m;
    __syncthreads();
    if (tid < 32) {
        m = (tid < (blockDim.x + 31) / 32) ? smax[tid] : -__int_as_float(0x7f800000);
        m = warp_reduce_max_f32(m);
    }
    __syncthreads();
    if (tid == 0) smax[0] = m;
    __syncthreads();
    m = smax[0];

    // Pass 2: sum of exp(x - max)
    float s = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += __expf(x[i] - m);
    s = warp_reduce_sum_f32(s);
    __shared__ float ssum[32];
    if (tid % 32 == 0) ssum[tid / 32] = s;
    __syncthreads();
    if (tid < 32) {
        s = (tid < (blockDim.x + 31) / 32) ? ssum[tid] : 0.0f;
        s = warp_reduce_sum_f32(s);
    }
    __syncthreads();
    if (tid == 0) ssum[0] = s;
    __syncthreads();
    s = ssum[0];

    // Pass 3: write softmax output
    float inv_s = 1.0f / s;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = __expf(x[i] - m) * inv_s;
}

extern "C" __global__ void k_softmax_f64(const double* __restrict__ in,
                                          double* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const double* x = in + row * cols;
    double* y = out + row * cols;
    unsigned tid = threadIdx.x;

    double m = __longlong_as_double(0xFFF0000000000000LL);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmax(m, x[i]);
    m = warp_reduce_max_f64(m);
    __shared__ double smax[32];
    if (tid % 32 == 0) smax[tid / 32] = m;
    __syncthreads();
    if (tid < 32) {
        m = (tid < (blockDim.x + 31) / 32) ? smax[tid] : __longlong_as_double(0xFFF0000000000000LL);
        m = warp_reduce_max_f64(m);
    }
    __syncthreads();
    if (tid == 0) smax[0] = m;
    __syncthreads();
    m = smax[0];

    double s = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += exp(x[i] - m);
    s = warp_reduce_sum_f64(s);
    __shared__ double ssum[32];
    if (tid % 32 == 0) ssum[tid / 32] = s;
    __syncthreads();
    if (tid < 32) {
        s = (tid < (blockDim.x + 31) / 32) ? ssum[tid] : 0.0;
        s = warp_reduce_sum_f64(s);
    }
    __syncthreads();
    if (tid == 0) ssum[0] = s;
    __syncthreads();
    s = ssum[0];

    double inv_s = 1.0 / s;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = exp(x[i] - m) * inv_s;
}

// ── Fused Layer Norm (one block per row) f32 ─────────────────────────────
// out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]

extern "C" __global__ void k_layer_norm_f32(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    unsigned rows, unsigned cols, float eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + row * cols;
    float* y = out + row * cols;
    unsigned tid = threadIdx.x;

    // Mean
    float sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sum += x[i];
    sum = warp_reduce_sum_f32(sum);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        sum = warp_reduce_sum_f32(sum);
    }
    __syncthreads();
    if (tid == 0) sdata[0] = sum;
    __syncthreads();
    float mean = sdata[0] / (float)cols;

    // Variance
    float var_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float d = x[i] - mean;
        var_sum += d * d;
    }
    var_sum = warp_reduce_sum_f32(var_sum);
    if (tid % 32 == 0) sdata[tid / 32] = var_sum;
    __syncthreads();
    if (tid < 32) {
        var_sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        var_sum = warp_reduce_sum_f32(var_sum);
    }
    __syncthreads();
    if (tid == 0) sdata[0] = var_sum;
    __syncthreads();
    float inv_std = 1.0f / sqrtf(sdata[0] / (float)cols + eps);

    // Normalize + affine
    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

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
    sum = warp_reduce_sum_f64(sum);
    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
        sum = warp_reduce_sum_f64(sum);
    }
    __syncthreads();
    if (tid == 0) sdata[0] = sum;
    __syncthreads();
    double mean = sdata[0] / (double)cols;

    double var_sum = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        double d = x[i] - mean;
        var_sum += d * d;
    }
    var_sum = warp_reduce_sum_f64(var_sum);
    if (tid % 32 == 0) sdata[tid / 32] = var_sum;
    __syncthreads();
    if (tid < 32) {
        var_sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
        var_sum = warp_reduce_sum_f64(var_sum);
    }
    __syncthreads();
    if (tid == 0) sdata[0] = var_sum;
    __syncthreads();
    double inv_std = 1.0 / sqrt(sdata[0] / (double)cols + eps);

    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

// ── Fused RMS Norm (one block per row) ──────────────────────────────────
// out[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]

extern "C" __global__ void k_rms_norm_f32(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    unsigned rows, unsigned cols, float eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + row * cols;
    float* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sq_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sq_sum += x[i] * x[i];
    sq_sum = warp_reduce_sum_f32(sq_sum);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = sq_sum;
    __syncthreads();
    if (tid < 32) {
        sq_sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        sq_sum = warp_reduce_sum_f32(sq_sum);
    }
    __syncthreads();
    if (tid == 0) sdata[0] = sq_sum;
    __syncthreads();
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)cols + eps);

    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = x[i] * inv_rms * gamma[i];
}

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
    sq_sum = warp_reduce_sum_f64(sq_sum);
    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = sq_sum;
    __syncthreads();
    if (tid < 32) {
        sq_sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
        sq_sum = warp_reduce_sum_f64(sq_sum);
    }
    __syncthreads();
    if (tid == 0) sdata[0] = sq_sum;
    __syncthreads();
    double inv_rms = 1.0 / sqrt(sdata[0] / (double)cols + eps);

    for (unsigned i = tid; i < cols; i += blockDim.x)
        y[i] = x[i] * inv_rms * gamma[i];
}

// ── Axis reduction (sum along rows → column vector, or along cols → row) ──

// Sum along axis=1 (cols): one block per row, output is (rows, 1)
extern "C" __global__ void k_sum_axis1_f32(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += in[row * cols + i];
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
    }
    if (tid == 0) out[row] = acc;
}

extern "C" __global__ void k_sum_axis1_f64(const double* __restrict__ in,
                                            double* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    double acc = 0.0;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += in[row * cols + i];
    acc = warp_reduce_sum_f64(acc);
    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
        acc = warp_reduce_sum_f64(acc);
    }
    if (tid == 0) out[row] = acc;
}

// Max along axis=1: one block per row, output is (rows, 1)
extern "C" __global__ void k_max_axis1_f32(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmaxf(acc, in[row * cols + i]);
    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : -__int_as_float(0x7f800000);
        acc = warp_reduce_max_f32(acc);
    }
    if (tid == 0) out[row] = acc;
}

extern "C" __global__ void k_max_axis1_f64(const double* __restrict__ in,
                                            double* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    double acc = __longlong_as_double(0xFFF0000000000000LL);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmax(acc, in[row * cols + i]);
    acc = warp_reduce_max_f64(acc);
    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : __longlong_as_double(0xFFF0000000000000LL);
        acc = warp_reduce_max_f64(acc);
    }
    if (tid == 0) out[row] = acc;
}

// ── Embedding gather kernel ─────────────────────────────────────────────
// indices: (batch, seq_len) as integer indices stored as float/double
// weight: (vocab_size, embed_dim), output: (batch*seq_len, embed_dim)

extern "C" __global__ void k_embedding_f32(
    const float* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned n_tokens, unsigned embed_dim) {
    unsigned tid = THREAD_ID;
    unsigned total = n_tokens * embed_dim;
    if (tid >= total) return;
    unsigned tok = tid / embed_dim;
    unsigned dim = tid % embed_dim;
    unsigned idx = (unsigned)indices[tok];
    out[tid] = weight[idx * embed_dim + dim];
}

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

"#;

// Total: 62 kernels (19 unary + 4 binary + 3 scalar + 1 transpose + 1 matmul + 3 reduction) x 2 types
//        + 2 softmax + 2 layer_norm + 2 rms_norm + 4 axis_reduce + 2 embedding

// ── WMMA tensor-core matmul ─────────────────────────────────────────────────
// Separate compilation unit — requires <mma.h> (CUDA) or rocWMMA (HIP).
// CUDA: Volta+ (sm_70), F16 inputs → F32 accumulator, 16x16x16 tiles.
// HIP: CDNA2+ (gfx90a), uses rocWMMA intrinsics.

#[cfg(feature = "cuda")]
pub(crate) const WMMA_KERNELS: &str = r#"
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void k_matmul_wmma_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    unsigned M, unsigned K, unsigned N
) {
    // Each warp computes one WMMA_M x WMMA_N output tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (unsigned k = 0; k < K; k += WMMA_K) {
        unsigned aRow = warpM * WMMA_M;
        unsigned bCol = warpN * WMMA_N;

        if (aRow < M && k + WMMA_K <= K)
            wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        if (k + WMMA_K <= K && bCol < N)
            wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    unsigned cRow = warpM * WMMA_M;
    unsigned cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}
"#;

#[cfg(feature = "hip")]
pub(crate) const WMMA_KERNELS: &str = r#"
#include <rocwmma/rocwmma.hpp>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void k_matmul_wmma_f16(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    float* __restrict__ C,
    unsigned M, unsigned K, unsigned N
) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 64;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    auto a_frag = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, _Float16, rocwmma::row_major>();
    auto b_frag = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, _Float16, rocwmma::row_major>();
    auto c_frag = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
    rocwmma::fill_fragment(c_frag, 0.0f);

    for (unsigned k = 0; k < K; k += WMMA_K) {
        unsigned aRow = warpM * WMMA_M;
        unsigned bCol = warpN * WMMA_N;

        if (aRow < M && k + WMMA_K <= K)
            rocwmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        if (k + WMMA_K <= K && bCol < N)
            rocwmma::load_matrix_sync(b_frag, B + k * N + bCol, N);

        rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    unsigned cRow = warpM * WMMA_M;
    unsigned cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N)
        rocwmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, rocwmma::mem_row_major);
}
"#;

// CPU-only: WMMA kernels not available
#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub(crate) const WMMA_KERNELS: &str = "";
