// kernels_cu.rs — CUDA/HIP C kernel source strings compiled at runtime via nvrtc/hiprtc.
//
// All kernels are plain C compatible with both CUDA nvcc and HIP hipcc.
// Type-suffixed kernel names: k_{op}_f32, k_{op}_f64.
// THREAD_ID = global thread index, BLOCK_SIZE = 256.

#![allow(dead_code)]

pub(crate) const BLOCK_SIZE: u32 = 256;

// Combined kernel source — unary, binary, scalar, reduction, transpose, matmul for f32+f64.
pub(crate) const KERNELS: &str = r#"
#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define TILE 16

// float4 is CUDA built-in with __attribute__((aligned(16))) → emits LDG.E.128
#define LOAD_F4(ptr, i)  (((const float4*)(ptr))[i])
#define STORE_F4(ptr, i, v) (((float4*)(ptr))[i] = (v))
#define VEC4_IDX (blockIdx.x * blockDim.x + threadIdx.x)

// ── Unary f32 (float4 + fast math) ──────────────────────────────────────

extern "C" __global__ void k_neg_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = -v.x; v.y = -v.y; v.z = -v.z; v.w = -v.w;
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = -in[j];
    }
}
extern "C" __global__ void k_recip_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = 1.0f/v.x; v.y = 1.0f/v.y; v.z = 1.0f/v.z; v.w = 1.0f/v.w;
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = 1.0f/in[j];
    }
}
extern "C" __global__ void k_exp_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __expf(v.x); v.y = __expf(v.y); v.z = __expf(v.z); v.w = __expf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = __expf(in[j]);
    }
}
extern "C" __global__ void k_ln_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __logf(v.x); v.y = __logf(v.y); v.z = __logf(v.z); v.w = __logf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = __logf(in[j]);
    }
}
extern "C" __global__ void k_log1p_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __logf(1.0f+v.x); v.y = __logf(1.0f+v.y); v.z = __logf(1.0f+v.z); v.w = __logf(1.0f+v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = log1pf(in[j]);
    }
}
extern "C" __global__ void k_sin_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __sinf(v.x); v.y = __sinf(v.y); v.z = __sinf(v.z); v.w = __sinf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = __sinf(in[j]);
    }
}
extern "C" __global__ void k_cos_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __cosf(v.x); v.y = __cosf(v.y); v.z = __cosf(v.z); v.w = __cosf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = __cosf(in[j]);
    }
}
extern "C" __global__ void k_tanh_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = tanhf(v.x); v.y = tanhf(v.y); v.z = tanhf(v.z); v.w = tanhf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = tanhf(in[j]);
    }
}
extern "C" __global__ void k_sqrt_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __fsqrt_rn(v.x); v.y = __fsqrt_rn(v.y); v.z = __fsqrt_rn(v.z); v.w = __fsqrt_rn(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = sqrtf(in[j]);
    }
}
extern "C" __global__ void k_abs_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = fabsf(v.x); v.y = fabsf(v.y); v.z = fabsf(v.z); v.w = fabsf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = fabsf(in[j]);
    }
}
extern "C" __global__ void k_ceil_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = ceilf(v.x); v.y = ceilf(v.y); v.z = ceilf(v.z); v.w = ceilf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = ceilf(in[j]);
    }
}
extern "C" __global__ void k_floor_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = floorf(v.x); v.y = floorf(v.y); v.z = floorf(v.z); v.w = floorf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = floorf(in[j]);
    }
}
extern "C" __global__ void k_round_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = roundf(v.x); v.y = roundf(v.y); v.z = roundf(v.z); v.w = roundf(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = roundf(in[j]);
    }
}
// erf — A&S polynomial (max error ~1.5e-7) with float4 + __expf
__device__ float erf_approx_f32(float x) {
    float ax = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = t * (0.254829592f + t * (-0.284496736f +
              t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    float r = 1.0f - p * __expf(-x * x);
    return (x >= 0.0f) ? r : -r;
}
extern "C" __global__ void k_erf_f32(const float* in, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = erf_approx_f32(v.x); v.y = erf_approx_f32(v.y);
        v.z = erf_approx_f32(v.z); v.w = erf_approx_f32(v.w);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = erf_approx_f32(in[j]);
    }
}

// ── Binary f32 (float4) ─────────────────────────────────────────────────

extern "C" __global__ void k_add_f32(const float* a, const float* b, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4);
        float4 vo = make_float4(va.x+vb.x, va.y+vb.y, va.z+vb.z, va.w+vb.w);
        STORE_F4(out, i4, vo);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = a[j]+b[j];
    }
}
extern "C" __global__ void k_sub_f32(const float* a, const float* b, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4);
        float4 vo = make_float4(va.x-vb.x, va.y-vb.y, va.z-vb.z, va.w-vb.w);
        STORE_F4(out, i4, vo);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = a[j]-b[j];
    }
}
extern "C" __global__ void k_emul_f32(const float* a, const float* b, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4);
        float4 vo = make_float4(va.x*vb.x, va.y*vb.y, va.z*vb.z, va.w*vb.w);
        STORE_F4(out, i4, vo);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = a[j]*b[j];
    }
}
extern "C" __global__ void k_ediv_f32(const float* a, const float* b, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4);
        float4 vo = make_float4(va.x/vb.x, va.y/vb.y, va.z/vb.z, va.w/vb.w);
        STORE_F4(out, i4, vo);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = a[j]/b[j];
    }
}

// ── Scalar ops f32 (float4 + fast math) ─────────────────────────────────

extern "C" __global__ void k_scale_f32(const float* in, float s, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x *= s; v.y *= s; v.z *= s; v.w *= s;
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = in[j]*s;
    }
}
extern "C" __global__ void k_powf_f32(const float* in, float p, float* out, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = LOAD_F4(in, i4);
        v.x = __expf(p*__logf(v.x)); v.y = __expf(p*__logf(v.y));
        v.z = __expf(p*__logf(v.z)); v.w = __expf(p*__logf(v.w));
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = powf(in[j], p);
    }
}
extern "C" __global__ void k_fill_f32(float* out, float val, unsigned n) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned n4 = n >> 2;
    for (unsigned i4 = VEC4_IDX; i4 < n4; i4 += stride) {
        float4 v = make_float4(val, val, val, val);
        STORE_F4(out, i4, v);
    }
    for (unsigned j = n4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; j < n; j += stride) {
        out[j] = val;
    }
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

// ── Reduction f32 (warp shuffle) ──────────────────────────────────────────

extern "C" __global__ void k_sum_f32(const float* in, float* out, unsigned n) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? in[i] : 0.0f;

    val = warp_reduce_sum_f32(val);

    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
    }
    if (tid == 0) atomicAdd(&out[0], val);
}

extern "C" __global__ void k_max_f32(const float* in, float* out,
                                      unsigned n, const float* init) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? in[i] : *init;

    val = warp_reduce_max_f32(val);

    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : *init;
        val = warp_reduce_max_f32(val);
    }
    if (tid == 0) out[blockIdx.x] = val;
}

extern "C" __global__ void k_min_f32(const float* in, float* out,
                                      unsigned n, const float* init) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? in[i] : *init;

    val = warp_reduce_min_f32(val);

    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : *init;
        val = warp_reduce_min_f32(val);
    }
    if (tid == 0) out[blockIdx.x] = val;
}

// ── Unary f64 ──────────────────────────────────────────────────────────────

extern "C" __global__ void k_neg_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = -in[i];
}
extern "C" __global__ void k_recip_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = 1.0 / in[i];
}
extern "C" __global__ void k_exp_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = exp(in[i]);
}
extern "C" __global__ void k_ln_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = log(in[i]);
}
extern "C" __global__ void k_log1p_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = log1p(in[i]);
}
extern "C" __global__ void k_sin_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = sin(in[i]);
}
extern "C" __global__ void k_cos_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = cos(in[i]);
}
extern "C" __global__ void k_tanh_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = tanh(in[i]);
}
extern "C" __global__ void k_sqrt_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = sqrt(in[i]);
}
extern "C" __global__ void k_abs_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = fabs(in[i]);
}
extern "C" __global__ void k_ceil_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = ceil(in[i]);
}
extern "C" __global__ void k_floor_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = floor(in[i]);
}
extern "C" __global__ void k_round_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = round(in[i]);
}
extern "C" __global__ void k_erf_f64(const double* in, double* out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        double x = in[i];
        double ax = fabs(x);
        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double p = t * (0.254829592 + t * (-0.284496736 +
                   t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        double r = 1.0 - p * exp(-x * x);
        out[i] = (x >= 0.0) ? r : -r;
    }
}

// ── Binary f64 ─────────────────────────────────────────────────────────────

extern "C" __global__ void k_add_f64(const double* a, const double* b, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void k_sub_f64(const double* a, const double* b, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = a[i] - b[i];
}
extern "C" __global__ void k_emul_f64(const double* a, const double* b, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = a[i] * b[i];
}
extern "C" __global__ void k_ediv_f64(const double* a, const double* b, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = a[i] / b[i];
}

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

// ── Reduction f64 (warp shuffle) ──────────────────────────────────────────

extern "C" __global__ void k_sum_f64(const double* in, double* out, unsigned n) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    double val = (i < n) ? in[i] : 0.0;

    val = warp_reduce_sum_f64(val);

    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0;
        val = warp_reduce_sum_f64(val);
    }
    // atomicAdd for doubles requires compute >= 6.0
    if (tid == 0) atomicAdd(&out[0], val);
}

extern "C" __global__ void k_max_f64(const double* in, double* out,
                                      unsigned n, const double* init) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    double val = (i < n) ? in[i] : *init;

    val = warp_reduce_max_f64(val);

    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : *init;
        val = warp_reduce_max_f64(val);
    }
    if (tid == 0) out[blockIdx.x] = val;
}

extern "C" __global__ void k_min_f64(const double* in, double* out,
                                      unsigned n, const double* init) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    double val = (i < n) ? in[i] : *init;

    val = warp_reduce_min_f64(val);

    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : *init;
        val = warp_reduce_min_f64(val);
    }
    if (tid == 0) out[blockIdx.x] = val;
}

"#;

// Total: 42 kernels (14 unary + 4 binary + 3 scalar + 1 transpose + 1 matmul + 3 reduction) x 2 types

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
