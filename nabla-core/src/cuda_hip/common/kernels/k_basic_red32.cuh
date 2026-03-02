// k_basic_red32.cuh — Warp-level f32 reduction helpers (warp_reduce_sum/max, SHFL_DOWN)

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

extern "C" __global__ void __launch_bounds__(256) k_sum_f16(
    const __half* __restrict__ in,
    float* __restrict__ partial,
    unsigned n,
    __half* __restrict__ out) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        acc += from_half(in[i]);
    }
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata_sum[32];
    if (tid % 32 == 0) sdata_sum[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata_sum[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
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
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata_sum[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata_sum[tid] : 0.0f;
                val = warp_reduce_sum_f32(val);
            }
        }
        if (tid == 0) out[0] = to_half(val);
    }
}

extern "C" __global__ void __launch_bounds__(256) k_max_f16(
    const __half* __restrict__ in,
    float* __restrict__ partial,
    unsigned n,
    __half* __restrict__ out) {
    float neg_inf = -__int_as_float(0x7f800000);
    float acc = neg_inf;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        acc = fmaxf(acc, from_half(in[i]));
    }
    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata_max[32];
    if (tid % 32 == 0) sdata_max[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata_max[tid] : neg_inf;
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
            if (tid % 32 == 0) sdata_max[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata_max[tid] : neg_inf;
                val = warp_reduce_max_f32(val);
            }
        }
        if (tid == 0) out[0] = to_half(val);
    }
}

extern "C" __global__ void k_min_f16(const __half* __restrict__ in,
                                      float* __restrict__ partial,
                                      unsigned n,
                                      __half* __restrict__ out) {
    float pos_inf = __int_as_float(0x7f800000);
    float acc = pos_inf;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) {
        unsigned* counter = (unsigned*)&partial[gridDim.x];
        *counter = 0u;
    }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        acc = fminf(acc, from_half(in[i]));
    }
    acc = warp_reduce_min_f32(acc);
    __shared__ float sdata_min[32];
    if (tid % 32 == 0) sdata_min[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata_min[tid] : pos_inf;
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
        float val = (tid < gridDim.x) ? partial[tid] : pos_inf;
        val = warp_reduce_min_f32(val);
        if (blockDim.x > 32) {
            if (tid % 32 == 0) sdata_min[tid / 32] = val;
            __syncthreads();
            if (tid < 32) {
                val = (tid < (blockDim.x + 31) / 32) ? sdata_min[tid] : pos_inf;
                val = warp_reduce_min_f32(val);
            }
        }
        if (tid == 0) out[0] = to_half(val);
    }
}

#define REDUCE_FP8_SUM(name, to_f32, from_f32) \
extern "C" __global__ void __launch_bounds__(256) k_sum_##name( \
    const uint8_t* __restrict__ in, float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out) { \
    float acc = 0.0f; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    if (blockIdx.x == 0 && tid == 0) { \
        unsigned* counter = (unsigned*)&partial[gridDim.x]; \
        *counter = 0u; \
    } \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) { \
        acc += to_f32(in[i]); \
    } \
    acc = warp_reduce_sum_f32(acc); \
    __shared__ float sdata_sum[32]; \
    if (tid % 32 == 0) sdata_sum[tid / 32] = acc; \
    __syncthreads(); \
    if (tid < 32) { \
        acc = (tid < (blockDim.x + 31) / 32) ? sdata_sum[tid] : 0.0f; \
        acc = warp_reduce_sum_f32(acc); \
    } \
    if (tid == 0) partial[blockIdx.x] = acc; \
    __threadfence(); \
    __shared__ bool is_last; \
    if (tid == 0) { \
        unsigned* counter = (unsigned*)&partial[gridDim.x]; \
        unsigned ticket = atomicInc(counter, gridDim.x); \
        is_last = (ticket == gridDim.x - 1); \
    } \
    __syncthreads(); \
    if (is_last) { \
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f; \
        val = warp_reduce_sum_f32(val); \
        if (blockDim.x > 32) { \
            if (tid % 32 == 0) sdata_sum[tid / 32] = val; \
            __syncthreads(); \
            if (tid < 32) { \
                val = (tid < (blockDim.x + 31) / 32) ? sdata_sum[tid] : 0.0f; \
                val = warp_reduce_sum_f32(val); \
            } \
        } \
        if (tid == 0) out[0] = from_f32(val); \
    } \
}

#define REDUCE_FP8_MAX(name, to_f32, from_f32) \
extern "C" __global__ void __launch_bounds__(256) k_max_##name( \
    const uint8_t* __restrict__ in, float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out) { \
    float neg_inf = -__int_as_float(0x7f800000); \
    float acc = neg_inf; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    if (blockIdx.x == 0 && tid == 0) { \
        unsigned* counter = (unsigned*)&partial[gridDim.x]; \
        *counter = 0u; \
    } \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) { \
        acc = fmaxf(acc, to_f32(in[i])); \
    } \
    acc = warp_reduce_max_f32(acc); \
    __shared__ float sdata_max[32]; \
    if (tid % 32 == 0) sdata_max[tid / 32] = acc; \
    __syncthreads(); \
    if (tid < 32) { \
        acc = (tid < (blockDim.x + 31) / 32) ? sdata_max[tid] : neg_inf; \
        acc = warp_reduce_max_f32(acc); \
    } \
    if (tid == 0) partial[blockIdx.x] = acc; \
    __threadfence(); \
    __shared__ bool is_last; \
    if (tid == 0) { \
        unsigned* counter = (unsigned*)&partial[gridDim.x]; \
        unsigned ticket = atomicInc(counter, gridDim.x); \
        is_last = (ticket == gridDim.x - 1); \
    } \
    __syncthreads(); \
    if (is_last) { \
        float val = (tid < gridDim.x) ? partial[tid] : neg_inf; \
        val = warp_reduce_max_f32(val); \
        if (blockDim.x > 32) { \
            if (tid % 32 == 0) sdata_max[tid / 32] = val; \
            __syncthreads(); \
            if (tid < 32) { \
                val = (tid < (blockDim.x + 31) / 32) ? sdata_max[tid] : neg_inf; \
                val = warp_reduce_max_f32(val); \
            } \
        } \
        if (tid == 0) out[0] = from_f32(val); \
    } \
}

#define REDUCE_FP8_MIN(name, to_f32, from_f32) \
extern "C" __global__ void __launch_bounds__(256) k_min_##name( \
    const uint8_t* __restrict__ in, float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out) { \
    float pos_inf = __int_as_float(0x7f800000); \
    float acc = pos_inf; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    if (blockIdx.x == 0 && tid == 0) { \
        unsigned* counter = (unsigned*)&partial[gridDim.x]; \
        *counter = 0u; \
    } \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) { \
        acc = fminf(acc, to_f32(in[i])); \
    } \
    acc = warp_reduce_min_f32(acc); \
    __shared__ float sdata_min[32]; \
    if (tid % 32 == 0) sdata_min[tid / 32] = acc; \
    __syncthreads(); \
    if (tid < 32) { \
        acc = (tid < (blockDim.x + 31) / 32) ? sdata_min[tid] : pos_inf; \
        acc = warp_reduce_min_f32(acc); \
    } \
    if (tid == 0) partial[blockIdx.x] = acc; \
    __threadfence(); \
    __shared__ bool is_last; \
    if (tid == 0) { \
        unsigned* counter = (unsigned*)&partial[gridDim.x]; \
        unsigned ticket = atomicInc(counter, gridDim.x); \
        is_last = (ticket == gridDim.x - 1); \
    } \
    __syncthreads(); \
    if (is_last) { \
        float val = (tid < gridDim.x) ? partial[tid] : pos_inf; \
        val = warp_reduce_min_f32(val); \
        if (blockDim.x > 32) { \
            if (tid % 32 == 0) sdata_min[tid / 32] = val; \
            __syncthreads(); \
            if (tid < 32) { \
                val = (tid < (blockDim.x + 31) / 32) ? sdata_min[tid] : pos_inf; \
                val = warp_reduce_min_f32(val); \
            } \
        } \
        if (tid == 0) out[0] = from_f32(val); \
    } \
}

REDUCE_FP8_SUM(fp8e4m3, fp8e4m3_to_f32, fp8e4m3_from_f32)
REDUCE_FP8_MAX(fp8e4m3, fp8e4m3_to_f32, fp8e4m3_from_f32)
REDUCE_FP8_MIN(fp8e4m3, fp8e4m3_to_f32, fp8e4m3_from_f32)
REDUCE_FP8_SUM(fp8e5m2, fp8e5m2_to_f32, fp8e5m2_from_f32)
REDUCE_FP8_MAX(fp8e5m2, fp8e5m2_to_f32, fp8e5m2_from_f32)
REDUCE_FP8_MIN(fp8e5m2, fp8e5m2_to_f32, fp8e5m2_from_f32)
REDUCE_FP8_SUM(fp4e2m1, fp4e2m1_to_f32, fp4e2m1_from_f32)
REDUCE_FP8_MAX(fp4e2m1, fp4e2m1_to_f32, fp4e2m1_from_f32)
REDUCE_FP8_MIN(fp4e2m1, fp4e2m1_to_f32, fp4e2m1_from_f32)

#undef REDUCE_FP8_SUM
#undef REDUCE_FP8_MAX
#undef REDUCE_FP8_MIN

UNARY_F64(neg,   _NEG)
UNARY_F64(recip, _RECIP_D)
UNARY_F64(exp,   exp)
UNARY_F64(ln,    log)
UNARY_F64(log1p, log1p)
UNARY_F64(sin,   sin)
UNARY_F64(cos,   cos)
UNARY_F64(tan,   tan)
UNARY_F64(tanh,  tanh)
UNARY_F64(sqrt,  sqrt)
UNARY_F64(abs,   fabs)
UNARY_F64(ceil,  ceil)
UNARY_F64(floor, floor)
UNARY_F64(round, round)
UNARY_F64(erf,   erf_approx_f64)
UNARY_F64(asin,  asin)
UNARY_F64(acos,  acos)
UNARY_F64(atan,  atan)
UNARY_F64(sinh,  sinh)
UNARY_F64(cosh,  cosh)
UNARY_F64(asinh, asinh)
UNARY_F64(acosh, acosh)
UNARY_F64(atanh, atanh)
UNARY_F64(log2,  log2)
UNARY_F64(log10, log10)

UNARY_F64(sigmoid,    sigmoid_f64)
UNARY_F64(silu,       silu_f64)
UNARY_F64(mish,       mish_f64)
UNARY_F64(leaky_relu, leaky_relu_f64)
UNARY_F64(elu,        elu_f64)
UNARY_F64(hardswish,  hardswish_f64)

// --- Backward activation kernels f64 ---
extern "C" __global__ __launch_bounds__(256) void k_relu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); out[i] = __ldg(&input[i]) > 0.0 ? g : 0.0; }
}
extern "C" __global__ __launch_bounds__(256) void k_leaky_relu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); out[i] = __ldg(&input[i]) > 0.0 ? g : 0.01*g; }
}
extern "C" __global__ __launch_bounds__(256) void k_elu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); double x = __ldg(&input[i]); out[i] = x > 0.0 ? g : g*exp(x); }
}
extern "C" __global__ __launch_bounds__(256) void k_gelu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        double g = __ldg(&grad[i]), x = __ldg(&input[i]);
        double cdf = 0.5 * (1.0 + erf_approx_f64(x * 0.7071067811865476));
        double pdf = exp(-0.5 * x * x) * 0.3989422804014327;
        out[i] = g * (cdf + x * pdf);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_abs_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); double x = __ldg(&input[i]); out[i] = x > 0.0 ? g : x < 0.0 ? -g : 0.0; }
}

BINARY_F64(add,  +)
BINARY_F64(sub,  -)
BINARY_F64(emul, *)
BINARY_F64(ediv, /)

extern "C" __global__ __launch_bounds__(256) void k_atan2_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = atan2(__ldg(&a[i]), __ldg(&b[i]));
}

extern "C" __global__ void k_axpy_f64(double* y, double alpha, const double* x, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) y[i] += alpha * x[i];
}
extern "C" __global__ void k_scale_f64(const double* in, double s, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = in[i] * s;
}
extern "C" __global__ void k_powf_f64(const double* in, double p, double* out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = pow(in[i], p);
}
extern "C" __global__ void k_fill_f64(double* out, double val, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = val;
}

extern "C" __global__ void k_transpose_f64(const double* in, double* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}

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

