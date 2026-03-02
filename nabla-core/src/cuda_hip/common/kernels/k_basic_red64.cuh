// k_basic_red64.cuh — f64 global reduction kernels (sum, max)

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

extern "C" __global__ __launch_bounds__(256) void k_expand_f32(float* __restrict__ out, const float* __restrict__ src, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) {
    unsigned i = THREAD_ID;
    unsigned n = dst_rows * dst_cols;
    if (i < n) {
        unsigned r = i / dst_cols, c = i % dst_cols;
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c;
        out[i] = __ldg(&src[sr * src_cols + sc]);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_expand_f64(double* __restrict__ out, const double* __restrict__ src, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) {
    unsigned i = THREAD_ID;
    unsigned n = dst_rows * dst_cols;
    if (i < n) {
        unsigned r = i / dst_cols, c = i % dst_cols;
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c;
        out[i] = __ldg(&src[sr * src_cols + sc]);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_expand_f16(__half* __restrict__ out, const __half* __restrict__ src, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) {
    unsigned i = THREAD_ID;
    unsigned n = dst_rows * dst_cols;
    if (i < n) {
        unsigned r = i / dst_cols, c = i % dst_cols;
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c;
        out[i] = src[sr * src_cols + sc];
    }
}
extern "C" __global__ __launch_bounds__(256) void k_expand_fp8e4m3(uint8_t* __restrict__ out, const uint8_t* __restrict__ src, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) {
    unsigned i = THREAD_ID;
    unsigned n = dst_rows * dst_cols;
    if (i < n) {
        unsigned r = i / dst_cols, c = i % dst_cols;
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c;
        out[i] = src[sr * src_cols + sc];
    }
}
extern "C" __global__ __launch_bounds__(256) void k_expand_fp8e5m2(uint8_t* __restrict__ out, const uint8_t* __restrict__ src, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) {
    unsigned i = THREAD_ID;
    unsigned n = dst_rows * dst_cols;
    if (i < n) {
        unsigned r = i / dst_cols, c = i % dst_cols;
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c;
        out[i] = src[sr * src_cols + sc];
    }
}
extern "C" __global__ __launch_bounds__(256) void k_expand_fp4e2m1(uint8_t* __restrict__ out, const uint8_t* __restrict__ src, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) {
    unsigned i = THREAD_ID;
    unsigned n = dst_rows * dst_cols;
    if (i < n) {
        unsigned r = i / dst_cols, c = i % dst_cols;
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c;
        out[i] = src[sr * src_cols + sc];
    }
}

// --- Fused MSE sum forward: sum((pred-target)^2) with last-block aggregation ---
extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_f32(
    const float* __restrict__ pred, const float* __restrict__ target,
    float* __restrict__ partial, unsigned n, float* __restrict__ out
) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; *ctr = 0u; }
    unsigned n4 = n / 4;
    const float4* p4 = (const float4*)pred;
    const float4* t4 = (const float4*)target;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n4; i += grid_stride) {
        float4 vp = p4[i], vt = t4[i];
        float d0 = vp.x - vt.x, d1 = vp.y - vt.y, d2 = vp.z - vt.z, d3 = vp.w - vt.w;
        acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    for (unsigned j = n4 * 4 + blockIdx.x * blockDim.x + tid; j < n; j += grid_stride) {
        float d = __ldg(&pred[j]) - __ldg(&target[j]); acc += d * d;
    }
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) { acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f; acc = warp_reduce_sum_f32(acc); }
    if (tid == 0) partial[blockIdx.x] = acc;
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; unsigned ticket = atomicInc(ctr, gridDim.x); is_last = (ticket == gridDim.x - 1); }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) { if (tid % 32 == 0) sdata[tid / 32] = val; __syncthreads(); if (tid < 32) { val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f; val = warp_reduce_sum_f32(val); } }
        if (tid == 0) out[0] = val;
    }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_f16(
    const __half* __restrict__ pred, const __half* __restrict__ target,
    float* __restrict__ partial, unsigned n, __half* __restrict__ out
) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; *ctr = 0u; }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        float d = from_half(pred[i]) - from_half(target[i]);
        acc += d * d;
    }
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata_f16[32];
    if (tid % 32 == 0) sdata_f16[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) { acc = (tid < (blockDim.x + 31) / 32) ? sdata_f16[tid] : 0.0f; acc = warp_reduce_sum_f32(acc); }
    if (tid == 0) partial[blockIdx.x] = acc;
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; unsigned ticket = atomicInc(ctr, gridDim.x); is_last = (ticket == gridDim.x - 1); }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) { if (tid % 32 == 0) sdata_f16[tid / 32] = val; __syncthreads(); if (tid < 32) { val = (tid < (blockDim.x + 31) / 32) ? sdata_f16[tid] : 0.0f; val = warp_reduce_sum_f32(val); } }
        if (tid == 0) out[0] = to_half(val);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_fp8e4m3(
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ target,
    float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out
) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; *ctr = 0u; }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        float d = fp8e4m3_to_f32(pred[i]) - fp8e4m3_to_f32(target[i]);
        acc += d * d;
    }
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata_fp8[32];
    if (tid % 32 == 0) sdata_fp8[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) { acc = (tid < (blockDim.x + 31) / 32) ? sdata_fp8[tid] : 0.0f; acc = warp_reduce_sum_f32(acc); }
    if (tid == 0) partial[blockIdx.x] = acc;
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; unsigned ticket = atomicInc(ctr, gridDim.x); is_last = (ticket == gridDim.x - 1); }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) { if (tid % 32 == 0) sdata_fp8[tid / 32] = val; __syncthreads(); if (tid < 32) { val = (tid < (blockDim.x + 31) / 32) ? sdata_fp8[tid] : 0.0f; val = warp_reduce_sum_f32(val); } }
        if (tid == 0) out[0] = fp8e4m3_from_f32(val);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_fp8e5m2(
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ target,
    float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out
) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; *ctr = 0u; }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        float d = fp8e5m2_to_f32(pred[i]) - fp8e5m2_to_f32(target[i]);
        acc += d * d;
    }
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata_fp8[32];
    if (tid % 32 == 0) sdata_fp8[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) { acc = (tid < (blockDim.x + 31) / 32) ? sdata_fp8[tid] : 0.0f; acc = warp_reduce_sum_f32(acc); }
    if (tid == 0) partial[blockIdx.x] = acc;
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; unsigned ticket = atomicInc(ctr, gridDim.x); is_last = (ticket == gridDim.x - 1); }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) { if (tid % 32 == 0) sdata_fp8[tid / 32] = val; __syncthreads(); if (tid < 32) { val = (tid < (blockDim.x + 31) / 32) ? sdata_fp8[tid] : 0.0f; val = warp_reduce_sum_f32(val); } }
        if (tid == 0) out[0] = fp8e5m2_from_f32(val);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_fp4e2m1(
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ target,
    float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out
) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; *ctr = 0u; }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        float d = fp4e2m1_to_f32(pred[i]) - fp4e2m1_to_f32(target[i]);
        acc += d * d;
    }
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata_fp8[32];
    if (tid % 32 == 0) sdata_fp8[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) { acc = (tid < (blockDim.x + 31) / 32) ? sdata_fp8[tid] : 0.0f; acc = warp_reduce_sum_f32(acc); }
    if (tid == 0) partial[blockIdx.x] = acc;
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; unsigned ticket = atomicInc(ctr, gridDim.x); is_last = (ticket == gridDim.x - 1); }
    __syncthreads();
    if (is_last) {
        float val = (tid < gridDim.x) ? partial[tid] : 0.0f;
        val = warp_reduce_sum_f32(val);
        if (blockDim.x > 32) { if (tid % 32 == 0) sdata_fp8[tid / 32] = val; __syncthreads(); if (tid < 32) { val = (tid < (blockDim.x + 31) / 32) ? sdata_fp8[tid] : 0.0f; val = warp_reduce_sum_f32(val); } }
        if (tid == 0) out[0] = fp4e2m1_from_f32(val);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_f64(
    const double* __restrict__ pred, const double* __restrict__ target,
    double* __restrict__ partial, unsigned n, double* __restrict__ out
) {
    double acc = 0.0;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; *ctr = 0u; }
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        double d = __ldg(&pred[i]) - __ldg(&target[i]); acc += d * d;
    }
    acc = warp_reduce_sum_f64(acc);
    __shared__ double sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) { acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0; acc = warp_reduce_sum_f64(acc); }
    if (tid == 0) partial[blockIdx.x] = acc;
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) { unsigned* ctr = (unsigned*)&partial[gridDim.x]; unsigned ticket = atomicInc(ctr, gridDim.x); is_last = (ticket == gridDim.x - 1); }
    __syncthreads();
    if (is_last) {
        double val = (tid < gridDim.x) ? partial[tid] : 0.0;
        val = warp_reduce_sum_f64(val);
        if (blockDim.x > 32) { if (tid % 32 == 0) sdata[tid / 32] = val; __syncthreads(); if (tid < 32) { val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0; val = warp_reduce_sum_f64(val); } }
        if (tid == 0) out[0] = val;
    }
}

// --- Fused MSE sum backward: out[i] = 2*(pred[i]-target[i])*grad_ptr[0] ---
extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_f32(
    const float* __restrict__ pred, const float* __restrict__ target,
    const float* __restrict__ grad_ptr, float* __restrict__ out, unsigned n
) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    float two_g = 2.0f * __ldg(grad_ptr);
    if (i + 3 < n) {
        float4 vp = LOAD_F4(pred, i4), vt = LOAD_F4(target, i4);
        float4 vo = make_float4((vp.x-vt.x)*two_g, (vp.y-vt.y)*two_g, (vp.z-vt.z)*two_g, (vp.w-vt.w)*two_g);
        STORE_F4(out, i4, vo);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = (__ldg(&pred[j]) - __ldg(&target[j])) * two_g; }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_f16(
    const __half* __restrict__ pred, const __half* __restrict__ target,
    const __half* __restrict__ grad_ptr, __half* __restrict__ out, unsigned n
) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = from_half(grad_ptr[0]);
        float d = from_half(pred[i]) - from_half(target[i]);
        out[i] = to_half(2.0f * d * g);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_fp8e4m3(
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ target,
    const uint8_t* __restrict__ grad_ptr, uint8_t* __restrict__ out, unsigned n
) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = fp8e4m3_to_f32(grad_ptr[0]);
        float d = fp8e4m3_to_f32(pred[i]) - fp8e4m3_to_f32(target[i]);
        out[i] = fp8e4m3_from_f32(2.0f * d * g);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_fp8e5m2(
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ target,
    const uint8_t* __restrict__ grad_ptr, uint8_t* __restrict__ out, unsigned n
) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = fp8e5m2_to_f32(grad_ptr[0]);
        float d = fp8e5m2_to_f32(pred[i]) - fp8e5m2_to_f32(target[i]);
        out[i] = fp8e5m2_from_f32(2.0f * d * g);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_fp4e2m1(
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ target,
    const uint8_t* __restrict__ grad_ptr, uint8_t* __restrict__ out, unsigned n
) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = fp4e2m1_to_f32(grad_ptr[0]);
        float d = fp4e2m1_to_f32(pred[i]) - fp4e2m1_to_f32(target[i]);
        out[i] = fp4e2m1_from_f32(2.0f * d * g);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_f64(
    const double* __restrict__ pred, const double* __restrict__ target,
    const double* __restrict__ grad_ptr, double* __restrict__ out, unsigned n
) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = 2.0 * (__ldg(&pred[i]) - __ldg(&target[i])) * __ldg(grad_ptr);
}

// --- Multi-param AXPY: y0+=a*x0, y1+=a*x1, y2+=a*x2 ---
extern "C" __global__ __launch_bounds__(256) void k_multi_axpy3_f32(
    float* __restrict__ y0, const float* __restrict__ x0, unsigned n0,
    float* __restrict__ y1, const float* __restrict__ x1, unsigned n1,
    float* __restrict__ y2, const float* __restrict__ x2, unsigned n2, float alpha
) {
    unsigned idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    unsigned total = (n0 > n1 ? n0 : n1); total = (total > n2 ? total : n2);
    if (idx + 3 < total) {
        if (idx + 3 < n0) { unsigned i4 = idx / 4; float4 vy = LOAD_F4(y0, i4), vx = LOAD_F4(x0, i4); vy.x += alpha*vx.x; vy.y += alpha*vx.y; vy.z += alpha*vx.z; vy.w += alpha*vx.w; STORE_F4(y0, i4, vy); }
        else { for (unsigned j = idx; j < n0 && j < idx+4; j++) y0[j] += alpha * __ldg(&x0[j]); }
        if (idx + 3 < n1) { unsigned i4 = idx / 4; float4 vy = LOAD_F4(y1, i4), vx = LOAD_F4(x1, i4); vy.x += alpha*vx.x; vy.y += alpha*vx.y; vy.z += alpha*vx.z; vy.w += alpha*vx.w; STORE_F4(y1, i4, vy); }
        else { for (unsigned j = idx; j < n1 && j < idx+4; j++) y1[j] += alpha * __ldg(&x1[j]); }
        if (idx + 3 < n2) { unsigned i4 = idx / 4; float4 vy = LOAD_F4(y2, i4), vx = LOAD_F4(x2, i4); vy.x += alpha*vx.x; vy.y += alpha*vx.y; vy.z += alpha*vx.z; vy.w += alpha*vx.w; STORE_F4(y2, i4, vy); }
        else { for (unsigned j = idx; j < n2 && j < idx+4; j++) y2[j] += alpha * __ldg(&x2[j]); }
    } else {
        for (unsigned j = idx; j < n0 && j < idx+4; j++) y0[j] += alpha * __ldg(&x0[j]);
        for (unsigned j = idx; j < n1 && j < idx+4; j++) y1[j] += alpha * __ldg(&x1[j]);
        for (unsigned j = idx; j < n2 && j < idx+4; j++) y2[j] += alpha * __ldg(&x2[j]);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_multi_axpy3_f16(
    __half* __restrict__ y0, const __half* __restrict__ x0, unsigned n0,
    __half* __restrict__ y1, const __half* __restrict__ x1, unsigned n1,
    __half* __restrict__ y2, const __half* __restrict__ x2, unsigned n2, __half alpha
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = (n0 > n1 ? n0 : n1);
    total = (total > n2 ? total : n2);
    float a = from_half(alpha);
    if (idx < total) {
        if (idx < n0) {
            float v = from_half(y0[idx]) + a * from_half(x0[idx]);
            y0[idx] = to_half(v);
        }
        if (idx < n1) {
            float v = from_half(y1[idx]) + a * from_half(x1[idx]);
            y1[idx] = to_half(v);
        }
        if (idx < n2) {
            float v = from_half(y2[idx]) + a * from_half(x2[idx]);
            y2[idx] = to_half(v);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void k_multi_axpy3_f64(
    double* __restrict__ y0, const double* __restrict__ x0, unsigned n0,
    double* __restrict__ y1, const double* __restrict__ x1, unsigned n1,
    double* __restrict__ y2, const double* __restrict__ x2, unsigned n2, double alpha
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = (n0 > n1 ? n0 : n1); total = (total > n2 ? total : n2);
    if (idx < total) {
        if (idx < n0) y0[idx] += alpha * __ldg(&x0[idx]);
        if (idx < n1) y1[idx] += alpha * __ldg(&x1[idx]);
        if (idx < n2) y2[idx] += alpha * __ldg(&x2[idx]);
    }
}
