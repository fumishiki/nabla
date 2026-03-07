// k_ops.cuh -- Elementwise + reduction kernels (merged from k_elementwise + k_reduction)
// Section 1: Elementwise (unary/binary/activation, all types including f64)
// Section 2: Reductions (sum/max/min, expand, MSE, multi_axpy3)

// ============================================================
// Section 1: Elementwise operations
// ============================================================

// --- Cast kernels ---
CAST_KERNEL(k_cast_f32_to_f16,     float,          __half,          to_half)
CAST_KERNEL(k_cast_f16_to_f32,     __half,          float,          from_half)
CAST_KERNEL(k_cast_f32_to_bf16,    float,          __nv_bfloat16,   to_bf16)
CAST_KERNEL(k_cast_bf16_to_f32,    __nv_bfloat16,   float,          from_bf16)
CAST_KERNEL(k_cast_f64_to_f32,     double,          float,          _F32_CAST)
CAST_KERNEL(k_cast_f32_to_f64,     float,           double,         _F64_CAST)
CAST_KERNEL(k_cast_f32_to_fp8e4m3, float,           uint8_t,        fp8e4m3_from_f32)
CAST_KERNEL(k_cast_fp8e4m3_to_f32, uint8_t,         float,          fp8e4m3_to_f32)
CAST_KERNEL(k_cast_f32_to_fp8e5m2, float,           uint8_t,        fp8e5m2_from_f32)
CAST_KERNEL(k_cast_fp8e5m2_to_f32, uint8_t,         float,          fp8e5m2_to_f32)
CAST_KERNEL(k_cast_f32_to_fp4e2m1, float,           uint8_t,        fp4e2m1_from_f32)
CAST_KERNEL(k_cast_fp4e2m1_to_f32, uint8_t,         float,          fp4e2m1_to_f32)

// --- Unary math ops (25 ops x 7 types = 175 kernels) ---
UNARY_ALL_TYPES(neg,   _NEG,            _NEG)
UNARY_ALL_TYPES(recip, _RECIP_F,        _RECIP_F)
UNARY_ALL_TYPES(exp,   __expf,          __expf)
UNARY_ALL_TYPES(ln,    __logf,          __logf)
UNARY_ALL_TYPES(log1p, _LOG1P_FAST,     log1pf)
UNARY_ALL_TYPES(sin,   __sinf,          __sinf)
UNARY_ALL_TYPES(cos,   __cosf,          __cosf)
UNARY_ALL_TYPES(tan,   tanf,            tanf)
UNARY_ALL_TYPES(tanh,  tanhf,           tanhf)
UNARY_ALL_TYPES(sqrt,  __fsqrt_rn,      sqrtf)
UNARY_ALL_TYPES(abs,   fabsf,           fabsf)
UNARY_ALL_TYPES(ceil,  ceilf,           ceilf)
UNARY_ALL_TYPES(floor, floorf,          floorf)
UNARY_ALL_TYPES(round, roundf,          roundf)
UNARY_ALL_TYPES(erf,   erf_approx_f32,  erf_approx_f32)
UNARY_ALL_TYPES(asin,  asinf,           asinf)
UNARY_ALL_TYPES(acos,  acosf,           acosf)
UNARY_ALL_TYPES(atan,  atanf,           atanf)
UNARY_ALL_TYPES(sinh,  sinhf,           sinhf)
UNARY_ALL_TYPES(cosh,  coshf,           coshf)
UNARY_ALL_TYPES(asinh, asinhf,          asinhf)
UNARY_ALL_TYPES(acosh, acoshf,          acoshf)
UNARY_ALL_TYPES(atanh, atanhf,          atanhf)
UNARY_ALL_TYPES(log2,  __log2f,         __log2f)
UNARY_ALL_TYPES(log10, __log10f,        __log10f)

// --- f64 unary math ops ---
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

// --- Activation unary ops (6 ops x 7 types = 42 kernels) ---
UNARY_ALL_TYPES(sigmoid,    sigmoid_f32,    sigmoid_f32)
UNARY_ALL_TYPES(silu,       silu_f32,       silu_f32)
UNARY_ALL_TYPES(mish,       mish_f32,       mish_f32)
UNARY_ALL_TYPES(leaky_relu, leaky_relu_f32, leaky_relu_f32)
UNARY_ALL_TYPES(elu,        elu_f32,        elu_f32)
UNARY_ALL_TYPES(hardswish,  hardswish_f32,  hardswish_f32)

// --- f64 activation unary ops ---
UNARY_F64(sigmoid,    sigmoid_f64)
UNARY_F64(silu,       silu_f64)
UNARY_F64(mish,       mish_f64)
UNARY_F64(leaky_relu, leaky_relu_f64)
UNARY_F64(elu,        elu_f64)
UNARY_F64(hardswish,  hardswish_f64)

// --- Backward activation kernels (f32 with vec4, f16/bf16/f64 via dispatch) ---
extern "C" __global__ __launch_bounds__(256) void k_relu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : 0.f, x.y > 0.f ? g.y : 0.f, x.z > 0.f ? g.z : 0.f, x.w > 0.f ? g.w : 0.f);
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); out[j] = __ldg(&input[j]) > 0.f ? gv : 0.f; } }
}
extern "C" __global__ __launch_bounds__(256) void k_leaky_relu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float alpha, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : alpha*g.x, x.y > 0.f ? g.y : alpha*g.y, x.z > 0.f ? g.z : alpha*g.z, x.w > 0.f ? g.w : alpha*g.w);
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); out[j] = __ldg(&input[j]) > 0.f ? gv : alpha*gv; } }
}
extern "C" __global__ __launch_bounds__(256) void k_elu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float alpha, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : g.x*alpha*__expf(x.x), x.y > 0.f ? g.y : g.y*alpha*__expf(x.y),
                               x.z > 0.f ? g.z : g.z*alpha*__expf(x.z), x.w > 0.f ? g.w : g.w*alpha*__expf(x.w));
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); float xv = __ldg(&input[j]); out[j] = xv > 0.f ? gv : gv*alpha*__expf(xv); } }
}
extern "C" __global__ __launch_bounds__(256) void k_gelu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = __ldg(&grad[i]), x = __ldg(&input[i]);
        float cdf = 0.5f * (1.0f + erf_approx_f32(x * 0.7071067811865476f));
        float pdf = __expf(-0.5f * x * x) * 0.3989422804014327f;
        out[i] = g * (cdf + x * pdf);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_abs_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : x.x < 0.f ? -g.x : 0.f, x.y > 0.f ? g.y : x.y < 0.f ? -g.y : 0.f,
                               x.z > 0.f ? g.z : x.z < 0.f ? -g.z : 0.f, x.w > 0.f ? g.w : x.w < 0.f ? -g.w : 0.f);
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); float xv = __ldg(&input[j]); out[j] = xv > 0.f ? gv : xv < 0.f ? -gv : 0.f; } }
}

BWD_ALL_CONV(f16, __half, from_half, to_half)
BWD_ALL_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16)

// --- f64 backward activation kernels ---
extern "C" __global__ __launch_bounds__(256) void k_relu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); out[i] = __ldg(&input[i]) > 0.0 ? g : 0.0; }
}
extern "C" __global__ __launch_bounds__(256) void k_leaky_relu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double alpha, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); out[i] = __ldg(&input[i]) > 0.0 ? g : alpha*g; }
}
extern "C" __global__ __launch_bounds__(256) void k_elu_bwd_f64(const double* __restrict__ grad, const double* __restrict__ input, double alpha, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) { double g = __ldg(&grad[i]); double x = __ldg(&input[i]); out[i] = x > 0.0 ? g : g*alpha*exp(x); }
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

// --- masked_fill and where (7 types each) ---
MASKED_FILL_ALL_TYPES()
WHERE_ALL_TYPES()

// --- Binary arithmetic ops (4 ops x 7 types = 28 kernels) ---
BINARY_ALL_TYPES(add,  +)
BINARY_ALL_TYPES(sub,  -)
BINARY_ALL_TYPES(emul, *)
BINARY_ALL_TYPES(ediv, /)

BINARY_F64(add,  +)
BINARY_F64(sub,  -)
BINARY_F64(emul, *)
BINARY_F64(ediv, /)

// --- atan2 (binary function, 7 types) ---
BINARY_FN_ALL_TYPES(atan2, atan2f)

extern "C" __global__ __launch_bounds__(256) void k_atan2_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID; if (i < n) out[i] = atan2(__ldg(&a[i]), __ldg(&b[i]));
}

// --- axpy, scale, powf, fill (7 types each) ---
AXPY_ALL_TYPES()
SCALE_ALL_TYPES()
POWF_ALL_TYPES()
FILL_ALL_TYPES()

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

// --- transpose (7 types) ---
TRANSPOSE_ALL_TYPES()

extern "C" __global__ void k_transpose_f64(const double* in, double* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}

// --- matmul tiled GEMM (7 types) ---
MATMUL_ALL_TYPES()

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

// ============================================================
// Section 2: Reduction operations
// ============================================================

// --- Ops-local reduction macros ---

#define REDUCE_NARROW(op_name, name, T, from_fn, to_fn, init_expr, acc_update, warp_fn) \
extern "C" __global__ void __launch_bounds__(256) k_##op_name##_##name( \
    const T* __restrict__ in, float* __restrict__ partial, \
    unsigned n, T* __restrict__ out) { \
    float acc = (init_expr); \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) \
        acc_update; \
    __shared__ float sdata[32]; \
    FULL_REDUCE_F32(acc, warp_fn, (init_expr), partial, sdata, tid, out, to_fn); \
}

#define EXPAND_TYPED(name, T) \
extern "C" __global__ __launch_bounds__(256) void k_expand_##name( \
    T* __restrict__ out, const T* __restrict__ src, \
    unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned dst_cols) { \
    unsigned i = THREAD_ID; \
    unsigned n = dst_rows * dst_cols; \
    if (i < n) { \
        unsigned r = i / dst_cols, c = i % dst_cols; \
        unsigned sr = src_rows == 1 ? 0 : r, sc = src_cols == 1 ? 0 : c; \
        out[i] = src[sr * src_cols + sc]; \
    } \
}

#define MSE_SUM_BWD_NARROW(name, T, to_f32, from_f32) \
extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_##name( \
    const T* __restrict__ pred, const T* __restrict__ target, \
    const T* __restrict__ grad_ptr, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float g = to_f32(grad_ptr[0]); \
        float d = to_f32(pred[i]) - to_f32(target[i]); \
        out[i] = from_f32(2.0f * d * g); \
    } \
}

#define REDUCE_F64_VEC2(kernel_name, acc_init, acc_vec_op, acc_scalar_op, warp_fn, identity) \
extern "C" __global__ void kernel_name(const double* __restrict__ in, \
    double* __restrict__ partial, unsigned n, double* __restrict__ out) { \
    double acc = acc_init; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    unsigned n2 = n / 2; \
    const double2* in2 = (const double2*)in; \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n2; i += grid_stride) { \
        double2 v = in2[i]; \
        acc_vec_op; \
    } \
    for (unsigned i = n2 * 2 + blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) \
        acc_scalar_op; \
    __shared__ double sdata[32]; \
    FULL_REDUCE_F64(acc, warp_fn, identity, partial, sdata, tid, out, _IDENTITY); \
}

#define MULTI_AXPY3_NARROW(name, T, from_fn, to_fn) \
extern "C" __global__ __launch_bounds__(256) void k_multi_axpy3_##name( \
    T* __restrict__ y0, const T* __restrict__ x0, unsigned n0, \
    T* __restrict__ y1, const T* __restrict__ x1, unsigned n1, \
    T* __restrict__ y2, const T* __restrict__ x2, unsigned n2, T alpha) { \
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned total = (n0 > n1 ? n0 : n1); \
    total = (total > n2 ? total : n2); \
    float a = from_fn(alpha); \
    if (idx < total) { \
        if (idx < n0) { float v = from_fn(y0[idx]) + a * from_fn(x0[idx]); y0[idx] = to_fn(v); } \
        if (idx < n1) { float v = from_fn(y1[idx]) + a * from_fn(x1[idx]); y1[idx] = to_fn(v); } \
        if (idx < n2) { float v = from_fn(y2[idx]) + a * from_fn(x2[idx]); y2[idx] = to_fn(v); } \
    } \
}

// ============================================================
// f32 vectorized reductions (hand-optimized, not macro-izable)
// ============================================================

extern "C" __global__ void __launch_bounds__(256) k_sum_f32(
    const float* __restrict__ in,
    float* __restrict__ partial,
    unsigned n,
    float* __restrict__ out) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    unsigned n4 = n / 4;
    const float4* in4 = (const float4*)in;
    unsigned i = blockIdx.x * blockDim.x + tid;
    unsigned stride4 = grid_stride * 4;
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

    __shared__ float sdata[32];
    FULL_REDUCE_F32(acc, warp_reduce_sum_f32, 0.0f, partial, sdata, tid, out, _IDENTITY);
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

    __shared__ float sdata[32];
    FULL_REDUCE_F32(acc, warp_reduce_max_f32, neg_inf, partial, sdata, tid, out, _IDENTITY);
}

extern "C" __global__ void k_min_f32(const float* __restrict__ in,
                                      float* __restrict__ partial,
                                      unsigned n,
                                      float* __restrict__ out) {
    float pos_inf = __int_as_float(0x7f800000);
    float acc = pos_inf;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    const float4* in4 = (const float4*)in;
    unsigned n4 = n / 4;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n4; i += grid_stride) {
        float4 v = in4[i];
        acc = fminf(acc, fminf(fminf(v.x, v.y), fminf(v.z, v.w)));
    }
    for (unsigned i = n4 * 4 + blockIdx.x * blockDim.x + tid; i < n; i += grid_stride)
        acc = fminf(acc, in[i]);

    __shared__ float sdata[32];
    FULL_REDUCE_F32(acc, warp_reduce_min_f32, pos_inf, partial, sdata, tid, out, _IDENTITY);
}

// ============================================================
// f16/bf16 reductions (via unified REDUCE_NARROW macro)
// ============================================================

REDUCE_NARROW(sum, f16,  __half,        from_half, to_half, 0.0f,                              acc += from_half(in[i]),             warp_reduce_sum_f32)
REDUCE_NARROW(sum, bf16, __nv_bfloat16, from_bf16, to_bf16, 0.0f,                              acc += from_bf16(in[i]),             warp_reduce_sum_f32)
REDUCE_NARROW(max, f16,  __half,        from_half, to_half, -__int_as_float(0x7f800000),        acc = fmaxf(acc, from_half(in[i])),  warp_reduce_max_f32)
REDUCE_NARROW(max, bf16, __nv_bfloat16, from_bf16, to_bf16, -__int_as_float(0x7f800000),        acc = fmaxf(acc, from_bf16(in[i])),  warp_reduce_max_f32)
REDUCE_NARROW(min, f16,  __half,        from_half, to_half, __int_as_float(0x7f800000),          acc = fminf(acc, from_half(in[i])),  warp_reduce_min_f32)
REDUCE_NARROW(min, bf16, __nv_bfloat16, from_bf16, to_bf16, __int_as_float(0x7f800000),          acc = fminf(acc, from_bf16(in[i])),  warp_reduce_min_f32)

// ============================================================
// fp8/fp4 reductions (via REDUCE_FP8 macros from k_defs.cuh)
// ============================================================

REDUCE_FP8_SUM(fp8e4m3, fp8e4m3_to_f32, fp8e4m3_from_f32)
REDUCE_FP8_MAX(fp8e4m3, fp8e4m3_to_f32, fp8e4m3_from_f32)
REDUCE_FP8_MIN(fp8e4m3, fp8e4m3_to_f32, fp8e4m3_from_f32)
REDUCE_FP8_SUM(fp8e5m2, fp8e5m2_to_f32, fp8e5m2_from_f32)
REDUCE_FP8_MAX(fp8e5m2, fp8e5m2_to_f32, fp8e5m2_from_f32)
REDUCE_FP8_MIN(fp8e5m2, fp8e5m2_to_f32, fp8e5m2_from_f32)
REDUCE_FP8_SUM(fp4e2m1, fp4e2m1_to_f32, fp4e2m1_from_f32)
REDUCE_FP8_MAX(fp4e2m1, fp4e2m1_to_f32, fp4e2m1_from_f32)
REDUCE_FP8_MIN(fp4e2m1, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ============================================================
// f64 reductions (vectorized double2, via REDUCE_F64_VEC2 macro)
// ============================================================

REDUCE_F64_VEC2(k_sum_f64, 0.0,
    acc += v.x + v.y, acc += in[i],
    warp_reduce_sum_f64, 0.0)

REDUCE_F64_VEC2(k_max_f64, __longlong_as_double(0xFFF0000000000000LL),
    acc = fmax(acc, fmax(v.x, v.y)), acc = fmax(acc, in[i]),
    warp_reduce_max_f64, __longlong_as_double(0xFFF0000000000000LL))

REDUCE_F64_VEC2(k_min_f64, __longlong_as_double(0x7FF0000000000000LL),
    acc = fmin(acc, fmin(v.x, v.y)), acc = fmin(acc, in[i]),
    warp_reduce_min_f64, __longlong_as_double(0x7FF0000000000000LL))

// ============================================================
// Expand (broadcast) kernels -- all types via EXPAND_TYPED
// ============================================================

EXPAND_TYPED(f32,      float)
EXPAND_TYPED(f64,      double)
EXPAND_TYPED(f16,      __half)
EXPAND_TYPED(bf16,     __nv_bfloat16)
EXPAND_TYPED(fp8e4m3,  uint8_t)
EXPAND_TYPED(fp8e5m2,  uint8_t)
EXPAND_TYPED(fp4e2m1,  uint8_t)

// ============================================================
// Fused MSE sum forward
// ============================================================

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_f32(
    const float* __restrict__ pred, const float* __restrict__ target,
    float* __restrict__ partial, unsigned n, float* __restrict__ out) {
    float acc = 0.0f;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
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
    __shared__ float sdata[32];
    FULL_REDUCE_F32(acc, warp_reduce_sum_f32, 0.0f, partial, sdata, tid, out, _IDENTITY);
}

MSE_SUM_FWD(k_mse_sum_fwd_f16,      __half,          __half,          from_half,       to_half,          float, warp_reduce_sum_f32, 0.0f, FULL_REDUCE_F32)
MSE_SUM_FWD(k_mse_sum_fwd_bf16,     __nv_bfloat16,   __nv_bfloat16,   from_bf16,       to_bf16,          float, warp_reduce_sum_f32, 0.0f, FULL_REDUCE_F32)
MSE_SUM_FWD(k_mse_sum_fwd_fp8e4m3,  uint8_t,         uint8_t,         fp8e4m3_to_f32,  fp8e4m3_from_f32, float, warp_reduce_sum_f32, 0.0f, FULL_REDUCE_F32)
MSE_SUM_FWD(k_mse_sum_fwd_fp8e5m2,  uint8_t,         uint8_t,         fp8e5m2_to_f32,  fp8e5m2_from_f32, float, warp_reduce_sum_f32, 0.0f, FULL_REDUCE_F32)
MSE_SUM_FWD(k_mse_sum_fwd_fp4e2m1,  uint8_t,         uint8_t,         fp4e2m1_to_f32,  fp4e2m1_from_f32, float, warp_reduce_sum_f32, 0.0f, FULL_REDUCE_F32)

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_fwd_f64(
    const double* __restrict__ pred, const double* __restrict__ target,
    double* __restrict__ partial, unsigned n, double* __restrict__ out) {
    double acc = 0.0;
    unsigned tid = threadIdx.x;
    unsigned grid_stride = blockDim.x * gridDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) {
        double d = __ldg(&pred[i]) - __ldg(&target[i]); acc += d * d;
    }
    __shared__ double sdata[32];
    FULL_REDUCE_F64(acc, warp_reduce_sum_f64, 0.0, partial, sdata, tid, out, _IDENTITY);
}

// ============================================================
// Fused MSE sum backward
// ============================================================

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_f32(
    const float* __restrict__ pred, const float* __restrict__ target,
    const float* __restrict__ grad_ptr, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    float two_g = 2.0f * __ldg(grad_ptr);
    if (i + 3 < n) {
        float4 vp = LOAD_F4(pred, i4), vt = LOAD_F4(target, i4);
        float4 vo = make_float4((vp.x-vt.x)*two_g, (vp.y-vt.y)*two_g, (vp.z-vt.z)*two_g, (vp.w-vt.w)*two_g);
        STORE_F4(out, i4, vo);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = (__ldg(&pred[j]) - __ldg(&target[j])) * two_g; }
}

MSE_SUM_BWD_NARROW(f16,      __half,          from_half,       to_half)
MSE_SUM_BWD_NARROW(bf16,     __nv_bfloat16,   from_bf16,       to_bf16)
MSE_SUM_BWD_NARROW(fp8e4m3,  uint8_t,         fp8e4m3_to_f32,  fp8e4m3_from_f32)
MSE_SUM_BWD_NARROW(fp8e5m2,  uint8_t,         fp8e5m2_to_f32,  fp8e5m2_from_f32)
MSE_SUM_BWD_NARROW(fp4e2m1,  uint8_t,         fp4e2m1_to_f32,  fp4e2m1_from_f32)

extern "C" __global__ __launch_bounds__(256) void k_mse_sum_bwd_f64(
    const double* __restrict__ pred, const double* __restrict__ target,
    const double* __restrict__ grad_ptr, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = 2.0 * (__ldg(&pred[i]) - __ldg(&target[i])) * __ldg(grad_ptr);
}

// ============================================================
// Multi-param AXPY3
// ============================================================

extern "C" __global__ __launch_bounds__(256) void k_multi_axpy3_f32(
    float* __restrict__ y0, const float* __restrict__ x0, unsigned n0,
    float* __restrict__ y1, const float* __restrict__ x1, unsigned n1,
    float* __restrict__ y2, const float* __restrict__ x2, unsigned n2, float alpha) {
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

MULTI_AXPY3_NARROW(f16,  __half,          from_half, to_half)
MULTI_AXPY3_NARROW(bf16, __nv_bfloat16,   from_bf16, to_bf16)

extern "C" __global__ __launch_bounds__(256) void k_multi_axpy3_f64(
    double* __restrict__ y0, const double* __restrict__ x0, unsigned n0,
    double* __restrict__ y1, const double* __restrict__ x1, unsigned n1,
    double* __restrict__ y2, const double* __restrict__ x2, unsigned n2, double alpha) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = (n0 > n1 ? n0 : n1); total = (total > n2 ? total : n2);
    if (idx < total) {
        if (idx < n0) y0[idx] += alpha * __ldg(&x0[idx]);
        if (idx < n1) y1[idx] += alpha * __ldg(&x1[idx]);
        if (idx < n2) y2[idx] += alpha * __ldg(&x2[idx]);
    }
}

// ============================================================
// Section 3: Misc ops (repeat, pad, triu/tril, roll, flip, from_diag, meshgrid, kron)
// ============================================================

#define REPEAT_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_repeat_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, \
    unsigned in_rows, unsigned in_cols, unsigned out_rows, unsigned out_cols) { \
    unsigned i = THREAD_ID; \
    unsigned total = out_rows * out_cols; \
    if (i >= total) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    unsigned src_r = r % in_rows; \
    unsigned src_c = c % in_cols; \
    out[i] = a[src_r * in_cols + src_c]; \
}

#define PAD_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_pad_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, \
    unsigned in_rows, unsigned in_cols, unsigned out_cols, \
    unsigned left, unsigned right, unsigned top, unsigned bottom, T val) { \
    unsigned i = THREAD_ID; \
    unsigned out_rows = in_rows + top + bottom; \
    unsigned out_cols_l = out_cols; \
    unsigned total = out_rows * out_cols_l; \
    if (i >= total) return; \
    unsigned r = i / out_cols_l; \
    unsigned c = i - r * out_cols_l; \
    if (r >= top && r < top + in_rows && c >= left && c < left + in_cols) { \
        unsigned src_r = r - top; \
        unsigned src_c = c - left; \
        out[i] = a[src_r * in_cols + src_c]; \
    } else { \
        out[i] = val; \
    } \
}

#define TRIU_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_triu_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, unsigned rows, unsigned cols, int diag) { \
    unsigned i = THREAD_ID; \
    unsigned total = rows * cols; \
    if (i >= total) return; \
    unsigned r = i / cols; \
    unsigned c = i - r * cols; \
    out[i] = ((int)c >= (int)r + diag) ? a[i] : (T)0; \
}

#define TRIL_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_tril_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, unsigned rows, unsigned cols, int diag) { \
    unsigned i = THREAD_ID; \
    unsigned total = rows * cols; \
    if (i >= total) return; \
    unsigned r = i / cols; \
    unsigned c = i - r * cols; \
    out[i] = ((int)c <= (int)r + diag) ? a[i] : (T)0; \
}

#define ROLL_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_roll_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, \
    unsigned rows, unsigned cols, int shift, unsigned axis) { \
    unsigned i = THREAD_ID; \
    unsigned total = rows * cols; \
    if (i >= total) return; \
    unsigned r = i / cols; \
    unsigned c = i - r * cols; \
    if (axis == 0) { \
        int dim = (int)rows; \
        int s = shift % dim; \
        if (s < 0) s += dim; \
        unsigned src_r = (unsigned)((((int)r - s) % dim + dim) % dim); \
        out[i] = a[src_r * cols + c]; \
    } else { \
        int dim = (int)cols; \
        int s = shift % dim; \
        if (s < 0) s += dim; \
        unsigned src_c = (unsigned)((((int)c - s) % dim + dim) % dim); \
        out[i] = a[r * cols + src_c]; \
    } \
}

#define FLIP_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_flip_##suffix( \
    const T* __restrict__ a, T* __restrict__ out, \
    unsigned rows, unsigned cols, unsigned axis) { \
    unsigned i = THREAD_ID; \
    unsigned total = rows * cols; \
    if (i >= total) return; \
    unsigned r = i / cols; \
    unsigned c = i - r * cols; \
    if (axis == 0) { \
        unsigned src_r = rows - 1 - r; \
        out[i] = a[src_r * cols + c]; \
    } else { \
        unsigned src_c = cols - 1 - c; \
        out[i] = a[r * cols + src_c]; \
    } \
}

#define FROM_DIAG_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_from_diag_##suffix( \
    const T* __restrict__ v, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    unsigned total = n * n; \
    if (i >= total) return; \
    unsigned r = i / n; \
    unsigned c = i - r * n; \
    out[i] = (r == c) ? v[r] : (T)0; \
}

#define MESHGRID_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_meshgrid_##suffix( \
    const T* __restrict__ x, const T* __restrict__ y, \
    T* __restrict__ out_x, T* __restrict__ out_y, \
    unsigned nx, unsigned ny, unsigned out_cols) { \
    unsigned i = THREAD_ID; \
    unsigned total = nx * ny; \
    if (i >= total) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    out_x[i] = x[c]; \
    out_y[i] = y[r]; \
}

#define KRON_KERNEL(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_kron_##suffix( \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, \
    unsigned m, unsigned n, unsigned p, unsigned q, unsigned out_cols) { \
    unsigned i = THREAD_ID; \
    unsigned total = (m * p) * (n * q); \
    if (i >= total) return; \
    unsigned r = i / out_cols; \
    unsigned c = i - r * out_cols; \
    unsigned ar = r / p; \
    unsigned ac = c / q; \
    unsigned br = r - ar * p; \
    unsigned bc = c - ac * q; \
    out[i] = a[ar * n + ac] * b[br * q + bc]; \
}

REPEAT_KERNEL(f32, float)
REPEAT_KERNEL(f64, double)

PAD_KERNEL(f32, float)
PAD_KERNEL(f64, double)

TRIU_KERNEL(f32, float)
TRIU_KERNEL(f64, double)

TRIL_KERNEL(f32, float)
TRIL_KERNEL(f64, double)

ROLL_KERNEL(f32, float)
ROLL_KERNEL(f64, double)

FLIP_KERNEL(f32, float)
FLIP_KERNEL(f64, double)

FROM_DIAG_KERNEL(f32, float)
FROM_DIAG_KERNEL(f64, double)

MESHGRID_KERNEL(f32, float)
MESHGRID_KERNEL(f64, double)

KRON_KERNEL(f32, float)
KRON_KERNEL(f64, double)

// Cleanup ops-local macros
#undef REDUCE_NARROW
#undef REDUCE_F64_VEC2
#undef EXPAND_TYPED
#undef MSE_SUM_BWD_NARROW
#undef MULTI_AXPY3_NARROW
#undef REPEAT_KERNEL
#undef PAD_KERNEL
#undef TRIU_KERNEL
#undef TRIL_KERNEL
#undef ROLL_KERNEL
#undef FLIP_KERNEL
#undef FROM_DIAG_KERNEL
#undef MESHGRID_KERNEL
#undef KRON_KERNEL
