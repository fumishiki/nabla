// k_defs.cuh — Shared macros, type definitions, device helpers, and reduce primitives.
// Must be first in KERNELS concat order. All other .cuh files depend on definitions here.

// ============================================================
// === Infrastructure ===
// ============================================================

#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define TILE 16

#define LOAD_F4(ptr, i)  __ldg(((const float4*)(ptr)) + (i))
#define STORE_F4(ptr, i, v) (((float4*)(ptr))[i] = (v))
#define VEC4_IDX (blockIdx.x * blockDim.x + threadIdx.x)

#ifndef __nabla_stdint_defined__
#define __nabla_stdint_defined__
typedef unsigned char      uint8_t;
typedef signed char        int8_t;
typedef unsigned short     uint16_t;
typedef signed short       int16_t;
typedef unsigned int       uint32_t;
typedef signed int         int32_t;
typedef unsigned long long uint64_t;
typedef signed long long   int64_t;
#endif
#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#endif

// ============================================================
// === Type Conversion Helpers ===
// ============================================================

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ __half to_half(float v) { return __float2half(v); }
__device__ __forceinline__ float from_half(__half v) { return __half2float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf16(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float from_bf16(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

__device__ __forceinline__ uint8_t fp8e4m3_from_f32(float v) {
    const int M = 3, BIAS = 7, EMIN = -6, EMAX = 7;
    if (v == 0.0f) return 0;
    uint8_t sign = v < 0.0f;
    float av = fabsf(v);
    if (!isfinite(av)) {
        uint8_t exp_bits = (uint8_t)(EMAX + BIAS);
        uint8_t mant = (uint8_t)((1 << M) - 1);
        return (uint8_t)((sign << 7) | (exp_bits << M) | mant);
    }
    int exp = (int)floorf(log2f(av));
    if (exp < EMIN) return 0;
    int mant_bits = 0;
    if (exp > EMAX) {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        float base = ldexpf(1.0f, exp);
        float mant = av / base - 1.0f;
        if (mant < 0.0f) mant = 0.0f;
        int mant_i = (int)floorf(mant * (float)(1 << M) + 0.5f);
        if (mant_i >= (1 << M)) {
            mant_i = 0;
            exp += 1;
            if (exp > EMAX) { exp = EMAX; mant_i = (1 << M) - 1; }
        }
        mant_bits = mant_i;
    }
    uint8_t exp_bits = (uint8_t)(exp + BIAS);
    return (uint8_t)((sign << 7) | (exp_bits << M) | (uint8_t)mant_bits);
}

__device__ __forceinline__ float fp8e4m3_to_f32(uint8_t bits) {
    const int M = 3, BIAS = 7;
    if (bits == 0) return 0.0f;
    uint8_t sign = (bits >> 7) & 1;
    int exp_bits = (int)((bits >> M) & 0x0f);
    int mant_bits = (int)(bits & ((1 << M) - 1));
    int exp = exp_bits - BIAS;
    float mant = 1.0f + (float)mant_bits / (float)(1 << M);
    float v = ldexpf(mant, exp);
    return sign ? -v : v;
}

__device__ __forceinline__ uint8_t fp8e5m2_from_f32(float v) {
    const int M = 2, BIAS = 15, EMIN = -14, EMAX = 15;
    if (v == 0.0f) return 0;
    uint8_t sign = v < 0.0f;
    float av = fabsf(v);
    if (!isfinite(av)) {
        uint8_t exp_bits = (uint8_t)(EMAX + BIAS);
        uint8_t mant = (uint8_t)((1 << M) - 1);
        return (uint8_t)((sign << 7) | (exp_bits << M) | mant);
    }
    int exp = (int)floorf(log2f(av));
    if (exp < EMIN) return 0;
    int mant_bits = 0;
    if (exp > EMAX) {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        float base = ldexpf(1.0f, exp);
        float mant = av / base - 1.0f;
        if (mant < 0.0f) mant = 0.0f;
        int mant_i = (int)floorf(mant * (float)(1 << M) + 0.5f);
        if (mant_i >= (1 << M)) {
            mant_i = 0;
            exp += 1;
            if (exp > EMAX) { exp = EMAX; mant_i = (1 << M) - 1; }
        }
        mant_bits = mant_i;
    }
    uint8_t exp_bits = (uint8_t)(exp + BIAS);
    return (uint8_t)((sign << 7) | (exp_bits << M) | (uint8_t)mant_bits);
}

__device__ __forceinline__ float fp8e5m2_to_f32(uint8_t bits) {
    const int M = 2, BIAS = 15;
    if (bits == 0) return 0.0f;
    uint8_t sign = (bits >> 7) & 1;
    int exp_bits = (int)((bits >> M) & 0x1f);
    int mant_bits = (int)(bits & ((1 << M) - 1));
    int exp = exp_bits - BIAS;
    float mant = 1.0f + (float)mant_bits / (float)(1 << M);
    float v = ldexpf(mant, exp);
    return sign ? -v : v;
}

__device__ __forceinline__ uint8_t fp4e2m1_from_f32(float v) {
    const int M = 1, BIAS = 1, EMIN = -1, EMAX = 2;
    if (v == 0.0f) return 0;
    uint8_t sign = v < 0.0f;
    float av = fabsf(v);
    if (!isfinite(av)) {
        uint8_t exp_bits = (uint8_t)(EMAX + BIAS);
        uint8_t mant = (uint8_t)((1 << M) - 1);
        return (uint8_t)(((sign << 3) | (exp_bits << M) | mant) & 0x0f);
    }
    int exp = (int)floorf(log2f(av));
    if (exp < EMIN) return 0;
    int mant_bits = 0;
    if (exp > EMAX) {
        exp = EMAX;
        mant_bits = (1 << M) - 1;
    } else {
        float base = ldexpf(1.0f, exp);
        float mant = av / base - 1.0f;
        if (mant < 0.0f) mant = 0.0f;
        int mant_i = (int)floorf(mant * (float)(1 << M) + 0.5f);
        if (mant_i >= (1 << M)) {
            mant_i = 0;
            exp += 1;
            if (exp > EMAX) { exp = EMAX; mant_i = (1 << M) - 1; }
        }
        mant_bits = mant_i;
    }
    uint8_t exp_bits = (uint8_t)(exp + BIAS);
    return (uint8_t)(((sign << 3) | (exp_bits << M) | (uint8_t)mant_bits) & 0x0f);
}

__device__ __forceinline__ float fp4e2m1_to_f32(uint8_t bits) {
    const int M = 1, BIAS = 1;
    uint8_t b = bits & 0x0f;
    if (b == 0) return 0.0f;
    uint8_t sign = (b >> 3) & 1;
    int exp_bits = (int)((b >> M) & 0x03);
    int mant_bits = (int)(b & ((1 << M) - 1));
    int exp = exp_bits - BIAS;
    float mant = 1.0f + (float)mant_bits / (float)(1 << M);
    float v = ldexpf(mant, exp);
    return sign ? -v : v;
}

// ============================================================
// === Device Helper Functions ===
// ============================================================

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

// ============================================================
// === Constants ===
// ============================================================

#define _NEG_INF_F32 (-__int_as_float(0x7f800000))
#define _NEG_INF_F64 (__longlong_as_double(0xFFF0000000000000LL))
#define _IDENTITY(x) (x)
#define _INV_SQRT_F32(x) (1.0f / sqrtf(x))
#define _RSQRT_F32(x) rsqrtf(x)

// FlashAttention tile sizes
#define FA_BLOCK_M 64
#define FA_BLOCK_N 64
#define FA_HEAD_DIM_MAX 128

// ============================================================
// === Scalar Op Helpers ===
// ============================================================

#define _NEG(x) (-(x))
#define _RECIP_F(x) (1.0f/(x))
#define _RECIP_D(x) (1.0/(x))
#define _LOG1P_FAST(x) (__logf(1.0f+(x)))

#define CAST_KERNEL(name, IN, OUT, conv) \
extern "C" __global__ __launch_bounds__(256) void name(const IN* __restrict__ in, OUT* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = conv(in[i]); \
}
#define _F32_CAST(x)  ((float)(x))
#define _F64_CAST(x)  ((double)(x))
#define MATMUL_F32_IDENTITY(x) (x)

// ============================================================
// === Per-Type Kernel Template Macros ===
// ============================================================

// ---- Unary ----

#define UNARY_F32(name, vop, sop) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f32(const float* __restrict__ in, float* __restrict__ out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 v = LOAD_F4(in, i4); \
        v.x = vop(v.x); v.y = vop(v.y); v.z = vop(v.z); v.w = vop(v.w); \
        STORE_F4(out, i4, v); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = sop(__ldg(&in[j])); } \
}

#define UNARY_F64(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f64(const double* __restrict__ in, double* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = op(__ldg(&in[i])); \
}

#define UNARY_F16(name, vop, sop) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f16(const __half* __restrict__ in, __half* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float x = from_half(in[i]); \
        out[i] = to_half(sop(x)); \
    } \
}

#define UNARY_BF16(name, vop, sop) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_bf16(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float x = from_bf16(in[i]); \
        out[i] = to_bf16(sop(x)); \
    } \
}

#define UNARY_U8(suffix, name, op, to_f32, from_f32) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_##suffix(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { out[i] = from_f32(op(to_f32(in[i]))); } \
}
#define UNARY_FP8E4M3(name, op) UNARY_U8(fp8e4m3, name, op, fp8e4m3_to_f32, fp8e4m3_from_f32)
#define UNARY_FP8E5M2(name, op) UNARY_U8(fp8e5m2, name, op, fp8e5m2_to_f32, fp8e5m2_from_f32)
#define UNARY_FP4E2M1(name, op) UNARY_U8(fp4e2m1, name, op, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ---- Binary ----

#define BINARY_F32(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4); \
        float4 vo = make_float4(va.x op vb.x, va.y op vb.y, va.z op vb.z, va.w op vb.w); \
        STORE_F4(out, i4, vo); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = __ldg(&a[j]) op __ldg(&b[j]); } \
}

#define BINARY_F64(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = __ldg(&a[i]) op __ldg(&b[i]); \
}

#define BINARY_F16(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float ax = from_half(a[i]); \
        float bx = from_half(b[i]); \
        out[i] = to_half(ax op bx); \
    } \
}

#define BINARY_BF16(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float ax = from_bf16(a[i]); \
        float bx = from_bf16(b[i]); \
        out[i] = to_bf16(ax op bx); \
    } \
}

#define BINARY_U8(suffix, name, op, to_f32, from_f32) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_##suffix(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { out[i] = from_f32(to_f32(a[i]) op to_f32(b[i])); } \
}
#define BINARY_FP8E4M3(name, op) BINARY_U8(fp8e4m3, name, op, fp8e4m3_to_f32, fp8e4m3_from_f32)
#define BINARY_FP8E5M2(name, op) BINARY_U8(fp8e5m2, name, op, fp8e5m2_to_f32, fp8e5m2_from_f32)
#define BINARY_FP4E2M1(name, op) BINARY_U8(fp4e2m1, name, op, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ============================================================
// === Type Dispatch Meta-Macros ===
// ============================================================

#define UNARY_ALL_TYPES(name, vop, sop) \
    UNARY_F32(name, vop, sop) \
    UNARY_F16(name, vop, sop) \
    UNARY_BF16(name, vop, sop) \
    UNARY_FP8E4M3(name, sop) \
    UNARY_FP8E5M2(name, sop) \
    UNARY_FP4E2M1(name, sop)

#define BINARY_ALL_TYPES(name, op) \
    BINARY_F32(name, op) \
    BINARY_F16(name, op) \
    BINARY_BF16(name, op) \
    BINARY_FP8E4M3(name, op) \
    BINARY_FP8E5M2(name, op) \
    BINARY_FP4E2M1(name, op)

#define BINARY_FN_F32(name, fn) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4); \
        float4 vo = make_float4(fn(va.x, vb.x), fn(va.y, vb.y), fn(va.z, vb.z), fn(va.w, vb.w)); \
        STORE_F4(out, i4, vo); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = fn(__ldg(&a[j]), __ldg(&b[j])); } \
}

#define BINARY_FN_CONV(suffix, T, to_f32, from_f32, name, fn) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_##suffix(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { out[i] = from_f32(fn(to_f32(a[i]), to_f32(b[i]))); } \
}

#define BINARY_FN_ALL_TYPES(name, fn) \
    BINARY_FN_F32(name, fn) \
    BINARY_FN_CONV(f16, __half, from_half, to_half, name, fn) \
    BINARY_FN_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16, name, fn) \
    BINARY_FN_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32, name, fn) \
    BINARY_FN_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32, name, fn) \
    BINARY_FN_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32, name, fn)

// ---- Masked fill / where ----

#define MASKED_FILL_F16_BF16_FP(name, T, to_f32, from_f32_unused) \
extern "C" __global__ __launch_bounds__(256) void k_masked_fill_##name(const T* __restrict__ in, const T* __restrict__ mask, T value, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { float m = to_f32(mask[i]); out[i] = (m == 0.0f) ? in[i] : value; } \
}

#define MASKED_FILL_ALL_TYPES() \
extern "C" __global__ __launch_bounds__(256) void k_masked_fill_f32(const float* __restrict__ in, const float* __restrict__ mask, float value, float* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = (mask[i] == 0.0f) ? in[i] : value; \
} \
extern "C" __global__ __launch_bounds__(256) void k_masked_fill_f64(const double* __restrict__ in, const double* __restrict__ mask, double value, double* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = (mask[i] == 0.0) ? in[i] : value; \
} \
MASKED_FILL_F16_BF16_FP(f16, __half, from_half, to_half) \
MASKED_FILL_F16_BF16_FP(bf16, __nv_bfloat16, from_bf16, to_bf16) \
MASKED_FILL_F16_BF16_FP(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32) \
MASKED_FILL_F16_BF16_FP(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32) \
MASKED_FILL_F16_BF16_FP(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

#define WHERE_F16_BF16_FP(name, T, to_f32) \
extern "C" __global__ __launch_bounds__(256) void k_where_##name(const T* __restrict__ a, const T* __restrict__ cond, const T* __restrict__ b, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { float c = to_f32(cond[i]); out[i] = (c == 0.0f) ? b[i] : a[i]; } \
}

#define WHERE_ALL_TYPES() \
extern "C" __global__ __launch_bounds__(256) void k_where_f32(const float* __restrict__ a, const float* __restrict__ cond, const float* __restrict__ b, float* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = (cond[i] == 0.0f) ? b[i] : a[i]; \
} \
extern "C" __global__ __launch_bounds__(256) void k_where_f64(const double* __restrict__ a, const double* __restrict__ cond, const double* __restrict__ b, double* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = (cond[i] == 0.0) ? b[i] : a[i]; \
} \
WHERE_F16_BF16_FP(f16, __half, from_half) \
WHERE_F16_BF16_FP(bf16, __nv_bfloat16, from_bf16) \
WHERE_F16_BF16_FP(fp8e4m3, uint8_t, fp8e4m3_to_f32) \
WHERE_F16_BF16_FP(fp8e5m2, uint8_t, fp8e5m2_to_f32) \
WHERE_F16_BF16_FP(fp4e2m1, uint8_t, fp4e2m1_to_f32)

// ---- Scale / powf / fill / axpy / transpose / matmul ----

#define SCALE_CONV(name, T, to_f32, from_f32) \
extern "C" __global__ __launch_bounds__(256) void k_scale_##name(const T* __restrict__ in, T s, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    float sf = to_f32(s); \
    if (i < n) { out[i] = from_f32(to_f32(in[i]) * sf); } \
}

#define SCALE_ALL_TYPES() \
extern "C" __global__ __launch_bounds__(256) void k_scale_f32(const float* __restrict__ in, float s, float* __restrict__ out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 v = LOAD_F4(in, i4); v.x *= s; v.y *= s; v.z *= s; v.w *= s; STORE_F4(out, i4, v); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = __ldg(&in[j])*s; } \
} \
SCALE_CONV(f16, __half, from_half, to_half) \
SCALE_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16) \
SCALE_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32) \
SCALE_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32) \
SCALE_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

#define POWF_CONV(name, T, to_f32, from_f32) \
extern "C" __global__ __launch_bounds__(256) void k_powf_##name(const T* __restrict__ in, T p, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    float pf = to_f32(p); \
    if (i < n) { out[i] = from_f32(powf(to_f32(in[i]), pf)); } \
}

#define POWF_ALL_TYPES() \
extern "C" __global__ __launch_bounds__(256) void k_powf_f32(const float* __restrict__ in, float p, float* __restrict__ out, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 v = LOAD_F4(in, i4); \
        v.x = __expf(p*__logf(v.x)); v.y = __expf(p*__logf(v.y)); \
        v.z = __expf(p*__logf(v.z)); v.w = __expf(p*__logf(v.w)); \
        STORE_F4(out, i4, v); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = powf(__ldg(&in[j]), p); } \
} \
POWF_CONV(f16, __half, from_half, to_half) \
POWF_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16) \
POWF_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32) \
POWF_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32) \
POWF_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

#define FILL_SCALAR(suffix, T) \
extern "C" __global__ __launch_bounds__(256) void k_fill_##suffix(T* __restrict__ out, T val, unsigned n) { \
    unsigned i = THREAD_ID; if (i < n) out[i] = val; \
}
#define FILL_ALL_TYPES() \
extern "C" __global__ __launch_bounds__(256) void k_fill_f32(float* __restrict__ out, float val, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { STORE_F4(out, i4, make_float4(val, val, val, val)); } \
    else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = val; } \
} \
FILL_SCALAR(f16, __half) \
FILL_SCALAR(bf16, __nv_bfloat16) \
FILL_SCALAR(fp8e4m3, uint8_t) \
FILL_SCALAR(fp8e5m2, uint8_t) \
FILL_SCALAR(fp4e2m1, uint8_t)

#define AXPY_CONV(name, T, to_f32, from_f32) \
extern "C" __global__ __launch_bounds__(256) void k_axpy_##name(T* __restrict__ y, T alpha, const T* __restrict__ x, unsigned n) { \
    unsigned i = THREAD_ID; \
    float a = to_f32(alpha); \
    if (i < n) { y[i] = from_f32(to_f32(y[i]) + a * to_f32(x[i])); } \
}

#define AXPY_ALL_TYPES() \
extern "C" __global__ __launch_bounds__(256) void k_axpy_f32(float* __restrict__ y, float alpha, const float* __restrict__ x, unsigned n) { \
    unsigned i4 = VEC4_IDX, i = i4 * 4; \
    if (i + 3 < n) { \
        float4 vy = LOAD_F4(y, i4), vx = LOAD_F4(x, i4); \
        vy.x += alpha * vx.x; vy.y += alpha * vx.y; vy.z += alpha * vx.z; vy.w += alpha * vx.w; \
        STORE_F4(y, i4, vy); \
    } else { for (unsigned j = i; j < n && j < i+4; j++) y[j] += alpha * __ldg(&x[j]); } \
} \
AXPY_CONV(f16, __half, from_half, to_half) \
AXPY_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16) \
AXPY_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32) \
AXPY_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32) \
AXPY_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

#define TRANSPOSE_TYPED(name, T) \
extern "C" __global__ void k_transpose_##name(const T* in, T* out, unsigned rows, unsigned cols) { \
    unsigned i = THREAD_ID; \
    if (i < rows * cols) { unsigned r = i / cols, c = i % cols; out[c * rows + r] = in[r * cols + c]; } \
}

#define TRANSPOSE_ALL_TYPES() \
TRANSPOSE_TYPED(f32, float) \
TRANSPOSE_TYPED(f16, __half) \
TRANSPOSE_TYPED(bf16, __nv_bfloat16) \
TRANSPOSE_TYPED(fp8e4m3, uint8_t) \
TRANSPOSE_TYPED(fp8e5m2, uint8_t) \
TRANSPOSE_TYPED(fp4e2m1, uint8_t)

#define MATMUL_CONV(name, T, to_f32, from_f32) \
extern "C" __global__ void k_matmul_##name(const T* A, const T* B, T* C, unsigned M, unsigned K, unsigned N) { \
    __shared__ float sA[TILE][TILE], sB[TILE][TILE]; \
    unsigned row = blockIdx.y * TILE + threadIdx.y; \
    unsigned col = blockIdx.x * TILE + threadIdx.x; \
    float acc = 0.0f; \
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) { \
        unsigned ak = t * TILE + threadIdx.x, bk = t * TILE + threadIdx.y; \
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? to_f32(A[row * K + ak]) : 0.0f; \
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? to_f32(B[bk * N + col]) : 0.0f; \
        __syncthreads(); \
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x]; \
        __syncthreads(); \
    } \
    if (row < M && col < N) C[row * N + col] = from_f32(acc); \
}

#define MATMUL_ALL_TYPES() \
MATMUL_CONV(f32, float, MATMUL_F32_IDENTITY, MATMUL_F32_IDENTITY) \
MATMUL_CONV(f16, __half, from_half, to_half) \
MATMUL_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16) \
MATMUL_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32) \
MATMUL_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32) \
MATMUL_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ---- Backward activation helpers ----

#define BWD_RELU_CONV(name, T, from_t, to_t) \
extern "C" __global__ __launch_bounds__(256) void k_relu_bwd_##name(const T* __restrict__ grad, const T* __restrict__ input, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { float g = from_t(grad[i]); float x = from_t(input[i]); out[i] = to_t(x > 0.0f ? g : 0.0f); } \
}

#define BWD_LEAKY_RELU_CONV(name, T, from_t, to_t) \
extern "C" __global__ __launch_bounds__(256) void k_leaky_relu_bwd_##name(const T* __restrict__ grad, const T* __restrict__ input, float alpha, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { float g = from_t(grad[i]); float x = from_t(input[i]); out[i] = to_t(x > 0.0f ? g : alpha * g); } \
}

#define BWD_ELU_CONV(name, T, from_t, to_t) \
extern "C" __global__ __launch_bounds__(256) void k_elu_bwd_##name(const T* __restrict__ grad, const T* __restrict__ input, float alpha, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { float g = from_t(grad[i]); float x = from_t(input[i]); out[i] = to_t(x > 0.0f ? g : g * alpha * __expf(x)); } \
}

#define BWD_GELU_CONV(name, T, from_t, to_t) \
extern "C" __global__ __launch_bounds__(256) void k_gelu_bwd_##name(const T* __restrict__ grad, const T* __restrict__ input, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float g = from_t(grad[i]); float x = from_t(input[i]); \
        float cdf = 0.5f * (1.0f + erf_approx_f32(x * 0.7071067811865476f)); \
        float pdf = __expf(-0.5f * x * x) * 0.3989422804014327f; \
        out[i] = to_t(g * (cdf + x * pdf)); \
    } \
}

#define BWD_ABS_CONV(name, T, from_t, to_t) \
extern "C" __global__ __launch_bounds__(256) void k_abs_bwd_##name(const T* __restrict__ grad, const T* __restrict__ input, T* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { float g = from_t(grad[i]); float x = from_t(input[i]); out[i] = to_t(x > 0.0f ? g : x < 0.0f ? -g : 0.0f); } \
}

#define BWD_ALL_CONV(name, T, from_t, to_t) \
    BWD_RELU_CONV(name, T, from_t, to_t) \
    BWD_LEAKY_RELU_CONV(name, T, from_t, to_t) \
    BWD_ELU_CONV(name, T, from_t, to_t) \
    BWD_GELU_CONV(name, T, from_t, to_t) \
    BWD_ABS_CONV(name, T, from_t, to_t)

// ============================================================
// === Reduce Primitives ===
// ============================================================

// ---- Shuffle intrinsics (platform-dependent) ----
#ifdef __HIP_PLATFORM_AMD__
#define SHFL_DOWN_F32(val, offset) __shfl_down(val, offset)
#define SHFL_DOWN_F64(val, offset) __shfl_down(val, offset)
#else
#define SHFL_DOWN_F32(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#define SHFL_DOWN_F64(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#endif

// ---- Warp-level reductions (f32) ----

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += SHFL_DOWN_F32(val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, SHFL_DOWN_F32(val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_min_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, SHFL_DOWN_F32(val, offset));
    return val;
}

// ---- Warp-level reductions (f64) ----

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += SHFL_DOWN_F64(val, offset);
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmax(val, SHFL_DOWN_F64(val, offset));
    return val;
}

__device__ __forceinline__ double warp_reduce_min_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmin(val, SHFL_DOWN_F64(val, offset));
    return val;
}

// ---- Block-level reduce (tid==0 only) ----

#define BLOCK_REDUCE_F32(val, warp_fn, identity, sdata, tid) \
    do { \
        val = warp_fn(val); \
        if ((tid) % 32 == 0) (sdata)[(tid) / 32] = val; \
        __syncthreads(); \
        if ((tid) < 32) { \
            val = ((tid) < (blockDim.x + 31) / 32) ? (sdata)[(tid)] : (identity); \
            val = warp_fn(val); \
        } \
    } while (0)

#define BLOCK_REDUCE_F64(val, warp_fn, identity, sdata, tid) \
    BLOCK_REDUCE_F32(val, warp_fn, identity, sdata, tid)

// ---- Block-level reduce with broadcast (ALL threads get result) ----

#define BLOCK_REDUCE_BCAST_F32(val, warp_fn, identity, sdata, tid) \
    do { \
        val = warp_fn(val); \
        if (tid % 32 == 0) sdata[tid / 32] = val; \
        __syncthreads(); \
        if (tid < 32) { \
            val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : (identity); \
            val = warp_fn(val); \
        } \
        __syncthreads(); \
        if (tid == 0) sdata[0] = val; \
        __syncthreads(); \
        val = sdata[0]; \
    } while (0)

#define BLOCK_REDUCE_BCAST_F64(val, warp_fn, identity, sdata, tid) \
    do { \
        val = warp_fn(val); \
        if (tid % 32 == 0) sdata[tid / 32] = val; \
        __syncthreads(); \
        if (tid < 32) { \
            val = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : (identity); \
            val = warp_fn(val); \
        } \
        __syncthreads(); \
        if (tid == 0) sdata[0] = val; \
        __syncthreads(); \
        val = sdata[0]; \
    } while (0)

// ---- Grid-level reduce (last-block aggregation) ----

#define GRID_REDUCE_LAST_BLOCK(T, val, warp_fn, identity, partial, sdata, tid, out, convert_out) \
    do { \
        __threadfence(); \
        __shared__ bool _is_last; \
        if ((tid) == 0) { \
            unsigned* _counter = (unsigned*)&(partial)[gridDim.x]; \
            unsigned _ticket = atomicInc(_counter, gridDim.x); \
            _is_last = (_ticket == gridDim.x - 1); \
        } \
        __syncthreads(); \
        if (_is_last) { \
            T _v = ((tid) < gridDim.x) ? (partial)[(tid)] : (identity); \
            _v = warp_fn(_v); \
            if (blockDim.x > 32) { \
                if ((tid) % 32 == 0) (sdata)[(tid) / 32] = _v; \
                __syncthreads(); \
                if ((tid) < 32) { \
                    _v = ((tid) < (blockDim.x + 31) / 32) ? (sdata)[(tid)] : (identity); \
                    _v = warp_fn(_v); \
                } \
            } \
            if ((tid) == 0) (out)[0] = convert_out(_v); \
        } \
    } while (0)

// ---- Full reduce (block + grid) ----

#define FULL_REDUCE_F32(val, warp_fn, identity, partial, sdata, tid, out, convert_out) \
    do { \
        BLOCK_REDUCE_F32(val, warp_fn, identity, sdata, tid); \
        if ((tid) == 0) (partial)[blockIdx.x] = val; \
        GRID_REDUCE_LAST_BLOCK(float, val, warp_fn, identity, partial, sdata, tid, out, convert_out); \
    } while (0)

#define FULL_REDUCE_F64(val, warp_fn, identity, partial, sdata, tid, out, convert_out) \
    do { \
        BLOCK_REDUCE_F64(val, warp_fn, identity, sdata, tid); \
        if ((tid) == 0) (partial)[blockIdx.x] = val; \
        GRID_REDUCE_LAST_BLOCK(double, val, warp_fn, identity, partial, sdata, tid, out, convert_out); \
    } while (0)

// ============================================================
// === Reduce Kernel Generators ===
// ============================================================

#define REDUCE_FP8_SUM(name, to_f32, from_f32) \
extern "C" __global__ void __launch_bounds__(256) k_sum_##name( \
    const uint8_t* __restrict__ in, float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out) { \
    float acc = 0.0f; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) \
        acc += to_f32(in[i]); \
    __shared__ float sdata[32]; \
    FULL_REDUCE_F32(acc, warp_reduce_sum_f32, 0.0f, partial, sdata, tid, out, from_f32); \
}

#define REDUCE_FP8_MAX(name, to_f32, from_f32) \
extern "C" __global__ void __launch_bounds__(256) k_max_##name( \
    const uint8_t* __restrict__ in, float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out) { \
    float neg_inf = -__int_as_float(0x7f800000); \
    float acc = neg_inf; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) \
        acc = fmaxf(acc, to_f32(in[i])); \
    __shared__ float sdata[32]; \
    FULL_REDUCE_F32(acc, warp_reduce_max_f32, neg_inf, partial, sdata, tid, out, from_f32); \
}

#define REDUCE_FP8_MIN(name, to_f32, from_f32) \
extern "C" __global__ void __launch_bounds__(256) k_min_##name( \
    const uint8_t* __restrict__ in, float* __restrict__ partial, unsigned n, uint8_t* __restrict__ out) { \
    float pos_inf = __int_as_float(0x7f800000); \
    float acc = pos_inf; \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) \
        acc = fminf(acc, to_f32(in[i])); \
    __shared__ float sdata[32]; \
    FULL_REDUCE_F32(acc, warp_reduce_min_f32, pos_inf, partial, sdata, tid, out, from_f32); \
}

#define MSE_SUM_FWD(kernel_name, in_type, out_type, from_fn, to_fn, partial_type, warp_fn, identity, full_reduce) \
extern "C" __global__ __launch_bounds__(256) void kernel_name( \
    const in_type* __restrict__ pred, const in_type* __restrict__ target, \
    partial_type* __restrict__ partial, unsigned n, out_type* __restrict__ out) { \
    partial_type acc = (identity); \
    unsigned tid = threadIdx.x; \
    unsigned grid_stride = blockDim.x * gridDim.x; \
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += grid_stride) { \
        partial_type d = from_fn(pred[i]) - from_fn(target[i]); \
        acc += d * d; \
    } \
    __shared__ partial_type sdata[32]; \
    full_reduce(acc, warp_fn, identity, partial, sdata, tid, out, to_fn); \
}
