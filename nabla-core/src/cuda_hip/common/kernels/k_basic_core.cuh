// k_basic_core.cuh — Shared macros and stdint definitions (included first in KERNELS concat)

#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define TILE 16

#define LOAD_F4(ptr, i)  __ldg(((const float4*)(ptr)) + (i))
#define STORE_F4(ptr, i, v) (((float4*)(ptr))[i] = (v))
#define VEC4_IDX (blockIdx.x * blockDim.x + threadIdx.x)

// Define stdint types inline to avoid NVRTC system header path issues.
// Including <stdint.h> from /usr/include on aarch64 interferes with CUDA macros.
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
#endif
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ __half to_half(float v) { return __float2half(v); }
__device__ __forceinline__ float from_half(__half v) { return __half2float(v); }
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

extern "C" __global__ __launch_bounds__(256) void k_cast_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = to_half(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_f16_to_f32(const __half* __restrict__ in, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = from_half(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_f64_to_f32(const double* __restrict__ in, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = (float)in[i];
}

extern "C" __global__ __launch_bounds__(256) void k_cast_f32_to_f64(const float* __restrict__ in, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = (double)in[i];
}

extern "C" __global__ __launch_bounds__(256) void k_cast_f32_to_fp8e4m3(const float* __restrict__ in, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = fp8e4m3_from_f32(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_fp8e4m3_to_f32(const uint8_t* __restrict__ in, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = fp8e4m3_to_f32(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_f32_to_fp8e5m2(const float* __restrict__ in, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = fp8e5m2_from_f32(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_fp8e5m2_to_f32(const uint8_t* __restrict__ in, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = fp8e5m2_to_f32(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_f32_to_fp4e2m1(const float* __restrict__ in, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = fp4e2m1_from_f32(in[i]);
}

extern "C" __global__ __launch_bounds__(256) void k_cast_fp4e2m1_to_f32(const uint8_t* __restrict__ in, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = fp4e2m1_to_f32(in[i]);
}

#define _NEG(x) (-(x))
#define _RECIP_F(x) (1.0f/(x))
#define _RECIP_D(x) (1.0/(x))
#define _LOG1P_FAST(x) (__logf(1.0f+(x)))

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

#define UNARY_FP8E4M3(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_fp8e4m3(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float x = fp8e4m3_to_f32(in[i]); \
        out[i] = fp8e4m3_from_f32(op(x)); \
    } \
}

#define UNARY_FP8E5M2(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_fp8e5m2(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float x = fp8e5m2_to_f32(in[i]); \
        out[i] = fp8e5m2_from_f32(op(x)); \
    } \
}

#define UNARY_FP4E2M1(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_fp4e2m1(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float x = fp4e2m1_to_f32(in[i]); \
        out[i] = fp4e2m1_from_f32(op(x)); \
    } \
}

#define BINARY_FP8E4M3(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_fp8e4m3(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float ax = fp8e4m3_to_f32(a[i]); \
        float bx = fp8e4m3_to_f32(b[i]); \
        out[i] = fp8e4m3_from_f32(ax op bx); \
    } \
}

#define BINARY_FP8E5M2(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_fp8e5m2(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float ax = fp8e5m2_to_f32(a[i]); \
        float bx = fp8e5m2_to_f32(b[i]); \
        out[i] = fp8e5m2_from_f32(ax op bx); \
    } \
}

#define BINARY_FP4E2M1(name, op) \
extern "C" __global__ __launch_bounds__(256) void k_##name##_fp4e2m1(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) { \
    unsigned i = THREAD_ID; \
    if (i < n) { \
        float ax = fp4e2m1_to_f32(a[i]); \
        float bx = fp4e2m1_to_f32(b[i]); \
        out[i] = fp4e2m1_from_f32(ax op bx); \
    } \
}

extern "C" __global__ __launch_bounds__(256) void k_masked_fill_f32(const float* __restrict__ in, const float* __restrict__ mask, float value, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = (mask[i] == 0.0f) ? in[i] : value;
}

extern "C" __global__ __launch_bounds__(256) void k_masked_fill_f64(const double* __restrict__ in, const double* __restrict__ mask, double value, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = (mask[i] == 0.0) ? in[i] : value;
}

extern "C" __global__ __launch_bounds__(256) void k_masked_fill_f16(const __half* __restrict__ in, const __half* __restrict__ mask, __half value, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float m = from_half(mask[i]);
        out[i] = (m == 0.0f) ? in[i] : value;
    }
}

extern "C" __global__ __launch_bounds__(256) void k_masked_fill_fp8e4m3(const uint8_t* __restrict__ in, const uint8_t* __restrict__ mask, uint8_t value, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float m = fp8e4m3_to_f32(mask[i]);
        out[i] = (m == 0.0f) ? in[i] : value;
    }
}

extern "C" __global__ __launch_bounds__(256) void k_masked_fill_fp8e5m2(const uint8_t* __restrict__ in, const uint8_t* __restrict__ mask, uint8_t value, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float m = fp8e5m2_to_f32(mask[i]);
        out[i] = (m == 0.0f) ? in[i] : value;
    }
}

extern "C" __global__ __launch_bounds__(256) void k_masked_fill_fp4e2m1(const uint8_t* __restrict__ in, const uint8_t* __restrict__ mask, uint8_t value, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float m = fp4e2m1_to_f32(mask[i]);
        out[i] = (m == 0.0f) ? in[i] : value;
    }
}

extern "C" __global__ __launch_bounds__(256) void k_where_f32(const float* __restrict__ a, const float* __restrict__ cond, const float* __restrict__ b, float* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = (cond[i] == 0.0f) ? b[i] : a[i];
}

extern "C" __global__ __launch_bounds__(256) void k_where_f64(const double* __restrict__ a, const double* __restrict__ cond, const double* __restrict__ b, double* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) out[i] = (cond[i] == 0.0) ? b[i] : a[i];
}

extern "C" __global__ __launch_bounds__(256) void k_where_f16(const __half* __restrict__ a, const __half* __restrict__ cond, const __half* __restrict__ b, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float c = from_half(cond[i]);
        out[i] = (c == 0.0f) ? b[i] : a[i];
    }
}

extern "C" __global__ __launch_bounds__(256) void k_where_fp8e4m3(const uint8_t* __restrict__ a, const uint8_t* __restrict__ cond, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float c = fp8e4m3_to_f32(cond[i]);
        out[i] = (c == 0.0f) ? b[i] : a[i];
    }
}

extern "C" __global__ __launch_bounds__(256) void k_where_fp8e5m2(const uint8_t* __restrict__ a, const uint8_t* __restrict__ cond, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float c = fp8e5m2_to_f32(cond[i]);
        out[i] = (c == 0.0f) ? b[i] : a[i];
    }
}

extern "C" __global__ __launch_bounds__(256) void k_where_fp4e2m1(const uint8_t* __restrict__ a, const uint8_t* __restrict__ cond, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float c = fp4e2m1_to_f32(cond[i]);
        out[i] = (c == 0.0f) ? b[i] : a[i];
    }
}

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

UNARY_F32(neg,   _NEG,            _NEG)
UNARY_F32(recip, _RECIP_F,        _RECIP_F)
UNARY_F32(exp,   __expf,          __expf)
UNARY_F32(ln,    __logf,          __logf)
UNARY_F32(log1p, _LOG1P_FAST,     log1pf)
UNARY_F32(sin,   __sinf,          __sinf)
UNARY_F32(cos,   __cosf,          __cosf)
UNARY_F32(tan,   tanf,            tanf)
UNARY_F32(tanh,  tanhf,           tanhf)
UNARY_F32(sqrt,  __fsqrt_rn,      sqrtf)
UNARY_F32(abs,   fabsf,           fabsf)
UNARY_F32(ceil,  ceilf,           ceilf)
UNARY_F32(floor, floorf,          floorf)
UNARY_F32(round, roundf,          roundf)
UNARY_F32(erf,   erf_approx_f32,  erf_approx_f32)
UNARY_F32(asin,  asinf,           asinf)
UNARY_F32(acos,  acosf,           acosf)
UNARY_F32(atan,  atanf,           atanf)
UNARY_F32(sinh,  sinhf,           sinhf)
UNARY_F32(cosh,  coshf,           coshf)
UNARY_F32(asinh, asinhf,          asinhf)
UNARY_F32(acosh, acoshf,          acoshf)
UNARY_F32(atanh, atanhf,          atanhf)
UNARY_F32(log2,  __log2f,         __log2f)
UNARY_F32(log10, __log10f,        __log10f)

UNARY_F16(neg,   _NEG,            _NEG)
UNARY_F16(recip, _RECIP_F,        _RECIP_F)
UNARY_F16(exp,   __expf,          __expf)
UNARY_F16(ln,    __logf,          __logf)
UNARY_F16(log1p, _LOG1P_FAST,     log1pf)
UNARY_F16(sin,   __sinf,          __sinf)
UNARY_F16(cos,   __cosf,          __cosf)
UNARY_F16(tan,   tanf,            tanf)
UNARY_F16(tanh,  tanhf,           tanhf)
UNARY_F16(sqrt,  __fsqrt_rn,      sqrtf)
UNARY_F16(abs,   fabsf,           fabsf)
UNARY_F16(ceil,  ceilf,           ceilf)
UNARY_F16(floor, floorf,          floorf)
UNARY_F16(round, roundf,          roundf)
UNARY_F16(erf,   erf_approx_f32,  erf_approx_f32)
UNARY_F16(asin,  asinf,           asinf)
UNARY_F16(acos,  acosf,           acosf)
UNARY_F16(atan,  atanf,           atanf)
UNARY_F16(sinh,  sinhf,           sinhf)
UNARY_F16(cosh,  coshf,           coshf)
UNARY_F16(asinh, asinhf,          asinhf)
UNARY_F16(acosh, acoshf,          acoshf)
UNARY_F16(atanh, atanhf,          atanhf)
UNARY_F16(log2,  __log2f,         __log2f)
UNARY_F16(log10, __log10f,        __log10f)

UNARY_FP8E4M3(neg,   _NEG)
UNARY_FP8E4M3(recip, _RECIP_F)
UNARY_FP8E4M3(exp,   __expf)
UNARY_FP8E4M3(ln,    __logf)
UNARY_FP8E4M3(log1p, log1pf)
UNARY_FP8E4M3(sin,   __sinf)
UNARY_FP8E4M3(cos,   __cosf)
UNARY_FP8E4M3(tan,   tanf)
UNARY_FP8E4M3(tanh,  tanhf)
UNARY_FP8E4M3(sqrt,  sqrtf)
UNARY_FP8E4M3(abs,   fabsf)
UNARY_FP8E4M3(ceil,  ceilf)
UNARY_FP8E4M3(floor, floorf)
UNARY_FP8E4M3(round, roundf)
UNARY_FP8E4M3(erf,   erf_approx_f32)
UNARY_FP8E4M3(asin,  asinf)
UNARY_FP8E4M3(acos,  acosf)
UNARY_FP8E4M3(atan,  atanf)
UNARY_FP8E4M3(sinh,  sinhf)
UNARY_FP8E4M3(cosh,  coshf)
UNARY_FP8E4M3(asinh, asinhf)
UNARY_FP8E4M3(acosh, acoshf)
UNARY_FP8E4M3(atanh, atanhf)
UNARY_FP8E4M3(log2,  __log2f)
UNARY_FP8E4M3(log10, __log10f)

UNARY_FP8E5M2(neg,   _NEG)
UNARY_FP8E5M2(recip, _RECIP_F)
UNARY_FP8E5M2(exp,   __expf)
UNARY_FP8E5M2(ln,    __logf)
UNARY_FP8E5M2(log1p, log1pf)
UNARY_FP8E5M2(sin,   __sinf)
UNARY_FP8E5M2(cos,   __cosf)
UNARY_FP8E5M2(tan,   tanf)
UNARY_FP8E5M2(tanh,  tanhf)
UNARY_FP8E5M2(sqrt,  sqrtf)
UNARY_FP8E5M2(abs,   fabsf)
UNARY_FP8E5M2(ceil,  ceilf)
UNARY_FP8E5M2(floor, floorf)
UNARY_FP8E5M2(round, roundf)
UNARY_FP8E5M2(erf,   erf_approx_f32)
UNARY_FP8E5M2(asin,  asinf)
UNARY_FP8E5M2(acos,  acosf)
UNARY_FP8E5M2(atan,  atanf)
UNARY_FP8E5M2(sinh,  sinhf)
UNARY_FP8E5M2(cosh,  coshf)
UNARY_FP8E5M2(asinh, asinhf)
UNARY_FP8E5M2(acosh, acoshf)
UNARY_FP8E5M2(atanh, atanhf)
UNARY_FP8E5M2(log2,  __log2f)
UNARY_FP8E5M2(log10, __log10f)

UNARY_FP4E2M1(neg,   _NEG)
UNARY_FP4E2M1(recip, _RECIP_F)
UNARY_FP4E2M1(exp,   __expf)
UNARY_FP4E2M1(ln,    __logf)
UNARY_FP4E2M1(log1p, log1pf)
UNARY_FP4E2M1(sin,   __sinf)
UNARY_FP4E2M1(cos,   __cosf)
UNARY_FP4E2M1(tan,   tanf)
UNARY_FP4E2M1(tanh,  tanhf)
UNARY_FP4E2M1(sqrt,  sqrtf)
UNARY_FP4E2M1(abs,   fabsf)
UNARY_FP4E2M1(ceil,  ceilf)
UNARY_FP4E2M1(floor, floorf)
UNARY_FP4E2M1(round, roundf)
UNARY_FP4E2M1(erf,   erf_approx_f32)
UNARY_FP4E2M1(asin,  asinf)
UNARY_FP4E2M1(acos,  acosf)
UNARY_FP4E2M1(atan,  atanf)
UNARY_FP4E2M1(sinh,  sinhf)
UNARY_FP4E2M1(cosh,  coshf)
UNARY_FP4E2M1(asinh, asinhf)
UNARY_FP4E2M1(acosh, acoshf)
UNARY_FP4E2M1(atanh, atanhf)
UNARY_FP4E2M1(log2,  __log2f)
UNARY_FP4E2M1(log10, __log10f)

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

UNARY_F32(sigmoid,    sigmoid_f32,    sigmoid_f32)
UNARY_F32(silu,       silu_f32,       silu_f32)
UNARY_F32(mish,       mish_f32,       mish_f32)
UNARY_F32(leaky_relu, leaky_relu_f32, leaky_relu_f32)
UNARY_F32(elu,        elu_f32,        elu_f32)
UNARY_F32(hardswish,  hardswish_f32,  hardswish_f32)

UNARY_F16(sigmoid,    sigmoid_f32,    sigmoid_f32)
UNARY_F16(silu,       silu_f32,       silu_f32)
UNARY_F16(mish,       mish_f32,       mish_f32)
UNARY_F16(leaky_relu, leaky_relu_f32, leaky_relu_f32)
UNARY_F16(elu,        elu_f32,        elu_f32)
UNARY_F16(hardswish,  hardswish_f32,  hardswish_f32)

UNARY_FP8E4M3(sigmoid,    sigmoid_f32)
UNARY_FP8E4M3(silu,       silu_f32)
UNARY_FP8E4M3(mish,       mish_f32)
UNARY_FP8E4M3(leaky_relu, leaky_relu_f32)
UNARY_FP8E4M3(elu,        elu_f32)
UNARY_FP8E4M3(hardswish,  hardswish_f32)

UNARY_FP8E5M2(sigmoid,    sigmoid_f32)
UNARY_FP8E5M2(silu,       silu_f32)
UNARY_FP8E5M2(mish,       mish_f32)
UNARY_FP8E5M2(leaky_relu, leaky_relu_f32)
UNARY_FP8E5M2(elu,        elu_f32)
UNARY_FP8E5M2(hardswish,  hardswish_f32)

UNARY_FP4E2M1(sigmoid,    sigmoid_f32)
UNARY_FP4E2M1(silu,       silu_f32)
UNARY_FP4E2M1(mish,       mish_f32)
UNARY_FP4E2M1(leaky_relu, leaky_relu_f32)
UNARY_FP4E2M1(elu,        elu_f32)
UNARY_FP4E2M1(hardswish,  hardswish_f32)

// --- Backward activation kernels (binary: grad, input -> grad_input) ---
extern "C" __global__ __launch_bounds__(256) void k_relu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : 0.f, x.y > 0.f ? g.y : 0.f, x.z > 0.f ? g.z : 0.f, x.w > 0.f ? g.w : 0.f);
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); out[j] = __ldg(&input[j]) > 0.f ? gv : 0.f; } }
}
extern "C" __global__ __launch_bounds__(256) void k_leaky_relu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : 0.01f*g.x, x.y > 0.f ? g.y : 0.01f*g.y, x.z > 0.f ? g.z : 0.01f*g.z, x.w > 0.f ? g.w : 0.01f*g.w);
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); out[j] = __ldg(&input[j]) > 0.f ? gv : 0.01f*gv; } }
}
extern "C" __global__ __launch_bounds__(256) void k_elu_bwd_f32(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 g = LOAD_F4(grad, i4), x = LOAD_F4(input, i4);
        float4 o = make_float4(x.x > 0.f ? g.x : g.x*__expf(x.x), x.y > 0.f ? g.y : g.y*__expf(x.y),
                               x.z > 0.f ? g.z : g.z*__expf(x.z), x.w > 0.f ? g.w : g.w*__expf(x.w));
        STORE_F4(out, i4, o);
    } else { for (unsigned j = i; j < n && j < i+4; j++) { float gv = __ldg(&grad[j]); float xv = __ldg(&input[j]); out[j] = xv > 0.f ? gv : gv*__expf(xv); } }
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

extern "C" __global__ __launch_bounds__(256) void k_relu_bwd_f16(const __half* __restrict__ grad, const __half* __restrict__ input, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = from_half(grad[i]);
        float x = from_half(input[i]);
        out[i] = to_half(x > 0.0f ? g : 0.0f);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_leaky_relu_bwd_f16(const __half* __restrict__ grad, const __half* __restrict__ input, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = from_half(grad[i]);
        float x = from_half(input[i]);
        out[i] = to_half(x > 0.0f ? g : 0.01f * g);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_elu_bwd_f16(const __half* __restrict__ grad, const __half* __restrict__ input, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = from_half(grad[i]);
        float x = from_half(input[i]);
        out[i] = to_half(x > 0.0f ? g : g * __expf(x));
    }
}
extern "C" __global__ __launch_bounds__(256) void k_gelu_bwd_f16(const __half* __restrict__ grad, const __half* __restrict__ input, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = from_half(grad[i]);
        float x = from_half(input[i]);
        float cdf = 0.5f * (1.0f + erf_approx_f32(x * 0.7071067811865476f));
        float pdf = __expf(-0.5f * x * x) * 0.3989422804014327f;
        out[i] = to_half(g * (cdf + x * pdf));
    }
}
extern "C" __global__ __launch_bounds__(256) void k_abs_bwd_f16(const __half* __restrict__ grad, const __half* __restrict__ input, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float g = from_half(grad[i]);
        float x = from_half(input[i]);
        float v = x > 0.0f ? g : x < 0.0f ? -g : 0.0f;
        out[i] = to_half(v);
    }
}

BINARY_F32(add,  +)
BINARY_F32(sub,  -)
BINARY_F32(emul, *)
BINARY_F32(ediv, /)
BINARY_F16(add,  +)
BINARY_F16(sub,  -)
BINARY_F16(emul, *)
BINARY_F16(ediv, /)
BINARY_FP8E4M3(add,  +)
BINARY_FP8E4M3(sub,  -)
BINARY_FP8E4M3(emul, *)
BINARY_FP8E4M3(ediv, /)
BINARY_FP8E5M2(add,  +)
BINARY_FP8E5M2(emul, *)
BINARY_FP8E5M2(ediv, /)
BINARY_FP4E2M1(add,  +)
BINARY_FP4E2M1(sub,  -)
BINARY_FP4E2M1(emul, *)
BINARY_FP4E2M1(ediv, /)

