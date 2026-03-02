
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
BINARY_FP8E5M2(sub,  -)
BINARY_FP8E5M2(emul, *)
BINARY_FP8E5M2(ediv, /)
BINARY_FP4E2M1(add,  +)
BINARY_FP4E2M1(sub,  -)
BINARY_FP4E2M1(emul, *)
BINARY_FP4E2M1(ediv, /)

extern "C" __global__ __launch_bounds__(256) void k_atan2_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 va = LOAD_F4(a, i4), vb = LOAD_F4(b, i4);
        float4 vo = make_float4(atan2f(va.x, vb.x), atan2f(va.y, vb.y), atan2f(va.z, vb.z), atan2f(va.w, vb.w));
        STORE_F4(out, i4, vo);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = atan2f(__ldg(&a[j]), __ldg(&b[j])); }
}

extern "C" __global__ __launch_bounds__(256) void k_atan2_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float ax = from_half(a[i]);
        float bx = from_half(b[i]);
        out[i] = to_half(atan2f(ax, bx));
    }
}

extern "C" __global__ __launch_bounds__(256) void k_atan2_fp8e4m3(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float ax = fp8e4m3_to_f32(a[i]);
        float bx = fp8e4m3_to_f32(b[i]);
        out[i] = fp8e4m3_from_f32(atan2f(ax, bx));
    }
}

extern "C" __global__ __launch_bounds__(256) void k_atan2_fp8e5m2(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float ax = fp8e5m2_to_f32(a[i]);
        float bx = fp8e5m2_to_f32(b[i]);
        out[i] = fp8e5m2_from_f32(atan2f(ax, bx));
    }
}

extern "C" __global__ __launch_bounds__(256) void k_atan2_fp4e2m1(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        float ax = fp4e2m1_to_f32(a[i]);
        float bx = fp4e2m1_to_f32(b[i]);
        out[i] = fp4e2m1_from_f32(atan2f(ax, bx));
    }
}


extern "C" __global__ __launch_bounds__(256) void k_axpy_f32(float* __restrict__ y, float alpha, const float* __restrict__ x, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 vy = LOAD_F4(y, i4), vx = LOAD_F4(x, i4);
        vy.x += alpha * vx.x; vy.y += alpha * vx.y; vy.z += alpha * vx.z; vy.w += alpha * vx.w;
        STORE_F4(y, i4, vy);
    } else { for (unsigned j = i; j < n && j < i+4; j++) y[j] += alpha * __ldg(&x[j]); }
}

extern "C" __global__ __launch_bounds__(256) void k_axpy_f16(__half* __restrict__ y, __half alpha, const __half* __restrict__ x, unsigned n) {
    unsigned i = THREAD_ID;
    float a = from_half(alpha);
    if (i < n) {
        float yv = from_half(y[i]);
        float xv = from_half(x[i]);
        y[i] = to_half(yv + a * xv);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_axpy_fp8e4m3(uint8_t* __restrict__ y, uint8_t alpha, const uint8_t* __restrict__ x, unsigned n) {
    unsigned i = THREAD_ID;
    float a = fp8e4m3_to_f32(alpha);
    if (i < n) {
        float yv = fp8e4m3_to_f32(y[i]);
        float xv = fp8e4m3_to_f32(x[i]);
        y[i] = fp8e4m3_from_f32(yv + a * xv);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_axpy_fp8e5m2(uint8_t* __restrict__ y, uint8_t alpha, const uint8_t* __restrict__ x, unsigned n) {
    unsigned i = THREAD_ID;
    float a = fp8e5m2_to_f32(alpha);
    if (i < n) {
        float yv = fp8e5m2_to_f32(y[i]);
        float xv = fp8e5m2_to_f32(x[i]);
        y[i] = fp8e5m2_from_f32(yv + a * xv);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_axpy_fp4e2m1(uint8_t* __restrict__ y, uint8_t alpha, const uint8_t* __restrict__ x, unsigned n) {
    unsigned i = THREAD_ID;
    float a = fp4e2m1_to_f32(alpha);
    if (i < n) {
        float yv = fp4e2m1_to_f32(y[i]);
        float xv = fp4e2m1_to_f32(x[i]);
        y[i] = fp4e2m1_from_f32(yv + a * xv);
    }
}

extern "C" __global__ __launch_bounds__(256) void k_scale_f32(const float* __restrict__ in, float s, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = LOAD_F4(in, i4);
        v.x *= s; v.y *= s; v.z *= s; v.w *= s;
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = __ldg(&in[j])*s; }
}
extern "C" __global__ __launch_bounds__(256) void k_scale_f16(const __half* __restrict__ in, __half s, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float sf = from_half(s);
    if (i < n) {
        float v = from_half(in[i]);
        out[i] = to_half(v * sf);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_scale_fp8e4m3(const uint8_t* __restrict__ in, uint8_t s, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float sf = fp8e4m3_to_f32(s);
    if (i < n) {
        float v = fp8e4m3_to_f32(in[i]);
        out[i] = fp8e4m3_from_f32(v * sf);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_scale_fp8e5m2(const uint8_t* __restrict__ in, uint8_t s, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float sf = fp8e5m2_to_f32(s);
    if (i < n) {
        float v = fp8e5m2_to_f32(in[i]);
        out[i] = fp8e5m2_from_f32(v * sf);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_scale_fp4e2m1(const uint8_t* __restrict__ in, uint8_t s, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float sf = fp4e2m1_to_f32(s);
    if (i < n) {
        float v = fp4e2m1_to_f32(in[i]);
        out[i] = fp4e2m1_from_f32(v * sf);
    }
}
extern "C" __global__ __launch_bounds__(256) void k_powf_f32(const float* __restrict__ in, float p, float* __restrict__ out, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = LOAD_F4(in, i4);
        v.x = __expf(p*__logf(v.x)); v.y = __expf(p*__logf(v.y));
        v.z = __expf(p*__logf(v.z)); v.w = __expf(p*__logf(v.w));
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = powf(__ldg(&in[j]), p); }
}
extern "C" __global__ __launch_bounds__(256) void k_powf_f16(const __half* __restrict__ in, __half p, __half* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float pf = from_half(p);
    if (i < n) {
        float v = from_half(in[i]);
        out[i] = to_half(powf(v, pf));
    }
}
extern "C" __global__ __launch_bounds__(256) void k_powf_fp8e4m3(const uint8_t* __restrict__ in, uint8_t p, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float pf = fp8e4m3_to_f32(p);
    if (i < n) {
        float v = fp8e4m3_to_f32(in[i]);
        out[i] = fp8e4m3_from_f32(powf(v, pf));
    }
}
extern "C" __global__ __launch_bounds__(256) void k_powf_fp8e5m2(const uint8_t* __restrict__ in, uint8_t p, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float pf = fp8e5m2_to_f32(p);
    if (i < n) {
        float v = fp8e5m2_to_f32(in[i]);
        out[i] = fp8e5m2_from_f32(powf(v, pf));
    }
}
extern "C" __global__ __launch_bounds__(256) void k_powf_fp4e2m1(const uint8_t* __restrict__ in, uint8_t p, uint8_t* __restrict__ out, unsigned n) {
    unsigned i = THREAD_ID;
    float pf = fp4e2m1_to_f32(p);
    if (i < n) {
        float v = fp4e2m1_to_f32(in[i]);
        out[i] = fp4e2m1_from_f32(powf(v, pf));
    }
}
extern "C" __global__ __launch_bounds__(256) void k_fill_f32(float* __restrict__ out, float val, unsigned n) {
    unsigned i4 = VEC4_IDX, i = i4 * 4;
    if (i + 3 < n) {
        float4 v = make_float4(val, val, val, val);
        STORE_F4(out, i4, v);
    } else { for (unsigned j = i; j < n && j < i+4; j++) out[j] = val; }
}
extern "C" __global__ __launch_bounds__(256) void k_fill_f16(__half* __restrict__ out, __half val, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        out[i] = val;
    }
}
extern "C" __global__ __launch_bounds__(256) void k_fill_fp8e4m3(uint8_t* __restrict__ out, uint8_t val, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        out[i] = val;
    }
}
extern "C" __global__ __launch_bounds__(256) void k_fill_fp8e5m2(uint8_t* __restrict__ out, uint8_t val, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        out[i] = val;
    }
}
extern "C" __global__ __launch_bounds__(256) void k_fill_fp4e2m1(uint8_t* __restrict__ out, uint8_t val, unsigned n) {
    unsigned i = THREAD_ID;
    if (i < n) {
        out[i] = val;
    }
}


extern "C" __global__ void k_transpose_f32(const float* in, float* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}
extern "C" __global__ void k_transpose_f16(const __half* in, __half* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}
extern "C" __global__ void k_transpose_fp8e4m3(const uint8_t* in, uint8_t* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}
extern "C" __global__ void k_transpose_fp8e5m2(const uint8_t* in, uint8_t* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}
extern "C" __global__ void k_transpose_fp4e2m1(const uint8_t* in, uint8_t* out,
                                            unsigned rows, unsigned cols) {
    unsigned i = THREAD_ID;
    if (i < rows * cols) {
        unsigned r = i / cols;
        unsigned c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}


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
extern "C" __global__ void k_matmul_f16(const __half* A, const __half* B, __half* C,
                                         unsigned M, unsigned K, unsigned N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    unsigned row = blockIdx.y * TILE + threadIdx.y;
    unsigned col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned ak = t * TILE + threadIdx.x;
        unsigned bk = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? from_half(A[row * K + ak]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? from_half(B[bk * N + col]) : 0.0f;
        __syncthreads();
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = to_half(acc);
}

extern "C" __global__ void k_matmul_fp8e4m3(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                         unsigned M, unsigned K, unsigned N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    unsigned row = blockIdx.y * TILE + threadIdx.y;
    unsigned col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned ak = t * TILE + threadIdx.x;
        unsigned bk = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? fp8e4m3_to_f32(A[row * K + ak]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? fp8e4m3_to_f32(B[bk * N + col]) : 0.0f;
        __syncthreads();
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = fp8e4m3_from_f32(acc);
}

extern "C" __global__ void k_matmul_fp8e5m2(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                         unsigned M, unsigned K, unsigned N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    unsigned row = blockIdx.y * TILE + threadIdx.y;
    unsigned col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned ak = t * TILE + threadIdx.x;
        unsigned bk = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? fp8e5m2_to_f32(A[row * K + ak]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? fp8e5m2_to_f32(B[bk * N + col]) : 0.0f;
        __syncthreads();
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = fp8e5m2_from_f32(acc);
}

extern "C" __global__ void k_matmul_fp4e2m1(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                         unsigned M, unsigned K, unsigned N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    unsigned row = blockIdx.y * TILE + threadIdx.y;
    unsigned col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (unsigned t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned ak = t * TILE + threadIdx.x;
        unsigned bk = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? fp4e2m1_to_f32(A[row * K + ak]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? fp4e2m1_to_f32(B[bk * N + col]) : 0.0f;
        __syncthreads();
        for (unsigned k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = fp4e2m1_from_f32(acc);
}


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
