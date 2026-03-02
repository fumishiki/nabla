// k_basic_math.cuh — Elementwise math kernels: atan2, exp, log, pow, sin/cos/tanh, clamp, ...

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

