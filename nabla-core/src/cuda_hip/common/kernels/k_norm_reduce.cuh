// k_norm_reduce.cuh — Axis reductions: sum/max axis0/1, argmax/argmin, cumsum/cumprod

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

extern "C" __global__ void k_sum_axis1_f16(const __half* __restrict__ in,
                                            __half* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += from_half(in[row * cols + i]);
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
    }
    if (tid == 0) out[row] = to_half(acc);
}

extern "C" __global__ void k_sum_axis1_fp8e4m3(const uint8_t* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += fp8e4m3_to_f32(in[row * cols + i]);
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
    }
    if (tid == 0) out[row] = fp8e4m3_from_f32(acc);
}

extern "C" __global__ void k_sum_axis1_fp8e5m2(const uint8_t* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += fp8e5m2_to_f32(in[row * cols + i]);
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
    }
    if (tid == 0) out[row] = fp8e5m2_from_f32(acc);
}

extern "C" __global__ void k_sum_axis1_fp4e2m1(const uint8_t* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc += fp4e2m1_to_f32(in[row * cols + i]);
    acc = warp_reduce_sum_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        acc = warp_reduce_sum_f32(acc);
    }
    if (tid == 0) out[row] = fp4e2m1_from_f32(acc);
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

extern "C" __global__ void k_max_axis1_f16(const __half* __restrict__ in,
                                            __half* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmaxf(acc, from_half(in[row * cols + i]));
    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : -__int_as_float(0x7f800000);
        acc = warp_reduce_max_f32(acc);
    }
    if (tid == 0) out[row] = to_half(acc);
}

extern "C" __global__ void k_max_axis1_fp8e4m3(const uint8_t* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmaxf(acc, fp8e4m3_to_f32(in[row * cols + i]));
    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : -__int_as_float(0x7f800000);
        acc = warp_reduce_max_f32(acc);
    }
    if (tid == 0) out[row] = fp8e4m3_from_f32(acc);
}

extern "C" __global__ void k_max_axis1_fp8e5m2(const uint8_t* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmaxf(acc, fp8e5m2_to_f32(in[row * cols + i]));
    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : -__int_as_float(0x7f800000);
        acc = warp_reduce_max_f32(acc);
    }
    if (tid == 0) out[row] = fp8e5m2_from_f32(acc);
}

extern "C" __global__ void k_max_axis1_fp4e2m1(const uint8_t* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned tid = threadIdx.x;
    float acc = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        acc = fmaxf(acc, fp4e2m1_to_f32(in[row * cols + i]));
    acc = warp_reduce_max_f32(acc);
    __shared__ float sdata[32];
    if (tid % 32 == 0) sdata[tid / 32] = acc;
    __syncthreads();
    if (tid < 32) {
        acc = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : -__int_as_float(0x7f800000);
        acc = warp_reduce_max_f32(acc);
    }
    if (tid == 0) out[row] = fp4e2m1_from_f32(acc);
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

extern "C" __global__ void k_embedding_f16(
    const __half* __restrict__ indices,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    unsigned n_tokens, unsigned embed_dim) {
    unsigned tid = THREAD_ID;
    unsigned total = n_tokens * embed_dim;
    if (tid >= total) return;
    unsigned tok = tid / embed_dim;
    unsigned dim = tid % embed_dim;
    unsigned idx = (unsigned)from_half(indices[tok]);
    out[tid] = weight[idx * embed_dim + dim];
}

extern "C" __global__ void k_embedding_fp8e4m3(
    const uint8_t* __restrict__ indices,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ out,
    unsigned n_tokens, unsigned embed_dim) {
    unsigned tid = THREAD_ID;
    unsigned total = n_tokens * embed_dim;
    if (tid >= total) return;
    unsigned tok = tid / embed_dim;
    unsigned dim = tid % embed_dim;
    unsigned idx = (unsigned)fp8e4m3_to_f32(indices[tok]);
    out[tid] = weight[idx * embed_dim + dim];
}

extern "C" __global__ void k_embedding_fp8e5m2(
    const uint8_t* __restrict__ indices,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ out,
    unsigned n_tokens, unsigned embed_dim) {
    unsigned tid = THREAD_ID;
    unsigned total = n_tokens * embed_dim;
    if (tid >= total) return;
    unsigned tok = tid / embed_dim;
    unsigned dim = tid % embed_dim;
    unsigned idx = (unsigned)fp8e5m2_to_f32(indices[tok]);
    out[tid] = weight[idx * embed_dim + dim];
}

extern "C" __global__ void k_embedding_fp4e2m1(
    const uint8_t* __restrict__ indices,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ out,
    unsigned n_tokens, unsigned embed_dim) {
    unsigned tid = THREAD_ID;
    unsigned total = n_tokens * embed_dim;
    if (tid >= total) return;
    unsigned tok = tid / embed_dim;
    unsigned dim = tid % embed_dim;
    unsigned idx = (unsigned)fp4e2m1_to_f32(indices[tok]);
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

extern "C" __global__ void k_cumsum_axis1_f32(const float* in, float* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cs_f32[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        // Blelloch parallel inclusive prefix sum
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cs_f32[i1] = (i1 < n) ? in[r * n + i1] : 0.0f;
        smem_cs_f32[i2] = (i2 < n) ? in[r * n + i2] : 0.0f;
        __syncthreads();
        // Up-sweep (reduce)
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cs_f32[idx] += smem_cs_f32[idx - stride];
            __syncthreads();
        }
        // Clear last element for exclusive scan
        if (threadIdx.x == 0) smem_cs_f32[2 * bx - 1] = 0.0f;
        __syncthreads();
        // Down-sweep
        for (unsigned stride = bx; stride >= 1; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) {
                float t = smem_cs_f32[idx - stride];
                smem_cs_f32[idx - stride] = smem_cs_f32[idx];
                smem_cs_f32[idx] += t;
            }
            __syncthreads();
        }
        // smem holds exclusive prefix; add original to make inclusive
        if (i1 < n) out[r * n + i1] = smem_cs_f32[i1] + in[r * n + i1];
        if (i2 < n) out[r * n + i2] = smem_cs_f32[i2] + in[r * n + i2];
    } else {
        // Sequential fallback for rows longer than 2*BLOCK_SIZE
        if (threadIdx.x == 0) {
            float acc = 0.0f;
            for (unsigned c = 0; c < n; c++) {
                acc += in[r * n + c];
                out[r * n + c] = acc;
            }
        }
    }
}
extern "C" __global__ void k_cumsum_axis1_f64(const double* in, double* out, unsigned rows, unsigned cols) {
    extern __shared__ double smem_cs_f64[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cs_f64[i1] = (i1 < n) ? in[r * n + i1] : 0.0;
        smem_cs_f64[i2] = (i2 < n) ? in[r * n + i2] : 0.0;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cs_f64[idx] += smem_cs_f64[idx - stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) smem_cs_f64[2 * bx - 1] = 0.0;
        __syncthreads();
        for (unsigned stride = bx; stride >= 1; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) {
                double t = smem_cs_f64[idx - stride];
                smem_cs_f64[idx - stride] = smem_cs_f64[idx];
                smem_cs_f64[idx] += t;
            }
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = smem_cs_f64[i1] + in[r * n + i1];
        if (i2 < n) out[r * n + i2] = smem_cs_f64[i2] + in[r * n + i2];
    } else {
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (unsigned c = 0; c < n; c++) {
                acc += in[r * n + c];
                out[r * n + c] = acc;
            }
        }
    }
}

extern "C" __global__ void k_cumsum_axis1_f16(const __half* in, __half* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cs_f16[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cs_f16[i1] = (i1 < n) ? from_half(in[r * n + i1]) : 0.0f;
        smem_cs_f16[i2] = (i2 < n) ? from_half(in[r * n + i2]) : 0.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cs_f16[idx] += smem_cs_f16[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cs_f16[idx + stride] += smem_cs_f16[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = to_half(smem_cs_f16[i1]);
        if (i2 < n) out[r * n + i2] = to_half(smem_cs_f16[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 0.0f;
            for (unsigned c = 0; c < n; c++) {
                acc += from_half(in[r * n + c]);
                out[r * n + c] = to_half(acc);
            }
        }
    }
}

extern "C" __global__ void k_cumsum_axis1_fp8e4m3(const uint8_t* in, uint8_t* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cs_fp8[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cs_fp8[i1] = (i1 < n) ? fp8e4m3_to_f32(in[r * n + i1]) : 0.0f;
        smem_cs_fp8[i2] = (i2 < n) ? fp8e4m3_to_f32(in[r * n + i2]) : 0.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cs_fp8[idx] += smem_cs_fp8[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cs_fp8[idx + stride] += smem_cs_fp8[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = fp8e4m3_from_f32(smem_cs_fp8[i1]);
        if (i2 < n) out[r * n + i2] = fp8e4m3_from_f32(smem_cs_fp8[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 0.0f;
            for (unsigned c = 0; c < n; c++) {
                acc += fp8e4m3_to_f32(in[r * n + c]);
                out[r * n + c] = fp8e4m3_from_f32(acc);
            }
        }
    }
}

extern "C" __global__ void k_cumsum_axis1_fp8e5m2(const uint8_t* in, uint8_t* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cs_fp8[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cs_fp8[i1] = (i1 < n) ? fp8e5m2_to_f32(in[r * n + i1]) : 0.0f;
        smem_cs_fp8[i2] = (i2 < n) ? fp8e5m2_to_f32(in[r * n + i2]) : 0.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cs_fp8[idx] += smem_cs_fp8[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cs_fp8[idx + stride] += smem_cs_fp8[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = fp8e5m2_from_f32(smem_cs_fp8[i1]);
        if (i2 < n) out[r * n + i2] = fp8e5m2_from_f32(smem_cs_fp8[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 0.0f;
            for (unsigned c = 0; c < n; c++) {
                acc += fp8e5m2_to_f32(in[r * n + c]);
                out[r * n + c] = fp8e5m2_from_f32(acc);
            }
        }
    }
}

extern "C" __global__ void k_cumsum_axis1_fp4e2m1(const uint8_t* in, uint8_t* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cs_fp8[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cs_fp8[i1] = (i1 < n) ? fp4e2m1_to_f32(in[r * n + i1]) : 0.0f;
        smem_cs_fp8[i2] = (i2 < n) ? fp4e2m1_to_f32(in[r * n + i2]) : 0.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cs_fp8[idx] += smem_cs_fp8[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cs_fp8[idx + stride] += smem_cs_fp8[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = fp4e2m1_from_f32(smem_cs_fp8[i1]);
        if (i2 < n) out[r * n + i2] = fp4e2m1_from_f32(smem_cs_fp8[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 0.0f;
            for (unsigned c = 0; c < n; c++) {
                acc += fp4e2m1_to_f32(in[r * n + c]);
                out[r * n + c] = fp4e2m1_from_f32(acc);
            }
        }
    }
}
extern "C" __global__ void k_cumprod_axis1_f32(const float* in, float* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cp_f32[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cp_f32[i1] = (i1 < n) ? in[r * n + i1] : 1.0f;
        smem_cp_f32[i2] = (i2 < n) ? in[r * n + i2] : 1.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cp_f32[idx] *= smem_cp_f32[idx - stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) smem_cp_f32[2 * bx - 1] = 1.0f;
        __syncthreads();
        for (unsigned stride = bx; stride >= 1; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) {
                float t = smem_cp_f32[idx - stride];
                smem_cp_f32[idx - stride] = smem_cp_f32[idx];
                smem_cp_f32[idx] *= t;
            }
            __syncthreads();
        }
        // Inclusive: smem[i] (exclusive) * original[i]
        if (i1 < n) out[r * n + i1] = smem_cp_f32[i1] * in[r * n + i1];
        if (i2 < n) out[r * n + i2] = smem_cp_f32[i2] * in[r * n + i2];
    } else {
        if (threadIdx.x == 0) {
            float acc = 1.0f;
            for (unsigned c = 0; c < n; c++) {
                acc *= in[r * n + c];
                out[r * n + c] = acc;
            }
        }
    }
}
extern "C" __global__ void k_cumprod_axis1_f64(const double* in, double* out, unsigned rows, unsigned cols) {
    extern __shared__ double smem_cp_f64[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cp_f64[i1] = (i1 < n) ? in[r * n + i1] : 1.0;
        smem_cp_f64[i2] = (i2 < n) ? in[r * n + i2] : 1.0;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cp_f64[idx] *= smem_cp_f64[idx - stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) smem_cp_f64[2 * bx - 1] = 1.0;
        __syncthreads();
        for (unsigned stride = bx; stride >= 1; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) {
                double t = smem_cp_f64[idx - stride];
                smem_cp_f64[idx - stride] = smem_cp_f64[idx];
                smem_cp_f64[idx] *= t;
            }
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = smem_cp_f64[i1] * in[r * n + i1];
        if (i2 < n) out[r * n + i2] = smem_cp_f64[i2] * in[r * n + i2];
    } else {
        if (threadIdx.x == 0) {
            double acc = 1.0;
            for (unsigned c = 0; c < n; c++) {
                acc *= in[r * n + c];
                out[r * n + c] = acc;
            }
        }
    }
}

extern "C" __global__ void k_cumprod_axis1_f16(const __half* in, __half* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cp_f16[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cp_f16[i1] = (i1 < n) ? from_half(in[r * n + i1]) : 1.0f;
        smem_cp_f16[i2] = (i2 < n) ? from_half(in[r * n + i2]) : 1.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cp_f16[idx] *= smem_cp_f16[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cp_f16[idx + stride] *= smem_cp_f16[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = to_half(smem_cp_f16[i1]);
        if (i2 < n) out[r * n + i2] = to_half(smem_cp_f16[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 1.0f;
            for (unsigned c = 0; c < n; c++) {
                acc *= from_half(in[r * n + c]);
                out[r * n + c] = to_half(acc);
            }
        }
    }
}

extern "C" __global__ void k_cumprod_axis1_fp8e4m3(const uint8_t* in, uint8_t* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cp_fp8[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cp_fp8[i1] = (i1 < n) ? fp8e4m3_to_f32(in[r * n + i1]) : 1.0f;
        smem_cp_fp8[i2] = (i2 < n) ? fp8e4m3_to_f32(in[r * n + i2]) : 1.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cp_fp8[idx] *= smem_cp_fp8[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cp_fp8[idx + stride] *= smem_cp_fp8[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = fp8e4m3_from_f32(smem_cp_fp8[i1]);
        if (i2 < n) out[r * n + i2] = fp8e4m3_from_f32(smem_cp_fp8[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 1.0f;
            for (unsigned c = 0; c < n; c++) {
                acc *= fp8e4m3_to_f32(in[r * n + c]);
                out[r * n + c] = fp8e4m3_from_f32(acc);
            }
        }
    }
}

extern "C" __global__ void k_cumprod_axis1_fp8e5m2(const uint8_t* in, uint8_t* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cp_fp8[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cp_fp8[i1] = (i1 < n) ? fp8e5m2_to_f32(in[r * n + i1]) : 1.0f;
        smem_cp_fp8[i2] = (i2 < n) ? fp8e5m2_to_f32(in[r * n + i2]) : 1.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cp_fp8[idx] *= smem_cp_fp8[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cp_fp8[idx + stride] *= smem_cp_fp8[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = fp8e5m2_from_f32(smem_cp_fp8[i1]);
        if (i2 < n) out[r * n + i2] = fp8e5m2_from_f32(smem_cp_fp8[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 1.0f;
            for (unsigned c = 0; c < n; c++) {
                acc *= fp8e5m2_to_f32(in[r * n + c]);
                out[r * n + c] = fp8e5m2_from_f32(acc);
            }
        }
    }
}

extern "C" __global__ void k_cumprod_axis1_fp4e2m1(const uint8_t* in, uint8_t* out, unsigned rows, unsigned cols) {
    extern __shared__ float smem_cp_fp8[];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    unsigned bx = blockDim.x;
    unsigned n = cols;
    if (n <= 2 * bx) {
        unsigned i1 = 2 * threadIdx.x;
        unsigned i2 = 2 * threadIdx.x + 1;
        smem_cp_fp8[i1] = (i1 < n) ? fp4e2m1_to_f32(in[r * n + i1]) : 1.0f;
        smem_cp_fp8[i2] = (i2 < n) ? fp4e2m1_to_f32(in[r * n + i2]) : 1.0f;
        __syncthreads();
        for (unsigned stride = 1; stride < 2 * bx; stride <<= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx < 2 * bx) smem_cp_fp8[idx] *= smem_cp_fp8[idx - stride];
            __syncthreads();
        }
        for (unsigned stride = bx; stride > 0; stride >>= 1) {
            unsigned idx = (threadIdx.x + 1) * 2 * stride - 1;
            if (idx + stride < 2 * bx) smem_cp_fp8[idx + stride] *= smem_cp_fp8[idx];
            __syncthreads();
        }
        if (i1 < n) out[r * n + i1] = fp4e2m1_from_f32(smem_cp_fp8[i1]);
        if (i2 < n) out[r * n + i2] = fp4e2m1_from_f32(smem_cp_fp8[i2]);
    } else {
        if (threadIdx.x == 0) {
            float acc = 1.0f;
            for (unsigned c = 0; c < n; c++) {
                acc *= fp4e2m1_to_f32(in[r * n + c]);
                out[r * n + c] = fp4e2m1_from_f32(acc);
            }
        }
    }
}
