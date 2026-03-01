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

extern "C" __global__ void k_prod_partial_f32(
    const float* __restrict__ in, float* __restrict__ partial, unsigned N
) {
    __shared__ float smem_pp_f32[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem_pp_f32[tid] = (idx < N) ? in[idx] : 1.0f;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_pp_f32[tid] *= smem_pp_f32[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem_pp_f32[0];
}
extern "C" __global__ void k_prod_partial_f64(
    const double* __restrict__ in, double* __restrict__ partial, unsigned N
) {
    __shared__ double smem_pp_f64[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem_pp_f64[tid] = (idx < N) ? in[idx] : 1.0;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_pp_f64[tid] *= smem_pp_f64[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem_pp_f64[0];
}

extern "C" __global__ void k_max_pool2d_with_idx_f32(
    const float* __restrict__ in, float* __restrict__ out, float* __restrict__ idx_out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned pos = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (pos >= total) return;
    unsigned ow = pos % outW;
    unsigned oh = (pos / outW) % outH;
    unsigned n  = pos / (outH * outW);
    float max_val = -3.402823466e+38f;
    unsigned best_idx = 0;
    for (unsigned kh = 0; kh < kH; kh++) {
        for (unsigned kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * sH + kh) - (int)pH;
            int iw = (int)(ow * sW + kw) - (int)pW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                unsigned flat = n * H * W + (unsigned)ih * W + (unsigned)iw;
                float v = in[flat];
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    }
    out[pos] = max_val;
    idx_out[pos] = (float)best_idx;
}
extern "C" __global__ void k_max_pool2d_with_idx_f64(
    const double* __restrict__ in, double* __restrict__ out, double* __restrict__ idx_out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned pos = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (pos >= total) return;
    unsigned ow = pos % outW;
    unsigned oh = (pos / outW) % outH;
    unsigned n  = pos / (outH * outW);
    double max_val = -1.7976931348623157e+308;
    unsigned best_idx = 0;
    for (unsigned kh = 0; kh < kH; kh++) {
        for (unsigned kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * sH + kh) - (int)pH;
            int iw = (int)(ow * sW + kw) - (int)pW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                unsigned flat = n * H * W + (unsigned)ih * W + (unsigned)iw;
                double v = in[flat];
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    }
    out[pos] = max_val;
    idx_out[pos] = (double)best_idx;
}

extern "C" __global__ void k_max_pool2d_f32(
    const float* __restrict__ in, float* __restrict__ out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned idx = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (idx >= total) return;
    unsigned ow = idx % outW;
    unsigned oh = (idx / outW) % outH;
    unsigned n  = idx / (outH * outW);
    float max_val = -3.402823466e+38f;
    for (unsigned kh = 0; kh < kH; kh++) {
        for (unsigned kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * sH + kh) - (int)pH;
            int iw = (int)(ow * sW + kw) - (int)pW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                float v = in[n * H * W + ih * W + iw];
                if (v > max_val) max_val = v;
            }
        }
    }
    out[idx] = max_val;
}
extern "C" __global__ void k_max_pool2d_f64(
    const double* __restrict__ in, double* __restrict__ out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned idx = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (idx >= total) return;
    unsigned ow = idx % outW;
    unsigned oh = (idx / outW) % outH;
    unsigned n  = idx / (outH * outW);
    double max_val = -1.7976931348623157e+308;
    for (unsigned kh = 0; kh < kH; kh++) {
        for (unsigned kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * sH + kh) - (int)pH;
            int iw = (int)(ow * sW + kw) - (int)pW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                double v = in[n * H * W + ih * W + iw];
                if (v > max_val) max_val = v;
            }
        }
    }
    out[idx] = max_val;
}

extern "C" __global__ void k_avg_pool2d_f32(
    const float* __restrict__ in, float* __restrict__ out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned idx = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (idx >= total) return;
    unsigned ow = idx % outW;
    unsigned oh = (idx / outW) % outH;
    unsigned n  = idx / (outH * outW);
    float sum = 0.0f; unsigned cnt = 0;
    for (unsigned kh = 0; kh < kH; kh++) {
        for (unsigned kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * sH + kh) - (int)pH;
            int iw = (int)(ow * sW + kw) - (int)pW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                sum += in[n * H * W + ih * W + iw]; cnt++;
            }
        }
    }
    out[idx] = cnt > 0 ? sum / (float)cnt : 0.0f;
}
extern "C" __global__ void k_avg_pool2d_f64(
    const double* __restrict__ in, double* __restrict__ out,
    unsigned H, unsigned W, unsigned kH, unsigned kW,
    unsigned sH, unsigned sW, unsigned pH, unsigned pW,
    unsigned outH, unsigned outW, unsigned NC
) {
    unsigned idx = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (idx >= total) return;
    unsigned ow = idx % outW;
    unsigned oh = (idx / outW) % outH;
    unsigned n  = idx / (outH * outW);
    double sum = 0.0; unsigned cnt = 0;
    for (unsigned kh = 0; kh < kH; kh++) {
        for (unsigned kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * sH + kh) - (int)pH;
            int iw = (int)(ow * sW + kw) - (int)pW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                sum += in[n * H * W + ih * W + iw]; cnt++;
            }
        }
    }
    out[idx] = cnt > 0 ? sum / (double)cnt : 0.0;
}

extern "C" __global__ void k_adaptive_avg_pool2d_f32(
    const float* __restrict__ in, float* __restrict__ out,
    unsigned inH, unsigned inW, unsigned outH, unsigned outW, unsigned NC
) {
    unsigned idx = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (idx >= total) return;
    unsigned ow = idx % outW;
    unsigned oh = (idx / outW) % outH;
    unsigned n  = idx / (outH * outW);
    unsigned ih_start = oh * inH / outH;
    unsigned ih_end = (oh + 1) * inH / outH; if (ih_end <= ih_start) ih_end = ih_start + 1;
    unsigned iw_start = ow * inW / outW;
    unsigned iw_end = (ow + 1) * inW / outW; if (iw_end <= iw_start) iw_end = iw_start + 1;
    float sum = 0.0f; unsigned cnt = 0;
    for (unsigned ih = ih_start; ih < ih_end && ih < inH; ih++) {
        for (unsigned iw = iw_start; iw < iw_end && iw < inW; iw++) {
            sum += in[n * inH * inW + ih * inW + iw]; cnt++;
        }
    }
    out[idx] = cnt > 0 ? sum / (float)cnt : 0.0f;
}
extern "C" __global__ void k_adaptive_avg_pool2d_f64(
    const double* __restrict__ in, double* __restrict__ out,
    unsigned inH, unsigned inW, unsigned outH, unsigned outW, unsigned NC
) {
    unsigned idx = THREAD_ID;
    unsigned total = NC * outH * outW;
    if (idx >= total) return;
    unsigned ow = idx % outW;
    unsigned oh = (idx / outW) % outH;
    unsigned n  = idx / (outH * outW);
    unsigned ih_start = oh * inH / outH;
    unsigned ih_end = (oh + 1) * inH / outH; if (ih_end <= ih_start) ih_end = ih_start + 1;
    unsigned iw_start = ow * inW / outW;
    unsigned iw_end = (ow + 1) * inW / outW; if (iw_end <= iw_start) iw_end = iw_start + 1;
    double sum = 0.0; unsigned cnt = 0;
    for (unsigned ih = ih_start; ih < ih_end && ih < inH; ih++) {
        for (unsigned iw = iw_start; iw < iw_end && iw < inW; iw++) {
            sum += in[n * inH * inW + ih * inW + iw]; cnt++;
        }
    }
    out[idx] = cnt > 0 ? sum / (double)cnt : 0.0;
}

