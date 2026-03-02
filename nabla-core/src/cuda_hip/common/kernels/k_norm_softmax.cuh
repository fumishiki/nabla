// k_norm_softmax.cuh — Softmax fwd/bwd (row-wise, shared-memory warp reduction)

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

extern "C" __global__ void k_softmax_f16(const __half* __restrict__ in,
                                          __half* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const __half* x = in + row * cols;
    __half* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float m = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmaxf(m, from_half(x[i]));
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

    float s = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += __expf(from_half(x[i]) - m);
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
    float inv_s = 1.0f / ssum[0];
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = __expf(from_half(x[i]) - m) * inv_s;
        y[i] = to_half(v);
    }
}

extern "C" __global__ void k_softmax_fp8e4m3(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float m = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmaxf(m, fp8e4m3_to_f32(x[i]));
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

    float s = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += __expf(fp8e4m3_to_f32(x[i]) - m);
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
    float inv_s = 1.0f / ssum[0];
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = __expf(fp8e4m3_to_f32(x[i]) - m) * inv_s;
        y[i] = fp8e4m3_from_f32(v);
    }
}

extern "C" __global__ void k_softmax_fp8e5m2(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float m = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmaxf(m, fp8e5m2_to_f32(x[i]));
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

    float s = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += __expf(fp8e5m2_to_f32(x[i]) - m);
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
    float inv_s = 1.0f / ssum[0];
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = __expf(fp8e5m2_to_f32(x[i]) - m) * inv_s;
        y[i] = fp8e5m2_from_f32(v);
    }
}

extern "C" __global__ void k_softmax_fp4e2m1(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          unsigned rows, unsigned cols) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float m = -__int_as_float(0x7f800000);
    for (unsigned i = tid; i < cols; i += blockDim.x)
        m = fmaxf(m, fp4e2m1_to_f32(x[i]));
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

    float s = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        s += __expf(fp4e2m1_to_f32(x[i]) - m);
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
    float inv_s = 1.0f / ssum[0];
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = __expf(fp4e2m1_to_f32(x[i]) - m) * inv_s;
        y[i] = fp4e2m1_from_f32(v);
    }
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

extern "C" __global__ void k_layer_norm_f16(
    const __half* __restrict__ in,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const __half* x = in + row * cols;
    __half* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sum += from_half(x[i]);
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

    float var_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float d = from_half(x[i]) - mean;
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
    float inv_std = 1.0f / sqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = from_half(x[i]);
        float gv = from_half(gamma[i]);
        float bv = from_half(beta[i]);
        y[i] = to_half((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_layer_norm_fp8e4m3(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sum += fp8e4m3_to_f32(x[i]);
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

    float var_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float d = fp8e4m3_to_f32(x[i]) - mean;
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
    float inv_std = 1.0f / sqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = fp8e4m3_to_f32(x[i]);
        float gv = fp8e4m3_to_f32(gamma[i]);
        float bv = fp8e4m3_to_f32(beta[i]);
        y[i] = fp8e4m3_from_f32((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_layer_norm_fp8e5m2(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sum += fp8e5m2_to_f32(x[i]);
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

    float var_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float d = fp8e5m2_to_f32(x[i]) - mean;
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
    float inv_std = 1.0f / sqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = fp8e5m2_to_f32(x[i]);
        float gv = fp8e5m2_to_f32(gamma[i]);
        float bv = fp8e5m2_to_f32(beta[i]);
        y[i] = fp8e5m2_from_f32((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_layer_norm_fp4e2m1(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x)
        sum += fp4e2m1_to_f32(x[i]);
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

    float var_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float d = fp4e2m1_to_f32(x[i]) - mean;
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
    float inv_std = 1.0f / sqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = fp4e2m1_to_f32(x[i]);
        float gv = fp4e2m1_to_f32(gamma[i]);
        float bv = fp4e2m1_to_f32(beta[i]);
        y[i] = fp4e2m1_from_f32((xv - mean) * inv_std * gv + bv);
    }
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

