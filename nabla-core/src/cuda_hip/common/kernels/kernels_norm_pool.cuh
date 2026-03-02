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

extern "C" __global__ void k_rms_norm_f16(
    const __half* __restrict__ in,
    const __half* __restrict__ gamma,
    __half* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const __half* x = in + row * cols;
    __half* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sq_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = from_half(x[i]);
        sq_sum += v * v;
    }
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
    float inv_rms = rsqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = from_half(x[i]);
        float gv = from_half(gamma[i]);
        y[i] = to_half(xv * inv_rms * gv);
    }
}

extern "C" __global__ void k_rms_norm_fp8e4m3(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sq_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = fp8e4m3_to_f32(x[i]);
        sq_sum += v * v;
    }
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
    float inv_rms = rsqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = fp8e4m3_to_f32(x[i]);
        float gv = fp8e4m3_to_f32(gamma[i]);
        y[i] = fp8e4m3_from_f32(xv * inv_rms * gv);
    }
}

extern "C" __global__ void k_rms_norm_fp8e5m2(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sq_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = fp8e5m2_to_f32(x[i]);
        sq_sum += v * v;
    }
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
    float inv_rms = rsqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = fp8e5m2_to_f32(x[i]);
        float gv = fp8e5m2_to_f32(gamma[i]);
        y[i] = fp8e5m2_from_f32(xv * inv_rms * gv);
    }
}

extern "C" __global__ void k_rms_norm_fp4e2m1(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, double eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t* x = in + row * cols;
    uint8_t* y = out + row * cols;
    unsigned tid = threadIdx.x;

    float sq_sum = 0.0f;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float v = fp4e2m1_to_f32(x[i]);
        sq_sum += v * v;
    }
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
    float inv_rms = rsqrtf(sdata[0] / (float)cols + (float)eps);

    for (unsigned i = tid; i < cols; i += blockDim.x) {
        float xv = fp4e2m1_to_f32(x[i]);
        float gv = fp4e2m1_to_f32(gamma[i]);
        y[i] = fp4e2m1_from_f32(xv * inv_rms * gv);
    }
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

extern "C" __global__ void k_group_norm_f32(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, float eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const float* x = in + row * cols + g_start;
    float* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
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
    float mean = sdata[0] / (float)g_size;

    float var_sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
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
    float inv_std = rsqrtf(sdata[0] / (float)g_size + eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        y[i] = (x[i] - mean) * inv_std * gamma[ci] + beta[ci];
    }
}

extern "C" __global__ void k_group_norm_f16(
    const __half* __restrict__ in,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, double eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const __half* x = in + row * cols + g_start;
    __half* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
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
    float mean = sdata[0] / (float)g_size;

    float var_sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
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
    float inv_std = rsqrtf(sdata[0] / (float)g_size + (float)eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        float xv = from_half(x[i]);
        float gv = from_half(gamma[ci]);
        float bv = from_half(beta[ci]);
        y[i] = to_half((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_group_norm_fp8e4m3(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, double eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const uint8_t* x = in + row * cols + g_start;
    uint8_t* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
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
    float mean = sdata[0] / (float)g_size;

    float var_sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
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
    float inv_std = rsqrtf(sdata[0] / (float)g_size + (float)eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        float xv = fp8e4m3_to_f32(x[i]);
        float gv = fp8e4m3_to_f32(gamma[ci]);
        float bv = fp8e4m3_to_f32(beta[ci]);
        y[i] = fp8e4m3_from_f32((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_group_norm_fp8e5m2(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, double eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const uint8_t* x = in + row * cols + g_start;
    uint8_t* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
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
    float mean = sdata[0] / (float)g_size;

    float var_sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
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
    float inv_std = rsqrtf(sdata[0] / (float)g_size + (float)eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        float xv = fp8e5m2_to_f32(x[i]);
        float gv = fp8e5m2_to_f32(gamma[ci]);
        float bv = fp8e5m2_to_f32(beta[ci]);
        y[i] = fp8e5m2_from_f32((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_group_norm_fp4e2m1(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    uint8_t* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, double eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const uint8_t* x = in + row * cols + g_start;
    uint8_t* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
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
    float mean = sdata[0] / (float)g_size;

    float var_sum = 0.0f;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
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
    float inv_std = rsqrtf(sdata[0] / (float)g_size + (float)eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        float xv = fp4e2m1_to_f32(x[i]);
        float gv = fp4e2m1_to_f32(gamma[ci]);
        float bv = fp4e2m1_to_f32(beta[ci]);
        y[i] = fp4e2m1_from_f32((xv - mean) * inv_std * gv + bv);
    }
}

extern "C" __global__ void k_group_norm_f64(
    const double* __restrict__ in,
    const double* __restrict__ gamma,
    const double* __restrict__ beta,
    double* __restrict__ out,
    unsigned rows, unsigned cols, unsigned groups, double eps) {
    unsigned gid = blockIdx.x;
    unsigned row = gid / groups;
    if (row >= rows) return;
    unsigned g = gid % groups;
    unsigned g_size = cols / groups;
    unsigned g_start = g * g_size;
    const double* x = in + row * cols + g_start;
    double* y = out + row * cols + g_start;
    unsigned tid = threadIdx.x;

    double sum = 0.0;
    for (unsigned i = tid; i < g_size; i += blockDim.x)
        sum += x[i];
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
    double mean = sdata[0] / (double)g_size;

    double var_sum = 0.0;
    for (unsigned i = tid; i < g_size; i += blockDim.x) {
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
    double inv_std = 1.0 / sqrt(sdata[0] / (double)g_size + eps);

    for (unsigned i = tid; i < g_size; i += blockDim.x) {
        unsigned ci = g_start + i;
        y[i] = (x[i] - mean) * inv_std * gamma[ci] + beta[ci];
    }
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

extern "C" __global__ void k_prod_partial_f16(
    const __half* __restrict__ in, __half* __restrict__ partial, unsigned N
) {
    __shared__ float smem_pp_f16[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem_pp_f16[tid] = (idx < N) ? from_half(in[idx]) : 1.0f;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_pp_f16[tid] *= smem_pp_f16[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = to_half(smem_pp_f16[0]);
}

extern "C" __global__ void k_prod_partial_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ partial, unsigned N
) {
    __shared__ float smem_pp_fp8[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem_pp_fp8[tid] = (idx < N) ? fp8e4m3_to_f32(in[idx]) : 1.0f;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_pp_fp8[tid] *= smem_pp_fp8[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = fp8e4m3_from_f32(smem_pp_fp8[0]);
}

extern "C" __global__ void k_prod_partial_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ partial, unsigned N
) {
    __shared__ float smem_pp_fp8[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem_pp_fp8[tid] = (idx < N) ? fp8e5m2_to_f32(in[idx]) : 1.0f;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_pp_fp8[tid] *= smem_pp_fp8[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = fp8e5m2_from_f32(smem_pp_fp8[0]);
}

extern "C" __global__ void k_prod_partial_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ partial, unsigned N
) {
    __shared__ float smem_pp_fp8[256];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    smem_pp_fp8[tid] = (idx < N) ? fp4e2m1_to_f32(in[idx]) : 1.0f;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_pp_fp8[tid] *= smem_pp_fp8[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = fp4e2m1_from_f32(smem_pp_fp8[0]);
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

extern "C" __global__ void k_max_pool2d_with_idx_f16(
    const __half* __restrict__ in, __half* __restrict__ out, __half* __restrict__ idx_out,
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
                float v = from_half(in[flat]);
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    }
    out[pos] = to_half(max_val);
    idx_out[pos] = to_half((float)best_idx);
}

extern "C" __global__ void k_max_pool2d_with_idx_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out, uint8_t* __restrict__ idx_out,
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
                float v = fp8e4m3_to_f32(in[flat]);
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    }
    out[pos] = fp8e4m3_from_f32(max_val);
    idx_out[pos] = fp8e4m3_from_f32((float)best_idx);
}

extern "C" __global__ void k_max_pool2d_with_idx_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out, uint8_t* __restrict__ idx_out,
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
                float v = fp8e5m2_to_f32(in[flat]);
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    }
    out[pos] = fp8e5m2_from_f32(max_val);
    idx_out[pos] = fp8e5m2_from_f32((float)best_idx);
}

extern "C" __global__ void k_max_pool2d_with_idx_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out, uint8_t* __restrict__ idx_out,
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
                float v = fp4e2m1_to_f32(in[flat]);
                if (v > max_val) { max_val = v; best_idx = flat; }
            }
        }
    }
    out[pos] = fp4e2m1_from_f32(max_val);
    idx_out[pos] = fp4e2m1_from_f32((float)best_idx);
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

extern "C" __global__ void k_max_pool2d_f16(
    const __half* __restrict__ in, __half* __restrict__ out,
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
                float v = from_half(in[n * H * W + ih * W + iw]);
                if (v > max_val) max_val = v;
            }
        }
    }
    out[idx] = to_half(max_val);
}

extern "C" __global__ void k_max_pool2d_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
                float v = fp8e4m3_to_f32(in[n * H * W + ih * W + iw]);
                if (v > max_val) max_val = v;
            }
        }
    }
    out[idx] = fp8e4m3_from_f32(max_val);
}

extern "C" __global__ void k_max_pool2d_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
                float v = fp8e5m2_to_f32(in[n * H * W + ih * W + iw]);
                if (v > max_val) max_val = v;
            }
        }
    }
    out[idx] = fp8e5m2_from_f32(max_val);
}

extern "C" __global__ void k_max_pool2d_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
                float v = fp4e2m1_to_f32(in[n * H * W + ih * W + iw]);
                if (v > max_val) max_val = v;
            }
        }
    }
    out[idx] = fp4e2m1_from_f32(max_val);
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

extern "C" __global__ void k_avg_pool2d_f16(
    const __half* __restrict__ in, __half* __restrict__ out,
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
                sum += from_half(in[n * H * W + ih * W + iw]); cnt++;
            }
        }
    }
    out[idx] = to_half(cnt > 0 ? sum / (float)cnt : 0.0f);
}

extern "C" __global__ void k_avg_pool2d_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
                sum += fp8e4m3_to_f32(in[n * H * W + ih * W + iw]); cnt++;
            }
        }
    }
    float v = cnt > 0 ? sum / (float)cnt : 0.0f;
    out[idx] = fp8e4m3_from_f32(v);
}

extern "C" __global__ void k_avg_pool2d_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
                sum += fp8e5m2_to_f32(in[n * H * W + ih * W + iw]); cnt++;
            }
        }
    }
    float v = cnt > 0 ? sum / (float)cnt : 0.0f;
    out[idx] = fp8e5m2_from_f32(v);
}

extern "C" __global__ void k_avg_pool2d_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
                sum += fp4e2m1_to_f32(in[n * H * W + ih * W + iw]); cnt++;
            }
        }
    }
    float v = cnt > 0 ? sum / (float)cnt : 0.0f;
    out[idx] = fp4e2m1_from_f32(v);
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

extern "C" __global__ void k_adaptive_avg_pool2d_f16(
    const __half* __restrict__ in, __half* __restrict__ out,
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
            sum += from_half(in[n * inH * inW + ih * inW + iw]); cnt++;
        }
    }
    out[idx] = to_half(cnt > 0 ? sum / (float)cnt : 0.0f);
}

extern "C" __global__ void k_adaptive_avg_pool2d_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
            sum += fp8e4m3_to_f32(in[n * inH * inW + ih * inW + iw]); cnt++;
        }
    }
    float v = cnt > 0 ? sum / (float)cnt : 0.0f;
    out[idx] = fp8e4m3_from_f32(v);
}

extern "C" __global__ void k_adaptive_avg_pool2d_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
            sum += fp8e5m2_to_f32(in[n * inH * inW + ih * inW + iw]); cnt++;
        }
    }
    float v = cnt > 0 ? sum / (float)cnt : 0.0f;
    out[idx] = fp8e5m2_from_f32(v);
}

extern "C" __global__ void k_adaptive_avg_pool2d_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
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
            sum += fp4e2m1_to_f32(in[n * inH * inW + ih * inW + iw]); cnt++;
        }
    }
    float v = cnt > 0 ? sum / (float)cnt : 0.0f;
    out[idx] = fp4e2m1_from_f32(v);
}
