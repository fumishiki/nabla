// k_norm_group.cuh — rms_norm, layer_norm, group_norm (fwd + bwd)

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

