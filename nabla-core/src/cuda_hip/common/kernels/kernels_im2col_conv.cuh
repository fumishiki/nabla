extern "C" __global__ void k_im2col_f32(
    const float* __restrict__ in, float* __restrict__ col,
    unsigned C_in, unsigned H, unsigned W,
    unsigned kH, unsigned kW,
    unsigned sH, unsigned sW,
    unsigned pH, unsigned pW,
    unsigned dH, unsigned dW,
    unsigned out_H, unsigned out_W
) {
    unsigned col_elem = C_in * kH * kW * out_H * out_W;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp = tmp / out_H;
    unsigned kw = tmp % kW;
    tmp = tmp / kW;
    unsigned kh = tmp % kH;
    unsigned c  = tmp / kH;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    float val = 0.0f;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = in[n * C_in * H * W + c * H * W + ih * W + iw];
    }
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val;
}

extern "C" __global__ void k_im2col_f64(
    const double* __restrict__ in, double* __restrict__ col,
    unsigned C_in, unsigned H, unsigned W,
    unsigned kH, unsigned kW,
    unsigned sH, unsigned sW,
    unsigned pH, unsigned pW,
    unsigned dH, unsigned dW,
    unsigned out_H, unsigned out_W
) {
    unsigned col_elem = C_in * kH * kW * out_H * out_W;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp = tmp / out_H;
    unsigned kw = tmp % kW;
    tmp = tmp / kW;
    unsigned kh = tmp % kH;
    unsigned c  = tmp / kH;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    double val = 0.0;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = in[n * C_in * H * W + c * H * W + ih * W + iw];
    }
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val;
}

extern "C" __global__ void k_im2col_f16(
    const __half* __restrict__ in, __half* __restrict__ col,
    unsigned C_in, unsigned H, unsigned W,
    unsigned kH, unsigned kW,
    unsigned sH, unsigned sW,
    unsigned pH, unsigned pW,
    unsigned dH, unsigned dW,
    unsigned out_H, unsigned out_W
) {
    unsigned col_elem = C_in * kH * kW * out_H * out_W;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp = tmp / out_H;
    unsigned kw = tmp % kW;
    tmp = tmp / kW;
    unsigned kh = tmp % kH;
    unsigned c  = tmp / kH;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    __half val = to_half(0.0f);
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = in[n * C_in * H * W + c * H * W + ih * W + iw];
    }
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val;
}

extern "C" __global__ void k_im2col_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned H, unsigned W,
    unsigned kH, unsigned kW,
    unsigned sH, unsigned sW,
    unsigned pH, unsigned pW,
    unsigned dH, unsigned dW,
    unsigned out_H, unsigned out_W
) {
    unsigned col_elem = C_in * kH * kW * out_H * out_W;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp = tmp / out_H;
    unsigned kw = tmp % kW;
    tmp = tmp / kW;
    unsigned kh = tmp % kH;
    unsigned c  = tmp / kH;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    uint8_t val = fp8e4m3_from_f32(0.0f);
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = in[n * C_in * H * W + c * H * W + ih * W + iw];
    }
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val;
}

extern "C" __global__ void k_im2col_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned H, unsigned W,
    unsigned kH, unsigned kW,
    unsigned sH, unsigned sW,
    unsigned pH, unsigned pW,
    unsigned dH, unsigned dW,
    unsigned out_H, unsigned out_W
) {
    unsigned col_elem = C_in * kH * kW * out_H * out_W;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp = tmp / out_H;
    unsigned kw = tmp % kW;
    tmp = tmp / kW;
    unsigned kh = tmp % kH;
    unsigned c  = tmp / kH;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    uint8_t val = fp8e5m2_from_f32(0.0f);
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = in[n * C_in * H * W + c * H * W + ih * W + iw];
    }
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val;
}

extern "C" __global__ void k_im2col_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned H, unsigned W,
    unsigned kH, unsigned kW,
    unsigned sH, unsigned sW,
    unsigned pH, unsigned pW,
    unsigned dH, unsigned dW,
    unsigned out_H, unsigned out_W
) {
    unsigned col_elem = C_in * kH * kW * out_H * out_W;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp = tmp / out_H;
    unsigned kw = tmp % kW;
    tmp = tmp / kW;
    unsigned kh = tmp % kH;
    unsigned c  = tmp / kH;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    uint8_t val = fp4e2m1_from_f32(0.0f);
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = in[n * C_in * H * W + c * H * W + ih * W + iw];
    }
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val;
}

extern "C" __global__ void k_batch_norm_stats_f32(
    const float* __restrict__ in,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    unsigned N, unsigned C
) {
    unsigned c = THREAD_ID;
    if (c >= C) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (unsigned n = 0; n < N; n++) {
        float v = in[n * C + c];
        sum += v; sum2 += v * v;
    }
    float m = sum / (float)N;
    mean_out[c] = m;
    var_out[c] = sum2 / (float)N - m * m;
}

extern "C" __global__ void k_batch_norm_stats_f64(
    const double* __restrict__ in,
    double* __restrict__ mean_out,
    double* __restrict__ var_out,
    unsigned N, unsigned C
) {
    unsigned c = THREAD_ID;
    if (c >= C) return;
    double sum = 0.0, sum2 = 0.0;
    for (unsigned n = 0; n < N; n++) {
        double v = in[n * C + c];
        sum += v; sum2 += v * v;
    }
    double m = sum / (double)N;
    mean_out[c] = m;
    var_out[c] = sum2 / (double)N - m * m;
}

extern "C" __global__ void k_batch_norm_stats_f16(
    const __half* __restrict__ in,
    __half* __restrict__ mean_out,
    __half* __restrict__ var_out,
    unsigned N, unsigned C
) {
    unsigned c = THREAD_ID;
    if (c >= C) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (unsigned n = 0; n < N; n++) {
        float v = from_half(in[n * C + c]);
        sum += v; sum2 += v * v;
    }
    float m = sum / (float)N;
    mean_out[c] = to_half(m);
    var_out[c] = to_half(sum2 / (float)N - m * m);
}

extern "C" __global__ void k_batch_norm_stats_fp8e4m3(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ mean_out,
    uint8_t* __restrict__ var_out,
    unsigned N, unsigned C
) {
    unsigned c = THREAD_ID;
    if (c >= C) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (unsigned n = 0; n < N; n++) {
        float v = fp8e4m3_to_f32(in[n * C + c]);
        sum += v; sum2 += v * v;
    }
    float m = sum / (float)N;
    mean_out[c] = fp8e4m3_from_f32(m);
    var_out[c] = fp8e4m3_from_f32(sum2 / (float)N - m * m);
}

extern "C" __global__ void k_batch_norm_stats_fp8e5m2(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ mean_out,
    uint8_t* __restrict__ var_out,
    unsigned N, unsigned C
) {
    unsigned c = THREAD_ID;
    if (c >= C) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (unsigned n = 0; n < N; n++) {
        float v = fp8e5m2_to_f32(in[n * C + c]);
        sum += v; sum2 += v * v;
    }
    float m = sum / (float)N;
    mean_out[c] = fp8e5m2_from_f32(m);
    var_out[c] = fp8e5m2_from_f32(sum2 / (float)N - m * m);
}

extern "C" __global__ void k_batch_norm_stats_fp4e2m1(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ mean_out,
    uint8_t* __restrict__ var_out,
    unsigned N, unsigned C
) {
    unsigned c = THREAD_ID;
    if (c >= C) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (unsigned n = 0; n < N; n++) {
        float v = fp4e2m1_to_f32(in[n * C + c]);
        sum += v; sum2 += v * v;
    }
    float m = sum / (float)N;
    mean_out[c] = fp4e2m1_from_f32(m);
    var_out[c] = fp4e2m1_from_f32(sum2 / (float)N - m * m);
}

extern "C" __global__ void k_batch_norm_fwd_f32(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ out,
    float eps, unsigned total, unsigned C
) {
    unsigned idx = THREAD_ID;
    if (idx >= total) return;
    unsigned c = idx % C;
    float inv_std = rsqrtf(var[c] + eps);
    out[idx] = gamma[c] * (in[idx] - mean[c]) * inv_std + beta[c];
}

extern "C" __global__ void k_batch_norm_fwd_f64(
    const double* __restrict__ in,
    const double* __restrict__ gamma,
    const double* __restrict__ beta,
    const double* __restrict__ mean,
    const double* __restrict__ var,
    double* __restrict__ out,
    double eps, unsigned total, unsigned C
) {
    unsigned idx = THREAD_ID;
    if (idx >= total) return;
    unsigned c = idx % C;
    double inv_std = 1.0 / sqrt(var[c] + eps);
    out[idx] = gamma[c] * (in[idx] - mean[c]) * inv_std + beta[c];
}

extern "C" __global__ void k_batch_norm_fwd_f16(
    const __half* __restrict__ in,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    const __half* __restrict__ mean,
    const __half* __restrict__ var,
    __half* __restrict__ out,
    double eps, unsigned total, unsigned C
) {
    unsigned idx = THREAD_ID;
    if (idx >= total) return;
    unsigned c = idx % C;
    float inv_std = rsqrtf(from_half(var[c]) + (float)eps);
    float v = from_half(gamma[c]) * (from_half(in[idx]) - from_half(mean[c])) * inv_std + from_half(beta[c]);
    out[idx] = to_half(v);
}

extern "C" __global__ void k_batch_norm_fwd_fp8e4m3(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    const uint8_t* __restrict__ mean,
    const uint8_t* __restrict__ var,
    uint8_t* __restrict__ out,
    float eps, unsigned total, unsigned C
) {
    unsigned idx = THREAD_ID;
    if (idx >= total) return;
    unsigned c = idx % C;
    float inv_std = rsqrtf(fp8e4m3_to_f32(var[c]) + eps);
    float v = fp8e4m3_to_f32(gamma[c]) * (fp8e4m3_to_f32(in[idx]) - fp8e4m3_to_f32(mean[c])) * inv_std + fp8e4m3_to_f32(beta[c]);
    out[idx] = fp8e4m3_from_f32(v);
}

extern "C" __global__ void k_batch_norm_fwd_fp8e5m2(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    const uint8_t* __restrict__ mean,
    const uint8_t* __restrict__ var,
    uint8_t* __restrict__ out,
    float eps, unsigned total, unsigned C
) {
    unsigned idx = THREAD_ID;
    if (idx >= total) return;
    unsigned c = idx % C;
    float inv_std = rsqrtf(fp8e5m2_to_f32(var[c]) + eps);
    float v = fp8e5m2_to_f32(gamma[c]) * (fp8e5m2_to_f32(in[idx]) - fp8e5m2_to_f32(mean[c])) * inv_std + fp8e5m2_to_f32(beta[c]);
    out[idx] = fp8e5m2_from_f32(v);
}

extern "C" __global__ void k_batch_norm_fwd_fp4e2m1(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ gamma,
    const uint8_t* __restrict__ beta,
    const uint8_t* __restrict__ mean,
    const uint8_t* __restrict__ var,
    uint8_t* __restrict__ out,
    float eps, unsigned total, unsigned C
) {
    unsigned idx = THREAD_ID;
    if (idx >= total) return;
    unsigned c = idx % C;
    float inv_std = rsqrtf(fp4e2m1_to_f32(var[c]) + eps);
    float v = fp4e2m1_to_f32(gamma[c]) * (fp4e2m1_to_f32(in[idx]) - fp4e2m1_to_f32(mean[c])) * inv_std + fp4e2m1_to_f32(beta[c]);
    out[idx] = fp4e2m1_from_f32(v);
}

extern "C" __global__ void k_cross_entropy_f32(
    const float* __restrict__ in,
    const float* __restrict__ target,
    float* __restrict__ loss_out,
    unsigned N, unsigned C
) {
    unsigned n = THREAD_ID;
    if (n >= N) return;
    const float* row = in + n * C;
    float mx = row[0];
    for (unsigned c = 1; c < C; c++) if (row[c] > mx) mx = row[c];
    float sum_exp = 0.0f;
    for (unsigned c = 0; c < C; c++) sum_exp += expf(row[c] - mx);
    int t = (int)target[n];
    loss_out[n] = -(row[t] - mx - logf(sum_exp));
}

extern "C" __global__ void k_cross_entropy_f64(
    const double* __restrict__ in,
    const double* __restrict__ target,
    double* __restrict__ loss_out,
    unsigned N, unsigned C
) {
    unsigned n = THREAD_ID;
    if (n >= N) return;
    const double* row = in + n * C;
    double mx = row[0];
    for (unsigned c = 1; c < C; c++) if (row[c] > mx) mx = row[c];
    double sum_exp = 0.0;
    for (unsigned c = 0; c < C; c++) sum_exp += exp(row[c] - mx);
    int t = (int)target[n];
    loss_out[n] = -(row[t] - mx - log(sum_exp));
}

extern "C" __global__ void k_cross_entropy_f16(
    const __half* __restrict__ in,
    const __half* __restrict__ target,
    __half* __restrict__ loss_out,
    unsigned N, unsigned C
) {
    unsigned n = THREAD_ID;
    if (n >= N) return;
    const __half* row = in + n * C;
    float mx = from_half(row[0]);
    for (unsigned c = 1; c < C; c++) {
        float v = from_half(row[c]);
        if (v > mx) mx = v;
    }
    float sum_exp = 0.0f;
    for (unsigned c = 0; c < C; c++) sum_exp += expf(from_half(row[c]) - mx);
    int t = (int)from_half(target[n]);
    float loss = -(from_half(row[t]) - mx - logf(sum_exp));
    loss_out[n] = to_half(loss);
}

extern "C" __global__ void k_cross_entropy_fp8e4m3(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ target,
    uint8_t* __restrict__ loss_out,
    unsigned N, unsigned C
) {
    unsigned n = THREAD_ID;
    if (n >= N) return;
    const uint8_t* row = in + n * C;
    float mx = fp8e4m3_to_f32(row[0]);
    for (unsigned c = 1; c < C; c++) {
        float v = fp8e4m3_to_f32(row[c]);
        if (v > mx) mx = v;
    }
    float sum_exp = 0.0f;
    for (unsigned c = 0; c < C; c++) sum_exp += expf(fp8e4m3_to_f32(row[c]) - mx);
    int t = (int)fp8e4m3_to_f32(target[n]);
    float loss = -(fp8e4m3_to_f32(row[t]) - mx - logf(sum_exp));
    loss_out[n] = fp8e4m3_from_f32(loss);
}

extern "C" __global__ void k_cross_entropy_fp8e5m2(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ target,
    uint8_t* __restrict__ loss_out,
    unsigned N, unsigned C
) {
    unsigned n = THREAD_ID;
    if (n >= N) return;
    const uint8_t* row = in + n * C;
    float mx = fp8e5m2_to_f32(row[0]);
    for (unsigned c = 1; c < C; c++) {
        float v = fp8e5m2_to_f32(row[c]);
        if (v > mx) mx = v;
    }
    float sum_exp = 0.0f;
    for (unsigned c = 0; c < C; c++) sum_exp += expf(fp8e5m2_to_f32(row[c]) - mx);
    int t = (int)fp8e5m2_to_f32(target[n]);
    float loss = -(fp8e5m2_to_f32(row[t]) - mx - logf(sum_exp));
    loss_out[n] = fp8e5m2_from_f32(loss);
}

extern "C" __global__ void k_cross_entropy_fp4e2m1(
    const uint8_t* __restrict__ in,
    const uint8_t* __restrict__ target,
    uint8_t* __restrict__ loss_out,
    unsigned N, unsigned C
) {
    unsigned n = THREAD_ID;
    if (n >= N) return;
    const uint8_t* row = in + n * C;
    float mx = fp4e2m1_to_f32(row[0]);
    for (unsigned c = 1; c < C; c++) {
        float v = fp4e2m1_to_f32(row[c]);
        if (v > mx) mx = v;
    }
    float sum_exp = 0.0f;
    for (unsigned c = 0; c < C; c++) sum_exp += expf(fp4e2m1_to_f32(row[c]) - mx);
    int t = (int)fp4e2m1_to_f32(target[n]);
    float loss = -(fp4e2m1_to_f32(row[t]) - mx - logf(sum_exp));
    loss_out[n] = fp4e2m1_from_f32(loss);
}


#define FA_BLOCK_M 64
#define FA_BLOCK_N 64
#define FA_HEAD_DIM_MAX 128

extern "C" __global__ void k_sdpa_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    float scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;   // 0..FA_BLOCK_M-1
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ float smem_f32[];
    float* smem_k = smem_f32;
    float* smem_v = smem_k + FA_BLOCK_N * D;

    // Per-thread registers: query row, running output, running stats.
    float qi[FA_HEAD_DIM_MAX];
    float oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0f; oi[d] = 0.0f; }
    float mi = -3.402823466e+38f;
    float li = 0.0f;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = Q[base + d];
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        // Cooperative load K/V tiles into shared memory.
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float kv = (k_row < seq_k && bh < BH)
                ? K[bh * seq_k * D + k_row * D + d] : 0.0f;
            smem_k[j] = kv;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float vv = (k_row < seq_k && bh < BH)
                ? V[bh * seq_k * D + k_row * D + d] : 0.0f;
            smem_v[j] = vv;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            float sij[FA_BLOCK_N];
            // Sij = qi @ Kj^T * scale
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                unsigned k_row = k_start + kj;
                float dot = 0.0f;
                if (k_row < seq_k) {
                    for (unsigned d = 0; d < D; d++)
                        dot += qi[d] * smem_k[kj * D + d];
                    sij[kj] = dot * scale;
                } else {
                    sij[kj] = -3.402823466e+38f;
                }
            }

            // Row max for this tile.
            float mij = sij[0];
            for (unsigned kj = 1; kj < FA_BLOCK_N; kj++)
                if (sij[kj] > mij) mij = sij[kj];

            // Pij = exp(sij - mij), lij = rowsum.
            float pij[FA_BLOCK_N];
            float lij = 0.0f;
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                pij[kj] = __expf(sij[kj] - mij);
                lij += pij[kj];
            }

            // FA2 online update.
            float mi_new     = (mi > mij) ? mi : mij;
            float scale_old  = __expf(mi - mi_new);
            float scale_new  = __expf(mij - mi_new);

            for (unsigned d = 0; d < D; d++) {
                float pv = 0.0f;
                for (unsigned kj = 0; kj < FA_BLOCK_N; kj++)
                    pv += pij[kj] * smem_v[kj * D + d];
                oi[d] = scale_old * oi[d] + scale_new * pv;
            }
            li = scale_old * li + scale_new * lij;
            mi = mi_new;
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        float inv_li = (li > 0.0f) ? 1.0f / li : 0.0f;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = oi[d] * inv_li;
    }
}

extern "C" __global__ void k_sdpa_f16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    float scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ float smem_f16[];
    float* smem_k = smem_f16;
    float* smem_v = smem_k + FA_BLOCK_N * D;

    float qi[FA_HEAD_DIM_MAX];
    float oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0f; oi[d] = 0.0f; }
    float mi = -3.402823466e+38f;
    float li = 0.0f;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = from_half(Q[base + d]);
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float kv = (k_row < seq_k && bh < BH)
                ? from_half(K[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_k[j] = kv;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float vv = (k_row < seq_k && bh < BH)
                ? from_half(V[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_v[j] = vv;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            float sij[FA_BLOCK_N];
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                unsigned k_row = k_start + kj;
                float dot = 0.0f;
                if (k_row < seq_k) {
                    for (unsigned d = 0; d < D; d++)
                        dot += qi[d] * smem_k[kj * D + d];
                    sij[kj] = dot * scale;
                } else {
                    sij[kj] = -3.402823466e+38f;
                }
            }

            float mij = sij[0];
            for (unsigned kj = 1; kj < FA_BLOCK_N; kj++)
                if (sij[kj] > mij) mij = sij[kj];

            float pij[FA_BLOCK_N];
            float lij = 0.0f;
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                pij[kj] = __expf(sij[kj] - mij);
                lij += pij[kj];
            }

            float mi_new     = (mi > mij) ? mi : mij;
            float scale_old  = __expf(mi - mi_new);
            float scale_new  = __expf(mij - mi_new);

            for (unsigned d = 0; d < D; d++) {
                float pv = 0.0f;
                for (unsigned kj = 0; kj < FA_BLOCK_N; kj++)
                    pv += pij[kj] * smem_v[kj * D + d];
                oi[d] = scale_old * oi[d] + scale_new * pv;
            }
            li = scale_old * li + scale_new * lij;
            mi = mi_new;
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        float inv_li = (li > 0.0f) ? 1.0f / li : 0.0f;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = to_half(oi[d] * inv_li);
    }
}

extern "C" __global__ void k_sdpa_fp8e4m3(
    const uint8_t* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ V,
    uint8_t* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    float scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ float smem_fp8[];
    float* smem_k = smem_fp8;
    float* smem_v = smem_k + FA_BLOCK_N * D;

    float qi[FA_HEAD_DIM_MAX];
    float oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0f; oi[d] = 0.0f; }
    float mi = -3.402823466e+38f;
    float li = 0.0f;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = fp8e4m3_to_f32(Q[base + d]);
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float kv = (k_row < seq_k && bh < BH)
                ? fp8e4m3_to_f32(K[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_k[j] = kv;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float vv = (k_row < seq_k && bh < BH)
                ? fp8e4m3_to_f32(V[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_v[j] = vv;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            float sij[FA_BLOCK_N];
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                unsigned k_row = k_start + kj;
                float dot = 0.0f;
                if (k_row < seq_k) {
                    for (unsigned d = 0; d < D; d++)
                        dot += qi[d] * smem_k[kj * D + d];
                    sij[kj] = dot * scale;
                } else {
                    sij[kj] = -3.402823466e+38f;
                }
            }

            float mij = sij[0];
            for (unsigned kj = 1; kj < FA_BLOCK_N; kj++)
                if (sij[kj] > mij) mij = sij[kj];

            float pij[FA_BLOCK_N];
            float lij = 0.0f;
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                pij[kj] = __expf(sij[kj] - mij);
                lij += pij[kj];
            }

            float mi_new     = (mi > mij) ? mi : mij;
            float scale_old  = __expf(mi - mi_new);
            float scale_new  = __expf(mij - mi_new);

            for (unsigned d = 0; d < D; d++) {
                float pv = 0.0f;
                for (unsigned kj = 0; kj < FA_BLOCK_N; kj++)
                    pv += pij[kj] * smem_v[kj * D + d];
                oi[d] = scale_old * oi[d] + scale_new * pv;
            }
            li = scale_old * li + scale_new * lij;
            mi = mi_new;
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        float inv_li = (li > 0.0f) ? 1.0f / li : 0.0f;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = fp8e4m3_from_f32(oi[d] * inv_li);
    }
}

extern "C" __global__ void k_sdpa_fp8e5m2(
    const uint8_t* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ V,
    uint8_t* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    float scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ float smem_fp8[];
    float* smem_k = smem_fp8;
    float* smem_v = smem_k + FA_BLOCK_N * D;

    float qi[FA_HEAD_DIM_MAX];
    float oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0f; oi[d] = 0.0f; }
    float mi = -3.402823466e+38f;
    float li = 0.0f;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = fp8e5m2_to_f32(Q[base + d]);
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float kv = (k_row < seq_k && bh < BH)
                ? fp8e5m2_to_f32(K[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_k[j] = kv;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float vv = (k_row < seq_k && bh < BH)
                ? fp8e5m2_to_f32(V[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_v[j] = vv;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            float sij[FA_BLOCK_N];
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                unsigned k_row = k_start + kj;
                float dot = 0.0f;
                if (k_row < seq_k) {
                    for (unsigned d = 0; d < D; d++)
                        dot += qi[d] * smem_k[kj * D + d];
                    sij[kj] = dot * scale;
                } else {
                    sij[kj] = -3.402823466e+38f;
                }
            }

            float mij = sij[0];
            for (unsigned kj = 1; kj < FA_BLOCK_N; kj++)
                if (sij[kj] > mij) mij = sij[kj];

            float pij[FA_BLOCK_N];
            float lij = 0.0f;
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                pij[kj] = __expf(sij[kj] - mij);
                lij += pij[kj];
            }

            float mi_new     = (mi > mij) ? mi : mij;
            float scale_old  = __expf(mi - mi_new);
            float scale_new  = __expf(mij - mi_new);

            for (unsigned d = 0; d < D; d++) {
                float pv = 0.0f;
                for (unsigned kj = 0; kj < FA_BLOCK_N; kj++)
                    pv += pij[kj] * smem_v[kj * D + d];
                oi[d] = scale_old * oi[d] + scale_new * pv;
            }
            li = scale_old * li + scale_new * lij;
            mi = mi_new;
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        float inv_li = (li > 0.0f) ? 1.0f / li : 0.0f;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = fp8e5m2_from_f32(oi[d] * inv_li);
    }
}

extern "C" __global__ void k_sdpa_fp4e2m1(
    const uint8_t* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ V,
    uint8_t* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    float scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ float smem_fp8[];
    float* smem_k = smem_fp8;
    float* smem_v = smem_k + FA_BLOCK_N * D;

    float qi[FA_HEAD_DIM_MAX];
    float oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0f; oi[d] = 0.0f; }
    float mi = -3.402823466e+38f;
    float li = 0.0f;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = fp4e2m1_to_f32(Q[base + d]);
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float kv = (k_row < seq_k && bh < BH)
                ? fp4e2m1_to_f32(K[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_k[j] = kv;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            float vv = (k_row < seq_k && bh < BH)
                ? fp4e2m1_to_f32(V[bh * seq_k * D + k_row * D + d]) : 0.0f;
            smem_v[j] = vv;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            float sij[FA_BLOCK_N];
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                unsigned k_row = k_start + kj;
                float dot = 0.0f;
                if (k_row < seq_k) {
                    for (unsigned d = 0; d < D; d++)
                        dot += qi[d] * smem_k[kj * D + d];
                    sij[kj] = dot * scale;
                } else {
                    sij[kj] = -3.402823466e+38f;
                }
            }

            float mij = sij[0];
            for (unsigned kj = 1; kj < FA_BLOCK_N; kj++)
                if (sij[kj] > mij) mij = sij[kj];

            float pij[FA_BLOCK_N];
            float lij = 0.0f;
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                pij[kj] = __expf(sij[kj] - mij);
                lij += pij[kj];
            }

            float mi_new     = (mi > mij) ? mi : mij;
            float scale_old  = __expf(mi - mi_new);
            float scale_new  = __expf(mij - mi_new);

            for (unsigned d = 0; d < D; d++) {
                float pv = 0.0f;
                for (unsigned kj = 0; kj < FA_BLOCK_N; kj++)
                    pv += pij[kj] * smem_v[kj * D + d];
                oi[d] = scale_old * oi[d] + scale_new * pv;
            }
            li = scale_old * li + scale_new * lij;
            mi = mi_new;
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        float inv_li = (li > 0.0f) ? 1.0f / li : 0.0f;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = fp4e2m1_from_f32(oi[d] * inv_li);
    }
}

extern "C" __global__ void k_sdpa_f64(
    const double* __restrict__ Q,
    const double* __restrict__ K,
    const double* __restrict__ V,
    double* __restrict__ Out,
    unsigned seq_q, unsigned seq_k, unsigned D, unsigned BH,
    double scale
) {
    unsigned num_q_tiles = (seq_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    unsigned bh      = blockIdx.x / num_q_tiles;
    unsigned qi_tile = blockIdx.x % num_q_tiles;
    unsigned tid     = threadIdx.x;
    unsigned qi_row  = qi_tile * FA_BLOCK_M + tid;

    extern __shared__ double smem_f64[];
    double* smem_k = smem_f64;
    double* smem_v = smem_k + FA_BLOCK_N * D;

    double qi[FA_HEAD_DIM_MAX];
    double oi[FA_HEAD_DIM_MAX];
    for (unsigned d = 0; d < D; d++) { qi[d] = 0.0; oi[d] = 0.0; }
    double mi = -1.7976931348623158e+308;
    double li = 0.0;

    if (qi_row < seq_q) {
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++) qi[d] = Q[base + d];
    }

    unsigned num_k_tiles = (seq_k + FA_BLOCK_N - 1) / FA_BLOCK_N;
    for (unsigned ki = 0; ki < num_k_tiles; ki++) {
        unsigned k_start = ki * FA_BLOCK_N;

        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            smem_k[j] = (k_row < seq_k && bh < BH)
                ? K[bh * seq_k * D + k_row * D + d] : 0.0;
        }
        for (unsigned j = tid; j < FA_BLOCK_N * D; j += FA_BLOCK_M) {
            unsigned kj    = j / D;
            unsigned d     = j % D;
            unsigned k_row = k_start + kj;
            smem_v[j] = (k_row < seq_k && bh < BH)
                ? V[bh * seq_k * D + k_row * D + d] : 0.0;
        }
        __syncthreads();

        if (qi_row < seq_q) {
            double sij[FA_BLOCK_N];
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                unsigned k_row = k_start + kj;
                double dot = 0.0;
                if (k_row < seq_k) {
                    for (unsigned d = 0; d < D; d++)
                        dot += qi[d] * smem_k[kj * D + d];
                    sij[kj] = dot * scale;
                } else {
                    sij[kj] = -1.7976931348623158e+308;
                }
            }

            double mij = sij[0];
            for (unsigned kj = 1; kj < FA_BLOCK_N; kj++)
                if (sij[kj] > mij) mij = sij[kj];

            double pij[FA_BLOCK_N];
            double lij = 0.0;
            for (unsigned kj = 0; kj < FA_BLOCK_N; kj++) {
                pij[kj] = exp(sij[kj] - mij);
                lij += pij[kj];
            }

            double mi_new    = (mi > mij) ? mi : mij;
            double scale_old = exp(mi - mi_new);
            double scale_new = exp(mij - mi_new);

            for (unsigned d = 0; d < D; d++) {
                double pv = 0.0;
                for (unsigned kj = 0; kj < FA_BLOCK_N; kj++)
                    pv += pij[kj] * smem_v[kj * D + d];
                oi[d] = scale_old * oi[d] + scale_new * pv;
            }
            li = scale_old * li + scale_new * lij;
            mi = mi_new;
        }
        __syncthreads();
    }

    if (qi_row < seq_q) {
        double inv_li = (li > 0.0) ? 1.0 / li : 0.0;
        unsigned base = bh * seq_q * D + qi_row * D;
        for (unsigned d = 0; d < D; d++)
            Out[base + d] = oi[d] * inv_li;
    }
}

extern "C" __global__ void k_im1col_f32(
    const float* __restrict__ in, float* __restrict__ col,
    unsigned C_in, unsigned L,
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L
) {
    unsigned col_elem = C_in * kL * out_L;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ol = idx % out_L;
    unsigned tmp = idx / out_L;
    unsigned kl = tmp % kL;
    unsigned c  = tmp / kL;
    int il = (int)(ol * sL + kl * dL) - (int)pL;
    float val = 0.0f;
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il];
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val;
}
extern "C" __global__ void k_im1col_f64(
    const double* __restrict__ in, double* __restrict__ col,
    unsigned C_in, unsigned L,
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L
) {
    unsigned col_elem = C_in * kL * out_L;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ol = idx % out_L;
    unsigned tmp = idx / out_L;
    unsigned kl = tmp % kL;
    unsigned c  = tmp / kL;
    int il = (int)(ol * sL + kl * dL) - (int)pL;
    double val = 0.0;
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il];
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val;
}

extern "C" __global__ void k_im1col_f16(
    const __half* __restrict__ in, __half* __restrict__ col,
    unsigned C_in, unsigned L,
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L
) {
    unsigned col_elem = C_in * kL * out_L;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ol = idx % out_L;
    unsigned tmp = idx / out_L;
    unsigned kl = tmp % kL;
    unsigned c  = tmp / kL;
    int il = (int)(ol * sL + kl * dL) - (int)pL;
    __half val = to_half(0.0f);
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il];
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val;
}

extern "C" __global__ void k_im1col_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned L,
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L
) {
    unsigned col_elem = C_in * kL * out_L;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ol = idx % out_L;
    unsigned tmp = idx / out_L;
    unsigned kl = tmp % kL;
    unsigned c  = tmp / kL;
    int il = (int)(ol * sL + kl * dL) - (int)pL;
    uint8_t val = fp8e4m3_from_f32(0.0f);
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il];
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val;
}

extern "C" __global__ void k_im1col_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned L,
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L
) {
    unsigned col_elem = C_in * kL * out_L;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ol = idx % out_L;
    unsigned tmp = idx / out_L;
    unsigned kl = tmp % kL;
    unsigned c  = tmp / kL;
    int il = (int)(ol * sL + kl * dL) - (int)pL;
    uint8_t val = fp8e5m2_from_f32(0.0f);
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il];
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val;
}

extern "C" __global__ void k_im1col_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned L,
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L
) {
    unsigned col_elem = C_in * kL * out_L;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ol = idx % out_L;
    unsigned tmp = idx / out_L;
    unsigned kl = tmp % kL;
    unsigned c  = tmp / kL;
    int il = (int)(ol * sL + kl * dL) - (int)pL;
    uint8_t val = fp4e2m1_from_f32(0.0f);
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il];
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val;
}

extern "C" __global__ void k_im3col_f32(
    const float* __restrict__ in, float* __restrict__ col,
    unsigned C_in, unsigned D, unsigned H, unsigned W,
    unsigned kD, unsigned kH, unsigned kW,
    unsigned sD, unsigned sH, unsigned sW,
    unsigned pD, unsigned pH, unsigned pW,
    unsigned dD, unsigned dH, unsigned dW,
    unsigned out_D, unsigned out_H, unsigned out_W
) {
    unsigned k_vol = C_in * kD * kH * kW;
    unsigned out_vol = out_D * out_H * out_W;
    unsigned col_elem = k_vol * out_vol;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp /= out_H;
    unsigned od = tmp % out_D;
    tmp /= out_D;
    unsigned kw = tmp % kW;
    tmp /= kW;
    unsigned kh = tmp % kH;
    tmp /= kH;
    unsigned kd = tmp % kD;
    unsigned c  = tmp / kD;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    int id = (int)(od * sD + kd * dD) - (int)pD;
    float val = 0.0f;
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow;
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val;
}
extern "C" __global__ void k_im3col_f64(
    const double* __restrict__ in, double* __restrict__ col,
    unsigned C_in, unsigned D, unsigned H, unsigned W,
    unsigned kD, unsigned kH, unsigned kW,
    unsigned sD, unsigned sH, unsigned sW,
    unsigned pD, unsigned pH, unsigned pW,
    unsigned dD, unsigned dH, unsigned dW,
    unsigned out_D, unsigned out_H, unsigned out_W
) {
    unsigned k_vol = C_in * kD * kH * kW;
    unsigned out_vol = out_D * out_H * out_W;
    unsigned col_elem = k_vol * out_vol;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp /= out_H;
    unsigned od = tmp % out_D;
    tmp /= out_D;
    unsigned kw = tmp % kW;
    tmp /= kW;
    unsigned kh = tmp % kH;
    tmp /= kH;
    unsigned kd = tmp % kD;
    unsigned c  = tmp / kD;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    int id = (int)(od * sD + kd * dD) - (int)pD;
    double val = 0.0;
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow;
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val;
}

extern "C" __global__ void k_im3col_f16(
    const __half* __restrict__ in, __half* __restrict__ col,
    unsigned C_in, unsigned D, unsigned H, unsigned W,
    unsigned kD, unsigned kH, unsigned kW,
    unsigned sD, unsigned sH, unsigned sW,
    unsigned pD, unsigned pH, unsigned pW,
    unsigned dD, unsigned dH, unsigned dW,
    unsigned out_D, unsigned out_H, unsigned out_W
) {
    unsigned k_vol = C_in * kD * kH * kW;
    unsigned out_vol = out_D * out_H * out_W;
    unsigned col_elem = k_vol * out_vol;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp /= out_H;
    unsigned od = tmp % out_D;
    tmp /= out_D;
    unsigned kw = tmp % kW;
    tmp /= kW;
    unsigned kh = tmp % kH;
    tmp /= kH;
    unsigned kd = tmp % kD;
    unsigned c  = tmp / kD;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    int id = (int)(od * sD + kd * dD) - (int)pD;
    __half val = to_half(0.0f);
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow;
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val;
}

extern "C" __global__ void k_im3col_fp8e4m3(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned D, unsigned H, unsigned W,
    unsigned kD, unsigned kH, unsigned kW,
    unsigned sD, unsigned sH, unsigned sW,
    unsigned pD, unsigned pH, unsigned pW,
    unsigned dD, unsigned dH, unsigned dW,
    unsigned out_D, unsigned out_H, unsigned out_W
) {
    unsigned k_vol = C_in * kD * kH * kW;
    unsigned out_vol = out_D * out_H * out_W;
    unsigned col_elem = k_vol * out_vol;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp /= out_H;
    unsigned od = tmp % out_D;
    tmp /= out_D;
    unsigned kw = tmp % kW;
    tmp /= kW;
    unsigned kh = tmp % kH;
    tmp /= kH;
    unsigned kd = tmp % kD;
    unsigned c  = tmp / kD;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    int id = (int)(od * sD + kd * dD) - (int)pD;
    uint8_t val = fp8e4m3_from_f32(0.0f);
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow;
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val;
}

extern "C" __global__ void k_im3col_fp8e5m2(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned D, unsigned H, unsigned W,
    unsigned kD, unsigned kH, unsigned kW,
    unsigned sD, unsigned sH, unsigned sW,
    unsigned pD, unsigned pH, unsigned pW,
    unsigned dD, unsigned dH, unsigned dW,
    unsigned out_D, unsigned out_H, unsigned out_W
) {
    unsigned k_vol = C_in * kD * kH * kW;
    unsigned out_vol = out_D * out_H * out_W;
    unsigned col_elem = k_vol * out_vol;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp /= out_H;
    unsigned od = tmp % out_D;
    tmp /= out_D;
    unsigned kw = tmp % kW;
    tmp /= kW;
    unsigned kh = tmp % kH;
    tmp /= kH;
    unsigned kd = tmp % kD;
    unsigned c  = tmp / kD;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    int id = (int)(od * sD + kd * dD) - (int)pD;
    uint8_t val = fp8e5m2_from_f32(0.0f);
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow;
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val;
}

extern "C" __global__ void k_im3col_fp4e2m1(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ col,
    unsigned C_in, unsigned D, unsigned H, unsigned W,
    unsigned kD, unsigned kH, unsigned kW,
    unsigned sD, unsigned sH, unsigned sW,
    unsigned pD, unsigned pH, unsigned pW,
    unsigned dD, unsigned dH, unsigned dW,
    unsigned out_D, unsigned out_H, unsigned out_W
) {
    unsigned k_vol = C_in * kD * kH * kW;
    unsigned out_vol = out_D * out_H * out_W;
    unsigned col_elem = k_vol * out_vol;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = blockIdx.y;
    if (idx >= col_elem) return;
    unsigned ow = idx % out_W;
    unsigned tmp = idx / out_W;
    unsigned oh = tmp % out_H;
    tmp /= out_H;
    unsigned od = tmp % out_D;
    tmp /= out_D;
    unsigned kw = tmp % kW;
    tmp /= kW;
    unsigned kh = tmp % kH;
    tmp /= kH;
    unsigned kd = tmp % kD;
    unsigned c  = tmp / kD;
    int iw = (int)(ow * sW + kw * dW) - (int)pW;
    int ih = (int)(oh * sH + kh * dH) - (int)pH;
    int id = (int)(od * sD + kd * dD) - (int)pD;
    uint8_t val = fp4e2m1_from_f32(0.0f);
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw];
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow;
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val;
}

extern "C" __global__ void k_conv_transpose2d_f32(
    const float* x, const float* w, float* out,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int out_H, int out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_H * out_W;
    if (idx >= total) return;
    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / (out_W * out_H)) % C_out;
    int b  = idx / (out_W * out_H * C_out);
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int khr = 0; khr < kH; khr++) {
            for (int kwc = 0; kwc < kW; kwc++) {
                int ih_pad = oh + pad_h;
                int iw_pad = ow + pad_w;
                if (ih_pad >= khr && iw_pad >= kwc
                    && (ih_pad - khr) % stride_h == 0
                    && (iw_pad - kwc) % stride_w == 0) {
                    int ih = (ih_pad - khr) / stride_h;
                    int iw = (iw_pad - kwc) / stride_w;
                    if (ih < H && iw < W) {
                        float xv = x[(b*C_in + ic)*H*W + ih*W + iw];
                        float wv = w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc];
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = acc;
}

extern "C" __global__ void k_conv_transpose2d_f64(
    const double* x, const double* w, double* out,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int out_H, int out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_H * out_W;
    if (idx >= total) return;
    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / (out_W * out_H)) % C_out;
    int b  = idx / (out_W * out_H * C_out);
    double acc = 0.0;
    for (int ic = 0; ic < C_in; ic++) {
        for (int khr = 0; khr < kH; khr++) {
            for (int kwc = 0; kwc < kW; kwc++) {
                int ih_pad = oh + pad_h;
                int iw_pad = ow + pad_w;
                if (ih_pad >= khr && iw_pad >= kwc
                    && (ih_pad - khr) % stride_h == 0
                    && (iw_pad - kwc) % stride_w == 0) {
                    int ih = (ih_pad - khr) / stride_h;
                    int iw = (iw_pad - kwc) / stride_w;
                    if (ih < H && iw < W) {
                        double xv = x[(b*C_in + ic)*H*W + ih*W + iw];
                        double wv = w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc];
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = acc;
}

extern "C" __global__ void k_conv_transpose2d_f16(
    const __half* x, const __half* w, __half* out,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int out_H, int out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_H * out_W;
    if (idx >= total) return;
    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / (out_W * out_H)) % C_out;
    int b  = idx / (out_W * out_H * C_out);
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int khr = 0; khr < kH; khr++) {
            for (int kwc = 0; kwc < kW; kwc++) {
                int ih_pad = oh + pad_h;
                int iw_pad = ow + pad_w;
                if (ih_pad >= khr && iw_pad >= kwc
                    && (ih_pad - khr) % stride_h == 0
                    && (iw_pad - kwc) % stride_w == 0) {
                    int ih = (ih_pad - khr) / stride_h;
                    int iw = (iw_pad - kwc) / stride_w;
                    if (ih < H && iw < W) {
                        float xv = from_half(x[(b*C_in + ic)*H*W + ih*W + iw]);
                        float wv = from_half(w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc]);
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = to_half(acc);
}

extern "C" __global__ void k_conv_transpose2d_fp8e4m3(
    const uint8_t* x, const uint8_t* w, uint8_t* out,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int out_H, int out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_H * out_W;
    if (idx >= total) return;
    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / (out_W * out_H)) % C_out;
    int b  = idx / (out_W * out_H * C_out);
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int khr = 0; khr < kH; khr++) {
            for (int kwc = 0; kwc < kW; kwc++) {
                int ih_pad = oh + pad_h;
                int iw_pad = ow + pad_w;
                if (ih_pad >= khr && iw_pad >= kwc
                    && (ih_pad - khr) % stride_h == 0
                    && (iw_pad - kwc) % stride_w == 0) {
                    int ih = (ih_pad - khr) / stride_h;
                    int iw = (iw_pad - kwc) / stride_w;
                    if (ih < H && iw < W) {
                        float xv = fp8e4m3_to_f32(x[(b*C_in + ic)*H*W + ih*W + iw]);
                        float wv = fp8e4m3_to_f32(w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc]);
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = fp8e4m3_from_f32(acc);
}

extern "C" __global__ void k_conv_transpose2d_fp8e5m2(
    const uint8_t* x, const uint8_t* w, uint8_t* out,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int out_H, int out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_H * out_W;
    if (idx >= total) return;
    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / (out_W * out_H)) % C_out;
    int b  = idx / (out_W * out_H * C_out);
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int khr = 0; khr < kH; khr++) {
            for (int kwc = 0; kwc < kW; kwc++) {
                int ih_pad = oh + pad_h;
                int iw_pad = ow + pad_w;
                if (ih_pad >= khr && iw_pad >= kwc
                    && (ih_pad - khr) % stride_h == 0
                    && (iw_pad - kwc) % stride_w == 0) {
                    int ih = (ih_pad - khr) / stride_h;
                    int iw = (iw_pad - kwc) / stride_w;
                    if (ih < H && iw < W) {
                        float xv = fp8e5m2_to_f32(x[(b*C_in + ic)*H*W + ih*W + iw]);
                        float wv = fp8e5m2_to_f32(w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc]);
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = fp8e5m2_from_f32(acc);
}

extern "C" __global__ void k_conv_transpose2d_fp4e2m1(
    const uint8_t* x, const uint8_t* w, uint8_t* out,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int out_H, int out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * out_H * out_W;
    if (idx >= total) return;
    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / (out_W * out_H)) % C_out;
    int b  = idx / (out_W * out_H * C_out);
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int khr = 0; khr < kH; khr++) {
            for (int kwc = 0; kwc < kW; kwc++) {
                int ih_pad = oh + pad_h;
                int iw_pad = ow + pad_w;
                if (ih_pad >= khr && iw_pad >= kwc
                    && (ih_pad - khr) % stride_h == 0
                    && (iw_pad - kwc) % stride_w == 0) {
                    int ih = (ih_pad - khr) / stride_h;
                    int iw = (iw_pad - kwc) / stride_w;
                    if (ih < H && iw < W) {
                        float xv = fp4e2m1_to_f32(x[(b*C_in + ic)*H*W + ih*W + iw]);
                        float wv = fp4e2m1_to_f32(w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc]);
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = fp4e2m1_from_f32(acc);
}
