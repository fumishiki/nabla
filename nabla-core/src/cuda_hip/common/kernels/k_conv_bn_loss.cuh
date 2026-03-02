// k_conv_bn_loss.cuh — im2col/col2im (2d/3d), batch_norm_train, fused cross_entropy + MSE loss

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

