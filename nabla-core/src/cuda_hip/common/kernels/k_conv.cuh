// k_conv.cuh — im2col/col2im (1d/2d/3d), batch_norm_train, fused cross_entropy + MSE loss, conv_transpose2d, WHT quantization

// ---- im2col 2D ----

#define IM2COL_KERNEL(suffix, T, ZERO) \
extern "C" __global__ void k_im2col_##suffix( \
    const T* __restrict__ in, T* __restrict__ col, \
    unsigned C_in, unsigned H, unsigned W, \
    unsigned kH, unsigned kW, \
    unsigned sH, unsigned sW, \
    unsigned pH, unsigned pW, \
    unsigned dH, unsigned dW, \
    unsigned out_H, unsigned out_W \
) { \
    unsigned col_elem = C_in * kH * kW * out_H * out_W; \
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned n = blockIdx.y; \
    if (idx >= col_elem) return; \
    unsigned ow = idx % out_W; \
    unsigned tmp = idx / out_W; \
    unsigned oh = tmp % out_H; \
    tmp = tmp / out_H; \
    unsigned kw = tmp % kW; \
    tmp = tmp / kW; \
    unsigned kh = tmp % kH; \
    unsigned c  = tmp / kH; \
    int iw = (int)(ow * sW + kw * dW) - (int)pW; \
    int ih = (int)(oh * sH + kh * dH) - (int)pH; \
    T val = ZERO; \
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) { \
        val = in[n * C_in * H * W + c * H * W + ih * W + iw]; \
    } \
    col[n * C_in * kH * kW * out_H * out_W + (c * kH * kW + kh * kW + kw) * out_H * out_W + oh * out_W + ow] = val; \
}

IM2COL_KERNEL(f32, float, 0.0f)
IM2COL_KERNEL(f64, double, 0.0)
IM2COL_KERNEL(f16, __half, to_half(0.0f))
IM2COL_KERNEL(bf16, __nv_bfloat16, to_bf16(0.0f))
IM2COL_KERNEL(fp8e4m3, uint8_t, fp8e4m3_from_f32(0.0f))
IM2COL_KERNEL(fp8e5m2, uint8_t, fp8e5m2_from_f32(0.0f))
IM2COL_KERNEL(fp4e2m1, uint8_t, fp4e2m1_from_f32(0.0f))

// ---- batch_norm_stats (convert-via-f32 variants) ----

#define BN_STATS_NATIVE(suffix, T, ZERO) \
extern "C" __global__ void k_batch_norm_stats_##suffix( \
    const T* __restrict__ in, \
    T* __restrict__ mean_out, \
    T* __restrict__ var_out, \
    unsigned N, unsigned C \
) { \
    unsigned c = THREAD_ID; \
    if (c >= C) return; \
    T sum = ZERO, sum2 = ZERO; \
    for (unsigned n = 0; n < N; n++) { \
        T v = in[n * C + c]; \
        sum += v; sum2 += v * v; \
    } \
    T m = sum / (T)N; \
    mean_out[c] = m; \
    var_out[c] = sum2 / (T)N - m * m; \
}

BN_STATS_NATIVE(f32, float, 0.0f)
BN_STATS_NATIVE(f64, double, 0.0)

#define BN_STATS_CONV(suffix, T, to_f32, from_f32) \
extern "C" __global__ void k_batch_norm_stats_##suffix( \
    const T* __restrict__ in, \
    T* __restrict__ mean_out, \
    T* __restrict__ var_out, \
    unsigned N, unsigned C \
) { \
    unsigned c = THREAD_ID; \
    if (c >= C) return; \
    float sum = 0.0f, sum2 = 0.0f; \
    for (unsigned n = 0; n < N; n++) { \
        float v = to_f32(in[n * C + c]); \
        sum += v; sum2 += v * v; \
    } \
    float m = sum / (float)N; \
    mean_out[c] = from_f32(m); \
    var_out[c] = from_f32(sum2 / (float)N - m * m); \
}

BN_STATS_CONV(f16, __half, from_half, to_half)
BN_STATS_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16)
BN_STATS_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
BN_STATS_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
BN_STATS_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ---- batch_norm_fwd ----

#define BN_FWD_NATIVE(suffix, T, RSQRT) \
extern "C" __global__ void k_batch_norm_fwd_##suffix( \
    const T* __restrict__ in, \
    const T* __restrict__ gamma, \
    const T* __restrict__ beta, \
    const T* __restrict__ mean, \
    const T* __restrict__ var, \
    T* __restrict__ out, \
    T eps, unsigned total, unsigned C \
) { \
    unsigned idx = THREAD_ID; \
    if (idx >= total) return; \
    unsigned c = idx % C; \
    T inv_std = RSQRT(var[c] + eps); \
    out[idx] = gamma[c] * (in[idx] - mean[c]) * inv_std + beta[c]; \
}

BN_FWD_NATIVE(f32, float, rsqrtf)
#define _rsqrt_f64(x) (1.0 / sqrt(x))
BN_FWD_NATIVE(f64, double, _rsqrt_f64)
#undef _rsqrt_f64

#define BN_FWD_CONV(suffix, T, eps_type, to_f32, from_f32) \
extern "C" __global__ void k_batch_norm_fwd_##suffix( \
    const T* __restrict__ in, \
    const T* __restrict__ gamma, \
    const T* __restrict__ beta, \
    const T* __restrict__ mean, \
    const T* __restrict__ var, \
    T* __restrict__ out, \
    eps_type eps, unsigned total, unsigned C \
) { \
    unsigned idx = THREAD_ID; \
    if (idx >= total) return; \
    unsigned c = idx % C; \
    float inv_std = rsqrtf(to_f32(var[c]) + (float)eps); \
    float v = to_f32(gamma[c]) * (to_f32(in[idx]) - to_f32(mean[c])) * inv_std + to_f32(beta[c]); \
    out[idx] = from_f32(v); \
}

BN_FWD_CONV(f16, __half, double, from_half, to_half)
BN_FWD_CONV(bf16, __nv_bfloat16, double, from_bf16, to_bf16)
BN_FWD_CONV(fp8e4m3, uint8_t, float, fp8e4m3_to_f32, fp8e4m3_from_f32)
BN_FWD_CONV(fp8e5m2, uint8_t, float, fp8e5m2_to_f32, fp8e5m2_from_f32)
BN_FWD_CONV(fp4e2m1, uint8_t, float, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ---- batch_norm_update_running ----

#define BN_UPDATE_RUNNING(suffix, T, ONE) \
extern "C" __global__ void k_batch_norm_update_running_##suffix( \
    T* __restrict__ running_mean, T* __restrict__ running_var, \
    const T* __restrict__ mean, const T* __restrict__ var, \
    T momentum, unsigned C \
) { \
    unsigned c = THREAD_ID; \
    if (c >= C) return; \
    T onem = ONE - momentum; \
    running_mean[c] = onem * running_mean[c] + momentum * mean[c]; \
    running_var[c]  = onem * running_var[c]  + momentum * var[c]; \
}

BN_UPDATE_RUNNING(f32, float, 1.0f)
BN_UPDATE_RUNNING(f64, double, 1.0)

// ---- cross_entropy ----

#define CROSS_ENTROPY_NATIVE(suffix, T, ZERO, EXPF, LOGF) \
extern "C" __global__ void k_cross_entropy_##suffix( \
    const T* __restrict__ in, \
    const T* __restrict__ target, \
    T* __restrict__ loss_out, \
    unsigned N, unsigned C \
) { \
    unsigned n = THREAD_ID; \
    if (n >= N) return; \
    const T* row = in + n * C; \
    T mx = row[0]; \
    for (unsigned c = 1; c < C; c++) if (row[c] > mx) mx = row[c]; \
    T sum_exp = ZERO; \
    for (unsigned c = 0; c < C; c++) sum_exp += EXPF(row[c] - mx); \
    int t = (int)target[n]; \
    loss_out[n] = -(row[t] - mx - LOGF(sum_exp)); \
}

CROSS_ENTROPY_NATIVE(f32, float, 0.0f, expf, logf)
CROSS_ENTROPY_NATIVE(f64, double, 0.0, exp, log)

#define CROSS_ENTROPY_CONV(suffix, T, to_f32, from_f32) \
extern "C" __global__ void k_cross_entropy_##suffix( \
    const T* __restrict__ in, \
    const T* __restrict__ target, \
    T* __restrict__ loss_out, \
    unsigned N, unsigned C \
) { \
    unsigned n = THREAD_ID; \
    if (n >= N) return; \
    const T* row = in + n * C; \
    float mx = to_f32(row[0]); \
    for (unsigned c = 1; c < C; c++) { \
        float v = to_f32(row[c]); \
        if (v > mx) mx = v; \
    } \
    float sum_exp = 0.0f; \
    for (unsigned c = 0; c < C; c++) sum_exp += expf(to_f32(row[c]) - mx); \
    int t = (int)to_f32(target[n]); \
    float loss = -(to_f32(row[t]) - mx - logf(sum_exp)); \
    loss_out[n] = from_f32(loss); \
}

CROSS_ENTROPY_CONV(f16, __half, from_half, to_half)
CROSS_ENTROPY_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16)
CROSS_ENTROPY_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
CROSS_ENTROPY_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
CROSS_ENTROPY_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

// ---- im1col 1D ----

#define IM1COL_KERNEL(suffix, T, ZERO) \
extern "C" __global__ void k_im1col_##suffix( \
    const T* __restrict__ in, T* __restrict__ col, \
    unsigned C_in, unsigned L, \
    unsigned kL, unsigned sL, unsigned pL, unsigned dL, unsigned out_L \
) { \
    unsigned col_elem = C_in * kL * out_L; \
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned n = blockIdx.y; \
    if (idx >= col_elem) return; \
    unsigned ol = idx % out_L; \
    unsigned tmp = idx / out_L; \
    unsigned kl = tmp % kL; \
    unsigned c  = tmp / kL; \
    int il = (int)(ol * sL + kl * dL) - (int)pL; \
    T val = ZERO; \
    if (il >= 0 && il < (int)L) val = in[n * C_in * L + c * L + il]; \
    col[n * C_in * kL * out_L + (c * kL + kl) * out_L + ol] = val; \
}

IM1COL_KERNEL(f32, float, 0.0f)
IM1COL_KERNEL(f64, double, 0.0)
IM1COL_KERNEL(f16, __half, to_half(0.0f))
IM1COL_KERNEL(bf16, __nv_bfloat16, to_bf16(0.0f))
IM1COL_KERNEL(fp8e4m3, uint8_t, fp8e4m3_from_f32(0.0f))
IM1COL_KERNEL(fp8e5m2, uint8_t, fp8e5m2_from_f32(0.0f))
IM1COL_KERNEL(fp4e2m1, uint8_t, fp4e2m1_from_f32(0.0f))

// ---- im3col 3D ----

#define IM3COL_KERNEL(suffix, T, ZERO) \
extern "C" __global__ void k_im3col_##suffix( \
    const T* __restrict__ in, T* __restrict__ col, \
    unsigned C_in, unsigned D, unsigned H, unsigned W, \
    unsigned kD, unsigned kH, unsigned kW, \
    unsigned sD, unsigned sH, unsigned sW, \
    unsigned pD, unsigned pH, unsigned pW, \
    unsigned dD, unsigned dH, unsigned dW, \
    unsigned out_D, unsigned out_H, unsigned out_W \
) { \
    unsigned k_vol = C_in * kD * kH * kW; \
    unsigned out_vol = out_D * out_H * out_W; \
    unsigned col_elem = k_vol * out_vol; \
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned n = blockIdx.y; \
    if (idx >= col_elem) return; \
    unsigned ow = idx % out_W; \
    unsigned tmp = idx / out_W; \
    unsigned oh = tmp % out_H; \
    tmp /= out_H; \
    unsigned od = tmp % out_D; \
    tmp /= out_D; \
    unsigned kw = tmp % kW; \
    tmp /= kW; \
    unsigned kh = tmp % kH; \
    tmp /= kH; \
    unsigned kd = tmp % kD; \
    unsigned c  = tmp / kD; \
    int iw = (int)(ow * sW + kw * dW) - (int)pW; \
    int ih = (int)(oh * sH + kh * dH) - (int)pH; \
    int id = (int)(od * sD + kd * dD) - (int)pD; \
    T val = ZERO; \
    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) \
        val = in[n * C_in * D * H * W + c * D * H * W + id * H * W + ih * W + iw]; \
    unsigned k_idx = c * kD * kH * kW + kd * kH * kW + kh * kW + kw; \
    unsigned out_idx = od * out_H * out_W + oh * out_W + ow; \
    col[n * k_vol * out_vol + k_idx * out_vol + out_idx] = val; \
}

IM3COL_KERNEL(f32, float, 0.0f)
IM3COL_KERNEL(f64, double, 0.0)
IM3COL_KERNEL(f16, __half, to_half(0.0f))
IM3COL_KERNEL(bf16, __nv_bfloat16, to_bf16(0.0f))
IM3COL_KERNEL(fp8e4m3, uint8_t, fp8e4m3_from_f32(0.0f))
IM3COL_KERNEL(fp8e5m2, uint8_t, fp8e5m2_from_f32(0.0f))
IM3COL_KERNEL(fp4e2m1, uint8_t, fp4e2m1_from_f32(0.0f))

// ---- conv_transpose2d ----

#define CONV_TRANSPOSE2D_NATIVE(suffix, T, ZERO) \
extern "C" __global__ void k_conv_transpose2d_##suffix( \
    const T* x, const T* w, T* out, \
    int N, int C_in, int H, int W, \
    int C_out, int kH, int kW, \
    int out_H, int out_W, \
    int stride_h, int stride_w, \
    int pad_h, int pad_w) \
{ \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    int total = N * C_out * out_H * out_W; \
    if (idx >= total) return; \
    int ow = idx % out_W; \
    int oh = (idx / out_W) % out_H; \
    int oc = (idx / (out_W * out_H)) % C_out; \
    int b  = idx / (out_W * out_H * C_out); \
    T acc = ZERO; \
    for (int ic = 0; ic < C_in; ic++) { \
        for (int khr = 0; khr < kH; khr++) { \
            for (int kwc = 0; kwc < kW; kwc++) { \
                int ih_pad = oh + pad_h; \
                int iw_pad = ow + pad_w; \
                if (ih_pad >= khr && iw_pad >= kwc \
                    && (ih_pad - khr) % stride_h == 0 \
                    && (iw_pad - kwc) % stride_w == 0) { \
                    int ih = (ih_pad - khr) / stride_h; \
                    int iw = (iw_pad - kwc) / stride_w; \
                    if (ih < H && iw < W) { \
                        T xv = x[(b*C_in + ic)*H*W + ih*W + iw]; \
                        T wv = w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc]; \
                        acc += xv * wv; \
                    } \
                } \
            } \
        } \
    } \
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = acc; \
}

CONV_TRANSPOSE2D_NATIVE(f32, float, 0.0f)
CONV_TRANSPOSE2D_NATIVE(f64, double, 0.0)

#define CONV_TRANSPOSE2D_CONV(suffix, T, to_f32, from_f32) \
extern "C" __global__ void k_conv_transpose2d_##suffix( \
    const T* x, const T* w, T* out, \
    int N, int C_in, int H, int W, \
    int C_out, int kH, int kW, \
    int out_H, int out_W, \
    int stride_h, int stride_w, \
    int pad_h, int pad_w) \
{ \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    int total = N * C_out * out_H * out_W; \
    if (idx >= total) return; \
    int ow = idx % out_W; \
    int oh = (idx / out_W) % out_H; \
    int oc = (idx / (out_W * out_H)) % C_out; \
    int b  = idx / (out_W * out_H * C_out); \
    float acc = 0.0f; \
    for (int ic = 0; ic < C_in; ic++) { \
        for (int khr = 0; khr < kH; khr++) { \
            for (int kwc = 0; kwc < kW; kwc++) { \
                int ih_pad = oh + pad_h; \
                int iw_pad = ow + pad_w; \
                if (ih_pad >= khr && iw_pad >= kwc \
                    && (ih_pad - khr) % stride_h == 0 \
                    && (iw_pad - kwc) % stride_w == 0) { \
                    int ih = (ih_pad - khr) / stride_h; \
                    int iw = (iw_pad - kwc) / stride_w; \
                    if (ih < H && iw < W) { \
                        float xv = to_f32(x[(b*C_in + ic)*H*W + ih*W + iw]); \
                        float wv = to_f32(w[ic*(C_out*kH*kW) + oc*kH*kW + khr*kW + kwc]); \
                        acc += xv * wv; \
                    } \
                } \
            } \
        } \
    } \
    out[b*C_out*out_H*out_W + oc*out_H*out_W + oh*out_W + ow] = from_f32(acc); \
}

CONV_TRANSPOSE2D_CONV(f16, __half, from_half, to_half)
CONV_TRANSPOSE2D_CONV(bf16, __nv_bfloat16, from_bf16, to_bf16)
CONV_TRANSPOSE2D_CONV(fp8e4m3, uint8_t, fp8e4m3_to_f32, fp8e4m3_from_f32)
CONV_TRANSPOSE2D_CONV(fp8e5m2, uint8_t, fp8e5m2_to_f32, fp8e5m2_from_f32)
CONV_TRANSPOSE2D_CONV(fp4e2m1, uint8_t, fp4e2m1_to_f32, fp4e2m1_from_f32)

// === Quantization ===

// ---- WHT (Walsh-Hadamard Transform) — butterfly, one block per row ----
// Forward: unnormalized. Inverse: same butterfly + divide by cols.
// WHT_BUTTERFLY_BODY: shared butterfly loop used by all WHT variants.

#define WHT_BUTTERFLY_BODY(STYPE, S) \
    for (unsigned half = 1; half < cols; half <<= 1) { \
        for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) { \
            unsigned j = i ^ half; \
            if (j > i) { \
                STYPE a = S[i], b = S[j]; \
                S[i] = a + b; \
                S[j] = a - b; \
            } \
        } \
        __syncthreads(); \
    }

#define WHT_KERNEL_NATIVE(NAME, T, SCALE, SMEM_NAME) \
extern "C" __global__ void NAME( \
    const T* __restrict__ in, T* __restrict__ out, unsigned rows, unsigned cols \
) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    extern __shared__ T SMEM_NAME[]; \
    const T* row_in = in + row * cols; \
    T* row_out = out + row * cols; \
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) \
        SMEM_NAME[i] = row_in[i]; \
    __syncthreads(); \
    WHT_BUTTERFLY_BODY(T, SMEM_NAME) \
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) \
        row_out[i] = SMEM_NAME[i] * SCALE; \
}

WHT_KERNEL_NATIVE(k_wht_f32, float, 1.0f, smem_wht_f32)
WHT_KERNEL_NATIVE(k_wht_f64, double, 1.0, smem_wht_f64)
WHT_KERNEL_NATIVE(k_wht_inverse_f32, float, (1.0f / (float)cols), smem_wht_inv_f32)
WHT_KERNEL_NATIVE(k_wht_inverse_f64, double, (1.0 / (double)cols), smem_wht_inv_f64)

#if defined(__CUDACC__)

#define WHT_KERNEL_CONV(NAME, T, LOAD, STORE, SCALE, SMEM_NAME) \
extern "C" __global__ void NAME( \
    const T* __restrict__ in, T* __restrict__ out, unsigned rows, unsigned cols \
) { \
    unsigned row = blockIdx.x; \
    if (row >= rows) return; \
    extern __shared__ float SMEM_NAME[]; \
    const T* row_in = in + row * cols; \
    T* row_out = out + row * cols; \
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) \
        SMEM_NAME[i] = LOAD(row_in[i]); \
    __syncthreads(); \
    WHT_BUTTERFLY_BODY(float, SMEM_NAME) \
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) \
        row_out[i] = STORE(SMEM_NAME[i] * SCALE); \
}

WHT_KERNEL_CONV(k_wht_bf16, __nv_bfloat16, from_bf16, to_bf16, 1.0f, smem_wht_bf16)
WHT_KERNEL_CONV(k_wht_inverse_bf16, __nv_bfloat16, from_bf16, to_bf16, (1.0f / (float)cols), smem_wht_inv_bf16)

#endif
