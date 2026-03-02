// k_norm_pool.cuh — Pooling kernels: max/avg/adaptive pool2d, prod_partial

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
