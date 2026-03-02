// k_conv_misc.cuh — 1d convolution im1col/col2im + conv_transpose1d scatter-add

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
