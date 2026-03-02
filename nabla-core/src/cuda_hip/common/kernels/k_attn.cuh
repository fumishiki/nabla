// k_attn.cuh — FlashAttention-2 SDPA kernel (online softmax, O(seq_len) HBM)

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

